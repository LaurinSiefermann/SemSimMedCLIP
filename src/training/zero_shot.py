import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from .mimic_zeroshot_data import CHEXPERT_CLASS_PROMPTS, CHEXPERT_COMPETITION_TASKS


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname)
                     for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def generate_class_prompts(n=None):
    '''
    Generate class prompts for zero-shot evaluation for mimic and chexpert eval set.
    From medClip paper
    '''
    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(
            f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def zero_shot_classifier_mimic(model, args):
    '''
    Create zero-shot classifier/zeroshot weights for downstream classification task.
    '''
    tokenizer = get_tokenizer(args.model)
    # create prompts for each class
    cls_prompts = generate_class_prompts(args.zeroshot_num_prompts)

    with torch.no_grad():
        zeroshot_weights = []
        for cls in tqdm(cls_prompts):
            cls_text = cls_prompts[cls]
            texts = tokenizer(cls_text).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        # returns a torch.Size([512, 5]) tensor
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def run_mimic(model, classifier, dataloader, args):
    """
    Evaluate the model's performance on a zero-shot task.

    target: 
        0 = "Atelectasis",
        1 = "Cardiomegaly",
        2 = "Consolidation",
        3 = "Edema",
        4 = "Pleural Effusion",
    """
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    total, top1_correct, top3_correct = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            labels = labels.to(args.device)

            with autocast():
                # Get image embeddings from the model
                image_embeddings = model.encode_image(images)
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                logits = 100. * image_embeddings @ classifier

            # Top-1 accuracy
            _, preds = torch.max(logits, dim=1)
            top1_correct += (preds == labels).sum().item()

            # Get top-3 predictions
            _, top3_preds = torch.topk(logits, k=3, dim=1)
            top3_correct += sum([labels[i] in top3_preds[i]
                                for i in range(len(labels))])

            total += labels.size(0)

            # For class-wise accuracy, confusion matrix, and precision/recall
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate top-1 and top-3 accuracy
    top1 = 100.0 * top1_correct / total
    top3 = 100.0 * top3_correct / total

    # Calculate class-wise accuracy
    class_accuracies = {}
    targets = ["Atelectasis", "Cardiomegaly",
               "Consolidation", "Edema", "Pleural Effusion"]
    for idx, label in enumerate(targets):
        class_accuracies[label] = 100.0 * sum([1 for p, l in zip(
            all_preds, all_labels) if p == idx and l == idx]) / all_labels.count(idx)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')

    return top1, top3, class_accuracies, conf_matrix, precision, recall, f1


def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'mimic-5x200' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    if 'imagenet-val' in data or 'imagenet-v2' in data:

        logging.info('Starting zero-shot imagenet')
        logging.info('Building zero-shot classifier')

        classifier = zero_shot_classifier(
            model, imagenet_classnames, openai_imagenet_template, args)

        logging.info('Using classifier')
        results = {}
        if 'imagenet-val' in data:
            top1, top5 = run(model, classifier,
                             data['imagenet-val'].dataloader, args)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
        if 'imagenet-v2' in data:
            top1, top5 = run(model, classifier,
                             data['imagenet-v2'].dataloader, args)
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

        logging.info('Finished zero-shot imagenet.')

    if 'mimic-5x200' in data:
        logging.info('Starting zero-shot mimic')
        logging.info('Building zero-shot classifier')

        # Classifier = zero shot weights for all classes
        classifier = zero_shot_classifier_mimic(
            model, args)

        logging.info('Using classifier')
        results = {}

        top1, top3, class_accuracies, conf_matrix, precision, recall, f1 = run_mimic(model, classifier,
                                                                                     data['mimic-5x200'].dataloader, args)
        results['mimic-zeroshot-top1'] = top1
        results['mimic-zeroshot-top3'] = top3
        results['mimic-zeroshot-class_accuracies'] = class_accuracies
        results['mimic-zeroshot-conf_matrix'] = conf_matrix.tolist()
        results['mimic-zeroshot-precision'] = precision
        results['mimic-zeroshot-recall'] = recall
        results['mimic-zeroshot-f1'] = f1

        logging.info('Finished zero-shot mimic-5x200.')

    return results
