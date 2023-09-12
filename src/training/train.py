import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns  # For a better-looking heatmap
from sentence_transformers import util
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import MultipleTextFeaturesClip, SingleTextFeatureClip
from open_clip import ClipLoss, get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def get_loss_weights(epoch, total_epochs, start_weight_local, end_weight_local, transition_1, transition_2):
    """
    Calculate the weights for local and global losses based on the current epoch and given transition points.

    Parameters:
    - epoch (int): The current training epoch.
    - total_epochs (int): Total number of training epochs.
    - start_weight_local (float): Weight for the local loss at the beginning of training. (local = sentence)
    - end_weight_local (float): Weight for the local loss after the second transition point.
    - transition_1 (float): Fraction of total_epochs after which the first transition in weights occurs.
    - transition_2 (float): Fraction of total_epochs after which the second transition in weights occurs.

    Returns:
    - (float, float): Tuple containing weights for local and global losses.

    Description:
    The function determines the weightage of the local and global losses based on the current epoch.
    - Up to the first transition point (`transition_1`), the local loss is given full weight (`start_weight_local`), and the global loss weight is its complement to 1.
    - Between the first and second transition points, the weight for the local loss linearly transitions from `start_weight_local` to `end_weight_local`.
    - After the second transition point, the weights remain constant at their values at the second transition point.
    The global loss weight is always the complement of the local loss weight such that their sum is 1. This ensures that the total weightage (local + global) remains constant throughout training.
    """

    # Calculate transition epochs based on the given fractions and total epochs
    first_transition_epoch = transition_1 * total_epochs
    second_transition_epoch = transition_2 * total_epochs

    # During the first phase (up to the first transition epoch)
    if epoch < first_transition_epoch:
        return start_weight_local, 1.0 - start_weight_local

    # During the second phase (between the two transition epochs)
    elif epoch < second_transition_epoch:
        # Calculate the progress within this phase as a fraction [0, 1]
        progress = (epoch - first_transition_epoch) / \
            (second_transition_epoch - first_transition_epoch)

        # Linearly interpolate between the start and end weights based on progress
        current_weight_local = start_weight_local + \
            progress * (end_weight_local - start_weight_local)

        return current_weight_local, 1.0 - current_weight_local

    # During the final phase (after the second transition epoch)
    else:
        return end_weight_local, 1.0 - end_weight_local


def get_log_file_path(logger):
    """Retrieve the file path from a logger's FileHandler."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


def append_to_json_file(data, file_path):
    """Append positive pairings to an existing JSON file or create a new one if it doesn't exist."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


def positive_pairs_to_dict(positive_pairs, instance_identifier):
    paired_indices = torch.nonzero(positive_pairs).tolist()
    paired_dict = defaultdict(list)
    for i, j in paired_indices:
        paired_dict[instance_identifier[i]].append(instance_identifier[j])
    return paired_dict


def compute_text_sim_positive_pairs(text_similarity_model, raw_texts):
    if isinstance(text_similarity_model, dict) and "tokenizer" in text_similarity_model and "model" in text_similarity_model:
        tokenizer = text_similarity_model["tokenizer"]
        model = text_similarity_model["model"]

        inputs = tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt")
        # Get the embeddings
        with torch.no_grad():
            sbert_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    else:
        # Compute sBERT embeddings for each text feature
        sbert_embeddings = text_similarity_model.encode(raw_texts, convert_to_tensor=True, show_progress_bar=False)

    threshold = 0.7151462626262626  # -> Sarah: 0.65 / run a sweep?
    # Compute cosine similarity between every pair of sentences
    similarity_matrix = util.pytorch_cos_sim(sbert_embeddings, sbert_embeddings)
    positive_pairs = (similarity_matrix > threshold).float()
    return positive_pairs


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, text_similarity_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    if args.loss_type == 'single_feature':
        loss = SingleTextFeatureClip(local_loss=args.local_loss,
                                     gather_with_grad=args.gather_with_grad,
                                     cache_labels=False,  # dont cache labels - in our use case they have to change every time
                                     rank=args.rank,
                                     world_size=args.world_size,
                                     use_horovod=args.horovod)
    elif args.loss_type == 'multiple_features':
        # Compute the loss weights for the current epoch ### Continue here.
        lambda_sentence, lambda_report = get_loss_weights(epoch, args.epochs, args.start_weight_sentence, args.end_weight_sentence, args.transition_1, args.transition_2)

        loss = MultipleTextFeaturesClip(local_loss=args.local_loss,
                                        gather_with_grad=args.gather_with_grad,
                                        cache_labels=False,  # don't cache labels - in our use case they have to change every time
                                        rank=args.rank,
                                        world_size=args.world_size,
                                        use_horovod=args.horovod)
    elif args.loss_type == 'clip':
        loss = ClipLoss(local_loss=args.local_loss,
                        gather_with_grad=args.gather_with_grad,
                        cache_labels=False,  # don't cache labels - in our use case they have to change every time
                        rank=args.rank,
                        world_size=args.world_size,
                        use_horovod=args.horovod)

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    #  used to track the average value of some quantity over time, such as the loss value during training (loss_m), the time it takes to process a batch of data
    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    # 1 Report, 2 Sentences
    positive_pairs_1_m = AverageMeter()
    positive_pairs_2_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        # TODO: rename loss_type = singleSNN / multipleSNN
        # TODO: Check if everything for calc is on the GPU
        if args.loss_type == 'single_feature':
            images, raw_texts, tokenized_texts, chexpert_groups, instance_identifier = batch

            # define positive pairs
            if args.similarity_decision_1 == 'chexPert-labels':
                positive_pairs_1 = (chexpert_groups[:, None] == chexpert_groups[None, :]).float()

            elif args.similarity_decision_1 == 'text_similarity_model':
                positive_pairs_1 = compute_text_sim_positive_pairs(text_similarity_model, raw_texts)

            # prep log positive pairs
            paired_dict_1 = positive_pairs_to_dict(positive_pairs_1, instance_identifier)

            # put positive pairs on GPU
            positive_pairs_1 = positive_pairs_1.to(device=device, non_blocking=True)

            # tokenize texts for forward pass
            tokenized_texts = tokenized_texts.to(device=device, non_blocking=True)

        elif args.loss_type == 'multiple_features':
            images, raw_sentences, raw_reports, tokenized_sentences, tokenized_reports, chexpert_sentence_groups, chexpert_report_groups, instance_identifier = batch
            # define positive pairs
            # Report-level will alawys be handled on the first level
            if args.similarity_decision_1 == 'chexPert-labels':
                positive_pairs_1 = (chexpert_report_groups[:, None] == chexpert_report_groups[None, :]).float()
            elif args.similarity_decision_2 == 'text_similarity_model':
                positive_pairs_1 = compute_text_sim_positive_pairs(text_similarity_model, raw_reports)

            # Sentence-level will alawys be handled secondly
            if args.similarity_decision_2 == 'chexPert-labels':
                positive_pairs_2 = (chexpert_sentence_groups[:, None] == chexpert_sentence_groups[None, :]).float()
            elif args.similarity_decision_2 == 'text_similarity_model':
                positive_pairs_2 = compute_text_sim_positive_pairs(text_similarity_model, raw_sentences)

            # prep log positive pairs
            paired_dict_1 = positive_pairs_to_dict(positive_pairs_1, instance_identifier)
            paired_dict_2 = positive_pairs_to_dict(positive_pairs_2, instance_identifier)

            # put positive pairs on GPU
            positive_pairs_1 = positive_pairs_1.to(device=device, non_blocking=True)
            positive_pairs_2 = positive_pairs_2.to(device=device, non_blocking=True)

            # tokenize texts for forward pass
            tokenized_sentences = tokenized_sentences.to(device=device, non_blocking=True)
            tokenized_reports = tokenized_reports.to(device=device, non_blocking=True)
        elif args.loss_type == 'clip':
            images, _, tokenized_texts, _, _ = batch
            tokenized_texts = tokenized_texts.to(device=device, non_blocking=True)

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.loss_type != 'clip':
            positive_pairs_1_m.update(positive_pairs_1.sum().item())
            if args.loss_type == 'multiple_features':
                positive_pairs_2_m.update(positive_pairs_2.sum().item())

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        # enables automatic mixed precision (AMP)
        with autocast():
            if args.loss_type == 'single_feature' or args.loss_type == 'clip':
                image_features, text_features, logit_scale = model(images, tokenized_texts)
            elif args.loss_type == 'multiple_features':
                image_features, sentence_features, report_features, logit_scale = model(images, tokenized_sentences, tokenized_reports)

            if args.loss_type == 'single_feature':
                total_loss = loss(image_features, text_features, positive_pairs_1, logit_scale)
            elif args.loss_type == 'multiple_features':
                total_loss, sentence_loss, report_loss = loss(image_features, sentence_features, report_features, positive_pairs_1,
                                                              positive_pairs_2, logit_scale, lambda_sentence, lambda_report)
            elif args.loss_type == 'clip':
                total_loss, image_loss, text_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        # store positive pairings
        if (epoch == 0 or epoch == 1) and args.save_pairings and args.loss_type != 'clip':

            # Pair each instance with its corresponding raw text/sentence
            if args.loss_type == 'single_feature':

                instance_text_pairs = [{"instance": instance, "text": text}
                                       for instance, text in zip(instance_identifier, raw_texts)]
                pairings_data = {
                    f"epoch_{epoch}_batch_{batch_count}": {
                        "instances": instance_text_pairs,
                        "paired_dict": paired_dict_1
                    }
                }

            elif args.loss_type == 'multiple_features':
                pairings_data_report = {
                    f"epoch_{epoch}_batch_{batch_count}": {
                        "instances": instance_identifier,
                        "paired_dict": paired_dict_1
                    }
                }
                instance_sentence_pairs = [{"instance": instance, "sentence": sentence}
                                           for instance, sentence in zip(instance_identifier, raw_sentences)]
                pairings_data_sentence = {
                    f"epoch_{epoch}_batch_{batch_count}": {
                        "instances": instance_sentence_pairs,
                        "paired_dict": paired_dict_2
                    }
                }

            log_file_path = get_log_file_path(logging.getLogger())
            if log_file_path:
                log_dir = os.path.dirname(log_file_path)
                if args.loss_type == 'single_feature':
                    pairings_file_path = os.path.join(log_dir, f'paired_dict.json')
                else:
                    sentence_file_path = os.path.join(log_dir, f'paired_dict_sentence.json')
                    report_file_path = os.path.join(log_dir, f'paired_dict_report.json')
            else:
                pairings_file_path = f'paired_dict.json'
                sentence_file_path = f'paired_dict_sentence.json'
                report_file_path = f'paired_dict_report.json'

            # Save to the respective files
            if args.loss_type == 'single_feature':
                append_to_json_file(pairings_data, pairings_file_path)
            elif args.loss_type == 'multiple_features':
                append_to_json_file(pairings_data_sentence, sentence_file_path)
                append_to_json_file(pairings_data_report, report_file_path)

        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()

            log_string = (
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loading Data (t): {data_time_m.avg:.3f} "
                f"Processing Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )

            if args.loss_type == 'single_feature':
                single_feature_log_string = (
                    f"Positive Pairs: {positive_pairs_1_m.val:.2f} "
                )
                log_string += single_feature_log_string

            if args.loss_type == 'multiple_features':
                multiple_features_log_string = (
                    f"Sentence Loss: {sentence_loss.item()} "
                    f"Report Loss: {report_loss.item()} "
                    f"Lambda Sentence: {lambda_sentence:.3f} "
                    f"Lambda Report: {lambda_report:.3f}"
                    f"Report Positive Pairs: {positive_pairs_1_m.val:.2f} "
                    f"Sentence Positive Pairs: {positive_pairs_2_m.val:.2f} "
                )
                log_string += multiple_features_log_string

            logging.info(log_string)

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            if args.loss_type == 'single_features':
                single_features_log_data = {
                    "positive_pairs": positive_pairs_1_m.val,
                }
                log_data.update(single_features_log_data)

            if args.loss_type == 'multiple_features':
                multiple_features_log_data = {
                    "Sentence Loss": sentence_loss.item(),
                    "Report Loss": report_loss.item(),
                    "Lambda Sentence": lambda_sentence,
                    "Lambda Report": lambda_report,
                    "Report positive pairs": positive_pairs_1_m.val,
                    "Sentence positive pairs": positive_pairs_2_m.val,
                }
                log_data.update(multiple_features_log_data)

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step, 'epoch': epoch})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, text_similarity_model, args, tb_writer=None):
    if args.loss_type == 'single_feature':
        loss = SingleTextFeatureClip(local_loss=args.local_loss,
                                     gather_with_grad=args.gather_with_grad,
                                     cache_labels=False,  # dont cache labels - in our use case they have to change every time
                                     rank=args.rank,
                                     world_size=args.world_size,
                                     use_horovod=args.horovod)
    elif args.loss_type == 'multiple_features':
        loss = MultipleTextFeaturesClip(local_loss=args.local_loss,
                                        gather_with_grad=args.gather_with_grad,
                                        cache_labels=False,  # don't cache labels - in our use case they have to change every time
                                        rank=args.rank,
                                        world_size=args.world_size,
                                        use_horovod=args.horovod)
    elif args.loss_type == 'clip':
        loss = ClipLoss(local_loss=args.local_loss,
                        gather_with_grad=args.gather_with_grad,
                        cache_labels=False,  # don't cache labels - in our use case they have to change every time
                        rank=args.rank,
                        world_size=args.world_size,
                        use_horovod=args.horovod)

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                if args.loss_type == 'single_feature':
                    images, raw_texts, tokenized_texts, chexpert_groups, instance_identifier = batch

                    # define positive pairs
                    if args.similarity_decision_1 == 'chexPert-labels':
                        positive_pairs_1 = (chexpert_groups[:, None] == chexpert_groups[None, :]).float()

                    elif args.similarity_decision_1 == 'text_similarity_model':
                        positive_pairs_1 = compute_text_sim_positive_pairs(text_similarity_model, raw_texts)

                    tokenized_texts = tokenized_texts.to(device=device, non_blocking=True)

                elif args.loss_type == 'multiple_features':
                    images, raw_sentences, raw_reports, tokenized_sentences, tokenized_reports, chexpert_sentence_groups, chexpert_report_groups, instance_identifier = batch

                    # Report-level will alawys be handled on the first level
                    if args.similarity_decision_1 == 'chexPert-labels':
                        positive_pairs_1 = (chexpert_report_groups[:, None] == chexpert_report_groups[None, :]).float()
                    elif args.similarity_decision_2 == 'text_similarity_model':
                        positive_pairs_1 = compute_text_sim_positive_pairs(text_similarity_model, raw_reports)

                    # Sentence-level will alawys be handled secondly
                    if args.similarity_decision_2 == 'chexPert-labels':
                        positive_pairs_2 = (chexpert_sentence_groups[:, None] == chexpert_sentence_groups[None, :]).float()
                    elif args.similarity_decision_2 == 'text_similarity_model':
                        positive_pairs_2 = compute_text_sim_positive_pairs(text_similarity_model, raw_sentences)

                    # put positive pairs on GPU
                    positive_pairs_1 = positive_pairs_1.to(device=device, non_blocking=True)
                    positive_pairs_2 = positive_pairs_2.to(device=device, non_blocking=True)

                    tokenized_sentences = tokenized_sentences.to(device=device, non_blocking=True)
                    tokenized_reports = tokenized_reports.to(device=device, non_blocking=True)
                elif args.loss_type == 'clip':
                    images, _, tokenized_texts, _, _ = batch
                    tokenized_texts = tokenized_texts.to(device=device, non_blocking=True)

                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)

                with autocast():
                    if args.loss_type == 'single_feature' or args.loss_type == 'clip':
                        image_features, text_features, logit_scale = model(images, tokenized_texts)
                    elif args.loss_type == 'multiple_features':
                        image_features, sentence_features, report_features, logit_scale = model(images, tokenized_sentences, tokenized_reports)

                    if args.loss_type == 'single_feature':
                        total_loss = loss(image_features, text_features, positive_pairs_1, logit_scale)
                    elif args.loss_type == 'multiple_features':
                        lambda_sentence, lambda_report = get_loss_weights(epoch, args.epochs, args.start_weight_sentence,
                                                                          args.end_weight_sentence, args.transition_1, args.transition_2)
                        total_loss, sentence_loss, report_loss = loss(image_features, sentence_features, report_features, positive_pairs_1,
                                                                      positive_pairs_2, logit_scale, lambda_sentence, lambda_report)
                    elif args.loss_type == 'clip':
                        total_loss, image_loss, text_loss = loss(image_features, text_features, logit_scale)

                batch_size = len(images)
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            loss = cumulative_loss / num_samples
            metrics.update({"val_loss": loss.item(), "epoch": epoch,
                           "num_samples": num_samples})

    # TODO: Deleted the part for the metrics -> see how to include it again.
    # -> look at notion explaination for metrics and adapt later
    if not metrics:
        return metrics

    targets = ["Atelectasis", "Cardiomegaly",
               "Consolidation", "Edema", "Pleural Effusion"]

    log_entries = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):  # Scalar values
            log_entries.append(f"{k}: {round(v, 4):.4f}")
        elif isinstance(v, dict):  # Dictionary values (e.g., class accuracies)
            summary = ', '.join(
                [f"{key}: {round(val, 4):.4f}" for key, val in v.items()])
            log_entries.append(f"{k}: {summary}")
        elif isinstance(v, list):  # List values (e.g., confusion matrix)
            trace = sum(v[i][i] for i in range(len(targets)))
            log_entries.append(f"{k} Trace: {trace}")
        else:
            log_entries.append(f"{k}: {v}")

    logging.info(f"Eval Epoch: {epoch} " + "\t".join(log_entries))

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'

        # add if staetement for mimic zeroshot metrics
        if epoch % args.zeroshot_frequency == 0:
            # Logging class accuracies as a bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(targets, [metrics['mimic-zeroshot-class_accuracies'][t]
                    for t in targets])
            plt.xlabel('Classes')
            plt.ylabel('Accuracy')
            plt.title('Class-wise Accuracies')
            plt.tight_layout()
            wandb.log({"Class Accuracies": [wandb.Image(plt)], 'step': epoch})

            # Logging the confusion matrix as a heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(metrics['mimic-zeroshot-conf_matrix'], annot=True,
                        fmt='d', cmap='Blues', xticklabels=targets, yticklabels=targets)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            wandb.log({"Confusion Matrix": [wandb.Image(plt)], 'step': epoch})

        # Logging other metrics
        for name, val in metrics.items():
            # Skip logging these since we're handling them separately
            if name not in ['mimic-zeroshot-class_accuracies', 'mimic-zeroshot-conf_matrix']:
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @
                        text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image,
              "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
