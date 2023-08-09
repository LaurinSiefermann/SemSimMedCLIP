import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ModifiedClipLoss
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


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    if args.loss_type == 'soft_clip':
        loss = ModifiedClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=False,  # don't cache labels - in our use case they have to change every time
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)
    else:
        loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=False,  # dont cache labels - in our use case they have to change every time
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
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, raw_sentences, tokenized_sentences, tokenized_reports, chexpert_report_groups = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        tokenized_sentences = tokenized_sentences.to(
            device=device, non_blocking=True)
        tokenized_reports = tokenized_reports.to(
            device=device, non_blocking=True)
        chexpert_report_groups = chexpert_report_groups.to(
            device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        # enables automatic mixed precision (AMP)
        with autocast():
            # forward pass of the model. The model is applied to the images and texts data to produce image features, text features, and the logit scale.
            image_features, sentence_features, report_features, logit_scale = model(
                images, tokenized_sentences, tokenized_reports)

            # loss function is applied to the outputs of the model to calculate the total loss.
            if args.loss_type == 'soft_clip':
                total_loss, local_loss, local_loss_v_to_u, local_loss_u_to_v, global_loss, global_snn_loss_v_to_u, global_snn_loss_u_to_v, local_positive_pairs, global_positive_pairs = loss(image_features, sentence_features, report_features,
                                                                                                                                                                                              raw_sentences, chexpert_report_groups, logit_scale)
            else:
                total_loss, image_loss, text_loss = loss(image_features, sentence_features,
                                                         logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loading Data (t): {data_time_m.avg:.3f} "
                f"Processing Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"Local Loss: {local_loss.item()} "
                f"local_loss_v_to_u: {local_loss_v_to_u.item()} "
                f"local_loss_u_to_v: {local_loss_u_to_v.item()} "
                f"Global Loss: {global_loss.item()} "
                f"global_snn_loss_v_to_u: {global_snn_loss_v_to_u.item()} "
                f"global_snn_loss_u_to_v: {global_snn_loss_u_to_v.item()} "
                f"extra_local_positive_pairs: {(local_positive_pairs.item() - args.batch_size)/2} "
                f"extra_global_positive_pairs: {(global_positive_pairs.item() - args.batch_size)/2} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "Local Loss": local_loss.item(),
                "local_loss_v_to_u": local_loss_v_to_u.item(),
                "local_loss_u_to_v": local_loss_u_to_v.item(),
                "Global Loss": global_loss.item(),
                "global_snn_loss_v_to_u": global_snn_loss_v_to_u.item(),
                "global_snn_loss_u_to_v": global_snn_loss_u_to_v.item(),
                "extra_local_positive_pairs": (local_positive_pairs.item() - args.batch_size)/2,
                "extra_global_positive_pairs": (global_positive_pairs.item() - args.batch_size)/2,
            }
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


def evaluate(model, data, epoch, args, tb_writer=None):

    loss = ModifiedClipLoss(
        local_loss=args.local_loss,
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
                images, raw_sentences, tokenized_sentences, tokenized_reports, chexpert_report_groups = batch
                images = images.to(
                    device=device, dtype=cast_dtype, non_blocking=True)
                tokenized_sentences = tokenized_sentences.to(
                    device=device, non_blocking=True)
                tokenized_reports = tokenized_reports.to(
                    device=device, non_blocking=True)
                chexpert_report_groups = chexpert_report_groups.to(
                    device=device, non_blocking=True)

                with autocast():
                    image_features, sentence_features, report_features, logit_scale = model(
                        images, tokenized_sentences, tokenized_reports)

                    total_loss, local_loss, local_loss_v_to_u, local_loss_u_to_v, global_loss, global_snn_loss_v_to_u, global_snn_loss_u_to_v, local_positive_pairs, global_positive_pairs = loss(image_features, sentence_features, report_features,
                                                                                                                                                                                                  raw_sentences, chexpert_report_groups, logit_scale)

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

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
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
