import torch
import torch.nn as nn
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(
                image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(
                text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # log this loss
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2

        return total_loss, image_loss, text_loss


class MultipleTextFeaturesClip(ClipLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = 0.7151462626262626

    def compute_similarity(self, raw_texts):
        # Compute sBERT embeddings for each text feature
        sbert_embeddings = self.sbert_model.encode(
            raw_texts, convert_to_tensor=True, show_progress_bar=False)
        # Compute cosine similarity between every pair of sentences
        similarity_matrix = util.pytorch_cos_sim(
            sbert_embeddings, sbert_embeddings)
        return similarity_matrix

    def compute_local_loss(self, image_features, text_features, raw_texts, instance_identifier, logit_scale):
        similarity_matrix = self.compute_similarity(raw_texts)
        positive_pairs = (
            similarity_matrix > self.threshold).float()

        local_positive_pairs = positive_pairs.sum()

        logits_per_image = (image_features @ text_features.T) * logit_scale
        logits_per_text = (text_features @ image_features.T) * logit_scale

        snn_loss_v_to_u = -torch.log(torch.sum(torch.exp(logits_per_image)
                                     * positive_pairs) / torch.sum(torch.exp(logits_per_image)))
        snn_loss_u_to_v = -torch.log(torch.sum(torch.exp(logits_per_text.T)
                                     * positive_pairs) / torch.sum(torch.exp(logits_per_text.T)))
        snn_loss = (snn_loss_v_to_u + snn_loss_u_to_v) / 2

        paired_indices_local = torch.nonzero(positive_pairs).tolist()
        paired_dict_local = defaultdict(list)
        for i, j in paired_indices_local:
            paired_dict_local[instance_identifier[i]].append(
                instance_identifier[j])

        return snn_loss, snn_loss_v_to_u, snn_loss_u_to_v, local_positive_pairs, paired_dict_local

    def compute_global_loss(self, image_features, report_features, chexpert_report_groups, instance_identifier, logit_scale):
        # Compute similarity matrix
        similarity_matrix = (
            chexpert_report_groups[:, None] == chexpert_report_groups[None, :]).float()
        positive_pairs = similarity_matrix

        global_positive_pairs = positive_pairs.sum()

        logits_per_image = (image_features @ report_features.T) * logit_scale
        logits_per_report = (report_features @ image_features.T) * logit_scale

        snn_loss_v_to_u = -torch.log(torch.sum(torch.exp(
            logits_per_image) * positive_pairs) / torch.sum(torch.exp(logits_per_image)))
        snn_loss_u_to_v = -torch.log(torch.sum(torch.exp(
            logits_per_report.T) * positive_pairs) / torch.sum(torch.exp(logits_per_report.T)))
        snn_loss = (snn_loss_v_to_u + snn_loss_u_to_v) / 2

        paired_indices_global = torch.nonzero(positive_pairs).tolist()
        paired_dict_global = defaultdict(list)
        for i, j in paired_indices_global:
            paired_dict_global[instance_identifier[i]].append(
                instance_identifier[j])

        return snn_loss, snn_loss_v_to_u, snn_loss_u_to_v, global_positive_pairs, paired_dict_global

    def forward(self, image_features, sentence_features, report_features, raw_sentences, chexpert_report_groups, instance_identifier, logit_scale, lambda_local=0.5, lambda_global=0.5):
        local_loss, local_loss_v_to_u, local_loss_u_to_v, local_positive_pairs, paired_dict_local = self.compute_local_loss(
            image_features, sentence_features, raw_sentences, instance_identifier, logit_scale)

        global_loss, global_snn_loss_v_to_u, global_snn_loss_u_to_v, global_positive_pairs, paired_dict_global = self.compute_global_loss(
            image_features, report_features, chexpert_report_groups, instance_identifier, logit_scale)

        total_loss = lambda_local * local_loss + lambda_global * global_loss

        return total_loss, local_loss, local_loss_v_to_u, local_loss_u_to_v, global_loss, global_snn_loss_v_to_u, global_snn_loss_u_to_v, local_positive_pairs, global_positive_pairs, paired_dict_local, paired_dict_global


class SingleTextFeatureClip(ClipLoss):
    def __init__(self, sBERT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if sBERT == Report = all distilroberta-v1 = more tokens
        if sBERT == "REPORT":
            self.sbert_model = SentenceTransformer("all-distilroberta-v1")
        else:
            self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = 0.7151462626262626

    def compute_similarity(self, raw_text):
        # Compute sBERT embeddings for each text feature
        sbert_embeddings = self.sbert_model.encode(
            raw_text, convert_to_tensor=True, show_progress_bar=False)
        # Compute cosine similarity between every pair of sentences
        similarity_matrix = util.pytorch_cos_sim(
            sbert_embeddings, sbert_embeddings)
        return similarity_matrix

    def forward(self, image_features, text_features, raw_text, logit_scale):
        similarity_matrix = self.compute_similarity(raw_text)
        positive_pairs = (
            similarity_matrix > self.threshold).float()

        logits_per_image = (image_features @ text_features.T) * logit_scale
        logits_per_text = (text_features @ image_features.T) * logit_scale

        snn_loss_v_to_u = -torch.log(torch.sum(torch.exp(logits_per_image)
                                     * positive_pairs) / torch.sum(torch.exp(logits_per_image)))
        snn_loss_u_to_v = -torch.log(torch.sum(torch.exp(logits_per_text.T)
                                     * positive_pairs) / torch.sum(torch.exp(logits_per_text.T)))
        loss = (snn_loss_v_to_u + snn_loss_u_to_v) / 2

        return loss
