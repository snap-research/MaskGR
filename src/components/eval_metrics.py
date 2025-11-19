from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall
## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        percentage_list: Optional[List[int]] = None,
        seq_len_list: Optional[List[int]] = None,
        should_sample_negatives_from_vocab: bool = False,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
        generated_id_mask: Optional[torch.Tensor] = None,
        codebook: Optional[torch.Tensor] = None,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            if generated_id_mask is not None:
                preds = (torch.mm(query_embeddings, key_embeddings.t()) * generated_id_mask).reshape(-1)
            else:
                preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        # import pdb; pdb.set_trace()
        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, we get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # we initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # we set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )