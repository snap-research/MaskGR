import torch
import torch.nn as nn
from src.models.components.network_blocks.aggregation_strategy import (
    AggregationStrategy,
)
class EmbeddingAggregator(nn.Module):
    """Embedding aggregator function. this determins how user embeddings are aggregated to form the final user embedding.

    Parameters
    ----------
    aggregation_type: str
        aggregation function type
    """

    def __init__(
        self,
        aggregation_strategy: AggregationStrategy,
    ):
        super(EmbeddingAggregator, self).__init__()
        self.aggregation_strategy = aggregation_strategy

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # we -1 here because the token index starts from 0
        last_item_index = attention_mask.sum(dim=1) - 1
        row_ids = torch.arange(embeddings.size(0))

        return self.aggregation_strategy.aggregate(embeddings, row_ids, last_item_index)