from typing import Optional

import torch
import torch.nn.functional as F


class FullDenseRetrievalLoss(torch.nn.Module):
    """
    Dense retrieval loss.
    """
    
    
    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tau = contrastive_tau
        self.normalize = normalize
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.
        """
        # query embeddings shape: (batch_size, embedding_dim)
        # key embeddings shape: (total number of items, embedding_dim)
        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)
        
        logits = torch.mm(query_embeddings, key_embeddings.t()) / self.tau

        loss = self.cross_entropy_loss(logits, labels.long())

        return loss.mean()
        


class FullBatchCrossEntropyLoss(torch.nn.Module):
    """
    Contrastive loss with negative samples being all candidates in the embedding table.
    """

    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the FullBatchContrastiveLoss.

        Parameters
        ----------
        contrastive_tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the logits via dot product.
        """
        super().__init__()
        self.normalize = normalize
        self.tau = contrastive_tau
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.

        Parameters
        ----------
        query_embeddings: torch.Tensor (batch_size x sequence length x embedding_dim)
            The embeddings of the query items.
        key_embeddings: torch.Tensor (total number of items x embedding_dim)
            The embeddings of all items, i.e the full embedding table.
        label_locations: torch.Tensor (number of labels x 2)
            The locations of the labels in the input sequences.
        labels: torch.Tensor (number of labels)
            The labels for the input sequences.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t()) / self.tau

        loss = self.cross_entropy_loss(logits, labels.long())

        return (loss * weights).sum()

