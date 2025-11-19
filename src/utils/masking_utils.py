from typing import Optional
import torch
def create_causal_attention_mask(
    batch_size: int, sequence_length: int, attention_mask: Optional[torch.tensor] = None
) -> torch.tensor:
    """
    Creates a causal attention mask for a given batch size and sequence length.
    Args:
        batch_size (int): The size of the batch.
        sequence_length (int): The length of the sequence.
        attention_mask (Optional[torch.tensor]): An optional attention mask tensor of shape
            (batch_size, sequence_length). If provided, it will be multiplie with the causal max to ensure only valid tokens are attended to.
            attention mask. If None, only the causal mask will be returned.
    Returns:
        torch.tensor: A tensor representing the causal attention mask of shape
            (batch_size, sequence_length, sequence_length).
    """

    causal_attention_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=attention_mask.dtype if attention_mask is not None else torch.bool,
            device=attention_mask.device if attention_mask is not None else None,
        )
    )  # Shape: (seq_length, seq_length)

    # Expand causal_mask to (batch_size, seq_length, seq_length)
    causal_attention_mask = causal_attention_mask.unsqueeze(0).expand(
        batch_size, -1, -1
    )  # Shape: (batch_size, seq_length, seq_length)

    if attention_mask is not None:
        # Multiply causal mask with attention mask to ensure every token attends to only valid tokens.
        # Note that for self attention, the expected shape of the final causal mask is (batch_size, seq_length, seq_length), where the last two dimensions are the query and key dimensions.
        # Thus, the attention mask is broadcasted along the second last dimension to ensure each token in the query attends to only valid tokens in the key
        causal_attention_mask = (
            causal_attention_mask * attention_mask[:, None, :]
        )  # Shape: (batch_size, seq_length, seq_length)
    return causal_attention_mask
def create_last_k_mask(
    sequence_length: int, last_item_index: torch.Tensor, last_k: Optional[int] = None
) -> torch.tensor:
    """
    Creates a mask to select the last K items of sequences.
    If a sequence has less than K items, all items are considered for the row.
    If last_k is None, all items are considered for all rows.

    Args:
        sequence_length (int): The length of the sequences.
        last_item_index (torch.Tensor) of shape (batch_size,).
            The tensor containing the indices of the last items in the each row
        last_k (Optional[int]): The number of last K items to consider.
            If None, all items are considered.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, sequence_length) with
            True for the last K items in each row and False for the rest.
    """

    if last_k is None:
        start_index = torch.zeros_like(last_item_index)
    else:
        if last_k < 1:
            raise ValueError("last_k must be None or greater than or equal to 1")
        start_index = torch.clamp(
            last_item_index - last_k + 1, min=0
        )  # Shape (batch_size,)

    indices = (
        torch.arange(sequence_length, device=last_item_index.device)
        .unsqueeze(0)
        .expand(last_item_index.size(0), -1)
    )  # shape (batch_size, sequence_length)

    mask = (indices >= start_index.unsqueeze(1)) & (
        indices <= last_item_index.unsqueeze(1)
    )  # Shape (batch_size, sequence_length)
    return mask