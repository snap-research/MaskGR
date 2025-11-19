from abc import ABC, abstractmethod
from typing import Optional
import torch
from src.data.loading.components.interfaces import LabelFunctionOutput
class LabelFunction(ABC):
    """
    An interface for the LabelFunction classes. The LabelFunction classes are used to transform the input sequence for training or inference and collecting the labels and label prediction locations.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform_label(self, sequence: torch.Tensor) -> LabelFunctionOutput:
        """
        Function to transform the input sequence for training or inference and collecting the labels and label prediction locations.
        The exact functionality needs to be implemented in the child class.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, new_sequence_length)
                The original input sequence transformed for training or inference. The new_sequence_length can be the same as the original sequence length or different depending on the LabelFunction.
            labels: torch.Tensor of size (number of labels,)
                The labels across the sequences in the batch stacked together as labels for training.
            label_location: torch.Tensor of size (number of masked tokens, 2)
                The (row, col) location of the label_prediction corresponding to the labels in the batch stacked together.
        """
        raise NotImplementedError("Need to implement in the child class.")
class PrecomputedLabel(LabelFunction):
    """
    No-op label function that returns the precomputed labels.
    """

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int = 0
    ) -> LabelFunctionOutput:
        return LabelFunctionOutput(sequence=None, labels=sequence, label_location=None)
class Identity(LabelFunction):
    """
    LabelFunction class to return the original input for the non-masked values of the sequence. It's useful in situations where we don't to transform the input sequence.
    """

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int = 0, masking_token: int = 0
    ) -> LabelFunctionOutput:
        """
        Returns the original input for the non-masked values of the sequence.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
            The padding_token is defined in the DataloaderConfig of LightningDataModule and is passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
            Thus, to change the padding_token, the user needs to change the padding_token in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length)
                The original input sequence.
            labels: torch.Tensor of size (number of non-padding tokens in batch,)
                Collecting all the non-padding tokens in the batch stacked together as labels for training.
            label_location: torch.Tensor of size (number of non-padding tokens in batch, 2)
                The location of the label predictions, that is the (row, col) location of all non-padding tokens in the batch stacked together.
        """
        content_mask = sequence != padding_token
        labels = sequence[content_mask]
        label_location = content_mask.nonzero()

        return LabelFunctionOutput(
            sequence=sequence, labels=labels, label_location=label_location
        )
class RandomMasking(LabelFunction):
    """
    LabelFunction class to implement `Masked Language Model` style masking similar to Bert4Rec (https://arxiv.org/pdf/1904.06690).
    It randomly masks a proportion of tokens from the sequence with the masking token.
    """

    def __init__(
        self,
        masking_probability: float = 0.15,
        masking_tolerance: float = 0.05,
        max_trials_for_masking: int = 5,
    ):
        """
        Initialize the RandomMasking LabelFunction.

        Parameters
        ----------
        masking_probability: float
            The probability of masking any token in the sequence. Each token is masked with this probability.
        masking_tolerance: float
            The tolerance for the masking proportion. If the sampled masked proportion is not close to the desired masking probability within the tolerance, we re-sample the masking.
        max_trials_for_masking: int
            The maximum number of times we sample the masking in case the masking proportion is not within the tolerance. A ValueError is raised if the masking proportion is not within the tolerance after max_trials_for_masking.
        """
        self.masking_probability = masking_probability
        self.masking_tolerance = masking_tolerance
        self.max_trials_for_masking = max_trials_for_masking

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int, masking_token: int
    ) -> LabelFunctionOutput:
        """
        Function to transform the input sequence by randomly masking a proportion of tokens with the masking token and collecting the labels and label prediction locations.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
        masking_token: int
            The index of token for masked tokens.

        The padding_token and the masking_token are defined in the DataloaderConfig of LightningDataModule and are passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
        Thus, to change the padding_token and the masking_token, the user needs to change them in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length)
                The original input sequence transformed such that the tokens are randomly masked with the masking token.
            labels: torch.Tensor of size (number of masked tokens,)
                The original tokens that were masked across the sequences in the batch stacked together as labels for training.
            label_location: torch.Tensor of size (number of masked tokens, 2)
                The location of the label predictions, that is the (row, col) location of all masked tokens in the batch stacked together.
        """
        content_mask = sequence != padding_token
        # for sequence with no tokens due to oov handling
        # previously we removed oov tokens.
        if (content_mask.sum(1) == 0).any():
            sequence = sequence[content_mask.sum(1) > 0]
            content_mask = sequence != padding_token

        row_ids = torch.arange(sequence.size(0))
        masking_proportion = 0

        # If we are unlucky and the masking proportion is not close to the desired masking_probability, we re-sample
        # NOTE: This is a very rare case, but it can happen.
        counter = 0
        while (
            not self.masking_probability - self.masking_tolerance
            < masking_proportion
            < self.masking_probability + self.masking_tolerance
        ):
            masking_mask = torch.rand(sequence.size()) < self.masking_probability
            # we only mask the tokens that are not padding
            masking_mask = masking_mask & content_mask
            masking_proportion = (
                (masking_mask.sum(dim=1) / content_mask.sum(dim=1)).mean().item()
            )
            counter += 1
            if counter > self.max_trials_for_masking:
                raise ValueError(
                    f"Cannot get the desired masking proportion after {counter} trials."
                )

        assert masking_mask is not None, "masking_mask is None"
        # making sure each row has at least one masked token
        one_random_index_by_session = torch.multinomial(
            content_mask.float(), num_samples=1
        ).squeeze()
        masking_mask[row_ids, one_random_index_by_session] = True

        # getting the location of masks and their corresponding tokens (labels)
        labels = sequence[masking_mask]
        sequence[masking_mask] = masking_token

        label_location = (
            masking_mask.nonzero()
        )  # these are the (i,j) coordinates in the masking_mask

        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )
class NextTokenMasking(LabelFunction):
    """
    LabelFunction class to mask one token at the end of the sequence. It implements the `Next Action` training objective from Pinnerformer (https://arxiv.org/pdf/2205.04507).
    If for inference, we add one more masking token at the end of the sequence. If for eval, we replace the last token with the masking token and regard the last token as the label.
    """

    def __init__(self, is_for_inference: bool = False):
        """
        Initialize the NextTokenMasking LabelFunction.

        Parameters
        ----------
        is_for_inference: bool
            Denotes if the masking is for inference or evaluation. If is_for_inference is true, we just add a masked token at the end of the sequence otherwise we replace the last token with the masking token and regard the last token as the label.
        """
        self.is_for_inference = is_for_inference

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int, masking_token: int
    ) -> LabelFunctionOutput:
        """
        For each row of sequence, we mask the last element of the sequence and regard it as the label.
        If is_for_inference is true, then we just add an additional mask token at the end of the sequence.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
        masking_token: int
            The index of token for masked tokens.

        The padding_token and the masking_token are defined in the DataloaderConfig of LightningDataModule and are passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
        Thus, to change the padding_token and the masking_token, the user needs to change them in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length) if is_for_inference is False, else (batch_size, sequence_length + 1)
                The original input sequence transformed such that the last token is replaced with 1 masking token. If is_for_inference is True, then we add an additional mask token at the end of the sequence.
            labels: torch.Tensor of size (batch_size,)
                The last tokens of each row chosen as labels stacked together. If is_for_inference is True, then all values are equal to the masking_token.
            label_location: torch.Tensor of size (batch_size * self.next_k, 2)
                The location of the label predictions, that is the location of the masking token for each row.
        """
        row_ids = torch.arange(sequence.size(0))
        content_mask = sequence != padding_token
        last_indices = content_mask.sum(1)

        if self.is_for_inference:
            # first adding a padding to the sequence tensor such that full length sequence will have
            # one more token at the end
            padding_tensor = torch.full((sequence.shape[0], 1), padding_token)
            sequence = torch.cat([sequence, padding_tensor], dim=-1)
            # Adding a masking_token at the end to conduct prediction
            sequence[row_ids, last_indices] = masking_token
        else:
            # for eval mode, we regard the last token as the label,
            # whose location is the last token (i.e., num_of_tokens - 1)
            last_indices -= 1

        labels = sequence[row_ids, last_indices]
        # masking the token after the last token as 1 to conduct prediction
        sequence[row_ids, last_indices] = masking_token

        label_location = (sequence == masking_token).nonzero()
        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )
class NextKTokenMasking(LabelFunction):
    """
    LabelFunction to create masking to use the last K tokens at the end of each sequence as labels.
    Implements the `All Action` training objective from Pinnerformer (https://arxiv.org/pdf/2205.04507)
    """

    def __init__(self, next_k: int = 5):
        """
        Initialize the LabelFunction with the number of tokens to mask as labels.

        Parameters
        ----------
        next_k: int
            Number of tokens K to mask as labels.
        """
        self.next_k = next_k

    def transform_label(
        self, sequence: torch.Tensor, padding_token: int, masking_token: int
    ) -> LabelFunctionOutput:
        """
        For each row of sequence, we save the original last next_k tokens as labels and replace them with 1 masking token and next_k - 1 padding tokens.
        For all the next_k tokens, we use the first masked token as the label prediction.

        Parameters
        ----------
        sequence: torch.Tensor of size (batch_size, sequence_length)
            The matrix for input sequences.
        padding_token: int
            The index of token for padding.
        masking_token: int
            The index of token for masked tokens.

        The padding_token and the masking_token are defined in the DataloaderConfig of LightningDataModule and are passed to the collate_fn which then passes it to the LabelFunction's transform_label function.
        Thus, to change the padding_token and the masking_token, the user needs to change them in the DataloaderConfig.

        Returns
        ----------
        A LabelFunctionOutput object with the following attributes:
            sequence: torch.Tensor of size (batch_size, sequence_length),
                The original input sequence transformed such that for each row, the last next_k non-padding tokens are replaced with 1 masking token and next_k - 1 padding tokens.
            labels: torch.Tensor of size (batch_size * self.next_k,)
                The original last next_k tokens of each row chosen as labels stacked together.
            label_location: torch.Tensor of size (batch_size * self.next_k, 2)
                The location of the label predictions, for each row the first masked token's location is returned as the label prediction location for all the next_k tokens.
        """
        content_mask = sequence != padding_token

        # check each sequence in batch is greater than next_k + 1
        unpadded_seq_lengths = content_mask.sum(1)  # shape: (batch_size,)
        if torch.any(unpadded_seq_lengths < self.next_k + 1):
            raise ValueError(
                f"Sequence lengths: {unpadded_seq_lengths[unpadded_seq_lengths < self.next_k + 1]} should be greater than next_k + 1: {self.next_k + 1}"
            )

        # for each row, we want elements from unpadded_seq_lengths - next_k to unpadded_seq_lengths - 1 as labels
        label_start_indices = unpadded_seq_lengths - self.next_k  # shape: (batch_size,)

        # for each row, we select [label_start_indices, label_start_indices + 1, ..., label_start_indices + next_k - 1]
        label_col_offset = torch.arange(self.next_k)  # shape: (next_k,)
        label_col_indices = (
            label_start_indices.unsqueeze(1) + label_col_offset
        )  # shape: (batch_size, next_k)
        label_col_indices = label_col_indices.reshape(
            -1
        )  # shape: (batch_size * next_k,)

        # To get the row indices, we repeat each row index next_k times
        row_orig_indices = torch.arange(sequence.size(0))  # shape: (batch_size,)
        row_interleaved_indices = row_orig_indices.repeat_interleave(
            self.next_k
        )  # shape: (batch_size * next_k,)

        labels = sequence[row_interleaved_indices, label_col_indices]

        # for prediction, for each row, we replace the last next_k tokens with
        # 1 masking token at the label_start_indices, and next_k - 1 with padding tokens
        sequence[row_interleaved_indices, label_col_indices] = padding_token
        sequence[row_orig_indices, label_start_indices] = masking_token

        # for each row, we use the label_start_index (which was masked) as label prediction for all the next_k labels
        label_location = torch.stack(
            (
                row_interleaved_indices,
                label_start_indices.repeat_interleave(self.next_k),
            ),
            dim=1,
        )

        return LabelFunctionOutput(
            sequence=sequence,
            labels=labels,
            label_location=label_location,
        )

