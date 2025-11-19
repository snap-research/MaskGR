from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

from src.data.loading.components.interfaces import (
    LabelFunctionOutput,
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.loading.utils import combine_list_of_tensor_dicts, pad_or_trim_sequence


def identity_collate_fn(batch: Any) -> Any:
    """The default collate function that does nothing."""
    return batch


def return_first_element_collate_fn(batch: List[Any]) -> Any:
    """The collate function that returns the first element of the batch.
    Useful when a dataset outputs a tuple or list of elements and we only want the first one.
    """
    return batch[0]


def collate_with_sid_causal_duplicate(
    # batch can be a list or a dict
    # this function is used to create the generate contiguous sequences as data augmentation to improve the performance
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    sequence_field_name: str,
    sid_hierarchy: int,
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    max_batch_size: int = 128,
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
        this collate fn is used to create the generate contiguous sequences as data augmentation to improve the performance.
        It does three things
        1. augment the input sequences by creating all possible contiguous sequences
        2. random sample max_batch_size sequences from the augmented sequences to prevent OOM
        3. run regular collate_fn_train
    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_field_name : str
        The name of the field in the batch that contains the sequence to be augmented.
    sid_hierarchy : int
        The length of Semantic IDs
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to. (not used in this function, passed to collate_fn_train)
    masking_token : int
        The token used for masking. (not used in this function, passed to collate_fn_train)
    padding_token : int
        The token used for padding. (not used in this function, passed to collate_fn_train)
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence. (not used in this function, passed to collate_fn_train)
    max_batch_size : int
        The maximum batch size to be used after the data augmentation.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    # calculating the total number of contiguous sub-sequences in the batch
    total_num_seqs = torch.sum(
        (
            (
                k := torch.tensor([s.shape[0] for s in batch[sequence_field_name]])
                // sid_hierarchy
            )
            - 1
        )
        * k
        // 2
    )

    if total_num_seqs > max_batch_size:
        select_seqs = torch.randint(
            low=0,
            high=total_num_seqs,
            size=(max_batch_size,),
        )
    else:
        select_seqs = torch.arange(total_num_seqs)

    new_batch = {field_name: [] for field_name in batch}
    current_idx = 0
    for row_index, sequence in enumerate(batch[sequence_field_name]):
        end_indices = torch.arange(
            2 * sid_hierarchy, sequence.shape[0] + 1, sid_hierarchy
        )
        for end_index in end_indices:
            start_indices = torch.arange(
                0, end_index - 2 * sid_hierarchy + 1, sid_hierarchy
            )  # we have a -2 here because we want to have at least two items in the sequence
            for start_index in start_indices:
                if current_idx in select_seqs:
                    new_batch[sequence_field_name].append(
                        sequence[start_index:end_index]
                    )
                    for field_name in new_batch:
                        if field_name != sequence_field_name:
                            new_batch[field_name].append(batch[field_name][row_index])
                current_idx += 1

    return collate_fn_train(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
    )

def collate_fn_train(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    sid_hierarchy: int = 0,
    max_batch_size: int = 0,
    sequence_field_name: Optional[str] = None,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    data_augmentation_functions: Optional[
        List[Dict[str, callable]]
    ] = None,  # type: ignore
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """The collate function passed to dataloader. It can do training masking and padding for the input sequence.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    data_augmentation_functions : Optional[List[Dict[str, callable]]]
        The list of functions to apply to augment the data.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    if data_augmentation_functions:
        for data_augmentation_function in data_augmentation_functions:
            batch = data_augmentation_function(batch)

    model_input_data = SequentialModelInputData()
    model_label_data = SequentialModuleLabelData()

    for field_name, field_sequence in batch.items():  # type: ignore
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=min(sequence_length, current_sequence.shape[1]),
            padding_token=padding_token,
        )

        # creating labels if the field is in the labels list
        if field_name in labels:
            label_function = labels[field_name].transform
            label_function_output: LabelFunctionOutput = label_function.transform_label(
                sequence=current_sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )
            model_label_data.labels[field_name] = label_function_output.labels
            model_label_data.label_location[
                field_name
            ] = label_function_output.label_location
            model_label_data.attention_mask[
                field_name
            ] = label_function_output.attention_mask
            model_input_data.transformed_sequences[
                field_name
            ] = label_function_output.sequence
        else:
            model_input_data.transformed_sequences[field_name] = current_sequence

        # Currently supports a single masking per sequence
        if model_input_data.mask is None:
            model_input_data.mask = (current_sequence != padding_token).long()

    return model_input_data, model_label_data  # type: ignore
