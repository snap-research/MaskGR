from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import torch

from src.data.loading.components.interfaces import (
    BaseDatasetConfig,
    SemanticIDDatasetConfig,
)

# support functions


def is_feature_in_features_to_apply(features_to_apply: List[str], k: str) -> bool:
    if len(features_to_apply) > 0 and k not in features_to_apply:
        return False
    return True


# All pre_processing functions must have the following signature:
# def my_pre_processing_function(batch_or_row : Dict[str, Any],  dataset_config: DatasetConfig, features_to_apply: Optional[List[str]]=[], **kwargs) -> Any:
# If the function only works for batch or for a row, make it explicit in the documentation and/or function name


def filter_features_to_consider(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, tf.Tensor]:
    batch_or_row = map_feature_names(batch_or_row, dataset_config)
    features_to_consider = dataset_config.features_to_consider
    if hasattr(dataset_config, "keep_user_id") and dataset_config.keep_user_id:
        if dataset_config.user_id_field not in features_to_consider:
            features_to_consider.append(dataset_config.user_id_field)
    if hasattr(dataset_config, "keep_item_id") and dataset_config.keep_item_id:
        if dataset_config.item_id_field not in features_to_consider:
            features_to_consider.append(dataset_config.item_id_field)
    if len(dataset_config.features_to_consider):
        # Given a batch or row, filter the features to consider.
        return {k: v for k, v in batch_or_row.items() if k in features_to_consider}
    # if not specified, we consider all features
    return batch_or_row


def convert_to_dense_numpy_array(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # Transform a tfrecord example to a dictionary of numpy arrays, converting sparse tensors to dense numpy arrays.

    for k in batch_or_row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = tf.sparse.to_dense(batch_or_row[k]).numpy()
    return batch_or_row


def map_feature_names(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # Given a batch or row, map the feature names to the desired feature names.
    if dataset_config.feature_map:
        batch_or_row = {
            v: batch_or_row[k]
            for k, v in dataset_config.feature_map.items()
            if k in batch_or_row
        }
    return batch_or_row


def convert_fields_to_tensors(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    # Given a batch or row, convert all fields to torch tensors. Uses the field type map to determine the dtype, defaulting to torch.long
    # if no dtype is specified.
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if isinstance(v, int) or isinstance(v, float):
                v = [int(v)]
            batch_or_row[k] = torch.tensor(v, dtype=dataset_config.field_type_map.get(k, torch.long))  # type: ignore
    return batch_or_row


def add_placeholder_tokens(
    batch_or_row: Dict[str, torch.Tensor],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Given a batch or row of torch Tensors, values in the tensor are incremented
    by the number of placeholder tokens to account for the new tokens based on
    the num_placeholder_tokens_map in the dataset config.
    If a feature is not specified in the map, its values remain unchanged.
    """
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            num_placeholder_tokens = dataset_config.num_placeholder_tokens_map.get(k, 0)
            if torch.any(v < 0):
                raise ValueError(
                    f"Negative tokens: {torch.unique(v[v < 0])} present in feature: {k}. "
                    "Ensure that all tokens are non-negative before adding placeholder tokens."
                )
            batch_or_row[k] = v + num_placeholder_tokens
    return batch_or_row


## Row only


def remove_oov_tokens(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
    # Given a row, remove the oov tokens from the row.
    oov_token = kwargs.get("oov_token", -1)
    # We only need to access the first element of the values, so we use next(iter()) to speed it up
    mask = torch.ones_like(next(iter(row.values())), dtype=torch.bool)
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            mask = mask & (v != oov_token)
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            row[k] = v[mask]
    return row


def filter_sequence_length_row(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with sequence length smaller than the min threshold.
    for _, tensor in row.items():
        if len(tensor) < dataset_config.min_sequence_length:
            return None
    return row


def filter_empty_feature(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with empty tensors.
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if len(v) == 0:
                return None
    return row


def filter_only_oov(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with only the oov token.
    oov_token = kwargs.get("oov_token", -1)
    for _, tensor in row.items():
        if (tensor == oov_token).all():
            return None
    return row


def map_sparse_id_to_semantic_id(
    row: Dict[str, torch.Tensor],
    dataset_config: SemanticIDDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    num_hierarchies: Optional[int] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Given a row of data, maps the sparse ids to semantic ids
    based on the id_map in the dataset config.
    """

    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            id_map: torch.Tensor = dataset_config.semantic_id_map.get(k, None)
            # id_map is a D x N tensor
            # where N is the number of unique items in the dataset
            # and D is the number of hierarchies (semantic id digits)
            if id_map is not None:
                # flatten the semantic id sequence
                if num_hierarchies is None:
                    row[k] = id_map.t()[v].view(-1)
                else:
                    assert num_hierarchies <= id_map.size(
                        0
                    ), "num_hierarchies must be less than or equal to the number of hierarchies in the semantic id map."
                    row[k] = id_map[:num_hierarchies].t()[v].view(-1)
            else:
                raise ValueError(f"Semantic id map not found for feature {k}")
    return row


def convert_bytes_to_string(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # For each feature to apply, cast its np.ndarray of bytes to string.
    for k in batch_or_row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = batch_or_row[k].astype(str)
    return batch_or_row


def convert_sparse_float_list_string_to_array(
    row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            row[k] = np.array(
                list(
                    map(
                        float,
                        tf.sparse.to_dense(v).numpy()[0].decode("utf-8").split(","),
                    )
                )
            )
    return row
