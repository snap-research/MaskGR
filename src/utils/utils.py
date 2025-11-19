import warnings
from typing import Any, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from transformers.cache_utils import DynamicCache

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def delete_module(module: torch.nn.Module, module_name: str) -> None:
    if hasattr(module, module_name):
        delattr(module, module_name)

    for name, submodule in module.named_children():
        delete_module(submodule, module_name)


def find_module_shape(
    module: torch.nn.Module, module_name: str
) -> Optional[torch.Size]:
    if hasattr(module, module_name):
        return getattr(module, module_name).weight.shape

    for name, submodule in module.named_children():
        shape = find_module_shape(submodule, module_name)
        if shape:
            return shape
    return None


def reset_parameters(module: torch.nn.Module) -> None:

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for layer in module.children():
            reset_parameters(layer)


def get_var_if_not_none(value: Optional[Any], default_value: Any) -> Any:
    return value if value is not None else default_value


def get_class_name_str(class_definition: Any) -> str:
    return ".".join([class_definition.__module__, class_definition.__name__])


def has_class_object_inside_list(obj_list: list, class_type: Any) -> bool:
    return any(isinstance(obj, class_type) for obj in obj_list)


def convert_legacy_kv_cache_to_dynamic(
    kv_cache: Union[DynamicCache, Tuple[torch.Tensor]]
) -> DynamicCache:
    if isinstance(kv_cache, DynamicCache):
        return kv_cache

    return DynamicCache.from_legacy_cache(kv_cache)


def get_parent_module_and_attr(
    model: torch.nn.Module, module_name: str
) -> Tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def lightning_precision_to_dtype(precision: str) -> torch.dtype:
    # Mapping from Lightning precision identifiers to PyTorch dtypes
    precision_map = {
        "32": torch.float32,  # Single precision (float32)
        "32-true": torch.float32,  # Also maps to float32, useful for clarity when specifying defaults
        "64": torch.float64,  # Double precision
        "16": torch.float16,  # Half precision
        "16-mixed": torch.float16,  # Mixed precision typically uses torch.float16
        "bf16": torch.bfloat16,  # BFloat16 precision
        "half": torch.float16,  # Alias for half precision
    }

    if precision in precision_map:
        return precision_map[precision]
    else:
        raise ValueError(
            f"Unsupported precision type: '{precision}'. "
            "Supported precision types are: '32', '32-true', '64', '16', '16-mixed', 'bf16', 'half'."
        )
