from omegaconf import DictConfig, ListConfig, OmegaConf

""""
Hydra allows for custom resolvers, which are functions that can be used to resolve values in the config.
For example, one can manipulate strings or apply simple python functions to the config values.

"""


def remove_chars_from_string(s: str, chars: str) -> str:
    return s.translate(str.maketrans("", "", chars))


def conditional_expression(
    condition_expression, value_if_true, value_if_false, **kwargs
):
    try:
        # Evaluate the condition expression with the config context
        result = eval(condition_expression, {}, kwargs)
        return value_if_true if result else value_if_false
    except Exception as e:
        raise ValueError(
            f"Error evaluating condition: {condition_expression}. Error: {e}"
        )


def extract_fields_from_list_of_dicts(
    list_of_dicts: ListConfig,
    key: str,
    default: str = None,
    filter_key: str = None,
    filter_value: str = None,
) -> ListConfig:
    if filter_key and filter_value:
        filtered_dicts = [
            d for d in list_of_dicts if d.get(filter_key) == eval(filter_value)
        ]
    else:
        filtered_dicts = list_of_dicts

    return ListConfig([d.get(key, default) for d in filtered_dicts])


def create_map_from_list_of_dicts(
    list_of_dicts: ListConfig, key: str, value: str
) -> DictConfig:
    return DictConfig(
        {d[key]: d[value] for d in list_of_dicts if key in d and value in d}
    )


def get_gpu_count_to_machine_map(gpu_type: str, gpu_count: int) -> str:
    # please refer to https://cloud.google.com/compute/docs/general-purpose-machines
    # for a complete lits of setups
    gpu_count = int(gpu_count)
    gpu_count_machine_mapping = {
        "NVIDIA_H100_80GB": {
            1: "a3-highgpu-1g",
            2: "a3-highgpu-2g",
            4: "a3-highgpu-4g",
            8: "a3-highgpu-8g",
        },
        "NVIDIA_TESLA_A100": {
            1: "a2-highgpu-1g",
            2: "a2-highgpu-2g",
            4: "a2-highgpu-4g",
            8: "a2-highgpu-8g",
        },
        "NVIDIA_L4": {
            1: "g2-standard-16",
            2: "g2-standard-24",
            4: "g2-standard-48",
            8: "g2-standard-96",
        },
        "NVIDIA_TESLA_V100": {
            1: "n1-standard-8",
            2: "n1-standard-16",
            4: "n1-standard-32",
            8: "n1-standard-96",
        },
        "NVIDIA_TESLA_T4": {
            1: "n1-standard-16",
            2: "n1-standard-32",
            4: "n1-standard-64",
        },
    }

    if gpu_type not in gpu_count_machine_mapping:
        raise ValueError(f"GPU type {gpu_type} not supported.")

    if gpu_count not in gpu_count_machine_mapping[gpu_type]:
        raise ValueError(
            f"GPU count {gpu_count} not supported for GPU type {gpu_type}."
        )
    machine_type = gpu_count_machine_mapping[gpu_type][gpu_count]
    print(
        f"Using machine type {machine_type} for {gpu_count} {gpu_type} GPUs. If a different machine type is needed, please override the machine type using the config or command line."
    )
    return machine_type


# resolvers need to be registered to be accessible during config composition.
# The resolver name is the function name without the type annotations.
OmegaConf.register_new_resolver("remove_chars_from_string", remove_chars_from_string)
OmegaConf.register_new_resolver("conditional_expression", conditional_expression)
OmegaConf.register_new_resolver(
    "extract_fields_from_list_of_dicts", extract_fields_from_list_of_dicts
)
OmegaConf.register_new_resolver(
    "create_map_from_list_of_dicts", create_map_from_list_of_dicts
)

OmegaConf.register_new_resolver(
    "get_gpu_count_to_machine_map", get_gpu_count_to_machine_map
)
