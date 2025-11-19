# dataset_config

Find below the existing configs for the dataset_config folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## base_text_dataset.yaml

Configurations text based datasets.


## language_modeling_dataset.yaml

Configurations for language modeling datasets.


## random_rec_dataset.yaml

Configuration to create a random dataset based on torchrec format.
It will generate sparse and dense features if those are passed below.
Useful for testing and benchmarking purposes.
For more information about the parameters, please refer to the RandomRecDataset class documentation:
https://github.com/pytorch/torchrec/blob/c2f7d61/torchrec/datasets/random.py#L125


## text_classification_dataset.yaml

Configurations text based datasets doing classification tasks.
