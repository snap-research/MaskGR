# features_config

Find below the existing configs for the features_config folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## criteo_features.yaml

Configuration for the full Criteo dataset. Used for
local or gfs testing since Criteo datamodule is not able to load from GCS.


## criteo_features_small.yaml

Configuration for the Criteo dataset with only 13 dense features and 4 sparse id ones. Used for
local or gfs testing since Criteo datamodule is not able to load from GCS.


## language_modeling_features.yaml

Features configuration for language modeling tasks.


## random_sparse_features.yaml

Random dataset with 3 sparse features and 30 dense features. Used mainly to test e2e pipelines and stress test
TorchRec.


## text_classification_features.yaml

Features configuration for text classification tasks.
