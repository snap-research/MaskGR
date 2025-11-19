# data_loading

Find below the existing configs for the data_loading folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## language_modeling.yaml

Configuration for datamodule that loads text and prepares it for
language modeling tasks. Can be used for Masked Language Modeling (MLM)
or Causal Language Modeling (CLM).


## language_modeling_inference.yaml

Configuration for datamodule that loads text and prepares it
for language modeling inference.


## text_classification.yaml

Configuration for datamodule that loads text and prepares it for
text classification tasks.


## text_classification_inference.yaml

Configuration for datamodule that loads text and prepares it for
text classification inference.
