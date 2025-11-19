# amazon_p5

Find below the existing configs for the amazon_p5 folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## amazon_p5_dataloader.yaml

Dataloader config for reading Amazon P5 data from an UnboundedSequenceIterable dataset.
For more details on the Amazon dataset please see https://amazon-reviews-2023.github.io/index.html
The dataset is preprocessed as in the P5 paper: https://arxiv.org/pdf/2203.13366.


## amazon_p5_items_dataloader.yaml

Dataloader config for reading Amazon P5 items features from an UnboundedSequenceIterable dataset.
