# callbacks

Find below the existing configs for the callbacks folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## buffered_bq_writer.yaml

Inference callback to write predictions to BigQuery.


## buffered_parquet_writer.yaml

Inference callback to write predictions to GCS in Parquet format.
gcs path must start with gs://


## early_stopping.yaml

EarlyStopping is used to stop the training if the monitored quantity does not improve.
implements callback from https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html


## model_checkpoint.yaml

Callback to save the model with the best score.
Implements callback from https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html


## model_summary.yaml

Callback that prints a summary of the model to the console.
Implements callback from https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html


## rich_progress_bar.yaml

Callback to show a progress bar using the Rich library.
Implements callback from https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.RichProgressBar.html
Does not work well for unbounded datasets.
