# Masked Diffusion for Generative Recommendation

This repository implements a masked diffusion model on semantic-ID (SID) representations of items. It contains training and evaluation code, configuration files (Hydra), and utilities for data loading and model checkpoints.

This README explains how to prepare the environment, curate the SID data (using the GRID pipeline), point the config to your curated SID file, and run training and inference with the provided Makefile targets.

## Quick start

- Train (GPU):

```bash
make train-gpu ARGS="experiment=discrete_diffusion_train"
```

- Resume training from a checkpoint (GPU):

```bash
make train-gpu ARGS="experiment=discrete_diffusion_train ckpt_path=<path/to/checkpoint.ckpt>"
```

Note: the project also provides `make inference` and `make inference-gpu` targets which call `src/inference.py` directly. The `train-gpu` command above will run the training script with the `trainer=ddp` override as the project Makefile is configured.

## Prerequisites

We recommend Python 3.10. The following steps can be used to create an environment and setup the dependencies.

```bash
conda create -n madrec python=3.10 -y
conda activate madrec
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository layout (short)

- `src/` — application code: training script, model modules, data loaders, evaluation, utilities.
- `configs/` — Hydra config files. The main training experiment is `configs/experiment/discrete_diffusion_train.yaml`.
- `Makefile` — convenient targets for training, inference, testing and development tasks.

## Two-part data setup (SID creation + training sequences)

We recommend organizing data preparation into two separate steps so the SID creation and the dataset split / sequence formatting are clearly separated.

### SID Creation

This project expects a curated SID (semantic id) tensor saved to disk (typically a `.pt` file). For data curation, follow the GRID repository and its instructions:

- GRID (GitHub): https://github.com/snap-research/GRID

Steps:

1. Clone GRID and follow its README to run the curation pipeline. GRID contains the scripts and documentation for creating semantic-ID datasets (feature extraction, clustering/quantization, and saving semantic id maps).
2. After running the GRID pipeline, you should have a file containing the SID mapping (example: `flan-t5-xl_rkmeans_4_256_seed43.pt`).
3. Move or copy the resulting SID file into your project or to a location accessible from the training environment.

Example curated SID file path (project-relative):

```
./data/amazon/beauty/sids/flan-t5-xl_rkmeans_4_256_seed43.pt
```

4. Open `configs/experiment/discrete_diffusion_train.yaml` and set the `sid_data_path` key to point to your curated `.pt` file. For example:

```yaml
sid_data_path: ./data/amazon/beauty/sids/flan-t5-xl_rkmeans_4_256_seed43.pt
```




### Set training / evaluation / testing sequences

We provide pre-processed Amazon data explored in the P5 paper ([P5 paper](https://arxiv.org/abs/2203.13366)). The data can be downloaded from this Google Drive link: [Google Drive download](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view). The code expects a dataset folder structured like the examples under `./data/amazon/beauty/`.

```
./data/amazon/beauty/
├─ sids/
│  └─ flan-t5-xxl_rkmeans_4_256_seed43.pt       # SID tensor produced by GRID
├─ training/
│  └─ *.tfrecord.gz                             # Training sequences (GZIP format)
├─ validation/
│  └─ *.tfrecord.gz                             # Validation sequences (GZIP format)
└─ testing/
   └─ *.tfrecord.gz                             # Testing sequences (GZIP format)
```

For the MovieLens-1M dataset, we use a script `src/data/ml1m_preparation.py`. This script will download, preprocess and then store the data in `data/ml1m` folder. 

## Training

Use the Makefile training target which wraps `src/train.py` with Hydra.

Train on GPU(s):

```bash
make train-gpu ARGS="experiment=discrete_diffusion_train"
```

You can pass any Hydra config override in the ARGS string such as `optim.optimizer.lr` or `batch_size`.

### Checkpoints

Checkpoints are managed by the callbacks configured in the experiment YAML. By default checkpoints are saved under `${paths.output_dir}/checkpoints` (see `configs/*/paths`).

To resume or run inference from a checkpoint, provide `ckpt_path` (Hydra override) in the ARGS string. Example:

```bash
make train-gpu ARGS="experiment=discrete_diffusion_train ckpt_path=/path/to/checkpoint.ckpt"
```

## Inference / Evaluation

You can run the inference script directly using the Makefile `train-gpu` targets (these call `src/train.py`):

```bash
make train-gpu ARGS="experiment=discrete_diffusion_train train=false ckpt_path=/path/to/checkpoint.ckpt"
```

Evaluation metrics and retrieval settings are configured in `configs/eval/sid_retrieval.yaml` and the experiment YAML; you can modify `eval.evaluator` related keys to adjust top-k and sequence length evaluation parameters.

## Acknowledgements

This work builds upon the GRID framework for creating semantic-ID representations: [Paper](https://arxiv.org/abs/2507.22224) and [Github repository](https://github.com/snap-research/GRID).