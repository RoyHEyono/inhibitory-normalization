# Inhibitory Normalization

Code accompanying the paper: **Inhibitory normalization of error signals improves learning in neural circuits**.

## Overview

This repository provides:
- **`inhib_norm/`** — core library implementing E-I layers, normalization modules, and optimizers
- **`experiments/dense_mnist/`** — training scripts and figure notebooks for the MNIST experiments
- **`tests/`** — unit tests for all components

## Installation

```bash
git clone https://github.com/yourusername/inhibitory-normalization.git
cd inhibitory-normalization
pip install -e .
```

Dependencies are listed in `requirements.txt`. A GPU is recommended but not required.

## Running Experiments

Training scripts are in `experiments/dense_mnist/src/`.

**E-I network (non-homeostatic):**
```bash
python experiments/dense_mnist/src/train_EI_network.py \
  --opt.lr=1 --opt.wd=1e-6 \
  --opt.inhib_lrs.wei=1e-4 --opt.inhib_lrs.wix=0.5 \
  --opt.momentum=0.5 --opt.inhib_momentum=0 \
  --train.batch_size=32 --exp.name="ei_network"
```

**Homeostatic E-I network:**
```bash
python experiments/dense_mnist/src/train_homeostatic_network.py \
  --opt.lr=0.2 --opt.wd=1e-6 \
  --opt.inhib_lrs.wei=0.5 --opt.inhib_lrs.wix=1e-4 \
  --opt.momentum=0.5 --opt.inhib_momentum=0.9 \
  --train.batch_size=32 --exp.name="homeostatic_dann"
```

Experiment tracking uses Weights & Biases (`wandb`). Set `--exp.use_wandb=True` and configure `--exp.wandb_project` and `--exp.wandb_entity` with your own account.

## Reproducing Figures

Figure notebooks are in `experiments/dense_mnist/figures/`. Open and run each notebook after completing training runs.

## Tests

```bash
python -m pytest tests/
```
