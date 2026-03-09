# Inhibitory Normalization

Code accompanying the paper: **Inhibitory normalization of error signals improves learning in neural circuits**.

## Overview

This repository provides:
- **`inhib_norm/`** — core library implementing E-I layers, normalization modules, and optimizers
- **`experiments/perceptual_invariant_fashionmnist/`** — training scripts and figure notebooks for the MNIST experiments
- **`tests/`** — unit tests for all components

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
git clone https://github.com/yourusername/inhibitory-normalization.git
cd inhibitory-normalization
bash create_venv.sh
```

To activate the environment in a new shell:

```bash
source .venv/bin/activate
```

Requires Python 3.9.15 (uv will install it automatically). A GPU is recommended but not required.

## Running Experiments

Training scripts are in `experiments/perceptual_invariant_fashionmnist/src/`.

**E-I network:**
```bash
python experiments/perceptual_invariant_fashionmnist/src/train_EI_network.py \
  --opt.lr=1 --opt.wd=1e-6 \
  --opt.inhib_lrs.wei=1e-4 --opt.inhib_lrs.wix=0.5 \
  --opt.momentum=0.5 --opt.inhib_momentum=0 \
  --train.batch_size=32 --exp.name="ei_network"
```

**I-Norm network:**
```bash
python experiments/perceptual_invariant_fashionmnist/src/train_homeostatic_network.py \
  --opt.lr=0.2 --opt.wd=1e-6 \
  --opt.inhib_lrs.wei=0.5 --opt.inhib_lrs.wix=1e-4 \
  --opt.momentum=0.5 --opt.inhib_momentum=0.9 \
  --train.batch_size=32 --exp.name="homeostatic_dann"
```

Experiment tracking uses Weights & Biases (`wandb`). Set `--exp.use_wandb=True` and configure `--exp.wandb_project` and `--exp.wandb_entity` with your own account.

## Reproducing Figures

Figures in the paper are generated from results logged to Weights & Biases (`wandb`). The main project containing the experiment runs used for the paper is [`Luminosity_LNHomeostasis` on W&B](https://wandb.ai/project_danns/Luminosity_LNHomeostasis).

Figure notebooks are in `experiments/perceptual_invariant_fashionmnist/figures/`. Open and run each notebook after completing training runs, using the data from the corresponding `wandb` runs.

## Tests

```bash
python -m pytest tests/
```
