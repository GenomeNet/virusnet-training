# VirusNet Training

Fine-tune a pre-trained BERT model for binary classification of DNA sequences
(virus vs. non-virus) using [deepG](https://github.com/GenomeNet/deepG).

## Quick Start

```bash
git clone https://github.com/GenomeNet/virusnet-training.git
cd virusnet-training

# 1. Install environment (GPU)
bash install.sh

# 2. Download base model + training data
mamba activate virusnet
python download.py model archaea non-virus virus

# 3. Train (submit to SLURM)
mkdir -p logs
sbatch train.slurm
```

## Installation

No root/sudo required. The install script sets up
[Miniforge](https://github.com/conda-forge/miniforge) (mamba), creates a conda
environment with R, TensorFlow, and deepG.

```bash
bash install.sh          # GPU (CUDA 11.8 + cuDNN 8.6 + TensorFlow 2.12)
bash install.sh --cpu    # CPU-only (for testing without GPU)
```

Verify the installation:

```bash
mamba activate virusnet
$CONDA_PREFIX/bin/Rscript test_deepg.R
```

Quick smoke test on SLURM:

```bash
sbatch tests/test_quick.slurm          # CPU only, ~2 min
sbatch tests/test_gpu.slurm            # requires GPU + downloaded model
```

## Downloading Models and Data

The `download.py` script fetches the base model and training datasets:

```bash
python download.py --list     # show all available assets with sizes
python download.py            # download everything (~135 GB total)
```

Individual assets:

| Asset | Description | Size |
|-------|-------------|------|
| `model` | Pre-trained BERT base model (`llm_1k_bert.h5`) | 2.8 GB |
| `virus` | Virus sequences (positive class) | — |
| `archaea` | Additional archaea sequences (non-virus) | 190 MB |
| `non-virus` | Non-virus sequences | 3.7 GB |
| `virusnet-sim` | Subsampled simulated data (non-virus) | 6.1 GB |
| `bio-bakery` | BioBakery reference data (non-virus) | 125 GB |

```bash
# Minimum required for training:
python download.py model virus non-virus archaea

# Specific datasets:
python download.py archaea non-virus
```

After downloading, the directory structure is:

```
virusnet-training/
├── models/
│   └── llm_1k_bert.h5                    # base BERT model
├── data/
│   ├── VirusNet_data_virus/{train,validation}/   # virus fasta files
│   ├── non_virus/{train,validation}/              # non-virus fasta files
│   ├── additional-archaea-merged/{train,validation}/
│   └── bio_bakery/{train,validation}/             # (optional, large)
```

## Training

### Option A: Fasta-based training (default)

The simplest approach. Reads fasta files directly during training:

```bash
# Interactive (single GPU):
mamba activate virusnet
Rscript train_virusnet.R

# SLURM submission:
sbatch train.slurm
```

### Option B: RDS-based training (faster)

Pre-generates training batches as `.rds` files, then trains from them.
Significantly faster because fasta parsing is done once upfront:

```bash
# Step 1: Generate RDS batches (run until you have enough, then Ctrl+C)
Rscript generate_rds.R

# Step 2: Train from RDS data
Rscript train_virusnet_rds.R

# Or via SLURM:
sbatch --export=ALL,VIRUSNET_TRAIN_SCRIPT=train_virusnet_rds.R train.slurm
```

### Configuration

All training scripts accept environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIRUSNET_BASE_MODEL` | `models/llm_1k_bert.h5` | Path to pre-trained BERT model |
| `VIRUSNET_DATA` | `data/` | Directory with training datasets |
| `VIRUSNET_CHECKPOINTS` | `checkpoints/` | Where to save model checkpoints |
| `VIRUSNET_LOGS` | `logs/` | Training log directory |
| `VIRUSNET_RUN_NAME` | `virusnet_bert_finetune_1` | Name for this training run |

### Resuming from a checkpoint

Checkpoints are saved in `checkpoints/<run_name>/`. To resume:

```r
library(deepG)
model <- load_cp("checkpoints/virusnet_bert_finetune_1", compile = TRUE)
```

### GPU Memory Guidelines

| GPU    | VRAM  | Recommended batch_size |
|--------|-------|------------------------|
| T4     | 16 GB | 8–16                   |
| V100   | 32 GB | 16–32                  |
| A100   | 80 GB | 32–64                  |
| H100   | 80 GB | 32–64                  |


## Model Architecture and Input Encoding

### Pre-trained BERT model (`llm_1k_bert.h5`)

A BERT-style transformer pre-trained with **masked language modeling (MLM)** on
virus genome sequences. The model learns to predict randomly masked nucleotides,
building general representations of DNA sequence patterns.

| Parameter | Value |
|-----------|-------|
| Input shape | `(batch, 1000)` |
| Encoding | **1-gram integer tokens** |
| Vocabulary | 6 tokens: `0`=pad/N, `1`=A, `2`=C, `3`=G, `4`=T, `5`=MASK |
| Embedding dim | 650 |
| Transformer blocks | 18 |
| Attention heads | 20 per block |
| Head size | 200 |
| Feed-forward dim | 1950 |
| Dropout | 0.1 |
| Output (pre-training) | Dense(256, relu) → Dense(6, softmax) |
| Size | ~2.8 GB |

### Input encoding

The BERT models use **1-gram integer encoding** (`n_gram = 1`,
`return_int = TRUE` in deepG). Each nucleotide is mapped to a single integer:

```
DNA sequence:   A  C  G  T  A  C  N  G  T  ...
Integer tokens: 1  2  3  4  1  2  0  3  4  ...
```

The embedding layer maps each integer to a learned dense vector (dim 650).
The vocabulary has 6 values: `4 nucleotides + 1 padding/ambiguous + 1 mask = 6`.

Auto-detection from a loaded model:

```r
# output_shape[[3]] == 6  → 1-gram (4^1 + 2 = 6)
# output_shape[[3]] == 4098 → 6-gram (4^6 + 2 = 4098)
n_gram <- ifelse(model$output_shape[[3]] == 6, 1, 6)
```

### Fine-tuning process

The binary VirusNet classifier is built by:

1. **Loading** the pre-trained BERT (`llm_1k_bert.h5`)
2. **Removing** the MLM output head (final Dense layers)
3. **Adding** new classification layers (e.g. `Dense(128) → Dense(2, softmax)`)
4. **Training** with focal loss to handle class imbalance

Two strategies are available:

| | `train_virusnet.R` | `train_virusnet_rds.R` |
|---|---|---|
| Data format | Fasta (on-the-fly) | Pre-generated RDS batches |
| Base frozen? | No (all layers trainable) | Yes (only last block + head) |
| Learning rate | 5e-08 (low, full fine-tune) | 5e-05 (higher, mostly frozen) |
| Pooling | Flatten | Global max pooling |
| Dense head | (128, 2) | (64, 2) |
| Speed | Slower (fasta I/O) | Faster (pre-batched) |

Both use **focal loss** (gamma=2–2.5, alpha=[0.5, 5]) to upweight the virus
(minority) class, and the same 1-gram integer encoding as the base BERT model.

### Comparison: BERT binary vs. CNN/LSTM genus model

| | Binary classifier (BERT) | Genus classifier (CNN/LSTM) |
|---|---|---|
| Architecture | Transformer (18 blocks) | CNN + optional LSTM |
| Input shape | `(batch, 1000)` | `(batch, 2000, 4)` |
| Encoding | 1-gram integer tokens | One-hot encoded |
| `return_int` | `TRUE` | `FALSE` |
| Sequence length | 1000 bp | 2000 bp |
| Pre-training | MLM on virus genomes | None (from scratch) |

## Training Scripts Overview

| Script | Purpose |
|--------|---------|
| `train_virusnet.R` | Main fine-tuning script (fasta-based, default) |
| `train_virusnet_rds.R` | RDS-based fine-tuning (faster, needs `generate_rds.R` first) |
| `generate_rds.R` | Pre-generate RDS training batches from fasta data |
| `train.slurm` | SLURM job submission script |
| `download.py` | Download base model and training datasets |
| `install.sh` | Set up conda environment with deepG |
| `test_deepg.R` | Comprehensive test suite |
