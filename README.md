# VirusNet: Retraining the BERT Masked Language Model for Viral Genomes

Retrain the VirusNet BERT masked language model using
[deepG](https://github.com/GenomeNet/deepG) with GPU-accelerated TensorFlow.

## Setup

No root/sudo required. The install script sets up
[Miniforge](https://github.com/conda-forge/miniforge) (mamba), creates a conda
environment, and installs TensorFlow, R, and deepG.

```bash
git clone https://github.com/GenomeNet/virusnet-training.git
cd virusnet-training
bash install.sh          # GPU
bash install.sh --cpu    # CPU-only (for testing)
```

Verify:

```bash
mamba activate virusnet
$CONDA_PREFIX/bin/Rscript test_deepg.R
```

## Base Model

The base model (`llm_1k_bert.h5`) is a BERT-like masked language model we
pre-trained on genomic sequences using deepG. It serves as the starting point
for fine-tuning the binary VirusNet classifier.

Download the pre-trained weights:

```bash
python download_model.py            # saves to models/llm_1k_bert.h5
```

Or manually:

```
https://research.bifo.helmholtz-hzi.de/downloads/genomenet/llm_1k_bert.h5
```

## Training Data

<!-- TODO: Add download URL once data is published -->

Training data will be provided as a downloadable archive.

## Training

Example SLURM submission:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate virusnet
export RETICULATE_PYTHON=$(which python)

$CONDA_PREFIX/bin/Rscript train_virusnet.R \
  --data_dir "$VIRUSNET_DATA" \
  --checkpoint_dir ./checkpoints
```

Resume from a checkpoint:

```r
model <- deepG::load_cp("/path/to/checkpoint_dir", compile = TRUE)
```

## GPU Memory Guidelines

| GPU    | VRAM  | batch_size |
|--------|-------|------------|
| T4     | 16 GB | 8-16       |
| V100   | 32 GB | 16-32      |
| A100   | 80 GB | 32-64      |
| H100   | 80 GB | 32-64      |
