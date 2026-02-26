# VirusNet: Retraining the BERT Masked Language Model for Viral Genomes

This guide explains how to set up the environment and retrain the VirusNet BERT
masked language model using the [deepG](https://github.com/GenomeNet/deepG) R package
with GPU-accelerated TensorFlow.

## Prerequisites

- A Linux machine (or SLURM cluster) with NVIDIA GPUs
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
  [Mamba](https://mamba.readthedocs.io/) installed
- At least one GPU with >= 16 GB VRAM (32 GB+ recommended for the full model)
- Training data: FASTA files organized into `train/`, `validation/`, and `test/` subdirectories

## 1. Prepare Training Data

VirusNet expects a directory with three subdirectories containing FASTA files:

```
training_data/
├── train/           # training FASTA files
├── validation/      # validation FASTA files
└── test/            # held-out test FASTA files
```

Each FASTA file contains viral genome sequences. If you already have a dataset,
set the path as an environment variable for the rest of this guide:

```bash
export VIRUSNET_DATA=/path/to/your/training_data
```

## 2. Create the Conda Environment

```bash
conda create -n virusnet_gpu \
  python=3.10 \
  r-base=4.3 \
  r-essentials \
  -c conda-forge -y

conda activate virusnet_gpu
```

## 3. Install TensorFlow with GPU Support

deepG requires TensorFlow 2.12. Install it with CUDA/cuDNN support:

```bash
# Install CUDA toolkit and cuDNN via conda (most reliable method)
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y

# Install TensorFlow
pip install tensorflow==2.12.1
pip install h5py==3.9.0

# Verify GPU detection
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPUs:', tf.config.list_physical_devices('GPU'))
"
```

If the above does not detect GPUs, try one of these alternatives:

```bash
# Alternative A: TF with bundled CUDA (pip-only, no conda CUDA needed)
pip install tensorflow[and-cuda]==2.12.1

# Alternative B: TF 2.15 with bundled CUDA (newer, but check deepG compat)
pip install tensorflow[and-cuda]==2.15.0

# Alternative C: Manual cuDNN via pip
pip install nvidia-cudnn-cu11==8.6.0.163
pip install tensorflow==2.12.1
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__.rsplit('/', 1)[0])")
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
```

## 4. Install R Packages

```bash
Rscript -e '
install.packages(c(
  "reticulate", "magrittr", "dplyr", "ggplot2",
  "hdf5r", "zoo", "optparse", "data.table",
  "yardstick", "microseq"
), repos = "https://cran.r-project.org")
'

Rscript -e '
install.packages(c("tensorflow", "keras"), repos = "https://cran.r-project.org")
'
```

## 5. Install deepG

```bash
Rscript -e '
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("GenomeNet/deepG")
'
```

If that fails (e.g., no internet on compute nodes), clone and install locally:

```bash
git clone https://github.com/GenomeNet/deepG.git
Rscript -e 'install.packages("./deepG", repos = NULL, type = "source")'
```

## 6. Configure `RETICULATE_PYTHON`

R's `reticulate` package must know which Python to use. Set this **before** running
any R scripts that load TensorFlow:

```bash
export RETICULATE_PYTHON=$(conda run -n virusnet_gpu which python)
```

Add this to your SLURM job scripts as well.

## 7. Verify the Setup

Run the included test script to confirm everything works:

```bash
# Set required env vars
export RETICULATE_PYTHON=$(conda run -n virusnet_gpu which python)
export VIRUSNET_DATA=/path/to/your/training_data

# On a SLURM cluster with GPUs:
srun --partition=gpu --gres=gpu:1 --time=10 \
  $(conda run -n virusnet_gpu which Rscript) test_deepg.R

# Or locally / CPU-only test:
Rscript test_deepg.R
```

The test script checks: R package loading, TensorFlow backend, GPU detection,
training data access, data generators, and model creation.

## 8. Training

### Model Architecture

VirusNet uses a BERT-style transformer trained with masked language modeling (MLM)
on viral genome sequences. Key hyperparameters from the existing model:

- **Input**: nucleotide sequences (vocabulary: a, c, g, t)
- **Max sequence length**: 1000 bp
- **Training type**: masked language model (mask_rate=10%, random_rate=5%, identity_rate=5%)
- **Architecture**: transformer (via `deepG::create_model_transformer`)
- **Checkpoint format**: HDF5 (`.hdf5`)

### Running Training

Below is an example SLURM job script (`run_training.sh`). Adapt the paths and
SLURM parameters to your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=virusnet_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=virusnet_train_%j.out
#SBATCH --error=virusnet_train_%j.err

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate virusnet_gpu

# Tell reticulate where Python is
export RETICULATE_PYTHON=$(which python)

# Run training (replace with your actual training script)
Rscript train_virusnet.R \
  --data_dir "$VIRUSNET_DATA" \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

Submit with:

```bash
sbatch run_training.sh
```

### Resuming from a Checkpoint

To continue training from an existing checkpoint, use `deepG::load_cp()`:

```r
model <- deepG::load_cp("/path/to/checkpoint_dir", compile = TRUE)
```

The checkpoint directory should contain an `.hdf5` file (e.g., `Ep.018.hdf5`).

### Data Generator Configuration

The data generator for masked LM training is configured as follows:

```r
gen <- deepG::get_generator(
  path        = file.path(data_dir, "train"),
  train_type  = "masked_lm",
  format      = "fasta",
  batch_size  = 32,
  return_int  = TRUE,
  maxlen      = 1000,
  step        = 1000,
  n_gram      = 1,
  n_gram_stride = 1,
  padding     = TRUE,
  masked_lm   = list(
    mask_rate     = 0.10,
    random_rate   = 0.05,
    identity_rate = 0.05,
    include_sw    = TRUE,
    block_len     = 1
  ),
  ambiguous_nuc = "zero",
  vocabulary    = c("a", "c", "g", "t")
)
```

Adjust `batch_size` based on your GPU memory.

## Troubleshooting

### "Installation of Python not found" / reticulate errors

Set `RETICULATE_PYTHON` before running R:

```bash
export RETICULATE_PYTHON=$(conda run -n virusnet_gpu which python)
```

### No GPUs detected

1. Make sure you requested GPUs in SLURM (`--gres=gpu:1`)
2. Verify the GPU is visible: `nvidia-smi`
3. Check if TensorFlow was built with CUDA:
   ```bash
   python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
   ```
4. If `False`, reinstall TensorFlow with GPU support (see Step 3)

### cuDNN version mismatch

TF 2.12 requires cuDNN 8.6. If you see cuDNN errors:

```bash
conda install -c conda-forge cudnn=8.6.0 cudatoolkit=11.8 -y
```

### h5py version warning

The "h5py running against HDF5 X when built against Y" warning is usually harmless.
To fix: `pip install h5py --no-binary h5py` (slow — compiles from source).

### deepG install fails from GitHub

Check connectivity and install from a local clone instead:

```bash
git clone https://github.com/GenomeNet/deepG.git
Rscript -e 'install.packages("./deepG", repos = NULL, type = "source")'
```

## GPU Memory Guidelines

| GPU         | VRAM   | Recommended batch_size |
|-------------|--------|------------------------|
| NVIDIA T4   | 16 GB  | 8–16                   |
| NVIDIA V100 | 32 GB  | 16–32                  |
| NVIDIA A100 | 80 GB  | 32–64                  |
| NVIDIA H100 | 80 GB  | 32–64                  |

For the full BERT model (~1 GB checkpoint), at least 32 GB VRAM (V100+) is recommended.
