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
bash install.sh          # GPU (installs CUDA 11.8 + cuDNN 8.6 + tensorflow 2.12)
bash install.sh --cpu    # CPU-only tensorflow (for testing without GPU)
```

Verify:

```bash
mamba activate virusnet
$CONDA_PREFIX/bin/Rscript test_deepg.R
```

## Testing on SLURM

Quick smoke test (CPU, no GPU needed, ~2 min):

```bash
sbatch tests/test_quick.slurm
```

Full GPU test (loads base BERT model, checks GPU, ~10 min):

```bash
python download_model.py model    # download base model first
sbatch tests/test_gpu.slurm

# with training data:
sbatch --export=ALL,VIRUSNET_DATA=/path/to/data tests/test_gpu.slurm
```

## Base Model

The base model (`llm_1k_bert.h5`) is a BERT-like masked language model we
pre-trained on genomic sequences using deepG. It serves as the starting point
for fine-tuning the binary VirusNet classifier.

Download the pre-trained weights:

```bash
python download_model.py model      # saves to models/llm_1k_bert.h5
```

## Training Data

Training datasets for the binary VirusNet classifier:

```bash
python download_model.py --list     # show all available assets with sizes
python download_model.py            # download everything (~135 GB total)
```

Individual datasets:

| Asset | Description | Size |
|-------|-------------|------|
| `archaea` | Additional archaea sequences | 190 MB |
| `non-virus` | Non-virus sequences for binary classification | 3.7 GB |
| `virusnet-sim` | Subsampled simulated training data | 6.1 GB |
| `bio-bakery` | BioBakery reference data | 125 GB |

```bash
python download_model.py archaea non-virus   # download specific datasets
```

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


## Model Architecture and Input Encoding

### Pre-trained BERT model (`llm_1k_bert.h5`)

The base model is a BERT-style transformer trained with **masked language
modeling (MLM)** on virus genome sequences. The model learns to predict
randomly masked nucleotides, building general representations of DNA sequence
patterns.

Architecture (from the saved model config):

| Parameter | Value |
|-----------|-------|
| Input shape | `(batch, 1000)` |
| Encoding | **1-gram integer tokens** |
| Vocabulary size | 6 tokens: `0`=padding/N, `1`=A, `2`=C, `3`=G, `4`=T, `5`=MASK |
| Embedding dim | 650 |
| Transformer blocks | 18 |
| Attention heads | 20 per block |
| Head size | 200 |
| Feed-forward dim | 1950 |
| Dropout | 0.1 |
| Output | Dense(256, relu) -> Dense(6, softmax) -- predicts masked nucleotide |
| Parameters | ~2.8 GB on disk |

A second pre-trained BERT checkpoint (`bert_1k_3.h5`) uses a similar
architecture but with 24 transformer blocks, 16 heads, head_size=250,
embed_dim=600, ff_dim=2400, and no dropout.

### Input encoding explained

The BERT models use **1-gram integer encoding** (`n_gram = 1`,
`return_int = TRUE` in deepG). Each nucleotide in the input sequence is mapped
to a single integer:

```
DNA sequence:   A  C  G  T  A  C  N  G  T  ...
Integer tokens: 1  2  3  4  1  2  0  3  4  ...
```

The embedding layer inside the transformer then maps each integer to a dense
vector of dimension 650 (or 600 for `bert_1k_3.h5`). This is conceptually
similar to word embeddings in NLP -- each nucleotide "token" gets a learned
vector representation.

The vocabulary has 6 values because:
`4 nucleotides + 1 padding/ambiguous + 1 mask token = 6`.

You can verify this from any loaded model using the auto-detection logic
in the training scripts:

```r
# output_shape[[3]] == 6 means 1-gram (4^1 + 2 = 6)
# output_shape[[3]] == 4098 would mean 6-gram (4^6 + 2 = 4098)
n_gram <- ifelse(model$output_shape[[3]] == 6, 1, 6)
```

### Fine-tuning for binary classification (virus vs. non-virus)

The binary VirusNet classifier is built by:

1. Loading the pre-trained BERT (`llm_1k_bert.h5`)
2. Removing the MLM output head (the last Dense layers)
3. Adding new classification layers
   (e.g. `Dense(1024) -> Dense(128) -> Dense(2, softmax)`)
4. Freezing the transformer base, training only the new layers
5. Optionally unfreezing and fine-tuning the full model at a lower learning rate

The input encoding stays the same as the base BERT: 1-gram integer tokens with
`maxlen = 1000`. The fine-tuned model input shape is `(batch, 1000)`.

The relevant deepG generator parameters for this model:

```r
get_generator(
  train_type    = "label_folder",
  maxlen        = 1000,
  vocabulary    = c("a", "c", "g", "t"),
  return_int    = TRUE,    # integer encoding, not one-hot
  n_gram        = 1,       # single nucleotide tokens
  n_gram_stride = 1
)
```

### Comparison with the CNN/LSTM genus model

A separate CNN/LSTM model (`create_model_lstm_cnn` in deepG) was trained for
genus-level virus classification. This model uses a **completely different**
input encoding:

| | Binary classifier (BERT) | Genus classifier (CNN/LSTM) |
|---|---|---|
| Architecture | Transformer (18-24 blocks) | CNN + optional LSTM |
| Input shape | `(batch, 1000)` | `(batch, 2000, 4)` |
| Encoding | 1-gram integer tokens | One-hot encoded |
| `return_int` | `TRUE` | `FALSE` |
| Sequence length | 1000 bp | 2000 bp |
| Each position | 1 integer (0-5) | 4-element vector `[1,0,0,0]` |
| Pre-training | MLM on virus genomes | None (trained from scratch) |

The CNN model takes one-hot encoded sequences directly into convolutional
layers, while the BERT model uses integer tokens that pass through a learned
embedding layer before the transformer blocks.
