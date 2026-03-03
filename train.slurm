#!/bin/bash
#SBATCH --job-name=virusnet-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Fine-tune the BERT base model for virus vs non-virus classification.
#
# Prerequisites:
#   1. bash install.sh
#   2. python download.py model archaea non-virus virus
#
# Usage (from the virusnet-training directory):
#   mkdir -p logs
#   sbatch train.slurm
#
# With custom data path:
#   sbatch --export=ALL,VIRUSNET_DATA=/path/to/data train.slurm
#
# RDS-based training (faster, requires generate_rds.R first):
#   sbatch --export=ALL,VIRUSNET_TRAIN_SCRIPT=train_virusnet_rds.R train.slurm
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-virusnet}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${VIRUSNET_TRAIN_SCRIPT:-train_virusnet.R}"

echo "=== VirusNet Training ==="
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-?}"
echo "  Script:     $TRAIN_SCRIPT"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  echo "--- GPU ---"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo ""
fi

# ── Activate conda ───────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
  if [[ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
    source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Run install.sh first." >&2
    exit 1
  fi
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate "$ENV_NAME" || { echo "ERROR: env '$ENV_NAME' not found." >&2; exit 1; }
export RETICULATE_PYTHON="$(which python)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "  Conda env:  $CONDA_PREFIX"
echo "  Python:     $RETICULATE_PYTHON"
echo ""

# ── Create output dirs ───────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/checkpoints" "$SCRIPT_DIR/logs"

# ── Run training ─────────────────────────────────────────────────────
echo "Starting: $TRAIN_SCRIPT"
echo "=========================================="

"$CONDA_PREFIX/bin/Rscript" "$SCRIPT_DIR/$TRAIN_SCRIPT"

echo ""
echo "=== Training finished ==="
echo "  $(date)"
