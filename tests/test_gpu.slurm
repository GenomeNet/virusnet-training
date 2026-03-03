#!/bin/bash
#SBATCH --job-name=virusnet-gpu-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=tests/test_gpu_%j.out
#SBATCH --error=tests/test_gpu_%j.err
#
# Full GPU test — loads base BERT model with deepG, checks GPU visibility,
# runs all test sections including model summary.
#
# Prerequisites:
#   python download.py model   # downloads models/llm_1k_bert.h5
#
# Usage:
#   cd /vol/hpcprojects/BIFO/genomenet/virusnet-training
#   sbatch tests/test_gpu.slurm
#
#   # With training data:
#   sbatch --export=ALL,VIRUSNET_DATA=/path/to/data tests/test_gpu.slurm
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-virusnet}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Full GPU test ==="
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "  Job:  ${SLURM_JOB_ID:-local}"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo ""
fi

# ── Activate conda environment ───────────────────────────────────────
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

conda activate "$ENV_NAME" || { echo "ERROR: env '$ENV_NAME' not found. Run install.sh first." >&2; exit 1; }
export RETICULATE_PYTHON="$(which python)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ── Point to the base BERT model ─────────────────────────────────────
if [[ -z "${VIRUSNET_BASE_MODEL:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/models/llm_1k_bert.h5" ]]; then
    export VIRUSNET_BASE_MODEL="$SCRIPT_DIR/models/llm_1k_bert.h5"
  fi
fi

echo "  Conda env:  $CONDA_PREFIX"
echo "  Python:     $RETICULATE_PYTHON"
echo "  Base model: ${VIRUSNET_BASE_MODEL:-not set}"
echo "  Data:       ${VIRUSNET_DATA:-not set}"
echo ""

# ── Run full test suite ───────────────────────────────────────────────
"$CONDA_PREFIX/bin/Rscript" "$SCRIPT_DIR/test_deepg.R"

echo ""
echo "=== GPU test finished ==="
