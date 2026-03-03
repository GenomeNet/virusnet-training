#!/bin/bash
#SBATCH --job-name=virusnet-quick
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=tests/test_quick_%j.out
#SBATCH --error=tests/test_quick_%j.err
#
# Quick smoke test — CPU only, no GPU required.
# Validates: R packages, TensorFlow backend, small model creation.
# Skips: GPU, training data, base BERT model (env vars not set).
#
# Usage:
#   cd /vol/hpcprojects/BIFO/genomenet/virusnet-training
#   sbatch tests/test_quick.slurm
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-virusnet}"

echo "=== Quick smoke test (CPU) ==="
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "  Job:  ${SLURM_JOB_ID:-local}"
echo ""

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

"$CONDA_PREFIX/bin/Rscript" test_deepg.R

echo ""
echo "=== Quick test finished ==="
