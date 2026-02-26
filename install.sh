#!/usr/bin/env bash
# install.sh — Set up the virusnet environment (no sudo required)
#
# Usage:
#   bash install.sh                    # GPU install, env name "virusnet"
#   bash install.sh --cpu              # CPU-only TensorFlow
#   bash install.sh myenv              # GPU install, custom env name
#   bash install.sh --cpu myenv        # CPU-only, custom env name
#
set -euo pipefail

ENV_NAME="virusnet"
CPU_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --cpu) CPU_ONLY=true ;;
    -*) echo "Unknown option: $arg"; exit 1 ;;
    *) ENV_NAME="$arg" ;;
  esac
done

# ── 1. Install Miniforge (mamba) if not available ──────────────────────

if ! command -v mamba &>/dev/null; then
  echo "==> mamba not found, installing Miniforge..."
  MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
  rm /tmp/miniforge.sh

  eval "$("$MINIFORGE_DIR/bin/conda" shell.bash hook)"
  echo ""
  echo "Miniforge installed to $MINIFORGE_DIR"
  echo "To use it in future shells, run:"
  echo "  $MINIFORGE_DIR/bin/mamba init"
  echo ""
else
  echo "==> mamba found: $(command -v mamba)"
  eval "$(conda shell.bash hook)"
fi

# ── 2. Create conda environment with R packages ───────────────────────

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "==> Environment '${ENV_NAME}' already exists, updating..."
else
  echo "==> Creating environment '${ENV_NAME}'..."
  mamba create -n "$ENV_NAME" \
    python=3.10 \
    r-base=4.3 \
    r-essentials \
    r-reticulate \
    r-keras \
    r-tensorflow \
    r-magrittr \
    r-dplyr \
    r-ggplot2 \
    r-hdf5r \
    r-zoo \
    r-optparse \
    r-data.table \
    r-yardstick \
    r-remotes \
    r-testthat \
    r-purrr \
    -c conda-forge -y
fi

conda activate "$ENV_NAME"

# Ensure R uses the conda env's library, not the system one
export R_LIBS_SITE="$CONDA_PREFIX/lib/R/library"
export R_LIBS_USER="$CONDA_PREFIX/lib/R/library"

# ── 3. Install CUDA + TensorFlow ──────────────────────────────────────

if [[ "$CPU_ONLY" == true ]]; then
  echo "==> Installing TensorFlow (CPU-only)..."
  pip install --quiet tensorflow-cpu==2.12.1
else
  echo "==> Installing CUDA toolkit, cuDNN, and TensorFlow (GPU)..."
  mamba install -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y
  pip install --quiet tensorflow==2.12.1
fi

pip install --quiet h5py==3.9.0

# ── 4. Install R packages not on conda-forge ──────────────────────────

echo "==> Installing additional R packages..."
R_LIB="$CONDA_PREFIX/lib/R/library"
Rscript -e "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
lib <- '$R_LIB'
.libPaths(lib)
for (pkg in c('microseq')) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, lib = lib)
  }
}
"

# ── 5. Install deepG ──────────────────────────────────────────────────

echo "==> Installing deepG..."
Rscript -e "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
lib <- '$R_LIB'
.libPaths(lib)
if (!requireNamespace('deepG', quietly = TRUE)) {
  remotes::install_github('GenomeNet/deepG', lib = lib, upgrade = 'never')
}
"

# ── 6. Verify ─────────────────────────────────────────────────────────

echo ""
echo "==> Verifying installation..."
RETICULATE_PYTHON="$(which python)" Rscript -e '
library(reticulate)
library(tensorflow)
library(deepG)
cat("TensorFlow version:", as.character(tf$version$VERSION), "\n")
gpus <- tf$config$list_physical_devices("GPU")
cat("GPUs available:", length(gpus), "\n")
cat("deepG loaded successfully\n")
'

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Environment name: ${ENV_NAME}"
echo ""
echo "  Activate with:"
echo "    mamba activate ${ENV_NAME}"
echo ""
echo "  Run tests with:"
echo "    Rscript test_deepg.R"
echo "============================================"
