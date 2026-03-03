#!/usr/bin/env Rscript
# test_deepg.R — Smoke-test for deepG + TensorFlow environment
#
# Usage:
#   # Set these before running:
#   export RETICULATE_PYTHON=$(conda run -n virusnet_gpu which python)
#   export VIRUSNET_DATA=/path/to/training_data   # dir with train/, validation/, test/
#
#   # On a SLURM cluster with GPUs:
#   srun --partition=gpu --gres=gpu:1 --time=10 Rscript test_deepg.R
#
#   # Or locally / CPU-only:
#   Rscript test_deepg.R

# ── 0. Configuration ─────────────────────────────────────────────────────

# Check that we're using the conda env's R, not the system one
r_home <- R.home()
conda_prefix <- Sys.getenv("CONDA_PREFIX")
if (conda_prefix != "" && !startsWith(r_home, conda_prefix)) {
  cat("WARNING: Using system R at", r_home, "\n")
  cat("  Expected conda env R from", conda_prefix, "\n")
  cat("  Run with the conda env's Rscript:\n")
  cat("    $CONDA_PREFIX/bin/Rscript test_deepg.R\n\n")
}

# RETICULATE_PYTHON should be set as an env var before running this script.
# If not set, reticulate will try to auto-detect Python (may pick the wrong one).
if (Sys.getenv("RETICULATE_PYTHON") == "") {
  cat("WARNING: RETICULATE_PYTHON is not set.\n")
  cat("  Set it to your conda env Python, e.g.:\n")
  cat("  export RETICULATE_PYTHON=$(which python)\n\n")
}

# Training data path — set via VIRUSNET_DATA env var or change this default
data_base <- Sys.getenv("VIRUSNET_DATA", unset = "")
if (data_base == "") {
  cat("WARNING: VIRUSNET_DATA is not set. Skipping data and generator tests.\n")
  cat("  Set it to the directory containing train/, validation/, test/ subdirs.\n\n")
}

passed <- 0
failed <- 0

check <- function(label, expr) {
  cat(sprintf("%-50s", paste0("[TEST] ", label, " ... ")))
  tryCatch({
    result <- eval(expr)
    if (isTRUE(result) || !is.logical(result)) {
      cat("PASS\n")
      passed <<- passed + 1
    } else {
      cat("FAIL\n")
      failed <<- failed + 1
    }
  }, error = function(e) {
    cat(paste0("FAIL (", conditionMessage(e), ")\n"))
    failed <<- failed + 1
  })
}

# ── 1. Library loading ───────────────────────────────────────────────────
cat("\n=== 1. Loading R packages ===\n")
check("library(reticulate)", library(reticulate))
check("library(tensorflow)", library(tensorflow))
check("library(keras)",      library(keras))
check("library(magrittr)",   library(magrittr))
check("library(deepG)",      library(deepG))

# ── 2. TensorFlow backend ───────────────────────────────────────────────
cat("\n=== 2. TensorFlow backend ===\n")
check("tf$constant works", {
  x <- tensorflow::tf$constant(42.0)
  as.numeric(x) == 42.0
})

tf_version <- tryCatch(as.character(tensorflow::tf$version$VERSION), error = function(e) "unknown")
cat("  TensorFlow version:", tf_version, "\n")

# ── 3. GPU detection ────────────────────────────────────────────────────
cat("\n=== 3. GPU detection ===\n")
gpus <- tryCatch(tensorflow::tf$config$list_physical_devices("GPU"), error = function(e) list())
n_gpus <- length(gpus)
cat("  GPUs found:", n_gpus, "\n")
if (n_gpus > 0) {
  for (i in seq_along(gpus)) {
    cat("  GPU", i, ":", as.character(gpus[[i]]$name), "\n")
  }
  check("At least 1 GPU available", n_gpus > 0)
} else {
  cat("  WARNING: No GPUs detected. TensorFlow may be CPU-only build.\n")
  cat("  GPU training will NOT work with this setup.\n")
  cat("  Make sure you requested a GPU (--gres=gpu:1) and that TF has CUDA support.\n")
}

cuda_built <- tryCatch(tensorflow::tf$test$is_built_with_cuda(), error = function(e) FALSE)
cat("  TF built with CUDA:", cuda_built, "\n")

# ── 4. Training data paths ──────────────────────────────────────────────
cat("\n=== 4. Training data ===\n")
if (data_base != "") {
  check("Train dir exists",      dir.exists(file.path(data_base, "train")))
  check("Validation dir exists", dir.exists(file.path(data_base, "validation")))
  check("Test dir exists",       dir.exists(file.path(data_base, "test")))

  n_train <- length(list.files(file.path(data_base, "train")))
  n_val   <- length(list.files(file.path(data_base, "validation")))
  n_test  <- length(list.files(file.path(data_base, "test")))
  cat("  Train files:", n_train, " | Validation files:", n_val, " | Test files:", n_test, "\n")

  # Read first few lines of a training fasta
  train_files <- list.files(file.path(data_base, "train"), full.names = TRUE)
  if (length(train_files) > 0) {
    first_lines <- readLines(train_files[1], n = 4, warn = FALSE)
    cat("  First training file:", basename(train_files[1]), "\n")
    cat("  Header:", first_lines[1], "\n")
    cat("  Seq length:", nchar(first_lines[2]), "bp (first line)\n")
  }
} else {
  cat("  SKIPPED (VIRUSNET_DATA not set)\n")
}

# ── 5. Base BERT model loading ────────────────────────────────────────
cat("\n=== 5. Base BERT model ===\n")
# Default: models/llm_1k_bert.h5 next to this script; override with VIRUSNET_BASE_MODEL
script_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
default_model_path <- file.path(script_dir, "models", "llm_1k_bert.h5")
base_model_path <- Sys.getenv("VIRUSNET_BASE_MODEL", unset = default_model_path)

if (file.exists(base_model_path)) {
  cat("  Model file:", base_model_path, "\n")
  cat("  Size:", round(file.info(base_model_path)$size / 1e6, 1), "MB\n")
  check("load base BERT model", {
    base_model <- keras::load_model_hdf5(base_model_path, compile = FALSE)
    cat("  Input shape:", paste(unlist(base_model$input_shape), collapse = " x "), "\n")
    cat("  Output shape:", paste(unlist(base_model$output_shape), collapse = " x "), "\n")

    total_params     <- base_model$count_params()
    trainable_params <- sum(sapply(base_model$trainable_weights, function(w) prod(w$shape$as_list())))
    frozen_params    <- total_params - trainable_params
    cat("  Total parameters:     ", format(total_params, big.mark = ","), "\n")
    cat("  Trainable parameters: ", format(trainable_params, big.mark = ","), "\n")
    cat("  Frozen parameters:    ", format(frozen_params, big.mark = ","), "\n")

    n_layers <- length(base_model$layers)
    cat("  Number of layers:", n_layers, "\n")

    # Layer-type breakdown
    layer_types <- sapply(base_model$layers, function(l) class(l)[1])
    type_counts <- sort(table(layer_types), decreasing = TRUE)
    cat("  Layer types:\n")
    for (i in seq_along(type_counts)) {
      cat(sprintf("    %-30s %d\n", names(type_counts)[i], type_counts[i]))
    }

    # Full model summary to stdout
    cat("\n  --- Model summary ---\n")
    base_model$summary()
    cat("  ---------------------\n")

    TRUE
  })
} else {
  cat("  SKIPPED (model file not found)\n")
  cat("  Download with:  python download_model.py\n")
  cat("  Or set VIRUSNET_BASE_MODEL=/path/to/llm_1k_bert.h5\n")
}

# ── 6. Checkpoint loading ─────────────────────────────────────────────
cat("\n=== 6. Checkpoint loading ===\n")
cp_dir <- Sys.getenv("VIRUSNET_CHECKPOINT", unset = "")
if (cp_dir != "" && dir.exists(cp_dir)) {
  hdf5_files <- list.files(cp_dir, pattern = "\\.hdf5$", full.names = TRUE)
  if (length(hdf5_files) > 0) {
    cat("  Found checkpoint(s):", paste(basename(hdf5_files), collapse = ", "), "\n")
    check("load_cp works", {
      model <- deepG::load_cp(cp_dir, compile = TRUE)
      cat("  Model input shape:", paste(unlist(model$input_shape), collapse = " x "), "\n")
      cat("  Model output shape:", paste(unlist(model$output_shape), collapse = " x "), "\n")
      cat("  Model parameters:", format(model$count_params(), big.mark = ","), "\n")
      TRUE
    })
  } else {
    cat("  No .hdf5 files found in", cp_dir, "\n")
  }
} else {
  cat("  SKIPPED (VIRUSNET_CHECKPOINT not set or dir does not exist)\n")
  cat("  To test checkpoint loading, set:\n")
  cat("  export VIRUSNET_CHECKPOINT=/path/to/checkpoint_dir\n")
}

# ── 7. Generator test ───────────────────────────────────────────────────
cat("\n=== 7. Data generator test ===\n")
if (data_base != "" && dir.exists(file.path(data_base, "validation"))) {
  check("get_generator works", {
    gen <- deepG::get_generator(
      path        = file.path(data_base, "validation"),
      train_type  = "masked_lm",
      format      = "fasta",
      batch_size  = 4,
      return_int  = TRUE,
      maxlen      = 1000,
      step        = 1000,
      n_gram      = 1,
      n_gram_stride = 1,
      padding     = TRUE,
      masked_lm   = list(mask_rate = 0.10, random_rate = 0.05,
                          identity_rate = 0.05, include_sw = TRUE, block_len = 1),
      max_samples = 4,
      ambiguous_nuc = "zero",
      vocabulary  = c("a", "c", "g", "t")
    )
    z <- gen()
    x <- z[[1]]
    y <- z[[2]]
    sw <- z[[3]]
    cat("  x shape:", paste(dim(x), collapse = " x "), "\n")
    cat("  y shape:", paste(dim(y), collapse = " x "), "\n")
    cat("  sw shape:", paste(dim(sw), collapse = " x "), "\n")
    TRUE
  })
} else {
  cat("  SKIPPED (VIRUSNET_DATA not set or validation/ not found)\n")
}

# ── 8. Model creation test ──────────────────────────────────────────────
cat("\n=== 8. Model creation test (small) ===\n")
check("create_model_transformer works", {
  small_model <- deepG::create_model_transformer(
    maxlen          = 100,
    vocabulary_size = c(6),
    pos_encoding    = "embedding",
    head_size       = c(32),
    num_heads       = c(2),
    dropout         = c(0),
    verbose         = FALSE,
    ff_dim          = c(64),
    learning_rate   = 0.0001,
    embed_dim       = 32,
    flatten_method  = "none",
    last_layer_activation = "softmax",
    n               = 10000,
    loss_fn         = "sparse_categorical_crossentropy",
    bal_acc         = FALSE,
    layer_dense     = c(6)
  )
  cat("  Small test model created successfully\n")
  cat("  Parameters:", format(small_model$count_params(), big.mark = ","), "\n")
  TRUE
})

# ── Summary ──────────────────────────────────────────────────────────────
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat(sprintf("SUMMARY: %d passed, %d failed\n", passed, failed))
if (failed == 0) {
  cat("All tests passed! Environment is ready.\n")
} else {
  cat("Some tests failed. Check output above for details.\n")
}
cat(paste(rep("=", 60), collapse = ""), "\n\n")
