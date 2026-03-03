#!/usr/bin/env Rscript
# generate_rds.R — Pre-generate RDS training batches from fasta files
#
# This optional step converts fasta training data into pre-batched .rds files
# for faster training I/O. Use train_virusnet_rds.R to train from these.
#
# The generator samples sequences from fasta files, encodes them as integer
# tokens, and saves ready-to-use (x, y) batches as .rds files. This avoids
# repeated fasta parsing during training.
#
# Prerequisites:
#   Same as train_virusnet.R — install deepG and download datasets.
#
# Usage:
#   Rscript generate_rds.R              # generate to rds_data/
#   VIRUSNET_DATA=/path/to/data Rscript generate_rds.R
#
# This script runs an infinite loop for training data. Kill it (Ctrl+C) once
# you have generated enough batches. Validation data is generated first
# (1000 batches, then stops).
#

library(deepG)
library(keras)

# ── Resolve script directory ─────────────────────────────────────────
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)

# ── Configuration ────────────────────────────────────────────────────

data_dir <- Sys.getenv("VIRUSNET_DATA", file.path(script_dir, "data"))
rds_dir  <- Sys.getenv("VIRUSNET_RDS_DIR", file.path(script_dir, "rds_data"))

base_model_path <- Sys.getenv("VIRUSNET_BASE_MODEL",
                              file.path(script_dir, "models", "llm_1k_bert.h5"))

maxlen     <- 1000
batch_size <- c(30, 3)   # 30 non-virus, 3 virus per batch

rds_train_dir <- file.path(rds_dir, "train")
rds_val_dir   <- file.path(rds_dir, "validation")
dir.create(rds_train_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(rds_val_dir, showWarnings = FALSE, recursive = TRUE)

# ── Data paths ───────────────────────────────────────────────────────

path_non_virus_train <- list(
  file.path(data_dir, "bio_bakery", "train"),
  file.path(data_dir, "non_virus", "train"),
  file.path(data_dir, "additional-archaea-merged", "train")
)
path_non_virus_val <- list(
  file.path(data_dir, "bio_bakery", "validation"),
  file.path(data_dir, "non_virus", "validation"),
  file.path(data_dir, "additional-archaea-merged", "validation")
)
path_virus_train <- file.path(data_dir, "VirusNet_data_virus", "train")
path_virus_val   <- file.path(data_dir, "VirusNet_data_virus", "validation")

path_train <- list(path_non_virus_train, path_virus_train)
path_val   <- list(path_non_virus_val, path_virus_val)

# Verify paths exist
all_dirs <- c(unlist(path_non_virus_train), path_virus_train)
missing <- all_dirs[!dir.exists(all_dirs)]
if (length(missing) > 0) {
  cat("ERROR: Missing data directories:\n")
  for (d in missing) cat("  ", d, "\n")
  stop("Download datasets first: python download.py")
}

# ── Load base model to confirm maxlen ────────────────────────────────
cat("Loading base model to confirm input shape...\n")
base_model <- load_cp(base_model_path, compile = FALSE, mirrored_strategy = FALSE)
model_maxlen <- base_model$input_shape[[2]]
if (model_maxlen != maxlen) {
  cat("WARNING: model input length is", model_maxlen, "but maxlen is set to", maxlen, "\n")
  maxlen <- model_maxlen
}
n_gram <- ifelse(base_model$output_shape[[3]] == 6, 1, 6)
cat("Using maxlen =", maxlen, ", n_gram =", n_gram, "\n")
rm(base_model)

# ── Shared generator parameters ──────────────────────────────────────

gen_params <- list(
  train_type          = "label_folder",
  maxlen              = maxlen,
  batch_size          = batch_size,
  step                = maxlen,
  vocabulary          = c("a", "c", "g", "t"),
  vocabulary_label    = c("non_virus", "virus"),
  shuffle_file_order  = TRUE,
  shuffle_input       = TRUE,
  format              = "fasta",
  output_format       = "target_right",
  ambiguous_nuc       = "zero",
  proportion_per_seq  = 0.99,
  padding             = TRUE,
  max_samples         = c(5, 15),
  return_int          = TRUE,
  n_gram              = n_gram,
  n_gram_stride       = 1,
  reverse_complement  = FALSE,
  reverse_complement_encoding = FALSE,
  read_data           = FALSE,
  use_quality_score   = FALSE,
  random_sampling     = FALSE,
  add_noise           = NULL,
  concat_seq          = NULL,
  target_len          = 1,
  sample_by_file_size = FALSE
)

# ── Generate validation data (finite) ────────────────────────────────
cat("\n=== Generating validation RDS batches ===\n")

val_params <- gen_params
val_params$path <- path_val
val_params$seed <- c(7234)

gen_val <- do.call(get_generator, val_params)

n_val_batches <- 1000
for (i in seq_len(n_val_batches)) {
  z <- gen_val()
  rds_file <- file.path(rds_val_dir, sprintf("batch_%05d.rds", i))
  saveRDS(z, rds_file)
  if (i %% 100 == 0) cat("  Validation batch", i, "/", n_val_batches, "\n")
}
cat("Validation data: ", n_val_batches, "batches in", rds_val_dir, "\n")

# ── Generate training data (infinite loop) ───────────────────────────
cat("\n=== Generating training RDS batches (runs until killed) ===\n")
cat("Press Ctrl+C to stop once you have enough batches.\n\n")

train_params <- gen_params
train_params$path <- path_train
train_params$seed <- sample(50:5000, 2)

gen_train <- do.call(get_generator, train_params)

batch_num <- 0
while (TRUE) {
  # Generate 30 batches per loop iteration
  for (j in 1:30) {
    batch_num <- batch_num + 1
    z <- gen_train()
    rds_file <- file.path(rds_train_dir, sprintf("batch_%06d.rds", batch_num))
    saveRDS(z, rds_file)
  }
  cat("  Training batches generated:", batch_num, "\n")
}
