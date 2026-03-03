#!/usr/bin/env Rscript
# train_virusnet.R — Fine-tune BERT base model for virus vs non-virus classification
#
# This script loads a pre-trained BERT model (llm_1k_bert.h5) and fine-tunes it
# as a binary classifier: virus vs non-virus. Training reads fasta files from
# the data/ directory (label_folder mode in deepG).
#
# Prerequisites:
#   1. Install environment:  bash install.sh
#   2. Download base model:  python download.py model
#   3. Download datasets:    python download.py archaea non-virus virus
#      (optionally also: python download.py virusnet-sim bio-bakery)
#
# Usage:
#   Rscript train_virusnet.R
#
# Environment variables (all optional):
#   VIRUSNET_BASE_MODEL  path to base BERT model (default: models/llm_1k_bert.h5)
#   VIRUSNET_DATA        path to data directory (default: data/)
#   VIRUSNET_CHECKPOINTS path for saving checkpoints (default: checkpoints/)
#   VIRUSNET_LOGS        path for training logs (default: logs/)
#   VIRUSNET_RUN_NAME    run name for this training (default: virusnet_bert_finetune_1)
#

library(deepG)
library(keras)

# ── Resolve script directory ─────────────────────────────────────────
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)

# ── Configuration (edit these or set env vars) ────────────────────────

# Base model
base_model_path <- Sys.getenv("VIRUSNET_BASE_MODEL",
                              file.path(script_dir, "models", "llm_1k_bert.h5"))

# Data directory (expects subdirs: bio_bakery/, non_virus/,
# additional-archaea-merged/, VirusNet_data_virus/, each with train/ and
# validation/ containing .fasta files)
data_dir <- Sys.getenv("VIRUSNET_DATA",
                       file.path(script_dir, "data"))

# Output paths
checkpoint_dir <- Sys.getenv("VIRUSNET_CHECKPOINTS",
                             file.path(script_dir, "checkpoints"))
log_dir <- Sys.getenv("VIRUSNET_LOGS",
                      file.path(script_dir, "logs"))

dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

# ── Hyperparameters ──────────────────────────────────────────────────

run_name        <- Sys.getenv("VIRUSNET_RUN_NAME", "virusnet_bert_finetune_1")
learning_rate   <- 5e-08       # very low LR since all layers are trainable
batch_size      <- c(30, 3)    # 30 non-virus, 3 virus samples per batch
maxlen          <- 1000        # sequence length (must match base model)
steps_per_epoch <- 50000
epochs          <- 10000
dense_layers    <- c(128, 2)   # classification head
freeze_base     <- FALSE       # FALSE = fine-tune all layers
focal_gamma     <- 2.5
focal_alpha     <- c(0.5, 5)   # upweight virus class (minority)
max_samples     <- c(450, 225) # max samples read per file (non-virus, virus)
seed            <- sample(50:5000, 2)

cat("Run name:      ", run_name, "\n")
cat("Base model:    ", base_model_path, "\n")
cat("Data dir:      ", data_dir, "\n")
cat("Checkpoint dir:", checkpoint_dir, "\n")
cat("Learning rate: ", learning_rate, "\n")
cat("Batch size:    ", paste(batch_size, collapse = ", "), "\n")
cat("Seed:          ", paste(seed, collapse = ", "), "\n")

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
all_train_dirs <- c(unlist(path_non_virus_train), path_virus_train)
missing <- all_train_dirs[!dir.exists(all_train_dirs)]
if (length(missing) > 0) {
  cat("\nWARNING: Missing training directories:\n")
  for (d in missing) cat("  ", d, "\n")
  cat("\nDownload datasets first: python download.py\n")
  cat("Or set VIRUSNET_DATA to point to your data directory.\n\n")
  stop("Missing training data directories.")
}

# ── Focal loss ───────────────────────────────────────────────────────
# Addresses class imbalance: non-virus sequences are far more abundant than
# virus sequences. gamma focuses training on hard examples, alpha upweights
# the minority (virus) class.

focal_loss_multiclass <- function(y_true, y_pred,
                                  gamma = focal_gamma,
                                  alpha = focal_alpha) {
  y_pred <- keras::k_clip(y_pred, keras::k_epsilon(), 1.0 - keras::k_epsilon())
  cd_loss <- -y_true * keras::k_log(y_pred)
  fl_loss <- alpha * keras::k_pow(1.0 - y_pred, gamma) * cd_loss
  return(keras::k_sum(fl_loss, axis = -1))
}

# ── Build model ──────────────────────────────────────────────────────
cat("\nLoading base model:", base_model_path, "\n")

mirrored_strategy <- tensorflow::tf$distribute$MirroredStrategy()
with(mirrored_strategy$scope(), {

  base_model <- load_cp(base_model_path, compile = FALSE, mirrored_strategy = FALSE)

  # Auto-detect n_gram from model output
  n_gram <- ifelse(base_model$output_shape[[3]] == 6, 1, 6)
  cat("Auto-detected n_gram:", n_gram, "\n")

  # Determine where to cut: remove the final dense layers from the pre-trained
  # model, keeping all transformer blocks intact
  num_layers <- length(base_model$get_config()$layers)
  layer_name <- base_model$get_config()$layers[[num_layers - 3]]$name
  cat("Cutting at layer:", layer_name, "\n")

  # Swap the MLM head for a binary classification head
  model <- remove_add_layers(
    model              = base_model,
    layer_name         = layer_name,
    dense_layers       = dense_layers,
    last_activation    = list("softmax"),
    output_names       = NULL,
    losses             = list("categorical_crossentropy"),
    verbose            = FALSE,
    dropout            = NULL,
    freeze_base_model  = freeze_base,
    compile            = FALSE,
    learning_rate      = learning_rate,
    flatten            = TRUE,
    solver             = "adam",
    mirrored_strategy  = FALSE,
    model_seed         = 3
  )

  # Compile with focal loss
  keras_optimizer <- keras::optimizer_adam(learning_rate = learning_rate)
  model %>% keras::compile(
    loss      = focal_loss_multiclass,
    optimizer = keras_optimizer,
    metrics   = "acc"
  )
})

cat("\n")
print(model)

# ── Sanity check ─────────────────────────────────────────────────────
cat("\nRunning sanity check on dummy data...\n")
x <- array(0L, dim = c(10, maxlen))
y <- matrix(0, ncol = 2, nrow = 10)
y[1:9, 1] <- 1
y[10, 2]  <- 1
model$evaluate(x = x, y = y, verbose = 0)
cat("Sanity check passed.\n\n")

# ── Train ────────────────────────────────────────────────────────────
cat("Starting training: ", run_name, "\n")

train_model(
  train_type          = "label_folder",
  model               = model,
  path                = path_train,
  path_val            = path_val,
  path_checkpoint     = checkpoint_dir,
  path_log            = log_dir,
  run_name            = run_name,
  batch_size          = batch_size,
  epochs              = epochs,
  steps_per_epoch     = steps_per_epoch,
  step                = maxlen,
  train_val_ratio     = 0.2,
  reduce_lr_on_plateau = TRUE,
  lr_plateau_factor   = 0.9,
  patience            = 18,
  cooldown            = 14,
  shuffle_file_order  = TRUE,
  vocabulary          = c("a", "c", "g", "t"),
  vocabulary_label    = c("non_virus", "virus"),
  save_best_only      = TRUE,
  save_weights_only   = FALSE,
  seed                = seed,
  shuffle_input       = TRUE,
  format              = "fasta",
  output_format       = "target_right",
  ambiguous_nuc       = "zero",
  proportion_per_seq  = 0.99,
  padding             = TRUE,
  max_samples         = max_samples,
  class_weight        = NULL,
  concat_seq          = NULL,
  target_len          = 1,
  print_scores        = TRUE,
  reverse_complement  = FALSE,
  reverse_complement_encoding = FALSE,
  read_data           = FALSE,
  use_quality_score   = FALSE,
  random_sampling     = FALSE,
  max_queue_size      = 100,
  n_gram              = NULL,
  n_gram_stride       = 1,
  return_int          = FALSE,
  add_noise           = NULL
)
