#!/usr/bin/env Rscript
# train_virusnet_rds.R — Fine-tune BERT model using pre-generated RDS batches
#
# This is an alternative to train_virusnet.R that uses pre-generated RDS data
# instead of reading fasta files on the fly. RDS-based training is significantly
# faster because it skips fasta parsing during training.
#
# Prerequisites:
#   1. Install environment:  bash install.sh
#   2. Download base model:  python download.py model
#   3. Download datasets:    python download.py archaea non-virus virus
#   4. Generate RDS data:    Rscript generate_rds.R
#
# Usage:
#   Rscript train_virusnet_rds.R
#
# Environment variables (all optional):
#   VIRUSNET_BASE_MODEL  path to base BERT model (default: models/llm_1k_bert.h5)
#   VIRUSNET_RDS_DIR     path to RDS data dir (default: rds_data/)
#   VIRUSNET_CHECKPOINTS path for saving checkpoints (default: checkpoints/)
#   VIRUSNET_LOGS        path for training logs (default: logs/)
#   VIRUSNET_RUN_NAME    run name (default: virusnet_bert_rds_1)
#

library(deepG)
library(keras)

# ── Resolve script directory ─────────────────────────────────────────
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)

# ── Configuration ────────────────────────────────────────────────────

base_model_path <- Sys.getenv("VIRUSNET_BASE_MODEL",
                              file.path(script_dir, "models", "llm_1k_bert.h5"))

rds_dir <- Sys.getenv("VIRUSNET_RDS_DIR",
                      file.path(script_dir, "rds_data"))

checkpoint_dir <- Sys.getenv("VIRUSNET_CHECKPOINTS",
                             file.path(script_dir, "checkpoints"))
log_dir <- Sys.getenv("VIRUSNET_LOGS",
                      file.path(script_dir, "logs"))

dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

path_train <- file.path(rds_dir, "train")
path_val   <- file.path(rds_dir, "validation")

if (!dir.exists(path_train) || !dir.exists(path_val)) {
  stop("RDS data not found at ", rds_dir,
       "\nGenerate it first: Rscript generate_rds.R")
}

# ── Hyperparameters ──────────────────────────────────────────────────
# RDS training uses frozen base + selective unfreezing. This is more
# parameter-efficient: only the last transformer block and the classification
# head are trained. A higher learning rate is possible since most parameters
# are frozen.

run_name         <- Sys.getenv("VIRUSNET_RUN_NAME", "virusnet_bert_rds_1")
learning_rate    <- 5e-05       # higher LR possible with frozen base
batch_size       <- 33
maxlen           <- 1000
steps_per_epoch  <- 10000
epochs           <- 10000
dense_layers     <- c(64, 2)    # smaller head (base is frozen, less overfitting)
freeze_base      <- TRUE
focal_gamma      <- 2.0
focal_alpha      <- c(0.5, 5)
delete_used_files <- TRUE       # delete RDS files after reading (save disk)
seed             <- sample(50:5000, 2)

cat("Run name:       ", run_name, "\n")
cat("Base model:     ", base_model_path, "\n")
cat("RDS data dir:   ", rds_dir, "\n")
cat("Checkpoint dir: ", checkpoint_dir, "\n")
cat("Learning rate:  ", learning_rate, "\n")
cat("Batch size:     ", batch_size, "\n")
cat("Seed:           ", paste(seed, collapse = ", "), "\n")

# ── Focal loss ───────────────────────────────────────────────────────

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
  n_gram <- ifelse(base_model$output_shape[[3]] == 6, 1, 6)
  cat("Auto-detected n_gram:", n_gram, "\n")

  num_layers <- length(base_model$get_config()$layers)
  layer_name <- base_model$get_config()$layers[[num_layers - 4]]$name
  cat("Cutting at layer:", layer_name, "\n")

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
    global_pooling     = "max_ch_last",
    solver             = "adam",
    mirrored_strategy  = FALSE,
    model_seed         = 3
  )

  # Selectively unfreeze the last transformer block for fine-tuning.
  # This gives the model some ability to adapt its representations while
  # keeping most parameters frozen for stability.
  w <- model$get_layer("layer_transformer_block_20")
  w$trainable <- TRUE
  cat("Unfroze: layer_transformer_block_20\n")

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
cat("\nSanity check on dummy data...\n")
x <- array(0L, dim = c(10, maxlen))
y <- matrix(0, ncol = 2, nrow = 10)
y[1:9, 1] <- 1
y[10, 2]  <- 1
model$evaluate(x = x, y = y, verbose = 0)
cat("Sanity check passed.\n\n")

# ── Train ────────────────────────────────────────────────────────────
cat("Starting RDS-based training:", run_name, "\n")

train_model(
  train_type           = "label_rds",
  model                = model,
  path                 = path_train,
  path_val             = path_val,
  path_checkpoint      = checkpoint_dir,
  path_log             = log_dir,
  run_name             = run_name,
  batch_size           = batch_size,
  epochs               = epochs,
  steps_per_epoch      = steps_per_epoch,
  step                 = maxlen,
  train_val_ratio      = 0.1,
  reduce_lr_on_plateau = TRUE,
  lr_plateau_factor    = 0.9,
  patience             = 50,
  cooldown             = 50,
  shuffle_file_order   = TRUE,
  save_best_only       = NULL,
  seed                 = seed,
  format               = "rds",
  delete_used_files    = delete_used_files,
  vocabulary_label     = c("non_virus", "virus"),
  print_scores         = TRUE
)
