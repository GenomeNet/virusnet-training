test_that("model with custom deepG layers can be saved and reloaded", {
  skip_if_not(reticulate::py_module_available("tensorflow"),
              "TensorFlow not available")

  # Create a small transformer model that uses positional embedding
  # (the same layer type that caused the BERT loading failure)
  model <- deepG::create_model_transformer(
    maxlen              = 50,
    vocabulary_size     = c(6),
    pos_encoding        = "embedding",
    head_size           = c(16),
    num_heads           = c(1),
    dropout             = c(0),
    verbose             = FALSE,
    ff_dim              = c(32),
    learning_rate       = 0.001,
    embed_dim           = 16,
    flatten_method      = "none",
    last_layer_activation = "softmax",
    n                   = 100,
    loss_fn             = "sparse_categorical_crossentropy",
    bal_acc             = FALSE,
    layer_dense         = c(6)
  )

  expect_gt(model$count_params(), 0)

  # Save to a temp .h5 file
  tmp_model <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp_model), add = TRUE)
  keras::save_model_hdf5(model, tmp_model)
  expect_true(file.exists(tmp_model))

  # Reload with deepG::load_cp — this registers custom layers like
  # layer_pos_embedding. Using plain keras::load_model_hdf5 would fail
  # with "ValueError: Unknown layer: 'layer_pos_embedding'"
  reloaded <- deepG::load_cp(tmp_model, compile = FALSE)

  expect_gt(reloaded$count_params(), 0)
  expect_equal(model$count_params(), reloaded$count_params())
})
