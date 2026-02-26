test_that("VirusNet transformer model can be created", {
  skip_if_not(reticulate::py_module_available("tensorflow"),
              "TensorFlow not available")

  model <- deepG::create_model_transformer(
    maxlen              = 100,
    vocabulary_size     = c(6),
    pos_encoding        = "embedding",
    head_size           = c(32),
    num_heads           = c(2),
    dropout             = c(0),
    verbose             = FALSE,
    ff_dim              = c(64),
    learning_rate       = 0.0001,
    embed_dim           = 32,
    flatten_method      = "none",
    last_layer_activation = "softmax",
    n                   = 10000,
    loss_fn             = "sparse_categorical_crossentropy",
    bal_acc             = FALSE,
    layer_dense         = c(6)
  )

  expect_true(inherits(model, "keras.engine.training.Model") ||
              inherits(model, "keras.src.models.model.Model") ||
              !is.null(model$count_params))
  expect_gt(model$count_params(), 0)
})
