test_that("masked LM generator produces correct output shapes", {
  skip_if_not(reticulate::py_module_available("tensorflow"),
              "TensorFlow not available")

  # Create minimal test FASTA data
  test_dir <- tempdir()
  fasta_dir <- file.path(test_dir, "test_fasta")
  dir.create(fasta_dir, showWarnings = FALSE)

  # Write a small synthetic FASTA file
  seq <- paste(sample(c("a", "c", "g", "t"), 5000, replace = TRUE), collapse = "")
  writeLines(c(">test_sequence_1", seq), file.path(fasta_dir, "test.fasta"))

  batch_size <- 4
  maxlen <- 1000

  gen <- deepG::get_generator(
    path          = fasta_dir,
    train_type    = "masked_lm",
    format        = "fasta",
    batch_size    = batch_size,
    return_int    = TRUE,
    maxlen        = maxlen,
    step          = maxlen,
    n_gram        = 1,
    n_gram_stride = 1,
    padding       = TRUE,
    masked_lm     = list(
      mask_rate     = 0.10,
      random_rate   = 0.05,
      identity_rate = 0.05,
      include_sw    = TRUE,
      block_len     = 1
    ),
    max_samples   = batch_size,
    ambiguous_nuc  = "zero",
    vocabulary    = c("a", "c", "g", "t")
  )

  z <- gen()
  x <- z[[1]]
  y <- z[[2]]
  sw <- z[[3]]

  # Check dimensions
  expect_equal(dim(x)[1], batch_size)
  expect_equal(dim(x)[2], maxlen)
  expect_equal(dim(y)[1], batch_size)
  expect_equal(dim(y)[2], maxlen)
  expect_equal(dim(sw)[1], batch_size)

  # Clean up
  unlink(fasta_dir, recursive = TRUE)
})
