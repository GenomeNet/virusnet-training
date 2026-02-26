test_that("required R packages are loadable", {
  expect_no_error(library(reticulate))
  expect_no_error(library(tensorflow))
  expect_no_error(library(keras))
  expect_no_error(library(magrittr))
  expect_no_error(library(deepG))
})

test_that("TensorFlow backend works", {
  skip_if_not(reticulate::py_module_available("tensorflow"),
              "TensorFlow not available")
  x <- tensorflow::tf$constant(42.0)
  expect_equal(as.numeric(x), 42.0)
})
