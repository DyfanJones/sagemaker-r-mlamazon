<!-- badges: start -->
[![:name status badge](https://dyfanjones.r-universe.dev/badges/:name)](https://dyfanjones.r-universe.dev)
[![R-CMD-check](https://github.com/DyfanJones/sagemaker-r-mlframework/workflows/R-CMD-check/badge.svg)](https://github.com/DyfanJones/sagemaker-r-mlframework/actions)
[![sagemaker.mlframework status badge](https://dyfanjones.r-universe.dev/badges/sagemaker.mlframework)](https://dyfanjones.r-universe.dev)
<!-- badges: end -->

# Intro:

This package contains lower level functionality for [sagemaker](https://github.com/DyfanJones/sagemaker-r-sdk).

# Installation:

You can install the development version of sagemaker from Github with:
```r
# install.packages("remotes)
remotes::install_github("DyfanJones/sagemaker-r-mlframework")
```

Alternatively you can install development from [r-universe](https://dyfanjones.r-universe.dev/ui#builds).
```r
# Enable universe(s) by dyfanjones
options(repos = c(
  dyfanjones = 'https://dyfanjones.r-universe.dev',
  CRAN = 'https://cloud.r-project.org')
)

# Install some packages
install.packages('sagemaker.mlframework')
```
