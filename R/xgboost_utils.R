# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/dev/src/sagemaker/xgboost/utils.py

#' @include xgboost_default.R

#' @import sagemaker.core

validate_py_version = function(py_version){
  if (py_version != "py3")
    ValueError$new(sprintf("Unsupported Python version: %s.",py_version))
}

validate_framework_version = function(framework_version){
  xgboost_version = split_str(framework_version, "-")[1]
  if (xgboost_version %in% XGBOOST_UNSUPPORTED_VERSIONS){
    msg = XGBOOST_UNSUPPORTED_VERSIONS[[xgboost_version]]
    ValueError$new(msg)
  }
}
