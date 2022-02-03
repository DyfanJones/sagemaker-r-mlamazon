# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/xgboost/defaults.py

XGBOOST_NAME <- "xgboost"
XGBOOST_UNSUPPORTED_VERSIONS = list(
  "1.1"=paste(
    "XGBoost 1.1 is not supported on SageMaker because XGBoost 1.1 has broken capability to",
    "run prediction when the test input has fewer features than the training data in LIBSVM",
    "inputs. This capability has been restored in XGBoost 1.2",
    "(https://github.com/dmlc/xgboost/pull/5955). Consider using SageMaker XGBoost 1.2-1."
  )
)
