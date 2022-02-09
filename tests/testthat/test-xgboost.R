# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_xgboost.py

context("xgboost")

library(sagemaker.core)
library(sagemaker.common)
library(sagemaker.mlcore)

ENDPOINT_DESC = list("EndpointConfigName"= "test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

LIST_TAGS_RESULT = list("Tags"= list(list("Key"= "TagtestKey", "Value"= "TagtestValue")))

EXPERIMENT_CONFIG = list(
  "ExperimentName"= "exp",
  "TrialName"= "trial",
  "TrialComponentDisplayName"= "tc")

DATA_DIR = file.path(getwd(), "data")
SCRIPT_PATH = file.path(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1507167947
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
DIST_INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
GPU_INSTANCE_TYPE = "ml.p2.xlarge"
PYTHON_VERSION = "py3"
IMAGE_URI = "sagemaker-xgboost"
JOB_NAME = sprintf("%s-%s", IMAGE_URI, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "246618743249.dkr.ecr.%s.amazonaws.com/%s:%s-%s-%s"
ROLE = "Dummy"
REGION = "us-west-2"
CPU = "ml.c4.xlarge"
xgboost_framework_version ="1.0-1"

.get_full_cpu_image_uri = function(version){
  return(sprintf(IMAGE_URI_FORMAT_STRING, REGION, IMAGE_URI, version, "cpu", PYTHON_VERSION))
}

sagemaker_session <- function(){
  paws_mock <- Mock$new(name = "PawsCredentials", region_name = REGION)
  sms <- Mock$new(
    name = "Session",
    paws_credentials = paws_mock,
    paws_region_name=REGION,
    config=NULL,
    local_mode=FALSE,
    s3 = NULL
  )

  s3_client <- Mock$new()
  s3_client$.call_args("put_object")
  s3_client$.call_args("get_object", list(Body = BIN_OBJ))

  sagemaker_client <- Mock$new()
  describe = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://m/m.tar.gz"))
  describe_compilation = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://m/model_c5.tar.gz"))

  sagemaker_client$.call_args("describe_training_job", describe)
  sagemaker_client$.call_args("describe_endpoint", ENDPOINT_DESC)
  sagemaker_client$.call_args("describe_endpoint_config", ENDPOINT_CONFIG_DESC)
  sagemaker_client$.call_args("list_tags", LIST_TAGS_RESULT)

  sms$.call_args("default_bucket", BUCKET_NAME)
  sms$.call_args("expand_role", ROLE)
  sms$.call_args("train", list(TrainingJobArn = "sagemaker-xgboost-dummy"))
  sms$.call_args("create_model", "sagemaker-xgboost")
  sms$.call_args("endpoint_from_production_variants", "sagemaker-xgboost-endpoint")
  sms$.call_args("logs_for_job")
  sms$.call_args("wait_for_job")
  sms$.call_args("wait_for_compilation_job", describe_compilation)
  sms$.call_args("compile_model")

  sms$s3 <- s3_client
  sms$sagemaker <- sagemaker_client

  return(sms)
}

test_that("test create model", {
  source_dir = "s3://mybucket/source"
  sms <- sagemaker_session()
  xgboost_model = XGBoostModel$new(
    model_data=source_dir,
    role=ROLE,
    sagemaker_session=sms,
    entry_point=SCRIPT_PATH,
    framework_version=xgboost_framework_version)
  default_image_uri = .get_full_cpu_image_uri(xgboost_framework_version)
  model_values = xgboost_model$prepare_container_def(CPU)
  expect_equal(model_values$Image, default_image_uri)
})

test_that("test create model from estimator",{
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  base_job_name = "job"
  sms <- sagemaker_session()

  xgboost = XGBoost$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sms,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    framework_version=xgboost_framework_version,
    container_log_level=container_log_level,
    py_version=PYTHON_VERSION,
    base_job_name=base_job_name,
    source_dir=source_dir
  )
  xgboost$fit(inputs="s3://mybucket/train", job_name="new_name")
  xgboost$uploaded_code
  model_name = "model_name"
  model = xgboost$create_model()

  expect_equal(model$sagemaker_session, sms)
  expect_equal(model$framework_version, xgboost_framework_version)
  expect_equal(model$py_version, xgboost$py_version)
  expect_equal(model$entry_point, basename(SCRIPT_PATH))
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level, "20")
  expect_equal(model$source_dir, source_dir)
  expect_null(model$vpc_config)
})

test_that("test deploy model", {
  sms <- sagemaker_session()

  model = XGBoostModel$new(
    "s3://some/data.tar.gz",
    role=ROLE,
    framework_version=xgboost_framework_version,
    entry_point=SCRIPT_PATH,
    sagemaker_session=sms)
  predictor = model$deploy(1, CPU)
  expect_true(inherits(predictor, "XGBoostPredictor"))
})

test_that("test training image uri", {
  sms <- sagemaker_session()

  xgboost = XGBoost$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    framework_version=xgboost_framework_version,
    sagemaker_session=sms,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    py_version=PYTHON_VERSION)
  default_image_uri = .get_full_cpu_image_uri(xgboost_framework_version)
  model_uri = xgboost$training_image_uri()
  expect_equal(default_image_uri, model_uri)
})
