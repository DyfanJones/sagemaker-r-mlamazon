# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_sparkml_serving.py

context("spark ml")

library(sagemaker.core)
library(sagemaker.common)
library(sagemaker.mlcore)

MODEL_DATA = "s3://bucket/model.tar.gz"
ROLE = "myrole"
TRAIN_INSTANCE_TYPE = "ml.c4.xlarge"

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
ENDPOINT = "some-endpoint"

ENDPOINT_DESC = list("EndpointConfigName"= ENDPOINT)

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

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
  sms$.call_args("train", list(TrainingJobArn = "sagemaker-sparkml-dummy"))
  sms$.call_args("create_model", "sagemaker-sparkml")
  sms$.call_args("endpoint_from_production_variants", "sagemaker-sparkml-endpoint")
  sms$.call_args("logs_for_job")
  sms$.call_args("wait_for_job")
  sms$.call_args("wait_for_compilation_job", describe_compilation)
  sms$.call_args("compile_model")

  sms$s3 <- s3_client
  sms$sagemaker <- sagemaker_client

  return(sms)
}

test_that("test sparkml model", {
  sms <- sagemaker_session()
  sparkml = SparkMLModel$new(sagemaker_session=sms, model_data=MODEL_DATA, role=ROLE)
  expect_equal(sparkml$image_uri, ImageUris$new()$retrieve("sparkml-serving", REGION, version="2.4"))
})

test_that("test auto ml default channel name", {
  sms <- sagemaker_session()
  sparkml = SparkMLModel$new(sagemaker_session=sms, model_data=MODEL_DATA, role=ROLE)
  predictor = sparkml$deploy(1, TRAIN_INSTANCE_TYPE)
  expect_true(inherits(predictor, "SparkMLPredictor"))
})

test_that("test auto ml default channel name", {
  sms <- sagemaker_session()
  sparkml = SparkMLModel$new(sagemaker_session=sms, model_data=MODEL_DATA, role=ROLE)
  custom_serializer = Mock$new(name="BaseSerializer")
  predictor = sparkml$deploy(1, TRAIN_INSTANCE_TYPE, serializer=custom_serializer)
  expect_true(inherits(predictor, "SparkMLPredictor"))
  expect_equal(predictor$serializer, custom_serializer)
})
