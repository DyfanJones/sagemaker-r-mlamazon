# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/sagemaker/automl/test_auto_ml.py

context("automl")

library(lgr)
library(sagemaker.core)
library(sagemaker.common)
library(sagemaker.mlcore)

lg <- get_logger("sagemaker")

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

TIMESTAMP = "2017-11-06-14:14:15.671"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.2xlarge"
RESOURCE_POOLS = list(list("InstanceType"= INSTANCE_TYPE, "PoolSize"= INSTANCE_COUNT))
ROLE = "DummyRole"
TARGET_ATTRIBUTE_NAME = "target"
REGION = "us-west-2"
DEFAULT_S3_INPUT_DATA = sprintf("s3://%s/data", BUCKET_NAME)
DEFAULT_OUTPUT_PATH = sprintf("s3://%s/", BUCKET_NAME)
LOCAL_DATA_PATH = "file://data"
DEFAULT_MAX_CANDIDATES = NULL
DEFAULT_JOB_NAME = sprintf("automl-%s", TIMESTAMP)

JOB_NAME = "default-job-name"
JOB_NAME_2 = "banana-auto-ml-job"
JOB_NAME_3 = "descriptive-auto-ml-job"
VOLUME_KMS_KEY = "volume-kms-key-id-string"
OUTPUT_KMS_KEY = "output-kms-key-id-string"
OUTPUT_PATH = "s3://my_other_bucket/"
BASE_JOB_NAME = "banana"
PROBLEM_TYPE = "BinaryClassification"
BLACKLISTED_ALGORITHM = list("xgboost")
LIST_TAGS_RESULT = list("Tags"= list(list("Key"= "key1", "Value"= "value1")))
MAX_CANDIDATES = 10
MAX_RUNTIME_PER_TRAINING_JOB = 3600
TOTAL_JOB_RUNTIME = 36000
TARGET_OBJECTIVE = "0.01"
JOB_OBJECTIVE = list("fake job objective")
TAGS = list(list("Name"= "some-tag", "Value"= "value-for-tag"))
VPC_CONFIG = list("SecurityGroupIds"= list("group"), "Subnets"= list("subnet"))
COMPRESSION_TYPE = "Gzip"
ENCRYPT_INTER_CONTAINER_TRAFFIC = FALSE
GENERATE_CANDIDATE_DEFINITIONS_ONLY = FALSE
BEST_CANDIDATE = list("best-candidate" = "best-trial")
BEST_CANDIDATE_2 = list("best-candidate" = "best-trial-2")
AUTO_ML_DESC = list("AutoMLJobName"= JOB_NAME, "BestCandidate"= BEST_CANDIDATE)
AUTO_ML_DESC_2 = list("AutoMLJobName"= JOB_NAME_2, "BestCandidate"= BEST_CANDIDATE_2)
AUTO_ML_DESC_3 = list(
  "AutoMLJobArn"= "automl_job_arn",
  "AutoMLJobConfig"= list(
    "CompletionCriteria"= list(
      "MaxAutoMLJobRuntimeInSeconds"= 3000,
      "MaxCandidates"= 28,
      "MaxRuntimePerTrainingJobInSeconds"= 100
    ),
    "SecurityConfig"= list("EnableInterContainerTrafficEncryption"= TRUE)
  ),
  "AutoMLJobName"= "mock_automl_job_name",
  "AutoMLJobObjective"= list("MetricName" = "Auto"),
  "AutoMLJobSecondaryStatus" = "Completed",
  "AutoMLJobStatus" = "Completed",
  "GenerateCandidateDefinitionsOnly" = FALSE,
  "InputDataConfig" = list(
    list(
      "DataSource" = list(
        "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= "s3://input/prefix")
      ),
      "TargetAttributeName"= "y"
      )
    ),
  "OutputDataConfig"= list("KmsKeyId"= "string", "S3OutputPath"= "s3://output_prefix"),
  "ProblemType"= "Auto",
  "RoleArn"= "mock_role_arn"
)

INFERENCE_CONTAINERS = list(
  list(
    "Environment"= list("SAGEMAKER_PROGRAM"= "sagemaker_serve"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output"
    ),
  list(
    "Environment"= list("MAX_CONTENT_LENGTH"= "20000000"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output"
    ),
  list(
    "Environment"= list("INVERSE_LABEL_TRANSFORM"= "1"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output"
    )
)

CLASSIFICATION_INFERENCE_CONTAINERS = list(
  list(
    "Environment"= list("SAGEMAKER_PROGRAM"= "sagemaker_serve"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output"
    ),
  list(
    "Environment"= list(
      "MAX_CONTENT_LENGTH"= "20000000",
      "SAGEMAKER_INFERENCE_SUPPORTED"= "probability,probabilities,predicted_label",
      "SAGEMAKER_INFERENCE_OUTPUT"= "predicted_label"
      ),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output"
    ),
  list(
    "Environment"= list(
      "INVERSE_LABEL_TRANSFORM"= "1",
      "SAGEMAKER_INFERENCE_SUPPORTED"= "probability,probabilities,predicted_label,labels",
      "SAGEMAKER_INFERENCE_OUTPUT"= "predicted_label",
      "SAGEMAKER_INFERENCE_INPUT"= "predicted_label"
      ),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output"
    )
)

CANDIDATE_STEPS = list(
  list(
    "CandidateStepName"= "training-job/sagemaker-auto-ml-gamma/data-processing",
    "CandidateStepType"= "AWS::Sagemaker::TrainingJob"
    ),
  list(
    "CandidateStepName"= "transform-job/sagemaker-auto-ml-gamma/transform",
    "CandidateStepType"= "AWS::Sagemaker::TransformJob"
    ),
  list(
    "CandidateStepName"= "training-job/sagemaker-auto-ml-gamma/training",
    "CandidateStepType"= "AWS::Sagemaker::TrainingJob"
    )
)

CANDIDATE_DICT = list(
  "CandidateName"= "candidate_mock",
  "InferenceContainers"= INFERENCE_CONTAINERS,
  "CandidateSteps"= CANDIDATE_STEPS
)

CLASSIFICATION_CANDIDATE_DICT = list(
  "CandidateName"= "candidate_mock",
  "InferenceContainers"= CLASSIFICATION_INFERENCE_CONTAINERS,
  "CandidateSteps"= CANDIDATE_STEPS
)

TRAINING_JOB = list(
  "AlgorithmSpecification"= list(
    "AlgorithmName"= "string",
    "TrainingImage"= "string",
    "TrainingInputMode"= "string"
    ),
  "CheckpointConfig"= list("LocalPath"= "string", "S3Uri"= "string"),
  "EnableInterContainerTrafficEncryption" = FALSE,
  "EnableManagedSpotTraining" = FALSE,
  "EnableNetworkIsolation" = FALSE,
  "InputDataConfig" = list(
    list("DataSource"= list("S3DataSource"= list("S3DataType"= "string", "S3Uri"= "string")))
    ),
  "OutputDataConfig"= list("KmsKeyId"= "string", "S3OutputPath"= "string"),
  "ResourceConfig"= list(),
  "RoleArn"= "string",
  "StoppingCondition"= list(),
  "TrainingJobArn"= "string",
  "TrainingJobName"= "string",
  "TrainingJobStatus"= "Completed",
  "VpcConfig"= list()
)

TRANSFORM_JOB = list(
  "BatchStrategy"= "string",
  "DataProcessing"= list(),
  "Environment"= list("string"= "string"),
  "FailureReason"= "string",
  "LabelingJobArn"= "string",
  "MaxConcurrentTransforms"= 1,
  "MaxPayloadInMB"= 2000,
  "ModelName"= "string",
  "TransformInput"= list("DataSource"= list("S3DataSource"= list("S3DataType"= "string", "S3Uri"= "string"))),
  "TransformJobStatus"= "Completed",
  "TransformJobArn"= "string",
  "TransformJobName"= "string",
  "TransformOutput"= list(),
  "TransformResources"= list()
)

describe_auto_ml_job_mock = function(job_name=NULL){
  if (is.null(job_name) || job_name == JOB_NAME)
    return(AUTO_ML_DESC)
  else if (job_name == JOB_NAME_2)
    return(AUTO_ML_DESC_2)
  else if (job_name == JOB_NAME_3)
    return(AUTO_ML_DESC_3)
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

  sagemaker_client$.call_args("describe_training_job", TRAINING_JOB)
  sagemaker_client$.call_args("describe_transform_job", TRANSFORM_JOB)
  sagemaker_client$.call_args("describe_endpoint", ENDPOINT_DESC)
  sagemaker_client$.call_args("describe_endpoint_config", ENDPOINT_CONFIG_DESC)
  sagemaker_client$.call_args("list_tags", LIST_TAGS_RESULT)

  sms$.call_args("default_bucket", BUCKET_NAME)
  sms$.call_args("expand_role", ROLE)
  sms$.call_args("train")
  sms$.call_args("create_model", "sagemaker-xgboost")
  sms$.call_args("endpoint_from_production_variants", "sagemaker-xgboost-endpoint")
  sms$.call_args("logs_for_job")
  sms$.call_args("wait_for_job")
  sms$.call_args("wait_for_compilation_job", describe_compilation)
  sms$.call_args("compile_model")
  sms$.call_args("upload_data", DEFAULT_S3_INPUT_DATA)
  sms$.call_args("describe_auto_ml_job", side_effect=describe_auto_ml_job_mock)
  sms$.call_args("list_candidates", list("Candidates"= list()))
  sms$.call_args("auto_ml")
  sms$.call_args("transform")
  sms$.call_args("logs_for_auto_ml_job")

  sms$s3 <- s3_client
  sms$sagemaker <- sagemaker_client

  return(sms)
}

candidate = Mock$new(
  name="candidate_mock",
  containers=INFERENCE_CONTAINERS,
  steps=CANDIDATE_STEPS,
  sagemaker_session=sagemaker_session()
)

test_that("test auto ml default channel name", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  inputs = DEFAULT_S3_INPUT_DATA
  AutoMLJob$new(sms)$start_new(auto_ml, inputs)

  args = auto_ml$sagemaker_session$auto_ml(..return_value = T)
  args$input_config

  expect_equal(args$input_config,
    list(
      list(
        "DataSource"= list(
          "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
        ),
        "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
      )
    )
  )
})

test_that("test auto ml invalid input data format", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  inputs = 1

  expect_error(
    AutoMLJob$new(sms)$start_new(auto_ml, inputs),
    sprintf("Cannot format input %s. Expecting a string or a list of strings.", inputs),
    class = "ValueError"
  )
})

test_that("test auto ml only one of problem type and job objective provided", {
  msg = paste0("One of problem type and objective metric provided. Either both of them ",
               "should be provided or none of them should be provided.")
  expect_error(
    AutoML$new(
      role=ROLE,
      target_attribute_name=TARGET_ATTRIBUTE_NAME,
      sagemaker_session=sagemaker_session(),
      problem_type=PROBLEM_TYPE),
    msg,
    class = "ValueError"
  )
})

test_that("test auto ml fit set logs to false", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  inputs = DEFAULT_S3_INPUT_DATA

  auto_ml$fit(inputs, job_name=JOB_NAME, wait=FALSE, logs=TRUE)

  expect_equal(
    lg$last_event$msg,
    "Setting `logs` to FALSE. `logs` is only meaningful when `wait` is TRUE."
  )
})

test_that("test auto ml additional optional params", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE,
    target_attribute_name=TARGET_ATTRIBUTE_NAME,
    sagemaker_session=sms,
    volume_kms_key=VOLUME_KMS_KEY,
    vpc_config=VPC_CONFIG,
    encrypt_inter_container_traffic=ENCRYPT_INTER_CONTAINER_TRAFFIC,
    compression_type=COMPRESSION_TYPE,
    output_kms_key=OUTPUT_KMS_KEY,
    output_path=OUTPUT_PATH,
    problem_type=PROBLEM_TYPE,
    max_candidates=MAX_CANDIDATES,
    max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
    job_objective=JOB_OBJECTIVE,
    generate_candidate_definitions_only=GENERATE_CANDIDATE_DEFINITIONS_ONLY,
    tags=TAGS)

  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs, job_name=JOB_NAME)
  args = sms$auto_ml(..return_value = T)
  expect_equal(args, list(
      "input_config"= list(
        list(
          "DataSource"= list(
            "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
          ),
          "CompressionType"= COMPRESSION_TYPE,
          "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
          )
        ),
      "output_config"= list("S3OutputPath"= OUTPUT_PATH, "KmsKeyId"= OUTPUT_KMS_KEY),
      "auto_ml_job_config"= list(
        "CompletionCriteria"= list(
          "MaxCandidates"= MAX_CANDIDATES,
          "MaxRuntimePerTrainingJobInSeconds"= MAX_RUNTIME_PER_TRAINING_JOB,
          "MaxAutoMLJobRuntimeInSeconds"= TOTAL_JOB_RUNTIME
          ),
        "SecurityConfig"= list(
          "EnableInterContainerTrafficEncryption"= ENCRYPT_INTER_CONTAINER_TRAFFIC,
          "VolumeKmsKeyId"= VOLUME_KMS_KEY,
          "VpcConfig"= VPC_CONFIG
          )
        ),
      "role"= ROLE,
      "generate_candidate_definitions_only"= GENERATE_CANDIDATE_DEFINITIONS_ONLY,
      "job_name"= JOB_NAME,
      "problem_type"= PROBLEM_TYPE,
      "job_objective"= JOB_OBJECTIVE,
      "tags"= TAGS
    )
  )
})

test_that("test auto ml default fit", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs)
  args = sms$auto_ml(..return_value=T)
  # remove timestamp job name
  args$job_name = NULL
  expect_equal(args, list(
    "input_config"= list(
      list(
        "DataSource"= list(
          "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
          ),
        "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
        )
      ),
    "output_config"= list("S3OutputPath"= DEFAULT_OUTPUT_PATH),
    "auto_ml_job_config"= list(
      "CompletionCriteria"= list("MaxCandidates"= DEFAULT_MAX_CANDIDATES),
      "SecurityConfig"= list(
        "EnableInterContainerTrafficEncryption"= ENCRYPT_INTER_CONTAINER_TRAFFIC
        )
      ),
    "role"= ROLE,
    "generate_candidate_definitions_only"= GENERATE_CANDIDATE_DEFINITIONS_ONLY
    )
  )
})

test_that("test auto ml local input", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs)
  args = sms$auto_ml(..return_value = T)
  expect_equal(
    args$input_config[[1]]$DataSource$S3DataSource$S3Uri,
    DEFAULT_S3_INPUT_DATA)
})

test_that("test auto ml input", {
  sms <- sagemaker_session()
  inputs = AutoMLInput$new(
    inputs=DEFAULT_S3_INPUT_DATA, target_attribute_name="target", compression="Gzip"
  )
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  auto_ml$fit(inputs)
  args = sms$auto_ml(..return_value = T)
  expect_equal(args$input_config, list(
    list(
      "DataSource"= list(
        "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
        ),
      "TargetAttributeName"= TARGET_ATTRIBUTE_NAME,
      "CompressionType"= "Gzip")
    )
  )
})

test_that("test describe auto ml job", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  expect_equal(auto_ml$describe_auto_ml_job(job_name=JOB_NAME),
               AUTO_ML_DESC)
})

test_that("test list candidates default", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  auto_ml$current_job_name = "current_job_name"
  expect_equal(auto_ml$list_candidates(), list())
})

test_that("test list candidates with optional args", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  auto_ml$list_candidates(
    job_name=JOB_NAME,
    status_equals="Completed",
    candidate_name="candidate-name",
    candidate_arn="candidate-arn",
    sort_order="Ascending",
    sort_by="Status",
    max_results=99
  )

  args = sms$list_candidates(..return_value = T)
  expect_equal(args, list(
    "job_name"= JOB_NAME,
    "status_equals"= "Completed",
    "candidate_name"= "candidate-name",
    "candidate_arn"= "candidate-arn",
    "sort_order"= "Ascending",
    "sort_by"= "Status",
    "max_results"= 99)
    )
})

test_that("test best candidate with existing best candidate", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  auto_ml$.best_candidate = BEST_CANDIDATE
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate default job name", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  auto_ml$current_job_name = JOB_NAME
  auto_ml$.auto_ml_job_desc = AUTO_ML_DESC
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate job no desc", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  auto_ml$current_job_name = JOB_NAME
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate no desc no job name", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  best_candidate = auto_ml$best_candidate(job_name=JOB_NAME)
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate job name not match", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  auto_ml$current_job_name = JOB_NAME
  auto_ml$.auto_ml_job_desc = AUTO_ML_DESC
  best_candidate = auto_ml$best_candidate(job_name=JOB_NAME_2)

  expect_equal(best_candidate, BEST_CANDIDATE_2)
})

test_that("test_deploy", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  mock_pipeline = Mock$new(name="pipeline_model")
  mock_pipeline$.call_args("deploy")

  unlockEnvironmentBinding(auto_ml$.__enclos_env__$self)
  auto_ml$best_candidate = mock_fun(CANDIDATE_DICT)
  auto_ml$create_model = mock_fun(mock_pipeline)

  auto_ml$deploy(
    initial_instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    sagemaker_session=sagemaker_session,
    model_kms_key=OUTPUT_KMS_KEY
  )

  expect_equal(auto_ml$create_model(..count = T), 1)
  expect_equal(mock_pipeline$deploy(..count = T), 1)
})

test_that("test_deploy_optional_args", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  mock_pipeline = Mock$new(name="pipeline_model")
  mock_pipeline$.call_args("deploy")
  unlockEnvironmentBinding(auto_ml$.__enclos_env__$self)
  auto_ml$best_candidate = mock_fun(CANDIDATE_DICT)
  auto_ml$create_model = mock_fun(mock_pipeline)

  auto_ml$deploy(
    initial_instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    candidate=CANDIDATE_DICT,
    sagemaker_session=sms,
    name=JOB_NAME,
    endpoint_name=JOB_NAME,
    tags=TAGS,
    wait=FALSE,
    vpc_config=VPC_CONFIG,
    enable_network_isolation=TRUE,
    model_kms_key=OUTPUT_KMS_KEY,
    predictor_cls=Predictor,
    inference_response_keys=NULL
  )
  expect_equal(auto_ml$create_model(..count = T), 1)
  expect_equal(auto_ml$create_model(..return_value = T), list(
    name=JOB_NAME,
    sagemaker_session=sms,
    candidate=CANDIDATE_DICT,
    inference_response_keys=NULL,
    vpc_config=VPC_CONFIG,
    enable_network_isolation=TRUE,
    model_kms_key=OUTPUT_KMS_KEY,
    predictor_cls=Predictor
  ))
  expect_equal(mock_pipeline$deploy(..count = T), 1)

  expect_equal(mock_pipeline$deploy(..return_value = T), list(
    initial_instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    serializer=NULL,
    deserializer=NULL,
    endpoint_name=JOB_NAME,
    kms_key=OUTPUT_KMS_KEY,
    tags=TAGS,
    wait=FALSE
  ))
})

test_that("test candidate estimator get steps", {
  sms <- sagemaker_session()
  candidate_estimator = CandidateEstimator$new(CANDIDATE_DICT, sagemaker_session=sms)
  steps = candidate_estimator$get_steps()

  expect_equal(length(steps), 3)
})

test_that("test validate and update inference response", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  cic = auto_ml$validate_and_update_inference_response(
    inference_containers=CLASSIFICATION_INFERENCE_CONTAINERS,
    inference_response_keys=c("predicted_label", "labels", "probabilities", "probability")
  )

  expect_equal(cic[[3]]$Environment$SAGEMAKER_INFERENCE_OUTPUT, "predicted_label,labels,probabilities,probability")
  expect_equal(cic[[3]]$Environment$SAGEMAKER_INFERENCE_INPUT, "predicted_label,probabilities,probability")
  expect_equal(cic[[2]]$Environment$SAGEMAKER_INFERENCE_OUTPUT, "predicted_label,probabilities,probability")
})

test_that("test validate and update inference response wrong input", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  expect_error(
    auto_ml$validate_and_update_inference_response(
      inference_containers=CLASSIFICATION_INFERENCE_CONTAINERS,
      inference_response_keys=c("wrong_key", "wrong_label", "probabilities", "probability")
    ),
    "Requested inference output keys*.",
    class = "ValueError"
  )
})

test_that("test create model", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )

  pipeline_model = auto_ml$create_model(
    name=JOB_NAME,
    sagemaker_session=sms,
    candidate=CLASSIFICATION_CANDIDATE_DICT,
    vpc_config=VPC_CONFIG,
    enable_network_isolation=TRUE,
    model_kms_key=NULL,
    predictor_cls=NULL,
    inference_response_keys=NULL
  )

  expect_true(inherits(pipeline_model, "PipelineModel"))
})

test_that("test attach", {
  sms <- sagemaker_session()
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sms
  )
  aml = auto_ml$attach(auto_ml_job_name=JOB_NAME_3, sagemaker_session=sms)

  expect_equal(aml$current_job_name, JOB_NAME_3)
  expect_equal(aml$role, "mock_role_arn")
  expect_equal(aml$target_attribute_name, "y")
  expect_equal(aml$problem_type, "Auto")
  expect_equal(aml$output_path, "s3://output_prefix")
  expect_equal(aml$tags, LIST_TAGS_RESULT$Tags)
})
