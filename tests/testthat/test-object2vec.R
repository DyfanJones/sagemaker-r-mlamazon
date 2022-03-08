# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_object2vec.py
context("object2vec")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
EPOCHS = 5
ENC0_MAX_SEQ_LEN = 100
ENC0_VOCAB_SIZE = 500

MINI_BATCH_SIZE = 32

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list("epochs"= EPOCHS, "enc0_max_seq_len"= ENC0_MAX_SEQ_LEN, "enc0_vocab_size"= ENC0_VOCAB_SIZE),
                 COMMON_TRAIN_ARGS)

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"

DESCRIBE_TRAINING_JOB_RESULT = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://bucket/model.tar.gz"))

ENDPOINT_DESC = list("EndpointConfigName"= "test-endpoint")

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

  sagemaker_client <- Mock$new()
  sagemaker_client$.call_args("describe_training_job", DESCRIBE_TRAINING_JOB_RESULT)
  sagemaker_client$.call_args("describe_endpoint", ENDPOINT_DESC)
  sagemaker_client$.call_args("describe_endpoint_config", ENDPOINT_CONFIG_DESC)

  sms$.call_args("default_bucket", BUCKET_NAME)
  sms$.call_args("expand_role", ROLE)
  sms$.call_args("train", list(TrainingJobArn = "sagemaker-object2vec-dummy"))
  sms$.call_args("create_model", "sagemaker-object2vec")
  sms$.call_args("endpoint_from_production_variants", "sagemaker-object2vec-endpoint")
  sms$.call_args("logs_for_job")

  sms$s3 <- s3_client
  sms$sagemaker <- sagemaker_client

  return(sms)
}

test_that("test init required positional", {
  object2vec = Object2Vec$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    EPOCHS,
    ENC0_MAX_SEQ_LEN,
    ENC0_VOCAB_SIZE,
    sagemaker_session=sagemaker_session()
  )
  expect_equal(object2vec$role, COMMON_TRAIN_ARGS$role)
  expect_equal(object2vec$instance_count, INSTANCE_COUNT)
  expect_equal(object2vec$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(object2vec$epochs, EPOCHS)
  expect_equal(object2vec$enc0_max_seq_len, ENC0_MAX_SEQ_LEN)
  expect_equal(object2vec$enc0_vocab_size, ENC0_VOCAB_SIZE)
})

test_that("test init required named", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec = do.call(Object2Vec$new, object2vec_args)

  expect_equal(object2vec$role, COMMON_TRAIN_ARGS$role)
  expect_equal(object2vec$instance_count, INSTANCE_COUNT)
  expect_equal(object2vec$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(object2vec$epochs, ALL_REQ_ARGS$epochs)
  expect_equal(object2vec$enc0_max_seq_len, ALL_REQ_ARGS$enc0_max_seq_len)
  expect_equal(object2vec$enc0_vocab_size, ALL_REQ_ARGS$enc0_vocab_size)
})

test_that("test all hyperparameters", {
  object2vec_args = c(sagemaker_session=sagemaker_session(),
                      enc_dim=1024,
                      mini_batch_size=100,
                      early_stopping_patience=3,
                      early_stopping_tolerance=0.001,
                      dropout=0.1,
                      weight_decay=0.001,
                      bucket_width=0,
                      num_classes=5,
                      mlp_layers=3,
                      mlp_dim=1024,
                      mlp_activation="tanh",
                      output_layer="softmax",
                      optimizer="adam",
                      learning_rate=0.0001,
                      negative_sampling_rate=1,
                      comparator_list="hadamard, abs_diff",
                      tied_token_embedding_weight=TRUE,
                      token_embedding_storage_type="row_sparse",
                      enc0_network="bilstm",
                      enc1_network="hcnn",
                      enc0_cnn_filter_width=3,
                      enc1_cnn_filter_width=3,
                      enc1_max_seq_len=300,
                      enc0_token_embedding_dim=300,
                      enc1_token_embedding_dim=300,
                      enc1_vocab_size=300,
                      enc0_layers=3,
                      enc1_layers=3,
                      enc0_freeze_pretrained_embedding=TRUE,
                      enc1_freeze_pretrained_embedding=FALSE,
                      ALL_REQ_ARGS)
  object2vec = do.call(Object2Vec$new, object2vec_args)

  hp = object2vec$hyperparameters()
  expect_equal(hp$epochs, EPOCHS)
  expect_equal(hp$mlp_activation , "tanh")
})

test_that("test image", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec = do.call(Object2Vec$new, object2vec_args)

  expect_equal(object2vec$training_image_uri(), ImageUris$new()$retrieve("object2vec", REGION))
})

test_that("test required hyper parameters type", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec_args$epochs = NULL
  test_param = list(num_topics = "string")

  for(i in seq_along(test_param)){
    test_args = c(object2vec_args, test_param[i])
    expect_error(do.call(Object2Vec$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec_args$enc0_vocab_size = NULL
  test_param = list("enc0_vocab_size"=0,
                    "enc0_vocab_size"=1000000000)

  for(i in seq_along(test_param)){
    test_args = c(object2vec_args, test_param[i])
    expect_error(do.call(Object2Vec$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec_args$epochs = NULL
  test_param = list("epochs"="string",
                    "optimizer"=0,
                    "enc0_cnn_filter_width"="string",
                    "weight_decay"="string",
                    "learning_rate"="string",
                    "negative_sampling_rate"="some_string",
                    "comparator_list"= 0,
                    "comparator_list"= list("foobar"),
                    "token_embedding_storage_type"= 123)

  for(i in seq_along(test_param)){
    test_args = c(object2vec_args, test_param[i])
    expect_error(do.call(Object2Vec$new, test_args))
  }
})

test_that("test error optional hyper parameters value", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec_args$epochs = NULL
  test_param = list("epochs"=0,
                    "epochs"=1000,
                    "optimizer"="string",
                    "early_stopping_tolerance"=0,
                    "early_stopping_tolerance"=0.5,
                    "early_stopping_patience"=0,
                    "early_stopping_patience"=100,
                    "weight_decay"=-1,
                    "weight_decay"=200000,
                    "enc0_cnn_filter_width"=2000,
                    "learning_rate"=0,
                    "learning_rate"=2,
                    "negative_sampling_rate"=-1,
                    "comparator_list"="hadamard,foobar",
                    "token_embedding_storage_type"="foobar")

  for(i in seq_along(test_param)){
    test_args = c(object2vec_args, test_param[i])
    expect_error(do.call(Object2Vec$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10

test_that("test call fit", {
  object2vec_args = c(base_job_name="object2vec", sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  object2vec$fit(data, MINI_BATCH_SIZE)

  expect_equal(object2vec$latest_training_job , "sagemaker-object2vec-dummy")
  expect_equal(object2vec$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training none mini batch_size", {
  object2vec_args = c(base_job_name="object2vec", sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  object2vec$fit(data)

  expect_equal(object2vec$latest_training_job , "sagemaker-object2vec-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  object2vec_args = c(base_job_name="object2vec", sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(object2vec$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  object2vec_args = c(base_job_name="object2vec", sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(object2vec$.prepare_for_training(data, 0))
})

test_that("test model image", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  object2vec$fit(data, MINI_BATCH_SIZE)
  model = object2vec$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("object2vec", REGION))
})

test_that("test predictor type", {
  object2vec_args = c(sagemaker_session=sagemaker_session(), ALL_REQ_ARGS)
  object2vec=do.call(Object2Vec$new, object2vec_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  object2vec$fit(data, MINI_BATCH_SIZE)
  model = object2vec$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "Predictor"))
})
