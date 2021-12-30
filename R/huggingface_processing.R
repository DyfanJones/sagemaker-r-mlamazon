# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/huggingface/processing.py

#' @import R6
#' @import sagemaker.common
#' @import sagemaker.mlcore

#' @title HuggingFaceProcessor class
#' @description Handles Amazon SageMaker processing tasks for jobs using HuggingFace containers.
#' @export
HuggingFaceProcessor = R6Class("HuggingFaceProcessor",
  inherit = sagemaker.common::FrameworkProcessor,
  public = list(

    #' @field estimator_cls
    #' Estimator object
    estimator_cls = HuggingFace,

    #' @description This processor executes a Python script in a HuggingFace execution environment.
    #'              Unless ``image_uri`` is specified, the environment is an Amazon-built Docker container
    #'              that executes functions defined in the supplied ``code`` Python script.
    #'              The arguments have the same meaning as in ``FrameworkProcessor``, with the following
    #'              exceptions.
    #' @param role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing uses
    #'              this role to access AWS resources, such as data stored in Amazon S3.
    #' @param instance_count (int): The number of instances to run a processing job with.
    #' @param instance_type (str): The type of EC2 instance to use for processing, for
    #'              example, 'ml.c4.xlarge'.
    #' @param transformers_version (str): Transformers version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided. The current supported version is ``4.4.2``.
    #' @param tensorflow_version (str): TensorFlow version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``pytorch_version`` is provided. The current supported version is ``1.6.0``.
    #' @param pytorch_version (str): PyTorch version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``tensorflow_version`` is provided. The current supported version is ``2.4.1``.
    #' @param py_version (str): Python version you want to use for executing your model training
    #'              code. Defaults to ``None``. Required unless ``image_uri`` is provided.  If
    #'              using PyTorch, the current supported version is ``py36``. If using TensorFlow,
    #'              the current supported version is ``py37``.
    #' @param image_uri (str): The URI of the Docker image to use for the
    #'              processing jobs (default: None).
    #' @param command ([str]): The command to run, along with any command-line flags
    #'              to *precede* the ```code script```. Example: ["python3", "-v"]. If not
    #'              provided, ["python"] will be chosen (default: None).
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing volume (default: None).
    #' @param output_kms_key (str): The KMS key ID for processing job outputs (default: None).
    #' @param code_location (str): The S3 prefix URI where custom code will be
    #'              uploaded (default: None). The code file uploaded to S3 is
    #'              'code_location/job-name/source/sourcedir.tar.gz'. If not specified, the
    #'              default ``code location`` is 's3://{sagemaker-default-bucket}'
    #' @param max_runtime_in_seconds (int): Timeout in seconds (default: None).
    #'              After this amount of time, Amazon SageMaker terminates the job,
    #'              regardless of its current status. If `max_runtime_in_seconds` is not
    #'              specified, the default value is 24 hours.
    #' @param base_job_name (str): Prefix for processing name. If not specified,
    #'              the processor generates a default job name, based on the
    #'              processing image name and current timestamp (default: None).
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain (default: None).
    #' @param env (dict[str, str]): Environment variables to be passed to
    #'              the processing jobs (default: None).
    #' @param tags (list[dict]): List of tags to be passed to the processing job
    #'              (default: None). For more, see
    #'              \url{https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html}.
    #' @param network_config (:class:`~sagemaker.network.NetworkConfig`):
    #'              A :class:`~sagemaker.network.NetworkConfig`
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets (default: None).
    initialize = function(role,
                          instance_count,
                          instance_type,
                          transformers_version=NULL,
                          tensorflow_version=NULL,
                          pytorch_version=NULL,
                          py_version="py36",
                          image_uri=NULL,
                          command=NULL,
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          code_location=NULL,
                          max_runtime_in_seconds=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      self$pytorch_version = pytorch_version
      self$tensorflow_version = tensorflow_version
      super$initialize(
        self$estimator_cls,
        transformers_version,
        role,
        instance_count,
        instance_type,
        py_version,
        image_uri,
        command,
        volume_size_in_gb,
        volume_kms_key,
        output_kms_key,
        code_location,
        max_runtime_in_seconds,
        base_job_name,
        sagemaker_session,
        env,
        tags,
        network_config
      )
    }
  ),
  private = list(

    # Override default estimator factory function for HuggingFace's different parameters
    # HuggingFace estimators have 3 framework version parameters instead of one: The version for
    # Transformers, PyTorch, and TensorFlow.
    .create_estimator = function(entry_point="",
                                 source_dir=NULL,
                                 dependencies=NULL,
                                 git_config=NULL){
      return(self$estimator_cls$new(
        transformers_version=self$framework_version,
        tensorflow_version=self$tensorflow_version,
        pytorch_version=self$pytorch_version,
        py_version=self$py_version,
        entry_point=entry_point,
        source_dir=source_dir,
        dependencies=dependencies,
        git_config=git_config,
        code_location=self$code_location,
        enable_network_isolation=FALSE,
        image_uri=self$image_uri,
        role=self$role,
        instance_count=self$instance_count,
        instance_type=self$instance_type,
        sagemaker_session=self$sagemaker_session,
        debugger_hook_config=FALSE,
        disable_profiler=TRUE)
      )
    }
  )
)
