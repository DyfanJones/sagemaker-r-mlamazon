# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/huggingface/model.py

#' @include r_utils.R

#' @import R6
#' @import sagemaker.core
#' @import sagemaker.common
#' @import sagemaker.mlcore

#' @title A Predictor for inference against Hugging Face Endpoints.
#' @description This is able to serialize Python lists, dictionaries, and numpy arrays to
#'             multidimensional tensors for Hugging Face inference.
#' @export
HuggingFacePredictor = R6Class("HuggingFacePredictor",
  inherit = sagemaker.mlcore::Predictor,
  public = list(

    #' @description Initialize an ``HuggingFacePredictor``.
    #' @param endpoint_name (str): The name of the endpoint to perform inference
    #'              on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object that
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param serializer (sagemaker.serializers.BaseSerializer): Optional. Default
    #'              serializes input data to .npy format. Handles lists and numpy
    #'              arrays.
    #' @param deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
    #'              Default parses the response from .npy format to numpy array.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL,
                          serializer=JSONSerializer$new(),
                          deserializer=JSONDeserializer$new()){
      super$initialize(
        endpoint_name,
        sagemaker_session,
        serializer=serializer,
        deserializer=deserializer
      )
    }
  )
)

.validate_pt_tf_versions = function(pytorch_version, tensorflow_version, image_uri){
  if (!is.null(image_uri))
    return(NULL)

  if (!is.null(tensorflow_version) && !is.null(pytorch_version))
    ValueError$new(
      "tensorflow_version and pytorch_version are both not None. ",
      "Specify only tensorflow_version or pytorch_version."
    )
  if (is.null(tensorflow_version) && is.null(pytorch_version))
    ValueError$new(
      "tensorflow_version and pytorch_version are both None. ",
      "Specify either tensorflow_version or pytorch_version."
    )
}

#' @title HuggingFaceModel Class
#' @description A Hugging Face SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``.
#' @export
HuggingFaceModel = R6Class("HuggingFaceModel",
  inherit = sagemaker.mlcore::FrameworkModel,
  public = list(

    #' @description Initialize a HuggingFaceModel.
    #' @param model_data (str): The Amazon S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param entry_point (str): The absolute or relative path to the Python source
    #'              file that should be executed as the entry point to model
    #'              hosting. If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #'              Defaults to None.
    #' @param transformers_version (str): Transformers version you want to use for
    #'              executing your model training code. Defaults to None. Required
    #'              unless ``image_uri`` is provided.
    #' @param tensorflow_version (str): TensorFlow version you want to use for
    #'              executing your inference code. Defaults to ``None``. Required unless
    #'              ``pytorch_version`` is provided. List of supported versions:
    #'              \url{https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators}.
    #' @param pytorch_version (str): PyTorch version you want to use for
    #'              executing your inference code. Defaults to ``None``. Required unless
    #'              ``tensorflow_version`` is provided. List of supported versions:
    #'              \url{https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators}.
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided.
    #' @param image_uri (str): A Docker image URI. Defaults to None. If not specified, a
    #'              default image for PyTorch will be used. If ``framework_version``
    #'              or ``py_version`` are ``None``, then ``image_uri`` is required. If
    #'              also ``None``, then a ``ValueError`` will be raised.
    #' @param predictor_cls (callable[str, sagemaker.session.Session]): A function
    #'              to call to create a predictor with an endpoint name and
    #'              SageMaker ``Session``. If specified, ``deploy()`` returns the
    #'              result of invoking this function on the created endpoint name.
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one
    #'              worker per vCPU.
    #' @param ... : Keyword arguments passed to the superclass
    #'              :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
    #'              superclass :class:`~sagemaker.model.Model`.,
    initialize = function(role,
                          model_data=NULL,
                          entry_point=NULL,
                          transformers_version=NULL,
                          tensorflow_version=NULL,
                          pytorch_version=NULL,
                          py_version=NULL,
                          image_uri=NULL,
                          predictor_cls=HuggingFacePredictor,
                          model_server_workers=NULL,
                          ...){
      validate_version_or_image_args(transformers_version, py_version, image_uri)
      .validate_pt_tf_versions(
        pytorch_version=pytorch_version,
        tensorflow_version=tensorflow_version,
        image_uri=image_uri
      )
      if (py_version == "py2")
        ValueError$new("py2 is not supported with HuggingFace images")
      self$framework_version = transformers_version
      self$pytorch_version = pytorch_version
      self$tensorflow_version = tensorflow_version
      self$py_version = py_version

      super$initialize(model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, ...)
      self$model_server_workers = model_server_workers

      attr(self, "_framework_name") = "huggingface"
    },

    #' @description Creates a model package for creating SageMaker models or listing on Marketplace.
    #' @param content_types (list): The supported MIME types for the input data.
    #' @param response_types (list): The supported MIME types for the output data.
    #' @param inference_instances (list): A list of the instance types that are used to
    #'              generate inferences in real-time.
    #' @param transform_instances (list): A list of the instance types on which a transformation
    #'              job can be run or on which an endpoint can be deployed.
    #' @param model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
    #'              using `model_package_name` makes the Model Package un-versioned.
    #'              Defaults to ``None``.
    #' @param model_package_group_name (str): Model Package Group name, exclusive to
    #'              `model_package_name`, using `model_package_group_name` makes the Model Package
    #'              versioned. Defaults to ``None``.
    #' @param image_uri (str): Inference image URI for the container. Model class' self.image will
    #'              be used if it is None. Defaults to ``None``.
    #' @param model_metrics (ModelMetrics): ModelMetrics object. Defaults to ``None``.
    #' @param metadata_properties (MetadataProperties): MetadataProperties object.
    #'              Defaults to ``None``.
    #' @param marketplace_cert (bool): A boolean value indicating if the Model Package is certified
    #'              for AWS Marketplace. Defaults to ``False``.
    #' @param approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
    #'              or "PendingManualApproval". Defaults to ``PendingManualApproval``.
    #' @param description (str): Model Package description. Defaults to ``None``.
    #' @param drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None)
    #' @return A `sagemaker.model.ModelPackage` instance.
    register = function(content_types,
                        response_types,
                        inference_instances,
                        transform_instances,
                        model_package_name=NULL,
                        model_package_group_name=NULL,
                        image_uri=NULL,
                        model_metrics=NULL,
                        metadata_properties=NULL,
                        marketplace_cert=FALSE,
                        approval_status=NULL,
                        description=NULL,
                        drift_check_baselines=NULL){
      instance_type = inference_instances[[1]]
      private$.init_sagemaker_session_if_does_not_exist(instance_type)

      if (!is.null(image_uri))
        self$image_uri = image_uri
      if (is.null(self$image_uri))
        self$image_uri = self$serving_image_uri(
          region_name=self$sagemaker_session$paws_region_name,
          instance_type=instance_type
        )
      return(super$register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name,
        model_package_group_name,
        image_uri,
        model_metrics,
        metadata_properties,
        marketplace_cert,
        approval_status,
        description,
        drift_check_baselines=drift_check_baselines
        )
      )
    },

    #' @description A container definition with framework configuration set in model environment variables.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge'.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model.
    #' @return dict[str, str]: A container definition object usable with the
    #'              CreateModel API.
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
      deploy_image = self$image_uri
      if (is.null(deploy_image)){
        if (is.null(instance_type))
          ValueError$new(
            "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
          )
        region_name = self$sagemaker_session$paws_region_name
        deploy_image = self$serving_image_uri(
          region_name, instance_type, accelerator_type=accelerator_type
        )
      }
      deploy_key_prefix = model_code_key_prefix(self$key_prefix, self.name, deploy_image)
      private$.upload_code(deploy_key_prefix, repack=True)
      deploy_env = list(self$env)
      deploy_env= modifyList(deploy_env, private$.framework_env_vars())

      if (!is.null(self$model_server_workers))
        deploy_env[[toupper(model_parameters$MODEL_SERVER_WORKERS_PARAM_NAME)]] = as.character(self$model_server_workers)
      return(container_def(
        deploy_image, self$repacked_model_data %||% self.model_data, deploy_env)
      )
    },

    #' @description Create a URI for the serving image.
    #' @param region_name (str): AWS region where the image is uploaded.
    #' @param instance_type (str): SageMaker instance type. Used to determine device type
    #'              (cpu/gpu/family-specific optimized).
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model.
    #' @return str: The appropriate image URI based on the given parameters.
    serving_image_uri = function(region_name, instance_type, accelerator_type=NULL){
      if(!is.null(self$tensorflow_version)) {
        base_framework_version = sprintf("tensorflow%s", self$tensorflow_version)
      } else {
        base_framework_version = sprintf("pytorch%s", self$pytorch_version)
      }
      return(sagemaker.core::ImageUris$new()$retrieve(
        attr(self, "_framework_name"),
        region_name,
        version=self.framework_version,
        py_version=self.py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        image_scope="inference",
        base_framework_version=base_framework_version)
      )
    }
  ),
  lock_objects = F
)
