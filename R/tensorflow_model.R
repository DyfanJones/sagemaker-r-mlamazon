# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/model.py

#' @include r_utils.R

#' @import R6
#' @import sagemaker.core
#' @import sagemaker.common
#' @import sagemaker.mlcore
#' @import lgr

#' @title TensorFlowPredictor Class
#' @description A ``Predictor`` implementation for inference against TensorFlow
#'              Serving endpoints.
#' @export
TensorFlowPredictor = R6Class("TensorFlowPredictor",
  inherit = sagemaker.mlcore::Predictor,
  public = list(

    #' @description Initialize a ``TensorFlowPredictor``.
    #'              See :class:`~sagemaker.predictor.Predictor` for more info about parameters.
    #' @param endpoint_name (str): The name of the endpoint to perform inference
    #'              on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param serializer (callable): Optional. Default serializes input data to
    #'              json. Handles dicts, lists, and numpy arrays.
    #' @param deserializer (callable): Optional. Default parses the response using
    #'              ``json.load(...)``.
    #' @param model_name (str): Optional. The name of the SavedModel model that
    #'              should handle the request. If not specified, the endpoint's
    #'              default model will handle the request.
    #' @param model_version (str): Optional. The version of the SavedModel model
    #'              that should handle the request. If not specified, the latest
    #'              version of the model will be used.
    #' @param ... : Additional parameters passed to the Predictor constructor.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL,
                          serializer=JSONSerializer$new(),
                          deserializer=JSONDeserializer$new(),
                          model_name=NULL,
                          model_version=NULL,
                          ...){
      kwargs = list(...)
      removed_kwargs("content_type", kwargs)
      removed_kwargs("accept", kwargs)
      super$initialize(
        endpoint_name,
        sagemaker_session,
        serializer,
        deserializer)

      attributes = list()
      if (!is.null(model_name))
        attributes = list.append(attributes, sprintf("tfs-model-name=%s",model_name))
      if (!is.null(model_version))
        attributes = list.append(attributes, sprintf("tfs-model-version=%s",model_version))
      private$.model_attributes =  if (!islistempty(attributes)) paste(attributes, collapse = ",") else NULL
    },

    #' @description PlaceHolder
    #' @param data :
    classify = function(data){
      return (private$.classify_or_regress(data, "classify"))
    },

    #' @description PlaceHolder
    #' @param data :
    regress = function(data){
      return(private$.classify_or_regress(data, "regress"))
    },

    #' @description Return the inference from the specified endpoint.
    #' @param data (object): Input data for which you want the model to provide
    #'              inference. If a serializer was specified when creating the
    #'              Predictor, the result of the serializer is sent as input
    #'              data. Otherwise the data must be sequence of bytes, and the
    #'              predict method then sends the bytes in the request body as is.
    #' @param initial_args (list[str,str]): Optional. Default arguments for boto3
    #'              ``invoke_endpoint`` call. Default is NULL (no default
    #'              arguments).
    predict = function(data,
                       initial_args=NULL){
      args = if (!islistempty(initial_args)) initial_args else list()
      if (!islistempty(private$.model_attributes)){
        if ("CustomAttributes" %in% names(args))
          args$CustomAttributes = paste0(args$CustomAttributes, ",", private$.model_attributes)
        else
          args$CustomAttributes = private$.model_attributes
      }
      return(super$predict(data, args))
    }
  ),
  private = list(
    .model_attributes = NULL,
    .classify_or_regress = function(data,
                                    method){
      if (!(method %in% c("classify", "regress")))
        ValueError$new(sprintf("invalid TensorFlow Serving method: %s", method))

      if (self$content_type != "application/json")
        ValueError$new(sprintf("The %s api requires json requests.", method))

      args = list("CustomAttributes"= sprintf("tfs-method=%s", method))

      return(self$predict(data, args))
    }
  ),
  lock_objects = F
)

#' @title TensorFlowModel Class
#' @description A ``FrameworkModel`` implementation for inference with TensorFlow Serving.
#' @export
TensorFlowModel = R6Class("TensorFlowModel",
  inherit = sagemaker.mlcore::FrameworkModel,
  public = list(

    #' @field LOG_LEVEL_PARAM_NAME
    #' logging level
    LOG_LEVEL_PARAM_NAME = "SAGEMAKER_TFS_NGINX_LOGLEVEL",

    #' @field LOG_LEVEL_MAP
    #' logging level map
    LOG_LEVEL_MAP = list(
      "10"="debug",
      "20"="info",
      "30"="warn",
      "40"="error",
      "50"="crit"
      ),

    #' @field LATEST_EIA_VERSION
    #' latest eia version supported
    LATEST_EIA_VERSION = "2.3",

    #' @description Initialize a Model.
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to model
    #'              hosting. If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param image_uri (str): A Docker image URI (default: None). If not specified, a
    #'              default image for TensorFlow Serving will be used. If
    #'              ``framework_version`` is ``None``, then ``image_uri`` is required.
    #'              If also ``None``, then a ``ValueError`` will be raised.
    #' @param framework_version (str): Optional. TensorFlow Serving version you
    #'              want to use. Defaults to ``None``. Required unless ``image_uri`` is
    #'              provided.
    #' @param container_log_level (int): Log level to use within the container
    #'              (default: logging.ERROR). Valid values are defined in the Python
    #'              logging module.
    #' @param predictor_cls (callable[str, sagemaker.session.Session]): A function
    #'              to call to create a predictor with an endpoint name and
    #'              SageMaker ``Session``. If specified, ``deploy()`` returns the
    #'              result of invoking this function on the created endpoint name.
    #' @param ... : Keyword arguments passed to the superclass
    #'              :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
    #'              superclass :class:`~sagemaker.model.Model`.
    #'              .. tip::
    #'              You can find additional parameters for initializing this class at
    #'              :class:`~sagemaker.model.FrameworkModel` and
    #'              :class:`~sagemaker.model.Model`.
    initialize = function(model_data,
                          role,
                          entry_point=NULL,
                          image_uri=NULL,
                          framework_version=NULL,
                          container_log_level=NULL,
                          predictor_cls=TensorFlowPredictor,
                          ...){

      self$framework_version = framework_version

      super$initialize(
        model_data=model_data,
        role=role,
        image_uri=image_uri,
        predictor_cls=predictor_cls,
        entry_point=entry_point,
        ...)

      attr(self, "_framework_name") = "tensorflow"

      if (is.null(framework_version) && is.null(image_uri))
        ValueError$new(
          "Both framework_version and image_uri were NULL. ",
          "Either specify framework_version or specify image_uri."
        )

      self$.container_log_level = container_log_level
    },

    #' @description Creates a model package for creating SageMaker models or listing on Marketplace.
    #' @param content_types (list): The supported MIME types for the input data.
    #' @param response_types (list): The supported MIME types for the output data.
    #' @param inference_instances (list): A list of the instance types that are used to
    #'              generate inferences in real-time.
    #' @param transform_instances (list): A list of the instance types on which a transformation
    #'              job can be run or on which an endpoint can be deployed.
    #' @param model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
    #'              using `model_package_name` makes the Model Package un-versioned (default: None).
    #' @param model_package_group_name (str): Model Package Group name, exclusive to
    #'              `model_package_name`, using `model_package_group_name` makes the Model Package
    #'              versioned (default: None).
    #' @param image_uri (str): Inference image uri for the container. Model class' self.image will
    #'              be used if it is None (default: None).
    #' @param model_metrics (ModelMetrics): ModelMetrics object (default: None).
    #' @param metadata_properties (MetadataProperties): MetadataProperties object (default: None).
    #' @param marketplace_cert (bool): A boolean value indicating if the Model Package is certified
    #'              for AWS Marketplace (default: False).
    #' @param approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
    #'              or "PendingManualApproval" (default: "PendingManualApproval").
    #' @param description (str): Model Package description (default: None).
    #' @return str: A string of SageMaker Model Package ARN.
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
                        description=NULL){
      instance_type = inference_instances[[1]]
      private$.init_sagemaker_session_if_does_not_exist(instance_type)

      if (!is.null(image_uri))
        self$image_uri = image_uri
      if (is.null(self$image_uri))
        self$image_uri = self$serving_image_uri(
          region_name=self$sagemaker_session$paw_region_name,
          instance_type=instance_type)

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
        description)
      )
    },

    #' @description Deploy a Tensorflow ``Model`` to a SageMaker ``Endpoint``.
    #' @param initial_instance_count (int): The initial number of instances to run
    #'              in the ``Endpoint`` created from this ``Model``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge', or 'local' for local mode.
    #' @param serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
    #'              serializer object, used to encode data for an inference endpoint
    #'              (default: None). If ``serializer`` is not None, then
    #'              ``serializer`` will override the default serializer. The
    #'              default serializer is set by the ``predictor_cls``.
    #' @param deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
    #'              deserializer object, used to decode data from an inference
    #'              endpoint (default: None). If ``deserializer`` is not None, then
    #'              ``deserializer`` will override the default deserializer. The
    #'              default deserializer is set by the ``predictor_cls``.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              deploy this model for model loading and inference, for example,
    #'              'ml.eia1.medium'. If not specified, no Elastic Inference
    #'              accelerator will be attached to the endpoint. For more
    #'              information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param endpoint_name (str): The name of the endpoint to create (Default:
    #'              NULL). If not specified, a unique endpoint name will be created.
    #' @param tags (List[dict[str, str]]): The list of tags to attach to this
    #'              specific endpoint.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              data on the storage volume attached to the instance hosting the
    #'              endpoint.
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              this model completes (default: True).
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @param update_endpoint : Placeholder
    #' @param serverless_inference_config (ServerlessInferenceConfig):
    #'              Specifies configuration related to serverless endpoint. Use this configuration
    #'              when trying to create serverless endpoint and make serverless inference. If
    #'              empty object passed through, we will use pre-defined values in
    #'              ``ServerlessInferenceConfig`` class to deploy serverless endpoint (default: None)
    #' @return callable[string, sagemaker.session.Session] or None: Invocation of
    #'              ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
    #'              is not None. Otherwise, return None.
    deploy = function(initial_instance_count=NULL,
                      instance_type=NULL,
                      serializer=NULL,
                      deserializer=NULL,
                      accelerator_type=NULL,
                      endpoint_name=NULL,
                      tags=NULL,
                      kms_key=NULL,
                      wait=TRUE,
                      data_capture_config=NULL,
                      update_endpoint=NULL,
                      serverless_inference_config=NULL){
      if (!is.null(accelerator_type) && !private$.eia_supported()){
        msg = sprintf("The TensorFlow version %s doesn't support EIA.", self$framework_version)
        AttributeError$new(msg)
      }
      return(super$deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        serializer=serializer,
        deserializer=deserializer,
        accelerator_type=accelerator_type,
        endpoint_name=endpoint_name,
        tags=tags,
        kms_key=kms_key,
        wait=wait,
        data_capture_config=data_capture_config,
        serverless_inference_config=serverless_inference_config,
        update_endpoint=update_endpoint
        )
      )
    },

    #' @description Prepare the container definition.
    #' @param instance_type : Instance type of the container.
    #' @param accelerator_type : Accelerator type, if applicable.
    #' @return A container definition for deploying a ``Model`` to an ``Endpoint``.
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
      if (is.null(self$image_uri) && is.null(instance_type))
        ValueError$new(
          "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
        )
      image_uri = private$.get_image_uri(instance_type, accelerator_type)
      env = private$.get_container_env()

      if (!is.null(self$entry_point)){
        key_prefix = model_code_key_prefix(
          self$key_prefix, self$name, image_uri
        )

        bucket = self$bucket %||% self$sagemaker_session$default_bucket()
        model_data = s3_path_join("s3://", bucket, key_prefix, "model.tar.gz")

        repack_model(
          self$entry_point,
          self$source_dir,
          self$dependencies,
          self$model_data,
          model_data,
          self$sagemaker_session,
          kms_key=self$model_kms_key
        )
      } else {
        model_data = self$model_data
      }
      return(container_def(image_uri, model_data, env))
    },

    #' @description Create a URI for the serving image.
    #' @param region_name (str): AWS region where the image is uploaded.
    #' @param instance_type (str): SageMaker instance type. Used to determine device type
    #'              (cpu/gpu/family-specific optimized).
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #' @param model (default: None). For example, 'ml.eia1.medium'.
    #' @return str: The appropriate image URI based on the given parameters.
    serving_image_uri = function(){
      return(private$.get_image_uri(instance_type=instance_type, accelerator_type=accelerator_type))
    }
  ),
  private = list(
    # Return true if TF version is EIA enabled
    .eia_supported = function(){
      framework_version = package_version(self$framework_version)
      return(
        framework_version != package_version("2.1") &&
        framework_version !=package_version("2.2") &&
        package_version(framework_version) <= package_version(LATEST_EIA_VERSION)
      )
    },

    .get_container_env = function(){
      if (is.null(private$.container_log_level))
        return(self$env)

      if (!(private$.container_log_level %in% self$LOG_LEVEL_MAP)){
        LOGGER$warn("ignoring invalid container log level: %s", private$.container_log_level)
        return(self$env)
      }
      env = list(self$env)
      env[[self$LOG_LEVEL_PARAM_NAME]] = self$LOG_LEVEL_MAP[[as.character(private$.container_log_level)]]
      return(env)
    },

    .get_image_uri = function(instance_type, accelerator_type = NULL){
      if (!is.null(self$image_uri))
        return(self$image_uri)
      return(sagemaker.core::ImageUris$new()$retrieve(
        attr(self, "_framework_name"),
        self$sagemaker_session$paws_region_name,
        version=self$framework_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        image_scope="inference"
        )
      )
    }
  ),
  lock_objects = F
)
