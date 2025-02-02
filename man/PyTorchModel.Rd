% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pytorch_model.R
\name{PyTorchModel}
\alias{PyTorchModel}
\title{PyTorchModel class}
\description{
An PyTorch SageMaker ``Model`` that can be deployed to a SageMaker
             ``Endpoint``.
}
\section{Super classes}{
\code{\link[sagemaker.mlcore:ModelBase]{sagemaker.mlcore::ModelBase}} -> \code{\link[sagemaker.mlcore:Model]{sagemaker.mlcore::Model}} -> \code{\link[sagemaker.mlcore:FrameworkModel]{sagemaker.mlcore::FrameworkModel}} -> \code{PyTorchModel}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{.LOWEST_MMS_VERSION}}{Lowest Multi Model Server PyTorch version that can be executed}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{PyTorchModel$new()}}
\item \href{#method-register}{\code{PyTorchModel$register()}}
\item \href{#method-prepare_container_def}{\code{PyTorchModel$prepare_container_def()}}
\item \href{#method-serving_image_uri}{\code{PyTorchModel$serving_image_uri()}}
\item \href{#method-clone}{\code{PyTorchModel$clone()}}
}
}
\if{html}{
\out{<details ><summary>Inherited methods</summary>}
\itemize{
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="ModelBase" data-id="format">}\href{../../sagemaker.mlcore/html/ModelBase.html#method-format}{\code{sagemaker.mlcore::ModelBase$format()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="ModelBase" data-id="help">}\href{../../sagemaker.mlcore/html/ModelBase.html#method-help}{\code{sagemaker.mlcore::ModelBase$help()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id=".create_sagemaker_model">}\href{../../sagemaker.mlcore/html/Model.html#method-.create_sagemaker_model}{\code{sagemaker.mlcore::Model$.create_sagemaker_model()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="check_neo_region">}\href{../../sagemaker.mlcore/html/Model.html#method-check_neo_region}{\code{sagemaker.mlcore::Model$check_neo_region()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="compile">}\href{../../sagemaker.mlcore/html/Model.html#method-compile}{\code{sagemaker.mlcore::Model$compile()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="delete_model">}\href{../../sagemaker.mlcore/html/Model.html#method-delete_model}{\code{sagemaker.mlcore::Model$delete_model()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="deploy">}\href{../../sagemaker.mlcore/html/Model.html#method-deploy}{\code{sagemaker.mlcore::Model$deploy()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="enable_network_isolation">}\href{../../sagemaker.mlcore/html/Model.html#method-enable_network_isolation}{\code{sagemaker.mlcore::Model$enable_network_isolation()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="package_for_edge">}\href{../../sagemaker.mlcore/html/Model.html#method-package_for_edge}{\code{sagemaker.mlcore::Model$package_for_edge()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Model" data-id="transformer">}\href{../../sagemaker.mlcore/html/Model.html#method-transformer}{\code{sagemaker.mlcore::Model$transformer()}}\out{</span>}
}
\out{</details>}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
Initialize a PyTorchModel.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PyTorchModel$new(
  model_data,
  role,
  entry_point,
  framework_version = NULL,
  py_version = NULL,
  image_uri = NULL,
  predictor_cls = PyTorchPredictor,
  model_server_workers = NULL,
  ...
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{model_data}}{(str): The S3 location of a SageMaker model data
``.tar.gz`` file.}

\item{\code{role}}{(str): An AWS IAM role (either name or full ARN). The Amazon
SageMaker training jobs and APIs that create Amazon SageMaker
endpoints use this role to access training data and model
artifacts. After the endpoint is created, the inference code
might use the IAM role, if it needs to access an AWS resource.}

\item{\code{entry_point}}{(str): Path (absolute or relative) to the Python source
file which should be executed as the entry point to model
hosting. If ``source_dir`` is specified, then ``entry_point``
must point to a file located at the root of ``source_dir``.}

\item{\code{framework_version}}{(str): PyTorch version you want to use for
executing your model training code. Defaults to None. Required
unless ``image_uri`` is provided.}

\item{\code{py_version}}{(str): Python version you want to use for executing your
model training code. Defaults to ``None``. Required unless
``image_uri`` is provided.}

\item{\code{image_uri}}{(str): A Docker image URI (default: None). If not specified, a
default image for PyTorch will be used. If ``framework_version``
or ``py_version`` are ``None``, then ``image_uri`` is required. If
also ``None``, then a ``ValueError`` will be raised.}

\item{\code{predictor_cls}}{(callable[str, sagemaker.session.Session]): A function
to call to create a predictor with an endpoint name and
SageMaker ``Session``. If specified, ``deploy()`` returns the
result of invoking this function on the created endpoint name.}

\item{\code{model_server_workers}}{(int): Optional. The number of worker processes
used by the inference server. If None, server will use one
worker per vCPU.}

\item{\code{...}}{: Keyword arguments passed to the superclass
:class:`~sagemaker.model.FrameworkModel` and, subsequently, its
superclass :class:`~sagemaker.model.Model`.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-register"></a>}}
\if{latex}{\out{\hypertarget{method-register}{}}}
\subsection{Method \code{register()}}{
Creates a model package for creating SageMaker models or listing on Marketplace.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PyTorchModel$register(
  content_types,
  response_types,
  inference_instances,
  transform_instances,
  model_package_name = NULL,
  model_package_group_name = NULL,
  image_uri = NULL,
  model_metrics = NULL,
  metadata_properties = NULL,
  marketplace_cert = FALSE,
  approval_status = NULL,
  description = NULL,
  drift_check_baselines = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{content_types}}{(list): The supported MIME types for the input data.}

\item{\code{response_types}}{(list): The supported MIME types for the output data.}

\item{\code{inference_instances}}{(list): A list of the instance types that are used to
generate inferences in real-time.}

\item{\code{transform_instances}}{(list): A list of the instance types on which a transformation
job can be run or on which an endpoint can be deployed.}

\item{\code{model_package_name}}{(str): Model Package name, exclusive to `model_package_group_name`,
using `model_package_name` makes the Model Package un-versioned (default: None).}

\item{\code{model_package_group_name}}{(str): Model Package Group name, exclusive to
`model_package_name`, using `model_package_group_name` makes the Model Package
versioned (default: None).}

\item{\code{image_uri}}{(str): Inference image uri for the container. Model class' self.image will
be used if it is None (default: None).}

\item{\code{model_metrics}}{(ModelMetrics): ModelMetrics object (default: None).}

\item{\code{metadata_properties}}{(MetadataProperties): MetadataProperties object (default: None).}

\item{\code{marketplace_cert}}{(bool): A boolean value indicating if the Model Package is certified
for AWS Marketplace (default: False).}

\item{\code{approval_status}}{(str): Model Approval Status, values can be "Approved", "Rejected",
or "PendingManualApproval" (default: "PendingManualApproval").}

\item{\code{description}}{(str): Model Package description (default: None).}

\item{\code{drift_check_baselines}}{(DriftCheckBaselines): DriftCheckBaselines object (default: None).}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A `sagemaker.model.ModelPackage` instance.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-prepare_container_def"></a>}}
\if{latex}{\out{\hypertarget{method-prepare_container_def}{}}}
\subsection{Method \code{prepare_container_def()}}{
Return a container definition with framework configuration set in
             model environment variables.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PyTorchModel$prepare_container_def(
  instance_type = NULL,
  accelerator_type = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{instance_type}}{(str): The EC2 instance type to deploy this Model to.
For example, 'ml.p2.xlarge'.}

\item{\code{accelerator_type}}{(str): The Elastic Inference accelerator type to
deploy to the instance for loading and making inferences to the
model.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
dict[str, str]: A container definition object usable with the
             CreateModel API.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-serving_image_uri"></a>}}
\if{latex}{\out{\hypertarget{method-serving_image_uri}{}}}
\subsection{Method \code{serving_image_uri()}}{
Create a URI for the serving image.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PyTorchModel$serving_image_uri(
  region_name,
  instance_type,
  accelerator_type = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{region_name}}{(str): AWS region where the image is uploaded.}

\item{\code{instance_type}}{(str): SageMaker instance type. Used to determine device type
(cpu/gpu/family-specific optimized).}

\item{\code{accelerator_type}}{(str): The Elastic Inference accelerator type to
deploy to the instance for loading and making inferences to the
model.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
str: The appropriate image URI based on the given parameters
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PyTorchModel$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
