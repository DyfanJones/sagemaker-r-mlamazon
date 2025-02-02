% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/chainer_estimator.R
\name{Chainer}
\alias{Chainer}
\title{Chainer Class}
\description{
Handle end-to-end training and deployment of custom Chainer code.
}
\section{Super classes}{
\code{\link[sagemaker.mlcore:EstimatorBase]{sagemaker.mlcore::EstimatorBase}} -> \code{\link[sagemaker.mlcore:Framework]{sagemaker.mlcore::Framework}} -> \code{Chainer}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{.use_mpi}}{Entry point is run as an MPI script.}

\item{\code{.num_processes}}{Total number of processes to run the entry
point with}

\item{\code{.process_slots_per_host}}{The number of processes that can run
on each instance.}

\item{\code{.additional_mpi_options}}{String of options to the 'mpirun' command used
to run the entry point.}

\item{\code{.module}}{mimic python module}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{Chainer$new()}}
\item \href{#method-hyperparameters}{\code{Chainer$hyperparameters()}}
\item \href{#method-create_model}{\code{Chainer$create_model()}}
\item \href{#method-clone}{\code{Chainer$clone()}}
}
}
\if{html}{
\out{<details ><summary>Inherited methods</summary>}
\itemize{
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="compile_model">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-compile_model}{\code{sagemaker.mlcore::EstimatorBase$compile_model()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="delete_endpoint">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-delete_endpoint}{\code{sagemaker.mlcore::EstimatorBase$delete_endpoint()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="deploy">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-deploy}{\code{sagemaker.mlcore::EstimatorBase$deploy()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="describe">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-describe}{\code{sagemaker.mlcore::EstimatorBase$describe()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="disable_profiling">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-disable_profiling}{\code{sagemaker.mlcore::EstimatorBase$disable_profiling()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="enable_default_profiling">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-enable_default_profiling}{\code{sagemaker.mlcore::EstimatorBase$enable_default_profiling()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="enable_network_isolation">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-enable_network_isolation}{\code{sagemaker.mlcore::EstimatorBase$enable_network_isolation()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="fit">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-fit}{\code{sagemaker.mlcore::EstimatorBase$fit()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="format">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-format}{\code{sagemaker.mlcore::EstimatorBase$format()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="get_vpc_config">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-get_vpc_config}{\code{sagemaker.mlcore::EstimatorBase$get_vpc_config()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="help">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-help}{\code{sagemaker.mlcore::EstimatorBase$help()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="latest_job_debugger_artifacts_path">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-latest_job_debugger_artifacts_path}{\code{sagemaker.mlcore::EstimatorBase$latest_job_debugger_artifacts_path()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="latest_job_profiler_artifacts_path">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-latest_job_profiler_artifacts_path}{\code{sagemaker.mlcore::EstimatorBase$latest_job_profiler_artifacts_path()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="latest_job_tensorboard_artifacts_path">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-latest_job_tensorboard_artifacts_path}{\code{sagemaker.mlcore::EstimatorBase$latest_job_tensorboard_artifacts_path()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="logs">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-logs}{\code{sagemaker.mlcore::EstimatorBase$logs()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="prepare_workflow_for_training">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-prepare_workflow_for_training}{\code{sagemaker.mlcore::EstimatorBase$prepare_workflow_for_training()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="register">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-register}{\code{sagemaker.mlcore::EstimatorBase$register()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="rule_job_summary">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-rule_job_summary}{\code{sagemaker.mlcore::EstimatorBase$rule_job_summary()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="update_profiler">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-update_profiler}{\code{sagemaker.mlcore::EstimatorBase$update_profiler()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="EstimatorBase" data-id="wait">}\href{../../sagemaker.mlcore/html/EstimatorBase.html#method-wait}{\code{sagemaker.mlcore::EstimatorBase$wait()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Framework" data-id=".prepare_for_training">}\href{../../sagemaker.mlcore/html/Framework.html#method-.prepare_for_training}{\code{sagemaker.mlcore::Framework$.prepare_for_training()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Framework" data-id="attach">}\href{../../sagemaker.mlcore/html/Framework.html#method-attach}{\code{sagemaker.mlcore::Framework$attach()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Framework" data-id="training_image_uri">}\href{../../sagemaker.mlcore/html/Framework.html#method-training_image_uri}{\code{sagemaker.mlcore::Framework$training_image_uri()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="sagemaker.mlcore" data-topic="Framework" data-id="transformer">}\href{../../sagemaker.mlcore/html/Framework.html#method-transformer}{\code{sagemaker.mlcore::Framework$transformer()}}\out{</span>}
}
\out{</details>}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
This ``Estimator`` executes an Chainer script in a managed Chainer
             execution environment, within a SageMaker Training Job. The managed
             Chainer environment is an Amazon-built Docker container that executes
             functions defined in the supplied ``entry_point`` Python script.
             Training is started by calling
             :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
             After training is complete, calling
             :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
             SageMaker endpoint and returns an
             :class:`~sagemaker.amazon.chainer.model.ChainerPredictor` instance that
             can be used to perform inference against the hosted model.
             Technical documentation on preparing Chainer scripts for SageMaker
             training and using the Chainer Estimator is available on the project
             home-page: https://github.com/aws/sagemaker-python-sdk
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Chainer$new(
  entry_point,
  use_mpi = NULL,
  num_processes = NULL,
  process_slots_per_host = NULL,
  additional_mpi_options = NULL,
  source_dir = NULL,
  hyperparameters = NULL,
  framework_version = NULL,
  py_version = NULL,
  image_uri = NULL,
  ...
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{entry_point}}{(str): Path (absolute or relative) to the Python source
file which should be executed as the entry point to training.
If ``source_dir`` is specified, then ``entry_point``
must point to a file located at the root of ``source_dir``.}

\item{\code{use_mpi}}{(bool): If true, entry point is run as an MPI script. By
default, the Chainer Framework runs the entry point with
'mpirun' if more than one instance is used.}

\item{\code{num_processes}}{(int): Total number of processes to run the entry
point with. By default, the Chainer Framework runs one process
per GPU (on GPU instances), or one process per host (on CPU
instances).}

\item{\code{process_slots_per_host}}{(int): The number of processes that can run
on each instance. By default, this is set to the number of GPUs
on the instance (on GPU instances), or one (on CPU instances).}

\item{\code{additional_mpi_options}}{(str): String of options to the 'mpirun'
command used to run the entry point. For example, '-X
NCCL_DEBUG=WARN' will pass that option string to the mpirun
command.}

\item{\code{source_dir}}{(str): Path (absolute or relative) to a directory with
any other training source code dependencies aside from the entry
point file (default: None). Structure within this directory are
preserved when training on Amazon SageMaker.}

\item{\code{hyperparameters}}{(dict): Hyperparameters that will be used for
training (default: None). The hyperparameters are made
accessible as a dict[str, str] to the training code on
SageMaker. For convenience, this accepts other types for keys
and values, but ``str()`` will be called to convert them before
training.}

\item{\code{framework_version}}{(str): Chainer version you want to use for
executing your model training code. Defaults to ``None``. Required unless
``image_uri`` is provided. List of supported versions:
https://github.com/aws/sagemaker-python-sdk#chainer-sagemaker-estimators.}

\item{\code{py_version}}{(str): Python version you want to use for executing your
model training code. Defaults to ``None``. Required unless ``image_uri``
is provided.}

\item{\code{image_uri}}{(str): If specified, the estimator will use this image
for training and hosting, instead of selecting the appropriate
SageMaker official image based on framework_version and
py_version. It can be an ECR url or dockerhub image and tag.
Examples
* ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
* ``custom-image:latest``
If ``framework_version`` or ``py_version`` are ``None``, then
``image_uri`` is required. If also ``None``, then a ``ValueError``
will be raised.}

\item{\code{...}}{: Additional kwargs passed to the
:class:`~sagemaker.estimator.Framework` constructor.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-hyperparameters"></a>}}
\if{latex}{\out{\hypertarget{method-hyperparameters}{}}}
\subsection{Method \code{hyperparameters()}}{
Return hyperparameters used by your custom Chainer code during
             training.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Chainer$hyperparameters()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-create_model"></a>}}
\if{latex}{\out{\hypertarget{method-create_model}{}}}
\subsection{Method \code{create_model()}}{
Create a SageMaker ``ChainerModel`` object that can be deployed to an
             ``Endpoint``.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Chainer$create_model(
  model_server_workers = NULL,
  role = NULL,
  vpc_config_override = "VPC_CONFIG_DEFAULT",
  entry_point = NULL,
  source_dir = NULL,
  dependencies = NULL,
  ...
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{model_server_workers}}{(int): Optional. The number of worker processes
used by the inference server. If None, server will use one
worker per vCPU.}

\item{\code{role}}{(str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
which is also used during transform jobs. If not specified, the
role from the Estimator will be used.}

\item{\code{vpc_config_override}}{(dict[str, list[str]]): Optional override for VpcConfig set on
the model. Default: use subnets and security groups from this Estimator.
* 'Subnets' (list[str]): List of subnet ids.
* 'SecurityGroupIds' (list[str]): List of security group ids.}

\item{\code{entry_point}}{(str): Path (absolute or relative) to the local Python source file which
should be executed as the entry point to training. If ``source_dir`` is specified,
then ``entry_point`` must point to a file located at the root of ``source_dir``.
If not specified, the training entry point is used.}

\item{\code{source_dir}}{(str): Path (absolute or relative) to a directory with any other serving
source code dependencies aside from the entry point file.
If not specified, the model source directory from training is used.}

\item{\code{dependencies}}{(list[str]): A list of paths to directories (absolute or relative) with
any additional libraries that will be exported to the container.
If not specified, the dependencies from training are used.
This is not supported with "local code" in Local Mode.}

\item{\code{...}}{: Additional kwargs passed to the ChainerModel constructor.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
sagemaker.chainer.model.ChainerModel: A SageMaker ``ChainerModel``
             object. See :func:`~sagemaker.chainer.model.ChainerModel` for full details.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Chainer$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
