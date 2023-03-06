# AML Optimization and Profiling Recipe

## Overview

The process of accelerating models, deploying models to a competent platform and tuning deployment parameters to make the best use of compute resources and reduce cost to reach the desired performance SLA (e.g. latency, throughput) is not only necessary but also vital for the production of machine learning services. This recipe is aiming at providing a one-stop experience for users to execute the complete process from optimization to profiling on azureml.

AML Optimization and Profiling (Preview) provides fully managed experience that makes it easy to benchmark your model performance.

* Use the benchmarking tool of your choice.

* Easy to use CLI experience.
  
* Support for CI/CD MLOps pipelines to automate profiling.
  
* Thorough performance report containing latency percentiles and resource utilization metrics.

## A brief introduction on the aml optimization and profiling tools

The aml optimization and profiling recipe is currently consisted of the following 5 tools:

* `aml-wrk-profiler`: A profiler based on "wrk". "wrk" is a modern HTTP benchmarking tool capable of generating significant load when run on a single multi-core CPU. It combines a multithreaded design with scalable event notification systems such as epoll and kqueue. For detailed info please refer to this link: https://github.com/wg/wrk.

* `aml-wrk2-profiler`: A profiler based on "wrk2". "wrk2" is "wrk" modified to produce a constant throughput load, and accurate latency details to the high 9s (i.e. can produce accuracy 99.9999% if run long enough). In addition to wrk's arguments, wrk2 takes a throughput argument (in total requests per second) via either the --rate or -R parameters (default is 1000). For detailed info please refer to this link: https://github.com/giltene/wrk2.

* `aml-labench-profiler`: A profiler based on "LaBench". "LaBench" (for LAtency BENCHmark) is a tool that measures latency percentiles of HTTP GET or POST requests under very even and steady load. For detailed info please refer to this link: https://github.com/microsoft/LaBench.

* `aml-olive-optimizer`: An optimizer based on "OLive". "OLive" (for ONNX Runtime(ORT) Go Live)

* `aml-online-endpoints-deployer`: A deployer that deploys models as aml online-endpoints.

* `aml-online-endpoints-deleter`: A deleter that deletes aml online-endpoints and online-deployments.
  
## Prerequisites

* Azure subscription. If you don't have an Azure subscription, sign up to try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/) today.

* Azure CLI and ML extension. For more information, see [Install, set up, and use the CLI (v2) (preview)](how-to-configure-cli.md).

Prepare model files:

## Get started

Please follow this [example](https://github.com/Azure/azureml-examples/blob/xiyon/mir-profiling/cli/how-to-profile-online-endpoint.sh) and get started with the model profiling experience.

### Step 1: Optimize your model

#### Create a compute to host the optimizer

#### Create an optimization job

Prepare an optimization configuration json file. Below is a sample configuration file.

```json
{
  "version": "1.0",
  "optimizer_config": {
    "inputs_spec": {"attention_mask": [1, 128], "input_ids": [1, 128]},
    "model_file_path": "distilbert-base-cased-distilled-squad.onnx",
    "providers_list": ["cpu"],
    "inter_thread_num_list": [1],
    "intra_thread_num_list": [1,2,3,4,5,6],
    "quantization_enabled": false,
    "trt_fp16_enabled": false,
    "transformer_enabled": true,
    "transformer_args": "--model_type bert --num_heads 12 --hidden_size 768"
  }
}
```

Below is a template yaml file that defines an olive optimization job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python -m aml_olive_optimizer --config_path ${{inputs.config}} --model_path ${{inputs.model}}
experiment_name: demo-optimization-job
name: optimize-demo-model
tags: 
  optimizationTool: olive
environment:
  image: mcr.microsoft.com/azureml/aml-olive-optimizer-cpu:latest
compute: azureml:optimizerCompute
inputs:
  config:
    type: uri_file
    path: config.yml
  model:
    type: uri_folder
    path: ../distilbert_model/model
```

#### Understand and download job output

### Step 2: Deploy an online-endpoint with the optimized model and optimized parameters

#### Create a compute to host the deployer

#### Create an online-endpoints deployer job

Prepare an online-endpoint configuration yaml file. Below is a sample configuration file.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: distilbert-optimized-endpt
auth_mode: key
```

Prepare an online-deployment configuration yaml file. Below is a sample configuration file.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: distilbert-optimized-dep
endpoint_name: distilbert-optimized-endpt
model:
  name: optimized-distilbert-model
  version: 2
  path: <% MODEL_FOLDER_PATH %>
code_configuration:
  code: <% CODE_FOLDER_PATH %>
  scoring_script: score.py
environment: 
  conda_file: <% ENVIRONMENT_FOLDER_PATH %>/conda.yml
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
instance_type: Standard_F8s_v2
instance_count: 1
```

Below is a sample yaml file that defines a wrk profiling job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command:
  python -m aml_online_endpoints_deployer --endpoint_yaml_path ${{inputs.endpoint}} --deployment_yaml_path ${{inputs.deployment}} --model_folder_path ${{inputs.model}} --environment_folder_path ${{inputs.environment}} --code_folder_path ${{inputs.code}} --optimized_parameters_path ${{inputs.optimized_parameters}}
experiment_name: demo-deployment-jobs
environment:
  image: profilervalidationacr.azurecr.io/aml-online-endpoints-deployer:20230401.80737331
name: deploy-optimized-1
tags:
  endpoint: distilbert-optimized-endpt
  deployment: distilbert-optimized-dep
compute: azureml:deploymentTest
inputs:
  endpoint:
    type: uri_file
    path: endpoint.yml
  deployment: 
    type: uri_file
    path: deployment.yml
  model: 
    type: uri_folder
    path: ../distilbert_model/optimized_model
  environment: 
    type: uri_folder
    path: ../distilbert_model/environment
  code:
    type: uri_folder    
    path: ../distilbert_model/code
  optimized_parameters:
    type: uri_file
    path: optimized_parameters.json
```

#### Understand and download job output

### Step 3: Profile your online-endpoint

#### Create a compute to host the profiler

#### Create a profiling job

Prepare a profiling configuration json file. Below is a sample configuration file.

```json
{
  "version": 1.0,
  "profiler_config": {
    "duration_sec": 300,
    "connections": 1
  }
}
```

Below is a sample yaml file that defines a wrk profiling job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python -m aml_wrk_profiler --config_path ${{inputs.config}} --scoring_target_path ${{inputs.scoring_target}} --payload_path ${{inputs.payload}}
experiment_name: demo-profiling-jobs
name: profile-optimized
environment:
  image: profilervalidationacr.azurecr.io/aml-wrk-profiler:20230401.80751377
tags: 
  deployment: distilbert-optimized-dep
compute: azureml:profilingTest
inputs:
  config:
    type: uri_file
    path: config.json
  scoring_target:
    type: uri_file
    path: deployment_settings.json
  payload:
    type: uri_file
    path: payload.jsonl
```

#### Understand and download job output

### Step 4: Delete your online-endpoint

#### Create a compute to host the deleter

#### Create an online-endpoints deleter job

Below is a sample yaml file that defines a wrk profiling job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command:
  python -m aml_online_endpoints_deleter --config_path ${{inputs.config}} --delete_endpoint ${{inputs.delete_endpoint}}
experiment_name: deletion-job
environment:
  image: profilervalidationacr.azurecr.io/aml-online-endpoints-deleter:20230401.80737331
name: delete-bertsquad-optimized-4
tags:
  action: delete
  endpoint: xiyononnx-endpt
  deployment: xiyononnx-optimized-dep-4
compute: azureml:deploymentTest
inputs:
  config:
    type: uri_file
    path: deployment_settings_optimized.json
  delete_endpoint: False
```

#### Understand job output



# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Contact us

For any questions, bugs and requests of new features, please contact us at [miroptprof@microsoft.com](mailto:miroptprof@microsoft.com)

Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.