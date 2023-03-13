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

You will need a compute to host the optimizer, run the optimization program and generate final reports. We would suggest you to use the same sku type that you intend to deploy your model with.

  ```bash
  az ml compute create --name $OPTIMIZER_COMPUTE_NAME --size $INFERENCE_SERVICE_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
  ```

#### Create an optimization job

Prepare an optimization configuration json file. Below is a sample configuration file. For detailed configuration definitions, please refer to [Job yaml syntax and configuration definitions](#Job yaml syntax and configuration definitions).

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
name: $OPTIMIZER_JOB_NAME
tags: 
  optimizationTool: olive
environment:
  image: mcr.microsoft.com/azureml/aml-olive-optimizer:20230306.2_cpu
compute: azureml:$OPTIMIZER_COMPUTE_NAME
inputs:
  config:
    type: uri_file
    path: config.json
  model:
    type: uri_folder
    path: ../distilbert_model/model
```

You may create this olive optimizer job with the following command:

  ```bash
  az ml job create --name $OPTIMIZER_JOB_NAME --file optimizer_job.yml
  ```

#### Understand and download job output

The olive optimizer job will generate 3 output files.

* olive_result.json: This is the result file generated by OLive, it contains detailed profiling results for all optimization options.
* optimized_model.onnx: This is the optimized model file generated by OLive.
* optimized_parameters.json: This file contains the optimized parameters for the `best_test_name` extracted from the olive_result.json file.

You may download the optimized_parameters.json file and optimized_model.onnx file with the following command, they can be used as inputs of the online-endpoints deployer job.

  ```bash
  az ml job download --name $OPTIMIZER_JOB_NAME --all --download-path $OPTIMIZER_DOWNLOAD_FOLDER
  ```

### Step 2: Deploy an online-endpoint with the optimized model and optimized parameters

#### Create a compute to host the deployer

* You will need a compute to host the deployer, run the online-endpoints deployment program and generate final reports. You may choose any sku type for this job.

  ```bash
  az ml compute create --name $DEPLOYER_COMPUTE_NAME --size $DEPLOYER_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
  ```

* Create proper role assignment for accessing online endpoint resources. The compute needs to have contributor role to the machine learning workspace. For more information, see [Assign Azure roles using Azure CLI](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-cli).

  ```bash
  compute_info=`az ml compute show --name $DEPLOYER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
  workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
  identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
  az role assignment create --role Contributor --assignee-object-id $identity_object_id --assignee-principal-type ServicePrincipal --scope $workspace_resource_id
  if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $DEPLOYER_COMPUTE_NAME" && exit 1; fi
  ```

#### Create an online-endpoints deployer job

Prepare an online-endpoint configuration yaml file. Below is a sample configuration file.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: $ENDPOINT_NAME
auth_mode: key
```

Prepare an online-deployment configuration yaml file. Below is a sample configuration file.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $DEPLOYMENT_NAME
endpoint_name: $ENDPOINT_NAME
model:
  name: optimized-distilbert-model
  version: 1
  path: <% MODEL_FOLDER_PATH %>
code_configuration:
  code: <% CODE_FOLDER_PATH %>
  scoring_script: score.py
environment: 
  conda_file: <% ENVIRONMENT_FOLDER_PATH %>/conda.yml
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
instance_type: $INFERENCE_SERVICE_COMPUTE_SIZE
instance_count: 1
```

Below is a sample yaml file that defines a wrk profiling job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command:
  python -m aml_online_endpoints_deployer --endpoint_yaml_path ${{inputs.endpoint}} --deployment_yaml_path ${{inputs.deployment}} --model_folder_path ${{inputs.model}} --environment_folder_path ${{inputs.environment}} --code_folder_path ${{inputs.code}} --optimized_parameters_path ${{inputs.optimized_parameters}}
experiment_name: deployment-demo-jobs
environment:
  image: mcr.microsoft.com/azureml/aml-online-endpoints-deployer:20230306.1
name: $DEPLOYER_JOB_NAME
tags:
  endpoint: $ENDPOINT_NAME
  deployment: $DEPLOYMENT_NAME
compute: azureml:$DEPLOYER_COMPUTE_NAME
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

You may create this online-endpoints deployer job with the following command:

  ```bash
  az ml job create --name $DEPLOYER_JOB_NAME --file deployer_job.yml
  ```

#### Understand and download job output

The online-endpoints deployer job will generate 1 output file.

* deployment_settings.json: This file contains the detailed online-deployment information.

You may download the deployment_settings.json file with the following command, this file can later be used as an input of the profiler job.

  ```bash
  az ml job download --name $DEPLOYER_JOB_NAME --all --download-path $DEPLOYER_DOWNLOAD_FOLDER
  ```

### Step 3: Profile your online-endpoint

#### Create a compute to host the profiler

* You will need a compute to host the profiler, run the profiling program and generate final reports. Please choose a compute SKU with proper network bandwidth (considering the inference request payload size and profiling traffic, we'd recommend Standard_F4s_v2) in the same region as the online endpoint or your model inference service.

  ```bash
  az ml compute create --name $PROFILER_COMPUTE_NAME --size $PROFILER_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
  ```

* Create proper role assignment for accessing online endpoint resources. The compute needs to have contributor role to the machine learning workspace. For more information, see [Assign Azure roles using Azure CLI](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-cli).

  ```bash
  compute_info=`az ml compute show --name $PROFILER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
  workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
  identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
  az role assignment create --role Contributor --assignee-object-id $identity_object_id --assignee-principal-type ServicePrincipal --scope $workspace_resource_id
  if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
  ```

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
experiment_name: profiling-demo-jobs
name: $PROFILER_JOB_NAME
environment:
  image: mcr.microsoft.com/azureml/aml-wrk-profiler:20230303.2
tags: 
  deployment: $DEPLOYMENT_NAME
compute: azureml:$PROFILER_COMPUTE_NAME
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

You may create this profiling job with the following command:

  ```bash
  az ml job create --file profiler_job.yml
  ```

#### Understand and download job output

The profiler job will generate 1 output file.

* report.json: This file contains detailed profiling results.

You may download the report.json file with the following command.

  ```bash
  az ml job download --name $PROFILER_JOB_NAME --all --download-path $PROFILER_DOWNLOAD_FOLDER
  ```

### Step 4: Delete your online-endpoint

#### Create a compute to host the deleter

* You will need a compute to host the deleter, run the online-endpoints deletion program and generate final reports. You may choose any sku type for this job, and you may also choose to reuse the compute for the online-endpoints deployer job.

  ```bash
  az ml compute create --name $DELETER_COMPUTE_NAME --size $DELETER_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
  ```

* Create proper role assignment for accessing online endpoint resources. The compute needs to have contributor role to the machine learning workspace. For more information, see [Assign Azure roles using Azure CLI](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-cli).

  ```bash
  compute_info=`az ml compute show --name $DELETER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
  workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
  identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
  az role assignment create --role Contributor --assignee-object-id $identity_object_id --assignee-principal-type ServicePrincipal --scope $workspace_resource_id
  if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $DELETER_COMPUTE_NAME" && exit 1; fi
  ```

#### Create an online-endpoints deleter job

Below is a sample yaml file that defines an online-endpoints deleter job.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command:
  python -m aml_online_endpoints_deleter --config_path ${{inputs.config}} --delete_endpoint ${{inputs.delete_endpoint}}
experiment_name: deletion-demo-job
environment:
  image: profilervalidationacr.azurecr.io/aml-online-endpoints-deleter:20230401.80737331
name: $DELETER_JOB_NAME
tags:
  action: delete
  endpoint: $ENDPOINT_NAME
  deployment: $DEPLOYMENT_NAME
compute: azureml:$DELETER_COMPUTE_NAME
inputs:
  config:
    type: uri_file
    path: deployment_settings.json
  delete_endpoint: True
```

You may create this online-endpoints deleter job with the following command:

  ```bash
  az ml job create --file deleter_job.yml
  ```

#### Understand job output

The deleter job won't generate any output files, but you may also use the below command for downloading all std_out logs.
  
  ```bash
  az ml job download --name $DELETER_JOB_NAME --all --download-path $DELETER_DOWNLOAD_FOLDER
  

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