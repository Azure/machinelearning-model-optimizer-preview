# Azure Machine Learning Model Optimizer

## Overview

The process of accelerating models, deploying models to a competent platform and tuning deployment parameters to make the best use of compute resources and reduce cost to reach the desired performance SLA (e.g. latency, throughput) is not only necessary but also vital for the production of machine learning services. This recipe is aiming at providing a one-stop experience for users to execute the complete process from optimization to profiling on Azure Machine Learning.

Azure Machine Learning Model Optimizer (preview) provides fully managed experience that makes it easy to benchmark your model performance.

* Use the benchmarking tool of your choice.

* Easy to use CLI experience.
  
* Support for CI/CD MLOps pipelines to automate profiling.
  
* Thorough performance report containing latency percentiles and resource utilization metrics.

## A brief introduction on the Azure Machine Learning Model Optimizer

The Azure Machine Learning Model Optimizer currently consistes of the following 5 tools:

* `aml-olive-optimizer`: An optimizer based on "OLive". "OLive" (for ONNX Runtime(ORT) Go Live) is a python package that automates the process of accelerating models with ONNX Runtime(ORT). For detailed info please refer to this [link](https://github.com/microsoft/OLive).

* `aml-wrk-profiler`: A profiler based on "wrk". "wrk" is a modern HTTP benchmarking tool capable of generating significant load when run on a single multi-core CPU. It combines a multithreaded design with scalable event notification systems such as epoll and kqueue. For detailed info please refer to this [link]( https://github.com/wg/wrk).

* `aml-wrk2-profiler`: A profiler based on "wrk2". "wrk2" is "wrk" modified to produce a constant throughput load, and accurate latency details to the high 9s (i.e. can produce accuracy 99.9999% if run long enough). In addition to wrk's arguments, wrk2 takes a throughput argument (in total requests per second) via either the --rate or -R parameters (default is 1000). For detailed info please refer to this [link](https://github.com/giltene/wrk2).

* `aml-labench-profiler`: A profiler based on "LaBench". "LaBench" (for LAtency BENCHmark) is a tool that measures latency percentiles of HTTP GET or POST requests under very even and steady load. For detailed info please refer to this [link](https://github.com/microsoft/LaBench).

* `aml-online-endpoints-deployer`: A deployer that deploys models as azureml online-endpoints. The deployer uses `az cli` and `ml` extension for the deployment job. For detailed info regarding using `az cli` for deploying online-endpoints, please refer to this [link](https://learn.microsoft.com/en-us/cli/azure/ml/online-endpoint?view=azure-cli-latest#az-ml-online-endpoint-create).
  
* `aml-online-endpoints-deleter`: A deleter that deletes azureml online-endpoints and online-deployments. The deleter also uses `az cli` and `ml` extension for the deletion job. For detailed info regarding using `az cli` for deleting online-endpoints, please refer to this [link](https://learn.microsoft.com/en-us/cli/azure/ml/online-endpoint?view=azure-cli-latest#az-ml-online-endpoint-delete).
  
## Prerequisites

* Azure subscription. If you don't have an Azure subscription, sign up to try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/) today.

* Azure CLI and ML extension. For more information, see [Install, set up, and use the CLI (v2) (preview)](how-to-configure-cli.md).

## Get started

Please follow this [example](example/azureml_optimization_and_profiling.ipynb) and get started with the Azure Machine Learning Model Optimizer experience.

### Step 1: Optimize your model

#### Create a compute to host the optimizer

You will need a compute to host the optimizer, run the optimization program and generate final reports. We would suggest you to use the same sku type that you intend to deploy your model with.

  ```bash
  az ml compute create --name $OPTIMIZER_COMPUTE_NAME --size $INFERENCE_SERVICE_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
  ```

#### Create an optimization job

Prepare an optimization configuration json file. Below is a sample configuration file. For detailed configuration definitions, please refer to [OLive Optimizer Configuration](#olive-optimizer-configuration).

```json
{
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "hf_config": {
                "model_name": "Intel/bert-base-uncased-mrpc",
                "task": "text-classification",
                "dataset": {
                    "data_name":"glue",
                    "subset": "mrpc",
                    "split": "validation",
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"],
                    "batch_size": 1
                }
            },
            "io_config" : {
                "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                "input_shapes": [[1, 128], [1, 128], [1, 128]],
                "input_types": ["int64", "int64", "int64"],
                "output_names": ["output"],
                "dynamic_axes": {
                    "input_ids": {"0": "batch_size", "1": "seq_length"},
                    "attention_mask": {"0": "batch_size", "1": "seq_length"},
                    "token_type_ids": {"0": "batch_size", "1": "seq_length"}
                }
            }
        }
    },
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "config": {
                "accelerators": ["CPU"]
            }
        }
    },
    "evaluators": {
        "common_evaluator": {
            "metrics":[
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "sub_types": [
                        {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}}
                    ]
                },
                {
                    "name": "latency",
                    "type": "latency",
                    "sub_types": [
                        {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}}
                    ]
                }
            ]
        }
    },
    "passes": {
        "conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 13
            }
        },
        "transformers_optimization": {
            "type": "OrtTransformersOptimization",
            "config": {
                "model_type": "bert",
                "num_heads": 12,
                "hidden_size": 768,
                "float16": false
            }
        },
        "perf_tuning": {
            "type": "OrtPerfTuning",
            "config": {
                "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                "input_shapes": [[1, 128], [1, 128], [1, 128]],
                "input_types": ["int64", "int64", "int64"]
            }
        }
    },
    "engine": {
        "log_severity_level": 0,
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "tpe",
            "search_algorithm_config": {
                "num_samples": 3,
                "seed": 0
            }
        },
        "evaluator": "common_evaluator",
        "host": "local_system",
        "target": "local_system",
        "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        "clean_cache": true,
        "cache_dir": "cache"
    }
}
```
Optimization job use olive configuration directy, but still have some limitations:

* Systems Information: only support `LocalSystem` now,  if config include system type `AzureML` and/or `Docker`, will fail.

* Engine Information:

  1. `search_strategy`: set `output_model_num` to 1 if config not include. If user want to have more than one best candidate models, please set `output_model_num` in advance.
       
  2. `packaging_config`: set as default packaging config, and don's support override.
       
  ```json
   "packaging_config": {
      "type": "Zipfile",
      "name": "OutputModels"
   },
  ```

  3. `plot_pareto_frontier`: set as `true`, which means will plot the pareto frontier of the search results.

  4. `output_dir`: set to job's default artifact output files `./outputs`. All default output files will put here. Suggest user set outputs `optimized_parameters` and `optimized_model` in job template.

Below is a template yaml file that defines an olive optimization job. For detailed info regarding how to construct a command job yaml file, see [AzureML Job Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)

**Template1**:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python -m aml_olive_optimizer
  --config_path ${{inputs.config}}
  --code ${{inputs.code_path}}
  --model_path ${{inputs.model}}
  --optimized_parameters_path ${{outputs.optimized_parameters}}
  --optimized_model_path ${{outputs.optimized_model}}
experiment_name: optimization-demo-job
name: $OPTIMIZER_JOB_NAME
tags:
  optimizationTool: olive
environment:
  image: mcr.microsoft.com/azureml/aml-olive-optimizer:20230602.8_cpu
compute: azureml:$OPTIMIZER_COMPUTE_NAME
inputs:
  config:
    type: uri_file
    path: ../resnet/resnet.json
  model:
    type: uri_folder
    path: ../resnet/models
  code_path:
    type: uri_folder
    path: ../resnet
outputs:
  optimized_parameters:
    type: uri_folder
  optimized_model:
    type: uri_folder
```

**HINT: `config_path` and `model_path` are optional input parameters. User decides use either or both of them.**

**Template2**:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python -m aml_olive_optimizer
  --config_path ${{inputs.config}}
  --code ${{inputs.code_path}}
  --optimized_parameters_path ${{outputs.optimized_parameters}}
  --optimized_model_path ${{outputs.optimized_model}}
experiment_name: optimization-demo-job
name: $OPTIMIZER_JOB_NAME
tags: 
  optimizationTool: olive
environment:
  image: mcr.microsoft.com/azureml/aml-olive-optimizer:20230602.8_cpu
compute: azureml:$OPTIMIZER_COMPUTE_NAME
inputs:
  config:
    type: uri_file
    path: ../bert/bert.json
  code_path:
    type: uri_folder
    path: ../bert/code
outputs:
  optimized_parameters:
    type: uri_folder
  optimized_model:
    type: uri_folder
```

You may create this olive optimizer job with the following command:

  ```bash
  az ml job create --name $OPTIMIZER_JOB_NAME --file optimizer_job.yml
  ```

#### Understand and download job output

The olive optimizer job will generate below output files into job's default artifact output folder `./outputs`.

* `*_footprints.json`: A dictionary of all the footprints generated during the optimization process.
* `*_pareto_frontier_footprints.json`: A dictionary of the footprints that are on the Pareto frontier based on the metrics goal you set in config of `evaluators.metrics`.
* `*_pareto_frontier_footprints_footprints_chart.html`: Dump pareto_frontier points to html.
* `OutputModels.zip`: Generate a ZIP file which includes 3 folders: `CandidateModels`, `SampleCode` and `ONNXRuntimePackages`. Details please find in: [Packaging Olive artifacts](https://github.com/microsoft/Olive/blob/2deaf333a7b1a36c56c748c03b3bdbbc4a7095c7/docs/source/tutorials/packaging_output_models.md).
* `OutputModels`: Decompressed zip package.

You may download the optimized_parameters file and optimized_model with the following command, they can be used as inputs of the online-endpoints deployer job.

  ```bash
  az ml job download --name $OPTIMIZER_JOB_NAME --output-name optimized_parameters --download-path $OPTIMIZER_DOWNLOAD_FOLDER

  az ml job download --name $OPTIMIZER_JOB_NAME --output-name optimized_model  --download-path $OPTIMIZER_DOWNLOAD_FOLDER
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

Prepare an online-endpoint configuration yaml file. Below is a sample configuration file. For detailed info regarding how to construct an online-endpoint yaml file, see [AzureML Online-Endpoint Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-endpoint-online).

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: $ENDPOINT_NAME
auth_mode: key
```

Prepare an online-deployment configuration yaml file. Below is a sample configuration file. For detailed info regarding how to construct an online-deployment yaml file, see [AzureML Online-Deployment Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online).

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

Below is a sample yaml file that defines an online-endpoints deployer job. For detailed info regarding how to construct a command job yaml file, see [AzureML Job Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)

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

* `deployment_settings.json`: This file contains the detailed online-deployment information.

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

Prepare a profiling configuration json file. Below is a sample configuration file. For detailed info about profiler job configs, see [AzureML Wrk Profiler Configs](#azureml-wrk-profiler-configs), [AzureML Wrk2 Profiler Configs](#azureml-wrk2-profiler-configs) and [AzureML Labench Profiler Configs](#azureml-labench-profiler-configs)

```json
{
  "version": 1.0,
  "profiler_config": {
    "duration_sec": 300,
    "connections": 1
  }
}
```

Prepare a scoring target configuration json file. You may use the `deployment_settings.json` file from the deployer job outputs. Below is a sample configuration file. For detailed info about scoring target configurations, please refer to [Scoring target configuration](#scoring-target-configs)

```json
{
  "version": "1.0",
  "deployment_settings": {
    "subscription_id": "636d700c-4412-48fa-84be-452ac03d34a1",
    "resource_group": "model-profiler",
    "workspace_name": "profilervalidation",
    "endpoint_name": "distilbert-endpt",
    "deployment_name": "distilbert-dep",
    "sku": "Standard_F8s_v2",
    "location": "eastus",
    "instance_count": 1,
    "worker_count": 1,
    "max_concurrent_requests_per_instance": 1
  }
}
```

Below is a sample yaml file that defines a wrk profiling job. For detailed info regarding how to construct a command job yaml file, see [AzureML Job Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)

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

* `report.json`: This file contains detailed profiling results.

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

Below is a sample yaml file that defines an online-endpoints deleter job. For detailed info regarding how to construct a command job yaml file, see [AzureML Job Yaml Schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)

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
  ```

## Job configuration specifications

### OLive optimizer configuration

Currently we support most OLive configs, but still have below special settings and limitations:

* Systems Information:

  <table>
  <tr>
  <th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
  </tr>
  <tr>
  <td> <code>type</code> </td> <td> The type of the system. Only support "LocalSystem" for olive optimizer, "AzureML" and "Docker" are not supported. </td> <td> "LocalSystem" </td> <td> - </td>
  </tr>
  </table>

* Engine Information:

  <table>
  <tr>
  <th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
  </tr>
  <tr>
  <td> <code> packaging_config </code> </td> <td> Olive artifacts packaging configurations. If not specified, Olive will not package artifacts </td>
  <td> DON'T SET, WILL BE OVERRIDED BY DEFAULT VALUE </td>
  <td> 

  ```json
    "packaging_config": {
        "type": "Zipfile",
        "name": "OutputModels"
    }
  ```
  </td>
  </tr>
  <tr>
  <td> <code> output_model_num </code> </td> <td>  The number of output models from the engine based on metric priority. If user want to have more than one best candidate models, please set "output_model_num" below "search_strategy" in advance.</td> <td> 1 </td> <td> 1 </td>
  </tr>
  <tr>
  <td> <code> plot_pareto_frontier </code> </td> <td> This decides whether to plot the pareto frontier of the search results. </td> <td> true </td> <td> true </td>
  </tr>
  <tr>
  <td> <code> output_dir </code> </td> <td> The directory to store the output of the engine. Set to job's default artifact output files "./outputs" </td> <td> DON'T SET, WILL BE OVERRIDED BY DEFAULT VALUE </td> <td> "./outputs" </td>
  </tr>
  </table>

### AzureML profiler configuration

#### Scoring target configs

<table>
<tr>
<th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
</tr>
<tr>
<td> <code>subscription_id</code> </td> <td> [Optional] the subscription id of the online endpoint </td> <td> ea4faa5b-5e44-4236-91f6-5483d5b17d14 </td> <td> subscription id of the profiling job </td>
</tr>
<tr>
<td> <code>resource_group</code> </td> <td> [Optional] the resource group of the online endpoint </td> <td> my-rg </td> <td> resource group of the profiling job </td>
</tr>
<tr>
<td> <code>workspace_name</code> </td> <td> [Optional] the workspace name of the online endpoint </td> <td> my-ws </td> <td> workspace of the profiling job </td>
</tr>
<tr>
<td> <code>endpoint_name</code> </td>
<td>

[Optional] the name of the online endpoint

Required, if users want to get resource usage metrics in the profiling reports, such as CpuUtilizationPercentage, CpuMemoryUtilizationPercentage, etc.

If <code>scoring_uri</code> is not provided, the system will try to get the scoring_uri from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code> and <code>endpoint_name</code>

</td> <td> my-endpoint </td> <td> - </td>
</tr>
<tr>
<td> <code>deployment_name</code> </td> 
<td> 

[Optional] the name of the online deployment

Required, if users want to get resource usage metrics in the profiling reports, such as CpuUtilizationPercentage, CpuMemoryUtilizationPercentage, etc.

If <code>scoring_uri</code> is not provided, the system will try to get the scoring_uri from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code> and <code>endpoint_name</code>

</td> <td> my-deployment </td> <td> - </td>
</tr>
<tr>
<td> <code>identity_access_token</code> </td>
<td>

[Optional] an optional aad token for retrieving endpoint scoring_uri, access_key, and resource usage metrics. This will not be necessary for the following scenario:

- The aml compute that is used to run the profiling job has contributor access to the workspace of the online endpoint.

It's recommended to assign appropriate permissions to the aml compute rather than providing this aad token, since the aad token might be expired during the profiling job

</td> <td> - </td> <td> - </td>
</tr>
<tr>
<td> <code>scoring_uri</code> </td>
<td> 

[Optional] users are optional to provide this env var as instead of the <code>subscription_id</code> / <code>resource_group</code> / <code>workspace_name</code> / <code>endpoint_name</code> / <code>deployment_name</code> combination to define the profiling target. 

If <code>scoring_uri</code> is not provided, the system will try to get the scoring_uri from the endpoint info provided by the user, including subscription_id, resource_group, workspace_name and endpoint_name

If both <code>scoring_uri</code> and endpoint info are provided, the profiling tool will honor the value of the <code>scoring_uri</code>.

</td> <td> https://my-inference-service-uri.com </td> <td> - </td>
</tr>
<tr>
<td> <code>scoring_headers</code> </td>
<td>

[Optional] users may use this env var to provide any headers necessary when invoking the profiling target. One Required field inside the scoring_headers dict is “Authorization”.

If <code>scoring_headers</code> is not provided, the system will try to get the scoring_headers from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

If both <code>scoring_headers</code> and endpoint info are provided, the profiling tool will honor the value of the <code>scoring_headers</code>

</td>
<td>

```json
{
  "Content-Type": "application/json",
  "Authorization": "Bearer < auth_key >",
  "azureml-model-deployment": "< deployment_name >"
}
```

</td> <td> - </td>
</tr>
<tr>
<td> <code>sku</code> </td>
<td>

[Optional] used together with <code>location</code> and <code>instance_count</code> for calculating <code>core_hour_per_million_requests</code>. Missing either one of the 3 values will result in failure when calculating <code>core_hour_per_million_requests</code>.

If <code>sku</code> is not provided, the system will try to get the sku from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

</td> <td> Standard_F2s_v2 </td> <td> - </td>
</tr>
<tr>
<td> <code>location</code> </td>
<td> 

[Optional] used together with <code>sku</code> and <code>instance_count</code> for calculating <code>core_hour_per_million_requests</code>. Missing either one of the 3 values will result in failure when calculating <code>core_hour_per_million_requests</code>.

If <code>location</code> is not provided, the system will try to get the location from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

</td> <td> eastus2 </td> <td> - </td>
</tr>
<tr>
<td> <code>instance_count</code> </td>
<td> 

[Optional] used together with <code>sku</code> and <code>instance_count</code> for calculating <code>core_hour_per_million_requests</code>. Missing either one of the 3 values will result in failure when calculating <code>core_hour_per_million_requests</code>.

If <code>instance_count</code> is not provided, the system will try to get the instance_count from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

</td> <td> 1 </td> <td> - </td>
</tr>
<tr>
<td> <code>worker_count</code> </td>
<td> 

[Optional] the profiling tool would set the default traffic concurrency basing on <code>worker_count</code> and <code>max_concurrent_requests_per_instance</code>.

If users choose to provide the concurrency setting specifically in the profiling_config.json file, then neither <code>worker_count</code> nor <code>max_concurrent_requests_per_instance</code> is necessary.

If concurrency is not provided specifically by the user, and the user did not provide <code>worker_count</code> or <code>max_concurrent_requests_per_instance</code>, the system would try to get the <code>worker_count</code> and <code>max_concurrent_requests_per_instance</code> info from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

Basic logic for setting the default traffic concurrency: if <code>max_concurrent_requests_per_instance</code> is provided, then the default concurrency would be the same as <code>max_concurrent_requests_per_instance</code>; if <code>max_concurrent_requests_per_instance</code> is not provided, while <code>worker_count</code> is provided, the default concurrency would be the same as <code>worker_count</code>; if neither is provided, the default concurrency would be 1.

</td> <td> 1 </td> <td> - </td>
</tr>
<tr>
<td> <code>max_concurrent_requests_per_instance</code> </td> 
<td> 

[Optional] the profiling tool would set the default traffic concurrency basing on <code>worker_count</code> and <code>max_concurrent_requests_per_instance</code>.

If users choose to provide the concurrency setting specifically in the profiling_config.json file, then neither <code>worker_count</code> nor <code>max_concurrent_requests_per_instance</code> is necessary.

If concurrency is not provided specifically by the user, and the user did not provide <code>worker_count</code> or <code>max_concurrent_requests_per_instance</code>, the system would try to get the <code>worker_count</code> and <code>max_concurrent_requests_per_instance</code> info from the endpoint info provided by the user, including <code>subscription_id</code>, <code>resource_group</code>, <code>workspace_name</code>, <code>endpoint_name</code> and <code>deployment_name</code>.

Basic logic for setting the default traffic concurrency: if <code>max_concurrent_requests_per_instance</code> is provided, then the default concurrency would be the same as <code>max_concurrent_requests_per_instance</code>; if <code>max_concurrent_requests_per_instance</code> is not provided, while <code>worker_count</code> is provided, the default concurrency would be the same as <code>worker_count</code>; if neither is provided, the default concurrency would be 1.

</td> <td> 1 </td> <td> - </td>
</tr>
</table>

#### AzureML wrk profiler configs

<table>
<tr>
<th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
</tr>
<tr>
<td> <code>duration_sec</code> </td> <td> [Optional] duration in seconds for running the profiler </td> <td> 600 </td> <td> 300 </td>
</tr>
<tr>
<td> <code>connections</code> </td>
<td>

[Optional] no. of connections for the profiler

The default value will be set to the value of max_concurrent_requests_per_instance

The value is [Required] if the online-endpoint/online-deployment info is not provided, otherwise an error will be thrown

</td> <td> 10 </td> <td> - </td>
</tr>
<tr>
<td> <code>threads</code> </td> <td> [Optional] no. of threads allocated for the profiler </td> <td> 3 </td> <td> 1 </td>
</tr>
<tr>
<td> <code>payload</code> </td>
<td>

[Optional] users may use this param to provide a single string format payload data for invoking the scoring target.

If inputs.payload is provided in the profiler_job.yml file, this env var will be ignored.

</td> <td> '{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}' </td> <td> - </td>
</tr>
</table>

#### AzureML wrk2 profiler configs

<table>
<tr>
<th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
</tr>
<tr>
<td> <code>duration_sec</code> </td> <td> [Optional] duration in seconds for running the profiler </td> <td> 600 </td> <td> 300 </td>
</tr>
<tr>
<td> <code>connections</code> </td>
<td>

[Optional] no. of connections for the profiler

The default value will be set to the value of max_concurrent_requests_per_instance

The value is [Required] if the online-endpoint/online-deployment info is not provided, otherwise an error will be thrown

</td> <td> 10 </td> <td> - </td>
</tr>
<tr>
<td> <code>threads</code> </td> <td> [Optional] no. of threads allocated for the profiler </td> <td> 3 </td> <td> 1 </td>
</tr>
<tr>
<td> <code>payload</code> </td>
<td>

[Optional] users may use this param to provide a single string format payload data for invoking the scoring target.

If inputs.payload is provided in the profiler_job.yml file, this env var will be ignored.

</td> <td> '{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}' </td> <td> - </td>
</tr>
<tr>
<td> <code>target_rps</code> </td> <td> [Optional] target requests per second for the profiler </td> <td> 100 </td> <td> 50 </td>
</tr>
</table>

#### AzureML labench profiler configs

<table>
<tr>
<th> Configuration </th> <th> Definition </th> <th> Example </th> <th> Default Values </th>
</tr>
<tr>
<td> <code>duration_sec</code> </td> <td> [Optional] duration in seconds for running the profiler </td> <td> 600 </td> <td> 300 </td>
</tr>
<tr>
<td> <code>clients</code> </td>
<td>

[Optional] no. of clients for the profiler.

The default value will be set to the value of max_concurrent_requests_per_instance

</td> <td> 10 </td> <td> - </td>
</tr>
<tr>
<td> <code>timeout_sec</code> </td> <td> [Optional] timeout in seconds for each request </td> <td> 20 </td> <td> 10 </td>
</tr>
<tr>
<td> <code>payload</code> </td>
<td>

[Optional] users may use this param to provide a single string format payload data for invoking the scoring target.

If inputs.payload is provided in the profiler_job.yml file, this env var will be ignored.

</td><td> '{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}' </td> <td> - </td>
</tr>
<tr>
<td> <code>target_rps</code> </td> <td> [Optional] target requests per second for the profiler </td> <td> 100 </td> <td> 50 </td>
</tr>
</table>

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
