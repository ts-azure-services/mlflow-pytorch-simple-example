#!/bin/bash
set -e

# Source subscription ID, and prep config file
source ./variables.env
number=$[ ( $RANDOM % 10000 ) + 1 ]
sub_id=$SUB_ID
wspace=$WORKSPACE_NAME
location=$LOCATION
rg=$RESOURCE_GROUP
endpoint_name='endpoint'$number

# Set the default subscription 
az account set -s $sub_id

# Set workspace defaults
az configure --defaults workspace=$wspace group=$rg location=$location

# Register a local model
az ml model create --name "linearmodel" \
                   --type "mlflow_model" \
                   --path "./model"

# Create online endpoint
az ml online-endpoint create --name $endpoint_name -f ./inference/endpoint.yaml

# Create deployment
az ml online-deployment create --name "linear-deployment" \
  --endpoint $endpoint_name \
  -f "./inference/instances.yaml" --all-traffic
