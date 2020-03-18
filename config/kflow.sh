# # initial setup
# mkdir kubeflow
# cd kubeflow
# curl -LO https://github.com/kubeflow/kfctl/releases/download/v1.0.1/kfctl_v1.0.1-0-gf3edb9b_linux.tar.gz
# tar -xvf kfctl_v1.0.1-0-gf3edb9b_linux.tar.gz
# sudo mv ./kfctl /usr/local/bin

# # Create user credentials
# # If using GCE, this command is not necessary
# gcloud auth application-default login

# Set GCP project ID and the zone where you want to create the Kubeflow deployment
export PROJECT=xxx
export ZONE=asia-east1-a
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

# google cloud storage bucket
export GCP_BUCKET=gs://kf-test1234

# Make the same version as kfctl.
# Use the following kfctl configuration file for authentication with 
# Cloud IAP (recommended):
export CONFIG="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_gcp_iap.v1.0.0.yaml"

# For using Cloud IAP for authentication, create environment variables
# from the OAuth client ID and secret that you obtained earlier:
export CLIENT_ID=<ADD OAuth CLIENT ID HERE>
export CLIENT_SECRET=<ADD OAuth CLIENT SECRET HERE>

# Set KF_NAME to the name of your Kubeflow deployment. You also use this
# value as directory name when creating your configuration directory. 
# For example, your deployment name can be 'my-kubeflow' or 'kf-test'
export KF_NAME=kf-test

# # Set up name of the service account that should be created and used
# # while creating the Kubeflow cluster
# SA_NAME=

export BASE_DIR=~/kubeflow
export KF_DIR=${BASE_DIR}/${KF_NAME}
mkdir -p $KF_DIR