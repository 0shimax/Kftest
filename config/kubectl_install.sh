curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.17.0/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

curl https://sdk.cloud.google.com | bash
source ~/.bashrc
gcloud components install kubectl