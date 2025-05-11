#!/usr/bin/env bash
set -euo pipefail

#----------------------------------------
# Params
#----------------------------------------
account_id=${1:-}        # e.g. 650906427567
ecr_repo_name=${2:-}     # e.g. catboost_training
region=${3:-}            # e.g. us-east-1
tag_name=${4:-inference-miniconda}    # optional, defaults to "latest"

if [[ -z "$account_id" || -z "$ecr_repo_name" || -z "$region" ]]; then
  echo "Usage: $0 <account_id> <ecr_repo_name> <region> [tag_name]"
  exit 1
fi

#----------------------------------------
# Compute variables
#----------------------------------------
ecr_repo_url="${account_id}.dkr.ecr.${region}.amazonaws.com/${ecr_repo_name}"
# Where this script lives, so paths work whether you run it from project root or src/
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#----------------------------------------
# 1) AWS ECR login
#----------------------------------------
# Must run as the same user that has ~/.aws/credentials configured
aws ecr get-login-password --region "$region" \
  | docker login --username AWS --password-stdin "$ecr_repo_url"

#----------------------------------------
# 2) Build Docker image
#----------------------------------------
# Context: the directory with Dockerfile and code
docker build \
  -t "$ecr_repo_name:$tag_name" \
  -f "${script_dir}/Dockerfile" \
  "$script_dir"

#----------------------------------------
# 3) Tag & push
#----------------------------------------
docker tag "$ecr_repo_name:$tag_name" "$ecr_repo_url:$tag_name"
docker push "$ecr_repo_url:$tag_name"

#----------------------------------------
# 4) Export for Azure DevOps
#----------------------------------------
echo "##vso[task.setvariable variable=IMAGE_URI]$ecr_repo_url:$tag_name"
echo "##vso[task.setvariable variable=IMAGE_URI;isOutput=true]$ecr_repo_url:$tag_name"

#----------------------------------------
# 5) Summary
#----------------------------------------
echo "Built and pushed: $ecr_repo_url:$tag_name"
