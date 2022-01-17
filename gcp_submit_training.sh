#!/bin/bash

# Define region
REGION=us-central1

# Define image URI
IMAGE_URI=gcr.io/dtumlops-338009/testing:latest

# Define unique job name
JOB_NAME=model_training_job_$(date +%Y%m%d_%H%M%S)

# Submit job to AI- Platform
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI 