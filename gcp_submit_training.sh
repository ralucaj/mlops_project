#!/bin/bash

# wandb API key
WANDB_KEY=$(gcloud secrets versions access 2 --secret="wandb_api_key")
# Define region
REGION=us-central1

# Define image URI
IMAGE_URI=gcr.io/mlops-project-skin-cancer/training_images:latest

# Define unique job name
JOB_NAME=model_training_job_$(date +%Y%m%d_%H%M%S)

# Submit job to AI- Platform
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --lr=0.001 \
  --batch_size=32 \
  --epochs=20 \
  --seed=30 \
  --wandb_key=$WANDB_KEY

