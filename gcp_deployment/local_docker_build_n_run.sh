#!/bin/sh

## Build docker Image
docker build \
	-f gcp_deployment/Deployment.dockerfile \
	--tag=us-central1-docker.pkg.dev/mlops-project-skin-cancer/deployment-docker-repo/serve-visual-transformer \
	.


## Run docker
docker run \
	-d -p 8080:8080 \
	--name=local_deployment \
	us-central1-docker.pkg.dev/mlops-project-skin-cancer/deployment-docker-repo/serve-visual-transformer