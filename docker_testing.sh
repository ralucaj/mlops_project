#!/bin/bash

# Build docker file
docker build -f Dockerfile . -t mlops_prj:latest

# Run docker file
docker run --name prj_test4 mlops_prj:latest

