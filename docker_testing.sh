#!/bin/bash

# Build docker file
docker build -f Dockerfile . -t mlops_prj:latest

# Run docker file
docker run --name prj_test10 mlops_prj:latest

