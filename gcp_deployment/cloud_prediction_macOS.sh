#!/bin/sh

## Create instances of encoded test image
cat > instances.json <<END
{
    "instances": [
      {
        "data": {
            "image": "$(base64 ${1?Error: No image directory given})"
        }
    }]
}
END


## Send prediction request
curl -X POST \
	-H "Authorization: Bearer $(gcloud auth print-access-token)" \
	-H "Content-Type: application/json; charset=utf-8" \
	-d @instances.json \
	https://us-central1-ml.googleapis.com/v1/projects/mlops-project-skin-cancer/models/docker_deployed_model/versions/v10:predict

printf "\n"