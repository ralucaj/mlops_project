#!/bin/sh

## Create instances of encoded test image
cat > instances.json <<END
{
    "instances": [
      {
        "data": {
            "image": "$(base64 --wrap=0 ${1?Error: No image directory given})"
        }
    }]
}
END


## Send prediction request
curl -X POST \
	-H "Content-Type: application/json; charset=utf-8" \
	-d @instances.json \
	localhost:8080/predictions/visual_transformer

printf "\nStopping docker..."

## Stop docker
docker stop local_deployment
