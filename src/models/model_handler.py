import io
import logging
import os
import base64

import torch
from PIL import Image
from torchvision import transforms

# Create model object
model = None

def model_handler(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.

    Example archiver:
    torch-model-archiver --model-name transformer_model --version 1.0 \
            --model-file src/models/visual_transformer_model.py \
            --serialized-file=model_store/deployable_model.pt \
            --handler=src/models/model_handler:model_handler \
            --export-path=model_store -f

    Example torchserve:
    torchserve --start --ncs --model-store model_store \
            --models transformer_model=transformer_model.mar

    Create request file (Linux):
cat > instances.json <<END
{
    "instances": [
      {
        "data": {
            "image": "$(base64 --wrap=0 reports/figures/isic.jpg)"
        }
    }]
}
END

    Create request file (macOs):
cat > instances.json <<END
{
    "instances": [
      {
        "data": {
            "image": "$(base64 reports/figures/isic.jpg)"
        }
    }]
}
END

    Request a prediction from the server:
     curl -X POST \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @instances.json\
     http://127.0.0.1:8080/predictions/transformer_model


    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.jit.load(model_pt_path)
    else:
        # Read bytes array as PIL image
        # print(data) 
        decoded_image = base64.b64decode(data[0]['data']['image'])
        image = Image.open(io.BytesIO(decoded_image))
        # Transform PIL image to tensor
        image = transforms.ToTensor()(image)
        # Resize to 512 x 512
        image = transforms.Resize((512, 512))(image)
        image = torch.unsqueeze(image, 0)
        #infer and return result
        pred = model(image).cpu().detach()[0].argmax().item()
        return [pred]
