import io
import logging
import os

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
            --handler=src/models/model_handler:model_handler

    Example torchserve:
    torchserve --start --ncs --model-store model_store \
            --models transformer_model=transformer_model.mar

    Example request:
     curl http://127.0.0.1:8080/predictions/transformer_model -T reports/figures/isic.jpg


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
        data = Image.open(io.BytesIO(data[0]['body']))
        # Transform PIL image to tensor
        data = transforms.ToTensor()(data)
        # Resize to 512 x 512
        data = transforms.Resize((512, 512))(data)
        data = torch.unsqueeze(data, 0)
        #infer and return result
        pred = model(data).cpu().detach()[0].argmax().item()
        return [pred]
