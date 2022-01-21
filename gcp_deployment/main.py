from google.cloud import storage
import os
import io

import kornia as K
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
from PIL import Image
from torchvision import transforms as T



class model_config_attributes:
  image_size = 512
  patch_size = 64
  embed_dim = 128
  num_heads = 3
  num_classes = 2
  lr = 1e-3


class VisualTransformer(LightningModule):
    def __init__(self, cfg, lr):
        super().__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.num_classes = cfg.num_classes
        self.lr = lr

        self.vision_transformer = K.contrib.VisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )
        self.classification_head = K.contrib.ClassificationHead(
            embed_size=self.embed_dim, num_classes=self.num_classes
        )
        self.classifier = nn.Sequential(
            self.vision_transformer, self.classification_head
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        # make sure input tensor is flattened
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Perform a training iteration.

        Args:
            batch (tensor): input data batch of size [batch_size, 512, 512]
            batch_idx ([type]): [description]

        Returns:
            loss (tensor): array of losses for the particular input batch
        """
        data, target = batch
        preds = self.forward(data)
        loss = self.criterion(preds, target)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation iteration.

        Args:
            batch (tensor): input data batch of size [batch_size, 512, 512]
            batch_idx ([type]): [description]

        Returns:
            loss (tensor): array of losses for the particular input batch
        """
        data, target = batch
        preds = self.forward(data)
        loss = self.criterion(preds, target)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


def run_model(models_bucket_name, test_image_bucket_name, test_image_dir, model_dir):

    ## Load pre-trained model
    storage_client_model = storage.Client()
    model_bucket = storage_client_model.bucket(models_bucket_name)
    blob_model = model_bucket.get_blob(model_dir)
    
    model_dict = torch.load(io.BytesIO(blob_model.download_as_string()))
    
    model = VisualTransformer(model_config_attributes, model_config_attributes.lr)
    model.state_dict(model_dict)
    model.eval()
    
    ## Load test image
    storage_client_image = storage.Client()
    image_bucket = storage_client_image.bucket(test_image_bucket_name)
    blob_image = image_bucket.get_blob(test_image_dir)
    
    test_image_as_pil = Image.open(io.BytesIO(blob_image.download_as_string()))
    test_image = T.ToTensor()(test_image_as_pil)
    test_image = T.Resize((512, 512))(test_image)
    test_image = torch.unsqueeze(test_image, 0)
    
    ## Predict
    pred = model(test_image).cpu().detach()[0].argmax().item()
    
    ## Print Result
    print('The model prediction is: {}'.format(pred))


if __name__ == "__main__":
    models_bucket_name = 'melanoma-classification-models'
    test_image_bucket_name = 'raw-dataset'
    
    test_image_dir = 'test/test_images/ISIC_0052060.jpg'
    model_dir = 'trained_models/trained_model_20220119-173852.pt'
    
    run_model(models_bucket_name, test_image_bucket_name, test_image_dir, model_dir)