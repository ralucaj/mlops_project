import os
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
import kornia as K
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


class VisualTransformer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.num_classes = cfg.num_classes

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
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(self, x):
        # make sure input tensor is flattened
        x = self.classifier(x)
        '''
        # Profile Tracking (uncomment and use with debug breakpoint at `return x`)
        with profile(activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    on_trace_ready=tensorboard_trace_handler(os.getcwd())) as prof:
            with record_function("model_inference"):
                # make sure input tensor is flattened
                x = self.classifier(x)
        '''
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
