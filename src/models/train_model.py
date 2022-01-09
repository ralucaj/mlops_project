import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from visual_transformer_model import VisualTransformer
from torch import nn, optim
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import logging
log = logging.getLogger(__name__)
from src.data.isic import ISIC


@hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg):
    print("Training day and night")
    model = VisualTransformer(cfg.model)
    train_dataset = ISIC(
        cfg.data.train_label_map_path,
        cfg.data.class_map,
        image_dir=cfg.data.image_dir,
        filename_col=cfg.data.filename_col,
        label_col=cfg.data.label_col,
        transforms=None,
    )
    valid_dataset = ISIC(
        cfg.data.valid_label_map_path,
        cfg.data.class_map,
        image_dir=cfg.data.image_dir,
        filename_col=cfg.data.filename_col,
        label_col=cfg.data.label_col,
        transforms=None,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    validation_loader = DataLoader(valid_dataset, batch_size=cfg.training.batch_size)

    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", patience=10, verbose=True, mode="min"
    )
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='gpu',
        gpus=1,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[early_stopping_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)


    # Save model
    torch.save(model.state_dict(), cfg.training.model_path)


train()
