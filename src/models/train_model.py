import logging
import os
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader
from visual_transformer_model import VisualTransformer

log = logging.getLogger(__name__)
from src.data.isic import ISIC

# Loading realtive directories
train_label_map_path = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/train.csv'))
valid_label_map_path = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/valid.csv'))
image_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/images'))

@hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg):
    print("Training day and night")
    
    # Specify model
    model = VisualTransformer(cfg.model)
    
    # Fetch dataset for training
    train_dataset = ISIC(
        train_label_map_path,
        cfg.data.class_map,
        image_dir,
        filename_col=cfg.data.filename_col,
        label_col=cfg.data.label_col,
        transforms=None,
    )
    
    # Fetch dataset for validation
    valid_dataset = ISIC(
        valid_label_map_path,
        cfg.data.class_map,
        image_dir,
        filename_col=cfg.data.filename_col,
        label_col=cfg.data.label_col,
        transforms=None,
    )
    
    # Create train/validation dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    validation_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size,
    )

    # Initialize early stopping
    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", patience=10, verbose=True, mode="min"
    )
    
    # Initialize model trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        #gpus=1,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[early_stopping_callback],
    )
    
    # Train model (with simultaneous validation)
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    # Save trained model
    torch.save(model.state_dict(), cfg.training.model_path)

    # Save deployable model
    script_model = torch.jit.trace(
        model,
        torch.rand(1, 3, cfg.model.image_size, cfg.model.image_size)
    )
    deployable_model_path = os.path.abspath(
        os.path.join(os.getcwd(), cfg.training.deployable_model_path)
    )
    script_model.save(deployable_model_path)

    # Save quantized model
    model_int8 = torch.quantization.quantize_dynamic(model)
    torch.save(model_int8.state_dict(), cfg.training.quantized_model_path)

train()
