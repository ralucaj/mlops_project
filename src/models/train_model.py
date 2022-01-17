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
import pdb
import subprocess

import logging
import os
import time

log = logging.getLogger(__name__)
from src.data.isic import ISIC

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
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    validation_loader = DataLoader(valid_dataset, batch_size=cfg.training.batch_size)

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

    model_name = cfg.training.model_path + '_' + experiment_time + '.pt'
    model_path_docker = os.path.join('/models', model_name)
    # Save trained model
    torch.save(model.state_dict(), model_path_docker)

    bucket_name = 'gs://melanoma-classification-models'
    model_path_bucket = os.path.join(bucket_name, 'trained_models', model_name)
    subprocess.check_call(['gsutil', 'cp', model_path_docker, model_path_bucket])

if __name__ == "__main__":

    # Create directory to store the downloaded data
    # destination_path = os.path.abspath(os.path.join(os.getcwd(),'data/processed')) #'root/data/processed'

    destination_path =  '/tmp' #'root/data/processed'
    # print("Created directory: {0}".format(destination_path))
    # Download data from cloud storage bucket
    bucket_name = 'gs://raw-dataset/processed'
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', bucket_name, destination_path])

    # Define train/test/images map paths
    # train_label_map_path = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/train.csv'))
    # valid_label_map_path = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/valid.csv'))
    # image_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/images'))
    train_label_map_path = os.path.join(destination_path, 'train.csv')
    valid_label_map_path = os.path.join(destination_path, 'train.csv')
    image_dir = os.path.join(destination_path, 'images')
    experiment_time = time.strftime("%Y%m%d-%H%M%S")

    

    # Train model
    train()
