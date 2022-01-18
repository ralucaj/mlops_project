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
import argparse
from omegaconf import OmegaConf

import logging
import os
import time

log = logging.getLogger(__name__)
from src.data.isic import ISIC

def get_args():
    """Argument parser.
    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Melanoma classification model parameters')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='Learning rate')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        metavar='N',
        help='Input batch size for training')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='Number of epoch to train')

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='S',
        help='Random seed')
    
    parser.add_argument(
        '--limit_train_batches',
        type=float,
        default=0.2,
        metavar='LB',
        help='Limit train batches')
    
    args = parser.parse_args()
    return args

    

# @hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg, args):
    print("Training day and night")

    # Specify model
    model = VisualTransformer(cfg.model, args.lr)
    
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Initialize early stopping
    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", patience=10, verbose=True, mode="min"
    )
    
    # Initialize model trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        #gpus=1,
        limit_train_batches=args.limit_train_batches,
        callbacks=[early_stopping_callback],
    )

    # Train model (with simultaneous validation)
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    model_name = cfg.model.model_path + '_' + experiment_time + '.pt'
    model_path_docker = os.path.join(models_dir_path, model_name)
    print("Model path docker: {0}".format(model_path_docker))
    # Save trained model
    torch.save(model.state_dict(), model_path_docker)

    bucket_name = 'gs://melanoma-classification-models'
    model_path_bucket = os.path.join(bucket_name, 'trained_models', model_name)
    subprocess.check_call(['gsutil', 'cp', model_path_docker, model_path_bucket])


if __name__ == "__main__":

    args = get_args()
    # pdb.set_trace()
    # Define data/train/test/images map paths
    destination_path = os.path.abspath(os.path.join(os.getcwd(),'data'))
    train_label_map_path = os.path.join(destination_path, 'processed', 'train.csv')
    valid_label_map_path = os.path.join(destination_path, 'processed', 'valid.csv')
    image_dir = os.path.join(destination_path,'processed', 'images')
    models_dir_path = os.path.abspath(os.path.join(os.getcwd(),'models'))
    config_path = os.path.abspath(os.path.join(os.getcwd(), 'src', 'models', 'configs', 'config.yaml'))
    experiment_time = time.strftime("%Y%m%d-%H%M%S")

    # Google Cloud bucket name 
    bucket_name = 'gs://raw-dataset/processed'

    if not os.listdir(destination_path):
        print("Data directory empty, downloading the data...")
        # Download data from cloud storage bucket
        command = "gsutil -m cp -r {bucketname} {localpath}".format(bucketname = bucket_name, localpath = destination_path)
        os.system(command)
        #print(os.listdir(os.path.join(destination_path, 'processed')))
    else:
        print("Data already stored.")

    cfg = OmegaConf.load(config_path)
    # Train model
    train(cfg, args)
