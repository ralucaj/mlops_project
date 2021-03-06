# -*- coding: utf-8 -*-
import logging
import os
import pdb
import shutil
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("kaggle_dataset")
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(kaggle_dataset, input_filepath, output_filepath):
    """
        Prepares the datasets. Download and process raw data from
        (./data/raw) into cleaned ready preprocessed (.data/processed).

    Args:
        kaggle_dataset (string): kaggle dataset id, in the format [UserName/DatasetName]
        input_filepath (string): location where the raw data is stored
        output_filepath (string): location where the processed data will be stored
    """
    logger = logging.getLogger(__name__)
    # Check and create directories if missing
    input_filepath_full = os.path.abspath(
        os.path.join(
            os.getcwd(), input_filepath.split("/")[0], input_filepath.split("/")[1]
        )
    )
    if not os.path.exists(input_filepath_full):
        logger.info(f"Path {input_filepath} not found, creating it now")
        os.makedirs(input_filepath_full)

    output_filepath_full = os.path.abspath(
        os.path.join(
            os.getcwd(), output_filepath.split("/")[0], output_filepath.split("/")[1]
        )
    )
    if not os.path.exists(output_filepath_full):
        logger.info(f"Path {output_filepath} not found, creating it now")
        os.makedirs(output_filepath_full)

    logger.info(f"Downloading dataset from {kaggle_dataset}")
    os.system(f"kaggle datasets download {kaggle_dataset} -p {input_filepath}")

    dataset_name = kaggle_dataset.split("/")[1]
    with zipfile.ZipFile(f"{input_filepath}/{dataset_name}.zip", "r") as zip_ref:
        zip_ref.extractall(input_filepath)

    logger.info("making final data set from raw data")

    # Train/valid split
    seed = 42
    split = 0.8
    data_df = pd.read_csv(f"{input_filepath}/train.csv")

    # Shuffle dataset and set the first 1 - split as the train set, and the rest as the test set
    data_df = data_df.sample(frac=1, random_state=seed)
    split_mask = np.random.rand(len(data_df)) < split
    train_df = data_df[split_mask]
    valid_df = data_df[~split_mask]
    train_df.to_csv(f"{output_filepath}/train.csv", index=False)
    valid_df.to_csv(f"{output_filepath}/valid.csv", index=False)

    # Copy images to processed
    shutil.copytree(
        f"{input_filepath}/train", f"{output_filepath}/images", dirs_exist_ok=True
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
