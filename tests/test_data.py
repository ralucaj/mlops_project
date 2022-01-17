import os
from tests import _PATH_DATA
from src.data.isic import ISIC
import pytest

class data_config_attributes:
    class_map = {"benign": 0, "malignant": 1}
    filename_col = "image_name"
    label_col = "benign_malignant"
    
class data_paths:
    processed_data_dir = os.path.join(_PATH_DATA,'processed')
    train_label_map = os.path.join(_PATH_DATA,'processed/train.csv')
    valid_label_map = os.path.join(_PATH_DATA,'processed/valid.csv')
    images_dir = os.path.join(_PATH_DATA,'processed/images')

@pytest.mark.skipif(
    not os.path.exists(data_paths.processed_data_dir),
    reason="Data files not found, skipping data tests")
def test_dataset_dimensions():
    
    # Fetch dataset for training
    train_dataset = ISIC(
        data_paths.train_label_map,
        data_config_attributes.class_map,
        data_paths.images_dir,
        filename_col=data_config_attributes.filename_col,
        label_col=data_config_attributes.label_col,
        transforms=None,
    )
    
    # Fetch dataset for training
    valid_dataset = ISIC(
        data_paths.valid_label_map,
        data_config_attributes.class_map,
        data_paths.images_dir,
        filename_col=data_config_attributes.filename_col,
        label_col=data_config_attributes.label_col,
        transforms=None,
    )
    
    assert train_dataset[0][0].shape[0] == 3, 'Train images should have 3 channels!'
    assert list(train_dataset[0][0].shape[1:]) == [512,512], 'Train images should be of dimension 512x512!'
    
    assert valid_dataset[0][0].shape[0] == 3, 'Validation images should have 3 channels!'
    assert list(valid_dataset[0][0].shape[1:]) == [512,512], 'Validation images should be of dimension 512x512!'
