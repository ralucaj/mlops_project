import os
from tests import _PATH_DATA
from src.data.isic import ISIC
from src.models.visual_transformer_model import VisualTransformer
import torch
from torch.utils.data import DataLoader

class model_config_attributes:
  image_size = 512
  patch_size = 64
  embed_dim = 128
  num_heads = 3
  num_classes = 2
  lr = 1e-3

   
def test_model_dimensions():
    
    input_data = torch.randn(1, 3, 512, 512)
    model = VisualTransformer(model_config_attributes)
    output = model(input_data)
    
    assert list(output[0].shape) == [2], "The model's output should be 2D."
    
    
    
    