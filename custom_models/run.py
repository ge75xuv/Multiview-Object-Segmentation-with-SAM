import os
import sys

import numpy as np
from PIL import Image
# import sam2
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_model_builder import build_sam2
from dataset.mini_dataset import MiniDataset
 

if __name__ == '__main__':
    # Dataset
    md = MiniDataset()
    # Model
    config='configs/custom_sam2.1_hiera_l.yaml'
    ck='/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'
    submodel = build_sam2(config, ck, mode='train')
    print('OVER')