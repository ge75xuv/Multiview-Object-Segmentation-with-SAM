import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from PIL import Image
import sam2
from sam2.build_sam import build_sam2
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2.sam2.modeling import memory_encoder, memory_attention
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from custom_models.debugging.show import *
# from models.Mask2Former.mask2former.modeling.transformer_decoder import mask2former_transformer_decoder

class customSam2(nn.Module):
    custom = True

    def __init__(self, sub_model:nn.Module):
        self.image_encoder = sub_model.image_encoder
        self.mask_downsample = sub_model.mask_downsample 
        self.memory_attention =sub_model.memory_attention 
        self.memory_encoder = sub_model.memory_encoder 
        
        for p in sub_model.parameters():
            print(p)

    def forward():
        pass

if __name__ == '__main__':
    # Model
    config='configs/sam2.1/sam2.1_hiera_l.yaml'
    ck='/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'
    submodel = build_sam2(config, ck)
    im_pred = SAM2ImagePredictor(submodel)

    # Image
    image = Image.open('custom_models/camera02.jpg').convert('RGB')
    im_pred.set_image(image)

    # Grid
    steps = 500
    y = np.arange(image.size[0], step=steps)
    x = np.arange(image.size[1], step=steps)
    grid_x, grid_y = np.meshgrid(y,x)
    grid_x = grid_x.flatten(order='C')[:,None]
    grid_y = grid_y.flatten(order='F')[:,None]
    grid = np.concatenate([grid_x, grid_y], axis=1)

    # Point
    # grid = np.array([[500,400]])

    # Point Labels: 1 Foreground 0 Background
    point_labels = np.ones([grid.shape[0]], dtype=np.int8)
    
    up_dim = True
    if up_dim:
        grid[:,None,:]
        point_labels[:,None]

    # Predict
    masks, scores, logits = im_pred.predict(
    point_coords=grid,
    point_labels=point_labels,
    multimask_output=True,
)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    show_masks(image, masks, scores, point_coords=grid, input_labels=point_labels, borders=False)
    model = customSam2(submodel)
    del submodel
    print('Over!')

''''
What do we need to do?
1. We need the pre-trained image encoder
2. We need the mask decoder from the mask2former (It does not need to be pre-trained!)
3. We need the memory related blocks (They don't need to be pre-trained)
4. The dimensions should be matching!
'''