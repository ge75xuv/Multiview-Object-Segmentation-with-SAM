import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2.sam2.modeling import memory_encoder, memory_attention
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from custom_models.debugging.show import *
# from models.Mask2Former.mask2former.modeling.transformer_decoder import mask2former_transformer_decoder

if __name__ == '__main__':
    # Model
    config='configs/sam2.1/sam2.1_hiera_l.yaml'
    ck='/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'
    device = 'cuda'
    predictor = build_sam2_video_predictor(config, ck, device=device)

    # Video
    video_dir = '/home/guests/tuna_gurbuz/prototype/models/sam2/notebooks/videos/bedroom.mp4'
    
    # Predict Init
    inference_state = predictor.init_state(video_path=video_dir)

    # Prompt
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[210, 350]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
    predictor.propagate_in_video(inference_state)

''''
What do we need to do?
1. We need the pre-trained image encoder
3. We need the memory related blocks (They don't need to be pre-trained)
4. The dimensions should be matching!
'''