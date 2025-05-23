import os
import sys
from copy import deepcopy
import json

project_dir = os.path.dirname(os.getcwd())
print(project_dir)
sys.path.append(project_dir)

import numpy as np
# from sam2.build_sam import build_sam2
import torch
from torchvision.transforms import ToPILImage, ToTensor, Normalize
from training.dataset.transforms import ComposeAPI, NormalizeAPI
from tqdm import tqdm

from .dataset.collate_fn import collate_fn
from .dataset.mini_dataset import MiniDataset


# Dataset
object_labels = [13, 15, 16]
len_video = 1
input_image_size = 512
batch_size = 1
shuffle = False
mean = [0.3551, 0.3500, 0.3469]
std = [0.2921, 0.2716, 0.2742]
transforms = [ComposeAPI([NormalizeAPI(mean=mean, std=std, v2=True)])]
revert_mean=[-.3551/.2921, -.3500/.2716, -.3469/.2742]
revert_std=[1/.2921, 1/.2716, 1/.2742]
revert_transform = Normalize(mean=revert_mean, std=revert_std)
test_dataset = MiniDataset('train',
                           num_frames=len_video,
                           input_image_size=input_image_size,
                           object_labels=object_labels,
                           transforms=transforms,
                           collate_fn=collate_fn,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           get_seg_mask=True)
print(f'Lenght of the dataset! {len(test_dataset)}')

seed = 123  # Check seed 123 index 19966
torch.manual_seed(seed)
np.random.seed(seed)
# 035 idx 2761 has a problem!
# Image
len_objects = len(object_labels)
toPILimage = ToPILImage()
exist = False
exist_idx = []
idx = 0

for idx in tqdm(range(len(test_dataset))):
    # if len(exist_idx) == 1:
    #     break
    # idx = 25424
    # print(f'Index: {idx}')
    frame_obj_list, frames_segmentation_mask = test_dataset[idx]
    image = frame_obj_list.frames[0].data
    # print('Input Image:')
    # toPILimage(revert_transform(image)).show()
    for j in range(len_objects):
        exist = torch.any(frame_obj_list.frames[0].objects[j].segment == True)
        if exist:
            exist_idx.append(idx)
            del frame_obj_list
            break

with open('./temp/temp.json', 'w') as f:
    json.dump({'train_13_15_16': exist_idx}, f)