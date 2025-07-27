import os
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from training.dataset.transforms import ComposeAPI, NormalizeAPI

project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
print(project_dir)
sys.path.append(project_dir)

from mini_dataset import MiniDataset
from collate_fn import collate_fn

num_frames = 1
input_image_size = 512
transforms = [ComposeAPI([NormalizeAPI(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], v2=True)])]
object_labels = [10]
batch_size = 1

train_dataset = MiniDataset('train',
                             num_frames=num_frames,
                             input_image_size=input_image_size,
                             object_labels=object_labels,
                             transforms=transforms,
                             collate_fn=collate_fn,
                             batch_size=batch_size,)

len_dataset = len(train_dataset)
print(f"Dataset length: {len_dataset}")

nb_samples = 0
mean = 0.0
std = 0.0
batch_size = 100
im_list = []

with open('mean_std.txt', 'w') as f:
    f.write('Mean and Std for training dataset\n')

for idx, im_path in tqdm(enumerate(train_dataset.images)):
    img = np.array(Image.open(im_path[0]).convert("RGB"))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    im_list.append(img)
    if len(im_list) == batch_size:
        try:
            img = torch.stack(im_list, dim=0).to('cuda')
        except RuntimeError as e:
            print(f"Error processing batch {idx}: {e}")
            for im in im_list:
                img = im.to('cuda')
                im_vec = img.flatten(-2)
                mean += im_vec.mean(-1)
                std += im_vec.std(-1)
                nb_samples += 1
            im_list = []
            continue
        im_list = []
        im_vec = img.flatten(-2)
        mean += im_vec.mean(-1).sum(0)
        std += im_vec.std(-1).sum(0)
        nb_samples += batch_size
        with open('mean_std.txt', 'a') as f:
            f.write(f'Idx: {idx}\n')
            f.write(f'   Mean: {mean/nb_samples}\n')
            f.write(f'   Std: {std/nb_samples}\n')
    elif idx == len_dataset - 1:
        try:
            img = torch.stack(im_list, dim=0).to('cuda')
        except RuntimeError as e:
            print(f"Error processing batch {idx}: {e}")
            for im in im_list:
                img = im.to('cuda')
                im_vec = img.flatten(-2)
                mean += im_vec.mean(-1)
                std += im_vec.std(-1)
                nb_samples += 1
            im_list = []
            continue
        im_list = []
        im_vec = img.flatten(-2)
        mean += im_vec.mean(-1).sum(0)
        std += im_vec.std(-1).sum(0)
        nb_samples += len(im_vec)

mean /= nb_samples
std /= nb_samples

print(f'Sample size: {nb_samples}')
print(f'Mean: {mean}')
print(f'Std: {std}')