import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .custom_model_builder import build_sam2
from .dataset.mini_dataset import MiniDataset
from .dataset.collate_fn import collate_fn
from .helpers.configurations import TRACK_TO_METAINFO
# from optimizer import *
from .process_output import process_output
 

def train():
    # Hyperparameters
    epochs = 10
    batch_size = 1
    lr = 1e-5
    shuffle = False
    len_objects = len(TRACK_TO_METAINFO.keys())
    len_video = 2
    model_type = 'large'  #TODO add this as an option

    # Dataset
    train_dataset = MiniDataset(split_type='mini_train', len_video=len_video)
    valid_dataset = MiniDataset(split_type='mini_train', len_video=len_video)

    # Show the data to test
    debug = True
    if debug:
        idx = np.random.random_integers(0, len(train_dataset))
        frame_obj_list, frames_segmentation_mask = train_dataset[idx]
        for i in range(len_video):
            image = frame_obj_list.frames[i].data
            segmentation_mask = frames_segmentation_mask[i]
            # Images are double precision tensors so we multiply 255 and convert to numpy uint8
            Image.fromarray((image*255).permute(1,2,0).numpy().astype('uint8')).save(f'temp/image{i}.png')
            segmentation_mask.save(f'temp/segmentation_mask{i}.png')
            for j in range(len_objects):
                Image.fromarray((frame_obj_list.frames[i].objects[j].segment.numpy().T.astype('uint8') * 255)).save(f'temp/one_hot{i}{j}.png')

    # TODO Data Augmentation in the Dataset
    # TODO We have to obey the structure of the video data DONE
    # TODO We need to use class BatchedVideoDatapoint see Data utils DONE

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    config = 'custom_sam2.1_hiera_l.yaml'  # This config is per default the trainSAM2
    ck = '/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, loss = build_sam2(config, ck, mode='train', _load_partial=False, device=device)

    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    # loss = MultiStepMultiMasksAndIous()  # Initialized with the model
    
    # Initialize the loss
    train_loss = 0
    val_loss = 0
    epoch_train_loss = []
    epoch_val_loss = []
    
    # Initialize the writer
    writer = SummaryWriter(log_dir='tb_logs')

    # Iterate
    for epoch in tqdm(range(epochs), ncols=50):
        model.train()
        for data_t in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batched_video_data = data_t[0].to(device)
            seg_mask = data_t[1]  # List of PIL Image
            all_frame_outputs = model(batched_video_data)
            # What is the difference between
            # multistep_pred_masks_high_res - multistep_pred_multimasks_high_res - pred_masks_high_res
            # Implement something like run_step as in trainer
            # process_output(all_frame_outputs, data_t[0].masks)
            iter_loss = loss(all_frame_outputs, data_t[0].masks)
            iter_loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", iter_loss, epoch)

        model.eval()
        with torch.no_grad():
            for data_v in valid_loader:
                model()
                writer.add_scalar("Loss/eval", loss, epoch)


if __name__ == '__main__':
    train()