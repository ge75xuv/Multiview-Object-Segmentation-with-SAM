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
# from loss_fns import *
# from optimizer import *
from .models.custom_sam2_tune import SAM2Tune
 

def train():
    # Hyperparameters
    epochs = 10
    batch_size = 1
    lr = 1e-5
    shuffle = False
    num_multimask_outputs = 8  # This is num mask queries
    model_type = 'large'  #TODO add this as an option

    # Dataset
    train_dataset = MiniDataset(split_type='small_train', num_multimask_outputs=num_multimask_outputs)
    valid_dataset = MiniDataset(split_type='val', num_multimask_outputs=num_multimask_outputs)

    # Show the data to test
    debug = False
    if debug:
        idx = np.random.random_integers(0, len(train_dataset))
        image, segmentation_mask, one_hot_mask = train_dataset[idx]
        image.save('temp/image.png')
        segmentation_mask.save('temp/segmentation_mask.png')
        for i in range(num_multimask_outputs):
            Image.fromarray((one_hot_mask[i].numpy().T.astype('uint8') * 255)).save(f'temp/one_hot{i}.png')

    # TODO Data Augmentation in the Dataset

    # TODO We have to obey the structure of the video data
    
    # TODO We need to use class BatchedVideoDatapoint see Data utils

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model
    config = 'custom_sam2.1_hiera_l.yaml'  # This config is per default the trainSAM2
    ck = '/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'
 
    model = build_sam2(config, ck, mode='train')
    device = model.device

    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    loss = MSELoss()
    
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
            im = data_t[0].to(device)
            seg_mask = data_t[1].to(device)
            one_hot_mask = data_t[2].to(device)
            output = model(im)
            
            # Implement something like run_step as in trainer
            loss(output, one_hot_mask)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)

        model.eval()
        with torch.no_grad():
            for data_v in valid_loader:
                model()
                writer.add_scalar("Loss/eval", loss, epoch)


if __name__ == '__main__':
    train()