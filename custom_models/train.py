import os
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='tb_logs')

from custom_model_builder import build_sam2
from or_dataset import ORDataset
# from loss_fns import *
# from optimizer import *
from models.custom_sam2_tune import SAM2Tune
 

def train():
    # Hyperparameters
    epochs = 10
    batch_size = 1
    lr = 1e-5
    shuffle = False
    config = {}
    load_4dor = False
    load_mmor = True
    model_type = 'large'  #TODO add this as an option
    
    # Dataset
    # train_dataset = ORDataset(config=config, split='train', load_4dor=load_4dor, load_mmor=load_mmor)
    # valid_dataset = ORDataset(config=config, split='val', load_4dor=load_4dor, load_mmor=load_mmor)
    
    # Data Augmentation
    
    # Dataloader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    config='configs/custom_sam2.1_hiera_l.yaml'
    ck='/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt'

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
    
    # Iterate
    for epoch in epochs:
        model.train()
        for data_t in train_loader:
            optimizer.zero_grad(set_to_none=True)
            data_t.to(device)
            model()
            # Implement something like run_step as in trainer
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