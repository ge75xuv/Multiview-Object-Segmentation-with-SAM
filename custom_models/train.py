import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training.optimizer import GradientClipper

from .custom_model_builder import build_sam2
from .dataset.mini_dataset import MiniDataset
from .dataset.collate_fn import collate_fn
from .helpers.configurations import TRACK_TO_METAINFO
# from optimizer import *
from .process_output import process_output
 

def train():
    # Hyperparameters
    epochs = 1
    batch_size = 1
    lr = 1e-5
    shuffle = False
    len_objects = len(TRACK_TO_METAINFO.keys())
    len_video = 2
    model_type = 'large'  #TODO add this as an option
    
    # Empty GPU cache
    if False:
        torch.cuda.empty_cache()
        import gc
        del variables
        gc.collect()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        device = "cuda"
        with torch.cuda.device(device):
            torch.cuda.mem_get_info()  # (free_memory_usage, total_memory)

    # Dataset
    train_dataset = MiniDataset(split_type='mini_train', len_video=len_video)
    valid_dataset = MiniDataset(split_type='mini_train', len_video=len_video)

    # Show the data to test
    debug = False
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
    print(f'Device: {device}\n')
    print(f'Length of the dataloader: {len(train_loader)}\n')
    model, loss = build_sam2(config, ck, mode='train', _load_partial=False, device=device)

    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(device=device, enabled=True, )
    gradient_clipper = GradientClipper(max_norm=0.1, norm_type=2)
    # !!!!! CHECK OPTIMCONF in the trainer they made everything float16 
    # with torch.cuda.amp.autocast( 
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
        for idx_t, data_t in enumerate(train_loader):
            if idx_t > 10:
                break
            optimizer.zero_grad(set_to_none=True)
            batched_video_data = data_t[0].to(device)
            seg_mask = data_t[1]  # List of PIL Image for debug
            masks = data_t[0].masks.to(device)
            all_frame_outputs = model(batched_video_data)
            # What is the difference between
            # multistep_pred_masks_high_res - multistep_pred_multimasks_high_res - pred_masks_high_res
            # the loss handles it itself technically multimask refers to all 3 estimations
            loss_dict = loss(all_frame_outputs, masks)
            iter_loss = loss_dict['core_loss']
            scaler.scale(iter_loss).backward()
            # They use core loss for backprop and use others for logging, see _log_loss_detailed_and_return_core_loss()
            if gradient_clipper is not None:
                scaler.unscale_(optimizer)
                gradient_clipper(model=model)
            # Optimizer step: the scaler will make sure gradients are not
            # applied if the gradients are infinite
            scaler.step(optimizer)
            scaler.update()
            writer.add_scalar("Loss/train", iter_loss, epoch)
            print(f'Epoch: {epoch}\nIter: {idx_t}')
            print(f'Train Loss: {iter_loss}\n')

        model.eval()
        with torch.no_grad():
            for idx_v, data_v in enumerate(valid_loader):
                if idx_v > 10:
                    break
                batched_video_data_val = data_v[0].to(device)
                seg_mask = data_v[1]  # List of PIL Image for debug
                masks_val = data_v[0].masks.to(device)
                all_frame_outputs_val = model(batched_video_data_val)
                loss_dict_val = loss(all_frame_outputs_val, masks_val)
                iter_loss_val = loss_dict_val['core_loss']
                writer.add_scalar("Loss/eval", iter_loss_val, epoch)
                print(f'Epoch: {epoch}\nIter: {idx_v}')
                print(f'Val Loss: {iter_loss_val}')

        # Flush and close
        writer.flush()
        writer.close()
        # Save the model weights
        strf = time.strftime('%m_%d_%H_%M', time.gmtime())
        save_path = f'custom_models/finetune_checkpoints/model{strf}.pt'
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train()