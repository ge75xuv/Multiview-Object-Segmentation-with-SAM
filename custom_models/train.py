import time

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from training.optimizer import GradientClipper
from training.dataset.transforms import ComposeAPI, NormalizeAPI

from .custom_model_builder import build_sam2
from .dataset.mini_dataset import MiniDataset
from .dataset.collate_fn import collate_fn
from .helpers.configurations import TRACK_TO_METAINFO
 
model_size_dict = {
    'small': {
        'config': 'custom_sam2.1_hiera_s.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_small.pt',
        },
    'base': {
        'config': 'custom_sam2.1_hiera_b+.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_base_plus.pt',
        },
    'large': {
        'config': 'custom_sam2.1_hiera_l.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/models/sam2/checkpoints/sam2.1_hiera_large.pt',
        },
}

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

def train():
    # Hyperparameters
    epochs = 1
    batch_size = 1
    lr = 5e-6
    shuffle = False
    len_video = 2
    model_size = 'small'
    input_image_size = 512
    object_labels = [8, 9, 10]
    len_objects = len(object_labels)
    #  The list structure is just the way hydra parses the .yaml
    transforms = [ComposeAPI([NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], v2=True)])]

    # Dataset
    train_dataset = MiniDataset('train', len_video, input_image_size, object_labels, transforms, collate_fn, batch_size, shuffle, get_seg_mask=True)
    valid_dataset = MiniDataset('mini_train', len_video, input_image_size, object_labels, transforms, collate_fn, batch_size, shuffle, get_seg_mask=True)

    # Show the data to test
    debug = True
    if debug:
        toPILimage = ToPILImage()
        idx = np.random.randint(0, len(train_dataset))
        idx = 29930
        frame_obj_list, frames_segmentation_mask = train_dataset[idx]
        for i in range(len_video):
            image = frame_obj_list.frames[i].data
            segmentation_mask = frames_segmentation_mask[i]
            # Images are double precision tensors so we multiply 255 and convert to numpy uint8
            toPILimage(image).save(f'temp/image{i}.png')
            segmentation_mask.save(f'temp/segmentation_mask{i}.png')
            for j in range(len_objects):
                toPILimage(frame_obj_list.frames[i].objects[j].segment).save(f'temp/one_hot{i}{j}.png')

    # TODO Data Augmentation in the Dataset

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    iters_per_epoch = len(train_loader)
    print(f'Device: {device}\n')
    print(f'Length of the dataloader: {iters_per_epoch}\n')
    config = model_size_dict[model_size]['config']
    ck = model_size_dict[model_size]['ck']
    model, loss, optim = build_sam2(config, ck, mode='train', _load_partial=False, device=device)
    print('Model building completed')

    scaler = GradScaler(device=device, enabled=True, )
    gradient_clipper = GradientClipper(max_norm=0.1, norm_type=2)
    where = 0

    # Initialize the writer
    start_time = time.strftime('%m_%d_%H_%M', time.gmtime())
    writer = SummaryWriter(log_dir=f'tb_logs/{start_time}')

    # Iterate
    print(f'Training starts for {epochs} epochs!')
    for epoch in range(epochs):
        model.train()
        for idx_t, data_t in tqdm(enumerate(train_loader), ncols=100):
            optim.optimizer.zero_grad(set_to_none=True)
            batched_video_data = data_t[0].to(device)
            seg_mask = data_t[1]  # List of PIL Image for debug
            masks = data_t[0].masks.to(device)
            with torch.autocast(
                device_type=device,
                enabled=True,
                dtype=autocast_dtype,
            ):
                all_frame_outputs = model(batched_video_data)
                loss_dict = loss(all_frame_outputs, masks)
            # What is the difference between
            # multistep_pred_masks_high_res - multistep_pred_multimasks_high_res - pred_masks_high_res
            # the loss handles it itself technically multimask refers to all 3 estimations
            iter_loss = loss_dict['core_loss']
            scaler.scale(iter_loss).backward()
            # They use core loss for backprop and use others for logging, see _log_loss_detailed_and_return_core_loss()
            exact_epoch = epoch + float(idx_t) / iters_per_epoch
            where = float(exact_epoch) / epochs
            assert where <= 1 + 1e-5
            if where < 1.0:
                optim.step_schedulers(
                    where, step=int(exact_epoch * iters_per_epoch)
                )
            if gradient_clipper is not None:
                scaler.unscale_(optim.optimizer)
                gradient_clipper(model=model)
            # Optimizer step: the scaler will make sure gradients are not
            # applied if the gradients are infinite
            scaler.step(optim.optimizer)
            scaler.update()
            writer.add_scalar("Loss/train", iter_loss, epoch)

        model.eval()
        with torch.no_grad():
            for idx_v, data_v in enumerate(valid_loader):
                batched_video_data_val = data_v[0].to(device)
                seg_mask = data_v[1]  # List of PIL Image for debug
                masks_val = data_v[0].masks.to(device)
                all_frame_outputs_val = model(batched_video_data_val)
                loss_dict_val = loss(all_frame_outputs_val, masks_val)
                iter_loss_val = loss_dict_val['core_loss']
                writer.add_scalar("Loss/eval", iter_loss_val, epoch)

        # Save Checkpoint
        save_path = f'custom_models/finetune_checkpoints/model{start_time}_cp{epoch}.pt'
        torch.save(model.state_dict(), save_path)
    # Flush and close
    writer.flush()
    writer.close()
    # Save the model weights
    save_path = f'custom_models/finetune_checkpoints/model{start_time}.pt'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train()