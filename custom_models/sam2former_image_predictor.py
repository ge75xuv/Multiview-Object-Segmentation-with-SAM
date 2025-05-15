import os
import sys
# from copy import deepcopy

project_dir = os.path.dirname(os.getcwd())
print(project_dir)
sys.path.append(project_dir)

from hydra import initialize
try:
    initialize(version_base=None, config_path="../sam2_logs/", job_name="predict_run")
except ValueError:
    pass

import numpy as np
import torch
from torch.amp import autocast
from torchvision.transforms import ToPILImage, Normalize, Resize
from torchmetrics.detection import PanopticQuality
from training.dataset.transforms import ComposeAPI, NormalizeAPI
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .helpers.configurations import TRACK_TO_METAINFO, LABEL_PROJECTION_MAP
from .dataset.collate_fn import collate_fn
from .dataset.mini_dataset import MiniDataset
from .custom_model_builder import build_sam2former

model_size_dict = {
    'base': {
        'config': '04_28_00_50/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/04_28_00_50/checkpoints/checkpoint_33.pt',
        },
}

# Tensorboard
model_name = '04_28_00_50'
writer = SummaryWriter(f'./tb_logs/{model_name}_eval/')

# Model
model_size = 'base'
config = model_size_dict[model_size]['config']
ck = model_size_dict[model_size]['ck']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
amp_type = torch.bfloat16 if device == 'cuda' else torch.float16
submodel, object_labels, _, loss = build_sam2former(config, ck, device=device)
print(device)

# Load Dict
sd = torch.load(ck, map_location="cpu", weights_only=True)["model"]
missing_keys, unexpected_keys = submodel.load_state_dict(sd, strict=False)

# Dataset
len_video = 1
input_image_size = 512
batch_size = 5
shuffle = True
mean = [0.3551, 0.3500, 0.3469]
std = [0.2921, 0.2716, 0.2742]
revert_mean=[-.3551/.2921, -.3500/.2716, -.3469/.2742]
revert_std=[1/.2921, 1/.2716, 1/.2742]
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# revert_mean=[-.485/.229, -.456/.224, -.406/.225]
# revert_std=[1/.229, 1/.224, 1/.225]
transforms = [ComposeAPI([NormalizeAPI(mean=mean, std=std, v2=True)])]
revert_transform = Normalize(mean=revert_mean, std=revert_std)
test_dataset = MiniDataset('test',
                           num_frames=len_video,
                           input_image_size=input_image_size,
                           object_labels=object_labels,
                           transforms=transforms,
                           collate_fn=collate_fn,
                           batch_size=batch_size,
                           get_seg_mask=True,
                           shuffle=shuffle,)
print(f'Lenght of the dataset! {len(test_dataset)}')

# Seed
seed = 2 # Check seed 123 index 19966
torch.manual_seed(seed)
np.random.seed(seed)
flag_print_logs = False

# Image
len_objects = len(object_labels)
toPILimage = ToPILImage()
exist = False
if_break = False
sample_idx = 10
test_loader = test_dataset.get_loader()

del test_dataset

# Prepare panoptic quality metric
# Stuff: 16, Things: 17 Instance labels: 0-15
background = 16
things = {LABEL_PROJECTION_MAP[idx]['label'] for idx in object_labels}
stuff = {background, background+1}  # Putting 17 does not change anything for the last value
pq_list = []
panoptic_quality = PanopticQuality(stuffs=stuff, things=things).to(device)

# Run the model
with torch.no_grad():
    submodel.eval()
    for idx, batch in tqdm(enumerate(test_loader)):
        # print(f"Processing {idx}th sample")
        batched_video_data_val = batch[0].to(device)
        # batch_seg_mask_gt = batch[1]  # List of PIL Image for debug
        masks_val = batch[0].masks.to(device)
        with autocast(device_type=device, dtype=amp_type):
            all_frame_outputs_val = submodel(batched_video_data_val)

        print('CP: 1') if flag_print_logs else None

        # Convert mask prediction logits to binary masks
        binary_pred_masks = all_frame_outputs_val[0]['pred_masks_high_res'].sigmoid() > 0.5
        binary_pred_masks = binary_pred_masks.cpu().numpy()

        # Convert class logits to class predictions
        pred_logits = all_frame_outputs_val[0]['pred_logits'].type(torch.float32).softmax(-1).cpu().numpy()
        pred_class = pred_logits.argmax(-1)
        
        # Open Memory
        del batch
        del all_frame_outputs_val

        B, N, H, W = binary_pred_masks.shape  # Size with padding
        gt_labels = batched_video_data_val.metadata.unique_objects_identifier[:,:,1]

        # Prepare segmentation mask and RGB mask with color encoding
        pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 16  # 16 is the background class
        # pred_rgb_mask = np.zeros((B, H, W, 3), dtype=np.uint8)
        for _b_idx, (_b_binary_pred_masks, _b_pred_class) in enumerate(zip(binary_pred_masks, pred_class)):
            for mask, class_id in zip(_b_binary_pred_masks, _b_pred_class):
                pos = np.where(mask == True)
                if len(pos[0]) > 0:
                    pred_seg_class_mask[_b_idx, 0, pos[0], pos[1]] = class_id
                else:
                    continue

        gt_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 16  # 16 is the background class
        # gt_rgb_mask = np.zeros((B, H, W, 3), dtype=np.uint8)
        for batch_idx in range(B):
            pos = batched_video_data_val.obj_to_frame_idx[0,:,1] == batch_idx  # The objects in the batch are mixed
            gt_mask = masks_val[0, pos, :, :].cpu().numpy()  # Get the GT masks in the batch
            gt_class_id = gt_labels[0, pos].cpu().numpy()  # Get the GT class id in the batch
            for mask, class_id in zip(gt_mask, gt_class_id):
                pos = np.where(mask == True)
                if len(pos[0]) > 0:
                    gt_seg_class_mask[batch_idx, 0, pos[0], pos[1]] = class_id
                else:
                    continue

        # Open Memory
        del binary_pred_masks
        del pred_class
        del gt_mask
        del gt_labels
    
    
        print('CP: 2') if flag_print_logs else None
    
        # Initialize lists for masks
        # pred_rgb_mask_list_azure = []
        pred_seg_class_mask_list_azure = []
        # gt_rgb_mask_list_azure = []
        gt_seg_class_mask_list_azure = []

        # Revert the padding
        for batch_idx in range(B):
            # Load the corresponding rgb image to look for padding 
            img = batched_video_data_val.img_batch[0][batch_idx].permute(1,2,0).cpu().numpy()
            
            # Check if the image is black (0,0,0) along the height dimension
            pos_padding = np.where((img[:,:,0] <= 0) == False)
            start = pos_padding[0][0]
            end = pos_padding[0][-1]
            
            # Slice the predictions and ground truth masks
            if start == 64:
                # pred_rgb_mask_list_azure.append(pred_rgb_mask[batch_idx, start:end+1, :, :])
                pred_seg_class_mask_list_azure.append(pred_seg_class_mask[batch_idx, 0:1, start:end+1, :])
                # gt_rgb_mask_list_azure.append(gt_rgb_mask[batch_idx, start:end+1, :, :])
                gt_seg_class_mask_list_azure.append(gt_seg_class_mask[batch_idx, 0:1, start:end+1, :])
            elif start == 0:
                pass
            else:
                raise ValueError("Padding not found in the image")


        print('CP: 3') if flag_print_logs else None

        # Concatenate the masks
        # pred_rgb_mask = np.array(pred_rgb_mask_list_azure)
        pred_seg_class_mask = np.array(pred_seg_class_mask_list_azure)
        # gt_rgb_mask = np.array(gt_rgb_mask_list_azure)
        gt_seg_class_mask = np.array(gt_seg_class_mask_list_azure)
        
        # Open Memory
        del batched_video_data_val
        del masks_val
        del pred_seg_class_mask_list_azure
        del gt_seg_class_mask_list_azure


        print('CP: 4') if flag_print_logs else None

        # Prepare stuff and things mask for prediction
        preds = pred_seg_class_mask[:, 0, :, :, None]
        stuff_things_mask = np.ones_like(preds) * (background + 1)  # 17 is the stuff label
        pos_stuff = np.where(preds[:, :, :, 0] == background)  # Find where the background is
        stuff_things_mask[pos_stuff[0], pos_stuff[1], pos_stuff[2], 0] = background  # Set the stuff label background
        preds = np.concatenate((stuff_things_mask, preds), axis=3)

        # Prepare stuff and things mask for GT
        target = gt_seg_class_mask[:, 0, :, :, None]
        stuff_things_mask_gt = np.ones_like(target) * (background + 1)  # 17 is the stuff label
        pos_stuff = np.where(target[:, :, :, 0] == background)  # Find where the background is
        stuff_things_mask_gt[pos_stuff[0], pos_stuff[1], pos_stuff[2], 0] = background  # Set the stuff label background
        target = np.concatenate((stuff_things_mask_gt, target), axis=3)

        # Convert to torch tensors
        preds = torch.tensor(preds, device=device)
        target = torch.tensor(target, device=device)

        print('CP: 5') if flag_print_logs else None

        # Compute PQ
        pq = panoptic_quality(preds, target)
        pq_list.append(pq.item())
        print(f"PQ: {pq}")
        writer.add_scalar(f"PQ/test_{batch_size}", pq, idx)
        writer.flush() if idx % 10 == 0 else None
        torch.cuda.empty_cache()

average_pq = sum(pq_list) / len(pq_list)
print(f"Average PQ: {average_pq}")
writer.add_scalar(f"PQ/test_{batch_size}_average", average_pq, 1)
writer.flush()
writer.close()
print("Finished!")
