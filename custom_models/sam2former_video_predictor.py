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
from .dataset.collate_fn import collate_fn_wrapper
from .dataset.mini_dataset import MiniDataset
from .custom_model_builder import build_sam2former

model_size_dict = {
    'base': {
        'config': '04_28_00_50/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/04_28_00_50/checkpoints/checkpoint_33.pt',
        },
    'base1': {  # THIS IS VIDEO MODEL 23 Q
        'config': '08_03_21_22/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/08_03_21_22/checkpoints/checkpoint_23.pt',
        },
    'base2': {  # THIS IS VIDEO MODEL30 Q
        'config': '09_04_11_56_vid_model1/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_04_11_56_vid_model1/checkpoints/checkpoint_17.pt',
        },
    'base3': {  # THIS IS VIDEO MODEL30 Q cont
        'config': '09_04_11_56_vid_model1/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_07_16_40/checkpoints/checkpoint_18.pt',
        },
}

# Model
model_size = 'base1'
len_video = 8
# Tensorboard
model_name = model_size_dict[model_size]['config'].split('/')[0]
writer = SummaryWriter(f'./tb_logs/{model_name}_video{len_video}_eval/')

# Hydra
config = model_size_dict[model_size]['config']
ck = model_size_dict[model_size]['ck']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
amp_type = torch.bfloat16 if device == 'cuda' else torch.float16
submodel, object_labels, _, loss, mean, std = build_sam2former(config, ck, device=device)
print(device)

# Load Dict
sd = torch.load(ck, map_location="cpu", weights_only=True)["model"]
missing_keys, unexpected_keys = submodel.load_state_dict(sd, strict=False)

print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Dataset
input_image_size = 256
batch_size = 1
shuffle = False
revert_mean = [-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]]
revert_std = [1/std[0], 1/std[1], 1/std[2]]
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
                           collate_fn= collate_fn_wrapper,
                           batch_size=batch_size,
                           num_workers=0,
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
BACKGROUND = 23
ADD_RATIO_THR = 0.2
things = {LABEL_PROJECTION_MAP['default'][idx]['label'] for idx in object_labels}
stuff = {BACKGROUND}  # Putting 17 does not change anything for the last value
print(f'Things: {things}\nStuff: {stuff}')
panoptic_quality = PanopticQuality(stuffs=stuff, things=things).to(device)
pq_list = []

# Run the model
with torch.no_grad():
    submodel.eval()
    for idx, batch in tqdm(enumerate(test_loader)):
        # print(f"Processing {idx}th sample")
        batched_video_data_val = batch.to(device)
        # batch_seg_mask_gt = batch[1]  # List of PIL Image for debug
        masks_val = batch.masks.to(device)
        with autocast(device_type=device, dtype=amp_type):
            all_frame_outputs_val = submodel(batched_video_data_val)
        all_frame_outputs_val = all_frame_outputs_val[0]  # It is a multview output, get only single view

        print('CP: 1') if flag_print_logs else None
        
        # Concatenate frame outputs as batch
        for i in range(len(all_frame_outputs_val)):
            if i == 0:
                all_masks = all_frame_outputs_val[i]['pred_masks_high_res']
                all_labels = all_frame_outputs_val[i]['pred_logits']
            else:
                all_masks = torch.cat((all_masks, all_frame_outputs_val[i]['pred_masks_high_res']), dim=0)
                all_labels = torch.cat((all_labels, all_frame_outputs_val[i]['pred_logits']), dim=0)

        # Convert mask prediction logits to binary masks
        mask_logits = all_masks.float()           # [B,Q,H,W]
        mask_prob = mask_logits.sigmoid()                                               # [B,Q,H,W]
        binary_pred_masks = (mask_prob > 0.5).cpu().numpy()                             # [B,Q,H,W]

        # Convert class logits to class predictions
        pred_logits = all_labels.type(torch.float32)
        probs = pred_logits.softmax(-1)
        class_probs = probs[..., :-1]
        pred_class = class_probs.argmax(-1).cpu().numpy()
        pred_class_prob = class_probs.max(-1).values.cpu().numpy()

        # Open Memory
        del batch
        del all_frame_outputs_val

        B, N, H, W = binary_pred_masks.shape  # Size with padding
        gt_labels = batched_video_data_val.metadata.unique_objects_identifier[:,:,1]

        # Compute maskness score
        mask_prob_np = mask_prob.cpu().numpy()
        B, Q, H, W = binary_pred_masks.shape
        maskness = np.zeros((B, Q), dtype=np.float32)
        for b in range(B):
            for q in range(Q):
                m = binary_pred_masks[b, q]
                maskness[b, q] = mask_prob_np[b, q][m].mean() if m.any() else mask_prob_np[b, q].mean()

        # query score for ordering
        query_score = pred_class_prob * maskness   

        # # Prepare segmentation mask and RGB mask with color encoding
        # pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 23 is the background class
        # for _b_idx, (_b_binary_pred_masks, _b_pred_class) in enumerate(zip(binary_pred_masks, pred_class)):
        #     for mask, class_id in zip(_b_binary_pred_masks, _b_pred_class):
        #         if class_id == BACKGROUND:
        #             continue
        #         pos = np.where(mask == True)
        #         if len(pos[0]) > 0:
        #             pred_seg_class_mask[_b_idx, 0, pos[0], pos[1]] = class_id
        #         else:
        #             continue

        # ---- Greedy fill: highest-score queries claim pixels first ----
        pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 16 is the background class
        pred_rgb_mask = np.zeros((B, H, W, 3), dtype=np.uint8)

        for b in range(B):
            order = np.argsort(-query_score[b])                                         # best first
            taken = np.zeros((H, W), dtype=bool)

            for q in order:
                cls = int(pred_class[b, q])
                if cls == BACKGROUND:
                    continue

                m = binary_pred_masks[b, q]
                if not m.any():
                    continue

                new_pixels = m & (~taken)
                added = int(new_pixels.sum())
                total = int(m.sum())
                if total == 0 or added / (total + 1e-6) < ADD_RATIO_THR:
                    # near-duplicate or negligible contribution
                    continue

                ys, xs = np.where(new_pixels)
                pred_seg_class_mask[b, 0, ys, xs] = cls
                taken[ys, xs] = True

        gt_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 23 is the background class
        # gt_rgb_mask = np.zeros((B, H, W, 3), dtype=np.uint8)
        for batch_idx in range(B):
            pos = batched_video_data_val.obj_to_frame_idx[:,:,0] == batch_idx  # The objects in the batch are mixed
            gt_mask = masks_val[pos, :, :].cpu().numpy()  # Get the GT masks in the batch
            gt_class_id = gt_labels[pos].cpu().numpy()  # Get the GT class id in the batch
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
            img = batched_video_data_val.img_batch[batch_idx][0].permute(1,2,0).cpu().numpy()
            
            # Check if the image is black (0,0,0) along the height dimension
            pos_padding = np.where((img[:,:,0] <= 0) == False)
            start = pos_padding[0][0]
            end = pos_padding[0][-1]
            
            # Slice the predictions and ground truth masks
            if start == 32:
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
        instance_ids = np.zeros_like(preds)  # Almost like semantic segmentation, therefore all objects are the first instances
        preds = np.concatenate((preds, instance_ids), axis=3)

        # Prepare stuff and things mask for GT
        target = gt_seg_class_mask[:, 0, :, :, None]
        instance_ids_gt = np.zeros_like(target)  # Almost like semantic segmentation, therefore all objects are the first instances
        target = np.concatenate((target, instance_ids_gt), axis=3)

        # Convert to torch tensors
        preds = torch.tensor(preds, device=device)
        target = torch.tensor(target, device=device)

        # test_pq_list = []
        # for i in range(B):
        #     test_pred = preds[i]
        #     test_target = target[i]
        #     test_pq = panoptic_quality(test_pred, test_target)
        #     test_pq_list.append(test_pq)
        # mean_pq = torch.tensor(test_pq_list).mean()
        # print(f'Mean PQ: {mean_pq}')
        # print(test_pq_list)

        # Compute PQ
        pq = panoptic_quality(preds, target)
        pq_list.append(pq)
        print(f'Iter {idx} PQ: {pq}')
        writer.add_scalar(f"PQ/test_{batch_size}", pq, idx)
        writer.flush() if idx % 10 == 0 else None
        torch.cuda.empty_cache()

average_pq = sum(pq_list) / len(pq_list)
print(f"Average PQ: {average_pq}")
writer.add_scalar(f"PQ/test_{batch_size}_average", average_pq, 1)
writer.flush()
writer.close()
print("Finished!")
