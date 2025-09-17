import os
import sys
from functools import partial
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

def process_outputs(view_output, unique_object_identfier):
    # Convert mask prediction logits to binary masks
    mask_logits = torch.cat([view_output[k]['pred_masks_high_res'].float() 
                               for k in view_output.keys()], dim=0)  # [B,Q,H,W]
    mask_prob = mask_logits.sigmoid()                                # [B,Q,H,W]
    binary_pred_masks = (mask_prob > 0.5).cpu().numpy()  

    # Convert class logits to class predictions
    pred_logits = torch.cat([view_output[k]['pred_logits'].type(torch.float32) 
                               for k in view_output.keys()], dim=0)
    probs = pred_logits.softmax(-1)
    class_probs = probs[..., :-1]
    pred_class = class_probs.argmax(-1).cpu().numpy()
    pred_class_prob = class_probs.max(-1).values.cpu().numpy()

    gt_labels = unique_object_identfier[:,:,1]
    return binary_pred_masks, mask_prob, pred_class, pred_class_prob, gt_labels

def _prepare_seg_rgb_masks(binary_pred_masks, pred_class):
    """
    Prepare segmentation mask and RGB mask with color encoding.
    """
    B, H, W = binary_pred_masks.shape[0], binary_pred_masks.shape[2], binary_pred_masks.shape[3]
    pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 16 is the background class
    
    for _b_idx, (_b_binary_pred_masks, _b_pred_class) in enumerate(zip(binary_pred_masks, pred_class)):
        for mask, class_id in zip(_b_binary_pred_masks, _b_pred_class):
            if class_id == BACKGROUND:
                continue
            pos = np.where(mask == True)
            if len(pos[0]) > 0:
                pred_seg_class_mask[_b_idx, 0, pos[0], pos[1]] = class_id
            else:
                continue
    return pred_seg_class_mask, None

def _prepare_gt_seg_rgb_masks(masks_val, gt_labels, obj_to_frame_idx):
    """
    Prepare ground truth segmentation mask and RGB mask with color encoding.
    """
    B, H, W = masks_val.shape[0], masks_val.shape[2], masks_val.shape[3]
    # Initialize masks
    gt_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 16 is the background class
    for batch_idx in range(B):
        pos = obj_to_frame_idx[0,:,1] == batch_idx  # The objects in the batch are mixed
        gt_mask = masks_val[0, pos, :, :].cpu().numpy()  # Get the GT masks in the batch
        gt_class_id = gt_labels[0, pos].cpu().numpy()  # Get the GT class id in the batch
        for mask, class_id in zip(gt_mask, gt_class_id):
            pos = np.where(mask == True)
            if len(pos[0]) > 0:
                gt_seg_class_mask[batch_idx, 0, pos[0], pos[1]] = class_id
            else:
                continue
    return gt_seg_class_mask, None

def calculate_panoptic_quality(panoptic_quality_fn, pred_seg_class_mask, gt_seg_class_mask):
    """
    Calculate panoptic quality metric.
    """
    # Prepare stuff and things mask for prediction
    preds = pred_seg_class_mask[:, 0, :, :, None]
    instance_ids = np.zeros_like(preds)  # Almost like semantic segmentation, therefore all objects are the first instances
    preds = np.concatenate((preds, instance_ids), axis=3)

    # Prepare stuff and things mask for GT
    target = gt_seg_class_mask[:, 0, :, :, None]
    instance_ids_gt = np.zeros_like(target)  # Almost like semantic segmentation, therefore all objects are the first instances
    target = np.concatenate((target, instance_ids_gt), axis=3)

    # Convert to torch tensors
    preds = torch.tensor(preds)
    target = torch.tensor(target)

    # Prepare panoptic quality metric
    pq = panoptic_quality_fn(preds, target)
    print(f'Prediction: {preds.shape}, Target: {target.shape}')
    print(f"PQ: {pq}")
    return pq

def _prepare_seg_mask_with_masksness(mask_prob, binary_pred_masks, pred_class, pred_class_prob,
                                     BACKGROUND=23, ADD_RATIO_THR=0.2):
    mask_prob_np = mask_prob.cpu().numpy()
    B, Q, H, W = binary_pred_masks.shape
    maskness = np.zeros((B, Q), dtype=np.float32)
    for b in range(B):
        for q in range(Q):
            m = binary_pred_masks[b, q]
            maskness[b, q] = mask_prob_np[b, q][m].mean() if m.any() else mask_prob_np[b, q].mean()

    # query score for ordering
    query_score = pred_class_prob * maskness

    # ---- Greedy fill: highest-score queries claim pixels first ----
    pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 16 is the background class

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
    return pred_seg_class_mask, None

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
        'config': '09_09_11_28_vid_model_1c/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_09_11_28_vid_model_1c/checkpoints/checkpoint_18.pt',
        },
    'base4': { # THIS IS MV MODEL 23 Q
        'config': '09_11_18_13_MV/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_11_18_13_MV/checkpoints/checkpoint_3.pt',
        },
    'base5': { # THIS IS MV MODEL 23 Q video trained
        'config': '09_16_13_47_MV/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_16_13_47_MV/checkpoints/checkpoint_1.pt',
        },
}

# Model
model_size = 'base5'
len_video = 4
# Tensorboard
model_name = model_size_dict[model_size]['config'].split('/')[0]
writer = SummaryWriter(f'./tb_logs/{model_name}_multiview_video{len_video}_eval/')

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
                           collate_fn=partial(collate_fn_wrapper, num_frames=len_video),
                           batch_size=batch_size,
                           multiview=True,
                           depth_image=True,
                           num_workers=0,
                           shuffle=shuffle,)
print(f'Length of the dataset! {len(test_dataset)}')

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
panoptic_quality_fn = PanopticQuality(stuffs=stuff, things=things)
pq_list = []

# Run the model
with torch.no_grad():
    submodel.eval()
    for idx, batch in tqdm(enumerate(test_loader)):
        # print(f"Processing {idx}th sample")
        batched_video_data_val = batch.to(device)
        masks_val0 = batch[0].masks.to(device)
        masks_val1 = batch[1].masks.to(device)
        masks_val2 = batch[2].masks.to(device)
        with autocast(device_type=device, dtype=amp_type):
            all_frame_outputs_val = submodel(batched_video_data_val)

        print('CP: 1') if flag_print_logs else None

        # Process outputs, get binary masks and class ids
        query_post_processing = True
        binary_pred_masks0, mask_prob0, pred_class0, pred_class_prob0, gt_labels0 = process_outputs(all_frame_outputs_val[0], 
                                                    batched_video_data_val[0].metadata.unique_objects_identifier)

        binary_pred_masks1, mask_prob1, pred_class1, pred_class_prob1, gt_labels1 = process_outputs(all_frame_outputs_val[1], 
                                                    batched_video_data_val[1].metadata.unique_objects_identifier)

        binary_pred_masks2, mask_prob2, pred_class2, pred_class_prob2, gt_labels2 = process_outputs(all_frame_outputs_val[2], 
                                                    batched_video_data_val[2].metadata.unique_objects_identifier)

        B, N, H, W = binary_pred_masks0.shape  # Size with padding

        # Open Memory
        del batch
        del all_frame_outputs_val

        # Prepare segmentation masks
        if query_post_processing:
            pred_seg_class_mask0, pred_rgb_mask0 = _prepare_seg_mask_with_masksness(mask_prob0, binary_pred_masks0, pred_class0, pred_class_prob0,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
            pred_seg_class_mask1, pred_rgb_mask1 = _prepare_seg_mask_with_masksness(mask_prob1, binary_pred_masks1, pred_class1, pred_class_prob1,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
            pred_seg_class_mask2, pred_rgb_mask2 = _prepare_seg_mask_with_masksness(mask_prob2, binary_pred_masks2, pred_class2, pred_class_prob2,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
        else:
            pred_seg_class_mask0, pred_rgb_mask0 = _prepare_seg_rgb_masks(binary_pred_masks0, pred_class0)
            pred_seg_class_mask1, pred_rgb_mask1 = _prepare_seg_rgb_masks(binary_pred_masks1, pred_class1)
            pred_seg_class_mask2, pred_rgb_mask2 = _prepare_seg_rgb_masks(binary_pred_masks2, pred_class2)

        gt_seg_class_mask0, gt_rgb_mask0 = _prepare_gt_seg_rgb_masks(masks_val0, gt_labels0, batched_video_data_val[0].obj_to_frame_idx)
        gt_seg_class_mask1, gt_rgb_mask1 = _prepare_gt_seg_rgb_masks(masks_val1, gt_labels1, batched_video_data_val[1].obj_to_frame_idx)
        gt_seg_class_mask2, gt_rgb_mask2 = _prepare_gt_seg_rgb_masks(masks_val2, gt_labels2, batched_video_data_val[2].obj_to_frame_idx)

        print('CP: 2') if flag_print_logs else None

        # Check if the image is black (0,0,0) along the height dimension
        ref_img = batched_video_data_val[0][0].img_batch[0].cpu().numpy()
        is_black_row = (ref_img[0,:,:] <= 0)  # Check along width and channels
        pos_padding = np.where(is_black_row == False)
        start = pos_padding[0][0]

        print('CP: 3') if flag_print_logs else None

        # Open Memory
        del batched_video_data_val

        print('CP: 4') if flag_print_logs else None

        # Calculate PQ
        pq0 = calculate_panoptic_quality(panoptic_quality_fn, pred_seg_class_mask0, gt_seg_class_mask0)
        pq1 = calculate_panoptic_quality(panoptic_quality_fn, pred_seg_class_mask1, gt_seg_class_mask1)
        pq2 = calculate_panoptic_quality(panoptic_quality_fn, pred_seg_class_mask2, gt_seg_class_mask2)

        mean_pq = (pq0 + pq1 + pq2) / 3
        print(f"Mean PQ: {mean_pq}")

        # Compute PQ
        # pq = panoptic_quality(preds, target)
        pq_list.append(mean_pq)
        print(f'Iter {idx} PQ: {mean_pq}')
        writer.add_scalar(f"PQ/test_{batch_size}", mean_pq, idx)
        writer.flush() if idx % 10 == 0 else None
        torch.cuda.empty_cache()

average_pq = sum(pq_list) / len(pq_list)
print(f"Average PQ: {average_pq}")
writer.add_scalar(f"PQ/test_{batch_size}_average", average_pq, 1)
writer.flush()
writer.close()
print("Finished!")
