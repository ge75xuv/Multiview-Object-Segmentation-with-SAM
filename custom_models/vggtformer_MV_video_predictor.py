import os
import sys
from functools import partial
# from copy import deepcopy

project_dir = os.path.dirname(os.getcwd())
print(project_dir)
sys.path.append(project_dir)

from hydra import initialize
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
from .dataset.mini_dataset_vggt import MiniDatasetVGGT
from .custom_model_builder import build_sam2former

# torch.multiprocessing.set_start_method(
#         "spawn"
#     )  # CUDA runtime does not support `fork`

def process_outputs(view_output, view_idx, unique_object_identfier):
    # Convert mask prediction logits to binary masks
    binary_pred_masks = view_output['pred_masks_high_res'][:,view_idx].sigmoid() > 0.5
    binary_pred_masks = binary_pred_masks.cpu().numpy()

    gt_labels = unique_object_identfier[:,:,1]
    return binary_pred_masks, gt_labels

def _prepare_seg_rgb_masks(binary_pred_masks):
    """
    Prepare segmentation mask and RGB mask with color encoding.
    """
    B, H, W = binary_pred_masks.shape[0], binary_pred_masks.shape[2], binary_pred_masks.shape[3]
    pred_seg_class_mask = np.ones((B, 1, H, W), dtype=np.int8) * 23  # 16 is the background class
    
    for _b_idx, _b_binary_pred_masks in enumerate(binary_pred_masks):
        for class_id, mask in enumerate(_b_binary_pred_masks):
            pos = np.where(mask == True)
            if len(pos[0]) > 0:
                pred_seg_class_mask[_b_idx, 0, pos[0], pos[1]] = class_id
            else:
                continue

    return pred_seg_class_mask

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
    # print(f'Prediction: {preds.shape}, Target: {target.shape}')
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
    'base1': {
        'config': '09_04_15_01_VGGT/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_04_15_01_VGGT/checkpoints/checkpoint_9.pt',
        },
    'base2': {
        'config': '09_10_16_45/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_10_16_45/checkpoints/checkpoint_1.pt',
        },
    'base3': {
        'config': '09_14_07_30/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_14_07_30/checkpoints/checkpoint_5.pt',
        },
    'base4': {  # MAIN CAMERAS
        'config': '09_18_16_17_VGGT/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_18_16_17_VGGT/checkpoints/checkpoint_19.pt',  # 9
        },
    'base5': {  # ALL CAMERAS
        'config': '09_24_10_35_VGGT/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_24_10_35_VGGT/checkpoints/checkpoint_1.pt',
        },
    'base6': {  # ALL CAMERAS
        'config': '09_24_14_29_VGGT/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/09_24_14_29_VGGT/checkpoints/checkpoint_15.pt',
        },
    'base7': {
        'config': '10_03_22_41/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/10_03_22_41/checkpoints/checkpoint_66.pt',
        },
    'base8': {
        'config': '10_20_10_06/config_resolved.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/10_20_10_06/checkpoints/checkpoint_16.pt',
        },
}
# Model
model_size = 'base8'
config = model_size_dict[model_size]['config']
ck = model_size_dict[model_size]['ck']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Hydra init
config_path = f'../sam2_logs/{config.split("/")[0]}/'
print("Config path:", config_path)
config = config.split('/')[1]
try:
    initialize(version_base=None, config_path=config_path, job_name="predict_run")
except ValueError:
    pass
# hydra_overrides_extra = ['scratch.multiview=true', 'trainer.model.multiview=true']
hydra_overrides_extra = []
amp_type = torch.bfloat16 if device == 'cuda' else torch.float16

# Read config and init model
cfg = compose(config_name=config, overrides=hydra_overrides_extra)
# Register new operation becaise of learning rate
try:
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
except ValueError:
    print('Operation divide already registered')
# Resolve
OmegaConf.resolve(cfg)
print(f'OmegaConf resolved successfully')
# Instantiate model, loss, load weights, freeze backbone
submodel = instantiate(cfg.trainer.model, _recursive_=True)
obj_labels = cfg['scratch']['obj_labels']
submodel.eval()

# Load weights
sd = torch.load(ck, map_location="cpu", weights_only=True)["model"]
missing_keys, unexpected_keys = submodel.load_state_dict(sd, strict=False)
submodel = submodel.to(device)
print(missing_keys, unexpected_keys)

# All cameras flag
ALL_CAMERAS = True
submodel.all_cameras = ALL_CAMERAS
print(f"ALL_CAMERAS is set to {ALL_CAMERAS} for the model and the dataset")

# Tensorboard
model_name = model_size_dict[model_size]['config'].split('/')[0]
writer = SummaryWriter(f'./tb_logs/{model_name}_multiview_AC_{ALL_CAMERAS}_v2_eval_test/')

# Dataset
len_video = 1
input_image_size = 518
batch_size = 4
shuffle = False
transforms = [ComposeAPI([torch.nn.Identity(),])]
test_dataset = MiniDatasetVGGT('test',
                           num_frames=len_video,
                           input_image_size=input_image_size,
                           object_labels=obj_labels,
                           transforms=transforms,
                           collate_fn=partial(collate_fn_wrapper, num_frames=len_video),
                           batch_size=batch_size,
                           multiview=True,
                           all_cameras=ALL_CAMERAS,
                           num_workers=0,
                           shuffle=shuffle,)
print(f'Length of the dataset! {len(test_dataset)}')

# Seed
seed = 2 # Check seed 123 index 19966
torch.manual_seed(seed)
np.random.seed(seed)
flag_print_logs = False

# Image
len_objects = len(obj_labels)
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
things = {LABEL_PROJECTION_MAP['default'][idx]['label'] for idx in obj_labels}
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
        query_post_processing = False
        binary_pred_masks0, gt_labels0 = process_outputs(all_frame_outputs_val, 0, batched_video_data_val[0].metadata.unique_objects_identifier)
        binary_pred_masks1, gt_labels1 = process_outputs(all_frame_outputs_val, 1, batched_video_data_val[1].metadata.unique_objects_identifier)
        binary_pred_masks2, gt_labels2 = process_outputs(all_frame_outputs_val, 2, batched_video_data_val[2].metadata.unique_objects_identifier)

        B, N, H, W = binary_pred_masks0.shape  # Size with padding

        # Open Memory
        del batch
        del all_frame_outputs_val

        # Prepare segmentation masks
        if query_post_processing and False:
            pred_seg_class_mask0, pred_rgb_mask0 = _prepare_seg_mask_with_masksness(mask_prob0, binary_pred_masks0, pred_class0, pred_class_prob0,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
            pred_seg_class_mask1, pred_rgb_mask1 = _prepare_seg_mask_with_masksness(mask_prob1, binary_pred_masks1, pred_class1, pred_class_prob1,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
            pred_seg_class_mask2, pred_rgb_mask2 = _prepare_seg_mask_with_masksness(mask_prob2, binary_pred_masks2, pred_class2, pred_class_prob2,
                                            BACKGROUND=BACKGROUND, ADD_RATIO_THR=ADD_RATIO_THR)
        else:
            pred_seg_class_mask0 = _prepare_seg_rgb_masks(binary_pred_masks0)
            pred_seg_class_mask1 = _prepare_seg_rgb_masks(binary_pred_masks1)
            pred_seg_class_mask2 = _prepare_seg_rgb_masks(binary_pred_masks2)


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
