#%%
import time
from typing import List, Tuple, Optional, Dict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402

import torch

from helpers.configurations import OBJECTS_EPIPOLAR, LABEL_PROJECTION_MAP, OBJECTS_DEPTH_PROJ
from models.sam2former.point_cloud_mask import point_cloud_mask

def fundamental_from_ext_intr(ext_AB, K_B, K_A):
    # ext_AB maps from A to B (your ext_* variables already do this)
    R = ext_AB[:3, :3]
    t = ext_AB[:3, 3]
    tx = torch.tensor([[0., -t[2],  t[1]],
                       [t[2],  0., -t[0]],
                       [-t[1], t[0],  0.]], device=ext_AB.device, dtype=ext_AB.dtype)
    E = tx @ R
    F = torch.linalg.inv(K_B).T @ E @ torch.linalg.inv(K_A)
    return F

def soft_epipolar_project(
    logits_src,           # (H,W) source logits (not thresholded)
    K_src, ext_src,       # intrinsics/extrinsics of source
    K_tgt, ext_tgt,       # intrinsics/extrinsics of target
    out_hw=None,          # output size (h,w); default = src size
    sigma_px=2.0,         # band thickness at the (possibly downsampled) resolution
    chunk_size=None       # optional: process source pixels in chunks to save RAM
):
    """
    Returns a (H_tgt, W_tgt) heatmap. Completely differentiable.
    """
    device = logits_src.device
    dtype  = logits_src.dtype
    Hs, Ws = logits_src.shape
    if out_hw is None:
        Ht, Wt = Hs, Ws
    else:
        Ht, Wt = out_hw

    # Optionally downsample logits_src first, to keep compute low.
    # E.g., if (Hs,Ws) is big, do a bilinear downsample here, adjust sigma_px accordingly.

    # Fundamental matrix F_{src->tgt}
    # ext_tgt <- world <- src : you already compute ext_src_to_tgt = inv(ext_tgt) @ ext_src; reuse that
    inv_tgt = torch.eye(4, device=device, dtype=dtype)
    inv_tgt[:3,:3] = ext_tgt[:3,:3].T
    inv_tgt[:3, 3] = -ext_tgt[:3,:3].T @ ext_tgt[:3,3]
    ext_src_to_tgt = inv_tgt @ ext_src
    F = fundamental_from_ext_intr(ext_src_to_tgt, K_tgt, K_src)   # (3,3)

    # Source homogeneous grid (Hs*Ws, 3)
    ys, xs = torch.meshgrid(torch.arange(Hs, device=device),
                            torch.arange(Ws, device=device), indexing='ij')
    ones = torch.ones_like(xs, dtype=dtype)
    Xs = torch.stack([xs, ys, ones], dim=-1).reshape(-1,3)        # (Ns,3)

    # Weights from logits (soft, no threshold!)
    p = torch.sigmoid(logits_src).reshape(-1)                     # (Ns,)

    # Lines in target for each source pixel: l' = F x
    L = (F @ Xs.T).T                                             # (Ns,3) rows are (a,b,c)

    # Target grid (Ht,Wt,3)
    y2, x2 = torch.meshgrid(torch.arange(Ht, device=device),
                            torch.arange(Wt, device=device), indexing='ij')
    X2 = torch.stack([x2, y2, torch.ones_like(x2, dtype=dtype)], dim=-1)  # (Ht,Wt,3)

    # We’ll accumulate into this heatmap
    heat = torch.zeros((Ht, Wt), device=device, dtype=dtype)

    # Chunk over source pixels if needed to limit memory
    Ns = L.shape[0]
    if chunk_size is None:
        chunk_size = Ns  # single chunk

    for start in range(0, Ns, chunk_size):
        end = min(start + chunk_size, Ns)
        a = L[start:end, 0].view(-1, 1, 1)
        b = L[start:end, 1].view(-1, 1, 1)
        c = L[start:end, 2].view(-1, 1, 1)

        # distance from each target pixel to each line
        num = (a * X2[...,0] + b * X2[...,1] + c).abs()
        den = torch.clamp((a*a + b*b).sqrt(), min=1e-8)
        d = num / den                                            # (Nc, Ht, Wt)

        # soft band
        band = torch.exp(-0.5 * (d / sigma_px)**2)               # (Nc, Ht, Wt)

        # weight by p (source pixel probability)
        w = p[start:end].view(-1, 1, 1)
        heat = heat + torch.sum(w * band, dim=0)

    # Optional: normalize (e.g., divide by sum of p or clamp)
    # heat = heat / (p.sum().clamp_min(1e-6))

    return heat  # (Ht,Wt)


def random_sample_mask_points(mask: torch.Tensor, n_points: int, threshold: float = 0.5) -> Optional[torch.Tensor]:
    # Handle multi-channel masks by summing across channels
    binary_mask = torch.sum(mask.sigmoid() > threshold, dim=0)

    # Get all valid pixel coordinates
    y, x = torch.where(binary_mask)

    if len(x) == 0:
        return None

    # If we have fewer points than requested, return all available points
    if len(x) <= n_points:
        return torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)

    # Randomly sample n_points indices
    indices = torch.randperm(len(x))[:n_points]
    sampled_x = x[indices]
    sampled_y = y[indices]

    return torch.cat([sampled_x.unsqueeze(1), sampled_y.unsqueeze(1)], dim=1)

@torch.no_grad()
def epipolar_mask_preprocess(
    mask_cam0: torch.Tensor,
    mask_cam1: torch.Tensor,
    mask_cam2: torch.Tensor,
    n_points: Optional[int] = 4096,
    ):
    # Another reason for us to use temporality. It is not possible to make a differentiable masking here
    # The batch operation for different objects cannot be done here since the size of the masks are going to be varying size
    pts_cam0 = random_sample_mask_points(mask_cam0, n_points=n_points, threshold=0.5)
    pts_cam1 = random_sample_mask_points(mask_cam1, n_points=n_points, threshold=0.5)
    pts_cam2 = random_sample_mask_points(mask_cam2, n_points=n_points, threshold=0.5)    

    return pts_cam0, pts_cam1, pts_cam2

def epipolar_mask_post_process(
    epi_masks_for_views: Dict[int, Tuple[torch.Tensor, float]],
    _size: Tuple[int, int],
    epipolar_weight: List[float],
    device: Optional[str] = 'cpu'
):
    mask = torch.zeros([3, 2, _size[0], _size[1]], dtype=torch.float32, device=device)
    # epi_masks keys for views and values are the intersection points from 2 other views
    for view_id in range(3):
        other_views = [v for v in range(3) if v != view_id]        # e.g. if view=0 → [1,2]
        weights = [epipolar_weight[v] for v in other_views]
        for i in range(2):  # two sets of epipolar intersections
            coords = epi_masks_for_views[view_id][i]  # [N, 2]
            if coords.numel() == 0:
                continue

            coords_rounded = coords.floor().long()
            x, y = coords_rounded[:, 0], coords_rounded[:, 1]
            mask[view_id, i, y, x] += weights[i]  # Add the weight to the mask

    return mask

@torch.no_grad()
def epipolar_main(camera_int_ext: List[torch.Tensor],
                  predictions0: Dict[str, torch.Tensor],
                  predictions1: Dict[str, torch.Tensor],
                  predictions2: Dict[str, torch.Tensor],
                  num_queries: int = 23,
                  n_points: Optional[int] = 4096,
                  step: Optional[int] = 2,
                  depth_images=None,
                  ):
    """Wrap the epipolar functionality to be used in the training class."""
    # Get the logits from each camera's predictions.
    pred_logits0 = torch.Tensor(predictions0['pred_logits']).softmax(dim=-1)
    pred_logits1 = torch.Tensor(predictions1['pred_logits']).softmax(dim=-1)
    pred_logits2 = torch.Tensor(predictions2['pred_logits']).softmax(dim=-1)
    # Class
    pred_class0 = pred_logits0.argmax(dim=-1)
    pred_class1 = pred_logits1.argmax(dim=-1)
    pred_class2 = pred_logits2.argmax(dim=-1)

    # Initialize the masks
    device = pred_class0.device
    o, h, w = predictions0['pred_masks_high_res'].shape[-3:]
    unique_projected_labels = set([obj['label'] for obj in LABEL_PROJECTION_MAP['default'].values()])
    # unique_projected_labels = [i for i in range(num_queries)]  # TODO: get it from the config num_classes
    num_classes = len(unique_projected_labels)
    masks = torch.zeros([num_classes, 3, 2, h, w], dtype=torch.float32, device=device)

    # Send camera matrices to cuda
    camera_int_ext = [(cam_mtrcs[0].to(device), cam_mtrcs[1].to(device)) for cam_mtrcs in camera_int_ext]

    # Object position dictionary (It is bad solution to put num_class - 1 because cemeter (class id 22) is not in the dataset)
    # Real solution is to increase the queries, we have models like that too.
    object_pos_label = {view_idx:{pos_idx: num_classes-1 for pos_idx in range(o)} for view_idx in range(3)}

    # Iterate over the objects of interest and get the masks.
    for obj_idx in LABEL_PROJECTION_MAP['default'].values():  #TODO get it from the config num_classes
        # Get the object index from the label projection map
        obj_idx = obj_idx['label']

        # Find the positions of the object in each camera's predictions
        pos_obj0 = torch.where(pred_class0 == obj_idx)  # pos_obj0 is a tuple of (batch_idx, class_idx)
        pos_obj1 = torch.where(pred_class1 == obj_idx)  # pos_obj1 is a tuple of (batch_idx, class_idx)
        pos_obj2 = torch.where(pred_class2 == obj_idx)  # pos_obj2 is a tuple of (batch_idx, class_idx)

        # Put the obj_idx in the corresponding position, use fromkeys in case there are multiple instances
        object_pos_label[0].update(dict.fromkeys(pos_obj0[1].tolist(), obj_idx))
        object_pos_label[1].update(dict.fromkeys(pos_obj1[1].tolist(), obj_idx))
        object_pos_label[2].update(dict.fromkeys(pos_obj2[1].tolist(), obj_idx))

        # Get the position and continue if the object is not epipolarly tracked
        epi_object = obj_idx in OBJECTS_EPIPOLAR
        depth_object = obj_idx in OBJECTS_DEPTH_PROJ
        if not epi_object and not depth_object:
            masks[obj_idx] = torch.zeros([3, 2, h, w], dtype=torch.float32)
            continue

        # Get the masks for the first three cameras
        mask_cam0 = predictions0['pred_masks_high_res'][pos_obj0]  # Slice the correct objects from the indeces.
        mask_cam1 = predictions1['pred_masks_high_res'][pos_obj1]  # Sum is only needed if multiple
        mask_cam2 = predictions2['pred_masks_high_res'][pos_obj2]  # instances of the same object.

        # Preprocess the masks to get points
        pts_cam0, pts_cam1, pts_cam2 = epipolar_mask_preprocess(mask_cam0, mask_cam1, mask_cam2, n_points=n_points)

        # Technically we dont need to check if the points are None, since the epipolar mask function will handle it
        # But we do it here to avoid unnecessary computation if the points are not available
        if pts_cam0 is None and pts_cam1 is None and pts_cam2 is None:
            masks[obj_idx] = torch.zeros([3, 2, h, w], dtype=torch.float32)
            continue
        
        # Intrinsic and extrinsic
        K0, ext0 = camera_int_ext[0]
        K1, ext1 = camera_int_ext[1]
        K2, ext2 = camera_int_ext[2]

        # Get the logits for the specific object query in each view (H,W)
        # If you have per-query logits, pick the right slice; if you have per-class logits, use the object’s logit map.
        logits0 = predictions0['pred_masks_high_res'][pos_obj0]   # shape (n_inst, H, W)
        logits1 = predictions1['pred_masks_high_res'][pos_obj1]
        logits2 = predictions2['pred_masks_high_res'][pos_obj2]

        # If multiple instances, you can sum their contributions (still differentiable).
        log0 = logits0.sum(dim=0) if logits0.numel() else torch.zeros(h, w, device=device)
        log1 = logits1.sum(dim=0) if logits1.numel() else torch.zeros(h, w, device=device)
        log2 = logits2.sum(dim=0) if logits2.numel() else torch.zeros(h, w, device=device)

        # Build the three pairwise projections as needed
        # send K/ext to device/dtype once outside the loop
        heat_1to0 = soft_epipolar_project(log1, K1, ext1, K0, ext0, out_hw=(h,w), sigma_px=2.0, chunk_size=8192)
        heat_2to0 = soft_epipolar_project(log2, K2, ext2, K0, ext0, out_hw=(h,w), sigma_px=2.0, chunk_size=8192)

        # Combine into your 3x2 structure if you want to keep that API,
        # but now they’re *dense* heatmaps, not sparse points:
        masks[obj_idx, 0, 0] = heat_1to0
        masks[obj_idx, 0, 1] = heat_2to0

        # Find the prob of that object
        prob_obj0 = pred_logits0[pos_obj0].max()  if pos_obj0[0].numel() > 0 else 0.0
        prob_obj1 = pred_logits1[pos_obj1].max()  if pos_obj1[0].numel() > 0 else 0.0
        prob_obj2 = pred_logits2[pos_obj2].max()  if pos_obj2[0].numel() > 0 else 0.0

        # Epipolar mask post processing
        # mask = epipolar_mask_post_process(epi_mask_for_views, mask_cam0.shape[-2:], [prob_obj0, prob_obj1, prob_obj2], device)
        # masks[obj_idx] = mask
    return masks, object_pos_label

 #%%
if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))    

    # Load predictions and camera intrinsics/extrinsics
    try:
        with open('./temp/predictions.json', 'r') as f:
            predictions = json.load(f)
        with open('./temp/camera_int_ext.json', 'r') as f:
            camera_int_ext_json = json.load(f)
    except FileNotFoundError:
        with open('../../../temp/predictions.json', 'r') as f:
            predictions = json.load(f)
        with open('../../../temp/camera_int_ext.json', 'r') as f:
            camera_int_ext_json = json.load(f)
#%%
    # Convert camera intrinsics and extrinsics to tensors
    camera_int_ext = [(torch.tensor(cam[0]), torch.tensor(cam[1])) for cam in camera_int_ext_json.values()]

    # Reshape the logits
    pred_class = torch.Tensor(predictions['pred_logits']).softmax(dim=-1).argmax(dim=-1)
    h,w = torch.Tensor(predictions['pred_masks_high_res'][0]).shape[-2:]
    device = 'cpu'
    masks = torch.zeros(23, 3, 2, h, w, dtype=torch.float32, device=device)  # Assuming 23 classes

    OBJECTS_EPIPOLAR = [10]  # NOTE: Test for multi instances
#%%
    for obj_idx in OBJECTS_EPIPOLAR:
        # NOTE by visial observation, we know that the first camera is the 1st in the batch
        cam0_ids = torch.where(pred_class[0] == obj_idx)[0]
        cam1_ids = torch.where(pred_class[1] == obj_idx)[0]
        cam2_ids = torch.where(pred_class[2] == obj_idx)[0]

        # Get the masks for the first three cameras
        # NOTE by visial observation, we know that the first camera is the 1st in the batch
        mask_cam0 = torch.Tensor(predictions['pred_masks_high_res'][0])[tuple(cam0_ids), :, :]  # Slice the correct objects
        mask_cam1 = torch.Tensor(predictions['pred_masks_high_res'][1])[tuple(cam1_ids), :, :]  # from the indeces
        mask_cam2 = torch.Tensor(predictions['pred_masks_high_res'][2])[tuple(cam2_ids), :, :]

        # NOTE: test lacking visibility
        # mask_cam0 = torch.zeros_like(mask_cam0)
        # Intrinsic and extrinsic
        K0, ext0 = camera_int_ext[0]
        K1, ext1 = camera_int_ext[1]
        K2, ext2 = camera_int_ext[2]

        # Get the logits for the specific object query in each view (H,W)
        # If you have per-query logits, pick the right slice; if you have per-class logits, use the object’s logit map.
        logits0 = mask_cam0   # shape (n_inst, H, W)
        logits1 = mask_cam1
        logits2 = mask_cam2

        # If multiple instances, you can sum their contributions (still differentiable).
        log0 = logits0.sum(dim=0) if logits0.numel() else torch.zeros(h, w, device=device)
        log1 = logits1.sum(dim=0) if logits1.numel() else torch.zeros(h, w, device=device)
        log2 = logits2.sum(dim=0) if logits2.numel() else torch.zeros(h, w, device=device)

        # Build the three pairwise projections as needed
        # send K/ext to device/dtype once outside the loop
        heat_1to0 = soft_epipolar_project(log1, K1, ext1, K0, ext0, out_hw=(h,w), sigma_px=2.0, chunk_size=1024)
        heat_2to0 = soft_epipolar_project(log2, K2, ext2, K0, ext0, out_hw=(h,w), sigma_px=2.0, chunk_size=1024)

        # Combine into your 3x2 structure if you want to keep that API,
        # but now they’re *dense* heatmaps, not sparse points:
        masks[obj_idx, 0, 0] = heat_1to0
        masks[obj_idx, 0, 1] = heat_2to0
        
        start_time_all = time.time()
        

# TODO: Add the visualization of the epipolar lines and intersection points
# NOTE: Visualization is added by ipynb debug
# TODO: Batch computation of the epipolar lines for multiple objects
# NOTE: Batch computation cannot be done since the number of points may not be varying
# among the batches
# TODO: Convert points to masks
# NOTE: Done
# TODO: Test single epipolar line sampling
# NOTE: Single epipolar line sampling works
# TODO: Intersection points are extremely repetitive, need to filter them
# NOTE: Filtering works

# %%
# plt.imshow(masks[0].cpu().numpy(), cmap='gray')
# plt.title('Epipolar Mask for View 0')
# plt.show()