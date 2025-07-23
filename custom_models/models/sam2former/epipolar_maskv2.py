#%%
import time
from typing import List, Tuple, Optional, Dict

import torch

from custom_models.helpers.configurations import OBJECTS_EPIPOLAR, LABEL_PROJECTION_MAP

VIEW_COMBINATIONS = {
    0: ["epi_lines_1_to_0", "epi_lines_2_to_0"],
    1: ["epi_lines_0_to_1", "epi_lines_2_to_1"],
    2: ["epi_lines_0_to_2", "epi_lines_1_to_2"],
}

def skew(t):
    return torch.tensor([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def compute_epipolar_lines(ext, K_line, K_pt, pts1):
    # pts1: Nx2 array
    R, t = ext[:3, :3], ext[:3, 3]
    E = skew(t) @ R
    # NOTE inverse can be changed by custom computation
    inv_K1 = torch.linalg.inv(K_line)
    inv_K0 = torch.linalg.inv(K_pt)
    device = pts1.device
    F = (inv_K1.mT @ E @ inv_K0).to(device)  # TODO: K0, K1 replaced
    pts1_h = torch.hstack((pts1, torch.ones((pts1.shape[0], 1), device=device)))  # homogeneous
    lines2 = (F @ pts1_h.mT).mT  # Each row is [a, b, c]
    return lines2

def compute_inverse_transformation(ext):
    inv = torch.zeros((4, 4), dtype=torch.float32)
    inv[:3, :3] = ext[:3,:3].T
    inv[:3, 3] = -ext[:3,:3].T @ ext[:3, 3]
    return inv

def intersection(a, b, c, k, l, m):
    x = torch.round((-c * l + m * b) / (a * l - k * b)).type(torch.int32)
    y = torch.round((-c - a * x) / b).type(torch.int32)
    return torch.concatenate((x, y), axis=1)

def single_epiline(_size: Tuple[int, int], lines: torch.Tensor, step: int = 10):
    h, w = _size
    device = lines.device

    # Create x positions (1D), shape: [num_samples]
    x_vals = torch.arange(0, w, step, device=device)

    # lines: [N, 3], where each line is [a, b, c]
    a = lines[:, 0].unsqueeze(1)  # [N, 1]
    b = lines[:, 1].unsqueeze(1)
    c = lines[:, 2].unsqueeze(1)

    y_vals = (-a * x_vals + -c) / b  # [N, num_x]

    # Mask valid (on-image) points
    valid = (y_vals >= 0) & (y_vals < h)
    x_repeat = x_vals.unsqueeze(0).expand_as(y_vals)  # [N, num_x]

    x_valid = x_repeat[valid]
    y_valid = y_vals[valid]

    # Round or floor to nearest pixel if needed
    pts = torch.stack([x_valid, y_valid], dim=1)
    return pts

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
    epipolar_weight: float = 0.5,
    device: Optional[str] = 'cpu'
):
    mask = torch.zeros([3, _size[0], _size[1]], dtype=torch.float32, device=device)
    # epi_masks keys for views and values are the intersection points from 2 other views
    for view_id in range(3):
        for i in range(2):  # two sets of epipolar intersections
            coords = epi_masks_for_views[view_id][i]  # [N, 2]
            if coords.numel() == 0:
                continue

            coords_rounded = coords.floor().long()
            x, y = coords_rounded[:, 0], coords_rounded[:, 1]
            mask[view_id, y, x] += epipolar_weight

    return mask

@torch.no_grad()
def epipolar_mask(
    camera_int_ext: List[torch.Tensor],
    pts_cam0: torch.Tensor,
    pts_cam1: torch.Tensor,
    pts_cam2: torch.Tensor,
    _size: Tuple[int, int],
    step: int = 2,
    ):

    # Unpack camera intrinsics and extrinsics
    K0, ext0 = camera_int_ext[0]
    K1, ext1 = camera_int_ext[1]
    K2, ext2 = camera_int_ext[2]

    # Compute relative transforms #TODO: This part is repetitive, can be optimized
    inv0 = compute_inverse_transformation(ext0)
    inv1 = compute_inverse_transformation(ext1)
    inv2 = compute_inverse_transformation(ext2)

    ext_1_to_0 = inv0 @ ext1
    ext_2_to_0 = inv0 @ ext2

    ext_0_to_1 = inv1 @ ext0
    ext_2_to_1 = inv1 @ ext2

    ext_0_to_2 = inv2 @ ext0
    ext_1_to_2 = inv2 @ ext1

    # Compute epipolar lines
    epi_lines_1_to_0 = compute_epipolar_lines(ext_1_to_0, K0, K1, pts_cam1) if pts_cam1 is not None else None
    epi_lines_2_to_0 = compute_epipolar_lines(ext_2_to_0, K0, K2, pts_cam2) if pts_cam2 is not None else None

    epi_lines_0_to_1 = compute_epipolar_lines(ext_0_to_1, K1, K0, pts_cam0) if pts_cam0 is not None else None
    epi_lines_2_to_1 = compute_epipolar_lines(ext_2_to_1, K1, K2, pts_cam2) if pts_cam2 is not None else None

    epi_lines_0_to_2 = compute_epipolar_lines(ext_0_to_2, K2, K0, pts_cam0) if pts_cam0 is not None else None
    epi_lines_1_to_2 = compute_epipolar_lines(ext_1_to_2, K2, K1, pts_cam1) if pts_cam1 is not None else None

    # Put the lines in the form ax + by + c = 0
    epi_masks_for_views = {}
    for view in [0, 1, 2]:
        _first_epi_lines = eval(VIEW_COMBINATIONS[view][0])
        _second_epi_lines = eval(VIEW_COMBINATIONS[view][1])
        _check_first = _first_epi_lines is not None and _first_epi_lines.shape[0] > 0
        _check_second = _second_epi_lines is not None and _second_epi_lines.shape[0] > 0
        # Discriminate between the cases If both are None, skip this view
        if _check_first == False and _check_second == False:
            epi_masks_for_views[view] = torch.tensor([[]]), torch.tensor([[]])
            continue

        # If both lines exist, compute the intersection
        elif _check_first == True and _check_second == True:
            intersect_point1 = single_epiline(_size, _first_epi_lines, step)
            intersect_point2 = single_epiline(_size, _second_epi_lines, step)
            epi_masks_for_views[view] = intersect_point1, intersect_point2

        # If only one line exists, sample along the epipolar line
        else:
            intersect_point = single_epiline(_size, 
                           _first_epi_lines if _first_epi_lines is not None else _second_epi_lines
                        )
            epi_masks_for_views[view] = intersect_point, torch.tensor([[]])
    return epi_masks_for_views

@torch.no_grad()
def epipolar_main(camera_int_ext: List[torch.Tensor],
                  predictions0: Dict[str, torch.Tensor],
                  predictions1: Dict[str, torch.Tensor],
                  predictions2: Dict[str, torch.Tensor],
                  num_queries: int = 23,
                  epipolar_weight: float = 0.5,
                  n_points: Optional[int] = 4096,
                  step: Optional[int] = 2,
                  ):
    """Wrap the epipolar functionality to be used in the training class."""

    # Get the logits from each camera's predictions.
    pred_class0 = torch.Tensor(predictions0['pred_logits']).softmax(dim=-1).argmax(dim=-1)
    pred_class1 = torch.Tensor(predictions1['pred_logits']).softmax(dim=-1).argmax(dim=-1)
    pred_class2 = torch.Tensor(predictions2['pred_logits']).softmax(dim=-1).argmax(dim=-1)

    # Initialize the masks
    device = pred_class0.device
    o, h, w = predictions0['pred_masks_high_res'].shape[-3:]
    unique_projected_labels = set([obj['label'] for obj in LABEL_PROJECTION_MAP['default'].values()])
    # unique_projected_labels = [i for i in range(num_queries)]  # TODO: get it from the config num_classes
    masks = torch.zeros([len(unique_projected_labels), 3, h, w], dtype=torch.float32, device=device)

    # Object position dictionary
    object_pos_label = {view_idx:{pos_idx:None for pos_idx in range(o)} for view_idx in range(3)}

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
        if obj_idx not in OBJECTS_EPIPOLAR:
            masks[obj_idx] = torch.zeros([3, h, w], dtype=torch.float32)
            continue

        # Get the masks for the first three cameras
        mask_cam0 = predictions0['pred_masks_high_res'][pos_obj0]  # Slice the correct objects from the indeces.
        mask_cam1 = predictions1['pred_masks_high_res'][pos_obj1]  # Sum is only needed if multiple
        mask_cam2 = predictions2['pred_masks_high_res'][pos_obj2]  # instances of the same object.

        # Preprocess the masks to get points
        pts_cam0, pts_cam1, pts_cam2 = epipolar_mask_preprocess(mask_cam0, mask_cam1, mask_cam2, n_points=n_points)

        # Technically we dont need to check if the points are None, since the epipolar mask function will handle it
        # But we do it here to avoid unnecessary computation if the points are not available
        if pts_cam0 is None or pts_cam1 is None or pts_cam2 is None:
            masks[obj_idx] = torch.zeros([3, h, w], dtype=torch.float32)
            continue

        # Run the epipolar mask function
        epi_mask_for_views = epipolar_mask(camera_int_ext, pts_cam0, pts_cam1, pts_cam2, mask_cam0.shape[-2:], step=step)

        # Epipolar mask post processing
        mask = epipolar_mask_post_process(epi_mask_for_views, mask_cam0.shape[-2:], epipolar_weight, device)
        masks[obj_idx] = mask
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

    # Convert camera intrinsics and extrinsics to tensors
    camera_int_ext = [(torch.tensor(cam[0]), torch.tensor(cam[1])) for cam in camera_int_ext_json.values()]

    # Reshape the logits
    pred_class = torch.Tensor(predictions['pred_logits']).softmax(dim=-1).argmax(dim=-1)

    OBJECTS_EPIPOLAR = [10]  # NOTE: Test for multi instances

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
        mask_cam0 = torch.zeros_like(mask_cam0)
        
        start_time = time.time()

        # Preprocess the masks to get points
        pts_cam0, pts_cam1, pts_cam2 = epipolar_mask_preprocess(mask_cam0, mask_cam1, mask_cam2, num_points_idx=-1)

        # Run the epipolar mask function
        epi_mask_for_views = epipolar_mask(camera_int_ext, pts_cam0, pts_cam1, pts_cam2, mask_cam0.shape[-2:])

        # Epipolar mask post processing
        masks = epipolar_mask_post_process(epi_mask_for_views, mask_cam0.shape[-2:])
        
        print(f"Time taken for epipolar mask computation for object {obj_idx}: {time.time() - start_time:.4f} seconds")

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