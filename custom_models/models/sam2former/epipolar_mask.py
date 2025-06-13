from typing import List, Tuple, Optional, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# OBJECTS_EPIPOLAR = [10, 12]
OBJECTS_EPIPOLAR = [10]

VIEW_COMBINATIONS = {
    0: ["epi_lines_1_to_0", "epi_lines_2_to_0"],
    1: ["epi_lines_0_to_1", "epi_lines_2_to_1"],
    2: ["epi_lines_0_to_2", "epi_lines_1_to_2"],
}

def draw_epilines(img2, lines, padding=0):
    num_points = lines.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, num_points))[:, :] * 255  # RGB 0-255
    colors = colors.astype(np.uint8)
    h, w = img2.shape[:2]
    for r, color in zip(lines, colors):
        color = color.tolist() if isinstance(color, np.ndarray) else color
        # a, b, c = r
        # x0, x1 = 0, w
        # y0 = int((-c - a * x0) / b)
        # y1 = int((-c - a * x1) / b)
        x0,y0 = map(int, [0, -(r[2])/r[1] - padding])
        x1,y1 = map(int, [w, -(r[2] + r[0]*w)/r[1] - padding])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 2)
    return img2

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
    F = inv_K0.mT @ E @ inv_K1
    pts1_h = torch.hstack((pts1, torch.ones((pts1.shape[0], 1))))  # homogeneous
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

def single_epiline(_size: Tuple[int, int], lines: torch.Tensor):
    step = 10  # Step size for sampling points along the epipolar line
    h, w = _size
    eval_func = lambda x: (-lines[:, 2] - lines[:, 0] * x) / lines[:, 1]
    pts = torch.tensor([(x, y) for x in range(0, w, step) for y in eval_func(x) if y >= 0 and y <= h])
    return pts

@torch.no_grad()
def epipolar_mask_preprocess(
    mask_cam0: torch.Tensor,
    mask_cam1: torch.Tensor,
    mask_cam2: torch.Tensor,
    num_points_idx: Optional[int] = -1,
    ):
    # Another reason for us to use temporality. It is not possible to make a differentiable masking here
    # The batch operation for different objects cannot be done here since the size of the masks are going to be varying size
    y, x = torch.where(mask_cam0.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam0 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)[0:num_points_idx] if len(x) > 0 else None

    y, x = torch.where(mask_cam1.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam1 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)[0:num_points_idx] if len(x) > 0 else None

    y, x = torch.where(mask_cam2.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam2 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)[0:num_points_idx] if len(x) > 0 else None
    return pts_cam0, pts_cam1, pts_cam2

def epipolar_mask_post_process(
    epi_masks_for_views: Dict[int, Tuple[torch.Tensor, float]],
    _size: Tuple[int, int],
):
    mask = torch.zeros([3, _size[0], _size[1]], dtype=torch.float32)
    # epi_masks keys for views and values are the intersection points and the case weights
    # We use points as indices to fill the mask, so seperate them as x y convert them to tuples
    pos_x, pos_y = tuple(epi_masks_for_views[0][0][:,0].tolist()), tuple(epi_masks_for_views[0][0][:,1].tolist())
    mask[0, pos_y, pos_x] = epi_masks_for_views[0][1]
    pos_x, pos_y = tuple(epi_masks_for_views[1][0][:,0].tolist()), tuple(epi_masks_for_views[1][0][:,1].tolist())
    mask[1, pos_y, pos_x] = epi_masks_for_views[1][1]
    pos_x, pos_y = tuple(epi_masks_for_views[2][0][:,0].tolist()), tuple(epi_masks_for_views[2][0][:,1].tolist())
    mask[2, pos_y, pos_x] = epi_masks_for_views[2][1]
    return mask

    # This is very inefficient
    # masks = np.zeros([3, _size[0], _size[1]], dtype=np.float32)
    # for k, v in epi_mask_for_views.items():
    #     w = v[1]
    #     pts = v[0]
    #     for pt in pts:
    #         pt_int = tuple(np.round(pt.numpy()).astype(int))
    #         masks[k] = cv2.circle(masks[k], pt_int, 2, [255 * w, 0, 0], 1)
    # return masks

@torch.no_grad()
def epipolar_mask(
    camera_int_ext: List[torch.Tensor],
    pts_cam0: torch.Tensor,
    pts_cam1: torch.Tensor,
    pts_cam2: torch.Tensor,
    _size: Tuple[int, int],
    _case_weights: Optional[Tuple[float, float]] = (0.8, 0.2)
    ):

    # Unpack camera intrinsics and extrinsics
    K0, ext0 = camera_int_ext[0]
    K1, ext1 = camera_int_ext[1]
    K2, ext2 = camera_int_ext[2]

    # Compute relative transforms
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

    # NOTE: DEBUG
    # test_img1 = draw_epilines(np.zeros((_size[0], _size[1], 1)), epi_lines_1_to_0, padding=0)
    # test_img2 = draw_epilines(np.zeros((_size[0], _size[1], 1)), epi_lines_2_to_0, padding=0)

    # Put the lines in the form ax + by + c = 0
    skip = 1
    epi_masks_for_views = {}
    for view in [0, 1, 2]:
        _first_epi_lines = eval(VIEW_COMBINATIONS[view][0])
        _second_epi_lines = eval(VIEW_COMBINATIONS[view][1])
        # Discriminate between the cases If both are None, skip this view
        if _first_epi_lines is None and _second_epi_lines is None:
            continue

        # If both lines exist, compute the intersection
        elif not _first_epi_lines is None and not _second_epi_lines is None:
            a, b, c = _first_epi_lines.T[:, ::skip]
            a, b, c = a[:, None], b[:, None], c[:, None]
            k, l, m = _second_epi_lines.T[:, ::skip]
            k, l, m = k[:, None], l[:, None], m[:, None]

            # Calculate the intersection point, every line to all lines
            for i in range(len(a)):
                if i == 0:
                    intersect_point = intersection(a[0:1], b[0:1], c[0:1], k, l, m)
                else:
                    curr_point = intersection(a[i:i+1], b[i:i+1], c[i:i+1], k, l, m)
                    intersect_point = torch.concatenate((intersect_point, curr_point), axis=0)
            # Remove duplicates
            intersect_point = torch.unique(intersect_point, dim=0)
            # Remove points outside the image bounds
            intersect_point = intersect_point[(intersect_point[:, 0] >= 0) & (intersect_point[:, 0] < _size[1]) &
                                              (intersect_point[:, 1] >= 0) & (intersect_point[:, 1] < _size[0])]
            epi_masks_for_views[view] = intersect_point, _case_weights[0]

        # If only one line exists, sample along the epipolar line
        else:
            intersect_point = single_epiline(_size, 
                           _first_epi_lines if _first_epi_lines is not None else _second_epi_lines
                        )
            epi_masks_for_views[view] = intersect_point, _case_weights[1]
    return epi_masks_for_views

@torch.no_grad()
def epipolar_main(camera_int_ext: List[torch.Tensor],
                  predictions0: Dict[str, torch.Tensor],
                  predictions1: Dict[str, torch.Tensor],
                  predictions2: Dict[str, torch.Tensor],
                  ):
    """Wrap the epipolar functionality to be used in the training class."""
    # Change camera_int_ext to a list of tensors for each camera
    #TODO save initially as tensor not numpy
    [[torch.tensor(int_ext[0]), torch.tensor(int_ext[1])] for int_ext in zip(*camera_int_ext)]

    # Get the logits from each camera's predictions.
    pred_class0 = torch.Tensor(predictions0['pred_logits']).softmax(dim=-1).argmax(dim=-1)
    pred_class1 = torch.Tensor(predictions1['pred_logits']).softmax(dim=-1).argmax(dim=-1)
    pred_class2 = torch.Tensor(predictions2['pred_logits']).softmax(dim=-1).argmax(dim=-1)

    # Iterate over the objects of interest and get the masks.
    for obj_idx in OBJECTS_EPIPOLAR:
        # Find the positions of the object in each camera's predictions
        pos_obj0 = torch.where(pred_class0 == obj_idx)  # pos_obj0 is a tuple of (batch_idx, class_idx)
        pos_obj1 = torch.where(pred_class1 == obj_idx)  # pos_obj1 is a tuple of (batch_idx, class_idx)
        pos_obj2 = torch.where(pred_class2 == obj_idx)  # pos_obj2 is a tuple of (batch_idx, class_idx)

        # Get the masks for the first three cameras
        mask_cam0 = predictions0['pred_masks_high_res'][pos_obj0].squeeze(0)  # Slice the correct objects
        mask_cam1 = predictions1['pred_masks_high_res'][pos_obj1].squeeze(0)  # from the indeces
        mask_cam2 = predictions2['pred_masks_high_res'][pos_obj2].squeeze(0)

        # Preprocess the masks to get points
        pts_cam0, pts_cam1, pts_cam2 = epipolar_mask_preprocess(mask_cam0, mask_cam1, mask_cam2, num_points_idx=-1)

        # Run the epipolar mask function
        epi_mask_for_views = epipolar_mask(camera_int_ext, pts_cam0, pts_cam1, pts_cam2, mask_cam0.shape[-2:])

        # Epipolar mask post processing
        masks = epipolar_mask_post_process(epi_mask_for_views, mask_cam0.shape[-2:])
    return masks


if __name__ == '__main__':
    import json

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
    for obj_idx in OBJECTS_EPIPOLAR:
        index_10 = torch.where(pred_class == obj_idx)
        cam0_ids = index_10[1][2]  # It is a tuple of batch index and class index, 
        cam1_ids = index_10[1][3]  # so we need to access the second element
        cam2_ids = index_10[1][4]

        # Get the masks for the first three cameras
        # NOTE by visial observation, we know that the first camera is the 1st in the batch
        mask_cam0 = torch.Tensor(predictions['pred_masks_high_res'][0])[cam0_ids, :, :]  # Slice the correct objects
        mask_cam1 = torch.Tensor(predictions['pred_masks_high_res'][1])[cam1_ids, :, :]  # from the indeces
        mask_cam2 = torch.Tensor(predictions['pred_masks_high_res'][2])[cam2_ids, :, :]

        # NOTE: test lacking visibility
        # mask_cam0 = torch.zeros_like(mask_cam0)

        # Preprocess the masks to get points
        pts_cam0, pts_cam1, pts_cam2 = epipolar_mask_preprocess(mask_cam0, mask_cam1, mask_cam2, num_points_idx=-1)

        # Run the epipolar mask function
        epi_mask_for_views, test1, test2 = epipolar_mask(camera_int_ext, pts_cam0, pts_cam1, pts_cam2, mask_cam0.shape[-2:])

        # Epipolar mask post processing
        masks = epipolar_mask_post_process(epi_mask_for_views, mask_cam0.shape[-2:])

        # Print the results
        print("Intersection Points:", epi_mask_for_views[0][0].shape, epi_mask_for_views[0][1])
        print("Intersection Points:", epi_mask_for_views[1][0].shape, epi_mask_for_views[1][1])
        print("Intersection Points:", epi_mask_for_views[2][0].shape, epi_mask_for_views[2][1])

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

    # plt.imshow(masks[0].cpu().numpy(), cmap='gray')
    # plt.title('Epipolar Mask for View 0')
    # plt.show()