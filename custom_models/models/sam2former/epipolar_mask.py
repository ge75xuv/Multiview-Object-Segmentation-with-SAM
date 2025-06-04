from typing import List, Tuple, Optional

import numpy as np
import torch

def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def compute_epipolar_lines(ext, K_line, K_pt, pts1):
    # pts1: Nx2 array
    R, t = ext[:3, :3], ext[:3, 3]
    E = skew(t) @ R
    inv_K1 = np.linalg.inv(K_line)
    inv_K0 = np.linalg.inv(K_pt)
    F = inv_K0.T @ E @ inv_K1
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  # homogeneous
    lines2 = (F @ pts1_h.T).T  # Each row is [a, b, c]
    return lines2

def compute_inverse_transformation(ext):
    inv = np.zeros((4, 4), dtype=np.float32)
    inv[:3, :3] = ext[:3,:3].T
    inv[:3, 3] = -ext[:3,:3].T @ ext[:3, 3]
    return inv

def intersection(a, b, c, k, l, m):
    x = (-c * l + m * b) / (a * l - k * b)
    y = (-c - a * x) / b
    return np.concatenate((x, y), axis=1)

@torch.no_grad()
def epipolar_mask_preprocess(
    mask_cam0: torch.Tensor,
    mask_cam1: torch.Tensor,
    mask_cam2: torch.Tensor,
    num_points_idx: Optional[int] = -1,
    ):
    # Another reason for us to use temporality. It is not possible to make a differentiable masking here
    y, x = torch.where(mask_cam0.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam0 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).numpy()[0:num_points_idx]

    y, x = torch.where(mask_cam1.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam1 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).numpy()[0:num_points_idx]

    y, x = torch.where(mask_cam2.sigmoid() > 0.5)  # Is torch where a problem for the gradients? Yes
    pts_cam2 = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).numpy()[0:num_points_idx]
    return pts_cam0, pts_cam1, pts_cam2

def epipolar_mask(
    camera_int_ext: List[torch.Tensor],
    pts_cam0: torch.Tensor,
    pts_cam1: torch.Tensor,
    pts_cam2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
    epi_lines_1_to_0 = compute_epipolar_lines(ext_1_to_0, K0, K1, pts_cam1)
    epi_lines_2_to_0 = compute_epipolar_lines(ext_2_to_0, K0, K2, pts_cam2)

    epi_lines_0_to_1 = compute_epipolar_lines(ext_0_to_1, K1, K0, pts_cam0)
    epi_lines_2_to_1 = compute_epipolar_lines(ext_2_to_1, K1, K2, pts_cam2)

    epi_lines_0_to_2 = compute_epipolar_lines(ext_0_to_2, K2, K0, pts_cam0)
    epi_lines_1_to_2 = compute_epipolar_lines(ext_1_to_2, K2, K1, pts_cam1)

    # Put the lines in the form ax + by + c = 0
    skip = 1
    a, b, c = epi_lines_1_to_0.T[:, ::skip]
    a, b, c = a[:, None], b[:, None], c[:, None]
    k, l, m = epi_lines_2_to_0.T[:, ::skip]
    k, l, m = k[:, None], l[:, None], m[:, None]

    # Calculate the intersection point
    for i in range(len(a)):
        if i == 0:
            intersect_point = intersection(a[0:1], b[0:1], c[0:1], k, l, m)
        else:
            curr_point = intersection(a[i:i+1], b[i:i+1], c[i:i+1], k, l, m)
            intersect_point = np.concatenate((intersect_point, curr_point), axis=0)
    