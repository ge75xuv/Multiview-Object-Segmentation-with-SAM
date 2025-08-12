import torch

def depth_preprocess(pts_cam:torch.Tensor, gt_depth_image_view:torch.Tensor, padding:int, depth_trunc:int, img_size:tuple, device:torch.device):
    # Get individual x, y, coordinates
    x_, y_ = pts_cam.T  # Put it first into 2xN
    # Create depth masks for each camera
    H, W = img_size
    mask_depth = torch.zeros([H, W], dtype=float, device=device)
    mask_depth[padding:-padding] = gt_depth_image_view.squeeze(0)
    mask_depth = mask_depth[y_, x_] / 1000.0  # Convert mm to m
    # Truncate
    mask_depth[mask_depth > depth_trunc] = depth_trunc
    return mask_depth.type(torch.float32)  # Convert to float32 for further processing

def backproject_to_world(pts_cam, K, T, mask_depth):
    # Get the mask points
    x_, y_ = pts_cam.T.type(torch.float)  # Put it first into 2xN
    # Homogenous coordinates
    h_coords = torch.stack([x_, y_, torch.ones_like(x_)], axis=1)  # Homogeneous coordinates Nx3
    # Image coordinates
    im_coords = torch.linalg.inv(K) @ h_coords.T  # Inverse camera matrix multiplication 3xN
    # 3D coordinates
    im_coords_depth_scaled = im_coords * mask_depth
    # 3D coordinates in world frame   
    obj_pc = T @ torch.concatenate([im_coords_depth_scaled, torch.ones_like(x_)[None,:]], axis=0)  # Transforming to world coordinates 4xN
    return obj_pc

def reprojection_to_image_frame(K, T, obj_pc, H, W):
    # Projecting back to image coordinates
    projected_pts = K @ (torch.linalg.inv(T) @ obj_pc)[:3, :]  # 3xN
    normalized_pts = projected_pts / projected_pts[2, :]
    # Pixel coordinates
    pixel_pts = normalized_pts[:2, :]  # 2xN
    pixel_pts = pixel_pts.round().astype(int)  # Round to nearest integer for pixel coordinates
    # Refinement
    refinement = (pixel_pts[1] > (H-1)) | (pixel_pts[1] < 0) | (pixel_pts[0] > (W-1)) | (pixel_pts[0] < 0)
    pixel_pts[:, refinement] = 0  # Set invalid pixel coordinates to 0
    return pixel_pts

def point_cloud_main(cam_int_ext, pts_cam0, pts_cam1, pts_cam2, gt_depth_image, img_size):
    # Store Intrinsics and extrinsics
    K0, K1, K2 = cam_int_ext[0][0], cam_int_ext[1][0], cam_int_ext[2][0]
    T0, T1, T2 = cam_int_ext[0][1], cam_int_ext[1][1], cam_int_ext[2][1]  # Transformations are from camera to world coordinates

    # Create empty point clouds
    device = cam_int_ext[0][0].device
    obj_pc0 = torch.empty(4,0).to(device)
    obj_pc1 = torch.empty(4,0).to(device)
    obj_pc2 = torch.empty(4,0).to(device)

    # Compute the depth values of the points in the mask
    H_depth, W_depth = gt_depth_image[0].shape[-2:]  # Assuming all depth images have the same shape
    padding = (W_depth-H_depth) // 2
    depth_trunc = 10  # 10 meters max

    # Calculate the backprojection to 3D world frame
    if pts_cam0 is not None:
        mask0_depth = depth_preprocess(pts_cam0, gt_depth_image[0], padding, depth_trunc, img_size, device)
        obj_pc0 = backproject_to_world(pts_cam0, K0, T0, mask0_depth)
    if pts_cam1 is not None:
        mask1_depth = depth_preprocess(pts_cam1, gt_depth_image[1], padding, depth_trunc, img_size, device)
        obj_pc1 = backproject_to_world(pts_cam1, K1, T1, mask1_depth)
    if pts_cam2 is not None:
        mask2_depth = depth_preprocess(pts_cam2, gt_depth_image[2], padding, depth_trunc, img_size, device)
        obj_pc2 = backproject_to_world(pts_cam2, K2, T2, mask2_depth)

    # Overall point cloud
    obj_pc12 = torch.concatenate([obj_pc1, obj_pc2], axis=1)
    obj_pc02 = torch.concatenate([obj_pc0, obj_pc2], axis=1)
    obj_pc01 = torch.concatenate([obj_pc0, obj_pc1], axis=1)

    # Reprojection to the image frame
    H, W = img_size
    pixel_pts0 = reprojection_to_image_frame(K0, T0, obj_pc12, H, W) if obj_pc12.shape[1] > 0 else None
    pixel_pts1 = reprojection_to_image_frame(K1, T1, obj_pc02, H, W) if obj_pc02.shape[1] > 0 else None
    pixel_pts2 = reprojection_to_image_frame(K2, T2, obj_pc01, H, W) if obj_pc01.shape[1] > 0 else None

    pc_masks_for_views = {
        0: (pixel_pts0, torch.tensor([[]])),
        1: (pixel_pts1, torch.tensor([[]])),
        2: (pixel_pts2, torch.tensor([[]])),
    }

    return pc_masks_for_views