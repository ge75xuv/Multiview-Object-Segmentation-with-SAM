import torch

def depth_preprocess(gt_depth_image, pts_cam0, pts_cam1, pts_cam2, H, W):
    '''Preprocess depth images for point cloud generation.'''
    # Get individual x, y, coordinates
    x0_, y0_ = pts_cam0
    x1_, y1_ = pts_cam1
    x2_, y2_ = pts_cam2

    # Create depth masks for each camera
    mask0_depth = torch.zeros([H, W], dtype=float)
    mask1_depth = torch.zeros([H, W], dtype=float)
    mask2_depth = torch.zeros([H, W], dtype=float)

    padding = (W-H) // 2

    mask0_depth[padding:-padding] = gt_depth_image[0]
    mask1_depth[padding:-padding] = gt_depth_image[1]
    mask2_depth[padding:-padding] = gt_depth_image[2]

    mask0_depth = mask0_depth.squeeze()[y0_, x0_] / 1000.0  # Convert mm to m
    mask1_depth = mask1_depth.squeeze()[y1_, x1_] / 1000.0  # Convert mm to m
    mask2_depth = mask2_depth.squeeze()[y2_, x2_] / 1000.0  # Convert mm to m

    # Truncate
    depth_trunc = 10
    mask0_depth[mask0_depth > depth_trunc] = depth_trunc
    mask1_depth[mask1_depth > depth_trunc] = depth_trunc
    mask2_depth[mask2_depth > depth_trunc] = depth_trunc
    return mask0_depth, mask1_depth, mask2_depth

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

def point_cloud_main(cam_int, cam_ext, pts_cam0, pts_cam1, pts_cam2, mask0_depth, mask1_depth, mask2_depth, H, W):
    # Store Intrinsics and extrinsics
    K0, K1, K2 = cam_int[0], cam_int[1], cam_int[2]
    T0, T1, T2 = cam_ext[0], cam_ext[1], cam_ext[2]  # Transformations are from camera to world coordinates

    # Get the mask points
    x0_, y0_ = pts_cam0
    x1_, y1_ = pts_cam1
    x2_, y2_ = pts_cam2

    # Homogenous coordinates
    h_coords0 = torch.stack([x0_, y0_, torch.ones_like(x0_)], axis=1)  # Homogeneous coordinates Nx3
    h_coords1 = torch.stack([x1_, y1_, torch.ones_like(x1_)], axis=1)  # Homogeneous coordinates Nx3
    h_coords2 = torch.stack([x2_, y2_, torch.ones_like(x2_)], axis=1)  # Homogeneous coordinates Nx3

    # Image coordinates
    im_coords0 = torch.linalg.inv(K0) @ h_coords0.T  # Inverse camera matrix multiplication 3xN
    im_coords1 = torch.linalg.inv(K1) @ h_coords1.T  # Inverse camera matrix multiplication 3xN
    im_coords2 = torch.linalg.inv(K2) @ h_coords2.T  # Inverse camera matrix multiplication 3xN

    # 3D coordinates
    im_coords0_depth_scaled = im_coords0 * mask0_depth
    im_coords1_depth_scaled = im_coords1 * mask1_depth
    im_coords2_depth_scaled = im_coords2 * mask2_depth

    # 3D coordinates in world frame    
    obj_pc0 = T0 @ torch.concatenate([im_coords0_depth_scaled, torch.ones_like(x0_)[None,:]], axis=0)  # Transforming to world coordinates 4xN
    obj_pc1 = T1 @ torch.concatenate([im_coords1_depth_scaled, torch.ones_like(x1_)[None,:]], axis=0)  # Transforming to world coordinates 4xN
    obj_pc2 = T2 @ torch.concatenate([im_coords2_depth_scaled, torch.ones_like(x2_)[None,:]], axis=0)  # Transforming to world coordinates 4xN

    # Overall point cloud
    obj_pc12 = torch.concatenate([obj_pc1, obj_pc2], axis=1)
    obj_pc02 = torch.concatenate([obj_pc0, obj_pc2], axis=1)
    obj_pc01 = torch.concatenate([obj_pc0, obj_pc1], axis=1)

    # Reprojection to the image frame
    pixel_pts0 = reprojection_to_image_frame(K0, T0, obj_pc12, H, W)
    pixel_pts1 = reprojection_to_image_frame(K1, T1, obj_pc02, H, W)
    pixel_pts2 = reprojection_to_image_frame(K2, T2, obj_pc01, H, W)

    return pixel_pts0, pixel_pts1, pixel_pts2