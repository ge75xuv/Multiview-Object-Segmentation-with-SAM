from scipy.spatial.transform import Rotation
import torch

def load_camera_data(camera_data, downscale=1):
    h, w = camera_data['value0']['color_parameters']['height'], camera_data['value0']['color_parameters']['width']
    h_, w_ = h // downscale, w // downscale
    padding = (w_ - h_) // 2
    # padding = 0
    intrinsics_json = camera_data['value0']['color_parameters']['intrinsics_matrix']
    K = torch.tensor([[intrinsics_json['m00'] / downscale, intrinsics_json['m10'], intrinsics_json['m20'] / downscale],
                    [intrinsics_json['m01'], intrinsics_json['m11'] / downscale, intrinsics_json['m21'] / downscale + padding],
                    [0, 0, 1]])
    # Extrinsics Depth
    extrinsics_json = camera_data['value0']['camera_pose']
    trans = extrinsics_json['translation']
    rot = extrinsics_json['rotation']
    extrinsics = torch.zeros((4, 4), dtype=torch.float32)
    R = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
    extrinsics[:3, :3] = torch.tensor(R)
    t = [trans['m00'], trans['m10'], trans['m20'], 1]
    extrinsics[:, 3] = torch.tensor(t)

    # Extrinsics Color
    color2depth_json = camera_data['value0']['color2depth_transform']
    trans = color2depth_json['translation']
    rot = color2depth_json['rotation']
    color2depth_transform = torch.zeros((4, 4), dtype=torch.float32)
    rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
    color2depth_transform[:3, :3] = torch.tensor(rot_matrix)
    color2depth_transform[:, 3] = torch.tensor([trans['m00'], trans['m10'], trans['m20'], 1])
    depth_extrinsics = extrinsics.clone()
    extrinsics = extrinsics @ color2depth_transform  # Extrinsics were given for the depth camera, convert them to color camera
    extrinsics = extrinsics @ torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

    # Distortion
    rad_dist = camera_data['value0']['color_parameters']['radial_distortion']
    tan_dist = camera_data['value0']['color_parameters']['tangential_distortion']
    dist = [rad_dist.pop('m00'), rad_dist.pop('m10'), tan_dist.pop('m00'), tan_dist.pop('m10')]
    dist += [rd for rd in rad_dist.values()]
    dist = torch.tensor(dist, dtype=torch.float32)

    return K, extrinsics, padding, dist, depth_extrinsics