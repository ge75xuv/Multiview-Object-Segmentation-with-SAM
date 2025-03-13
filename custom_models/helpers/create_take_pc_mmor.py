import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.configurations import MMOR_TAKE_NAMES, MMOR_DATA_ROOT_PATH, MMOR_TAKE_NAME_TO_FOLDER
from scipy import stats

from visualize_panoptic_seg_for_take import TRACK_TO_METAINFO


def load_cam_infos(root_path: Path, cam_count=5):
    cam_infos = {}
    for c_idx in range(1, cam_count + 1):
        cam_json_path = root_path / f'camera0{c_idx}.json'
        with cam_json_path.open() as f:
            cam_info = json.load(f)['value0']
            intrinsics_json = cam_info['color_parameters']['intrinsics_matrix']
            intrinsics = np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                                     [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                                     [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

            extrinsics_json = cam_info['camera_pose']
            trans = extrinsics_json['translation']
            rot = extrinsics_json['rotation']
            extrinsics = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            extrinsics[:3, :3] = rot_matrix
            extrinsics[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]

            color2depth_json = cam_info['color2depth_transform']
            trans = color2depth_json['translation']
            rot = color2depth_json['rotation']
            color2depth_transform = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            color2depth_transform[:3, :3] = rot_matrix
            color2depth_transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
            depth_extrinsics = np.copy(extrinsics)
            extrinsics = np.matmul(extrinsics, color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

            fov_x = cam_info['color_parameters']['fov_x']
            fov_y = cam_info['color_parameters']['fov_y']
            c_x = cam_info['color_parameters']['c_x']
            c_y = cam_info['color_parameters']['c_y']
            width = cam_info['color_parameters']['width']
            height = cam_info['color_parameters']['height']

            params = cam_info['color_parameters']['radial_distortion']
            radial_params = params['m00'], params['m10'], params['m20'], params['m30'], params['m40'], params['m50']
            params = cam_info['color_parameters']['tangential_distortion']
            tangential_params = params['m00'], params['m10']

            cam_infos[f'camera0{c_idx}'] = {'intrinsics': intrinsics, 'extrinsics': extrinsics, 'fov_x': fov_x, 'fov_y': fov_y,
                                            'c_x': c_x, 'c_y': c_y, 'width': width, 'height': height, 'radial_params': radial_params,
                                            'tangential_params': tangential_params, 'depth_extrinsics': depth_extrinsics}

    return cam_infos


def _helper_export_point_cloud(timestamp_info, color_image_path, depth_image_path, export_pcd_path, cam_infos, N_CAMERAS, POINT_COUNT, MIN_BOUNDS, MAX_BOUNDS, IMAGE_RES):
    idx, info = timestamp_info
    idx_str = str(idx).zfill(6)
    azure_idx = info['azure']
    azure_idx_str = str(azure_idx).zfill(6)
    if (export_pcd_path / f'{idx_str}.pcd').exists() and (export_pcd_path / f'{idx_str}_seg.pcd').exists():
        return
    pcds = []
    pcds_seg = []
    for c_idx in range(1, N_CAMERAS + 1):
        cam_info = cam_infos[f'camera0{c_idx}']
        rgb_path = color_image_path / f'camera0{c_idx}_colorimage-{azure_idx_str}.jpg'
        segmentation_image_export_dir = color_image_path.parent / f'segmentation_export_{c_idx}'
        s_path = segmentation_image_export_dir / f'{rgb_path.stem}.png'
        if not s_path.exists():
            s_path = segmentation_image_export_dir / f'{rgb_path.stem}_interpolated.png'
        if not rgb_path.exists():
            continue
        d_path = depth_image_path / f'camera0{c_idx}_depthimage-{azure_idx_str}.tiff'

        intrinsics = o3d.camera.PinholeCameraIntrinsic(cam_info['width'], cam_info['height'], cam_info['fov_x'], cam_info['fov_y'], cam_info['c_x'],
                                                       cam_info['c_y'])
        extrinsics = cam_info['extrinsics']

        rgb = Image.open(rgb_path)
        rgb = o3d.geometry.Image(np.asarray(rgb))
        d_tmp = Image.open(d_path)
        # Optionally scale it up to the original size
        d_tmp = d_tmp.resize((d_tmp.size[0] * 2, d_tmp.size[1] * 2))
        d = o3d.geometry.Image(np.asarray(d_tmp))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, convert_rgb_to_intensity=False, depth_scale=1, depth_trunc=10000000)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 1000)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform(extrinsics)

        pcds.append(pcd)

        if s_path.exists():
            s = cv2.imread(s_path.as_posix())
            s = cv2.resize(s, IMAGE_RES, interpolation=cv2.INTER_NEAREST)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            depth_array = np.asarray(d_tmp)
            # Median depth based filtering to avoid outliers
            unique_labels = np.unique(s)
            for label in unique_labels:
                if label == 0:
                    continue  # Skip background or unlabeled pixels
                # Get mask of pixels belonging to the current class
                class_mask = (s == label)
                # Get corresponding depth values
                class_depths = depth_array[class_mask]
                valid_depths = class_depths[class_depths > 0]
                # Check if there are enough pixels to compute statistics
                if len(valid_depths) < 10:
                    continue
                # Compute median depth
                median_depth = np.median(valid_depths)
                std_depth = np.std(valid_depths)
                threshold = 2 * std_depth
                abs_diff = np.abs(class_depths - median_depth)
                outlier_mask = abs_diff > threshold
                s[class_mask] = np.where(outlier_mask, 0, s[class_mask])

            # Convert s to 3 channels
            s = np.stack([s, s, s], axis=-1)
            sd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.asarray(s)), d, convert_rgb_to_intensity=False, depth_scale=1, depth_trunc=10000000)
            pcd_seg = o3d.geometry.PointCloud.create_from_rgbd_image(sd, intrinsics)
            pcd_seg.points = o3d.utility.Vector3dVector(np.asarray(pcd_seg.points) / 1000)
            pcd_seg.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_seg.transform(extrinsics)
            pcds_seg.append(pcd_seg)

    if len(pcds) == 0:
        return
    merged_pcd = pcds[0]
    for i in range(1, N_CAMERAS):
        merged_pcd += pcds[i]

    merged_pcd_seg = pcds_seg[0]
    for i in range(1, len(pcds_seg)):
        merged_pcd_seg += pcds_seg[i]

    merged_pcd = merged_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(MIN_BOUNDS, MAX_BOUNDS))
    # empty merged_pcd_seg_ready_to_be filled.
    merged_pcd_seg_ready = o3d.geometry.PointCloud()
    merged_pcd_seg_ready.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd.points))
    # color should be all 0,0,0 for now
    merged_pcd_seg_ready.colors = o3d.utility.Vector3dVector(np.zeros_like(np.asarray(merged_pcd.points)))

    # downsample
    choices = np.random.choice(len(merged_pcd.points), POINT_COUNT, replace=len(merged_pcd.points) < POINT_COUNT)
    merged_pcd.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd.points)[choices])
    merged_pcd.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd.colors)[choices])
    merged_pcd_seg_ready.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd_seg_ready.points)[choices])
    merged_pcd_seg_ready.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd_seg_ready.colors)[choices])

    choices = np.random.choice(len(merged_pcd_seg.points), 10000, replace=len(merged_pcd_seg.points) < 10000)
    merged_pcd_seg.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd_seg.points)[choices])
    merged_pcd_seg.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd_seg.colors)[choices])

    # Find the N closest points in merged_pcd_seg for every point in merged_pcd_seg_ready vectorized implementation using kdtree
    N = 10  # the number of nearest points you want to find
    # build the k-d tree
    tree = cKDTree(np.asarray(merged_pcd_seg.points))
    # find the N nearest neighbors for each point
    distances, indices = tree.query(np.asarray(merged_pcd_seg_ready.points), k=N)

    # distances[i, j] is the distance from the i-th point to its j-th nearest neighbor
    # indices[i, j] is the index of the j-th nearest neighbor of the i-th point
    indices_of_closest_points = indices
    colors_of_closest_points = np.round(np.asarray(merged_pcd_seg.colors)[:, 0] * 255).astype(np.uint8)[indices_of_closest_points]
    # Take the most common color among the N closest points
    colors = stats.mode(colors_of_closest_points, axis=1)[0].flatten()
    # Make 3 channels
    colors = np.stack([colors, colors, colors], axis=-1)

    new_colors = colors.copy().astype(np.float32)
    for track, metainfo in TRACK_TO_METAINFO.items():
        corresponding_points = np.where(colors == metainfo['label'])[0]
        new_colors[corresponding_points, 0] = metainfo['color'][0] / 255
        new_colors[corresponding_points, 1] = metainfo['color'][1] / 255
        new_colors[corresponding_points, 2] = metainfo['color'][2] / 255

    # convert RGB to BGR
    merged_pcd_seg_ready.colors = o3d.utility.Vector3dVector(new_colors)

    merged_pcd.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd.points) * 500)
    merged_pcd_seg_ready.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd_seg_ready.points) * 500)

    # o3d.io.write_point_cloud(str(export_pcd_path / f'{idx_str}.pcd'), merged_pcd, compressed=True)
    o3d.io.write_point_cloud(str(export_pcd_path / f'{idx_str}_seg.pcd'), merged_pcd_seg_ready, compressed=True)


def main():
    for take in tqdm(MMOR_TAKE_NAMES):
        print(f'Processing take {take}')
        JSON_PATH = MMOR_DATA_ROOT_PATH / 'take_jsons' / f'{take}.json'
        EXPORT_PC_PATH = MMOR_DATA_ROOT_PATH / 'take_point_clouds_seg_lowres' / take
        EXPORT_PC_PATH.mkdir(exist_ok=True, parents=True)

        with JSON_PATH.open() as f:
            take_json = json.load(f)
            timestamps = take_json['timestamps']
            timestamps = {int(k): v for k, v in timestamps.items()}

        folder_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take, take)
        assert folder_path.exists(), f'Could not find folder for take {take}'

        print('Exporting Point Clouds')
        color_image_path = folder_path / 'colorimage'
        depth_image_path = folder_path / 'depthimage'
        N_CAMERAS = 5
        try:
            cam_infos = load_cam_infos(folder_path, N_CAMERAS)
        except Exception as e:
            print(f'Could not load cam infos for take {take}, {e}')
            continue
        POINT_COUNT = 5_000
        # MIN_BOUNDS = [-2.7, -0.06, -2.7]  # with floor
        MIN_BOUNDS = [-2.7, 0.10, -2.7]  # without floor
        MAX_BOUNDS = [2.21, 1.8, 3]
        IMAGE_RES = (2048, 1536)

        process_map(partial(_helper_export_point_cloud, color_image_path=color_image_path, depth_image_path=depth_image_path,
                            export_pcd_path=EXPORT_PC_PATH, cam_infos=cam_infos, N_CAMERAS=N_CAMERAS, POINT_COUNT=POINT_COUNT, MIN_BOUNDS=MIN_BOUNDS,
                            MAX_BOUNDS=MAX_BOUNDS, IMAGE_RES=IMAGE_RES),
                    sorted(timestamps.items()), max_workers=32, chunksize=10)


if __name__ == '__main__':
    main()
