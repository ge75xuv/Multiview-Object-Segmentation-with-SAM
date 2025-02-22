import json
import os
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from training.utils.data_utils import VideoDatapoint
from tqdm import tqdm

from ..helpers.configurations import *

class MiniDataset(Dataset):
    
    def __init__(self, split_type:str, num_multimask_outputs:int, len_video:int):
        '''Initialzie the class open the data folders and store them.
        TODO Data Augmentatation
        TODO Video Batches:
                Let's store images in self.images as [[fr0, fr1, fr2, frK-1], [fr0+K, fr1+K, fr2+K, frK+k-1], ...]
                so a list of small videos. Thus during __getitem__ I can load those images and return them as BatchedVideoDatapoint
                img_batch: A [TxBxCxHxW]
                masks: A [TxOxHxW] <-- This is going to be weird I would say.
                T: frames per video
                B: videos in the batch
        '''
        super().__init__()
        # Arguments
        self.num_multimask_outputs = num_multimask_outputs
        self.len_video = len_video

        # Get root path and split folders
        root_path = MMOR_DATA_ROOT_PATH
        split_take_keys = MMOR_SPLIT_TO_TAKES.keys()
        assert split_type in split_take_keys, "Provided split type is not valid!"
        split_folder_names = MMOR_SPLIT_TO_TAKES[split_type]

        # Data containers
        self.images = []
        self.segmentation_masks = []

        ### You can use Ege's however we dont wanna open the images beforehand but we will open on the run.
        # Iterate over the take folders
        for take_name in split_folder_names:
            # Folder names
            json_path = root_path / 'take_jsons' / f'{take_name}.json'
            take_path = root_path / take_name
            
            # Read MMOR/Simstation JSON file for timestamps and image paths
            with json_path.open() as f:
                take_json = json.load(f)
                timestamps = take_json['timestamps']
                timestamps = {int(k): v for k, v in timestamps.items()}
                timestamps = sorted(timestamps.items())
            
            # Iterate over the time stamps (The order of the for loops is changed to have a video)
            for c_idx in tqdm([1, 4, 5]):
                segmasks = []
                for idx, (timestamp, image_files) in enumerate(timestamps):
                    if idx % len_video == 0:
                        video_batch_image = []
                        video_batch_seg_mask = []
                    # Data paths
                    rgb_path = take_path / 'colorimage' / f'camera0{c_idx}_colorimage-{image_files["azure"]}.jpg'
                    mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
                    interpolated_mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}_interpolated.png'

                    # Store the data paths
                    video_batch_image.append(rgb_path)
                    video_batch_seg_mask.append(mask_path) if mask_path.exists() else video_batch_seg_mask.append(interpolated_mask_path)
                    if (idx % len_video == len_video - 1) or (idx == len(timestamps)-1):
                        self.images.append(video_batch_image)
                        self.segmentation_masks.append(video_batch_seg_mask)

                # assume azure is not available, use simstation instead
                if len(segmasks) == 0:
                    for c_idx in [0, 2, 3]:
                        # Simstation data paths
                        simstation_rgb_path = take_path / 'simstation' / f'camera0{c_idx}_{image_files["simstation"]}.jpg'
                        simstation_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{simstation_rgb_path.stem}.png'
                        simstation_interpolated_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{simstation_rgb_path.stem}_interpolated.png'

                        # Store simstation data paths
                        self.images.append(simstation_rgb_path)
                        if simstation_mask_path.exists():
                            self.segmentation_masks.append(simstation_mask_path)
                        elif interpolated_mask_path.exists():
                            self.segmentation_masks.append(simstation_interpolated_mask_path)
        
        if False:
            # Create paths to data directories and data path containers
            data_paths = [root_path / ii for ii in split_folder_names]
            seg_data = []
            im_data = []
            frame_indices = []

            # Iterate the data take dirs 001_PKA, 003_TKA etc.
            for dpath in data_paths:
                # I dont understand somehow it is sim something 
                if dpath == root_path / '005_TKA':
                    continue
                print(f'Currently extracting from folder: {dpath}\n')
                # Get segmentation folders and camera indeces
                seg_export_data_list = [dpath / dir_idx for dir_idx in os.listdir(dpath) if dir_idx.startswith('segmentation_export_')]  # Like [/.../segmetation_export_1, /.../segmetation_export_4]
                camera_indices = [seg_exp_idx.name[-1] for seg_exp_idx in seg_export_data_list]

                # Segemntation dictionary has the keys as take indices and value as the data path
                # seg_dict = [{png_idx.split('.')[0].split('-')[1].split('-')[0]: seg_exp_idx / png_idx for png_idx in os.listdir(seg_exp_idx) if png_idx.startswith('camera')} for seg_exp_idx in seg_export_data_list]
                seg_data_sub = [[ seg_exp_idx / png_idx for png_idx in os.listdir(seg_exp_idx) if png_idx.startswith('camera')] for seg_exp_idx in seg_export_data_list]
                frame_indices_sub = [[re.findall(r'\d+', png_idx)[1] for png_idx in os.listdir(seg_exp_idx) if png_idx.startswith('camera')] for seg_exp_idx in seg_export_data_list]

                # Print Log
                num_data_in_take = sum([len(ii) for ii in frame_indices_sub])
                print(f'{num_data_in_take} samples of segmentation data extracted from the folder: {dpath}\n')

                # Set color image folder
                color_image_path = dpath / "colorimage"
                im_data_sub = [[color_image_path / f'camera0{camera_indices[ii]}_colorimage-{idx}.jpg' for idx in frame_idx] for ii, frame_idx in enumerate(frame_indices_sub)]

                # Merge into main data storage
                seg_data.extend(seg_data_sub)
                frame_indices.extend(frame_indices_sub)
                im_data.extend(im_data_sub)
            print('Loading of the data path completed!')

            # Store the data as self and as a whole list not list of lists
            self.segmentation_masks = [x for xb in seg_data for x in xb]
            self.images = [x for xb in im_data for x in xb]
            assert len(self.segmentation_masks) == len(self.images), "There is a problem with the lengths of the data containers"

    def __len__(self):
        return len(self.segmentation_masks)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # segmentation_mask = cv2.imread(self.segmentation_masks[index])
        segmentation_mask = Image.open(self.segmentation_masks[index]).convert("RGB")
        one_hot_mask = self._convert_to_one_hot_mask(segmentation_mask)
        return pil_to_tensor(image), pil_to_tensor(segmentation_mask), one_hot_mask

    def _convert_to_one_hot_mask(self, segmentation_mask):
        seg_np = np.array(segmentation_mask)[:,:,0].T  # We only need one channel, transpose since numpy reverts the positions
        one_hot_mask = torch.zeros((self.num_multimask_outputs , *segmentation_mask.size))
        i = 0
        for obj_keys, obj_values in TRACK_TO_METAINFO.items():
            if obj_keys == '__background__':
                continue
            if i == self.num_multimask_outputs:
                print('Small Demo num mask tokens are not enough!')
                ## SHAL BE REMOVED AFTER INCREASING MASK TOKEN NUMBER
                continue
            label = obj_values['label']
            mask1 = seg_np == label
            set_flag = np.any(mask1)
            one_hot_mask[i] = torch.tensor(mask1, dtype=torch.uint8)
            i += 1 if set_flag else 0
        return one_hot_mask


if __name__ == '__main__':
    md = MiniDataset("small_train")
    print(len(md))
    print(md[6700])