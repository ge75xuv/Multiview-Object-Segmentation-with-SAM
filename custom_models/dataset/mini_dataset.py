import json
import os
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from training.utils.data_utils import Object, Frame, VideoDatapoint
from tqdm import tqdm

from ..helpers.configurations import *

class MiniDataset(Dataset):
    
    def __init__(self, split_type:str, num_multimask_outputs:int, len_video:int):
        '''Initialzie the class open the data folders and store them.
        TODO Data Augmentatation
        TODO Video Batches: DONE
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
            print(f'Loading the take {take_name}!\n')
            
            # Read MMOR/Simstation JSON file for timestamps and image paths
            with json_path.open() as f:
                take_json = json.load(f)
                timestamps = take_json['timestamps']
                timestamps = {int(k): v for k, v in timestamps.items()}
                timestamps = sorted(timestamps.items())

            # Check if the take needs to use simstation or azure
            flag_simstation = False
            seg_export_folders = [dir_name for dir_name in os.listdir(take_path) if dir_name.startswith('segmentation_export')]
            if len(seg_export_folders) == 0:
                flag_simstation = True
                seg_export_folders = [dir_name for dir_name in os.listdir(take_path) if dir_name.startswith('simstation_segmentation_export')]
                assert len(seg_export_folders) != 0, "Azure and Simstation data does not exist!"
            camera_indices = [int(idx[-1]) for idx in seg_export_folders]
            
            # Iterate over the time stamps (The order of the for loops is changed to have a video from a cameraview first)
            for c_idx in tqdm(camera_indices):
                for idx, (timestamp, image_files) in enumerate(timestamps):
                    # Create the video frames lists
                    if idx % len_video == 0:
                        video_batch_image = []
                        video_batch_seg_mask = []

                    # Data paths
                    if not flag_simstation:
                        rgb_path = take_path / 'colorimage' / f'camera0{c_idx}_colorimage-{image_files["azure"]}.jpg'
                        mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
                        interpolated_mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}_interpolated.png'
                    else:
                        # Simstation data paths
                        rgb_path = take_path / 'simstation' / f'camera0{c_idx}_{image_files["simstation"]}.jpg'
                        mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
                        interpolated_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{rgb_path.stem}_interpolated.png'

                    # Store the data paths
                    if mask_path.exists():
                        video_batch_image.append(rgb_path)
                        video_batch_seg_mask.append(mask_path)  
                    elif interpolated_mask_path.exists():
                        video_batch_image.append(rgb_path)
                        video_batch_seg_mask.append(interpolated_mask_path)
                    else:
                        break

                    # Complete the video sequence
                    if (idx % len_video == len_video - 1) or (idx == len(timestamps)-1):
                        self.images.append(video_batch_image)
                        self.segmentation_masks.append(video_batch_seg_mask)
   
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
        video_frames = self.images[index]
        video_frames_segmentation_mask = self.segmentation_masks[index]
        frame_obj_list, frames_segmentation_mask = self._convert_to_one_hot_mask(video_frames, video_frames_segmentation_mask)
        return frame_obj_list, frames_segmentation_mask

    def _convert_to_one_hot_mask(self, video_frames, video_frames_segmentation_mask):        
        frame_obj_list = []
        frames_segmentation_mask = []

        # Iterate over the frames of a video
        for frame_idx, frame in enumerate(video_frames):
            # Open frame and segmentation mask as pillow image
            im_frame = Image.open(frame).convert("RGB")
            segmentation_mask = Image.open(video_frames_segmentation_mask[frame_idx]).convert("RGB")
            seg_np = np.array(segmentation_mask)[:,:,0].T  # We only need one channel, transpose since numpy reverts the positions
            
            # Initialize one-hot-mask and obj list
            one_hot_mask = torch.zeros((self.num_multimask_outputs , *segmentation_mask.size))
            obj_list = []
            
            # Iterate over the object keys and values
            i = 0
            for obj_keys, obj_values in TRACK_TO_METAINFO.items():
                if obj_keys == '__background__':
                    continue
                if i == self.num_multimask_outputs - 1:
                    print('Small Demo num mask tokens are not enough!')
                    ## SHAL BE REMOVED AFTER INCREASING MASK TOKEN NUMBER
                    continue
                # Get label, find regions with the label and set the mask
                label = obj_values['label']
                mask1 = seg_np == label
                set_flag = np.any(mask1)
                one_hot_mask[i] = torch.tensor(mask1, dtype=torch.uint8)
                i += 1 if set_flag else 0  # if the object does not occur, do not increase
                # Occupy obj_list with the objects in the scene
                obj_list.append(Object(label, frame_idx, one_hot_mask[i]))

            # Occupy the frames 
            frame_obj_list.append(Frame(pil_to_tensor(im_frame), obj_list))
            frames_segmentation_mask.append(segmentation_mask)
        return frame_obj_list, frames_segmentation_mask


if __name__ == '__main__':
    md = MiniDataset("small_train")
    print(len(md))
    print(md[6700])