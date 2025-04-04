import json
from multiprocessing import Manager
import os
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from training.utils.data_utils import Object, Frame, VideoDatapoint
from tqdm import tqdm

from custom_models.helpers.configurations import *

class MiniDataset(Dataset):
    
    def __init__(self,
                 split_type:str,
                 len_video:int,
                 input_image_size:int,
                 object_labels: List[int],
                 transforms,
                 collate_fn,
                 batch_size:int=1,
                 num_workers:int=0,
                 **kwargs):
        '''Initialzie the class open the data folders and store them.
            Store images in self.images as [[fr0, fr1, fr2, frK-1], [fr0+K, fr1+K, fr2+K, frK+k-1], ...]
            so a list of small videos. Thus during __getitem__ I can load those images and return them as BatchedVideoDatapoint
            img_batch: A [TxBxCxHxW]
            masks: A [TxOxHxW] <-- This is going to be weird I would say.
            T: frames per video
            B: videos in the batch
        '''
        super().__init__()
        # Arguments
        self.len_video = len_video
        self.batch_size = batch_size
        self.num_workers= num_workers
        self.collate_fn=collate_fn
        self.input_image_size = input_image_size
        self.transforms_ = transforms
        self.object_labels = list(set(object_labels))  # Get rid of the repeating values if they exist
        true_labels = [obj['label'] for obj in iter(TRACK_TO_METAINFO.values())]
        assert all([obj in true_labels for obj in object_labels]), 'Unidentified key in the obj labels'

        # Initialize image resizer
        original_image_size = (1536, 2048)  # (H,W)
        scaling_factor = original_image_size[1] // input_image_size  # Depends on the long edge, short will be padded.
        resize_shape = [original_image_size[0] // scaling_factor, input_image_size]
        self.resize_image = Resize(resize_shape)

        # Initialize image transforms
        self.to_tensor = ToTensor()

        # Adjustment for the trainer
        self.get_seg_mask = kwargs.get('get_seg_mask', False)

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
                        break  # in 010_PKA there are some problems with the first images: image_files['azure']=None, same for 011 TKA and 035 PKA, for only few time_stamps

                    # Complete the video sequence
                    if (idx % len_video == len_video - 1) or (idx == len(timestamps)-1):
                        self.images.append(video_batch_image)
                        self.segmentation_masks.append(video_batch_seg_mask)
        if split_type == 'over_train':
            cam_switch = len(self.images) // 3
            start_idx = 2700
            # end_idx = 2900
            end_idx = 2730
            # idx_range_free = [i for i in range(1900, 2100)]
            idx_range_free = []
            idx_range_cam1 = [i for i in range(start_idx, end_idx)]
            idx_range_cam4 = [ii for ii in range(start_idx + cam_switch, end_idx + cam_switch)]
            idx_range = idx_range_free + idx_range_cam1 + idx_range_cam4
            self.images = [self.images[i] for i in idx_range]
            self.segmentation_masks = [self.segmentation_masks[i] for i in idx_range]

        # Python multiprocessing manager
        # This is a workaround for the multiprocessing issue with PyTorch
        manager = Manager()
        self.images = manager.list(self.images)
        self.segmentation_masks = manager.list(self.segmentation_masks)
        #TODO Another solution is to use np.array(self.images) however if len_video > 1 and there is an uncomplete video
        # left, then thre will be inhomogeneous shapes in the array. So we may need to dump it.
        # Convert the list into a numpy array
        # self.images = np.array(self.images)
        # self.segmentation_masks = np.array(self.segmentation_masks)

    def __len__(self):
        return len(self.segmentation_masks)

    def __getitem__(self, index):
        # Get file paths
        video_frames = self.images[index]
        video_frames_segmentation_mask = self.segmentation_masks[index]

        # Open the images and process (resizing to input size, padding, etc.)
        frame_obj_list, frames_segmentation_mask = self._open_and_process(video_frames, video_frames_segmentation_mask)

        # Create the VideoDatapoint
        size_x_y = frame_obj_list[0].data.shape[-2:]
        video_datapoint = VideoDatapoint(frame_obj_list, index, size_x_y)

        # Apply transforms
        for transform in self.transforms_:
            video_datapoint = transform(video_datapoint)
        if self.get_seg_mask:
            return video_datapoint, frames_segmentation_mask
        return video_datapoint

    def _open_and_process(self, video_frames, video_frames_segmentation_mask):
        frame_obj_list = []
        frames_segmentation_mask = []

        # Iterate over the frames of a video
        for frame_idx, frame in enumerate(video_frames):
            # Open frame and segmentation mask as pillow image
            im_frame = Image.open(frame).convert("RGB")
            segmentation_mask = Image.open(video_frames_segmentation_mask[frame_idx]).convert("RGB")
            seg_np = np.array(segmentation_mask)[:,:,0]  # We only need one channel, it's same anyways

            # Initialize one-hot-mask and obj list
            obj_list = []

            # Iterate over the objects in the dataset
            for label in self.object_labels:
                # Get label, find regions with the label and set the mask
                # This would absolutely work here
                # correct_class_probs = softmax_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                mask1 = seg_np == label
                mask1 = self.resize_image(torch.tensor(mask1[None,:,:], dtype=torch.uint8) * 255)
                mask1 = self._add_padding(mask1, self.input_image_size).squeeze(0)

                # Occupy obj_list with the objects in the scene.
                # Temporary solution, 
                pseudo_label = LABEL_PROJECTION_MAP[int(label)]['label']
                obj_list.append(Object(pseudo_label, frame_idx, mask1))

            # Occupy the frames
            im_frame = self.resize_image(self.to_tensor(im_frame))
            im_frame = self._add_padding(im_frame, self.input_image_size)
            frame_obj_list.append(Frame(im_frame, obj_list))
            frames_segmentation_mask.append(segmentation_mask)
        return frame_obj_list, frames_segmentation_mask

    def _add_padding(self, input_image:torch.Tensor, out_shape:int):
        H = out_shape - input_image.shape[-2]
        out_image = torch.zeros([input_image.shape[0], out_shape, out_shape], dtype=input_image.dtype)
        out_image[:, H//2:-H//2, :] = input_image
        return out_image

    def get_loader(self, **kwargs):
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def load_checkpoint_state(*args, **kwargs):
        pass


if __name__ == '__main__':
    md = MiniDataset("small_train")
    print(len(md))
    print(md[6700])