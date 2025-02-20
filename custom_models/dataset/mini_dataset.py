import os
import re
import torch
from torch.utils.data import Dataset

from ..helpers.configurations import *

class MiniDataset(Dataset):
    
    def __init__(self, split_type:str):
        super().__init__()
        # Get root path and split folders
        root_path = MMOR_DATA_ROOT_PATH
        split_take_keys = MMOR_SPLIT_TO_TAKES.keys()
        assert split_type in split_take_keys, "Provided split type is not valid!"
        split_folder_names = MMOR_SPLIT_TO_TAKES[split_type]
        
        # Create paths to data directories
        data_paths = [root_path / ii for ii in split_folder_names]
        
        # Create data path containers for segmentation and image
        seg_data = []
        im_data = []
        frame_indices = []
        
        # Iterate the data dirs 001_PKA, 003_TKA etc.
        for dpath in data_paths:
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

    def __len__():
        pass
    
    def __getitem__():
        pass


if __name__ == '__main__':
    md = MiniDataset("small_train")