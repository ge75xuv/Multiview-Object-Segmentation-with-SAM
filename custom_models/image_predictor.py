from hydra import initialize
initialize(version_base=None, config_path="./configs", job_name="predict_run")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from torchvision.transforms import ToPILImage, ToTensor

from .helpers.configurations import TRACK_TO_METAINFO
from .dataset.collate_fn import collate_fn
from .dataset.mini_dataset import MiniDataset
from .debugging.show import *
# from .custom_model_builder import build_sam2

model_size_dict = {
    'base': {
        'config': 'custom_sam2.1_hiera_b+.yaml',
        'ck': '/home/guests/tuna_gurbuz/prototype/sam2_logs/sam2.1_hiera_b+_promptless.yaml/checkpoints/checkpoint.pt',
        },
}
seed = 6
torch.manual_seed(seed)
np.random.seed(seed)

# seed 6 idx 25424 val

def predict():
    # Model
    model_size = 'base'
    config = model_size_dict[model_size]['config']
    ck = model_size_dict[model_size]['ck']
    submodel = build_sam2(config, ck, 'cpu')
    im_pred = SAM2ImagePredictor(submodel)

    # Dataset
    len_video = 1
    input_image_size = 512
    batch_size = 1
    shuffle = False
    test_dataset = MiniDataset('val', len_video, input_image_size, collate_fn, batch_size, shuffle, get_seg_mask=True)

    # Image
    len_objects = len(TRACK_TO_METAINFO.keys())
    toPILimage = ToPILImage()
    exist = False
    while True:
        if exist:
            break
        idx = np.random.randint(0, len(test_dataset))
        idx = 25424
        print(f'Index: {idx}')
        frame_obj_list, frames_segmentation_mask = test_dataset[idx]
        for i in range(len_video):
            if exist:
                break
            image = frame_obj_list.frames[i].data
            segmentation_mask = frames_segmentation_mask[i]
            # Images are double precision tensors so we multiply 255 and convert to numpy uint8
            image = toPILimage(image)
            image.save(f'temp/image{i}.png')
            segmentation_mask.save(f'temp/segmentation_mask{i}.png')
            for j in range(len_objects):
                exist = torch.any(frame_obj_list.frames[i].objects[j].segment == True)
                print(f'Exist Flag: {exist}')
                if exist:
                    break
                toPILimage(frame_obj_list.frames[i].objects[j].segment/255).save(f'temp/one_hot{i}{j}.png')
    im_pred.set_image(image)

    # Point Copy-Paste from the sam2_tune
    O = len_objects
    points = torch.tensor([[i,i] for i in range(O)]).unsqueeze(1)
    labels = torch.tensor([[1]]).tile((O,1))

    # Predict
    masks, scores, logits = im_pred.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=True,
)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    show_masks(image, masks, scores, point_coords=points, input_labels=labels, borders=False)

if __name__ == '__main__':
    predict()
