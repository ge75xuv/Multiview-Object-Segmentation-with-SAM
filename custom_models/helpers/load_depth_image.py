from PIL import Image
from torchvision.transforms import ToTensor

def load_depth_image(video_frames, size_x_y):
    depth_images = []
    to_tensor = ToTensor()
    for frame in video_frames:
        last_part = str(frame.parts[-1].split('/')[-1].replace('.jpg', '.tiff').replace('color', 'depth'))
        depth_path = frame.parents[1] / 'depthimage' / last_part
        # Load depth image
        gt_depth_image = Image.open(depth_path)
        # Compute the scale factor
        scaling_factor = gt_depth_image.size[0] // size_x_y[1]  # for PIL it is W,H for tensor it is H,W
        resize_shape = [size_x_y[0], gt_depth_image.size[1] // scaling_factor]  # It is PIL reshape give it as W,H
        # Resize depth image to the specified size
        gt_depth_image = gt_depth_image.resize(resize_shape)
        gt_depth_image = to_tensor(gt_depth_image)
        depth_images.append(gt_depth_image)
    return depth_images