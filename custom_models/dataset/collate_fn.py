# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
The document is slightly adjusted from the SAM2's original code base.
'''
from dataclasses import dataclass
from tensordict import tensorclass
import torch
from typing import List, Tuple, Optional

from training.utils.data_utils import VideoDatapoint, BatchedVideoDatapoint, BatchedVideoMetaData

@tensorclass
class MultiviewBatchedVideoDatapoint:
    """This class represents a batch of videos from 3 views with associated annotations and metadata.
    Attributes:
        view1_batchvideo: A BatchedVideoDatapoint instance for the first view.
        view2_batchvideo: A BatchedVideoDatapoint instance for the second view.
        view3_batchvideo: A BatchedVideoDatapoint instance for the third view.
    """
    view1_batchvideo: BatchedVideoDatapoint
    view2_batchvideo: BatchedVideoDatapoint
    view3_batchvideo: BatchedVideoDatapoint

    def __getitem__(self, idx: int) -> BatchedVideoDatapoint:
        """
        Allows indexing into the multiview batch to get a specific view's BatchedVideoDatapoint.
        """
        if idx == 1:
            return self.view1_batchvideo
        elif idx == 2:
            return self.view2_batchvideo
        elif idx == 3:
            return self.view3_batchvideo
        else:
            raise IndexError("Index out of range for multiview batch.")

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.view1_batchvideo.batch_size[0]

def collate_fn_wrapper(
    batch: List[Tuple[VideoDatapoint]],
    num_frames: int = 0,
    dict_key: Optional[str] = 'all',
    ) -> MultiviewBatchedVideoDatapoint:
    """
    A wrapper function for multiview VideoDataPoint.
    """
    # Not multiview case
    if isinstance(batch[0], VideoDatapoint):
        return collate_fn(batch, dict_key)
    
    # Multiview case
    assert isinstance(batch[0], list), "Batch should be a list of lists for multiview data."
    assert len(batch) == 1, "Cannot support batch size > 1 for multiview data."
    assert len(batch[0]) == 3, "Batch should contain exactly 3 views for multiview data."
    view_batchvideo = [None, None, None]
    for idx, bc_view in enumerate(batch[0]):
        view_batchvideo[idx] = collate_fn([bc_view], dict_key=dict_key)
    return MultiviewBatchedVideoDatapoint(*view_batchvideo)

def collate_fn(
    batch: List[VideoDatapoint],
    dict_key: Optional[str] = 'all',
) -> BatchedVideoDatapoint:
    """
    Args:
        batch_container: A list of VideoDatapoint instances AND segmentation_masks.
    """
    # frames_segmentation_mask = [bcc[1] for bcc in batch]
    # batch = [bcc[0] for bcc in batch]
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )  # , frames_segmentation_mask