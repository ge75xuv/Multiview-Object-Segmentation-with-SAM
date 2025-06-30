# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import numpy as np
import torch
import torch.distributed
from .sam2former_base import SAM2FormerBase
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points
from training.utils.data_utils import BatchedVideoDatapoint

from custom_models.models.sam2former.epipolar_maskv2 import epipolar_main

class SAM2FormerTrain(SAM2FormerBase):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        multiview=False,
        epipolar_encoder=None,
        epipolar_attention=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        freeze_sam_mask_decoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, multiview, epipolar_attention, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            # HEADS UP
            for p in self.image_encoder.trunk.parameters():
            # for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_sam_mask_decoder:
            # HEADS UP
            for p in self.sam_mask_decoder.parameters():
                p.requires_grad = False

        if self.multiview:
            self.epipolar_encoder = epipolar_encoder
        else:
            self.epipolar_encoder = None
            del epipolar_encoder  # avoid unused argument warning

    def forward(self, input: BatchedVideoDatapoint):
        num_frames = input.num_frames if not self.multiview else input[0].num_frames
        if not self.multiview and (self.training or not self.forward_backbone_per_frame_for_eval):
        # if num_frames == 1 and (self.training or not self.forward_backbone_per_frame_for_eval):
            # precompute image features on all frames before tracking
            # input.img_batch.shape = (num_frames, B, 3, H, W)
            # input.flat_img_batch = (B * num_frames, 3, H, W)
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            num_init_cond_frames = 1
            rand_init_cond_frames = False
        else:
            num_init_cond_frames = 1
            rand_init_cond_frames = False

        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1

        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        # output_dict has view indeces as keys and frame outputs as values, where frame outputs are also dictionaries
        output_dict = {0: {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},}  # dict containing {frame_idx: <out>}
        } if not self.multiview else {view_idx: {
            "cond_frame_outputs": {},  # dict containing {view_idx: {frame_idx: <out>}}
            "non_cond_frame_outputs": {},  # dict containing {view_idx: {frame_idx: <out>}}
            }
            for view_idx in range(3)
        }
        view_inputs = [input[idx] for idx in range(3)] if self.multiview else [input]
        camera_int_ext = input.camera_intrinsics_extrinsics if self.multiview else None
        feature_container_for_multiview_fusion = {}
        encoded_epipolar_dict = {0: {}, 1: {}, 2: {}}
        # t_stage = time.time()
        for stage_id in processing_order:
            for view_idx, input in enumerate(view_inputs):
                # Get the image features for the current frames
                # HEADS UP, we need the index of the frame but we dont want to repeat it O (object number) many times for the batch size
                # This part is fairly hard to understand but basically we need to select the features in the given frame among
                # (B*L, C, H, W). Since B*L is a flat dimension.
                img_ids_full = input.flat_obj_to_img_idx[stage_id]
                img_ids = img_ids_full.unique()
                if img_feats_already_computed:
                    # Retrieve image features according to img_ids (if they are already computed).
                    current_vision_feats = [x[:, img_ids] for x in vision_feats]
                    current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
                else:
                    # Otherwise, compute the image features on the fly for the given img_ids
                    # (this might be used for evaluation on long videos to avoid backbone OOM).
                    (
                        _,
                        current_vision_feats,
                        current_vision_pos_embeds,
                        feat_sizes,
                    ) = self._prepare_backbone_features_per_frame(
                        input.flat_img_batch, img_ids
                    )

                # Get output masks based on this frame's prompts and previous memory
                current_out = self.track_step(
                    frame_idx=stage_id,
                    is_init_cond_frame=stage_id in init_cond_frames,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    output_dict=output_dict[view_idx],
                    encoded_epipolar_dict = encoded_epipolar_dict[view_idx] if self.multiview else None,
                    num_frames=num_frames,
                    run_mem_encoder=True if num_frames > 1 else False,
                )

                add_output_as_cond_frame = stage_id in init_cond_frames
                if add_output_as_cond_frame:
                    output_dict[view_idx]["cond_frame_outputs"][stage_id] = current_out
                else:
                    output_dict[view_idx]["non_cond_frame_outputs"][stage_id] = current_out

                # We need the vision features for epipolar encoding
                if self.multiview:
                    HW, B, D = current_vision_feats[-1].shape
                    H = W = torch.sqrt(torch.tensor(HW)).int()
                    last_vision_feat = current_vision_feats[-1].detach().clone()  #NOTE NOT SURE IF DETACH
                    feature_container_for_multiview_fusion[view_idx] = last_vision_feat.permute(1, 2, 0).view(B, D, H, W)
            # end view loop
            # print(f"Processed stage {stage_id} in {time.time() - t_stage:.2f} seconds")
            # Fuse the epipolar lines
            if self.multiview:
                pred0 = output_dict[0]["cond_frame_outputs"][stage_id] if add_output_as_cond_frame else output_dict[0]["non_cond_frame_outputs"][stage_id]
                pred1 = output_dict[1]["cond_frame_outputs"][stage_id] if add_output_as_cond_frame else output_dict[1]["non_cond_frame_outputs"][stage_id]
                pred2 = output_dict[2]["cond_frame_outputs"][stage_id] if add_output_as_cond_frame else output_dict[2]["non_cond_frame_outputs"][stage_id]
                t = time.time()
                epipolar_masks, object_pos_label = epipolar_main(camera_int_ext, pred0, pred1, pred2)
                print(f"Epipolar lines computed in {time.time() - t} seconds")  #TODO: make it faster
                epipolar_masks = epipolar_masks.to(self.device)
                # Scale the epipolar masks and add bias (Use the same scale and bias as in the memory encoder)
                epipolar_masks = epipolar_masks.sigmoid() * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc
                pix_feat = torch.vstack(list(feature_container_for_multiview_fusion.values()))
                # NOTE pix_feats = (3, 256, 16, 16), epipolar_masks = (O, 3, H, W)
                # NOTE pix_feat_for_mem = (1, 256, 16, 16) mask_for_mem = (23, 1, 256, 256)
                # Encode the epipolar features
                for view_idx in range(3):
                    encoded_epipolars = self.epipolar_encoder(pix_feat[view_idx:view_idx+1], epipolar_masks[:,view_idx:view_idx+1])
                    # dictionary with keys "vision_features" and "vision_pos_enc"
                    encoded_epipolars['object_positions_labels'] = object_pos_label[view_idx]
                    encoded_epipolar_dict[view_idx] = encoded_epipolars

        # end stage loop

        if return_dict:
            return output_dict

        # HEADS UP Get rid of unnecessary keys in the output_dict
        used_keys = ['pred_masks', 'pred_masks_high_res', 'pred_logits', 'aux_outputs']
        for view_idx in output_dict:
            for key, value in output_dict[view_idx].items():  # cond_frame_outputs, non_cond_frame_outputs
                for k, v in value.items():  # frame_idx
                    frame_out = output_dict[view_idx][key][k]
                    output_dict[view_idx][key][k] = {out_type: out_val for out_type, out_val in frame_out.items() if out_type in used_keys}

        # turn `output_dict` into a list for loss function
        # all_frame_outputs = {view_idx: {} for view_idx in output_dict.keys()}
        # for view_idx in all_frame_outputs:
        #     all_frame_outputs[view_idx].update(output_dict[view_idx]["cond_frame_outputs"])
        #     all_frame_outputs[view_idx].update(output_dict[view_idx]["non_cond_frame_outputs"])

        # turn `output_dict` into a list for loss function. It does the above but much faster
        all_frame_outputs = {
            view_idx: {stage_id: view_out["cond_frame_outputs"][stage_id] 
                       if stage_id in view_out["cond_frame_outputs"].keys() 
                       else view_out["non_cond_frame_outputs"][stage_id] for stage_id in processing_order} 
                for view_idx, view_out in output_dict.items()
            }

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        # point_inputs,
        # mask_inputs,
        output_dict,
        num_frames,
        encoded_epipolar_dict=None,  # dictionary containing encoded epipolar features
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            # point_inputs,
            # mask_inputs,
            output_dict,
            encoded_epipolar_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_masks,
            high_res_masks,
            obj_ptr,
            pred_logits,
            aux_outputs,
        ) = sam_outputs

        # We can leave these too since they are the same as pred masks, for sam2 correction points make the
        # difference and we don't have them here
        # current_out["multistep_pred_masks"] = low_res_masks
        # current_out["multistep_pred_masks_high_res"] = high_res_masks
        # current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        # current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        # current_out["multistep_pred_ious"] = [ious]
        # current_out["multistep_point_inputs"] = [point_inputs]
        # current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        # if frame_idx in frames_to_add_correction_pt:
        #     point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
        #         is_init_cond_frame,
        #         point_inputs,
        #         gt_masks,
        #         high_res_features,
        #         pix_feat,
        #         low_res_multimasks,
        #         high_res_multimasks,
        #         ious,
        #         low_res_masks,
        #         high_res_masks,
        #         object_score_logits,
        #         current_out,
        #     )
        #     (
        #         _,
        #         _,
        #         _,
        #         low_res_masks,
        #         high_res_masks,
        #         obj_ptr,
        #         object_score_logits,
        #     ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["pred_logits"] = pred_logits
        current_out["aux_outputs"] = aux_outputs

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        # Since python subscriptibles are passed by reference, it updates the dictionary even though
        # there is no return
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            # point_inputs,
            run_mem_encoder,
            high_res_masks,
            # object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs
