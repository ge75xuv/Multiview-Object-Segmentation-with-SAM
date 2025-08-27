# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable

from fvcore.common.param_scheduler import ParamScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# fvcore overrides
class LinearParamScheduler(ParamScheduler):
    """
    Linearly interpolates parameter between ``start_value`` and ``end_value``.
    Can be used for either warmup or decay based on start and end values.
    The schedule is updated after every train step by default.

    Example:

        .. code-block:: python

            LinearParamScheduler(start_value=0.0001, end_value=0.01)

    Corresponds to a linear increasing schedule with values in [0.0001, 0.01)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        warmup_length: float
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value
        self._warmup_length = warmup_length
        assert warmup_length > 0 and warmup_length < 1, "Warmup length must be positive and smaller than 1"

    def __call__(self, where: float) -> float:
        # interpolate between start and end values
        where = where / self._warmup_length
        if where > 1:
            return self._end_value
        return self._end_value * where + self._start_value * (1 - where)

# training overrides
def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """
    Loads a state dict into the given model. Be advised this is not the function that is called 
    when resuming training, but rather starting from begining with given model weights.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
        ignore_missing_keys: unix pattern of keys to ignore
    """
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)

    # Sanity check
    try:
        model_fuse = model.multi_object_memory_proj.score_mlp[0].weight.shape[1]
        sd_fuse = state_dict['multi_object_memory_proj.score_mlp.0.weight'].shape[1]
        if model_fuse != sd_fuse:
            Warning(f"Model fuse size {model_fuse} does not match state dict fuse size {sd_fuse}.")
            # Remove the multi_object_memory_proj from the state dict
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('multi_object_memory_proj')}
    except:
        Warning(f"Base model checkpoint is used")

    # The pixel decoder in earlier versions have 3 layers now we increase it to 4.
    try:
        if model.sam_mask_decoder.pixel_decoder.transformer.level_embed.shape[0] != state_dict['sam_mask_decoder.pixel_decoder.transformer.level_embed'].shape[0]:
            Warning(f"Model level embed size {model.sam_mask_decoder.pixel_decoder.transformer.level_embed.shape[0]} does not match state dict level embed size {state_dict['sam_mask_decoder.pixel_decoder.transformer.level_embed'].shape[0]}.")
            # Remove the pixel decoder transformer from the state dict
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('sam_mask_decoder.pixel_decoder')}
    except KeyError:
        Warning(f"Base model checkpoint is used")

    # Load state dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Error loading state dict into model: {e}")
        
    #TODO maybe we just use one projection layer
    # Load the multi object memory proj as multi object epipolar projection
    len_missing_keys = len(missing_keys)
    missing_keys = [k for k in missing_keys if not (k.startswith("multi_object_epi"))]  # Custom multi-object epipolar
    epipolar_proj_sd_missing = len_missing_keys - len(missing_keys) > 0
    if epipolar_proj_sd_missing:
        epipolar_proj_sd = {'.'.join(key.split('.')[1:]) : value for key, value in state_dict.items() 
                             if key.startswith('multi_object_memory')}
        model.multi_object_epi_proj.load_state_dict(epipolar_proj_sd)

    # Ignore check_load_state_dict_errors
    missing_keys = [k for k in missing_keys if not (k.startswith("sam_mask_decoder") or
                                                  k.startswith("multi_object_memory") or  # Custom multi-object memory
                                                  k.startswith("multi_object_epi") or  # Custom multi-object epipolar
                                                  k.startswith("view_spatial_prior") or  # Custom view spatial prior
                                                  k.startswith("reliability_and_bias") or  # Custom reliability and bias
                                                  k.startswith("source_target_reliability"))  # Custom source target
                    ]

    # Parts that do not belong to the model anymore
    unexpected_keys = [k for k in unexpected_keys 
                       if not (k.startswith("sam_mask_decoder") or
                               k.startswith("sam_prompt_encoder") or
                               k.startswith("memory") or  # Included in base model checkpoint
                               k.startswith("multi_object_memory") or  # Memory part projection is changed
                               k.startswith('sam_mask_decoder'))
                       ]

    # Check if the epipolar encoder state dict exists, if not load the memory encoder state dict
    len_missing_keys = len(missing_keys)
    missing_keys = [k for k in missing_keys 
                    if not k.startswith("epipolar_encoder")]
    epipolar_sd_missing = len_missing_keys - len(missing_keys) > 0
    # For epipolar mask encoding we use the same structure as the memory encoder
    # Get the state dict keys remove memory encider from the name
    memory_encoder_sd = {'.'.join(key.split('.')[1:]) : value for key, value in state_dict.items() if key.startswith('memory_encoder')}
    model.epipolar_encoder.load_state_dict(memory_encoder_sd, strict=True) if epipolar_sd_missing else None

    # Check if the epipolar attention state dict exists, if not load the memory attention state dict
    len_missing_keys = len(missing_keys)
    missing_keys = [k for k in missing_keys 
                    if not k.startswith("epipolar_attention")]
    epipolar_attn_sd_missing = len_missing_keys - len(missing_keys) > 0
    # For epipolar attention we use the same structure as the memory attention
    # Get the state dict keys remove memory attention from the name
    memory_attention_sd = {'.'.join(key.split('.')[1:]) : value for key, value in state_dict.items() if key.startswith('memory_attention')}
    model.epipolar_attention.load_state_dict(memory_attention_sd, strict=True) if epipolar_attn_sd_missing else None

    # For the parameters no_epipolar_embed and no_epipolar_pos_enc load the no_mem_embed and no_mem_pos_enc
    len_missing_keys = len(missing_keys)
    missing_keys = [k for k in missing_keys 
                    if not k.startswith("no_epipolar")]
    no_epipolar_missing = len_missing_keys - len(missing_keys) > 0
    if no_epipolar_missing:
        no_epipolar_embed_sd = state_dict['no_mem_embed']
        no_epipolar_pos_enc_sd = state_dict['no_mem_pos_enc']
        model.no_epipolar_embed.data.copy_(no_epipolar_embed_sd)
        model.no_epipolar_pos_enc.data.copy_(no_epipolar_pos_enc_sd)

    assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    return model

# detectron2 overrides
@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def sample_from_obj_id(target_labels, target_masks, obj_id, point_coords, num_points) -> torch.Tensor:
    '''Check if the object id is in the target if so sample num_points many points 
    from it and concatenate to the point coords, if not do not do anything.'''
    if obj_id not in target_labels:
        return point_coords
    # Get the mask for the object id
    obj_mask = target_masks[target_labels == obj_id]  # Shape: [H, W] or [1, H, W]
    # Ensure mask is 3D
    obj_mask = obj_mask.squeeze(1)
    # Get the shapes
    B, H, W = obj_mask.shape
    # Get the points where the mask is 1
    mask_pts = torch.where(obj_mask == 1)
    mask_size = mask_pts[0].numel()
    if mask_size == 0:
        return point_coords
    # Randomly sample num_points from the mask
    # If the mask size is smaller than num_points, sample all points
    num_points_to_sample = min(num_points, mask_size)
    if mask_size > num_points_to_sample:
        indeces = torch.randperm(mask_size, device=obj_mask.device)[:num_points_to_sample]
        w_coords = mask_pts[2][indeces]
        h_coords = mask_pts[1][indeces]
    else:
        w_coords = mask_pts[2]
        h_coords = mask_pts[1]
    # Normalize the coordinates to [0, 1]
    w_normalized = (w_coords.float() + 0.5) / (W)
    h_normalized = (h_coords.float() + 0.5) / (H)

    sampled_points = torch.stack([w_normalized, h_normalized], dim=1)  # [N_sampled, 2]

    # Add batch dimension and concatenate
    batch_size = point_coords.shape[0]
    sampled_points = sampled_points.unsqueeze(0).expand(batch_size, -1, -1)
    
    point_coords = torch.cat((point_coords, sampled_points), dim=1)
    return point_coords

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)