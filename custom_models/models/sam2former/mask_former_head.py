# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import fvcore.nn.weight_init as weight_init
from hydra.utils import instantiate
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d

from .lib import ShapeSpec


class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: Dict[str, Any],
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: Dict[str, Any],
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape_original = deepcopy(input_shape)
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        pixel_decoder['_target_'] = pixel_decoder.pop('target_holder')
        self.pixel_decoder = instantiate(pixel_decoder, _convert_="partial", input_shape=input_shape_original)

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        transformer_predictor['_target_'] = transformer_predictor.pop('target_holder')
        self.predictor = instantiate(transformer_predictor)
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    def forward(self, features, mask=None, epipolar_attn_bias=None):
        return self.layers(features, mask, epipolar_attn_bias)

    def layers(self, features, mask=None, epipolar_attn_bias=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask, epipolar_attn_bias=epipolar_attn_bias)
        return predictions
