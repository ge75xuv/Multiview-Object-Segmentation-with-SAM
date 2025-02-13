# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re

from hydra import compose, initialize
from hydra.utils import instantiate
initialize(config_path="../custom_models", job_name="my_app")
from omegaconf import OmegaConf
import torch


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        # Clean the unused keywords from the loaded state dict
        sd, ignored_keys = _find_keys_in_sd(sd)
        # Load the weights
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        # Ignore the missing keys related to the mask decoder mlps
        missing_keys = [ii for ii in missing_keys if ii not in ignored_keys]
        missing_keys = [jj for jj in missing_keys if jj not in model.state_dict().keys()]
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

def _find_keys_in_sd(sd: dict):
    '''Find the parameters keys which belong to the decoder hypernetworks and mask tokens. 
    Remove them from the state dict of the loaded parameters.'''
    hyper_mlp = []
    iou_pred = []
    mask_tokens = []
    for key in sd.keys():
        if re.search("sam_mask_decoder.output_hypernetworks_mlps.*", key):
            hyper_mlp.append(key)
        elif re.search("sam_mask_decoder.iou_prediction_head.*", key):
            iou_pred.append(key)
            ## IoU Prediction in sadece outputu degisiyor N+1 olarak. Weightleri almamiz sart degil 
            ## cunku zaten baska bir sey icin kullanacagiz
        elif re.search("sam_mask_decoder.mask_tokens.*", key):
            mask_tokens.append(key)
    for key in hyper_mlp + iou_pred + mask_tokens:
        sd.pop(key)
    return sd, hyper_mlp + iou_pred + mask_tokens