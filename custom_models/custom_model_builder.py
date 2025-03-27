# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re

from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="./configs", job_name="train_run")
from omegaconf import OmegaConf
import torch
import torch.nn as nn


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
    # Register new operation becaise of learning rate
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    # Resolve
    OmegaConf.resolve(cfg)
    print(f'OmegaConf resolved successfully')
    # Instantiate model, loss, load weights, freeze backbone
    model = instantiate(cfg.model, _recursive_=True)
    loss = instantiate(cfg.loss, _recursive_=True)
    _load_checkpoint(model, ckpt_path, kwargs['_load_partial'])
    _remove_parameters_of_backbone(model)
    # Load custom trainer only for the purpose of schedulers for the optimizer
    trainer = instantiate(cfg.trainer, model=model)
    trainer._construct_optimizers(cfg.optimizer)
    optim = trainer.optim
    # send model to device
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model, loss, optim

def build_sam2_predict(
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
    # Register new operation becaise of learning rate
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    # Resolve
    OmegaConf.resolve(cfg)
    print(f'OmegaConf resolved successfully')
    # Instantiate model, loss, load weights, freeze backbone
    training_key = list(cfg.keys())[0]
    model = instantiate(cfg[training_key].trainer.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, False)
    # send model to device
    model = model.to(device)
    model.eval()
    return model, cfg[training_key]['scratch']['obj_labels']

def build_sam2former(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    # Register new operation becaise of learning rate
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    # Resolve
    OmegaConf.resolve(cfg)
    print(f'OmegaConf resolved successfully')
    # Instantiate model, loss, load weights, freeze backbone
    training_key = list(cfg.keys())[0]
    model = instantiate(cfg[training_key].trainer.model, _recursive_=True)
    # _load_checkpoint(model, ckpt_path, kwargs['_load_partial'])
    _remove_parameters_of_backbone(model)
    # send model to device
    model = model.to(device)
    if mode == "eval":
        model.eval()
    optim = lambda:0
    optim.optimizer = torch.optim.AdamW(model.parameters())
    loss = torch.nn.MSELoss()
    return model, cfg[training_key]['scratch']['obj_labels'], optim, loss

def _load_checkpoint(model, ckpt_path, _load_partial:bool=False):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        # Clean the unused keywords from the loaded state dict
        if _load_partial:
            sd, ignored_keys = _find_keys_in_sd(sd)
        # Load the weights
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        # Ignore the missing keys related to the mask decoder mlps
        if _load_partial:
            missing_keys = [ii for ii in missing_keys if ii not in ignored_keys]
            missing_keys = [jj for jj in missing_keys if jj not in model.state_dict().keys()]
        # The issue is that the missing keys return the keys that exist in the model but havent been found in the sd
        # So the ones we remove (mask decoder 123) and also the ones we added newly(mask decoder 4-N).
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
    else:
        raise RuntimeError('Provide a path for checkpoint!')

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

def _remove_parameters_of_backbone(model: nn.Module):
    '''Remove the parameters of the model backbone from the trainable
    parameters for the optimizer.
    '''
    for ii in model.image_encoder.parameters():
        ii.requires_grad = False
    #TODO This freezes the whole encoder, however in the example Ege mentioned, 
    # they freeze the trunk and keep the neck. That could have been done as well.
    #TODO The forward pass of the whole encoder is done with torch.no_grad(), which
    # means, in case we need to unfreeze the neck, we need to modify the no_grad