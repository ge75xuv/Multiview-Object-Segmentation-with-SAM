import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_models.models.sam2former.point_cloud_mask import apply_padding_on_depth

class SourceTargetReliability(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.num_views = num_views
        self.W = nn.Parameter(torch.zeros(num_views, num_views))  # target x source

    def forward(self, epipolar_masks):
        '''Apply source-target reliability to epipolar masks.
        epipolar_masks: B, 3, 2, H, W'''
        W = torch.sigmoid(self.W)  # (0,1)
        # Apply reliability scores in the corresponding epipolar masks
        for i in range(self.num_views):
            other_views = [j for j in range(self.num_views) if j != i]
            epipolar_masks[:,i] *= W[i, other_views][None, :, None, None]  # Bx2xHxW
        return epipolar_masks

class ViewSpatialPrior(nn.Module):
    """
    One coarse learnable map per fixed camera view.
    Zero-initialized -> starts as no-op.
    """
    def __init__(self, num_views: int, hc: int = 32, wc: int = 32):
        super().__init__()
        self.maps = nn.Parameter(torch.zeros(num_views, 1, hc, wc))  # learnable priors

    def forward(self, view_ids: torch.Tensor, out_h: int, out_w: int):
        """
        view_ids: [B] ints in {0..num_views-1}
        returns:  [B,1,out_h,out_w]
        """
        prior = self.maps[view_ids]  # [B,1,hc,wc]
        prior = F.interpolate(prior, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return prior

class ReliabilityAndBias(nn.Module):
    """
    Pixelwise reliability + capped attention-bias for Mask2Former/SAM2 pipelines.

    Inputs (all tensors are BxCxHxW):
      - self_logits:         Bx1xHxW   (mask logits from the current view's decoder)
      - pseudo_maps:         BxMxHxW   (stack of pseudo masks from other views; logits or probabilities)
      - depth_rgb:           Bx3xHxW   (depth image encoded as RGB; will be normalized to [0,1])

    Returns:
      - r:                   Bx1xHxW   (pixelwise reliability in [0,1], with epsilon floor)
      - bias:                Bx1xHxW   (capped attention-bias derived from consensus pseudo + r)
    """

    def __init__(
        self,
        pseudo_are_logits: bool = True,
        eps_floor: float = 0.02,         # floor to avoid exact 0/1
        temperature: float = 1.0,        # gate softness; >1 is softer
        lambda_scale: float = 0.75,      # how strongly the bias nudges attention
        beta_clip: float = 3.0,          # cap on bias magnitude (in logit units)
        use_pseudo_dropout: bool = True,
        pseudo_dropout_p: float = 0.3,   # drop pseudo channels at train time to prevent over-reliance
        norm_groups: int = 16,
        in_augment_depth_valid: bool = True,  # add a simple depth-valid channel from depth_rgb>0
    ):
        super().__init__()
        self.pseudo_are_logits = pseudo_are_logits
        self.eps_floor = eps_floor
        self.temperature = temperature
        self.lambda_scale = lambda_scale
        self.beta_clip = beta_clip
        self.use_pseudo_dropout = use_pseudo_dropout
        self.pseudo_dropout_p = pseudo_dropout_p
        self.in_augment_depth_valid = in_augment_depth_valid

        # Channels: [self_logits(1), consensus(1), abs(self-consensus)(1),
        #            pseudo stack (M), depth_rgb(3), depth_valid(1 optional)]
        # We'll build layers to accept a variable number of pseudo channels by instantiating
        # the first conv at runtime once we see M. To keep nn.Module static, we use a small
        # subnetwork with 1x1 "adapter" first.
        self.adapter = nn.Conv2d(0, 0, 1)  # placeholder; will be replaced at first forward()

        # Core head after adapter brings inputs -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(num_groups=min(norm_groups, 32), num_channels=32),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(num_groups=min(norm_groups, 16), num_channels=16),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(16, 1, 1)

        # Lazily replaced when we know input channels
        depth_included = True
        if depth_included:
            self._build_adapter(9)
        else:
            self._build_adapter(6)

    @staticmethod
    def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(min=eps, max=1 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _build_adapter(self, in_ch: int, device: torch.device='cpu'):
        # Bring arbitrary in_ch to 32 channels with a 1x1 conv
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1),
            nn.GroupNorm(num_groups=min(16, 32), num_channels=32),
            nn.GELU(),
        )
        self.adapter = self.adapter.to(device)
        self._adapter_built = True

    @staticmethod
    @torch.no_grad()
    def depth_feats_from_mm(depth_mm: torch.Tensor,
                            zmin_m: float = 0.2,
                            zmax_m: float = 10.0,
                            use_percentile_clip: bool = False):
        """
        depth_mm: Bx1xHxW (int/float), millimeters. 0 (or <=0) = invalid.
        Returns:
        d_norm:  Bx1xHxW  normalized inverse depth in [0,1], 0 where invalid
        valid:   Bx1xHxW  {0,1}
        gradmag: Bx1xHxW  |∇(d_norm)|, 0 where invalid
        """
        x = depth_mm.float()
        valid = (x > 0).float()
        z_m = (x / 1000.0).clamp(min=zmin_m)  # avoid inf when inverting

        if use_percentile_clip:
            # Robust per-batch clipping within [1st, 99th] percentile of valid depths
            v = z_m[valid.bool()]
            if v.numel() > 0:
                lo = torch.quantile(v, 0.01).item()
                hi = torch.quantile(v, 0.99).item()
                zmin_m, zmax_m = float(max(zmin_m, lo)), float(min(zmax_m, hi))

        z_m = z_m.clamp(max=zmax_m)
        inv = 1.0 / z_m
        inv_min, inv_max = 1.0 / zmax_m, 1.0 / zmin_m
        d_norm = ((inv - inv_min) / (inv_max - inv_min)).clamp(0.0, 1.0) * valid

        # Simple Sobel gradient magnitude on normalized inverse depth
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3) / 8.0
        ky = kx.transpose(2,3)
        gx = F.conv2d(d_norm, kx, padding=1)
        gy = F.conv2d(d_norm, ky, padding=1)
        gradmag = (gx.abs() + gy.abs()) * valid

        return d_norm, valid, gradmag

    def forward(
        self,
        self_logits: torch.Tensor,     # Bx1xHxW
        pseudo_maps: torch.Tensor,     # BxMxHxW
        depth_mm: torch.Tensor,       # Bx1xHxW
        view_spatial_prior: torch.Tensor,  # Bx1xHxW
        *,
        detach_pseudo: bool = False,   # optionally stop grads through pseudo path early in training
    ):
        # Check the input size
        if len(self_logits.shape) == 3:
            self_logits = self_logits.unsqueeze(1)
        B, _, H, W = self_logits.shape

        # Depth specific preprocessing
        H_depth, W_depth = depth_mm.shape[-2:]  # Assuming all depth images have the same shape
        assert W == W_depth, "Image size width must match depth image width"
        padding = (W_depth-H_depth) // 2
        depth_trunc = 10 # 10 meters max
        depth_mm = apply_padding_on_depth(depth_mm, padding=padding, img_size=(H, W), device=self_logits.device)
        depth_mm = depth_mm[None, None, :].repeat(B, 1, 1, 1)  # Bx1xHxW

        # Check the view prior
        if view_spatial_prior.shape[0] != B:
            view_spatial_prior = view_spatial_prior.unsqueeze(1).repeat(B, 1, 1, 1)

        # Normalize depth_mm to [0, 1] per batch (robust to 0..255 or 0..1 inputs)
        depth_mm_norm, depth_valid, depth_gradmag = self.depth_feats_from_mm(depth_mm, zmax_m=depth_trunc)

        # Handle pseudo logits/probabilities
        if detach_pseudo:
            pseudo_maps = pseudo_maps.detach()

        if self.pseudo_are_logits:
            pseudo_logits = pseudo_maps
        else:
            pseudo_logits = self._safe_logit(pseudo_maps)

        # Optional pseudo-channel dropout during training
        if self.training and self.use_pseudo_dropout and pseudo_logits.shape[1] > 0:
            drop_mask = (torch.rand(B, pseudo_logits.shape[1], 1, 1, device=pseudo_logits.device) >
                         self.pseudo_dropout_p).float()
            pseudo_logits = pseudo_logits * drop_mask

        # Consensus: pixelwise median over views (robust). If M==1, it's just that map.
        if pseudo_logits.shape[1] == 0:
            # No pseudo channels? fallback: zero consensus
            consensus_logit = torch.zeros_like(self_logits)
        else:
            consensus_logit, _ = pseudo_logits.median(dim=1, keepdim=True)

        # Agreement magnitude (helps the head learn where pseudo & self disagree)
        agreement = (self_logits - consensus_logit).abs()

        # Build input feature stack
        feats = [self_logits, consensus_logit, agreement, view_spatial_prior, pseudo_logits, depth_mm_norm, depth_gradmag, depth_valid]
        x = torch.cat(feats, dim=1)

        # Lazy-build adapter with correct channel count
        if not self._adapter_built:
            self._build_adapter(x.shape[1], device=x.device)

        # Tiny CNN → reliability logits
        z = self.adapter(x)
        z = self.block1(z)
        z = self.block2(z)
        r_raw = self.out_conv(z)  # Bx1xHxW

        # Temperatured sigmoid with epsilon floor/ceiling
        r = torch.sigmoid(r_raw / self.temperature)
        if self.eps_floor > 0:
            r = self.eps_floor + (1 - 2 * self.eps_floor) * r  # keep in [eps, 1-eps]

        # ---- Attention bias from consensus + reliability ----
        # If inputs are logits, logit(sigmoid(x)) == x. If probs, we already converted to logits.
        bias = consensus_logit * r
        bias = bias.clamp(min=-self.beta_clip, max=self.beta_clip)
        bias = bias * self.lambda_scale  # final bias to add to attention logits

        return r, bias