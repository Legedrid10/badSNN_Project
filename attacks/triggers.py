import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def T_p(x, q=None):
    """Power transformation trigger (Equation 3)
    Normalizes each image to [0, 1] independently, applies power q, and denormalizes.
    Handles input shapes [B, C, H, W] and [T, B, C, H, W].
    """
    if q is None:
        q = Config.POWER_Q

    orig_shape = x.shape
    if x.dim() == 5:
        # [T, B, C, H, W] -> [T*B, C*H*W] for per-image min/max
        x_flat = x.reshape(orig_shape[0] * orig_shape[1], -1)
    else:
        # [B, C, H, W] -> [B, C*H*W]
        x_flat = x.reshape(orig_shape[0], -1)

    x_min = x_flat.min(dim=1, keepdim=True).values
    x_max = x_flat.max(dim=1, keepdim=True).values

    constant = (x_max == x_min)
    denom = torch.where(constant, torch.ones_like(x_max), x_max - x_min)
    x_norm = (x_flat - x_min) / denom
    x_transformed = x_norm ** q
    result = x_transformed * (x_max - x_min) + x_min
    result = torch.where(constant, x_flat, result)

    return result.reshape(orig_shape)

def T_s(x, beta=0.03):
    """Neuromorphic noise trigger (Equation 7)
    Generates and applies valid bounded neuromorphic noise.
    """
    epsilon = torch.rand_like(x) * 2 * beta - beta
    return torch.clip(x + epsilon, 0, 1)

def adaptive_blending(x, T_p_x, deepfool_noise, alpha):
    """Adaptive blending (Equation 4) combining T_p and DeepFool noise."""
    return (1 - alpha) * (x - T_p_x) + alpha * (x + deepfool_noise)
