import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def random_action_if_within_delta(qs, delta=0.0001):
    q_diff = qs.max(-1).values - qs.min(-1).values
    random_action_mask = q_diff < delta
    if random_action_mask.sum() == 0:
        return None
    argmax_q = qs.max(-1)[1]
    random_actions = torch.randint(0, qs.size(-1), random_action_mask.shape).to(
        qs.device
    )
    argmax_q = torch.where(random_action_mask, random_actions, argmax_q)
    return argmax_q


def encode_action(
        continuous_action: torch.Tensor,
        initial_low: torch.Tensor,
        initial_high: torch.Tensor,
        levels: int,
        bins: int,
):
    """Encode continuous action to discrete action

    Args:
        continuous_action: [..., D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
    Returns:
        discrete_action: [..., L, D] shape tensor where L is the level
    """
    low = initial_low.repeat(*continuous_action.shape[:-1], 1) # [..., D]
    high = initial_high.repeat(*continuous_action.shape[:-1], 1) # [..., D]

    initial_low = low.clone()
    initial_high = high.clone()

    idxs = []
    for _ in range(levels):
        # Put continuous values into bin
        slice_range = (high - low) / bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, bins - 1) # [..., D]
        idxs.append(idx)

        # Re-compute low/high for each bin (i.e., Zoom-in)
        recalculated_action = low + slice_range * idx
        recalculated_action = torch.clip(recalculated_action, -1.0, 1.0)
        low = recalculated_action
        high = recalculated_action + slice_range
        low = torch.maximum(initial_low, low)
        high = torch.minimum(initial_high, high)
    discrete_action = torch.stack(idxs, -2) # [..., L, D]
    return discrete_action


def decode_action(
        discrete_action: torch.Tensor,
        initial_low: torch.Tensor,
        initial_high: torch.Tensor,
        levels: int,
        bins: int,
):
    """Decode discrete action to continuous action

    Args:
        discrete_action: [..., L, D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
    Returns:
        continuous_action: [..., D] shape tensor
    """
    low = initial_low.repeat(*discrete_action.shape[:-2], 1)
    high = initial_high.repeat(*discrete_action.shape[:-2], 1)

    initial_low = low.clone()
    initial_high = high.clone()

    for i in range(levels):
        slice_range = (high - low) / bins
        continuous_action = low + slice_range * discrete_action[..., i, :]
        low = continuous_action
        high = continuous_action + slice_range
        low = torch.maximum(initial_low, low)
        high = torch.minimum(initial_high, high)
    continuous_action = (high + low) / 2.0
    return continuous_action


def zoom_in(low: torch.Tensor, high: torch.Tensor, argmax_q: torch.Tensor, bins: int):
    """Zoom-in to the selected interval

    Args:
        low: [D] shape tensor that denotes minimum of the current interval
        high: [D] shape tensor that denotes maximum of the current interval
    Returns:
        low: [D] shape tensor that denotes minimum of the *next* interval
        high: [D] shape tensor that denotes maximum of the *next* interval
    """
    slice_range = (high - low) / bins
    continuous_action = low + slice_range * argmax_q
    low = continuous_action
    high = continuous_action + slice_range
    # low = torch.maximum(-torch.ones_like(low), low)
    # high = torch.minimum(torch.ones_like(high), high)
    return low, high


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

