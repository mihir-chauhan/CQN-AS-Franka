from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensordict import TensorDict

import arsq_rlb.util.utils as utils
from arsq_rlb.alg.cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
)


class RandomShiftsAug(nn.Module):
    """
    Random shift augmentation for rgb observations
    """

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]
        self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded

        self.conv_nets = nn.ModuleList()
        for _ in range(self.num_views):
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                ImgChLayerNorm(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                ImgChLayerNorm(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                ImgChLayerNorm(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                ImgChLayerNorm(256),
                nn.SiLU(),
            )
            self.conv_nets.append(conv_net)

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor):
        # obs: [B, V, C, H, W]
        obs = obs / 255.0 - 0.5
        hs = []
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
        gru_layers: int,
        rgb_encoder_layers: int,
        levels: int,
        bins: int,
        atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._action_sequence, self._actor_dim = 1, action_shape[0] if len(action_shape) == 1 else action_shape
        self._bins = bins

        # Advantage stream in Dueling network
        ## RGB encoder for advantage stream
        adv_rgb_encoder_net = []
        input_dim = repr_dim
        for i in range(rgb_encoder_layers):
            adv_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        adv_rgb_encoder_net = adv_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.adv_rgb_encoder = nn.Sequential(*adv_rgb_encoder_net)

        ## Low-dimensional encoder for advantage stream
        self.adv_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        ## Main network for advantage stream
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.adv_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.adv_head = nn.Linear(
            hidden_dim,
            self._actor_dim * bins * atoms,
        )
        self.adv_output_shape = (self._action_sequence * self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        ## RGB encoder for advantage stream
        value_rgb_encoder_net = []
        input_dim = repr_dim
        for i in range(rgb_encoder_layers):
            value_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        value_rgb_encoder_net = value_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.value_rgb_encoder = nn.Sequential(*value_rgb_encoder_net)

        ## Low-dimensional encoder for advantage stream
        self.value_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        ## Main network for advantage stream
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.value_head = nn.Linear(
            hidden_dim,
            self._actor_dim * 1 * atoms,
        )
        self.value_output_shape = (self._action_sequence * self._actor_dim, 1, atoms)

        self.apply(utils.weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def encode(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        value_h = torch.cat(
            [self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs)],
            -1,
        )
        adv_h = torch.cat(
            [self.adv_rgb_encoder(rgb_obs), self.adv_low_dim_encoder(low_dim_obs)],
            -1,
        )
        return value_h, adv_h

    def forward_each_level(
        self,
        level: int,
        features: tuple[torch.Tensor, torch.Tensor],
        prev_action: torch.Tensor,
    ):
        """
        Implementation that processes forward step for each level

        Inputs:
        - features: ([B, D], [B, D]) for value_h and adv_h, shared for all levels
        - prev_actions: [B, D] shaped prev action for the current level

        Outputs:
        - q_logits: [B, action_sequence * action_dimensions, bins, atoms]
        """
        value_h, adv_h = features

        level_id = (
            torch.eye(self._levels, device=value_h.device, dtype=value_h.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )
        level_id = level_id.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        prev_action = prev_action.view(
            -1, self._action_sequence, self._actor_dim
        )  # [B, T, D]
        action_sequence_id = (
            torch.eye(self._action_sequence, device=value_h.device, dtype=value_h.dtype)
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )  # [B, T, T]

        # Value
        value_h = value_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        value_x = torch.cat(
            [value_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats, _ = self.value_gru(value_feats)
        values = self.value_head(value_feats).view(-1, *self.value_output_shape)

        # Advantage
        adv_h = adv_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        adv_x = torch.cat(
            [adv_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats, _ = self.adv_gru(adv_feats)
        advs = self.adv_head(adv_feats).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        prev_actions: torch.Tensor,
    ):
        """
        Optimized implementation that processes forward step for all the levels at once
        This is possible because we have pre-computed prev_actions for all the levels
        when we are computing Q(s,a) for given action a.
        But this is not possible when we want to get actions for given s, as we have to
        compute action for each level. In this case, use `forward_each_level`

        Inputs:
        - rgb_obs: [B, D]
        - low_dim_obs: [B, D]
        - prev_actions: [B, L, D]

        Outputs:
        - q_logits: [B, L, action_sequence * action_dimensions, bins, atoms]
        """
        device, dtype = rgb_obs.device, rgb_obs.dtype
        levels = prev_actions.size(1)
        B, L, T = prev_actions.size(0), levels, self._action_sequence

        # Reshape previous actions
        prev_actions = prev_actions.view(-1, L, T, self._actor_dim)  # [B, L, T, D]

        # Action sequence id - [T, T] -> [B, L, T, T]
        action_sequence_id = torch.eye(T, device=device, dtype=dtype)[
            None, None, :, :
        ].repeat(B, L, 1, 1)

        # level id - [L, L] -> [B, L, T, L]
        level_id = torch.eye(L, device=device, dtype=dtype)[None, :, None, :].repeat(
            B, 1, T, 1
        )

        # Encode features
        value_h, adv_h = self.encode(rgb_obs, low_dim_obs)

        # Value
        value_h = value_h[:, None, None, :].repeat(1, L, T, 1)
        value_x = torch.cat([value_h, prev_actions, action_sequence_id, level_id], -1)
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats = value_feats.view(B * L, T, -1)
        value_feats = self.value_gru(value_feats)[0]
        values = self.value_head(value_feats).view(B, L, *self.value_output_shape)

        # Advantage
        adv_h = adv_h[:, None, None, :].repeat(1, L, T, 1)
        adv_x = torch.cat([adv_h, prev_actions, action_sequence_id, level_id], -1)
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats = adv_feats.view(B * L, T, -1)
        adv_feats = self.adv_gru(adv_feats)[0]
        advs = self.adv_head(adv_feats).view(B, L, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        repr_dim: int,
        low_dim: int,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
        gru_layers: int,
        rgb_encoder_layers: int,
        use_parallel_impl: bool,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_parallel_impl = use_parallel_impl
        # Safely handle both 1D (standard) and 2D (sequence) action shapes
        if len(action_shape) == 1:
            actor_dim = action_shape[0]
        else:
            actor_dim = action_shape[0] * action_shape[1]
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms), requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.network = C2FCriticNetwork(
            repr_dim,
            low_dim,
            action_shape,
            feature_dim,
            hidden_dim,
            gru_layers,
            rgb_encoder_layers,
            levels,
            bins,
            atoms,
        )

    def get_action(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            q_logits = self.network.forward_each_level(
                level, features, (low + high) / 2
            )
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            rgb_obs: [B, repr_dim] shaped feature tensor
            low_dim_obs: [B, low_dim] shaped feature tensor
            continuous_action: [B, D] shaped action tensor

        Return:
            q_probs: [B, L, D, bins, atoms] for value distribution at all bins
            q_probs_a: [B, L, D, atoms] for value distribution at given bin
            log_q_probs: [B, L, D, bins, atoms] with log probabilities
            log_q_probs_a: [B, L, D, atoms] with log probabilities
        """

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        if self.use_parallel_impl:
            # Pre-compute previous actions for all the levels
            prev_actions = []
            for level in range(self.levels):
                prev_actions.append((low + high) / 2)
                argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]
                low, high = zoom_in(low, high, argmax_q, self.bins)
            q_logits_all = self.network(rgb_obs, low_dim_obs, torch.stack(prev_actions, 1))
        else:
            features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            if self.use_parallel_impl:
                q_logits = q_logits_all[:, level]
            else:
                q_logits = self.network.forward_each_level(level, features, (low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # (Log) Probs [..., D, bins, atoms]
            # (Log) Probs_a [..., D, atoms]
            q_probs = F.softmax(q_logits, 3)  # [B, D, bins, atoms]
            q_probs_a = torch.gather(
                q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            q_probs_a = q_probs_a[..., 0, :]  # [B, D, atoms]

            log_q_probs = F.log_softmax(q_logits, 3)  # [B, D, bins, atoms]
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            log_q_probs_a = log_q_probs_a[..., 0, :]  # [B, D, atoms]

            q_probs_per_level.append(q_probs)
            q_probs_a_per_level.append(q_probs_a)
            log_q_probs_per_level.append(log_q_probs)
            log_q_probs_a_per_level.append(log_q_probs_a)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_probs = torch.stack(q_probs_per_level, -4)  # [B, L, D, bins, atoms]
        q_probs_a = torch.stack(q_probs_a_per_level, -3)  # [B, L, D, atoms]
        log_q_probs = torch.stack(log_q_probs_per_level, -4)
        log_q_probs_a = torch.stack(log_q_probs_a_per_level, -3)
        return q_probs, q_probs_a, log_q_probs, log_q_probs_a

    def compute_target_q_dist(
        self,
        next_rgb_obs: torch.Tensor,
        next_low_dim_obs: torch.Tensor,
        next_continuous_action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_rgb_obs: [B, repr_dim] shaped feature tensor
            next_low_dim_obs: [B, low_dim] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, atoms] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(
            next_rgb_obs, next_low_dim_obs, next_continuous_action
        )[1]

        shape = next_q_probs_a.shape  # [B, L, D, atoms]
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        # Compute Tz for [B, atoms]
        Tz = reward + discount * self.support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.v_min) / self.delta_z
        # Mask for conditions
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        lower_mask = (upper > 0) & (lower == upper)
        upper_mask = (lower < (self.atoms - 1)) & (lower == upper)
        # Apply masks separately
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        # Repeat Tz for (L * D) times -> [B * L * D, atoms]
        multiplier = batch_size // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        # Distribute probability of Tz
        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )
        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )  # m_u = m_u + p(s_t+n, a*)(b - l)

        m = m.view(*shape)  # [B, L, D, atoms]
        return m

    def encode_decode_action(self, continuous_action: torch.Tensor):
        """Encode and decode actions"""
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        return continuous_action


class CQNASAgent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        levels,
        bins,
        atoms,
        v_min,
        v_max,
        bc_lambda,
        bc_margin,
        gru_layers,
        rgb_encoder_layers,
        use_parallel_impl,
        critic_lambda,
        critic_target_tau,
        critic_target_interval,
        weight_decay,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        nu = 0.01,
        kappa = 0.1,
        num_walks = 10,
        use_fk_reg = True,
        use_eikonal = False,
        fk_weight = 1.0,
        **kwargs,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.critic_target_interval = critic_target_interval
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda
        self.nu = nu
        self.kappa = kappa
        self.num_walks = num_walks
        self.use_fk_reg = use_fk_reg
        self.fk_weight = fk_weight
        self.use_eikonal = use_eikonal

        # models
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)
        self.critic = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
            gru_layers,
            rgb_encoder_layers,
            use_parallel_impl,
        ).to(device)
        self.critic_target = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
            gru_layers,
            rgb_encoder_layers,
            use_parallel_impl,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

        print(self.encoder)
        print(self.critic)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.critic_target.get_action(
            rgb_obs, low_dim_obs
        )  # use critic_target
        stddev = torch.ones_like(action) * stddev
        dist = utils.TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()[0]

    def add_noise_to_action(self, action: np.array, step: int):
        if step < self.num_expl_steps:
            action = np.random.uniform(-1.0, 1.0, size=action.shape).astype(
                action.dtype
            )
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            action = np.clip(
                action
                + np.random.normal(0, stddev, size=action.shape).astype(action.dtype),
                -1.0,
                1.0,
            )
        return action

    # def update_critic(
    #     self,
    #     rgb_obs,
    #     low_dim_obs,
    #     action,
    #     reward,
    #     discount,
    #     next_rgb_obs,
    #     next_low_dim_obs,
    #     demos,
    # ):
    #     with torch.no_grad():
    #         next_action = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
    #         target_q_probs_a = self.critic_target.compute_target_q_dist(
    #             next_rgb_obs, next_low_dim_obs, next_action, reward, discount
    #         )

    #     # Cross entropy loss for C51
    #     q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
    #         rgb_obs, low_dim_obs, action
    #     )
    #     q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
    #     critic_loss = self.critic_lambda * q_critic_loss

    #     demos = demos.float().squeeze(1)  # [B,]

    #     # BC - First-order stochastic dominance loss
    #     # q_probs: [B, L, D, bins, atoms], q_probs_a: [B, L, D, atoms]
    #     q_probs_cdf = torch.cumsum(q_probs, -1)
    #     q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
    #     # q_probs_{a_{i}} is stochastically dominant over q_probs_{a_{-i}}
    #     bc_fosd_loss = (
    #         (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
    #         .clamp(min=0)
    #         .sum(-1)
    #         .mean([-1, -2, -3])
    #     )
    #     bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
    #     critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss

    #     # BC - Margin loss
    #     support = self.critic.support.to(q_probs.device)
    #     qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
    #     qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(-1)
    #     margin_loss = torch.clamp(
    #         self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
    #     ).mean([-1, -2, -3])
    #     margin_loss = (margin_loss * demos).sum() / demos.sum()
    #     critic_loss = critic_loss + self.bc_lambda * margin_loss


    #     # =========================================================================
    #     # --- 4. NEW: Stochastic Eikonal Regularization (Speed Limit) ---
    #     # =========================================================================
    #     # nu = self.nu
    #     # sigma = (2.0*nu)**0.5
        
    #     # #grad_epsilon = 0.05
    #     # num_walks = self.num_walks
    #     # kappa = self.kappa
        
    #     # # A. Define Speed (Constraint)
    #     # # Using placeholder 1.0 unless velocity is available in low_dim_obs
    #     # current_speed = torch.ones((low_dim_obs.shape[0], 1), device=low_dim_obs.device)
    #     # # If low_dim_obs has velocity at indices [-2:], uncomment:
    #     # # current_speed = torch.norm(low_dim_obs[:, -2:], dim=1, keepdim=True)

    #     # q_s = kappa / (current_speed + 1e-6)

    #     # # B. Generate Neighbors (Perturb Low-Dim State)
    #     # B, D = low_dim_obs.shape
    #     # noise = torch.randn(B, num_walks, D, device=low_dim_obs.device) * sigma
    #     # #noise = F.normalize(noise, dim=-1) * grad_epsilon # Unit vectors * epsilon
        
    #     # flat_low_dim = (low_dim_obs.unsqueeze(1) + noise).view(B * num_walks, D)
    #     # flat_rgb = (rgb_obs.unsqueeze(1)).expand(-1, num_walks, *rgb_obs.shape[1:]).reshape(B * num_walks, *rgb_obs.shape[1:])
    #     # flat_action = action.unsqueeze(1).expand(-1, num_walks, *action.shape[1:]).reshape(B * num_walks, *action.shape[1:])

    #     # # D. Query Neighbors (s + epsilon)
    #     # # -----------------------------------------------------------
    #     # # Output shape: [B*Walks, D1, D2, ..., Atoms]
    #     # _, _, flat_log_probs_all_bins,_ = self.critic(flat_rgb, flat_low_dim, flat_action)
        
    #     # # 1. Convert to Probabilities
    #     # probs_all_bins = flat_log_probs_all_bins.exp()
    #     # support = self.critic.support.to(probs_all_bins.device)
        
    #     # # 2. Collapse Atoms -> Expected Q
    #     # # Sum over the last dimension (Atoms)
    #     # # Shape: [B*Walks, D1, D2, ...]
    #     # support_expanded = support.expand_as(probs_all_bins)
    #     # q_expected_grid = (probs_all_bins * support_expanded).sum(dim=-1)

    #     # v_neighbors_flat = q_expected_grid.amax(dim=-1)
    #     # v_neighbors = v_neighbors_flat.view(B, num_walks, *v_neighbors_flat.shape[1:])
        
    #     # # -----------------------------------------------------------
        
    #     # # E. Get Anchor Value V(s)
    #     # # We re-use 'qs' from the earlier Margin Loss.
    #     # # qs contains Expected Q for all actions: [B, D1, D2, ...]
        
    #     # v_anchor_scalar = qs.amax(dim=-1) # [B, L, D]
    #     # v_anchor = v_anchor_scalar.unsqueeze(1) # [B, 1, L, D]
        
    #     # # F. Compute Gradient Penalty
    #     # # |V(s') - V(s)| / epsilon
    #     # # We detach v_anchor so we don't pull the anchor up, only push neighbors down.
    #     # diff = (v_neighbors - v_anchor.detach()).abs()
    #     # grad_epsilon = torch.linalg.norm(noise, dim=-1) + 1e-6 # [B, Walks]
    #     # grad_estimate = diff / grad_epsilon.detach().unsqueeze(-1).unsqueeze(-1)    # [B, Walks, L, D]
        
        
        
    #     # # Hinge Loss
    #     # slope_excess = torch.relu(max_slope.unsqueeze(-1).unsqueeze(-1) - grad_estimate)
    #     # #slope_excess = grad_estimate - max_slope
    #     # eikonal_loss = slope_excess.pow(2).mean()

    #     # # Add to total loss (Weight 0.1)
    #     # critic_loss = critic_loss + self.fk_weight * eikonal_loss
    #     # # print("Debug - Eikonal Loss:", eikonal_loss.item(), "Avg Grad Estimate:", grad_estimate.mean().item())
    #     # # print("Debug - Max Slope (Speed Limit):", max_slope.mean().item())
    #     # # print("Debug - Current Speed:", current_speed.mean().item())
    #     # # print("Debug - v anchor (V(s)):", v_anchor_scalar.mean().item())

    #     # # =========================================================================

    #     # =====================================================================
    #     # --- Exact Autograd Eikonal Regularizer (with E_a [Q]) ---
    #     # =====================================================================

    #     # 2. Compute Expectation over Actions (Marginalizing over bins)
    #     # We define the implicit policy \pi(a|s) as a softmax over the bin Q-values.
    #     # This converts qs from [B, L, D, bins] -> expected Q [B, L, D]
    #     temperature = 1.0 # Or self.soft_alpha if you use one
    #     pi_a = torch.softmax(qs / temperature, dim=-1) 
        
    #     # E_{a ~ \pi} [Q(s, a)]
    #     expected_q_per_dim = (pi_a * qs).sum(dim=-1) # Shape: [B, L, D]
        
    #     # Sum over levels and dimensions to get total V(s)
    #     v_s = expected_q_per_dim.sum(dim=(1, 2)) # Shape: [B]

    #     # 3. Compute the Exact Spatial Gradient: \nabla_s E_a [Q(s, a)]
    #     # create_graph=True allows the Eikonal loss to backpropagate through the critic
    #     low_dim_obs = low_dim_obs.detach().clone().requires_grad_(True)
    #     # --- DIAGNOSTIC CHECKS (These will catch the bug before it crashes) ---
    #     assert low_dim_obs.requires_grad, "low_dim_obs lost its requires_grad flag!"
    #     assert q_probs.requires_grad, "The critic output lost its gradients! Check your no_grad() indentation."
    #     assert v_s.requires_grad, "v_s lost its gradients during the soft-maximum calculation!"
    #     grad_v = torch.autograd.grad(
    #         outputs=v_s.sum(), 
    #         inputs=low_dim_obs,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0] # Shape: [B, State_Dim]

    #     # 4. Compute the PDE Residual
    #     grad_norm = torch.linalg.norm(grad_v, dim=-1) # Shape: [B]

    #     # Define maximum slope based on current speed
    #     # If speed is at index -2:, uncomment the following line:
    #     # current_speed = torch.norm(low_dim_obs[:, -2:], dim=1, keepdim=False)
    #     current_speed = torch.ones((low_dim_obs.shape[0],), device=self.device)
    #     max_slope = self.kappa / (current_speed + 1e-6)

    #     # One-sided penalty
    #     slope_excess = torch.relu(grad_norm - max_slope)
    #     eikonal_loss = slope_excess.pow(2).mean()

    #     critic_loss = critic_loss + self.fk_weight * eikonal_loss

    #     # =====================================================================

    #     # optimize encoder and critic
    #     self.encoder_opt.zero_grad(set_to_none=True)
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.critic_opt.step()
    #     self.encoder_opt.step()

    #     return TensorDict(
    #         ratio_of_demos=demos.mean().detach(),
    #         q_critic_loss=q_critic_loss.detach(),
    #         bc_margin_loss=margin_loss.detach(),
    #         bc_fosd_loss=bc_fosd_loss.detach(),
    #         eikonal_loss=eikonal_loss.detach(),
    #         avg_grad_norm=grad_estimate.mean().detach(),
    #     )
    #     # optimize encoder and critic
    #     self.encoder_opt.zero_grad(set_to_none=True)
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.critic_opt.step()
    #     self.encoder_opt.step()

    #     return TensorDict(
    #         ratio_of_demos=demos.mean().detach(),
    #         q_critic_loss=q_critic_loss.detach(),
    #         bc_margin_loss=margin_loss.detach(),
    #         bc_fosd_loss=bc_fosd_loss.detach(),
    #     )
    def update_critic(
        self,
        rgb_obs,
        low_dim_obs,
        action,
        reward,
        discount,
        next_rgb_obs,
        next_low_dim_obs,
        demos,
    ):
        # =====================================================================
        # 1. AUTOGRAD PREP: Track gradients for BOTH modalities
        # Using .detach().clone() prevents altering the upstream data loader graph
        # =====================================================================
        low_dim_obs = low_dim_obs.detach().clone().requires_grad_(True)
        rgb_obs = rgb_obs.detach().clone().requires_grad_(True)

        # Target network calculations must be strictly isolated from the gradient tape
        with torch.no_grad():
            next_action = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, reward, discount
            )

        # Forward pass through the main critic
        with torch.backends.cudnn.flags(enabled=False):
            q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
                rgb_obs, low_dim_obs, action
            )

        # =====================================================================
        # --- Base CQN Losses (C51, FOSD, Margin) ---
        # =====================================================================
        
        # Cross entropy loss for C51 distributional RL
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        demos = demos.float().squeeze(1)  # [B,]

        # BC - First-order stochastic dominance loss (FOSD)
        q_probs_cdf = torch.cumsum(q_probs, -1)
        q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
        bc_fosd_loss = (
            (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
            .clamp(min=0)
            .sum(-1)
            .mean([-1, -2, -3])
        )
        bc_fosd_loss = (bc_fosd_loss * demos).sum() / (demos.sum() + 1e-6)
        critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss

        # BC - Margin loss
        support = self.critic.support.to(q_probs.device)
        
        # Collapse the 51 atoms into expected scalar Q-values for all bins
        qs = (q_probs * support.expand_as(q_probs)).sum(-1) # Shape: [B, L, D, bins]
        qs_a = (q_probs_a * support.expand_as(q_probs_a)).sum(-1) # Shape: [B, L, D]
        
        margin_loss = torch.clamp(
            self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
        ).mean([-1, -2, -3])
        margin_loss = (margin_loss * demos).sum() / (demos.sum() + 1e-6)
        critic_loss = critic_loss + self.bc_lambda * margin_loss
        
        if self.use_fk_reg:
            # =====================================================================
            # --- Stochastic Eikonal Regularization (Finite Differences) ---
            # =====================================================================
            
            nu = self.nu
            sigma = (2.0 * nu) ** 0.5
            num_walks = self.num_walks
            kappa = self.kappa
            
            # A. Define Speed Limit
            B, D_obs = low_dim_obs.shape
            current_speed = torch.ones((B,), device=low_dim_obs.device)
            # current_speed = torch.norm(low_dim_obs[:, -2:], dim=1, keepdim=False) # If velocity is at -2:
            max_slope = kappa / (current_speed + 1e-6) # [B]

            # B. Generate Neighbors (Perturb Low-Dim State)
            noise = torch.randn(B, num_walks, D_obs, device=low_dim_obs.device) * sigma
            flat_low_dim = (low_dim_obs.unsqueeze(1) + noise).view(B * num_walks, D_obs)
            
            # Expand RGB and Actions to match the batch size of the random walks
            flat_rgb = rgb_obs.unsqueeze(1).expand(-1, num_walks, *rgb_obs.shape[1:]).reshape(B * num_walks, *rgb_obs.shape[1:])
            flat_action = action.unsqueeze(1).expand(-1, num_walks, *action.shape[1:]).reshape(B * num_walks, *action.shape[1:])

            # C. Query Neighbors (Keep gradients ON so the critic weights are updated!)
            flat_q_probs, _, _, _ = self.critic(flat_rgb, flat_low_dim, flat_action)
            # flat_q_probs shape: [B*Walks, L, D_act, bins, atoms]
            
            # 1. Collapse Atoms -> Expected Q for all bins
            support = self.critic.support.to(flat_q_probs.device)
            flat_qs = (flat_q_probs * support.expand_as(flat_q_probs)).sum(dim=-1) # [B*Walks, L, D_act, bins]
            
            # 2. Marginalize over Actions (Softmax Policy)
            temperature = 1.0
            pi_a_neighbors = torch.softmax(flat_qs / temperature, dim=-1)
            expected_q_neighbors = (pi_a_neighbors * flat_qs).sum(dim=-1) # [B*Walks, L, D_act]
            
            # 3. Sum over Hierarchy to get Scalar V(s')
            v_neighbors_flat = expected_q_neighbors.sum(dim=(1, 2)) # [B*Walks]
            v_neighbors = v_neighbors_flat.view(B, num_walks) # [B, Walks]
            
            # D. Get Anchor Value V(s)
            # Reuse 'qs' from the earlier Margin Loss: [B, L, D_act, bins]
            pi_a_anchor = torch.softmax(qs / temperature, dim=-1)
            expected_q_anchor = (pi_a_anchor * qs).sum(dim=-1) # [B, L, D_act]
            v_anchor = expected_q_anchor.sum(dim=(1, 2)) # [B]
            
            # E. Compute Gradient Penalty (|V(s') - V(s)| / epsilon)
            diff = (v_neighbors - v_anchor.unsqueeze(1)).abs() # [B, Walks]
            grad_epsilon = torch.linalg.norm(noise, dim=-1) + 1e-6 # [B, Walks]
            grad_estimate = diff / grad_epsilon # [B, Walks]
            
            # F. Hinge Loss
            # Penalize only when the estimated gradient exceeds the max slope
            slope_excess = torch.relu(grad_estimate - max_slope.unsqueeze(1)) # [B, Walks]
            eikonal_loss = slope_excess.pow(2).mean()

            # Add to total loss
            critic_loss = critic_loss + self.fk_weight * eikonal_loss
            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()

            # Return a dictionary matching what the SQAR runner expects
            return dict(
                ratio_of_demos=demos.mean().detach().item(),
                q_critic_loss=q_critic_loss.detach().item(),
                bc_margin_loss=margin_loss.detach().item(),
                bc_fosd_loss=bc_fosd_loss.detach().item(),
                fk_loss=eikonal_loss.detach().item(),
            )
            # =====================================================================
            

        # =====================================================================
        # --- Exact Autograd Eikonal Regularizer (PDE Constraint) ---
        # =====================================================================
        if self.use_eikonal:

            # 2. Compute Expectation over Actions: E_{a ~ \pi} [Q(s, a)]
            # Define the implicit policy \pi(a|s) as a softmax over the bin Q-values
            temperature = 1.0 
            pi_a = torch.softmax(qs / temperature, dim=-1) 
            
            # Multiply probabilities by the bin values and sum
            expected_q_per_dim = (pi_a * qs).sum(dim=-1) # Shape: [B, L, D]
            
            # Sum over coarse-to-fine levels and action dimensions to get total V(s)
            v_s = expected_q_per_dim.sum(dim=(1, 2)) # Shape: [B]

            
            # 3. Compute the Exact Spatial Gradient with Fallback
            # allow_unused=True prevents crashes if self.critic drops one of the inputs
            grads = torch.autograd.grad(
                outputs=v_s.sum(), 
                inputs=(low_dim_obs, rgb_obs),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True 
            )

            grad_low_dim = grads[0]
            grad_rgb = grads[1]

            # 4. Gracefully handle whichever computation graph survived
            if grad_low_dim is not None:
                # Proprioception graph is intact
                grad_v = grad_low_dim
            elif grad_rgb is not None:
                # Proprioception severed, fallback to the latent visual state
                grad_v = grad_rgb
            else:
                raise RuntimeError("CRITICAL: Both low_dim and rgb_obs are disconnected from the output inside C2FCritic!")

            # Flatten if grad_v is a spatial feature map (e.g., [B, C, H, W] -> [B, N])
            if len(grad_v.shape) > 2:
                grad_v = grad_v.reshape(grad_v.shape[0], -1)

            # 5. Compute the PDE Residual
            grad_norm = torch.linalg.norm(grad_v, dim=-1) # Shape: [B]

            # Define maximum slope based on current speed
            current_speed = torch.ones((low_dim_obs.shape[0],), device=self.device)
            max_slope = self.kappa / (current_speed + 1e-6)

            # One-sided penalty: upper bound the Lipschitz constant
            slope_excess = torch.relu(grad_norm - max_slope)
            eikonal_loss = slope_excess.pow(2).mean()

            # Add to total loss
            critic_loss = critic_loss + self.fk_weight * eikonal_loss

            # =====================================================================
            # --- Optimization Step ---
            # =====================================================================
            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()

            # Return a dictionary matching what the SQAR runner expects
            return dict(
                ratio_of_demos=demos.mean().detach().item(),
                q_critic_loss=q_critic_loss.detach().item(),
                bc_margin_loss=margin_loss.detach().item(),
                bc_fosd_loss=bc_fosd_loss.detach().item(),
                eikonal_loss=eikonal_loss.detach().item(),
                avg_grad_norm=grad_norm.mean().detach().item(),
            )

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # Return a dictionary matching what the SQAR runner expects
        return dict(
            ratio_of_demos=demos.mean().detach().item(),
            q_critic_loss=q_critic_loss.detach().item(),
            bc_margin_loss=margin_loss.detach().item(),
            bc_fosd_loss=bc_fosd_loss.detach().item(),
        )
        

    def update(self, replay_iter, step):
        metrics = dict()

        # 1. Skip updates based on the update interval
        if step % self.update_every_steps != 0:
            return metrics

        # 2. Fetch and unpack the batch the SQAR way (Tuple instead of Dict)
        batch = next(replay_iter)
        (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
        ) = utils.to_torch(batch, self.device)

        # 3. Augment and Encode (Preserved from both agents)
        rgb_obs = rgb_obs.float()
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        
        next_rgb_obs = next_rgb_obs.float()
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        
        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        metrics["batch_reward"] = reward.mean().item()

        # 4. Update the critic using CQNAS's specific logic
        critic_metrics = self.update_critic(
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
        )
        metrics.update(critic_metrics)

        # 5. Handle Target Critic Updates (Bridging CQNAS logic into the SQAR loop)
        self.update_target_critic(step)

        return metrics

    def update_target_critic(self, step):
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
