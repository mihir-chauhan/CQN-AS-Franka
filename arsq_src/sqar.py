"""
sqar.py — SQAR (Soft Q Auto-Regressive) agent for real-robot training.

Adapted from ARSQ-main/rlbench/arsq_rlb/alg/sqar.py with imports adjusted
to work from the project root.

Key differences from CQN-AS:
  - No action sequences / GRU — single-step actions
  - No distributional RL (no atoms / C51) — uses soft Q-learning
  - update() handles target network updates internally
  - Batch comes as a tuple, not a dict
"""

import numpy as np
import torch
import torch.nn as nn

from arsq_src.encoder import (
    MultiViewCNNEncoder,
    RandomShiftsAug,
    weight_init,
)
from arsq_src.cqn_utils import (
    encode_action,
    decode_action,
    zoom_in,
)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def metrics_full_log(metrics: dict, key: str, value: torch.Tensor):
    metrics[f"{key}_mean"] = value.mean().item()
    metrics[f"{key}_std"] = value.std().item()
    metrics[f"{key}_min"] = value.min().item()
    metrics[f"{key}_max"] = value.max().item()
    return metrics


# ── Network components ────────────────────────────────────────────────


class FullyConnectedNetwork(nn.Module):
    def __init__(self, cfg, repr_dim: int, low_dim: int, ar_dim: int, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim

        self.rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, self.feature_dim, bias=False),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, self.feature_dim, bias=False),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )
        self.adv_net = nn.Sequential(
            nn.Linear(
                self.feature_dim * 2 + ar_dim, self.hidden_dim, bias=False
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.adv_head = nn.Linear(self.hidden_dim, self.out_dim)

        self.apply(weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)

    def forward(self, rgb_obs, low_dim_obs, ar=None):
        adv_h = torch.cat(
            [self.rgb_encoder(rgb_obs), self.low_dim_encoder(low_dim_obs)], dim=-1
        )
        if ar is not None:
            adv_h = torch.cat([adv_h, ar], dim=-1)
        adv_h = self.adv_net(adv_h)
        adv = self.adv_head(adv_h)
        return adv


class NN_mlpc_qc(nn.Module):
    def __init__(self, config, repr_dim: int, low_dim: int, actor_dim: int):
        super().__init__()
        self.cfg = config
        self.repr_dim = repr_dim
        self.low_dim = low_dim
        self.actor_dim = actor_dim

        self.levels = config.levels
        self.bins = config.bins
        self.act_alpha = config.act_alpha
        self.abl_skip = config.abl_skip

        self.max_seq_len = self.levels * self.actor_dim
        self.qchunk_size = (
            config.qchunk_size if config.qchunk_size > 0 else self.max_seq_len
        )
        self.num_qchunks = self.max_seq_len // self.qchunk_size
        assert self.max_seq_len % self.qchunk_size == 0

        self.v_network = FullyConnectedNetwork(
            self.cfg, self.repr_dim, self.low_dim, 0, 1
        )
        self.q_networks = nn.ModuleList([
            FullyConnectedNetwork(
                self.cfg, self.repr_dim, self.low_dim,
                self.actor_dim, self.qchunk_size * self.bins,
            )
            for _ in range(self.num_qchunks)
        ])

    def forward_value_soft(self, rgb_obs, low_obs):
        return self.v_network(rgb_obs, low_obs)  # [B, 1]

    def forward(self, rgb_obs, low_obs, category):
        assert category.shape[1] * category.shape[2] == self.max_seq_len

        initial_low = torch.tensor([-1.0] * self.actor_dim).to(rgb_obs.device)
        initial_low = initial_low.unsqueeze(0).repeat(rgb_obs.shape[0], 1)
        initial_high = torch.tensor([1.0] * self.actor_dim).to(rgb_obs.device)
        initial_high = initial_high.unsqueeze(0).repeat(rgb_obs.shape[0], 1)

        actions = []
        action = (initial_high + initial_low) / 2.0
        actions.append(action)
        for l in range(self.levels):
            for d in range(self.actor_dim):
                low, high = zoom_in(
                    initial_low[:, d], initial_high[:, d],
                    category[:, l, d], self.bins,
                )
                initial_low[:, d] = low
                initial_high[:, d] = high
                idx = l * self.actor_dim + d
                if self.abl_skip > 0 and idx % self.abl_skip != 0:
                    pass
                else:
                    action = (initial_high + initial_low) / 2.0
                actions.append(action)
        actions = torch.stack(actions[:-1], dim=1)  # [B, L * D, D]

        rgb_obs_repeat = rgb_obs.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        low_obs_repeat = low_obs.unsqueeze(1).repeat(1, self.max_seq_len, 1)

        value = self.v_network(rgb_obs, low_obs)  # [B, 1]

        advs = []
        for idx in range(self.num_qchunks):
            s = idx * self.qchunk_size
            e = (idx + 1) * self.qchunk_size
            rgb_i = rgb_obs_repeat[:, s:e]
            low_i = low_obs_repeat[:, s:e]
            act_i = actions[:, s:e]

            adv_i = self.q_networks[idx](rgb_i, low_i, act_i)
            adv_i = adv_i.reshape(-1, self.qchunk_size, self.qchunk_size, self.bins)

            idx_range = torch.arange(self.qchunk_size).to(rgb_obs.device)
            idx_range = idx_range.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            idx_range = idx_range.expand(rgb_obs.shape[0], -1, -1, self.bins)
            adv_i = torch.gather(adv_i, 2, idx_range).squeeze(2)

            advs.append(adv_i)

        adv = torch.cat(advs, dim=1)  # [B, L * D, bins]
        adv = adv.view(-1, self.levels, self.actor_dim, self.bins)

        return value, adv

    @classmethod
    @torch.no_grad()
    def infer(cls, rgb_obs, low_obs, nn1, nn2=None, deterministic=False):
        initial_low = torch.tensor([-1.0] * nn1.actor_dim).to(rgb_obs.device)
        initial_low = initial_low.unsqueeze(0).repeat(rgb_obs.shape[0], 1)
        initial_high = torch.tensor([1.0] * nn1.actor_dim).to(rgb_obs.device)
        initial_high = initial_high.unsqueeze(0).repeat(rgb_obs.shape[0], 1)

        res_action = []
        action = (initial_high + initial_low) / 2.0

        for l in range(nn1.levels):
            for d in range(nn1.actor_dim):
                idx = l * nn1.actor_dim + d
                if nn1.abl_skip > 0 and idx % nn1.abl_skip != 0:
                    pass
                else:
                    action = (initial_high + initial_low) / 2.0

                q_idx = (d + l * nn1.actor_dim) // nn1.qchunk_size
                offset_idx = (d + l * nn1.actor_dim) % nn1.qchunk_size

                q1 = nn1.q_networks[q_idx](rgb_obs, low_obs, action)
                q1 = q1.reshape(-1, nn1.qchunk_size, nn1.bins)
                q1 = q1[:, offset_idx]
                qs1 = q1 - nn1.act_alpha * torch.logsumexp(
                    q1 / nn1.act_alpha, dim=-1, keepdim=True
                )
                qs1 = qs1.unsqueeze(1)

                if nn2 is not None:
                    q2 = nn2.q_networks[q_idx](rgb_obs, low_obs, action)
                    q2 = q2.reshape(-1, nn2.qchunk_size, nn2.bins)
                    q2 = q2[:, offset_idx]
                    qs2 = q2 - nn2.act_alpha * torch.logsumexp(
                        q2 / nn2.act_alpha, dim=-1, keepdim=True
                    )
                    qs2 = qs2.unsqueeze(1)
                    qs = torch.minimum(qs1, qs2)
                else:
                    qs = qs1

                if deterministic:
                    action_final = qs.max(-1)[1]
                else:
                    logits = qs / nn1.act_alpha
                    action_final = torch.distributions.Categorical(
                        logits=logits
                    ).sample()

                res_action.append(action_final)

                low, high = zoom_in(
                    initial_low[:, d], initial_high[:, d],
                    action_final.squeeze(-1), nn1.bins,
                )
                initial_low[:, d] = low
                initial_high[:, d] = high

        res_action = torch.cat(res_action, 1)
        res_action = res_action.view(-1, nn1.levels, nn1.actor_dim)
        return res_action


# ── Main agent ────────────────────────────────────────────────────────


class SQARAgent:
    """SQAR agent matching the CQN-AS interface for real-robot training.

    Key API:
        act(rgb_obs, low_dim_obs, step, eval_mode) -> np.ndarray (action_dim,)
        update(batch_tuple, step) -> dict   (batch is a tuple of numpy arrays)
        train(training=True)
    """

    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        lr,
        weight_decay,
        feature_dim,
        hidden_dim,
        levels,
        bins,
        soft_alpha=0.001,
        act_alpha=None,
        qchunk_size=-1,
        abl_skip=-1,
        bellman_loss_coef=0.1,
        cql_type="margin",
        cql_min_q_weight=1.0,
        cql_clip_diff_min=-0.01,
        cql_clip_diff_max=10_000_000,
        cql_temp=1.0,
        critic_target_tau=0.02,
        update_every_steps=1,
        num_expl_steps=0,
        use_logger=True,
    ):
        self.device = device
        self.action_dim = action_shape[0]
        self.use_logger = use_logger

        self.soft_alpha = soft_alpha
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.levels = levels
        self.bins = bins

        # Build a simple namespace for config params needed by NN_mlpc_qc
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.feature_dim = feature_dim
        cfg.hidden_dim = hidden_dim
        cfg.levels = levels
        cfg.bins = bins
        cfg.soft_alpha = soft_alpha
        cfg.act_alpha = act_alpha if act_alpha is not None else soft_alpha
        cfg.qchunk_size = qchunk_size
        cfg.abl_skip = abl_skip
        # Store loss config as dict for update_critic
        self._loss_cfg = {
            "bellman_loss_coef": bellman_loss_coef,
            "cql_type": cql_type,
            "cql_min_q_weight": cql_min_q_weight,
            "cql_clip_diff_min": cql_clip_diff_min,
            "cql_clip_diff_max": cql_clip_diff_max,
            "cql_temp": cql_temp,
        }

        # Models
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)

        self.qf1 = NN_mlpc_qc(
            cfg, self.encoder.repr_dim, low_dim_obs_shape[-1], action_shape[0]
        ).to(device)
        self.qf2 = NN_mlpc_qc(
            cfg, self.encoder.repr_dim, low_dim_obs_shape[-1], action_shape[0]
        ).to(device)

        self.qf1_target = NN_mlpc_qc(
            cfg, self.encoder.repr_dim, low_dim_obs_shape[-1], action_shape[0]
        ).to(device)
        self.qf2_target = NN_mlpc_qc(
            cfg, self.encoder.repr_dim, low_dim_obs_shape[-1], action_shape[0]
        ).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.critic_opt = torch.optim.AdamW(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        # Data augmentation
        self.aug = RandomShiftsAug(pad=4)

        # Fixed action bounds
        self.low = torch.tensor(
            [-1.0] * action_shape[0], requires_grad=False
        ).to(device)
        self.high = torch.tensor(
            [1.0] * action_shape[0], requires_grad=False
        ).to(device)

        self.training = True
        self.train()
        self.qf1_target.eval()
        self.qf2_target.eval()

        print(self.encoder)
        print(self.qf1)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.qf1.train(training)
        self.qf2.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        """Select action. Returns np.ndarray of shape (action_dim,)."""
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs_rep = self.encoder(rgb_obs)

        nn1 = self.qf1_target
        nn2 = self.qf2_target
        action_d = nn1.infer(
            rgb_obs_rep, low_dim_obs, nn1, nn2, deterministic=eval_mode
        )
        action_c = decode_action(
            action_d, self.low, self.high, self.levels, self.bins
        )
        return action_c.cpu().numpy()[0]

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
        metrics = dict()
        cfg = self._loss_cfg

        actions_d = encode_action(
            action, self.low, self.high, self.levels, self.bins
        ).long()
        actions_d = actions_d.unsqueeze(-1)  # [B, L, D, 1]

        idx_temp = (
            torch.ones_like(actions_d)
            .repeat(1, 1, 1, self.bins)
            .scatter(-1, actions_d, 0)
        )
        actions_d_i = torch.nonzero(idx_temp)
        actions_d_i = actions_d_i[:, -1].reshape(
            -1, self.levels, self.action_dim, self.bins - 1
        )

        # Q loss
        qf1_loss = 0
        qf2_loss = 0

        v1_pred, ad1_pred = self.qf1.forward(
            rgb_obs, low_dim_obs, actions_d.squeeze(-1)
        )
        v2_pred, ad2_pred = self.qf2.forward(
            rgb_obs, low_dim_obs, actions_d.squeeze(-1)
        )

        adv1_pred = ad1_pred - self.soft_alpha * torch.logsumexp(
            ad1_pred / self.soft_alpha, dim=-1, keepdim=True
        )
        adv2_pred = ad2_pred - self.soft_alpha * torch.logsumexp(
            ad2_pred / self.soft_alpha, dim=-1, keepdim=True
        )
        adv1a = adv1_pred.gather(-1, actions_d)
        adv2a = adv2_pred.gather(-1, actions_d)
        q1a_pred = v1_pred + adv1a.sum(1).sum(1)
        q2a_pred = v2_pred + adv2a.sum(1).sum(1)

        if cfg["bellman_loss_coef"] > 0.0:
            with torch.no_grad():
                next_v1_t = self.qf1_target.forward_value_soft(
                    next_rgb_obs, next_low_dim_obs
                )
                next_v2_t = self.qf2_target.forward_value_soft(
                    next_rgb_obs, next_low_dim_obs
                )
                next_v_t = torch.minimum(next_v1_t, next_v2_t)
                td_target = reward + discount * next_v_t

            qf1_bellman_loss = torch.mean((q1a_pred - td_target) ** 2)
            qf2_bellman_loss = torch.mean((q2a_pred - td_target) ** 2)

            metrics["qf1_bellman_loss"] = qf1_bellman_loss.item()
            metrics["qf2_bellman_loss"] = qf2_bellman_loss.item()

            qf1_loss = qf1_bellman_loss * cfg["bellman_loss_coef"]
            qf2_loss = qf2_bellman_loss * cfg["bellman_loss_coef"]

        # CQL
        demos_sum = torch.sum(demos)
        if cfg["cql_min_q_weight"] > 0.0 and demos_sum > 0:
            cql_adv1_other = adv1_pred.gather(-1, actions_d_i)
            cql_adv2_other = adv2_pred.gather(-1, actions_d_i)

            if cfg["cql_type"] == "cql":
                cql_temp = cfg["cql_temp"]
                cql_adv1_ood = (
                    torch.logsumexp(cql_adv1_other / cql_temp, dim=-1, keepdim=True)
                    * cql_temp
                )
                cql_adv2_ood = (
                    torch.logsumexp(cql_adv2_other / cql_temp, dim=-1, keepdim=True)
                    * cql_temp
                )
                cql_adv1_diff = torch.clamp(
                    cql_adv1_ood - adv1a,
                    cfg["cql_clip_diff_min"], cfg["cql_clip_diff_max"],
                ).sum(dim=1).sum(dim=1)
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(
                    cql_adv2_ood - adv2a,
                    cfg["cql_clip_diff_min"], cfg["cql_clip_diff_max"],
                ).sum(dim=1).sum(dim=1)
                cql_adv2_diff = (cql_adv2_diff * demos).mean()
            elif cfg["cql_type"] == "margin":
                cql_adv1_diff = torch.clamp(
                    cql_adv1_other - adv1a,
                    cfg["cql_clip_diff_min"], cfg["cql_clip_diff_max"],
                ).sum(dim=1).sum(dim=1)
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(
                    cql_adv2_other - adv2a,
                    cfg["cql_clip_diff_min"], cfg["cql_clip_diff_max"],
                ).sum(dim=1).sum(dim=1)
                cql_adv2_diff = (cql_adv2_diff * demos).mean()
            else:
                raise NotImplementedError(f"Unknown cql_type: {cfg['cql_type']}")

            metrics["cql_qf1_loss"] = cql_adv1_diff.item()
            metrics["cql_qf2_loss"] = cql_adv2_diff.item()

            qf1_loss += cql_adv1_diff * cfg["cql_min_q_weight"]
            qf2_loss += cql_adv2_diff * cfg["cql_min_q_weight"]

        metrics["qf1_loss"] = (qf1_loss.mean().item()
                               if torch.is_tensor(qf1_loss) else qf1_loss)
        metrics["qf2_loss"] = (qf2_loss.mean().item()
                               if torch.is_tensor(qf2_loss) else qf2_loss)

        critic_loss = qf1_loss + qf2_loss

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, batch, step):
        """Update from a batch tuple (matches ARSQ API).

        batch: tuple of numpy arrays
            (rgb_obs, low_dim_obs, action, reward, discount,
             next_rgb_obs, next_low_dim_obs, demos)

        Returns: dict of metrics
        """
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        # Convert to torch
        (
            rgb_obs, low_dim_obs, action, reward, discount,
            next_rgb_obs, next_low_dim_obs, demos,
        ) = tuple(torch.as_tensor(x, device=self.device) for x in batch)

        # Encode next obs
        next_rgb_obs = next_rgb_obs.float()
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        # Encode current obs
        rgb_obs = rgb_obs.float()
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        rgb_obs = self.encoder(rgb_obs)

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        # Update critic
        metrics.update(
            self.update_critic(
                rgb_obs, low_dim_obs, action, reward, discount,
                next_rgb_obs, next_low_dim_obs, demos,
            )
        )

        # Update target networks
        soft_update_params(self.qf1, self.qf1_target, self.critic_target_tau)
        soft_update_params(self.qf2, self.qf2_target, self.critic_target_tau)

        return metrics
