import numpy as np
import torch
import torch.nn as nn

import arsq_rlb.util.utils as utils
from arsq_rlb.alg.cqn import RandomShiftsAug, MultiViewCNNEncoder
from arsq_rlb.alg.cqn_utils import (
    encode_action,
    decode_action,
    zoom_in
)
from arsq_rlb.util.utils import metrics_full_log


class FullyConnectedNetwork(nn.Module):
    def __init__(self, cfg, repr_dim: int, low_dim: int, ar_dim: int, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim

        # Advantage stream in Dueling network
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

        self.apply(utils.weight_init)
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

        # qchunk_size
        self.max_seq_len = self.levels * self.actor_dim
        self.qchunk_size = config.qchunk_size if config.qchunk_size > 0 else self.max_seq_len
        self.num_qchunks = self.max_seq_len // self.qchunk_size
        assert self.max_seq_len % self.qchunk_size == 0

        self.v_network = FullyConnectedNetwork(
            self.cfg, self.repr_dim, self.low_dim, 0, 1
        )
        self.q_networks = nn.ModuleList([
            FullyConnectedNetwork(self.cfg, self.repr_dim, self.low_dim, self.actor_dim, self.qchunk_size * self.bins)
            for _ in range(self.num_qchunks)
        ])

    def forward_value_soft(self, rgb_obs: torch.Tensor, low_obs: torch.Tensor):
        """
        Args:
        - rgb_obs: observations [B, repr_dim]
        - low_obs: low-dim observations [B, low_dim]

        Return:
        - value: (B, 1)
        """
        value = self.v_network(rgb_obs, low_obs)  # [B, 1]

        return value

    def forward(self, rgb_obs: torch.Tensor, low_obs: torch.Tensor, category: torch.Tensor):
        """
        Args:
        - rgb_obs: observations [B, repr_dim]
        - low_obs: low-dim observations [B, low_dim]
        - category: actions [B, L, D]

        Return:
        - value: (B, 1)
        - adv: (B, L, D, bins)
        """
        assert category.shape[1] * category.shape[2] == self.max_seq_len

        initial_low = torch.tensor([-1.0] * self.actor_dim).to(rgb_obs.device)  # [D]
        initial_low = initial_low.unsqueeze(0).repeat(rgb_obs.shape[0], 1)  # [B, D]
        initial_high = torch.tensor([1.0] * self.actor_dim).to(rgb_obs.device)  # [D]
        initial_high = initial_high.unsqueeze(0).repeat(rgb_obs.shape[0], 1)  # [B, D]

        actions = []

        action = (initial_high + initial_low) / 2.0  # [B, D]
        actions.append(action)
        for l in range(self.levels):
            for d in range(self.actor_dim):
                low, high = zoom_in(initial_low[:, d], initial_high[:, d], category[:, l, d], self.bins)  # [B]
                initial_low[:, d] = low
                initial_high[:, d] = high

                idx = l * self.actor_dim + d
                if self.abl_skip > 0 and idx % self.abl_skip != 0:
                    pass
                else:
                    action = (initial_high + initial_low) / 2.0  # [B, D]
                actions.append(action)
        actions = torch.stack(actions[:-1], dim=1)  # [B, L * D, D]

        # process obs
        rgb_obs_repeat = rgb_obs.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # [B, L * D, obs_dim]
        low_obs_repeat = low_obs.unsqueeze(1).repeat(1, self.max_seq_len, 1)

        # forward value
        value = self.v_network(rgb_obs, low_obs)  # [B, 1]

        # forward q
        advs = []
        for idx in range(self.num_qchunks):
            rgb_i = rgb_obs_repeat[:, idx * self.qchunk_size: (idx + 1) * self.qchunk_size]  # [B, qchunk, obs_dim]
            low_i = low_obs_repeat[:, idx * self.qchunk_size: (idx + 1) * self.qchunk_size]  # [B, qchunk, low_dim]
            act_i = actions[:, idx * self.qchunk_size: (idx + 1) * self.qchunk_size]  # [B, qchunk, D]

            adv_i = self.q_networks[idx](rgb_i, low_i, act_i)  # [B, qchunk, qchunk * bins]
            adv_i = adv_i.reshape(-1, self.qchunk_size, self.qchunk_size, self.bins)  # [B, qchunk, qchunk, bins]

            idx_range = torch.arange(self.qchunk_size).to(rgb_obs.device)  # [qchunk]
            idx_range = idx_range.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, qchunk, 1, 1]
            idx_range = idx_range.expand(rgb_obs.shape[0], -1, -1, self.bins)  # [B, qchunk, 1, bins]
            adv_i = torch.gather(adv_i, 2, idx_range)  # [B, qchunk, 1, bins]
            adv_i = adv_i.squeeze(2)  # [B, qchunk, bins]

            advs.append(adv_i)

        adv = torch.cat(advs, dim=1)  # [B, L * D, bins]
        adv = adv.view(-1, self.levels, self.actor_dim, self.bins)  # [B, L, D, bins]

        return value, adv

    @classmethod
    @torch.no_grad()
    def infer(cls, rgb_obs: torch.Tensor, low_obs: torch.Tensor, nn1, nn2=None, deterministic=False):
        """
        Args:
        - rgb_obs: observations [B, repr_dim]
        - low_obs: low-dim observations [B, low_dim]

        Return:
        - res_action: (B, L, D)
        """

        initial_low = torch.tensor([-1.0] * nn1.actor_dim).to(rgb_obs.device)  # [D]
        initial_low = initial_low.unsqueeze(0).repeat(rgb_obs.shape[0], 1)  # [B, D]
        initial_high = torch.tensor([1.0] * nn1.actor_dim).to(rgb_obs.device)  # [D]
        initial_high = initial_high.unsqueeze(0).repeat(rgb_obs.shape[0], 1)  # [B, D]

        # result
        res_action = []
        action = (initial_high + initial_low) / 2.0

        for l in range(nn1.levels):
            for d in range(nn1.actor_dim):
                # process action
                idx = l * nn1.actor_dim + d
                if nn1.abl_skip > 0 and idx % nn1.abl_skip != 0:
                    pass
                else:
                    action = (initial_high + initial_low) / 2.0  # [B, D]

                # forward
                q_idx = (d + l * nn1.actor_dim) // nn1.qchunk_size
                offset_idx = (d + l * nn1.actor_dim) % nn1.qchunk_size

                q1 = nn1.q_networks[q_idx](rgb_obs, low_obs, action)  # [B, qchunk * bins]
                q1 = q1.reshape(-1, nn1.qchunk_size, nn1.bins)  # [B, qchunk, bins]
                q1 = q1[:, offset_idx]  # [B, bins]
                qs1 = q1 - nn1.act_alpha * torch.logsumexp(q1 / nn1.act_alpha, dim=-1, keepdim=True)  # [B, bins]
                qs1 = qs1.unsqueeze(1)  # [B, 1, bins]

                if nn2 is not None:
                    q2 = nn2.q_networks[q_idx](rgb_obs, low_obs, action)  # [B, qchunk * bins]
                    q2 = q2.reshape(-1, nn2.qchunk_size, nn2.bins)  # [B, qchunk, bins]
                    q2 = q2[:, offset_idx]
                    qs2 = q2 - nn2.act_alpha * torch.logsumexp(q2 / nn2.act_alpha, dim=-1, keepdim=True)  # [B, bins]
                    qs2 = qs2.unsqueeze(1)  # [B, 1, bins]

                    qs = torch.minimum(qs1, qs2)  # [B, 1, bins]
                else:
                    qs = qs1

                # soft action selection
                if deterministic:
                    action_final = qs.max(-1)[1]  # [B, 1]
                else:
                    logits = qs / nn1.act_alpha
                    action_final = torch.distributions.Categorical(logits=logits).sample()  # [B, 1]

                # log
                res_action.append(action_final)

                # new round
                low, high = zoom_in(initial_low[:, d], initial_high[:, d], action_final.squeeze(-1), nn1.bins)  # [B]
                initial_low[:, d] = low
                initial_high[:, d] = high

        res_action = torch.cat(res_action, 1)  # [B, L * D]
        res_action = res_action.view(-1, nn1.levels, nn1.actor_dim)  # [B, L, D]

        return res_action


class SQARAgent:
    def __init__(self, cfg, rgb_obs_shape, low_obs_shape, action_shape, use_logger):
        self.cfg = cfg
        self.action_dim = action_shape[0]
        self.use_logger = use_logger

        self.soft_alpha = cfg.soft_alpha

        self.device = cfg.device
        self.critic_target_tau = cfg.critic_target_tau
        self.update_every_steps = cfg.update_every_steps
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.levels = cfg.levels
        self.bins = cfg.bins

        # models
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(self.device)

        NN = NN_mlpc_qc

        self.qf1 = NN(cfg, self.encoder.repr_dim, low_obs_shape[-1], action_shape[0]).to(self.device)
        self.qf2 = NN(cfg, self.encoder.repr_dim, low_obs_shape[-1], action_shape[0]).to(self.device)

        self.qf1_target = NN(cfg, self.encoder.repr_dim, low_obs_shape[-1], action_shape[0]).to(self.device)
        self.qf2_target = NN(cfg, self.encoder.repr_dim, low_obs_shape[-1], action_shape[0]).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        # fix params
        self.low = torch.tensor([-1.0] * action_shape[0], requires_grad=False).to(self.device)  # [D]
        self.high = torch.tensor([1.0] * action_shape[0], requires_grad=False).to(self.device)  # [D]

        # mode
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
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs_rep = self.encoder(rgb_obs)

        nn1 = self.qf1_target
        nn2 = self.qf2_target
        action_d = nn1.infer(rgb_obs_rep, low_dim_obs, nn1, nn2, deterministic=eval_mode)  # [1, L, D]

        action_c = decode_action(action_d, self.low, self.high, self.levels, self.bins)  # [B, D]

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

        assert len(low_dim_obs.shape) == 2
        assert len(action.shape) == 2
        assert len(reward.shape) == 2
        assert len(discount.shape) == 2
        assert len(demos.shape) == 2

        actions_d = encode_action(action, self.low, self.high, self.levels, self.bins).long()  # [B, L, D]
        actions_d = actions_d.unsqueeze(-1)  # [B, L, D, 1]
        # data in action_dim is bin idx

        idx_temp = torch.ones_like(actions_d).repeat(1, 1, 1, self.bins).scatter(-1, actions_d, 0)  # [B, L, D, bins]
        actions_d_i = torch.nonzero(idx_temp)  # [B * L * D * (bins - 1), 4]
        actions_d_i = actions_d_i[:, -1].reshape(-1, self.levels, self.action_dim, self.bins - 1)  # [B, L, D, bins - 1]

        # check: (0, ..., bins - 1) all in (actions_d_i | actions_d)
        assert actions_d_i.shape == (actions_d.shape[0], actions_d.shape[1], actions_d.shape[2], self.bins - 1)
        assert actions_d_i[0, 0, 0].sum().item() + actions_d[0, 0, 0].item() == self.bins * (self.bins - 1) // 2

        # Q function loss
        qf1_loss = 0
        qf2_loss = 0

        # Q function loss
        v1_pred, ad1_pred = self.qf1.forward(rgb_obs, low_dim_obs, actions_d.squeeze(-1))  # [B, 1], [B, L, D, bins]
        v2_pred, ad2_pred = self.qf2.forward(rgb_obs, low_dim_obs, actions_d.squeeze(-1))  # [B, 1], [B, L, D, bins]
        metrics_full_log(metrics, 'v1_pred', v1_pred)
        metrics_full_log(metrics, 'v2_pred', v2_pred)

        adv1_pred = ad1_pred - self.soft_alpha * torch.logsumexp(ad1_pred / self.soft_alpha, dim=-1, keepdim=True)
        adv2_pred = ad2_pred - self.soft_alpha * torch.logsumexp(ad2_pred / self.soft_alpha, dim=-1, keepdim=True)
        adv1a = adv1_pred.gather(-1, actions_d)  # [B, L, D, 1]
        adv2a = adv2_pred.gather(-1, actions_d)  # [B, L, D, 1]
        q1a_pred = v1_pred + adv1a.sum(1).sum(1)  # [B, 1]
        q2a_pred = v2_pred + adv2a.sum(1).sum(1)  # [B, 1]
        metrics_full_log(metrics, 'q1_pred', q1a_pred)
        metrics_full_log(metrics, 'q2_pred', q2a_pred)

        if self.cfg['bellman_loss_coef'] > 0.0:
            with torch.no_grad():
                adv1i = adv1_pred.gather(-1, actions_d_i)  # [B, L, D, bins - 1]
                adv2i = adv2_pred.gather(-1, actions_d_i)  # [B, L, D, bins - 1]
                metrics["adv1_0_mean"] = adv1_pred[:, 0, 0].mean().item()
                metrics["adv1_0_std"] = adv1_pred[:, 0, 0].std(dim=-1).mean().item()
                metrics["adv1a_0_mean"] = adv1a[:, 0, 0].mean().item()
                metrics["adv1i_0_mean"] = adv1i[:, 0, 0].mean().item()
                metrics["adv1i_0_std"] = adv1i[:, 0, 0].std(dim=-1).mean().item()

                metrics["adv1_n1_mean"] = adv1_pred[:, -1, -1].mean().item()
                metrics["adv1_n1_std"] = adv1_pred[:, -1, -1].std(dim=-1).mean().item()
                metrics["adv1a_n1_mean"] = adv1a[:, -1, -1].mean().item()
                metrics["adv1i_n1_mean"] = adv1i[:, -1, -1].mean().item()
                metrics["adv1i_n1_std"] = adv1i[:, -1, -1].std(dim=-1).mean().item()

                metrics["adv2_0_mean"] = adv2_pred[:, 0, 0].mean().item()
                metrics["adv2_0_std"] = adv2_pred[:, 0, 0].std(dim=-1).mean().item()
                metrics["adv2a_0_mean"] = adv2a[:, 0, 0].mean().item()
                metrics["adv2i_0_mean"] = adv2i[:, 0, 0].mean().item()
                metrics["adv2i_0_std"] = adv2i[:, 0, 0].std(dim=-1).mean().item()

                metrics["adv2_n1_mean"] = adv2_pred[:, -1, -1].mean().item()
                metrics["adv2_n1_std"] = adv2_pred[:, -1, -1].std(dim=-1).mean().item()
                metrics["adv2a_n1_mean"] = adv2a[:, -1, -1].mean().item()
                metrics["adv2i_n1_mean"] = adv2i[:, -1, -1].mean().item()
                metrics["adv2i_n1_std"] = adv2i[:, -1, -1].std(dim=-1).mean().item()

                next_v1_t = self.qf1_target.forward_value_soft(next_rgb_obs, next_low_dim_obs)  # [B, 1]
                next_v2_t = self.qf2_target.forward_value_soft(next_rgb_obs, next_low_dim_obs)  # [B, 1]
                next_v_t = torch.minimum(next_v1_t, next_v2_t)  # [B, 1]

                td_target = reward + discount * next_v_t  # [B, 1]
                assert q1a_pred.shape == td_target.shape

                metrics_full_log(metrics, 'target_q_values', next_v_t)
                metrics_full_log(metrics, 'td_target', td_target)

            qf1_bellman_loss = torch.mean((q1a_pred - td_target) ** 2)
            qf2_bellman_loss = torch.mean((q2a_pred - td_target) ** 2)

            metrics['qf1_bellman_diff_v'] = torch.mean(q1a_pred - td_target).item()
            metrics['qf2_bellman_diff_v'] = torch.mean(q2a_pred - td_target).item()

            metrics['qf1_bellman_loss'] = qf1_bellman_loss.item()
            metrics['qf2_bellman_loss'] = qf2_bellman_loss.item()

            qf1_loss = qf1_bellman_loss * self.cfg['bellman_loss_coef']
            qf2_loss = qf2_bellman_loss * self.cfg['bellman_loss_coef']

        # CQL
        demos_sum = torch.sum(demos)
        if self.cfg["cql_min_q_weight"] > 0.0 and demos_sum > 0:
            cql_adv1_other = adv1_pred.gather(-1, actions_d_i)  # [B, L, D, bins - 1]
            cql_adv2_other = adv2_pred.gather(-1, actions_d_i)  # [B, L, D, bins - 1]

            metrics_full_log(metrics, 'cql/cql_cat_q1', cql_adv1_other.std(dim=-1))
            metrics_full_log(metrics, 'cql/cql_cat_q2', cql_adv2_other.std(dim=-1))

            assert cql_adv1_other.shape[-1] == self.bins - 1

            if self.cfg["cql_type"] == "cql":
                cql_temp = self.cfg['cql_temp']
                cql_adv1_ood = torch.logsumexp(cql_adv1_other / cql_temp, dim=-1,
                                               keepdim=True) * cql_temp  # [B, L, D, 1]
                cql_adv2_ood = torch.logsumexp(cql_adv2_other / cql_temp, dim=-1,
                                               keepdim=True) * cql_temp  # [B, L, D, 1]

                cql_adv1_diff = torch.clamp(cql_adv1_ood - adv1a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  # [B, 1]
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(cql_adv2_ood - adv2a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  # [B, 1]
                cql_adv2_diff = (cql_adv2_diff * demos).mean()
            elif self.cfg["cql_type"] == "margin":
                cql_adv1_diff = torch.clamp(cql_adv1_other - adv1a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  # [B, 1]
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(cql_adv2_other - adv2a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)
                cql_adv2_diff = (cql_adv2_diff * demos).mean()
            else:
                raise NotImplementedError

            metrics["cql/cql_qf1_loss"] = cql_adv1_diff.item()
            metrics["cql/cql_qf2_loss"] = cql_adv2_diff.item()

            qf1_loss += cql_adv1_diff * self.cfg["cql_min_q_weight"]
            qf2_loss += cql_adv2_diff * self.cfg["cql_min_q_weight"]

        metrics['qf1_loss'] = qf1_loss.mean().item()
        metrics['qf2_loss'] = qf2_loss.mean().item()

        # optimize encoder and critic
        critic_loss = qf1_loss + qf2_loss

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

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

        # encode
        next_rgb_obs = next_rgb_obs.float()
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        rgb_obs = rgb_obs.float()
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        rgb_obs = self.encoder(rgb_obs)

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                rgb_obs,
                low_dim_obs,
                action,
                reward,
                discount,
                next_rgb_obs,
                next_low_dim_obs,
                demos,
            )
        )

        # update critic target
        utils.soft_update_params(self.qf1, self.qf1_target, self.critic_target_tau)
        utils.soft_update_params(self.qf2, self.qf2_target, self.critic_target_tau)

        return metrics
