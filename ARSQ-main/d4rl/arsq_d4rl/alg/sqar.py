import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arsq_d4rl.alg.cqn_utils import encode_action, decode_action, zoom_in, weight_init
from arsq_d4rl.util.util import metrics_full_log


class FCNetNaive(nn.Module):
    def __init__(self, repr_dim: int, ar_dim: int, out_dim: int):
        super().__init__()

        # hyperparam
        arch = "512-512-512"
        activation = "tanh"
        orthogonal_init = True

        hidden_sizes = [int(h) for h in arch.split("-")]

        if activation == "relu":
            activation_func = nn.ReLU
        elif activation == "tanh":
            activation_func = nn.Tanh
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        self.layers = nn.ModuleList()
        last_dim = None

        for h in hidden_sizes:
            self.layers.append(nn.Linear(last_dim or (repr_dim + ar_dim), h))
            self.layers.append(activation_func())
            last_dim = h

        self.output = nn.Linear(last_dim, out_dim)

        # orthogonal initialization
        if orthogonal_init:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias.data)
            nn.init.orthogonal_(self.output.weight.data, gain=0.01)
            nn.init.zeros_(self.output.bias.data)
        else:
            nn.init.xavier_uniform_(self.output.weight.data)
            nn.init.zeros_(self.output.bias.data)

    def forward(self, obs, ar=None):
        if ar is not None:
            obs = torch.cat([obs, ar], dim=-1)

        for layer in self.layers:
            obs = layer(obs)
        obs = self.output(obs)  # [B, (repeat), output_dim]
        return obs


class NN_mlpc_qc(nn.Module):
    def __init__(self, config, observation_dim: int, action_dim: int):
        super().__init__()
        self.cfg = config
        self.obs_dim = observation_dim
        self.actor_dim = action_dim
        self.levels = config.levels
        self.bins = config.bins
        self.act_alpha = config.act_alpha
        self.use_nbias = config.use_nbias
        self.abl_skip = config.abl_skip

        # qchunk_size
        self.max_seq_len = self.levels * self.actor_dim
        self.qchunk_size = config.qchunk_size if config.qchunk_size > 0 else self.max_seq_len
        self.num_qchunks = self.max_seq_len // self.qchunk_size
        assert self.max_seq_len % self.qchunk_size == 0

        nn_class = FCNetNaive
        self.v_network = nn_class(self.obs_dim, 0, 1)
        self.q_networks = nn.ModuleList([
            nn_class(self.obs_dim, self.actor_dim, self.bins * self.qchunk_size)
            for _ in range(self.num_qchunks)
        ])

        if self.use_nbias:
            self.nbias = nn_class(self.obs_dim, 0, 1)

    def forward_value_soft(self, obs: torch.Tensor):
        """
        Args:
        - obs: observations [B, obs_dim]

        Return:
        - value: (B, 1)
        """
        value = self.v_network(obs)  # [B, 1]

        if self.use_nbias:
            nbias = self.nbias(obs).sum(dim=-1, keepdim=True)  # [B, 1]
            value = value + nbias  # [B, 1]

        return value

    def forward_value(self, obs: torch.Tensor):
        """
        Args:
        - obs: observations [B, obs_dim]

        Return:
        - value: (B, 1)
        """
        value = self.v_network(obs)  # [B, 1]

        return value

    def forward(self, obs: torch.Tensor, category: torch.Tensor):
        """
        Args:
        - obs: observations [B, obs_dim]
        - category: actions [B, L, D]

        Return:
        - value: (B, 1)
        - adv: (B, L, D, bins)
        - nbias: (B, L, D, 1)
        """
        assert category.shape[1] * category.shape[2] == self.max_seq_len

        initial_low = torch.tensor([-1.0] * self.actor_dim).to(obs.device)  # [D]
        initial_low = initial_low.unsqueeze(0).repeat(obs.shape[0], 1)  # [B, D]
        initial_high = torch.tensor([1.0] * self.actor_dim).to(obs.device)  # [D]
        initial_high = initial_high.unsqueeze(0).repeat(obs.shape[0], 1)  # [B, D]

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
        obs_repeat = obs.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # [B, L * D, obs_dim]

        # forward value
        value = self.v_network(obs)  # [B, 1]

        if self.use_nbias:
            nbias = self.nbias(obs).unsqueeze(-1)  # [B, L * D, 1]
        else:
            nbias = torch.zeros(obs.shape[0], self.max_seq_len, 1, dtype=torch.float32).to(obs.device)  # [B, L * D, 1]

        # forward q
        advs = []
        for idx in range(self.num_qchunks):
            obs_i = obs_repeat[:, idx * self.qchunk_size: (idx + 1) * self.qchunk_size]  # [B, qchunk, obs_dim]
            act_i = actions[:, idx * self.qchunk_size: (idx + 1) * self.qchunk_size]  # [B, qchunk, D]
            adv_i = self.q_networks[idx](obs_i, ar=act_i)  # [B, qchunk, qchunk * bins]
            adv_i = adv_i.reshape(-1, self.qchunk_size, self.qchunk_size, self.bins)  # [B, qchunk, qchunk, bins]

            idx_range = torch.arange(self.qchunk_size).to(obs.device)  # [qchunk]
            idx_range = idx_range.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, qchunk, 1, 1]
            idx_range = idx_range.expand(obs.shape[0], -1, -1, self.bins)  # [B, qchunk, 1, bins]
            adv_i = torch.gather(adv_i, 2, idx_range)  # [B, qchunk, 1, bins]
            adv_i = adv_i.squeeze(2)  # [B, qchunk, bins]

            advs.append(adv_i)

        adv = torch.cat(advs, dim=1)  # [B, L * D, bins]

        adv = adv.view(-1, self.levels, self.actor_dim, self.bins)  # [B, L, D, bins]
        nbias = nbias.view(-1, self.levels, self.actor_dim, 1)  # [B, L, D, 1]

        return value, adv, nbias

    @classmethod
    @torch.no_grad()
    def infer(cls, obs: torch.Tensor, nn1, nn2=None, deterministic=False):
        """
        Args:
        - obs: observations [B, obs_dim]

        Return:
        - res_action: (B, L, D)
        """

        initial_low = torch.tensor([-1.0] * nn1.actor_dim).to(obs.device)  # [D]
        initial_low = initial_low.unsqueeze(0).repeat(obs.shape[0], 1)  # [B, D]
        initial_high = torch.tensor([1.0] * nn1.actor_dim).to(obs.device)  # [D]
        initial_high = initial_high.unsqueeze(0).repeat(obs.shape[0], 1)  # [B, D]

        # result
        res_action = []
        action = None

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

                q1 = nn1.q_networks[q_idx](obs, ar=action)  # [B, qchunk * bins]
                q1 = q1.reshape(-1, nn1.qchunk_size, nn1.bins)  # [B, qchunk, bins]
                q1 = q1[:, offset_idx]  # [B, bins]
                qs1 = q1 - nn1.act_alpha * torch.logsumexp(q1 / nn1.act_alpha, dim=-1, keepdim=True)  # [B, bins]
                qs1 = qs1.unsqueeze(1)  # [B, 1, bins]

                if nn2 is not None:
                    q2 = nn2.q_networks[q_idx](obs, ar=action)  # [B, qchunk * bins]
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

        res_action = torch.cat(res_action, 1)  # [B, level * D]

        res_action = res_action.view(-1, nn1.levels, nn1.actor_dim)  # [B, L, D]

        return res_action



class SQAR:
    def __init__(self, config, observation_dim, action_dim):
        self.cfg = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.levels = config.levels
        self.bins = config.bins
        self.soft_alpha = config.soft_alpha
        self.act_alpha = config.act_alpha

        # device: policy.device
        self.device = torch.device("cuda")

        NN = NN_mlpc_qc
        self.qf1 = NN(config, observation_dim, action_dim).to(self.device)
        self.qf2 = NN(config, observation_dim, action_dim).to(self.device)

        # Initialize optimizers
        optimizer_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
        }[self.cfg['optimizer_type']]

        qf1_params = [
            {'params': self.qf1.v_network.parameters(), 'lr': self.cfg['vf_lr']},
            {'params': self.qf1.q_networks.parameters(), 'lr': self.cfg['qf_lr']},
        ]
        qf2_params = [
            {'params': self.qf2.v_network.parameters(), 'lr': self.cfg['vf_lr']},
            {'params': self.qf2.q_networks.parameters(), 'lr': self.cfg['qf_lr']},
        ]

        if self.cfg['use_nbias']:
            qf1_params.append({'params': self.qf1.nbias.parameters(), 'lr': self.cfg['vf_lr']})
            qf2_params.append({'params': self.qf2.nbias.parameters(), 'lr': self.cfg['vf_lr']})

        self.qf1_optimizer = optimizer_class(qf1_params)
        self.qf2_optimizer = optimizer_class(qf2_params)

        # Initialize target networks
        self.target_qf1 = deepcopy(self.qf1)
        self.target_qf2 = deepcopy(self.qf2)
        self.target_qf1.train(False)
        self.target_qf2.train(False)

        # cqn low high
        self.low = torch.tensor([-1.0] * self.action_dim, requires_grad=False).to(self.device)  # [D]
        self.high = torch.tensor([1.0] * self.action_dim, requires_grad=False).to(self.device)  # [D]

        self._total_steps = 0

    def set_training(self, is_training: bool):
        self.qf1.train(is_training)
        self.qf2.train(is_training)

    def update(self, batch, is_pretrain):
        self.set_training(True)

        cql_min_q_weight = self.cfg.cql_min_q_weight

        self._total_steps += 1

        metrics = self._update_step(batch, cql_min_q_weight)

        return metrics

    def _update_step(self, batch, cql_m_q_w):
        metrics = {}

        # Convert data to torch tensors
        rewards = batch['rewards'].reshape(-1, 1)
        dones = batch['dones'].reshape(-1, 1)
        mc_returns = batch['mc_returns'].reshape(-1, 1)
        demos = batch['demos'].reshape(-1, 1)
        assert len(batch['observations'].shape) == 2
        assert len(batch['actions'].shape) == 2
        assert len(batch['next_observations'].shape) == 2

        observations = torch.tensor(batch['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.device)
        next_observations = torch.tensor(batch['next_observations'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        mc_returns = torch.tensor(mc_returns, dtype=torch.float32).to(self.device)
        demos = torch.tensor(demos, dtype=torch.float32).to(self.device)

        # discrete
        actions_d = encode_action(actions, self.low, self.high, self.levels, self.bins).long()  # [B, L, D]
        actions_d = actions_d.unsqueeze(-1)  # [B, L, D, 1]
        # data in action_dim is bin idx

        idx_temp = torch.ones_like(actions_d).repeat(1, 1, 1, self.bins).scatter(-1, actions_d, 0)  # [B, L, D, bins]
        actions_d_i = torch.nonzero(idx_temp)  # [B * L * D * (bins - 1), 4]
        actions_d_i = actions_d_i[:, -1].reshape(-1, self.levels, self.action_dim, self.bins - 1)  # [B, L, D, bins - 1]

        idx_temp = torch.ones_like(actions_d).repeat(1, 1, 1, self.bins)  # [B, L, D, bins]
        actions_d_all = torch.nonzero(idx_temp)  # [B * L * D * bins, 4]
        actions_d_all = actions_d_all[:, -1].reshape(-1, self.levels, self.action_dim, self.bins)  # [B, L, D, bins]

        # check: (0, ..., bins - 1) all in (actions_d_i | actions_d)
        assert actions_d_i.shape == (actions_d.shape[0], actions_d.shape[1], actions_d.shape[2], self.bins - 1)
        assert actions_d_i[0, 0, 0].sum().item() + actions_d[0, 0, 0].item() == self.bins * (self.bins - 1) // 2

        # Q function loss
        qf1_loss = 0
        qf2_loss = 0

        # Q function loss
        v1_pred, ad1_pred, nb1_pred = self.qf1.forward(observations, actions_d.squeeze(-1))
        # [B, 1], [B, L, D, bins], [B, L, D, 1]
        v2_pred, ad2_pred, nb2_pred = self.qf2.forward(observations, actions_d.squeeze(-1))
        # [B, 1], [B, L, D, bins], [B, L, D, 1]
        metrics_full_log(metrics, 'v1_pred', v1_pred)
        metrics_full_log(metrics, 'v2_pred', v2_pred)

        sa = self.soft_alpha
        adv1_pred = ad1_pred - sa * torch.logsumexp(ad1_pred / sa, dim=-1, keepdim=True)
        adv2_pred = ad2_pred - sa * torch.logsumexp(ad2_pred / sa, dim=-1, keepdim=True)

        adv1a = adv1_pred.gather(-1, actions_d)  # [B, L, D, 1]
        adv2a = adv2_pred.gather(-1, actions_d)  # [B, L, D, 1]
        nb1 = nb1_pred.sum(1).sum(1)  # [B, 1]
        nb2 = nb2_pred.sum(1).sum(1)  # [B, 1]
        q1a_pred = v1_pred + nb1 + adv1a.sum(1).sum(1)  # [B, 1]
        q2a_pred = v2_pred + nb2 + adv2a.sum(1).sum(1)  # [B, 1]
        metrics_full_log(metrics, 'q1_pred', q1a_pred)
        metrics_full_log(metrics, 'q2_pred', q2a_pred)

        if self.cfg['bellman_loss_coef'] > 0:
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

                next_v1_t = self.target_qf1.forward_value_soft(next_observations)  # [B, 1]
                next_v2_t = self.target_qf2.forward_value_soft(next_observations)  # [B, 1]
                next_v_t = torch.minimum(next_v1_t, next_v2_t)  # [B, 1]

                td_target = rewards + (1. - dones) * self.cfg['discount'] * next_v_t  # [B, 1]
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
        demos_sum = demos.sum()
        if demos_sum > 0:
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
                cql_adv2_diff = torch.clamp(cql_adv2_ood - adv2a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  # [B, 1]
            elif self.cfg["cql_type"] == "margin":
                cql_adv1_diff = torch.clamp(cql_adv1_other - adv1a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1).mean(dim=1)  # [B, 1]
                cql_adv2_diff = torch.clamp(cql_adv2_other - adv2a, self.cfg['cql_clip_diff_min'],
                                            self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1).mean(dim=1)  # [B, 1]
            else:
                raise NotImplementedError

            cql_adv1_diff = (cql_adv1_diff * demos).sum() / demos_sum
            cql_adv2_diff = (cql_adv2_diff * demos).sum() / demos_sum

            metrics["cql/cql_qf1_diff"] = cql_adv1_diff.item()
            metrics["cql/cql_qf2_diff"] = cql_adv2_diff.item()

            cql_min_qf1_loss = cql_adv1_diff * cql_m_q_w
            cql_min_qf2_loss = cql_adv2_diff * cql_m_q_w

            metrics['cql/cql_min_qf1_loss'] = cql_min_qf1_loss.item()
            metrics['cql/cql_min_qf2_loss'] = cql_min_qf2_loss.item()

            qf1_loss += cql_min_qf1_loss
            qf2_loss += cql_min_qf2_loss

        metrics['qf1_loss'] = qf1_loss.mean().item()
        metrics['qf2_loss'] = qf2_loss.mean().item()

        # update
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        qf_loss = qf1_loss + qf2_loss
        qf_loss.backward(retain_graph=False)

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # Update target networks
        tau = self.cfg['soft_target_update_rate']
        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return metrics

    def act(self, observation: np.ndarray, deterministic=False) -> np.ndarray:
        self.set_training(False)

        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)  # [B, obs_dim]

        if self.cfg["act_network"] == "default":
            nn1 = self.qf1
            nn2 = self.qf2
        elif self.cfg["act_network"] == "target":
            nn1 = self.target_qf1
            nn2 = self.target_qf2
        else:
            raise NotImplementedError

        nad = self.qf1.infer(observation, nn1, nn2, deterministic=deterministic)  # [B, L, D]

        action_c = decode_action(nad, self.low, self.high, self.levels, self.bins)  # [B, D]

        return action_c.cpu().numpy()

    def save(self, path: Path):
        state_dict = {
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),

            'qf1_optimizer': self.qf1_optimizer.state_dict(),
            'qf2_optimizer': self.qf2_optimizer.state_dict(),
        }

        os.makedirs(path.parent, exist_ok=True)
        torch.save(state_dict, path)

    def load(self, path: Path):
        checkpoint = torch.load(path)
        self.qf1.load_state_dict(checkpoint['qf1'])
        self.qf2.load_state_dict(checkpoint['qf2'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2'])

        self.qf1_optimizer.load_state_dict(checkpoint['qf1_optimizer'])
        self.qf2_optimizer.load_state_dict(checkpoint['qf2_optimizer'])

        for param_group in self.qf1_optimizer.param_groups:
            param_group['lr'] = self.cfg['qf_lr']

        for param_group in self.qf2_optimizer.param_groups:
            param_group['lr'] = self.cfg['qf_lr']

