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
    def __init__(self, cfg, rgb_obs_shape, low_obs_shape, action_shape, use_logger,  nu = 0.01,
        kappa = 0.1,
        num_walks = 10,
        use_fk_reg = True,
        use_eikonal = False,
        fk_weight = 1.0,
        **kwargs,):
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
        self.nu = nu
        self.kappa = kappa
        self.num_walks = num_walks
        self.use_fk_reg = use_fk_reg
        self.fk_weight = fk_weight
        self.use_eikonal = use_eikonal

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

        # =====================================================================
        # 1. AUTOGRAD PREP: Enable gradient tracking if Eikonal is active
        # =====================================================================
        if getattr(self.cfg, 'use_eikonal', getattr(self, 'use_eikonal', False)):
            low_dim_obs = low_dim_obs.detach().clone().requires_grad_(True)
            rgb_obs = rgb_obs.detach().clone().requires_grad_(True)

        actions_d = encode_action(action, self.low, self.high, self.levels, self.bins).long()  # [B, L, D]
        actions_d = actions_d.unsqueeze(-1)  # [B, L, D, 1]

        idx_temp = torch.ones_like(actions_d).repeat(1, 1, 1, self.bins).scatter(-1, actions_d, 0)  # [B, L, D, bins]
        actions_d_i = torch.nonzero(idx_temp)  # [B * L * D * (bins - 1), 4]
        actions_d_i = actions_d_i[:, -1].reshape(-1, self.levels, self.action_dim, self.bins - 1)  # [B, L, D, bins - 1]

        # Q function loss tracking
        qf1_loss = 0
        qf2_loss = 0

        # Forward passes (v_pred is exactly V(s), ad_pred is A(s, a))
        v1_pred, ad1_pred = self.qf1.forward(rgb_obs, low_dim_obs, actions_d.squeeze(-1))  
        v2_pred, ad2_pred = self.qf2.forward(rgb_obs, low_dim_obs, actions_d.squeeze(-1))  
        
        metrics_full_log(metrics, 'v1_pred', v1_pred)
        metrics_full_log(metrics, 'v2_pred', v2_pred)

        adv1_pred = ad1_pred - self.soft_alpha * torch.logsumexp(ad1_pred / self.soft_alpha, dim=-1, keepdim=True)
        adv2_pred = ad2_pred - self.soft_alpha * torch.logsumexp(ad2_pred / self.soft_alpha, dim=-1, keepdim=True)
        adv1a = adv1_pred.gather(-1, actions_d)  # [B, L, D, 1]
        adv2a = adv2_pred.gather(-1, actions_d)  # [B, L, D, 1]
        
        q1a_pred = v1_pred + adv1a.sum(1).sum(1)  # [B, 1]
        q2a_pred = v2_pred + adv2a.sum(1).sum(1)  # [B, 1]

        # --- Standard Bellman & CQL Loss Block ---
        if self.cfg['bellman_loss_coef'] > 0.0:
            with torch.no_grad():
                adv1i = adv1_pred.gather(-1, actions_d_i)  
                adv2i = adv2_pred.gather(-1, actions_d_i)  
                
                next_v1_t = self.qf1_target.forward_value_soft(next_rgb_obs, next_low_dim_obs)  
                next_v2_t = self.qf2_target.forward_value_soft(next_rgb_obs, next_low_dim_obs)  
                next_v_t = torch.minimum(next_v1_t, next_v2_t)  

                td_target = reward + discount * next_v_t  

            qf1_bellman_loss = torch.mean((q1a_pred - td_target) ** 2)
            qf2_bellman_loss = torch.mean((q2a_pred - td_target) ** 2)

            metrics['qf1_bellman_loss'] = qf1_bellman_loss.item()
            metrics['qf2_bellman_loss'] = qf2_bellman_loss.item()

            qf1_loss = qf1_bellman_loss * self.cfg['bellman_loss_coef']
            qf2_loss = qf2_bellman_loss * self.cfg['bellman_loss_coef']

        # CQL
        demos_sum = torch.sum(demos)
        if self.cfg["cql_min_q_weight"] > 0.0 and demos_sum > 0:
            cql_adv1_other = adv1_pred.gather(-1, actions_d_i)  
            cql_adv2_other = adv2_pred.gather(-1, actions_d_i)  

            if self.cfg["cql_type"] == "cql":
                cql_temp = self.cfg['cql_temp']
                cql_adv1_ood = torch.logsumexp(cql_adv1_other / cql_temp, dim=-1, keepdim=True) * cql_temp  
                cql_adv2_ood = torch.logsumexp(cql_adv2_other / cql_temp, dim=-1, keepdim=True) * cql_temp  

                cql_adv1_diff = torch.clamp(cql_adv1_ood - adv1a, self.cfg['cql_clip_diff_min'], self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(cql_adv2_ood - adv2a, self.cfg['cql_clip_diff_min'], self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  
                cql_adv2_diff = (cql_adv2_diff * demos).mean()
            elif self.cfg["cql_type"] == "margin":
                cql_adv1_diff = torch.clamp(cql_adv1_other - adv1a, self.cfg['cql_clip_diff_min'], self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)  
                cql_adv1_diff = (cql_adv1_diff * demos).mean()
                cql_adv2_diff = torch.clamp(cql_adv2_other - adv2a, self.cfg['cql_clip_diff_min'], self.cfg['cql_clip_diff_max']).sum(dim=1).sum(dim=1)
                cql_adv2_diff = (cql_adv2_diff * demos).mean()

            metrics["cql/cql_qf1_loss"] = cql_adv1_diff.item()
            metrics["cql/cql_qf2_loss"] = cql_adv2_diff.item()

            qf1_loss += cql_adv1_diff * self.cfg["cql_min_q_weight"]
            qf2_loss += cql_adv2_diff * self.cfg["cql_min_q_weight"]

        # If 0.0 weights were used, ensure these are tensors so we can add physics losses
        if isinstance(qf1_loss, int): qf1_loss = torch.tensor(0.0, device=self.device)
        if isinstance(qf2_loss, int): qf2_loss = torch.tensor(0.0, device=self.device)

        critic_loss = qf1_loss + qf2_loss

        # =====================================================================
        # --- EXACT AUTOGRAD EIKONAL LOSS ---
        # =====================================================================
        if getattr(self.cfg, 'use_eikonal', getattr(self, 'use_eikonal', False)):
            # Average the scalar state-values from both dueling heads
            v_s = (v1_pred + v2_pred) / 2.0  # [B, 1]

            grads = torch.autograd.grad(
                outputs=v_s.sum(), 
                inputs=(low_dim_obs, rgb_obs),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )

            grad_v = grads[0] if grads[0] is not None else grads[1]
            if len(grad_v.shape) > 2:
                grad_v = grad_v.reshape(grad_v.shape[0], -1)

            grad_norm = torch.linalg.norm(grad_v, dim=-1) # [B]
            
            # Using 1.0 as a static proxy speed limit (adjust as needed)
            current_speed = torch.ones((low_dim_obs.shape[0],), device=self.device)
            kappa = getattr(self.cfg, 'kappa', getattr(self, 'kappa', 0.1))
            max_slope = kappa / (current_speed + 1e-6)

            slope_excess = torch.relu(grad_norm - max_slope)
            eikonal_loss = slope_excess.pow(2).mean()

            fk_weight = getattr(self.cfg, 'fk_weight', getattr(self, 'fk_weight', 1.0))
            critic_loss = critic_loss + fk_weight * eikonal_loss
            metrics['eikonal_loss'] = eikonal_loss.item()

        # =====================================================================
        # --- STOCHASTIC FK (Walk-on-Spheres) LOSS ---
        # =====================================================================
        if getattr(self.cfg, 'use_fk_reg', getattr(self, 'use_fk_reg', False)):
            B, D_obs = low_dim_obs.shape
            nu = getattr(self.cfg, 'nu', getattr(self, 'nu', 0.01))
            sigma = (2.0 * nu) ** 0.5
            num_walks = getattr(self.cfg, 'num_walks', getattr(self, 'num_walks', 10))
            kappa = getattr(self.cfg, 'kappa', getattr(self, 'kappa', 0.1))
            
            # Perturb States
            noise = torch.randn(B, num_walks, D_obs, device=self.device) * sigma
            flat_low_dim = (low_dim_obs.unsqueeze(1) + noise).view(B * num_walks, D_obs)
            flat_rgb = rgb_obs.unsqueeze(1).expand(-1, num_walks, *rgb_obs.shape[1:]).reshape(B * num_walks, *rgb_obs.shape[1:])
            
            # Because SQAR isolates V(s), we can just query forward_value_soft directly!
            # No need to marginalize over actions or bins here.
            v1_neighbors = self.qf1.forward_value_soft(flat_rgb, flat_low_dim).view(B, num_walks)
            v2_neighbors = self.qf2.forward_value_soft(flat_rgb, flat_low_dim).view(B, num_walks)
            v_neighbors = (v1_neighbors + v2_neighbors) / 2.0  # [B, Walks]

            # Re-use anchor values (detach to prevent pulling the anchor up)
            v_anchor = ((v1_pred + v2_pred) / 2.0).detach()  # [B, 1]

            diff = (v_neighbors - v_anchor).abs()
            grad_epsilon = torch.linalg.norm(noise, dim=-1) + 1e-6
            grad_estimate = diff / grad_epsilon  # [B, Walks]

            current_speed = torch.ones((B,), device=self.device)
            max_slope = kappa / (current_speed + 1e-6)

            slope_excess = torch.relu(grad_estimate - max_slope.unsqueeze(1))
            fk_loss = slope_excess.pow(2).mean()

            fk_weight = getattr(self.cfg, 'fk_weight', getattr(self, 'fk_weight', 1.0))
            critic_loss = critic_loss + fk_weight * fk_loss
            metrics['fk_loss'] = fk_loss.item()
            metrics['avg_grad_norm'] = grad_estimate.mean().item()

        # =====================================================================

        metrics['qf1_loss'] = qf1_loss.mean().item() if isinstance(qf1_loss, torch.Tensor) else qf1_loss
        metrics['qf2_loss'] = qf2_loss.mean().item() if isinstance(qf2_loss, torch.Tensor) else qf2_loss

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
