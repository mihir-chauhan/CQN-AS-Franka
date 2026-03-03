import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from arsq_d4rl.alg.cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
    weight_init,
    schedule,
    TruncatedNormal,
    soft_update_params,
)
from arsq_d4rl.util.util import metrics_full_log


class C2FCriticNetwork(nn.Module):
    def __init__(
            self,
            repr_dim: int,
            action_dim: int,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._actor_dim = action_dim
        self._bins = bins

        # Advantage stream in Dueling network
        self.adv_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.adv_head = nn.Linear(hidden_dim, self._actor_dim * bins * atoms)
        self.adv_output_shape = (self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        self.value_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.value_head = nn.Linear(hidden_dim, self._actor_dim * 1 * atoms)
        self.value_output_shape = (self._actor_dim, 1, atoms)

        self.apply(weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def forward(
            self, level: int, obs: torch.Tensor, prev_action: torch.Tensor
    ):
        """
        Inputs:
        - level: level index
        - obs: features from visual encoder
        - prev_action: actions from previous level

        Outputs:
        - q_logits: (batch_size, action_dim, bins, atoms)
        """
        level_id = (
            torch.eye(self._levels, device=obs.device, dtype=obs.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(obs.shape[0], 0)
        )

        value_h = self.value_trunk(obs)
        value_x = torch.cat([value_h, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        adv_h = self.adv_trunk(obs)
        adv_x = torch.cat([adv_h, prev_action, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    def __init__(
            self,
            action_dim: int,
            repr_dim: int,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
            v_min: float,
            v_max: float,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        actor_dim = action_dim
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
            repr_dim, action_dim, feature_dim, hidden_dim, levels, bins, atoms
        )

    def get_action(self, obs: torch.Tensor):
        metrics = dict()
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, obs, (low + high) / 2)
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

            # for logging
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[
                ..., 0
            ]  # [..., D]
            metrics_full_log(metrics, f"critic_target_q_level{level}", qs_a)
        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action, metrics

    def forward(
            self,
            obs: torch.Tensor,
            continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            obs: [B, F] shaped feature tensor
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

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()
        for level in range(self.levels):
            q_logits = self.network(level, obs, (low + high) / 2)
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
            next_obs: torch.Tensor,
            next_continuous_action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_obs: [B, F] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, atoms] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(
            next_obs, next_continuous_action
        )[1]

        shape = next_q_probs_a.shape  # [B, L, D, atoms]
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        # Compute Tz for [B, atoms]
        Tz = reward + discount * self.support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.v_min) / self.delta_z
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l =b = u (b is int)
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

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


class Critic(nn.Module):
    def __init__(
            self,
            action_dim: int,
            obs_dim: int,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            v_min: float,
            v_max: float,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.v_min = v_min
        self.v_max = v_max
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * action_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * action_dim), requires_grad=False
        )

        self.network = C2FCriticNetwork(
            obs_dim,
            action_dim,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms=1,
        )

    def get_action(self, obs: torch.Tensor):
        metrics = dict()
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, obs, (low + high) / 2)
            qs = q_logits.squeeze(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

            # for logging
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[
                ..., 0
            ]  # [..., D]
            metrics_full_log(metrics, f"critic_target_q_level{level}", qs_a)
        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action, metrics

    def forward(
            self,
            obs: torch.Tensor,
            continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            obs: [B, obs_dim] shaped feature tensor
            continuous_action: [B, D] shaped action tensor

        Return:
            qs: [B, L, D, bins, 1] for value distribution at all bins
            qa: [B, L, D, 1] for value distribution at given bin
        """

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        qs_per_level = []
        qa_per_level = []

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()
        for level in range(self.levels):
            qs = self.network(level, obs, (low + high) / 2)  # [B, D, bins, 1]
            argmax_q = discrete_action[..., level, :].long()  # [B, L, D] -> [B, D]

            qa = torch.gather(
                qs,
                dim=-2,
                index=argmax_q.unsqueeze(-1).unsqueeze(-1)
            )  # [B, D, 1, 1]
            qa = qa[..., 0, :]  # [B, D, 1]

            qs_per_level.append(qs)
            qa_per_level.append(qa)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        qs = torch.stack(qs_per_level, -4)  # [B, L, D, bins, 1]
        qa = torch.stack(qa_per_level, -3)  # [B, L, D, 1]
        return qs, qa

    def compute_target_q_dist(
            self,
            next_obs: torch.Tensor,
            next_continuous_action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_obs: [B, obs_dim] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, 1] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(
            next_obs, next_continuous_action
        )[1]  # [B, L, D, 1]

        r = reward.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        dc = discount.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        q_target = r + dc * next_q_probs_a  # [B, L, D, 1]

        return q_target

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


class CQN:
    def __init__(self, cfg, obs_shape, action_shape):
        self.cfg = cfg

        self.discount = cfg.discount

        self.lr = cfg.lr
        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim
        self.levels = cfg.levels
        self.bins = cfg.bins
        self.atoms = cfg.atoms
        self.v_min = cfg.v_min
        self.v_max = cfg.v_max
        self.critic_target_tau = cfg.critic_target_tau
        self.num_expl_steps = cfg.num_expl_steps
        # self.update_every_steps = cfg.update_every_steps
        self.stddev_schedule = cfg.stddev_schedule

        self.bc_lambda = cfg.bc_lambda
        self.bc_fosd = cfg.bc_fosd
        self.bc_margin = cfg.bc_margin
        self.critic_lambda = cfg.critic_lambda
        self.weight_decay = cfg.weight_decay

        self.device = "cuda"

        # models
        if self.atoms == 1:
            self.critic = Critic(
                action_shape,
                obs_shape,
                self.feature_dim,
                self.hidden_dim,
                self.levels,
                self.bins,
                self.v_min,
                self.v_max,
            ).to(self.device)
            self.critic_target = Critic(
                action_shape,
                obs_shape,
                self.feature_dim,
                self.hidden_dim,
                self.levels,
                self.bins,
                self.v_min,
                self.v_max,
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.critic = C2FCritic(
                action_shape,
                obs_shape,
                self.feature_dim,
                self.hidden_dim,
                self.levels,
                self.bins,
                self.atoms,
                self.v_min,
                self.v_max,
            ).to(self.device)
            self.critic_target = C2FCritic(
                action_shape,
                obs_shape,
                self.feature_dim,
                self.hidden_dim,
                self.levels,
                self.bins,
                self.atoms,
                self.v_min,
                self.v_max,
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.weight_decay > 0:
            self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self._set_training_mode(True)
        self.critic_target.eval()

        print(self.critic)

    def _set_training_mode(self, training=True):
        self.critic.train(training)

    def act(self, obs, deterministic):
        self._set_training_mode(False)
        step = 0

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        stddev = schedule(self.stddev_schedule, step)
        action, _ = self.critic_target.get_action(obs)  # use critic_target
        stddev = torch.ones_like(action) * stddev
        dist = TruncatedNormal(action, stddev)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            # if step < self.num_expl_steps:
            #     action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()

    def _update_critic(self, obs, action, reward, discount, next_obs, demos):
        metrics = dict()

        with torch.no_grad():
            next_action, mets = self.critic.get_action(next_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_obs, next_action, reward, discount
            )
            metrics.update(**mets)

        if self.atoms == 1:
            q_probs = q_probs_a = log_q_probs = log_q_probs_a = None
            qs, qa = self.critic(obs, action)
            q_critic_loss = F.mse_loss(qa, target_q_probs_a)
        else:
            qs = qa = None
            q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(obs, action)
            q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()

        critic_loss = self.critic_lambda * q_critic_loss

        metrics["q_critic_loss"] = q_critic_loss.item()

        if self.bc_lambda > 0.0:
            demos = demos.float().squeeze(1)  # [B,]
            demos_sum = torch.sum(demos)
            metrics["ratio_of_demos"] = demos.mean().item()

            if demos_sum > 0 and self.atoms > 1 and self.bc_fosd > 0:
                # q_probs: [B, L, D, bins, atoms], q_probs_a: [B, L, D, atoms]
                q_probs_cdf = torch.cumsum(q_probs, -1)
                q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
                # q_probs_{a_{i}} is stochastically dominant over q_probs_{a_{-i}}
                bc_fosd_loss = (
                                   (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
                                   .clamp(min=0)
                                   .sum(-1)
                                   .mean([-1, -2, -3])
                               ) * self.bc_fosd
                bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
                critic_loss = critic_loss + self.bc_fosd * bc_fosd_loss
                metrics["bc_fosd_loss"] = bc_fosd_loss.item()

            if demos_sum > 0 and self.bc_margin > 0:
                if self.atoms == 1:
                    qs = qs.squeeze(-1)
                    qs_a = qa.squeeze(-1)
                else:
                    qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
                    qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(-1)
                margin_loss = torch.clamp(
                    self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
                ).mean([-1, -2, -3])
                margin_loss = (margin_loss * demos).sum() / demos.sum()
                critic_loss = critic_loss + self.bc_lambda * margin_loss
                metrics["bc_margin_loss"] = margin_loss.item()

        # optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return metrics

    def update(self, batch, step):
        self._set_training_mode(True)
        metrics = {}

        # if step % self.update_every_steps != 0:
        #     return metrics

        # Convert data to torch tensors
        rewards = batch['rewards'].reshape(-1, 1)
        dones = batch['dones'].reshape(-1, 1)
        demos = batch['demos'].reshape(-1, 1)
        assert len(batch['observations'].shape) == 2
        assert len(batch['actions'].shape) == 2
        assert len(batch['next_observations'].shape) == 2

        obs = torch.tensor(batch['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(batch['next_observations'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        demos = torch.tensor(demos, dtype=torch.float32).to(self.device)

        discount = torch.zeros_like(dones, dtype=torch.float32).to(self.device) + self.discount

        # encode
        metrics["batch_reward"] = rewards.mean().item()

        # update critic
        metrics.update(
            self._update_critic(
                obs,
                actions,
                rewards,
                discount,
                next_obs,
                demos,
            )
        )

        # update critic target
        soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def save(self, path: Path):
        state_dict = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic.state_dict(),
        }

        os.makedirs(path.parent, exist_ok=True)
        torch.save(state_dict, path)

    def load(self, path: Path):
        state_dict = torch.load(path)

        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])

        self.critic_opt.load_state_dict(state_dict['critic_optimizer'])
