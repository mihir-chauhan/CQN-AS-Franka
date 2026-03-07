"""
replay_buffer.py — In-memory replay buffer for ARSQ real-robot training.

Adapted from ARSQ-main/rlbench/arsq_rlb/util/replay_buffer.py.
Uses the ReplayBufferBatch class which stores all data contiguously in numpy
arrays and samples batches directly (no DataLoader / disk I/O).
"""

from collections import defaultdict
import numpy as np


def episode_len(episode):
    """Number of transitions = total stored steps - 1 (dummy first transition)."""
    return next(iter(episode.values())).shape[0] - 1


class ReplayBufferBatch:
    """In-memory replay buffer that stores episodes contiguously.

    Compatible with the real-robot env's TimeStep interface.

    Data is inserted one TimeStep at a time via add(). When
    time_step.last() is True the current episode is committed.
    sample(bs) returns a tuple of numpy arrays ready for the ARSQ agent.
    """

    def __init__(
        self,
        data_specs,
        use_relabeling: bool,
        is_demo_buffer: bool,
        max_size: int,
        nstep: int,
        discount: float,
        do_always_bootstrap: bool,
        frame_stack: int,
        low_dim_raw: int = 8,
        rgb_channels: int = 3,
    ):
        self._data_specs = data_specs
        self._use_relabeling = use_relabeling
        self._is_demo_buffer = is_demo_buffer
        self._max_size = max_size
        self._nstep = nstep
        self._discount = discount
        self._do_always_bootstrap = do_always_bootstrap
        self._frame_stack = frame_stack
        self._low_dim_raw = low_dim_raw
        self._rgb_channels = rgb_channels

        self._online_buffer = True
        self._current_episode = defaultdict(list)

        self._num_steps = 0
        self._ep_start: list[int] = []
        self._ep_len: list[int] = []
        self._fs: dict[str, np.ndarray | None] = {
            spec.name: None for spec in self._data_specs
        }

    def set_online_buffer(self, online_buffer: bool):
        self._online_buffer = online_buffer

    # ── Insert ──────────────────────────────────────────────────────

    def add(self, time_step):
        if not self._online_buffer:
            return

        for spec in self._data_specs:
            value = time_step[spec.name]
            # Remove frame stacking (store raw single-frame data)
            if spec.name == "low_dim_obs":
                value = value[..., -self._low_dim_raw:]
            elif spec.name == "rgb_obs":
                value = value[:, -self._rgb_channels:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype, (
                spec.name, spec.shape, value.shape, spec.dtype, value.dtype,
            )
            self._current_episode[spec.name].append(value)

        if time_step.last():
            episode = {}
            for spec in self._data_specs:
                episode[spec.name] = np.array(
                    self._current_episode[spec.name], spec.dtype
                )
            self._current_episode = defaultdict(list)

            if self._use_relabeling:
                episode = self._relabel_episode(episode)
            if self._is_demo_buffer:
                if self._check_if_successful(episode):
                    self._store_episode(episode)
            else:
                self._store_episode(episode)

    def _relabel_episode(self, episode):
        if self._check_if_successful(episode):
            episode["demo"] = np.ones_like(episode["demo"])
        return episode

    @staticmethod
    def _check_if_successful(episode):
        return np.isclose(episode["reward"][-1], 1.0)

    def _store_episode(self, episode):
        eplen = episode_len(episode) + 1  # total steps
        assert eplen > 2

        self._ep_start.append(self._num_steps)
        self._ep_len.append(eplen)
        for spec in self._data_specs:
            if self._fs[spec.name] is None:
                self._fs[spec.name] = episode[spec.name]
            else:
                self._fs[spec.name] = np.concatenate(
                    [self._fs[spec.name], episode[spec.name]], axis=0
                )
        self._num_steps += eplen

        # Evict old episodes if over capacity
        while self._num_steps - len(self._ep_len) > self._max_size:
            eplen = self._ep_len[0]
            self._num_steps -= eplen
            self._ep_start.pop(0)
            self._ep_start = [x - eplen for x in self._ep_start]
            self._ep_len.pop(0)
            for spec in self._data_specs:
                self._fs[spec.name] = self._fs[spec.name][eplen:]

    # ── Sample ──────────────────────────────────────────────────────

    def sample(self, bs: int):
        ep_idxs = np.random.randint(0, len(self._ep_start), size=(bs,))
        ep_lens = np.array(self._ep_len)[ep_idxs]
        ep_starts = np.array(self._ep_start)[ep_idxs]

        idxs_step = np.random.randint(1, ep_lens - self._nstep + 1)
        next_idxs_step = idxs_step + self._nstep - 1

        idxs = idxs_step + ep_starts
        # next_idxs = next_idxs_step + ep_starts  # not used directly below

        # ── Frame-stacked obs ──
        obs_idxs = idxs_step.reshape(-1, 1) - 1
        obs_idxs = obs_idxs + np.arange(-self._frame_stack + 1, 1).reshape(1, -1)
        obs_idxs = np.clip(obs_idxs, 0, None) + ep_starts.reshape(-1, 1)
        obs_idxs_flat = obs_idxs.reshape(-1)

        obs_next_idxs = next_idxs_step.reshape(-1, 1)
        obs_next_idxs = obs_next_idxs + np.arange(
            -self._frame_stack + 1, 1
        ).reshape(1, -1)
        obs_next_idxs = np.clip(obs_next_idxs, 0, None) + ep_starts.reshape(-1, 1)
        obs_next_idxs_flat = obs_next_idxs.reshape(-1)

        # RGB: [B*F, cams, C, H, W] → [B, cams, F*C, H, W]
        rgb_obs = self._fs["rgb_obs"][obs_idxs_flat]
        rgb_obs = rgb_obs.reshape(bs, self._frame_stack, *rgb_obs.shape[1:])
        rgb_obs = rgb_obs.swapaxes(0, 1)
        rgb_obs = np.concatenate(rgb_obs, 2)

        next_rgb_obs = self._fs["rgb_obs"][obs_next_idxs_flat]
        next_rgb_obs = next_rgb_obs.reshape(
            bs, self._frame_stack, *next_rgb_obs.shape[1:]
        )
        next_rgb_obs = next_rgb_obs.swapaxes(0, 1)
        next_rgb_obs = np.concatenate(next_rgb_obs, 2)

        # Low-dim: [B*F, D] → [B, F*D]
        low_dim_obs = self._fs["low_dim_obs"][obs_idxs_flat]
        low_dim_obs = low_dim_obs.reshape(bs, self._frame_stack, *low_dim_obs.shape[1:])
        low_dim_obs = low_dim_obs.swapaxes(0, 1)
        low_dim_obs = np.concatenate(low_dim_obs, -1)

        next_low_dim_obs = self._fs["low_dim_obs"][obs_next_idxs_flat]
        next_low_dim_obs = next_low_dim_obs.reshape(
            bs, self._frame_stack, *next_low_dim_obs.shape[1:]
        )
        next_low_dim_obs = next_low_dim_obs.swapaxes(0, 1)
        next_low_dim_obs = np.concatenate(next_low_dim_obs, -1)

        # Action, reward, discount, demo
        action = self._fs["action"][idxs]
        reward = np.zeros((bs, 1), dtype=self._fs["reward"].dtype)
        discount = np.ones((bs, 1), dtype=self._fs["discount"].dtype)

        for i in range(self._nstep):
            step_reward = self._fs["reward"][idxs + i].reshape(-1, 1)
            reward += discount * step_reward
            if self._do_always_bootstrap:
                _discount = np.ones((bs, 1), dtype=self._fs["discount"].dtype)
            else:
                _discount = self._fs["discount"][idxs + i].reshape(-1, 1)
            discount *= _discount * self._discount

        demo = self._fs["demo"][idxs]

        return (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demo,
        )

    def __len__(self):
        return max(self._num_steps - len(self._ep_start), 0)
