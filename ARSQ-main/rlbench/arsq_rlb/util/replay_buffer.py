from collections import defaultdict

import numpy as np
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


class ReplayBuffer(IterableDataset):
    def __init__(
            self,
            data_specs,
            use_relabeling,
            is_demo_buffer,
            max_size,
            nstep,
            discount,
            do_always_bootstrap,
            frame_stack,
    ):
        self._data_specs = data_specs
        self._use_relabeling = use_relabeling
        self._is_demo_buffer = is_demo_buffer

        self._max_size = max_size
        self._nstep = nstep
        self._discount = discount
        self._do_always_bootstrap = do_always_bootstrap
        self._frame_stack = frame_stack

        self._current_episode = defaultdict(list)
        self._num_transitions = 0
        self._fs = []
        self._fs_eplen = []
        self._online_buffer = True

    def set_online_buffer(self, online_buffer):
        print(f"Setting online buffer to {online_buffer}")
        self._online_buffer = online_buffer

    def add(self, time_step):
        if not self._online_buffer:
            return

        for spec in self._data_specs:
            value = time_step[spec.name]
            # Remove frame stacking
            if spec.name == "low_dim_obs":
                low_dim = 8  # hard-coded
                value = value[..., -low_dim:]
            elif spec.name == "rgb_obs":
                rgb_dim = 3  # hard-coded
                value = value[:, -rgb_dim:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype, (
                spec.name,
                spec.shape,
                value.shape,
                spec.dtype,
                value.dtype,
            )
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            if self._use_relabeling:
                episode = self._relabel_episode(episode)
            if self._is_demo_buffer:
                # If this is demo replay buffer, save only when it's successful
                if self._check_if_successful(episode):
                    self._store_episode(episode)
            else:
                self._store_episode(episode)

    def _relabel_episode(self, episode):
        if self._check_if_successful(episode):
            episode["demo"] = np.ones_like(episode["demo"])
        return episode

    def _check_if_successful(self, episode):
        reward = episode["reward"]
        return np.isclose(reward[-1], 1.0)

    def _store_episode(self, episode):
        eplen = episode_len(episode)

        self._num_transitions += eplen
        self._fs.append(episode)
        self._fs_eplen.append(eplen)

        while self._num_transitions > self._max_size:
            ep = self._fs.pop(0)
            eplen = self._fs_eplen.pop(0)
            self._num_transitions -= eplen

    def _sample(self):
        ep_idx = np.random.randint(0, len(self._fs))
        episode = self._fs[ep_idx]
        episode_len = self._fs_eplen[ep_idx]
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len - self._nstep + 1) + 1
        next_idx = idx + self._nstep - 1

        obs_idxs = list(
            map(
                lambda x: np.clip(x, 0, None),
                range((idx - 1) - self._frame_stack + 1, (idx - 1) + 1),
            )
        )
        obs_next_idxs = list(
            map(
                lambda x: np.clip(x, 0, None),
                range(next_idx - self._frame_stack + 1, next_idx + 1),
            )
        )

        # rgb_obs stacking -- channel-wise concat
        rgb_obs = np.concatenate(episode["rgb_obs"][obs_idxs], 1)
        next_rgb_obs = np.concatenate(episode["rgb_obs"][obs_next_idxs], 1)
        # low_dim_obs stacking -- last-dim-wise concat
        low_dim_obs = np.concatenate(episode["low_dim_obs"][obs_idxs], -1)
        next_low_dim_obs = np.concatenate(episode["low_dim_obs"][obs_next_idxs], -1)

        action = episode["action"][idx]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            if self._do_always_bootstrap:
                _discount = 1.0
            else:
                _discount = episode["discount"][idx + i]
            discount *= _discount * self._discount
        demo = episode["demo"][idx]
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
        return self._num_transitions

    def __iter__(self):
        while True:
            yield self._sample()

class ReplayBufferBatch:
    def __init__(
            self,
            data_specs,
            use_relabeling,
            is_demo_buffer,
            max_size,
            nstep,
            discount,
            do_always_bootstrap,
            frame_stack,
    ):
        self._data_specs = data_specs
        self._use_relabeling = use_relabeling
        self._is_demo_buffer = is_demo_buffer

        self._max_size = max_size
        self._nstep = nstep
        self._discount = discount
        self._do_always_bootstrap = do_always_bootstrap
        self._frame_stack = frame_stack

        self._online_buffer = True

        self._current_episode = defaultdict(list)

        self._num_steps = 0
        self._ep_len = []
        self._episodes = []


    def set_online_buffer(self, online_buffer):
        print(f"Setting online buffer to {online_buffer}")
        self._online_buffer = online_buffer

    def add(self, time_step):
        if not self._online_buffer:
            return

        for spec in self._data_specs:
            value = time_step[spec.name]
            # Remove frame stacking
            if spec.name == "low_dim_obs":
                low_dim = 8  # hard-coded
                value = value[..., -low_dim:]
            elif spec.name == "rgb_obs":
                rgb_dim = 3  # hard-coded
                value = value[:, -rgb_dim:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype, (
                spec.name,
                spec.shape,
                value.shape,
                spec.dtype,
                value.dtype,
            )
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            if self._use_relabeling:
                episode = self._relabel_episode(episode)
            if self._is_demo_buffer:
                # If this is demo replay buffer, save only when it's successful
                if self._check_if_successful(episode):
                    self._store_episode(episode)
            else:
                self._store_episode(episode)

    def _relabel_episode(self, episode):
        if self._check_if_successful(episode):
            episode["demo"] = np.ones_like(episode["demo"])
        return episode

    def _check_if_successful(self, episode):
        reward = episode["reward"]
        return np.isclose(reward[-1], 1.0)

    def _store_episode(self, episode):
        eplen = episode_len(episode) + 1 # total len
        assert eplen > 2

        self._ep_len.append(eplen)
        self._episodes.append(episode)
        self._num_steps += eplen

        while self._num_steps - len(self._ep_len) > self._max_size: # transitions = num_steps - num_episodes
            evicted_eplen = self._ep_len.pop(0)
            self._episodes.pop(0)
            self._num_steps -= evicted_eplen

    def sample(self, bs):
        ep_idxs = np.random.randint(0, len(self._episodes), size=(bs,))  # [B]
        ep_lens = np.array(self._ep_len)[ep_idxs]  # [B]

        idxs_step = np.random.randint(1, ep_lens - self._nstep + 1)  # [B], 1-indexed local step

        # Local frame indices within each episode: [B, frame_stack]
        obs_local = idxs_step.reshape(-1, 1) - 1 + np.arange(-self._frame_stack + 1, 1).reshape(1, -1)
        obs_local = np.clip(obs_local, 0, None)  # [B, frame_stack]

        next_idxs_step = idxs_step + self._nstep - 1  # [B]
        obs_next_local = next_idxs_step.reshape(-1, 1) + np.arange(-self._frame_stack + 1, 1).reshape(1, -1)
        obs_next_local = np.clip(obs_next_local, 0, None)  # [B, frame_stack]

        # rgb_obs stacking -- gather per episode then channel-wise concat
        rgb_obs = np.array([self._episodes[ep_idxs[b]]["rgb_obs"][obs_local[b]] for b in range(bs)])
        # [B, frame_stack, cams, C, H, W] -> [B, cams, frame_stack*C, H, W]
        rgb_obs = rgb_obs.swapaxes(1, 2)  # [B, cams, frame_stack, C, H, W]
        rgb_obs = rgb_obs.reshape(bs, rgb_obs.shape[1], -1, *rgb_obs.shape[4:])

        next_rgb_obs = np.array([self._episodes[ep_idxs[b]]["rgb_obs"][obs_next_local[b]] for b in range(bs)])
        next_rgb_obs = next_rgb_obs.swapaxes(1, 2)
        next_rgb_obs = next_rgb_obs.reshape(bs, next_rgb_obs.shape[1], -1, *next_rgb_obs.shape[4:])

        # low_dim_obs stacking -- last-dim-wise concat
        low_dim_obs = np.array([self._episodes[ep_idxs[b]]["low_dim_obs"][obs_local[b]] for b in range(bs)])
        # [B, frame_stack, low_dim] -> [B, frame_stack*low_dim]
        low_dim_obs = low_dim_obs.swapaxes(0, 1)  # [frame_stack, B, low_dim]
        low_dim_obs = np.concatenate(low_dim_obs, -1)  # [B, frame_stack*low_dim]

        next_low_dim_obs = np.array([self._episodes[ep_idxs[b]]["low_dim_obs"][obs_next_local[b]] for b in range(bs)])
        next_low_dim_obs = next_low_dim_obs.swapaxes(0, 1)
        next_low_dim_obs = np.concatenate(next_low_dim_obs, -1)

        action = np.array([self._episodes[ep_idxs[b]]["action"][idxs_step[b]] for b in range(bs)])

        reward = np.zeros((bs, 1), dtype=np.float32)  # [B, 1]
        discount = np.ones((bs, 1), dtype=np.float32)  # [B, 1]

        for i in range(self._nstep):
            step_reward = np.array([self._episodes[ep_idxs[b]]["reward"][idxs_step[b] + i]
                                    for b in range(bs)]).reshape(-1, 1)
            reward += discount * step_reward
            if self._do_always_bootstrap:
                _discount = np.ones((bs, 1), dtype=np.float32)
            else:
                _discount = np.array([self._episodes[ep_idxs[b]]["discount"][idxs_step[b] + i]
                                      for b in range(bs)]).reshape(-1, 1)
            discount *= _discount * self._discount

        demo = np.array([self._episodes[ep_idxs[b]]["demo"][idxs_step[b]] for b in range(bs)])

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
        return self._num_steps - len(self._episodes)

