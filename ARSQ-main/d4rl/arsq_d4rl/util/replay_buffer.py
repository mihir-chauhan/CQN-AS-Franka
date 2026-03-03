import collections

import gym
import numpy as np

ENV_CONFIG = {

    "hopper-random-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "halfcheetah-random-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "walker2d-random-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "hopper-medium-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "halfcheetah-medium-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "walker2d-medium-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "hopper-medium-replay-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "halfcheetah-medium-replay-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "walker2d-medium-replay-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "hopper-medium-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "halfcheetah-medium-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "walker2d-medium-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "hopper-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "halfcheetah-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },
    "walker2d-expert-v2": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": False,
    },

    "door-human-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "pen-human-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "hammer-human-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "relocate-human-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "door-cloned-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "pen-cloned-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "hammer-cloned-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },
    "relocate-cloned-v1": {
        "reward_pos": -100000,
        "reward_neg": -100000,
        "is_sparse": False,
        "use_goal": False,
        "manual_render": True,
    },

    "maze2d-medium-v1": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
        "is_sparse": True,
        "use_goal": False,
        "manual_render": False,
    },
    "antmaze-medium-diverse-v2": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
        "is_sparse": True,
        "use_goal": False,
        "manual_render": False,
    },
    "adroit-binary": {
        "reward_pos": 0.0,
        "reward_neg": -1.0,
        "is_sparse": True,
        "use_goal": True,
        "manual_render": False,
    }
}


class ReplayBuffer(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self._dones = np.zeros((self._max_size, 1), dtype=np.float32)
        self._mc_returns = np.zeros((self._max_size, 1), dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def from_dataset(self, dataset):
        # obs = np.concatenate(dataset["observations"], axis=0)
        # next_obs = np.concatenate(dataset["next_observations"], axis=0)
        # actions = np.concatenate(dataset["actions"], axis=0)
        # rewards = np.concatenate(dataset["rewards"], axis=0)
        # dones = np.concatenate(dataset["terminals"], axis=0)
        # mc_returns = np.concatenate(dataset["mc_returns"], axis=0)
        obs = dataset["observations"]
        next_obs = dataset["next_observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        dones = dataset["terminals"]
        mc_returns = dataset["mc_returns"]

        assert obs.shape[0] == next_obs.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0] == \
               mc_returns.shape[0]
        assert rewards.shape[1] == dones.shape[1] == mc_returns.shape[1] == 1

        if not self._initialized:
            self._init_storage(obs.shape[1], actions.shape[1])

        dataset_len = obs.shape[0]

        self._size = min(self._max_size, dataset_len)
        self._next_idx = self._size % self._max_size
        self._total_steps = 0

        self._observations[:self._size, :] = obs[-self._size:, :]
        self._next_observations[:self._size, :] = next_obs[-self._size:, :]
        self._actions[:self._size, :] = actions[-self._size:, :]
        self._rewards[:self._size] = rewards[-self._size:]
        self._dones[:self._size] = dones[-self._size:]
        self._mc_returns[:self._size] = mc_returns[-self._size:]

    def add_sample(self, observation, action, reward, next_observation, done, mc_returns):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = done
        self._mc_returns[self._next_idx] = mc_returns

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def sample(self, batch_size):
        indices = np.random.randint(self._size, size=batch_size)
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
            mc_returns=self._mc_returns[indices, ...],
        )

    @property
    def total_steps(self):
        return self._total_steps


# dataset functions
def subsample_batch(batch, size):
    indices = np.random.randint(batch['rewards'].shape[0], size=size)

    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices]

    return indexed


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def calc_mc(rewards, terminals, gamma, is_sparse, reward_neg, reward_scale, reward_bias):
    if len(rewards) == 0:
        return np.array([])

    reward_neg = reward_neg * reward_scale + reward_bias

    if is_sparse and np.allclose(np.array(rewards) - reward_neg, 0):
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


def get_hand_dataset_with_mc_calculation(env_name, gamma, add_expert_demos=True, add_bc_demos=True, reward_scale=1.0,
                                         reward_bias=0.0, pos_ind=-1, clip_action=None):
    assert env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", "pen-binary", "door-binary",
                        "relocate-binary"]

    expert_demo_paths = {
        "pen-binary-v0": "demonstrations/offpolicy_hand_data/pen2_sparse.npy",
        "door-binary-v0": "demonstrations/offpolicy_hand_data/door2_sparse.npy",
        "relocate-binary-v0": "demonstrations/offpolicy_hand_data/relocate2_sparse.npy",
    }

    bc_demo_paths = {
        "pen-binary-v0": "demonstrations/offpolicy_hand_data/pen_bc_sparse4.npy",
        "door-binary-v0": "demonstrations/offpolicy_hand_data/door_bc_sparse4.npy",
        "relocate-binary-v0": "demonstrations/offpolicy_hand_data/relocate_bc_sparse4.npy",
    }

    def truncate_traj(env_name, dataset, i, reward_scale, reward_bias, gamma, start_index=None, end_index=None):
        """
        This function truncates the i'th trajectory in dataset from start_index to end_index.
        Since in Adroit-binary datasets, we have trajectories like [-1, -1, -1, -1, 0, 0, 0, -1, -1] which transit from neg -> pos -> neg,
        we truncate the trajcotry from the beginning to the last positive reward, i.e., [-1, -1, -1, -1, 0, 0, 0]
        """
        is_sparse = ENV_CONFIG["adroit-binary"]["is_sparse"]
        reward_pos = ENV_CONFIG["adroit-binary"]["reward_pos"]
        reward_neg = ENV_CONFIG["adroit-binary"]["reward_neg"]

        observations = np.array(dataset[i]["observations"])[start_index:end_index]
        next_observations = np.array(dataset[i]["next_observations"])[start_index:end_index]
        rewards = dataset[i]["rewards"][start_index:end_index]
        dones = (rewards == reward_pos)
        rewards = rewards * reward_scale + reward_bias
        actions = np.array(dataset[i]["actions"])[start_index:end_index]
        mc_returns = calc_mc(rewards, dones, gamma, is_sparse, reward_neg, reward_scale, reward_bias)

        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            mc_returns=mc_returns,
        )

    dataset_list = []
    dataset_bc_list = []
    if add_expert_demos:
        print("loading expert demos from:", expert_demo_paths[env_name])
        dataset = np.load(expert_demo_paths[env_name], allow_pickle=True)

        for i in range(len(dataset)):
            N = len(dataset[i]["observations"])
            for j in range(len(dataset[i]["observations"])):
                dataset[i]["observations"][j] = dataset[i]["observations"][j]['state_observation']
                dataset[i]["next_observations"][j] = dataset[i]["next_observations"][j]['state_observation']
            if np.array(dataset[i]["rewards"]).shape != np.array(dataset[i]["terminals"]).shape:
                dataset[i]["rewards"] = dataset[i]["rewards"][:N]

            if clip_action:
                dataset[i]["actions"] = np.clip(dataset[i]["actions"], -clip_action, clip_action)

            assert np.array(dataset[i]["rewards"]).shape == np.array(dataset[i]["terminals"]).shape
            dataset[i].pop('terminals', None)

            if not (0 in dataset[i]["rewards"]):
                continue

            trunc_ind = np.where(dataset[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(env_name, dataset, i, reward_scale, reward_bias, gamma, start_index=None,
                                  end_index=trunc_ind)
            dataset_list.append(d_pos)

    if add_bc_demos:
        print("loading BC demos from:", bc_demo_paths[env_name])
        dataset_bc = np.load(bc_demo_paths[env_name], allow_pickle=True)
        for i in range(len(dataset_bc)):
            dataset_bc[i]["rewards"] = dataset_bc[i]["rewards"].squeeze()
            dataset_bc[i]["dones"] = dataset_bc[i]["terminals"].squeeze()
            dataset_bc[i].pop('terminals', None)
            if clip_action:
                dataset_bc[i]["actions"] = np.clip(dataset_bc[i]["actions"], -clip_action, clip_action)

            if not (0 in dataset_bc[i]["rewards"]):
                continue
            trunc_ind = np.where(dataset_bc[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(env_name, dataset_bc, i, reward_scale, reward_bias, gamma, start_index=None,
                                  end_index=trunc_ind)
            dataset_bc_list.append(d_pos)

    dataset = np.concatenate([dataset_list, dataset_bc_list])

    print("num offline trajs:", len(dataset))
    concatenated = {}
    for key in dataset[0].keys():
        if key in ['agent_infos', 'env_infos']:
            continue
        concatenated[key] = np.concatenate([batch[key] for batch in dataset], axis=0).astype(np.float32)
    return concatenated


def _qlearning_dataset_chunks(env: gym.Env) -> dict[str, list[np.ndarray]]:
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # first process by traj
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if 'timeouts' in dataset:  # for forward compatibility
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        if final_timestep or i == N - 1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in ['actions', 'next_observations', 'observations', 'rewards', 'terminals', 'timeouts']:
                    data_[k].append(dataset[k][i])
            if 'next_observations' not in dataset.keys():
                data_['next_observations'].append(dataset['observations'][i + 1])
            episode_step += 1

        if (done_bool or final_timestep or i == N - 1) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    concatenated = {}
    for key in episodes_dict_list[0].keys():
        concatenated[key] = [batch[key].astype(np.float32) for batch in episodes_dict_list]

    return concatenated


def qlearning_dataset_mc(env_name, reward_scale, reward_bias, clip_action, gamma):
    # setup
    is_sparse = ENV_CONFIG[env_name]["is_sparse"]
    reward_neg = ENV_CONFIG[env_name]["reward_neg"]
    env = gym.make(env_name).unwrapped

    dataset_chunks = _qlearning_dataset_chunks(env)  # all float32

    # assert check
    k0 = list(dataset_chunks.keys())[0]
    for k in dataset_chunks:
        assert len(dataset_chunks[k]) == len(dataset_chunks[k0])  # same number of episodes
        for idx in range(len(dataset_chunks[k0])):  # same number of steps
            assert dataset_chunks[k][idx].shape[0] == dataset_chunks[k0][idx].shape[0]

    # rewards
    for idx, r in enumerate(dataset_chunks["rewards"]):
        dataset_chunks["rewards"][idx] = r * reward_scale + reward_bias

    # mc_returns
    dataset_chunks["mc_returns"] = []
    for idx, (r, t) in enumerate(zip(dataset_chunks["rewards"], dataset_chunks["terminals"])):
        mc = calc_mc(r, t, gamma, is_sparse, reward_neg, reward_scale, reward_bias)
        dataset_chunks["mc_returns"].append(mc)

    # action
    for idx, a in enumerate(dataset_chunks["actions"]):
        dataset_chunks["actions"][idx] = np.clip(a, -clip_action, clip_action)

    # reshape
    dataset_chunks["rewards"] = [r.reshape(-1, 1) for r in dataset_chunks["rewards"]]
    dataset_chunks["terminals"] = [r.reshape(-1, 1) for r in dataset_chunks["terminals"]]
    dataset_chunks["mc_returns"] = [r.reshape(-1, 1) for r in dataset_chunks["mc_returns"]]
    dataset_chunks["timeouts"] = [r.reshape(-1, 1) for r in dataset_chunks["timeouts"]]

    return dataset_chunks
