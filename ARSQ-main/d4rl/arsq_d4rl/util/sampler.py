import gym
import numpy as np

from .replay_buffer import calc_mc, ENV_CONFIG


class TrajSampler(object):
    def __init__(self, env_name, gamma=0.99, reward_scale=1.0, reward_bias=0.0):
        self.env_name = env_name
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

        self._env = gym.make(env_name).unwrapped

        self.use_goal = ENV_CONFIG[env_name]["use_goal"]
        self.max_traj_length = self._env.spec.max_episode_steps

    def render(self):
        manual_render = ENV_CONFIG[self.env_name]["manual_render"]
        if manual_render:
            rgb = self._env.sim.render(width=500, height=500, mode='offscreen', device_id=0)
            rgb = rgb[::-1, :, :]
        else:
            rgb = self._env.render(mode='rgb_array')
        return rgb

    def sample(self, policy, n_trajs, deterministic=False, render_num: int = 0) \
            -> list[dict[str, np.ndarray]]:
        trajs = []

        for idx in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            # goal
            goal_achieved_list = []
            # render
            imgs = []

            # sample
            observation = self.env.reset()
            # render
            if idx < render_num:
                imgs.append(self.render())

            for _ in range(self.max_traj_length):
                action = policy.act(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)

                next_observation, reward, done, env_infos = self.env.step(action)

                # goal
                if self.use_goal:
                    goal_achieved = env_infos['goal_achieved']
                    goal_achieved_list.append(1 if goal_achieved else 0)
                    # terminate the episode when goal is achieved in Adroit envs
                    done = goal_achieved or done

                # render
                if idx < render_num:
                    imgs.append(self.render())

                observations.append(observation)
                actions.append(action)
                rewards.append(reward * self.reward_scale + self.reward_bias)
                dones.append(done)
                next_observations.append(next_observation)

                observation = next_observation

                if done:
                    break

            mc_returns = calc_mc(rewards, dones, self.gamma, True, 0.0,
                                 self.reward_scale, self.reward_bias)

            traj_dict = dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32).reshape(-1, 1),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32).reshape(-1, 1),
                mc_returns=np.array(mc_returns, dtype=np.float32).reshape(-1, 1)
            )
            assert len(traj_dict["observations"].shape) == 2
            assert len(traj_dict["actions"].shape) == 2
            assert len(traj_dict["next_observations"].shape) == 2

            # goal
            if goal_achieved_list:
                traj_dict["goal_achieved"] = np.array(goal_achieved_list, dtype=np.float32).reshape(-1, 1)

            # render
            if idx < render_num:
                traj_dict["imgs"] = np.array(imgs)

            trajs.append(traj_dict)

        return trajs

    @property
    def env(self):
        return self._env
