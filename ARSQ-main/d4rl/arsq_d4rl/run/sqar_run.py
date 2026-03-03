import os
from pathlib import Path

import hydra
import numpy as np
import torch.random
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

import d4rl
from arsq_d4rl.alg.sqar import SQAR
from arsq_d4rl.run.base_run import print_metrics, get_time_metrics, log, stat_traj, print_metrics_str
from arsq_d4rl.util.replay_buffer import concatenate_batches, qlearning_dataset_mc, \
    ReplayBuffer
from arsq_d4rl.util.sampler import TrajSampler
from arsq_d4rl.util.util import Timer


class ARSQRunner:
    def __init__(self, cfg):
        self.cfg = cfg

        # logger
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        # random
        np.random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)

        # replay buffer
        dataset = qlearning_dataset_mc(cfg.env, cfg.reward_scale, cfg.reward_bias, cfg.clip_action, cfg.discount)

        # sort dataset by reward, ascending
        sort_idx = np.argsort([r.sum() for r in dataset["rewards"]])
        for k, v in dataset.items():
            cat = np.concatenate([v[i] for i in sort_idx], axis=0)
            assert len(cat.shape) == 2, f"{k}: {cat.shape}"
            dataset[k] = cat

        # demo reward
        pct = cfg.demo_percentile
        pct = max(pct, cfg.demo_ratio / 2)
        pct = min(pct, 100 - cfg.demo_ratio / 2)
        pct_min = pct - cfg.demo_ratio / 2
        pct_max = pct + cfg.demo_ratio / 2

        len_ds = dataset["rewards"].shape[0]
        idx_min = int(len_ds * pct_min / 100)
        idx_max = int(len_ds * pct_max / 100)

        dataset_il = {}
        for k, v in dataset.items():
            dataset_il[k] = v[idx_min:idx_max]
        demo_reward = dataset_il["rewards"].mean()

        self.buffer_il = ReplayBuffer(len_ds * 2)
        self.buffer_il.from_dataset(dataset_il)

        self.buffer_rl = ReplayBuffer(cfg.replay_buffer_size)
        if cfg.demo_only:
            self.buffer_rl.from_dataset(dataset_il)
        else:
            self.buffer_rl.from_dataset(dataset)

        print(f"buffer_rl: {self.buffer_rl._size}")
        print(f"buffer_il: {self.buffer_il._size}")
        print(f"demo_reward: {cfg.demo_percentile}: {demo_reward}")
        wandb.log({"demo_reward": demo_reward})

        self.demo_reward = demo_reward

        self.il_batch_size = int(cfg.batch_size * cfg.mixing_ratio)
        self.rl_batch_size = cfg.batch_size - self.il_batch_size

        # sampler policy
        self.eval_sampler = TrajSampler(cfg.env, gamma=cfg.discount)
        self.train_sampler = TrajSampler(cfg.env, gamma=cfg.discount,
                                         reward_scale=cfg.reward_scale, reward_bias=cfg.reward_bias)

        observation_dim = self.eval_sampler.env.observation_space.shape[0]
        action_dim = self.eval_sampler.env.action_space.shape[0]


        # policy
        self.alg = SQAR(cfg.alg, observation_dim, action_dim)

        # metrics
        self.total_grad_steps = 0

    def pretrain(self):
        cfg = self.cfg
        max_norm_return = -10000

        for epoch in range(cfg.n_pretrain_epochs):
            # train
            with Timer() as train_timer:
                for train_step in range(cfg.n_train_step_per_epoch_offline):
                    offline_batch = self.buffer_il.sample(self.il_batch_size)
                    offline_batch["demos"] = np.ones_like(offline_batch["rewards"])

                    online_batch = self.buffer_rl.sample(self.rl_batch_size)
                    online_batch["demos"] = np.zeros_like(online_batch["rewards"])

                    batch = concatenate_batches([offline_batch, online_batch])
                    train_metrics = self.alg.update(batch, is_pretrain=True)

            self.total_grad_steps += cfg.n_train_step_per_epoch_offline
            train_metrics['grad_steps'] = self.total_grad_steps
            print_metrics(train_metrics, "train", epoch, train_timer(), cfg.wandb.name)

            # eval
            do_eval = ((epoch % cfg.offline_eval_every_n_epoch == 0)
                       or (epoch == cfg.n_pretrain_epochs))
            if do_eval:
                with Timer() as eval_timer:
                    trajs = self.eval_sampler.sample(self.alg, self.cfg.eval_n_trajs, deterministic=True, render_num=1)
                    eval_metrics = stat_traj(trajs, self.eval_sampler.env,
                                             render_path=Path(f"render") / f"pt{epoch:05}")
                    if "average_normalized_return" in eval_metrics:
                        max_norm_return = max(max_norm_return, eval_metrics["average_normalized_return"])
                        eval_metrics["max_average_normalized_return"] = max_norm_return
                # save
                self.alg.save(Path(f"glob/pt_{epoch:05}-r_{eval_metrics['average_return']:.1f}.pt"))
                print_metrics(eval_metrics, "eval", epoch, eval_timer(), cfg.wandb.name)
            else:
                eval_metrics = None

            # log
            time_metrics = get_time_metrics(None, train_timer, eval_timer)
            log(epoch, 0, None, train_metrics, eval_metrics, time_metrics)

        # end
        wandb.log({"pt_max_norm_return": max_norm_return})

    def finetune(self):
        cfg = self.cfg
        max_norm_return = -10000

        online_eval_cnt = -1
        for epoch in range(100000):
            # rollouts
            with Timer() as rollout_timer:
                trajs = self.train_sampler.sample(self.alg, n_trajs=cfg.n_online_traj_per_epoch, deterministic=False)
                for traj in trajs:
                    for i in range(len(traj["rewards"])):
                        self.buffer_rl.add_sample(
                            traj["observations"][i], traj["actions"][i], traj["rewards"][i],
                            traj["next_observations"][i], traj["dones"][i], traj["mc_returns"][i])
                    if self.cfg.demo_relabel and traj["rewards"].sum() >= self.demo_reward:
                        for i in range(len(traj["rewards"])):
                            self.buffer_il.add_sample(
                                traj["observations"][i], traj["actions"][i], traj["rewards"][i],
                                traj["next_observations"][i], traj["dones"][i], traj["mc_returns"][i])

                expl_metrics = stat_traj(trajs, self.eval_sampler.env)
                expl_metrics["demo_steps"] = self.buffer_il.total_steps
                expl_metrics["buffer_il"] = self.buffer_il._size
                expl_metrics["buffer_rl"] = self.buffer_rl._size
            print_metrics(expl_metrics, "expl", epoch, rollout_timer(), cfg.wandb.name)

            # train
            with Timer() as train_timer:
                n_train_step_per_epoch = np.sum([len(t["rewards"]) for t in trajs]) * cfg.online_utd_ratio

                for train_step in range(n_train_step_per_epoch):
                    offline_batch = self.buffer_il.sample(self.il_batch_size)
                    offline_batch["demos"] = np.ones_like(offline_batch["rewards"])

                    online_batch = self.buffer_rl.sample(self.rl_batch_size)
                    online_batch["demos"] = np.zeros_like(online_batch["rewards"])

                    batch = concatenate_batches([offline_batch, online_batch])

                    train_metrics = self.alg.update(batch, is_pretrain=False)
            self.total_grad_steps += n_train_step_per_epoch
            # train_metrics["mixing_ratio"] = cfg.mixing_ratio
            train_metrics['grad_steps'] = self.total_grad_steps
            print_metrics(train_metrics, "train", epoch, train_timer(), cfg.wandb.name)

            # eval
            do_eval = ((self.buffer_rl.total_steps // cfg.online_eval_every_n_env_steps > online_eval_cnt)
                       or (self.buffer_rl.total_steps >= cfg.max_online_env_steps))
            if do_eval:
                with Timer() as eval_timer:
                    trajs = self.eval_sampler.sample(self.alg, self.cfg.eval_n_trajs, deterministic=True, render_num=1)
                    eval_metrics = stat_traj(trajs, self.eval_sampler.env,
                                             render_path=Path(f"render") / f"ft{epoch:05}")
                    if "average_normalized_return" in eval_metrics:
                        max_norm_return = max(max_norm_return, eval_metrics["average_normalized_return"])
                        eval_metrics["max_average_normalized_return"] = max_norm_return
                self.alg.save(Path(f"glob/ft_{epoch:05}-r_{eval_metrics['average_return']:.1f}.pt"))
                online_eval_cnt = self.buffer_rl.total_steps // cfg.online_eval_every_n_env_steps
                print_metrics(eval_metrics, "eval", epoch, eval_timer(), cfg.wandb.name)
            else:
                eval_metrics = None

            # log
            time_metrics = get_time_metrics(rollout_timer, train_timer, eval_timer)
            log(epoch, self.buffer_rl.total_steps, expl_metrics, train_metrics, eval_metrics, time_metrics)

            # stop
            if self.buffer_rl.total_steps >= cfg.max_online_env_steps:
                print("Finished Training")
                break


@hydra.main(config_path="../cfgs", config_name="sqar", version_base=None)
def main(cfg):
    os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    runner = ARSQRunner(cfg)

    if cfg.model_path:
        runner.alg.load(cfg.model_path)
    runner.pretrain()

    if cfg.max_online_env_steps > 0:
        runner.finetune()


if __name__ == '__main__':
    print("d4rl version:", d4rl.__version__)
    main()
