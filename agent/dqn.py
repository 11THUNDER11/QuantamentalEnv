from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from agent.network import QNetwork, ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        seed:           Optional[int] = None,
        checkpoint_dir: str           = "checkpoints/",
        device:         str           = "cpu",
    ) -> None:
        self.device         = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self._sync_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.LEARNING_RATE
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer  = ReplayBuffer()

        self.total_steps: int   = 0
        self.epsilon:     float = config.EPSILON_START
        self._best_nav:   float = -np.inf

        self.loss_history:    list[float] = []
        self.reward_history:  list[float] = []
        self.nav_history:     list[float] = []
        self.epsilon_history: list[float] = []

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, config.N_ACTIONS_TOTAL))
        self.policy_net.eval()
        action = self.policy_net.best_action(obs)
        self.policy_net.train()
        return action

    def _decay_epsilon(self) -> None:
        progress     = min(self.total_steps / config.EPSILON_DECAY_STEPS, 1.0)
        self.epsilon = config.EPSILON_START + progress * (
            config.EPSILON_END - config.EPSILON_START
        )

    def train_step(self) -> Optional[float]:
        if not self.buffer.ready:
            return None

        obs_np, act_np, rew_np, next_obs_np, done_np = self.buffer.sample(
            config.BATCH_SIZE, self.rng
        )

        obs_t      = torch.from_numpy(obs_np).float().to(self.device)
        act_t      = torch.from_numpy(act_np).long().to(self.device)
        rew_t      = torch.from_numpy(rew_np).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs_np).float().to(self.device)
        done_t     = torch.from_numpy(done_np).float().to(self.device)

        with torch.no_grad():
            q_next     = self.target_net(next_obs_t)
            q_next_max = q_next.max(dim=1).values
            q_target   = rew_t + config.GAMMA_DISCOUNT * q_next_max * (1.0 - done_t)

        q_all  = self.policy_net(obs_t)
        q_pred = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.total_steps % config.TARGET_UPDATE_FREQ == 0:
            self._sync_target()

        return float(loss.item())

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, n_episodes: int = 10, reporter=None) -> None:
        print(f"\n{'═'*60}")
        print(f"  TRAINING  —  {n_episodes} episodi  |  device={self.device}")
        print(f"  OBS_DIM={config.OBS_DIM}  N_ACTIONS={config.N_ACTIONS_TOTAL}")
        print(f"  Buffer={config.REPLAY_BUFFER_SIZE}  Batch={config.BATCH_SIZE}")
        print(f"  LR={config.LEARNING_RATE}  γ={config.GAMMA_DISCOUNT}")
        print(f"  ε: {config.EPSILON_START}→{config.EPSILON_END} "
              f"in {config.EPSILON_DECAY_STEPS:,} step "
              f"(≈{config.EPSILON_DECAY_STEPS // config.MAX_STEPS} episodi)")
        print(f"{'═'*60}\n")

        for episode in range(1, n_episodes + 1):
            obs, info = env.reset(seed=episode)
            if reporter:
                reporter.reset()

            ep_reward   = 0.0
            ep_loss_sum = 0.0
            ep_loss_cnt = 0
            done        = False

            while not done:
                action = self.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, reward, next_obs, done)

                loss = self.train_step()
                self.total_steps += 1
                self._decay_epsilon()

                if loss is not None:
                    ep_loss_sum += loss
                    ep_loss_cnt += 1

                if reporter:
                    reporter.record(info, action=action)
                
                ep_reward += reward
                obs = next_obs

            final_nav = info.get("nav", 0.0)
            avg_loss  = ep_loss_sum / ep_loss_cnt if ep_loss_cnt > 0 else 0.0

            self.reward_history.append(ep_reward)
            self.nav_history.append(final_nav)
            self.epsilon_history.append(self.epsilon)
            if avg_loss > 0:
                self.loss_history.append(avg_loss)

            print(
                f"Ep {episode:>4}/{n_episodes}  "
                f"NAV={final_nav:>10.2f}  "
                f"Reward={ep_reward:>+10.4f}  "
                f"Loss={avg_loss:.5f}  "
                f"ε={self.epsilon:.4f}  "
                f"Steps={self.total_steps:,}"
            )

            if reporter:
                reporter.save_final_report(episode=episode)

            if episode % 50 == 0:
                self.save_checkpoint(tag=f"ep{episode:04d}")

            if final_nav > self._best_nav:
                self._best_nav = final_nav
                self.save_checkpoint(tag="best")
                print(f"  → Nuovo best model  (NAV={final_nav:.2f})\n")

        self._save_training_log()
        print(f"\nTraining completed. Checkpoint in: {self.checkpoint_dir}")

    def save_checkpoint(self, tag: str = "latest") -> None:
        path = os.path.join(self.checkpoint_dir, f"model_{tag}.pt")
        self.policy_net.save(path)

    @classmethod
    def load_for_eval(cls, checkpoint_path: str, device: str = "cpu") -> "DQNAgent":
        agent = cls.__new__(cls)
        agent.device         = torch.device(device)
        agent.checkpoint_dir = os.path.dirname(checkpoint_path)
        agent.rng            = np.random.default_rng()
        agent.policy_net     = QNetwork.load(checkpoint_path, device=device)
        agent.target_net     = None
        agent.buffer         = None
        agent.optimizer      = None
        agent.loss_fn        = None
        agent.epsilon        = 0.0
        agent.total_steps    = 0
        agent._best_nav      = -np.inf
        agent.loss_history    = []
        agent.reward_history  = []
        agent.nav_history     = []
        agent.epsilon_history = []
        return agent

    def _save_training_log(self) -> None:
        path = os.path.join(self.checkpoint_dir, "training_log.json")
        log  = {
            "total_steps":         self.total_steps,
            "nav_per_episode":     self.nav_history,
            "reward_per_episode":  self.reward_history,
            "loss_per_update":     self.loss_history,
            "epsilon_per_episode": self.epsilon_history,
        }
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log salvato in: {path}")
