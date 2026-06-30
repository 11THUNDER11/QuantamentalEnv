from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config

class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config.N_ACTIONS_TOTAL),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def q_values_np(self, obs_np: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device 
        x = torch.from_numpy(obs_np).float().to(device)
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        return self.forward(x).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def best_action(self, obs_np: np.ndarray) -> int:
        return int(np.argmax(self.q_values_np(obs_np)))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "QNetwork":
        net = cls()
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        net.eval()
        return net


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = config.REPLAY_BUFFER_SIZE,
        obs_dim:  int = config.OBS_DIM,
    ) -> None:
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self._ptr     = 0
        self._size    = 0

        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity,             dtype=np.int64)
        self._rewards  = np.zeros(capacity,             dtype=np.float32)
        self._dones    = np.zeros(capacity,             dtype=np.float32)

    def push(
        self,
        obs: np.ndarray, action: int, reward: float,
        next_obs: np.ndarray, done: bool,
    ) -> None:
        i = self._ptr
        self._obs[i]      = obs
        self._next_obs[i] = next_obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._dones[i]    = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple:
        idx = rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    @property
    def ready(self) -> bool:
        return self._size >= config.BATCH_SIZE

    def __len__(self) -> int:
        return self._size
