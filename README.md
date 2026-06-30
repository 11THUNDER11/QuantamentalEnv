# QuantamentalEnv — Reinforcement Learning for Financial Asset Allocation

A custom single-agent [Gymnasium](https://gymnasium.farama.org/) environment simulating a stochastic financial market, paired with a from-scratch Deep Q-Network (DQN) agent trained to learn an optimal asset allocation policy.

Course project — *Autonomous and Adaptive Systems*, University of Bologna (UNIBO), 2025-26.

## Overview

The agent manages a portfolio across three synthetic assets with distinct lifecycle profiles a high-growth stock, a stable dividend-paying value stock, and a structurally declining stock under monthly inflation erosion and quarterly dividend distributions. The objective is to maximize real (inflation-adjusted) net asset value over a one-year trading horizon.

The DQN implementation separates concerns deliberately: the Q-network is built with PyTorch, while the experience replay buffer, the epsilon-greedy exploration schedule, and the full training loop (Bellman targets, target network synchronization, gradient clipping) are implemented from scratch in NumPy and plain Python.

## Project Structure

```
QuantamentalEnv/
├── config.py              # Single source of truth for all hyperparameters
├── train.py                # Training entry point (CLI)
├── evaluate.py              # Evaluation entry point (CLI)
├── demo.py                  # Interactive Plotly dashboard for a single episode
├── requirements.txt
├── QuantamentalEnv_report.pdf      # Report of the project
│
├── env/
│   ├── entities.py          # Company and Investor dataclasses
│   ├── market_engine.py     # GBM price update + dividend payout logic
│   └── env.py                # QuantamentalEnv (gymnasium.Env implementation)
│
├── agent/
│   ├── network.py           # QNetwork (PyTorch) + ReplayBuffer (NumPy)
│   └── dqn.py                 # DQNAgent: manual training loop, checkpointing
│
├── utils/
│   ├── reporter.py            # Sharpe ratio, max drawdown, JSON episode reports
│   └── plotter.py             # Matplotlib 4-panel PNG + narrative TXT report
│
├── scripts/
│   └── check_env.py           # Environment sanity check
│
├── checkpoints/               # Model weights saved during training (.pt files)
└── reports/                   # Generated training/eval reports, plots, logs
    ├── eval/
    └── baseline/
```

## MDP Formulation

| Component | Description |
|---|---|
| **Observation** | 18-dimensional vector: 2 macro features (time, inflation) + 4 features × 3 assets (normalized price, daily return, dividend yield, days to next dividend) + 4 portfolio features (cash ratio + allocation per asset) |
| **Action** | `Discrete(5)`: go all-in on GROW, VALU, or DECL; hold cash; or split equally across the three assets |
| **Reward** | Step-wise percentage change in real (inflation-deflated) net asset value |
| **Horizon** | 252 steps (one simulated trading year) |
| **Price dynamics** | Geometric Brownian Motion: `P(t+1) = P(t) · exp((μ - σ²/2) + σ·ε)`, `ε ~ N(0,1)` |
| **Inflation** | Fixed monthly rate applied to cash holdings every 30 steps |
| **Dividends** | Paid every 90 steps, proportional to shares held, with ±10% multiplicative noise |

### Asset Profiles

| Ticker | Profile | Daily μ | Daily σ | Payout Ratio | Initial Yield |
|---|---|---|---|---|---|
| GROW | High-Growth | +0.18% | 0.5% | 10% | 0.2% |
| VALU | Stable-Value | +0.005% | 0.2% | 60% | 2.5% |
| DECL | Corporate-Decline | -0.15% | 0.4% | 80% | 1.0% |

## Installation

```bash
git clone https://github.com/11THUNDER11/QuantamentalEnv
cd QuantamentalEnv
pip install -r requirements.txt
```

For Google Colab, run the same `pip install` command in a notebook cell prefixed with `!`.

## Usage

### 1. Verify the environment

```bash
python -m scripts.check_env
```

### 2. Train the agent

```bash
python train.py --episodes 2000 --seed 42 --device cuda
```

Key arguments: `--episodes` (number of training episodes), `--seed` (reproducibility), `--device` (`cpu` or `cuda`), `--checkpoint-dir` (default `checkpoints/`), `--resume` (path to a checkpoint to continue training from).

Checkpoints are saved every 50 episodes as `checkpoints/model_epNNNN.pt`, plus a running `checkpoints/model_best.pt` tracking the highest NAV ever reached, and `checkpoints/training_log.json` with per-episode NAV, reward, loss, and epsilon history.

### 3. Evaluate the trained model

```bash
python evaluate.py --model checkpoints/model_best.pt --episodes 6000 --detail-episodes 200 --plot --baseline
```

- `--episodes`: number of evaluation episodes (greedy policy, ε = 0), run on seeds never seen during training.
- `--detail-episodes N`: for the first N episodes, additionally saves a full step-by-step JSON log (prices, NAV, action taken at every step).
- `--plot`: generates, for each detailed episode, a 4-panel PNG (NAV trajectory, normalized asset prices, action scatter plot, action distribution) plus a narrative TXT report interpreting the agent's policy.
- `--baseline`: also runs an equal-size evaluation with a uniformly random policy and reports the alpha (DQN return minus random return) at the end.

Output is written to `reports/eval/` (DQN) and `reports/baseline/` (random agent), including `eval_summary.json` / `baseline_summary.json` with aggregate statistics (mean return, Sharpe ratio, max drawdown).

### 4. Interactive live demo

For a single episode visualized as an interactive, real-time updating dashboard (NAV, capital allocation, asset prices, and reward — all live), run the following directly inside a Jupyter/Colab cell (not via `!python`, since the widget must render in the notebook kernel):

```python
import sys

for mod_name in list(sys.modules.keys()):
    if mod_name == "demo" or mod_name.startswith("demo."):
        del sys.modules[mod_name]

import os; os.chdir('/content/QuantamentalEnv')
import sys; sys.path.insert(0, os.getcwd())

from demo import run_episode_live
from env.env import QuantamentalEnv
from agent.dqn import DQNAgent

env   = QuantamentalEnv()
agent = DQNAgent.load_for_eval("checkpoints/model_best.pt")
history = run_episode_live(agent, env, seed=10000, delay=0.1)
```

Adjust `seed` to replay a specific market scenario, and `delay` (seconds between steps) to control playback speed — set it to `0` for the fastest possible run.

For a non-live, fully interactive Plotly dashboard saved as a standalone HTML file (useful when running `demo.py` as a script via `!python`), use:

```bash
python demo.py --model checkpoints/model_best.pt --seed 123
```

This produces `demo_dashboard.html`, viewable in any browser without needing Python installed.

## Requirements

See `requirements.txt`. Core dependencies: `gymnasium`, `numpy`, `torch`, `matplotlib`, `plotly`, `anywidget` (the latter required only for the live interactive demo).
