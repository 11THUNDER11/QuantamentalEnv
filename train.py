import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim

from env.env import QuantamentalEnv
from agent.dqn import DQNAgent
from agent.network import ReplayBuffer, QNetwork
from utils.reporter import Reporter
import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,   default=10)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",         type=str,   default="cpu",
                        help="cpu | cuda  (auto-detect se non specificato)")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to .pt file to resume training")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints/")
    args = parser.parse_args()

    if args.device == "cpu" and torch.cuda.is_available():
        print("CUDA available , use --device cuda to speed up training")
    device = args.device

    env      = QuantamentalEnv()
    reporter = Reporter(output_dir=config.REPORT_OUTPUT_DIR)

    if args.resume:
        print(f"Riprendendo da: {args.resume}")
        agent = DQNAgent.load_for_eval(args.resume, device=device)
        agent.buffer    = ReplayBuffer()
        agent.optimizer = optim.Adam(
            agent.policy_net.parameters(), lr=config.LEARNING_RATE
        )
        agent.loss_fn   = nn.SmoothL1Loss()
        agent.target_net = QNetwork().to(agent.device)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.target_net.eval()
        agent.epsilon        = config.EPSILON_END
        agent.checkpoint_dir = args.checkpoint_dir
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    else:
        agent = DQNAgent(
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )

    agent.train(env=env, n_episodes=args.episodes, reporter=reporter)


if __name__ == "__main__":
    main()
