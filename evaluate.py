import argparse
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from env.env import QuantamentalEnv
from agent.dqn import DQNAgent
from utils.reporter import Reporter, sharpe_ratio, max_drawdown, cumulative_return
import config


def run_episode(
    agent,
    env:          QuantamentalEnv,
    seed:         int,
    out_dir:      str,
    save_steplog: bool = False,
    plot:         bool = False,
    episode_idx:  int  = 0,
) -> dict:
    obs, info = env.reset(seed=seed)
    initial_nav = info.get("nav", 0.0)   # NAV al reset, prima del primo step

    reporter = Reporter(output_dir=out_dir)
    done      = False
    ep_reward = 0.0

    while not done:
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        reporter.record(info, action=action)

    nav_s = reporter._nav_series
    rew_s = reporter._reward_series

    if save_steplog:
        reporter.print_episode_summary(episode=episode_idx)
        log_path = reporter.save_step_log(episode=episode_idx)

        if plot and log_path and os.path.exists(log_path):
            from utils.plotter import plot_from_file
            plot_dir = os.path.join(out_dir, "plots")
            with open(log_path, encoding="utf-8") as f:
                step_log = json.load(f)
            plot_episode_from_log(step_log, episode=episode_idx, out_dir=plot_dir)

    return {
        "seed":              seed,
        "initial_nav":       initial_nav,
        "final_nav":         nav_s[-1] if nav_s else 0.0,
        "cumulative_return": cumulative_return([initial_nav] + nav_s) * 100,
        "sharpe":            sharpe_ratio(rew_s),
        "max_drawdown":      max_drawdown(nav_s) * 100,
        "total_reward":      ep_reward,
        "_reporter":         reporter,
    }


def plot_episode_from_log(
    step_log: dict,
    episode:  int,
    out_dir:  str,
) -> None:
    """Chiama il plotter e stampa i percorsi dei file generati."""
    from utils.plotter import plot_episode
    png_path, txt_path = plot_episode(step_log, episode=episode, out_dir=out_dir)
    print(f"  ✓ Grafico PNG:       {png_path}")
    print(f"  ✓ Report narrativo:  {txt_path}")


def evaluate(
    model_path:      str,
    n_episodes:      int  = 5,
    detail_episodes: int  = 0,
    plot:            bool = False,
) -> list:
    agent = DQNAgent.load_for_eval(model_path)
    env   = QuantamentalEnv()
    out   = os.path.join(config.REPORT_OUTPUT_DIR, "eval")
    results = []

    print(f"\n{'═'*55}")
    print(f"  VALUTAZIONE  |  {model_path}")
    print(f"  {n_episodes} episodi greedy (ε=0)")
    if detail_episodes > 0:
        print(f"  Step log + grafici per i primi {detail_episodes} episodi")
    print(f"{'═'*55}")

    for ep in range(1, n_episodes + 1):
        save_detail = (ep <= detail_episodes)
        r = run_episode(
            agent, env,
            seed         = 9000 + ep,
            out_dir      = out,
            save_steplog = save_detail,
            plot         = plot and save_detail,
            episode_idx  = ep,
        )
        r["episode"] = ep
        results.append(r)
        r["_reporter"].save_final_report(episode=ep)
        print(
            f"  Ep {ep:>3}  "
            f"NAV {r['initial_nav']:.0f}→{r['final_nav']:.0f}  "
            f"ret={r['cumulative_return']:>+7.2f}%  "
            f"Sharpe={r['sharpe']:>+.3f}  "
            f"MDD={r['max_drawdown']:>+.1f}%"
        )

    _save_summary(results, label="DQN", out_dir=out, fname="eval_summary.json")
    return results


def evaluate_random_baseline(
    n_episodes:      int  = 5,
    detail_episodes: int  = 0,
    plot:            bool = False,
) -> list:
    class RandomAgent:
        def __init__(self, s): self.rng = np.random.default_rng(s + 50000)
        def select_action(self, obs, training=False):
            return int(self.rng.integers(0, config.N_ACTIONS_TOTAL))

    env  = QuantamentalEnv()
    out  = os.path.join(config.REPORT_OUTPUT_DIR, "baseline")
    results = []

    print(f"\n{'─'*55}")
    print(f"  BASELINE RANDOM  |  {n_episodes} episodi")
    print(f"{'─'*55}")

    for ep in range(1, n_episodes + 1):
        seed_env    = 9000 + ep
        agent       = RandomAgent(seed_env)
        save_detail = (ep <= detail_episodes)
        r = run_episode(
            agent, env,
            seed         = seed_env,
            out_dir      = out,
            save_steplog = save_detail,
            plot         = plot and save_detail,
            episode_idx  = ep,
        )
        r["episode"] = ep
        results.append(r)
        print(f"  Ep {ep:>3}  ret={r['cumulative_return']:>+7.2f}%  Sharpe={r['sharpe']:>+.3f}")

    _save_summary(results, label="Random", out_dir=out, fname="baseline_summary.json")
    return results


def _save_summary(results: list, label: str, out_dir: str, fname: str) -> None:
    rets  = [r["cumulative_return"] for r in results]
    sh    = [r["sharpe"]            for r in results]
    mdds  = [r["max_drawdown"]      for r in results]
    summary = {
        "label":             label,
        "n_episodes":        len(results),
        "mean_return_pct":   round(float(np.mean(rets)), 3),
        "std_return_pct":    round(float(np.std(rets)),  3),
        "mean_sharpe":       round(float(np.mean(sh)),   4),
        "mean_max_drawdown": round(float(np.mean(mdds)), 3),
        "episodes": [{k: v for k, v in r.items() if k != "_reporter"} for r in results],
    }
    print(
        f"\n  ── {label} summary ──────────────────────────\n"
        f"  Rend. medio : {summary['mean_return_pct']:>+8.2f}%  (±{summary['std_return_pct']:.2f}%)\n"
        f"  Sharpe medio: {summary['mean_sharpe']:>+8.4f}\n"
        f"  MDD medio   : {summary['mean_max_drawdown']:>+8.2f}%\n"
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary salvato in: {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta un modello DQN addestrato")
    parser.add_argument("--model",           type=str, required=True)
    parser.add_argument("--episodes",        type=int, default=5)
    parser.add_argument("--baseline",        action="store_true")
    parser.add_argument("--detail-episodes", type=int, default=0,
                        dest="detail_episodes",
                        help="Per i primi N episodi: step log + traiettoria console")
    parser.add_argument("--plot",            action="store_true",
                        help="Genera PNG + report narrativo TXT per gli episodi "
                             "in --detail-episodes (richiede matplotlib)")
    args = parser.parse_args()

    dqn_results = evaluate(
        args.model,
        n_episodes      = args.episodes,
        detail_episodes = args.detail_episodes,
        plot            = args.plot,
    )

    if args.baseline:
        base_results = evaluate_random_baseline(
            n_episodes      = args.episodes,
            detail_episodes = args.detail_episodes,
            plot            = args.plot,
        )
        dqn_ret  = np.mean([r["cumulative_return"] for r in dqn_results])
        base_ret = np.mean([r["cumulative_return"] for r in base_results])
        print(f"Alpha DQN vs Random: {dqn_ret - base_ret:>+.2f}%")
