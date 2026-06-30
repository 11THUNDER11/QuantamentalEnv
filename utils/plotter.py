from __future__ import annotations

import os
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import config
from utils.reporter import ACTION_LABELS

ACTION_COLORS: Dict[int, str] = {
    0: "#2ECC71",
    1: "#3498DB",
    2: "#E74C3C",
    3: "#F39C12",
    4: "#9B59B6",
}

ACTION_DESCRIPTIONS: Dict[int, str] = {
    0: ("ALL_IN_GROW",
        "Invests 100% of capital in GROW.\n"
        "Maximizes exposure to the asset with the highest drift (+13.4%/year).\n"
        "High volatility (σ=0.8%/day). Optimal when GROW is trending."),
    1: ("ALL_IN_VALU",
        "Invests 100% of capital in VALU.\n"
        "Stable asset (+2.6%/year, σ=0.3%/day) with a high dividend (1.5%/quarter).\n"
        "Defensive: captures dividend yield and protects against downside risk."),
    2: ("ALL_IN_DECL",
        "Invests 100% of capital in DECL.\n"
        "Asset in structural decline (-7.3%/year). Dividend initially high\n"
        "but unsustainable. A rational policy should avoid it."),
    3: ("CASH",
        "Holds 100% in cash.\n"
        "No market exposure. Suffers from inflationary erosion\n"
        "(~0.2%/month). Rational only as temporary protection."),
    4: ("EQUAL_WEIGHT",
        "Equally divides capital among the 3 assets (33% each).\n"
        "Maximum diversification. Expected return: average of the 3 drifts.\n"
        "Reduces volatility compared to concentrated positions."),
}

def plot_episode(
    step_log:   Dict[str, Any],
    episode:    int = 0,
    out_dir:    str = "reports/eval/plots",
) -> tuple[str, str]:

    os.makedirs(out_dir, exist_ok=True)
    steps_data = step_log["steps"]

    steps       = [e["step"]     for e in steps_data]
    navs        = [e["nav"]      for e in steps_data]
    real_navs   = [e["real_nav"] for e in steps_data]
    rewards     = [e["reward"]   for e in steps_data]
    actions     = [e.get("action", -1)  for e in steps_data]
    act_labels  = [e.get("action_label", "N/A") for e in steps_data]

    prices: Dict[str, List[float]] = {t: [] for t in config.TICKERS}
    for e in steps_data:
        for t in config.TICKERS:
            prices[t].append(e["prices"].get(t, 0.0))

    prices_norm: Dict[str, List[float]] = {}
    for t in config.TICKERS:
        base = prices[t][0] if prices[t][0] != 0 else 1.0
        prices_norm[t] = [p / base * 100 for p in prices[t]]

    div_steps = [e["step"] for e in steps_data if e.get("dividends_paid")]

    n_steps  = len(steps)
    act_counts: Dict[int, int] = {k: 0 for k in range(config.N_ACTIONS_TOTAL)}
    act_rewards: Dict[int, List[float]] = {k: [] for k in range(config.N_ACTIONS_TOTAL)}
    for e in steps_data:
        a = e.get("action", -1)
        if a >= 0:
            act_counts[a]  += 1
            act_rewards[a].append(e["reward"])

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Episode {episode}  —  {n_steps} steps  |  "
        f"NAV {navs[0]:.0f} → {navs[-1]:.0f}  "
        f"({(navs[-1]-navs[0])/navs[0]*100:>+.1f}%)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, navs,      color="#2C3E50", linewidth=1.8, label="Nominal NAV")
    ax1.plot(steps, real_navs, color="#7F8C8D", linewidth=1.2,
             linestyle="--", label="Real NAV (deflated)")
    for ds in div_steps:
        ax1.axvline(x=ds, color="#F39C12", linewidth=0.8, alpha=0.7, linestyle=":")
    if div_steps:
        ax1.axvline(x=div_steps[0], color="#F39C12", linewidth=0.8,
                    alpha=0.7, linestyle=":", label="Dividend")
    ax1.set_title("NAV Evolution", fontweight="bold")
    ax1.set_xlabel("Step (Day)")
    ax1.set_ylabel("Value (€)")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    PRICE_COLORS = {"GROW": "#27AE60", "VALU": "#2980B9", "DECL": "#C0392B"}
    ax2 = fig.add_subplot(gs[0, 1])
    for t in config.TICKERS:
        ax2.plot(steps, prices_norm[t], color=PRICE_COLORS[t],
                 linewidth=1.6, label=t)
    ax2.axhline(y=100, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_title("Asset Prices (Base 100)", fontweight="bold")
    ax2.set_xlabel("Step (Day)")
    ax2.set_ylabel("Normalized Price")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for a_idx in range(config.N_ACTIONS_TOTAL):
        mask = [i for i, a in enumerate(actions) if a == a_idx]
        if mask:
            ax3.scatter(
                [steps[i] for i in mask],
                [a_idx]   * len(mask),
                color  = ACTION_COLORS[a_idx],
                s      = 18,
                alpha  = 0.8,
                label  = ACTION_LABELS[a_idx],
                zorder = 3,
            )
    ax3.set_yticks(range(config.N_ACTIONS_TOTAL))
    ax3.set_yticklabels(
        [ACTION_LABELS[i] for i in range(config.N_ACTIONS_TOTAL)],
        fontsize=8
    )
    ax3.set_title("Actions Selected per Step", fontweight="bold")
    ax3.set_xlabel("Step (Day)")
    ax3.grid(True, alpha=0.2, axis="x")
    for ds in div_steps:
        ax3.axvline(x=ds, color="#F39C12", linewidth=0.8, alpha=0.5, linestyle=":")

    ax4 = fig.add_subplot(gs[1, 1])
    labels_bar = [ACTION_LABELS[i] for i in range(config.N_ACTIONS_TOTAL)]
    counts_bar = [act_counts[i]    for i in range(config.N_ACTIONS_TOTAL)]
    colors_bar = [ACTION_COLORS[i] for i in range(config.N_ACTIONS_TOTAL)]
    bars = ax4.bar(labels_bar, counts_bar, color=colors_bar, edgecolor="white",
                   linewidth=0.8, alpha=0.85)

    for bar, cnt in zip(bars, counts_bar):
        pct = cnt / n_steps * 100
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )
    """
    for i, (bar, a_idx) in enumerate(zip(bars, range(config.N_ACTIONS_TOTAL))):
        if act_rewards[a_idx]:
            mean_r = np.mean(act_rewards[a_idx]) * 100
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                f"r̄={mean_r:>+.3f}%",
                ha="center", va="bottom", fontsize=7,
                color="white", fontweight="bold"
            )
    """
    ax4.set_title("Action Distribution", fontweight="bold")
    ax4.set_ylabel("Number of Steps")
    ax4.set_xticks(range(config.N_ACTIONS_TOTAL))
    ax4.set_xticklabels(labels_bar, rotation=20, ha="right", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    png_path = os.path.join(out_dir, f"episode_{episode:04d}.png")
    fig.savefig(png_path, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    txt_path = os.path.join(out_dir, f"narrative_ep{episode:04d}.txt")
    _write_narrative(
        txt_path  = txt_path,
        episode   = episode,
        steps     = steps,
        navs      = navs,
        real_navs = real_navs,
        rewards   = rewards,
        actions   = actions,
        prices    = prices,
        act_counts   = act_counts,
        act_rewards  = act_rewards,
        div_steps    = div_steps,
        n_steps      = n_steps,
    )

    print(f"  Plot:            {png_path}")
    print(f"  Narrative Report:{txt_path}")
    return png_path, txt_path


def _write_narrative(
    txt_path:   str,
    episode:    int,
    steps:      List[int],
    navs:       List[float],
    real_navs:  List[float],
    rewards:    List[float],
    actions:    List[int],
    prices:     Dict[str, List[float]],
    act_counts: Dict[int, int],
    act_rewards:Dict[int, List[float]],
    div_steps:  List[int],
    n_steps:    int,
) -> None:

    from utils.reporter import sharpe_ratio, max_drawdown, cumulative_return

    ret      = (navs[-1] - navs[0]) / navs[0] * 100
    real_ret = (real_navs[-1] - real_navs[0]) / real_navs[0] * 100
    sr       = sharpe_ratio(rewards)
    mdd      = max_drawdown(navs) * 100

    lines: List[str] = []
    W = 62

    def sec(title: str) -> None:
        lines.append("")
        lines.append("═" * W)
        lines.append(f"  {title}")
        lines.append("═" * W)

    def sub(title: str) -> None:
        lines.append("")
        lines.append(f"  ── {title} " + "─" * max(0, W - len(title) - 6))

    lines.append("=" * W)
    lines.append(f"  NARRATIVE REPORT EPISODE {episode}")
    lines.append(f"  Generated by QuantamentalEnv — DQN Agent")
    lines.append("=" * W)

    sec("1. SUMMARY METRICS")
    lines.append(f"  Initial NAV     : {navs[0]:>12,.2f}")
    lines.append(f"  Final NAV       : {navs[-1]:>12,.2f}")
    lines.append(f"  Final Real NAV  : {real_navs[-1]:>12,.2f}")
    lines.append(f"  Return          : {ret:>+11.2f} %")
    lines.append(f"  Real Return     : {real_ret:>+11.2f} %")
    lines.append(f"  Sharpe Ratio    : {sr:>+11.4f}")
    lines.append(f"  Max Drawdown    : {mdd:>+11.2f} %")
    lines.append(f"  Total Steps     : {n_steps}")
    lines.append(f"  Dividends Collected: {len(div_steps)} payments")
    if div_steps:
        lines.append(f"  Dividend Steps  : {div_steps}")

    sec("2. ASSET PRICE EVOLUTION")
    for t in config.TICKERS:
        p0  = prices[t][0]
        pf  = prices[t][-1]
        pct = (pf - p0) / p0 * 100
        pmax = max(prices[t])
        pmin = min(prices[t])
        lines.append(
            f"  {t:<6}:  {p0:.2f} → {pf:.2f}  ({pct:>+.1f}%)  "
            f"[min={pmin:.2f}  max={pmax:.2f}]"
        )

    sec("3. ACTION ANALYSIS")
    lines.append(f"  {'Action':<22}  {'Steps':>5}  {'%':>6}  "
                 f"{'Mean Reward':>13}  {'Total Reward':>14}")
    lines.append(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*13}  {'─'*14}")

    for a_idx in range(config.N_ACTIONS_TOTAL):
        cnt  = act_counts[a_idx]
        pct  = cnt / n_steps * 100 if n_steps > 0 else 0
        rews = act_rewards[a_idx]
        r_mean  = np.mean(rews) * 100 if rews else 0.0
        r_total = np.sum(rews)  * 100 if rews else 0.0
        bar  = "█" * int(pct / 5)
        lines.append(
            f"  {ACTION_LABELS[a_idx]:<22}  {cnt:>5}  {pct:>5.1f}%  "
            f"{r_mean:>+12.4f}%  {r_total:>+13.4f}%  {bar}"
        )

    sec("4. DESCRIPTION OF AVAILABLE ACTIONS")
    for a_idx in range(config.N_ACTIONS_TOTAL):
        label, desc = ACTION_DESCRIPTIONS[a_idx]
        lines.append(f"\n  [{a_idx}] {label}")
        for line in desc.split("\n"):
            lines.append(f"      {line}")
        cnt = act_counts[a_idx]
        pct = cnt / n_steps * 100 if n_steps > 0 else 0
        rews = act_rewards[a_idx]
        r_mean = np.mean(rews) * 100 if rews else 0.0
        lines.append(f"      → In this episode: {cnt} times ({pct:.1f}%),  "
                     f"mean reward {r_mean:>+.4f}%")

    sec("5. ACTION SEQUENCE (Every 10 Steps)")
    lines.append(f"  {'Step':>5}  {'Action':<22}  {'NAV':>10}  "
                 f"{'Reward':>10}  {'GROW':>7}  {'VALU':>7}  {'DECL':>7}")
    lines.append(f"  {'─'*5}  {'─'*22}  {'─'*10}  {'─'*10}  "
                 f"{'─'*7}  {'─'*7}  {'─'*7}")

    idx_sample = list(range(0, n_steps, 10))
    if idx_sample[-1] != n_steps - 1:
        idx_sample.append(n_steps - 1)

    for idx in idx_sample:
        from utils.reporter import ACTION_LABELS as AL
        a    = actions[idx]
        step = steps[idx]
        nav  = navs[idx]
        rew  = rewards[idx] * 100
        pg   = prices["GROW"][idx]
        pv   = prices["VALU"][idx]
        pd_  = prices["DECL"][idx]
        lbl  = AL.get(a, "N/A")
        div  = " [D]" if step in div_steps else ""
        lines.append(
            f"  {step:>5}  {lbl:<22}  {nav:>10,.2f}  "
            f"{rew:>+9.4f}%  {pg:>7.2f}  {pv:>7.2f}  {pd_:>7.2f}{div}"
        )

    sec("6. POLICY INTERPRETATION")
    dominant_action = max(act_counts, key=lambda k: act_counts[k])
    dom_pct = act_counts[dominant_action] / n_steps * 100

    lines.append(f"  Dominant Action: {ACTION_LABELS[dominant_action]} "
                 f"({dom_pct:.1f}% of steps)")

    avoid_decl = act_counts[2] / n_steps * 100
    if avoid_decl < 10:
        lines.append(f"  ✓ Agent avoids DECL ({avoid_decl:.1f}% steps) — "
                     f"rational behavior given negative drift")
    elif avoid_decl > 30:
        lines.append(f"  ✗ Agent frequently uses DECL ({avoid_decl:.1f}% steps) — "
                     f"possible sub-optimality")

    cash_pct = act_counts[3] / n_steps * 100
    if cash_pct > 20:
        lines.append(f"  ⚠ High CASH utilization ({cash_pct:.1f}%) — "
                     f"suffers from inflation, potential excessive risk-aversion")

    grow_pct = act_counts[0] / n_steps * 100
    if grow_pct > 40:
        lines.append(f"  ✓ Strong preference for GROW ({grow_pct:.1f}%) — "
                     f"captures the asset with the highest drift")

    if sr > 1.0:
        lines.append(f"  ✓ Sharpe {sr:.2f} > 1.0: highly stable reward relative to risk")
    if mdd > -5.0:
        lines.append(f"  ✓ Low drawdown ({mdd:.1f}%): conservative policy")

    lines.append("")
    lines.append("═" * W)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_from_file(json_path: str, out_dir: Optional[str] = None) -> tuple[str, str]:
    with open(json_path, encoding="utf-8") as f:
        step_log = json.load(f)

    episode = step_log.get("episode", 0)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(json_path), "plots")

    return plot_episode(step_log, episode=episode, out_dir=out_dir)