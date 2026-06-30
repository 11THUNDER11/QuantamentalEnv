from __future__ import annotations

import os
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import config

# Mappa indice azione → etichetta leggibile
ACTION_LABELS: Dict[int, str] = {
    i: f"ALL_IN_{config.TICKERS[i]}" for i in range(config.N_COMPANIES)
}
ACTION_LABELS[config.ACTION_CASH_IDX]        = "CASH"
ACTION_LABELS[config.ACTION_EQUALWEIGHT_IDX] = "EQUAL_WEIGHT"


# ── Funzioni standalone ───────────────────────────────────────────────────────

def sharpe_ratio(returns: List[float], risk_free_daily: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr    = np.array(returns, dtype=np.float64)
    excess = arr - risk_free_daily
    std    = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * math.sqrt(config.STEPS_PER_YEAR))


def max_drawdown(nav_series: List[float]) -> float:
    if len(nav_series) < 2:
        return 0.0
    arr      = np.array(nav_series, dtype=np.float64)
    peak     = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / np.where(peak > 0, peak, 1)
    return float(drawdown.min())


def cumulative_return(nav_series: List[float]) -> float:
    if len(nav_series) < 2 or nav_series[0] == 0:
        return 0.0
    return float((nav_series[-1] - nav_series[0]) / nav_series[0])

class Reporter:

    def __init__(self, output_dir: str = config.REPORT_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._nav_series:      List[float]      = []
        self._real_nav_series: List[float]      = []
        self._reward_series:   List[float]      = []
        self._price_series:    Dict[str, List]  = {t: [] for t in config.TICKERS}
        self._steps:           List[int]        = []

        self._step_log: List[Dict[str, Any]] = []

    def record(
        self,
        info:   Dict[str, Any],
        action: Optional[int] = None,
    ) -> None:
        
        step = info.get("step", 0)
        nav  = info.get("nav",  0.0)

        self._steps.append(step)
        self._nav_series.append(nav)
        self._real_nav_series.append(info.get("real_nav", 0.0))
        self._reward_series.append(info.get("reward",   0.0))
        for t in config.TICKERS:
            self._price_series[t].append(info.get("prices", {}).get(t, 0.0))

        # Step log dettagliato
        log_entry: Dict[str, Any] = {
            "step":            step,
            "nav":             round(nav, 2),
            "real_nav":        round(info.get("real_nav", 0.0), 2),
            "reward":          round(info.get("reward",   0.0), 6),
            "cash":            round(info.get("cash",     0.0), 2),
            "prices":          {t: round(info.get("prices", {}).get(t, 0.0), 4)
                                for t in config.TICKERS},
            "daily_returns":   {t: round(info.get("daily_returns", {}).get(t, 0.0), 6)
                                for t in config.TICKERS},
            "portfolio_value": {t: round(info.get("portfolio", {}).get(t, 0.0), 2)
                                for t in config.TICKERS},
            "dividends_paid":  info.get("dividends_paid", {}),
        }
        if action is not None:
            log_entry["action"]       = int(action)
            log_entry["action_label"] = ACTION_LABELS.get(int(action), str(action))

        self._step_log.append(log_entry)

    def save_final_report(self, episode: int = 0) -> str:
        
        sr       = sharpe_ratio(self._reward_series)
        mdd      = max_drawdown(self._nav_series)
        ret      = cumulative_return(self._nav_series)
        real_ret = cumulative_return(self._real_nav_series)

        # Conteggio azioni (se disponibile)
        action_counts: Dict[str, int] = {}
        if self._step_log and "action" in self._step_log[0]:
            for label in ACTION_LABELS.values():
                action_counts[label] = 0
            for entry in self._step_log:
                lbl = entry.get("action_label", "UNKNOWN")
                action_counts[lbl] = action_counts.get(lbl, 0) + 1

        report: Dict[str, Any] = {
            "episode":     episode,
            "total_steps": self._steps[-1] if self._steps else 0,
            "metrics": {
                "initial_nav":                round(self._nav_series[0],  2) if self._nav_series else 0,
                "final_nav":                  round(self._nav_series[-1], 2) if self._nav_series else 0,
                "final_real_nav":             round(self._real_nav_series[-1], 2) if self._real_nav_series else 0,
                "cumulative_return_pct":      round(ret      * 100, 4),
                "cumulative_real_return_pct": round(real_ret * 100, 4),
                "annualized_sharpe":          round(sr,             4),
                "max_drawdown_pct":           round(mdd      * 100, 4),
                "avg_daily_reward":           round(float(np.mean(self._reward_series)), 6) if self._reward_series else 0,
            },
            "price_evolution": {
                t: {
                    "initial":    round(self._price_series[t][0],  2) if self._price_series[t] else 0,
                    "final":      round(self._price_series[t][-1], 2) if self._price_series[t] else 0,
                    "return_pct": round(cumulative_return(self._price_series[t]) * 100, 2),
                }
                for t in config.TICKERS
            },
            "action_counts": action_counts,
        }

        m = report["metrics"]
        print(
            f"\n{'═'*55}\n"
            f"  REPORT FINALE  |  Episodio {episode}\n"
            f"{'═'*55}\n"
            f"  NAV iniziale  : {m['initial_nav']:>12.2f}\n"
            f"  NAV finale    : {m['final_nav']:>12.2f}\n"
            f"  NAV reale     : {m['final_real_nav']:>12.2f}\n"
            f"  Rendimento    : {m['cumulative_return_pct']:>+11.2f} %\n"
            f"  Rend. reale   : {m['cumulative_real_return_pct']:>+11.2f} %\n"
            f"  Sharpe        : {m['annualized_sharpe']:>+11.4f}\n"
            f"  Max drawdown  : {m['max_drawdown_pct']:>+11.2f} %\n"
            f"{'─'*55}\n"
        )
        for t, pd in report["price_evolution"].items():
            print(f"  {t}: {pd['initial']:.2f} → {pd['final']:.2f}  ({pd['return_pct']:>+.1f}%)")

        if action_counts:
            total = sum(action_counts.values())
            print(f"\n  Azioni effettuate ({total} step):")
            for lbl, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100
                bar = "█" * int(pct / 5)
                print(f"    {lbl:<22} {cnt:>4}  ({pct:>5.1f}%)  {bar}")
        print(f"{'═'*55}\n")

        fname = os.path.join(self.output_dir, f"report_ep{episode:04d}.json")
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report salvato in: {fname}\n")
        return fname

    def save_step_log(self, episode: int = 0) -> str:
    
        if not self._step_log:
            return ""

        log_doc = {
            "episode":    episode,
            "n_steps":    len(self._step_log),
            "tickers":    config.TICKERS,
            "action_map": ACTION_LABELS,
            "steps":      self._step_log,
        }

        fname = os.path.join(self.output_dir, f"step_log_ep{episode:04d}.json")
        with open(fname, "w") as f:
            json.dump(log_doc, f, indent=2)
        print(f"  Step log salvato in: {fname}  ({len(self._step_log)} step)\n")
        return fname

    def print_episode_summary(self, episode: int = 0, n_sample: int = 20) -> None:
        
        if not self._step_log:
            print("  Nessun log disponibile.")
            return

        total = len(self._step_log)
        step_size = max(1, total // n_sample)
        indices   = list(range(0, total, step_size))
        if indices[-1] != total - 1:
            indices.append(total - 1)

        header = (f"{'Step':>5}  {'NAV':>10}  "
                  + "  ".join(f"{t:>8}" for t in config.TICKERS)
                  + f"  {'Azione':<22}  {'Reward':>9}")
        print(f"\n{'─'*55}")
        print(f"  TRAIETTORIA EPISODIO {episode} ({total} step totali, mostra ogni ~{step_size})")
        print(f"{'─'*55}")
        print(f"  {header}")
        print(f"  {'─'*len(header)}")

        for idx in indices:
            e      = self._step_log[idx]
            prices = "  ".join(f"{e['prices'].get(t, 0):>8.2f}" for t in config.TICKERS)
            action = e.get("action_label", "N/A")
            reward = e.get("reward", 0.0)
            div    = " [DIV]" if e.get("dividends_paid") else ""
            print(f"  {e['step']:>5}  {e['nav']:>10.2f}  {prices}  {action:<22}  {reward:>+9.6f}{div}")

        print(f"{'─'*55}\n")

    def reset(self) -> None:
        self._nav_series.clear()
        self._real_nav_series.clear()
        self._reward_series.clear()
        self._steps.clear()
        self._step_log.clear()
        for t in config.TICKERS:
            self._price_series[t].clear()
