"""
demo.py
=======
Esegue un episodio dell'agente (DQN o random) e genera un dashboard
interattivo Plotly con l'andamento di NAV, allocazione, prezzi, reward
e azioni scelte. Nessun risultato viene salvato su disco.

Compatibile con esecuzione via `!python demo.py` su Colab: il dashboard
viene salvato come file HTML autonomo e, se il notebook lo supporta,
mostrato anche inline.

Uso:
    python demo.py --model checkpoints/model_best.pt
    python demo.py --model checkpoints/model_best.pt --seed 42
    python demo.py --random --seed 123
    python demo.py --model checkpoints/model_best.pt --seed 7 --no-open
"""

from __future__ import annotations

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env.env import QuantamentalEnv
from agent.dqn import DQNAgent
from utils.reporter import ACTION_LABELS
import config

# ── Palette ───────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    0: "#2ECC71",   # ALL_IN_GROW
    1: "#3498DB",   # ALL_IN_VALU
    2: "#E74C3C",   # ALL_IN_DECL
    3: "#F39C12",   # CASH
    4: "#9B59B6",   # EQUAL_WEIGHT
}
PRICE_COLORS = {"GROW": "#27AE60", "VALU": "#2980B9", "DECL": "#C0392B"}


def run_episode(agent, env: QuantamentalEnv, seed: int) -> dict:
    """Esegue un episodio completo e raccoglie la storia step-by-step."""
    obs, info = env.reset(seed=seed)
    initial_nav = info["nav"]

    history = {
        "steps":      [],
        "navs":       [],
        "real_navs":  [],
        "rewards":    [],
        "cash":       [],
        "actions":    [],
        "prices":     {t: [] for t in config.TICKERS},
        "alloc":      {t: [] for t in config.TICKERS},
        "alloc_cash": [],
        "dividends":  [],   # step in cui sono stati pagati dividendi
    }

    done = False
    while not done:
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio = info["portfolio"]
        total_pv  = sum(portfolio.values()) + info["cash"]
        pct = lambda v: v / total_pv * 100 if total_pv > 0 else 0.0

        history["steps"].append(info["step"])
        history["navs"].append(info["nav"])
        history["real_navs"].append(info["real_nav"])
        history["rewards"].append(info["reward"])
        history["cash"].append(info["cash"])
        history["actions"].append(action)
        for t in config.TICKERS:
            history["prices"][t].append(info["prices"][t])
            history["alloc"][t].append(pct(portfolio.get(t, 0)))
        history["alloc_cash"].append(pct(info["cash"]))
        if info.get("dividends_paid"):
            history["dividends"].append(info["step"])

    history["initial_nav"] = initial_nav
    return history


def build_dashboard(history: dict, seed: int, agent_label: str) -> go.Figure:
    """Costruisce il dashboard Plotly a 5 pannelli interattivi."""
    steps = history["steps"]
    n     = len(steps)

    # Sharpe e MDD finali per il titolo
    rew_arr = np.array(history["rewards"])
    sr = float((rew_arr.mean()/rew_arr.std())*math.sqrt(config.STEPS_PER_YEAR)) \
         if rew_arr.std() > 0 else 0.0
    nav_arr = np.array(history["navs"])
    peak    = np.maximum.accumulate(nav_arr)
    mdd     = float(((nav_arr-peak)/np.where(peak>0,peak,1)).min())*100
    ret_pct = (history["navs"][-1]-history["initial_nav"])/history["initial_nav"]*100

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.32, 0.22, 0.22, 0.24],
        subplot_titles=(
            "NAV nel tempo (nominale vs reale)",
            "Allocazione del capitale (%)",
            "Prezzi titoli (base 100)",
            "Reward giornaliera e azione scelta",
        ),
    )

    # ── Pannello 1: NAV ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=steps, y=history["navs"], name="NAV nominale",
        line=dict(color="#E8E8FF", width=2),
        hovertemplate="Step %{x}<br>NAV: %{y:,.0f} €<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=history["real_navs"], name="NAV reale",
        line=dict(color="#888899", width=1.4, dash="dash"),
        hovertemplate="Step %{x}<br>NAV reale: %{y:,.0f} €<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=history["initial_nav"], line=dict(color="#555577", dash="dot"),
                  row=1, col=1)
    for d in history["dividends"]:
        fig.add_vline(x=d, line=dict(color="#F39C12", width=1, dash="dot"),
                      opacity=0.5, row=1, col=1)

    # ── Pannello 2: Allocazione stacked ───────────────────────────────────────
    for t in config.TICKERS:
        fig.add_trace(go.Scatter(
            x=steps, y=history["alloc"][t], name=t,
            stackgroup="alloc", line=dict(width=0.5, color=PRICE_COLORS[t]),
            fillcolor=PRICE_COLORS[t],
            hovertemplate=f"{t}: " + "%{y:.1f}%<extra></extra>",
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=history["alloc_cash"], name="CASH",
        stackgroup="alloc", line=dict(width=0.5, color="#F39C12"),
        fillcolor="#F39C12",
        hovertemplate="CASH: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── Pannello 3: Prezzi normalizzati ───────────────────────────────────────
    for t in config.TICKERS:
        base = history["prices"][t][0] or 1.0
        norm = [p / base * 100 for p in history["prices"][t]]
        fig.add_trace(go.Scatter(
            x=steps, y=norm, name=f"{t} (prezzo)",
            line=dict(color=PRICE_COLORS[t], width=1.6),
            hovertemplate=f"{t}: " + "%{y:.1f}<extra></extra>",
        ), row=3, col=1)
    fig.add_hline(y=100, line=dict(color="#555577", dash="dot"), row=3, col=1)

    # ── Pannello 4: Reward colorata + marker azione ───────────────────────────
    rew_colors = ["#2ECC71" if r >= 0 else "#E74C3C" for r in history["rewards"]]
    fig.add_trace(go.Bar(
        x=steps, y=history["rewards"], name="Reward",
        marker_color=rew_colors,
        hovertemplate="Step %{x}<br>Reward: %{y:.5f}<extra></extra>",
    ), row=4, col=1)

    # Markers azione sopra le barre, colorati per tipo
    action_y_offset = max(abs(r) for r in history["rewards"]) * 1.3 or 0.01
    for a_idx in range(config.N_ACTIONS_TOTAL):
        mask = [i for i, a in enumerate(history["actions"]) if a == a_idx]
        if not mask:
            continue
        fig.add_trace(go.Scatter(
            x=[steps[i] for i in mask],
            y=[action_y_offset] * len(mask),
            mode="markers",
            name=ACTION_LABELS[a_idx],
            marker=dict(color=ACTION_COLORS[a_idx], size=5, symbol="square"),
            hovertemplate=f"{ACTION_LABELS[a_idx]}<br>Step " + "%{x}<extra></extra>",
        ), row=4, col=1)

    # ── Layout generale ───────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=950,
        title=dict(
            text=(f"<b>Alpha Demo</b> — {agent_label}  |  Seed {seed}<br>"
                  f"<sup>Rendimento: {ret_pct:+.2f}%  |  Sharpe: {sr:+.3f}  |  "
                  f"Max DD: {mdd:+.2f}%</sup>"),
            font=dict(size=16),
            x=0.01, xanchor="left",
        ),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        margin=dict(t=90, b=40, l=60, r=140),
    )
    fig.update_xaxes(title_text="Step (giorno)", row=4, col=1)
    fig.update_yaxes(title_text="Valore (€)",   row=1, col=1)
    fig.update_yaxes(title_text="%",            row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Prezzo norm.", row=3, col=1)
    fig.update_yaxes(title_text="Reward",       row=4, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Demo episodio con dashboard interattivo")
    parser.add_argument("--model",   type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--random",  action="store_true",
                        help="Usa agente random invece del DQN")
    parser.add_argument("--out",     type=str, default="demo_dashboard.html",
                        help="Percorso del file HTML generato")
    parser.add_argument("--no-open", action="store_true",
                        help="Non tentare di aprire/mostrare il dashboard automaticamente")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else \
           int(np.random.default_rng().integers(1000, 9999))

    env = QuantamentalEnv()
    if args.random:
        class RandomAgent:
            rng = np.random.default_rng(seed + 1)
            def select_action(self, obs, training=False):
                return int(self.rng.integers(0, config.N_ACTIONS_TOTAL))
        agent       = RandomAgent()
        agent_label = "Random Baseline"
    else:
        agent       = DQNAgent.load_for_eval(args.model)
        agent_label = f"DQN ({args.model})"

    print(f"Esecuzione episodio — seed={seed}  agente={agent_label}")
    history = run_episode(agent, env, seed=seed)
    print(f"Episodio completato: {len(history['steps'])} step  "
          f"NAV {history['initial_nav']:,.0f} → {history['navs'][-1]:,.0f}")

    fig = build_dashboard(history, seed=seed, agent_label=agent_label)

    # Salva sempre il file HTML autonomo (funziona ovunque, anche offline)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Dashboard salvato in: {args.out}")

    if not args.no_open:
        # Su Colab/Jupyter: prova a mostrarlo inline nella cella corrente.
        # Se eseguito come script standalone (!python demo.py), questo
        # non avrà effetto visibile nella cella perché siamo in un
        # sottoprocesso — in tal caso usare il file HTML salvato sopra,
        # oppure chiamare run_episode()+build_dashboard() direttamente
        # in una cella Python (non con !python).
        try:
            fig.show()
        except Exception:
            pass

    return fig, history


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Versione LIVE per notebook (Colab/Jupyter): aggiorna il grafico step-by-step
# usando plotly.graph_objects.FigureWidget, che si ridisegna automaticamente
# nella cella senza bisogno di richiamare fig.show() o salvare file.
#
# Uso (dentro una cella Colab, NON con !python):
#
#   import os; os.chdir('/content/qrl_v4/qrl_v4')
#   import sys; sys.path.insert(0, os.getcwd())
#   from demo import run_episode_live
#   from env.env import QuantamentalEnv
#   from agent.dqn import DQNAgent
#
#   env   = QuantamentalEnv()
#   agent = DQNAgent.load_for_eval("checkpoints/model_best.pt")
#   run_episode_live(agent, env, seed=123, delay=0.05)
# ─────────────────────────────────────────────────────────────────────────────

def build_live_widget(agent_label: str, seed: int) -> "go.FigureWidget":
    """Crea il FigureWidget vuoto con la stessa struttura a 4 pannelli."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.32, 0.22, 0.22, 0.24],
        subplot_titles=(
            "NAV nel tempo (nominale vs reale)",
            "Allocazione del capitale (%)",
            "Prezzi titoli (base 100)",
            "Reward giornaliera",
        ),
    )

    # Traccia 0-1: NAV nominale / reale
    fig.add_trace(go.Scatter(x=[], y=[], name="NAV nominale",
                  line=dict(color="#E8E8FF", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], name="NAV reale",
                  line=dict(color="#888899", width=1.4, dash="dash")), row=1, col=1)

    # Tracce 2-5: allocazione stacked (GROW, VALU, DECL, CASH)
    for t in config.TICKERS:
        fig.add_trace(go.Scatter(x=[], y=[], name=t, stackgroup="alloc",
                      line=dict(width=0.5, color=PRICE_COLORS[t]),
                      fillcolor=PRICE_COLORS[t]), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], name="CASH", stackgroup="alloc",
                  line=dict(width=0.5, color="#F39C12"),
                  fillcolor="#F39C12"), row=2, col=1)

    # Tracce 6-8: prezzi normalizzati
    for t in config.TICKERS:
        fig.add_trace(go.Scatter(x=[], y=[], name=f"{t} (prezzo)",
                      line=dict(color=PRICE_COLORS[t], width=1.6)), row=3, col=1)

    # Traccia 9: reward bar
    fig.add_trace(go.Bar(x=[], y=[], name="Reward",
                  marker_color="#2ECC71"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=950,
        title=dict(text=f"<b>Alpha Demo LIVE</b> — {agent_label}  |  Seed {seed}",
                  font=dict(size=16), x=0.01, xanchor="left"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                   font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=40, l=60, r=140),
    )
    fig.update_xaxes(title_text="Step (giorno)", row=4, col=1)
    fig.update_yaxes(title_text="Valore (€)", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Prezzo norm.", row=3, col=1)
    fig.update_yaxes(title_text="Reward", row=4, col=1)

    return go.FigureWidget(fig)


def run_episode_live(agent, env: QuantamentalEnv, seed: int, delay: float = 0.05):
    """
    Esegue l'episodio aggiornando il FigureWidget ad ogni step.
    Deve essere chiamata da dentro una cella Jupyter/Colab (non da script .py
    lanciato con !python), perché il widget si renderizza nel kernel del notebook.
    """
    import time
    from IPython.display import display

    obs, info = env.reset(seed=seed)
    initial_nav = info["nav"]

    fig = build_live_widget(agent_label="DQN" if hasattr(agent, "policy_net")
                            else "Random", seed=seed)
    display(fig)   # mostra il widget UNA volta; verrà aggiornato in-place

    history = {
        "steps": [], "navs": [], "real_navs": [], "rewards": [],
        "alloc": {t: [] for t in config.TICKERS}, "alloc_cash": [],
        "prices": {t: [] for t in config.TICKERS},
    }

    done = False
    while not done:
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio = info["portfolio"]
        total_pv  = sum(portfolio.values()) + info["cash"]
        pct = lambda v: v / total_pv * 100 if total_pv > 0 else 0.0

        history["steps"].append(info["step"])
        history["navs"].append(info["nav"])
        history["real_navs"].append(info["real_nav"])
        history["rewards"].append(info["reward"])
        history["alloc_cash"].append(pct(info["cash"]))
        for t in config.TICKERS:
            history["prices"][t].append(info["prices"][t])
            history["alloc"][t].append(pct(portfolio.get(t, 0)))

        steps = history["steps"]

        # Aggiorna ogni traccia in batch (evita un redraw per ogni singola riga)
        with fig.batch_update():
            fig.data[0].x = steps; fig.data[0].y = history["navs"]
            fig.data[1].x = steps; fig.data[1].y = history["real_navs"]

            for i, t in enumerate(config.TICKERS):
                fig.data[2+i].x = steps
                fig.data[2+i].y = history["alloc"][t]
            fig.data[5].x = steps
            fig.data[5].y = history["alloc_cash"]

            for i, t in enumerate(config.TICKERS):
                base = history["prices"][t][0] or 1.0
                norm = [p / base * 100 for p in history["prices"][t]]
                fig.data[6+i].x = steps
                fig.data[6+i].y = norm

            rew_colors = ["#2ECC71" if r >= 0 else "#E74C3C" for r in history["rewards"]]
            fig.data[9].x = steps
            fig.data[9].y = history["rewards"]
            fig.data[9].marker.color = rew_colors

            ret_pct = (info["nav"] - initial_nav) / initial_nav * 100
            fig.layout.title.text = (
                f"<b>Alpha Demo LIVE</b> — Seed {seed}  |  "
                f"Step {info['step']}/{config.MAX_STEPS}  |  "
                f"NAV {info['nav']:,.0f} €  ({ret_pct:+.2f}%)"
            )

        if delay > 0:
            time.sleep(delay)

    return history