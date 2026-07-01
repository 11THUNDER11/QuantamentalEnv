import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env.env import QuantamentalEnv
import config


def check_gymnasium_api():
    """Test minimo richiesto dalle linee guida del corso."""
    env = QuantamentalEnv()
    _, _ = env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    print(f"✓ API Gymnasium OK  —  reward: {reward:.6f}")


def run_random_episode(max_steps: int = 252, verbose: bool = True):
    env = QuantamentalEnv(render_mode="human")
    obs, info = env.reset(seed=42)

    assert obs.shape == (config.OBS_DIM,), f"Shape errata: {obs.shape}"
    print(f"✓ reset() OK  —  obs shape={obs.shape}  NAV={info['nav']:.2f}")
    print(f"  Azioni disponibili: {config.N_ACTIONS_TOTAL}")
    print(f"    0..{config.N_COMPANIES-1}: all-in su {config.TICKERS}")
    print(f"    {config.ACTION_CASH_IDX}: tutto in cash")
    print(f"    {config.ACTION_EQUALWEIGHT_IDX}: equal-weight")

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        if verbose and step % 30 == 0:
            env.render()

        # Verifica shape costante
        assert obs.shape == (config.OBS_DIM,)

    print(f"\n✓ {step} step completati senza errori")
    print(f"  Reward cumulata: {total_reward:.6f}")
    print(f"  NAV finale: {info['nav']:.2f}")
    print(f"  Prezzi finali: { {t: f'{p:.2f}' for t, p in info['prices'].items()} }")

    # Test dividendi al step 90
    env2 = QuantamentalEnv()
    obs2, _ = env2.reset(seed=1)
    for _ in range(89):
        env2.step(1)   # all-in VALU per accumulare azioni
    _, _, _, _, info90 = env2.step(1)
    print(f"\n✓ Step 90 (dividendi): {info90['dividends_paid']}")
    print(f"  (dividendi pagati se portafoglio non vuoto)")


if __name__ == "__main__":
    print("=" * 55)
    print("  SANITY CHECK  —  QuantamentalEnv v2")
    print("=" * 55)
    check_gymnasium_api()
    run_random_episode(max_steps=252, verbose=True)
    print("\nTutti i test passati. Ambiente pronto.")
