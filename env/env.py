from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

import config
from env.entities import Company, Investor
from env.market_engine import update_price, pay_dividends


class QuantamentalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.action_space      = spaces.Discrete(config.N_ACTIONS_TOTAL)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.OBS_DIM,), dtype=np.float32
        )
        self.companies:    Dict[str, Company] = {}
        self.investor:     Optional[Investor] = None
        self.current_step: int = 0
        self.rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng          = np.random.default_rng(seed)
        self.current_step = 0

        self.companies = {}
        for ticker in config.TICKERS:
            pname = config.TICKER_PROFILES[ticker]
            prof  = config.COMPANY_PROFILES[pname]
            p0    = prof["initial_price"] * float(self.rng.uniform(0.95, 1.05))
            self.companies[ticker] = Company(
                ticker           = ticker,
                profile          = pname,
                mu               = prof["mu"],
                sigma            = prof["sigma"],
                price            = p0,
                initial_price    = p0,
                payout_ratio     = prof["payout_ratio"],
                dividend_yield   = prof["initial_dividend_yield"],
                days_to_dividend = config.STEPS_PER_QUARTER,
            )

        init_cash = float(self.rng.uniform(config.INITIAL_CASH_MIN, config.INITIAL_CASH_MAX))
        self.investor = Investor(
            cash               = init_cash,
            initial_nav        = init_cash,
            portfolio          = {t: 0.0 for t in config.TICKERS},
            nav                = init_cash,
            inflation_deflator = 1.0,
        )

        return self._build_obs(), self._get_info()

    
    def step(self, action: int):
        prices = {t: c.price for t, c in self.companies.items()}
        self.investor.compute_nav(prices)
        old_real_nav = self.investor.real_nav

        self.current_step += 1
        self._execute_action(action)

        for comp in self.companies.values():
            update_price(comp, self.rng)

        dividends_paid: Dict[str, float] = {}
        if self.current_step % config.STEPS_PER_QUARTER == 0:
            for comp in self.companies.values():
                paid = pay_dividends(comp, self.investor, self.rng)
                if paid > 0:
                    dividends_paid[comp.ticker] = paid

        self.investor.inflation_deflator *= (1.0 + config.DAILY_INFLATION_RATE)

        prices_now = {t: c.price for t, c in self.companies.items()}
        self.investor.compute_nav(prices_now)
        new_real_nav = self.investor.real_nav
        
        reward = (new_real_nav - old_real_nav) / old_real_nav if old_real_nav > 0 else 0.0

        terminated = False
        truncated  = self.current_step >= config.MAX_STEPS

        obs  = self._build_obs()
        info = self._get_info(dividends_paid=dividends_paid, reward=reward)

        return obs, float(reward), terminated, truncated, info
    
    def _execute_action(self, action: int) -> None:
        prices  = {t: c.price for t, c in self.companies.items()}

        total_nav = self.investor.cash
        for ticker, qty in self.investor.portfolio.items():
            total_nav += qty * prices.get(ticker, 0.0)

        self.investor.cash      = total_nav
        self.investor.portfolio = {t: 0.0 for t in config.TICKERS}

        if action == config.ACTION_CASH_IDX:
            pass

        elif action == config.ACTION_EQUALWEIGHT_IDX:
            per_company = total_nav / config.N_COMPANIES
            for ticker in config.TICKERS:
                p = prices[ticker]
                if p > 0:
                    self.investor.portfolio[ticker] = per_company / p   # qty
                    self.investor.cash -= per_company

        else:
            ticker = config.TICKERS[action]
            p = prices[ticker]
            if p > 0:
                self.investor.portfolio[ticker] = total_nav / p   # qty
                self.investor.cash = 0.0

    def _build_obs(self) -> np.ndarray:
        obs: List[float] = []
        
        prices = {t: c.price for t, c in self.companies.items()}
        for ticker in config.TICKERS:
            c  = self.companies[ticker]
            p0 = c.initial_price if c.initial_price > 0 else 1.0
            obs.append(float(np.clip(c.price / p0, 0.0, 10.0)))
            obs.append(float(np.clip(c.daily_return, -0.15, 0.15)))
            obs.append(float(np.clip(c.dividend_yield, 0.0, 0.10)))
            obs.append(c.days_to_dividend / config.STEPS_PER_QUARTER)

        inv0 = self.investor.initial_nav if self.investor.initial_nav > 0 else 1.0
        obs.append(float(np.clip(self.investor.cash / inv0, 0.0, 5.0)))
        for ticker in config.TICKERS:
            qty = self.investor.portfolio.get(ticker, 0.0)
            val = qty * prices.get(ticker, 0.0)
            obs.append(float(np.clip(val / inv0, 0.0, 10.0)))

        return np.array(obs, dtype=np.float32)

    def _get_info(self, dividends_paid=None, reward=0.0) -> Dict[str, Any]:
        prices = {t: c.price for t, c in self.companies.items()}
        portfolio_values = {
            t: self.investor.portfolio.get(t, 0.0) * prices.get(t, 0.0)
            for t in config.TICKERS
        }
        return {
            "step":           self.current_step,
            "nav":            self.investor.nav,
            "real_nav":       self.investor.real_nav,
            "cash":           self.investor.cash,
            "portfolio":      portfolio_values,
            "prices":         prices,
            "daily_returns":  {t: c.daily_return   for t, c in self.companies.items()},
            "div_yields":     {t: c.dividend_yield for t, c in self.companies.items()},
            "dividends_paid": dividends_paid or {},
            "reward":         reward,
        }

    def render(self) -> None:
        if self.render_mode != "human":
            return
        prices = {t: f"{c.price:.2f}" for t, c in self.companies.items()}
        nav    = self.investor.nav
        pv     = {t: f"{v:.0f}" for t, v in self._get_info()["portfolio"].items()}
        print(f"[Step {self.current_step:>5}]  NAV={nav:>10.2f}  Prices={prices}  Alloc={pv}")
