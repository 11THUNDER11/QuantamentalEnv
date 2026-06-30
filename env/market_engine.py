from __future__ import annotations
import math
import numpy as np
from env.entities import Company, Investor
import config


def update_price(company: Company, rng: np.random.Generator) -> None:
    eps               = rng.standard_normal()
    log_return        = (company.mu - 0.5 * company.sigma ** 2) + company.sigma * eps
    old_price         = company.price
    company.price     = old_price * math.exp(log_return)
    company.daily_return      = (company.price - old_price) / old_price
    company.days_to_dividend  = max(company.days_to_dividend - 1, 0)


def pay_dividends(
    company: Company,
    investor: Investor,
    rng: np.random.Generator,
) -> float:
    
    noise_factor = max(1.0 + rng.normal(0.0, config.DIVIDEND_NOISE_STD), 0.0)
    company.dividend_per_share = company.price * company.dividend_yield * noise_factor
    company.days_to_dividend   = config.STEPS_PER_QUARTER

    qty = investor.portfolio.get(company.ticker, 0.0)   # quantità di azioni
    if qty <= 0 or company.price <= 0:
        return 0.0

    payment = qty * company.dividend_per_share
    investor.cash += payment
    return payment
