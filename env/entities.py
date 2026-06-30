from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Company:
    ticker:          str
    profile:         str

    mu:              float = 0.0 
    sigma:           float = 0.0

    price:           float = 100.0
    initial_price:   float = 100.0
    daily_return:    float = 0.0

    payout_ratio:       float = 0.0
    dividend_yield:     float = 0.0
    dividend_per_share: float = 0.0
    days_to_dividend:   int   = 90


@dataclass
class Investor:
    
    cash:              float = 0.0
    initial_nav:       float = 0.0
    portfolio:         Dict[str, float] = field(default_factory=dict)
    nav:               float = 0.0
    inflation_deflator: float = 1.0

    def compute_nav(self, prices: Dict[str, float]) -> float:
        total_stock_value = 0.0
        
        for ticker, quantity in self.portfolio.items():
            current_price = prices.get(ticker, 0.0)
            total_stock_value += quantity * current_price
            
        self.nav = self.cash + total_stock_value
        return self.nav

    @property
    def real_nav(self) -> float:
        return self.nav / self.inflation_deflator if self.inflation_deflator > 0 else 0.0
