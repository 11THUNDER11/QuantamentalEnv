MAX_STEPS: int          = 252
STEPS_PER_YEAR: int     = 252
STEPS_PER_QUARTER: int  = 90    
STEPS_PER_MONTH: int    = 30    

MONTHLY_INFLATION_RATE: float = 0.002         
DAILY_INFLATION_RATE: float   = (1 + MONTHLY_INFLATION_RATE) ** (1 / STEPS_PER_MONTH) - 1

TICKERS: list[str] = ["GROW", "VALU", "DECL"]

COMPANY_PROFILES: dict = {
    "Growth": {
        "mu":             0.0018,
        "sigma":          0.005,
        "initial_price":  100.0,
        "payout_ratio":   0.10,      
        "initial_dividend_yield": 0.002,
    },
    "Value": {
        "mu":             0.00005,
        "sigma":          0.002,
        "initial_price":  100.0,
        "payout_ratio":   0.60,
        "initial_dividend_yield": 0.025,
    },
    "Decline": {
        "mu":            -0.0015,
        "sigma":          0.004,
        "initial_price":  100.0,
        "payout_ratio":   0.80,      
        "initial_dividend_yield": 0.010,
    },
}

TICKER_PROFILES: dict[str, str] = {
    "GROW": "Growth",
    "VALU": "Value",
    "DECL": "Decline",
}

DIVIDEND_NOISE_STD: float = 0.10

INITIAL_CASH_MIN: float = 50_000.0
INITIAL_CASH_MAX: float = 150_000.0

N_COMPANIES: int      = len(TICKERS)
N_ACTIONS_TOTAL: int  = N_COMPANIES + 2

ACTION_CASH_IDX: int        = N_COMPANIES
ACTION_EQUALWEIGHT_IDX: int = N_COMPANIES + 1

N_FEATURES_PER_COMPANY: int = 4
N_FEATURES_PRIVATE: int     = 1 + N_COMPANIES
OBS_DIM: int = (
    N_FEATURES_PER_COMPANY * N_COMPANIES
    + N_FEATURES_PRIVATE
)

REPLAY_BUFFER_SIZE: int  = 100_000
BATCH_SIZE: int           = 128
LEARNING_RATE: float      = 3e-5
GAMMA_DISCOUNT: float     = 0.99
EPSILON_START: float      = 1.0
EPSILON_END: float        = 0.05
EPSILON_DECAY_STEPS: int  = 75000
TARGET_UPDATE_FREQ: int   = 1_000

REPORT_INTERVAL_STEPS: int = STEPS_PER_YEAR
REPORT_OUTPUT_DIR: str     = "reports/"
