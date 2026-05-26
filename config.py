import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# API
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
API_BASE_URL = "https://api.football-data.org/v4"

# Match type weights (parameterables)
MATCH_WEIGHTS = {
    "FRIENDLY":   0.3,
    "QUALIFIER":  0.8,
    "GROUP":      1.2,
    "KNOCKOUT":   1.5,
    "WORLD_CUP":  2.0,
}

# Temporal decay: weight × exp(-LAMBDA_DECAY × years_elapsed)
LAMBDA_DECAY = 0.3

# Reference date for decay calculation
REFERENCE_DATE = date(2026, 6, 11)  # WC 2026 kick-off

# Global average goals per match (calibrated from international data)
MU = 2.55

# Max goals for Poisson PMF table
MAX_GOALS = 15

# Database
DB_PATH = "wc2026.db"

# Monte Carlo defaults
DEFAULT_N_SIMULATIONS = 10_000

# Training / validation windows (ISO dates)
TRAIN_START = "2022-01-01"
TRAIN_END   = "2026-05-01"
VAL_START   = "2024-01-01"
VAL_END     = "2026-05-01"

# W/D/L accuracy threshold (prob A wins > threshold → predict A wins)
WDL_THRESHOLD = 0.5

# Mixed loss weights (Poisson exact-score vs W/D/L outcome log-likelihood)
LOSS_ALPHA = 0.4   # weight of Poisson exact-score loss
LOSS_BETA  = 0.6   # weight of W/D/L outcome loss

# L2 regularisation: penalises distance from the initial coefficient vector
L2_LAMBDA = 0.3

# Max goals used in the outcome probability table (truncated bivariate Poisson)
MAX_GOALS_OUTCOME = 8

# Weak-opponent weight cap for coefficients and MLE
# Applied symmetrically to both attack and defence contributions:
#   attack  — goals scored vs weak opp count less (prevents blowout inflation)
#   defence — goals conceded from weak opp count less (prevents clean-sheet deflation)
ELO_WEAK_OPP_THRESHOLD  = 1400   # ELO below which a team is considered very weak
ELO_WEAK_OPP_CAP_FACTOR = 0.20   # max fraction of normal weight kept for such matches

# Post-training coefficient bounds: values outside these are clipped then re-normalised.
# A def_rating of 0.27 (Tunisia) or 0.34 (Ecuador) is statistically impossible —
# it would mean opponents score ~0 goals regardless of their attack rating.
ATT_MIN = 0.4
ATT_MAX = 2.5
DEF_MIN = 0.4
DEF_MAX = 2.5

# Head-to-head adjustment
H2H_SCALING = 0.5   # tanh scaling: raw_score × H2H_SCALING before tanh()
H2H_WEIGHT  = 0.15  # max lambda shift from H2H (±15% when score = ±1.0)

# Competition fetch config is defined in src/fetcher.py (COMPETITION_CONFIG).
# Free-tier string codes: "WC", "EC", "WCQ" — numeric IDs cause HTTP 400.
