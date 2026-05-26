"""Match simulation via Poisson draws + penalty shootout."""
from __future__ import annotations

import numpy as np

import config


def _apply_h2h(
    lam_h: float,
    lam_a: float,
    home_team: str | None,
    away_team: str | None,
    h2h_table: dict | None,
) -> tuple[float, float]:
    """Apply H2H adjustment to lambdas. No-op when table is None or pair unknown."""
    if h2h_table is None or not home_team or not away_team:
        return lam_h, lam_a
    score = h2h_table.get((home_team, away_team), 0.0)
    if score == 0.0:
        return lam_h, lam_a
    factor = 1.0 + config.H2H_WEIGHT * score
    return max(lam_h * factor, 1e-6), max(lam_a / factor, 1e-6)


def simulate_match(
    att_h: float,
    def_h: float,
    att_a: float,
    def_a: float,
    mu: float = config.MU,
    is_knockout: bool = False,
    rng: np.random.Generator | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    h2h_table: dict | None = None,
) -> tuple[int, int, bool]:
    """
    Simulate a single match.

    Returns (home_goals, away_goals, went_to_penalties).
    In knockout mode a draw triggers a penalty shootout via Bernoulli.

    h2h_table: {(home, away): score ∈ (−1, +1)} — boosts λ_h and dampens λ_a
               when score > 0, and vice-versa.  Pass None to disable.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam_h = max(att_h * def_a * mu, 1e-6)
    lam_a = max(att_a * def_h * mu, 1e-6)
    lam_h, lam_a = _apply_h2h(lam_h, lam_a, home_team, away_team, h2h_table)

    hg = int(rng.poisson(lam_h))
    ag = int(rng.poisson(lam_a))

    if is_knockout and hg == ag:
        p_home = (att_h / def_a) / (att_h / def_a + att_a / def_h)
        p_home = float(np.clip(p_home, 0.35, 0.65))
        if rng.random() < p_home:
            return hg + 1, ag, True
        return hg, ag + 1, True

    return hg, ag, False


# ---------------------------------------------------------------------------
# Vectorised batch simulation (for Monte Carlo group stage)
# ---------------------------------------------------------------------------

def simulate_matches_vec(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate M matches × N simulations simultaneously.

    lam_h, lam_a: shape (M, N) or broadcastable.
    Returns (home_goals, away_goals) arrays of shape (M, N).
    """
    return rng.poisson(lam_h), rng.poisson(lam_a)


def outcome_probs(
    att_h: float,
    def_h: float,
    att_a: float,
    def_a: float,
    mu: float = config.MU,
    max_goals: int = config.MAX_GOALS,
    home_team: str | None = None,
    away_team: str | None = None,
    h2h_table: dict | None = None,
) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) analytically via Poisson PMF.

    h2h_table: optional H2H table; same semantics as simulate_match().
    """
    from scipy.stats import poisson as sp_poisson

    lam_h = max(att_h * def_a * mu, 1e-6)
    lam_a = max(att_a * def_h * mu, 1e-6)
    lam_h, lam_a = _apply_h2h(lam_h, lam_a, home_team, away_team, h2h_table)

    goals = np.arange(max_goals + 1)
    ph    = sp_poisson.pmf(goals, lam_h)
    pa    = sp_poisson.pmf(goals, lam_a)

    prob_matrix = np.outer(ph, pa)
    p_win  = float(np.tril(prob_matrix, -1).sum())
    p_draw = float(np.trace(prob_matrix))
    p_loss = float(np.triu(prob_matrix, 1).sum())

    total = p_win + p_draw + p_loss
    if total > 0:
        p_win  /= total
        p_draw /= total
        p_loss /= total

    return p_win, p_draw, p_loss
