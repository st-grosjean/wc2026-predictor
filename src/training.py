"""
Dixon-Coles MLE calibration — mixed Poisson + outcome loss with L2 regularisation.

Loss  = α × L_poisson  +  β × L_outcome  +  λ_L2 × ‖params − params_init‖²

L_poisson  = −Σ_k w_k [hg_k·log(λ_h) − λ_h + ag_k·log(λ_a) − λ_a]
L_outcome  = −Σ_k w_k log P(actual_outcome_k)   (W/D/L from bivariate Poisson)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

import config
from src.coefficients import match_weight, load_full_elo_map

logger = logging.getLogger(__name__)

_TEAMS_PATH = Path("data/teams.json")
_DEFAULT_ELO = 1750
_ELO_PRIOR_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Precomputed tables for the bivariate Poisson outcome model
# ---------------------------------------------------------------------------
_MAX_G = config.MAX_GOALS_OUTCOME          # 8
_GOALS = np.arange(_MAX_G + 1)            # [0 .. 8]
_LOG_FACT = np.array([math.lgamma(k + 1) for k in range(_MAX_G + 1)])

# Outcome masks, shape (9, 9) — home index i, away index j
_I = _GOALS[:, None].astype(float)        # (9, 1)
_J = _GOALS[None, :].astype(float)        # (1, 9)
_WIN_MASK  = (_I > _J)                    # home scores more
_DRAW_MASK = (_I == _J)
_LOSS_MASK = (_I < _J)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_in_window(date_str: str, start: str, end: str) -> bool:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return datetime.fromisoformat(start) <= d <= datetime.fromisoformat(end)
    except ValueError:
        return False


def _filter_matches(matches: list[dict], start: str, end: str) -> list[dict]:
    return [m for m in matches if _date_in_window(m.get("date", ""), start, end)]


def _build_arrays(
    matches: list[dict],
    teams: list[str],
    custom_weights: dict | None,
    lambda_decay: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (home_idx, away_idx, home_goals, away_goals, weights).

    Match weights are scaled by the geometric mean of the two teams' ELO
    relative to the field average — consistent with compute_coefficients().
    A match France vs Argentina counts more than Haiti vs Curaçao.
    Uses 3-tier ELO lookup (WC2026 teams → all FIFA → confederation avg).
    """
    t2i = {t: i for i, t in enumerate(teams)}

    # 3-tier ELO lookup shared with compute_coefficients()
    try:
        elo_map_raw, elo_mean = load_full_elo_map()
        elo_map: dict[str, float] = {k: float(v) for k, v in elo_map_raw.items()}
    except Exception:
        elo_map = {}
        elo_mean = float(_DEFAULT_ELO)

    rows: list[tuple] = []
    for m in matches:
        h, a = m.get("home_team", ""), m.get("away_team", "")
        if h not in t2i or a not in t2i:
            continue
        hg, ag = m.get("home_goals"), m.get("away_goals")
        if hg is None or ag is None:
            continue
        w = match_weight(m, custom_weights, lambda_decay)
        if w <= 0:
            continue
        # ELO scaling: geometric mean of both sides' relative strength
        elo_h = elo_map.get(h, elo_mean)
        elo_a = elo_map.get(a, elo_mean)
        elo_factor = math.sqrt((elo_h / elo_mean) * (elo_a / elo_mean))
        w *= elo_factor
        # Weak-opponent cap: if either team is very weak, cap the match weight
        _thresh = config.ELO_WEAK_OPP_THRESHOLD
        _cap    = config.ELO_WEAK_OPP_CAP_FACTOR
        if elo_h < _thresh or elo_a < _thresh:
            w *= _cap
        rows.append((t2i[h], t2i[a], int(hg), int(ag), w))
    if not rows:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    arr = np.array(rows, dtype=float)
    return (
        arr[:, 0].astype(int),
        arr[:, 1].astype(int),
        arr[:, 2].astype(int),
        arr[:, 3].astype(int),
        arr[:, 4],
    )


# ---------------------------------------------------------------------------
# Vectorised bivariate Poisson outcome model
# ---------------------------------------------------------------------------

def _poisson_pmf_table(lam: np.ndarray) -> np.ndarray:
    """
    Compute Poisson PMF for k = 0 … _MAX_G.
    lam : (N,)  →  returns (N, _MAX_G+1)
    """
    lam_c = np.maximum(lam[:, None], 1e-9)              # (N, 1)
    log_pmf = (
        -lam_c
        + _GOALS[None, :] * np.log(lam_c)
        - _LOG_FACT[None, :]
    )                                                    # (N, 9)
    return np.exp(log_pmf)


def _outcome_stats(lam_h: np.ndarray, lam_a: np.ndarray):
    """
    Vectorised computation of outcome probabilities and the moments needed
    for the analytic gradient.

    The bivariate Poisson is truncated at MAX_GOALS_OUTCOME goals, so the
    raw probabilities P_W + P_D + P_L = Z ≤ 1.  We normalise so they sum to
    1 and adjust the gradient formula accordingly.

    Returns
    -------
    P_norm    : (N, 3)  — normalised [P_win, P_draw, P_loss]  (sum = 1)
    M_h_out   : (N, 3)  — Σ_{(i,j)∈S} i·p_h(i)·p_a(j)  (un-normalised)
    M_a_out   : (N, 3)  — Σ_{(i,j)∈S} j·p_h(i)·p_a(j)  (un-normalised)
    M_h_total : (N,)    — sum of M_h over all outcome sets (= M_h_W+M_h_D+M_h_L)
    M_a_total : (N,)    — idem for away
    Z         : (N,)    — normalisation constant (P_W+P_D+P_L before normalising)
    """
    ph = _poisson_pmf_table(lam_h)   # (N, 9)
    pa = _poisson_pmf_table(lam_a)   # (N, 9)

    pjoint = ph[:, :, None] * pa[:, None, :]  # (N, 9, 9)

    masks = [_WIN_MASK, _DRAW_MASK, _LOSS_MASK]
    N = len(lam_h)
    P_raw   = np.empty((N, 3))
    M_h_out = np.empty((N, 3))
    M_a_out = np.empty((N, 3))

    for s, mask in enumerate(masks):
        masked         = pjoint * mask[None, :, :]
        P_raw[:, s]    = masked.sum(axis=(1, 2))
        M_h_out[:, s]  = (masked * _I[None, :, :]).sum(axis=(1, 2))
        M_a_out[:, s]  = (masked * _J[None, :, :]).sum(axis=(1, 2))

    Z         = np.maximum(P_raw.sum(axis=1), 1e-15)   # (N,)
    M_h_total = M_h_out.sum(axis=1)                     # (N,)
    M_a_total = M_a_out.sum(axis=1)                     # (N,)
    P_norm    = P_raw / Z[:, None]                       # (N, 3), sums to 1

    return P_norm, M_h_out, M_a_out, M_h_total, M_a_total, Z


# ---------------------------------------------------------------------------
# Mixed loss and gradient
# ---------------------------------------------------------------------------

def _mixed_loss(
    params: np.ndarray,
    n: int,
    home_idx, away_idx,
    home_goals: np.ndarray, away_goals: np.ndarray,
    weights: np.ndarray,
    mu: float,
    alpha: float, beta: float, l2_lam: float,
    x_init: np.ndarray,
    true_outcome_idx: np.ndarray,
) -> float:
    att  = params[:n]
    def_ = params[n:]

    lam_h = np.maximum(att[home_idx] * def_[away_idx] * mu, 1e-9)
    lam_a = np.maximum(att[away_idx] * def_[home_idx] * mu, 1e-9)

    # --- Poisson exact-score loss ---
    ll_h = home_goals * np.log(lam_h) - lam_h
    ll_a = away_goals * np.log(lam_a) - lam_a
    poisson_loss = -np.dot(weights, ll_h + ll_a)

    # --- Outcome log-likelihood (normalised for truncation) ---
    P_norm, _, _, _, _, _ = _outcome_stats(lam_h, lam_a)
    P_norm = np.maximum(P_norm, 1e-15)
    K = np.arange(len(home_idx))
    P_actual = P_norm[K, true_outcome_idx]
    outcome_loss = -np.dot(weights, np.log(P_actual))

    # --- L2 regularisation ---
    l2_loss = l2_lam * np.sum((params - x_init) ** 2)

    return alpha * poisson_loss + beta * outcome_loss + l2_loss


def _mixed_loss_grad(
    params: np.ndarray,
    n: int,
    home_idx, away_idx,
    home_goals: np.ndarray, away_goals: np.ndarray,
    weights: np.ndarray,
    mu: float,
    alpha: float, beta: float, l2_lam: float,
    x_init: np.ndarray,
    true_outcome_idx: np.ndarray,
) -> np.ndarray:
    att  = params[:n]
    def_ = params[n:]

    lam_h = np.maximum(att[home_idx] * def_[away_idx] * mu, 1e-9)
    lam_a = np.maximum(att[away_idx] * def_[home_idx] * mu, 1e-9)

    # ---- Poisson gradient ----
    res_h = weights * (1.0 - home_goals / lam_h)   # w·(1 − hg/λ_h)
    res_a = weights * (1.0 - away_goals / lam_a)

    grad_att_p = np.zeros(n)
    grad_def_p = np.zeros(n)
    np.add.at(grad_att_p, home_idx, res_h * def_[away_idx] * mu)
    np.add.at(grad_att_p, away_idx, res_a * def_[home_idx] * mu)
    np.add.at(grad_def_p, away_idx, res_h * att[home_idx] * mu)
    np.add.at(grad_def_p, home_idx, res_a * att[away_idx] * mu)

    # ---- Outcome gradient (accounts for truncation normalisation) ----
    # When we normalise p_S = P_S_raw / Z, the gradient of −log p_S w.r.t. λ is:
    #   ∂(−log p_S)/∂λ_h = (1/λ_h) × (M_h_total/Z − M_h_S/P_S_raw)
    P_norm, M_h_out, M_a_out, M_h_total, M_a_total, Z = _outcome_stats(lam_h, lam_a)

    K = np.arange(len(home_idx))
    s = true_outcome_idx
    P_norm_act = np.maximum(P_norm[K, s], 1e-15)
    M_h = M_h_out[K, s]
    M_a = M_a_out[K, s]
    P_S_raw = np.maximum(P_norm_act * Z, 1e-15)   # un-normalised P for actual outcome

    dL_dlh = weights * (M_h_total / (lam_h * Z) - M_h / (lam_h * P_S_raw))
    dL_dla = weights * (M_a_total / (lam_a * Z) - M_a / (lam_a * P_S_raw))

    grad_att_o = np.zeros(n)
    grad_def_o = np.zeros(n)
    np.add.at(grad_att_o, home_idx, dL_dlh * def_[away_idx] * mu)
    np.add.at(grad_att_o, away_idx, dL_dla * def_[home_idx] * mu)
    np.add.at(grad_def_o, away_idx, dL_dlh * att[home_idx] * mu)
    np.add.at(grad_def_o, home_idx, dL_dla * att[away_idx] * mu)

    # ---- L2 gradient ----
    grad_l2 = 2.0 * l2_lam * (params - x_init)

    grad_att = alpha * grad_att_p + beta * grad_att_o + grad_l2[:n]
    grad_def = alpha * grad_def_p + beta * grad_def_o + grad_l2[n:]
    return np.concatenate([grad_att, grad_def])


# ---------------------------------------------------------------------------
# Metrics  (unchanged logic — uses closed-form Poisson bivariate)
# ---------------------------------------------------------------------------

def _outcome_probs(att: np.ndarray, def_: np.ndarray, home_idx, away_idx, mu, max_g=10):
    """P(home win), P(draw), P(away win) per match — used only for metrics."""
    goals = np.arange(max_g + 1)
    n = len(home_idx)
    p_win, p_draw, p_loss = np.zeros(n), np.zeros(n), np.zeros(n)

    for k, (hi, ai) in enumerate(zip(home_idx, away_idx)):
        lh = max(float(att[hi] * def_[ai] * mu), 1e-9)
        la = max(float(att[ai] * def_[hi] * mu), 1e-9)
        ph = poisson.pmf(goals, lh)
        pa = poisson.pmf(goals, la)
        for g1, p1 in enumerate(ph):
            for g2, p2 in enumerate(pa):
                prob = p1 * p2
                if g1 > g2:
                    p_win[k] += prob
                elif g1 == g2:
                    p_draw[k] += prob
                else:
                    p_loss[k] += prob

    return p_win, p_draw, p_loss


def compute_metrics(
    matches: list[dict],
    teams: list[str],
    att: np.ndarray,
    def_: np.ndarray,
    mu: float,
    threshold: float = 0.5,
) -> dict:
    """Return accuracy/MAE/Brier on a set of matches."""
    if not matches:
        return {"accuracy": None, "mae": None, "brier": None, "n": 0}

    t2i = {t: i for i, t in enumerate(teams)}
    hi_list, ai_list, hg_list, ag_list = [], [], [], []
    for m in matches:
        h, a = m.get("home_team", ""), m.get("away_team", "")
        if h not in t2i or a not in t2i:
            continue
        hg, ag = m.get("home_goals"), m.get("away_goals")
        if hg is None or ag is None:
            continue
        hi_list.append(t2i[h])
        ai_list.append(t2i[a])
        hg_list.append(int(hg))
        ag_list.append(int(ag))

    if not hi_list:
        return {"accuracy": None, "mae": None, "brier": None, "n": 0}

    hi_arr = np.array(hi_list)
    ai_arr = np.array(ai_list)
    hg_arr = np.array(hg_list)
    ag_arr = np.array(ag_list)

    p_win, p_draw, p_loss = _outcome_probs(att, def_, hi_arr, ai_arr, mu)

    true_outcome = np.sign(hg_arr - ag_arr)
    pred_outcome = np.where(p_win > threshold, 1, np.where(p_loss > threshold, -1, 0))
    accuracy = float(np.mean(pred_outcome == true_outcome))

    lh = att[hi_arr] * def_[ai_arr] * mu
    la = att[ai_arr] * def_[hi_arr] * mu
    mae = float(np.mean(np.abs(hg_arr - lh) + np.abs(ag_arr - la)))

    y1 = (true_outcome == 1).astype(float)
    y0 = (true_outcome == 0).astype(float)
    ym = (true_outcome == -1).astype(float)
    brier = float(np.mean(
        (p_win - y1) ** 2 + (p_draw - y0) ** 2 + (p_loss - ym) ** 2
    ))

    return {"accuracy": accuracy, "mae": mae, "brier": brier, "n": len(hi_list)}


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def calibrate(
    matches: list[dict],
    teams: list[str],
    initial_coeffs: dict[str, dict],
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    custom_weights: dict | None = None,
    lambda_decay: float | None = None,
    mu: float | None = None,
    l2_lambda: float | None = None,
    callback=None,
) -> dict:
    """
    Run Dixon-Coles MLE calibration with mixed loss and L2 regularisation.

    Returns {att, def, metrics_before, metrics_after, success, message, iterations}.
    """
    mu = mu or config.MU
    alpha  = config.LOSS_ALPHA
    beta   = config.LOSS_BETA
    l2_lam = l2_lambda if l2_lambda is not None else config.L2_LAMBDA
    n = len(teams)

    train_matches = _filter_matches(matches, train_start, train_end)
    val_matches   = _filter_matches(matches, val_start, val_end)

    _empty = {
        "att": {t: initial_coeffs.get(t, {}).get("att_rating", 1.0) for t in teams},
        "def": {t: initial_coeffs.get(t, {}).get("def_rating", 1.0) for t in teams},
        "metrics_before": {"accuracy": None, "mae": None, "brier": None, "n": 0},
        "metrics_after":  {"accuracy": None, "mae": None, "brier": None, "n": 0},
        "success": False,
    }

    if not train_matches:
        return {**_empty, "message": "No training matches found in window.", "iterations": 0}

    hi, ai, hg, ag, ws = _build_arrays(train_matches, teams, custom_weights, lambda_decay)
    if len(hi) == 0:
        return {**_empty, "message": "No usable training matches (teams not in roster?).", "iterations": 0}

    # True outcome index per match: 0=home win, 1=draw, 2=away win
    true_outcome_idx = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

    # Initial parameter vector (anchors for L2 regularisation)
    x0 = np.array(
        [initial_coeffs.get(t, {}).get("att_rating", 1.0) for t in teams] +
        [initial_coeffs.get(t, {}).get("def_rating", 1.0) for t in teams]
    )
    x0 = np.clip(x0, 0.1, 5.0)

    # Metrics before optimisation
    att0, def0 = x0[:n], x0[n:]
    metrics_before = compute_metrics(val_matches, teams, att0, def0, mu)

    iteration_count = [0]
    args = (n, hi, ai, hg, ag, ws, mu, alpha, beta, l2_lam, x0, true_outcome_idx)

    def obj(params):
        iteration_count[0] += 1
        if callback and iteration_count[0] % 50 == 0:
            callback(iteration_count[0])
        return _mixed_loss(params, *args)

    def jac(params):
        return _mixed_loss_grad(params, *args)

    result = minimize(
        obj,
        x0,
        jac=jac,
        method="L-BFGS-B",
        bounds=[(0.1, 5.0)] * (2 * n),
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6},
    )

    att_opt = result.x[:n]
    def_opt = result.x[n:]

    # Normalise: mean att = 1.0, mean def = 1.0
    att_opt = att_opt / att_opt.mean()
    def_opt = def_opt / def_opt.mean()

    # --- ELO prior blending for data-sparse teams ---
    try:
        elo_map_prior, elo_mean_prior = load_full_elo_map()

        for i, t in enumerate(teams):
            n_m = initial_coeffs.get(t, {}).get("n_matches", 0)
            if n_m >= _ELO_PRIOR_THRESHOLD:
                continue
            blend = (_ELO_PRIOR_THRESHOLD - n_m) / _ELO_PRIOR_THRESHOLD
            elo_t = elo_map_prior.get(t, int(elo_mean_prior))
            att_prior = elo_t / elo_mean_prior
            def_prior = elo_mean_prior / elo_t
            att_opt[i] = (1.0 - blend) * att_opt[i] + blend * att_prior
            def_opt[i] = (1.0 - blend) * def_opt[i] + blend * def_prior

        att_opt = att_opt / att_opt.mean()
        def_opt = def_opt / def_opt.mean()
    except Exception as exc:
        logger.warning("ELO regularisation skipped: %s", exc)

    # Clip, re-normalise, then clip again — the second clip silently catches the
    # tiny drift that re-normalisation introduces when outliers were clipped.
    _att_min, _att_max = config.ATT_MIN, config.ATT_MAX
    _def_min, _def_max = config.DEF_MIN, config.DEF_MAX
    att_opt = np.clip(att_opt, _att_min, _att_max)
    def_opt = np.clip(def_opt, _def_min, _def_max)
    att_opt = att_opt / att_opt.mean()
    def_opt = def_opt / def_opt.mean()
    att_opt = np.clip(att_opt, _att_min, _att_max)
    def_opt = np.clip(def_opt, _def_min, _def_max)

    metrics_after = compute_metrics(val_matches, teams, att_opt, def_opt, mu)

    return {
        "att": {t: float(att_opt[i]) for i, t in enumerate(teams)},
        "def": {t: float(def_opt[i]) for i, t in enumerate(teams)},
        "metrics_before": metrics_before,
        "metrics_after":  metrics_after,
        "success":    result.success,
        "message":    str(result.message),
        "iterations": iteration_count[0],
        "loss_config": {"alpha": alpha, "beta": beta, "l2_lambda": l2_lam},
    }
