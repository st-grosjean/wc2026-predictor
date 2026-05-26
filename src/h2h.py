"""
Head-to-head (H2H) score table.

For each pair (A, B), computes a bounded score ∈ (-1, +1) representing
A's historical advantage over B, using log-scaled goal differences with
temporal and match-type weighting.

Score formula
-------------
  For each match A vs B:
    gd_A         = goals_A − goals_B
    norm_gd      = sign(gd_A) × log(1 + |gd_A|)   # log-scale compresses blowouts
    contribution = norm_gd × temporal_weight

  raw(A, B)    = Σ contributions / Σ weights
  h2h(A, B)    = tanh(raw × H2H_SCALING)          # bounded in (−1, +1)

Antisymmetry: h2h(B, A) = −h2h(A, B)
Absence:      0.0 for pairs with no shared matches
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import config
from src.coefficients import _years_elapsed


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_h2h_table(
    matches: list[dict],
    custom_weights: dict | None = None,
    lambda_decay: float | None = None,
    reference: date | None = None,
) -> dict[tuple[str, str], float]:
    """
    Compute H2H scores for all pairs seen in match history.

    Returns {(A, B): score} covering every encountered ordered pair.
    h2h[(B, A)] = −h2h[(A, B)] is guaranteed.
    """
    weights   = custom_weights or config.MATCH_WEIGHTS
    lam       = lambda_decay if lambda_decay is not None else config.LAMBDA_DECAY
    ref       = reference or config.REFERENCE_DATE
    scaling   = config.H2H_SCALING

    # [weighted_norm_sum, weight_sum] per ordered pair
    accum: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])

    for m in matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        hg   = m.get("home_goals")
        ag   = m.get("away_goals")
        if not home or not away or hg is None or ag is None:
            continue

        base_w = weights.get(m.get("competition_type", "FRIENDLY"), 0.3)
        years  = _years_elapsed(m.get("date", ""), ref)
        w      = base_w * math.exp(-lam * years)
        if w <= 0:
            continue

        gd   = int(hg) - int(ag)
        norm = math.copysign(math.log1p(abs(gd)), gd) if gd != 0 else 0.0

        accum[(home, away)][0] += norm * w
        accum[(home, away)][1] += w
        # Antisymmetric: away sees negative gd
        accum[(away, home)][0] -= norm * w
        accum[(away, home)][1] += w

    table: dict[tuple[str, str], float] = {}
    for pair, (wsum_norm, wsum) in accum.items():
        if wsum > 0:
            table[pair] = math.tanh(wsum_norm / wsum * scaling)

    return table


# ---------------------------------------------------------------------------
# Pair summary (for the UI "H2H notable" table)
# ---------------------------------------------------------------------------

def h2h_pairs_summary(
    matches: list[dict],
    known_teams: set[str],
    h2h_table: dict[tuple[str, str], float],
    custom_weights: dict | None = None,
    lambda_decay: float | None = None,
    reference: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Return the most imbalanced H2H pairs for the UI table.

    Only includes pairs where both teams are in known_teams.

    Each row: {team_a, team_b, wins_a, draws, wins_b, weighted_gd, h2h_score, n_matches}
    where team_a < team_b alphabetically and score is from team_a's perspective.
    Sorted by |h2h_score| descending.
    """
    weights = custom_weights or config.MATCH_WEIGHTS
    lam     = lambda_decay if lambda_decay is not None else config.LAMBDA_DECAY
    ref     = reference or config.REFERENCE_DATE

    # canonical pair (a < b) → stats
    pair_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"wins_a": 0, "draws": 0, "wins_b": 0, "gd_w": 0.0, "wsum": 0.0, "n": 0}
    )

    for m in matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        hg   = m.get("home_goals")
        ag   = m.get("away_goals")
        if not home or not away or hg is None or ag is None:
            continue
        if home not in known_teams or away not in known_teams:
            continue

        # Canonical: a < b alphabetically
        if home <= away:
            a, b   = home, away
            gd_a   = int(hg) - int(ag)
        else:
            a, b   = away, home
            gd_a   = int(ag) - int(hg)

        base_w = weights.get(m.get("competition_type", "FRIENDLY"), 0.3)
        years  = _years_elapsed(m.get("date", ""), ref)
        w      = base_w * math.exp(-lam * years)

        d = pair_stats[(a, b)]
        d["n"]    += 1
        d["wsum"] += w
        norm       = math.copysign(math.log1p(abs(gd_a)), gd_a) if gd_a != 0 else 0.0
        d["gd_w"] += norm * w

        if gd_a > 0:
            d["wins_a"] += 1
        elif gd_a < 0:
            d["wins_b"] += 1
        else:
            d["draws"] += 1

    result = []
    for (a, b), d in pair_stats.items():
        if d["n"] == 0:
            continue
        score = h2h_table.get((a, b), 0.0)
        wgd   = d["gd_w"] / d["wsum"] if d["wsum"] > 0 else 0.0
        result.append({
            "team_a":     a,
            "team_b":     b,
            "wins_a":     d["wins_a"],
            "draws":      d["draws"],
            "wins_b":     d["wins_b"],
            "weighted_gd": round(wgd, 3),
            "h2h_score":  round(score, 3),
            "n_matches":  d["n"],
        })

    result.sort(key=lambda x: -abs(x["h2h_score"]))
    return result[:limit]


# ---------------------------------------------------------------------------
# Per-team opponent breakdown (for "Focus team" panel)
# ---------------------------------------------------------------------------

def h2h_vs_team(
    matches: list[dict],
    team: str,
    h2h_table: dict[tuple[str, str], float],
    custom_weights: dict | None = None,
    lambda_decay: float | None = None,
    reference: date | None = None,
) -> list[dict]:
    """
    Return H2H stats for every opponent the team has faced.

    Each row: {opponent, h2h_score, wins, draws, losses,
               avg_scored, avg_conceded, n_matches}
    Sorted by |h2h_score| descending.
    avg_scored / avg_conceded are temporally weighted averages.
    """
    weights = custom_weights or config.MATCH_WEIGHTS
    lam     = lambda_decay if lambda_decay is not None else config.LAMBDA_DECAY
    ref     = reference or config.REFERENCE_DATE

    opp_data: dict[str, dict] = defaultdict(
        lambda: {"wins": 0, "draws": 0, "losses": 0,
                 "gs_w": 0.0, "gc_w": 0.0, "wsum": 0.0, "n": 0}
    )

    for m in matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        hg   = m.get("home_goals")
        ag   = m.get("away_goals")
        if hg is None or ag is None:
            continue

        if home == team:
            opp, gs, gc = away, int(hg), int(ag)
        elif away == team:
            opp, gs, gc = home, int(ag), int(hg)
        else:
            continue

        base_w = weights.get(m.get("competition_type", "FRIENDLY"), 0.3)
        years  = _years_elapsed(m.get("date", ""), ref)
        w      = base_w * math.exp(-lam * years)

        d = opp_data[opp]
        d["n"]    += 1
        d["wsum"] += w
        d["gs_w"] += gs * w
        d["gc_w"] += gc * w
        if gs > gc:
            d["wins"] += 1
        elif gs == gc:
            d["draws"] += 1
        else:
            d["losses"] += 1

    result = []
    for opp, d in opp_data.items():
        if d["n"] == 0:
            continue
        wsum         = d["wsum"] or 1.0
        avg_scored   = round(d["gs_w"] / wsum, 2)
        avg_conceded = round(d["gc_w"] / wsum, 2)
        score        = h2h_table.get((team, opp), 0.0)
        result.append({
            "opponent":       opp,
            "h2h_score":      round(score, 3),
            "wins":           d["wins"],
            "draws":          d["draws"],
            "losses":         d["losses"],
            "avg_scored":     avg_scored,
            "avg_conceded":   avg_conceded,
            "n_matches":      d["n"],
        })

    result.sort(key=lambda x: -abs(x["h2h_score"]))
    return result
