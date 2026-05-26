"""Compute weighted att/def ratings from match history."""
import json
import math
from datetime import date, datetime
from pathlib import Path

import config

# Streamlit cache — degrades gracefully outside Streamlit context (scripts, tests)
try:
    from streamlit import cache_data as _st_cache_data
except Exception:
    def _st_cache_data(fn=None, *, ttl=None, **kw):  # type: ignore[misc]
        def _dec(f):
            return f
        return fn if fn is not None else _dec

_TEAMS_PATH    = Path("data/teams.json")
_ELO_ALL_PATH  = Path("data/elo_all_teams.json")
_DEFAULT_ELO   = 1650   # global fallback when team & confederation both unknown


def _years_elapsed(match_date: str, reference: date | None = None) -> float:
    ref = reference or config.REFERENCE_DATE
    try:
        d = datetime.strptime(match_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return 2.0
    delta = (ref - d).days / 365.25
    return max(delta, 0.0)


def _temporal_weight(match_date: str, match_type: str, reference: date | None = None) -> float:
    base = config.MATCH_WEIGHTS.get(match_type, 0.3)
    years = _years_elapsed(match_date, reference)
    return base * math.exp(-config.LAMBDA_DECAY * years)


def _load_elo(teams_json: dict) -> dict[str, int]:
    """Return {team_name: elo} from a teams.json dict (tier-1 only, kept for compat)."""
    return {
        name: info.get("elo", _DEFAULT_ELO)
        for name, info in teams_json.get("teams", {}).items()
    }


def load_full_elo_map() -> tuple[dict[str, int], float]:
    """
    Build a complete ELO map using 3-tier fallback:

      1. data/teams.json        — 48 WC2026 qualifiers (highest precision)
      2. data/elo_all_teams.json — all ~210 FIFA nations
      3. confederation average  — from elo_all_teams._confederation_fallbacks
         (e.g. UEFA=1750, CAF=1650, …)

    Returns (elo_map, elo_mean).
    elo_mean is computed over all resolved entries and used to normalise
    opponent strength in compute_coefficients() and calibrate().
    """
    # --- Tier 1: WC2026 qualifiers ---
    elo_map: dict[str, int] = {}
    try:
        with _TEAMS_PATH.open("r", encoding="utf-8") as f:
            tj = json.load(f)
        for name, info in tj.get("teams", {}).items():
            elo_map[name] = int(info.get("elo", _DEFAULT_ELO))
    except Exception:
        pass

    # --- Tier 2 + 3: all other FIFA nations ---
    try:
        with _ELO_ALL_PATH.open("r", encoding="utf-8") as f:
            elo_all = json.load(f)
        conf_fallbacks: dict[str, int] = {
            k: int(v) for k, v in elo_all.get("_confederation_fallbacks", {}).items()
        }
        for name, info in elo_all.get("teams", {}).items():
            if name in elo_map:
                continue  # tier-1 takes precedence
            elo = info.get("elo")
            if elo is not None:
                elo_map[name] = int(elo)
            else:
                conf = info.get("confederation", "")
                elo_map[name] = conf_fallbacks.get(conf, _DEFAULT_ELO)
    except Exception:
        pass

    # elo_mean is anchored to WC2026 qualifiers only (tier-1).
    # Using all 197 FIFA teams would drag the mean to ~1550 (San Marino, Montserrat…)
    # and artificially inflate opponent-strength ratios for European teams.
    try:
        with _TEAMS_PATH.open("r", encoding="utf-8") as f:
            _wc = json.load(f)
        wc_elos = [int(v.get("elo", _DEFAULT_ELO)) for v in _wc.get("teams", {}).values()]
        elo_mean = sum(wc_elos) / len(wc_elos) if wc_elos else float(_DEFAULT_ELO)
    except Exception:
        elo_vals = list(elo_map.values())
        elo_mean = sum(elo_vals) / len(elo_vals) if elo_vals else float(_DEFAULT_ELO)
    return elo_map, elo_mean


@_st_cache_data
def compute_coefficients(
    matches: list[dict],
    teams: list[str],
    custom_weights: dict | None = None,
    lambda_decay: float | None = None,
    reference: date | None = None,
) -> dict[str, dict]:
    """
    Compute raw (unoptimised) att/def ratings for each team.

    Goals against stronger opponents are weighted up for attack and down for
    defence, using each opponent's ELO relative to the field mean.

    Returns dict[team] = {"att_rating": float, "def_rating": float, "n_matches": int}
    Ratings are normalised so global mean == 1.0.
    """
    weights = custom_weights or config.MATCH_WEIGHTS
    lam = lambda_decay if lambda_decay is not None else config.LAMBDA_DECAY
    ref = reference or config.REFERENCE_DATE

    # Load fallback ratings and full ELO map (3-tier: WC teams → all FIFA → confederation avg)
    with _TEAMS_PATH.open("r", encoding="utf-8") as f:
        teams_json = json.load(f)
    fallback = teams_json.get("teams", {})
    elo_map, elo_mean = load_full_elo_map()

    # Separate accumulators for att and def (different ELO weighting)
    scored_att:  dict[str, float] = {t: 0.0 for t in teams}
    scored_def:  dict[str, float] = {t: 0.0 for t in teams}
    wsum_att:    dict[str, float] = {t: 0.0 for t in teams}
    wsum_def:    dict[str, float] = {t: 0.0 for t in teams}
    counts:      dict[str, int]   = {t: 0   for t in teams}

    teams_set = set(teams)

    for m in matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        h_goals = m.get("home_goals")
        a_goals = m.get("away_goals")
        if h_goals is None or a_goals is None:
            continue
        if home not in teams_set and away not in teams_set:
            continue

        match_type = m.get("competition_type", "FRIENDLY")
        base_w = weights.get(match_type, 0.3)
        years = _years_elapsed(m.get("date", ""), ref)
        w = base_w * math.exp(-lam * years)

        # opponent_strength: > 1 for strong opponents, < 1 for weak ones
        elo_home = elo_map.get(home, int(elo_mean))
        elo_away = elo_map.get(away, int(elo_mean))
        opp_for_home = elo_away / elo_mean   # away team is home's opponent
        opp_for_away = elo_home / elo_mean   # home team is away's opponent

        # Weak-opponent cap: very weak opponents (ELO below threshold) contribute
        # at most cap_factor of the normal weight, regardless of the scoreline.
        # Applied per-perspective: home team's stats capped when *away* team is weak,
        # and vice-versa.
        _thresh = config.ELO_WEAK_OPP_THRESHOLD
        _cap    = config.ELO_WEAK_OPP_CAP_FACTOR
        w_h = w * (_cap if elo_away < _thresh else 1.0)
        w_a = w * (_cap if elo_home < _thresh else 1.0)

        if home in teams_set:
            # Attack: goals vs strong opponent worth more → multiply by opp_strength
            scored_att[home] += w_h * opp_for_home * h_goals
            wsum_att[home]   += w_h * opp_for_home
            # Defence: conceding vs strong opponent less penalising → divide by opp_strength.
            # w_h is the SAME capped weight as for attack — cap is symmetric for both sides.
            scored_def[home] += w_h / opp_for_home * a_goals
            wsum_def[home]   += w_h / opp_for_home
            counts[home] += 1

        if away in teams_set:
            scored_att[away] += w_a * opp_for_away * a_goals
            wsum_att[away]   += w_a * opp_for_away
            # Same w_a cap applies to defence (symmetric with attack).
            scored_def[away] += w_a / opp_for_away * h_goals
            wsum_def[away]   += w_a / opp_for_away
            counts[away] += 1

    raw_att: dict[str, float] = {}
    raw_def: dict[str, float] = {}

    for t in teams:
        fb = fallback.get(t, {})
        if wsum_att[t] > 0:
            raw_att[t] = scored_att[t] / wsum_att[t]
        else:
            raw_att[t] = fb.get("att_rating", 1.0)

        if wsum_def[t] > 0:
            raw_def[t] = scored_def[t] / wsum_def[t]
        else:
            raw_def[t] = fb.get("def_rating", 1.0)

    # Normalise: mean att = 1.0, mean def = 1.0
    att_mean = sum(raw_att.values()) / len(raw_att) or 1.0
    def_mean = sum(raw_def.values()) / len(raw_def) or 1.0

    _att_min, _att_max = config.ATT_MIN, config.ATT_MAX
    _def_min, _def_max = config.DEF_MIN, config.DEF_MAX

    result: dict[str, dict] = {}
    for t in teams:
        result[t] = {
            "att_rating": max(_att_min, min(_att_max, raw_att[t] / att_mean)),
            "def_rating": max(_def_min, min(_def_max, raw_def[t] / def_mean)),
            "n_matches":  counts[t],
        }

    return result


def match_weight(m: dict, custom_weights: dict | None = None, lambda_decay: float | None = None) -> float:
    """Return the temporal weight of a single match dict (no ELO adjustment)."""
    weights = custom_weights or config.MATCH_WEIGHTS
    lam = lambda_decay if lambda_decay is not None else config.LAMBDA_DECAY
    base_w = weights.get(m.get("competition_type", "FRIENDLY"), 0.3)
    years = _years_elapsed(m.get("date", ""))
    return base_w * math.exp(-lam * years)
