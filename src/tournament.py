"""
WC 2026 tournament structure and group-stage/KO simulation.

48 teams – 12 groups of 4 – top 2 + 8 best 3rd-place qualifiers → R32 (16 matches)
→ R16 (8) → QF (4) → SF (2) → Final.

Official R32 structure (FIFA WC 2026):
  8 WvT  (winner vs 3rd-place): M74, M77, M79, M80, M81, M82, M85, M87
  4 WvR  (winner vs runner-up): M75, M76, M84, M86
  4 RvR  (runner-up vs runner-up): M73, M78, M83, M88

Third-place slot assignment uses bipartite matching with official per-slot constraint sets
confirmed by the FIFA/ESPN bracket publication (6 of 8 slots explicitly listed; W-B and W-K
derived from the bracket structure and validated across all C(12,8)=495 combinations).
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import numpy as np

import config
from src.poisson import simulate_match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

def load_groups() -> dict[str, list[str]]:
    with (Path("data") / "teams.json").open("r", encoding="utf-8") as f:
        return json.load(f)["groups"]


def _load_ko_played() -> dict[tuple[str, int], str]:
    """
    Load played KO results from data/ko_results.json.
    Returns {(round, match_idx): 'home'|'away'} for each decisively played match.
    round ∈ {'r32','r16','qf','sf','final'}; match_idx is 0-based per R32_BRACKET/
    R16_PAIRS/QF_PAIRS/SF_PAIRS order.
    """
    path = Path("data") / "ko_results.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    played: dict[tuple[str, int], str] = {}

    def _side(m: dict) -> str | None:
        gh, ga = int(m["goals_h"]), int(m["goals_a"])
        if gh > ga:
            return "home"
        if ga > gh:
            return "away"
        # Draw resolved by penalty shootout
        if m.get("penalties"):
            ph = int(m.get("penalties_h") or 0)
            pa = int(m.get("penalties_a") or 0)
            if ph > pa:
                return "home"
            if pa > ph:
                return "away"
        return None

    for rnd in ("r32", "r16", "qf", "sf"):
        for idx, m in enumerate(data.get(rnd, [])):
            if (m.get("played")
                    and m.get("goals_h") is not None
                    and m.get("goals_a") is not None):
                side = _side(m)
                if side:
                    played[(rnd, idx)] = side
    fin = data.get("final", {})
    if (fin.get("played")
            and fin.get("goals_h") is not None
            and fin.get("goals_a") is not None):
        side = _side(fin)
        if side:
            played[("final", 0)] = side
    return played


# ---------------------------------------------------------------------------
# Official R32 bracket — third-place placement
# ---------------------------------------------------------------------------

# Eligible groups for each "Winner vs 3rd-place" slot.
# Slot key = group whose WINNER hosts; value = groups whose 3rd-place team may fill it.
# Own-group is always excluded.
# Sources: ESPN/FIFA official bracket (marked [ESPN]); bracket-structure inference ([inf]).
# All 495 combinations C(12,8) validated — every input yields a unique valid assignment.
_THIRD_PLACE_CONSTRAINTS: dict[str, frozenset[str]] = {
    "A": frozenset("CEFHI"),    # M79: W-A vs 3rd from C,E,F,H,I  [ESPN]
    "B": frozenset("EFGIJ"),    # M85: W-B vs 3rd from E,F,G,I,J  [inf]
    "D": frozenset("BEFIJ"),    # M81: W-D vs 3rd from B,E,F,I,J  [ESPN]
    "E": frozenset("ABCDF"),    # M74: W-E vs 3rd from A,B,C,D,F  [ESPN]
    "G": frozenset("AEHIJ"),    # M82: W-G vs 3rd from A,E,H,I,J  [ESPN]
    "I": frozenset("CDFGH"),    # M77: W-I vs 3rd from C,D,F,G,H  [ESPN]
    "K": frozenset("DEIJL"),    # M87: W-K vs 3rd from D,E,I,J,L  [inf]
    "L": frozenset("EHIJK"),    # M80: W-L vs 3rd from E,H,I,J,K  [ESPN]
}

# Official 3rd-place draw (FIFA WC 2026, announced post-group-stage).
# Annex C allows multiple valid bipartite matchings for some qualifying-group
# combinations; the backtracking finds the first valid solution (alphabetical order)
# which differs from the actual FIFA draw. This override stores confirmed results.
# Key = sorted tuple of 8 qualifying groups; value = {winner_slot: third_grp}.
_OFFICIAL_THIRD_PLACE_DRAW: dict[tuple[str, ...], dict[str, str]] = {
    # Actual WC2026: groups B,D,E,F,I,J,K,L produced the 8 best 3rd-place teams.
    # M79:1A-3E, M85:1B-3J, M81:1D-3B, M74:1E-3D, M82:1G-3I, M77:1I-3F,
    # M87:1K-3L, M80:1L-3K  (source: official FIFA bracket publication)
    ('B', 'D', 'E', 'F', 'I', 'J', 'K', 'L'): {
        "A": "E", "B": "J", "D": "B", "E": "D",
        "G": "I", "I": "F", "K": "L", "L": "K",
    },
}

# Sorted tuple of the 8 WvT slot labels (= groups whose winner hosts a 3rd-place team)
_WVT_SLOTS: tuple[str, ...] = tuple(sorted(_THIRD_PLACE_CONSTRAINTS))


@functools.lru_cache(maxsize=512)
def _assign_third_place(qualifying_groups: tuple[str, ...]) -> dict[str, str]:
    """
    Assign 8 qualifying 3rd-place groups to the 8 WvT R32 slots.

    Checks _OFFICIAL_THIRD_PLACE_DRAW first (known FIFA draw results).
    Falls back to backtracking bipartite matching against _THIRD_PLACE_CONSTRAINTS
    for hypothetical Monte Carlo combinations. Result is cached (≤495 inputs).

    Args:
        qualifying_groups: sorted tuple of 8 group letters (e.g. ('A','B','C',...))
    Returns:
        {winner_slot_group: qualifying_group} e.g. {'A': 'F', 'B': 'J', ...}
    """
    if qualifying_groups in _OFFICIAL_THIRD_PLACE_DRAW:
        return dict(_OFFICIAL_THIRD_PLACE_DRAW[qualifying_groups])

    remaining = list(qualifying_groups)
    assignment: dict[str, str] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(_WVT_SLOTS):
            return True
        slot = _WVT_SLOTS[idx]
        constraints = _THIRD_PLACE_CONSTRAINTS[slot]
        for i, grp in enumerate(remaining):
            if grp in constraints:
                assignment[slot] = grp
                del remaining[i]
                if backtrack(idx + 1):
                    return True
                remaining.insert(i, grp)
                del assignment[slot]
        return False

    if not backtrack(0):
        logger.error(
            "No valid 3rd-place assignment for groups %s — using naive fallback",
            qualifying_groups,
        )
        for slot, grp in zip(_WVT_SLOTS, qualifying_groups):
            assignment[slot] = grp

    return dict(assignment)


# ---------------------------------------------------------------------------
# R32 / R16 / QF / SF bracket constants
# ---------------------------------------------------------------------------

# Slot codes: "1X" = winner of group X, "2X" = runner-up, "3W_X" = 3rd-place
# team assigned to face the winner of group X (resolved per-sim by _assign_third_place).
#
# Pairs are ordered so every two consecutive entries share an R16 match:
#   indices [0,1]   → R16-M89   indices [2,3]   → R16-M90
#   indices [4,5]   → R16-M91   indices [6,7]   → R16-M92
#   indices [8,9]   → R16-M93   indices [10,11] → R16-M94
#   indices [12,13] → R16-M95   indices [14,15] → R16-M96
R32_BRACKET: list[tuple[str, str]] = [
    # R16-M89  (M74 + M77)
    ("1E", "3W_E"),   # M74: W-E vs 3rd-WvT
    ("1I", "3W_I"),   # M77: W-I vs 3rd-WvT
    # R16-M90  (M73 + M75)
    ("2A", "2B"),     # M73: RU-A vs RU-B
    ("1F", "2C"),     # M75: W-F vs RU-C
    # R16-M91  (M76 + M78)
    ("1C", "2F"),     # M76: W-C vs RU-F
    ("2E", "2I"),     # M78: RU-E vs RU-I
    # R16-M92  (M79 + M80)
    ("1A", "3W_A"),   # M79: W-A vs 3rd-WvT
    ("1L", "3W_L"),   # M80: W-L vs 3rd-WvT
    # R16-M93  (M83 + M84) — Spain (W-H) path
    ("2K", "2L"),     # M83: RU-K vs RU-L
    ("1H", "2J"),     # M84: W-H vs RU-J
    # R16-M94  (M81 + M82)
    ("1D", "3W_D"),   # M81: W-D vs 3rd-WvT
    ("1G", "3W_G"),   # M82: W-G vs 3rd-WvT
    # R16-M95  (M86 + M88) — Argentina (W-J) path
    ("1J", "2H"),     # M86: W-J vs RU-H
    ("2D", "2G"),     # M88: RU-D vs RU-G
    # R16-M96  (M85 + M87)
    ("1B", "3W_B"),   # M85: W-B vs 3rd-WvT
    ("1K", "3W_K"),   # M87: W-K vs 3rd-WvT
]
assert len(R32_BRACKET) == 16

# Pairs of R32 match indices (into r32_winners) that meet in each R16 match.
R16_PAIRS: list[tuple[int, int]] = [
    (0, 1),   # R16-M89: M74 winner vs M77 winner
    (2, 3),   # R16-M90: M73 winner vs M75 winner
    (4, 5),   # R16-M91: M76 winner vs M78 winner
    (6, 7),   # R16-M92: M79 winner vs M80 winner
    (8, 9),   # R16-M93: M83 winner vs M84 winner  (Spain's half)
    (10, 11), # R16-M94: M81 winner vs M82 winner
    (12, 13), # R16-M95: M86 winner vs M88 winner  (Argentina's half)
    (14, 15), # R16-M96: M85 winner vs M87 winner
]

# Pairs of R16 winner indices (0-7) that meet in each QF.
# M97=(R16-0+R16-1), M98=(R16-4+R16-5), M99=(R16-2+R16-3), M100=(R16-6+R16-7)
QF_PAIRS: list[tuple[int, int]] = [
    (0, 1),  # M97: Spain half upper
    (4, 5),  # M98: Spain half lower
    (2, 3),  # M99: Argentina half upper
    (6, 7),  # M100: Argentina half lower
]

# Pairs of QF winner indices (0-3) that meet in each SF.
# M101=(QF-0+QF-1) Spain's SF, M102=(QF-2+QF-3) Argentina's SF → Final only
SF_PAIRS: list[tuple[int, int]] = [
    (0, 1),  # M101: Spain's SF half
    (2, 3),  # M102: Argentina's SF half
]


# ---------------------------------------------------------------------------
# Group stage helpers
# ---------------------------------------------------------------------------

def _group_rank_key(row: dict) -> tuple:
    """Primary pts-only key — used by _simulate_group for initial grouping."""
    return (-row["pts"],)


def _build_sort_key(
    row: dict,
    tied_names: set[str],
    match_results: dict[tuple[str, str], tuple[int, int]],
    att_rating: float,
) -> tuple:
    """Full FIFA WC2026 tiebreaker key for a team within a same-pts group.

    Order (per official regulations):
      1. H2H pts among tied teams
      2. H2H GD among tied teams
      3. H2H GF among tied teams
      4. Overall GD (fallthrough when H2H doesn't resolve)
      5. Overall GF
      6. Fair-play: not tracked → constant 0
      7. ELO proxy (att_rating as FIFA-ranking surrogate)
      8. Alphabetical — absolute last resort, guarantees uniqueness
    """
    t = row["team"]
    opponents = tied_names - {t}
    h_pts = h_gd = h_gf = 0
    for (h, a), (hg, ag) in match_results.items():
        if h == t and a in opponents:
            h_gf += hg
            h_gd += hg - ag
            if hg > ag:   h_pts += 3
            elif hg < ag: pass
            else:         h_pts += 1
        elif a == t and h in opponents:
            h_gf += ag
            h_gd += ag - hg
            if ag > hg:   h_pts += 3
            elif hg > ag: pass
            else:         h_pts += 1
    return (-h_pts, -h_gd, -h_gf, -row["gd"], -row["gf"], 0, -att_rating, t)


def _simulate_group(
    teams: list[str],
    coeffs: dict[str, dict],
    mu: float,
    rng: np.random.Generator,
    played_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Simulate all 6 group matches. Return list of dicts sorted by rank.
    Each dict: {team, pts, gd, gf, ga, w, d, l}
    Matches already played (in played_results) use real scores; others use Poisson.
    """
    stats = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
             for t in teams}
    # Track per-match scores for H2H tiebreaking
    match_results: dict[tuple[str, str], tuple[int, int]] = {}

    played = played_results or {}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            if (h, a) in played:
                hg, ag = played[(h, a)]
            else:
                ch = coeffs[h]
                ca = coeffs[a]
                hg, ag, _ = simulate_match(
                    ch["att_rating"], ch["def_rating"],
                    ca["att_rating"], ca["def_rating"],
                    mu=mu, is_knockout=False, rng=rng,
                )
            match_results[(h, a)] = (hg, ag)
            stats[h]["gf"] += hg
            stats[h]["ga"] += ag
            stats[a]["gf"] += ag
            stats[a]["ga"] += hg
            stats[h]["gd"] += hg - ag
            stats[a]["gd"] += ag - hg
            if hg > ag:
                stats[h]["pts"] += 3; stats[h]["w"] += 1; stats[a]["l"] += 1
            elif hg < ag:
                stats[a]["pts"] += 3; stats[a]["w"] += 1; stats[h]["l"] += 1
            else:
                stats[h]["pts"] += 1; stats[h]["d"] += 1
                stats[a]["pts"] += 1; stats[a]["d"] += 1

    # FIFA WC2026 ranking: group by pts, then H2H → overall GD/GF → ELO → alpha
    from collections import defaultdict as _dd
    pts_grps: dict[int, list[dict]] = _dd(list)
    for row in stats.values():
        pts_grps[row["pts"]].append(row)

    final_ranked: list[dict] = []
    for pts_val in sorted(pts_grps.keys(), reverse=True):
        grp = pts_grps[pts_val]
        if len(grp) == 1:
            final_ranked.append(grp[0])
        else:
            tied_names = {r["team"] for r in grp}
            final_ranked.extend(sorted(
                grp,
                key=lambda r: _build_sort_key(
                    r, tied_names, match_results,
                    coeffs.get(r["team"], {}).get("att_rating", 1.0),
                ),
            ))

    for pos, row in enumerate(final_ranked):
        row["position"] = pos + 1
    return final_ranked


def _best_third_rank_key(row: dict) -> tuple:
    return (-row["pts"], -row["gd"], -row["gf"])


def _select_best_thirds(group_results: dict[str, list[dict]]) -> list[dict]:
    """Select the 8 best 3rd-place finishers from 12 groups."""
    thirds = []
    for grp, results in group_results.items():
        row = results[2].copy()
        row["group"] = grp
        thirds.append(row)
    thirds.sort(key=_best_third_rank_key)
    return thirds[:8]


# ---------------------------------------------------------------------------
# KO stage
# ---------------------------------------------------------------------------

def _ko_match(
    team_h: str,
    team_a: str,
    coeffs: dict[str, dict],
    mu: float,
    rng: np.random.Generator,
) -> str:
    ch = coeffs[team_h]
    ca = coeffs[team_a]
    hg, ag, _ = simulate_match(
        ch["att_rating"], ch["def_rating"],
        ca["att_rating"], ca["def_rating"],
        mu=mu, is_knockout=True, rng=rng,
    )
    return team_h if hg > ag else team_a


# ---------------------------------------------------------------------------
# Full tournament simulation (single run)
# ---------------------------------------------------------------------------

def simulate_tournament(
    coeffs: dict[str, dict],
    mu: float = config.MU,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Simulate one complete WC 2026.

    Returns {
        team: {
            "group": bool, "r32": bool, "r16": bool, "qf": bool,
            "sf": bool, "final": bool, "winner": bool,
            "group_position": int,
        }
    }
    """
    if rng is None:
        rng = np.random.default_rng()

    groups = load_groups()
    teams_set = set(coeffs.keys())

    # Ensure all group teams have coefficients (fallback to 1.0)
    for grp, members in groups.items():
        for t in members:
            if t not in coeffs:
                coeffs[t] = {"att_rating": 1.0, "def_rating": 1.0}

    # Load real match results — both orderings stored for O(1) lookup
    all_played: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    gr_path = Path("data") / "group_results.json"
    if gr_path.exists():
        with gr_path.open("r", encoding="utf-8") as f:
            gr_data = json.load(f)
        for grp, grp_data in gr_data.items():
            grp_played: dict[tuple[str, str], tuple[int, int]] = {}
            for match in grp_data.get("matches", []):
                if (match.get("played")
                        and match["home_goals"] is not None
                        and match["away_goals"] is not None):
                    h, a = match["home"], match["away"]
                    hg, ag = int(match["home_goals"]), int(match["away_goals"])
                    grp_played[(h, a)] = (hg, ag)
                    grp_played[(a, h)] = (ag, hg)
            if grp_played:
                all_played[grp] = grp_played

    ko_played = _load_ko_played()

    # ----- Group stage -----
    group_results: dict[str, list[dict]] = {}
    for grp, members in groups.items():
        group_results[grp] = _simulate_group(members, coeffs, mu, rng, all_played.get(grp))

    # Track which round each team reached
    results = {t: {
        "group": True, "r32": False, "r16": False, "qf": False,
        "sf": False, "final": False, "winner": False, "group_position": 0,
    } for t in coeffs}

    # Assign group positions
    for grp, ranked in group_results.items():
        for row in ranked:
            t = row["team"]
            if t in results:
                results[t]["group_position"] = row["position"]

    # ----- Build slot → team map for R32 -----
    slot_team: dict[str, str] = {}
    for grp, ranked in group_results.items():
        slot_team[f"1{grp}"] = ranked[0]["team"]
        slot_team[f"2{grp}"] = ranked[1]["team"]

    # Select best 8 3rd-place qualifiers and assign to WvT slots
    best_thirds = _select_best_thirds(group_results)
    qualifying_groups = tuple(sorted(row["group"] for row in best_thirds))
    third_by_group = {row["group"]: row["team"] for row in best_thirds}

    third_assign = _assign_third_place(qualifying_groups)
    for winner_grp, third_grp in third_assign.items():
        slot_team[f"3W_{winner_grp}"] = third_by_group[third_grp]

    # Mark R32 qualifiers
    for team in slot_team.values():
        if team in results:
            results[team]["r32"] = True

    # ----- R32 (16 matches) -----
    r32_winners: list[str] = []
    for k, (slot_h, slot_a) in enumerate(R32_BRACKET):
        th = slot_team[slot_h]
        ta = slot_team[slot_a]
        side = ko_played.get(("r32", k))
        if side == "home":
            winner = th
        elif side == "away":
            winner = ta
        else:
            winner = _ko_match(th, ta, coeffs, mu, rng)
        r32_winners.append(winner)

    # ----- R16 -----
    for w in r32_winners:
        if w in results:
            results[w]["r16"] = True

    r16_winners: list[str] = []
    for k, (i, j) in enumerate(R16_PAIRS):
        side = ko_played.get(("r16", k))
        if side == "home":
            winner = r32_winners[i]
        elif side == "away":
            winner = r32_winners[j]
        else:
            winner = _ko_match(r32_winners[i], r32_winners[j], coeffs, mu, rng)
        r16_winners.append(winner)

    # ----- QF -----
    for w in r16_winners:
        if w in results:
            results[w]["qf"] = True

    qf_winners: list[str] = []
    for k, (i, j) in enumerate(QF_PAIRS):
        side = ko_played.get(("qf", k))
        if side == "home":
            winner = r16_winners[i]
        elif side == "away":
            winner = r16_winners[j]
        else:
            winner = _ko_match(r16_winners[i], r16_winners[j], coeffs, mu, rng)
        qf_winners.append(winner)

    # ----- SF -----
    for w in qf_winners:
        if w in results:
            results[w]["sf"] = True

    sf_winners: list[str] = []
    for k, (i, j) in enumerate(SF_PAIRS):
        side = ko_played.get(("sf", k))
        if side == "home":
            winner = qf_winners[i]
        elif side == "away":
            winner = qf_winners[j]
        else:
            winner = _ko_match(qf_winners[i], qf_winners[j], coeffs, mu, rng)
        sf_winners.append(winner)

    # ----- Final -----
    for w in sf_winners:
        if w in results:
            results[w]["final"] = True

    side = ko_played.get(("final", 0))
    if side == "home":
        champion = sf_winners[0]
    elif side == "away":
        champion = sf_winners[1]
    else:
        champion = _ko_match(sf_winners[0], sf_winners[1], coeffs, mu, rng)
    if champion in results:
        results[champion]["winner"] = True

    return results
