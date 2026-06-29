"""
Monte Carlo engine: simulate N full tournaments and aggregate.
Uses numpy vectorisation for the group stage and sequential KO rounds.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

import config
from src.tournament import (
    load_groups,
    R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS,
    _group_rank_key, _best_third_rank_key,
    _assign_third_place, _WVT_SLOTS,
    _load_ko_played,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast vectorised group stage
# ---------------------------------------------------------------------------

def _vec_group_stage(
    groups: dict[str, list[str]],
    teams: list[str],
    att: np.ndarray,
    def_: np.ndarray,
    mu: float,
    N: int,
    rng: np.random.Generator,
    h2h_mat: np.ndarray,
    played_results: dict[str, dict[tuple[str, str], tuple[int, int]]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, tuple]]:
    """
    Simulate group stage for all N tournaments at once.

    Returns:
      - dict[grp] → np.ndarray shape (4, N) with team indices sorted by rank (best first)
      - dict[grp] → (mi, pts, gf, gd) raw arrays for stats accumulation
        where mi = list of global team indices, pts/gf/gd = np.ndarray shape (n_members, N)
    """
    t2i = {t: i for i, t in enumerate(teams)}
    group_rankings: dict[str, np.ndarray] = {}
    grp_stats: dict[str, tuple] = {}

    for grp, members in groups.items():
        n_teams = len(members)
        mi = [t2i[t] for t in members]  # member indices into att/def arrays

        # pts[k, sim] for each team k in group
        pts = np.zeros((n_teams, N), dtype=np.int32)
        gd = np.zeros((n_teams, N), dtype=np.int32)
        gf = np.zeros((n_teams, N), dtype=np.int32)

        # Per-pair match scores (local indices, i < j) — used for H2H tiebreaking
        match_hg: dict[tuple[int, int], np.ndarray] = {}
        match_ag: dict[tuple[int, int], np.ndarray] = {}

        grp_played = played_results.get(grp, {}) if played_results else {}
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                key = (members[i], members[j])
                if key in grp_played:
                    hg_real, ag_real = grp_played[key]
                    hg = np.full(N, hg_real, dtype=np.int32)
                    ag = np.full(N, ag_real, dtype=np.int32)
                else:
                    h2h_score = h2h_mat[mi[i], mi[j]]
                    h2h_factor = 1.0 + config.H2H_WEIGHT * h2h_score
                    lam_h = att[mi[i]] * def_[mi[j]] * mu * h2h_factor
                    lam_a = att[mi[j]] * def_[mi[i]] * mu / h2h_factor
                    hg = rng.poisson(lam_h, size=N).astype(np.int32)
                    ag = rng.poisson(lam_a, size=N).astype(np.int32)
                match_hg[(i, j)] = hg
                match_ag[(i, j)] = ag
                gf[i] += hg; gf[j] += ag
                diff = hg - ag
                gd[i] += diff; gd[j] -= diff
                home_wins = hg > ag
                draws = hg == ag
                away_wins = ag > hg
                pts[i] += 3 * home_wins + draws
                pts[j] += 3 * away_wins + draws

        # FIFA WC2026 ranking: pts primary, then H2H → overall GD/GF → ELO → alpha.
        # Encode as float64: pts * 1e12 so any sub-criterion adjustment (< 1) can't
        # overflow into the next pts level (1e12 apart).
        combined_f = pts.astype(np.float64) * 1e12  # (n_teams, N)

        # Detect simulations with pts ties — H2H tiebreaker must run for those.
        sorted_pts = np.sort(pts, axis=0)                           # (n_teams, N)
        has_pts_ties = np.any(np.diff(sorted_pts, axis=0) == 0, axis=0)  # (N,)
        tie_sims = np.where(has_pts_ties)[0]

        for sim_idx in tie_sims:
            pts_sim = pts[:, sim_idx]
            gd_sim  = gd[:, sim_idx]
            gf_sim  = gf[:, sim_idx]
            # Group local team indices by pts value
            by_pts: dict[int, list[int]] = {}
            for k, v in enumerate(pts_sim.tolist()):
                vi = int(v)
                if vi not in by_pts:
                    by_pts[vi] = []
                by_pts[vi].append(k)
            for pts_val, tied in by_pts.items():
                if len(tied) < 2:
                    continue
                n_tied = len(tied)
                h2h_pts_s = [0] * n_tied
                h2h_gd_s  = [0] * n_tied
                h2h_gf_s  = [0] * n_tied
                for pi in range(n_tied):
                    for pj in range(pi + 1, n_tied):
                        ti, tj = tied[pi], tied[pj]
                        # canonical key always has smaller local index first
                        if ti < tj:
                            hg_v = int(match_hg[(ti, tj)][sim_idx])
                            ag_v = int(match_ag[(ti, tj)][sim_idx])
                        else:
                            hg_v = int(match_ag[(tj, ti)][sim_idx])
                            ag_v = int(match_hg[(tj, ti)][sim_idx])
                        h2h_gf_s[pi] += hg_v
                        h2h_gf_s[pj] += ag_v
                        d = hg_v - ag_v
                        h2h_gd_s[pi] += d
                        h2h_gd_s[pj] -= d
                        if hg_v > ag_v:
                            h2h_pts_s[pi] += 3
                        elif ag_v > hg_v:
                            h2h_pts_s[pj] += 3
                        else:
                            h2h_pts_s[pi] += 1
                            h2h_pts_s[pj] += 1
                # Full tiebreaker key:
                #   H2H pts → H2H GD → H2H GF
                #   → overall GD → overall GF
                #   → att_rating (ELO proxy) → alphabetical
                order = sorted(
                    range(n_tied),
                    key=lambda p: (
                        -h2h_pts_s[p],
                        -h2h_gd_s[p],
                        -h2h_gf_s[p],
                        -int(gd_sim[tied[p]]),
                        -int(gf_sim[tied[p]]),
                        -float(att[mi[tied[p]]]),
                        members[tied[p]],
                    ),
                )
                # Inject sub-unit adjustments: top-ranked gets +(n-1)*0.1, etc.
                # Max adjustment for 4 teams = 0.3, well below the 1e12 pts gap.
                for rank, pos in enumerate(order):
                    combined_f[tied[pos], sim_idx] += (n_tied - 1 - rank) * 0.1

        rankings = np.argsort(-combined_f, axis=0)  # (n_teams, N)

        # Convert team local indices → global team indices
        mi_arr = np.array(mi)
        ranked_global = mi_arr[rankings]  # (n_teams, N) global team index
        group_rankings[grp] = ranked_global  # row 0 = best team

        # Store raw stats for position/pts/gf/ga accumulation
        grp_stats[grp] = (mi, pts, gf, gd)

    return group_rankings, grp_stats


def _vec_ko_round(
    matchups: list[tuple[int, int]],
    participants: np.ndarray,
    att: np.ndarray,
    def_: np.ndarray,
    mu: float,
    N: int,
    rng: np.random.Generator,
    h2h_mat: np.ndarray,
) -> np.ndarray:
    """
    Simulate one KO round for all N sims simultaneously.

    participants: shape (M*2, N) — pairs of team indices
    matchups: list of (i, j) pairs into participants axis 0
    Returns winners: shape (M, N)
    """
    n_matches = len(matchups)
    winners = np.empty((n_matches, N), dtype=np.int32)

    for k, (i, j) in enumerate(matchups):
        ti = participants[i]  # shape (N,) — home team index per sim
        tj = participants[j]  # shape (N,) — away team index per sim
        h2h_scores = h2h_mat[ti, tj]  # shape (N,)
        h2h_factors = 1.0 + config.H2H_WEIGHT * h2h_scores
        lam_h = att[ti] * def_[tj] * mu * h2h_factors
        lam_a = att[tj] * def_[ti] * mu / h2h_factors
        hg = rng.poisson(lam_h)
        ag = rng.poisson(lam_a)
        draw = hg == ag
        if draw.any():
            # Penalty shootout for draws
            p_h = (att[ti] / def_[tj]) / (att[ti] / def_[tj] + att[tj] / def_[ti])
            p_h = np.clip(p_h, 0.35, 0.65)
            penalty_home_wins = rng.random(N) < p_h
            # Override draw outcomes
            hg = np.where(draw, np.where(penalty_home_wins, 1, 0), hg)
            ag = np.where(draw, np.where(penalty_home_wins, 0, 1), ag)

        winners[k] = np.where(hg > ag, ti, tj)

    return winners


# ---------------------------------------------------------------------------
# Full Monte Carlo run
# ---------------------------------------------------------------------------

def run_simulations(
    coeffs: dict[str, dict],
    n_simulations: int = config.DEFAULT_N_SIMULATIONS,
    mu: float = config.MU,
    seed: int | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    batch_size: int = 5_000,
    h2h_table: dict | None = None,
) -> dict:
    """
    Run N Monte Carlo simulations.

    Returns aggregated result dict:
    {
        team: {
            "r32": float, "r16": float, "qf": float,
            "sf": float, "final": float, "winner": float,
            "opponent_counts": {"r32": Counter, ...}  ← for curiosities
        }
    }
    Also returns raw run metadata.
    """
    rng = np.random.default_rng(seed)
    groups = load_groups()

    # Load real match results from group_results.json — store both orderings so
    # lookup by (members[i], members[j]) always finds the right entry.
    played_results: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
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
                played_results[grp] = grp_played

    ko_played = _load_ko_played()

    teams = sorted(coeffs.keys())
    n_teams = len(teams)
    t2i = {t: i for i, t in enumerate(teams)}

    att = np.array([coeffs[t]["att_rating"] for t in teams], dtype=np.float64)
    def_ = np.array([coeffs[t]["def_rating"] for t in teams], dtype=np.float64)
    att = np.clip(att, 1e-3, None)
    def_ = np.clip(def_, 1e-3, None)

    # Build H2H matrix (n_teams × n_teams), zeros when no table provided
    h2h_mat = np.zeros((n_teams, n_teams), dtype=np.float64)
    if h2h_table:
        for (a, b), score in h2h_table.items():
            if a in t2i and b in t2i:
                h2h_mat[t2i[a], t2i[b]] = score

    # Stage qualification counts — shape (n_teams,), accumulated across all batches
    counts_arr: dict[str, np.ndarray] = {
        s: np.zeros(n_teams, dtype=np.int64)
        for s in ("r32", "r16", "qf", "sf", "final", "winner")
    }
    # Opponent matrices — opp_mat[stage][winner_idx, loser_idx] = count
    _OPP_STAGES = ("r32", "r16", "qf", "sf", "final")
    opp_mat: dict[str, np.ndarray] = {
        s: np.zeros((n_teams, n_teams), dtype=np.int64) for s in _OPP_STAGES
    }
    # Final pairs — pair_counts_flat[min_idx * n_teams + max_idx] = count
    # teams[] is sorted alphabetically so index order == alphabetical order
    pair_counts_flat = np.zeros(n_teams * n_teams, dtype=np.int64)
    # Canonical final pairs dict (built once from pair_counts_flat at the end)
    final_pair_counts: dict[tuple[str, str], int] = {}

    # New accumulation arrays for Predictions tab
    pos_counts  = np.zeros((n_teams, 4), dtype=np.int64)   # pos_counts[t_idx, k] = times finishing pos k+1
    pts_sums    = np.zeros(n_teams, dtype=np.float64)
    gf_sums     = np.zeros(n_teams, dtype=np.float64)
    ga_sums     = np.zeros(n_teams, dtype=np.float64)

    # Per-bracket-slot occupancy
    r32_slot_cts = np.zeros((16, n_teams), dtype=np.int64)
    r32_home_cts = np.zeros((16, n_teams), dtype=np.int64)
    r32_away_cts = np.zeros((16, n_teams), dtype=np.int64)
    r16_slot_cts = np.zeros((8,  n_teams), dtype=np.int64)
    qf_slot_cts  = np.zeros((4,  n_teams), dtype=np.int64)
    sf_slot_cts  = np.zeros((2,  n_teams), dtype=np.int64)
    fin_slot_cts = np.zeros((2,  n_teams), dtype=np.int64)

    n_processed = 0
    t_start = time.time()

    while n_processed < n_simulations:
        N = min(batch_size, n_simulations - n_processed)

        # === Group stage (vectorised) ===
        grp_rankings, grp_stats = _vec_group_stage(
            groups, teams, att, def_, mu, N, rng, h2h_mat, played_results
        )
        # grp_rankings[grp]: shape (4, N), row 0 = winner

        # Accumulate position counts and pts/gf/ga sums
        for grp, ranking in grp_rankings.items():
            for pos in range(4):
                np.add.at(pos_counts[:, pos], ranking[pos], 1)

        for grp, (mi_g, pts_b, gf_b, gd_b) in grp_stats.items():
            for k, ti in enumerate(mi_g):
                pts_sums[ti] += float(pts_b[k].sum())
                gf_sums[ti]  += float(gf_b[k].sum())
                ga_sums[ti]  += float((gf_b[k] - gd_b[k]).sum())  # ga = gf - gd

        # Slots: 1X, 2X → global team indices
        slot: dict[str, np.ndarray] = {}
        thirds: list[tuple[str, np.ndarray]] = []

        for grp, ranking in grp_rankings.items():
            slot[f"1{grp}"] = ranking[0]  # shape (N,)
            slot[f"2{grp}"] = ranking[1]
            thirds.append((grp, ranking[2]))

        # Select best 8 3rd-place teams by actual pts → gd → gf (FIFA rules).
        # att_rating proxy was wrong: DR Congo (att=0.4, 4pts/GD+1) was excluded
        # despite being the strongest 3rd-place finisher.
        thirds_groups = [grp for grp, _ in thirds]
        thirds_stack = np.stack([arr for _, arr in thirds], axis=0)  # (12, N)

        thirds_pts = np.zeros((12, N), dtype=np.int32)
        thirds_gd  = np.zeros((12, N), dtype=np.int32)
        thirds_gf  = np.zeros((12, N), dtype=np.int32)
        _sidx = np.arange(N)
        for _i, _grp in enumerate(thirds_groups):
            _mi, _pts_g, _gf_g, _gd_g = grp_stats[_grp]
            _mi_arr = np.array(_mi)
            _t3 = thirds_stack[_i]                                   # (N,) global team idx
            _loc = (_mi_arr[:, None] == _t3[None, :]).argmax(axis=0)  # (N,) local idx
            thirds_pts[_i] = _pts_g[_loc, _sidx]
            thirds_gd[_i]  = _gd_g[_loc, _sidx]
            thirds_gf[_i]  = _gf_g[_loc, _sidx]

        _thirds_key = (thirds_pts.astype(np.float64) * 1e12
                       + thirds_gd.astype(np.float64) * 1e6
                       + thirds_gf.astype(np.float64))
        thirds_order = np.argsort(-_thirds_key, axis=0)              # (12, N) best first

        best8_local = thirds_order[:8]                               # (8, N) index into thirds list
        best8 = np.take_along_axis(thirds_stack, best8_local, axis=0)  # (8, N) team indices

        # Resolve "3W_X" slots: batch per unique qual_groups (≤495 combos)
        for sv in _WVT_SLOTS:
            slot[f"3W_{sv}"] = np.empty(N, dtype=np.int32)

        thirds_groups_arr = np.array(thirds_groups)             # (12,) object/str
        best8_groups      = thirds_groups_arr[best8_local]      # (8, N) group letters
        best8_groups_srt  = np.sort(best8_groups, axis=0)       # (8, N) sorted

        # Group sim indices by their unique sorted qual_groups tuple
        from collections import defaultdict as _dd
        sim_by_qg: dict[tuple, list[int]] = _dd(list)
        for sim_idx in range(N):
            sim_by_qg[tuple(best8_groups_srt[:, sim_idx])].append(sim_idx)

        for qual_groups, sim_indices in sim_by_qg.items():
            assign    = _assign_third_place(qual_groups)          # cached lru_cache
            arr_idx   = np.asarray(sim_indices, dtype=np.intp)
            bg_sub    = best8_groups[:, arr_idx]                  # (8, n_batch)
            b8_sub    = best8[:, arr_idx]                         # (8, n_batch)
            n_batch   = len(arr_idx)
            batch_rng = np.arange(n_batch)
            for winner_grp, third_grp in assign.items():
                k_pos = (bg_sub == third_grp).argmax(axis=0)     # (n_batch,) row
                slot[f"3W_{winner_grp}"][arr_idx] = b8_sub[k_pos, batch_rng]

        if n_processed == 0 and logger.isEnabledFor(logging.DEBUG):
            q0 = tuple(best8_groups_srt[:, 0])
            logger.debug("3rd-place qualifying groups (sim 0): %s", q0)
            for _k, (_sh, _sa) in enumerate(R32_BRACKET):
                logger.debug(
                    "  R32[%2d] %-6s vs %-6s  →  %-24s vs %s",
                    _k, _sh, _sa,
                    teams[int(slot[_sh][0])],
                    teams[int(slot[_sa][0])],
                )

        # Mark R32 qualifiers — vectorised
        for sh, sa in R32_BRACKET:
            np.add.at(counts_arr["r32"], slot[sh], 1)
            np.add.at(counts_arr["r32"], slot[sa], 1)

        # === R32 (16 matches) ===
        # Build participant array: shape (32, N) — ordered by R32_BRACKET slots
        home_slots = np.stack([slot[sh] for sh, _ in R32_BRACKET], axis=0)  # (16, N)
        away_slots = np.stack([slot[sa] for _, sa in R32_BRACKET], axis=0)  # (16, N)

        r32_matchups = [(i, 16 + i) for i in range(16)]
        participants_r32 = np.concatenate([home_slots, away_slots], axis=0)  # (32, N)
        r32_winners = _vec_ko_round(r32_matchups, participants_r32, att, def_, mu, N, rng, h2h_mat)
        # r32_winners: (16, N)

        # Override played R32 results with actual scores
        for k in range(16):
            side = ko_played.get(("r32", k))
            if side == "home":
                r32_winners[k] = home_slots[k]
            elif side == "away":
                r32_winners[k] = away_slots[k]

        # Accumulate R32 slot occupancy
        for k in range(16):
            np.add.at(r32_home_cts[k], home_slots[k], 1)
            np.add.at(r32_away_cts[k], away_slots[k], 1)
            np.add.at(r32_slot_cts[k], r32_winners[k], 1)

        # Opponent tracking for R32 — vectorised
        for k in range(16):
            h = home_slots[k]; a = away_slots[k]; w = r32_winners[k]
            l = np.where(w == h, a, h)
            np.add.at(opp_mat["r32"], (w, l), 1)
            np.add.at(opp_mat["r32"], (l, w), 1)

        # R16 qualifiers (= R32 winners) — vectorised
        for k in range(16):
            np.add.at(counts_arr["r16"], r32_winners[k], 1)

        # === R16 ===
        r16_part = r32_winners  # (16, N)
        r16_matchups = [(i, j) for i, j in R16_PAIRS]
        r16_winners = _vec_ko_round(r16_matchups, r16_part, att, def_, mu, N, rng, h2h_mat)
        # r16_winners: (8, N)

        # Override played R16 results
        for k, (i, j) in enumerate(R16_PAIRS):
            side = ko_played.get(("r16", k))
            if side == "home":
                r16_winners[k] = r32_winners[i]
            elif side == "away":
                r16_winners[k] = r32_winners[j]

        # Accumulate R16 slot occupancy
        for k in range(8):
            np.add.at(r16_slot_cts[k], r16_winners[k], 1)

        # Opponent tracking for R16 — vectorised
        for k, (i, j) in enumerate(R16_PAIRS):
            w = r16_winners[k]; ri = r32_winners[i]; rj = r32_winners[j]
            l = np.where(w == ri, rj, ri)
            np.add.at(opp_mat["r16"], (w, l), 1)
            np.add.at(opp_mat["r16"], (l, w), 1)

        # QF qualifiers (= R16 winners) — vectorised
        for k in range(8):
            np.add.at(counts_arr["qf"], r16_winners[k], 1)

        # === QF ===
        qf_part = r16_winners  # (8, N)
        qf_matchups = [(i, j) for i, j in QF_PAIRS]
        qf_winners = _vec_ko_round(qf_matchups, qf_part, att, def_, mu, N, rng, h2h_mat)
        # qf_winners: (4, N)

        # Override played QF results
        for k, (i, j) in enumerate(QF_PAIRS):
            side = ko_played.get(("qf", k))
            if side == "home":
                qf_winners[k] = r16_winners[i]
            elif side == "away":
                qf_winners[k] = r16_winners[j]

        # Opponent tracking vectorised
        for k, (i, j) in enumerate(QF_PAIRS):
            w = qf_winners[k]; ri = r16_winners[i]; rj = r16_winners[j]
            l = np.where(w == ri, rj, ri)
            np.add.at(opp_mat["qf"], (w, l), 1)
            np.add.at(opp_mat["qf"], (l, w), 1)

        # Accumulate QF slot occupancy
        for k in range(4):
            np.add.at(qf_slot_cts[k], qf_winners[k], 1)

        # SF qualifiers (= QF winners) — vectorised
        for k in range(4):
            np.add.at(counts_arr["sf"], qf_winners[k], 1)

        # === SF ===
        sf_part = qf_winners  # (4, N)
        sf_matchups = [(i, j) for i, j in SF_PAIRS]
        sf_winners = _vec_ko_round(sf_matchups, sf_part, att, def_, mu, N, rng, h2h_mat)
        # sf_winners: (2, N)

        # Override played SF results
        for k, (i, j) in enumerate(SF_PAIRS):
            side = ko_played.get(("sf", k))
            if side == "home":
                sf_winners[k] = qf_winners[i]
            elif side == "away":
                sf_winners[k] = qf_winners[j]

        # Opponent tracking vectorised
        for k, (i, j) in enumerate(SF_PAIRS):
            w = sf_winners[k]; ri = qf_winners[i]; rj = qf_winners[j]
            l = np.where(w == ri, rj, ri)
            np.add.at(opp_mat["sf"], (w, l), 1)
            np.add.at(opp_mat["sf"], (l, w), 1)

        # Accumulate SF slot occupancy
        for k in range(2):
            np.add.at(sf_slot_cts[k], sf_winners[k], 1)

        # Finalists (= SF winners) — vectorised
        for k in range(2):
            np.add.at(counts_arr["final"], sf_winners[k], 1)

        # === Final ===
        fin_part = sf_winners  # (2, N)
        fin_winners = _vec_ko_round([(0, 1)], fin_part, att, def_, mu, N, rng, h2h_mat)
        # fin_winners: (1, N)

        # Override played Final result
        side = ko_played.get(("final", 0))
        if side == "home":
            fin_winners[0] = sf_winners[0]
        elif side == "away":
            fin_winners[0] = sf_winners[1]

        w_fin = fin_winners[0]                                          # (N,)
        l_fin = np.where(w_fin == sf_winners[0], sf_winners[1], sf_winners[0])

        # Final opponent tracking — vectorised
        np.add.at(opp_mat["final"], (w_fin, l_fin), 1)
        np.add.at(opp_mat["final"], (l_fin, w_fin), 1)

        # Champion counts — vectorised
        np.add.at(counts_arr["winner"], w_fin, 1)

        # Final pairs: store in upper triangle (teams[] is sorted → index = alpha order)
        pair_min = np.minimum(w_fin, l_fin)
        pair_max = np.maximum(w_fin, l_fin)
        flat_idx = pair_min * n_teams + pair_max                        # (N,)
        pair_counts_flat += np.bincount(flat_idx, minlength=n_teams * n_teams).astype(np.int64)

        # Accumulate finalist slot occupancy (sf_winners = finalists)
        for k in range(2):
            np.add.at(fin_slot_cts[k], sf_winners[k], 1)

        n_processed += N
        elapsed = time.time() - t_start
        if progress_cb:
            progress_cb(n_processed, n_simulations, elapsed)

    # Build final_pair_counts from flat array (upper triangle = alphabetically sorted)
    for flat_i in np.nonzero(pair_counts_flat)[0]:
        ti_a = int(flat_i) // n_teams
        ti_b = int(flat_i) % n_teams
        final_pair_counts[(teams[ti_a], teams[ti_b])] = int(pair_counts_flat[flat_i])

    # Normalise to probabilities
    ns = float(n_simulations)

    def _top10_opp(row: np.ndarray) -> dict[str, int]:
        top_idx = np.argsort(-row)[:10]
        return {teams[j]: int(row[j]) for j in top_idx if row[j] > 0}

    agg: dict[str, dict] = {}
    for t in teams:
        ti = t2i[t]
        agg[t] = {
            "r32":    int(counts_arr["r32"][ti])   / ns,
            "r16":    int(counts_arr["r16"][ti])   / ns,
            "qf":     int(counts_arr["qf"][ti])    / ns,
            "sf":     int(counts_arr["sf"][ti])    / ns,
            "final":  int(counts_arr["final"][ti]) / ns,
            "winner": int(counts_arr["winner"][ti]) / ns,
            "opp_r32":   _top10_opp(opp_mat["r32"][ti]),
            "opp_r16":   _top10_opp(opp_mat["r16"][ti]),
            "opp_qf":    _top10_opp(opp_mat["qf"][ti]),
            "opp_sf":    _top10_opp(opp_mat["sf"][ti]),
            "opp_final": _top10_opp(opp_mat["final"][ti]),
        }

    def _slot_top(counts_row: np.ndarray, teams_list: list[str], n: int, top_k: int = 3) -> list[tuple]:
        """Return [(team_name, probability), ...] for a slot occupancy row."""
        idxs = np.argsort(-counts_row)[:top_k]
        return [(teams_list[i], int(counts_row[i]) / n) for i in idxs if counts_row[i] > 0]

    # Build group_predictions
    group_predictions: dict[str, dict] = {}
    for t in teams:
        ti = t2i[t]
        group_predictions[t] = {
            "pos1": pos_counts[ti, 0] / n_simulations,
            "pos2": pos_counts[ti, 1] / n_simulations,
            "pos3": pos_counts[ti, 2] / n_simulations,
            "pos4": pos_counts[ti, 3] / n_simulations,
            "avg_pts": pts_sums[ti] / n_simulations,
            "avg_gf":  gf_sums[ti]  / n_simulations,
            "avg_ga":  ga_sums[ti]  / n_simulations,
        }

    # Build bracket_slots
    bracket_slots: dict[str, dict] = {
        "r32": {
            k: {
                "home":   _slot_top(r32_home_cts[k], teams, n_simulations),
                "away":   _slot_top(r32_away_cts[k], teams, n_simulations),
                "winner": _slot_top(r32_slot_cts[k], teams, n_simulations),
            }
            for k in range(16)
        },
        "r16": {k: _slot_top(r16_slot_cts[k], teams, n_simulations) for k in range(8)},
        "qf":  {k: _slot_top(qf_slot_cts[k],  teams, n_simulations) for k in range(4)},
        "sf":  {k: _slot_top(sf_slot_cts[k],  teams, n_simulations) for k in range(2)},
        "final": {k: _slot_top(fin_slot_cts[k], teams, n_simulations) for k in range(2)},
    }

    return {
        "probabilities": agg,
        "n_simulations": n_simulations,
        "elapsed_seconds": time.time() - t_start,
        "mu": mu,
        "final_pairs": final_pair_counts,   # {(team_a, team_b): count}, team_a < team_b
        "group_predictions": group_predictions,
        "bracket_slots": bracket_slots,
    }
