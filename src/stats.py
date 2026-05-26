"""SQLite persistence — simulation runs, training runs, team results, finals."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import config
import streamlit as st

DB_PATH = Path(config.DB_PATH)


@st.cache_resource
def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_connection() -> sqlite3.Connection:
    return _get_db_conn()


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            n_simulations INTEGER NOT NULL,
            config_json   TEXT,
            metrics_json  TEXT,
            elapsed_s     REAL
        );

        CREATE TABLE IF NOT EXISTS team_results (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id),
            team   TEXT NOT NULL,
            r32    REAL,
            r16    REAL,
            qf     REAL,
            sf     REAL,
            final  REAL,
            winner REAL
        );

        CREATE TABLE IF NOT EXISTS finals (
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id),
            team_a TEXT    NOT NULL,
            team_b TEXT    NOT NULL,
            count  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, team_a, team_b)
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            date              TEXT NOT NULL,
            train_start       TEXT,
            train_end         TEXT,
            val_start         TEXT,
            val_end           TEXT,
            accuracy_before   REAL,
            accuracy_after    REAL,
            mae_before        REAL,
            mae_after         REAL,
            brier_before      REAL,
            brier_after       REAL,
            coefficients_json TEXT
        );
        """)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def save_simulation_run(
    probabilities: dict[str, dict],
    n_simulations: int,
    elapsed_s: float,
    cfg: dict | None = None,
    final_pairs: dict | None = None,
) -> int:
    """Persist a simulation run + finals. Returns the new run_id."""
    init_db()
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO simulation_runs (date, n_simulations, config_json, elapsed_s) VALUES (?,?,?,?)",
            (now, n_simulations, json.dumps(cfg or {}), elapsed_s),
        )
        run_id = cur.lastrowid

        rows = [
            (run_id, team, d["r32"], d["r16"], d["qf"], d["sf"], d["final"], d["winner"])
            for team, d in probabilities.items()
        ]
        conn.executemany(
            "INSERT INTO team_results (run_id, team, r32, r16, qf, sf, final, winner) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )

        if final_pairs:
            fin_rows = [
                (run_id, a, b, cnt)
                for (a, b), cnt in final_pairs.items()
            ]
            conn.executemany(
                "INSERT INTO finals (run_id, team_a, team_b, count) VALUES (?,?,?,?)",
                fin_rows,
            )

    return run_id


def save_training_run(result: dict, train_start: str, train_end: str, val_start: str, val_end: str) -> int:
    init_db()
    mb = result.get("metrics_before", {})
    ma = result.get("metrics_after", {})
    coeffs = {"att": result.get("att", {}), "def": result.get("def", {})}
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO training_runs
               (date, train_start, train_end, val_start, val_end,
                accuracy_before, accuracy_after, mae_before, mae_after,
                brier_before, brier_after, coefficients_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.utcnow().isoformat(),
                train_start, train_end, val_start, val_end,
                mb.get("accuracy"), ma.get("accuracy"),
                mb.get("mae"), ma.get("mae"),
                mb.get("brier"), ma.get("brier"),
                json.dumps(coeffs),
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_simulation_runs() -> list[dict]:
    """Return all simulation runs ordered newest first."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM simulation_runs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=60)
def load_run_results(run_id: int) -> list[dict]:
    """Return team_results for a given run_id."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM team_results WHERE run_id=? ORDER BY winner DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=60)
def load_training_runs() -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=60)
def get_least_probable_finals(run_id: int, limit: int = 5) -> list[dict]:
    """
    Return the rarest finals observed for a given simulation run.

    Each row: {team_a, team_b, count, pct}
    """
    init_db()
    with get_connection() as conn:
        n_row = conn.execute(
            "SELECT n_simulations FROM simulation_runs WHERE id=?", (run_id,)
        ).fetchone()
        if n_row is None:
            return []
        n = n_row["n_simulations"]

        rows = conn.execute(
            """SELECT team_a, team_b, count
               FROM finals
               WHERE run_id=?
               ORDER BY count ASC
               LIMIT ?""",
            (run_id, limit),
        ).fetchall()

    return [
        {
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "count": r["count"],
            "pct": r["count"] / n * 100 if n > 0 else 0.0,
        }
        for r in rows
    ]


@st.cache_data(ttl=60)
def get_most_probable_finals(run_id: int, limit: int = 5) -> list[dict]:
    """Return the most common finals observed for a given simulation run."""
    init_db()
    with get_connection() as conn:
        n_row = conn.execute(
            "SELECT n_simulations FROM simulation_runs WHERE id=?", (run_id,)
        ).fetchone()
        if n_row is None:
            return []
        n = n_row["n_simulations"]

        rows = conn.execute(
            """SELECT team_a, team_b, count
               FROM finals
               WHERE run_id=?
               ORDER BY count DESC
               LIMIT ?""",
            (run_id, limit),
        ).fetchall()

    return [
        {
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "count": r["count"],
            "pct": r["count"] / n * 100 if n > 0 else 0.0,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# In-memory curiosity helpers (no DB required)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def curiosities(probabilities: dict[str, dict], initial_coeffs: dict[str, dict]) -> dict:
    """Compute curiosity stats from a simulation result."""
    sorted_by_winner = sorted(probabilities.items(), key=lambda x: -x[1]["winner"])
    top16 = [t for t, _ in sorted_by_winner[:16]]
    top8  = [t for t, _ in sorted_by_winner[:8]]
    top16_set = set(top16)

    # Outsider teams (not in top-16 by winner prob) who reach SF most often
    outsiders = [(t, d["sf"]) for t, d in probabilities.items() if t not in top16_set]
    outsiders.sort(key=lambda x: -x[1])
    surprise_sf = outsiders[:3]

    # Top-8 favourite with lowest chance of reaching R32 (eliminated earliest)
    fav_exit = [(t, d) for t, d in probabilities.items() if t in top8]
    fav_exit.sort(key=lambda x: x[1]["r32"])
    earliest_fav_exit = fav_exit[:3]

    # Group of death: group whose members have the highest combined winner prob
    groups_data: dict[str, list[str]] = {}
    try:
        from pathlib import Path as _Path
        with (_Path("data") / "teams.json").open(encoding="utf-8") as f:
            groups_data = json.load(f)["groups"]
    except Exception:
        pass

    death_group = None
    max_talent = 0.0
    for grp, members in groups_data.items():
        talent = sum(probabilities.get(t, {}).get("winner", 0.0) for t in members)
        if talent > max_talent:
            max_talent = talent
            death_group = grp

    return {
        "surprise_sf": [{"team": t, "sf_prob": p} for t, p in surprise_sf],
        "earliest_fav_exit": [
            {"team": t, "r32_prob": d["r32"], "winner_prob": d["winner"]}
            for t, d in earliest_fav_exit
        ],
        "death_group": death_group,
        "death_group_members": groups_data.get(death_group, []) if death_group else [],
    }


@st.cache_data(ttl=300)
def get_group_predictions(sim_result: dict, groups_data: dict) -> dict[str, list[dict]]:
    """
    Return per-group sorted predicted standings.
    Result: {grp: [{"team", "pos1","pos2","pos3","pos4","avg_pts","avg_gf","avg_ga"}, ...sorted by pos1 desc]}
    """
    gp = sim_result.get("group_predictions", {})
    result = {}
    for grp, members in groups_data.items():
        rows = []
        for t in members:
            d = gp.get(t, {})
            rows.append({
                "team": t,
                "pos1": d.get("pos1", 0.0),
                "pos2": d.get("pos2", 0.0),
                "pos3": d.get("pos3", 0.0),
                "pos4": d.get("pos4", 0.0),
                "avg_pts": d.get("avg_pts", 0.0),
                "avg_gf":  d.get("avg_gf",  0.0),
                "avg_ga":  d.get("avg_ga",  0.0),
            })
        rows.sort(key=lambda r: -r["pos1"])
        result[grp] = rows
    return result


@st.cache_data(ttl=300)
def get_probable_bracket(
    sim_result: dict,
    coeffs: dict[str, dict],
    mu: float,
    r32_labels: list[str],
    r16_labels: list[str],
    schedule: dict,
) -> dict:
    """
    Build the probable bracket for display.
    Returns a dict with r32/r16/qf/sf/final match data.
    Each match: {label, home_candidates, away_candidates, winner_candidates, sched, prob_score}
    """
    import math
    from src.tournament import R16_PAIRS, QF_PAIRS, SF_PAIRS
    bs = sim_result.get("bracket_slots", {})
    n = sim_result.get("n_simulations", 1)

    def poisson_mode(lam):
        return max(0, int(lam)) if lam >= 1.0 else 0

    def poisson_draw_prob(lam_h, lam_a, kmax=10):
        p = 0.0
        for k in range(kmax + 1):
            ph = math.exp(-lam_h) * (lam_h ** k) / math.factorial(k)
            pa = math.exp(-lam_a) * (lam_a ** k) / math.factorial(k)
            p += ph * pa
        return p

    def prob_score(team_h: str, team_a: str):
        if not team_h or not team_a:
            return None
        ch = coeffs.get(team_h, {"att_rating": 1.0, "def_rating": 1.0})
        ca = coeffs.get(team_a, {"att_rating": 1.0, "def_rating": 1.0})
        lam_h = ch["att_rating"] * ca["def_rating"] * mu
        lam_a = ca["att_rating"] * ch["def_rating"] * mu
        p_draw = poisson_draw_prob(lam_h, lam_a)
        ratio = (ch["att_rating"] / ca["def_rating"]) / (ca["att_rating"] / ch["def_rating"])
        p_pen_h = ratio / (1 + ratio)
        p_pen_h = min(max(p_pen_h, 0.35), 0.65)
        return {
            "score_h": poisson_mode(lam_h),
            "score_a": poisson_mode(lam_a),
            "p_draw": p_draw,
            "p_pen_h": p_pen_h,
        }

    def build_match(stage, slot_k, home_cands, away_cands, winner_cands, label, sched_entry):
        top_h = home_cands[0][0] if home_cands else ""
        top_a = away_cands[0][0] if away_cands else ""
        return {
            "stage": stage,
            "slot_k": slot_k,
            "label": label,
            "home_candidates": home_cands,
            "away_candidates": away_cands,
            "winner_candidates": winner_cands,
            "sched": sched_entry,
            "prob_score": prob_score(top_h, top_a),
        }

    r32_matches = []
    for k in range(16):
        slotdata = bs.get("r32", {}).get(k, {})
        r32_matches.append(build_match(
            "r32", k,
            slotdata.get("home", []),
            slotdata.get("away", []),
            slotdata.get("winner", []),
            r32_labels[k] if k < len(r32_labels) else f"R32-{k+1}",
            schedule.get(("r32", k)),
        ))

    r16_matches = []
    for k, (i, j) in enumerate(R16_PAIRS):
        r16_cands_h = bs.get("r32", {}).get(i, {}).get("winner", [])
        r16_cands_a = bs.get("r32", {}).get(j, {}).get("winner", [])
        r16_winner_cands = bs.get("r16", {}).get(k, [])
        r16_matches.append(build_match(
            "r16", k,
            r16_cands_h, r16_cands_a, r16_winner_cands,
            r16_labels[k] if k < len(r16_labels) else f"R16-{k+1}",
            schedule.get(("r16", k)),
        ))

    qf_matches = []
    for k, (i, j) in enumerate(QF_PAIRS):
        qf_cands_h = list(bs.get("r16", {}).get(i, []))
        qf_cands_a = list(bs.get("r16", {}).get(j, []))
        qf_winner_cands = bs.get("qf", {}).get(k, [])
        qf_matches.append(build_match(
            "qf", k,
            qf_cands_h, qf_cands_a, qf_winner_cands,
            f"QF-{k+1}", schedule.get(("qf", k)),
        ))

    # Deduplicate: a team may appear as top candidate in multiple QF slots when
    # it can reach different bracket paths (e.g. by finishing 1st or 2nd in its
    # group).  Keep each team in the slot where it has the highest probability;
    # fall back to the next candidate in other slots.
    _top_entries: list[tuple[float, str, int, str]] = []
    for mi, m in enumerate(qf_matches):
        for side in ("home_candidates", "away_candidates"):
            if m[side]:
                _t, _p = m[side][0]
                _top_entries.append((_p, _t, mi, side))
    _top_entries.sort(reverse=True)
    _qf_used: set[str] = set()
    for _, team, mi, side in _top_entries:
        if team in _qf_used:
            cands = qf_matches[mi][side]
            repl = next(((t, p) for t, p in cands if t not in _qf_used), None)
            if repl:
                qf_matches[mi][side] = [repl] + [(t, p) for t, p in cands if t != repl[0]]
                _qf_used.add(repl[0])
                th = (qf_matches[mi]["home_candidates"] or [("", 0)])[0][0]
                ta = (qf_matches[mi]["away_candidates"] or [("", 0)])[0][0]
                qf_matches[mi]["prob_score"] = prob_score(th, ta)
        else:
            _qf_used.add(team)

    # Synchronise R16 winner labels with post-dedup QF candidates.
    # After dedup, a QF slot may show team B even though R16 winner_candidates
    # still lists team A first (A had more raw R16 wins but was displaced).
    for qf_k, (r16_i, r16_j) in enumerate(QF_PAIRS):
        for r16_k, qf_side in [(r16_i, "home_candidates"), (r16_j, "away_candidates")]:
            qf_team = (qf_matches[qf_k][qf_side] or [(None, 0)])[0][0]
            if not qf_team:
                continue
            wc = r16_matches[r16_k]["winner_candidates"]
            if not wc or wc[0][0] == qf_team:
                continue
            qf_prob = next((p for t, p in wc if t == qf_team), None)
            if qf_prob is not None:
                r16_matches[r16_k]["winner_candidates"] = (
                    [(qf_team, qf_prob)] + [(t, p) for t, p in wc if t != qf_team]
                )

    sf_matches = []
    for k, (i, j) in enumerate(SF_PAIRS):
        sf_cands_h = bs.get("qf", {}).get(i, [])
        sf_cands_a = bs.get("qf", {}).get(j, [])
        sf_winner_cands = bs.get("sf", {}).get(k, [])
        sf_matches.append(build_match(
            "sf", k,
            sf_cands_h, sf_cands_a, sf_winner_cands,
            f"SF-{k+1}", schedule.get(("sf", k)),
        ))

    final_h = bs.get("final", {}).get(0, [])
    final_a = bs.get("final", {}).get(1, [])
    final_match = build_match(
        "final", None,
        final_h, final_a, [],
        "Finale — M104",
        schedule.get(("final", None)),
    )

    return {
        "r32": r32_matches,
        "r16": r16_matches,
        "qf": qf_matches,
        "sf": sf_matches,
        "final": final_match,
    }


@st.cache_data(ttl=300)
def get_upsets(sim_result: dict, top_n_fav: int = 8) -> dict:
    """
    Returns upset analysis:
    - upset_r32: top 5 favourites most often eliminated in R32
    - upset_r16: top 5 favourites most often eliminated in R16
    - outsider_qf: top 3 non-top8 teams reaching QF most often
    - biggest_surprise: non-top8 team with highest winner prob
    """
    probs = sim_result.get("probabilities", {})
    sorted_by_winner = sorted(probs.items(), key=lambda x: -x[1]["winner"])
    top8 = {t for t, _ in sorted_by_winner[:top_n_fav]}

    # Favourites (top-8 by winner prob) eliminated in R32
    upset_r32 = [
        {"team": t, "r32_pct": d["r32"], "elim_r32_pct": d["r32"] - d["r16"]}
        for t, d in sorted_by_winner[:top_n_fav]
    ]
    upset_r32.sort(key=lambda x: -x["elim_r32_pct"])

    # Favourites eliminated in R16
    upset_r16 = [
        {"team": t, "r16_pct": d["r16"], "elim_r16_pct": d["r16"] - d["qf"]}
        for t, d in sorted_by_winner[:top_n_fav]
    ]
    upset_r16.sort(key=lambda x: -x["elim_r16_pct"])

    # Outsider QF: non-top8 teams reaching QF most often
    outsider_qf = [
        {"team": t, "qf_pct": d["qf"], "winner_pct": d["winner"]}
        for t, d in probs.items() if t not in top8
    ]
    outsider_qf.sort(key=lambda x: -x["qf_pct"])

    # Biggest surprise: non-top8 with highest winner prob
    biggest_surprise = outsider_qf[0] if outsider_qf else None

    return {
        "upset_r32": upset_r32[:5],
        "upset_r16": upset_r16[:5],
        "outsider_qf": outsider_qf[:3],
        "biggest_surprise": biggest_surprise,
    }


@st.cache_data(ttl=60)
def get_exit_distribution(run_id: int, team: str) -> dict[str, float]:
    """
    Return cumulative reaching-probability for one team in a given run.
    Keys: Groupes, R32, R16, QF, SF, Finale, Champion.
    Values: % of simulations that reached (or won) that round (0-100).
    Groupes is always 100.0; values decrease monotonically left to right.
    """
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT r32, r16, qf, sf, final, winner FROM team_results WHERE run_id=? AND team=?",
            (run_id, team),
        ).fetchone()
    if not row:
        return {}
    r32, r16, qf, sf, fin, win = (float(v or 0) for v in row)
    return {
        "Groupes": 100.0,
        "R32":     round(r32 * 100, 2),
        "R16":     round(r16 * 100, 2),
        "QF":      round(qf  * 100, 2),
        "SF":      round(sf  * 100, 2),
        "Finale":  round(fin * 100, 2),
        "Champion":round(win * 100, 2),
    }


def finals_from_memory(
    final_pairs: dict,
    n_simulations: int,
    least: bool = True,
    limit: int = 5,
) -> list[dict]:
    """
    Sort final pairs from the in-memory simulation result.

    final_pairs: {(team_a, team_b): count}
    Returns list of {team_a, team_b, count, pct} sorted by count asc (least=True) or desc.
    """
    if not final_pairs:
        return []
    pairs = [
        {"team_a": a, "team_b": b, "count": cnt,
         "pct": cnt / n_simulations * 100 if n_simulations > 0 else 0.0}
        for (a, b), cnt in final_pairs.items()
    ]
    pairs.sort(key=lambda x: x["count"], reverse=not least)
    return pairs[:limit]
