"""Shared helpers, path constants, and persistence utilities for the UI layer."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import streamlit as st

import config
from src.i18n import fmt_time_local, t
from src.tournament import _assign_third_place

# ---------------------------------------------------------------------------
# Session-state defaults (shared with app.py)
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "step1_complete": False,
    "step2_complete": False,
    "matches": [],
    "n_matches_by_team": {},
    "coefficients": {},
    "calibrated_coeffs": {},
    "training_result": None,
    "sim_result": None,
    "current_run_id": None,
    "h2h_table": {},
    "h2h_top": [],
    "focus_team": "France",
    "lang": "fr",
    "tz": "Europe/Paris",
    "last_recalib_n_played": -1,
    "last_recalib_time": None,
}

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_GROUP_RESULTS_PATH = Path("data/group_results.json")
_KO_RESULTS_PATH    = Path("data/ko_results.json")
_SCHEDULE_PATH      = Path("data/schedule.json")
_USER_PREFS_PATH    = Path("data/user_prefs.json")

# ---------------------------------------------------------------------------
# R32 / R16 match labels (module-level, always available)
# ---------------------------------------------------------------------------
R32_LABELS: list[str] = [
    "M74: W-E vs 3e", "M77: W-I vs 3e",
    "M73: RU-A vs RU-B", "M75: W-F vs RU-C",
    "M76: W-C vs RU-F", "M78: RU-E vs RU-I",
    "M79: W-A vs 3e", "M80: W-L vs 3e",
    "M83: RU-K vs RU-L", "M84: W-H vs RU-J",
    "M81: W-D vs 3e", "M82: W-G vs 3e",
    "M86: W-J vs RU-H", "M88: RU-D vs RU-G",
    "M85: W-B vs 3e", "M87: W-K vs 3e",
]
R16_LABELS: list[str] = [
    "M89: V.M74-M77", "M90: V.M73-M75",
    "M91: V.M76-M78", "M92: V.M79-M80",
    "M93: V.M83-M84", "M94: V.M81-M82",
    "M95: V.M86-M88", "M96: V.M85-M87",
]

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_teams_json() -> dict:
    with (Path("data") / "teams.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _step_badge(label: str, done: bool, active: bool) -> str:
    if done:
        return f"**✅ {label}**"
    if active:
        return f"**🔄 {label}**"
    return f"🔒 {label}"


def _active_coeffs() -> dict[str, dict]:
    if st.session_state.calibrated_coeffs:
        return st.session_state.calibrated_coeffs
    return st.session_state.coefficients


# ---------------------------------------------------------------------------
# Preferences persistence
# ---------------------------------------------------------------------------

def _load_prefs() -> dict:
    if _USER_PREFS_PATH.exists():
        with _USER_PREFS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_prefs(updates: dict) -> None:
    data = _load_prefs()
    data.update(updates)
    with _USER_PREFS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Group-stage persistence
# ---------------------------------------------------------------------------

def _load_gr() -> dict:
    if _GROUP_RESULTS_PATH.exists():
        with _GROUP_RESULTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_gr(data: dict) -> None:
    with _GROUP_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _init_gr(groups: dict) -> dict:
    result: dict = {}
    for grp, teams in groups.items():
        matches = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matches.append({
                    "home": teams[i], "away": teams[j],
                    "home_goals": None, "away_goals": None, "played": False,
                })
        result[grp] = {"matches": matches}
    return result


def _group_standings(teams: list, matches: list) -> list:
    stats = {t_: {"team": t_, "pts": 0, "j": 0, "g": 0, "n": 0, "p": 0, "bp": 0, "bc": 0, "diff": 0}
             for t_ in teams}
    for m in matches:
        if not m.get("played"):
            continue
        h, a = m["home"], m["away"]
        hg, ag = int(m.get("home_goals") or 0), int(m.get("away_goals") or 0)
        stats[h]["j"] += 1; stats[a]["j"] += 1
        stats[h]["bp"] += hg; stats[h]["bc"] += ag; stats[h]["diff"] += hg - ag
        stats[a]["bp"] += ag; stats[a]["bc"] += hg; stats[a]["diff"] += ag - hg
        if hg > ag:
            stats[h]["pts"] += 3; stats[h]["g"] += 1; stats[a]["p"] += 1
        elif hg < ag:
            stats[a]["pts"] += 3; stats[a]["g"] += 1; stats[h]["p"] += 1
        else:
            stats[h]["pts"] += 1; stats[h]["n"] += 1
            stats[a]["pts"] += 1; stats[a]["n"] += 1
    ranked = sorted(stats.values(), key=lambda r: (-r["pts"], -r["diff"], -r["bp"], r["team"]))
    for i, r in enumerate(ranked):
        r["pos"] = i + 1
    return ranked


def _build_slot_map(gr_data: dict, groups: dict) -> dict:
    slot_team: dict[str, str] = {}
    all_thirds = []
    for grp, teams in groups.items():
        ranked = _group_standings(teams, gr_data.get(grp, {}).get("matches", []))
        if ranked and ranked[0]["j"] > 0:
            slot_team[f"1{grp}"] = ranked[0]["team"]
        if len(ranked) > 1 and ranked[1]["j"] > 0:
            slot_team[f"2{grp}"] = ranked[1]["team"]
        if len(ranked) > 2 and ranked[2]["j"] > 0:
            all_thirds.append({
                "grp": grp, "team": ranked[2]["team"],
                "pts": ranked[2]["pts"], "diff": ranked[2]["diff"], "bp": ranked[2]["bp"],
            })
    if len(all_thirds) >= 8:
        thirds_sorted = sorted(all_thirds, key=lambda r: (-r["pts"], -r["diff"], -r["bp"]))[:8]
        qual_groups = tuple(sorted(t_["grp"] for t_ in thirds_sorted))
        third_by_grp = {t_["grp"]: t_["team"] for t_ in thirds_sorted}
        assign = _assign_third_place(qual_groups)
        for winner_grp, third_grp in assign.items():
            slot_team[f"3W_{winner_grp}"] = third_by_grp.get(third_grp, f"3e-{third_grp}")
    return slot_team


# ---------------------------------------------------------------------------
# KO persistence
# ---------------------------------------------------------------------------

def _load_ko() -> dict:
    if _KO_RESULTS_PATH.exists():
        with _KO_RESULTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_ko(data: dict) -> None:
    with _KO_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _init_ko() -> dict:
    return {
        "r32":         [{"goals_h": None, "goals_a": None, "played": False} for _ in range(16)],
        "r16":         [{"goals_h": None, "goals_a": None, "played": False} for _ in range(8)],
        "qf":          [{"goals_h": None, "goals_a": None, "played": False} for _ in range(4)],
        "sf":          [{"goals_h": None, "goals_a": None, "played": False} for _ in range(2)],
        "final":       {"goals_h": None, "goals_a": None, "played": False},
        "third_place": {"goals_h": None, "goals_a": None, "played": False},
    }


@st.cache_data
def _load_schedule() -> dict[tuple, dict]:
    if not _SCHEDULE_PATH.exists():
        return {}
    with _SCHEDULE_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    lkp: dict[tuple, dict] = {}
    for m in raw.get("group_matches", []):
        lkp[(m["home"], m["away"])] = m
        lkp[(m["away"], m["home"])] = m
    return lkp


@st.cache_data
def _load_ko_sched() -> dict[tuple, dict]:
    if not _SCHEDULE_PATH.exists():
        return {}
    with _SCHEDULE_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {(m["round"], m["bracket_idx"]): m for m in raw.get("ko_matches", [])}


# ---------------------------------------------------------------------------
# KO match widget helpers
# ---------------------------------------------------------------------------

_KO_PLACEHOLDER_PREFIXES = ("V.", "1er ", "2e ", "3e->", "Perd.")


def _ko_is_placeholder(team: str) -> bool:
    return any(team.startswith(p) for p in _KO_PLACEHOLDER_PREFIXES)


def _ko_team_winner(res: dict, team_h: str, team_a: str) -> str | None:
    if not res.get("played"):
        return None
    hg, ag = int(res.get("goals_h") or 0), int(res.get("goals_a") or 0)
    if hg > ag:
        return team_h
    if ag > hg:
        return team_a
    return None


def _ko_match_input(
    stage: str, idx, team_h: str, team_a: str, label: str, ko: dict,
    sched: dict | None = None,
    lang: str = "fr",
) -> bool:
    res     = ko[stage] if idx is None else ko[stage][idx]
    cur_hg  = int(res.get("goals_h") or 0)
    cur_ag  = int(res.get("goals_a") or 0)
    cur_pl  = bool(res.get("played", False))
    wk      = f"{stage}_{idx}"
    teams_known = not (_ko_is_placeholder(team_h) or _ko_is_placeholder(team_a))

    with st.container(border=True):
        if sched and sched.get("date"):
            _d  = sched["date"]
            _t_ = sched.get("time_utc", "")
            _v  = sched.get("venue", "")
            _pts = _d.split("-")
            _ddmm = f"{_pts[2]}/{_pts[1]}" if len(_pts) == 3 else _d
            _tz  = st.session_state.get("tz", "Europe/Paris")
            _tl  = fmt_time_local(_d, _t_, _tz, lang) if _t_ else f"{_t_}Z"
            _date_label = f"🗓 {_ddmm} · {_tl}"
            try:
                _past = datetime.strptime(_d, "%Y-%m-%d").date() < date.today()
            except Exception:
                _past = False
            _missing = _past and not cur_pl
            _col = "#888" if not teams_known else ("#c0392b" if _missing else "#2c3e50")
            _esc_v = _v.replace('"', "&quot;")
            _warn  = " ⚠️" if _missing else ""
            st.markdown(
                f'<span title="{_esc_v}" style="color:{_col};font-size:0.78em">'
                f'{_date_label}{_warn}</span>',
                unsafe_allow_html=True,
            )
            if _missing:
                st.caption(t("score_missing_caption", lang))

        st.caption(label)
        _a, _b, _sep, _c, _d_col = st.columns([3.5, 1.2, 0.4, 1.2, 1.8])
        _a.markdown(f"**{team_h}** — **{team_a}**")
        hg = _b.number_input("h", min_value=0, max_value=30, value=cur_hg, step=1,
                              key=f"ko_h_{wk}", label_visibility="collapsed")
        _sep.markdown("<div style='text-align:center;padding-top:6px'>-</div>",
                      unsafe_allow_html=True)
        ag = _c.number_input("a", min_value=0, max_value=30, value=cur_ag, step=1,
                              key=f"ko_a_{wk}", label_visibility="collapsed")
        pl = _d_col.checkbox(t("played_checkbox", lang), value=cur_pl, key=f"ko_p_{wk}")

    changed = hg != cur_hg or ag != cur_ag or pl != cur_pl
    if changed:
        upd = {"goals_h": int(hg), "goals_a": int(ag), "played": pl}
        if idx is None:
            ko[stage].update(upd)
        else:
            ko[stage][idx].update(upd)
    return changed
