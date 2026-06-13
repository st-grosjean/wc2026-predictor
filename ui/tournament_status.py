"""Tournament status banner — recalibration pipeline button."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from src import stats as db
from src.i18n import t
from ui.common import _active_coeffs, _load_gr, _load_ko, load_teams_json


def _count_played() -> int:
    gr_data = _load_gr()
    ko_data = _load_ko()
    n = sum(1 for g in gr_data.values() for m in g.get("matches", []) if m.get("played"))
    for stage in ("r32", "r16", "qf", "sf"):
        items = ko_data.get(stage, [])
        if isinstance(items, list):
            n += sum(1 for m in items if m.get("played"))
    for stage in ("final", "third_place"):
        m = ko_data.get(stage, {})
        if isinstance(m, dict) and m.get("played"):
            n += 1
    return n


def render_tournament_status(lang: str) -> None:
    """Compact status banner + one-click recalibration pipeline."""
    n_played = _count_played()
    has_sim  = bool(st.session_state.get("sim_result"))
    last_n   = st.session_state.get("last_recalib_n_played", -1)
    last_ts  = st.session_state.get("last_recalib_time")

    if n_played == 0:
        status_md = f"🔴 **{t('status_pretournoi', lang)}**"
    elif has_sim and last_ts and last_n == n_played:
        status_md = f"🟢 **{t('status_calibrated', lang, n=n_played, ts=last_ts)}**"
    else:
        status_md = f"🟡 **{t('status_update_avail', lang, n=n_played)}**"

    with st.container(border=True):
        col_s, col_b = st.columns([4, 1])
        col_s.markdown(status_md)
        clicked = col_b.button(
            t("btn_recalib", lang),
            type="primary",
            key="btn_recalib_main",
        )

    if n_played == 0 or not has_sim:
        st.info(t("guide_no_sim", lang))

    if clicked:
        _run_pipeline(lang, n_played)


def _run_pipeline(lang: str, n_played: int) -> None:
    _log: list[str] = []
    _ph = st.empty()

    def _step(msg: str) -> None:
        _log.append(msg)
        _ph.markdown("  \n".join(_log))

    _step(t("recalib_step1", lang))

    teams_json = load_teams_json()
    teams_list = list(teams_json["teams"].keys())
    matches    = st.session_state.get("matches", [])

    # Bug 1 fix: anchor L2 to calibrated coefficients, not raw Step-1 values
    initial_coeffs = _active_coeffs() or st.session_state.get("coefficients", {})

    # Bug 2 fix: load live WC2026 results and add to training data
    _sched_lkp: dict[tuple, dict] = {}
    _sched_path = Path("data/schedule.json")
    if _sched_path.exists():
        with _sched_path.open(encoding="utf-8") as _f:
            for _sm in json.load(_f).get("group_matches", []):
                _sched_lkp[(_sm["home"], _sm["away"])] = _sm
                _sched_lkp[(_sm["away"], _sm["home"])] = _sm

    live_matches: list[dict] = []
    for _gd in _load_gr().values():
        for _m in _gd.get("matches", []):
            if not _m.get("played") or _m.get("home_goals") is None:
                continue
            _h, _a = _m["home"], _m["away"]
            _si = _sched_lkp.get((_h, _a), {})
            live_matches.append({
                "home_team":        _h,
                "away_team":        _a,
                "home_goals":       int(_m["home_goals"]),
                "away_goals":       int(_m["away_goals"]),
                "date":             _si.get("date", "2026-06-15"),
                "competition_type": "WC_LIVE",
            })

    augmented_matches = matches + live_matches

    if augmented_matches:
        _step(t("recalib_step2", lang))
        try:
            from src.training import calibrate
            _live_weights = {**config.MATCH_WEIGHTS, "WC_LIVE": config.LIVE_MATCH_WEIGHT}
            result = calibrate(
                matches=augmented_matches,
                teams=teams_list,
                initial_coeffs=initial_coeffs,
                train_start=config.TRAIN_START,
                train_end="2026-12-31",
                val_start=config.VAL_START,
                val_end=config.VAL_END,
                custom_weights=_live_weights,
                l2_lambda=config.LIVE_L2_LAMBDA,
            )
            calibrated = {
                team_: {
                    "att_rating": result["att"].get(team_, 1.0),
                    "def_rating": result["def"].get(team_, 1.0),
                    # Bug 3 fix: n_matches from active (calibrated) coefficients
                    "n_matches":  initial_coeffs.get(team_, {}).get("n_matches", 0),
                }
                for team_ in teams_list
            }
            st.session_state.calibrated_coeffs = calibrated
            st.session_state.training_result   = result
            db.load_training_runs.clear()
        except Exception as e:
            st.warning(t("recalib_calib_warn", lang, msg=e))
    else:
        _step(t("recalib_step2_skip", lang))

    _step(t("recalib_step3", lang))
    active = _active_coeffs()
    if not active:
        active = {
            team_: {
                "att_rating": d["att_rating"],
                "def_rating": d["def_rating"],
                "n_matches":  0,
            }
            for team_, d in teams_json["teams"].items()
        }

    try:
        from src.montecarlo import run_simulations
        sim = run_simulations(
            coeffs=active,
            n_simulations=500_000,
            mu=config.MU,
            h2h_table=st.session_state.get("h2h_table") or None,
        )
        st.session_state.sim_result       = sim
        st.session_state.current_run_id   = db.save_simulation_run(
            sim["probabilities"],
            500_000,
            sim["elapsed_seconds"],
            {"mu": config.MU, "source": "recalib"},
            final_pairs=sim.get("final_pairs", {}),
        )
        for _fn in (
            db.load_simulation_runs, db.load_run_results,
            db.curiosities, db.get_group_predictions,
            db.get_probable_bracket, db.get_upsets,
        ):
            _fn.clear()
    except Exception as e:
        _ph.empty()
        st.error(t("error_sims", lang, msg=e))
        return

    _step(t("recalib_step4", lang))
    st.session_state["last_recalib_n_played"] = n_played
    st.session_state["last_recalib_time"]     = datetime.now().strftime("%H:%M")
    st.rerun()
