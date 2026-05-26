"""Étape 3 — Monte Carlo + orchestration des 5 onglets de résultats."""
from __future__ import annotations

import streamlit as st

import config
from src import stats as db
from src.i18n import t
from ui.common import _active_coeffs, load_teams_json
from ui.tab_curiosities import render_tab_curiosities
from ui.tab_focus import render_tab_focus
from ui.tab_global import render_tab_global
from ui.tab_predictions import render_tab_predictions


def render_step3(lang: str, s2_done: bool) -> None:
    with st.expander(t("step3_expander", lang), expanded=s2_done):
        if not s2_done:
            st.warning(t("warn_complete_step2", lang))
            return

        active_coeffs = _active_coeffs()
        if not active_coeffs:
            teams_json = load_teams_json()
            for team_, d in teams_json["teams"].items():
                active_coeffs[team_] = {
                    "att_rating": d["att_rating"],
                    "def_rating": d["def_rating"],
                    "n_matches": 0,
                }
            st.session_state.calibrated_coeffs = active_coeffs

        n_sims = st.select_slider(
            t("n_sims_label", lang),
            options=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
            value=10_000,
        )
        mu_sim = st.slider(t("mu_sim_label", lang), 1.5, 4.0, config.MU, 0.05, key="mu_sim")

        if st.button(t("btn_run_sims", lang)):
            from src.montecarlo import run_simulations
            progress_bar = st.progress(0)
            eta_text     = st.empty()

            def prog_cb(curr, total, elapsed):
                pct = curr / total
                progress_bar.progress(pct)
                if elapsed > 0 and pct > 0:
                    eta = elapsed / pct * (1 - pct)
                    eta_text.text(t("sims_eta", lang,
                                    curr=f"{curr:,}", total=f"{total:,}", eta=f"{eta:.0f}"))

            with st.spinner(t("spinner_sims", lang)):
                try:
                    result = run_simulations(
                        coeffs=active_coeffs,
                        n_simulations=n_sims,
                        mu=mu_sim,
                        progress_cb=prog_cb,
                        h2h_table=st.session_state.h2h_table or None,
                    )
                    st.session_state.sim_result = result
                    run_id = db.save_simulation_run(
                        result["probabilities"],
                        n_sims,
                        result["elapsed_seconds"],
                        {"mu": mu_sim},
                        final_pairs=result.get("final_pairs", {}),
                    )
                    st.session_state.current_run_id = run_id
                    # Invalider les caches de présentation qui dépendent des résultats
                    db.load_simulation_runs.clear()
                    db.load_run_results.clear()
                    db.curiosities.clear()
                    db.get_group_predictions.clear()
                    db.get_probable_bracket.clear()
                    db.get_upsets.clear()
                    st.success(t("sims_success", lang,
                                 n=f"{n_sims:,}",
                                 elapsed=f"{result['elapsed_seconds']:.1f}",
                                 run_id=run_id))
                except Exception as e:
                    import traceback
                    st.error(t("error_sims", lang, msg=f"{e}\n{traceback.format_exc()}"))

            progress_bar.empty()
            eta_text.empty()

        if not st.session_state.sim_result:
            return

        import pandas as pd
        sim          = st.session_state.sim_result
        probs        = sim["probabilities"]
        n            = sim["n_simulations"]
        sorted_teams = sorted(probs.items(), key=lambda x: -x[1]["winner"])
        _focus_team  = st.session_state.get("focus_team", "France")

        tab_global, tab_france, tab_curiosities, tab_history, tab_predictions = st.tabs([
            t("tab_global", lang),
            t("tab_focus", lang, team=_focus_team),
            t("tab_curiosities_renamed", lang),
            t("tab_history", lang),
            t("tab_predictions", lang),
        ])

        with tab_global:
            render_tab_global(lang, probs, n, sorted_teams)

        with tab_france:
            render_tab_focus(lang, probs, n, active_coeffs, mu_sim)

        with tab_curiosities:
            render_tab_curiosities(lang, probs, n, sim, active_coeffs)

        with tab_history:
            _render_tab_history(lang)

        with tab_predictions:
            render_tab_predictions(lang, sim, probs, n, sorted_teams, active_coeffs)


def _render_tab_history(lang: str) -> None:
    import pandas as pd
    st.subheader(t("history_header", lang))
    runs = db.load_simulation_runs()
    if not runs:
        st.info(t("no_sims_recorded", lang))
        return

    runs_df = pd.DataFrame([{
        t("col_id", lang):         r["id"],
        t("col_date", lang):       r["date"][:19],
        t("col_n_sims", lang):     f"{r['n_simulations']:,}",
        t("col_config", lang):     r["config_json"],
        t("col_duration_s", lang): round(r.get("elapsed_s") or 0, 1),
    } for r in runs])
    st.dataframe(runs_df, width="stretch")

    if len(runs) < 2:
        return

    st.subheader(t("compare_runs_header", lang))
    run_ids = [r["id"] for r in runs]
    col_r1, col_r2 = st.columns(2)
    rid1 = col_r1.selectbox(t("run_a_label", lang), run_ids, index=0)
    rid2 = col_r2.selectbox(t("run_b_label", lang), run_ids, index=1)

    if st.button(t("btn_compare", lang)):
        res1 = {r["team"]: r for r in db.load_run_results(rid1)}
        res2 = {r["team"]: r for r in db.load_run_results(rid2)}
        teams_both = set(res1) & set(res2)
        delta_rows = []
        for team_ in sorted(teams_both):
            r1 = res1[team_]
            r2 = res2[team_]
            delta_rows.append({
                t("col_team", lang):         team_,
                t("col_winner_a", lang):     f"{r1['winner']*100:.2f}%",
                t("col_winner_b", lang):     f"{r2['winner']*100:.2f}%",
                t("col_delta_winner", lang): f"{(r2['winner']-r1['winner'])*100:+.2f}%",
                t("col_final_a", lang):      f"{r1['final']*100:.1f}%",
                t("col_final_b", lang):      f"{r2['final']*100:.1f}%",
            })
        compare_df = pd.DataFrame(delta_rows).sort_values(t("col_team", lang))
        st.dataframe(compare_df, width="stretch")
