"""Étape 1 — Données & Coefficients initiaux."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

import config
from src.coefficients import compute_coefficients
from src.fetcher import fetch_all_matches, load_cached_matches, load_csv_into_cache
from src.h2h import build_h2h_table, h2h_pairs_summary
from src.i18n import t
from ui.common import _save_prefs, load_teams_json


def render_step1(lang: str, s1_done: bool) -> None:
    with st.expander(t("step1_expander", lang), expanded=not s1_done):
        st.subheader(t("api_config_header", lang))
        api_key_input = st.text_input(
            t("api_key_label", lang),
            value=config.API_KEY,
            type="password",
            help=t("api_key_help", lang),
        )
        if api_key_input:
            config.API_KEY = api_key_input

        date_from = st.date_input(
            t("date_from_label", lang),
            value=date(2022, 1, 1),
            min_value=date(2018, 1, 1),
            max_value=date.today(),
        )

        col_load, col_csv, col_cache = st.columns(3)

        with col_load:
            if st.button(t("btn_load_api", lang), disabled=not config.API_KEY):
                progress_bar = st.progress(0)
                status_text  = st.empty()

                def progress_cb(curr, total, msg):
                    if total > 0:
                        progress_bar.progress(min(curr / total, 1.0))
                    status_text.text(msg)

                with st.spinner(t("spinner_loading_api", lang)):
                    try:
                        matches = fetch_all_matches(
                            date_from=date_from.isoformat(),
                            progress_cb=progress_cb,
                        )
                        st.session_state.matches = matches
                        st.success(t("matches_loaded", lang, n=len(matches)))
                    except Exception as e:
                        st.error(t("error_api", lang, msg=e))
                progress_bar.empty()
                status_text.empty()

        with col_csv:
            csv_path = Path("data") / "results.csv"
            csv_disabled = not csv_path.exists()
            if st.button(t("btn_load_csv", lang), disabled=csv_disabled,
                         help=t("csv_help", lang)):
                progress_bar_csv = st.progress(0)
                status_csv = st.empty()

                def csv_progress_cb(curr, total, msg):
                    if total > 0:
                        progress_bar_csv.progress(min(curr / total, 1.0))
                    status_csv.text(msg)

                with st.spinner(t("spinner_loading_csv", lang)):
                    try:
                        matches = load_csv_into_cache(
                            date_from=date_from.isoformat(),
                            progress_cb=csv_progress_cb,
                        )
                        st.session_state.matches = matches
                        load_cached_matches.clear()
                        compute_coefficients.clear()
                        st.success(t("matches_loaded_csv", lang, n=len(matches)))
                    except Exception as e:
                        st.error(t("error_csv", lang, msg=e))
                progress_bar_csv.empty()
                status_csv.empty()
            if csv_disabled:
                st.caption(t("csv_disabled_hint", lang))

        with col_cache:
            if st.button(t("btn_load_cache", lang)):
                matches = load_cached_matches()
                st.session_state.matches = matches
                if matches:
                    st.success(t("matches_from_cache", lang, n=len(matches)))
                else:
                    st.info(t("cache_empty", lang))

        if st.button(t("btn_compute_coeffs", lang)):
            teams_json = load_teams_json()
            teams_list = list(teams_json["teams"].keys())
            matches    = st.session_state.matches
            with st.spinner(t("spinner_coeffs", lang)):
                coeffs = compute_coefficients(matches, teams_list)
            st.session_state.coefficients = coeffs

            by_team: dict[str, int] = {}
            for m in matches:
                for team_ in (m.get("home_team", ""), m.get("away_team", "")):
                    if team_ in coeffs:
                        by_team[team_] = by_team.get(team_, 0) + 1
            st.session_state.n_matches_by_team = by_team

            with st.spinner(t("spinner_h2h", lang)):
                h2h_tbl = build_h2h_table(matches)
                h2h_top = h2h_pairs_summary(matches, set(teams_list), h2h_tbl)
            st.session_state.h2h_table = h2h_tbl
            st.session_state.h2h_top   = h2h_top

        if st.session_state.coefficients:
            import pandas as pd
            import plotly.express as px
            coeffs     = st.session_state.coefficients
            teams_json = load_teams_json()
            rows = []
            for team_, d in coeffs.items():
                t_data = teams_json["teams"].get(team_, {})
                rows.append({
                    t("col_team", lang):    team_,
                    t("col_group", lang):   t_data.get("group", "?"),
                    "ELO":                  t_data.get("elo", 1750),
                    t("col_att", lang):     round(d["att_rating"], 3),
                    t("col_def", lang):     round(d["def_rating"], 3),
                    t("col_matches", lang): st.session_state.n_matches_by_team.get(team_, 0),
                })
            df = pd.DataFrame(rows).sort_values("ELO", ascending=False).reset_index(drop=True)
            df.index = range(1, len(df) + 1)

            st.subheader(t("coeffs_table_title", lang))
            st.dataframe(df, width="stretch", height=400)

            att_col = t("col_att", lang)
            def_col = t("col_def", lang)
            fig = px.scatter(
                df,
                x=def_col, y=att_col,
                text=t("col_team", lang),
                color=t("col_group", lang),
                title=t("chart_att_def_title", lang),
                labels={
                    def_col: t("chart_def_label", lang),
                    att_col: t("chart_att_label", lang),
                },
            )
            fig.update_traces(textposition="top center", marker_size=8)
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig)

            if st.session_state.h2h_top:
                st.subheader(t("h2h_title", lang))
                h2h_df = pd.DataFrame([
                    {
                        t("col_team_a", lang):      r["team_a"],
                        t("col_team_b", lang):      r["team_b"],
                        t("col_wins_a", lang):      r["wins_a"],
                        t("col_draws", lang):       r["draws"],
                        t("col_wins_b", lang):      r["wins_b"],
                        t("col_gd_weighted", lang): r["weighted_gd"],
                        t("col_h2h_score", lang):   r["h2h_score"],
                        t("col_matches", lang):     r["n_matches"],
                    }
                    for r in st.session_state.h2h_top
                ])
                st.dataframe(h2h_df, width="stretch")

            st.subheader(t("focus_team_header", lang))
            _all_teams_sorted = sorted(load_teams_json()["teams"].keys())
            _cur_focus = st.session_state.get("focus_team", "France")
            _focus_idx = _all_teams_sorted.index(_cur_focus) if _cur_focus in _all_teams_sorted else 0
            _focus_sel = st.selectbox(
                t("focus_team_label", lang),
                options=_all_teams_sorted,
                index=_focus_idx,
                key="focus_team_select",
            )
            if _focus_sel != _cur_focus:
                st.session_state["focus_team"] = _focus_sel
                _save_prefs({"focus_team": _focus_sel})
                st.rerun()

            if st.button(t("btn_validate_step1", lang)):
                st.session_state.step1_complete = True
                st.rerun()
        else:
            st.info(t("info_no_coeffs", lang))
