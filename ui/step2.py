"""Étape 2 — Calibration Dixon-Coles (MLE)."""
from __future__ import annotations

from datetime import date

import streamlit as st

import config
from src import stats as db
from src.i18n import t
from src.training import calibrate
from ui.common import load_teams_json


def render_step2(lang: str, s1_done: bool, s2_done: bool) -> None:
    with st.expander(t("step2_expander", lang), expanded=s1_done and not s2_done):
        if not s1_done:
            st.warning(t("warn_complete_step1", lang))
            return

        st.subheader(t("training_params_header", lang))
        col_a, col_b = st.columns(2)
        with col_a:
            train_start = st.date_input(t("train_start_label", lang), value=date.fromisoformat(config.TRAIN_START), key="ts")
            train_end   = st.date_input(t("train_end_label", lang),   value=date.fromisoformat(config.TRAIN_END),   key="te")
        with col_b:
            val_start = st.date_input(t("val_start_label", lang), value=date.fromisoformat(config.VAL_START), key="vs")
            val_end   = st.date_input(t("val_end_label", lang),   value=date.fromisoformat(config.VAL_END),   key="ve")

        st.subheader(t("match_weights_header", lang))
        col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
        w_friendly  = col_w1.slider(t("w_friendly", lang),  0.1, 1.0, config.MATCH_WEIGHTS["FRIENDLY"],  0.05)
        w_qualifier = col_w2.slider(t("w_qualifier", lang), 0.1, 2.0, config.MATCH_WEIGHTS["QUALIFIER"],  0.1)
        w_group     = col_w3.slider(t("w_group", lang),     0.5, 2.5, config.MATCH_WEIGHTS["GROUP"],      0.1)
        w_knockout  = col_w4.slider(t("w_knockout", lang),  0.5, 3.0, config.MATCH_WEIGHTS["KNOCKOUT"],   0.1)
        w_wc        = col_w5.slider(t("w_wc", lang),        0.5, 4.0, config.MATCH_WEIGHTS["WORLD_CUP"],  0.1)
        lambda_decay = st.slider(t("lambda_label", lang), 0.05, 1.0, config.LAMBDA_DECAY, 0.05)
        mu_val       = st.slider(t("mu_label", lang),     1.5,  4.0, config.MU,           0.05)

        custom_weights = {
            "FRIENDLY": w_friendly, "QUALIFIER": w_qualifier,
            "GROUP": w_group, "KNOCKOUT": w_knockout, "WORLD_CUP": w_wc,
        }

        if st.button(t("btn_calibrate", lang)):
            teams_json = load_teams_json()
            teams_list = list(teams_json["teams"].keys())
            matches    = st.session_state.matches
            iteration_placeholder = st.empty()
            iter_count = [0]

            def iter_cb(n):
                iter_count[0] = n
                iteration_placeholder.text(t("iter_mle", lang, n=n))

            with st.spinner(t("spinner_calibrate", lang)):
                try:
                    result = calibrate(
                        matches=matches,
                        teams=teams_list,
                        initial_coeffs=st.session_state.coefficients,
                        train_start=train_start.isoformat(),
                        train_end=train_end.isoformat(),
                        val_start=val_start.isoformat(),
                        val_end=val_end.isoformat(),
                        custom_weights=custom_weights,
                        lambda_decay=lambda_decay,
                        mu=mu_val,
                        callback=iter_cb,
                    )
                    st.session_state.training_result = result

                    calibrated = {}
                    for team_ in teams_list:
                        calibrated[team_] = {
                            "att_rating": result["att"].get(team_, st.session_state.coefficients.get(team_, {}).get("att_rating", 1.0)),
                            "def_rating": result["def"].get(team_, st.session_state.coefficients.get(team_, {}).get("def_rating", 1.0)),
                            "n_matches":  st.session_state.coefficients.get(team_, {}).get("n_matches", 0),
                        }
                    st.session_state.calibrated_coeffs = calibrated

                    db.save_training_run(result, train_start.isoformat(), train_end.isoformat(),
                                         val_start.isoformat(), val_end.isoformat())
                    # Invalider les caches qui dépendent des coefficients calibrés
                    db.load_training_runs.clear()
                    db.curiosities.clear()
                    db.get_group_predictions.clear()
                    db.get_probable_bracket.clear()
                    db.get_upsets.clear()
                    st.success(t("calibrate_success", lang,
                                 n=result.get("iterations", "?"), status=result["message"]))
                except Exception as e:
                    st.error(t("error_calibrate", lang, msg=e))
            iteration_placeholder.empty()

        if st.session_state.training_result:
            import pandas as pd
            result = st.session_state.training_result
            mb = result.get("metrics_before", {})
            ma = result.get("metrics_after", {})

            lc = result.get("loss_config", {})
            if lc:
                st.caption(t("loss_config_caption", lang,
                             alpha=lc.get("alpha", "?"),
                             beta=lc.get("beta", "?"),
                             l2=lc.get("l2_lambda", "?")))

            st.subheader(t("metrics_header", lang))
            met_df = pd.DataFrame({
                t("col_metric", lang): ["Accuracy W/D/L", "MAE buts", "Brier score"],
                t("col_before", lang): [
                    f"{mb.get('accuracy', 'N/A'):.3f}" if mb.get("accuracy") is not None else "N/A",
                    f"{mb.get('mae', 'N/A'):.3f}"      if mb.get("mae")      is not None else "N/A",
                    f"{mb.get('brier', 'N/A'):.3f}"    if mb.get("brier")    is not None else "N/A",
                ],
                t("col_after", lang): [
                    f"{ma.get('accuracy', 'N/A'):.3f}" if ma.get("accuracy") is not None else "N/A",
                    f"{ma.get('mae', 'N/A'):.3f}"      if ma.get("mae")      is not None else "N/A",
                    f"{ma.get('brier', 'N/A'):.3f}"    if ma.get("brier")    is not None else "N/A",
                ],
                t("col_n_val", lang): [mb.get("n", 0), mb.get("n", 0), mb.get("n", 0)],
            })
            st.table(met_df)

            if st.session_state.coefficients and st.session_state.calibrated_coeffs:
                st.subheader(t("ranking_header", lang))
                st.caption(t("ranking_caption", lang))

                def _strength(c: dict) -> float:
                    return c["att_rating"] / max(c["def_rating"], 0.01)

                init_c = st.session_state.coefficients
                cal_c  = st.session_state.calibrated_coeffs
                ranked_before = sorted(init_c, key=lambda team_: -_strength(init_c[team_]))[:10]
                ranked_after  = sorted(cal_c,  key=lambda team_: -_strength(cal_c[team_]))[:10]
                rank_df = pd.DataFrame({
                    t("col_rank", lang):         list(range(1, 11)),
                    t("col_before_init", lang):  [f"{team_}  {_strength(init_c[team_]):.2f}" for team_ in ranked_before],
                    t("col_after_calib", lang):  [f"{team_}  {_strength(cal_c[team_]):.2f}"  for team_ in ranked_after],
                })
                st.table(rank_df)

                delta_rows = []
                for team_ in sorted(cal_c.keys()):
                    d_att = cal_c[team_]["att_rating"] - init_c.get(team_, {}).get("att_rating", 1.0)
                    d_def = cal_c[team_]["def_rating"] - init_c.get(team_, {}).get("def_rating", 1.0)
                    delta_rows.append({
                        t("col_team", lang):      team_,
                        t("col_att_calib", lang): round(cal_c[team_]["att_rating"], 3),
                        t("col_delta_att", lang): round(d_att, 3),
                        t("col_def_calib", lang): round(cal_c[team_]["def_rating"], 3),
                        t("col_delta_def", lang): round(d_def, 3),
                    })
                delta_df = pd.DataFrame(delta_rows)
                _abs_col = "|Δ|"
                delta_df[_abs_col] = (delta_df[t("col_delta_att", lang)].abs()
                                      + delta_df[t("col_delta_def", lang)].abs())
                delta_df = delta_df.sort_values(_abs_col, ascending=False).drop(columns=_abs_col)
                delta_df.index = range(1, len(delta_df) + 1)
                st.subheader(t("delta_header", lang))
                st.dataframe(
                    delta_df.style.background_gradient(
                        subset=[t("col_delta_att", lang), t("col_delta_def", lang)],
                        cmap="RdYlGn",
                    ),
                    width="stretch", height=400,
                )

            if st.button(t("btn_validate_step2", lang)):
                st.session_state.step2_complete = True
                if not st.session_state.calibrated_coeffs:
                    st.session_state.calibrated_coeffs = st.session_state.coefficients.copy()
                st.rerun()

        if not st.session_state.training_result:
            if st.button(t("btn_skip_step2", lang)):
                st.session_state.step2_complete = True
                st.session_state.calibrated_coeffs = st.session_state.coefficients.copy()
                st.rerun()
