"""Tab 5 — Prédictions: classements groupes, bracket visuel, upsets."""
from __future__ import annotations

import streamlit as st

import config
from src import stats as db
from src.i18n import t
from ui.common import R32_LABELS, R16_LABELS, _load_ko_sched, load_teams_json


def render_tab_predictions(
    lang: str,
    sim: dict,
    probs: dict,
    n: int,
    sorted_teams: list,
    active_coeffs: dict,
) -> None:
    import pandas as pd
    # Header caption
    _runs_pred = db.load_simulation_runs()
    if _runs_pred:
        _latest_pred = _runs_pred[0]
        _dt_str_pred = _latest_pred["date"][:16].replace("T", " ")
        st.caption(t("pred_based_on", lang, n=f"{n:,}", date=_dt_str_pred))

    # ---- SECTION 1: Group predictions ----
    st.subheader(t("pred_groups_header", lang))
    _tj_pred       = load_teams_json()
    _groups_pred   = _tj_pred["groups"]
    _gp_data       = db.get_group_predictions(sim, _groups_pred)

    _grp_list = list(_groups_pred.keys())
    for _row_start in range(0, len(_grp_list), 4):
        _row_grps_pred = _grp_list[_row_start:_row_start + 4]
        _pred_cols = st.columns(len(_row_grps_pred))
        for _ci_p, _grp_p in enumerate(_row_grps_pred):
            with _pred_cols[_ci_p]:
                _rows_p = _gp_data.get(_grp_p, [])
                if _rows_p:
                    _gap_p = (_rows_p[0]["pos1"] - _rows_p[1]["pos1"]) * 100 if len(_rows_p) > 1 else 0.0
                    _badge_p = t("pred_death_badge", lang) if _gap_p < 10 else t("pred_open_badge", lang)
                    st.markdown(f"**{t('pred_group_header_detail', lang, grp=_grp_p)}** {_badge_p}")
                    st.caption(t("pred_gap_label", lang, gap=f"{_gap_p:.1f}"))
                else:
                    st.markdown(f"**{t('pred_group_header_detail', lang, grp=_grp_p)}**")

                _pred_df_rows = []
                for _r_p in _rows_p:
                    _pred_df_rows.append({
                        t("col_team", lang):         _r_p["team"],
                        t("pred_col_pos1", lang):    f"{_r_p['pos1']*100:.1f}",
                        t("pred_col_pos2", lang):    f"{_r_p['pos2']*100:.1f}",
                        t("pred_col_pos3", lang):    f"{_r_p['pos3']*100:.1f}",
                        t("pred_col_elim", lang):    f"{_r_p['pos4']*100:.1f}",
                        t("pred_col_avg_pts", lang): f"{_r_p['avg_pts']:.1f}",
                        t("pred_col_avg_gf", lang):  f"{_r_p['avg_gf']:.1f}",
                        t("pred_col_avg_ga", lang):  f"{_r_p['avg_ga']:.1f}",
                    })
                if _pred_df_rows:
                    st.dataframe(pd.DataFrame(_pred_df_rows), hide_index=True, height=185)

    # ---- SECTION 2: Bracket ----
    st.subheader(t("pred_bracket_header", lang))
    _ko_sched_pred = _load_ko_sched()
    _bracket = db.get_probable_bracket(
        sim, active_coeffs, sim.get("mu", config.MU),
        R32_LABELS, R16_LABELS, _ko_sched_pred,
    )

    # R32 — compact table
    st.markdown(f"**{t('pred_round_r32', lang)}**")
    _r32_rows_pred = []
    for _m32 in _bracket["r32"]:
        _th32  = _m32["home_candidates"][0][0] if _m32["home_candidates"] else "?"
        _ta32  = _m32["away_candidates"][0][0] if _m32["away_candidates"] else "?"
        _tw32  = _m32["winner_candidates"][0][0] if _m32["winner_candidates"] else "?"
        _ps32  = _m32.get("prob_score")
        _score32 = f"{_ps32['score_h']}-{_ps32['score_a']}" if _ps32 else "-"
        _r32_rows_pred.append({
            "Match":      _m32["label"],
            "Domicile":   f"{_th32} ({_m32['home_candidates'][0][1]*100:.0f}%)" if _m32["home_candidates"] else "?",
            "Extérieur":  f"{_ta32} ({_m32['away_candidates'][0][1]*100:.0f}%)" if _m32["away_candidates"] else "?",
            t("pred_score_label", lang, h="?", a="?").split(":")[0]: _score32,
            "Vainqueur":  f"{_tw32} ({_m32['winner_candidates'][0][1]*100:.0f}%)" if _m32["winner_candidates"] else "?",
        })
    st.dataframe(pd.DataFrame(_r32_rows_pred), hide_index=True, width="stretch")

    # R16, QF, SF — cards
    for _stage_key, _stage_label, _matches_pred in [
        ("r16", t("pred_round_r16", lang), _bracket["r16"]),
        ("qf",  t("pred_round_qf",  lang), _bracket["qf"]),
        ("sf",  t("pred_round_sf",  lang), _bracket["sf"]),
    ]:
        st.markdown(f"**{_stage_label}**")
        _n_matches = len(_matches_pred)
        _card_cols = st.columns(min(_n_matches, 4))
        for _mi_p, _m_p in enumerate(_matches_pred):
            with _card_cols[_mi_p % len(_card_cols)]:
                with st.container(border=True):
                    st.caption(_m_p["label"])
                    _th_p   = _m_p["home_candidates"][0][0] if _m_p["home_candidates"] else "?"
                    _ta_p   = _m_p["away_candidates"][0][0] if _m_p["away_candidates"] else "?"
                    _th_pct = _m_p["home_candidates"][0][1] * 100 if _m_p["home_candidates"] else 0
                    _ta_pct = _m_p["away_candidates"][0][1] * 100 if _m_p["away_candidates"] else 0
                    st.markdown(f"**{_th_p}** ({_th_pct:.0f}%) vs **{_ta_p}** ({_ta_pct:.0f}%)")
                    _ps_p = _m_p.get("prob_score")
                    if _ps_p:
                        st.caption(t("pred_score_label", lang, h=_ps_p["score_h"], a=_ps_p["score_a"]))
                        if _ps_p["p_draw"] > 0.20:
                            st.caption(t("pred_draw_pct", lang, pct=f"{_ps_p['p_draw']*100:.0f}"))
                    if _m_p["winner_candidates"]:
                        _tw_p    = _m_p["winner_candidates"][0][0]
                        _tw_pct_p = _m_p["winner_candidates"][0][1] * 100
                        st.caption(f"→ **{_tw_p}** ({_tw_pct_p:.0f}%)")

    # Final card
    st.markdown(f"**{t('pred_round_final', lang)}**")
    _fin_m = _bracket["final"]
    with st.container(border=True):
        st.caption(_fin_m["label"])
        _th_fin     = _fin_m["home_candidates"][0][0] if _fin_m["home_candidates"] else "?"
        _ta_fin     = _fin_m["away_candidates"][0][0] if _fin_m["away_candidates"] else "?"
        _th_fin_pct = _fin_m["home_candidates"][0][1] * 100 if _fin_m["home_candidates"] else 0
        _ta_fin_pct = _fin_m["away_candidates"][0][1] * 100 if _fin_m["away_candidates"] else 0
        st.markdown(f"**{_th_fin}** ({_th_fin_pct:.0f}%) vs **{_ta_fin}** ({_ta_fin_pct:.0f}%)")
        _ps_fin = _fin_m.get("prob_score")
        if _ps_fin:
            st.caption(t("pred_score_label", lang, h=_ps_fin["score_h"], a=_ps_fin["score_a"]))
            if _ps_fin["p_draw"] > 0.20:
                st.caption(t("pred_draw_pct", lang, pct=f"{_ps_fin['p_draw']*100:.0f}"))

    # Champion path narrative
    _champ_pred = sorted_teams[0][0]
    _path_pred  = []
    for _stage_name, _opp_key in [
        ("R32", "opp_r32"), ("R16", "opp_r16"),
        ("QF", "opp_qf"), ("SF", "opp_sf"), ("Finale", "opp_final"),
    ]:
        _opp_dict_pred = probs.get(_champ_pred, {}).get(_opp_key, {})
        if _opp_dict_pred:
            _top_opp_pred = max(_opp_dict_pred, key=_opp_dict_pred.get)
            _path_pred.append(
                t("pred_champion_path", lang, team=_champ_pred, opp=_top_opp_pred, stage=_stage_name)
            )
    if _path_pred:
        st.markdown(f"**{t('pred_champion_path_header', lang)}:** " + " → ".join(_path_pred))
        _path_prob_pred = probs.get(_champ_pred, {}).get("winner", 0.0)
        st.caption(t("pred_path_global_prob", lang, pct=f"{_path_prob_pred*100:.1f}"))

    # ---- SECTION 3: Upsets ----
    st.subheader(t("pred_upsets_header", lang))
    _upsets = db.get_upsets(sim)

    st.markdown(f"**{t('pred_upset_r32_title', lang)}**")
    if _upsets["upset_r32"]:
        st.table(pd.DataFrame([{
            t("col_team", lang):          u["team"],
            t("pred_col_pos1", lang):     f"{u['r32_pct']*100:.1f}",
            t("pred_col_elim_r32", lang): f"{u['elim_r32_pct']*100:.1f}",
        } for u in _upsets["upset_r32"]]))

    st.markdown(f"**{t('pred_upset_r16_title', lang)}**")
    if _upsets["upset_r16"]:
        st.table(pd.DataFrame([{
            t("col_team", lang):          u["team"],
            t("pred_col_pos2", lang):     f"{u['r16_pct']*100:.1f}",
            t("pred_col_elim_r16", lang): f"{u['elim_r16_pct']*100:.1f}",
        } for u in _upsets["upset_r16"]]))

    st.markdown(f"**{t('pred_outsider_qf_title', lang)}**")
    if _upsets["outsider_qf"]:
        st.table(pd.DataFrame([{
            t("col_team", lang):            u["team"],
            t("pred_col_qf_reached", lang): f"{u['qf_pct']*100:.1f}",
            t("col_winner_pct2", lang):     f"{u['winner_pct']*100:.2f}",
        } for u in _upsets["outsider_qf"]]))

    if _upsets["biggest_surprise"]:
        _bs = _upsets["biggest_surprise"]
        st.info(f"**{t('pred_biggest_surprise', lang)}**: {_bs['team']} — "
                f"QF: {_bs['qf_pct']*100:.1f}% | "
                f"Winner: {_bs['winner_pct']*100:.2f}%")
