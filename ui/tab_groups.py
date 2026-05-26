"""Expander — Phase de groupes: saisie des scores, classements, tableau des 3es."""
from __future__ import annotations

import streamlit as st

from src.i18n import t
from ui.common import (
    _group_standings, _init_gr, _load_gr, _load_schedule, _save_gr,
    load_teams_json,
)


def render_tab_groups(lang: str) -> None:
    import pandas as pd
    with st.expander(t("groups_expander", lang), expanded=False):
        _tj_gs    = load_teams_json()
        _groups_gs = _tj_gs["groups"]
        _gr_data  = _load_gr()
        if not _gr_data or set(_gr_data.keys()) != set(_groups_gs.keys()):
            _gr_data = _init_gr(_groups_gs)
            _save_gr(_gr_data)

        _sched_lkp = _load_schedule()
        _gr_changed = False

        for _rs in range(0, 12, 3):
            _row_grps = list("ABCDEFGHIJKL")[_rs:_rs + 3]
            _gcols = st.columns(3)
            for _ci, _grp in enumerate(_row_grps):
                with _gcols[_ci]:
                    _teams_g  = _groups_gs[_grp]
                    _grp_ms   = _gr_data[_grp]["matches"]
                    _ranked   = _group_standings(_teams_g, _grp_ms)
                    _n_played = sum(1 for _mm in _grp_ms if _mm.get("played"))
                    _complete = _n_played == 6

                    st.markdown(f"#### {t('group_header', lang, grp=_grp)}")

                    _srows = []
                    for _r in _ranked:
                        if _r["pos"] == 1:
                            _ico = "🏆"
                        elif _r["pos"] == 2:
                            _ico = "✅"
                        elif _r["pos"] == 3 and not _complete:
                            _ico = "🟡"
                        else:
                            _ico = "❌"
                        _srows.append({
                            "": _ico,
                            t("col_team", lang):     _r["team"][:16],
                            t("col_pts", lang):      _r["pts"],
                            t("col_played_j", lang): _r["j"],
                            t("col_wins_g", lang):   _r["g"],
                            t("col_draws_n", lang):  _r["n"],
                            t("col_losses_p", lang): _r["p"],
                            t("col_gf_bp", lang):    _r["bp"],
                            t("col_ga_bc", lang):    _r["bc"],
                            t("col_gd_pm", lang):    _r["diff"],
                        })
                    st.dataframe(pd.DataFrame(_srows), hide_index=True, height=185)

                    for _mi, _m in enumerate(_grp_ms):
                        _mk   = f"{_grp}_{_mi}"
                        _si   = _sched_lkp.get((_m["home"], _m["away"]), {})
                        _md_lbl = f"J{_si['matchday']} · " if _si.get("matchday") else ""
                        _dt_lbl = f"{_si.get('date', '')} {_si.get('time_utc', '')}Z" if _si.get("date") else ""
                        _vn_lbl = _si.get("venue", "").split(",")[0] if _si.get("venue") else ""
                        _full_lbl = f"{_md_lbl}{_dt_lbl}" + (f" · {_vn_lbl}" if _vn_lbl else "")
                        _mca, _mcb, _mcc, _mcd, _mce = st.columns([3.8, 1.1, 0.3, 1.1, 1.7])
                        if _full_lbl:
                            _mca.caption(_full_lbl)
                        _mca.caption(f"{_m['home']} — {_m['away']}")
                        _chg = int(_m["home_goals"]) if _m["home_goals"] is not None else 0
                        _cag = int(_m["away_goals"]) if _m["away_goals"] is not None else 0
                        _cpl = bool(_m.get("played", False))
                        _nhg = _mcb.number_input("h", min_value=0, max_value=99, value=_chg, step=1,
                                                 key=f"ghg_{_mk}", label_visibility="collapsed")
                        _mcc.markdown("<div style='text-align:center;padding-top:6px'>-</div>",
                                      unsafe_allow_html=True)
                        _nag = _mcd.number_input("a", min_value=0, max_value=99, value=_cag, step=1,
                                                 key=f"gag_{_mk}", label_visibility="collapsed")
                        _npl = _mce.checkbox("✓", value=_cpl, key=f"gpl_{_mk}")
                        if _nhg != _chg or _nag != _cag or _npl != _cpl:
                            _gr_data[_grp]["matches"][_mi].update(
                                {"home_goals": int(_nhg), "away_goals": int(_nag), "played": _npl}
                            )
                            _gr_changed = True

                    if st.button(t("btn_fetch_group", lang, grp=_grp), key=f"fetch_gr_{_grp}"):
                        st.info(t("fetch_not_impl", lang))

        if _gr_changed:
            _save_gr(_gr_data)
            st.toast(t("toast_score_saved", lang))
            st.rerun()

        st.divider()
        st.subheader(t("thirds_header", lang))
        _all3 = []
        for _g3, _t3 in _groups_gs.items():
            _rk3 = _group_standings(_t3, _gr_data[_g3]["matches"])
            if len(_rk3) >= 3:
                _r3 = _rk3[2]
                _all3.append({
                    t("col_gr", lang):   _g3,
                    t("col_team", lang): _r3["team"],
                    t("col_pts", lang):  _r3["pts"],
                    t("col_gd_pm", lang):_r3["diff"],
                    t("col_gf_bp", lang): _r3["bp"],
                    "_s": (-_r3["pts"], -_r3["diff"], -_r3["bp"]),
                })
        if _all3:
            _all3.sort(key=lambda x: x["_s"])
            _gr_k  = t("col_gr", lang)
            _tm_k  = t("col_team", lang)
            _pts_k = t("col_pts", lang)
            _gd_k  = t("col_gd_pm", lang)
            _gf_k  = t("col_gf_bp", lang)
            _t3rows = [
                {
                    t("col_rank", lang):   i + 1,
                    _gr_k:                 _row[_gr_k],
                    _tm_k:                 _row[_tm_k],
                    _pts_k:                _row[_pts_k],
                    _gd_k:                 _row[_gd_k],
                    _gf_k:                 _row[_gf_k],
                    t("col_status", lang): t("status_qualified", lang) if i < 8 else t("status_eliminated", lang),
                }
                for i, _row in enumerate(_all3)
            ]
            st.dataframe(pd.DataFrame(_t3rows), hide_index=True, width="stretch")
        else:
            st.info(t("no_scores_yet", lang))
