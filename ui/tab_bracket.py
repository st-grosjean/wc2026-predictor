"""Expander — Tableau KO: saisie des scores, bracket R32→Finale, maj API."""
from __future__ import annotations

import streamlit as st

import config
from src.fetcher import fetch_live_wc_scores
from src.i18n import t
from src.tournament import R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS
from ui.common import (
    R32_LABELS, R16_LABELS,
    _build_slot_map, _init_ko, _ko_is_placeholder,
    _ko_match_input, _ko_team_winner,
    _load_gr, _load_ko, _load_ko_sched, _save_ko,
    load_teams_json,
)


def render_tab_bracket(lang: str) -> None:
    with st.expander(t("ko_expander", lang), expanded=False):
        _tj_ko  = load_teams_json()
        _grp_ko = _tj_ko["groups"]
        _sm     = _build_slot_map(_load_gr(), _grp_ko)

        def _slot(s: str) -> str:
            if s in _sm:
                return _sm[s]
            if s[:1] == "1":
                return f"1er Gr.{s[1:]}"
            if s[:1] == "2":
                return f"2e Gr.{s[1:]}"
            if s.startswith("3W_"):
                return f"3e->Gr.{s[3:]}"
            return s

        _ko     = _load_ko()
        _ko_sch = _load_ko_sched()
        if not _ko or "r32" not in _ko:
            _ko = _init_ko()
            _save_ko(_ko)
        _ko_ch = False

        _t32 = [(_slot(h), _slot(a)) for h, a in R32_BRACKET]
        _w32 = [_ko_team_winner(_ko["r32"][i], *_t32[i]) for i in range(16)]

        _t16 = [((_w32[i] or f"V.R32-{i+1}"), (_w32[j] or f"V.R32-{j+1}")) for i, j in R16_PAIRS]
        _w16 = [_ko_team_winner(_ko["r16"][i], *_t16[i]) for i in range(8)]

        _tqf = [((_w16[i] or f"V.R16-{i+1}"), (_w16[j] or f"V.R16-{j+1}")) for i, j in QF_PAIRS]
        _wqf = [_ko_team_winner(_ko["qf"][i], *_tqf[i]) for i in range(4)]

        _tsf = [((_wqf[i] or f"V.QF-{i+1}"), (_wqf[j] or f"V.QF-{j+1}")) for i, j in SF_PAIRS]
        _wsf = [_ko_team_winner(_ko["sf"][i], *_tsf[i]) for i in range(2)]
        _lsf = [
            (_tsf[i][1] if _wsf[i] == _tsf[i][0] else _tsf[i][0]) if _wsf[i] else None
            for i in range(2)
        ]

        _tfin = (_wsf[0] or "V.SF-1",      _wsf[1] or "V.SF-2")
        _t3rd = (_lsf[0] or "Perd.SF-1",  _lsf[1] or "Perd.SF-2")

        # Global update button
        _upd_col1, _upd_col2 = st.columns([3, 1])
        with _upd_col1:
            st.caption(t("ko_instructions", lang))
        with _upd_col2:
            if st.button(t("btn_update_all", lang), key="ko_update_all"):
                if not config.API_KEY:
                    st.warning(t("warn_api_key", lang))
                else:
                    with st.spinner(t("live_fetching", lang)):
                        try:
                            _live = fetch_live_wc_scores()
                            _api_lkp: dict[tuple, tuple] = {
                                (_m["home_team"], _m["away_team"]): (
                                    _m["home_goals"], _m["away_goals"],
                                    _m.get("penalties_home"), _m.get("penalties_away"),
                                )
                                for _m in _live
                                if _m["status"] == "FINISHED" and _m["home_goals"] is not None
                            }
                            _in_play_ko = [
                                _m for _m in _live
                                if _m["status"] in ("IN_PLAY", "PAUSED")
                                and _m["home_goals"] is not None
                            ]

                            def _apply_score(stg: str, ki, th: str, ta: str) -> bool:
                                _sc = _api_lkp.get((th, ta)) or _api_lkp.get((ta, th))
                                if not _sc:
                                    return False
                                _entry = _ko[stg] if ki is None else _ko[stg][ki]
                                if _entry.get("played"):
                                    return False
                                _hg_new, _ag_new, _ph_new, _pa_new = _sc
                                if _api_lkp.get((ta, th)) and not _api_lkp.get((th, ta)):
                                    _hg_new, _ag_new = _ag_new, _hg_new
                                    _ph_new, _pa_new = _pa_new, _ph_new
                                _upd: dict = {"goals_h": _hg_new, "goals_a": _ag_new, "played": True}
                                if (_hg_new == _ag_new
                                        and _ph_new is not None and _pa_new is not None):
                                    _upd["penalties"] = True
                                    _upd["penalties_h"] = _ph_new
                                    _upd["penalties_a"] = _pa_new
                                _entry.update(_upd)
                                return True

                            _updated = 0
                            for _stg, _teams_list in [
                                ("r32", _t32), ("r16", _t16), ("qf", _tqf), ("sf", _tsf),
                            ]:
                                for _ki, (_th, _ta) in enumerate(_teams_list):
                                    if _apply_score(_stg, _ki, _th, _ta):
                                        _updated += 1
                            for _stg, (_th, _ta) in [("final", _tfin), ("third_place", _t3rd)]:
                                if _apply_score(_stg, None, _th, _ta):
                                    _updated += 1

                            st.session_state["ko_live_in_play"] = _in_play_ko
                            _save_ko(_ko)
                            st.success(t("api_updated", lang, n=_updated))
                            if _updated:
                                _ko_ch = True
                                st.rerun()
                        except Exception as _e:
                            st.error(t("error_fetch_ko", lang, msg=_e))

        for _lm in st.session_state.get("ko_live_in_play", []):
            st.info(t("live_in_play", lang,
                      home=_lm["home_team"], hg=_lm["home_goals"],
                      ag=_lm["away_goals"], away=_lm["away_team"]))

        st.subheader(t("ko_r32_header", lang))
        _cols32 = st.columns(2)
        for _i32 in range(16):
            with _cols32[_i32 % 2]:
                if _ko_match_input(
                    "r32", _i32, _t32[_i32][0], _t32[_i32][1],
                    R32_LABELS[_i32], _ko, _ko_sch.get(("r32", _i32)), lang=lang,
                ):
                    _ko_ch = True

        st.subheader(t("ko_r16_header", lang))
        _cols16 = st.columns(2)
        for _i16 in range(8):
            with _cols16[_i16 % 2]:
                if _ko_match_input(
                    "r16", _i16, _t16[_i16][0], _t16[_i16][1],
                    R16_LABELS[_i16], _ko, _ko_sch.get(("r16", _i16)), lang=lang,
                ):
                    _ko_ch = True

        st.subheader(t("ko_qf_header", lang))
        _colsqf = st.columns(2)
        for _iqf in range(4):
            with _colsqf[_iqf % 2]:
                if _ko_match_input(
                    "qf", _iqf, _tqf[_iqf][0], _tqf[_iqf][1],
                    f"M{97 + _iqf}", _ko, _ko_sch.get(("qf", _iqf)), lang=lang,
                ):
                    _ko_ch = True

        st.subheader(t("ko_sf_header", lang))
        _colssf = st.columns(2)
        for _isf in range(2):
            with _colssf[_isf]:
                if _ko_match_input(
                    "sf", _isf, _tsf[_isf][0], _tsf[_isf][1],
                    f"M{101 + _isf}", _ko, _ko_sch.get(("sf", _isf)), lang=lang,
                ):
                    _ko_ch = True

        st.subheader(t("ko_final_header", lang))
        _colsfin = st.columns(2)
        with _colsfin[0]:
            st.markdown(f"**{t('ko_final_label', lang)}**")
            if _ko_match_input(
                "final", None, _tfin[0], _tfin[1],
                "M104", _ko, _ko_sch.get(("final", None)), lang=lang,
            ):
                _ko_ch = True
        with _colsfin[1]:
            st.markdown(f"**{t('ko_third_label', lang)}**")
            if _ko_match_input(
                "third_place", None, _t3rd[0], _t3rd[1],
                "M103", _ko, _ko_sch.get(("third_place", None)), lang=lang,
            ):
                _ko_ch = True

        if _ko_ch:
            _save_ko(_ko)
            st.toast(t("toast_score_saved", lang))
            st.rerun()
