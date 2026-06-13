"""Sidebar: language selector + quick stats."""
from __future__ import annotations

import streamlit as st

from src.i18n import t, SUPPORTED_LANGS, TZ_OPTIONS
from ui.common import DEFAULTS, _save_prefs


def render_sidebar(lang: str) -> None:
    with st.sidebar:
        _lang_opts = list(SUPPORTED_LANGS.keys())
        _cur_lang  = st.session_state.get("lang", "fr")
        _lang_idx  = _lang_opts.index(_cur_lang) if _cur_lang in _lang_opts else 0
        _sel_lang  = st.selectbox(
            t("lang_selector", _cur_lang),
            options=_lang_opts,
            format_func=lambda c: SUPPORTED_LANGS[c],
            index=_lang_idx,
            key="lang_select",
        )
        if _sel_lang != _cur_lang:
            st.session_state["lang"] = _sel_lang
            _save_prefs({"lang": _sel_lang})
            st.rerun()

        _tz_keys   = [tz for tz, _ in TZ_OPTIONS]
        _tz_labels = {tz: lbl for tz, lbl in TZ_OPTIONS}
        _cur_tz    = st.session_state.get("tz", "Europe/Paris")
        _tz_idx    = _tz_keys.index(_cur_tz) if _cur_tz in _tz_keys else 0
        _sel_tz    = st.selectbox(
            t("tz_selector", _cur_lang),
            options=_tz_keys,
            format_func=lambda k: _tz_labels[k],
            index=_tz_idx,
            key="tz_select",
        )
        if _sel_tz != _cur_tz:
            st.session_state["tz"] = _sel_tz
            _save_prefs({"tz": _sel_tz})
            st.rerun()

        st.divider()
        st.header(t("sidebar_header", lang))
        st.write(t("sidebar_cached_matches", lang, n=len(st.session_state.matches)))
        if st.session_state.calibrated_coeffs:
            st.write(t("sidebar_coeffs_calib", lang))
        elif st.session_state.coefficients:
            st.write(t("sidebar_coeffs_init", lang))
        else:
            st.write(t("sidebar_coeffs_default", lang))

        if st.session_state.sim_result:
            sr = st.session_state.sim_result
            st.write(t("sidebar_last_sim", lang, n=f"{sr['n_simulations']:,}"))
            st.write(t("sidebar_duration", lang, elapsed=f"{sr['elapsed_seconds']:.1f}"))

        st.divider()
        if st.button(t("btn_reset_session", lang)):
            for k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()

        st.caption(t("sidebar_stack", lang))
        st.caption(t("sidebar_data", lang))
