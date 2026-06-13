"""WC 2026 Predictor — entry point (orchestrateur)."""
from __future__ import annotations

import streamlit as st

from src import stats as db
from src.i18n import t, CJK_FONTS
from ui.common import DEFAULTS, _load_prefs, _save_prefs, _step_badge
from ui.sidebar import render_sidebar
from ui.step1 import render_step1
from ui.step2 import render_step2
from ui.step3 import render_step3
from ui.tab_bracket import render_tab_bracket
from ui.tab_groups import render_tab_groups
from ui.tab_methodology import render_tab_methodology
from ui.tournament_status import render_tournament_status

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

db.init_db()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Load persisted preferences (before any widget reads lang/focus_team)
# ---------------------------------------------------------------------------
_prefs = _load_prefs()
if "focus_team" in _prefs:
    st.session_state["focus_team"] = _prefs["focus_team"]
if "lang" in _prefs and st.session_state["lang"] == "fr":
    st.session_state["lang"] = _prefs["lang"]
if "tz" in _prefs and st.session_state["tz"] == "Europe/Paris":
    st.session_state["tz"] = _prefs["tz"]

lang = st.session_state.get("lang", "fr")

# CJK font injection (ja / ko)
if lang in CJK_FONTS:
    st.markdown(CJK_FONTS[lang], unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header & step status banner
# ---------------------------------------------------------------------------
st.title(t("app_title", lang))
st.caption(t("app_caption", lang))

s1_done = st.session_state.step1_complete
s2_done = st.session_state.step2_complete

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(_step_badge(t("step1_name", lang), s1_done, not s1_done))
with col_s2:
    st.markdown(_step_badge(t("step2_name", lang), s2_done, s1_done and not s2_done))
with col_s3:
    st.markdown(_step_badge(t("step3_name", lang), False, s2_done))

st.divider()

render_tournament_status(lang)

# ---------------------------------------------------------------------------
# Main pipeline (3 steps)
# ---------------------------------------------------------------------------
render_step1(lang, s1_done)
render_step2(lang, s1_done, s2_done)
render_step3(lang, s2_done)

st.divider()

# ---------------------------------------------------------------------------
# Standalone expanders (group stage + KO bracket)
# ---------------------------------------------------------------------------
render_tab_groups(lang)
render_tab_bracket(lang)

render_tab_methodology(lang)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
render_sidebar(lang)
