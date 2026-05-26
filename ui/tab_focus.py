"""Tab 2 — Focus équipe: parcours, opponents, exit dist, group probs, H2H."""
from __future__ import annotations

import streamlit as st

from src.h2h import h2h_vs_team
from src.i18n import t
from src.poisson import outcome_probs
from ui.common import _save_prefs, load_teams_json


def render_tab_focus(
    lang: str,
    probs: dict,
    n: int,
    active_coeffs: dict,
    mu_sim: float,
) -> None:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    _focus_team   = st.session_state.get("focus_team", "France")
    _sorted_keys  = sorted(probs.keys())
    _default_idx  = (_sorted_keys.index(_focus_team) if _focus_team in _sorted_keys else 0)
    focus_team = st.selectbox(
        t("team_analysis_label", lang),
        options=_sorted_keys,
        index=_default_idx,
    )
    if focus_team != _focus_team:
        st.session_state["focus_team"] = focus_team
        _save_prefs({"focus_team": focus_team})
    d = probs[focus_team]

    st.subheader(t("team_journey_header", lang, team=focus_team))
    round_cols = st.columns(6)
    rounds = [
        (t("round_r32", lang), "r32"),
        (t("round_r16", lang), "r16"),
        (t("round_qf", lang),  "qf"),
        (t("round_sf", lang),  "sf"),
        (t("round_final", lang), "final"),
        (t("round_winner", lang), "winner"),
    ]
    for col, (label, key) in zip(round_cols, rounds):
        col.metric(label, f"{d[key]*100:.1f}%")

    for stage_key, stage_label in [
        ("opp_qf",    t("round_qf", lang)),
        ("opp_sf",    t("round_sf", lang)),
        ("opp_final", t("round_final", lang)),
    ]:
        opp = d.get(stage_key, {})
        if opp:
            st.subheader(t("opp_title", lang, stage=stage_label))
            opp_sorted = sorted(opp.items(), key=lambda x: -x[1])[:5]
            opp_df = pd.DataFrame([
                {
                    t("col_opponent", lang): k,
                    t("col_opp_pct", lang, stage=stage_label): f"{v/n*100:.1f}",
                }
                for k, v in opp_sorted
            ])
            st.table(opp_df)

    _CONF_COLORS = {
        "UEFA": "#3498db", "CONMEBOL": "#2ecc71", "CAF": "#e74c3c",
        "AFC": "#e67e22", "CONCACAF": "#9b59b6", "OFC": "#95a5a6",
    }
    _tj_conf = load_teams_json()
    _conf    = _tj_conf["teams"].get(focus_team, {}).get("confederation", "UEFA")
    _col_hex = _CONF_COLORS.get(_conf, "#7f8c8d")
    _rv, _gv, _bv = int(_col_hex[1:3], 16), int(_col_hex[3:5], 16), int(_col_hex[5:7], 16)

    _rl = [
        t("exit_group_label", lang),
        t("round_r32", lang),
        t("round_r16", lang),
        t("round_qf", lang),
        t("round_sf", lang),
        t("exit_finalist_label", lang),
        t("exit_champion_label", lang),
    ]
    _vals = [
        100.0,
        d["r32"]    * 100,
        d["r16"]    * 100,
        d["qf"]     * 100,
        d["sf"]     * 100,
        d["final"]  * 100,
        d["winner"] * 100,
    ]

    fig_focus_dist = go.Figure()
    fig_focus_dist.add_trace(go.Scatter(
        x=_rl, y=_vals,
        name=focus_team,
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in _vals],
        textposition="top center",
        line=dict(shape="spline", color=_col_hex, width=3),
        fill="tozeroy",
        fillcolor=f"rgba({_rv},{_gv},{_bv},0.15)",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig_focus_dist.update_layout(
        title=t("dist_focus_title", lang, team=focus_team),
        xaxis_title=t("dist_round_label", lang),
        yaxis_title=t("dist_pct_label", lang),
        yaxis=dict(range=[0, 115]),
        height=440,
        showlegend=False,
        margin=dict(t=70, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_focus_dist)

    teams_json = load_teams_json()
    team_group = teams_json["teams"].get(focus_team, {}).get("group")
    if team_group:
        group_members = teams_json["groups"].get(team_group, [])
        opponents = [team_ for team_ in group_members if team_ != focus_team]
        if opponents and active_coeffs:
            st.subheader(t("group_probs_title", lang, grp=team_group))
            fh = active_coeffs.get(focus_team, {"att_rating": 1.0, "def_rating": 1.0})
            h2h_tbl = st.session_state.h2h_table or None
            match_rows = []
            for opp in opponents:
                fa = active_coeffs.get(opp, {"att_rating": 1.0, "def_rating": 1.0})
                pw, pd_prob, pl = outcome_probs(
                    fh["att_rating"], fh["def_rating"],
                    fa["att_rating"], fa["def_rating"],
                    mu=mu_sim,
                    home_team=focus_team,
                    away_team=opp,
                    h2h_table=h2h_tbl,
                )
                match_rows.append({
                    t("col_opponent", lang):                  opp,
                    t("col_pct_win_team", lang, team=focus_team): f"{pw*100:.1f}",
                    t("col_pct_draw", lang):                  f"{pd_prob*100:.1f}",
                    t("col_pct_win_opp", lang, opp=opp):     f"{pl*100:.1f}",
                })
            st.table(pd.DataFrame(match_rows))

    h2h_tbl = st.session_state.h2h_table or {}
    if h2h_tbl:
        h2h_rows = h2h_vs_team(st.session_state.matches, focus_team, h2h_tbl)
        wc_teams = set(probs.keys())
        h2h_rows = [r for r in h2h_rows if r["opponent"] in wc_teams]
        if h2h_rows:
            favourable = [r for r in h2h_rows if r["h2h_score"] > 0][:5]
            difficult  = sorted(
                [r for r in h2h_rows if r["h2h_score"] < 0],
                key=lambda x: x["h2h_score"],
            )[:5]
            col_fav, col_dif = st.columns(2)
            with col_fav:
                st.subheader(t("h2h_fav_title", lang, team=focus_team))
                if favourable:
                    st.dataframe(pd.DataFrame([{
                        t("col_opponent", lang):    r["opponent"],
                        t("col_h2h_score", lang):   r["h2h_score"],
                        t("col_vnd", lang):         f"{r['wins']}/{r['draws']}/{r['losses']}",
                        t("col_avg_scored", lang):  r["avg_scored"],
                        t("col_avg_conceded", lang):r["avg_conceded"],
                    } for r in favourable]), width="stretch")
                else:
                    st.info(t("no_h2h_advantage", lang))
            with col_dif:
                st.subheader(t("h2h_dif_title", lang, team=focus_team))
                if difficult:
                    st.dataframe(pd.DataFrame([{
                        t("col_opponent", lang):    r["opponent"],
                        t("col_h2h_score", lang):   r["h2h_score"],
                        t("col_vnd", lang):         f"{r['wins']}/{r['draws']}/{r['losses']}",
                        t("col_avg_scored", lang):  r["avg_scored"],
                        t("col_avg_conceded", lang):r["avg_conceded"],
                    } for r in difficult]), width="stretch")
                else:
                    st.info(t("no_h2h_disadvantage", lang))
