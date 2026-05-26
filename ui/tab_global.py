"""Tab 1 — Global MC results: bar chart, table, podium."""
from __future__ import annotations

import streamlit as st

from src.i18n import t


def render_tab_global(
    lang: str,
    probs: dict,
    n: int,
    sorted_teams: list,
) -> None:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    win_pct_label = t("col_win_pct", lang)
    team_label    = t("col_team", lang)
    bar_df = pd.DataFrame([
        {team_label: team_, win_pct_label: round(d["winner"] * 100, 2)}
        for team_, d in sorted_teams
    ])
    fig_bar = px.bar(
        bar_df,
        x=win_pct_label, y=team_label,
        orientation="h",
        title=t("chart_win_title", lang, n=f"{n:,}"),
        color=win_pct_label,
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=900)
    st.plotly_chart(fig_bar)

    table_rows = []
    for rank, (team_, d) in enumerate(sorted_teams, start=1):
        table_rows.append({
            t("col_rank", lang):      rank,
            t("col_team", lang):      team_,
            t("col_r32_pct", lang):   f"{d['r32']*100:.1f}",
            t("col_r16_pct", lang):   f"{d['r16']*100:.1f}",
            t("col_qf_pct", lang):    f"{d['qf']*100:.1f}",
            t("col_sf_pct", lang):    f"{d['sf']*100:.1f}",
            t("col_final_pct", lang): f"{d['final']*100:.1f}",
            t("col_winner_pct", lang):f"{d['winner']*100:.2f}",
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch")

    st.subheader(t("podium_title", lang))
    podium_cols = st.columns(4)
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (team_, data) in enumerate(sorted_teams[:4]):
        with podium_cols[i]:
            st.metric(
                label=f"{medals[i]} {team_}",
                value=f"{data['winner']*100:.2f}%",
                delta=t("podium_delta_label", lang, pct=f"{data['final']*100:.1f}"),
            )

    _col_labels = [
        t("exit_group_label", lang),
        t("round_r32", lang),
        t("round_r16", lang),
        t("round_qf", lang),
        t("round_sf", lang),
        t("exit_finalist_label", lang),
        t("exit_champion_label", lang),
    ]
    _podium4   = sorted_teams[:4]
    _y_labels  = [tm for tm, _ in _podium4]
    _z = [
        [100.0, td["r32"]*100, td["r16"]*100, td["qf"]*100,
         td["sf"]*100, td["final"]*100, td["winner"]*100]
        for _, td in _podium4
    ]

    fig_heat = go.Figure(go.Heatmap(
        z=_z,
        x=_col_labels,
        y=_y_labels,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        showscale=False,
        xgap=2, ygap=2,
    ))

    # Per-cell annotations — white text on dark cells, bold for Champion column
    for _ri, (_tm, _) in enumerate(_podium4):
        for _ci, _v in enumerate(_z[_ri]):
            _is_champ = (_ci == len(_col_labels) - 1)
            _txt_color = "white" if (_v > 90 or _v < 20) else "black"
            _txt = f"<b>{_v:.1f}%</b>" if _is_champ else f"{_v:.1f}%"
            fig_heat.add_annotation(
                x=_col_labels[_ci], y=_tm,
                text=_txt, showarrow=False,
                font=dict(color=_txt_color, size=12),
            )

    # Gold border around Champion column (categorical x position 6 → span [5.5, 6.5])
    fig_heat.add_shape(
        type="rect", xref="x", yref="y",
        x0=5.5, x1=6.5,
        y0=-0.5, y1=len(_podium4) - 0.5,
        line=dict(color="goldenrod", width=2.5),
        fillcolor="rgba(0,0,0,0)",
    )

    fig_heat.update_layout(
        title=t("dist_podium_title", lang),
        height=250,
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=30, l=130, r=20),
    )
    st.plotly_chart(fig_heat)
