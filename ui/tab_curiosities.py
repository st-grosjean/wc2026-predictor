"""Tab 3 — Curiosités: groupe de la mort, surprises SF, exits précoces, finales rares."""
from __future__ import annotations

import streamlit as st

from src import stats as db
from src.i18n import t


def render_tab_curiosities(
    lang: str,
    probs: dict,
    n: int,
    sim: dict,
    active_coeffs: dict,
) -> None:
    import pandas as pd
    cur = db.curiosities(probs, active_coeffs)

    st.subheader(t("death_group_title", lang))
    if cur["death_group"]:
        members = cur["death_group_members"]
        st.write(f"**{t('group_header', lang, grp=cur['death_group'])}** : {', '.join(members)}")
        grp_rows = [
            {
                t("col_team", lang):        team_,
                t("col_winner_pct2", lang): f"{probs.get(team_, {}).get('winner', 0)*100:.2f}",
                t("col_final_pct2", lang):  f"{probs.get(team_, {}).get('final', 0)*100:.1f}",
                t("col_sf_pct2", lang):     f"{probs.get(team_, {}).get('sf', 0)*100:.1f}",
            }
            for team_ in members if team_ in probs
        ]
        if grp_rows:
            st.table(pd.DataFrame(grp_rows))
    else:
        st.write(t("no_death_group", lang))

    st.subheader(t("surprise_sf_title", lang))
    if cur["surprise_sf"]:
        surp_rows = [
            {
                t("col_team", lang):        s["team"],
                t("col_sf_pct2", lang):     f"{s['sf_prob']*100:.1f}",
                t("col_final_pct2", lang):  f"{probs.get(s['team'], {}).get('final', 0)*100:.1f}",
                t("col_winner_pct2", lang): f"{probs.get(s['team'], {}).get('winner', 0)*100:.2f}",
            }
            for s in cur["surprise_sf"]
        ]
        st.table(pd.DataFrame(surp_rows))
    else:
        st.write(t("no_surprise_sf", lang))

    st.subheader(t("early_exit_title", lang))
    if cur["earliest_fav_exit"]:
        exit_df = pd.DataFrame([
            {
                t("col_team", lang):        e["team"],
                t("col_r32_prob", lang):    f"{e['r32_prob']*100:.1f}",
                t("col_sf_pct2", lang):     f"{probs.get(e['team'], {}).get('sf', 0)*100:.1f}",
                t("col_winner_pct2", lang): f"{e['winner_prob']*100:.2f}",
            }
            for e in cur["earliest_fav_exit"]
        ])
        st.table(exit_df)
    else:
        st.write(t("data_insufficient", lang))

    st.subheader(t("rare_finals_title", lang))
    final_pairs_mem = sim.get("final_pairs", {})
    rare_finals   = db.finals_from_memory(final_pairs_mem, n, least=True,  limit=5)
    common_finals = db.finals_from_memory(final_pairs_mem, n, least=False, limit=3)

    if rare_finals:
        rf = rare_finals[0]
        st.info(t("rarest_final_info", lang,
                  team_a=rf["team_a"], team_b=rf["team_b"],
                  count=rf["count"], n=f"{n:,}", pct=f"{rf['pct']:.3f}"))
        rare_df = pd.DataFrame([
            {
                t("col_team_a", lang):    r["team_a"],
                t("col_team_b", lang):    r["team_b"],
                t("col_count", lang):     r["count"],
                t("col_pct_sims", lang):  f"{r['pct']:.3f}",
            }
            for r in rare_finals
        ])
        st.table(rare_df)
        if common_finals:
            st.caption(t("common_finals_caption", lang))
            common_df = pd.DataFrame([
                {
                    t("col_team_a", lang):   r["team_a"],
                    t("col_team_b", lang):   r["team_b"],
                    t("col_count", lang):    r["count"],
                    t("col_pct_sims", lang): f"{r['pct']:.1f}",
                }
                for r in common_finals
            ])
            st.table(common_df)
    else:
        st.info(t("no_rare_finals", lang))
