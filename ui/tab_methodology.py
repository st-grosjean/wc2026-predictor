"""Tab — Methodology: pipeline, models, and references.

FR and EN only. Other languages fall back to EN via _ml().
Long explanatory text is kept in _TEXT[lang] rather than i18n.json
to avoid cluttering it with ~60 paragraph-length keys.
"""
from __future__ import annotations

import math

import streamlit as st

import config
from src import stats as db
from src.i18n import t
from src.coefficients import _years_elapsed
from ui.common import _active_coeffs, load_teams_json

# Languages that have full methodology text. Others fall back to "en".
_METH_LANGS = frozenset({"fr", "en"})


def _ml(lang: str) -> str:
    return lang if lang in _METH_LANGS else "en"


# ---------------------------------------------------------------------------
# Explanatory text blocks (not in i18n.json — paragraph-length)
# ---------------------------------------------------------------------------

_TEXT: dict[str, dict[str, str]] = {
    "fr": {
        "s1_caption": (
            "Le pipeline transforme des données historiques de matchs nationaux "
            "en probabilités de victoire pour les 48 équipes du Mondial 2026."
        ),
        "s1_data_desc": (
            "**Données** — Matchs internationaux depuis 2022 (API football-data.org "
            "ou CSV Kaggle). Chaque match porte : date, équipes, score, type de compétition."
        ),
        "s1_calib_desc": (
            "**Calibrage** — Optimisation Dixon-Coles MLE : on cherche les coefficients "
            "att/def qui maximisent la vraisemblance des scores observés, avec décroissance "
            "temporelle et pondération par type de compétition."
        ),
        "s1_mc_desc": (
            "**Monte Carlo** — On simule N tournois complets (jusqu'à 1 million). "
            "Dans chaque simulation, chaque match est tiré aléatoirement depuis une "
            "loi de Poisson paramétrée par att×def×μ. On compte les fréquences."
        ),
        "s2_why": (
            "**Pourquoi Poisson ?** Le football produit des buts rares et indépendants "
            "sur 90 minutes — la définition textuelle d'un processus de Poisson. "
            "Sur l'ensemble des matchs internationaux, la distribution des buts par "
            "équipe suit de très près une Poisson(λ) avec λ ≈ 1.27 buts/équipe/match."
        ),
        "s2_formula_intro": (
            "Pour un match A (domicile) contre B (extérieur), les taux de buts attendus sont :"
        ),
        "s2_example_pre": (
            "Avec μ = {mu:.2f}, att_France = {att_f:.3f}, def_France = {def_f:.3f}, "
            "att_Maroc = {att_m:.3f}, def_Maroc = {def_m:.3f} :"
        ),
        "s2_example_vals": "λ_France = {lf:.3f} buts attendus, λ_Maroc = {lm:.3f} buts attendus",
        "s2_heatmap_pick": "Choisissez deux équipes pour afficher la heatmap des scores :",
        "s2_heatmap_title": "Distribution des scores — {a} vs {b}  (λ_A={lA:.2f}, λ_B={lB:.2f})",
        "s2_no_coeffs": (
            "Coefficients non disponibles — calculez-les à l'Étape 1."
        ),
        "s3_weights_intro": (
            "Tous les matchs ne se valent pas. Un match de Coupe du Monde contre "
            "l'Argentine mérite plus de poids qu'un amical contre le Luxembourg."
        ),
        "s3_decay_desc": (
            "Le poids total d'un match est : **w = w_base × exp(−λ × années)** "
            "avec λ = {lam:.2f}. Un match WC vieux de 3 ans vaut autant "
            "qu'un amical récent d'il y a 6 mois."
        ),
        "s3_elo_desc": (
            "**Facteur ELO (attack) :** les buts marqués contre un adversaire fort "
            "(ELO élevé) augmentent le coefficient att. Formule : opp_strength = "
            "elo_adversaire / elo_moyen. Dans le MLE, c'est la moyenne géométrique "
            "√(elo_A/μ_elo × elo_B/μ_elo) qui pondère l'importance du match. "
            "Les matchs contre des adversaires très faibles (ELO < {thresh}) sont "
            "cappés à {cap:.0%} de leur poids normal."
        ),
        "s4_loglik": (
            "**Log-vraisemblance :** on cherche les paramètres θ = (att_i, def_i) "
            "qui rendent les scores observés les plus probables. "
            "ℓ(θ) = Σ_k w_k [hg_k·log(λ_h) − λ_h + ag_k·log(λ_a) − λ_a]. "
            "On maximise ℓ(θ), ce qui revient à minimiser −ℓ(θ)."
        ),
        "s4_loss_desc": (
            "**Perte mixte :** L = α·L_Poisson + β·L_W/D/L + λ_L2·‖θ−θ₀‖² "
            "avec α={alpha}, β={beta}. "
            "Le terme W/D/L évite de sur-pénaliser une équipe dominante dont le "
            "modèle prédit 3-0 mais qui a gagné 5-0 : le *résultat* (victoire) est correct. "
            "La régularisation L2 (λ={l2}) évite le sur-ajustement pour les équipes "
            "avec peu de matchs."
        ),
        "s6_why": (
            "**Pourquoi simuler plutôt que calculer ?** "
            "Le tournoi a 48 équipes, 4 tours éliminatoires + la finale. "
            "Calculer analytiquement la probabilité d'un vainqueur nécessite "
            "d'intégrer sur tous les chemins possibles — un espace combinatoire "
            "de l'ordre de 10^20 scénarios. Le Monte Carlo résout ce problème en "
            "N tirages aléatoires indépendants, chacun simulant un tournoi complet."
        ),
        "s6_convergence": (
            "**Convergence :** pour N simulations, l'intervalle de confiance à 95% "
            "sur une probabilité p est ±1.96×√(p(1−p)/N). "
            "Pour p=10% et N=1 000 000 : ±0.06%. Pour N=10 000 : ±0.59%."
        ),
        "s6_p_slider": "Probabilité de référence p",
        "s7_missing": (
            "**Données manquantes :** certaines fédérations (OFC, CONCACAF petites îles) "
            "ont très peu de matchs dans le dataset. Leurs coefficients sont blendés avec "
            "un prior ELO pour compenser."
        ),
        "s7_variance": (
            "**Haute variance intrinsèque :** avec 1 à 2 buts/équipe/match, un seul but "
            "suffit à changer l'issue. Même les meilleures équipes ont des probabilités "
            "de victoire finale inférieures à 25%."
        ),
        "s7_not_captured": (
            "**Non modélisé :** blessures en cours de tournoi, forme récente non linéaire, "
            "conditions climatiques (chaleur au Qatar 2022), pression psychologique des "
            "tirs au but, ni l'effet « surprise » des équipes qui découvrent ce niveau."
        ),
        "s7_bookmakers": (
            "**Calibration externe :** les bookmakers donnent généralement France/Argentine "
            "à 15-20% de chances de victoire. Notre modèle étant dans cette plage, "
            "la calibration globale est cohérente."
        ),
        "s8_intro": "Données, méthodologie et publications scientifiques ayant inspiré ce projet.",
    },
    "en": {
        "s1_caption": (
            "The pipeline transforms historical international match data into win "
            "probabilities for all 48 WC 2026 teams."
        ),
        "s1_data_desc": (
            "**Data** — International matches since 2022 (football-data.org API "
            "or Kaggle CSV). Each match carries: date, teams, score, competition type."
        ),
        "s1_calib_desc": (
            "**Calibration** — Dixon-Coles MLE optimisation: find att/def coefficients "
            "that maximise the likelihood of observed scores, with temporal decay and "
            "competition-type weighting."
        ),
        "s1_mc_desc": (
            "**Monte Carlo** — Simulate N complete tournaments (up to 1 million). "
            "In each simulation every match score is drawn from a Poisson distribution "
            "parametrised by att×def×μ. Win probabilities are estimated from frequencies."
        ),
        "s2_why": (
            "**Why Poisson?** Football produces rare, independent goals over 90 minutes — "
            "the textbook definition of a Poisson process. Across international matches, "
            "goals per team per match closely follow Poisson(λ) with λ ≈ 1.27."
        ),
        "s2_formula_intro": (
            "For a match A (home) vs B (away), expected goal rates are:"
        ),
        "s2_example_pre": (
            "With μ = {mu:.2f}, att_France = {att_f:.3f}, def_France = {def_f:.3f}, "
            "att_Morocco = {att_m:.3f}, def_Morocco = {def_m:.3f}:"
        ),
        "s2_example_vals": "λ_France = {lf:.3f} expected goals, λ_Morocco = {lm:.3f} expected goals",
        "s2_heatmap_pick": "Choose two teams to display the score probability heatmap:",
        "s2_heatmap_title": "Score distribution — {a} vs {b}  (λ_A={lA:.2f}, λ_B={lB:.2f})",
        "s2_no_coeffs": "Coefficients unavailable — compute them in Step 1.",
        "s3_weights_intro": (
            "Not all matches carry equal value. A World Cup match against Argentina "
            "deserves more weight than a friendly against Luxembourg."
        ),
        "s3_decay_desc": (
            "Total match weight: **w = w_base × exp(−λ × years)** "
            "with λ = {lam:.2f}. A 3-year-old WC match equals a friendly from 6 months ago."
        ),
        "s3_elo_desc": (
            "**ELO factor (attack):** goals scored against a strong opponent (high ELO) "
            "boost the att coefficient. Formula: opp_strength = elo_opponent / elo_mean. "
            "In MLE, the geometric mean √(elo_A/μ_elo × elo_B/μ_elo) weights each match "
            "by its difficulty. Matches against very weak opponents (ELO < {thresh}) are "
            "capped at {cap:.0%} of normal weight."
        ),
        "s4_loglik": (
            "**Log-likelihood:** find parameters θ = (att_i, def_i) that make "
            "observed scores most probable. "
            "ℓ(θ) = Σ_k w_k [hg_k·log(λ_h) − λ_h + ag_k·log(λ_a) − λ_a]. "
            "Maximising ℓ(θ) is equivalent to minimising −ℓ(θ)."
        ),
        "s4_loss_desc": (
            "**Mixed loss:** L = α·L_Poisson + β·L_W/D/L + λ_L2·‖θ−θ₀‖² "
            "with α={alpha}, β={beta}. "
            "The W/D/L term avoids over-penalising a dominant team whose model predicts "
            "3-0 but scored 5-0: the *outcome* (win) is correct. "
            "L2 regularisation (λ={l2}) prevents overfitting for data-sparse teams."
        ),
        "s6_why": (
            "**Why simulate instead of compute?** "
            "The 48-team tournament has 4 knockout rounds + final. Computing the exact "
            "probability of each outcome analytically means integrating over ~10^20 "
            "scenario paths. Monte Carlo solves this with N independent random draws, "
            "each simulating a full tournament."
        ),
        "s6_convergence": (
            "**Convergence:** for N simulations, the 95% confidence interval on "
            "probability p is ±1.96×√(p(1−p)/N). "
            "For p=10% and N=1,000,000: ±0.06%. For N=10,000: ±0.59%."
        ),
        "s6_p_slider": "Reference probability p",
        "s7_missing": (
            "**Missing data:** some federations (OFC, small CONCACAF islands) have very "
            "few matches in the dataset. Their coefficients are blended with an ELO prior."
        ),
        "s7_variance": (
            "**Intrinsic high variance:** with 1–2 goals/team/match, a single goal "
            "can flip the outcome. Even the best teams have tournament win probabilities "
            "below 25%."
        ),
        "s7_not_captured": (
            "**Not modelled:** in-tournament injuries, non-linear recent form, climate "
            "conditions, penalty-shootout psychology, or the 'surprise' factor of teams "
            "experiencing this level for the first time."
        ),
        "s7_bookmakers": (
            "**External calibration:** bookmakers typically give France/Argentina "
            "15–20% win probability. Our model is in this range, validating "
            "the overall calibration."
        ),
        "s8_intro": "Data sources, methodology and scientific publications that inspired this project.",
    },
}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_pipeline(lang: str) -> None:
    st.subheader(t("meth_s1_header", lang))
    st.caption(_TEXT[lang]["s1_caption"])

    col_d, col_arr1, col_c, col_arr2, col_m = st.columns([5, 1, 5, 1, 5])

    _box = (
        "background:{bg};border-radius:8px;padding:12px 10px;"
        "text-align:center;border:1px solid {border};"
    )
    with col_d:
        st.markdown(
            f"<div style='{_box.format(bg='#1e3a5f', border='#2d6a9f')}'>"
            "<b style='color:#7ec8e3'>📥 DONNÉES</b><br>"
            "<small style='color:#aad4e8'>API / CSV / Cache<br>"
            "football-data.org · Kaggle<br>matches_cache.json</small></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_TEXT[lang]["s1_data_desc"])

    with col_arr1:
        st.markdown(
            "<div style='text-align:center;padding-top:20px;"
            "font-size:28px;color:#888'>→</div>",
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            f"<div style='{_box.format(bg='#3a1f5f', border='#7a4db8')}'>"
            "<b style='color:#c8a0e3'>⚙️ CALIBRAGE</b><br>"
            "<small style='color:#d8c0f0'>Dixon-Coles MLE<br>"
            "src/training.py<br>src/coefficients.py</small></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_TEXT[lang]["s1_calib_desc"])

    with col_arr2:
        st.markdown(
            "<div style='text-align:center;padding-top:20px;"
            "font-size:28px;color:#888'>→</div>",
            unsafe_allow_html=True,
        )

    with col_m:
        st.markdown(
            f"<div style='{_box.format(bg='#1f4a2f', border='#3a9a5f')}'>"
            "<b style='color:#80e3a0'>🎲 MONTE CARLO</b><br>"
            "<small style='color:#a8f0c0'>N tournois simulés<br>"
            "src/montecarlo.py<br>wc2026.db</small></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_TEXT[lang]["s1_mc_desc"])


def _section_poisson(lang: str, coeffs: dict, teams_list: list[str]) -> None:
    import numpy as np
    import plotly.graph_objects as go

    st.subheader(t("meth_s2_header", lang))
    st.markdown(_TEXT[lang]["s2_why"])

    st.markdown(_TEXT[lang]["s2_formula_intro"])
    st.latex(r"\lambda_A = \text{att}_A \times \text{def}_B \times \mu")
    st.latex(r"\lambda_B = \text{att}_B \times \text{def}_A \times \mu")

    # Numeric example with France vs Morocco
    mu = config.MU
    c_fr  = coeffs.get("France",  {"att_rating": 1.68, "def_rating": 0.72})
    c_mor = coeffs.get("Morocco", {"att_rating": 1.28, "def_rating": 0.82})
    lf = c_fr["att_rating"]  * c_mor["def_rating"] * mu
    lm = c_mor["att_rating"] * c_fr["def_rating"]  * mu
    try:
        ex_txt = _TEXT[lang]["s2_example_pre"].format(
            mu=mu,
            att_f=c_fr["att_rating"],  def_f=c_fr["def_rating"],
            att_m=c_mor["att_rating"], def_m=c_mor["def_rating"],
        )
        st.caption(ex_txt)
        st.info(_TEXT[lang]["s2_example_vals"].format(lf=lf, lm=lm))
    except Exception:
        pass

    st.divider()
    st.markdown(_TEXT[lang]["s2_heatmap_pick"])

    sorted_teams = sorted(teams_list)
    default_a = sorted_teams.index("France") if "France" in sorted_teams else 0
    default_b = sorted_teams.index("Morocco") if "Morocco" in sorted_teams else min(1, len(sorted_teams) - 1)

    col_a, col_b = st.columns(2)
    team_a = col_a.selectbox(t("meth_s2_team_a", lang), sorted_teams, index=default_a, key="meth_ta")
    team_b = col_b.selectbox(t("meth_s2_team_b", lang), sorted_teams, index=default_b, key="meth_tb")

    ca = coeffs.get(team_a, {"att_rating": 1.0, "def_rating": 1.0})
    cb = coeffs.get(team_b, {"att_rating": 1.0, "def_rating": 1.0})
    lA = ca["att_rating"] * cb["def_rating"] * mu
    lB = cb["att_rating"] * ca["def_rating"] * mu

    max_g = 6
    goals = np.arange(max_g)

    def _pmf(lam: float) -> np.ndarray:
        return np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in goals])

    pmf_a = _pmf(lA)
    pmf_b = _pmf(lB)
    grid = np.outer(pmf_a, pmf_b) * 100  # percent

    p_win_a = float(np.sum(np.tril(grid, -1)))
    p_draw  = float(np.sum(np.diag(grid)))
    p_win_b = float(np.sum(np.triu(grid, 1)))

    text_grid = [[f"{grid[i, j]:.1f}%" for j in range(max_g)] for i in range(max_g)]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=[str(j) for j in goals],
        y=[str(i) for i in goals],
        text=text_grid,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="%"),
        hovertemplate=f"{team_a} %{{y}} — {team_b} %{{x}}: %{{text}}<extra></extra>",
    ))
    title = _TEXT[lang]["s2_heatmap_title"].format(a=team_a, b=team_b, lA=lA, lB=lB)
    fig.update_layout(
        title=title,
        xaxis_title=f"⚽ {team_b}",
        yaxis_title=f"⚽ {team_a}",
        height=450,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width="content")

    c1, c2, c3 = st.columns(3)
    c1.metric(t("meth_s2_win_a", lang, a=team_a), f"{p_win_a:.1f}%")
    c2.metric(t("meth_s2_draw",  lang), f"{p_draw:.1f}%")
    c3.metric(t("meth_s2_win_b", lang, b=team_b), f"{p_win_b:.1f}%")


def _section_weights(lang: str) -> None:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    st.subheader(t("meth_s3_header", lang))
    st.markdown(_TEXT[lang]["s3_weights_intro"])

    # Weight table
    weight_labels = {
        "fr": {
            "FRIENDLY": "Amical", "QUALIFIER": "Qualificatif",
            "GROUP": "Phase de groupes (compétitions)", "KNOCKOUT": "Phase finale",
            "WORLD_CUP": "Coupe du Monde",
        },
        "en": {
            "FRIENDLY": "Friendly", "QUALIFIER": "Qualifier",
            "GROUP": "Group stage (competitions)", "KNOCKOUT": "Knockout",
            "WORLD_CUP": "World Cup",
        },
    }
    wl = weight_labels[lang]
    why = {
        "fr": {
            "FRIENDLY": "Effectifs remaniés, enjeu limité",
            "QUALIFIER": "Enjeu fort, adversaires réels",
            "GROUP": "Compétition avec classement",
            "KNOCKOUT": "Pression d'élimination",
            "WORLD_CUP": "Niveau maximal, meilleur contexte",
        },
        "en": {
            "FRIENDLY": "Rotated squads, limited stakes",
            "QUALIFIER": "High stakes, real opponents",
            "GROUP": "Competitive group stage",
            "KNOCKOUT": "Elimination pressure",
            "WORLD_CUP": "Top level, best context",
        },
    }[lang]
    col_type = "Type" if lang == "en" else "Type"
    col_base = "w_base"
    col_why  = "Justification" if lang == "en" else "Justification"
    df_w = pd.DataFrame([
        {col_type: wl[k], col_base: v, col_why: why[k]}
        for k, v in config.MATCH_WEIGHTS.items()
    ])
    st.dataframe(df_w, hide_index=True, width="stretch")

    # Temporal decay chart
    st.markdown("**" + t("meth_s3_header", lang).split("—")[-1].strip() + "**")
    st.markdown(_TEXT[lang]["s3_decay_desc"].format(lam=config.LAMBDA_DECAY))

    years_max = st.slider(t("meth_s3_decay_years", lang), 1, 10, 5, key="meth_decay")
    years_arr = np.linspace(0, years_max, 300)

    fig = go.Figure()
    colors = {"FRIENDLY": "#888", "QUALIFIER": "#4a9", "GROUP": "#48c",
              "KNOCKOUT": "#a60", "WORLD_CUP": "#e33"}
    for mtype, base_w in config.MATCH_WEIGHTS.items():
        w_arr = base_w * np.exp(-config.LAMBDA_DECAY * years_arr)
        fig.add_trace(go.Scatter(
            x=years_arr, y=w_arr, name=wl[mtype],
            line=dict(color=colors[mtype], width=2),
        ))
    fig.update_layout(
        xaxis_title=t("meth_s3_decay_years", lang),
        yaxis_title="w",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, width="content")

    # ELO section
    st.markdown(_TEXT[lang]["s3_elo_desc"].format(
        thresh=config.ELO_WEAK_OPP_THRESHOLD,
        cap=config.ELO_WEAK_OPP_CAP_FACTOR,
    ))


def _section_dc(lang: str) -> None:
    import pandas as pd

    st.subheader(t("meth_s4_header", lang))

    st.markdown(_TEXT[lang]["s4_loglik"])
    st.latex(
        r"\ell(\theta) = \sum_k w_k "
        r"\bigl[hg_k \ln\lambda_h - \lambda_h + ag_k \ln\lambda_a - \lambda_a\bigr]"
    )

    st.markdown(_TEXT[lang]["s4_loss_desc"].format(
        alpha=config.LOSS_ALPHA, beta=config.LOSS_BETA, l2=config.L2_LAMBDA,
    ))
    st.latex(
        r"L = \alpha \cdot L_{\text{Poisson}} + \beta \cdot L_{\text{W/D/L}}"
        r"+ \lambda_{L2} \|\theta - \theta_0\|^2"
    )

    # Last calibration metrics from DB
    st.markdown(f"**{t('meth_s4_metrics', lang)}**")
    runs = db.load_training_runs()
    if not runs:
        st.info(t("meth_s4_no_calib", lang))
        return

    last = runs[0]
    mb = last.get("metrics_before") or {}
    ma = last.get("metrics_after")  or {}
    if isinstance(mb, str):
        import json
        try:
            mb = json.loads(mb)
        except Exception:
            mb = {}
    if isinstance(ma, str):
        import json
        try:
            ma = json.loads(ma)
        except Exception:
            ma = {}

    met_labels = (
        ["Accuracy W/D/L", "MAE buts", "Brier score"]
        if lang == "fr"
        else ["Accuracy W/D/L", "Goal MAE", "Brier score"]
    )
    col_b = "Avant" if lang == "fr" else "Before"
    col_a = "Après" if lang == "fr" else "After"
    rows = []
    for lbl, key in zip(met_labels, ["accuracy", "mae", "brier"]):
        bv = mb.get(key)
        av = ma.get(key)
        rows.append({
            "Métrique" if lang == "fr" else "Metric": lbl,
            col_b: f"{bv:.3f}" if bv is not None else "—",
            col_a: f"{av:.3f}" if av is not None else "—",
        })
    st.table(pd.DataFrame(rows))


def _section_h2h(lang: str, matches: list[dict], teams_list: list[str]) -> None:
    import pandas as pd

    st.subheader(t("meth_s5_header", lang))

    st.latex(
        r"\text{raw}(A,B) = \frac{\sum_k w_k \cdot \text{sign}(\Delta g_k)"
        r"\cdot \ln(1+|\Delta g_k|)}{\sum_k w_k}"
    )
    st.latex(r"h2h(A,B) = \tanh\!\bigl(\text{raw}(A,B) \times s\bigr), \quad s="
             + str(config.H2H_SCALING))

    _desc = {
        "fr": (
            "Le tanh borne le score dans (−1, +1). "
            "Dans la simulation, λ de chaque équipe est multiplié par "
            f"(1 ± {config.H2H_WEIGHT} × score)."
        ),
        "en": (
            "tanh bounds the score in (−1, +1). "
            "In simulation, each team's λ is multiplied by "
            f"(1 ± {config.H2H_WEIGHT} × score)."
        ),
    }
    st.markdown(_desc[lang])

    if not matches:
        st.info(t("meth_s5_no_match", lang))
        return

    sorted_teams = sorted(teams_list)
    default_a = sorted_teams.index("France")  if "France"  in sorted_teams else 0
    default_b = sorted_teams.index("Morocco") if "Morocco" in sorted_teams else min(1, len(sorted_teams) - 1)

    col_a, col_b = st.columns(2)
    ta = col_a.selectbox(t("meth_s5_team_a", lang), sorted_teams, index=default_a, key="meth_h2h_a")
    tb = col_b.selectbox(t("meth_s5_team_b", lang), sorted_teams, index=default_b, key="meth_h2h_b")

    # Filter relevant matches
    relevant = [
        m for m in matches
        if {m.get("home_team"), m.get("away_team")} == {ta, tb}
    ]

    if not relevant:
        st.info(t("meth_s5_no_match", lang))
        return

    relevant.sort(key=lambda m: m.get("date", ""))

    # Compute H2H score detail
    from src.h2h import build_h2h_table
    h2h_tbl = build_h2h_table(relevant)
    score_ab = h2h_tbl.get((ta, tb), 0.0)
    score_lbl = (
        f"**Score H2H ({ta}) : {score_ab:+.3f}**"
        if lang == "fr"
        else f"**H2H score ({ta}): {score_ab:+.3f}**"
    )
    st.markdown(score_lbl)

    rows = []
    for m in relevant:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        hg   = m.get("home_goals", 0)
        ag   = m.get("away_goals", 0)
        years = _years_elapsed(m.get("date", ""))
        base_w = config.MATCH_WEIGHTS.get(m.get("competition_type", "FRIENDLY"), 0.3)
        w = base_w * math.exp(-config.LAMBDA_DECAY * years)
        gd = int(hg) - int(ag) if home == ta else int(ag) - int(hg)
        contrib = math.copysign(math.log1p(abs(gd)), gd) * w if gd != 0 else 0.0
        rows.append({
            t("meth_s5_col_date",   lang): m.get("date", ""),
            t("meth_s5_col_score",  lang): f"{home} {hg}–{ag} {away}",
            t("meth_s5_col_comp",   lang): m.get("competition", ""),
            t("meth_s5_col_weight", lang): f"{w:.3f}",
            t("meth_s5_col_contrib",lang): f"{contrib:+.3f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _section_mc(lang: str) -> None:
    import numpy as np
    import plotly.graph_objects as go

    st.subheader(t("meth_s6_header", lang))
    st.markdown(_TEXT[lang]["s6_why"])
    st.markdown(_TEXT[lang]["s6_convergence"])

    # Interactive convergence chart
    p_ref = st.slider(
        _TEXT[lang]["s6_p_slider"],
        0.01, 0.50, 0.10, 0.01,
        key="meth_p_ref",
        format="%.0f%%",
    )
    n_values = np.logspace(3, 6, 300)
    se_values = 1.96 * np.sqrt(p_ref * (1 - p_ref) / n_values) * 100  # percent

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_values, y=se_values,
        mode="lines",
        line=dict(color="#4a9eff", width=2),
        name="±1.96·SE (%)",
        fill="tozeroy",
        fillcolor="rgba(74,158,255,0.15)",
    ))
    # Reference lines
    for n_ref, label in [(10_000, "10k"), (100_000, "100k"), (1_000_000, "1M")]:
        se_ref = 1.96 * math.sqrt(p_ref * (1 - p_ref) / n_ref) * 100
        fig.add_vline(x=n_ref, line_dash="dash", line_color="#888", line_width=1)
        fig.add_annotation(
            x=math.log10(n_ref), y=se_ref + 0.05,
            text=f"{label}: ±{se_ref:.2f}%",
            showarrow=False, font=dict(size=11, color="#aaa"),
            xref="x", yref="y",
        )
    fig.update_xaxes(type="log", title="N simulations")
    fig.update_yaxes(title="95% CI width (%)")
    fig.update_layout(
        title=t("meth_s6_ci_title", lang, p=p_ref),
        height=350,
        margin=dict(t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, width="content")


def _section_limits(lang: str) -> None:
    st.subheader(t("meth_s7_header", lang))
    st.markdown(_TEXT[lang]["s7_missing"])
    st.markdown(_TEXT[lang]["s7_variance"])
    st.markdown(_TEXT[lang]["s7_not_captured"])
    st.markdown(_TEXT[lang]["s7_bookmakers"])


def _section_refs(lang: str) -> None:
    st.subheader(t("meth_s8_header", lang))
    st.markdown(_TEXT[lang]["s8_intro"])
    st.markdown(
        "- **Dixon, M. J. & Coles, S. G.** (1997). *Modelling Association Football Scores "
        "and Inefficiencies in the Football Betting Market.* Applied Statistics, 46(2), 265–280.\n"
        "- **football-data.org** — Free-tier international match data API\n"
        "- **Kaggle** — martj42/international-football-results (>45 000 international results)\n"
        "- **eloratings.net** — World Football ELO Ratings\n"
        "- **Baio, G. & Blangiardo, M.** (2010). *Bayesian hierarchical model for the "
        "prediction of football results.* Journal of Applied Statistics, 37(2), 253–264.\n"
        "- **Maher, M. J.** (1982). *Modelling association football scores.* Statistica Neerlandica, "
        "36(3), 109–118."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_tab_methodology(lang: str) -> None:
    _lang = _ml(lang)

    with st.expander(t("meth_tab_label", _lang), expanded=False):
        # Collect data needed for interactive sections
        coeffs   = _active_coeffs()
        tj       = load_teams_json()
        teams_list = list(tj["teams"].keys())

        if not coeffs:
            for team_, d in tj["teams"].items():
                coeffs[team_] = {
                    "att_rating": d["att_rating"],
                    "def_rating": d["def_rating"],
                }

        matches: list[dict] = st.session_state.get("matches", [])

        _section_pipeline(_lang)
        st.divider()
        _section_poisson(_lang, coeffs, teams_list)
        st.divider()
        _section_weights(_lang)
        st.divider()
        _section_dc(_lang)
        st.divider()
        _section_h2h(_lang, matches, teams_list)
        st.divider()
        _section_mc(_lang)
        st.divider()
        _section_limits(_lang)
        st.divider()
        _section_refs(_lang)
