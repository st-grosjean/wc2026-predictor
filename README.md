# WC 2026 Predictor

Simulateur de la Coupe du Monde FIFA 2026 par méthode Monte Carlo + modèle de Poisson Dixon-Coles.  
FIFA World Cup 2026 simulator using Monte Carlo + Dixon-Coles Poisson model.

---

## Prérequis / Prerequisites

- [uv](https://docs.astral.sh/uv/) - gestionnaire de paquets Python
- Python 3.12+

## Installation

```bash
uv sync
```

## Données requises / Required data

Le fichier `data/results.csv` n'est pas inclus dans le repo.  
Pour l'obtenir / To get it :
1. Créer un compte gratuit sur / Create a free account at [kaggle.com](https://www.kaggle.com)
2. Télécharger le dataset / Download the dataset :  
   https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2026
3. Dézipper l'archive / Unzip the archive :  
   `international football results 1872 2026.zip`
4. Placer `results.csv` dans le dossier `data/`  
   Place `results.csv` in the `data/` folder

Sans ce fichier : l'app fonctionne avec les coefficients de fallback intégrés mais les prédictions seront moins précises

Without this file : the app works with built-in fallback coefficients but predictions will be less accurate.

---

## API football-data.org (optionnel / optional)

Pour enrichir les données avec les matchs les plus récents :  
To enrich data with the most recent matches :
1. S'inscrire gratuitement sur / Register for free at :  
   https://www.football-data.org/client/register
2. Copier la clé dans / Copy the key to `.env` :
   ```
   FOOTBALL_DATA_API_KEY=your_key_here
   ```

---

## Lancement / Run

```bash
uv run streamlit run app.py
```

---

## Pipeline en 3 étapes / 3-step pipeline

| Étape | Description (FR) | Description (EN) |
|-------|-----------------|-----------------|
| **1 - Coefficients** | Récupère les matchs via l'API football-data.org et/ou le .csv de Kaggle (ou fallback JSON) et calcule les coefficients att/def pondérés temporellement | Fetches matches via football-data.org API and/or .csv from Kaggle (or JSON fallback) and computes time-weighted att/def coefficients |
| **2 - Calibration** | Ajustement Dixon-Coles par MLE (L-BFGS-B via scipy) - affiche les métriques avant/après calibration | Dixon-Coles MLE calibration (L-BFGS-B via scipy) - shows before/after metrics |
| **3 - Simulation** | Monte Carlo vectorisé (jusqu'à 1 000 000 simulations) sur l'intégralité du tournoi - résultats en 4 sous-onglets | Vectorised Monte Carlo (up to 1 000 000 simulations) over the full tournament - results in 4 sub-tabs |

---

## Stack technique / Tech stack

| Composant | Technologie |
|-----------|------------|
| UI | Streamlit |
| Modèle statistique | Dixon-Coles (Poisson bivarié) |
| Optimisation | scipy `L-BFGS-B` |
| Simulation | NumPy vectorisé |
| Graphiques | Plotly |
| Persistance | SQLite (sqlite3) |
| Gestion des paquets | uv |
| Langues supportées | FR, EN, PT, ES, DE, JA, KO |

---

## Structure

```
app.py              # Interface Streamlit principale
config.py           # Paramètres globaux (poids, decay, API)
src/
  fetcher.py        # Récupération API + cache local
  coefficients.py   # Calcul att/def pondérés
  training.py       # Calibration Dixon-Coles MLE
  poisson.py        # Simulation match Poisson
  tournament.py     # Phases de groupes + tableau KO
  montecarlo.py     # Moteur Monte Carlo vectorisé
  stats.py          # SQLite CRUD + curiosités statistiques
  i18n.py           # Internationalisation (t(), tn(), fmt_*)
data/
  teams.json        # 48 équipes avec groupes A–L
  i18n.json         # ~130 clés UI × 7 langues
  team_names.json   # Noms d'équipes × 7 langues
  schedule.json     # Calendrier officiel FIFA 2026
ui/
  __init__.py           # Package Python
  common.py             # Constantes et helpers partagés
  sidebar.py            # Sélecteur langue + équipe focus
  step1.py              # Étape 1 — Données & coefficients
  step2.py              # Étape 2 — Training & calibrage
  step3.py              # Étape 3 — Monte Carlo & résultats
  tab_bracket.py        # Onglet bracket KO
  tab_curiosities.py    # Onglet curiosités statistiques
  tab_focus.py          # Onglet focus équipe
  tab_global.py         # Onglet résultats globaux
  tab_groups.py         # Onglet phase de groupes
  tab_methodology.py    # Onglet méthodologie & doc
  tab_predictions.py    # Onglet prédictions simulées
  tournament_status.py  # Bandeau statut tournoi live
```
