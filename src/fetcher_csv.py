"""
Load match data from Kaggle "International football results" CSV.

Expected file: data/results.csv
Columns: date, home_team, away_team, home_score, away_score,
         tournament, city, country, neutral

Returns the same dict structure as fetcher.py so both sources
can be merged transparently.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_PATH = Path("data/results.csv")

# ---------------------------------------------------------------------------
# Team name aliases: Kaggle CSV name → our canonical name
# ---------------------------------------------------------------------------
CSV_TEAM_ALIASES: dict[str, str] = {
    # CONCACAF
    "United States":                    "USA",
    "United States of America":         "USA",
    # CONMEBOL
    "Copa América":                     "Copa América",   # (tournament, not team)
    # UEFA
    "Bosnia and Herzegovina":           "Bosnia-Herzegovina",
    "Bosnia & Herzegovina":             "Bosnia-Herzegovina",
    "Czechia":                          "Czech Republic",
    # CAF
    "Cote d'Ivoire":                    "Ivory Coast",
    "Côte d'Ivoire":                    "Ivory Coast",
    "Côte D'Ivoire":                    "Ivory Coast",
    "Congo DR":                         "DR Congo",
    "Congo, DR":                        "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Democratic Republic of Congo":     "DR Congo",
    "Cabo Verde":                       "Cape Verde",
    "Cape Verde Islands":               "Cape Verde",
    # AFC
    "Korea Republic":                   "South Korea",
    "Republic of Korea":                "South Korea",
    "IR Iran":                          "Iran",
    "Korea DPR":                        "North Korea",
    # OFC / CONCACAF
    "Curacao":                          "Curaçao",
    "Curaçao":                          "Curaçao",
    "Aotearoa New Zealand":             "New Zealand",
    # Misc
    "North Macedonia":                  "North Macedonia",
    "FYR Macedonia":                    "North Macedonia",
    "Kyrgyz Republic":                  "Kyrgyzstan",
}


def _canon(name: str) -> str:
    return CSV_TEAM_ALIASES.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# Tournament → internal match type
# ---------------------------------------------------------------------------
# Big competitions whose matches get KNOCKOUT weight (1.5)
_KNOCKOUT_COMPETITIONS: frozenset[str] = frozenset({
    "FIFA World Cup",
    "AFC Asian Cup",
    "Africa Cup of Nations",
    "Copa América",
    "Copa America",
    "CONCACAF Gold Cup",
    "UEFA Euro",
    "UEFA European Championship",
    "CAF Champions League",  # club, but won't match national teams
    "CONMEBOL–UEFA Cup of Champions",
    "King's Cup",
    "Gold Cup",
})

# Big competitions whose matches get GROUP weight (1.2)
_GROUP_COMPETITIONS: frozenset[str] = frozenset({
    "UEFA Nations League",
    "CONCACAF Nations League",
    "AFC Challenge Cup",
    "COSAFA Cup",
    "CECAFA Cup",
    "WAFU Cup of Nations",
    "Arab Cup",
    "Gulf Cup",
    "EAFF E-1 Football Championship",
    "OFC Nations Cup",
    "Intercontinental Cup",
})


def _tournament_type(tournament: str) -> str:
    """Map Kaggle tournament name to internal match-type key."""
    t = tournament.strip()
    tl = t.lower()

    if t == "Friendly":
        return "FRIENDLY"

    # Qualifiers: any tournament containing "qualification" or "qualifier"
    if "qualification" in tl or "qualifier" in tl:
        return "QUALIFIER"

    # Exact World Cup match (not qualifier)
    if t == "FIFA World Cup":
        return "WORLD_CUP"

    # Major single-tournament competitions → full competition weight
    if t in _KNOCKOUT_COMPETITIONS:
        return "KNOCKOUT"

    # Nations Leagues and other organised multi-round competitions
    if t in _GROUP_COMPETITIONS:
        return "GROUP"

    # Everything else (minor cups, invitational, CONCACAF, AFC sub-qualifiers, etc.)
    return "GROUP"


def _stable_id(date: str, home: str, away: str) -> str:
    """Generate a stable unique ID for a CSV match (no API id available)."""
    key = f"csv:{date}:{home}:{away}"
    return "csv_" + hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv_matches(
    date_from: str = "2022-01-01",
    known_teams: set[str] | None = None,
) -> list[dict]:
    """
    Read data/results.csv and return a list of match dicts compatible
    with the API fetcher output.

    date_from: ISO date string — only matches on or after this date are included.
    known_teams: set of canonical team names to filter on; if None, all are returned.
    """
    if not CSV_PATH.exists():
        logger.warning(
            "CSV not found at %s. Download from Kaggle: "
            "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017",
            CSV_PATH,
        )
        return []

    cutoff = date_from[:10]
    matches: list[dict] = []
    skipped_teams = 0
    skipped_date = 0
    skipped_score = 0

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_date = row.get("date", "")[:10]

            # Date filter
            if match_date < cutoff:
                skipped_date += 1
                continue

            home_raw = row.get("home_team", "").strip()
            away_raw = row.get("away_team", "").strip()
            home = _canon(home_raw)
            away = _canon(away_raw)

            # Team filter
            if known_teams is not None:
                if home not in known_teams and away not in known_teams:
                    skipped_teams += 1
                    continue

            # Score
            try:
                hg = int(row["home_score"])
                ag = int(row["away_score"])
            except (KeyError, ValueError):
                skipped_score += 1
                continue

            tournament = row.get("tournament", "").strip()
            m_type = _tournament_type(tournament)

            # Infer season from date
            try:
                season = int(match_date[:4])
            except ValueError:
                season = 0

            matches.append({
                "id":               _stable_id(match_date, home, away),
                "date":             match_date,
                "home_team":        home,
                "away_team":        away,
                "home_goals":       hg,
                "away_goals":       ag,
                "competition":      tournament,
                "competition_code": "CSV",
                "competition_type": m_type,
                "stage":            "UNKNOWN",
                "season":           season,
                "source":           "kaggle_csv",
            })

    logger.info(
        "CSV loaded: %d matches kept (skipped: %d by date, %d by team, %d bad score)",
        len(matches), skipped_date, skipped_teams, skipped_score,
    )
    return matches


def csv_stats(matches: list[dict]) -> dict:
    """Return basic statistics about a list of CSV matches."""
    from collections import Counter
    types = Counter(m["competition_type"] for m in matches)
    teams: set[str] = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    return {
        "total": len(matches),
        "teams_seen": len(teams),
        "by_type": dict(types),
    }
