"""
Fetch national team matches from football-data.org v4 and cache locally.

Free-tier competitions accessible by code (string):
  WC  — FIFA World Cup
  EC  — UEFA European Championship
  WCQ — FIFA World Cup Qualifiers (all confederations, when available)

Use  /v4/competitions/{CODE}/matches?season={YEAR}&status=FINISHED
instead of numeric IDs + date-range, which produces HTTP 400.
"""
import json
import logging
import time
import hashlib
from datetime import datetime, date
from pathlib import Path

import requests

import config
from src.fetcher_csv import load_csv_matches

# Streamlit cache — degrades gracefully when running outside Streamlit (scripts, tests)
try:
    from streamlit import cache_data as _st_cache_data
except Exception:
    def _st_cache_data(fn=None, *, ttl=None, **kw):  # type: ignore[misc]
        def _dec(f):
            return f
        return fn if fn is not None else _dec

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/matches_cache.json")

# ---------------------------------------------------------------------------
# Competition config — free tier string codes
# ---------------------------------------------------------------------------
# Each entry: code (str), base match-type, list of seasons to try.
# Seasons are tried in order; 400/403 responses are skipped silently.
COMPETITION_CONFIG: list[dict] = [
    {"code": "WC",  "base_type": "WORLD_CUP", "seasons": [2022]},
    # WC 2026 group stage starts 11 Jun 2026 — add season 2026 once matches are FINISHED
    {"code": "EC",  "base_type": "KNOCKOUT",  "seasons": [2024, 2020]},
    {"code": "WCQ", "base_type": "QUALIFIER", "seasons": [2022, 2026]},
]

# ---------------------------------------------------------------------------
# Team name aliases: API name → our canonical name
# ---------------------------------------------------------------------------
TEAM_ALIASES: dict[str, str] = {
    # General
    "United States":                    "USA",
    "United States of America":         "USA",
    "Republic of Ireland":              "Ireland",
    "Côte d'Ivoire":                    "Ivory Coast",
    "Cote d'Ivoire":                    "Ivory Coast",
    "Côte D'Ivoire":                    "Ivory Coast",
    "Korea Republic":                   "South Korea",
    "Republic of Korea":                "South Korea",
    "IR Iran":                          "Iran",
    "Congo DR":                         "DR Congo",
    "Democratic Republic of Congo":     "DR Congo",
    "DR Congo":                         "DR Congo",
    "Czechia":                          "Czech Republic",
    "Czech Republic":                   "Czech Republic",
    "Bosnia and Herzegovina":           "Bosnia-Herzegovina",
    "Bosnia & Herzegovina":             "Bosnia-Herzegovina",
    "Bosnia-Herzegovina":               "Bosnia-Herzegovina",
    "Cabo Verde":                       "Cape Verde",
    "Cape Verde Islands":               "Cape Verde",
    "Curacao":                          "Curaçao",
    "Curaçao":                          "Curaçao",
    "New Zealand":                      "New Zealand",
    "Aotearoa New Zealand":             "New Zealand",
    "Saudi Arabia":                     "Saudi Arabia",
    "Kingdom of Saudi Arabia":          "Saudi Arabia",
    "Trinidad and Tobago":              "Trinidad & Tobago",
    "North Macedonia":                  "North Macedonia",
}


def _canonicalize(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": None, "api_key_hash": None, "matches": []}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _api_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Match type inference
# ---------------------------------------------------------------------------

def _infer_type(code: str, base_type: str, stage: str) -> str:
    """Map competition code + match stage to our internal weight key."""
    s = (stage or "").upper()
    if code == "WCQ":
        return "QUALIFIER"
    if code == "WC":
        return "WORLD_CUP"
    # EC and others: distinguish group phase from knockout
    if "GROUP" in s:
        return "GROUP"
    if any(k in s for k in ("FINAL", "QUARTER", "SEMI", "ROUND_OF", "KNOCKOUT")):
        return "KNOCKOUT"
    return base_type


# ---------------------------------------------------------------------------
# Single competition fetch
# ---------------------------------------------------------------------------

def _fetch_competition_season(
    session: requests.Session,
    code: str,
    season: int,
    base_type: str,
    known_teams: set[str],
) -> list[dict]:
    """
    Fetch FINISHED matches for one competition-season.
    Returns [] on HTTP 4xx (with a warning log) — never raises.
    """
    url = f"{config.API_BASE_URL}/competitions/{code}/matches"
    params = {"season": season, "status": "FINISHED"}

    try:
        resp = session.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        logger.warning("Network error fetching %s/%d: %s", code, season, exc)
        return []

    if resp.status_code == 429:
        logger.warning("%s/%d: rate-limited (429) — waiting 65s", code, season)
        time.sleep(65)
        try:
            resp = session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            logger.warning("Retry failed: %s", exc)
            return []

    if resp.status_code == 403:
        logger.warning(
            "%s/%d: HTTP 403 — competition not available on free tier. "
            "Upgrade your football-data.org plan or remove it from COMPETITION_CONFIG.",
            code, season,
        )
        return []

    if resp.status_code == 400:
        # Usually means the season doesn't exist for this competition
        try:
            detail = resp.json().get("message", "")
        except Exception:
            detail = resp.text[:120]
        logger.warning("%s/%d: HTTP 400 — %s", code, season, detail)
        return []

    if resp.status_code not in (200, 206):
        logger.warning("%s/%d: HTTP %d — skipping", code, season, resp.status_code)
        return []

    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("%s/%d: JSON parse error — %s", code, season, exc)
        return []

    comp_name = data.get("competition", {}).get("name", code)
    raw_matches = data.get("matches", [])
    logger.info("%s/%d (%s): %d raw matches returned", code, season, comp_name, len(raw_matches))

    matches: list[dict] = []
    for m in raw_matches:
        home_raw = (m.get("homeTeam") or {}).get("name", "")
        away_raw = (m.get("awayTeam") or {}).get("name", "")
        home = _canonicalize(home_raw)
        away = _canonicalize(away_raw)

        if home not in known_teams and away not in known_teams:
            continue

        score = m.get("score") or {}
        ft = score.get("fullTime") or {}
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            continue

        stage = m.get("stage", "REGULAR_SEASON")
        matches.append({
            "id":               m.get("id"),
            "date":             (m.get("utcDate") or "")[:10],
            "home_team":        home,
            "away_team":        away,
            "home_goals":       int(hg),
            "away_goals":       int(ag),
            "competition":      comp_name,
            "competition_code": code,
            "competition_type": _infer_type(code, base_type, stage),
            "stage":            stage,
            "season":           season,
        })

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_matches(
    date_from: str = "2022-01-01",   # kept for UI signature compat; not sent to API
    date_to: str | None = None,
    progress_cb=None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch matches for all 48 WC2026 teams via the free-tier API.

    Iterates over COMPETITION_CONFIG (code × season pairs).
    Returns the full cached match list after merging new results.

    progress_cb: optional callable(current, total, message_str).
    """
    api_key = config.API_KEY
    if not api_key:
        logger.info("No API key configured — returning cached matches only.")
        return load_cached_matches()

    cache = _load_cache()
    key_hash = _api_key_hash(api_key)

    with (Path("data") / "teams.json").open("r", encoding="utf-8") as f:
        teams_data = json.load(f)
    known_teams: set[str] = set(teams_data["teams"].keys())

    session = requests.Session()
    session.headers.update({
        "X-Auth-Token": api_key,
        "Accept":        "application/json",
    })

    # Load CSV as base layer first (large historical dataset)
    csv_base: list[dict] = []
    if (Path("data") / "results.csv").exists():
        if progress_cb:
            progress_cb(0, 1, "Lecture du CSV Kaggle comme base…")
        csv_base = load_csv_matches(date_from=date_from, known_teams=known_teams)
        logger.info("CSV base layer: %d matches", len(csv_base))

    # Build dedup index: CSV first, then existing cache, keyed by (date, home, away)
    seen_keys: dict[tuple, dict] = {}
    for m in csv_base:
        seen_keys[_dedup_key(m)] = m
    for m in cache.get("matches", []):
        k = _dedup_key(m)
        if k not in seen_keys:
            seen_keys[k] = m

    existing_ids: set = {m["id"] for m in seen_keys.values() if m.get("id")}
    new_matches: list[dict] = []

    # Build flat (code, season, base_type) list for progress tracking
    tasks = [
        (cfg["code"], season, cfg["base_type"])
        for cfg in COMPETITION_CONFIG
        for season in cfg["seasons"]
    ]
    total = len(tasks)

    for i, (code, season, base_type) in enumerate(tasks):
        if progress_cb:
            progress_cb(i, total, f"Fetching {code} {season}…")

        fetched = _fetch_competition_season(session, code, season, base_type, known_teams)
        added = 0
        for m in fetched:
            if m["id"] not in existing_ids:
                new_matches.append(m)
                existing_ids.add(m["id"])
                added += 1

        logger.info("%s/%d: %d new matches added (total new so far: %d)", code, season, added, len(new_matches))

        # Respect free-tier rate limit: 10 req/min → ≥6s between calls
        if i < total - 1:
            time.sleep(6.5)

    if progress_cb:
        progress_cb(total, total, f"Terminé — {len(new_matches)} nouveaux matchs")

    # Merge new API matches into dedup index (API data takes precedence)
    for m in new_matches:
        seen_keys[_dedup_key(m)] = m

    all_matches = sorted(seen_keys.values(), key=lambda m: m.get("date", ""))
    cache["matches"] = all_matches
    cache["last_updated"] = datetime.utcnow().isoformat()
    cache["api_key_hash"] = key_hash
    _save_cache(cache)

    logger.info(
        "Cache updated: %d total matches (%d new). Competitions tried: %s",
        len(all_matches), len(new_matches),
        ", ".join(f"{c}/{s}" for c, s, _ in tasks),
    )
    return all_matches


@_st_cache_data(ttl=3600)
def load_cached_matches() -> list[dict]:
    """Return cached matches without any API call."""
    return _load_cache().get("matches", [])


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _dedup_key(m: dict) -> tuple:
    return (m.get("date", ""), m.get("home_team", ""), m.get("away_team", ""))


def load_csv_into_cache(
    date_from: str = "2022-01-01",
    progress_cb=None,
) -> list[dict]:
    """
    Load CSV matches, merge with any existing API matches already in cache,
    deduplicate, save back to cache, and return the full list.

    This is called by the 'Charger depuis CSV Kaggle' button — no API call.
    """
    if progress_cb:
        progress_cb(0, 2, "Lecture du CSV Kaggle…")

    with (Path("data") / "teams.json").open("r", encoding="utf-8") as f:
        teams_data = json.load(f)
    known_teams: set[str] = set(teams_data["teams"].keys())

    csv_matches = load_csv_matches(date_from=date_from, known_teams=known_teams)

    if progress_cb:
        progress_cb(1, 2, f"{len(csv_matches)} matchs CSV lus — fusion avec le cache…")

    cache = _load_cache()
    existing = cache.get("matches", [])

    # Build dedup index from CSV matches first (primary source)
    seen: dict[tuple, dict] = {}
    for m in csv_matches:
        seen[_dedup_key(m)] = m

    # Layer existing API matches on top (they take precedence for data quality)
    added_from_cache = 0
    for m in existing:
        k = _dedup_key(m)
        if k not in seen:
            seen[k] = m
            added_from_cache += 1

    all_matches = list(seen.values())
    all_matches.sort(key=lambda m: m.get("date", ""))

    cache["matches"] = all_matches
    cache["last_updated"] = datetime.utcnow().isoformat()
    _save_cache(cache)

    if progress_cb:
        progress_cb(2, 2, f"Terminé — {len(all_matches)} matchs au total")

    logger.info(
        "CSV load: %d CSV matches + %d from existing cache = %d total",
        len(csv_matches), added_from_cache, len(all_matches),
    )
    return all_matches


def matches_by_team(matches: list[dict]) -> dict[str, list[dict]]:
    """Index match list by team name (each match appears under both teams)."""
    idx: dict[str, list[dict]] = {}
    for m in matches:
        for team in (m.get("home_team", ""), m.get("away_team", "")):
            if team:
                idx.setdefault(team, []).append(m)
    return idx
