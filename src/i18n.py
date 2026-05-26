"""Internationalisation helpers for WC 2026 Predictor."""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

SUPPORTED_LANGS: dict[str, str] = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "pt": "🇧🇷 Português",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
}

CJK_FONTS: dict[str, str] = {
    "ja": (
        "<style>@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap');"
        " body, .stApp { font-family: 'Noto Sans JP', sans-serif !important; }</style>"
    ),
    "ko": (
        "<style>@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');"
        " body, .stApp { font-family: 'Noto Sans KR', sans-serif !important; }</style>"
    ),
}

_DATE_FORMATS: dict[str, str] = {
    "fr": "%d/%m/%Y",
    "de": "%d.%m.%Y",
    "pt": "%d/%m/%Y",
    "es": "%d/%m/%Y",
    "en": "%m/%d/%Y",
    "ja": "%Y/%m/%d",
    "ko": "%Y/%m/%d",
}

_DECIMAL_SEP: dict[str, str] = {
    "fr": ",", "de": ",", "pt": ",", "es": ",",
    "en": ".", "ja": ".", "ko": ".",
}


@lru_cache(maxsize=1)
def _load_i18n() -> dict:
    with (Path("data") / "i18n.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_team_names() -> dict:
    p = Path("data") / "team_names.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def t(key: str, lang: str = "fr", **kwargs) -> str:
    """Look up a translation key; falls back to French, then to the key itself."""
    strings = _load_i18n()
    val = strings.get(lang, {}).get(key) or strings.get("fr", {}).get(key) or key
    if kwargs:
        try:
            val = val.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return val


def tn(team: str, lang: str = "fr") -> str:
    """Return the translated team name, falling back to the English name."""
    if lang == "en":
        return team
    names = _load_team_names()
    return names.get(team, {}).get(lang, team)


def fmt_pct(val: float, lang: str = "fr", decimals: int = 1) -> str:
    """Format a proportion (0–1) as a percentage string with locale decimal sep."""
    sep = _DECIMAL_SEP.get(lang, ".")
    formatted = f"{val * 100:.{decimals}f}".replace(".", sep)
    return f"{formatted}%"


def fmt_num(val: float | int, lang: str = "fr", decimals: int = 1) -> str:
    """Format a number with locale-aware decimal separator."""
    sep = _DECIMAL_SEP.get(lang, ".")
    return f"{val:.{decimals}f}".replace(".", sep)


def fmt_date(date_str: str, lang: str = "fr") -> str:
    """Convert YYYY-MM-DD to locale-formatted date string."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return d.strftime(_DATE_FORMATS.get(lang, "%d/%m/%Y"))
    except ValueError:
        return date_str
