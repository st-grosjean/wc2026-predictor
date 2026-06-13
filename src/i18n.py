"""Internationalisation helpers for WC 2026 Predictor."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
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

# ---------------------------------------------------------------------------
# Timezone support (fixed offsets for WC 2026 tournament period)
# ---------------------------------------------------------------------------

TZ_OPTIONS: list[tuple[str, str]] = [
    ("Europe/Paris",                    "🇫🇷 Paris (CEST) UTC+2"),
    ("Europe/London",                   "🇬🇧 Londres (BST) UTC+1"),
    ("America/Sao_Paulo",               "🇧🇷 Brasília UTC-3"),
    ("America/Argentina/Buenos_Aires",  "🇦🇷 Buenos Aires UTC-3"),
    ("America/New_York",                "🇺🇸 New York (EDT) UTC-4"),
    ("America/Chicago",                 "🇺🇸 Chicago (CDT) UTC-5"),
    ("America/Los_Angeles",             "🇺🇸 Los Angeles (PDT) UTC-7"),
    ("America/Mexico_City",             "🇲🇽 Mexico City (CST) UTC-6"),
    ("Asia/Tokyo",                      "🇯🇵 Tokyo (JST) UTC+9"),
    ("Asia/Seoul",                      "🇰🇷 Séoul (KST) UTC+9"),
    ("Europe/Berlin",                   "🇩🇪 Berlin (CEST) UTC+2"),
    ("Europe/Lisbon",                   "🇵🇹 Lisbonne (WEST) UTC+1"),
    ("UTC",                             "🌐 UTC (UTC+0)"),
]

# Fixed summer offsets for the June–July 2026 tournament window.
_TZ_OFFSETS: dict[str, tuple[int, str]] = {
    "Europe/Paris":                    (2,  "CEST"),
    "Europe/London":                   (1,  "BST"),
    "America/Sao_Paulo":               (-3, "BRT"),
    "America/Argentina/Buenos_Aires":  (-3, "ART"),
    "America/New_York":                (-4, "EDT"),
    "America/Chicago":                 (-5, "CDT"),
    "America/Los_Angeles":             (-7, "PDT"),
    "America/Mexico_City":             (-6, "CST"),
    "Asia/Tokyo":                      (9,  "JST"),
    "Asia/Seoul":                      (9,  "KST"),
    "Europe/Berlin":                   (2,  "CEST"),
    "Europe/Lisbon":                   (1,  "WEST"),
    "UTC":                             (0,  "UTC"),
}

_DAY_CHAR: dict[str, str] = {
    "fr": "j", "en": "d", "pt": "d", "es": "d",
    "de": "T", "ja": "日", "ko": "일",
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


def fmt_time_local(
    date_str: str, time_utc: str, tz_name: str = "UTC", lang: str = "fr"
) -> str:
    """Convert a UTC date+time pair to a local time string with TZ abbreviation.

    Returns e.g. "21:00 CEST", "01:00 CEST +1j", "23:00 BST -1d".
    Falls back to "<time>Z" on any parse error.
    """
    offset_h, abbr = _TZ_OFFSETS.get(tz_name, (0, "UTC"))
    day_char = _DAY_CHAR.get(lang, "d")
    try:
        dt_utc   = datetime.strptime(f"{date_str} {time_utc}", "%Y-%m-%d %H:%M")
        dt_local = dt_utc + timedelta(hours=offset_h)
        time_str = dt_local.strftime("%H:%M")
        day_diff = (dt_local.date() - dt_utc.date()).days
        if day_diff == 0:
            return f"{time_str} {abbr}"
        sign = "+" if day_diff > 0 else "-"
        return f"{time_str} {abbr} {sign}{abs(day_diff)}{day_char}"
    except Exception:
        return f"{time_utc}Z"
