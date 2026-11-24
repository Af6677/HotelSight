"""Display-currency conversion for the dashboard.

The model was trained on euro room rates and always works in euros. The data, the saved prediction history, the
exported files and the room-rate input therefore stay in EUR. Only the amounts shown on screen are converted, using a
dated rate table. Shipped rates live in assets/exchange_rates.json. The "Update rates" button can replace them with
the latest European Central Bank reference rates, saved to history/exchange_rates.json (or HOTELSIGHT_RATES_FILE).
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from .paths import ASSETS, ROOT

BASE = "EUR"
# code: (name, symbol, decimals)
CURRENCIES = {
    "EUR": ("Euro", "€", 2),
    "MYR": ("Malaysian ringgit", "RM ", 2),
    "USD": ("US dollar", "US$", 2),
    "SGD": ("Singapore dollar", "S$", 2),
    "GBP": ("British pound", "£", 2),
    "AUD": ("Australian dollar", "A$", 2),
    "CNY": ("Chinese yuan", "CN¥", 2),
    "THB": ("Thai baht", "฿", 2),
    "JPY": ("Japanese yen", "JP¥", 0),
}
API = "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=" + ",".join(c for c in CURRENCIES if c != BASE)
SHIPPED = ASSETS / "exchange_rates.json"


def user_file() -> Path:
    return Path(os.environ.get("HOTELSIGHT_RATES_FILE") or ROOT / "history" / "exchange_rates.json")


def _valid(table) -> bool:
    try:
        rates = table["rates"]
        return (table["base"] == BASE and bool(table["date"]) and
                all(float(rates[c]) > 0 for c in CURRENCIES if c != BASE))
    except (KeyError, TypeError, ValueError):
        return False


def load_rates() -> dict:
    """The saved update if it is valid, otherwise the rates shipped with the dashboard."""
    for path in (user_file(), SHIPPED):
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _valid(table):
            return table
    raise RuntimeError("No valid exchange rate table found (assets/exchange_rates.json).")


def rate(code: str, table: dict) -> float:
    return 1.0 if code == BASE else float(table["rates"][code])


def convert(value_eur, code: str, table: dict):
    """Works on a number or a pandas Series."""
    return value_eur * rate(code, table)


def fmt(value_eur: float, code: str, table: dict, digits: int | None = None) -> str:
    """Text such as 'RM 1,286' or '€95.00'. Without digits: no decimals from 1,000 up, two below it."""
    _, symbol, decimals = CURRENCIES[code]
    v = convert(value_eur, code, table)
    if digits is None:
        digits = 0 if abs(v) >= 1000 else decimals
    return f"{symbol}{v:,.{min(digits, decimals)}f}"


def column_format(code: str) -> str:
    """printf-style format for a Streamlit number column."""
    _, symbol, decimals = CURRENCIES[code]
    return f"{symbol}%.{decimals}f"


def fetch_latest(timeout: float = 8.0) -> dict:
    """Latest ECB reference rates from frankfurter.dev. Sends no user data, only the request for the rates."""
    with urllib.request.urlopen(API, timeout=timeout) as response:          # nosec - fixed https URL
        raw = json.loads(response.read().decode("utf-8"))
    table = {"base": BASE, "date": raw["date"], "source": "European Central Bank reference rates via frankfurter.dev",
             "rates": {BASE: 1.0, **{c: float(v) for c, v in raw["rates"].items()}}}
    if not _valid(table):
        raise ValueError("The rate service returned an unexpected answer.")
    return table


def refresh() -> dict:
    """Fetch the latest rates and keep them for the next start. Raises if the service cannot be reached."""
    table = fetch_latest()
    path = user_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table
