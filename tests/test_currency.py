"""Display-currency conversion: rates, formatting, the saved-rate fallback and the dashboard pages."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hotelsight import currency

APP = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture(autouse=True)
def isolated_rates_file(tmp_path, monkeypatch):
    """Never read or write the real history/exchange_rates.json while testing."""
    monkeypatch.setenv("HOTELSIGHT_RATES_FILE", str(tmp_path / "rates.json"))


def test_shipped_rates_cover_every_currency():
    table = currency.load_rates()
    assert table["base"] == "EUR" and table["date"]
    assert all(table["rates"][c] > 0 for c in currency.CURRENCIES)


def test_euro_is_unchanged_and_ringgit_uses_the_table():
    table = currency.load_rates()
    assert currency.convert(100.0, "EUR", table) == 100.0
    assert currency.convert(100.0, "MYR", table) == pytest.approx(100 * table["rates"]["MYR"])
    assert currency.fmt(95, "EUR", table) == "€95.00"
    assert currency.fmt(95, "MYR", table, 2) == f"RM {95 * table['rates']['MYR']:,.2f}"


def test_formatting_has_no_decimals_from_1000_and_yen_never_has_any():
    table = currency.load_rates()
    assert currency.fmt(1234.6, "EUR", table) == "€1,235"
    assert currency.fmt(999.5, "EUR", table) == "€999.50"
    assert "." not in currency.fmt(5, "JPY", table, 2)
    assert currency.column_format("MYR") == "RM %.2f" and currency.column_format("JPY") == "JP¥%.0f"


def test_a_broken_saved_rate_file_falls_back_to_the_shipped_rates(tmp_path):
    (tmp_path / "rates.json").write_text("not json", encoding="utf-8")
    assert currency.load_rates()["date"] == json.loads(currency.SHIPPED.read_text(encoding="utf-8"))["date"]
    (tmp_path / "rates.json").write_text(json.dumps({"base": "EUR", "date": "2030-01-01", "rates": {"MYR": -1}}),
                                         encoding="utf-8")
    assert currency.load_rates()["date"] != "2030-01-01"


def test_refresh_saves_the_new_rates_and_they_are_used_next_time(monkeypatch):
    new = {"base": "EUR", "date": "2030-01-01", "source": "test",
           "rates": {"EUR": 1.0, **{c: 2.0 for c in currency.CURRENCIES if c != "EUR"}}}
    monkeypatch.setattr(currency, "fetch_latest", lambda timeout=8.0: new)
    assert currency.refresh()["date"] == "2030-01-01"
    assert currency.load_rates()["rates"]["MYR"] == 2.0


def test_refresh_fails_cleanly_when_offline(monkeypatch):
    def offline(timeout=8.0):
        raise OSError("no internet")
    monkeypatch.setattr(currency, "fetch_latest", offline)
    with pytest.raises(OSError):
        currency.refresh()
    assert not currency.user_file().exists()               # nothing half-written


def test_overview_amounts_follow_the_chosen_currency(tmp_path, monkeypatch):
    from hotelsight import history
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    row = {"reference": "A", "hotel": "City Hotel", "arrival": "2099-01-01", "probability": 0.9,
           "top_raising": "Average daily rate: €100.00", "record": {"adr": 100.0, "stays_in_week_nights": 2}}
    history.save([row], "single", "Single prediction", 0.5, "abc")
    rate = currency.load_rates()["rates"]["MYR"]

    eur = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert next(m for m in eur.metric if m.label == "Revenue flagged").value == "€200.00"

    myr = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180)
    myr.session_state["currency"] = "MYR"
    myr.run()
    assert not myr.exception, [e.value for e in myr.exception]
    assert next(m for m in myr.metric if m.label == "Revenue flagged").value == f"RM {200 * rate:,.2f}"
    assert myr.dataframe[0].value["Stay value"].iloc[0] == pytest.approx(200 * rate)   # converted copy for display


def test_reservation_result_follows_the_chosen_currency(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    rate = currency.load_rates()["rates"]["USD"]
    at = AppTest.from_file(str(APP / "views/predict.py"), default_timeout=180)
    at.session_state["currency"] = "USD"
    at.run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(str(getattr(e, "value", "")) for e in at.get("html"))
    assert f"US${95 * rate:,.2f}" in html                    # the default booking has a room rate of EUR 95
    assert at.number_input(key="k_adr").value == 95.0        # the input itself stays in euros


def test_batch_revenue_follows_the_chosen_currency_and_downloads_stay_in_euros(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    rate = currency.load_rates()["rates"]["MYR"]
    runs = {}
    for code in ("EUR", "MYR"):
        at = AppTest.from_file(str(APP / "views/batch.py"), default_timeout=300)
        at.session_state["currency"] = code
        at.run()
        at.button(key="use_sample").click().run()
        at.button(key="run_batch").click().run()
        assert not at.exception, [e.value for e in at.exception]
        runs[code + "_df"] = at.session_state["batch_result"]["results"]
        outlook = AppTest.from_file(str(APP / "views/insights.py"), default_timeout=180)   # revenue lives here now
        outlook.session_state["currency"] = code
        outlook.session_state["ins_batch"] = at.session_state["batch_result"]["batch_id"]
        outlook.run()
        assert not outlook.exception, [e.value for e in outlook.exception]
        runs[code] = next(m for m in outlook.metric if m.label == "Revenue flagged").value
    eur = float(runs["EUR"].replace("€", "").replace(",", ""))
    myr = float(runs["MYR"].replace("RM", "").replace(",", ""))
    assert myr == pytest.approx(eur * rate, rel=2e-3)
    assert runs["EUR_df"]["Stay value (EUR)"].equals(runs["MYR_df"]["Stay value (EUR)"])   # the data is never converted
