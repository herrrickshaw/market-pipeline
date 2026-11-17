"""signal_tracker: THE ONE RULE is that signal_date is never backfilled —
record() only ever stamps today, re-runs are idempotent, and rows already in
the ledger are never re-stamped or re-priced. Harvesters, the ledger, the
watchlist and the price panels are all replaced with tmp fixtures.
"""
import datetime as real_dt

import numpy as np
import pandas as pd
import pytest

import signal_tracker as st


def _row(sym, mkt="IN", flt="technical", price=100.0, detail="grade A", score=80.0):
    return {"symbol": sym, "market": mkt, "filter": flt, "detail": detail,
            "score": score, "price_at_signal": price, "source": "fixture"}


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LEDGER", tmp_path / "cache_seed" / "signal_ledger.parquet")
    monkeypatch.setattr(st, "WATCHLIST", tmp_path / "watchlist.csv")   # absent
    monkeypatch.setattr(st, "PANELS", {})
    state = {"today": real_dt.date(2026, 10, 20), "tech": [], "fund": []}

    class FakeDate(real_dt.date):
        @classmethod
        def today(cls):
            d = state["today"]
            return cls(d.year, d.month, d.day)

    monkeypatch.setattr(st, "date", FakeDate)
    monkeypatch.setattr(st, "harvest_technical", lambda: pd.DataFrame(state["tech"]))
    monkeypatch.setattr(st, "harvest_fundamental", lambda: pd.DataFrame(state["fund"]))
    return state


def _ledger():
    return pd.read_parquet(st.LEDGER)


def test_record_stamps_today_even_if_a_harvester_supplies_an_old_date(ledger_env):
    r = _row("TCS")
    r["signal_date"] = pd.Timestamp("2025-01-02")        # a would-be backfill
    ledger_env["tech"] = [r]
    st.record()
    assert _ledger()["signal_date"].tolist() == [pd.Timestamp("2026-10-20")]


def test_rerun_same_day_is_idempotent_and_keeps_first_written_price(ledger_env):
    ledger_env["tech"] = [_row("TCS", price=100.0)]
    st.record()
    ledger_env["tech"] = [_row("TCS", price=125.0)]
    st.record()
    led = _ledger()
    assert len(led) == 1
    assert led["price_at_signal"].iloc[0] == 100.0


def test_next_day_appends_without_touching_earlier_rows(ledger_env):
    ledger_env["tech"] = [_row("TCS", price=100.0)]
    st.record()
    first = _ledger().copy()
    ledger_env["today"] = real_dt.date(2026, 10, 21)
    ledger_env["tech"] = [_row("TCS", price=111.0)]
    st.record()
    led = _ledger().sort_values("signal_date").reset_index(drop=True)
    assert len(led) == 2
    pd.testing.assert_frame_equal(led.iloc[[0]].reset_index(drop=True),
                                  first.reset_index(drop=True), check_dtype=False)
    assert led["signal_date"].iloc[1] == pd.Timestamp("2026-10-21")


def test_no_passes_writes_nothing(ledger_env):
    assert st.record() == 0
    assert not st.LEDGER.exists()


def test_same_symbol_in_two_markets_same_day_are_both_recorded(ledger_env):
    # INFY trades on NSE and NYSE under the same bare symbol. They are two
    # different instruments; the ledger must not collapse them.
    ledger_env["tech"] = [_row("TCS")]
    st.record()                                           # ledger now non-empty
    ledger_env["today"] = real_dt.date(2026, 10, 21)
    ledger_env["tech"] = [_row("INFY", "IN", price=1500.0), _row("INFY", "US", price=18.0)]
    st.record()
    today = _ledger()[lambda d: d["signal_date"] == pd.Timestamp("2026-10-21")]
    assert sorted(today["market"]) == ["IN", "US"]


def test_cap_technical_keeps_grade_a_top_five_per_market_and_all_fundamentals():
    tech = [_row(f"IN{i}", "IN", score=i) for i in range(8)] + \
           [_row(f"US{i}", "US", score=i) for i in range(3)] + \
           [_row("INB", "IN", detail="grade B", score=999)]
    fund = [_row("F1", "IN", flt="piotroski", detail="F=8", score=8)]
    out = st._cap_technical(pd.DataFrame(tech + fund))
    t = out[out["filter"] == "technical"]
    assert sorted(t[t.market == "IN"]["symbol"]) == ["IN3", "IN4", "IN5", "IN6", "IN7"]
    assert sorted(t[t.market == "US"]["symbol"]) == ["US0", "US1", "US2"]
    assert "INB" not in set(out["symbol"])
    assert "F1" in set(out["symbol"])


def test_sync_watchlist_never_touches_existing_rows_and_keeps_extra_columns(tmp_path, monkeypatch):
    wl = tmp_path / "watchlist.csv"
    pd.DataFrame([{"symbol": "TCS", "market": "IN", "status": "held",
                   "note": "core", "entry_date": "2024-01-01", "entry_price": 3500.0}]
                 ).to_csv(wl, index=False)
    monkeypatch.setattr(st, "WATCHLIST", wl)
    new = pd.DataFrame([_row("TCS", price=4000.0), _row("NEW", price=10.0),
                        _row("DR", flt="debt_reduction", detail="F=6")])
    new["signal_date"] = pd.Timestamp("2026-10-20")
    assert st.sync_watchlist(new) == 1
    out = pd.read_csv(wl).set_index("symbol")
    assert out.loc["TCS", "status"] == "held" and out.loc["TCS", "entry_price"] == 3500.0
    assert out.loc["NEW", "status"] == "signal"
    assert out.loc["NEW", "entry_date"] == "2026-10-20"
    assert out.loc["NEW", "entry_price"] == 10.0
    assert "DR" not in out.index                 # not a watchlist-worthy filter
