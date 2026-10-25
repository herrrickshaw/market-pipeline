"""yf_intl_pit_fundamentals: real announcement dates replace the fabricated
fy_end+90d `filed` column. yfinance is replaced by an in-memory fake; nothing
touches the network, the cache or Postgres.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

import yf_intl_pit_fundamentals as yp

NAN = np.nan


def earnings(rows, tz="Asia/Tokyo"):
    """rows: [(date, eps_estimate, reported_eps, surprise_pct)] newest-first like yfinance."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, *_ in rows]).tz_localize(tz)
    return pd.DataFrame({"EPS Estimate": [r[1] for r in rows],
                         "Reported EPS": [r[2] for r in rows],
                         "Surprise(%)": [r[3] for r in rows]}, index=idx)


def income(periods, revenue, net_income):
    cols = [pd.Timestamp(p) for p in periods]
    return pd.DataFrame([revenue, net_income], index=["Total Revenue", "Net Income"],
                        columns=cols)


@pytest.fixture
def fake_yf(monkeypatch):
    book = {}

    class Ticker:
        def __init__(self, sym):
            self.earnings_dates = book[sym].get("ed")
            self.quarterly_income_stmt = book[sym].get("qi")

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    return book


class TestFetchOne:
    def test_each_announcement_pairs_with_latest_period_end_on_or_before_it(self, fake_yf):
        fake_yf["7203.T"] = {
            "ed": earnings([("2026-02-05", 70.0, NAN, NAN),          # upcoming
                            ("2025-11-05", 60.0, 62.0, 3.3),
                            ("2025-08-06", 60.0, 64.56, 7.6)]),
            "qi": income(["2025-09-30", "2025-06-30"], [900, 800], [90, 80])}
        out, nxt = yp._fetch_one("JP", "7203.T")
        out = out.set_index("period_end")
        jun, sep = pd.Timestamp("2025-06-30").date(), pd.Timestamp("2025-09-30").date()
        assert out.loc[jun, "filing_date"] == pd.Timestamp("2025-08-06")
        assert out.loc[jun, "eps_reported"] == 64.56
        assert (out.loc[jun, "revenue"], out.loc[jun, "net_income"]) == (800.0, 80.0)
        assert out.loc[sep, "filing_date"] == pd.Timestamp("2025-11-05")
        assert out.loc[sep, "revenue"] == 900.0

    def test_upcoming_report_is_next_earnings_not_history(self, fake_yf):
        fake_yf["7203.T"] = {
            "ed": earnings([("2026-02-05", 70.0, NAN, NAN), ("2025-11-05", 60.0, 62.0, 3.3)]),
            "qi": income(["2025-09-30"], [900], [90])}
        out, nxt = yp._fetch_one("JP", "7203.T")
        assert len(out) == 1
        assert nxt == {"symbol": "7203.T",
                       "next_earnings_date": pd.Timestamp("2026-02-05").date(),
                       "eps_estimate": 70.0}

    def test_restated_quarter_keeps_the_earliest_announcement(self, fake_yf):
        fake_yf["000001.SZ"] = {
            "ed": earnings([("2025-10-24", NAN, 0.6, NAN), ("2025-08-22", NAN, 0.62, NAN)],
                           tz="Asia/Shanghai"),
            "qi": income(["2025-06-30"], [100], [10])}
        out, _ = yp._fetch_one("CN", "000001.SZ")
        assert len(out) == 1
        assert out["filing_date"].iloc[0] == pd.Timestamp("2025-08-22")
        assert out["eps_reported"].iloc[0] == 0.62

    def test_filing_dates_are_timezone_naive(self, fake_yf):
        fake_yf["005930.KS"] = {
            "ed": earnings([("2025-07-31", NAN, 1.0, NAN)], tz="Asia/Seoul"),
            "qi": income(["2025-06-30"], [1], [1])}
        out, _ = yp._fetch_one("KR", "005930.KS")
        assert out["filing_date"].iloc[0].tzinfo is None

    def test_no_earnings_dates_returns_empty(self, fake_yf):
        fake_yf["X.T"] = {"ed": pd.DataFrame(), "qi": None}
        out, nxt = yp._fetch_one("JP", "X.T")
        assert out.empty and nxt is None


def local_rows(*fy_ends, ticker="BHP.AX"):
    return pd.DataFrame([{"ticker": ticker, "fy_end": pd.Timestamp(f), "revenue": 10.0 * i,
                          "net_income": float(i), "shares": 5.0, "equity": 3.0}
                         for i, f in enumerate(fy_ends, 1)])


class TestFetchOneLocal:
    def test_each_fy_end_takes_the_next_announcement_on_or_after_it(self, fake_yf):
        fake_yf["BHP.AX"] = {"ed": earnings([("2024-08-20", NAN, 2.0, NAN),
                                              ("2024-02-20", NAN, 1.1, NAN),
                                              ("2023-08-22", NAN, 1.9, NAN)],
                                             tz="Australia/Sydney")}
        out, _ = yp._fetch_one_local("AU", "BHP.AX",
                                     local_rows("2023-06-30", "2024-06-30"))
        assert out["filing_date"].tolist() == [pd.Timestamp("2023-08-22"),
                                               pd.Timestamp("2024-08-20")]
        assert out["eps_reported"].tolist() == [1.9, 2.0]
        assert out["revenue"].tolist() == [10.0, 20.0]
        assert out["shares_out"].tolist() == [5.0, 5.0]   # local `shares` renamed

    def test_an_announcement_is_never_claimed_twice(self, fake_yf):
        fake_yf["BHP.AX"] = {"ed": earnings([("2024-08-06", NAN, 2.0, NAN)])}
        out, _ = yp._fetch_one_local("AU", "BHP.AX",
                                     local_rows("2024-03-31", "2024-06-30"))
        assert len(out) == 1
        assert out["period_end"].iloc[0] == pd.Timestamp("2024-03-31").date()

    def test_announcement_more_than_240_days_later_is_not_matched(self, fake_yf):
        fake_yf["BHP.AX"] = {"ed": earnings([("2025-03-01", NAN, 2.0, NAN)])}
        out, _ = yp._fetch_one_local("AU", "BHP.AX", local_rows("2024-06-30"))
        assert out.empty

    def test_filing_date_is_never_before_period_end(self, fake_yf):
        fake_yf["BHP.AX"] = {"ed": earnings([("2024-08-20", NAN, 2.0, NAN),
                                              ("2024-06-01", NAN, 1.0, NAN)])}
        out, _ = yp._fetch_one_local("AU", "BHP.AX", local_rows("2024-06-30"))
        assert (pd.to_datetime(out["filing_date"])
                >= pd.to_datetime(out["period_end"])).all()

    def test_symbol_absent_from_local_file_makes_no_yfinance_call(self, fake_yf):
        out, nxt = yp._fetch_one_local("AU", "NOPE.AX", local_rows("2024-06-30"))
        assert out.empty and nxt is None          # KeyError in the fake if it had called
