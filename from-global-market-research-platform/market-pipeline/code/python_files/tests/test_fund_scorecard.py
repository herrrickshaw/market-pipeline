"""fund_scorecard.py -- daily book curve, month-end resample, performance stats, peer query."""
import math

import numpy as np
import pandas as pd
import pytest

import fund_scorecard as FS


def events(rows):
    return pd.DataFrame(rows, columns=["evict_date", "days", "exit_ret_pct"])


# ── book_daily ────────────────────────────────────────────────────────────────
def test_single_trade_compounds_to_exact_total_return():
    out = FS.book_daily(events([("2024-01-01", 5, 10.0)]), cost_bps=0)
    assert list(out.columns) == ["date", "ret", "n_positions"]
    assert len(out) == 5 and (out["n_positions"] == 1).all()
    assert list(out["date"]) == list(pd.date_range("2024-01-01", periods=5, freq="D"))
    assert (1 + out["ret"]).prod() == pytest.approx(1.10)
    assert out["ret"].nunique() == 1                       # constant geometric daily rate


def test_cost_is_charged_once_off_the_total_return():
    out = FS.book_daily(events([("2024-01-01", 4, 10.0)]), cost_bps=50)
    assert (1 + out["ret"]).prod() == pytest.approx(1.10 * (1 - 0.005))
    zero = FS.book_daily(events([("2024-01-01", 4, 0.0)]), cost_bps=100)
    assert (1 + zero["ret"]).prod() == pytest.approx(0.99)


def test_zero_or_negative_days_are_clamped_to_one():
    for d in (0, -3):
        out = FS.book_daily(events([("2024-01-01", d, 5.0)]), cost_bps=0)
        assert len(out) == 1 and out["ret"].iloc[0] == pytest.approx(0.05)


def test_overlap_is_equal_weighted_by_open_positions():
    ev = events([("2024-01-01", 2, 21.0),          # g = 1.21**0.5 - 1 = 0.10
                 ("2024-01-02", 1, 0.0)])          # g = 0 on day 2 only
    out = FS.book_daily(ev, cost_bps=0)
    assert list(out["n_positions"]) == [1, 2]
    assert out["ret"].iloc[0] == pytest.approx(0.10)
    assert out["ret"].iloc[1] == pytest.approx((0.10 + 0.0) / 2)             # mean, not sum
    ev3 = events([("2024-01-01", 1, 21.0), ("2024-01-01", 1, 0.0), ("2024-01-01", 1, -19.0)])
    assert FS.book_daily(ev3, 0)["ret"].iloc[0] == pytest.approx((0.21 + 0.0 - 0.19) / 3)


def test_gap_days_are_cash_at_zero_return_and_zero_positions():
    ev = events([("2024-01-01", 1, 10.0), ("2024-01-05", 1, 10.0)])
    out = FS.book_daily(ev, cost_bps=0)
    assert len(out) == 5
    assert list(out["n_positions"]) == [1, 0, 0, 0, 1]
    assert list(out["ret"].iloc[1:4]) == [0.0, 0.0, 0.0]


def test_empty_events_gives_empty_frame_with_columns():
    out = FS.book_daily(events([]), cost_bps=50)
    assert out.empty and list(out.columns) == ["date", "ret", "n_positions"]


def test_accepts_timestamp_and_string_dates():
    a = FS.book_daily(events([(pd.Timestamp("2024-03-01"), 3, 6.0)]), 0)
    b = FS.book_daily(events([("2024-03-01", 3, 6.0)]), 0)
    assert a.equals(b)


# ── to_monthly ────────────────────────────────────────────────────────────────
def daily_const(start, end, r, n=1):
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": idx, "ret": r, "n_positions": n})


def test_to_monthly_nav_and_returns():
    d = daily_const("2024-01-15", "2024-03-10", 0.001)
    m = FS.to_monthly(d)
    assert [str(x) for x in m["month"]] == ["2024-01", "2024-02", "2024-03"]
    jan, feb = 1.001 ** 17, 1.001 ** (17 + 29)
    assert m["nav"].iloc[0] == pytest.approx(jan)
    assert m["nav"].iloc[1] == pytest.approx(feb)
    assert m["nav"].iloc[2] == pytest.approx(1.001 ** (17 + 29 + 10))
    assert m["ret"].iloc[0] == pytest.approx(jan - 1)                 # first month vs NAV 1.0
    assert m["ret"].iloc[1] == pytest.approx(feb / jan - 1)
    assert list(m.columns) == ["month", "ret", "nav", "n_positions"]


def test_to_monthly_position_count_is_monthly_mean():
    d = daily_const("2024-01-01", "2024-01-31", 0.0)
    d["n_positions"] = [0] * 15 + [4] * 16
    assert FS.to_monthly(d)["n_positions"].iloc[0] == pytest.approx(64 / 31)


def test_to_monthly_from_book_daily_total_return_is_preserved():
    ev = events([("2024-01-20", 30, 12.0), ("2024-02-05", 40, -6.0)])
    m = FS.to_monthly(FS.book_daily(ev, 0))
    assert m["nav"].iloc[-1] > 0
    assert m["nav"].iloc[0] == pytest.approx(1.12 ** (12 / 30), rel=1e-9)   # Jan 20..31, one position


# ── perf ──────────────────────────────────────────────────────────────────────
def test_perf_values():
    nav = pd.Series([100.0, 110.0, 99.0, 120.0, 130.0] + [130.0] * 8)    # 13 points -> 12 months
    p = FS.perf(nav)
    r = nav.pct_change().dropna()
    assert p["months"] == 12 and p["total"] == pytest.approx(0.30)
    assert p["cagr"] == pytest.approx(1.30 ** (1 / 1.0) - 1)
    assert p["vol"] == pytest.approx(r.std() * math.sqrt(12))
    assert p["maxdd"] == pytest.approx(99 / 110 - 1)


def test_perf_annualises_by_months_over_twelve():
    nav = pd.Series(1.01 ** np.arange(25))                            # 24 monthly steps, 2 years
    assert FS.perf(nav)["cagr"] == pytest.approx(1.01 ** 12 - 1)
    assert FS.perf(nav)["maxdd"] == 0.0


def test_perf_short_or_empty_series():
    assert FS.perf(pd.Series([1.0])) == {}
    assert FS.perf(pd.Series([], dtype=float)) == {}
    assert FS.perf(pd.Series([np.nan, 5.0, np.nan])) == {}
    two = FS.perf(pd.Series([1.0, np.nan, 1.5]))                      # NaN dropped first
    assert two["months"] == 1 and two["total"] == pytest.approx(0.5)
    assert two["cagr"] == pytest.approx(1.5 ** 12 - 1)


# ── peer_monthly ──────────────────────────────────────────────────────────────
class FakeCon:
    def __init__(self, df):
        self.df_, self.queries = df, []

    def execute(self, q):
        self.queries.append(q)
        return self

    def df(self):
        return self.df_


def raw_navs():
    return pd.DataFrame({
        "scheme_code": [1, 1, 1, 2, 2],
        "nav_date": ["2024-01-05", "2024-01-30", "2024-02-10", "2024-01-31", "2024-02-28"],
        "nav": [10.0, 11.0, 12.0, 20.0, 22.0],
        "category": ["Equity Scheme - Large Cap Fund", "Equity Scheme - Large Cap Fund",
                     "Equity Scheme - Large Cap Fund", "Equity Schemes - Large Cap Fund",
                     "Equity Schemes - Large Cap Fund"],
        "scheme_name": ["A", "A", "A", "B", "B"],
    })


def test_peer_monthly_takes_month_end_nav_and_normalises_category():
    con = FakeCon(raw_navs())
    out = FS.peer_monthly(con, None, "2024-01-01", "2024-02-29")
    got = out.set_index(["scheme_code", "month"])
    assert got.loc[(1, pd.Period("2024-01")), "nav"] == 11.0          # last NAV of the month
    assert got.loc[(1, pd.Period("2024-02")), "nav"] == 12.0
    assert got.loc[(2, pd.Period("2024-01")), "nav"] == 20.0
    # both AMFI spellings land in one peer set
    assert set(out["cat_norm"]) == {"Large Cap Fund"}
    assert len(out) == 4


def test_peer_monthly_sorts_by_date_before_taking_last():
    df = raw_navs().iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)      # shuffled: 01-30 before 01-05
    out = FS.peer_monthly(FakeCon(df), None, "a", "b")
    assert out.set_index(["scheme_code", "month"]).loc[(1, pd.Period("2024-01")), "nav"] == 11.0


def test_peer_monthly_category_uses_parenthesised_label():
    df = raw_navs().head(1).copy()
    df["category"] = "Equity Scheme - Sectoral/Thematic (Banking)"
    out = FS.peer_monthly(FakeCon(df), None, "a", "b")
    assert out["cat_norm"].iloc[0] == "Banking"


def test_peer_monthly_query_contents():
    con = FakeCon(pd.DataFrame())
    out = FS.peer_monthly(con, ["Small", "Mid"], "2020-01-01", "2024-12-31")
    assert out.empty
    q = con.queries[0]
    assert "s.plan = 'direct' AND s.option = 'growth'" in q
    assert "s.category NOT ILIKE '%Hybrid%'" in q
    assert "(s.category ILIKE '%Small%' OR s.category ILIKE '%Mid%')" in q
    assert "BETWEEN DATE '2020-01-01' AND DATE '2024-12-31'" in q and "n.nav > 0" in q
    FS.peer_monthly(con, None, "2020-01-01", "2024-12-31")
    assert "ILIKE '%Small%'" not in con.queries[1]
    FS.peer_monthly(con, [], "2020-01-01", "2024-12-31")
    assert con.queries[2].count("ILIKE") == con.queries[1].count("ILIKE")


def test_defaults():
    assert FS.DEFAULT_COST_BPS == 50.0 and FS.MONTHS_PER_YEAR == 12
