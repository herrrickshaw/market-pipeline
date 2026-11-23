"""Offline tests for scan_price_reconcile (session windows, quorum, stale-scan gate)."""

import datetime as real_dt
import subprocess
import sys
import types
from datetime import date

import pandas as pd
import pytest

import scan_price_reconcile as sr


def _freeze(monkeypatch, y, mo, d, hh, mm=0):
    now = real_dt.datetime(y, mo, d, hh, mm)

    class F(real_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(real_dt, "datetime", F)  # market_open_now imports it inside the function


# ---------------------------------------------------------------- market_open_now
@pytest.mark.parametrize(
    "mkt,hh,mm,expected",
    [("IN", 9, 14, False), ("IN", 9, 15, True), ("IN", 15, 30, True), ("IN", 15, 31, False),
     ("JP", 5, 29, False), ("JP", 5, 30, True), ("JP", 11, 30, True), ("JP", 11, 31, False),
     ("KR", 12, 0, True), ("KR", 12, 1, False), ("EU", 12, 29, False), ("EU", 12, 30, True),
     ("EU", 21, 0, True), ("EU", 21, 1, False), ("US", 19, 0, True), ("US", 18, 59, False),
     ("US", 23, 59, True), ("US", 1, 30, True), ("US", 1, 31, False), ("US", 3, 0, False),
     ("XX", 12, 0, False)],
)
def test_market_open_now_weekday_windows(monkeypatch, mkt, hh, mm, expected):
    _freeze(monkeypatch, 2026, 7, 29, hh, mm)  # a Wednesday
    assert sr.market_open_now(mkt) is expected


def test_market_open_now_weekend_is_closed(monkeypatch):
    _freeze(monkeypatch, 2026, 7, 25, 10)  # Saturday
    assert sr.market_open_now("IN") is False
    _freeze(monkeypatch, 2026, 7, 26, 20)  # Sunday
    assert sr.market_open_now("US") is False


@pytest.mark.parametrize("n,q", [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 3), (8, 4), (9, 5), (12, 6)])
def test_quorum(n, q):
    assert sr._quorum(n) == q


# ---------------------------------------------------------------- scan_age_days
def test_scan_age_days(monkeypatch):
    today = date.today().strftime("%Y%m%d")
    assert sr.scan_age_days(f"x/indian_full_scan_{today}_0300.xlsx") == 0
    assert sr.scan_age_days("x/indian_full_scan_20200101_0300.xlsx") == (date.today() - date(2020, 1, 1)).days
    assert sr.scan_age_days("x/indian_full_scan_20269999_0300.xlsx") is None
    assert sr.scan_age_days("x/other.xlsx") is None
    assert sr.scan_age_days("x/scan_20200101.xlsx") is None  # needs the _HHMM tail


# ---------------------------------------------------------------- warehouse_prev_close / fresh_quotes
def test_warehouse_prev_close_parses_csv_and_falls_back(monkeypatch):
    calls = []

    def run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="symbol,ltp\nTCS,100.5\nINFY,50\n")

    monkeypatch.setattr(sr.subprocess, "run", run)
    wh = sr.warehouse_prev_close("IN")
    assert wh["symbol"].tolist() == ["TCS", "INFY"] and wh["ltp"].tolist() == [100.5, 50]
    assert calls[0][:3] == ["psql", "-d", "market_data"] and "market='IN'" in calls[0][-1]
    monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="x"))
    assert sr.warehouse_prev_close("IN").empty and list(sr.warehouse_prev_close("IN").columns) == ["symbol", "ltp"]
    monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="  "))
    assert sr.warehouse_prev_close("US").empty

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("psql", 20)

    monkeypatch.setattr(sr.subprocess, "run", boom)
    assert sr.warehouse_prev_close("US").empty


def _yf(monkeypatch, data=None, exc=None):
    def download(tickers, **kw):
        if exc:
            raise exc
        return data

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))


def test_fresh_quotes_multi_single_and_missing(monkeypatch):
    closes = pd.DataFrame({"A": [1.0, 2.0], "B": [None, None], "C": [5.0, None]})
    _yf(monkeypatch, {"Close": closes})
    assert sr.fresh_quotes(["A", "B", "C", "D"]) == {"A": 2.0, "C": 5.0}
    _yf(monkeypatch, {"Close": pd.Series([3.0, 4.0])})
    assert sr.fresh_quotes(["Z"]) == {"Z": 4.0}


def test_fresh_quotes_without_close_key_and_errors(monkeypatch, capsys):
    _yf(monkeypatch, pd.DataFrame({"A": [7.0]}))  # no "Close" key: the frame itself is used
    assert sr.fresh_quotes(["A"]) == {"A": 7.0}
    _yf(monkeypatch, exc=RuntimeError("blocked"))
    assert sr.fresh_quotes(["A"]) == {}
    assert "fresh-quote batch failed: blocked" in capsys.readouterr().out


# ---------------------------------------------------------------- reconcile
@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "HERE", tmp_path)
    monkeypatch.setattr(sr, "market_open_now", lambda m: False)
    monkeypatch.setattr(sr, "warehouse_prev_close", lambda m: pd.DataFrame(columns=["symbol", "ltp"]))
    return tmp_path


def _scan(env, rows, market="US", name=None, columns=("Symbol", "LTP", "Turnover_USD")):
    pat = sr.MARKETS[market][0]
    d = env / pat.split("/")[0]
    d.mkdir(exist_ok=True)
    name = name or f"{pat.split('/')[1].replace('*', date.today().strftime('%Y%m%d') + '_0300')}"
    df = pd.DataFrame(rows, columns=list(columns))
    with pd.ExcelWriter(d / name) as w:
        df.to_excel(w, sheet_name="All_Stocks", index=False)
    return d / name


def test_reconcile_no_workbook_skips_as_pass(env, capsys):
    assert sr.reconcile("US", 8, 2.0) is True and "no scan workbook — SKIP" in capsys.readouterr().out


def test_reconcile_stale_workbook_fails_without_network(env, monkeypatch, capsys):
    _scan(env, [("A", 1.0, 5)], name="us_full_scan_20200101_0300.xlsx")
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: pytest.fail("must not fetch"))
    assert sr.reconcile("US", 8, 2.0) is False
    assert "STALE WORKBOOK" in capsys.readouterr().out


def test_reconcile_missing_columns_skips(env, capsys):
    _scan(env, [("A", 1.0)], columns=("Sym", "Price"))
    assert sr.reconcile("US", 8, 2.0) is True and "SKIP" in capsys.readouterr().out


def test_reconcile_passes_when_fresh_quotes_agree(env, monkeypatch, capsys):
    _scan(env, [("AAA", 100.0, 9), ("BBB", 50.0, 8), ("CCC", 10.0, 1)])
    asked = []
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: asked.append(list(t)) or {"AAA": 100.5, "BBB": 50.2})
    assert sr.reconcile("US", 2, 2.0) is True
    assert asked == [["AAA", "BBB"]]  # top-2 by turnover, not file order
    assert "2 checked · PASS" in capsys.readouterr().out


def test_reconcile_tolerance_boundary_and_failure_report(env, monkeypatch, capsys):
    _scan(env, [("AAA", 102.0, 9), ("BBB", 110.0, 8)])
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: {"AAA": 100.0, "BBB": 100.0})
    assert sr.reconcile("US", 8, 2.0) is False  # BBB at 10% is stale; AAA at exactly 2.0% is fine
    out = capsys.readouterr().out
    assert out.count("!!") == 1 and "BBB" in out and "diff=10.00%" in out
    assert "FAIL — 1/2 stale (worst BBB 10.0%)" in out


def test_reconcile_in_session_loosens_tolerance_and_uses_quorum(env, monkeypatch, capsys):
    monkeypatch.setattr(sr, "market_open_now", lambda m: True)
    _scan(env, [("A", 103.0, 9), ("B", 100.0, 8), ("C", 100.0, 7), ("D", 100.0, 6)])
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: {k: 100.0 for k in t})
    assert sr.reconcile("US", 4, 2.0) is True  # 3% < 4% (doubled tolerance)
    assert "tolerance loosened to 4.0%" in capsys.readouterr().out
    _scan(env, [("A", 110.0, 9), ("B", 100.0, 8), ("C", 100.0, 7), ("D", 100.0, 6)])
    assert sr.reconcile("US", 4, 2.0) is True  # one breach < quorum(4)=2
    assert "below the 2-name in-session quorum" in capsys.readouterr().out
    _scan(env, [("A", 110.0, 9), ("B", 120.0, 8), ("C", 100.0, 7), ("D", 100.0, 6)])
    assert sr.reconcile("US", 4, 2.0) is False  # two breaches reach the quorum


def test_reconcile_out_of_session_single_breach_fails(env, monkeypatch):
    _scan(env, [("A", 110.0, 9), ("B", 100.0, 8)])
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: {k: 100.0 for k in t})
    assert sr.reconcile("US", 2, 2.0) is False


def test_reconcile_suffix_added_once_and_fresh_misses_ignored(env, monkeypatch):
    _scan(env, [("TCS", 100.0, 9), ("INFY.NS", 100.0, 8), ("WIPRO", 100.0, 7)], market="IN")
    seen = []
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: seen.append(list(t)) or {"TCS.NS": 100.0, "INFY.NS": 0.0})
    assert sr.reconcile("IN", 3, 2.0) is True
    assert seen == [["TCS.NS", "INFY.NS", "WIPRO.NS"]]  # no double suffix; zero/missing quotes skipped


def test_reconcile_yfinance_unavailable_falls_back_to_signature(env, monkeypatch, capsys):
    _scan(env, [(f"S{i}", 100.0 + i, 9 - i) for i in range(4)])
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: {})
    assert sr.reconcile("US", 4, 2.0) is True
    assert "live spot-check unavailable" in capsys.readouterr().out


def test_reconcile_warehouse_signature_quorum(env, monkeypatch, capsys):
    _scan(env, [("A", 100.0, 9), ("B", 50.0, 8), ("C", 25.0, 7), ("D", 10.0, 6)])
    wh = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "ltp": [100.0, 50.0, 25.0, 10.5]})
    monkeypatch.setattr(sr, "warehouse_prev_close", lambda m: wh)
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: pytest.fail("short-circuits before the network"))
    assert sr.reconcile("US", 4, 2.0) is False  # 3 of 4 exact matches >= quorum(4)=2
    out = capsys.readouterr().out
    assert "3/4 liquid names EXACTLY equal" in out and "re-served yesterday's closes (3/4 exact matches)" in out


def test_reconcile_single_signature_match_is_tolerated_and_suffix_stripped(env, monkeypatch, capsys):
    _scan(env, [("TCS.NS", 100.0, 9), ("INFY", 55.0, 8)], market="IN")
    wh = pd.DataFrame({"symbol": ["TCS", "INFY"], "ltp": [100.0, 50.0]})
    monkeypatch.setattr(sr, "warehouse_prev_close", lambda m: wh)
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: {"TCS.NS": 100.0, "INFY.NS": 55.0})
    assert sr.reconcile("IN", 2, 2.0) is True  # 1 exact match < quorum(2)=2
    assert "1/2 liquid names EXACTLY equal" in capsys.readouterr().out


def test_reconcile_without_turnover_column_uses_head(env, monkeypatch):
    _scan(env, [("A", 1.0), ("B", 2.0), ("C", 3.0)], columns=("Symbol", "LTP"))
    seen = []
    monkeypatch.setattr(sr, "fresh_quotes", lambda t: seen.append(list(t)) or {k: 1.0 for k in t})
    sr.reconcile("US", 2, 50000.0)
    assert seen == [["A", "B"]]


# ---------------------------------------------------------------- main
def test_main_counts_failures_and_filters_market(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sr, "reconcile", lambda m, n, t: calls.append((m, n, t)) or (m != "JP"))
    monkeypatch.setattr(sys, "argv", ["scan_price_reconcile.py"])
    assert sr.main() == 1
    assert [c[0] for c in calls] == list(sr.MARKETS) and calls[0][1:] == (8, 2.0)
    assert "reconcile: 4/5 markets clean" in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["scan_price_reconcile.py", "--market", "US", "--sample", "3", "--tolerance", "1.5"])
    assert sr.main() == 0 and calls == [("US", 3, 1.5)]
