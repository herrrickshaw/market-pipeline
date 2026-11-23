"""Offline tests for fundamentals_offhours (universe cleaning, yfinance reshaping, store writes, main loop)."""

import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import fundamentals_offhours as fo


@pytest.fixture
def paths(monkeypatch, tmp_path):
    monkeypatch.setattr(fo, "HERE", tmp_path)
    monkeypatch.setattr(fo, "STORE", tmp_path / "store" / "IN_current.parquet")
    monkeypatch.setattr(fo, "QSTORE", tmp_path / "store" / "IN_quarterly.parquet")
    monkeypatch.setattr(fo, "UNIVERSE_PARQUET", tmp_path / "universe.parquet")
    monkeypatch.setattr(fo.time, "sleep", lambda s: None)
    return tmp_path


# ------------------------------------------------------------------ universe
def test_clean_equities_rules():
    out = fo._clean_equities(["tcs", " infy ", "AB", "1ABC", "A1234", "A12", "ABC12", "ABCD1", "", "X.NS"])
    # len<3, leading digits, and more than half digits (A1234, A12) are dropped
    assert out == ["TCS", "INFY", "ABC12", "ABCD1", "X.NS"]
    assert fo._clean_equities(["ABCD12"]) == ["ABCD12"]  # 2 of 6 digits
    assert fo._clean_equities(["AB1234"]) == []          # 4 of 6 digits > half
    assert fo._clean_equities(["ABC123"]) == ["ABC123"]  # exactly half is kept (strict >)


def test_hash_order_is_deterministic_dedup_and_not_alphabetical():
    syms = [f"{c}{c}{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    a = fo._hash_order(syms + syms)
    assert a == fo._hash_order(list(reversed(syms))) and len(a) == 26
    assert a != sorted(a)  # a hash order, so a partial run is not biased toward 'A'


def test_universe_from_latest_scan_workbook(paths):
    d = paths / "indian_full_scan"
    d.mkdir()
    for n, syms in (("indian_full_scan_1.xlsx", ["OLD"]), ("indian_full_scan_2.xlsx", ["TCS", "INFY", "AB", "123456"])):
        with pd.ExcelWriter(d / n) as w:
            pd.DataFrame({"Symbol": syms}).to_excel(w, sheet_name="All_Stocks", index=False)
    assert sorted(fo.universe()) == ["INFY", "TCS"]


def test_universe_falls_back_to_parquet_then_exits(paths):
    with pytest.raises(SystemExit, match="universe not found"):
        fo.universe()
    pd.DataFrame({"Symbol": ["reliance", "xy", "hdfc"]}).to_parquet(fo.UNIVERSE_PARQUET)
    assert sorted(fo.universe()) == ["HDFC", "RELIANCE"]
    d = paths / "indian_full_scan"
    d.mkdir()
    (d / "indian_full_scan_9.xlsx").write_text("corrupt")  # unreadable workbook -> parquet fallback
    assert sorted(fo.universe()) == ["HDFC", "RELIANCE"]


# ------------------------------------------------------------------ _pick
def _stmt(rows, cols=("2025-03-31", "2024-03-31")):
    return pd.DataFrame({pd.Timestamp(c): [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows))


def test_pick_exact_case_insensitive_and_substring():
    df = _stmt({"Total Assets": [100.0, 90.0], "net income common stockholders": [5.0, 4.0], "Cash And Cash Equivalents": [1.0, 2.0]})
    assert fo._pick(df, ["Total Assets"]).iloc[0] == 100.0
    assert fo._pick(df, ["Net Income", "Net Income Common Stockholders"]).iloc[0] == 5.0  # case-insensitive exact on the 2nd label
    assert fo._pick(df, ["cash"]).iloc[0] == 1.0  # substring match is the last resort
    assert fo._pick(df, ["Missing"]) is None
    assert fo._pick(None, ["Total Assets"]) is None and fo._pick(pd.DataFrame(), ["x"]) is None
    # exact beats substring even if the substring row comes first
    df2 = _stmt({"Total Assets Gross": [1.0, 1.0], "Total Assets": [2.0, 2.0]})
    assert fo._pick(df2, ["Total Assets"]).iloc[0] == 2.0


# ------------------------------------------------------------------ quarterly / yfinance
class _T:
    def __init__(self, bs=None, is_=None, cf=None, q=None, qf=None, raise_attr=None):
        self._bs, self._is, self._cf, self._q, self._qf, self._ra = bs, is_, cf, q, qf, raise_attr

    def _get(self, name, val):
        if self._ra == name:
            raise RuntimeError(name)
        return val

    balance_sheet = property(lambda s: s._get("bs", s._bs))
    income_stmt = property(lambda s: s._get("is", s._is))
    cashflow = property(lambda s: s._get("cf", s._cf))
    quarterly_income_stmt = property(lambda s: s._get("q", s._q))
    quarterly_financials = property(lambda s: s._get("qf", s._qf))


def _q():
    cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30")]
    return pd.DataFrame({cols[0]: [100.0, np.nan], cols[1]: [90.0, 7.0]}, index=["Total Revenue", "Net Income"])


def test_quarterly_rows_newest_first_with_missing_values():
    rows = fo._quarterly_rows("TCS", _T(q=_q()))
    assert [r["quarter_end"] for r in rows] == ["2025-12-31", "2025-09-30"]
    assert rows[0] == {"ticker": "TCS", "quarter_end": "2025-12-31", "revenue": 100.0, "net_income": None, "source": "yfinance"}
    assert rows[1]["net_income"] == 7.0


def test_quarterly_rows_fallbacks():
    assert fo._quarterly_rows("X", _T(q=pd.DataFrame(), qf=_q()))[0]["revenue"] == 100.0  # falls back to quarterly_financials
    assert fo._quarterly_rows("X", _T(q=None, qf=None)) == []
    assert fo._quarterly_rows("X", _T(raise_attr="q")) == []
    only_rev = pd.DataFrame({pd.Timestamp("2025-12-31"): [1.0]}, index=["Total Revenue"])
    r = fo._quarterly_rows("X", _T(q=only_rev))
    assert r[0]["net_income"] is None and r[0]["revenue"] == 1.0
    other = pd.DataFrame({pd.Timestamp("2025-12-31"): [1.0]}, index=["Interest Income"])
    assert fo._quarterly_rows("X", _T(q=other)) == []


def _fake_yf(monkeypatch, ticker_obj):
    mod = types.ModuleType("yfinance")
    seen = {}
    mod.Ticker = lambda s: seen.update(sym=s) or ticker_obj
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return seen


def test_from_yfinance_builds_one_row_per_fiscal_year(monkeypatch):
    bs = _stmt({"Total Assets": [100.0, 90.0], "Long Term Debt": [10.0, np.nan]})
    is_ = _stmt({"Net Income": [5.0, 4.0], "Total Revenue": [50.0, 40.0]}, cols=("2025-03-31", "2023-03-31"))
    cf = _stmt({"Operating Cash Flow": [7.0, 6.0]})
    seen = _fake_yf(monkeypatch, _T(bs=bs, is_=is_, cf=cf, q=_q()))
    sink = []
    rows = fo.from_yfinance("TCS", quarterly_sink=sink)
    assert seen["sym"] == "TCS.NS"
    assert [r["fy_end"] for r in rows] == ["2023-03-31", "2024-03-31", "2025-03-31"]
    r25 = rows[2]
    assert r25["total_assets"] == 100.0 and r25["net_income"] == 5.0 and r25["cfo"] == 7.0 and r25["source"] == "yfinance"
    assert rows[1]["total_assets"] == 90.0 and rows[1]["net_income"] is None  # statement lacks that year
    assert rows[0]["total_assets"] is None and rows[0]["net_income"] == 4.0
    assert r25["long_term_debt"] == 10.0 and rows[1]["long_term_debt"] is None  # NaN becomes None
    assert r25["gross_profit"] is None  # row absent from every statement
    assert len(sink) == 2 and sink[0]["ticker"] == "TCS"


def test_from_yfinance_empty_and_error_paths(monkeypatch):
    _fake_yf(monkeypatch, _T(bs=pd.DataFrame(), is_=None, cf=pd.DataFrame()))
    assert fo.from_yfinance("X") == []
    _fake_yf(monkeypatch, _T(raise_attr="bs"))
    assert fo.from_yfinance("X") == []
    bs = _stmt({"Total Assets": [100.0, 90.0]})
    _fake_yf(monkeypatch, _T(bs=bs, is_=None, cf=None))
    assert len(fo.from_yfinance("X", quarterly_sink=None)) == 2  # works without a sink and with missing statements


# ------------------------------------------------------------------ store
def test_load_store_and_fresh_set(paths):
    assert fo.load_store().empty
    assert fo._fresh_set(pd.DataFrame(), 30) == set()
    assert fo._fresh_set(pd.DataFrame({"ticker": ["A"]}), 30) == set()
    now = datetime.now(timezone.utc)
    store = pd.DataFrame({"ticker": ["new", "old", "bad", "edge"],
                          "collected_at": [now.isoformat(), (now - timedelta(days=60)).isoformat(), "not a date", (now - timedelta(days=29)).isoformat()]})
    assert fo._fresh_set(store, 30) == {"NEW", "EDGE"}
    assert fo._fresh_set(store, 1) == {"NEW"}


def test_write_store_replaces_tickers_and_keeps_backup(paths):
    assert fo.write_store(pd.DataFrame({"ticker": ["A"]}), [], "t0").shape == (1, 1)  # no new rows: untouched, nothing written
    assert not fo.STORE.exists()
    first = fo.write_store(pd.DataFrame(), [{"ticker": "AAA", "fy_end": "2024-03-31", "cfo": 1.0}], "t1")
    assert fo.STORE.exists() and not fo.STORE.with_suffix(".parquet.bak").exists()
    second = fo.write_store(first, [{"ticker": "aaa", "fy_end": "2025-03-31", "cfo": 2.0}, {"ticker": "BBB", "fy_end": "2025-03-31", "cfo": 3.0}], "t2")
    got = pd.read_parquet(fo.STORE)
    assert sorted(got["ticker"]) == ["BBB", "aaa"]  # AAA's old row replaced, case-insensitively
    assert got.set_index("ticker").loc["aaa", "collected_at"] == "t2" and len(second) == 2
    assert fo.STORE.with_suffix(".parquet.bak").exists() and not fo.STORE.with_suffix(".parquet.tmp").exists()


def test_write_quarterly_merges(paths):
    fo._write_quarterly([], "t")
    assert not fo.QSTORE.exists()
    fo._write_quarterly([{"ticker": "AAA", "quarter_end": "2025-12-31", "revenue": 1.0}], "t1")
    fo._write_quarterly([{"ticker": "aaa", "quarter_end": "2025-09-30", "revenue": 2.0}, {"ticker": "BBB", "quarter_end": "2025-09-30", "revenue": 3.0}], "t2")
    got = pd.read_parquet(fo.QSTORE)
    assert sorted(got["ticker"]) == ["BBB", "aaa"] and set(got["collected_at"]) == {"t2"}


def test_coverage_report(capsys):
    fo.coverage(pd.DataFrame())
    assert "store empty" in capsys.readouterr().out
    store = pd.DataFrame({"ticker": ["AAA", "aab", "BBB", "CCC"], "cfo": [1.0, 2.0, 3.0, np.nan], "total_assets": [1.0, 1.0, np.nan, 1.0]})
    fo.coverage(store)
    out = capsys.readouterr().out
    assert "usable tickers (cfo+total_assets): 2" in out and "distinct first-letters: 1   A-share: 100%" in out and "{'A': 2}" in out
    fo.coverage(pd.DataFrame({"ticker": ["BBB"], "cfo": [np.nan], "total_assets": [np.nan]}))
    assert "usable tickers (cfo+total_assets): 0" in capsys.readouterr().out


def test_self_test_passes(capsys):
    assert fo.self_test() == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and out.count("PASS") == 6 and "self-test: OK" in out


# ------------------------------------------------------------------ main
def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    return fo.main()


def test_main_self_test_flag(monkeypatch, capsys):
    assert _run(monkeypatch, "--self-test") == 0
    assert "self-test: OK" in capsys.readouterr().out


def test_main_collects_checkpoints_and_reports(paths, monkeypatch, capsys):
    monkeypatch.setattr(fo, "universe", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(fo, "CHECKPOINT_EVERY", 2)
    calls = []

    def fake(sym, quarterly_sink=None):
        calls.append(sym)
        if sym == "CCC":
            return []
        quarterly_sink.append({"ticker": sym, "quarter_end": "2025-12-31", "revenue": 1.0, "net_income": 1.0, "source": "yfinance"})
        return [{"ticker": sym, "fy_end": "2025-03-31", "cfo": 1.0, "total_assets": 1.0}]
    monkeypatch.setattr(fo, "from_yfinance", fake)
    assert _run(monkeypatch, "--rate", "0") == 0
    out = capsys.readouterr().out
    assert calls == ["AAA", "BBB", "CCC", "DDD"] and "[2/4] checkpoint · 2 collected, 0 empty" in out
    assert "done: 3 collected, 1 empty this run" in out and "to collect 4" in out
    assert sorted(pd.read_parquet(fo.STORE)["ticker"]) == ["AAA", "BBB", "DDD"]
    assert sorted(pd.read_parquet(fo.QSTORE)["ticker"]) == ["AAA", "BBB", "DDD"]


def test_main_skips_fresh_tickers_and_honours_limit(paths, monkeypatch, capsys):
    monkeypatch.setattr(fo, "universe", lambda: ["AAA", "BBB", "CCC", "DDD"])
    now = datetime.now(timezone.utc).isoformat()
    fo.write_store(pd.DataFrame(), [{"ticker": "AAA", "fy_end": "2025-03-31", "cfo": 1.0, "total_assets": 1.0},
                                    {"ticker": "BBB", "fy_end": "2025-03-31", "cfo": 1.0, "total_assets": 1.0}], now)
    fo._write_quarterly([{"ticker": "AAA", "quarter_end": "2025-12-31"}], now)  # only AAA has quarterly data
    got = []
    monkeypatch.setattr(fo, "from_yfinance", lambda s, quarterly_sink=None: got.append(s) or [])
    assert _run(monkeypatch, "--limit", "2", "--rate", "0") == 0
    # AAA is fresh in both stores; BBB lacks quarterly rows so it is re-collected
    assert got == ["BBB", "CCC"]
    assert "fresh (<30d) 1" in capsys.readouterr().out


def test_main_nothing_to_collect(paths, monkeypatch, capsys):
    monkeypatch.setattr(fo, "universe", lambda: ["AAA"])
    now = datetime.now(timezone.utc).isoformat()
    fo.write_store(pd.DataFrame(), [{"ticker": "AAA", "fy_end": "2025-03-31", "cfo": 1.0, "total_assets": 1.0}], now)
    fo._write_quarterly([{"ticker": "AAA", "quarter_end": "2025-12-31"}], now)
    monkeypatch.setattr(fo, "from_yfinance", lambda *a, **k: pytest.fail("everything is fresh"))
    assert _run(monkeypatch, "--max-age-days", "30") == 0
    out = capsys.readouterr().out
    assert "nothing to collect — store is current" in out and "usable tickers" in out


def test_main_stops_after_60_consecutive_empties(paths, monkeypatch, capsys):
    monkeypatch.setattr(fo, "universe", lambda: [f"SYM{i:03d}" for i in range(80)])
    calls = []
    monkeypatch.setattr(fo, "from_yfinance", lambda s, quarterly_sink=None: calls.append(s) or [])
    assert _run(monkeypatch, "--rate", "0") == 0
    assert len(calls) == 60 and "60 consecutive empties" in capsys.readouterr().out
