"""Offline tests for walk_forward_backtest (signals, splits, returns, stats, reports, orchestrator)."""

import importlib
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class _Ticker:
    """Stand-in for yfinance.Ticker: just attributes."""

    def __init__(self, **kw):
        self.info = {}
        self.fast_info = types.SimpleNamespace(market_cap=0)
        for k, v in kw.items():
            setattr(self, k, v)


def _fake_yf(download=None, tickers=None):
    m = types.ModuleType("yfinance")
    m.download = download or (lambda *a, **k: pd.DataFrame())
    tickers = tickers or {}
    m.Ticker = lambda t: tickers.get(t, _Ticker())
    return m


@pytest.fixture
def wf(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # OUT_DIR is created relative to cwd at import
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf())
    # no real nse_data_fetcher: force the ImportError branch
    monkeypatch.setitem(sys.modules, "nse_data_fetcher", None)
    sys.modules.pop("walk_forward_backtest", None)
    mod = importlib.import_module("walk_forward_backtest")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    sys.modules.pop("walk_forward_backtest", None)


# ---------------------------------------------------------------- helpers
def _ohlc(n, start="2020-01-01", close=100.0):
    idx = pd.bdate_range(start, periods=n)
    c = np.full(n, float(close))
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, 100.0)}, index=idx
    )


def _darvas_df(n=130, drift=0.0, second=True):
    """Flat 10 with a box top of 12 at bar 25 and a breakout close of 13 at bar 35
    (and, optionally, a second box top at 90 / breakout at 100)."""
    df = _ohlc(n, close=10.0)
    df["High"] = 10.0
    df.iloc[25, df.columns.get_loc("High")] = 12.0
    df.iloc[35, df.columns.get_loc("Close")] = 13.0
    df.iloc[35, df.columns.get_loc("Volume")] = 200.0
    if second:
        df.iloc[90, df.columns.get_loc("High")] = 12.0
        df.iloc[100, df.columns.get_loc("Close")] = 13.0
        df.iloc[100, df.columns.get_loc("Volume")] = 200.0
    if drift:
        for i in range(36, n):
            if i != 100:
                df.iloc[i, df.columns.get_loc("Close")] = 10.0 + drift * (i - 36)
    return df


def _index_df(n=400, start="2019-01-01", close=100.0, dma=100.0, slope=1.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"Close": np.full(n, close), "dma200": np.full(n, dma), "dma200_sl": np.full(n, slope)},
        index=idx,
    )


# ---------------------------------------------------------------- fetch_ohlc_bulk
def _multi(tickers, rows=60):
    idx = pd.bdate_range("2024-01-01", periods=rows)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    df = pd.DataFrame(np.ones((rows, len(cols))), index=idx, columns=cols)
    return df


def test_fetch_ohlc_bulk_multiindex_filters_short_and_missing(wf, monkeypatch, capsys):
    calls = []

    def dl(batch, **kw):
        calls.append((list(batch), kw))
        raw = _multi(["A.NS", "B.NS"])
        raw[("Close", "B.NS")] = np.nan  # B all-NaN -> dropped rows -> empty -> skipped
        raw[("Volume", "B.NS")] = np.nan
        return raw

    monkeypatch.setattr(wf.yf, "download", dl)
    out = wf.fetch_ohlc_bulk(["A.NS", "B.NS", "C.NS"], period="3y")
    assert set(out) == {"A.NS"}  # B empty after dropna, C absent (KeyError)
    assert len(calls) == 1 and calls[0][1]["period"] == "3y"
    assert "Batch 1/1" in capsys.readouterr().out


def test_fetch_ohlc_bulk_requires_50_rows_and_batches(wf, monkeypatch):
    seen = []

    def dl(batch, **kw):
        seen.append(len(batch))
        return _multi(list(batch), rows=49 if "SHORT.NS" in batch else 50)

    monkeypatch.setattr(wf.yf, "download", dl)
    out = wf.fetch_ohlc_bulk(["OK.NS"], period="3y")
    assert "OK.NS" in out  # exactly 50 rows is kept
    out = wf.fetch_ohlc_bulk(["SHORT.NS"], period="3y")
    assert out == {}  # 49 rows dropped
    # batch size: 100 for long periods, 200 otherwise
    seen.clear()
    wf.fetch_ohlc_bulk([f"T{i}.NS" for i in range(250)], period="10y")
    assert seen == [100, 100, 50]
    seen.clear()
    wf.fetch_ohlc_bulk([f"T{i}.NS" for i in range(250)], period="3y")
    assert seen == [200, 50]


def test_fetch_ohlc_bulk_empty_single_level_and_errors(wf, monkeypatch, capsys):
    monkeypatch.setattr(wf.yf, "download", lambda *a, **k: pd.DataFrame())
    assert wf.fetch_ohlc_bulk(["A.NS"]) == {}
    assert "empty" in capsys.readouterr().out

    flat = _ohlc(60)
    monkeypatch.setattr(wf.yf, "download", lambda *a, **k: flat)
    out = wf.fetch_ohlc_bulk(["A.NS", "B.NS"], period="3y")
    assert list(out) == ["A.NS"] and out["A.NS"] is flat  # single-frame -> first ticker

    def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(wf.yf, "download", boom)
    assert wf.fetch_ohlc_bulk(["A.NS"]) == {}
    assert "ERROR — kaput" in capsys.readouterr().out


def test_fetch_ohlc_bulk_rate_limit_retries_with_backoff(wf, monkeypatch):
    sleeps = []
    monkeypatch.setattr(wf.time, "sleep", lambda s: sleeps.append(s))
    state = {"n": 0}

    def dl(batch, **kw):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("429 Too Many Requests")
        return _multi(list(batch))

    monkeypatch.setattr(wf.yf, "download", dl)
    out = wf.fetch_ohlc_bulk(["A.NS"], period="3y")
    assert "A.NS" in out and state["n"] == 3
    assert sleeps == [30, 60]  # 30s * attempt, no inter-batch sleep after the last batch

    # persistent rate limiting gives up after 3 attempts
    state["n"] = -100
    sleeps.clear()
    assert wf.fetch_ohlc_bulk(["A.NS"]) == {}
    assert sleeps == [30, 60, 90]


def test_fetch_ohlc_bulk_sleeps_between_batches(wf, monkeypatch):
    sleeps = []
    monkeypatch.setattr(wf.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(wf.yf, "download", lambda *a, **k: pd.DataFrame())
    wf.fetch_ohlc_bulk([f"T{i}" for i in range(250)], period="3y")  # 2 batches
    assert sleeps == [1.5]
    sleeps.clear()
    wf.fetch_ohlc_bulk([f"T{i}" for i in range(250)], period="10y")  # 3 batches
    assert sleeps == [3.0, 3.0]


# ---------------------------------------------------------------- fetch_index
def test_fetch_index_adds_indicators(wf, monkeypatch, capsys):
    n = 260
    idx = pd.bdate_range("2023-01-02", periods=n)
    raw = pd.DataFrame({"Close": np.arange(n, dtype=float) + 100}, index=idx)
    monkeypatch.setattr(wf.yf, "download", lambda *a, **k: raw)
    df = wf.fetch_index("3y")
    assert df["dma50"].iloc[49] == pytest.approx(raw["Close"].iloc[:50].mean())
    assert np.isnan(df["dma50"].iloc[48])
    assert df["dma200"].iloc[199] == pytest.approx(raw["Close"].iloc[:200].mean())
    assert df["dma200_sl"].iloc[204] == pytest.approx(df["dma200"].iloc[204] - df["dma200"].iloc[199])
    assert "vol20" in df.columns
    assert "OK (260 bars" in capsys.readouterr().out


def test_fetch_index_multiindex_and_error(wf, monkeypatch):
    idx = pd.bdate_range("2023-01-02", periods=10)
    cols = pd.MultiIndex.from_product([["Close"], [wf.INDEX_SYM]])
    raw = pd.DataFrame(np.arange(10.0).reshape(10, 1) + 1, index=idx, columns=cols)
    monkeypatch.setattr(wf.yf, "download", lambda *a, **k: raw)
    df = wf.fetch_index()
    assert list(df["Close"]) == list(np.arange(10.0) + 1)

    def boom(*a, **k):
        raise ValueError("nope")

    monkeypatch.setattr(wf.yf, "download", boom)
    assert wf.fetch_index().empty


# ---------------------------------------------------------------- get_nse_symbols
def test_get_nse_symbols_primary_fetcher(wf, monkeypatch):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", True)
    syms = [f"S{i}" for i in range(150)]
    monkeypatch.setattr(wf, "_get_nse_syms", lambda: syms, raising=False)
    out = wf.get_nse_symbols()
    assert len(out) == 150 and out[0] == ("S0", ".NS")
    assert wf.get_nse_symbols(top=5) == [(f"S{i}", ".NS") for i in range(5)]


def test_get_nse_symbols_primary_too_small_or_failing_falls_back(wf, monkeypatch):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", True)
    monkeypatch.setitem(sys.modules, "nse", None)  # ImportError for the nse-library fallback
    monkeypatch.setattr(wf, "_get_nse_syms", lambda: ["A"] * 100, raising=False)  # needs > 100
    out = wf.get_nse_symbols()
    assert len(out) == 50 and ("RELIANCE", ".NS") in out  # Nifty-50 fallback list

    def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(wf, "_get_nse_syms", boom, raising=False)
    assert len(wf.get_nse_symbols()) == 50
    assert wf.get_nse_symbols(top=3) == [(s, ".NS") for s in
                                         ["ADANIENT", "ADANIPORTS", "APOLLOHOSP"]]


def test_get_nse_symbols_no_fetcher_uses_fallback_message(wf, monkeypatch, capsys):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", False)
    monkeypatch.setitem(sys.modules, "nse", None)
    assert len(wf.get_nse_symbols()) == 50
    assert "Nifty 50 fallback" in capsys.readouterr().out


class _FakeNSE:
    """nse.NSE context manager returning a bhavcopy csv for the 3rd day back."""

    def __init__(self, csvs, **kw):
        self.csvs = csvs
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def equityBhavcopy(self, d):
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("holiday")
        return self.csvs


def _install_nse(monkeypatch, csv_path):
    m = types.ModuleType("nse")
    m.NSE = lambda **kw: _FakeNSE(csv_path, **kw)
    monkeypatch.setitem(sys.modules, "nse", m)


def test_get_nse_symbols_bhavcopy_new_format(wf, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", False)
    p = tmp_path / "b.csv"
    pd.DataFrame(
        {"SctySrs": ["EQ", "BE", "EQ", "EQ"], "TckrSymb": [" ZED ", "SKIP", "ABC", None]}
    ).to_csv(p, index=False)
    _install_nse(monkeypatch, p)
    out = wf.get_nse_symbols()
    assert out == [("ABC", ".NS"), ("ZED", ".NS")]  # sorted, stripped, NaN dropped, EQ only
    assert "2 EQ symbols" in capsys.readouterr().out


def test_get_nse_symbols_bhavcopy_old_format_and_top(wf, monkeypatch, tmp_path):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", False)
    p = tmp_path / "b.csv"
    pd.DataFrame(
        {"SYMBOL": ["XX", "YY", "ZZ"], "SERIES": ["EQ", "EQ", "N1"]}
    ).to_csv(p, index=False)
    _install_nse(monkeypatch, p)
    assert wf.get_nse_symbols() == [("XX", ".NS"), ("YY", ".NS")]
    _install_nse(monkeypatch, p)
    assert wf.get_nse_symbols(top=1) == [("XX", ".NS")]


def test_get_nse_symbols_bhavcopy_missing_file_falls_back(wf, monkeypatch, tmp_path):
    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", False)
    _install_nse(monkeypatch, tmp_path / "does_not_exist.csv")  # .exists() False every day
    assert len(wf.get_nse_symbols()) == 50


# ---------------------------------------------------------------- classify_regime
def test_classify_regime_unknown_inputs(wf):
    d = pd.Timestamp("2020-06-01")
    assert wf.classify_regime(d, None) == "UNKNOWN"
    assert wf.classify_regime(d, pd.DataFrame()) == "UNKNOWN"
    assert wf.classify_regime(pd.Timestamp("2000-01-01"), _index_df()) == "UNKNOWN"  # before data
    nan_df = _index_df()
    nan_df["dma200"] = np.nan
    assert wf.classify_regime(d, nan_df) == "UNKNOWN"


def test_classify_regime_bull_bear_sideways_boundaries(wf):
    d = pd.Timestamp("2020-03-02")
    assert wf.classify_regime(d, _index_df(close=110, dma=100, slope=1)) == "BULL"
    assert wf.classify_regime(d, _index_df(close=90, dma=100, slope=-1)) == "BEAR"
    # above the DMA but falling slope, or below with rising slope, is sideways
    assert wf.classify_regime(d, _index_df(close=110, dma=100, slope=-1)) == "SIDEWAYS"
    assert wf.classify_regime(d, _index_df(close=90, dma=100, slope=1)) == "SIDEWAYS"
    assert wf.classify_regime(d, _index_df(close=110, dma=100, slope=0)) == "SIDEWAYS"
    # within 1.5% of the DMA is sideways even if trending
    assert wf.classify_regime(d, _index_df(close=101.4, dma=100, slope=1)) == "SIDEWAYS"
    assert wf.classify_regime(d, _index_df(close=101.6, dma=100, slope=1)) == "BULL"
    assert wf.classify_regime(d, _index_df(close=98.6, dma=100, slope=-1)) == "SIDEWAYS"
    assert wf.classify_regime(d, _index_df(close=98.0, dma=100, slope=-1)) == "BEAR"


def test_classify_regime_uses_last_bar_at_or_before_date_and_tz(wf):
    idx = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame(
        {
            "Close": [200.0, 200.0, 50.0, 50.0],
            "dma200": [100.0] * 4,
            "dma200_sl": [1.0, 1.0, -1.0, -1.0],
        },
        index=idx,
    )
    assert wf.classify_regime(idx[1], df) == "BULL"  # exact bar (searchsorted side=right)
    assert wf.classify_regime(idx[2], df) == "BEAR"
    assert wf.classify_regime(idx[1] + pd.Timedelta(hours=5), df) == "BULL"
    # tz-aware date and tz-aware index are normalised
    aware = df.copy()
    aware.index = aware.index.tz_localize("Asia/Kolkata")
    assert wf.classify_regime(idx[2].tz_localize("Asia/Kolkata"), aware) == "BEAR"


# ---------------------------------------------------------------- darvas / golden cross
def test_darvas_signals_short_or_none(wf):
    assert wf.darvas_signals(None) == []
    assert wf.darvas_signals(_ohlc(22)) == []  # needs confirm + 20 = 23 bars
    assert wf.darvas_signals(_ohlc(23)) == []  # long enough but flat -> no box breakout


def test_darvas_signal_on_breakout_with_volume(wf):
    sigs = wf.darvas_signals(_darvas_df(second=False))
    assert len(sigs) == 1
    s = sigs[0]
    assert s["idx"] == 35 and s["entry"] == 13.0 and s["box_top"] == 12.0
    assert s["date"] == _darvas_df().index[35]


def test_darvas_volume_confirmation_boundary(wf):
    df = _darvas_df(second=False)
    df.iloc[35, df.columns.get_loc("Volume")] = 109.0  # avg 100 -> needs >= 110
    assert wf.darvas_signals(df) == []
    df.iloc[35, df.columns.get_loc("Volume")] = 111.0
    assert len(wf.darvas_signals(df)) == 1


def test_darvas_requires_prior_close_below_box_top(wf):
    df = _darvas_df(second=False)
    df.iloc[34, df.columns.get_loc("Close")] = 13.0  # already above the box the bar before
    sigs = wf.darvas_signals(df)
    assert sigs == []  # bar 34 lacks volume; bar 35 is no longer a fresh cross of the box top


def test_darvas_cooldown_suppresses_close_repeat(wf):
    df = _darvas_df(second=False)
    df.iloc[36, df.columns.get_loc("Close")] = 11.0
    df.iloc[37, df.columns.get_loc("Close")] = 13.0
    df.iloc[37, df.columns.get_loc("Volume")] = 200.0
    assert [s["idx"] for s in wf.darvas_signals(df)] == [35]  # 37 is inside 10-bar cooldown
    assert [s["idx"] for s in wf.darvas_signals(df, cooldown=1)] == [35, 37]
    assert [s["idx"] for s in wf.darvas_signals(df, cooldown=2)] == [35, 37]  # gap 2 >= 2
    assert [s["idx"] for s in wf.darvas_signals(df, cooldown=3)] == [35]  # gap 2 < 3


def test_darvas_confirm_window_and_no_volume_column(wf):
    df = _darvas_df(second=False).drop(columns=["Volume"])  # falls back to ones -> avg*1.1 > 1
    assert wf.darvas_signals(df) == []
    df = _darvas_df(second=False)
    # a wider confirmation window needs more bars below the box top: still one signal
    assert [s["idx"] for s in wf.darvas_signals(df, confirm=5)] == [35]
    # confirm larger than the bars between peak and breakout kills the box
    assert wf.darvas_signals(df, confirm=15) == []


def test_darvas_second_box(wf):
    sigs = wf.darvas_signals(_darvas_df())
    assert [s["idx"] for s in sigs] == [35, 100]


def test_golden_cross_signals(wf):
    assert wf.golden_cross_signals(None) == []
    assert wf.golden_cross_signals(_ohlc(204)) == []  # needs 205 bars
    close = np.concatenate([np.linspace(100, 50, 250), np.linspace(50, 200, 150)])
    df = _ohlc(400)
    df["Close"] = close
    sigs = wf.golden_cross_signals(df)
    s50 = pd.Series(close).rolling(50).mean()
    s200 = pd.Series(close).rolling(200).mean()
    expected = [
        i for i in range(1, 400)
        if s50[i - 1] < s200[i - 1] and s50[i] > s200[i]
    ]
    assert len(expected) == 1
    assert [s["idx"] for s in sigs] == expected
    s = sigs[0]
    assert s["date"] == df.index[expected[0]]
    assert s["entry"] == pytest.approx(close[expected[0]])
    assert s["dma50"] == round(float(s50[expected[0]]), 2)
    assert s["dma200"] == round(float(s200[expected[0]]), 2)
    # a flat series never crosses (needs strict inequalities)
    assert wf.golden_cross_signals(_ohlc(300)) == []


# ---------------------------------------------------------------- is_financial
def test_is_financial(wf):
    assert wf.is_financial("HDFCBANK")
    assert wf.is_financial("BAJAJ FINSERV")
    assert wf.is_financial("MUTHOOTFIN")
    assert not wf.is_financial("RELIANCE")
    assert not wf.is_financial("RELIANCE", {"sector": "Energy"})
    assert wf.is_financial("RELIANCE", {"sector": "Financial Services"})
    assert wf.is_financial("XYZ", {"sector": "Insurance - Life"})
    assert wf.is_financial("XYZ", {"sector": "Capital Markets"})
    assert not wf.is_financial("XYZ", {"sector": None})
    assert not wf.is_financial("XYZ", {})
    assert not wf.is_financial("XYZ", None)


# ---------------------------------------------------------------- fundamental_signals
def _fin(rows, cols):
    """Statement frame: rows dict name -> values (newest first); columns = period ends."""
    return pd.DataFrame(rows).T.set_axis(pd.to_datetime(cols), axis=1)


FY = ["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31"]


def _piotroski_statements(good=True):
    inc = _fin(
        {
            "Net Income": [40, 30, 20, 10] if good else [10, 20, 30, 40],
            "Total Revenue": [300, 250, 200, 150] if good else [150, 200, 250, 300],
            "Gross Profit": [150, 110, 70, 40] if good else [40, 70, 110, 150],
        },
        FY,
    )
    bal = _fin(
        {
            "Total Assets": [100, 100, 100, 100],
            "Long Term Debt": [10, 20, 30, 40] if good else [40, 30, 20, 10],
            "Current Assets": [250, 230, 210, 190] if good else [190, 210, 230, 250],
            "Current Liabilities": [100, 100, 100, 100],
            "Share Issued": [90, 95, 100, 105] if good else [105, 100, 95, 90],
        },
        FY,
    )
    cf = _fin({"Operating Cash Flow": [60, 50, 40, 30] if good else [-5, -5, -5, -5]}, FY)
    return inc, bal, cf


def _tk(inc=None, bal=None, cf=None, inc_q=None, info=None, mcap=0):
    return _Ticker(
        income_stmt=inc,
        balance_sheet=bal,
        cash_flow=cf,
        quarterly_income_stmt=inc_q,
        info=info or {},
        fast_info=types.SimpleNamespace(market_cap=mcap),
    )


def _wf_ohlc():
    return _ohlc(1500, start="2020-01-01")  # through ~2025-10


def test_fundamental_piotroski_multi_offsets(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    ohlc = _wf_ohlc()
    sigs = wf.fundamental_signals("ACME", ".NS", "piotroski", ohlc)
    assert [s["col_offset"] for s in sigs] == [0, 1, 2]
    # FY ends in March (month <= 7) -> signal on 1 July of the same year
    dates = [s["date"] for s in sigs]
    # 1 Jul 2023 is a Saturday: the last trading bar at or before it is Fri 30 Jun
    assert dates == [pd.Timestamp("2025-07-01"), pd.Timestamp("2024-07-01"), pd.Timestamp("2023-06-30")]
    for s in sigs:
        assert s["entry"] == 100.0 and ohlc.index[s["idx"]] == s["date"]


def test_fundamental_piotroski_bad_and_threshold(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(False)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    assert wf.fundamental_signals("ACME", ".NS", "piotroski", _wf_ohlc()) == []

    # exactly 7/9 passes, 6/9 does not: break shares + gross margin (7), then asset turnover (6)
    inc, bal, cf = _piotroski_statements(True)
    bal7 = bal.copy()
    bal7.loc["Share Issued"] = [105, 100, 95, 90]
    inc7 = inc.copy()
    inc7.loc["Gross Profit"] = [40, 70, 110, 150]
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc7, bal7, cf))
    assert len(wf.fundamental_signals("ACME", ".NS", "piotroski", _wf_ohlc())) == 3
    inc6 = inc7.copy()
    inc6.loc["Total Revenue"] = [150, 200, 250, 300]  # asset turnover falls (and gp margin stays bad)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc6, bal7, cf))
    assert wf.fundamental_signals("ACME", ".NS", "piotroski", _wf_ohlc()) == []


def test_fundamental_signal_date_for_december_year_end(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    dec = ["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
    for df in (inc, bal, cf):
        df.columns = pd.to_datetime(dec)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    sigs = wf.fundamental_signals("ACME", ".NS", "piotroski", _wf_ohlc())
    # FY ends after July -> signal 1 July of the NEXT year; out-of-range ones are skipped
    assert [s["date"] for s in sigs] == [
        pd.Timestamp("2025-07-01"), pd.Timestamp("2024-07-01"), pd.Timestamp("2023-06-30")
    ]


def test_fundamental_skips_financials_for_roic_screeners_only(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    assert wf.fundamental_signals("HDFCBANK", ".NS", "piotroski", _wf_ohlc()) == []
    assert wf.fundamental_signals("HDFCBANK", ".NS", "magic_formula", _wf_ohlc()) == []
    assert wf.fundamental_signals("HDFCBANK", ".NS", "coffee_can", _wf_ohlc()) == []
    # bull_cartel is not in the skip list: it reaches the check (and fails for lack of quarterlies)
    assert wf.fundamental_signals("HDFCBANK", ".NS", "bull_cartel", _wf_ohlc()) == []


def test_fundamental_guards(wf, monkeypatch):
    assert wf.fundamental_signals("A", ".NS", "piotroski", None) == []
    assert wf.fundamental_signals("A", ".NS", "piotroski", pd.DataFrame()) == []

    def boom(t):
        raise RuntimeError("yf down")

    monkeypatch.setattr(wf.yf, "Ticker", boom)
    assert wf.fundamental_signals("A", ".NS", "piotroski", _wf_ohlc()) == []

    # info and fast_info raising are tolerated (marketCap falls back to info)
    class T(_Ticker):
        @property
        def info(self):
            raise RuntimeError("no info")

        @property
        def fast_info(self):
            raise RuntimeError("no fast")

    monkeypatch.setattr(wf.yf, "Ticker", lambda t: T())
    assert wf.fundamental_signals("A", ".NS", "piotroski", _wf_ohlc()) == []

    # no income statement: every screener fails closed
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk())
    for sc in ("piotroski", "coffee_can", "magic_formula", "bull_cartel", "nonsense"):
        assert wf.fundamental_signals("A", ".NS", sc, _wf_ohlc()) == []


def _coffee_statements():
    inc = _fin(
        {
            "Total Revenue": [200, 150, 120, 100],  # CAGR ~26%
            "EBIT": [60, 40, 30, 25],
            "Net Income": [30, 20, 15, 12],
        },
        FY,
    )
    bal = _fin(
        {"Total Assets": [200, 180, 160, 150], "Current Liabilities": [50, 40, 40, 30]}, FY
    )
    return inc, bal


def test_fundamental_coffee_can_boundaries(wf, monkeypatch):
    inc, bal = _coffee_statements()
    ohlc = _wf_ohlc()

    def run(info, mcap, inc=inc, bal=bal):
        monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, None, None, info, mcap))
        return wf.fundamental_signals("ACME", ".NS", "coffee_can", ohlc)

    assert len(run({"debtToEquity": 50}, 5e9)) == 3
    assert len(run({"debtToEquity": 0.5}, 5e9)) == 3  # already a ratio (<=10)
    assert run({"debtToEquity": 100}, 5e9) == []  # 100/100 = 1.0, must be < 1
    assert len(run({"debtToEquity": 99}, 5e9)) == 3
    assert run({"debtToEquity": 10}, 5e9) == []  # 10 is not > 10 so stays 10
    assert run({}, 5e9) == []  # no D/E
    assert len(run({"debtToEquity": 50}, 5e9)) == 3  # mcap exactly 500 Cr passes
    assert run({"debtToEquity": 50}, 4.99e9) == []
    # a loss-making year disqualifies
    inc_loss = inc.copy()
    inc_loss.loc["Net Income"] = [30, 20, -1, 12]
    assert run({"debtToEquity": 50}, 5e9, inc=inc_loss) == []
    # revenue CAGR not above 10%
    inc_slow = inc.copy()
    inc_slow.loc["Total Revenue"] = [105, 104, 102, 100]
    assert run({"debtToEquity": 50}, 5e9, inc=inc_slow) == []
    # ROCE not above 15
    inc_lowroce = inc.copy()
    inc_lowroce.loc["EBIT"] = [10, 8, 6, 5]
    assert run({"debtToEquity": 50}, 5e9, inc=inc_lowroce) == []


def test_fundamental_magic_formula(wf, monkeypatch):
    cr = 1e7
    inc = _fin({"EBIT": [60 * cr, 50 * cr, 40 * cr, 30 * cr], "Net Income": [30 * cr] * 4}, FY)
    bal = _fin({"Total Assets": [200 * cr] * 4, "Current Liabilities": [50 * cr] * 4}, FY)
    ohlc = _wf_ohlc()
    base = {"totalDebt": 100 * cr, "totalCash": 50 * cr, "bookValue": 10}

    def run(info, mcap, inc=inc):
        monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, None, None, info, mcap))
        return wf.fundamental_signals("ACME", ".NS", "magic_formula", ohlc)

    # roic = 60/150 = 40%; ey = 60/(300+100-50) = 17% -> all three offsets qualify
    assert len(run(base, 300 * cr)) == 3
    assert run({**base, "bookValue": 0}, 300 * cr) == []
    assert run({**base, "bookValue": None}, 300 * cr) == []
    assert run(base, 0) == []  # no market cap -> no EV
    # earnings yield must exceed 8%: ev = 750-50 -> 60/800 = 7.5% fails offset 0, offset 1 (6.3%)...
    assert run(base, 750 * cr) == []
    # mcap must exceed 100 Cr strictly
    nodebt = {"totalDebt": 0, "totalCash": 0, "bookValue": 10}
    assert run(nodebt, 100 * cr) == []
    assert len(run(nodebt, 100.1 * cr)) == 3
    # ROIC must exceed 15%: ebit 20 on capital 150 = 13%
    inc_low = inc.copy()
    inc_low.loc["EBIT"] = [20 * cr] * 4
    assert run(base, 100.1 * cr, inc=inc_low) == []
    # negative net income disqualifies
    inc_neg = inc.copy()
    inc_neg.loc["Net Income"] = [-1.0 * cr] * 4
    assert run(base, 300 * cr, inc=inc_neg) == []


def _bull_statements(rev0=130.0, rev4=100.0, ni0=3e7, ni4=2e7, ncols=6):
    cols = pd.date_range("2025-03-31", periods=ncols, freq="-3ME")
    vals_rev = [rev0, 0, 0, 0, rev4] + [100.0] * (ncols - 5)
    vals_ni = [ni0, 0, 0, 0, ni4] + [1.0] * (ncols - 5)
    return pd.DataFrame({"Total Revenue": vals_rev[:ncols], "Net Income": vals_ni[:ncols]}).T.set_axis(
        cols, axis=1
    )


def test_fundamental_bull_cartel_thresholds(wf, monkeypatch):
    inc, _bal, _cf = _piotroski_statements(True)
    ohlc = _wf_ohlc()

    def run(inc_q):
        monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, None, None, inc_q))
        return wf.fundamental_signals("ACME", ".NS", "bull_cartel", ohlc)

    # only the most recent quarter (offset 0) has a year-ago comparison in the fixture
    assert [s["col_offset"] for s in run(_bull_statements())] == [0]
    assert run(_bull_statements(ncols=4)) == []  # needs >= 5 quarters
    assert len(run(_bull_statements(ncols=5))) == 1
    assert run(_bull_statements(rev0=115.0)) == []  # sales growth must be > 15
    assert len(run(_bull_statements(rev0=115.5))) == 1
    assert run(_bull_statements(ni0=2.4e7)) == []  # profit growth must be > 20
    assert len(run(_bull_statements(ni0=2.5e7))) == 1
    assert run(_bull_statements(ni0=1e7, ni4=0.5e7)) == []  # net income must exceed 1 Cr
    assert run(_bull_statements(rev4=0.0)) == []  # no base revenue
    assert run(None) == []


def test_fundamental_without_income_uses_today_july(wf, monkeypatch):
    """With no annual income frame only offset 0 is tried and the date falls back to 1 July today."""

    class _DT(datetime):
        @classmethod
        def today(cls):
            return datetime(2023, 11, 5)

    monkeypatch.setattr(wf, "datetime", _DT)
    inc_q = _bull_statements()
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(None, None, None, inc_q))
    # bull_cartel passes the check without an annual frame (screener returns False for inc None only
    # in other branches)
    sigs = wf.fundamental_signals("ACME", ".NS", "bull_cartel", _wf_ohlc())
    assert len(sigs) == 1 and sigs[0]["date"] == pd.Timestamp("2023-06-30")  # 1 Jul is a Saturday
    assert sigs[0]["col_offset"] == 0


def test_fundamental_clamps_to_ohlc_history(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    # OHLC starting 2024-06-01: only the 2025-07-01 signal has an entry bar with idx >= 5
    ohlc = _ohlc(300, start="2024-06-03")
    sigs = wf.fundamental_signals("ACME", ".NS", "piotroski", ohlc)
    assert [s["col_offset"] for s in sigs] == [0, 1]
    assert all(5 <= s["idx"] < len(ohlc) for s in sigs)
    # tz-aware OHLC index is handled
    aware = _wf_ohlc()
    aware.index = aware.index.tz_localize("UTC")
    assert len(wf.fundamental_signals("ACME", ".NS", "piotroski", aware)) == 3


# ---------------------------------------------------------------- compute_filing_trend
def _quarters(rev_now, rev_ago, ni_now, ni_ago, n=8):
    cols = pd.date_range("2025-03-31", periods=n, freq="-3ME")
    rev = [rev_now] * 4 + [rev_ago] * (n - 4)
    ni = [ni_now] * 4 + [ni_ago] * (n - 4)
    return pd.DataFrame({"Total Revenue": rev, "Net Income": ni}).T.set_axis(cols, axis=1)


def _ft(wf, monkeypatch, inc_q, bal_q=None, cf_q=None, inc_a=None, cf_a=None, bal_a=None):
    t = _Ticker(
        quarterly_income_stmt=inc_q,
        quarterly_balance_sheet=bal_q,
        quarterly_cash_flow=cf_q,
        income_stmt=inc_a,
        balance_sheet=bal_a,
        cash_flow=cf_a,
    )
    monkeypatch.setattr(wf.yf, "Ticker", lambda s: t)
    return wf.compute_filing_trend("ACME", ".NS")


def test_filing_trend_default_when_no_data(wf, monkeypatch):
    r = _ft(wf, monkeypatch, None)
    assert r == {
        "symbol": "ACME", "rev_streak": 0, "profit_streak": 0, "ocf_streak": 0,
        "debt_streak": 0, "piotroski_trend": 0, "filing_score": 0, "filing_class": "WEAK",
    }
    # fewer than 8 quarters -> still the default
    assert _ft(wf, monkeypatch, _quarters(200, 100, 200, 100, n=7))["filing_score"] == 0

    def boom(t):
        raise RuntimeError("down")

    monkeypatch.setattr(wf.yf, "Ticker", boom)
    assert wf.compute_filing_trend("ACME", ".NS")["filing_class"] == "WEAK"


def test_filing_trend_streaks_and_classes(wf, monkeypatch):
    bal = _fin({"Long Term Debt": [10, 20, 30, 40]}, ["2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30"])
    inc_a = _fin({"Net Income": [10, -5, 1], "Total Assets": [100, 100, 100]}, FY[:3])
    bal_a = _fin({"Total Assets": [100, 100, 100]}, FY[:3])
    r = _ft(wf, monkeypatch, _quarters(200, 100, 200, 100), bal_q=bal, inc_a=inc_a, bal_a=bal_a)
    assert r["rev_streak"] == 4 and r["profit_streak"] == 4
    assert r["debt_streak"] == 3  # 3 comparisons across 4 columns
    assert r["ocf_streak"] == 0  # no cash-flow frame
    assert r["piotroski_trend"] == 2  # score 1 vs 0 -> +1 shifted by 1
    assert r["filing_score"] == 3 + 3 + 0 + 3 + 2 == 11
    assert r["filing_class"] == "STRONG"


def test_filing_trend_class_boundaries(wf, monkeypatch):
    qdates = ["2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30"]
    bal3 = _fin({"Long Term Debt": [10, 20, 30, 40]}, qdates)  # debt streak 3
    bal2 = _fin({"Long Term Debt": [10, 20, 30, 20]}, qdates)  # streak 3? q=2: 30<20 no -> 2
    bal1 = _fin({"Long Term Debt": [10, 20, 20, 20]}, qdates)  # streak 1
    q = _quarters(200, 100, 200, 100)
    # 3 + 3 + 3 = 9 -> STRONG ; 3 + 3 + 2 = 8 -> EMERGING
    assert _ft(wf, monkeypatch, q, bal_q=bal3)["filing_class"] == "STRONG"
    r = _ft(wf, monkeypatch, q, bal_q=bal2)
    assert r["debt_streak"] == 2 and r["filing_score"] == 8 and r["filing_class"] == "EMERGING"
    # revenue only: 3 -> WEAK; plus one debt quarter = 4 -> EMERGING
    q_rev = _quarters(200, 100, 100, 100)
    r = _ft(wf, monkeypatch, q_rev)
    assert r["filing_score"] == 3 and r["filing_class"] == "WEAK"
    r = _ft(wf, monkeypatch, q_rev, bal_q=bal1)
    assert r["debt_streak"] == 1 and r["filing_score"] == 4 and r["filing_class"] == "EMERGING"


def test_filing_trend_growth_thresholds(wf, monkeypatch):
    # revenue growth of exactly 10% is not > 10: no streak
    assert _ft(wf, monkeypatch, _quarters(110, 100, 100, 100))["rev_streak"] == 0
    assert _ft(wf, monkeypatch, _quarters(111, 100, 100, 100))["rev_streak"] == 4
    # profit growth must exceed 15% AND current profit positive
    assert _ft(wf, monkeypatch, _quarters(100, 100, 114, 100))["profit_streak"] == 0
    assert _ft(wf, monkeypatch, _quarters(100, 100, 116, 100))["profit_streak"] == 4
    # negative year-ago base: abs() used, but a still-negative profit does not count
    assert _ft(wf, monkeypatch, _quarters(100, 100, -50, -100))["profit_streak"] == 0
    assert _ft(wf, monkeypatch, _quarters(100, 100, 50, -100))["profit_streak"] == 4
    # zero revenue a year ago breaks the streak
    assert _ft(wf, monkeypatch, _quarters(100, 0, 100, 100))["rev_streak"] == 0


def test_filing_trend_piotroski_trend_clamped(wf, monkeypatch):
    inc_a = _fin({"Net Income": [-5, 10, 1]}, FY[:3])
    bal_a = _fin({"Total Assets": [100, 100, 100]}, FY[:3])
    r = _ft(wf, monkeypatch, _quarters(100, 100, 100, 100), inc_a=inc_a, bal_a=bal_a)
    assert r["piotroski_trend"] == 0  # 0 - 1 + 1
    # fewer than 3 annual columns -> no trend computed
    inc_short = _fin({"Net Income": [10, 1]}, FY[:2])
    r = _ft(wf, monkeypatch, _quarters(100, 100, 100, 100), inc_a=inc_short, bal_a=bal_a)
    assert r["piotroski_trend"] == 0


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="q_val(df, row, col) is called with two row names plus col for the OCF streak, "
    "so any filing with a quarterly cash-flow frame raises TypeError (walk_forward_backtest.py "
    "compute_filing_trend, 'Operating Cash Flow', 'Total Cash From Operating Activities', q)",
)
def test_filing_trend_ocf_streak_with_cash_flow_frame(wf, monkeypatch):
    cf_q = _fin(
        {"Operating Cash Flow": [5, 5, 5, 5, 5, 5, 5, 5]},
        pd.date_range("2025-03-31", periods=8, freq="-3ME"),
    )
    r = _ft(wf, monkeypatch, _quarters(100, 100, 100, 100), cf_q=cf_q)
    assert r["ocf_streak"] == 3  # capped at 3


# ---------------------------------------------------------------- assign_split
def test_assign_split_boundaries(wf):
    cfg = {"test_start": "2022-01-03", "val_start": "2024-01-02", "val_end": "2026-06-30"}
    f = wf.assign_split
    assert f(pd.Timestamp("2022-01-02"), cfg) == "TRAIN"
    assert f(pd.Timestamp("2022-01-03"), cfg) == "TEST"  # test_start itself is TEST
    assert f(pd.Timestamp("2024-01-01"), cfg) == "TEST"
    assert f(pd.Timestamp("2024-01-02"), cfg) == "VAL"
    assert f(pd.Timestamp("2026-06-30"), cfg) == "VAL"  # val_end inclusive
    assert f(pd.Timestamp("2026-07-01"), cfg) == "OUT_OF_RANGE"
    assert f("not a date", cfg) == "OUT_OF_RANGE"
    assert f(pd.Timestamp("2022-06-01"), {}) == "OUT_OF_RANGE"  # missing keys


# ---------------------------------------------------------------- forward_returns
def test_forward_returns_net_of_cost_and_horizons(wf):
    n = 300
    df = _ohlc(n)
    df["Close"] = 100.0 + np.arange(n)  # +1 per bar
    idx_df = _index_df(n=n, start=str(df.index[0].date()))
    idx_df["Close"] = 1000.0 + 10 * np.arange(n)
    fr = wf.forward_returns(df, 10, float(df["Close"].iloc[10]), idx_df, df.index[10])
    entry = 110.0
    assert fr["ret"]["T+1d"] == pytest.approx((111 - entry) / entry * 100 - 0.2)
    assert fr["ret"]["T+21d"] == pytest.approx((131 - entry) / entry * 100 - 0.2)
    bh = (1000 + 10 * 11 - (1000 + 10 * 10)) / (1000 + 10 * 10) * 100
    assert fr["alpha"]["T+1d"] == pytest.approx(fr["ret"]["T+1d"] - bh)
    assert set(fr["ret"]) == set(wf.HORIZONS)


def test_forward_returns_runs_off_the_end_gives_nan(wf):
    df = _ohlc(30)
    idx_df = _index_df(n=30, start=str(df.index[0].date()))
    fr = wf.forward_returns(df, 20, 100.0, idx_df, df.index[20])
    assert not np.isnan(fr["ret"]["T+5d"])  # 25 < 30
    assert np.isnan(fr["ret"]["T+10d"]) and np.isnan(fr["alpha"]["T+10d"])  # 30 >= n
    # boundary: entry_idx + offset == n - 1 is still valid, == n is NaN
    fr = wf.forward_returns(df, 28, 100.0, idx_df, df.index[28])
    assert not np.isnan(fr["ret"]["T+1d"]) and np.isnan(fr["ret"]["T+3d"])


def test_forward_returns_index_edge_cases(wf):
    df = _ohlc(60)
    # index shorter than the horizon: return exists, alpha NaN
    short = _index_df(n=3, start=str(df.index[0].date()))
    fr = wf.forward_returns(df, 0, 100.0, short, df.index[0])
    assert not np.isnan(fr["ret"]["T+5d"]) and np.isnan(fr["alpha"]["T+5d"])
    # signal date before the index starts: si = -1 -> NaN alpha
    idx_df = _index_df(n=60, start="2021-01-01")
    fr = wf.forward_returns(df, 0, 100.0, idx_df, pd.Timestamp("2019-01-01"))
    assert np.isnan(fr["alpha"]["T+1d"])
    # exception inside alpha (bad index frame) is swallowed
    fr = wf.forward_returns(df, 0, 100.0, pd.DataFrame(), df.index[0])
    assert np.isnan(fr["alpha"]["T+1d"]) and not np.isnan(fr["ret"]["T+1d"])
    # tz-aware signal date and index
    aware_idx = _index_df(n=60, start=str(df.index[0].date()))
    aware_idx.index = aware_idx.index.tz_localize("UTC")
    fr = wf.forward_returns(df, 0, 100.0, aware_idx, df.index[0].tz_localize("UTC"))
    assert not np.isnan(fr["alpha"]["T+1d"])


# ---------------------------------------------------------------- compute_stats
def test_compute_stats_values(wf):
    v = pd.Series([10.0, -5.0, 20.0, -10.0, 5.0])
    s = wf.compute_stats(v)
    assert s["N"] == 5
    assert s["Hit_Rate%"] == 60.0
    assert s["Avg_Return%"] == 4.0
    assert s["Median%"] == 5.0
    assert s["Std%"] == round(v.std(), 2)
    assert s["Sharpe"] == round(4.0 / v.std(), 3)
    losses = v[v < 0]
    assert s["Sortino"] == round(4.0 / losses.std(), 3)
    assert s["Profit_Factor"] == round(35 / 15, 2)
    assert s["Max_Win%"] == 20.0 and s["Max_Loss%"] == -10.0
    assert s["p25%"] == -5.0 and s["p75%"] == 10.0
    assert s["EV%"] == round(0.6 * (35 / 3) + 0.4 * (-7.5), 2)
    sv = v.sort_values().reset_index(drop=True)
    cum = (1 + sv / 100).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax() * 100).min()
    assert s["Max_DD%"] == round(dd, 2)
    assert s["Calmar"] == round(4.0 / abs(dd), 3)
    assert "BH_Avg%" not in s


def test_compute_stats_small_and_degenerate(wf):
    assert wf.compute_stats(pd.Series([1.0, 2.0])) == {}  # < 3 observations
    assert wf.compute_stats(pd.Series([1.0, np.nan, 2.0])) == {}  # NaN dropped first
    assert wf.compute_stats(pd.Series([1.0, 2.0, 3.0]))["N"] == 3
    # all winners: infinite profit factor reported as 999, zero drawdown -> calmar 0
    s = wf.compute_stats(pd.Series([1.0, 2.0, 3.0]))
    assert s["Profit_Factor"] == 999 and s["Calmar"] == 0 and s["Max_DD%"] == 0
    assert s["Hit_Rate%"] == 100.0
    # constant returns: zero std -> zero Sharpe
    s = wf.compute_stats(pd.Series([2.0, 2.0, 2.0]))
    assert s["Sharpe"] == 0 and s["Std%"] == 0
    # zero return is neither win nor loss
    s = wf.compute_stats(pd.Series([0.0, 0.0, 5.0]))
    assert s["Hit_Rate%"] == pytest.approx(33.3)
    # a single loser: downside std falls back to ~0 (1e-9 floor) giving a huge Sortino
    s = wf.compute_stats(pd.Series([5.0, 6.0, -1.0]))
    assert s["Sortino"] > 1e6


def test_compute_stats_benchmark(wf):
    v = pd.Series([1.0, 2.0, 3.0, 4.0])
    s = wf.compute_stats(v, bh_vals=pd.Series([0.5, 0.5, 0.5]))
    assert s["BH_Avg%"] == 0.5 and s["Alpha_Avg%"] == 2.0
    s = wf.compute_stats(v, bh_vals=pd.Series([0.5, 0.5]))  # < 3 benchmark points ignored
    assert "BH_Avg%" not in s
    s = wf.compute_stats(v, bh_vals=pd.Series([0.5, np.nan, 0.5, 0.5]))
    assert s["BH_Avg%"] == 0.5


# ---------------------------------------------------------------- analyze_all / overfitting / matrix
def _signals(n_train=4, n_val=4, with_filing=True, ret=None):
    rows = []
    for split, n, base in (("TRAIN", n_train, 5.0), ("VAL", n_val, 1.0)):
        for k in range(n):
            r = {
                "screener": "darvas", "split": split, "regime": "BULL",
                "ret_T+1d": base + k, "alpha_T+1d": base + k - 0.5,
                "ret_T+5d": np.nan,
            }
            if with_filing:
                r["filing_class"] = "STRONG"
            rows.append(r)
    return pd.DataFrame(rows)


def test_analyze_all_empty_and_missing_columns(wf):
    assert wf.analyze_all(pd.DataFrame(), "P").empty
    # too few rows in every cell
    assert wf.analyze_all(_signals(2, 0), "P").empty
    # no ret columns at all
    df = _signals().drop(columns=["ret_T+1d", "ret_T+5d"])
    assert wf.analyze_all(df, "P").empty


def test_analyze_all_cells_and_flags(wf):
    out = wf.analyze_all(_signals(), "Period X")
    assert set(out["Horizon"]) == {"T+1d"}  # T+5d is all-NaN
    assert set(out["Screener"]) == {"darvas"} and set(out["Period"]) == {"Period X"}
    key = out.set_index(["Split", "Regime", "FilingClass"])
    train = key.loc[("TRAIN", "BULL", "STRONG")]
    assert train["N"] == 4 and train["Avg_Return%"] == 6.5
    assert train["Stat_Flag"] == "⚠️ LOW N"  # N < 20
    assert train["Alpha_Avg%"] == 0.5  # mean(ret) - mean(alpha)
    allrow = key.loc[("ALL", "ALL", "ALL")]
    assert allrow["N"] == 8
    # filtering dimensions: TRAIN x ALL regime x ALL filing also present
    assert ("TRAIN", "ALL", "ALL") in key.index
    # no TEST rows (no TEST signals)
    assert "TEST" not in set(out["Split"])
    # without the filing_class column only the ALL class is used
    out2 = wf.analyze_all(_signals(with_filing=False), "P")
    assert set(out2["FilingClass"]) == {"ALL"}


def test_analyze_all_ok_flag_at_20(wf):
    out = wf.analyze_all(_signals(n_train=20, n_val=19), "P")
    key = out.set_index(["Split", "Regime", "FilingClass"])
    assert key.loc[("TRAIN", "BULL", "STRONG")]["Stat_Flag"] == "OK"
    assert key.loc[("VAL", "BULL", "STRONG")]["Stat_Flag"] == "⚠️ LOW N"


def _stats_row(split, sharpe, ev=1.0, n=10, sc="darvas", regime="BULL", hz="T+1d", period="P"):
    return {
        "Period": period, "Screener": sc, "Split": split, "Regime": regime, "FilingClass": "ALL",
        "Horizon": hz, "Sharpe": sharpe, "EV%": ev, "N": n, "Hit_Rate%": 55.0,
        "Avg_Return%": 2.0, "Stat_Flag": "OK",
    }


def test_overfitting_report_risk_bands(wf):
    assert wf.overfitting_report(pd.DataFrame()).empty
    rows = []
    # decay = (train - val)/|train| * 100
    for hz, tr, va in [("T+1d", 1.0, 0.4), ("T+3d", 1.0, 0.6), ("T+5d", 1.0, 0.8),
                       ("T+10d", 1.0, 0.5), ("T+21d", 1.0, 0.8001), ("T+63d", 1.0, 1.2)]:
        rows.append(_stats_row("TRAIN", tr, hz=hz))
        rows.append(_stats_row("VAL", va, hz=hz))
    rows.append(_stats_row("TEST", 0.7, hz="T+1d"))
    out = wf.overfitting_report(pd.DataFrame(rows)).set_index("Horizon")
    assert out.loc["T+1d", "Sharpe_Decay%"] == 60.0 and out.loc["T+1d", "Overfit_Risk"] == "HIGH"
    assert out.loc["T+1d", "Sharpe_TEST"] == 0.7
    assert out.loc["T+3d", "Overfit_Risk"] == "MEDIUM"  # 40%
    assert out.loc["T+5d", "Sharpe_Decay%"] == pytest.approx(20.0)
    assert out.loc["T+10d", "Overfit_Risk"] == "MEDIUM"  # exactly 50 is not > 50
    assert out.loc["T+21d", "Overfit_Risk"] == "LOW"
    assert out.loc["T+63d", "Overfit_Risk"] == "LOW" and out.loc["T+63d", "Sharpe_Decay%"] == -20.0
    assert pd.isna(out.loc["T+3d", "Sharpe_TEST"])


def test_overfitting_report_skips_missing_train_or_val_and_zero_train(wf):
    df = pd.DataFrame([_stats_row("TRAIN", 1.0, hz="T+1d")])  # no VAL
    assert wf.overfitting_report(df).empty
    df = pd.DataFrame([_stats_row("VAL", 1.0)])
    assert wf.overfitting_report(df).empty
    df = pd.DataFrame([_stats_row("TRAIN", 0.0), _stats_row("VAL", 0.5)])
    out = wf.overfitting_report(df)
    assert pd.isna(out["Sharpe_Decay%"].iloc[0]) and out["Overfit_Risk"].iloc[0] == "LOW"


def test_strategy_matrix_picks_best_ev_with_min_n(wf):
    assert wf.strategy_matrix(pd.DataFrame()).empty
    df = pd.DataFrame(
        [
            _stats_row("VAL", 1, ev=3.0, n=10, sc="darvas"),
            _stats_row("VAL", 1, ev=5.0, n=10, sc="piotroski"),
            _stats_row("ALL", 1, ev=9.0, n=4, sc="coffee_can"),  # N < 5 excluded
            _stats_row("ALL", 1, ev=6.0, n=5, sc="golden_cross"),  # N == 5 included
            _stats_row("TRAIN", 1, ev=99.0, n=50, sc="magic_formula"),  # wrong split
            _stats_row("VAL", 1, ev=2.0, n=10, sc="darvas", regime="BEAR", hz="T+5d"),
        ]
    )
    m = wf.strategy_matrix(df)
    bull = m[(m["Regime"] == "BULL") & (m["Horizon"] == "T+1d")].iloc[0]
    assert bull["Best_Screener"] == "golden_cross" and bull["EV%"] == 6.0
    assert bull["N_Signals"] == 5
    bear = m[m["Regime"] == "BEAR"].iloc[0]
    assert bear["Best_Screener"] == "darvas" and bear["Horizon"] == "T+5d"
    # sorted by Period, Regime, Horizon
    assert list(m["Regime"]) == sorted(m["Regime"])


# ---------------------------------------------------------------- save_excel / print_summary
def _report_frames(wf):
    stats = pd.DataFrame(
        [
            _stats_row("VAL", 0.5, ev=3.0, n=10, sc="darvas", period="3-Year (2023–2026)"),
            _stats_row("TRAIN", 1.0, ev=2.0, n=10, sc="darvas", period="3-Year (2023–2026)"),
            _stats_row("VAL", 0.2, ev=1.0, n=10, sc="golden_cross", hz="T+3d",
                       period="3-Year (2023–2026)"),
        ]
    )
    overfit = wf.overfitting_report(stats)
    matrix = wf.strategy_matrix(stats)
    sigs = pd.DataFrame(
        {"screener": ["darvas", "darvas"], "signal_date": pd.to_datetime(["2024-01-02", "2023-01-02"]),
         "symbol": ["A", "B"]}
    )
    filing = pd.DataFrame({"symbol": ["A", "B"], "filing_score": [3, 9]})
    return stats, overfit, matrix, sigs, filing


def test_save_excel_sheets(wf, tmp_path):
    stats, overfit, matrix, sigs, filing = _report_frames(wf)
    path = wf.save_excel(stats, overfit, matrix, sigs, filing)
    assert path.parent == wf.OUT_DIR and path.name.startswith("walk_forward_")
    x = pd.ExcelFile(path)
    names = x.sheet_names
    assert names[0] == "DISCLAIMER"
    for n in ("Strategy_Matrix", "Overfitting_Check", "HitRate_Heatmap_VAL", "EV_Heatmap_VAL",
              "Filing_Trends", "All_Signals"):
        assert n in names
    assert "Stats_3-Year_2023_2026" in names  # parentheses/spaces/en-dash stripped from the period
    ft = x.parse("Filing_Trends")
    assert list(ft["filing_score"]) == [9, 3]  # sorted descending
    al = x.parse("All_Signals")
    assert list(al["symbol"]) == ["B", "A"]  # sorted by screener then signal_date
    disc = x.parse("DISCLAIMER")
    assert disc["DISCLAIMER"].iloc[0] == wf.DISCLAIMER


def test_save_excel_skips_empty_sections(wf):
    path = wf.save_excel(*(pd.DataFrame(),) * 5)
    assert pd.ExcelFile(path).sheet_names == ["DISCLAIMER"]
    # TRAIN-only stats: no VAL rows so no heatmaps
    stats, overfit, matrix, sigs, filing = _report_frames(wf)
    only_train = stats[stats["Split"] == "TRAIN"]
    path = wf.save_excel(only_train, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    names = pd.ExcelFile(path).sheet_names
    assert "HitRate_Heatmap_VAL" not in names and "EV_Heatmap_VAL" not in names
    assert any(n.startswith("Stats_") for n in names)


def test_print_summary_empty_and_full(wf, capsys):
    wf.print_summary(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert "No results to display." in capsys.readouterr().out
    stats, overfit, matrix, _s, _f = _report_frames(wf)
    matrix.loc[matrix.index[0], "Stat_Flag"] = "⚠️ LOW N"
    wf.print_summary(matrix, overfit, stats)
    out = capsys.readouterr().out
    assert "STRATEGY RECOMMENDATION MATRIX" in out and "OVERFITTING CHECK" in out
    assert "⚠️ LOW N" in out
    assert "• BULL" in out and "darvas" in out
    assert "• BEAR" not in out  # no BEAR rows -> no key finding line
    # empty overfit table skips the rows but not the findings
    wf.print_summary(matrix, pd.DataFrame(), stats)
    assert "Key findings" in capsys.readouterr().out


# ---------------------------------------------------------------- run_period
def _cfg(df, test_i=60, val_i=80, val_end_i=125):
    idx = df.index
    return {
        "label": "TestPeriod", "start": str(idx[0].date()),
        "test_start": str(idx[test_i].date()), "val_start": str(idx[val_i].date()),
        "val_end": str(idx[val_end_i].date()),
    }


def test_run_period_technical_screeners(wf, monkeypatch):
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: [])
    df = _darvas_df(drift=0.0)
    idx_df = _index_df(n=400, start="2019-06-03", close=110.0)
    cfg = _cfg(df)
    out = wf.run_period(cfg, {"AAA.NS": df}, idx_df, [("AAA", ".NS")],
                        {"AAA": {"filing_class": "STRONG", "filing_score": 11}}, 2, True)
    assert list(out["screener"]) == ["darvas", "darvas"]
    assert list(out["symbol"]) == ["AAA", "AAA"]
    assert list(out["split"]) == ["TRAIN", "VAL"]
    assert list(out["regime"]) == ["BULL", "BULL"]
    assert set(out["filing_class"]) == {"STRONG"} and set(out["filing_score"]) == {11}
    assert out["entry_price"].tolist() == [13.0, 13.0]
    assert "ret_T+1d" in out.columns and "alpha_T+252d" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["signal_date"])
    # T+63d from bar 100 runs off the end of a 130-bar frame
    assert pd.isna(out["ret_T+63d"].iloc[1]) and not pd.isna(out["ret_T+63d"].iloc[0])


def test_run_period_trims_to_period_and_filters(wf, monkeypatch):
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: [])
    df = _darvas_df()
    cfg = _cfg(df)
    # a period window with < 50 bars yields nothing
    short = dict(cfg, start=str(df.index[100].date()))
    out = wf.run_period(short, {"AAA.NS": df}, _index_df(), [("AAA", ".NS")], {}, 1, False)
    assert out.empty
    # signals after val_end are OUT_OF_RANGE and dropped; default filing info is N/A / 0
    late = dict(cfg, val_end=str(df.index[60].date()))
    out = wf.run_period(late, {"AAA.NS": df}, _index_df(), [("AAA", ".NS")], {}, 1, False)
    assert list(out["split"]) == ["TRAIN"] and out["filing_class"].iloc[0] == "N/A"
    assert out["filing_score"].iloc[0] == 0


def test_run_period_fundamental_screeners_and_failures(wf, monkeypatch):
    df = _ohlc(130)
    idx_df = _index_df(n=400, start="2019-06-03")
    cfg = _cfg(df)
    calls = []

    def fs(sym, suffix, sc, ohlc_p):
        calls.append((sym, suffix, sc, len(ohlc_p)))
        if sym == "BAD":
            raise RuntimeError("worker blew up")
        if sym == "NODATA" or sc != "piotroski":
            return []
        return [
            {"date": df.index[30], "entry": 100.0, "idx": 30, "col_offset": 0},
            {"date": df.index[125] + pd.Timedelta(days=400), "entry": 100.0, "idx": 31},  # out of range
        ]

    monkeypatch.setattr(wf, "fundamental_signals", fs)
    ohlc_map = {"GOOD.NS": df, "BAD.NS": df, "TINY.NS": df.iloc[:10]}
    syms = [("GOOD", ".NS"), ("BAD", ".NS"), ("TINY", ".NS"), ("MISSING", ".NS")]
    out = wf.run_period(cfg, ohlc_map, idx_df, syms, {}, 2, False)
    pio = out[out["screener"] == "piotroski"]
    assert list(pio["symbol"]) == ["GOOD"] and list(pio["split"]) == ["TRAIN"]
    # TINY (<20 bars in window) and MISSING (no ohlc) never reach fundamental_signals
    assert {c[0] for c in calls} == {"GOOD", "BAD"}
    assert {c[2] for c in calls} == {"piotroski", "coffee_can", "magic_formula", "bull_cartel"}


def test_run_period_progress_cadence(wf, monkeypatch, capsys):
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: [])
    df = _ohlc(130)
    syms = [(f"S{i}", ".NS") for i in range(200)]
    wf.run_period(_cfg(df), {}, _index_df(), syms, {}, 4, False)
    out = capsys.readouterr().out
    # printed at done == 200 (a multiple of 200 and the total), once per screener
    assert out.count("200/200 stocks") == 4
    assert "0 total signals" in out


# ---------------------------------------------------------------- main
def _main_setup(wf, monkeypatch, filings=True):
    base = _darvas_df(drift=0.0)
    ohlc_map = {}
    for k in range(4):
        d = _darvas_df(drift=0.005 * k)
        ohlc_map[f"T{k}.NS"] = d
    cfg = _cfg(base)
    cfg["label"] = "Mini"
    monkeypatch.setattr(wf, "PERIODS", {"3y": cfg})
    monkeypatch.setattr(wf, "get_nse_symbols", lambda top=0: [(f"T{k}", ".NS") for k in range(4)][: top or 4])
    monkeypatch.setattr(wf, "fetch_ohlc_bulk", lambda tickers, period="10y": (
        wf._seen.update(period=period, tickers=list(tickers)) or ohlc_map))
    monkeypatch.setattr(wf, "fetch_index", lambda period="10y": _index_df(n=400, start="2019-06-03"))
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: [])
    monkeypatch.setattr(
        wf, "compute_filing_trend",
        lambda s, sfx: {"symbol": s, "filing_class": "STRONG", "filing_score": 10},
    )
    wf._seen = {}
    return ohlc_map


def test_main_end_to_end(wf, monkeypatch, capsys):
    _main_setup(wf, monkeypatch)
    res = wf.main(periods=["3y"], top=0, workers=2, run_filings=True)
    assert set(res) == {"signals", "stats", "matrix", "overfit", "filing", "excel"}
    assert wf._seen["period"] == "3y"  # only the 3y window requested
    assert wf._seen["tickers"] == ["T0.NS", "T1.NS", "T2.NS", "T3.NS"]
    sig = res["signals"]
    assert len(sig) == 8 and set(sig["split"]) == {"TRAIN", "VAL"}
    assert set(sig["filing_class"]) == {"STRONG"}
    assert len(res["filing"]) == 4
    assert not res["stats"].empty and not res["matrix"].empty
    assert pd.ExcelFile(res["excel"]).sheet_names[0] == "DISCLAIMER"
    out = capsys.readouterr().out
    assert "4/4 done — 4 STRONG trend stocks" in out
    assert "4 symbols" in out and "4 tickers with data" in out
    assert "WALK-FORWARD BACKTEST — STRATEGY RECOMMENDATION MATRIX" in out


def test_main_skip_filings_top_and_period_choice(wf, monkeypatch, capsys):
    _main_setup(wf, monkeypatch)
    res = wf.main(periods=["3y"], top=2, workers=1, run_filings=False)
    assert wf._seen["tickers"] == ["T0.NS", "T1.NS"]
    assert res["filing"].empty and set(res["signals"]["filing_class"]) == {"N/A"}
    assert "Filing trends skipped (--no-filings)" in capsys.readouterr().out
    # period selection chooses the longest download window needed
    cfg = wf.PERIODS["3y"]
    monkeypatch.setattr(wf, "PERIODS", {"3y": cfg, "5y": cfg, "10y": cfg})
    wf.main(periods=["3y", "5y"], run_filings=False)
    assert wf._seen["period"] == "5y"
    wf.main(periods=None, run_filings=False)
    assert wf._seen["period"] == "10y"


def test_main_liquid_override_and_live_context(wf, monkeypatch, capsys):
    _main_setup(wf, monkeypatch)
    monkeypatch.setattr(wf, "_LIQUID_OVERRIDE", [("T0", ".NS"), ("T1", ".NS")], raising=False)
    monkeypatch.setattr(
        wf, "get_nse_symbols", lambda top=0: (_ for _ in ()).throw(AssertionError("unused"))
    )
    live = {"n": 0}

    class F:
        def print_live_context(self):
            live["n"] += 1
            raise RuntimeError("context failed")  # must be swallowed

    monkeypatch.setattr(wf, "_USE_NSE_FETCHER", True)
    monkeypatch.setattr(wf, "_NSE_FETCHER", F())
    wf.main(periods=["3y"], run_filings=False)
    assert live["n"] == 1
    assert wf._seen["tickers"] == ["T0.NS", "T1.NS"]
    assert "2 liquid symbols" in capsys.readouterr().out


def test_main_no_signals(wf, monkeypatch, capsys):
    _main_setup(wf, monkeypatch)
    monkeypatch.setattr(wf, "fetch_ohlc_bulk", lambda tickers, period="10y": {})
    res = wf.main(periods=["3y"], run_filings=False)
    assert res["signals"].empty and res["stats"].empty and res["matrix"].empty
    out = capsys.readouterr().out
    assert "No signals for period 3y" in out and "No results to display." in out


def test_main_filing_worker_exception_is_isolated(wf, monkeypatch):
    _main_setup(wf, monkeypatch)

    def flaky(s, sfx):
        if s == "T1":
            raise RuntimeError("boom")
        return {"symbol": s, "filing_class": "WEAK", "filing_score": 1}

    monkeypatch.setattr(wf, "compute_filing_trend", flaky)
    res = wf.main(periods=["3y"], run_filings=True, workers=2)
    assert sorted(res["filing"]["symbol"]) == ["T0", "T2", "T3"]


# ------------------------------------------------------------------ mutation follow-ups
def test_regime_boundaries_are_exact(wf):
    d = pd.Timestamp("2020-03-02")
    assert wf.classify_regime(d, _index_df(close=101.5, dma=100, slope=1)) == "BULL"  # exactly 1.5% above: trend wins
    assert wf.classify_regime(d, _index_df(close=98.5, dma=100, slope=-1)) == "BEAR"
    assert wf.classify_regime(d, _index_df(close=90, dma=100, slope=0)) == "SIDEWAYS"  # below the DMA but flat slope


def test_darvas_equal_volume_and_prior_close_at_box_top_still_signal(wf):
    df = _darvas_df(second=False)
    df.iloc[35, df.columns.get_loc("Volume")] = 100 * 1.1  # exactly the 1.1x threshold: >= passes
    assert [s["idx"] for s in wf.darvas_signals(df)] == [35]
    df = _darvas_df(second=False)
    df.iloc[34, df.columns.get_loc("Close")] = 12.0  # prior close equal to the box top still counts as "below or at"
    assert [s["idx"] for s in wf.darvas_signals(df)] == [35]


def test_golden_cross_needs_strict_inversion_and_205_bars(wf):
    close = np.concatenate([np.full(250, 100.0), [300.0]])  # DMAs are exactly equal the bar before the jump
    df = _ohlc(len(close))
    df["Close"] = close
    assert wf.golden_cross_signals(df) == []
    close = np.concatenate([np.linspace(100, 90, 200), [200, 300, 400, 500]])  # a real cross at bar 201, 204 bars
    df = _ohlc(204)
    df["Close"] = close
    assert wf.golden_cross_signals(df) == []  # too short: needs 205 bars
    df = _ohlc(205)
    df["Close"] = np.concatenate([close, [500.0]])
    assert [s["idx"] for s in wf.golden_cross_signals(df)] == [201]


def test_magic_formula_roic_bar_is_15_not_25(wf, monkeypatch):
    cr = 1e7
    inc = _fin({"EBIT": [30 * cr] * 4, "Net Income": [10 * cr] * 4}, FY)  # ROIC 30/150 = 20%
    bal = _fin({"Total Assets": [200 * cr] * 4, "Current Liabilities": [50 * cr] * 4}, FY)
    info = {"totalDebt": 100 * cr, "totalCash": 50 * cr, "bookValue": 10}
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, None, None, info, 100.5 * cr))
    assert len(wf.fundamental_signals("ACME", ".NS", "magic_formula", _wf_ohlc())) == 3


def test_fiscal_year_ending_in_july_signals_same_year(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    jul = ["2024-07-31", "2023-07-31", "2022-07-31", "2021-07-31"]
    for df in (inc, bal, cf):
        df.columns = pd.to_datetime(jul)
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    sigs = wf.fundamental_signals("ACME", ".NS", "piotroski", _wf_ohlc())
    assert sigs[0]["date"] == pd.Timestamp("2024-07-01")  # month 7 is NOT 'after July'


def test_entry_bar_needs_five_prior_bars(wf, monkeypatch):
    inc, bal, cf = _piotroski_statements(True)
    for df in (inc, bal, cf):
        df.columns = pd.to_datetime(["2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31"])
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, cf))
    ohlc = _ohlc(200, start="2024-06-25")  # 1 Jul 2024 is bar 4 of this history
    assert wf.fundamental_signals("ACME", ".NS", "piotroski", ohlc) == []
    ohlc = _ohlc(200, start="2024-06-24")  # bar 5: accepted
    assert [s["idx"] for s in wf.fundamental_signals("ACME", ".NS", "piotroski", ohlc)] == [5]


def test_only_available_annual_columns_are_tried(wf, monkeypatch):
    inc, bal = _coffee_statements()
    inc, bal = inc.iloc[:, :3], bal.iloc[:, :3]  # three fiscal years -> offsets 0 and 1 only
    monkeypatch.setattr(wf.yf, "Ticker", lambda t: _tk(inc, bal, None, None, {"debtToEquity": 50}, 5e9))
    assert [s["col_offset"] for s in wf.fundamental_signals("ACME", ".NS", "coffee_can", _wf_ohlc())] == [0, 1]


def test_filing_trend_streak_window_is_six_quarters(wf, monkeypatch):
    cols = pd.date_range("2025-03-31", periods=12, freq="-3ME")
    rev = [800.0] * 4 + [400.0] * 4 + [200.0] * 4  # every quarter doubles its year-ago level
    q = pd.DataFrame({"Total Revenue": rev, "Net Income": rev}).T.set_axis(cols, axis=1)
    r = _ft(wf, monkeypatch, q)
    assert r["rev_streak"] == 6 and r["profit_streak"] == 6  # min(6, 12 - 4) = 6 comparisons, score caps at 3 each
    assert r["filing_score"] == 6


def test_overfit_risk_band_edges_are_exclusive(wf):
    rows = [_stats_row("TRAIN", 5.0), _stats_row("VAL", 4.0)]  # decay is exactly 20.0
    assert wf.overfitting_report(pd.DataFrame(rows)).iloc[0]["Overfit_Risk"] == "LOW"
    rows = [_stats_row("TRAIN", 2.0), _stats_row("VAL", 1.0)]  # decay exactly 50.0
    assert wf.overfitting_report(pd.DataFrame(rows)).iloc[0]["Overfit_Risk"] == "MEDIUM"


def test_run_period_window_and_history_minimums(wf, monkeypatch):
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: [])
    idx_df = _index_df(n=400, start="2019-06-03", close=110.0)

    def run(n):
        df = _darvas_df(n=n, second=False)
        cfg = {"label": "L", "start": str(df.index[0].date()), "test_start": str(df.index[20].date()),
               "val_start": str(df.index[30].date()), "val_end": str(df.index[-1].date())}
        return wf.run_period(cfg, {"AAA.NS": df}, idx_df, [], {}, 1, False)

    assert run(49).empty  # fewer than 50 bars in the window: skipped
    assert list(run(50)["symbol"]) == ["AAA"]
    calls = []
    monkeypatch.setattr(wf, "fundamental_signals", lambda *a, **k: calls.append(1) or [])
    for n, expected in ((19, 0), (20, 4)):
        calls.clear()
        df = _ohlc(n, start="2020-01-01")
        cfg = {"label": "L", "start": "2020-01-01", "test_start": "2020-02-01", "val_start": "2020-03-01", "val_end": "2020-12-31"}
        wf.run_period(cfg, {"AAA.NS": df}, idx_df, [("AAA", ".NS")], {}, 1, False)
        assert len(calls) == expected  # 4 fundamental screeners, only for >= 20 bars


def test_fetch_ohlc_bulk_batches_five_year_downloads_by_100(wf, monkeypatch):
    seen = []
    monkeypatch.setattr(wf.yf, "download", lambda batch, **k: seen.append(len(batch)) or pd.DataFrame())
    wf.fetch_ohlc_bulk([f"T{i}.NS" for i in range(250)], period="5y")
    assert seen == [100, 100, 50]
