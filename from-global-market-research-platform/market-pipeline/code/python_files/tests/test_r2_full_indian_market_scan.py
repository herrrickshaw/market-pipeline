"""Offline tests for full_indian_market_scan (NSE/BSE universe, bhavcopy OHLC, fundamentals, Excel, main)."""

import importlib
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ind(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # module creates ./indian_full_scan at import
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: pd.DataFrame()
    fake_yf.Ticker = lambda t: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    nsep = types.ModuleType("nsepython")
    nsep.get_bhavcopy = lambda d: pd.DataFrame()
    nsep.nse_eq_symbols = lambda: []
    monkeypatch.setitem(sys.modules, "nsepython", nsep)
    bse = types.ModuleType("bseindia")
    lib = types.ModuleType("bseindia.libutil")
    lib.SECURITY_MASTER_URL = "https://bse.example/master"
    bse.libutil = lib
    monkeypatch.setitem(sys.modules, "bseindia", bse)
    monkeypatch.setitem(sys.modules, "bseindia.libutil", lib)
    monkeypatch.setitem(sys.modules, "nse_data_fetcher", None)
    monkeypatch.setitem(sys.modules, "bhavcopy_fetcher", None)
    monkeypatch.setitem(sys.modules, "fundamentals_store_reader", None)
    sys.modules.pop("full_indian_market_scan", None)
    mod = importlib.import_module("full_indian_market_scan")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "indian_full_scan")
    monkeypatch.setattr(mod, "_SYMBOL_CACHE", tmp_path / "indian_full_scan" / ".symbols_cache.json")
    monkeypatch.setattr(mod, "SLEEP_BETWEEN", 0.0)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access in test")))
    yield mod
    sys.modules.pop("full_indian_market_scan", None)


def _ohlc(n=10, start=10.0):
    c = np.linspace(start, start + n - 1, n)
    return pd.DataFrame({"Close": c, "High": c + 0.5, "Low": c - 0.5, "Open": c, "Volume": 1000.0})


def test_import_flags_without_optional_modules(ind):
    assert ind._USE_NSE_FETCHER is False and ind._BHAV_OK is False
    assert ind._bq_fields is not None


# ------------------------------------------------------------------ symbol cache
def test_symbol_cache_roundtrip(ind):
    assert ind._load_symbol_cache() == (None, None)
    ind._save_symbol_cache(["TCS"], ["500001"])
    assert ind._load_symbol_cache() == (["TCS"], ["500001"])
    ind._SYMBOL_CACHE.write_text(json.dumps({"ts": time.time() - ind.SYMBOL_CACHE_TTL - 1, "nse": ["X"]}))
    assert ind._load_symbol_cache() == (None, None)
    ind._SYMBOL_CACHE.write_text("{")
    assert ind._load_symbol_cache() == (None, None)
    ind._SYMBOL_CACHE.write_text(json.dumps({"ts": time.time()}))
    assert ind._load_symbol_cache() == ([], [])


def test_save_symbol_cache_swallows(ind, monkeypatch, tmp_path):
    monkeypatch.setattr(ind, "_SYMBOL_CACHE", tmp_path / "nope" / "c.json")
    ind._save_symbol_cache([], [])


# ------------------------------------------------------------------ NSE / BSE universe
def test_fetch_nse_symbols_primary_source(ind, monkeypatch):
    syms = [f"S{i}" for i in range(150)][::-1]
    monkeypatch.setattr(sys.modules["nsepython"], "nse_eq_symbols", lambda: syms)
    out = ind.fetch_nse_symbols()
    assert out == sorted(syms) and len(out) == 150


def test_fetch_nse_symbols_small_list_falls_back_to_bhavcopy(ind, monkeypatch, capsys):
    nsep = sys.modules["nsepython"]
    nsep.nse_eq_symbols = lambda: ["A"] * 100  # needs >100 entries
    calls = []

    def bhav(d):
        calls.append(d)
        if len(calls) == 1:
            return pd.DataFrame()
        if len(calls) == 2:
            raise RuntimeError("holiday")
        if len(calls) == 3:
            return None
        return pd.DataFrame({" SERIES": ["EQ", "BE", "EQ", " EQ "], "SYMBOL ": [" TCS ", "JUNK", None, "INFY"]})
    monkeypatch.setattr(ind, "_nse_get_bhavcopy", bhav)
    out = ind.fetch_nse_symbols()
    assert out == ["INFY", "TCS"] and len(calls) == 4
    assert "nsepython bhavcopy" in capsys.readouterr().out


def test_fetch_nse_symbols_all_sources_fail(ind, monkeypatch, capsys):
    def boom():
        raise RuntimeError("nse down")
    sys.modules["nsepython"].nse_eq_symbols = boom
    monkeypatch.setattr(ind, "_nse_get_bhavcopy", lambda d: pd.DataFrame({"X": [1]}))  # no SERIES/SYMBOL columns
    assert ind.fetch_nse_symbols() == []
    out = capsys.readouterr().out
    assert "nse_eq_symbols failed: nse down" in out and "Could not fetch NSE symbol list" in out


def test_bse_session_sets_headers_and_tolerates_failure(ind, monkeypatch):
    got = []
    monkeypatch.setattr(ind.requests.Session, "get", lambda self, url, timeout: got.append(url))
    s = ind._bse_session()
    assert "Mozilla" in s.headers["User-Agent"] and s.headers["Referer"] == "https://www.bseindia.com/" and got
    monkeypatch.setattr(ind.requests.Session, "get", lambda self, url, timeout: (_ for _ in ()).throw(RuntimeError("x")))
    assert ind._bse_session() is not None


BSE_CSV = ("Security Code,Issuer Name,Security Id,Security Name,Status,Group,Face Value,ISIN No,Instrument,extra\n"
           "500001,A,AAA,A Ltd,Active,A,1,INE1,Equity,x\n"
           "500002,B,BBB,B Ltd,Suspended,A,1,INE2,Equity,x\n"
           "500003,C,TCS,TCS Ltd,Active,A,1,INE3,Equity,x\n"
           "500004,D,DDD,D Ltd,Active,A,1,INE4,Debentures,x\n"
           "500005,E, EEE ,E Ltd,Active,A,1,INE5,Equity,x\n")


def test_fetch_bse_only_symbols_filters(ind, monkeypatch):
    monkeypatch.setattr(ind, "_bse_session", lambda: types.SimpleNamespace(get=lambda url, params, timeout: types.SimpleNamespace(text=BSE_CSV)))
    assert ind.fetch_bse_only_symbols({"TCS"}) == ["AAA", "EEE"]


def test_fetch_bse_only_symbols_retries_then_gives_up(ind, monkeypatch, capsys):
    attempts = {"n": 0}

    def get(url, params, timeout):
        attempts["n"] += 1
        return types.SimpleNamespace(text="   ")
    monkeypatch.setattr(ind, "_bse_session", lambda: types.SimpleNamespace(get=get))
    assert ind.fetch_bse_only_symbols(set()) == []
    assert attempts["n"] == 3 and "Could not fetch BSE symbol list" in capsys.readouterr().out


# ------------------------------------------------------------------ OHLC
def test_bulk_download_ohlc_batches(ind, monkeypatch, capsys):
    monkeypatch.setattr(ind, "BATCH_SIZE", 2)
    cols = pd.MultiIndex.from_product([["Close", "High"], ["A.NS", "B.NS"]])
    good = pd.DataFrame(np.arange(40.0).reshape(10, 4), columns=cols)
    calls = {"n": 0}

    def dl(batch, **kw):
        calls["n"] += 1
        assert kw["period"] == "3mo"
        if calls["n"] == 1:
            return good
        if calls["n"] == 2:
            return pd.DataFrame()
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return pd.DataFrame({"Close": [1.0, 2.0]})
    ind.yf.download = dl
    out = ind.bulk_download_ohlc(["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS", "G.NS"])
    assert set(out) == {"A.NS", "B.NS", "G.NS"}
    text = capsys.readouterr().out
    assert "4 batches" in text and "empty" in text and "ERROR — boom" in text and "OK (2 usable)" in text


def _hist(symbols, n=20):
    rows = []
    for s in symbols:
        for i in range(n):
            rows.append({"symbol": s, "d": f"2026-01-{i + 1:02d}", "open": 10.0 + i, "high": 11.0 + i, "low": 9.0 + i,
                         "close": 10.5 + i, "volume": 100.0})
    return pd.DataFrame(rows)


def test_bulk_ohlc_without_bhavcopy_delegates(ind, monkeypatch):
    seen = {}
    monkeypatch.setattr(ind, "bulk_download_ohlc", lambda t, period: seen.update(t=t, p=period) or {"X": 1})
    assert ind.bulk_ohlc(["A.NS"], period="1y") == {"X": 1} and seen == {"t": ["A.NS"], "p": "1y"}
    monkeypatch.setattr(ind, "_BHAV_OK", True)
    monkeypatch.setattr(ind, "USE_BHAVCOPY", False)
    assert ind.bulk_ohlc(["A.NS"]) == {"X": 1}


def test_bulk_ohlc_bhavcopy_with_fallback(ind, monkeypatch, capsys):
    monkeypatch.setattr(ind, "_BHAV_OK", True)
    bh = types.SimpleNamespace(build_history=lambda days: (pd.concat([_hist(["TCS", "INFY"]), _hist(["SHORT"], n=7)]),
                                                           {"symbols": 3, "days": 20, "downloaded": 1, "fetch_secs": 0.1}))
    monkeypatch.setattr(ind, "_bhav", bh, raising=False)
    fb = {}
    monkeypatch.setattr(ind, "bulk_download_ohlc", lambda t, period: fb.update(t=t, p=period) or {x: _ohlc() for x in t})
    out = ind.bulk_ohlc(["TCS.NS", "INFY.NS", "SHORT.NS", "NOTIN.NS", "500001.BO"], period="1y")
    assert set(out) == {"TCS.NS", "INFY.NS", "SHORT.NS", "NOTIN.NS", "500001.BO"}
    assert fb["t"] == ["500001.BO", "SHORT.NS", "NOTIN.NS"]  # BSE first, then NSE names not in (or too short for) the cache
    assert list(out["TCS.NS"].columns) == ["Open", "High", "Low", "Close", "Volume"] and len(out["TCS.NS"]) == 20
    assert isinstance(out["TCS.NS"].index, pd.DatetimeIndex)
    text = capsys.readouterr().out
    assert "bhavcopy covered 2/4 NSE tickers" in text and "3 tickers (1 BSE-only, 2 NSE not in bhavcopy)" in text


def test_bulk_ohlc_bhavcopy_failure_falls_back_to_yfinance(ind, monkeypatch, capsys):
    monkeypatch.setattr(ind, "_BHAV_OK", True)

    def bad(days):
        raise RuntimeError("no archive")
    monkeypatch.setattr(ind, "_bhav", types.SimpleNamespace(build_history=bad), raising=False)
    monkeypatch.setattr(ind, "bulk_download_ohlc", lambda t, period: {"ALL": t})
    assert ind.bulk_ohlc(["A.NS", "B.BO"]) == {"ALL": ["A.NS", "B.BO"]}
    assert "bhavcopy build failed (no archive)" in capsys.readouterr().out


def test_bulk_ohlc_empty_history_all_fallback(ind, monkeypatch):
    monkeypatch.setattr(ind, "_BHAV_OK", True)
    monkeypatch.setattr(ind, "_bhav", types.SimpleNamespace(build_history=lambda d: (pd.DataFrame(), {"symbols": 0, "days": 0, "downloaded": 0, "fetch_secs": 0})), raising=False)
    monkeypatch.setattr(ind, "bulk_download_ohlc", lambda t, period: {x: _ohlc() for x in t})
    assert set(ind.bulk_ohlc(["A.NS"])) == {"A.NS"}


def test_darvas_wrapper_accepts_nse_columns(ind):
    df = pd.DataFrame({"CH_TRADE_HIGH_PRICE": np.linspace(10, 20, 30), "CH_TRADE_LOW_PRICE": np.linspace(9, 19, 30),
                       "CH_CLOSING_PRICE": np.linspace(9.5, 19.5, 30)})
    assert "signal" in ind.compute_darvas_box(df)


def test_golden_crossover(ind):
    assert ind.compute_golden_crossover(None)["gc_signal"] is False
    assert ind.compute_golden_crossover(pd.DataFrame())["dma200"] is None
    assert ind.compute_golden_crossover(_ohlc(200))["dma50"] is None
    r = ind.compute_golden_crossover(pd.DataFrame({"Close": np.linspace(10, 100, 260)}))
    assert r["dma50_above_200"] is True and r["gc_signal"] is False and r["dma_gap_%"] > 0
    r = ind.compute_golden_crossover(pd.DataFrame({"Close": np.linspace(100, 10, 260)}))
    assert r["dma50_above_200"] is False and r["dma_gap_%"] < 0
    c = np.concatenate([np.linspace(100, 40, 230), np.linspace(40, 200, 31)])
    hits = [ind.compute_golden_crossover(pd.DataFrame({"Close": c[:230 + k]}))["gc_signal"] for k in range(5, 31)]
    assert hits.count(True) == 1  # the cross fires on exactly one bar


# ------------------------------------------------------------------ fundamentals
def _fin(rows, ncols=3):
    cols = [f"y{i}" for i in range(ncols)]
    return pd.DataFrame({k: v + [v[-1]] * (ncols - len(v)) for k, v in rows.items()}).T.set_axis(cols, axis=1)


INC = _fin({"Net Income": [150e7, 100e7, 80e7], "Total Revenue": [1500e7, 1000e7, 800e7], "Gross Profit": [700e7, 400e7, 300e7], "EBIT": [300e7, 200e7, 150e7]})
BAL = _fin({"Total Assets": [1000e7, 1000e7, 900e7], "Long Term Debt": [50e7, 100e7, 100e7], "Current Assets": [500e7, 400e7, 300e7],
            "Current Liabilities": [200e7, 200e7, 200e7], "Share Issued": [100, 100, 100], "Stockholders Equity": [600e7, 500e7, 400e7]})
CF = _fin({"Operating Cash Flow": [200e7, 150e7, 100e7], "Free Cash Flow": [120e7, 100e7, 80e7]})
INC_Q = _fin({"Total Revenue": [130, 120, 115, 110, 100], "Net Income": [3e7, 2e7, 2e7, 1e7, 2e7]}, ncols=5)


class _Tk:
    def __init__(self, inc=INC, bal=BAL, cf=CF, inc_q=INC_Q, info=None, mcap=1e11, info_raises=False, fast_raises=False):
        self.income_stmt, self.balance_sheet, self.cash_flow, self.quarterly_income_stmt = inc, bal, cf, inc_q
        self._info, self._ir, self._fr, self._m = info if info is not None else {}, info_raises, fast_raises, mcap

    @property
    def fast_info(self):
        if self._fr:
            raise RuntimeError("fast")
        return types.SimpleNamespace(market_cap=self._m)

    @property
    def info(self):
        if self._ir:
            raise RuntimeError("info")
        return self._info


def test_fundamental_scan_live_all_screens(ind, monkeypatch):
    seen = {}
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: seen.update(s=s) or _Tk(info={"debtToEquity": 40.0, "totalDebt": 0, "totalCash": 0, "bookValue": 5}))
    r = ind.fundamental_scan("TCS")
    assert seen["s"] == "TCS.NS" and r["_source"] == "live"
    assert r["f_score"] == 9 and r["f_strong"] and r["qualifies_cc"] and r["cc_score"] == "6/6"
    assert r["ROCE_avg_%"] == pytest.approx(np.mean([300 / 800 * 100, 200 / 800 * 100, 150 / 700 * 100]), abs=0.01)
    assert r["qualifies_mf"] is False  # ROIC 37.5% but earnings yield only 3%
    assert r["ROIC_%"] == 37.5 and r["Earnings_Yield_%"] == pytest.approx(300e7 / 1e11 * 100, abs=0.01)
    assert r["qualifies_bc"] is True and r["Net_Profit_Cr"] == 3.0 and r["Sales_Growth_YoY_%"] == 30.0
    # a cheap small-cap (200 crore) passes magic formula but misses the 500 crore coffee-can floor
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(mcap=2e9, info={"debtToEquity": 40.0, "bookValue": 5}))
    r = ind.fundamental_scan("TCS")
    assert r["qualifies_mf"] is True and r["cc_score"] == "5/6" and r["qualifies_cc"] is False


def test_fundamental_scan_mcap_threshold_in_crore(ind, monkeypatch):
    # coffee-can wants market cap >= 500 crore (5e9), magic formula wants > 15 crore (1.5e8)
    info = {"debtToEquity": 5.0, "bookValue": 5}
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info=info, mcap=5e9 - 1))
    assert ind.fundamental_scan("X")["cc_score"] == "4/6"  # D/E=5 fails and cap just under 500 cr
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 50.0, "bookValue": 5}, mcap=5e9 - 1))
    assert ind.fundamental_scan("X")["cc_score"] == "5/6"
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 50.0, "bookValue": 5}, mcap=5e9))
    assert ind.fundamental_scan("X")["cc_score"] == "6/6"
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info={"bookValue": 5}, mcap=1.5e8))
    assert ind.fundamental_scan("X")["qualifies_mf"] is False  # exactly 15 cr is not > 15 cr (and EY tiny)


def test_fundamental_scan_debt_to_equity_percentage_boundary(ind, monkeypatch):
    # yfinance reports D/E in percent; values above 10 are rescaled, 10 itself is taken as a ratio (fails the <1 bar)
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 10.0}, mcap=5e9))
    assert ind.fundamental_scan("X")["cc_score"] == "5/6"
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 10.5}, mcap=5e9))
    assert ind.fundamental_scan("X")["cc_score"] == "6/6"


def test_fundamental_scan_bull_cartel_boundaries(ind, monkeypatch):
    def q(rev0, ni0):
        return _fin({"Total Revenue": [rev0, 0, 0, 0, 100], "Net Income": [ni0, 0, 0, 0, 2e7]}, ncols=5)
    for rev0, ni0, exp in ((115.0, 2.5e7, False), (116.0, 2.4e7, False), (116.0, 2.5e7, True)):
        monkeypatch.setattr(ind.yf, "Ticker", lambda s, rev0=rev0, ni0=ni0: _Tk(inc_q=q(rev0, ni0)))
        assert ind.fundamental_scan("X")["qualifies_bc"] is exp, (rev0, ni0)
    tiny = _fin({"Total Revenue": [116, 0, 0, 0, 100], "Net Income": [9e6, 0, 0, 0, 1e6]}, ncols=5)
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(inc_q=tiny))
    assert ind.fundamental_scan("X")["qualifies_bc"] is False  # net profit must exceed 1 crore
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk(inc_q=_fin({"Total Revenue": [1]}, ncols=3)))
    r = ind.fundamental_scan("X")
    assert r["qualifies_bc"] is False and r["Net_Profit_Cr"] is None


def test_fundamental_scan_bse_suffix_and_missing_data(ind, monkeypatch):
    seen = {}
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: seen.update(s=s) or _Tk(inc=None, info_raises=True, fast_raises=True))
    r = ind.fundamental_scan("500001", ".BO")
    assert seen["s"] == "500001.BO" and r["f_score"] is None and r["cc_score"] == "N/A" and r["Revenue_CAGR_%"] is None

    def boom(s):
        raise RuntimeError("yf broke")
    monkeypatch.setattr(ind.yf, "Ticker", boom)
    r = ind.fundamental_scan("X")
    assert r["error"] == "yf broke" and r["f_score"] is None and r["Sales_Growth_YoY_%"] is None


def test_fundamental_scan_uses_store_when_available(ind, monkeypatch):
    fsr = types.ModuleType("fundamentals_store_reader")
    seen = {}
    fsr.statements = lambda s: (INC, BAL, CF)
    fsr.mcap_and_info = lambda s, price: seen.update(price=price) or (1e11, {"debtToEquity": 40.0, "bookValue": 5})
    fsr.quarterly_statements = lambda s: INC_Q
    monkeypatch.setitem(sys.modules, "fundamentals_store_reader", fsr)
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: pytest.fail("store hit must not call yfinance"))
    r = ind.fundamental_scan("TCS", ".NS", 123.4)
    assert r["_source"] == "store" and r["f_score"] == 9 and seen["price"] == 123.4 and r["qualifies_bc"]
    # BSE-only names never use the store
    monkeypatch.setattr(ind.yf, "Ticker", lambda s: _Tk())
    assert ind.fundamental_scan("500001", ".BO")["_source"] == "live"
    # a store miss (no income statement) falls through to yfinance
    fsr.statements = lambda s: (None, None, None)
    assert ind.fundamental_scan("TCS", ".NS")["_source"] == "live"
    # a store exception also falls through
    fsr.statements = lambda s: (_ for _ in ()).throw(RuntimeError("store corrupt"))
    assert ind.fundamental_scan("TCS", ".NS")["_source"] == "live"


# ------------------------------------------------------------------ Excel
def test_save_excel_placeholder_sheets_and_sorting(ind):
    allr = [{"Symbol": "A", "Change%": 1.0, "GC_Signal": "GOLDEN_CROSS", "DMA_Gap%": 3.0, "Upside_to_Top%": 2.0},
            {"Symbol": "B", "Change%": 2.0, "GC_Signal": "BELOW_200DMA", "DMA_Gap%": -1.0, "Upside_to_Top%": 1.0}]
    fund = [{"Symbol": "A", "Piotroski_Score": 8, "Piotroski_Strong": "YES", "CoffeeCan": "PASS", "MagicFormula": "FAIL", "BullCartel": "FAIL", "Revenue_CAGR_%": 11},
            {"Symbol": "B", "Piotroski_Score": 9, "Piotroski_Strong": "NO", "CoffeeCan": "FAIL", "MagicFormula": "FAIL", "BullCartel": "FAIL"}]
    path = ind.save_excel(allr, allr, fund, fund[:1], None, tag="indian")
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets["All_Stocks"]["Symbol"]) == ["B", "A"]
    assert list(sheets["All_Fundamentals"]["Symbol"]) == ["B", "A"]
    assert list(sheets["Piotroski_Strong"]["Symbol"]) == ["A"] and list(sheets["Coffee_Can"]["Symbol"]) == ["A"]
    assert list(sheets["Golden_Crossover"]["Symbol"]) == ["A"]
    for empty in ("Magic_Formula", "Bull_Cartel", "Multi_Screen_Hits"):
        assert list(sheets[empty]["Note"]) == ["No qualifying stocks today"], empty
    assert path.name.startswith("indian_full_scan_")


# ------------------------------------------------------------------ main
_who = {}


def _frames(names):
    _who.clear()
    out = {}
    for i, n in enumerate(names):
        df = _ohlc(10, start=100.0 + 10 * i)
        _who[float(df["Close"].iloc[0])] = n
        out[n] = df
    return out


def _install(ind, monkeypatch, frames, signals):
    monkeypatch.setattr(ind, "bulk_ohlc", lambda t, period="1y": {k: v for k, v in frames.items() if k in t})
    monkeypatch.setattr(ind, "compute_darvas_box", lambda df, confirm=3: {"signal": signals[_who[float(df["Close"].iloc[0])]], "box_top": 20.0, "box_bottom": 5.0,
                                                                         "upside_to_top_pct": 4.0, "position_in_box_pct": 50.0, "data_points": 10})
    monkeypatch.setattr(ind, "compute_golden_crossover", lambda df: {"gc_signal": _who[float(df["Close"].iloc[0])] == "AAA.NS", "dma50_above_200": True,
                                                                     "dma50": 1.0, "dma200": 0.5, "dma_gap_%": 100.0})


def test_main_full_pipeline(ind, monkeypatch, capsys):
    monkeypatch.setattr(ind, "fetch_nse_symbols", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(ind, "fetch_bse_only_symbols", lambda s: ["500001"] if s == {"AAA", "BBB"} else [])
    frames = _frames(["AAA.NS", "BBB.NS", "500001.BO"])
    _install(ind, monkeypatch, frames, {"AAA.NS": "BREAKOUT_BUY", "BBB.NS": "BREAKDOWN_SELL", "500001.BO": "IN_BOX"})
    calls = []

    def fund(sym, suffix, price):
        calls.append((sym, suffix, price))
        if sym == "BBB":
            raise RuntimeError("scan failed")
        return {"f_score": 8, "f_strong": True, "qualifies_cc": sym in ("AAA", "500001"), "cc_score": "6/6", "qualifies_mf": sym == "AAA", "qualifies_bc": False,
                "Revenue_CAGR_%": 12.0}
    monkeypatch.setattr(ind, "fundamental_scan", fund)
    nse = types.SimpleNamespace(print_live_context=lambda: print("LIVE-CONTEXT"))
    monkeypatch.setattr(ind, "_USE_NSE_FETCHER", True)
    monkeypatch.setattr(ind, "_NSE_FETCHER", nse)
    res = ind.main(workers=1)
    out = capsys.readouterr().out
    assert "LIVE-CONTEXT" in out and "Total tickers to scan: 3" in out
    assert sorted(calls) == [("500001", ".BO", 129.0), ("AAA", ".NS", 109.0), ("BBB", ".NS", 119.0)]  # LTP = last close of each frame
    assert res["total_scanned"] == 3 and res["breakout_count"] == 1 and res["gc_count"] == 1
    assert "BBB: error — scan failed" in out
    assert len(res["triple_hits"]) == 1 and res["triple_hits"][0]["Symbol"] == "AAA"
    # AAA passes 5 screens (breakout, 200DMA, Piotroski, coffee can, magic formula); 500001 exactly 3 (the inclusive threshold); BBB's scan failed
    hits = {r["Symbol"]: r["Screens_Passed"] for r in res["six_screen_hits"]}
    assert hits == {"AAA": 5, "500001": 3}
    sheets = pd.read_excel(res["excel_path"], sheet_name=None, dtype={"Symbol": str})
    assert list(sheets["Triple_Hits"]["Symbol"]) == ["AAA"]


def test_main_nse_only_top_cache_and_no_scans(ind, monkeypatch, capsys):
    ind._save_symbol_cache(["AAA", "BBB", "CCC"], ["500001"])
    frames = _frames(["AAA.NS", "BBB.NS"])
    _install(ind, monkeypatch, frames, {"AAA.NS": "IN_BOX", "BBB.NS": "IN_BOX"})
    monkeypatch.setattr(ind, "fetch_nse_symbols", lambda: pytest.fail("cache must be used"))
    monkeypatch.setattr(ind, "fundamental_scan", lambda *a: pytest.fail("scans disabled"))
    monkeypatch.setattr(ind, "_USE_NSE_FETCHER", True)
    monkeypatch.setattr(ind, "_NSE_FETCHER", types.SimpleNamespace(print_live_context=lambda: (_ for _ in ()).throw(RuntimeError("ctx"))))
    res = ind.main(nse_only=True, top=2, run_scans=False)
    out = capsys.readouterr().out
    assert "Symbol cache hit: 3 NSE + 1 BSE-only" in out and "(limited to first 2 tickers)" in out
    assert "Total tickers to scan: 2" in out and "Live context unavailable: ctx" in out
    assert "Stage 4 — Skipped" in out and "Top Multi-Screen Hits" not in out
    assert res["total_scanned"] == 2 and res["triple_hits"] == []


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="main(): with nse_only=True and a cold symbol cache bse_only stays None (from _load_symbol_cache) and `for code in bse_only` raises")
def test_main_nse_only_skips_bse_fetch_and_caches(ind, monkeypatch):
    monkeypatch.setattr(ind, "fetch_nse_symbols", lambda: ["AAA"])
    monkeypatch.setattr(ind, "fetch_bse_only_symbols", lambda s: pytest.fail("BSE skipped for --nse-only"))
    monkeypatch.setattr(ind, "bulk_ohlc", lambda t, period="1y": {})
    ind.main(nse_only=True, run_scans=False)
    assert ind._load_symbol_cache() == (["AAA"], [])
