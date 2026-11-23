"""Offline tests for full_korea_market_scan (KIND universe, OHLC, DART/yfinance fundamentals, main)."""

import importlib
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def kr(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # module creates ./korea_scan at import
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: pd.DataFrame()
    fake_yf.Ticker = lambda t: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    pk = types.ModuleType("pykrx")
    stock = types.ModuleType("pykrx.stock")
    stock.get_market_ohlcv_by_date = lambda a, b, c: pd.DataFrame()
    pk.stock = stock
    monkeypatch.setitem(sys.modules, "pykrx", pk)
    monkeypatch.setitem(sys.modules, "pykrx.stock", stock)
    sys.modules.pop("full_korea_market_scan", None)
    mod = importlib.import_module("full_korea_market_scan")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "korea_scan")
    monkeypatch.setattr(mod, "_CACHE_FILE", tmp_path / "korea_scan" / ".symbols_cache.json")
    monkeypatch.setattr(mod, "SLEEP_BETWEEN_KRX", 0.0)
    # keep the real DART module out of the picture unless a test installs a fake
    monkeypatch.setitem(sys.modules, "dart_fundamentals", None)
    yield mod
    sys.modules.pop("full_korea_market_scan", None)


def _ohlc(n=10):
    return pd.DataFrame({"Close": np.arange(1.0, n + 1), "High": 1.0, "Low": 1.0, "Open": 1.0, "Volume": 1.0})


# ------------------------------------------------------------------ cache + KIND
def test_cache_roundtrip_ttl_and_corruption(kr, tmp_path):
    assert kr._load_cache() is None
    kr._save_cache([{"code": "005930"}])
    assert kr._load_cache() == [{"code": "005930"}]
    kr._CACHE_FILE.write_text(json.dumps({"ts": time.time() - kr.SYMBOL_CACHE_TTL - 1, "universe": [1]}))
    assert kr._load_cache() is None
    kr._CACHE_FILE.write_text("garbage")
    assert kr._load_cache() is None
    kr._CACHE_FILE.write_text(json.dumps({"ts": time.time()}))
    assert kr._load_cache() == []  # fresh cache without a universe key


def test_save_cache_swallows_errors(kr, monkeypatch, tmp_path):
    monkeypatch.setattr(kr, "_CACHE_FILE", tmp_path / "nope" / "c.json")
    kr._save_cache([1])


def test_fetch_kind_list_parses_rows(kr, monkeypatch):
    html_df = pd.DataFrame({"회사명": [" Samsung ", "BadCode", "Pad"], "종목코드": [5930, "ABC123", "660"], "업종": ["Semis", "x", None]})
    seen = {}
    resp = types.SimpleNamespace(text="<html/>", raise_for_status=lambda: None)

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return resp
    monkeypatch.setattr(kr.requests, "get", fake_get)
    monkeypatch.setattr(kr.pd, "read_html", lambda *a, **k: [html_df])
    out = kr._fetch_kind_list("stockMkt", "KOSPI", ".KS")
    assert [s["code"] for s in out] == ["005930", "000660"]
    assert out[0] == {"code": "005930", "name": "Samsung", "sector": "Semis", "market": "KOSPI", "yf_suffix": ".KS"}
    assert out[1]["sector"] == "None"
    assert seen["params"]["marketType"] == "stockMkt" and seen["timeout"] == 20


def test_fetch_kind_list_error_returns_empty(kr, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("KIND down")
    monkeypatch.setattr(kr.requests, "get", boom)
    assert kr._fetch_kind_list("stockMkt", "KOSPI", ".KS") == []
    assert "KIND fetch error (KOSPI): KIND down" in capsys.readouterr().out


def test_build_universe_cache_kospi_filter_and_fetch(kr, monkeypatch):
    kr._save_cache([{"code": "1", "market": "KOSPI"}, {"code": "2", "market": "KOSDAQ"}])
    assert len(kr.build_krx_universe()) == 2
    assert [s["code"] for s in kr.build_krx_universe(kospi_only=True)] == ["1"]
    kr._CACHE_FILE.unlink()
    calls = []

    def fake_kind(param, label, sfx):
        calls.append(param)
        return [{"code": param, "market": label, "yf_suffix": sfx}]
    monkeypatch.setattr(kr, "_fetch_kind_list", fake_kind)
    out = kr.build_krx_universe()
    assert calls == ["stockMkt", "kosdaqMkt"] and len(out) == 2
    assert kr._load_cache() == out
    kr._CACHE_FILE.unlink()
    calls.clear()
    assert len(kr.build_krx_universe(kospi_only=True)) == 1 and calls == ["stockMkt"]
    kr._CACHE_FILE.unlink()
    monkeypatch.setattr(kr, "_fetch_kind_list", lambda *a: [])
    with pytest.raises(SystemExit):
        kr.build_krx_universe()


# ------------------------------------------------------------------ OHLC
def test_normalise_krx_cols(kr):
    df = pd.DataFrame({"고가": [1], "저가": [2], "종가": [3], "시가": [4], "거래량": [5], "other": [6]})
    assert list(kr._normalise_krx_cols(df).columns) == ["High", "Low", "Close", "Open", "Volume", "other"]
    assert list(kr._normalise_krx_cols(pd.DataFrame({"x": [1]})).columns) == ["x"]


def test_fetch_krx_ohlc_uses_cache_with_suffix_mapping(kr, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    seen = {}

    def get(market, syms, yf_suffix, period):
        seen.update(market=market, syms=syms, period=period, sfx=yf_suffix)
        return {"005930.KS": _ohlc(10), "035720.KQ": _ohlc(7), "000660.KS": _ohlc(8)}
    oc.get = get
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    uni = [{"code": "005930", "yf_suffix": ".KS"}, {"code": "035720", "yf_suffix": ".KQ"},
           {"code": "000660"}, {"code": "999999"}]
    out = kr.fetch_krx_ohlc(uni, months=3)
    assert set(out) == {"005930", "000660"}  # 7 rows < DARVAS_CONFIRM+5=8 is dropped; missing ticker absent
    assert seen["syms"] == ["005930.KS", "035720.KQ", "000660.KS", "999999.KS"]
    assert seen["market"] == "KOREA" and seen["period"] == "12mo" and seen["sfx"] == ""
    assert "ohlcv_cache: 2/4" in capsys.readouterr().out


def test_fetch_krx_ohlc_period_uses_max_of_months_and_12(kr, monkeypatch):
    oc = types.ModuleType("ohlcv_cache")
    periods = []
    oc.get = lambda m, s, yf_suffix, period: periods.append(period) or {"1.KS": _ohlc(10)}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    kr.fetch_krx_ohlc([{"code": "1"}], months=18)
    assert periods == ["18mo"]


def test_fetch_krx_ohlc_falls_back_to_pykrx(kr, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    oc.get = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    kor = pd.DataFrame({"고가": np.ones(10), "저가": np.ones(10), "종가": np.arange(1.0, 11), "시가": 1.0, "거래량": 1.0})
    english_missing = pd.DataFrame({"종가": np.arange(1.0, 11)})

    def by_date(a, b, code):
        if code == "A":
            return kor
        if code == "B":
            return None
        if code == "C":
            return english_missing
        if code == "D":
            return kor.iloc[:3]
        raise RuntimeError("boom")
    kr.krx.get_market_ohlcv_by_date = by_date
    uni = [{"code": c} for c in "ABCDE"]
    out = kr.fetch_krx_ohlc(uni, months=3)
    assert list(out) == ["A"] and list(out["A"].columns[:3]) == ["High", "Low", "Close"]
    text = capsys.readouterr().out
    assert "returned nothing" in text and "1 tickers with usable OHLC data" in text


def test_fetch_krx_ohlc_cache_exception_then_progress_line(kr, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")

    def boom(*a, **k):
        raise RuntimeError("cache offline")
    oc.get = boom
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    kr.krx.get_market_ohlcv_by_date = lambda a, b, c: pd.DataFrame({"고가": [1], "저가": [1], "종가": [1]})  # too short
    out = kr.fetch_krx_ohlc([{"code": str(i)} for i in range(200)], months=1)
    assert out == {}
    text = capsys.readouterr().out
    assert "ohlcv_cache unavailable: cache offline" in text and "200/200 fetched (0 usable)" in text


# ------------------------------------------------------------------ helpers + fundamentals
def test_row_series_helpers(kr):
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]}, index=["X", "Y"])
    assert kr._row(df, "X") == 1.0 and kr._row(df, "X", col=1) == 2.0
    assert kr._row(df, "Y") is None and kr._row(df, "Q") is None and kr._row(None, "X") is None
    assert kr._row(df, "X", col=5) is None
    assert kr._series(df, "Y") == [3.0] and kr._series(df, "Q") == [] and kr._series(pd.DataFrame(), "X") == []
    assert kr.compute_darvas_box(_ohlc(30))["signal"] == "NO_BOX"


def _fin(rows):
    return pd.DataFrame(rows).T.rename(columns={0: "y0", 1: "y1", 2: "y2"})


INC = _fin({"Net Income": [150, 100, 80], "Total Revenue": [1500, 1000, 800], "Gross Profit": [700, 400, 300], "EBIT": [300, 200, 150]})
BAL = _fin({"Total Assets": [1000, 1000, 900], "Long Term Debt": [50, 100, 100], "Current Assets": [500, 400, 300],
            "Current Liabilities": [200, 200, 200], "Share Issued": [100, 100, 100], "Stockholders Equity": [600, 500, 400]})
CF = _fin({"Operating Cash Flow": [200, 150, 100], "Free Cash Flow": [120, 100, 80]})


def _ticker(inc=INC, bal=BAL, cf=CF, mcap=2e11):
    return types.SimpleNamespace(income_stmt=inc, balance_sheet=bal, cash_flow=cf,
                                 fast_info=types.SimpleNamespace(market_cap=mcap))


def _fake_dart(monkeypatch, score=None, raises=False):
    d = types.ModuleType("dart_fundamentals")

    def inputs(code):
        if raises:
            raise RuntimeError("dart down")
        return {"code": code} if score is not None else None
    d.piotroski_inputs = inputs
    d.f_score = lambda inp: {"f_score": score}
    monkeypatch.setitem(sys.modules, "dart_fundamentals", d)


def test_fundamental_scan_yfinance_full_pass(kr, monkeypatch):
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker())
    r = kr.fundamental_scan("005930", ".KS")
    assert r["yf_ticker"] == "005930.KS" and r["code"] == "005930"
    assert r["f_score"] == 9 and r["f_strong"] and r["source"] == "yfinance"
    assert r["cc_qualifies"] == "PASS" and r["cc_score"] == "6/6"
    assert r["market_cap_b_krw"] == 2000.0 and r["cc_debt_equity"] == pytest.approx(0.08)


def test_fundamental_scan_prefers_dart_score_when_available(kr, monkeypatch):
    _fake_dart(monkeypatch, score=7)
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker())
    r = kr.fundamental_scan("005930", ".KS")
    assert r["f_score"] == 7 and r["source"] == "dart" and r["yf_f_score"] == 9 and r["f_strong"]
    _fake_dart(monkeypatch, score=6)  # DART says 6: not strong even though yfinance says 9
    r = kr.fundamental_scan("005930", ".KS")
    assert r["f_score"] == 6 and not r["f_strong"] and r["cc_score"] == "—"


def test_fundamental_scan_dart_only_when_yfinance_empty(kr, monkeypatch):
    _fake_dart(monkeypatch, score=8)
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker(None, None, None))
    r = kr.fundamental_scan("1", ".KQ")
    assert r["f_score"] == 8 and r["f_strong"] and r["source"] == "dart" and r["cc_score"].startswith("n/a")
    _fake_dart(monkeypatch, score=7)
    r = kr.fundamental_scan("1", ".KQ")
    assert r["f_strong"] is True  # 7 is the strong boundary
    _fake_dart(monkeypatch, score=6)
    assert kr.fundamental_scan("1", ".KQ")["f_strong"] is False


def test_fundamental_scan_errors(kr, monkeypatch):
    _fake_dart(monkeypatch, raises=True)  # DART failure never breaks the scan
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker(None, None, None))
    assert kr.fundamental_scan("1", ".KS")["error"] == "no_financial_data"

    def boom(t):
        raise RuntimeError("y" * 100)
    monkeypatch.setattr(kr.yf, "Ticker", boom)
    assert kr.fundamental_scan("1", ".KS")["error"] == "y" * 80
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker())
    monkeypatch.setattr(kr, "_row", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    assert kr.fundamental_scan("1", ".KS")["f_score"] is None


def test_fundamental_scan_coffee_can_partial(kr, monkeypatch):
    cf = _fin({"Operating Cash Flow": [200, 1, 1], "Capital Expenditure": [-300, -1, -1]})
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: _ticker(cf=cf, mcap=1e10))
    r = kr.fundamental_scan("1", ".KS")
    assert r["f_strong"] and r["cc_qualifies"] == "FAIL" and r["cc_score"] == "4/6"

    class NoFast:
        income_stmt, balance_sheet, cash_flow = INC, BAL, CF

        @property
        def fast_info(self):
            raise RuntimeError("n/a")
    monkeypatch.setattr(kr.yf, "Ticker", lambda t: NoFast())
    r = kr.fundamental_scan("1", ".KS")
    assert r["market_cap_b_krw"] is None and r["cc_score"] == "5/6"


# ------------------------------------------------------------------ main
class _Liq:
    @staticmethod
    def gate(df, ticker):
        return (0.0, None) if ticker.startswith("999") else (4e6, "LIQUID")


signals_by_df = {}


def _setup(kr, monkeypatch, signals, argv=("prog",), cached=None):
    uni = [{"code": c, "name": f"N{c}", "market": "KOSPI" if i % 2 == 0 else "KOSDAQ",
            "yf_suffix": ".KS" if i % 2 == 0 else ".KQ"} for i, c in enumerate(signals)]
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(kr, "build_krx_universe", lambda kospi_only=False: [u for u in uni if not kospi_only or u["market"] == "KOSPI"])
    frames = {c: _ohlc(10) for c in signals}
    signals_by_df.clear()
    for rank, (c, sig) in enumerate(signals.items()):
        signals_by_df[id(frames[c])] = {"signal": sig, "box_top": 12.0, "box_bottom": 8.0, "upside_pct": float(rank + 1),
                                         "pos_in_box": 50.0, "data_points": 10}
    monkeypatch.setattr(kr, "fetch_krx_ohlc", lambda u, months=3: {s["code"]: frames[s["code"]] for s in u})
    monkeypatch.setattr(kr, "compute_darvas_box", lambda df, confirm=3: signals_by_df[id(df)])
    monkeypatch.setattr(kr, "_liq", _Liq)
    fc = types.ModuleType("fundamentals_cache")
    if cached is None:
        fc.load = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cache"))
    else:
        fc.load = lambda pat, key: (cached, "old.xlsx")
    monkeypatch.setitem(sys.modules, "fundamentals_cache", fc)
    return uni


def test_main_pipeline_with_reuse_and_failures(kr, monkeypatch, tmp_path, capsys):
    signals = {"000001": "BREAKOUT_BUY", "000002": "BREAKDOWN_SELL", "000003": "IN_BOX", "999999": "BREAKOUT_BUY",
               "000005": "BREAKOUT_BUY", "000006": "BREAKOUT_BUY"}
    cached = {"000005": {"Code": "000005", "Piotroski_Strong": "YES", "CoffeeCan": "PASS", "Darvas_Signal": "OLD",
                         "LTP_KRW": 1, "Piotroski_Score": 9, "Name": "N5", "Market": "KOSPI", "CC_Score": "6/6"}}
    _setup(kr, monkeypatch, signals, argv=("prog", "--workers", "1"), cached=cached)

    def fake_scan(code, sfx):
        if code == "000006":
            raise RuntimeError("scan blew up")
        return {"f_score": 8, "f_strong": True, "cc_qualifies": "PASS", "cc_score": "6/6", "cc_rev_cagr": 11.0,
                "cc_roce_avg": 20.0, "cc_debt_equity": 0.2, "market_cap_b_krw": 500.0}
    monkeypatch.setattr(kr, "fundamental_scan", fake_scan)
    kr.main()
    out = capsys.readouterr().out
    assert "Total: 6 tickers (3 KOSPI + 3 KOSDAQ)" in out
    assert "Breakout BUY:   3" in out and "Breakdown SELL: 1" in out and "In Box:         1" in out
    assert "reusing 1 fundamentals from old.xlsx" in out and "merged 1 reused" in out
    assert "000006: error — scan blew up" in out
    assert "TRIPLE HITS:         2" in out and "KRX stocks scanned:    5" in out
    f = next((tmp_path / "korea_scan").glob("korea_market_scan_*.xlsx"))
    sheets = pd.read_excel(f, sheet_name=None, dtype={"Code": str})
    assert set(sheets["Triple_Hits"]["Code"]) == {"000001", "000005"}
    fund = sheets["Fundamentals"].set_index("Code")
    assert fund.loc["000005", "Darvas_Signal"] == "BREAKOUT_BUY" and fund.loc["000001", "MCap_100M_KRW"] == 500.0
    assert "999999" not in set(sheets["All_Stocks"]["Code"])
    assert sheets["All_Stocks"].set_index("Code").loc["000001", "Trend_Signal"] == "Need 200+ days"


def test_main_kospi_only_top_and_no_scans(kr, monkeypatch, tmp_path, capsys):
    signals = {"000001": "BREAKOUT_BUY", "000002": "IN_BOX", "000003": "IN_BOX", "000004": "IN_BOX"}
    _setup(kr, monkeypatch, signals, argv=("prog", "--kospi-only", "--top", "1", "--no-scans"))
    monkeypatch.setattr(kr, "fundamental_scan", lambda *a: pytest.fail("scans disabled"))
    kr.main()
    out = capsys.readouterr().out
    assert "(limited to first 1 tickers)" in out and "Total: 1 tickers (1 KOSPI + 0 KOSDAQ)" in out
    assert "Stage 4 — Skipped" in out and "No Triple Hit stocks found today." in out


def test_main_caps_breakouts_to_freshest(kr, monkeypatch, tmp_path, capsys):
    signals = {f"00000{i}": "BREAKOUT_BUY" for i in range(1, 6)}
    _setup(kr, monkeypatch, signals, argv=("prog", "--workers", "1"))
    monkeypatch.setattr(kr, "MAX_FUND_CANDIDATES", 2)
    scanned = []
    monkeypatch.setattr(kr, "fundamental_scan", lambda c, s: scanned.append(c) or {"f_score": 1, "f_strong": False})
    kr.main()
    assert "(capped to 2 freshest breakouts)" in capsys.readouterr().out
    assert sorted(scanned) == ["000001", "000002"]  # smallest upside_pct first
