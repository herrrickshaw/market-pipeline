"""Offline tests for full_japan_market_scan (universe, bulk OHLC, fundamental scan, main)."""

import importlib
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def jp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # module creates ./japan_scan at import
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: pd.DataFrame()
    fake_yf.Ticker = lambda t: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    sys.modules.pop("full_japan_market_scan", None)
    mod = importlib.import_module("full_japan_market_scan")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "japan_scan")
    monkeypatch.setattr(mod, "_CACHE_FILE", tmp_path / "japan_scan" / ".symbols_cache.json")
    monkeypatch.setattr(mod, "SLEEP_BETWEEN", 0.0)
    yield mod
    sys.modules.pop("full_japan_market_scan", None)


# ------------------------------------------------------------------ cache
def test_cache_roundtrip_and_ttl(jp):
    assert jp._load_cache() is None
    jp._save_cache([{"yf_ticker": "7203.T"}])
    assert jp._load_cache() == [{"yf_ticker": "7203.T"}]
    jp._CACHE_FILE.write_text(json.dumps({"ts": time.time() - jp.SYMBOL_CACHE_TTL - 5, "tickers": [1]}))
    assert jp._load_cache() is None  # expired
    jp._CACHE_FILE.write_text(json.dumps({"ts": time.time() - 10, "tickers": [1]}))
    assert jp._load_cache() == [1]
    jp._CACHE_FILE.write_text("not json")
    assert jp._load_cache() is None


def test_save_cache_swallows_errors(jp, monkeypatch, tmp_path):
    monkeypatch.setattr(jp, "_CACHE_FILE", tmp_path / "missing_dir" / "c.json")
    jp._save_cache([1])  # must not raise


# ------------------------------------------------------------------ universe
def _fake_kabupy(issues):
    mod = types.ModuleType("kabupy")
    mod.Jpx = lambda: types.SimpleNamespace(issues=issues)
    return mod


def test_kabupy_universe_filters_categories_and_codes(jp, monkeypatch, capsys):
    issues = [
        {"category": "プライム（内国株式）", "security_code": "7203.0", "name": "Toyota", "33_industry_category": "Auto"},
        {"category": "ETF", "security_code": "1306", "name": "ETF"},
        {"category": "スタンダード（内国株式）", "security_code": "99", "name": "Pad", "17_industry_category": "Misc"},
        {"category": "内国株式", "security_code": "12345", "name": "TooLong"},
        {"category": "内国株式", "security_code": "ABCD", "name": "NotDigits"},
    ]
    monkeypatch.setitem(sys.modules, "kabupy", _fake_kabupy(issues))
    out = jp.fetch_tse_universe_kabupy()
    assert [s["code"] for s in out] == ["7203", "0099"]
    assert out[0]["sector"] == "Auto" and out[0]["market"] == "プライム"
    assert out[1]["sector"] == "Misc"
    assert "kabupy → 2 TSE domestic equities" in capsys.readouterr().out


def test_kabupy_universe_failure_modes(jp, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "kabupy", None)  # import raises ImportError
    assert jp.fetch_tse_universe_kabupy() == []
    assert "kabupy not installed" in capsys.readouterr().out
    bad = types.ModuleType("kabupy")

    def boom():
        raise RuntimeError("down")
    bad.Jpx = boom
    monkeypatch.setitem(sys.modules, "kabupy", bad)
    assert jp.fetch_tse_universe_kabupy() == []
    assert "kabupy error: down" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "kabupy", _fake_kabupy([]))
    assert jp.fetch_tse_universe_kabupy() == []  # nothing found: no success message


def _jpx_frame(cols):
    return pd.DataFrame(cols)


def _patch_jpx(jp, monkeypatch, df=None, exc=None):
    resp = types.SimpleNamespace(content=b"x", raise_for_status=lambda: (_ for _ in ()).throw(exc) if exc else None)
    monkeypatch.setattr(jp.requests, "get", lambda *a, **k: resp)
    if df is not None:
        monkeypatch.setattr(jp.pd, "read_excel", lambda *a, **k: df)


def test_jpx_universe_parses_and_filters_markets(jp, monkeypatch):
    df = _jpx_frame({
        " コード ": ["7203", "6758.0", "1", "abcd", "9999"],
        "銘柄名": ["Toyota", "Sony", "Tiny", "Bad", "Growth Co"],
        "33業種区分": ["Auto", "Elec", "X", "Y", "Z"],
        "市場・商品区分": ["プライム（内国株式）", "Standard", "プライム", "プライム", "グロース"],
    })
    _patch_jpx(jp, monkeypatch, df)
    out = jp.fetch_tse_universe_jpx()
    assert [s["code"] for s in out] == ["7203", "6758", "0001"]  # growth filtered, bad code dropped
    assert out[0]["name"] == "Toyota" and out[0]["sector"] == "Auto"


def test_jpx_universe_without_optional_columns(jp, monkeypatch):
    _patch_jpx(jp, monkeypatch, _jpx_frame({"Code": ["7203"]}))
    out = jp.fetch_tse_universe_jpx()
    assert out == [{"code": "7203", "name": "", "sector": "—"}]


def test_jpx_universe_no_code_column_and_http_error(jp, monkeypatch, capsys):
    _patch_jpx(jp, monkeypatch, _jpx_frame({"foo": ["1"]}))
    assert jp.fetch_tse_universe_jpx() == []
    assert "Cannot identify code column" in capsys.readouterr().out
    _patch_jpx(jp, monkeypatch, exc=RuntimeError("503"))
    assert jp.fetch_tse_universe_jpx() == []
    assert "JPX fetch error: 503" in capsys.readouterr().out


def test_build_ticker_universe_cache_dedup_and_exit(jp, monkeypatch):
    jp._save_cache([{"yf_ticker": "1.T"}])
    assert jp.build_ticker_universe() == [{"yf_ticker": "1.T"}]  # cache hit short-circuits fetch
    jp._CACHE_FILE.unlink()
    monkeypatch.setattr(jp, "fetch_tse_universe_kabupy", lambda: [
        {"code": "7203", "name": "A", "sector": "s"}, {"code": "7203", "name": "dup", "sector": "s"},
        {"code": "6758", "name": "B", "sector": "t"}])
    monkeypatch.setattr(jp, "fetch_tse_universe_jpx", lambda: pytest.fail("fallback must not run"))
    out = jp.build_ticker_universe()
    assert [s["yf_ticker"] for s in out] == ["7203.T", "6758.T"]
    assert out[0]["name"] == "A"
    assert jp._load_cache() == out  # persisted
    jp._CACHE_FILE.unlink()
    monkeypatch.setattr(jp, "fetch_tse_universe_kabupy", lambda: [])
    monkeypatch.setattr(jp, "fetch_tse_universe_jpx", lambda: [{"code": "1301", "name": "K", "sector": "x"}])
    assert jp.build_ticker_universe()[0]["yf_ticker"] == "1301.T"  # falls back to JPX
    jp._CACHE_FILE.unlink()
    monkeypatch.setattr(jp, "fetch_tse_universe_jpx", lambda: [])
    with pytest.raises(SystemExit):
        jp.build_ticker_universe()


# ------------------------------------------------------------------ bulk OHLC
def _ohlc(n=10):
    return pd.DataFrame({"Close": np.arange(1.0, n + 1), "High": 1.0, "Low": 1.0, "Open": 1.0, "Volume": 1.0})


def test_bulk_download_prefers_cache(jp, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    calls = []
    oc.get = lambda *a, **k: calls.append((a, k)) or {"A.T": _ohlc()}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    out = jp.bulk_download_ohlc(["A.T", "B.T"])
    assert list(out) == ["A.T"]
    assert calls[0][0][0] == "JAPAN" and calls[0][1] == {"yf_suffix": "", "period": "1y"}
    assert "ohlcv_cache: 1/2 tickers" in capsys.readouterr().out


def test_bulk_download_direct_multiindex_and_failures(jp, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    oc.get = lambda *a, **k: {}  # empty -> fall through to direct download
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    monkeypatch.setattr(jp, "BATCH_SIZE", 2)
    cols = pd.MultiIndex.from_product([["Close", "High"], ["A.T", "B.T"]])
    good = pd.DataFrame(np.arange(20.0).reshape(10, 2).repeat(2, axis=1)[:, :4], columns=cols)
    short = good.iloc[:4]
    calls = {"n": 0}

    def fake_download(batch, **kw):
        calls["n"] += 1
        assert kw["period"] == "1y" and kw["auto_adjust"] is True
        if calls["n"] == 1:
            return good
        if calls["n"] == 2:
            return pd.DataFrame()
        if calls["n"] == 3:
            raise RuntimeError("rate limited")
        return short

    jp.yf.download = fake_download
    out = jp.bulk_download_ohlc(["A.T", "B.T", "C.T", "D.T", "E.T", "F.T", "G.T", "H.T"])
    assert set(out) == {"A.T", "B.T"}  # 4th batch is too short (<DARVAS_CONFIRM+5 rows)
    text = capsys.readouterr().out
    assert "Batch 1/4" in text and "OK (2 usable)" in text and "empty" in text and "ERROR — rate limited" in text


def test_bulk_download_min_history_boundary(jp, monkeypatch):
    oc = types.ModuleType("ohlcv_cache")
    oc.get = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    cols = pd.MultiIndex.from_product([["Close"], ["A.T", "B.T"]])
    raw = pd.DataFrame(np.arange(1.0, 17.0).reshape(8, 2), columns=cols)
    raw.iloc[-1, 1] = np.nan  # B.T has 7 usable rows, A.T has exactly 8 (= DARVAS_CONFIRM + 5)
    jp.yf.download = lambda batch, **kw: raw
    out = jp.bulk_download_ohlc(["A.T", "B.T"])
    assert set(out) == {"A.T"} and len(out["A.T"]) == 8


def test_bulk_download_single_ticker_flat_frame_and_cache_exception(jp, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")

    def boom(*a, **k):
        raise RuntimeError("cache broke")
    oc.get = boom
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    jp.yf.download = lambda batch, **kw: _ohlc()
    out = jp.bulk_download_ohlc(["A.T"])
    assert list(out) == ["A.T"]
    assert "ohlcv_cache unavailable: cache broke" in capsys.readouterr().out


# ------------------------------------------------------------------ helpers
def test_row_and_series_helpers(jp):
    df = pd.DataFrame({"2024": [1.0, np.nan, 5.0], "2023": [2.0, 3.0, np.nan]}, index=["A", "B", "C"])
    assert jp._row(df, "A") == 1.0
    assert jp._row(df, "A", col=1) == 2.0
    assert jp._row(df, "B") is None          # NaN
    assert jp._row(df, "Z", "A") == 1.0      # first present name wins
    assert jp._row(df, "Z") is None
    assert jp._row(None, "A") is None and jp._row(pd.DataFrame(), "A") is None
    assert jp._row(df, "A", col=9) is None   # out of range
    assert jp._series(df, "A") == [1.0, 2.0]
    assert jp._series(df, "B") == [3.0]      # NaN dropped
    assert jp._series(df, "Z") == [] and jp._series(None, "A") == []


# ------------------------------------------------------------------ fundamental scan
def _fin(rows):
    return pd.DataFrame(rows).T.rename(columns={0: "y0", 1: "y1", 2: "y2"})


def _ticker(inc=None, bal=None, cf=None, mcap=2e11, raise_fast=False):
    class FI:
        @property
        def market_cap(self):
            if raise_fast:
                raise RuntimeError("no fast_info")
            return mcap
    return types.SimpleNamespace(income_stmt=inc, balance_sheet=bal, cash_flow=cf, fast_info=FI())


STRONG_INC = _fin({
    "Net Income": [150, 100, 80], "Total Revenue": [1500, 1000, 800], "Gross Profit": [700, 400, 300],
    "EBIT": [300, 200, 150]})
STRONG_BAL = _fin({
    "Total Assets": [1000, 1000, 900], "Long Term Debt": [50, 100, 100],
    "Current Assets": [500, 400, 300], "Current Liabilities": [200, 200, 200],
    "Share Issued": [100, 100, 100], "Stockholders Equity": [600, 500, 400]})
STRONG_CF = _fin({"Operating Cash Flow": [200, 150, 100], "Free Cash Flow": [120, 100, 80]})


def test_fundamental_scan_full_pass(jp, monkeypatch):
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(STRONG_INC, STRONG_BAL, STRONG_CF))
    r = jp.fundamental_scan("7203.T")
    assert r["f_score"] == 9 and r["f_strong"] is True
    assert r["cc_qualifies"] == "PASS" and r["cc_score"] == "6/6"
    assert r["cc_rev_cagr"] == pytest.approx(((1500 / 800) ** 0.5 - 1) * 100, abs=0.01)
    assert r["cc_debt_equity"] == pytest.approx(0.08)
    assert r["market_cap_b_jpy"] == 200.0


def test_fundamental_scan_coffee_can_failures_and_fcf_fallback(jp, monkeypatch):
    cf = _fin({"Operating Cash Flow": [200, 150, 100], "Capital Expenditure": [-250, -1, -1]})
    t = _ticker(STRONG_INC, STRONG_BAL, cf, mcap=5e10)  # mcap below 1e11, fcf negative via fallback
    monkeypatch.setattr(jp.yf, "Ticker", lambda t_: t)
    r = jp.fundamental_scan("X.T")
    assert r["f_strong"] is True and r["cc_qualifies"] == "FAIL" and r["cc_score"] == "4/6"
    # failing fast_info -> mcap 0 -> C4 fails and market cap omitted
    t2 = _ticker(STRONG_INC, STRONG_BAL, cf, raise_fast=True)
    monkeypatch.setattr(jp.yf, "Ticker", lambda t_: t2)
    r2 = jp.fundamental_scan("X.T")
    assert r2["market_cap_b_jpy"] is None and r2["cc_score"] == "4/6"
    # positive capex-adjusted fallback passes C6 (but mcap still too low)
    cf3 = _fin({"Operating Cash Flow": [200, 150, 100], "Capital Expenditure": [-50, -1, -1]})
    monkeypatch.setattr(jp.yf, "Ticker", lambda t_: _ticker(STRONG_INC, STRONG_BAL, cf3, mcap=5e10))
    assert jp.fundamental_scan("X.T")["cc_score"] == "5/6"


def test_fundamental_scan_weak_company_skips_coffee_can(jp, monkeypatch):
    weak_inc = _fin({"Net Income": [-5, 10, 12], "Total Revenue": [100, 120, 130], "Gross Profit": [10, 30, 35]})
    weak_bal = _fin({"Total Assets": [1000, 900, 900], "Long Term Debt": [300, 100, 100],
                     "Current Assets": [100, 300, 300], "Current Liabilities": [200, 200, 200],
                     "Share Issued": [120, 100, 100]})
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(weak_inc, weak_bal, None))
    r = jp.fundamental_scan("W.T")
    assert r["f_score"] == 0 and r["f_strong"] is False
    assert r["cc_score"] == "—" and r["cc_qualifies"] == "FAIL"


def test_fundamental_scan_f_score_boundary(jp, monkeypatch):
    # drop exactly two signals (f7 dilution, f9 turnover) -> 7 = strong boundary; drop a third -> 6
    bal = STRONG_BAL.copy()
    bal.loc["Share Issued", "y0"] = 101        # f7 fails
    inc = STRONG_INC.copy()
    inc.loc["Total Revenue", "y0"] = 1000      # asset turnover equal -> f9 fails (strict >), margin still up
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(inc, bal, STRONG_CF))
    r = jp.fundamental_scan("B.T")
    assert r["f_score"] == 7 and r["f_strong"] is True
    inc.loc["Gross Profit", "y0"] = 100        # f8 fails too
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(inc, bal, STRONG_CF))
    r = jp.fundamental_scan("B.T")
    assert r["f_score"] <= 6 and r["f_strong"] is False


def test_fundamental_scan_no_data_and_exceptions(jp, monkeypatch):
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(None, None, None))
    assert jp.fundamental_scan("N.T")["error"] == "no_financial_data"

    def boom(t):
        raise RuntimeError("x" * 200)
    monkeypatch.setattr(jp.yf, "Ticker", boom)
    r = jp.fundamental_scan("N.T")
    assert r["error"] == "x" * 80 and r["f_score"] is None
    # an exception while scoring leaves f_score unset and returns early
    monkeypatch.setattr(jp, "_row", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr(jp.yf, "Ticker", lambda t: _ticker(STRONG_INC, STRONG_BAL, STRONG_CF))
    assert jp.fundamental_scan("N.T")["f_score"] is None


# ------------------------------------------------------------------ main
class _Liq:
    @staticmethod
    def gate(df, ticker):
        return (0.0, None) if ticker == "SKIP.T" else (5e6, "LIQUID")


def _setup_main(jp, monkeypatch, signals, universe=None, cached=None, argv=("prog",)):
    universe = universe or [{"yf_ticker": t, "code": t[:-2], "name": f"N{t}", "sector": "S"} for t in signals]
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(jp, "build_ticker_universe", lambda: universe)
    monkeypatch.setattr(jp, "bulk_download_ohlc", lambda tk, period="1y": {t: _ohlc(10) for t in tk})
    monkeypatch.setattr(jp, "compute_darvas_box", lambda df, confirm=3: signals_by_df[id(df)])
    monkeypatch.setattr(jp, "_liq", _Liq)
    fc = types.ModuleType("fundamentals_cache")
    fc.load = lambda pat, key: (cached or {}, "cached.xlsx") if cached is not None else (_ for _ in ()).throw(RuntimeError("no cache"))
    monkeypatch.setitem(sys.modules, "fundamentals_cache", fc)
    return universe


signals_by_df = {}


def _bulk_with_signals(jp, monkeypatch, signals):
    frames = {t: _ohlc(10) for t in signals}
    signals_by_df.clear()
    for t, sig in signals.items():
        signals_by_df[id(frames[t])] = {"signal": sig, "box_top": 12.0, "box_bottom": 8.0, "upside_pct": 5.0 if sig == "BREAKOUT_BUY" else -3.0,
                                         "pos_in_box": 50.0, "data_points": 10}
    monkeypatch.setattr(jp, "bulk_download_ohlc", lambda tk, period="1y": {t: frames[t] for t in tk if t in frames})


def test_main_full_pipeline_writes_workbook(jp, monkeypatch, tmp_path, capsys):
    signals = {"AAA.T": "BREAKOUT_BUY", "BBB.T": "BREAKDOWN_SELL", "CCC.T": "IN_BOX", "SKIP.T": "BREAKOUT_BUY", "DDD.T": "BREAKOUT_BUY"}
    _setup_main(jp, monkeypatch, signals, cached={"DDD.T": {"YF_Ticker": "DDD.T", "Piotroski_Strong": "yes", "CoffeeCan": "pass",
                                                              "Darvas_Signal": "OLD", "LTP_JPY": 1, "Piotroski_Score": 8, "Code": "DDD", "Name": "D", "CC_Score": "6/6"}})
    _bulk_with_signals(jp, monkeypatch, signals)
    monkeypatch.setattr(jp, "fundamental_scan", lambda t: {"yf_ticker": t, "f_score": 8, "f_strong": True, "cc_qualifies": "PASS",
                                                          "cc_score": "6/6", "cc_rev_cagr": 12.0, "cc_roce_avg": 20.0, "cc_debt_equity": 0.1, "market_cap_b_jpy": 150.0})
    jp.main()
    out = capsys.readouterr().out
    assert "Breakout BUY:   2" in out          # SKIP.T gated out; AAA + DDD
    assert "Breakdown SELL: 1" in out and "In Box:         1" in out
    assert "reusing 1 fundamentals from cached.xlsx" in out and "merged 1 reused fundamentals" in out
    assert "TRIPLE HITS:         2" in out
    files = list((tmp_path / "japan_scan").glob("japan_market_scan_*.xlsx"))
    assert len(files) == 1
    sheets = pd.read_excel(files[0], sheet_name=None)
    assert set(sheets) == {"All_Stocks", "Darvas_Signals", "Fundamentals", "Triple_Hits"}
    assert set(sheets["All_Stocks"]["YF_Ticker"]) == {"AAA.T", "BBB.T", "CCC.T", "DDD.T"}
    assert set(sheets["Triple_Hits"]["YF_Ticker"]) == {"AAA.T", "DDD.T"}
    ddd = sheets["Fundamentals"].set_index("YF_Ticker").loc["DDD.T"]
    assert ddd["Darvas_Signal"] == "BREAKOUT_BUY"   # price-derived fields refreshed from today's scan
    assert ddd["LTP_JPY"] == 10
    allrows = sheets["All_Stocks"].set_index("YF_Ticker")
    assert allrows.loc["AAA.T", "Trend_Signal"] == "Insufficient History"
    assert allrows.loc["AAA.T", "Change%"] == pytest.approx(11.11, abs=0.01)


def test_main_no_scans_top_limit_and_no_triples(jp, monkeypatch, tmp_path, capsys):
    signals = {"AAA.T": "BREAKOUT_BUY", "BBB.T": "IN_BOX", "CCC.T": "IN_BOX"}
    _setup_main(jp, monkeypatch, signals, argv=("prog", "--top", "2", "--no-scans"))
    _bulk_with_signals(jp, monkeypatch, signals)
    monkeypatch.setattr(jp, "fundamental_scan", lambda t: pytest.fail("scans disabled"))
    jp.main()
    out = capsys.readouterr().out
    assert "(limited to first 2 tickers)" in out and "Stage 4 — Skipped" in out
    assert "TSE stocks scanned:    2" in out and "No Triple Hit stocks found today." in out
    f = next((tmp_path / "japan_scan").glob("japan_market_scan_*.xlsx"))
    sheets = pd.read_excel(f, sheet_name=None)
    assert sheets["Fundamentals"].empty and sheets["Triple_Hits"].empty


def test_main_caps_candidates_and_survives_failed_scan(jp, monkeypatch, tmp_path, capsys):
    signals = {f"T{i}.T": "BREAKOUT_BUY" for i in range(5)}
    _setup_main(jp, monkeypatch, signals, argv=("prog", "--workers", "1"))
    _bulk_with_signals(jp, monkeypatch, signals)
    # upside differs per ticker so the cap keeps the smallest |upside|
    for i, t in enumerate(signals):
        pass
    monkeypatch.setattr(jp, "MAX_FUND_CANDIDATES", 2)
    for df_id, d in list(signals_by_df.items()):
        d["upside_pct"] = 5.0
    ids = list(signals_by_df)
    for rank, df_id in enumerate(ids):
        signals_by_df[df_id]["upside_pct"] = float(rank + 1)

    def fake_scan(t):
        if t == "T1.T":
            raise RuntimeError("scan failed")
        return {"yf_ticker": t, "f_score": 3, "f_strong": False}

    monkeypatch.setattr(jp, "fundamental_scan", fake_scan)
    jp.main()
    out = capsys.readouterr().out
    assert "(capped to 2 freshest breakouts)" in out
    assert "T1.T: error — scan failed" in out
    f = next((tmp_path / "japan_scan").glob("japan_market_scan_*.xlsx"))
    fund = pd.read_excel(f, sheet_name="Fundamentals")
    assert list(fund["YF_Ticker"]) == ["T0.T"]  # the two lowest-upside names were scanned; T1 failed
    assert fund["Piotroski_Strong"].iloc[0] == "NO"
