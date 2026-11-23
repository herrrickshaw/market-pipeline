"""Offline tests for full_european_market_scan (universe CSV, bulk download, fundamental scan, styling, main)."""

import importlib
import sys
import types

import numpy as np
import openpyxl
import pandas as pd
import pytest


@pytest.fixture
def eu(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # module creates ./european_scan at import
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: pd.DataFrame()
    fake_yf.Ticker = lambda t: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    sys.modules.pop("full_european_market_scan", None)
    mod = importlib.import_module("full_european_market_scan")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "european_scan")
    yield mod
    sys.modules.pop("full_european_market_scan", None)


def _ohlc(n=30, start=10.0):
    c = np.linspace(start, start + n - 1, n)
    return pd.DataFrame({"Close": c, "High": c + 0.5, "Low": c - 0.5, "Open": c, "Volume": 1000.0})


# ------------------------------------------------------------------ universe
def test_builtin_meta_is_euro_stoxx_50(eu):
    assert len(eu.EURO_STOXX_50_META) == 50
    assert eu.EURO_STOXX_50_META["SAP.DE"] == ("SAP", "Technology")


def test_load_universe_skips_blank_tickers_and_strips_names(eu, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("yf_ticker,name,index,exchange\nSAP.DE,SAP SE,DAX,XETRA\n  ,Nameless,X,Y\nABC.PA,\"  Padded  \",CAC,PAR\n")
    meta = eu.load_universe(str(p))
    assert meta == {"SAP.DE": ("SAP SE", "—"), "ABC.PA": ("Padded", "—")}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="load_universe: a blank CSV name is NaN, str(NaN) is the truthy 'nan', so the `or tkr` fallback never fires")
def test_load_universe_blank_name_falls_back_to_ticker(eu, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("yf_ticker,name,index,exchange\nRIO.L,,FTSE,LSE\n")
    assert eu.load_universe(str(p)) == {"RIO.L": ("RIO.L", "—")}


# ------------------------------------------------------------------ bulk download
def test_bulk_download_prefers_cache(eu, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    seen = {}
    oc.get = lambda m, s, yf_suffix, period: seen.update(m=m, p=period, sfx=yf_suffix) or {"A": _ohlc()}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    assert list(eu.bulk_download(["A", "B"], 10)) == ["A"]
    assert seen == {"m": "EUROPE", "p": "1y", "sfx": ""}
    assert "ohlcv_cache: 1/2 tickers" in capsys.readouterr().out


def test_bulk_download_direct_batches_and_failures(eu, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")
    oc.get = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    cols = pd.MultiIndex.from_product([["A", "B"], ["Close"]])
    multi = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=cols)
    multi[("B", "Close")] = np.nan  # B has nothing -> dropna empties it
    calls = {"n": 0}

    def dl(batch, **kw):
        calls["n"] += 1
        assert kw["period"] == "1y" and kw["group_by"] == "ticker"
        if calls["n"] == 1:
            return multi
        if calls["n"] == 2:
            raise RuntimeError("throttled")
        return pd.DataFrame({"Close": [1.0, 2.0]})  # flat frame (single ticker)
    eu.yf.download = dl
    out = eu.bulk_download(["A", "B", "C", "D", "E"], 2)
    assert set(out) == {"A", "E"}
    text = capsys.readouterr().out
    assert "3 batches" in text and "OK (1/2)" in text and "FAILED (throttled)" in text and "OK (1/1)" in text


def test_bulk_download_cache_exception_then_missing_symbol(eu, monkeypatch, capsys):
    oc = types.ModuleType("ohlcv_cache")

    def boom(*a, **k):
        raise RuntimeError("x" * 90)
    oc.get = boom
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    cols = pd.MultiIndex.from_product([["A"], ["Close"]])
    eu.yf.download = lambda b, **k: pd.DataFrame([[1.0]], columns=cols)
    assert eu.bulk_download(["A", "ZZ"], 10).keys() == {"A"}  # ZZ -> KeyError swallowed
    assert "ohlcv_cache unavailable: " + "x" * 60 + " — direct" in capsys.readouterr().out


# ------------------------------------------------------------------ helpers
def test_row_series_and_darvas_wrapper(eu):
    fin = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]}, index=["X", "Y"])
    assert eu._row(fin, "X") == 1.0 and eu._row(fin, "X", col=1) == 2.0
    assert eu._row(fin, "Y") is None and eu._row(fin, "Q") is None
    assert eu._row(None, "X") is None and eu._row(pd.DataFrame(), "X") is None
    assert eu._row(fin, "Q", "X") == 1.0
    assert eu._series(fin, "Y") == [3.0] and eu._series(fin, "Q") == [] and eu._series(None, "X") == []
    # a price frame with no valid close at all is reported as NO_DATA by _row (price-frame guard)
    px = pd.DataFrame({"Close": [np.nan, np.nan]})
    assert eu._row(px, "Close") == {"signal": "NO_DATA", "box_top": None, "box_bottom": None, "current_price": None}
    assert eu.compute_darvas_box(_ohlc(30))["signal"] == "NO_BOX"


# ------------------------------------------------------------------ fundamental scan
def _fin(rows):
    return pd.DataFrame(rows).T.rename(columns={0: "y0", 1: "y1", 2: "y2"})


INC = _fin({"Net Income": [150, 100, 80], "Total Revenue": [1500, 1000, 800], "Gross Profit": [700, 400, 300], "EBIT": [300, 200, 150]})
BAL = _fin({"Total Assets": [1000, 1000, 900], "Long Term Debt": [50, 100, 100], "Current Assets": [500, 400, 300],
            "Current Liabilities": [200, 200, 200], "Share Issued": [100, 100, 100], "Stockholders Equity": [600, 500, 400]})
CF = _fin({"Operating Cash Flow": [200, 150, 100], "Free Cash Flow": [120, 100, 80]})


class _Tk:
    def __init__(self, inc=INC, bal=BAL, cf=CF, info=None, info_raises=False, fast_cap=2e9):
        self.income_stmt, self.balance_sheet, self.cash_flow = inc, bal, cf
        self._info, self._raises = info if info is not None else {}, info_raises
        self.fast_info = types.SimpleNamespace(market_cap=fast_cap)

    @property
    def info(self):
        if self._raises:
            raise RuntimeError("info blocked")
        return self._info


def test_fundamental_scan_all_pass_and_valuation(eu, monkeypatch):
    info = {"debtToEquity": 45.0, "marketCap": 5e9, "trailingPE": 18.456, "pegRatio": 1.2, "dividendYield": 0.03, "priceToBook": 2.345}
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info=info))
    r = eu.fundamental_scan("SAP.DE")
    assert r["f_score"] == 9 and r["cc_qualifies"] == "PASS" and r["cc_score"] == "6/6"
    assert r["cc_debt_equity"] == 0.45  # >10 treated as a percentage
    assert r["market_cap_b"] == 5.0 and r["pe_ratio"] == 18.46 and r["pb_ratio"] == 2.35
    assert r["dividend_yield"] == 3.0 and r["pegy_ratio"] == pytest.approx(4.2)


def test_fundamental_scan_dividend_yield_scale_boundary(eu, monkeypatch):
    for raw, expect in ((0.14, 14.0), (0.15, 0.15), (3.2, 3.2)):  # >=0.15 already a percent
        monkeypatch.setattr(eu.yf, "Ticker", lambda s, raw=raw: _Tk(info={"dividendYield": raw, "marketCap": 1e9}))
        assert eu.fundamental_scan("X")["dividend_yield"] == pytest.approx(expect)


def test_fundamental_scan_info_failure_uses_statements_and_fast_info(eu, monkeypatch):
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info_raises=True, fast_cap=2e9))
    r = eu.fundamental_scan("X")
    assert r["cc_debt_equity"] == 0.08  # fell back to long-term debt / equity from the statements
    assert r["cc_score"] == "6/6" and r["market_cap_b"] == 2.0
    assert r["pe_ratio"] is None and r["pegy_ratio"] is None and r["dividend_yield"] == 0.0


def test_fundamental_scan_mcap_boundary_and_de_none(eu, monkeypatch):
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"marketCap": 1e9 - 1}))
    assert eu.fundamental_scan("X")["cc_score"] == "5/6"
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"marketCap": 1e9}))
    assert eu.fundamental_scan("X")["cc_score"] == "6/6"
    # debtToEquity <=10 is taken as a ratio; 10 is exactly the boundary (not divided)
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"marketCap": 1e9, "debtToEquity": 10}))
    assert eu.fundamental_scan("X")["cc_debt_equity"] == 10
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"marketCap": 1e9, "debtToEquity": 10.5}))
    assert eu.fundamental_scan("X")["cc_debt_equity"] == 0.1


def test_fundamental_scan_pe_fallbacks(eu, monkeypatch):
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"forwardPE": 12.0, "marketCap": 1e9}))
    assert eu.fundamental_scan("X")["pe_ratio"] == 12.0
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"regularMarketPrice": 50.0, "trailingEps": 2.0, "marketCap": 1e9}))
    assert eu.fundamental_scan("X")["pe_ratio"] == 25.0
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(info={"trailingPE": -3.0, "marketCap": 1e9}))
    assert eu.fundamental_scan("X")["pe_ratio"] is None  # negative PE is not reported


def test_fundamental_scan_fcf_fallback_and_c1(eu, monkeypatch):
    cf = _fin({"Operating Cash Flow": [200, 1, 1], "Capital Expenditure": [-300, -1, -1]})
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(cf=cf, info={"marketCap": 5e9}))
    assert eu.fundamental_scan("X")["cc_score"] == "5/6"  # OCF - |capex| < 0 fails C6
    slow = _fin({"Net Income": [150, 100, 80], "Total Revenue": [1050, 1000, 1000], "Gross Profit": [700, 400, 300], "EBIT": [300, 200, 150]})
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(inc=slow, info={"marketCap": 5e9}))
    r = eu.fundamental_scan("X")
    assert r["cc_score"] == "5/6" and r["cc_rev_cagr"] == pytest.approx(2.47, abs=0.01)  # C1 needs >10% CAGR


def test_fundamental_scan_errors(eu, monkeypatch):
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(inc=None))
    assert eu.fundamental_scan("X")["error"] == "no_financial_statements"
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk(inc=pd.DataFrame()))
    assert eu.fundamental_scan("X")["error"] == "no_financial_statements"

    def boom(s):
        raise RuntimeError("z" * 120)
    monkeypatch.setattr(eu.yf, "Ticker", boom)
    assert eu.fundamental_scan("X")["error"] == "z" * 80
    # scoring exceptions are swallowed and leave f_score None
    monkeypatch.setattr(eu.yf, "Ticker", lambda s: _Tk())
    monkeypatch.setattr(eu, "_row", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    assert eu.fundamental_scan("X")["f_score"] is None


# ------------------------------------------------------------------ styling
def test_style_excel_sheet_formats(eu, tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Symbol", "Change%", "LTP", "PEG", "Name", "Dividend_Yield%"])
    ws.append(["SAP.DE", 1.5, 120.123, 1.2, "SAP", 3])
    ws.append(["X", None, None, None, "x", None])
    eu.style_excel_sheet(ws)
    assert ws["A1"].font.bold and ws["A1"].fill.start_color.rgb.endswith("1A365D")
    assert ws["B2"].number_format == '0.00"%"' and ws["F2"].number_format == '0.00"%"'
    assert ws["C2"].number_format == "#,##0.00" and ws["D2"].number_format == "0.00"
    assert ws["A2"].alignment.horizontal == "center" and ws["E2"].alignment.horizontal == "left"
    assert ws["B2"].alignment.horizontal == "right"
    assert ws.row_dimensions[1].height == 28 and ws.row_dimensions[2].height == 20
    assert ws.column_dimensions["A"].width == 12 and ws.column_dimensions["F"].width == len("Dividend_Yield%") + 4
    # zebra fill: odd row index (3) is shaded, even (2) is not
    assert ws["A3"].fill.start_color.rgb.endswith("F7FAFC") and ws["A2"].fill.fill_type is None


def test_style_excel_sheet_noop_without_openpyxl(eu):
    eu.OPENPYXL_OK = False
    ws = types.SimpleNamespace()
    assert eu.style_excel_sheet(ws) is None


# ------------------------------------------------------------------ main
class _Liq:
    @staticmethod
    def gate(df, ticker):
        return (0.0, None) if ticker == "SKIP.DE" else (3e6, "LIQUID")


def _setup(eu, monkeypatch, signals, argv=("prog",), cached=None, meta=None):
    monkeypatch.setattr(sys, "argv", list(argv))
    meta = meta or {s: (f"N{s}", "Sec") for s in signals}
    monkeypatch.setattr(eu, "EURO_STOXX_50_META", meta)
    # frames are told apart by their first close (df[sym] hands back a copy, so id() is useless)
    frames = {s: _ohlc(30, start=100.0 * (i + 1)) for i, s in enumerate(signals)}
    sig_by_id = {float(frames[s]["Close"].iloc[0]): {"signal": sig, "box_top": 40.0, "box_bottom": 30.0, "upside_pct": 5.0,
                                                    "pos_in_box": 20.0} for s, sig in signals.items()}
    monkeypatch.setattr(eu, "compute_darvas_box", lambda df, confirm=3: sig_by_id[float(df["Close"].iloc[0])])
    monkeypatch.setattr(eu, "_liq", _Liq)
    monkeypatch.setattr(eu.yf, "download", lambda syms, **k: pd.concat({s: frames[s] for s in syms}, axis=1))
    fc = types.ModuleType("fundamentals_cache")
    if cached is None:
        fc.load = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("none"))
    else:
        fc.load = lambda pat, key: (cached, "prev.xlsx")
    monkeypatch.setitem(sys.modules, "fundamentals_cache", fc)
    return frames


def test_main_builtin_universe_full_flow(eu, monkeypatch, tmp_path, capsys):
    signals = {"AAA.DE": "BREAKOUT_BUY", "BBB.DE": "BREAKDOWN_SELL", "SKIP.DE": "BREAKOUT_BUY",
               "CCC.DE": "BREAKOUT_BUY", "DDD.DE": "BREAKOUT_BUY", "SHORT.DE": "IN_BOX"}
    frames = _setup(eu, monkeypatch, signals, argv=("prog", "--workers", "1"),
                    cached={"DDD.DE": {"Symbol": "DDD.DE", "Piotroski_Score": 8, "LTP": 1.0, "Darvas_Signal": "OLD"}})
    # a too-short series is skipped before the liquidity gate
    orig_dl = eu.yf.download
    monkeypatch.setattr(eu.yf, "download", lambda syms, **k: orig_dl(syms, **k))
    frames["SHORT.DE"] = frames["SHORT.DE"].iloc[:10]

    def fund(sym):
        if sym == "CCC.DE":
            raise RuntimeError("bad scan")
        return {"f_score": 8, "cc_qualifies": "PASS", "cc_score": "6/6", "pe_ratio": 15.0, "pb_ratio": 2.0,
                "dividend_yield": 1.5, "pegy_ratio": 3.0, "market_cap_b": 4.0, "cc_rev_cagr": 12.0}
    monkeypatch.setattr(eu, "fundamental_scan", fund)
    eu.main()
    out = capsys.readouterr().out
    assert "OPTIMIZED EURO STOXX 50 SCANNER" in out
    assert "Processed 4 stocks. Found 3 Darvas breakouts." in out  # SKIP gated, SHORT too short -> 4 left; AAA, CCC, DDD
    assert "reusing 1 fundamentals from prev.xlsx" in out and "merged 1 reused" in out
    assert "Fundamental error on CCC.DE: bad scan" in out
    assert "TRIPLE HITS FOUND:          1" in out and "AAA.DE" in out
    f = next((tmp_path / "european_scan").glob("european_market_scan_*.xlsx"))
    sheets = pd.read_excel(f, sheet_name=None)
    assert set(sheets) == {"All_Stocks", "Darvas_Signals", "Fundamentals", "Triple_Hits"}
    assert set(sheets["All_Stocks"]["Symbol"]) == {"AAA.DE", "BBB.DE", "CCC.DE", "DDD.DE"}
    assert set(sheets["Darvas_Signals"]["Symbol"]) == {"AAA.DE", "BBB.DE", "CCC.DE", "DDD.DE"}
    allstk = sheets["All_Stocks"].set_index("Symbol")
    assert allstk.loc["AAA.DE", "P_E_Ratio"] == 15.0 and allstk.loc["AAA.DE", "Market_Cap_B"] == 4.0
    assert allstk.loc["AAA.DE", "Trend_Signal"] == "Insufficient History"
    fund_sheet = sheets["Fundamentals"].set_index("Symbol")
    assert set(fund_sheet.index) == {"AAA.DE", "CCC.DE", "DDD.DE"} or set(fund_sheet.index) == {"AAA.DE", "DDD.DE"}
    assert fund_sheet.loc["DDD.DE", "Darvas_Signal"] == "BREAKOUT_BUY"
    assert list(sheets["Triple_Hits"]["Symbol"]) == ["AAA.DE"]


def test_main_broad_universe_label_top_and_no_scans(eu, monkeypatch, tmp_path, capsys):
    csv = tmp_path / "broad.csv"
    csv.write_text("yf_ticker,name,index,exchange\nAAA.L,Alpha,FTSE,LSE\nBBB.L,Beta,FTSE,LSE\nCCC.L,Gamma,FTSE,LSE\n")
    signals = {"AAA.L": "BREAKOUT_BUY", "BBB.L": "IN_BOX", "CCC.L": "IN_BOX"}
    frames = _setup(eu, monkeypatch, signals, argv=("prog", "--universe", str(csv), "--label", "broad", "--top", "2", "--no-scans", "--batch", "1"))
    oc = types.ModuleType("ohlcv_cache")
    oc.get = lambda m, s, yf_suffix, period: {x: frames[x] for x in s}
    monkeypatch.setitem(sys.modules, "ohlcv_cache", oc)
    monkeypatch.setattr(eu, "fundamental_scan", lambda s: pytest.fail("scans disabled"))
    eu.main()
    out = capsys.readouterr().out
    assert "BROADER EUROPE SCANNER" in out and "for 2 stocks" in out
    assert "Stage 4 — Skipped" in out and "No Triple Hit stocks found today." in out
    f = next((tmp_path / "european_scan").glob("european_market_scan_broad_*.xlsx"))
    sheets = pd.read_excel(f, sheet_name=None)
    assert set(sheets["All_Stocks"]["Symbol"]) == {"AAA.L", "BBB.L"}
    assert sheets["Fundamentals"].empty and sheets["Triple_Hits"].empty
    assert eu.MAX_WORKERS == 10


def test_main_bulk_download_failure_exits(eu, monkeypatch):
    _setup(eu, monkeypatch, {"AAA.DE": "IN_BOX"})

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(eu.yf, "download", boom)
    with pytest.raises(SystemExit) as ei:
        eu.main()
    assert "Failed to download bulk OHLC data: network down" in str(ei.value)


def test_main_flat_frame_download_and_technical_error(eu, monkeypatch, tmp_path, capsys):
    _setup(eu, monkeypatch, {"AAA.DE": "IN_BOX"}, argv=("prog", "--no-scans"))
    flat = _ohlc(30)
    monkeypatch.setattr(eu.yf, "download", lambda syms, **k: flat)  # non-MultiIndex frame
    monkeypatch.setattr(eu, "compute_darvas_box", lambda df, confirm=3: (_ for _ in ()).throw(ValueError("darvas broke")))
    eu.main()
    out = capsys.readouterr().out
    assert "Technical error on AAA.DE: darvas broke" in out and "Processed 0 stocks" in out


@pytest.mark.parametrize("score,expected_hits", [(7, ["AAA.DE"]), (6, [])])
def test_main_triple_hit_requires_f_score_of_at_least_7(eu, monkeypatch, tmp_path, score, expected_hits):
    _setup(eu, monkeypatch, {"AAA.DE": "BREAKOUT_BUY"}, argv=("prog", "--workers", "1"))
    monkeypatch.setattr(eu, "fundamental_scan", lambda s: {"f_score": score, "cc_qualifies": "PASS", "cc_score": "6/6"})
    eu.main()
    f = next((tmp_path / "european_scan").glob("european_market_scan_*.xlsx"))
    hits = pd.read_excel(f, sheet_name="Triple_Hits")
    assert list(hits["Symbol"]) == expected_hits if expected_hits else hits.empty
