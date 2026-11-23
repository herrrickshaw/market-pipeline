"""ipo_tracker.py -- new-listing discovery by bhavcopy diff, OHLC download/cache, enrichment,
age-gated screening, Excel export and the main() pipeline. Every external dependency is faked;
the module does sys.exit() without yfinance and creates ./ipo_results at import, so it is imported
in a sandbox cwd with a fake yfinance."""

import importlib
import json
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def IT(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yf = types.SimpleNamespace(
        download=lambda *a, **k: pd.DataFrame(), Ticker=lambda s: types.SimpleNamespace(info={})
    )
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    sys.modules.pop("ipo_tracker", None)
    mod = importlib.import_module("ipo_tracker")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    sys.modules.pop("ipo_tracker", None)


def ohlc(n, start="2024-01-02", base=100.0):
    idx = pd.bdate_range(start, periods=n)
    c = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c + 2, "Low": c - 2, "Close": c, "Volume": 1000.0}, index=idx
    )


# ── discovery ─────────────────────────────────────────────────────────────────
def install_nse(monkeypatch, tmp_path, by_offset, boom_offsets=()):
    class NSE:
        def __init__(self, download_folder, server):
            NSE.folder, NSE.server = download_folder, server

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def equityBhavcopy(self, d):
            offset = (datetime.today().date() - d.date()).days
            if offset in boom_offsets:
                raise OSError("holiday")
            if offset not in by_offset:
                return None
            p = tmp_path / f"b{offset}.csv"
            pd.DataFrame(
                {
                    "SctySrs": ["EQ"] * len(by_offset[offset]) + ["BE"],
                    "TckrSymb": [f" {s} " for s in by_offset[offset]] + ["JUNK"],
                }
            ).to_csv(p, index=False)
            return p

    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=NSE))
    return NSE


def test_discover_new_listings_is_the_set_difference(IT, monkeypatch, tmp_path, capsys):
    NSE = install_nse(
        monkeypatch, tmp_path, {2: ["OLD", "NEW2", "NEW1", "OLD2"], 85: ["OLD", "OLD2", "GONE"]}
    )
    out = IT.discover_new_listings(lookback_days=90)
    assert out == ["NEW1", "NEW2"] and NSE.server is False
    text = capsys.readouterr().out
    assert "NEW LISTINGS in last 90 days: 2" in text and "Sample: ['NEW1', 'NEW2']" in text


def test_discover_walks_back_to_the_first_available_day(IT, monkeypatch, tmp_path):
    install_nse(monkeypatch, tmp_path, {5: ["A", "B"], 89: ["A"]}, boom_offsets={0, 1, 2, 3, 4})
    assert IT.discover_new_listings(lookback_days=90, verbose=False) == ["B"]


def test_discover_past_window_bounds(IT, monkeypatch, tmp_path):
    install_nse(monkeypatch, tmp_path, {0: ["A", "B"], 85: ["A"]})
    assert IT.discover_new_listings(90, verbose=False) == ["B"]  # lookback-5 is in range
    install_nse(monkeypatch, tmp_path, {0: ["A", "B"], 84: ["A"]})
    assert IT.discover_new_listings(90, verbose=False) == []  # lookback-6 is not
    install_nse(monkeypatch, tmp_path, {0: ["A", "B"], 99: ["A"]})
    assert IT.discover_new_listings(90, verbose=False) == ["B"]  # lookback+9 is the last day
    install_nse(monkeypatch, tmp_path, {0: ["A", "B"], 100: ["A"]})
    assert IT.discover_new_listings(90, verbose=False) == []


def test_discover_returns_empty_without_either_side_or_library(IT, monkeypatch, tmp_path, capsys):
    install_nse(monkeypatch, tmp_path, {2: ["A"]})
    assert IT.discover_new_listings(90) == []
    assert "Could not fetch bhavcopy for diff" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "nse", None)
    assert IT.discover_new_listings(90) == []
    assert "nse library not available" in capsys.readouterr().out


# ── download ──────────────────────────────────────────────────────────────────
def install_cache(monkeypatch, tmp_path, cached=None, ohlc_dir=None):
    ohlc_dir = ohlc_dir or tmp_path / "ohlc"
    ohlc_dir.mkdir(exist_ok=True)

    class MarketCache:
        def __init__(self, verbose=False):
            pass

        def get_ohlc(self, key):
            return (cached or {}).get(key)

    mod = types.SimpleNamespace(
        MarketCache=MarketCache, OHLC_DIR=ohlc_dir, META_FILE=tmp_path / "meta.json"
    )
    monkeypatch.setitem(sys.modules, "market_data_cache", mod)
    return mod


def multi_download(tickers, n=30):
    frames = {}
    for t in tickers:
        o = ohlc(n)
        for c in o.columns:
            frames[(c, t)] = o[c]
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_download_serves_cached_symbols_without_the_network(IT, monkeypatch, tmp_path, capsys):
    install_cache(monkeypatch, tmp_path, cached={"AAA.NS": ohlc(5), "BBB.NS": ohlc(6)})
    monkeypatch.setattr(IT.yf, "download", lambda *a, **k: pytest.fail("no download expected"))
    out = IT.download_ipo_ohlc(["AAA", "BBB"])
    assert (
        set(out) == {"AAA", "BBB"}
        and "All 2 IPO stocks loaded from cache" in capsys.readouterr().out
    )


def test_download_fetches_missing_symbols_and_persists_them(IT, monkeypatch, tmp_path):
    cache = install_cache(monkeypatch, tmp_path, cached={"AAA.NS": ohlc(5)})
    seen = {}

    def dl(batch, **kw):
        seen["batch"], seen["kw"] = list(batch), kw
        return multi_download(batch)

    monkeypatch.setattr(IT.yf, "download", dl)
    out = IT.download_ipo_ohlc(["AAA", "BBB", "CCC"], verbose=False)
    assert set(out) == {"AAA", "BBB", "CCC"} and seen["batch"] == ["BBB.NS", "CCC.NS"]
    assert seen["kw"]["period"] == "max" and list(out["BBB"].columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert (cache.OHLC_DIR / "BBB.NS.parquet").exists() and not (
        cache.OHLC_DIR / "AAA.NS.parquet"
    ).exists()
    meta = json.loads(cache.META_FILE.read_text())
    assert meta["ohlc:BBB.NS"]["ipo"] is True and meta["ohlc:BBB.NS"]["rows"] == 30
    assert meta["ohlc:BBB.NS"]["from"] == "2024-01-02"


def test_download_batches_by_fifty_and_handles_single_ticker_frames(IT, monkeypatch, tmp_path):
    install_cache(monkeypatch, tmp_path)
    batches = []
    monkeypatch.setattr(
        IT.yf,
        "download",
        lambda b, **k: batches.append(list(b)) or (ohlc(30) if len(b) == 1 else multi_download(b)),
    )
    syms = [f"S{i:03d}" for i in range(101)]
    out = IT.download_ipo_ohlc(syms, verbose=False)
    assert [len(b) for b in batches] == [50, 50, 1] and len(out) == 101


def test_download_without_a_cache_module_downloads_everything(IT, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "market_data_cache", None)
    monkeypatch.setattr(IT.yf, "download", lambda b, **k: multi_download(b))
    assert set(IT.download_ipo_ohlc(["AAA"], verbose=False)) == {"AAA"}


def test_download_retries_rate_limits_then_gives_up_on_other_errors(
    IT, monkeypatch, tmp_path, capsys
):
    install_cache(monkeypatch, tmp_path)
    waits, calls = [], []
    monkeypatch.setattr(IT.time, "sleep", lambda s: waits.append(s))

    def dl(b, **k):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 Too Many Requests")
        return multi_download(b)

    monkeypatch.setattr(IT.yf, "download", dl)
    out = IT.download_ipo_ohlc(["AAA"], verbose=True)
    assert set(out) == {"AAA"} and waits[:2] == [20, 40] and len(calls) == 3
    monkeypatch.setattr(
        IT.yf, "download", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    assert IT.download_ipo_ohlc(["ZZZ"], verbose=True) == {}
    assert "ERROR — boom" in capsys.readouterr().out


def test_download_empty_frame_is_skipped(IT, monkeypatch, tmp_path, capsys):
    install_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(IT.yf, "download", lambda *a, **k: pd.DataFrame())
    assert IT.download_ipo_ohlc(["AAA"], verbose=True) == {}
    assert "empty" in capsys.readouterr().out


def test_download_disabled_without_yfinance(IT, monkeypatch):
    monkeypatch.setattr(IT, "_YF_OK", False)
    assert IT.download_ipo_ohlc(["AAA"]) == {}


# ── enrichment ────────────────────────────────────────────────────────────────
def install_pe(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "stock_enricher",
        types.SimpleNamespace(
            _classify_pe=lambda pe, sector, mkt: {"emoji": "🟢", "zone": f"cheap-{sector}-{mkt}"}
        ),
    )


def test_enrich_derives_prices_gain_and_screener_availability(IT, monkeypatch):
    install_pe(monkeypatch)
    info = {"shortName": " Alpha Ltd ", "sector": "Tech", "marketCap": 5e9, "trailingPE": 23.456}
    monkeypatch.setattr(IT.yf, "Ticker", lambda s: types.SimpleNamespace(info=info))
    df = ohlc(250, start="2023-01-02")
    r = IT.enrich_ipo("ALPHA", df)
    assert r["Symbol"] == "ALPHA" and r["Company_Name"] == "Alpha Ltd" and r["Sector"] == "Tech"
    assert r["Trading_Bars"] == 250 and r["Listing_Date"] == "2023-01-02"
    assert (
        r["Listing_Price_₹"] == 100.0
        and r["Current_LTP_₹"] == 349.0
        and r["Listing_Gain_%"] == 249.0
    )
    assert r["ATH_₹"] == 351.0 and r["ATL_₹"] == 98.0
    assert r["MCap_Cr"] == 500.0 and r["Trailing_PE"] == 23.5 and r["PE_Zone"] == "🟢 cheap-Tech-IN"
    assert r["Days_Listed"] == (datetime.today() - df.index[0]).days
    assert r["Screeners_Available"] == "Darvas, MagicFormula, GoldenCross"


@pytest.mark.parametrize(
    "bars,expected",
    [
        (34, "Price only"),
        (35, "Darvas"),
        (59, "Darvas"),
        (60, "Darvas, MagicFormula"),
        (199, "Darvas, MagicFormula"),
        (200, "Darvas, MagicFormula, GoldenCross"),
        (251, "Darvas, MagicFormula, GoldenCross"),
        (252, "Darvas, MagicFormula, GoldenCross, BullCartel"),
        (504, "Darvas, MagicFormula, GoldenCross, BullCartel, Piotroski"),
        (756, "Darvas, MagicFormula, GoldenCross, BullCartel, Piotroski, CoffeeCan"),
    ],
)
def test_enrich_screener_gates_by_trading_bars(IT, monkeypatch, bars, expected):
    monkeypatch.setattr(IT.yf, "Ticker", lambda s: types.SimpleNamespace(info={}))
    assert IT.enrich_ipo("X", ohlc(bars))["Screeners_Available"] == expected


def test_enrich_without_data_or_info(IT, monkeypatch):
    for df in (None, pd.DataFrame()):
        r = IT.enrich_ipo("X", df)
        assert r["Note"] == "no_ohlc" and r["Trading_Bars"] == 0 and r["Listing_Gain_%"] is None
    monkeypatch.setattr(IT.yf, "Ticker", lambda s: 1 / 0)
    r = IT.enrich_ipo("X", ohlc(40))
    assert r["Company_Name"] == "X" and r["Sector"] == "—" and r["MCap_Cr"] is None
    assert r["PE_Zone"] == "⚪ N/A" and r["Screeners_Available"] == "Darvas"


def test_enrich_uses_long_name_industry_and_blank_numbers(IT, monkeypatch):
    install_pe(monkeypatch)
    info = {"longName": "L" * 80, "industry": "Software", "marketCap": 0, "trailingPE": None}
    monkeypatch.setattr(IT.yf, "Ticker", lambda s: types.SimpleNamespace(info=info))
    r = IT.enrich_ipo("X", ohlc(40))
    assert r["Company_Name"] == "L" * 50 and r["Sector"] == "Software"
    assert r["MCap_Cr"] is None and r["Trailing_PE"] is None


# ── screening ─────────────────────────────────────────────────────────────────
def install_screeners(monkeypatch, darvas=None, box=None, gc=None, fs=None, ml=None, boom=()):
    def guard(name, fn):
        def run(*a, **k):
            if name in boom:
                raise RuntimeError(name)
            return fn(*a, **k)

        return run

    monkeypatch.setitem(
        sys.modules,
        "backtest_screeners",
        types.SimpleNamespace(detect_darvas_signals=guard("darvas", lambda df: darvas or [])),
    )
    monkeypatch.setitem(
        sys.modules,
        "full_indian_market_scan",
        types.SimpleNamespace(
            compute_darvas_box=guard("box", lambda df: box or {}),
            compute_golden_crossover=guard("gc", lambda df: gc or {}),
            fundamental_scan=guard("fs", lambda s, sfx: fs or {}),
        ),
    )

    class Engine:
        def __init__(self, model_type):
            assert model_type == "ridge"

        def predict(self, sym, df):
            if "ml" in boom:
                raise RuntimeError("ml")
            return ml or {}

    monkeypatch.setitem(
        sys.modules, "ml_signal_engine", types.SimpleNamespace(MLSignalEngine=Engine)
    )


def test_screen_young_listing_gets_no_signals(IT, monkeypatch):
    install_screeners(monkeypatch)
    assert IT.screen_ipo("X", ohlc(20), {"Trading_Bars": 20}) == {"Symbol": "X"}
    assert IT.screen_ipo("X", None, {"Trading_Bars": 500}) == {"Symbol": "X"}
    assert IT.screen_ipo("X", pd.DataFrame(), {"Trading_Bars": 500}) == {"Symbol": "X"}


def test_screen_darvas_breakout_vs_current_box(IT, monkeypatch):
    install_screeners(
        monkeypatch,
        darvas=[
            {"box_top": 10.0, "date": "2024-02-01T00:00"},
            {"box_top": 12.0, "date": "2024-03-01T00:00"},
        ],
    )
    s = IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 40})
    assert (
        s["Darvas_Signal"] == "BREAKOUT_BUY"
        and s["Darvas_BoxTop"] == 12.0
        and s["Darvas_Date"] == "2024-03-01"
    )
    install_screeners(
        monkeypatch,
        box={"signal": "INSIDE_BOX", "box_top": 5, "box_bottom": 4, "upside_to_top_pct": 3.2},
    )
    s = IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 40})
    assert (s["Darvas_Signal"], s["Box_Top"], s["Box_Bottom"], s["Upside_%"]) == (
        "INSIDE_BOX",
        5,
        4,
        3.2,
    )
    install_screeners(monkeypatch)
    assert IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 40})["Darvas_Signal"] == "NO_BOX"
    install_screeners(monkeypatch, boom={"darvas"})
    assert IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 40})["Darvas_Signal"] == "error: darvas"


def test_screen_darvas_gate_boundary(IT, monkeypatch):
    install_screeners(monkeypatch)
    assert "Darvas_Signal" not in IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 34})
    assert "Darvas_Signal" in IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 35})


def test_screen_golden_cross_states(IT, monkeypatch):
    for gc, label in (
        ({"gc_signal": True, "dma50": 10, "dma200": 9}, "GOLDEN_CROSS"),
        ({"gc_signal": False, "dma50_above_200": True}, "ABOVE_200DMA"),
        ({}, "BELOW_200DMA"),
    ):
        install_screeners(monkeypatch, gc=gc)
        s = IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 200})
        assert s["GC_Signal"] == label
    assert (s["DMA50"], s["DMA200"]) == (None, None)
    install_screeners(monkeypatch, gc={"gc_signal": True, "dma50": 10, "dma200": 9})
    s = IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 200})
    assert (s["DMA50"], s["DMA200"]) == (10, 9)
    assert "GC_Signal" not in IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 199})
    install_screeners(monkeypatch, boom={"gc"})
    assert IT.screen_ipo("X", ohlc(40), {"Trading_Bars": 200})["GC_Signal"] == "error: gc"


def test_screen_ml_signal_gate_and_defaults(IT, monkeypatch):
    install_screeners(monkeypatch, ml={"direction": "UP", "predicted_ret%": 1.5, "confidence": 0.7})
    s = IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 60})
    assert (s["ML_Direction"], s["ML_Pred_Ret%"], s["ML_Confidence"]) == ("UP", 1.5, 0.7)
    assert "ML_Direction" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 59})
    install_screeners(monkeypatch)
    s = IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 60})
    assert (s["ML_Direction"], s["ML_Pred_Ret%"], s["ML_Confidence"]) == ("NEUTRAL", 0, 0)
    install_screeners(monkeypatch, boom={"ml"})
    assert "ML_Direction" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 60})


def magic_ticker(ebit=2e8, assets=1e9, cl=2e8, info=None):
    cols = pd.to_datetime(["2024-03-31", "2023-03-31"])
    inc = pd.DataFrame({cols[0]: [ebit], cols[1]: [1.0]}, index=["EBIT"])
    bal = pd.DataFrame(
        {cols[0]: [assets, cl], cols[1]: [1.0, 1.0]}, index=["Total Assets", "Current Liabilities"]
    )
    return types.SimpleNamespace(
        income_stmt=inc, financials=None, balance_sheet=bal, info=info if info is not None else {}
    )


def test_screen_magic_formula_pass_and_fail(IT, monkeypatch):
    install_screeners(monkeypatch)
    good = {"marketCap": 1e9, "totalDebt": 0, "totalCash": 0, "bookValue": 10}

    def run(**kw):
        info = {**good, **kw.pop("info", {})}
        monkeypatch.setattr(IT.yf, "Ticker", lambda s: magic_ticker(info=info, **kw))
        return IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 70})

    s = run()  # ROIC 2e8/8e8 = 25%, EY 2e8/1e9 = 20%, mcap 100 Cr
    assert (s["MF_ROIC_%"], s["MF_EY_%"], s["MF_Pass"]) == (25.0, 20.0, "PASS")
    assert run(info={"marketCap": 1.51e8})["MF_Pass"] == "PASS"  # just above 15 Cr
    assert run(info={"marketCap": 1.5e8})["MF_Pass"] == "FAIL"  # exactly 15 Cr is not enough
    weak = run(ebit=1e8)
    assert (weak["MF_ROIC_%"], weak["MF_EY_%"], weak["MF_Pass"]) == (
        12.5,
        10.0,
        "FAIL",
    )  # ROIC <= 15
    assert run(info={"marketCap": 5e9})["MF_Pass"] == "FAIL"  # EY 4% <= 8
    assert run(info={"bookValue": -1})["MF_Pass"] == "FAIL"
    assert run(info={"totalDebt": 1e9, "totalCash": 0})["MF_EY_%"] == 10.0  # EV includes net debt
    assert "MF_Pass" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 59})
    monkeypatch.setattr(
        IT.yf, "Ticker", lambda s: types.SimpleNamespace(income_stmt=None, financials=None, info={})
    )
    assert "MF_Pass" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 70})  # no statements


def test_screen_piotroski_gate_and_fields(IT, monkeypatch):
    install_screeners(
        monkeypatch,
        fs={"f_score": 8, "f_strong": True, "qualifies_cc": False, "qualifies_bc": True},
    )
    monkeypatch.setattr(
        IT.yf, "Ticker", lambda s: types.SimpleNamespace(income_stmt=None, financials=None, info={})
    )
    s = IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 504})
    assert (s["Piotroski_Score"], s["Piotroski_Strong"], s["CoffeeCan"], s["BullCartel"]) == (
        8,
        "YES",
        "FAIL",
        "PASS",
    )
    assert "Piotroski_Score" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 503})
    install_screeners(monkeypatch, boom={"fs"})
    assert "Piotroski_Score" not in IT.screen_ipo("X", ohlc(70), {"Trading_Bars": 504})


# ── Excel export ──────────────────────────────────────────────────────────────
def meta_rows():
    return [
        {
            "Symbol": "AAA",
            "Sector": "Tech",
            "Days_Listed": 100,
            "Listing_Gain_%": 50.0,
            "Trailing_PE": 20.0,
            "MCap_Cr": 500.0,
        },
        {
            "Symbol": "BBB",
            "Sector": "Tech",
            "Days_Listed": 300,
            "Listing_Gain_%": -10.0,
            "Trailing_PE": 30.0,
            "MCap_Cr": 100.0,
        },
        {
            "Symbol": "CCC",
            "Sector": "Auto",
            "Days_Listed": 50,
            "Listing_Gain_%": 5.0,
            "Trailing_PE": 10.0,
            "MCap_Cr": 50.0,
        },
    ]


def test_save_ipo_excel_sheets_and_sorting(IT):
    signals = [
        {
            "Symbol": "AAA",
            "Darvas_Signal": "BREAKOUT_BUY",
            "Upside_%": 2.0,
            "ML_Direction": "UP",
            "ML_Pred_Ret%": 1.0,
            "MF_Pass": "PASS",
            "MF_ROIC_%": 30.0,
        },
        {
            "Symbol": "BBB",
            "Darvas_Signal": "NO_BOX",
            "Upside_%": 9.0,
            "ML_Direction": "DOWN",
            "ML_Pred_Ret%": -2.0,
            "MF_Pass": "FAIL",
            "MF_ROIC_%": 10.0,
        },
        {"Symbol": "CCC", "Darvas_Signal": "", "ML_Direction": "", "MF_Pass": ""},
    ]
    path = IT.save_ipo_excel(meta_rows(), signals, 90)
    assert path.parent == IT.OUT_DIR and path.name.startswith("ipo_tracker_")
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == [
        "DISCLAIMER",
        "IPO_Overview",
        "Price_Analysis",
        "By_Sector",
        "Darvas_Signals",
        "ML_Signals",
        "Fundamental_Scan",
        "Screener_Gates",
    ]
    assert list(sheets["IPO_Overview"]["Symbol"]) == ["BBB", "AAA", "CCC"]  # oldest listing first
    assert list(sheets["Price_Analysis"]["Symbol"]) == ["AAA", "CCC", "BBB"]  # best gain first
    sector = sheets["By_Sector"].set_index("Sector")
    assert sector.loc["Tech", "Count"] == 2 and sector.loc["Tech", "Avg_Listing_Gain"] == 20.0
    assert sector.loc["Tech", "Total_MCap_Cr"] == 600.0 and list(sheets["By_Sector"]["Sector"]) == [
        "Tech",
        "Auto",
    ]
    assert list(sheets["Darvas_Signals"]["Symbol"]) == [
        "BBB",
        "AAA",
    ]  # by upside desc, blanks dropped
    assert list(sheets["ML_Signals"]["Symbol"]) == ["AAA", "BBB"]
    assert list(sheets["Fundamental_Scan"]["Symbol"]) == ["AAA", "BBB"]
    assert "last 90 days" in " ".join(sheets["DISCLAIMER"]["DISCLAIMER"].dropna())
    assert len(sheets["Screener_Gates"]) == 6


def test_save_ipo_excel_with_no_data(IT):
    sheets = pd.read_excel(IT.save_ipo_excel([], [], 30), sheet_name=None)
    assert list(sheets) == ["DISCLAIMER", "Screener_Gates"]


def test_save_ipo_excel_missing_signal_columns_gives_placeholder_sheets(IT):
    sheets = pd.read_excel(IT.save_ipo_excel(meta_rows(), [{"Symbol": "AAA"}], 30), sheet_name=None)
    assert (
        sheets["Darvas_Signals"].iloc[0, 0] == "No data"
        and sheets["ML_Signals"].iloc[0, 0] == "No data"
    )


def test_append_ipo_sheet_replaces_existing(IT, tmp_path):
    from openpyxl import Workbook, load_workbook

    p = tmp_path / "scan.xlsx"
    wb = Workbook()
    wb.active.title = "Main"
    wb.create_sheet("IPO_New_Listings").append(["stale"])
    wb.save(p)
    rows = meta_rows()
    rows[0]["MCap_Cr"] = float("nan")
    IT.append_ipo_to_scan_excel(str(p), rows)
    ws = load_workbook(p)["IPO_New_Listings"]
    header = [c.value for c in ws[1]]
    assert header[0] == "Symbol" and ws.max_row == 4
    assert [ws.cell(row=r, column=1).value for r in (2, 3, 4)] == [
        "BBB",
        "AAA",
        "CCC",
    ]  # by Days_Listed desc
    col = header.index("MCap_Cr") + 1
    assert ws.cell(row=3, column=col).value is None  # NaN written as blank
    IT.append_ipo_to_scan_excel(str(p), [])  # nothing to append
    assert len(load_workbook(p).sheetnames) == 2


def test_append_ipo_sheet_bad_path_is_reported(IT, tmp_path, capsys):
    IT.append_ipo_to_scan_excel(str(tmp_path / "missing.xlsx"), meta_rows())
    assert "Could not append IPO sheet" in capsys.readouterr().out


# ── main ──────────────────────────────────────────────────────────────────────
def wire_main(IT, monkeypatch, symbols):
    monkeypatch.setattr(IT, "discover_new_listings", lambda lookback_days: symbols)
    monkeypatch.setattr(
        IT, "download_ipo_ohlc", lambda syms: {s: ohlc(40 + 30 * i) for i, s in enumerate(syms)}
    )
    monkeypatch.setattr(
        IT,
        "enrich_ipo",
        lambda s, df: {
            "Symbol": s,
            "Company_Name": s * 2,
            "Days_Listed": len(df),
            "Current_LTP_₹": 100.0,
            "Trading_Bars": len(df),
            "Listing_Gain_%": float(len(df)) - 50,
            "PE_Zone": "⚪ N/A",
        },
    )
    monkeypatch.setattr(IT, "screen_ipo", lambda s, df, m: {"Symbol": s})


def test_main_pipeline_summary_and_outputs(IT, monkeypatch, capsys):
    wire_main(IT, monkeypatch, ["AAA", "BBB", "CCC"])
    res = IT.main(lookback_days=60, workers=2, append_to_scan=False)
    out = capsys.readouterr().out
    assert [m["Symbol"] for m in sorted(res["meta"], key=lambda m: m["Symbol"])] == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert {s["Company_Name"] for s in res["signals"]} == {"AAAAAA", "BBBBBB", "CCCCCC"}
    assert pd.read_excel(res["excel"], sheet_name="IPO_Overview").shape[0] == 3
    assert "New listings (last 60d): 3" in out and "With OHLC data:       3" in out
    assert "Darvas-eligible:      3" in out and "Full-screener-ready:  0" in out
    assert (
        "Avg gain:" in out
        and "Best:      +50.0%  (CCC)" in out
        and "Worst:     -10.0%  (AAA)" in out
    )
    assert "% positive:67%" in out and "Top 10 by listing gain" in out


def test_main_no_listings_stops_early(IT, monkeypatch, capsys):
    monkeypatch.setattr(IT, "discover_new_listings", lambda lookback_days: [])
    assert IT.main() is None
    assert "No new listings found" in capsys.readouterr().out


def scan_dir(tmp_path):
    (tmp_path / "indian_full_scan").mkdir()
    for n in ("1", "2"):
        (tmp_path / "indian_full_scan" / f"indian_full_scan_{n}.xlsx").write_text("x")


def test_main_appends_to_the_latest_scan_workbook(IT, monkeypatch, tmp_path, capsys):
    wire_main(IT, monkeypatch, ["AAA", "BBB"])
    scan_dir(tmp_path)
    appended = []
    monkeypatch.setattr(
        IT, "append_ipo_to_scan_excel", lambda p, rows: appended.append((p, len(rows)))
    )
    IT.main(lookback_days=30, workers=1, append_to_scan=True)
    assert appended == [("indian_full_scan/indian_full_scan_2.xlsx", 2)]
    assert (
        "Step 5 — Appending IPO sheet to indian_full_scan/indian_full_scan_2.xlsx"
        in capsys.readouterr().out
    )
    appended.clear()
    IT.main(lookback_days=30, workers=1, append_to_scan=False)
    assert appended == []


def test_main_keeps_failed_enrichments_as_note_rows(IT, monkeypatch):
    wire_main(IT, monkeypatch, ["AAA", "BAD"] + [f"G{i:02d}" for i in range(12)])  # >10 good rows

    def enrich(s, df):
        if s == "BAD":
            raise RuntimeError("enrich failed")
        return {
            "Symbol": s,
            "Company_Name": s,
            "Days_Listed": 5,
            "Current_LTP_₹": 1.0,
            "Trading_Bars": 40,
            "Listing_Gain_%": 1.0,
            "PE_Zone": "x",
        }

    monkeypatch.setattr(IT, "enrich_ipo", enrich)
    res = IT.main(lookback_days=30, workers=1, append_to_scan=False)
    assert {"Symbol": "BAD", "Note": "enrich failed"} in res["meta"] and len(res["meta"]) == 14


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="a failed enrichment row has no Company_Name, so the DataFrame holds NaN and "
    "`r.get('Company_Name', ...)[:25]` slices a float when that row makes the "
    "top-10 table (fewer than ten valid rows)",
)
def test_main_summary_survives_a_failed_enrichment_in_a_small_batch(IT, monkeypatch):
    wire_main(IT, monkeypatch, ["AAA", "BAD"])

    def enrich(s, df):
        if s == "BAD":
            raise RuntimeError("enrich failed")
        return {
            "Symbol": s,
            "Company_Name": s,
            "Days_Listed": 5,
            "Current_LTP_₹": 1.0,
            "Trading_Bars": 40,
            "Listing_Gain_%": 1.0,
            "PE_Zone": "x",
        }

    monkeypatch.setattr(IT, "enrich_ipo", enrich)
    IT.main(lookback_days=30, workers=1, append_to_scan=False)


# ── mutation follow-ups: progress output, pacing, index-free sheets ───────────
def test_download_progress_lines_and_pacing(IT, monkeypatch, tmp_path, capsys):
    install_cache(monkeypatch, tmp_path)
    waits = []
    monkeypatch.setattr(IT.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(
        IT.yf,
        "download",
        lambda b, **k: ohlc(30) if len(b) == 1 else multi_download(b),
    )
    IT.download_ipo_ohlc([f"S{i:03d}" for i in range(101)], verbose=True)
    out = capsys.readouterr().out
    assert "Batch 1/3 (50 tickers)" in out and "Batch 2/3 (50 tickers)" in out
    assert "Batch 3/3 (1 tickers)" in out
    assert out.count("OK (50/50)") == 2 and "OK (1/1)" in out
    assert waits == [IT.SLEEP_BETWEEN, IT.SLEEP_BETWEEN] and IT.SLEEP_BETWEEN == 2.0


def test_excel_sheets_have_no_index_column(IT):
    path = IT.save_ipo_excel(meta_rows(), [{"Symbol": "AAA"}], 60)
    for name, df in pd.read_excel(path, sheet_name=None).items():
        assert not [c for c in df.columns if str(c).startswith("Unnamed")], name
    empty = IT.save_ipo_excel([], [], 60)
    for name, df in pd.read_excel(empty, sheet_name=None).items():
        assert not [c for c in df.columns if str(c).startswith("Unnamed")], name


def test_main_progress_every_25(IT, monkeypatch, capsys):
    wire_main(IT, monkeypatch, [f"S{i:02d}" for i in range(26)])
    IT.main(lookback_days=60, workers=2, append_to_scan=False)
    out = capsys.readouterr().out
    assert "25/26 processed" in out and "26/26 processed" in out
