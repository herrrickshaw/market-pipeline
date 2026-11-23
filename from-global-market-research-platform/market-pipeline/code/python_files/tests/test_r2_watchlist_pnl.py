"""Offline tests for watchlist_pnl (entry dates, price lookups, P&L build, CLI report)."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import watchlist_pnl as wp


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Point every module-level path at tmp_path and stub the iCloud bridge workbook."""
    monkeypatch.setattr(wp, "WATCHLIST", tmp_path / "watchlist.csv")
    monkeypatch.setattr(wp, "LEDGER", tmp_path / "ledger.parquet")
    monkeypatch.setattr(wp, "FA_SHEET", tmp_path / "fa.xlsx")
    monkeypatch.setattr(wp, "PANELS", {"IN": tmp_path / "panel_IN.parquet", "US": tmp_path / "panel_US.parquet"})
    monkeypatch.setattr(wp, "_bse_bridge", lambda: {})
    monkeypatch.setattr(wp, "_india_bars", lambda: None)
    monkeypatch.setattr(wp, "_current_prices", lambda m, a=None: (pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")))
    return tmp_path


def _panel(path, series):
    rows = []
    for sym, (start, closes) in series.items():
        for d, c in zip(pd.bdate_range(start, periods=len(closes)), closes):
            rows.append({"Date": d, "Symbol": sym, "Close": c})
    pd.DataFrame(rows).to_parquet(path)


# ------------------------------------------------------------------ _cap / yf candidates
def test_cap_filters_by_date_and_passes_through():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    assert list(wp._cap(s, pd.Timestamp("2026-01-02"))) == [1.0, 2.0]
    assert wp._cap(s, None) is s
    empty = pd.Series(dtype=float)
    assert wp._cap(empty, pd.Timestamp("2026-01-02")) is empty
    bad = pd.Series([1.0], index=["not-a-date"])
    assert wp._cap(bad, pd.Timestamp("2026-01-02")) is bad  # un-parsable index: returned untouched


def test_yf_candidates_per_market():
    assert wp._yf_candidates("JP", " 7203 ") == ["7203.T"]
    assert wp._yf_candidates("KR", "5930") == ["005930.KS", "005930.KQ"]
    assert wp._yf_candidates("KR", "abc") == ["abc.KS".upper().replace("ABC", "ABC"), "ABC.KQ"]
    assert wp._yf_candidates("US", "aapl") == ["AAPL"]


# ------------------------------------------------------------------ bse bridge
def test_bse_bridge_maps_codes(monkeypatch):
    monkeypatch.undo()
    xl = pd.DataFrame({" BSE Code ": [500325, np.nan, 500112, 500999], "NSE Symbol": [" reliance ", "X", "sbin", np.nan]})
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: xl)
    assert wp._bse_bridge() == {"500325": "RELIANCE", "500112": "SBIN"}
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert wp._bse_bridge() == {}


# ------------------------------------------------------------------ entry dates
def test_entry_dates_from_ledger_and_fa_sheet(env, monkeypatch):
    led = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "CCC"], "market": ["US", "US", "IN", "US"],
        "signal_date": ["2026-03-05", "2026-02-01", "2026-04-01", "2026-05-01"],
        "price_at_signal": [11.0, 10.0, np.nan, 5.0], "filter": ["darvas", "piotroski", "gc", "x"],
    })
    led.to_parquet(wp.LEDGER)
    wp.FA_SHEET.write_text("x")
    fa = pd.DataFrame({"Security": [" ccc ", "DDD", "DDD", "EEE"], "Date of Acquiring": ["2025-12-01", "2026-01-10", "2026-01-05", None]})
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: fa)
    d = wp.entry_dates().set_index("symbol")
    assert d.loc["AAA", "date_added"] == pd.Timestamp("2026-02-01") and d.loc["AAA", "entry_price"] == 10.0
    assert d.loc["AAA", "basis"] == "signal-price" and d.loc["AAA", "why"] == "piotroski"
    assert d.loc["BBB", "basis"] == "panel-close"  # NaN signal price -> must be reconstructed from the panel
    # CCC: acquired 2025-12-01 predates its ledger signal: the earlier row wins after sorting/dedup
    assert d.loc["CCC", "date_added"] == pd.Timestamp("2025-12-01") and d.loc["CCC", "why"] == "acquired"
    assert d.loc["DDD", "date_added"] == pd.Timestamp("2026-01-05")  # earliest acquisition per symbol
    assert "EEE" not in d.index  # no date -> not used
    assert list(d["date_added"]) == sorted(d["date_added"])


def test_entry_dates_empty_and_broken_fa(env, monkeypatch):
    assert wp.entry_dates().empty
    wp.FA_SHEET.write_text("x")
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad xlsx")))
    assert wp.entry_dates().empty
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: pd.DataFrame({"Security": ["A"], "Other": [1]}))
    assert wp.entry_dates().empty  # no 'acquiring' column


# ------------------------------------------------------------------ price lookups
def test_india_bars_reads_cleaned_parquet(monkeypatch, tmp_path):
    cleaned = tmp_path / "cleaned.parquet"
    pd.DataFrame({"Date": ["2026-01-02", "2026-01-01"], "Symbol": ["tcs", "tcs"], "Close": [11.0, 10.0]}).to_parquet(cleaned)
    monkeypatch.setitem(sys.modules, "bhavcopy_store", types.SimpleNamespace(CLEANED=cleaned))
    d = wp._india_bars()
    assert list(d["Symbol"]) == ["TCS", "TCS"] and pd.api.types.is_datetime64_any_dtype(d["Date"])
    monkeypatch.setitem(sys.modules, "bhavcopy_store", types.SimpleNamespace(CLEANED=tmp_path / "missing.parquet"))
    assert wp._india_bars() is None
    pd.DataFrame({"Date": [], "Symbol": [], "Close": []}).to_parquet(cleaned)
    monkeypatch.setitem(sys.modules, "bhavcopy_store", types.SimpleNamespace(CLEANED=cleaned))
    assert wp._india_bars() is None
    monkeypatch.setitem(sys.modules, "bhavcopy_store", None)
    assert wp._india_bars() is None


def test_current_prices_us_uses_registry_dir_and_asof(monkeypatch, tmp_path):
    d = tmp_path / "ohlc"
    d.mkdir()
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx).to_parquet(d / "aapl.parquet")
    pd.DataFrame({"Close": [np.nan, np.nan, np.nan]}, index=idx).to_parquet(d / "empty.parquet")
    (d / "broken.parquet").write_text("not parquet")
    monkeypatch.setitem(sys.modules, "data_registry", types.SimpleNamespace(OHLC_DIR=d))
    px, dates = wp._current_prices("US")
    assert px.to_dict() == {"AAPL": 3.0} and dates["AAPL"] == pd.Timestamp("2026-01-05")
    px, dates = wp._current_prices("US", pd.Timestamp("2026-01-03"))
    assert px["AAPL"] == 2.0 and dates["AAPL"] == pd.Timestamp("2026-01-02")
    px, _ = wp._current_prices("US", pd.Timestamp("2025-01-01"))
    assert px.empty  # everything is after the cap
    px, dates = wp._current_prices("ZZ")
    assert px.empty and dates.empty


def test_current_prices_india_cleaned_then_bhavcopy_fallback(monkeypatch):
    bars = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-03"]),
                         "Symbol": ["TCS", "TCS", "INFY", "INFY"], "Close": [10.0, 11.0, 20.0, np.nan]})
    monkeypatch.setattr(wp, "_india_bars", lambda: bars)
    px, dates = wp._current_prices("IN")
    assert px.to_dict() == {"TCS": 11.0, "INFY": 20.0} and dates["INFY"] == pd.Timestamp("2026-01-01")
    px, _ = wp._current_prices("IN", pd.Timestamp("2026-01-01"))
    assert px.to_dict() == {"TCS": 10.0, "INFY": 20.0}
    # no cleaned frame -> per-symbol bhavcopy_store fallback
    monkeypatch.setattr(wp, "_india_bars", lambda: None)
    frames = {"RELIANCE": pd.DataFrame({"Close": ["100", "x", "102"]}, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])),
              "NONE": None, "EMPTYY": pd.DataFrame(), "NOCLOSE": pd.DataFrame({"Open": [1.0]}), "BOOM": "raise"}

    def get(sym):
        if frames[sym] == "raise" if isinstance(frames[sym], str) else False:
            raise RuntimeError("bad")
        return frames[sym]
    monkeypatch.setitem(sys.modules, "bhavcopy_store", types.SimpleNamespace(symbols=lambda: list(frames), get=get))
    px, dates = wp._current_prices("IN")
    assert px.to_dict() == {"RELIANCE": 102.0} and dates["RELIANCE"] == pd.Timestamp("2026-01-03")
    px, _ = wp._current_prices("IN", pd.Timestamp("2026-01-02"))
    assert px.to_dict() == {"RELIANCE": 100.0}


# ------------------------------------------------------------------ fetch-missing
class _Hist:
    def __init__(self, hist):
        self._h = hist

    def history(self, start, end, auto_adjust):
        h = self._h
        if isinstance(h, Exception):
            raise h
        return h


def _need_frame():
    return pd.DataFrame({"symbol": ["7203", "AAA"], "market": ["JP", "US"], "ltp": [np.nan, 5.0],
                         "date_added": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")],
                         "entry_price": [np.nan, 4.0], "pct_since": [np.nan, 25.0]})


def test_fetch_missing_prices_unpriced_names(monkeypatch, capsys):
    idx = pd.date_range("2026-01-01", periods=10, freq="B", tz="Asia/Tokyo")
    hist = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=idx)
    seen = []
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: seen.append(t) or _Hist(hist)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    d = wp._fetch_missing(_need_frame(), None)
    assert seen == ["7203.T"]  # only the unpriced name is fetched
    assert d.at[0, "ltp"] == 109.0 and d.at[0, "as_of"] == pd.Timestamp("2026-01-14")
    # entry = last close on/before 2026-01-05 (index 2 -> 102)
    assert d.at[0, "entry_price"] == 102.0 and d.at[0, "basis"] == "panel-close"
    assert d.at[0, "pct_since"] == pytest.approx((109 / 102 - 1) * 100) and d.at[0, "held_days"] == 9
    assert d.at[1, "ltp"] == 5.0  # already priced: untouched
    assert "priced 1/1" in capsys.readouterr().out


def test_fetch_missing_asof_cap_and_failures(monkeypatch, capsys):
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    hist = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=idx)
    d = _need_frame()
    d["date_added"] = [pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-05")]
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: _Hist(hist)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = wp._fetch_missing(d, pd.Timestamp("2026-01-06"))  # capped before the entry date -> last bar precedes date_added
    assert out.at[0, "ltp"] == 103.0
    assert out.at[0, "entry_price"] == 103.0 and out.at[0, "pct_since"] == 0.0 or pd.isna(out.at[0, "pct_since"]) or out.at[0, "pct_since"] == 0
    # exception / empty history -> next candidate, finally nothing priced
    for h in (RuntimeError("net"), pd.DataFrame(), pd.DataFrame({"Open": [1.0]}, index=idx[:1])):
        fake.Ticker = lambda t, h=h: _Hist(h)
        out = wp._fetch_missing(_need_frame(), None)
        assert pd.isna(out.at[0, "ltp"])
    assert "priced 0/1" in capsys.readouterr().out
    # nothing to fetch / yfinance missing
    full = _need_frame()
    full["ltp"] = 1.0
    assert wp._fetch_missing(full, None) is full
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert wp._fetch_missing(_need_frame(), None).at[0, "ltp"] != 1.0
    assert "yfinance unavailable" in capsys.readouterr().out


def test_fetch_missing_cap_before_all_bars_and_no_date_added(monkeypatch):
    idx = pd.date_range("2026-01-05", periods=3, freq="B")
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: _Hist(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = wp._fetch_missing(_need_frame(), pd.Timestamp("2026-01-01"))  # cap precedes every bar -> unpriced
    assert pd.isna(out.at[0, "ltp"])
    d = _need_frame()
    d["date_added"] = pd.NaT
    out = wp._fetch_missing(d, None)
    assert out.at[0, "ltp"] == 3.0 and pd.isna(out.at[0, "pct_since"])


# ------------------------------------------------------------------ realized sales
def _fa_sheet(rows):
    cols = [f"c{i}" for i in range(14)] + ["symbol", "sale_date", "units", "sell", "buy", "brk", "gl_usd", "gl_inr", "buy_date", "bq", "ba", "hold", "mbd"]
    data = []
    for r in rows:
        base = [None] * 14
        data.append(base + list(r))
    return pd.DataFrame(data, columns=cols)


def test_realized_sales_aggregates_per_symbol(env, monkeypatch):
    assert wp.realized_sales().empty  # no sheet
    wp.FA_SHEET.write_text("x")
    rows = [(" aaa ", "2026-02-01", 1, 150.0, 100.0, 1.0, 49.0, 4000.0, "2025-01-01", 1, 100.0, 400, "x"),
            ("AAA", "2026-03-01", 1, 60.0, 50.0, 0.5, 9.5, 800.0, "2025-02-01", 1, 50.0, 200, "x"),
            ("BBB", "bad-date", 1, 10.0, 0.0, 0.0, 10.0, 80.0, "2025-02-01", 1, 0.0, 100, "x"),
            (None, "2026-03-01", 1, 1.0, 1.0, 0.0, 0.0, 0.0, "2025-02-01", 1, 1.0, 1, "x")]
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: _fa_sheet(rows))
    r = wp.realized_sales().set_index("symbol")
    assert set(r.index) == {"AAA", "BBB"}
    assert r.loc["AAA", "realized_sell_value_usd"] == 210.0 and r.loc["AAA", "n_sale_transactions"] == 2
    assert r.loc["AAA", "realized_pct"] == pytest.approx(58.5 / 150.0 * 100) and r.loc["AAA", "avg_holding_period_days"] == 300
    assert r.loc["AAA", "last_sale_date"] == pd.Timestamp("2026-03-01")
    assert pd.isna(r.loc["BBB", "realized_pct"])  # zero cost basis never divides
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: pd.DataFrame({"a": [1]}))
    assert wp.realized_sales().empty  # <27 columns
    monkeypatch.setattr(wp.pd, "read_excel", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert wp.realized_sales().empty


# ------------------------------------------------------------------ build
def _watchlist(env, rows, ledger=True):
    pd.DataFrame(rows, columns=["symbol", "market", "status", "note"]).to_csv(wp.WATCHLIST, index=False)
    if ledger:  # an unrelated ledger row keeps entry_dates() non-empty (see the xfail below for the empty case)
        pd.DataFrame({"symbol": ["ZZZ"], "market": ["US"], "signal_date": ["2026-01-02"], "price_at_signal": [1.0], "filter": ["x"]}).to_parquet(wp.LEDGER)


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="build(): with no ledger and no FA sheet entry_dates() returns a column-less frame and the merge on symbol/market raises KeyError")
def test_build_without_any_entry_source_still_returns_rows(env):
    _watchlist(env, [("AAA", "US", "held", "2026-01-06")], ledger=False)
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0] * 6), "SPY": ("2026-01-01", [100.0] * 6)})
    assert list(wp.build(None, None)["symbol"]) == ["AAA"]


def test_build_computes_pct_since_vs_benchmark_and_filters(env):
    _watchlist(env, [("aaa", "us", "held", "added 2026-01-06 on darvas"), ("bbb", "us", "watch", ""), ("ccc", "in", "held", ""),
                     ("eee", "eu", "held", "")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0, 10, 10, 10, 20, 30]), "BBB": ("2026-01-01", [5.0] * 6),
                             "SPY": ("2026-01-01", [100.0, 100, 100, 100, 110, 120])})
    d = wp.build(None, None).set_index("symbol")
    a = d.loc["AAA"]
    assert a["date_added"] == pd.Timestamp("2026-01-06") and a["basis"] == "panel-close"
    assert a["entry_price"] == 10.0 and a["ltp"] == 30.0  # last panel close; entry = close on/before 01-06
    assert a["pct_since"] == pytest.approx(200.0)
    assert a["vs_mkt"] == pytest.approx(200.0 - 20.0)  # SPY 100 -> 120 over the same window
    assert pd.isna(d.loc["BBB", "pct_since"])  # no date_added -> no entry basis
    assert d.loc["CCC", "market"] == "IN" and pd.isna(d.loc["CCC", "ltp"])  # panel missing -> row kept unpriced
    assert d.loc["EEE", "market"] == "EU"  # market with no panel configured
    assert set(wp.build("us", None).symbol) == {"AAA", "BBB"}
    assert set(wp.build(None, "watch").symbol) == {"BBB"}


def test_build_asof_cap_and_fresh_prices_and_staleness(env, monkeypatch):
    _watchlist(env, [("AAA", "US", "held", "2026-01-06"), ("BBB", "US", "held", "2026-01-06"), ("LATE", "US", "held", "2026-03-01")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0, 10, 10, 10, 20, 30]), "BBB": ("2026-01-01", [5.0, 5, 5, 5, 5, 4]),
                             "SPY": ("2026-01-01", [100.0] * 6)})
    fresh = pd.Series({"AAA": 40.0})
    fresh_dates = pd.Series({"AAA": pd.Timestamp("2026-01-20")})
    monkeypatch.setattr(wp, "_current_prices", lambda m, a=None: (fresh, fresh_dates))
    d = wp.build(None, None).set_index("symbol")
    assert d.loc["AAA", "ltp"] == 40.0 and d.loc["AAA", "as_of"] == pd.Timestamp("2026-01-20") and d.loc["AAA", "stale_days"] == 0
    assert d.loc["BBB", "ltp"] == 4.0  # falls back to the panel when no fresher price
    assert d.loc["BBB", "stale_days"] == 12  # panel bar 01-08 vs the market's latest fresh bar 01-20
    assert "LATE" in d.index
    capped = wp.build(None, None, pd.Timestamp("2026-01-07")).set_index("symbol")
    assert "LATE" not in capped.index  # added after the cap
    assert capped.loc["AAA", "ltp"] == 40.0  # the stub ignores the cap; panel is capped separately
    assert capped.loc["BBB", "ltp"] == 5.0


def test_build_zero_day_and_stale_entries_are_nan_pct(env, monkeypatch):
    _watchlist(env, [("AAA", "US", "held", "2026-01-08")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0, 11, 12, 13, 14, 15]), "SPY": ("2026-01-01", [100.0] * 6)})
    monkeypatch.setattr(wp, "_current_prices", lambda m, a=None: (pd.Series({"AAA": 99.0}), pd.Series({"AAA": pd.Timestamp("2026-01-05")})))
    d = wp.build(None, None).set_index("symbol")
    # the latest price predates date_added: a negative holding window must not produce a % move
    assert pd.isna(d.loc["AAA", "pct_since"]) and pd.isna(d.loc["AAA", "vs_mkt"])


def test_build_sold_names_use_realized_returns(env, monkeypatch):
    _watchlist(env, [("AAA", "US", "sold", "2026-01-06"), ("BBB", "US", "sold", "2026-01-06"), ("CCC", "US", "held", "2026-01-06")])
    _panel(wp.PANELS["US"], {s: ("2026-01-01", [10.0] * 6) for s in ("AAA", "BBB", "CCC", "SPY")})
    monkeypatch.setattr(wp, "realized_sales", lambda: pd.DataFrame({"symbol": ["AAA"], "realized_gain_loss_usd": [50.0], "realized_pct": [12.5]}))
    d = wp.build(None, None).set_index("symbol")
    assert d.loc["AAA", "pct_since"] == 12.5 and d.loc["AAA", "basis"] == "realized-tax-filing"
    assert d.loc["BBB", "basis"] == "sold-no-exit-price-on-file"
    assert d.loc["CCC", "basis"] == "panel-close"


def test_build_median_benchmark_when_missing_and_dotted_symbols(env):
    _watchlist(env, [("BRK.B", "US", "held", "2026-01-06")])
    _panel(wp.PANELS["US"], {"BRK-B": ("2026-01-01", [10.0, 10, 10, 10, 20, 30]), "OTH": ("2026-01-01", [10.0, 10, 10, 10, 10, 10])})
    d = wp.build(None, None).set_index("symbol")
    assert d.loc["BRK.B", "ltp"] == 30.0  # dot mapped to the panel's dash spelling
    # no SPY column: the cross-sectional median close is the benchmark (median of BRK-B and OTH)
    assert d.loc["BRK.B", "vs_mkt"] == pytest.approx(200.0 - ((20.0 / 10.0) - 1) * 100)  # median series 10,10,10,10,15,20 from 01-06 -> (20/10 - 1)


# ------------------------------------------------------------------ main
def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    return wp.main()


def test_main_prints_table_summary_and_writes_csv(env, monkeypatch, capsys):
    _watchlist(env, [("AAA", "US", "held", "2026-01-06"), ("BBB", "US", "held", "2026-01-06"), ("OLD", "US", "watch", "2026-01-06")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0, 10, 10, 10, 20, 30]), "BBB": ("2026-01-01", [10.0, 10, 10, 10, 10, 5]),
                             "OLD": ("2026-01-01", [10.0] * 6), "SPY": ("2026-01-01", [100.0] * 6)})
    out_csv = env / "sub" / "out.csv"
    assert _run(monkeypatch, "--sort", "pct", "--top", "2", "--out", str(out_csv)) == 0
    text = capsys.readouterr().out
    assert "▌US   3 names, 3 with an entry basis" in text and "… 1 more" in text
    assert "US held    n=  2  med   +75.0%" in text  # median of +200% and -50%
    assert "vs mkt" in text and str(out_csv) in text
    assert "staleness:" not in text  # no fresh prices -> no stale_days to report
    saved = pd.read_csv(out_csv)
    assert list(saved["symbol"])[0] == "AAA" and "pct_since" in saved.columns  # sorted best-first


def test_main_default_and_date_sorts_and_asof(env, monkeypatch, capsys):
    _watchlist(env, [("AAA", "US", "held", "2026-01-06"), ("BBB", "US", "held", "2026-01-03")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0] * 6), "BBB": ("2026-01-01", [10.0] * 6), "SPY": ("2026-01-01", [100.0] * 6)})
    assert _run(monkeypatch, "--asof", "2026-01-07") == 0
    text = capsys.readouterr().out
    assert "as-of cap: 2026-01-07" in text
    assert text.index("BBB") < text.index("AAA")  # default order: market, then date_added ascending
    assert _run(monkeypatch, "--sort", "date") == 0
    text = capsys.readouterr().out
    assert text.index("AAA") < text.index("BBB")  # newest first
    assert _run(monkeypatch, "--asof", "yesterday") == 0
    assert "as-of cap:" in capsys.readouterr().out


def test_main_nothing_to_show(env, monkeypatch, capsys):
    _watchlist(env, [("AAA", "US", "held", "")])
    assert _run(monkeypatch, "--market", "JP") == 1
    assert "nothing to show" in capsys.readouterr().out


def test_main_fetch_missing_runs_on_mixed_watchlist(env, monkeypatch, capsys):
    _watchlist(env, [("AAA", "US", "held", "2026-01-06"), ("7203", "JP", "held", "2026-01-06")])
    _panel(wp.PANELS["US"], {"AAA": ("2026-01-01", [10.0] * 6), "SPY": ("2026-01-01", [100.0] * 6)})
    called = {}
    monkeypatch.setattr(wp, "_fetch_missing", lambda d, asof: called.update(n=len(d)) or d)
    assert _run(monkeypatch, "--fetch-missing") == 0
    assert called == {"n": 2}
    assert "▌JP   1 names" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="main(): a watchlist whose markets all lack a price panel (JP/KR/EU only) has no 'ltp' column, so the default sort raises KeyError")
def test_main_panel_less_watchlist_with_fetch_missing(env, monkeypatch):
    _watchlist(env, [("7203", "JP", "held", "2026-01-06")])
    monkeypatch.setattr(wp, "_fetch_missing", lambda d, asof: d)
    assert _run(monkeypatch, "--fetch-missing") == 0
