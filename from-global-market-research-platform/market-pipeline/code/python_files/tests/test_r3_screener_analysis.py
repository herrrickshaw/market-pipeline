"""Offline tests for screener_analysis (screener.in pre-filter, six screeners, Excel, orchestration)."""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
import requests


class _NoInfo:
    """Ticker whose .info access raises."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def info(self):
        raise RuntimeError("no info")


class _Ticker:
    def __init__(self, **kw):
        self.info = {}
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture
def sa(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # DOWNLOAD_DIR is created relative to cwd
    yf = types.ModuleType("yfinance")
    yf.download = lambda *a, **k: pd.DataFrame()
    yf.Ticker = lambda t: _Ticker()
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    monkeypatch.setitem(sys.modules, "nse_data_fetcher", None)  # ImportError branch
    sys.modules.pop("screener_analysis", None)
    mod = importlib.import_module("screener_analysis")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "screener_results")
    (tmp_path / "screener_results").mkdir(exist_ok=True)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    sys.modules.pop("screener_analysis", None)


def _frame(rows, cols=("2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31")):
    n = len(next(iter(rows.values())))
    return pd.DataFrame(rows).T.set_axis(pd.to_datetime(list(cols)[:n]), axis=1)


# ------------------------------------------------------------------ constants
def test_screener_definitions_are_complete(sa):
    assert set(sa.SCREENERS) == {"mf", "gc", "bc"}
    for meta in sa.SCREENERS.values():
        assert {"id", "slug", "name", "author", "logic", "criteria", "risk_note"} <= set(meta)
    assert sa.SCREENERS["mf"]["thresholds"] == {"roic_pct": 25, "earnings_yield_pct": 15, "min_mcap_cr": 15}
    assert sa.DARVAS_CONFIRM == 3 and sa.BATCH_SIZE == 200


# ------------------------------------------------------------------ screener.in fetch
class _MD:
    pages = {}
    default = ""

    def convert(self, stream, file_extension=None):
        body = stream.read().decode()
        return types.SimpleNamespace(text_content=body)


def _install_md(monkeypatch):
    m = types.ModuleType("markitdown")
    m.MarkItDown = _MD
    monkeypatch.setitem(sys.modules, "markitdown", m)


class _Resp:
    def __init__(self, text="", status=200):
        self.content = text.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_fetch_screener_in_without_markitdown(sa, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "markitdown", None)
    assert sa.fetch_screener_in_symbols(1, "x") == []
    assert "markitdown not installed" in capsys.readouterr().out


def test_fetch_screener_in_parses_hrefs_and_table_rows(sa, monkeypatch, capsys):
    _install_md(monkeypatch)
    pages = {
        1: "see [A](/company/RELIANCE/) and /company/TCS/ more\n| **INFY** | 10 | 20 |\n| Name | x |\n| CMP | y |\n| s | 1 |",
        2: "/company/WIPRO/ \n| RELIANCE | 1 |",
        3: "",
    }
    urls = []

    def get(url, timeout=None, headers=None):
        urls.append(url)
        page = int(url.rsplit("page=", 1)[1])
        assert timeout == 20 and "Mozilla" in headers["User-Agent"]
        return _Resp(pages[page])

    monkeypatch.setattr(sa.requests, "get", get)
    out = sa.fetch_screener_in_symbols(59, "magic-formula")
    # deduplicated, ordered; header words and 1-char cells excluded; stops when a page adds nothing new
    assert out == ["RELIANCE", "TCS", "INFY", "WIPRO"]
    assert urls[0] == "https://www.screener.in/screens/59/magic-formula/?page=1"
    assert len(urls) == 3
    assert "4 symbols" in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the 'no new symbols -> last page' stop compares the raw running list, which grows "
    "even when a page repeats already-seen symbols, so a repeated page never stops the loop "
    "(screener_analysis.py fetch_screener_in_symbols, prev_len check)",
)
def test_fetch_screener_in_stops_when_a_page_repeats(sa, monkeypatch):
    _install_md(monkeypatch)
    urls = []

    def get(url, timeout=None, headers=None):
        urls.append(url)
        return _Resp("/company/AAA/ /company/BBB/")  # every page returns the same rows

    monkeypatch.setattr(sa.requests, "get", get)
    assert sa.fetch_screener_in_symbols(1, "x", max_pages=10) == ["AAA", "BBB"]
    assert len(urls) <= 2


def test_fetch_screener_in_max_pages_and_errors(sa, monkeypatch, capsys):
    _install_md(monkeypatch)
    n = {"i": 0}

    def get(url, timeout=None, headers=None):
        n["i"] += 1
        return _Resp(f"/company/SYM{chr(64 + n['i'])}X/ ")

    monkeypatch.setattr(sa.requests, "get", get)
    out = sa.fetch_screener_in_symbols(1, "x", max_pages=2)
    assert out == ["SYMAX", "SYMBX"] and n["i"] == 2

    # 404 stops quietly; other HTTP errors report; generic errors report
    monkeypatch.setattr(sa.requests, "get", lambda *a, **k: _Resp("", 404))
    assert sa.fetch_screener_in_symbols(1, "x") == []
    assert "HTTP" not in capsys.readouterr().out.split("symbols")[0].replace("Fetching", "")
    monkeypatch.setattr(sa.requests, "get", lambda *a, **k: _Resp("", 500))
    sa.fetch_screener_in_symbols(1, "x")
    assert "HTTP 500 on page 1" in capsys.readouterr().out

    def boom(*a, **k):
        raise RuntimeError("socket")

    monkeypatch.setattr(sa.requests, "get", boom)
    assert sa.fetch_screener_in_symbols(1, "x") == []
    assert "Error on page 1: socket" in capsys.readouterr().out


# ------------------------------------------------------------------ NSE universe
def test_fetch_all_nse_symbols_primary(sa, monkeypatch):
    monkeypatch.setattr(sa, "_USE_NSE_FETCHER", True)
    syms = [f"S{i}" for i in range(101)]
    monkeypatch.setattr(sa, "_get_nse_syms", lambda: syms, raising=False)
    assert sa.fetch_all_nse_symbols() == syms  # > 100 required


def test_fetch_all_nse_symbols_fallbacks(sa, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sa, "_USE_NSE_FETCHER", True)
    monkeypatch.setattr(sa, "_get_nse_syms", lambda: ["A"] * 100, raising=False)
    monkeypatch.setitem(sys.modules, "nse", None)
    assert sa.fetch_all_nse_symbols() == []
    assert "Nifty 50 fallback" in capsys.readouterr().out

    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(sa, "_get_nse_syms", boom, raising=False)
    assert sa.fetch_all_nse_symbols() == []

    csv = tmp_path / "bhav.csv"
    pd.DataFrame({"SctySrs": ["EQ", "BE", "EQ"], "TckrSymb": [" ZED ", "SKIP", "ABC"]}).to_csv(csv, index=False)

    class FakeNSE:
        calls = 0

        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def equityBhavcopy(self, d):
            FakeNSE.calls += 1
            if FakeNSE.calls < 3:
                raise RuntimeError("holiday")
            return csv

    nse = types.ModuleType("nse")
    nse.NSE = FakeNSE
    monkeypatch.setitem(sys.modules, "nse", nse)
    assert sa.fetch_all_nse_symbols() == ["ABC", "ZED"]
    assert FakeNSE.calls == 3

    # bhavcopy without the new-format column yields nothing for any day
    pd.DataFrame({"x": [1]}).to_csv(csv, index=False)
    FakeNSE.calls = 10
    assert sa.fetch_all_nse_symbols() == []


# ------------------------------------------------------------------ bulk OHLC
def _multi(tickers, rows=12, nan_for=()):
    idx = pd.bdate_range("2024-01-01", periods=rows)
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    df = pd.DataFrame(np.ones((rows, len(cols))), index=idx, columns=cols)
    for t in nan_for:
        df[("Close", t)] = np.nan
    return df


def test_bulk_download_ohlc(sa, monkeypatch, capsys):
    monkeypatch.setattr(sa.yf, "download", lambda b, **k: _multi(list(b)[:2], nan_for=["B.NS"]))
    out = sa.bulk_download_ohlc(["A.NS", "B.NS", "C.NS"])
    assert list(out) == ["A.NS"]  # B all-NaN, C missing
    assert "OK (1 usable)" in capsys.readouterr().out
    # 10-row minimum
    monkeypatch.setattr(sa.yf, "download", lambda b, **k: _multi(list(b), rows=9))
    assert sa.bulk_download_ohlc(["A.NS"]) == {}
    monkeypatch.setattr(sa.yf, "download", lambda b, **k: _multi(list(b), rows=10))
    assert "A.NS" in sa.bulk_download_ohlc(["A.NS"])


def test_bulk_download_ohlc_batches_empty_flat_and_error(sa, monkeypatch, capsys):
    sizes = []

    def dl(b, **k):
        sizes.append(len(b))
        return pd.DataFrame()

    monkeypatch.setattr(sa.yf, "download", dl)
    sa.bulk_download_ohlc([f"T{i}" for i in range(450)])
    assert sizes == [200, 200, 50]
    assert "empty" in capsys.readouterr().out
    flat = _multi(["X"]).droplevel(1, axis=1)
    monkeypatch.setattr(sa.yf, "download", lambda b, **k: flat)
    out = sa.bulk_download_ohlc(["A.NS", "B.NS"])
    assert list(out) == ["A.NS"]

    def boom(b, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(sa.yf, "download", boom)
    assert sa.bulk_download_ohlc(["A.NS"]) == {}
    assert "ERROR — down" in capsys.readouterr().out


# ------------------------------------------------------------------ magic formula
def _mf_ticker(ebit=60e7, ta=200e7, cl=50e7, info=None, with_inc=True):
    inc = _frame({"EBIT": [ebit, 1]}) if with_inc else None
    bal = _frame({"Total Assets": [ta, 1], "Current Liabilities": [cl, 1]})
    return _Ticker(
        income_stmt=inc, balance_sheet=bal,
        info=info if info is not None else
        {"marketCap": 200e7, "totalDebt": 50e7, "totalCash": 10e7, "bookValue": 5.0, "shortName": "Acme"},
    )


def test_magic_formula_values(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda t: _mf_ticker())
    r = sa.compute_magic_formula("ACME")
    assert r["qualifies"] is True and r["name"] == "Acme" and r["screen"] == "mf" and r["error"] == ""
    assert r["ROIC_%"] == 40.0  # 60 / (200 - 50)
    assert r["Earnings_Yield_%"] == 25.0  # 60 / (200 + 50 - 10)
    assert r["Book_Value"] == 5.0 and r["MCap_Cr"] == 200.0


def test_magic_formula_thresholds(sa, monkeypatch):
    def run(**kw):
        info = {"marketCap": 200e7, "totalDebt": 50e7, "totalCash": 10e7, "bookValue": 5.0}
        info.update(kw.pop("info", {}))
        monkeypatch.setattr(sa.yf, "Ticker", lambda t: _mf_ticker(info=info, **kw))
        return sa.compute_magic_formula("ACME")

    assert run(ebit=50e7)["qualifies"] is True and run(ebit=50e7)["ROIC_%"] == 33.33
    # ROIC must exceed 25: ebit 37.5 / 150 = 25.0 exactly fails
    assert run(ebit=37.5e7)["qualifies"] is False
    assert run(ebit=38e7)["ROIC_%"] == 25.33
    # EY must exceed 15: mcap larger lowers it
    assert run(info={"marketCap": 500e7})["qualifies"] is False  # ev 540 -> 11%
    assert run(info={"marketCap": 360e7})["qualifies"] is False  # ey exactly 15.0 is not > 15
    assert run(info={"marketCap": 359e7})["qualifies"] is True
    assert run(info={"bookValue": 0})["qualifies"] is False
    assert run(info={"bookValue": None})["qualifies"] is False
    assert run(info={"marketCap": 15e7, "totalDebt": 0, "totalCash": 0})["qualifies"] is False  # 15 Cr is not > 15
    assert run(info={"marketCap": 15.1e7, "totalDebt": 0, "totalCash": 0})["qualifies"] is True
    r = run(info={"marketCap": None})
    assert r["qualifies"] is False and r["Earnings_Yield_%"] is None and r["MCap_Cr"] is None
    # missing current liabilities -> no capital employed -> no ROIC
    monkeypatch.setattr(sa.yf, "Ticker", lambda t: _mf_ticker(cl=0))
    r = sa.compute_magic_formula("ACME")
    assert r["ROIC_%"] is None and r["qualifies"] is False


def test_magic_formula_errors(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda t: _mf_ticker(with_inc=False))
    assert sa.compute_magic_formula("ACME") == {"symbol": "ACME", "screen": "mf", "qualifies": False, "error": "no_data"}

    def boom(t):
        raise RuntimeError("yf")

    monkeypatch.setattr(sa.yf, "Ticker", boom)
    assert sa.compute_magic_formula("ACME")["error"] == "yf"

    t = _NoInfo(income_stmt=_frame({"EBIT": [1, 1]}), balance_sheet=None)
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    r = sa.compute_magic_formula("ACME")
    assert r["qualifies"] is False and r["MCap_Cr"] is None


# ------------------------------------------------------------------ golden crossover
def _gc_df(close):
    idx = pd.bdate_range("2023-01-02", periods=len(close))
    return pd.DataFrame({"Close": close}, index=idx)


def test_golden_crossover_guards(sa):
    assert sa.compute_golden_crossover(None, "X")["error"] == "no_data"
    assert sa.compute_golden_crossover(pd.DataFrame(), "X")["error"] == "no_data"
    r = sa.compute_golden_crossover(_gc_df(np.ones(200)), "X")
    assert r["error"] == "need 201 bars, got 200" and r["qualifies"] is False
    r = sa.compute_golden_crossover(_gc_df(np.ones(201)), "X")
    assert r["error"] == "" and r["qualifies"] is False and r["dma50_above_200"] is False


def test_golden_crossover_detects_cross_on_last_bar(sa):
    # find a series whose 50-DMA crosses the 200-DMA exactly on the final bar
    base = np.concatenate([np.linspace(100, 60, 250), np.full(100, 60.0)])
    for jump_at in range(len(base), 260, -1):
        close = np.concatenate([base[: jump_at - 1], [500.0]])
        s50 = pd.Series(close).rolling(50).mean()
        s200 = pd.Series(close).rolling(200).mean()
        if s50.iloc[-2] < s200.iloc[-2] and s50.iloc[-1] > s200.iloc[-1]:
            break
    else:  # pragma: no cover
        pytest.fail("could not construct a cross")
    r = sa.compute_golden_crossover(_gc_df(close), "X")
    assert r["qualifies"] is True and r["recent_crossover"] is True and r["dma50_above_200"] is True
    assert r["DMA50"] == round(float(s50.iloc[-1]), 2) and r["DMA200"] == round(float(s200.iloc[-1]), 2)
    assert r["LTP"] == 500.0
    assert r["DMA_gap_%"] == round((s50.iloc[-1] - s200.iloc[-1]) / s200.iloc[-1] * 100, 2)


def test_golden_crossover_recent_but_not_today(sa):
    close = np.concatenate([np.linspace(100, 60, 250), np.linspace(60, 300, 60)])
    s50 = pd.Series(close).rolling(50).mean()
    s200 = pd.Series(close).rolling(200).mean()
    # first day above, and the days after it
    cross_idx = next(i for i in range(1, len(close)) if s50[i - 1] < s200[i - 1] and s50[i] > s200[i])
    # truncate so the cross happened 3 bars before the end
    cut = close[: cross_idx + 4]
    r = sa.compute_golden_crossover(_gc_df(cut), "X")
    assert r["qualifies"] is False  # strict: only the day of the cross
    assert r["recent_crossover"] is True and r["dma50_above_200"] is True
    # 10 bars after the cross is outside the 5-bar window
    r = sa.compute_golden_crossover(_gc_df(close[: cross_idx + 10]), "X")
    assert r["recent_crossover"] is False and r["dma50_above_200"] is True


def test_golden_crossover_downtrend(sa):
    close = np.linspace(200, 50, 300)
    r = sa.compute_golden_crossover(_gc_df(close), "X")
    assert r["qualifies"] is False and r["recent_crossover"] is False and r["dma50_above_200"] is False
    assert r["DMA_gap_%"] < 0


def test_golden_crossover_nan_closes_dropped(sa):
    close = list(np.linspace(100, 50, 210))
    close[5] = "bad"
    df = _gc_df(close)
    r = sa.compute_golden_crossover(df, "X")
    assert r["error"] == ""  # 209 numeric closes remain >= 201
    close2 = list(np.linspace(100, 50, 201))
    close2[5] = "bad"
    assert "need 201 bars, got 200" == sa.compute_golden_crossover(_gc_df(close2), "X")["error"]


# ------------------------------------------------------------------ bull cartel
def _q_frame(rev0=130.0, rev4=100.0, ni0=3e7, ni4=2e7, n=6):
    cols = pd.date_range("2025-03-31", periods=n, freq="-3ME")
    rev = [rev0, 1, 1, 1, rev4] + [1.0] * (n - 5)
    ni = [ni0, 1, 1, 1, ni4] + [1.0] * (n - 5)
    return pd.DataFrame({"Total Revenue": rev[:n], "Net Income": ni[:n]}).T.set_axis(cols, axis=1)


def test_bull_cartel_thresholds(sa, monkeypatch):
    def run(**kw):
        t = _Ticker(quarterly_income_stmt=_q_frame(**kw), info={"shortName": "Acme"})
        monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
        return sa.compute_bull_cartel("ACME")

    r = run()
    assert r["qualifies"] is True and r["name"] == "Acme" and r["screen"] == "bc"
    assert r["Sales_Growth_YoY_%"] == 30.0 and r["Profit_Growth_YoY_%"] == 50.0 and r["Net_Profit_Cr"] == 3.0
    assert run(rev0=115.0)["qualifies"] is False  # exactly 15% not > 15
    assert run(rev0=115.5)["qualifies"] is True
    assert run(ni0=2.4e7)["qualifies"] is False  # exactly 20%
    assert run(ni0=2.5e7)["qualifies"] is True
    r = run(ni0=1e7, ni4=0.5e7)  # exactly 1 Cr is not > 1
    assert r["qualifies"] is False and r["Net_Profit_Cr"] == 1.0
    r = run(rev4=0.0)
    assert r["qualifies"] is False and r["Sales_Growth_YoY_%"] is None
    # negative year-ago profit uses abs()
    r = run(ni0=3e7, ni4=-2e7)
    assert r["Profit_Growth_YoY_%"] == 250.0


def test_bull_cartel_guards(sa, monkeypatch):
    t = _Ticker(quarterly_income_stmt=_q_frame(n=4))
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    r = sa.compute_bull_cartel("ACME")
    assert r["error"] == "need ≥5 quarters of data" and r["qualifies"] is False
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _Ticker())
    assert sa.compute_bull_cartel("ACME")["error"] == "need ≥5 quarters of data"

    def boom(s):
        raise RuntimeError("yf")

    monkeypatch.setattr(sa.yf, "Ticker", boom)
    assert sa.compute_bull_cartel("ACME")["error"] == "yf"

    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _NoInfo(quarterly_income_stmt=_q_frame()))
    r = sa.compute_bull_cartel("ACME")
    assert r["qualifies"] is True and r["name"] == ""


# ------------------------------------------------------------------ darvas
def _box_df(close_last, n=25, lows=None, peak=10):
    highs = [10.0] * n
    highs[min(peak, n - 5)] = 20.0
    lows_ = list(lows) if lows is not None else [8.0] * n
    if lows is None:
        lows_[min(12, n - 3)] = 5.0
    closes = [9.0] * n
    closes[-1] = close_last
    return pd.DataFrame({"High": highs, "Low": lows_, "Close": closes})


def test_darvas_signals_and_boundaries(sa):
    r = sa.compute_darvas_box(_box_df(25.0), "X")
    assert r["signal"] == "BREAKOUT_BUY" and r["qualifies"] is True
    assert r["Box_Top"] == 20.0 and r["Box_Bottom"] == 5.0 and r["LTP"] == 25.0
    assert r["Upside_to_Top_%"] == -20.0 and r["Position_in_Box_%"] == 133.3
    r = sa.compute_darvas_box(_box_df(3.0), "X")
    assert r["signal"] == "BREAKDOWN_SELL" and r["qualifies"] is False
    r = sa.compute_darvas_box(_box_df(10.0), "X")
    assert r["signal"] == "IN_BOX" and r["Position_in_Box_%"] == 33.3 and r["Upside_to_Top_%"] == 100.0
    assert sa.compute_darvas_box(_box_df(20.0), "X")["signal"] == "IN_BOX"
    assert sa.compute_darvas_box(_box_df(5.0), "X")["signal"] == "IN_BOX"
    assert sa.compute_darvas_box(_box_df(20.01), "X")["signal"] == "BREAKOUT_BUY"
    assert sa.compute_darvas_box(_box_df(4.99), "X")["signal"] == "BREAKDOWN_SELL"


def test_darvas_guards_and_fallbacks(sa):
    assert sa.compute_darvas_box(_box_df(9, n=7), "X")["signal"] == "INSUFFICIENT_DATA"
    assert sa.compute_darvas_box(_box_df(9, n=8, peak=3), "X")["signal"] != "INSUFFICIENT_DATA"
    assert sa.compute_darvas_box(pd.DataFrame({"a": range(20)}), "X")["signal"] == "INSUFFICIENT_DATA"
    flat = pd.DataFrame({"High": [10.0] * 20, "Low": [8.0] * 20, "Close": [9.5] * 20})
    r = sa.compute_darvas_box(flat, "X")
    assert r["signal"] == "NO_BOX" and r["current_price"] == 9.5 and "box_top" not in r
    # flat lows -> falls back to the minimum of the segment
    r = sa.compute_darvas_box(_box_df(9.0, lows=[8.0] * 25), "X")
    assert r["Box_Bottom"] == 8.0
    # zero lows -> no bottom at all
    r = sa.compute_darvas_box(_box_df(9.0, lows=[0.0] * 25), "X")
    assert r["signal"] == "NO_BOX" and r["box_top"] == 20.0
    # the current bar never forms the box
    df = _box_df(9.0)
    df.loc[24, "High"] = 99.0
    assert sa.compute_darvas_box(df, "X")["Box_Top"] == 20.0
    assert sa.compute_darvas_box(_box_df(9.0), "X", confirm=2)["Box_Top"] == 20.0


# ------------------------------------------------------------------ piotroski
def _pio_ticker(good=True, with_inc=True):
    if good:
        inc = _frame({"Net Income": [40, 20], "Total Revenue": [300, 200], "Gross Profit": [150, 80]})
        bal = _frame({"Total Assets": [100, 100], "Long Term Debt": [10, 30], "Current Assets": [250, 190],
                      "Current Liabilities": [100, 100], "Share Issued": [90, 95]})
        cf = _frame({"Operating Cash Flow": [60, 30]})
    else:
        inc = _frame({"Net Income": [-5, 20], "Total Revenue": [100, 200], "Gross Profit": [10, 80]})
        bal = _frame({"Total Assets": [100, 100], "Long Term Debt": [40, 30], "Current Assets": [100, 190],
                      "Current Liabilities": [100, 100], "Share Issued": [100, 95]})
        cf = _frame({"Operating Cash Flow": [-5, 30]})
    return _Ticker(income_stmt=inc if with_inc else None, balance_sheet=bal, cash_flow=cf)


def test_piotroski_scores(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _pio_ticker(True))
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 9 and r["Piotroski_Score"] == 9 and r["qualifies"] is True
    assert r["interpretation"] == "STRONG" and r["error"] == ""
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _pio_ticker(False))
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 0 and r["qualifies"] is False and r["interpretation"] == "WEAK"


def test_piotroski_threshold_and_bands(sa, monkeypatch):
    # drop two criteria (F8, F9) -> 7 still qualifies; drop a third (F7) -> 6 does not
    tt = _pio_ticker(True)
    tt.income_stmt.loc["Gross Profit"] = [10, 80]
    tt.income_stmt.loc["Total Revenue"] = [100, 200]
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: tt)
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 7 and r["qualifies"] is True and r["interpretation"] == "STRONG"
    tt.balance_sheet.loc["Share Issued"] = [100, 95]
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 6 and r["qualifies"] is False and r["interpretation"] == "MODERATE"
    tt.balance_sheet.loc["Long Term Debt"] = [40, 30]
    tt.balance_sheet.loc["Current Assets"] = [100, 190]
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 4 and r["interpretation"] == "MODERATE"
    tt.cash_flow.loc["Operating Cash Flow"] = [-5, 30]
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 2 and r["interpretation"] == "WEAK"


def test_piotroski_missing_inputs(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _pio_ticker(with_inc=False))
    r = sa.compute_piotroski("ACME")
    assert r["error"] == "no_data" and r["f_score"] is None and r["qualifies"] is False
    # no balance sheet or cash flow: only F7 (assumed non-dilutive) and F8 can score
    good = _pio_ticker(True)
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _Ticker(income_stmt=good.income_stmt))
    r = sa.compute_piotroski("ACME")
    assert r["f_score"] == 2

    def boom(s):
        raise RuntimeError("yf")

    monkeypatch.setattr(sa.yf, "Ticker", boom)
    r = sa.compute_piotroski("ACME")
    assert r["error"] == "yf" and r["f_score"] is None


# ------------------------------------------------------------------ coffee can
def _cc_ticker(info=None, rev=(200, 150, 120, 100), ni=(30, 20, 15, 12), ebit=(60, 40, 30, 25), with_inc=True):
    inc = _frame({"Total Revenue": list(rev), "EBIT": list(ebit), "Net Income": list(ni)}) if with_inc else None
    bal = _frame({"Total Assets": [200, 180, 160, 150], "Current Liabilities": [50, 40, 40, 30],
                  "Long Term Debt": [20, 20, 20, 20], "Stockholders Equity": [100, 90, 80, 70]})
    return _Ticker(income_stmt=inc, balance_sheet=bal,
                   info=info if info is not None else {"debtToEquity": 45.0, "marketCap": 6e9})


def test_coffee_can_pass_and_each_criterion(sa, monkeypatch):
    def run(**kw):
        monkeypatch.setattr(sa.yf, "Ticker", lambda s: _cc_ticker(**kw))
        return sa.compute_coffee_can("ACME")

    r = run()
    assert r["qualifies"] is True and r["CC_Score"] == "5/5" and r["screen"] == "cc"
    assert r["CC_Criteria"] == {"C1": 1, "C2": 1, "C3": 1, "C4": 1, "C5": 1}
    assert run(rev=(105, 104, 102, 100))["CC_Criteria"]["C1"] == 0
    assert run(ebit=(10, 8, 6, 5))["CC_Criteria"]["C2"] == 0
    assert run(info={"debtToEquity": 100, "marketCap": 6e9})["CC_Criteria"]["C3"] == 0  # 1.0 not < 1
    assert run(info={"debtToEquity": 99, "marketCap": 6e9})["CC_Criteria"]["C3"] == 1
    assert run(info={"debtToEquity": 10, "marketCap": 6e9})["CC_Criteria"]["C3"] == 0  # 10 stays a ratio
    assert run(info={"debtToEquity": 0.5, "marketCap": 6e9})["CC_Criteria"]["C3"] == 1
    assert run(info={"debtToEquity": 50, "marketCap": 5e9})["CC_Criteria"]["C4"] == 1  # exactly 500 Cr
    assert run(info={"debtToEquity": 50, "marketCap": 4.9e9})["CC_Criteria"]["C4"] == 0
    assert run(info={"debtToEquity": 50})["CC_Criteria"]["C4"] == 0
    r = run(ni=(30, 20, 0, 12))
    assert r["CC_Criteria"]["C5"] == 0 and r["qualifies"] is False and r["CC_Score"] == "4/5"


def test_coffee_can_debt_fallback_and_errors(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _cc_ticker(info={"marketCap": 6e9}))
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C3"] == 1  # 20/100 from the balance sheet
    t = _cc_ticker(info={"marketCap": 6e9})
    t.balance_sheet.loc["Long Term Debt"] = [120, 120, 120, 120]
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C3"] == 0  # 1.2
    t.balance_sheet.loc["Stockholders Equity"] = [0, 0, 0, 0]
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C3"] == 0  # zero equity -> fail closed
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _cc_ticker(with_inc=False))
    assert sa.compute_coffee_can("ACME")["error"] == "no_data"

    def boom(s):
        raise RuntimeError("yf")

    monkeypatch.setattr(sa.yf, "Ticker", boom)
    assert sa.compute_coffee_can("ACME")["error"] == "yf"
    # single revenue point and zero oldest revenue -> C1 fails
    t = _cc_ticker(rev=(200, 100, 50, 0))
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C1"] == 0


# ------------------------------------------------------------------ enrich_symbol
def test_enrich_symbol_runs_requested_screens(sa, monkeypatch):
    calls = []
    for name in ("compute_magic_formula", "compute_bull_cartel", "compute_piotroski", "compute_coffee_can"):
        monkeypatch.setattr(sa, name, (lambda n: lambda s, suffix=".NS": calls.append(n) or {"qualifies": True})(name))
    monkeypatch.setattr(sa, "compute_darvas_box", lambda df, symbol="": {"signal": "IN_BOX"})
    monkeypatch.setattr(sa, "compute_golden_crossover", lambda df, symbol="": {"qualifies": False})
    t = _Ticker(income_stmt=_frame({"Net Income": [1, 1]}), quarterly_income_stmt=_q_frame())
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    ohlc = _gc_df(np.ones(250))
    out = sa.enrich_symbol("ACME", ohlc)
    assert out["symbol"] == "ACME" and out["suffix"] == ".NS"
    assert set(out) == {"symbol", "suffix", "darvas", "gc", "piotroski", "cc", "mf", "bc"}
    assert sorted(calls) == ["compute_bull_cartel", "compute_coffee_can", "compute_magic_formula", "compute_piotroski"]
    # active subset
    calls.clear()
    out = sa.enrich_symbol("ACME", None, ".BO", {"darvas", "mf"})
    assert out["suffix"] == ".BO" and "darvas" not in out and "gc" not in out  # no OHLC supplied
    assert set(out) == {"symbol", "suffix", "mf"} and calls == ["compute_magic_formula"]
    # technical-only screens do not touch yfinance financials
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: (_ for _ in ()).throw(AssertionError("no fetch")))
    out = sa.enrich_symbol("ACME", ohlc, ".NS", {"darvas", "gc"})
    assert set(out) == {"symbol", "suffix", "darvas", "gc"}


def test_enrich_symbol_without_statements_or_on_failure(sa, monkeypatch):
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _Ticker())  # no frames at all
    out = sa.enrich_symbol("ACME", None, ".NS", {"mf", "bc", "piotroski", "cc"})
    assert set(out) == {"symbol", "suffix"}

    def boom(s):
        raise RuntimeError("down")

    monkeypatch.setattr(sa.yf, "Ticker", boom)
    out = sa.enrich_symbol("ACME", None, ".NS", {"mf"})
    assert out["fetch_error"] == "down"

    monkeypatch.setattr(sa.yf, "Ticker", lambda s: _NoInfo(income_stmt=_frame({"Net Income": [1, 1]})))
    monkeypatch.setattr(sa, "compute_piotroski", lambda s, suffix=".NS": {"qualifies": False})
    assert "piotroski" in sa.enrich_symbol("ACME", None, ".NS", {"piotroski"})


# ------------------------------------------------------------------ excel
def test_flatten_for_sheet(sa):
    results = [
        {"symbol": "A", "suffix": ".NS", "mf": {"symbol": "A", "screen": "mf", "qualifies": True, "ROIC_%": 40, "error": ""}},
        {"symbol": "B", "mf": {"qualifies": False, "error": "no_data", "CC_Criteria": {"C1": 1}}},
        {"symbol": "C"},
        {"symbol": "D", "mf": {}},
    ]
    df = sa._flatten_for_sheet(results, "mf")
    assert list(df["Symbol"]) == ["A", "B"] and list(df["Qualifies"]) == ["✔", "✘"]
    assert df["ROIC_%"].iloc[0] == 40 and "screen" not in df.columns and "CC_Criteria" not in df.columns
    assert df["Error"].iloc[1] == "no_data" and pd.isna(df["Error"].iloc[0])
    assert list(df["Suffix"]) == [".NS", ""]
    assert sa._flatten_for_sheet(results, "bc").empty


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="sheets are meant to list qualifying stocks first, but sort_values(ascending=False) on the "
    "mapped key (check=0, cross=1) puts the non-qualifying rows first (screener_analysis.py "
    "save_results_excel)",
)
def test_save_results_excel_lists_qualifying_first(sa):
    results = [
        {"symbol": "A", "suffix": ".NS", "mf": {"qualifies": False, "ROIC_%": 1}},
        {"symbol": "B", "suffix": ".NS", "mf": {"qualifies": True, "ROIC_%": 2}},
    ]
    path = sa.save_results_excel(results, [], {"mf"})
    assert list(pd.ExcelFile(path).parse("Magic_Formula")["Symbol"]) == ["B", "A"]


def test_save_results_excel(sa):
    results = [
        {"symbol": "A", "suffix": ".NS", "mf": {"qualifies": False, "ROIC_%": 1}},
        {"symbol": "B", "suffix": ".NS", "mf": {"qualifies": True, "ROIC_%": 2}},
        {"symbol": "A", "suffix": ".NS", "darvas": {"qualifies": True, "signal": "BREAKOUT_BUY"}},
    ]
    hits = [{"Symbol": "B", "Screens_Count": 3}]
    path = sa.save_results_excel(results, hits, {"mf", "darvas", "cc"})
    x = pd.ExcelFile(path)
    assert path.parent == sa.DOWNLOAD_DIR and path.name.startswith("screener_analysis_")
    assert x.sheet_names == ["DISCLAIMER", "Magic_Formula", "Darvas_Box", "Multi_Screen_Hits", "Screener_Guide"]
    assert set(x.parse("Magic_Formula")["Symbol"]) == {"A", "B"}
    assert len(x.parse("Screener_Guide")) == 3
    # no hits, nothing active -> only disclaimer and guide
    path2 = sa.save_results_excel([], [], set())
    assert pd.ExcelFile(path2).sheet_names == ["DISCLAIMER", "Screener_Guide"]


# ------------------------------------------------------------------ run orchestration
def _stub_run(sa, monkeypatch, ohlc_map, nse_syms=("AAA", "BBB"), screener_in=None, enrich=None):
    monkeypatch.setattr(sa, "fetch_all_nse_symbols", lambda: list(nse_syms))
    monkeypatch.setattr(sa, "fetch_screener_in_symbols", lambda i, s, **k: (screener_in or {}).get(s, []))
    monkeypatch.setattr(sa, "bulk_download_ohlc", lambda t, period="1y": {f"{k}.NS": v for k, v in ohlc_map.items()})
    monkeypatch.setattr(sa, "compute_darvas_box", lambda df, symbol="": {"signal": "BREAKOUT_BUY" if symbol == "AAA" else "IN_BOX",
                                                                          "qualifies": symbol == "AAA"})
    monkeypatch.setattr(sa, "compute_golden_crossover", lambda df, symbol="": {"qualifies": symbol == "AAA", "DMA50": 1, "DMA200": 2})

    def default_enrich(sym, ohlc, suffix, active):
        return {"symbol": sym, "suffix": suffix,
                "mf": {"qualifies": sym == "AAA", "ROIC_%": 30.0, "Earnings_Yield_%": 20.0},
                "piotroski": {"qualifies": sym == "AAA", "f_score": 8},
                "cc": {"qualifies": False, "CC_Score": "3/5"}}

    monkeypatch.setattr(sa, "enrich_symbol", enrich or default_enrich)


def test_run_end_to_end_multi_screen_hits(sa, monkeypatch, capsys):
    big = _gc_df(np.ones(250))
    small = _gc_df(np.ones(100))
    _stub_run(sa, monkeypatch, {"AAA": big, "BBB": small},
              screener_in={"magic-formula": ["CCC"], "golden-crossover": []})
    res = sa.run(active_screens={"mf", "gc", "darvas", "piotroski", "cc"}, workers=2)
    out = capsys.readouterr().out
    allr = {r["symbol"]: r for r in res["all_results"]}
    assert set(allr) == {"AAA", "BBB", "CCC"} or set(allr) >= {"AAA", "BBB"}
    aaa = allr["AAA"]
    assert aaa["darvas"]["signal"] == "BREAKOUT_BUY" and aaa["gc"]["qualifies"] is True
    # BBB's frame has < 201 bars: no golden-crossover result
    assert "gc" not in allr["BBB"] and allr["BBB"]["darvas"]["signal"] == "IN_BOX"
    hits = res["multi_hits"]
    assert [h["Symbol"] for h in hits] == ["AAA"]
    h = hits[0]
    assert h["Screens_Count"] == 4 and h["Screens_Passed"] == "mf, gc, darvas, piotroski"
    assert h["Darvas_Signal"] == "BREAKOUT_BUY" and h["Piotroski_Score"] == 8 and h["CC_Score"] == "3/5"
    assert h["MF_ROIC_%"] == 30.0 and h["MF_EY_%"] == 20.0 and h["GC_DMA50"] == 1 and h["GC_DMA200"] == 2
    assert pd.ExcelFile(res["excel"]).sheet_names[0] == "DISCLAIMER"
    assert "MULTI-SCREEN HITS (3+ screens): 1" in out
    assert "Darvas Breakouts:    1" in out and "Golden Crossovers:   1" in out
    assert "Magic Formula" in out and "Bull Cartel" not in out  # inactive screens are not summarised
    assert "screener.in unavailable" in out  # golden-crossover screen came back empty


def test_run_three_screen_threshold(sa, monkeypatch):
    big = _gc_df(np.ones(250))

    def enrich(sym, ohlc, suffix, active):
        return {"symbol": sym, "suffix": suffix, "mf": {"qualifies": True}, "piotroski": {"qualifies": True}}

    _stub_run(sa, monkeypatch, {"AAA": big}, nse_syms=("AAA",), enrich=enrich)
    # AAA passes darvas + gc + mf + piotroski = 4; with only {mf, piotroski} active that's 2 -> no hit
    res = sa.run(active_screens={"mf", "piotroski"}, offline=True)
    assert res["multi_hits"] == []
    res = sa.run(active_screens={"mf", "piotroski", "gc"}, offline=True)
    assert [h["Symbol"] for h in res["multi_hits"]] == ["AAA"] and res["multi_hits"][0]["Screens_Count"] == 3


def test_run_modes_choose_sources(sa, monkeypatch):
    calls = {"si": [], "nse": 0}
    big = _gc_df(np.ones(250))
    _stub_run(sa, monkeypatch, {"AAA": big}, nse_syms=("AAA",))
    monkeypatch.setattr(sa, "fetch_screener_in_symbols", lambda i, s, **k: calls["si"].append(s) or [])
    real_fetch = sa.fetch_all_nse_symbols

    def counted():
        calls["nse"] += 1
        return real_fetch()

    monkeypatch.setattr(sa, "fetch_all_nse_symbols", counted)
    sa.run(active_screens={"mf", "gc", "bc"})
    assert sorted(calls["si"]) == ["golden-crossover", "magic-formula", "the-bull-cartel"]
    calls["si"].clear()
    sa.run(active_screens={"mf"}, offline=True)
    assert calls["si"] == []  # offline skips screener.in
    sa.run(active_screens={"mf"}, full_universe=True)
    assert calls["si"] == []  # full-universe also skips the pre-filter
    sa.run(active_screens={"bc"}, use_screener_in=False)
    assert calls["si"] == []


def test_run_universe_failure_and_worker_error(sa, monkeypatch, capsys):
    big = _gc_df(np.ones(250))

    def enrich(sym, ohlc, suffix, active):
        raise RuntimeError("worker died")

    _stub_run(sa, monkeypatch, {"AAA": big}, nse_syms=(), enrich=enrich)
    monkeypatch.setattr(sa, "fetch_screener_in_symbols", lambda *a, **k: [])
    res = sa.run(active_screens={"mf", "darvas"}, offline=True)
    out = capsys.readouterr().out
    assert "Full-universe fetch failed" in out
    # darvas breakout AAA is a fundamental candidate; its worker failure is recorded, not raised
    allr = {r["symbol"]: r for r in res["all_results"]}
    assert allr["AAA"]["error"] == "worker died" and allr["AAA"]["darvas"]["signal"] == "BREAKOUT_BUY"
    assert "1/1 done" in out


def test_run_progress_cadence(sa, monkeypatch, capsys):
    syms = [f"S{i:03d}" for i in range(50)]
    _stub_run(sa, monkeypatch, {s: _gc_df(np.ones(50)) for s in syms}, nse_syms=syms,
              enrich=lambda sym, o, sfx, a: {"symbol": sym})
    sa.run(active_screens={"mf"}, offline=True)
    out = capsys.readouterr().out
    assert "25/50 done" in out and "50/50 done" in out and "10/50 done" not in out


# ------------------------------------------------------------------ mutation follow-ups
def test_golden_cross_requires_a_strict_prior_inversion(sa):
    close = np.concatenate([np.full(250, 100.0), [200.0]])  # DMAs are exactly equal the bar before the jump
    r = sa.compute_golden_crossover(_gc_df(close), "X")
    assert r["dma50_above_200"] is True and r["qualifies"] is False


def test_golden_cross_recent_window_edge(sa):
    close = np.concatenate([np.linspace(100, 60, 250), np.linspace(60, 300, 60)])
    s50 = pd.Series(close).rolling(50).mean()
    s200 = pd.Series(close).rolling(200).mean()
    c = next(i for i in range(1, len(close)) if s50[i - 1] < s200[i - 1] and s50[i] > s200[i])
    # the last bar still below the 200 DMA must lie within the look-back window (offsets -2 .. -6)
    assert sa.compute_golden_crossover(_gc_df(close[: c + 5]), "X")["recent_crossover"] is True
    assert sa.compute_golden_crossover(_gc_df(close[: c + 6]), "X")["recent_crossover"] is False


def test_piotroski_equal_share_counts_are_not_dilution(sa, monkeypatch):
    t = _pio_ticker(True)
    t.balance_sheet.loc["Share Issued"] = [95, 95]
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    assert sa.compute_piotroski("ACME")["f_score"] == 9
    t.balance_sheet.loc["Share Issued"] = [96, 95]
    assert sa.compute_piotroski("ACME")["f_score"] == 8


def test_coffee_can_average_roce_must_exceed_15(sa, monkeypatch):
    cols = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31"])
    t = _cc_ticker()
    t.income_stmt = pd.DataFrame({"Total Revenue": [200, 150, 120, 100], "EBIT": [15, 15, 15, 15],
                                  "Net Income": [30, 20, 15, 12]}).T.set_axis(cols, axis=1)
    t.balance_sheet = pd.DataFrame({"Total Assets": [200] * 4, "Current Liabilities": [100] * 4}).T.set_axis(cols, axis=1)
    monkeypatch.setattr(sa.yf, "Ticker", lambda s: t)
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C2"] == 0  # exactly 15% is not enough
    t.income_stmt.loc["EBIT"] = [16, 16, 16, 16]
    assert sa.compute_coffee_can("ACME")["CC_Criteria"]["C2"] == 1
