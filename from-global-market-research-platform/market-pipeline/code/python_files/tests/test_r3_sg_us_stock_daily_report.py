"""Offline tests for sg_stock_daily_report and us_stock_daily_report (shared logic parametrized)."""

import importlib
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

MODS = ["sg_stock_daily_report", "us_stock_daily_report"]


class FakeFast:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeTicker:
    def __init__(self, symbol="X", **kw):
        self.symbol = symbol
        self.fast_info = FakeFast()
        self.info = {}
        self.actions = None
        self.calendar = {}
        self.options = ()
        self.insider_transactions = None
        self.institutional_holders = None
        self.news = []
        self._history = pd.DataFrame()
        self.history_calls = []
        self.income_stmt = self.financials = self.balance_sheet = None
        self.cash_flow = self.cashflow = None
        self.__dict__.update(kw)

    def history(self, period="1mo", auto_adjust=True):
        self.history_calls.append(period)
        if isinstance(self._history, Exception):
            raise self._history
        return self._history


def _install(monkeypatch, tmp_path, name):
    monkeypatch.chdir(tmp_path)
    yf = types.ModuleType("yfinance")
    yf.Ticker = lambda s: FakeTicker(s)
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(exist_ok=True)
    return mod


@pytest.fixture(params=MODS)
def m(request, monkeypatch, tmp_path):
    mod = _install(monkeypatch, tmp_path, request.param)
    yield mod
    sys.modules.pop(request.param, None)


@pytest.fixture
def sg(monkeypatch, tmp_path):
    mod = _install(monkeypatch, tmp_path, "sg_stock_daily_report")
    yield mod
    sys.modules.pop("sg_stock_daily_report", None)


@pytest.fixture
def us(monkeypatch, tmp_path):
    mod = _install(monkeypatch, tmp_path, "us_stock_daily_report")
    yield mod
    sys.modules.pop("us_stock_daily_report", None)


def _frame(rows, n=None):
    n = len(next(iter(rows.values())))
    cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"][:n])
    return pd.DataFrame(rows).T.set_axis(cols, axis=1)


def _statements(good=True):
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
    return inc, bal, cf


def _ticker(good=True, **kw):
    inc, bal, cf = _statements(good)
    return FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf, **kw)


def _ohlc(close_last, n=25, peak=10, lows=None):
    highs = [10.0] * n
    highs[min(peak, n - 5)] = 20.0
    lows_ = list(lows) if lows is not None else [8.0] * n
    if lows is None:
        lows_[min(12, n - 3)] = 5.0
    closes = [9.0] * n
    closes[-1] = close_last
    return pd.DataFrame({"High": highs, "Low": lows_, "Close": closes})


# ================================================================== shared helpers
def test_fmt_and_fmt_large(m):
    cur = m.fmt(1)[0:-4]  # currency prefix
    assert m.fmt(1234.5) == f"{cur}1,234.50"
    assert m.fmt("12", prefix="", decimals=0) == "12"
    assert m.fmt(None) == "N/A" and m.fmt("") == "N/A" and m.fmt("abc") == "abc"
    p = cur
    assert m.fmt_large(1.5e12) == f"{p}1.50T" and m.fmt_large(1e12) == f"{p}1.00T"
    assert m.fmt_large(999e9) == f"{p}999.00B" and m.fmt_large(1e9) == f"{p}1.00B"
    assert m.fmt_large(999e6) == f"{p}999.00M" and m.fmt_large(1e6) == f"{p}1.00M"
    assert m.fmt_large(999e3) == f"{p}999.00K" and m.fmt_large(1e3) == f"{p}1.00K"
    assert m.fmt_large(999) == f"{p}999.00" and m.fmt_large(-2e9) == f"{p}-2.00B"
    assert m.fmt_large(None) == "N/A" and m.fmt_large("x") == "N/A"


def test_pct_section_row(m, capsys):
    assert "▲ 1.50%" in m.pct(1.5) and "\033[92m" in m.pct(0)
    assert "▼ 2.00%" in m.pct(-2) and "\033[91m" in m.pct(-2)
    assert m.pct(None) == "N/A" and m.pct("x") == "N/A"
    m.section("title")
    m.row("Label", "v")
    out = capsys.readouterr().out
    assert "TITLE" in out and "-" * 60 in out and out.rstrip().endswith("v")


def test_yf_financials_real_dataframes_and_fallback_names(m):
    inc, bal, cf = _statements()
    t = FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf)
    got = m._yf_financials(t)
    assert got[0] is inc and got[1] is bal and got[2] is cf
    t = FakeTicker(income_stmt=pd.DataFrame(), financials=inc, cash_flow=None, cashflow=cf)
    got = m._yf_financials(t)
    assert got[0] is inc and got[1] is None and got[2] is cf  # empty frames are skipped
    t = FakeTicker(income_stmt="notadf")
    assert m._yf_financials(t) == (None, None, None)


def test_row_helper(m):
    df = pd.DataFrame({"a": [10.0, np.nan], "b": [5.0, 1.0]}, index=["A", "B"])
    assert m._row(df, "A") == 10.0 and m._row(df, "A", col=1) == 5.0
    assert m._row(df, "B") is None and m._row(df, "Z", "A") == 10.0
    assert m._row(df, "A", col=9) is None and m._row(None, "A") is None
    assert m._row(pd.DataFrame(), "A") is None


def test_fetch_quote_prefers_fast_info(m):
    t = FakeTicker(
        fast_info=FakeFast(last_price=11.0, previous_close=10.0, market_cap=5e9, year_year=None, year_high=15.0,
                           currency="XXX", year_change=0.1),
        info={"currentPrice": 99.0, "previousClose": 98.0, "marketCap": 1, "fiftyTwoWeekHigh": 50.0,
              "fiftyTwoWeekLow": 2.0, "trailingPE": 12.5, "shortName": "Acme", "sector": "Tech",
              "exchange": "NYQ", "currency": "AAA", "dividendYield": 0.02, "recommendationKey": "buy"},
    )
    q = m.fetch_quote(t)
    assert q["lastPrice"] == 11.0 and q["previousClose"] == 10.0 and q["marketCap"] == 5e9
    assert q["52wHigh"] == 15.0 and q["52wLow"] == 2.0  # low falls back to info
    assert q["currency"] == "XXX" and q["ytdChange"] == 0.1
    assert q["trailingPE"] == 12.5 and q["name"] == "Acme" and q["sector"] == "Tech" and q["exchange"] == "NYQ"
    assert q["dividendYield"] == 0.02


def test_fetch_quote_info_fallbacks_and_failures(m):
    class Boom:
        def __getattr__(self, n):
            raise RuntimeError("fast_info blocked")

    t = FakeTicker(fast_info=Boom(), info={"regularMarketPrice": 7.0, "longName": "Long Co", "volume": 100,
                                          "averageVolume": 50, "fiftyDayAverage": 6.0,
                                          "twoHundredDayAverage": 5.0, "recommendationKey": "hold"})
    q = m.fetch_quote(t)
    assert q["lastPrice"] == 7.0 and q["name"] == "Long Co" and q["volume"] == 100 and q["avgVolume"] == 50
    assert q["50dAvg"] == 6.0 and q["200dAvg"] == 5.0 and q["ytdChange"] is None

    class NoInfo(FakeTicker):
        @property
        def info(self):
            raise RuntimeError("blocked")

        @info.setter
        def info(self, v):
            pass

    q = m.fetch_quote(NoInfo(fast_info=FakeFast(last_price=3.0)))
    assert q["lastPrice"] == 3.0 and q["trailingPE"] is None and q["name"] == "—"


def test_fetch_historical(m, capsys):
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    t = FakeTicker(_history=df)
    assert m.fetch_historical(t, period="3mo") is df and t.history_calls == ["3mo"]
    assert m.fetch_historical(FakeTicker(_history=None)).empty
    assert m.fetch_historical(FakeTicker(_history=RuntimeError("down"))).empty
    assert "Historical error" in capsys.readouterr().out


def test_fetch_corporate_actions(m):
    idx = pd.date_range("2024-01-01", periods=10, freq="MS")
    acts = pd.DataFrame({"Dividends": [0.5, 0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 0, 0],
                         "Stock Splits": [0, 2, 0, 0, 3, 0, 0, 0, 4, 5]}, index=idx)
    t = FakeTicker(actions=acts, calendar={"Earnings Date": [pd.Timestamp("2026-02-01"), "x"],
                                           "Ex-Dividend Date": "2026-01-10", "Dividend Date": "2026-01-20"})
    r = m.fetch_corporate_actions(t)
    assert [d["amount"] for d in r["dividends"]] == [0.7, 0.8, 0.9, 1.0, 1.1]  # last five
    assert r["dividends"][0]["date"] == "2024-04-01"
    assert [s["ratio"] for s in r["splits"]] == [3, 4, 5]  # last three
    assert r["calendar"]["next_earnings"].startswith("2026-02-01")
    assert r["calendar"]["ex_dividend_date"] == "2026-01-10" and r["calendar"]["dividend_date"] == "2026-01-20"


def test_fetch_corporate_actions_empty_and_errors(m):
    r = m.fetch_corporate_actions(FakeTicker(actions=pd.DataFrame()))
    assert r["dividends"] == [] and r["splits"] == []
    assert r["calendar"]["next_earnings"] == "—" and r["calendar"]["ex_dividend_date"] == "—"
    r = m.fetch_corporate_actions(FakeTicker(actions=pd.DataFrame({"x": [1]})))  # missing columns -> swallowed
    assert r["dividends"] == []

    class BadCal(FakeTicker):
        @property
        def calendar(self):
            raise RuntimeError("no calendar")

        @calendar.setter
        def calendar(self, v):
            pass

    assert m.fetch_corporate_actions(BadCal())["calendar"] == {}


# ---------------------------------------------------------------- darvas
def test_darvas_signals_numbers_and_boundaries(m):
    r = m.compute_darvas_box(_ohlc(25.0))
    assert r["signal"] == "BREAKOUT_BUY" and r["box_top"] == 20.0 and r["box_bottom"] == 5.0
    assert r["current_price"] == 25.0 and r["box_range"] == 15.0 and r["position_in_box_pct"] == 133.3
    assert r["upside_to_top_pct"] == -20.0 and r["confirm_days"] == 3 and r["data_points"] == 25
    assert m.compute_darvas_box(_ohlc(3.0))["signal"] == "BREAKDOWN_SELL"
    r = m.compute_darvas_box(_ohlc(10.0))
    assert r["signal"] == "IN_BOX" and r["position_in_box_pct"] == 33.3 and r["upside_to_top_pct"] == 100.0
    assert m.compute_darvas_box(_ohlc(20.0))["signal"] == "IN_BOX"
    assert m.compute_darvas_box(_ohlc(5.0))["signal"] == "IN_BOX"
    assert m.compute_darvas_box(_ohlc(20.01))["signal"] == "BREAKOUT_BUY"
    assert m.compute_darvas_box(_ohlc(4.99))["signal"] == "BREAKDOWN_SELL"


def test_darvas_guards_and_fallbacks(m):
    r = m.compute_darvas_box(None)
    assert r["signal"] == "INSUFFICIENT_DATA" and r["note"] == "Empty OHLC DataFrame"
    assert m.compute_darvas_box(pd.DataFrame())["signal"] == "INSUFFICIENT_DATA"
    r = m.compute_darvas_box(pd.DataFrame({"a": range(20)}))
    assert r["signal"] == "INSUFFICIENT_DATA" and "OHLC columns" in r["note"]
    r = m.compute_darvas_box(_ohlc(9, n=7))
    assert r["note"] == "Need ≥ 8 bars; got 7"
    assert m.compute_darvas_box(_ohlc(9, n=8))["signal"] != "INSUFFICIENT_DATA"
    assert m.compute_darvas_box(_ohlc(9, n=8), confirm=4)["signal"] == "INSUFFICIENT_DATA"
    flat = pd.DataFrame({"High": [10.0] * 20, "Low": [8.0] * 20, "Close": [9.0] * 20})
    assert m.compute_darvas_box(flat)["signal"] == "NO_BOX"
    assert m.compute_darvas_box(_ohlc(9.0, lows=[8.0] * 25))["box_bottom"] == 8.0
    r = m.compute_darvas_box(_ohlc(9.0, lows=[0.0] * 25))
    assert r["signal"] == "NO_BOX" and r["box_top"] == 20.0 and "bottom" in r["note"]
    df = _ohlc(9.0)
    df.loc[24, "High"] = 99.0
    assert m.compute_darvas_box(df)["box_top"] == 20.0  # current bar never forms the box


def test_display_darvas_box(m, capsys):
    m.display_darvas_box(m.compute_darvas_box(_ohlc(25.0)))
    out = capsys.readouterr().out
    assert "BREAKOUT BUY" in out and "Box Top" in out and "133.3%" in out
    m.display_darvas_box({"signal": "NO_BOX", "note": "nothing"})
    out = capsys.readouterr().out
    assert "No confirmed Darvas box" in out
    m.display_darvas_box({"signal": "BREAKDOWN_SELL", "box_top": 5, "box_bottom": 4})
    assert "BREAKDOWN SELL" in capsys.readouterr().out
    m.display_darvas_box({"signal": "IN_BOX"})
    assert "IN BOX" in capsys.readouterr().out
    m.display_darvas_box({"signal": "WEIRD"})
    assert "Signal: WEIRD" in capsys.readouterr().out
    m.display_darvas_box({"signal": "INSUFFICIENT_DATA"})
    assert "Insufficient data" in capsys.readouterr().out


# ---------------------------------------------------------------- piotroski
def test_piotroski_perfect_and_failing(m):
    r = m.compute_piotroski_score(_ticker(True), "AAA")
    assert r["f_score"] == 9 and r["symbol"] == "AAA" and r["interpretation"].startswith("STRONG")
    assert all(v == 1 for v in r["component_scores"].values())
    assert r["details"]["ROA_current_%"] == 40.0 and r["details"]["ROA_prev_%"] == 20.0
    assert r["details"]["GrossMargin_curr_%"] == 50.0 and r["details"]["CurrentRatio_curr"] == 2.5
    r = m.compute_piotroski_score(_ticker(False), "AAA")
    assert r["f_score"] == 0 and r["interpretation"].startswith("WEAK")


def test_piotroski_each_criterion_and_bands(m):
    def comp(mods):
        t = _ticker(True)
        for (attr, row), vals in mods.items():
            getattr(t, attr).loc[row] = vals
        return m.compute_piotroski_score(t, "A")["component_scores"]

    assert comp({("income_stmt", "Net Income"): [-5, 20]})["F1_ROA_positive"] == 0
    assert comp({("cash_flow", "Operating Cash Flow"): [-5, 30]})["F2_OCF_positive"] == 0
    assert comp({("income_stmt", "Net Income"): [10, 20]})["F3_ROA_improving"] == 0
    assert comp({("cash_flow", "Operating Cash Flow"): [20, 30]})["F4_Accruals"] == 0
    assert comp({("balance_sheet", "Long Term Debt"): [40, 30]})["F5_Leverage_down"] == 0
    assert comp({("balance_sheet", "Long Term Debt"): [30, 30]})["F5_Leverage_down"] == 0
    assert comp({("balance_sheet", "Current Assets"): [100, 190]})["F6_CurrentRatio_up"] == 0
    assert comp({("balance_sheet", "Share Issued"): [100, 95]})["F7_No_dilution"] == 0
    assert comp({("balance_sheet", "Share Issued"): [95, 95]})["F7_No_dilution"] == 1
    assert comp({("income_stmt", "Gross Profit"): [30, 80]})["F8_GrossMargin_up"] == 0
    assert comp({("income_stmt", "Total Revenue"): [100, 200]})["F9_AssetTurnover_up"] == 0
    # interpretation bands: 7 STRONG, 6 MODERATE, 4 MODERATE, 3 WEAK
    t = _ticker(True)
    t.income_stmt.loc["Gross Profit"] = [30, 80]
    t.income_stmt.loc["Total Revenue"] = [100, 200]
    assert m.compute_piotroski_score(t)["f_score"] == 7
    assert m.compute_piotroski_score(t)["interpretation"].startswith("STRONG")
    t.balance_sheet.loc["Share Issued"] = [100, 95]
    r = m.compute_piotroski_score(t)
    assert r["f_score"] == 6 and r["interpretation"].startswith("MODERATE")
    t.balance_sheet.loc["Long Term Debt"] = [40, 30]
    t.balance_sheet.loc["Current Assets"] = [100, 190]
    assert m.compute_piotroski_score(t)["f_score"] == 4
    t.cash_flow.loc["Operating Cash Flow"] = [-5, 30]
    r = m.compute_piotroski_score(t)
    assert r["f_score"] == 2 and r["interpretation"].startswith("WEAK")


def test_piotroski_missing_data(m):
    r = m.compute_piotroski_score(FakeTicker(), "A")
    assert r == {"symbol": "A", "error": "No income statement data available"}
    t = FakeTicker(income_stmt=_statements()[0])  # no balance sheet / cash flow
    r = m.compute_piotroski_score(t, "A")
    assert r["f_score"] == 2 and r["component_scores"]["F7_No_dilution"] == 1
    assert r["details"]["ROA_current_%"] == "N/A"


def test_display_piotroski_score(m, capsys):
    m.display_piotroski_score(m.compute_piotroski_score(_ticker(True), "A"))
    out = capsys.readouterr().out
    assert "9/9" in out and "\033[92m9/9" in out
    m.display_piotroski_score({"error": "bad"})
    assert "bad" in capsys.readouterr().out
    base = {"interpretation": "x", "component_scores": {}, "details": {}}
    m.display_piotroski_score({**base, "f_score": 5})
    assert "\033[93m5/9" in capsys.readouterr().out
    m.display_piotroski_score({**base, "f_score": 3})
    assert "\033[91m3/9" in capsys.readouterr().out


def test_display_price_summary_values(m, capsys):
    q = {"lastPrice": 110.0, "previousClose": 100.0, "name": "Acme", "exchange": "X", "sector": "Tech",
         "dayHigh": 111, "dayLow": 99, "52wHigh": 120, "52wLow": 80, "200dAvg": 105, "volume": 12345,
         "marketCap": 5e9, "trailingPE": 12.0, "forwardPE": 11.0, "priceToBook": 2.0, "dividendYield": 0.025,
         "industry": "SW", "country": "ZZ", "open": 100, "ytdChange": 0.1, "50dAvg": 104, "avgVolume": 999,
         "pegRatio": 1.5, "trailingEps": 5, "forwardEps": 6, "beta": 1.1, "shortRatio": 2.0,
         "targetMean": 130, "recommendation": "BUY", "analystCount": 12}
    m.display_price_summary(q)
    out = capsys.readouterr().out
    assert "Acme" in out and "▲ 10.00%" in out and "12,345" in out and "2.50%" in out and "5.00B" in out
    m.display_price_summary({})
    out = capsys.readouterr().out
    assert out.count("N/A") >= 4


def test_display_corporate_actions_basic(m, capsys):
    m.display_corporate_actions({"calendar": {"ex_dividend_date": "2026-01-10", "dividend_date": "2026-01-20"},
                                 "dividends": [{"date": "2025-01-01", "amount": 0.5},
                                               {"date": "2025-07-01", "amount": 0.6}]})
    out = capsys.readouterr().out
    assert "2026-01-10" in out and "2025-01-01" in out and out.index("2025-07-01") < out.index("2025-01-01")
    m.display_corporate_actions({})
    assert "—" in capsys.readouterr().out


# ================================================================== SG specifics
def test_sg_universe_lists(sg):
    assert len(sg.STI_30) == 30 and sg.STI_30[0] == "D05.SI"
    assert len(sg.STI_TOP_15) == 15 and set(sg.STI_TOP_15) <= set(sg.STI_30)
    assert len(sg.SGX_ALL) > 150 and all(s.endswith(".SI") for s in sg.SGX_ALL)


def test_sg_get_ticker_adds_suffix(sg):
    assert sg._get_ticker("d05").symbol == "D05.SI"
    assert sg._get_ticker(" bn4.si ").symbol == "BN4.SI"


def _pegy_ticker(**info):
    base = {"pegRatio": 1.5, "dividendYield": 0.03, "trailingPE": 12.0, "currentPrice": 110.0,
            "twoHundredDayAverage": 100.0}
    base.update(info)
    return FakeTicker(info=base, _history=pd.DataFrame({"Close": [1.0]}))


def test_sg_pegy_and_breakout(sg):
    r = sg.compute_pegy_and_breakout(_pegy_ticker(), "A")
    assert r["cmp"] == 110.0 and r["pe_ratio"] == 12.0 and r["peg_ratio"] == 1.5
    assert r["dividend_yield_%"] == 3.0 and r["pegy_adjusted"] == 4.5  # peg + yield% (3.0)
    assert r["ma_200"] == 100.0
    assert r["breakout"] == {"distance_from_200ma_%": 10.0, "signal": "Above 200-day MA - Uptrend"}
    r = sg.compute_pegy_and_breakout(_pegy_ticker(currentPrice=90.0), "A")
    assert r["breakout"]["signal"] == "Below 200-day MA - Downtrend"
    # exactly +/-5% is consolidation
    r = sg.compute_pegy_and_breakout(_pegy_ticker(currentPrice=105.0), "A")
    assert r["breakout"]["signal"] == "Near 200-day MA - Consolidation"
    r = sg.compute_pegy_and_breakout(_pegy_ticker(currentPrice=95.0), "A")
    assert r["breakout"]["signal"] == "Near 200-day MA - Consolidation"
    r = sg.compute_pegy_and_breakout(_pegy_ticker(currentPrice=105.5), "A")
    assert r["breakout"]["signal"] == "Above 200-day MA - Uptrend"
    # missing PEG: no adjusted ratio; missing MA: no breakout block
    r = sg.compute_pegy_and_breakout(_pegy_ticker(pegRatio=None, twoHundredDayAverage=None), "A")
    assert r["pegy_adjusted"] is None and r["peg_ratio"] is None and r["breakout"] == {} and r["ma_200"] is None
    r = sg.compute_pegy_and_breakout(_pegy_ticker(dividendYield=None), "A")
    assert r["dividend_yield_%"] is None and r["pegy_adjusted"] == 1.5


def test_sg_pegy_data_failure(sg):
    class Bad(FakeTicker):
        @property
        def info(self):
            raise RuntimeError("blocked")

        @info.setter
        def info(self, v):
            pass

    assert sg.compute_pegy_and_breakout(Bad(), "A") == {"symbol": "A", "error": "Data fetch failed"}


def test_sg_display_pegy_and_breakout(sg, capsys):
    sg.display_pegy_and_breakout({"error": "x"})
    assert "x" in capsys.readouterr().out
    sg.display_pegy_and_breakout(sg.compute_pegy_and_breakout(_pegy_ticker(), "A"))
    out = capsys.readouterr().out
    assert "Current Market Price" in out and "3.00%" in out and "4.5" in out and "10.00%" in out
    assert "Uptrend" in out


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="compute_pegy_and_breakout stores dividend_yield_% = None for a non-payer, and "
    "display_pegy_and_breakout formats result.get('dividend_yield_%', 0) with :.2f, which raises "
    "TypeError for the None value (sg_stock_daily_report.py display_pegy_and_breakout)",
)
def test_sg_display_pegy_for_non_dividend_payer(sg, capsys):
    sg.display_pegy_and_breakout(sg.compute_pegy_and_breakout(_pegy_ticker(dividendYield=None), "A"))
    out = capsys.readouterr().out
    assert "Dividend Yield" in out and "PEGY Adjusted" in out


def test_sg_run_text_and_scans(sg, monkeypatch, capsys):
    t = _ticker(True, info={"shortName": "DBS", "currentPrice": 40.0, "previousClose": 39.0,
                            "pegRatio": 1.0, "dividendYield": 0.04, "twoHundredDayAverage": 30.0},
                _history=_ohlc(25.0))
    monkeypatch.setattr(sg.yf, "Ticker", lambda s: t)
    data = sg.run(" d05 ", output_format="text", run_scans=True)
    out = capsys.readouterr().out
    assert data["symbol"] == "D05.SI" and data["quote"]["name"] == "DBS"
    assert data["scans"]["darvas"]["signal"] == "BREAKOUT_BUY" and data["scans"]["piotroski"]["f_score"] == 9
    assert data["scans"]["pegy"]["cmp"] == 40.0
    assert t.history_calls == ["1mo", "6mo", "6mo"]  # report, darvas, and the PEGY scan's own fetch
    for h in ("LIVE QUOTE & VALUATION", "CORPORATE ACTIONS", "DARVAS BOX SCAN", "PIOTROSKI F-SCORE",
              "PEGY RATIO & BREAKOUT ANALYSIS", "SINGAPORE STOCK REPORT — D05.SI"):
        assert h in out
    t.history_calls.clear()
    data = sg.run("D05.SI", output_format="excel", run_scans=False)
    out = capsys.readouterr().out
    assert data["scans"] == {} and t.history_calls == ["1mo"]
    assert "LIVE QUOTE" not in out  # excel mode prints no console sections


def test_sg_run_scans_only(sg, monkeypatch, capsys):
    t = _ticker(True, info={"currentPrice": 40.0, "twoHundredDayAverage": 30.0, "dividendYield": 0.02,
                            "pegRatio": 1.0}, _history=_ohlc(25.0))
    monkeypatch.setattr(sg.yf, "Ticker", lambda s: t)
    out = sg.run_scans_only(" u11 ")
    assert out["symbol"] == "U11.SI" and set(out) == {"symbol", "darvas", "piotroski", "pegy"}
    assert "SCANS — U11.SI" in capsys.readouterr().out


def _sg_result(sym, with_scans=True):
    d = {"symbol": f"{sym}.SI", "quote": {"name": f"{sym} Co", "sector": "Fin", "lastPrice": 11.0,
                                           "previousClose": 10.0, "trailingPE": 9.0, "priceToBook": 1.1,
                                           "dividendYield": 0.04, "52wHigh": 12, "52wLow": 8, "volume": 1000,
                                           "200dAvg": 9.5, "marketCap": 1e9},
         "scans": {}}
    if with_scans:
        d["scans"] = {"darvas": {"signal": "IN_BOX", "box_top": 12, "box_bottom": 8,
                                 "position_in_box_pct": 50, "upside_to_top_pct": 9},
                      "piotroski": {"f_score": 7, "interpretation": "STRONG"},
                      "pegy": {"pegy_adjusted": 3.5, "peg_ratio": 1.0, "dividend_yield_%": 4.0,
                               "breakout": {"signal": "Above 200-day MA - Uptrend",
                                            "distance_from_200ma_%": 15.8}}}
    return d


def test_sg_write_excel_report(sg):
    from openpyxl import load_workbook

    sg._write_excel_report([_sg_result("AAA"), _sg_result("BBB")], include_scans=True)
    files = list(sg.DOWNLOAD_DIR.glob("sg_stock_report_*.xlsx"))
    assert len(files) == 1
    wb = load_workbook(files[0])
    assert wb.sheetnames == ["Summary", "AAA", "BBB"]
    ws = wb["Summary"]
    headers = [c.value for c in ws[1]]
    assert headers[:3] == ["Symbol", "Company", "Sector"] and headers[-1] == "Breakout Signal"
    assert len(headers) == 17
    row = [c.value for c in ws[2]]
    assert row[0] == "AAA" and row[3] == 11.0 and row[4] == pytest.approx(10.0)  # change %
    assert row[7] == pytest.approx(4.0) and row[11] == "IN_BOX" and row[14] == 7 and row[15] == 3.5
    assert row[16] == "Above 200-day MA - Uptrend"
    d = wb["AAA"]
    assert d["A1"].value == "AAA — AAA Co"
    labels = {d[f"A{r}"].value: d[f"B{r}"].value for r in range(1, d.max_row + 1)}
    assert labels["CMP (S$)"] == 11.0 and labels["Change %"] == pytest.approx(10.0)
    assert labels["Signal"] == "IN_BOX" and labels["Score (0-9)"] == 7 and labels["PEGY Adjusted"] == 3.5
    assert labels["Distance from 200MA %"] == 15.8


def test_sg_write_excel_report_without_scans(sg):
    from openpyxl import load_workbook

    res = _sg_result("CCC", with_scans=False)
    res["quote"].pop("lastPrice")  # no price: change% must be blank, not an error
    sg._write_excel_report([res], include_scans=False)
    wb = load_workbook(next(sg.DOWNLOAD_DIR.glob("sg_stock_report_*.xlsx")))
    assert [c.value for c in wb["Summary"][1]][-1] == "Volume" and wb.sheetnames == ["Summary", "CCC"]
    assert wb["Summary"]["E2"].value is None
    d = wb["CCC"]
    labels = [d[f"A{r}"].value for r in range(1, d.max_row + 1)]
    assert "TECHNICAL & FUNDAMENTAL SCANS" not in labels


def test_sg_run_batch(sg, monkeypatch, capsys):
    calls = []

    def fake_run(sym, output_format="excel", run_scans=False):
        calls.append((sym, output_format, run_scans))
        if sym.startswith("BAD"):
            raise RuntimeError("boom")
        return _sg_result(sym.replace(".SI", ""))

    monkeypatch.setattr(sg, "run", fake_run)
    wrote = []
    monkeypatch.setattr(sg, "_write_excel_report", lambda results, scans: wrote.append((len(results), scans)))
    res = sg.run_batch(["OK1.SI", "BAD.SI", "OK2"], output_format="excel", run_scans=True)
    out = capsys.readouterr().out
    assert [r["symbol"] for r in res] == ["OK1.SI", "OK2.SI"] and wrote == [(2, True)]
    assert "2 OK, 1 failed" in out and "Failed: BAD" in out and "[02/3] BAD" in out
    wrote.clear()
    sg.run_batch(["OK1"], output_format="text")
    assert wrote == []  # only the excel mode writes the workbook
    wrote.clear()
    sg.run_batch(["BAD1"], output_format="excel")
    assert wrote == []  # nothing to write
    seen = []
    monkeypatch.setattr(sg, "run", lambda s, output_format="excel", run_scans=False: seen.append(s) or _sg_result("Z"))
    sg.run_batch(output_format="text")
    assert seen == sg.STI_30


# ================================================================== US specifics
def test_us_get_ticker_and_lists(us):
    assert us._get_ticker(" aapl ").symbol == "AAPL"
    assert len(us.DOW_JONES_30) == 30 and len(us.NASDAQ_50) == 50


def _chain(call_oi, put_oi):
    calls = pd.DataFrame({"strike": list(call_oi), "openInterest": list(call_oi.values())})
    puts = pd.DataFrame({"strike": list(put_oi), "openInterest": list(put_oi.values())})
    return types.SimpleNamespace(calls=calls, puts=puts)


def test_us_options_pcr_and_max_pain(us):
    chain = _chain({90.0: 100, 100.0: 300, 110.0: 100}, {90.0: 100, 100.0: 300, 110.0: 200})
    t = FakeTicker(options=("2026-01-16", "2026-02-20"))
    t.option_chain = lambda exp: chain
    r = us.fetch_options_data(t)
    assert r["expiry_date"] == "2026-01-16" and r["call_oi"] == 500 and r["put_oi"] == 600
    assert r["pcr"] == 1.2 and r["sentiment"] == "Bullish (PCR>1)"
    # strike with minimal aggregate pain
    pain = {k: sum(max(0, k - c) * v for c, v in {90.0: 100, 100.0: 300, 110.0: 100}.items())
            + sum(max(0, p - k) * v for p, v in {90.0: 100, 100.0: 300, 110.0: 200}.items())
            for k in (90.0, 100.0, 110.0)}
    assert r["max_pain"] == min(pain, key=pain.get) == 100.0
    # PCR exactly 1 is bearish (strictly greater required)
    chain2 = _chain({100.0: 100}, {100.0: 100})
    t.option_chain = lambda exp: chain2
    assert us.fetch_options_data(t)["sentiment"] == "Bearish (PCR≤1)"
    # no call OI: pcr None
    chain3 = _chain({100.0: 0}, {100.0: 50})
    t.option_chain = lambda exp: chain3
    r = us.fetch_options_data(t)
    assert r["pcr"] is None and r["sentiment"] == "Bearish (PCR≤1)"


def test_us_options_empty_and_errors(us, capsys):
    assert us.fetch_options_data(FakeTicker(options=())) == {}

    def boom(exp):
        raise RuntimeError("no chain")

    t = FakeTicker(options=("2026-01-16",))
    t.option_chain = boom
    assert us.fetch_options_data(t) == {}
    assert "Options error" in capsys.readouterr().out


def test_us_insider_institutional_news(us, capsys):
    df = pd.DataFrame({
        "Insider": ["A", "B", "C", "D", "E", "F"], "Position": ["CEO"] * 6, "Transaction": ["Sale"] * 6,
        "Shares": [1, 2, 3, 4, 5, 6], "Value": [10, 20, 30, 40, 50, 60],
        "Start Date": pd.to_datetime(["2025-01-01", "2025-06-01", "2025-03-01", "2025-02-01", "2025-05-01",
                                      "2025-04-01"]),
    })
    r = us.fetch_insider_trades(FakeTicker(insider_transactions=df))
    assert [x["insider"] for x in r] == ["B", "E", "F", "C", "D"]  # newest five
    assert r[0]["date"].startswith("2025-06-01") and r[0]["shares"] == 2
    assert us.fetch_insider_trades(FakeTicker(insider_transactions=pd.DataFrame())) == []
    assert us.fetch_insider_trades(FakeTicker()) == []
    assert us.fetch_insider_trades(FakeTicker(insider_transactions=pd.DataFrame({"x": [1]}))) == []
    assert "Insider trades error" in capsys.readouterr().out

    inst = pd.DataFrame({"Holder": [f"H{i}" for i in range(7)], "Shares": range(7)})
    assert [h["Holder"] for h in us.fetch_institutional_holders(FakeTicker(institutional_holders=inst))] == \
        ["H0", "H1", "H2", "H3", "H4"]
    assert us.fetch_institutional_holders(FakeTicker()) == []
    assert us.fetch_institutional_holders(FakeTicker(institutional_holders=pd.DataFrame())) == []

    class BadInst(FakeTicker):
        @property
        def institutional_holders(self):
            raise RuntimeError("x")

        @institutional_holders.setter
        def institutional_holders(self, v):
            pass

    assert us.fetch_institutional_holders(BadInst()) == []
    assert "Institutional holders error" in capsys.readouterr().out


def test_us_fetch_news_formats(us, capsys):
    news = [
        {"content": {"title": "New style", "pubDate": "2026-01-02T10:30:00Z"}},
        {"title": "Old style", "providerPublishTime": 1_700_000_000},
        {"title": "No time"},
    ] + [{"title": f"n{i}"} for i in range(5)]
    r = us.fetch_news(FakeTicker(news=news))
    assert len(r) == 5
    assert r[0] == {"title": "New style", "published": "2026-01-02T10:30"}
    assert r[1]["title"] == "Old style" and r[1]["published"] == "2023-11-14 22:13"
    assert r[2] == {"title": "No time", "published": ""}
    assert us.fetch_news(FakeTicker(news=None)) == []

    class BadNews(FakeTicker):
        @property
        def news(self):
            raise RuntimeError("down")

        @news.setter
        def news(self, v):
            pass

    assert us.fetch_news(BadNews()) == []
    assert "News error" in capsys.readouterr().out


def _cc_ticker(info=None, fcf=(50, 40, 30, 20), cf_rows=None, ni=(60, 45, 30, 20), eq=(200, 180, 160, 140),
               rev=(400, 300, 200, 100)):
    inc = _frame({"Total Revenue": list(rev), "Net Income": list(ni)})
    bal = _frame({"Stockholders Equity": list(eq), "Long Term Debt": [20, 20, 20, 20]})
    cf = _frame(cf_rows if cf_rows is not None else {"Free Cash Flow": list(fcf)})
    return FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf,
                      info=info if info is not None else {"debtToEquity": 45.0, "marketCap": 5e9})


def test_us_coffee_can_pass_and_details(us):
    r = us.compute_coffee_can(_cc_ticker(), "A")
    assert r["qualifies"] is True and r["score"] == "6/6" and r["symbol"] == "A"
    d = r["details"]
    assert d["Revenue_CAGR_%"] == round(((400 / 100) ** (1 / 3) - 1) * 100, 2) and d["Revenue_years"] == 3
    assert d["Debt_to_Equity"] == 0.45 and d["Market_Cap"] == "$5.00B" and d["FCF_latest_$M"] == 0.0
    assert d["Loss_years"] == 0 and d["Years_analysed"] == 4
    assert d["ROE_avg_%"] == round(np.mean([60 / 200, 45 / 180, 30 / 160, 20 / 140]) * 100, 2)
    assert d["ROE_min_%"] == round(min([60 / 200, 45 / 180, 30 / 160, 20 / 140]) * 100, 2)


def test_us_coffee_can_criteria_boundaries(us):
    def crit(**kw):
        return us.compute_coffee_can(_cc_ticker(**kw), "A")["criteria"]

    assert crit(rev=(105, 104, 102, 100))["C1_Revenue_CAGR_gt10"] == 0
    assert crit(ni=(10, 8, 6, 5))["C2_ROE_gt15"] == 0
    assert crit(info={"debtToEquity": 100, "marketCap": 5e9})["C3_LowDebt"] == 0
    assert crit(info={"debtToEquity": 99, "marketCap": 5e9})["C3_LowDebt"] == 1
    assert crit(info={"debtToEquity": 10, "marketCap": 5e9})["C3_LowDebt"] == 0
    assert crit(info={"debtToEquity": 0.5, "marketCap": 5e9})["C3_LowDebt"] == 1
    assert crit(info={"debtToEquity": 50, "marketCap": 1e9})["C4_MCap_ge1B"] == 1  # exactly $1B
    assert crit(info={"debtToEquity": 50, "marketCap": 9.9e8})["C4_MCap_ge1B"] == 0
    assert crit(ni=(60, 45, 0, 20))["C5_NoProfitLoss"] == 0
    assert crit(fcf=(0, 40, 30, 20))["C6_FreeCashFlow_pos"] == 0
    assert crit(fcf=(-1, 40, 30, 20))["C6_FreeCashFlow_pos"] == 0
    r = us.compute_coffee_can(_cc_ticker(ni=(60, 45, 0, 20)), "A")
    assert r["qualifies"] is False and r["score"] == "5/6" and r["details"]["Loss_years"] == 1


def test_us_coffee_can_fallbacks(us):
    # no info: D/E from balance sheet, market cap from fast_info
    t = _cc_ticker(info={})
    t.fast_info = FakeFast(market_cap=3e9)
    r = us.compute_coffee_can(t, "A")
    assert r["criteria"]["C3_LowDebt"] == 1 and r["details"]["Debt_to_Equity"] == 0.1
    assert r["criteria"]["C4_MCap_ge1B"] == 1
    t = _cc_ticker(info={})
    r = us.compute_coffee_can(t, "A")
    assert r["criteria"]["C4_MCap_ge1B"] == 0 and r["details"]["Market_Cap"] == "N/A"
    # FCF derived from operating cash flow less capex (abs)
    t = _cc_ticker(cf_rows={"Operating Cash Flow": [100e6, 90e6], "Capital Expenditure": [-60e6, -50e6]})
    r = us.compute_coffee_can(t, "A")
    assert r["criteria"]["C6_FreeCashFlow_pos"] == 1 and r["details"]["FCF_latest_$M"] == 40.0
    t = _cc_ticker(cf_rows={"Operating Cash Flow": [100e6, 90e6]})
    r = us.compute_coffee_can(t, "A")
    assert r["criteria"]["C6_FreeCashFlow_pos"] == 0 and r["details"]["FCF_latest_$M"] == "N/A"
    # non-positive equity years are ignored in ROE; no equity at all -> N/A
    t = _cc_ticker(eq=(-1, -1, -1, -1))
    r = us.compute_coffee_can(t, "A")
    assert r["criteria"]["C2_ROE_gt15"] == 0 and r["details"]["ROE_avg_%"] == "N/A"
    r = us.compute_coffee_can(FakeTicker(income_stmt=_frame({"Total Revenue": [100.0]})), "A")
    assert r["details"]["Revenue_CAGR_%"] == "N/A" and r["criteria"]["C1_Revenue_CAGR_gt10"] == 0
    assert us.compute_coffee_can(FakeTicker(), "A") == {"symbol": "A", "error": "No income statement data available"}


def test_us_display_coffee_can_and_helpers(us, capsys):
    us.display_coffee_can(us.compute_coffee_can(_cc_ticker(), "A"))
    out = capsys.readouterr().out
    assert "QUALIFIES" in out and "DOES NOT" not in out and "6/6" in out
    r = us.compute_coffee_can(_cc_ticker(ni=(60, 45, 0, 20)), "A")
    us.display_coffee_can(r)
    assert "DOES NOT QUALIFY" in capsys.readouterr().out
    us.display_coffee_can({"error": "oops"})
    assert "oops" in capsys.readouterr().out


def test_us_display_options_insiders_holders_news_history(us, capsys):
    us.display_options_summary({})
    assert "Options data not available." in capsys.readouterr().out
    us.display_options_summary({"expiry_date": "2026-01-16", "call_oi": 1500, "put_oi": 2000, "pcr": 1.333,
                                "sentiment": "Bullish (PCR>1)", "max_pain": 100.0})
    out = capsys.readouterr().out
    assert "1,500" in out and "1.333" in out and "$100.00" in out
    us.display_options_summary({"expiry_date": "x"})
    assert out.count("N/A") <= 2 and "N/A" in capsys.readouterr().out

    us.display_insider_trades([], "AAPL")
    assert "No recent insider transactions found for AAPL." in capsys.readouterr().out
    us.display_insider_trades([{"transaction": "Sale", "insider": "Cook", "position": "CEO", "shares": 1000,
                                "value": 2.5e6, "date": "2025-06-01 00:00:00"},
                               {"insider": "Anon"}], "AAPL")
    out = capsys.readouterr().out
    assert "2025-06-01" in out and "1,000" in out and "$2.50M" in out and "Cook [CEO]" in out and "Anon" in out

    us.display_institutional_holders([])
    assert "No institutional holder data available." in capsys.readouterr().out
    us.display_institutional_holders([{"Holder": "Vanguard", "pctHeld": 0.0825, "Shares": 1.2e9, "pctChange": 0.01},
                                      {"Holder": "Other", "pctHeld": 0.01, "Shares": 5e6}])
    out = capsys.readouterr().out
    assert "Vanguard" in out and " 8.25%" in out and "1.20B" in out and "▲ 1.00%" in out and "Δ —" in out

    us.display_news([], "AAPL")
    assert "No news available." in capsys.readouterr().out
    us.display_news([{"published": "2026-01-02 10:30", "title": "Headline"}], "AAPL")
    out = capsys.readouterr().out
    assert "Latest News — AAPL".upper() in out and "Headline" in out and "2026-01-02 10:30" in out

    us.display_historical_summary(None)
    assert "No historical data available." in capsys.readouterr().out
    us.display_historical_summary(pd.DataFrame({"Close": [10.0]}))
    assert "Insufficient closing-price data." in capsys.readouterr().out
    us.display_historical_summary(pd.DataFrame({"Close": [10.0, 12.0, 11.0]}))
    out = capsys.readouterr().out
    assert "Trading days" in out and "$12.00" in out and "$10.00" in out and "$11.00" in out and "▲ 10.00%" in out


def _us_ticker_full():
    t = _ticker(True, info={"shortName": "Apple", "currentPrice": 110.0, "previousClose": 100.0,
                            "marketCap": 3e12, "dividendYield": 0.005, "recommendationKey": "buy"},
                _history=_ohlc(25.0))
    t._history.index = pd.date_range("2025-01-01", periods=len(t._history))
    return t


def test_us_run_text_and_json(us, monkeypatch, capsys):
    t = _us_ticker_full()
    t.options = ("2026-01-16",)
    t.option_chain = lambda e: _chain({100.0: 10}, {100.0: 20})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)
    data = us.run(" aapl ", show_options=True, output_format="text", run_scans=True)
    out = capsys.readouterr().out
    assert data["symbol"] == "AAPL" and "scans" not in data and data["timestamp"]
    for h in ("LIVE QUOTE & VALUATION", "CORPORATE ACTIONS & EARNINGS CALENDAR", "OPTIONS SUMMARY",
              "HISTORICAL SUMMARY", "RECENT INSIDER TRANSACTIONS", "TOP INSTITUTIONAL HOLDERS",
              "LATEST NEWS — AAPL", "DARVAS BOX SCAN", "PIOTROSKI F-SCORE", "COFFEE CAN PORTFOLIO SCREEN (US)"):
        assert h in out
    assert t.history_calls == ["1mo", "6mo"]
    # json mode saves a file and prints no sections
    t.history_calls.clear()
    data = us.run("aapl", show_options=False, output_format="json", run_scans=True)
    files = list(us.DOWNLOAD_DIR.glob("AAPL_report_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["quote"]["name"] == "Apple" and saved["options"] == {}
    assert saved["scans"]["darvas"]["signal"] == "BREAKOUT_BUY" and saved["scans"]["piotroski"]["f_score"] == 9
    assert len(saved["historical_30d"]) == 25 and "JSON saved" in capsys.readouterr().out
    assert data["scans"]["coffee_can"]["symbol"] == "AAPL"
    data = us.run("aapl", output_format="json", run_scans=False)
    assert data["scans"] == {}


def test_us_run_json_without_history(us, monkeypatch):
    t = _us_ticker_full()
    t._history = pd.DataFrame()
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)
    data = us.run("aapl", output_format="json")
    assert data["historical_30d"] == []


def test_us_run_scans_only(us, monkeypatch, capsys):
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _us_ticker_full())
    out = us.run_scans_only(" aapl ")
    assert set(out) == {"symbol", "darvas", "piotroski", "coffee_can"} and out["symbol"] == "AAPL"
    assert "SCANS — AAPL" in capsys.readouterr().out


def _us_result(sym, with_scans=True):
    d = {"symbol": sym, "timestamp": "t", "quote": {
        "name": f"{sym} Inc", "exchange": "NMS", "sector": "Tech", "lastPrice": 110.0, "previousClose": 100.0,
        "marketCap": 1e12, "trailingPE": 20.0, "forwardPE": 18.0, "beta": 1.2, "dividendYield": 0.0123456,
        "recommendation": "BUY", "targetMean": 130.0, "52wHigh": 120, "52wLow": 80}}
    if with_scans:
        d["scans"] = {"darvas": {"signal": "IN_BOX", "box_top": 12, "box_bottom": 8},
                      "piotroski": {"f_score": 7}, "coffee_can": {"qualifies": sym == "AAA", "score": "6/6"}}
    return d


def test_us_write_summary_csv(us):
    us._write_summary_csv([_us_result("AAA"), _us_result("BBB")], include_scans=True)
    df = pd.read_csv(next(us.DOWNLOAD_DIR.glob("us_scan_summary_*.csv")))
    assert list(df["Symbol"]) == ["AAA", "BBB"] and list(df["Change%"]) == [10.0, 10.0]
    assert list(df["DivYield%"]) == [1.235, 1.235]  # rounded to 3 dp
    assert list(df["CoffeeCan"]) == ["YES", "NO"] and list(df["Piotroski_Score"]) == [7, 7]
    assert list(df["Darvas_Signal"]) == ["IN_BOX", "IN_BOX"] and list(df["Darvas_BoxBottom"]) == [8, 8]
    us._write_summary_csv([_us_result("CCC", with_scans=False)], include_scans=False)
    df = pd.read_csv(next(us.DOWNLOAD_DIR.glob("us_batch_summary_*.csv")))
    assert "Darvas_Signal" not in df.columns and df["Symbol"].iloc[0] == "CCC"
    before = set(p.name for p in us.DOWNLOAD_DIR.glob("*.csv"))
    us._write_summary_csv([])  # nothing to write
    assert set(p.name for p in us.DOWNLOAD_DIR.glob("*.csv")) == before


def test_us_write_summary_csv_blank_quote_fields(us):
    us._write_summary_csv([{"symbol": "ZZZ", "quote": {}}])
    df = pd.read_csv(next(us.DOWNLOAD_DIR.glob("us_batch_summary_*.csv")))
    assert pd.isna(df["Price"].iloc[0]) and pd.isna(df["Change%"].iloc[0]) and pd.isna(df["DivYield%"].iloc[0])


def test_us_run_batch(us, monkeypatch, capsys):
    calls = []

    def fake_run(sym, show_options=False, output_format="json", run_scans=False):
        calls.append((sym, show_options, output_format, run_scans))
        if sym == "BAD":
            raise RuntimeError("explode")
        return _us_result(sym)

    monkeypatch.setattr(us, "run", fake_run)
    res = us.run_batch(["AAA", "BAD", "BBB"], show_options=True, output_format="text", run_scans=True)
    out = capsys.readouterr().out
    assert [r["symbol"] for r in res] == ["AAA", "BBB"] and calls[0] == ("AAA", True, "text", True)
    assert "2 OK, 1 failed" in out and "Failed: BAD" in out
    assert list(us.DOWNLOAD_DIR.glob("us_scan_summary_*.csv"))
    seen = []
    monkeypatch.setattr(us, "run", lambda s, **k: seen.append(s) or _us_result(s))
    us.run_batch()
    assert seen == us.DOW_JONES_30
