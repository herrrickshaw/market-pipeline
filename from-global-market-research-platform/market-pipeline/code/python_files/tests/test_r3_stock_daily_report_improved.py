"""Offline tests for stock_daily_report_improved (NSE/BSE report, Darvas, Piotroski, Coffee Can)."""

import importlib
import json
import sys
import types

import pandas as pd
import pytest


class _Ctx:
    """Context-manager base: ``with NSE(...) as nse`` returns self."""

    def __init__(self, *a, **k):
        self.init_kwargs = k

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeNSE(_Ctx):
    quote_val = {"trade_info": {"totalTradedVolume": 1234}}
    eq_val = {"priceInfo": {"lastPrice": 100.0}}
    meta_val = {"isin": "INE001"}
    actions_val = [{"exDate": "01-Jan", "purpose": "Dividend"}] * 7
    meetings_val = [{"meetingDate": "02-Jan", "purpose": "Results"}] * 5
    hist_val = [{"CH_CLOSING_PRICE": 10}, {"CH_CLOSING_PRICE": 12}]
    bulk_val = []
    raise_in = ()

    def _maybe(self, name, value):
        if name in self.raise_in:
            raise RuntimeError(f"{name} failed")
        return value

    def quote(self, symbol, section=None):
        return self._maybe("quote", self.quote_val)

    def equityQuote(self, symbol):
        return self._maybe("equityQuote", self.eq_val)

    def equityMetaInfo(self, symbol):
        return self._maybe("meta", self.meta_val)

    def actions(self, symbol=None):
        return self._maybe("actions", self.actions_val)

    def boardMeetings(self, symbol=None):
        return self._maybe("meetings", self.meetings_val)

    def equityHistoricalData(self, symbol=None, start=None, end=None):
        return self._maybe("hist", self.hist_val)

    def bulkdeals(self, option_type=None, fromdate=None, todate=None):
        return self._maybe("bulk", self.bulk_val)

    def optionChain(self, symbol):
        return self._maybe("oc", {"data": 1})

    def maxpain(self, oc):
        return 2500.0

    def compileOptionChain(self, oc):
        return {"pcr": 1.25}


class FakeBSE(_Ctx):
    scrip = "500325"
    quote_val = {"LastRate": 101.5, "Volume": 5000, "Mktcap": 2e10}
    actions_val = [{"exDate": "03-Jan", "purpose": "Bonus"}] * 6
    raises = False

    def getScripCode(self, ticker):
        if self.raises:
            raise RuntimeError("bse down")
        return self.scrip

    def quote(self, code):
        return self.quote_val

    def actions(self, scripcode=None):
        return self.actions_val


@pytest.fixture
def sd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # DOWNLOAD_DIR is created relative to cwd
    nse = types.ModuleType("nse")
    nse.NSE = FakeNSE
    bse = types.ModuleType("bse")
    bse.BSE = FakeBSE
    nspy = types.ModuleType("nsepython")
    nspy.get_bulkdeals = lambda: pd.DataFrame()
    nspy.equity_history = lambda *a, **k: pd.DataFrame()
    yf = types.ModuleType("yfinance")
    yf.Ticker = lambda t: types.SimpleNamespace(info={})
    for name, m in (("nse", nse), ("bse", bse), ("nsepython", nspy), ("yfinance", yf)):
        monkeypatch.setitem(sys.modules, name, m)
    for cls in (FakeNSE, FakeBSE):
        monkeypatch.setattr(cls, "raise_in", (), raising=False)
    sys.modules.pop("stock_daily_report_improved", None)
    mod = importlib.import_module("stock_daily_report_improved")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "nse_bse_data")
    (tmp_path / "nse_bse_data").mkdir(exist_ok=True)
    yield mod
    sys.modules.pop("stock_daily_report_improved", None)


# ------------------------------------------------------------------ import guards
def test_import_exits_without_nse_or_bse(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "nse", None)
    sys.modules.pop("stock_daily_report_improved", None)
    with pytest.raises(SystemExit):
        importlib.import_module("stock_daily_report_improved")
    nse = types.ModuleType("nse")
    nse.NSE = FakeNSE
    monkeypatch.setitem(sys.modules, "nse", nse)
    monkeypatch.setitem(sys.modules, "bse", None)
    with pytest.raises(SystemExit):
        importlib.import_module("stock_daily_report_improved")
    sys.modules.pop("stock_daily_report_improved", None)


def test_optional_deps_flags(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    for n, cls in (("nse", "NSE"), ("bse", "BSE")):
        m = types.ModuleType(n)
        setattr(m, cls, FakeNSE)
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "nsepython", None)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    sys.modules.pop("stock_daily_report_improved", None)
    mod = importlib.import_module("stock_daily_report_improved")
    out = capsys.readouterr().out
    assert mod.NSEPYTHON_OK is False and mod.YFINANCE_OK is False
    assert "nsepython not found" in out and "yfinance not found" in out
    # without nsepython: no OHLC history; without yfinance the scans report an error dict
    assert mod.fetch_ohlc_history("X").empty
    assert "yfinance not installed" in mod.compute_piotroski_score("X")["error"]
    assert "yfinance not installed" in mod.compute_coffee_can("X")["error"]
    sys.modules.pop("stock_daily_report_improved", None)


# ------------------------------------------------------------------ formatting
def test_fmt_pct_section_row(sd, capsys):
    assert sd.fmt(1234.5) == "₹1,234.50"
    assert sd.fmt("12", prefix="$", decimals=0) == "$12"
    assert sd.fmt(None) == "N/A"
    assert sd.fmt("abc") == "abc"
    assert sd.fmt(0) == "₹0.00"
    assert sd.fmt("") == "N/A"
    up, down = sd.pct(1.5), sd.pct(-2)
    assert "▲ 1.50%" in up and "\033[92m" in up
    assert "▼ 2.00%" in down and "\033[91m" in down
    assert "▲ 0.00%" in sd.pct(0)  # zero counts as up
    assert sd.pct("x") == "N/A" and sd.pct(None) == "N/A"
    sd.section("hello")
    sd.row("Label", "val")
    out = capsys.readouterr().out
    assert "HELLO" in out and "-" * 60 in out
    assert "  Label" in out and out.rstrip().endswith("val")


# ------------------------------------------------------------------ nse fetchers
def test_fetchers_truncate_and_default(sd):
    nse = FakeNSE()
    assert len(sd.fetch_nse_actions(nse, "X")) == 5
    assert len(sd.fetch_nse_board_meetings(nse, "X")) == 3
    assert sd.fetch_nse_quote(nse, "X") == FakeNSE.quote_val
    assert sd.fetch_nse_equity_quote(nse, "X") == FakeNSE.eq_val
    assert sd.fetch_nse_meta(nse, "X") == FakeNSE.meta_val
    FakeNSE.actions_val, FakeNSE.meetings_val = None, None
    try:
        assert sd.fetch_nse_actions(nse, "X") == []
        assert sd.fetch_nse_board_meetings(nse, "X") == []
    finally:
        FakeNSE.actions_val = [{"exDate": "01-Jan", "purpose": "Dividend"}] * 7
        FakeNSE.meetings_val = [{"meetingDate": "02-Jan", "purpose": "Results"}] * 5


def test_fetchers_swallow_errors(sd, monkeypatch, capsys):
    monkeypatch.setattr(FakeNSE, "raise_in", ("quote", "equityQuote", "meta", "actions", "meetings", "oc"))
    nse = FakeNSE()
    assert sd.fetch_nse_quote(nse, "X") == {}
    assert sd.fetch_nse_equity_quote(nse, "X") == {}
    assert sd.fetch_nse_meta(nse, "X") == {}
    assert sd.fetch_nse_actions(nse, "X") == []
    assert sd.fetch_nse_board_meetings(nse, "X") == []
    assert sd.fetch_nse_option_chain(nse, "X") == {}
    out = capsys.readouterr().out
    for tag in ("NSE quote error", "NSE equityQuote error", "NSE meta error",
                "NSE actions error", "NSE board meetings error", "NSE option chain error"):
        assert tag in out


def test_fetch_nse_historical_tries_both_method_names(sd, capsys):
    nse = FakeNSE()
    assert sd.fetch_nse_historical(nse, "X") == FakeNSE.hist_val

    class OnlyOld:
        def fetch_equity_historical_data(self, symbol=None, start=None, end=None):
            return [{"close": 1}]

    assert sd.fetch_nse_historical(OnlyOld(), "X") == [{"close": 1}]

    class Failing:
        def equityHistoricalData(self, **k):
            raise RuntimeError("first failed")

        def fetch_equity_historical_data(self, **k):
            return None  # falsy -> []

    assert sd.fetch_nse_historical(Failing(), "X") == []
    assert "equityHistoricalData" in capsys.readouterr().out
    assert sd.fetch_nse_historical(object(), "X") == []  # neither method exists


def test_fetch_nse_historical_window(sd):
    seen = {}

    class N:
        def equityHistoricalData(self, symbol=None, start=None, end=None):
            seen.update(start=start, end=end)
            return []

    sd.fetch_nse_historical(N(), "X", days=10)
    span = seen["end"] - seen["start"]
    assert abs(span.total_seconds() - 10 * 86400) < 5  # two datetime.today() calls, microseconds apart


def test_option_chain(sd):
    out = sd.fetch_nse_option_chain(FakeNSE(), "X")
    assert out["maxpain"] == 2500.0 and out["pcr"] == 1.25 and out["raw"] == {"data": 1}

    class NoChain(FakeNSE):
        def optionChain(self, symbol):
            return None

    assert sd.fetch_nse_option_chain(NoChain(), "X") == {}


# ------------------------------------------------------------------ bulk deals
def test_normalize_bulk_df(sd):
    df = pd.DataFrame(
        {
            " Symbol ": ["TCS"],
            "Client Name": ["ACME"],
            "Quantity Traded": ["1,26,922"],
            "Trade Price / Wght. Avg. Price": ["3,500.5"],
            "Buy/Sell": ["BUY"],
        }
    )
    out = sd._normalize_bulk_df(df)
    assert len(out) == 1
    d = out[0]
    assert d["symbol"] == "TCS" and d["clientname"] == "ACME" and d["buysell"] == "BUY"
    assert d["quantity"] == "126922"  # commas stripped
    assert pd.isna(d["price"])  # '3,500.5' is not numeric once to_numeric(errors=coerce)
    df2 = df.copy()
    df2["Trade Price / Wght. Avg. Price"] = ["3500.5"]
    assert sd._normalize_bulk_df(df2)[0]["price"] == 3500.5
    # original frame is not mutated
    assert list(df.columns)[0] == " Symbol "


def test_normalize_bulk_df_missing_columns(sd, capsys):
    out = sd._normalize_bulk_df(pd.DataFrame({"symbol": ["A"], "price": [1]}))
    assert out == []
    assert "Missing columns after rename" in capsys.readouterr().out


def _bulk_frame():
    return pd.DataFrame(
        {"Symbol": ["TCS"], "Client Name": ["Z"], "Quantity Traded": [10], "Price": [5.0], "Buy/Sell": ["SELL"]}
    )


def test_fetch_bulk_deals_primary_nsepython(sd, monkeypatch):
    monkeypatch.setattr(sd.nspy, "get_bulkdeals", _bulk_frame)
    out = sd.fetch_bulk_deals(FakeNSE())
    assert out[0]["symbol"] == "TCS" and out[0]["quantity"] == "10"


def test_fetch_bulk_deals_falls_to_nse_api(sd, monkeypatch, capsys):
    def boom():
        raise RuntimeError("bulk csv failed")

    monkeypatch.setattr(sd.nspy, "get_bulkdeals", boom)
    monkeypatch.setattr(FakeNSE, "bulk_val", [{"symbol": "INFY"}])
    assert sd.fetch_bulk_deals(FakeNSE()) == [{"symbol": "INFY"}]
    assert "nsepython bulk deals error" in capsys.readouterr().out
    # empty nsepython frame also falls through
    monkeypatch.setattr(sd.nspy, "get_bulkdeals", lambda: pd.DataFrame())
    assert sd.fetch_bulk_deals(FakeNSE()) == [{"symbol": "INFY"}]
    # nsepython missing entirely
    monkeypatch.setattr(sd, "NSEPYTHON_OK", False)
    assert sd.fetch_bulk_deals(FakeNSE()) == [{"symbol": "INFY"}]


def test_fetch_bulk_deals_local_csv_fallback(sd, monkeypatch, capsys):
    monkeypatch.setattr(sd, "NSEPYTHON_OK", False)
    monkeypatch.setattr(FakeNSE, "bulk_val", [])
    assert sd.fetch_bulk_deals(FakeNSE()) == []
    out = capsys.readouterr().out
    assert "No local fallback found" in out and "Trying local fallback" in out
    _bulk_frame().to_csv(sd.DOWNLOAD_DIR / "bulk_deals.csv", index=False)
    got = sd.fetch_bulk_deals(FakeNSE())
    assert got[0]["symbol"] == "TCS" and got[0]["price"] == 5.0
    (sd.DOWNLOAD_DIR / "bulk_deals.csv").write_bytes(b"")  # empty file -> pandas EmptyDataError
    assert sd.fetch_bulk_deals(FakeNSE()) == []
    assert "Bulk deals CSV error" in capsys.readouterr().out


def test_fetch_bulk_deals_nse_api_error_then_csv(sd, monkeypatch, capsys):
    monkeypatch.setattr(sd, "NSEPYTHON_OK", False)
    monkeypatch.setattr(FakeNSE, "raise_in", ("bulk",))
    assert sd.fetch_bulk_deals(FakeNSE()) == []
    assert "NSE bulk deals API error" in capsys.readouterr().out


# ------------------------------------------------------------------ BSE
def test_fetch_bse_data(sd, monkeypatch, capsys):
    out = sd.fetch_bse_data(FakeBSE(), "RELIANCE")
    assert out["scrip_code"] == "500325" and out["quote"]["LastRate"] == 101.5
    assert len(out["actions"]) == 5  # truncated to 5
    monkeypatch.setattr(FakeBSE, "scrip", None)
    assert sd.fetch_bse_data(FakeBSE(), "ZZZ") == {}
    assert "Scrip code not found for ZZZ" in capsys.readouterr().out
    monkeypatch.setattr(FakeBSE, "raises", True)
    assert sd.fetch_bse_data(FakeBSE(), "ZZZ") == {}
    assert "BSE error" in capsys.readouterr().out


# ------------------------------------------------------------------ OHLC history
def test_fetch_ohlc_history_sorts_by_date_column(sd, monkeypatch):
    raw = pd.DataFrame({"CH_TIMESTAMP": ["2024-03-02", "2024-03-01", "2024-03-03"], "CH_CLOSING_PRICE": [2, 1, 3]})
    monkeypatch.setattr(sd.nspy, "equity_history", lambda sym, series, start, end: raw)
    out = sd.fetch_ohlc_history("X")
    assert list(out["CH_CLOSING_PRICE"]) == [1, 2, 3]
    assert list(out.index) == [0, 1, 2]


def test_fetch_ohlc_history_date_format_args_and_fallbacks(sd, monkeypatch, capsys):
    seen = {}

    def eh(sym, series, start, end):
        seen.update(sym=sym, series=series, start=start, end=end)
        return pd.DataFrame({"A": [3, 1]})  # no date column -> unsorted

    monkeypatch.setattr(sd.nspy, "equity_history", eh)
    out = sd.fetch_ohlc_history("X", days=5)
    assert list(out["A"]) == [3, 1]
    assert seen["series"] == "EQ" and seen["sym"] == "X"
    assert len(seen["start"]) == 10 and seen["start"][2] == "-"  # dd-mm-YYYY
    monkeypatch.setattr(sd.nspy, "equity_history", lambda *a: None)
    assert sd.fetch_ohlc_history("X").empty

    def boom(*a):
        raise RuntimeError("down")

    monkeypatch.setattr(sd.nspy, "equity_history", boom)
    assert sd.fetch_ohlc_history("X").empty
    assert "Darvas OHLC fetch error" in capsys.readouterr().out


# ------------------------------------------------------------------ Darvas
def _box_df(close_last, highs_tail=10.0, lows=None, n=25, hi_col="High", lo_col="Low", cl_col="Close"):
    highs = [10.0] * n
    peak = min(10, n - 5)
    highs[peak] = 20.0
    lows_ = lows or ([8.0] * n)
    if lows is None:
        lows_[min(12, n - 3)] = 5.0
    closes = [9.0] * n
    closes[-1] = close_last
    return pd.DataFrame({hi_col: highs, lo_col: lows_, cl_col: closes})


def test_darvas_insufficient_and_missing_columns(sd):
    r = sd.compute_darvas_box(_box_df(9, n=7))  # needs confirm + 5 = 8
    assert r["signal"] == "INSUFFICIENT_DATA" and "Need ≥ 8 rows; got 7" in r["note"]
    assert sd.compute_darvas_box(_box_df(9, n=8))["signal"] != "INSUFFICIENT_DATA"
    assert sd.compute_darvas_box(pd.DataFrame({"x": range(30)}))["signal"] == "INSUFFICIENT_DATA"
    r = sd.compute_darvas_box(_box_df(9, n=9), confirm=4)  # 9 rows, needs 9
    assert r["signal"] != "INSUFFICIENT_DATA"
    assert sd.compute_darvas_box(_box_df(9, n=8), confirm=4)["signal"] == "INSUFFICIENT_DATA"


def test_darvas_signals_and_numbers(sd):
    r = sd.compute_darvas_box(_box_df(25.0))
    assert r["signal"] == "BREAKOUT_BUY" and r["box_top"] == 20.0 and r["box_bottom"] == 5.0
    assert r["upside_to_top_pct"] == -20.0 and r["position_in_box_pct"] == 133.3
    assert r["data_points"] == 25 and r["confirm_days"] == 3 and r["box_range"] == 15.0
    r = sd.compute_darvas_box(_box_df(3.0))
    assert r["signal"] == "BREAKDOWN_SELL" and r["current_price"] == 3.0
    r = sd.compute_darvas_box(_box_df(10.0))
    assert r["signal"] == "IN_BOX"
    assert r["position_in_box_pct"] == 33.3 and r["upside_to_top_pct"] == 100.0
    # boundaries: exactly at the top / bottom is still IN_BOX
    assert sd.compute_darvas_box(_box_df(20.0))["signal"] == "IN_BOX"
    assert sd.compute_darvas_box(_box_df(5.0))["signal"] == "IN_BOX"
    assert sd.compute_darvas_box(_box_df(20.01))["signal"] == "BREAKOUT_BUY"
    assert sd.compute_darvas_box(_box_df(4.99))["signal"] == "BREAKDOWN_SELL"


def test_darvas_current_bar_excluded_from_box(sd):
    df = _box_df(9.0)
    df.loc[24, "High"] = 99.0  # current bar's high must not become the box top
    df.loc[24, "Low"] = 0.5
    r = sd.compute_darvas_box(df)
    assert r["box_top"] == 20.0 and r["box_bottom"] == 5.0


def test_darvas_nse_column_names(sd):
    df = _box_df(25.0, hi_col="CH_TRADE_HIGH_PRICE", lo_col="CH_TRADE_LOW_PRICE", cl_col="CH_CLOSING_PRICE")
    assert sd.compute_darvas_box(df)["signal"] == "BREAKOUT_BUY"
    df = _box_df(25.0, hi_col="DayHigh", lo_col="DayLow", cl_col="LAST")
    assert sd.compute_darvas_box(df)["signal"] == "BREAKOUT_BUY"


def test_darvas_no_box_and_bottom_fallbacks(sd):
    flat = pd.DataFrame({"High": [10.0] * 20, "Low": [8.0] * 20, "Close": [9.0] * 20})
    r = sd.compute_darvas_box(flat)
    assert r["signal"] == "NO_BOX" and r["box_top"] is None and "No confirmed" in r["note"]
    # peak exists but lows are flat -> no confirmed bottom -> min of segment
    df = _box_df(9.0, lows=[8.0] * 25)
    r = sd.compute_darvas_box(df)
    assert r["box_bottom"] == 8.0 and r["signal"] == "IN_BOX"
    # all-zero lows -> no valid bottom at all
    df = _box_df(9.0, lows=[0.0] * 25)
    r = sd.compute_darvas_box(df)
    assert r["signal"] == "NO_BOX" and r["box_top"] == 20.0 and r["box_bottom"] is None
    assert "box bottom" in r["note"]


def test_darvas_confirm_parameter_and_zero_range(sd):
    r = sd.compute_darvas_box(_box_df(9.0), confirm=2)
    assert r["confirm_days"] == 2 and r["box_top"] == 20.0
    # a tight box (top == bottom) reports position 0 rather than dividing by zero
    df = pd.DataFrame({"High": [10.0] * 3 + [12.0] + [11.0] * 10, "Low": [10.0] * 14, "Close": [10.0] * 14})
    df.loc[3, "Low"] = 12.0
    out = sd.compute_darvas_box(df)
    assert out["signal"] in ("IN_BOX", "BREAKOUT_BUY", "BREAKDOWN_SELL", "NO_BOX")


def test_display_darvas_box(sd, capsys):
    sd.display_darvas_box(sd.compute_darvas_box(_box_df(25.0)))
    out = capsys.readouterr().out
    assert "BREAKOUT BUY" in out and "Box Top" in out and "₹20.00" in out
    assert "Position in Box" in out and "133.3%" in out and "Data points used" in out
    sd.display_darvas_box({"signal": "NO_BOX", "box_top": None, "note": "nothing here"})
    out = capsys.readouterr().out
    assert "No confirmed Darvas box found" in out and "Note: nothing here" in out
    sd.display_darvas_box({"signal": "BREAKDOWN_SELL", "box_top": 5, "box_bottom": 4})
    assert "BREAKDOWN SELL" in capsys.readouterr().out
    sd.display_darvas_box({"signal": "IN_BOX"})
    assert "IN BOX" in capsys.readouterr().out
    sd.display_darvas_box({"signal": "WEIRD"})
    assert "Signal: WEIRD" in capsys.readouterr().out
    sd.display_darvas_box({"signal": "INSUFFICIENT_DATA"})
    assert "Insufficient OHLC data" in capsys.readouterr().out


# ------------------------------------------------------------------ financial helpers
def _stmt(rows, ncols=2):
    return pd.DataFrame(rows, index=list(range(ncols))).T


def test_row_helper(sd):
    df = pd.DataFrame({"2025": [10.0, None], "2024": [5.0, 3.0]}, index=["A", "B"])
    assert sd._row(df, "A") == 10.0
    assert sd._row(df, "A", col=1) == 5.0
    assert sd._row(df, "B") is None  # NaN
    assert sd._row(df, "Z", "A") == 10.0  # falls through to the next name
    assert sd._row(df, "A", col=5) is None  # IndexError swallowed
    assert sd._row(None, "A") is None and sd._row(pd.DataFrame(), "A") is None


def test_yf_financials_attribute_fallbacks_and_bug(sd):
    t = types.SimpleNamespace(income_stmt=None, financials="OLD", balance_sheet="BS", cash_flow=None, cashflow="CF")
    assert sd._yf_financials(t) == ("OLD", "BS", "CF")
    t = types.SimpleNamespace()
    assert sd._yf_financials(t) == (None, None, None)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="_yf_financials uses `getattr(...) or getattr(...)` on DataFrames: bool(DataFrame) "
    "raises ValueError, so real yfinance frames always fail (stock_daily_report_improved.py "
    "_yf_financials)",
)
def test_yf_financials_with_real_dataframes(sd):
    inc = pd.DataFrame({"a": [1]})
    t = types.SimpleNamespace(income_stmt=inc, balance_sheet=inc, cash_flow=inc)
    assert sd._yf_financials(t)[0] is inc


def _install_ticker(sd, monkeypatch, inc=None, bal=None, cf=None, info=None):
    monkeypatch.setattr(sd, "_yf_financials", lambda t: (inc, bal, cf))
    monkeypatch.setattr(sd.yf, "Ticker", lambda s: types.SimpleNamespace(info=info))


def _cols(*names):
    return pd.to_datetime(["2025-03-31", "2024-03-31"])[: len(names)]


def _frame(rows):
    return pd.DataFrame(rows).T.set_axis(pd.to_datetime(["2025-03-31", "2024-03-31"]), axis=1)


def _good_piotroski():
    inc = _frame({"Net Income": [40, 20], "Total Revenue": [300, 200], "Gross Profit": [150, 80]})
    bal = _frame(
        {
            "Total Assets": [100, 100], "Long Term Debt": [10, 30], "Current Assets": [250, 190],
            "Current Liabilities": [100, 100], "Share Issued": [90, 95],
        }
    )
    cf = _frame({"Operating Cash Flow": [60, 30]})
    return inc, bal, cf


def test_piotroski_perfect_score(sd, monkeypatch):
    inc, bal, cf = _good_piotroski()
    _install_ticker(sd, monkeypatch, inc, bal, cf, {})
    r = sd.compute_piotroski_score("ACME")
    assert r["f_score"] == 9 and r["symbol"] == "ACME"
    assert r["interpretation"].startswith("STRONG")
    assert all(v == 1 for v in r["component_scores"].values())
    d = r["details"]
    assert d["ROA_current_%"] == 40.0 and d["ROA_prev_%"] == 20.0
    assert d["OCF_current_Cr"] == round(60 / 1e7, 2)
    assert d["LTD_ratio_curr_%"] == 10.0 and d["CurrentRatio_curr"] == 2.5
    assert d["GrossMargin_curr_%"] == 50.0 and d["AssetTurnover_curr"] == 3.0


def test_piotroski_each_criterion_can_fail(sd, monkeypatch):
    inc, bal, cf = _good_piotroski()

    def score(mods):
        i, b, c = inc.copy(), bal.copy(), cf.copy()
        for (frame, row), vals in mods.items():
            {"inc": i, "bal": b, "cf": c}[frame].loc[row] = vals
        _install_ticker(sd, monkeypatch, i, b, c, {})
        return sd.compute_piotroski_score("ACME")["component_scores"]

    assert score({("inc", "Net Income"): [-5, 20]})["F1_ROA_positive"] == 0
    assert score({("cf", "Operating Cash Flow"): [-5, 30]})["F2_OCF_positive"] == 0
    assert score({("inc", "Net Income"): [10, 20]})["F3_ROA_improving"] == 0
    assert score({("cf", "Operating Cash Flow"): [20, 30]})["F4_Accruals"] == 0
    assert score({("bal", "Long Term Debt"): [40, 30]})["F5_Leverage_down"] == 0
    assert score({("bal", "Current Assets"): [100, 190]})["F6_CurrentRatio_up"] == 0
    assert score({("bal", "Share Issued"): [100, 95]})["F7_No_dilution"] == 0
    assert score({("inc", "Gross Profit"): [100, 80]})["F8_GrossMargin_up"] == 0
    assert score({("inc", "Total Revenue"): [100, 200]})["F9_AssetTurnover_up"] == 0
    # equal values are not an improvement (strict >/<), but equal shares are non-dilutive (<=)
    assert score({("bal", "Long Term Debt"): [30, 30]})["F5_Leverage_down"] == 0
    assert score({("bal", "Share Issued"): [95, 95]})["F7_No_dilution"] == 1


def test_piotroski_missing_data_and_errors(sd, monkeypatch):
    inc, bal, cf = _good_piotroski()
    # no balance sheet / cash flow: everything but dilution (assumed OK) fails
    _install_ticker(sd, monkeypatch, inc, None, None, {})
    r = sd.compute_piotroski_score("ACME")
    # only F7 (assumed non-dilutive) and F8 (gross margin comes from the income statement) score
    assert r["f_score"] == 2 and r["component_scores"]["F7_No_dilution"] == 1
    assert r["component_scores"]["F8_GrossMargin_up"] == 1
    assert r["interpretation"].startswith("WEAK") and r["details"]["ROA_current_%"] == "N/A"
    # share count falls back to info.sharesOutstanding for the current year
    _install_ticker(sd, monkeypatch, inc, bal.drop(index="Share Issued"), cf, {"sharesOutstanding": 80})
    assert sd.compute_piotroski_score("ACME")["component_scores"]["F7_No_dilution"] == 1
    # no income statement
    _install_ticker(sd, monkeypatch, None, bal, cf, {})
    assert "No financial data found for ACME.NS" in sd.compute_piotroski_score("ACME")["error"]
    _install_ticker(sd, monkeypatch, pd.DataFrame(), bal, cf, {})
    assert "error" in sd.compute_piotroski_score("ACME")

    def boom(t):
        raise RuntimeError("net")

    monkeypatch.setattr(sd, "_yf_financials", boom)
    assert sd.compute_piotroski_score("ACME")["error"] == "yfinance error: net"


def test_piotroski_interpretation_bands(sd, monkeypatch):
    inc, bal, cf = _good_piotroski()
    # F7 assumed + F1 + F2 + F4 => 4 (MODERATE lower bound)
    inc2 = inc.copy()
    inc2.loc["Net Income"] = [40, 50]  # F3 fails
    bal2 = bal.copy()
    bal2.loc["Long Term Debt"] = [40, 30]
    bal2.loc["Current Assets"] = [100, 190]
    bal2.loc["Share Issued"] = [100, 95]  # F5, F6, F7 fail
    inc2.loc["Gross Profit"] = [30, 160]
    inc2.loc["Total Revenue"] = [100, 400]  # F8, F9 fail
    _install_ticker(sd, monkeypatch, inc2, bal2, cf, {})
    r = sd.compute_piotroski_score("ACME")
    assert r["f_score"] == 3 and r["interpretation"].startswith("WEAK")
    bal2.loc["Share Issued"] = [90, 95]  # F7 passes -> 4
    _install_ticker(sd, monkeypatch, inc2, bal2, cf, {})
    r = sd.compute_piotroski_score("ACME")
    assert r["f_score"] == 4 and r["interpretation"].startswith("MODERATE")


def test_display_piotroski_score(sd, monkeypatch, capsys):
    inc, bal, cf = _good_piotroski()
    _install_ticker(sd, monkeypatch, inc, bal, cf, {})
    sd.display_piotroski_score(sd.compute_piotroski_score("ACME"))
    out = capsys.readouterr().out
    assert "9/9" in out and "\033[92m9/9" in out and out.count("✔") == 9 and "ROA current %" in out
    sd.display_piotroski_score({"error": "bad"})
    assert "bad" in capsys.readouterr().out
    base = {"interpretation": "x", "component_scores": {"F1_ROA_positive": 0}, "details": {}}
    sd.display_piotroski_score({**base, "f_score": 5})
    assert "\033[93m5/9" in capsys.readouterr().out  # amber for 4-6
    sd.display_piotroski_score({**base, "f_score": 3})
    out = capsys.readouterr().out
    assert "\033[91m3/9" in out and "✘" in out
    sd.display_piotroski_score({**base, "f_score": 7})
    assert "\033[92m7/9" in capsys.readouterr().out


# ------------------------------------------------------------------ coffee can
def _cc_frames(rev=(200, 150, 120, 100), ebit=(60, 40, 30, 25), ni=(30, 20, 15, 12)):
    cols = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31"])
    inc = pd.DataFrame({"Total Revenue": rev, "EBIT": ebit, "Net Income": ni}).T.set_axis(cols, axis=1)
    bal = pd.DataFrame(
        {"Total Assets": [200, 180, 160, 150], "Current Liabilities": [50, 40, 40, 30],
         "Long Term Debt": [20, 20, 20, 20], "Stockholders Equity": [100, 90, 80, 70]}
    ).T.set_axis(cols, axis=1)
    return inc, bal


def test_coffee_can_qualifies_and_details(sd, monkeypatch):
    inc, bal = _cc_frames()
    _install_ticker(sd, monkeypatch, inc, bal, None, {"debtToEquity": 45.2, "marketCap": 6e9})
    r = sd.compute_coffee_can("ACME")
    assert r["qualifies"] is True and r["score"] == "5/5" and r["symbol"] == "ACME"
    d = r["details"]
    assert d["Revenue_CAGR_%"] == round(((200 / 100) ** (1 / 3) - 1) * 100, 2)
    assert d["Revenue_years"] == 3 and d["ROCE_years"] == 4
    assert d["Debt_to_Equity"] == 0.45 and d["Market_Cap_Cr"] == 600.0
    assert d["Loss_years"] == 0 and d["Years_analysed"] == 4
    assert d["ROCE_min_%"] == round(min(60 / 150, 40 / 140, 30 / 120, 25 / 120) * 100, 2)


def test_coffee_can_criteria_boundaries(sd, monkeypatch):
    inc, bal = _cc_frames()

    def crit(info=None, inc=inc, bal=bal):
        _install_ticker(sd, monkeypatch, inc, bal, None, info if info is not None else {"debtToEquity": 50, "marketCap": 5e9})
        return sd.compute_coffee_can("ACME")["criteria"]

    assert crit()["C4_MCap_ge500Cr"] == 1  # exactly 500 Cr passes (>=)
    assert crit({"debtToEquity": 50, "marketCap": 4.99e9})["C4_MCap_ge500Cr"] == 0
    assert crit({"debtToEquity": 50})["C4_MCap_ge500Cr"] == 0
    assert crit({"debtToEquity": 100, "marketCap": 5e9})["C3_LowDebt"] == 0  # 1.0 is not < 1
    assert crit({"debtToEquity": 99, "marketCap": 5e9})["C3_LowDebt"] == 1
    assert crit({"debtToEquity": 0.9, "marketCap": 5e9})["C3_LowDebt"] == 1  # already a ratio
    assert crit({"debtToEquity": 10, "marketCap": 5e9})["C3_LowDebt"] == 0  # 10 is not > 10
    inc_loss = inc.copy()
    inc_loss.loc["Net Income"] = [30, 20, 0, 12]  # zero profit counts as a loss year
    assert crit(inc=inc_loss)["C5_NoProfitLoss"] == 0
    inc_slow = inc.copy()
    inc_slow.loc["Total Revenue"] = [105, 104, 102, 100]
    assert crit(inc=inc_slow)["C1_RevenuCAGR_gt10"] == 0
    inc_low = inc.copy()
    inc_low.loc["EBIT"] = [10, 8, 6, 5]
    assert crit(inc=inc_low)["C2_ROCE_gt15"] == 0


def test_coffee_can_debt_from_balance_sheet_and_missing_data(sd, monkeypatch):
    inc, bal = _cc_frames()
    _install_ticker(sd, monkeypatch, inc, bal, None, {"marketCap": 5e9})
    r = sd.compute_coffee_can("ACME")
    assert r["criteria"]["C3_LowDebt"] == 1 and r["details"]["Debt_to_Equity"] == 0.2
    _install_ticker(sd, monkeypatch, inc, bal.drop(index="Stockholders Equity"), None, {})
    r = sd.compute_coffee_can("ACME")
    assert r["criteria"]["C3_LowDebt"] == 0 and r["details"]["Debt_to_Equity"] == "N/A"
    assert r["criteria"]["C4_MCap_ge500Cr"] == 0 and r["details"]["Market_Cap_Cr"] == "N/A"
    assert r["qualifies"] is False
    # single year of revenue / no balance sheet / no net income row
    inc1 = inc.iloc[:, :1]
    _install_ticker(sd, monkeypatch, inc1.drop(index="Net Income"), None, None, {})
    r = sd.compute_coffee_can("ACME")
    assert r["criteria"]["C1_RevenuCAGR_gt10"] == 0 and r["details"]["Revenue_CAGR_%"] == "N/A"
    assert r["criteria"]["C2_ROCE_gt15"] == 0 and r["criteria"]["C5_NoProfitLoss"] == 0
    # non-positive oldest revenue -> no CAGR
    inc_z = inc.copy()
    inc_z.loc["Total Revenue"] = [100, 50, 20, 0]
    _install_ticker(sd, monkeypatch, inc_z, bal, None, {})
    assert sd.compute_coffee_can("ACME")["details"]["Revenue_CAGR_%"] == "N/A"


def test_coffee_can_errors(sd, monkeypatch):
    _install_ticker(sd, monkeypatch, None, None, None, {})
    assert "No data for ACME.NS" in sd.compute_coffee_can("ACME")["error"]
    monkeypatch.setattr(sd, "_yf_financials", lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    assert sd.compute_coffee_can("ACME")["error"] == "yfinance error: x"


def test_display_coffee_can(sd, monkeypatch, capsys):
    inc, bal = _cc_frames()
    _install_ticker(sd, monkeypatch, inc, bal, None, {"debtToEquity": 50, "marketCap": 6e9})
    sd.display_coffee_can(sd.compute_coffee_can("ACME"))
    out = capsys.readouterr().out
    assert "QUALIFIES" in out and "DOES NOT" not in out and "5/5" in out and out.count("✔") >= 6
    r = sd.compute_coffee_can("ACME")
    r["qualifies"] = False
    sd.display_coffee_can(r)
    assert "DOES NOT QUALIFY" in capsys.readouterr().out
    sd.display_coffee_can({"error": "oops"})
    assert "oops" in capsys.readouterr().out


# ------------------------------------------------------------------ display helpers
def test_display_nse_price(sd, capsys):
    eq = {
        "priceInfo": {
            "lastPrice": 100.5, "previousClose": 99.0, "change": 1.5, "pChange": 1.52, "open": 99.5,
            "intraDayHighLow": {"max": 101.0, "min": 98.0}, "vwap": 100.1,
            "weekHighLow": {"max": 120.0, "min": 80.0},
            "priceBand": {"upper": 110.0, "lower": 90.0},
        },
        "metadata": {"companyName": "Acme Ltd", "isin": "INE1", "series": "EQ", "industry": "Tech"},
        "securityInfo": {"faceValue": 10},
    }
    ti = {"trade_info": {"totalTradedVolume": 1234567, "totalTradedValue": 5e8}}
    sd.display_nse_price(eq, ti)
    out = capsys.readouterr().out
    assert "Acme Ltd" in out and "₹100.50" in out and "1,234,567" in out and "₹50.00 Cr" in out
    assert "₹110.00" in out and "₹120.00" in out and "▲ 1.52%" in out


def test_display_nse_price_fallback_keys_and_empty(sd, capsys):
    eq = {"priceInfo": {"ltp": 7, "prevClose": 6, "dayHigh": 8, "dayLow": 5, "totalTradedVolume": 99,
                        "totalTradedValue": 2e7}, "metadata": {"symbol": "SYM"}}
    sd.display_nse_price(eq, {"trade_info": {"priceBandHigh": 9, "priceBandLow": 4}})
    out = capsys.readouterr().out
    assert "SYM" in out and "₹7.00" in out and "₹8.00" in out and "₹9.00" in out and "₹2.00 Cr" in out
    sd.display_nse_price({}, {})
    out = capsys.readouterr().out
    assert "N/A" in out and "—" in out


def test_display_bse_price(sd, capsys):
    sd.display_bse_price({})
    assert "No BSE data available." in capsys.readouterr().out
    sd.display_bse_price(
        {"scrip_code": "500325", "quote": {"CurrRate": 10, "Open": 9, "DayHigh": 11, "DayLow": 8,
                                           "PrevClose": 9.5, "TotalVolume": 1000, "MktCap": 3e10}}
    )
    out = capsys.readouterr().out
    assert "500325" in out and "₹10.00" in out and "₹11.00" in out and "1,000" in out and "₹3,000.00 Cr" in out
    sd.display_bse_price({"quote": {"LastRate": 1}})
    out = capsys.readouterr().out
    assert "Market Cap" in out and "N/A" in out and "—" in out  # no scrip code / volume 0


def test_display_corporate_actions_and_meetings(sd, capsys):
    sd.display_corporate_actions(
        [{"exDate": "01-Jan", "purpose": "Dividend"}, {"recordDate": "02-Jan", "subject": "Split"}, {}],
        [{"Ex_Date": "03-Jan", "Purpose": "Bonus"}],
    )
    out = capsys.readouterr().out
    assert "01-Jan" in out and "Dividend" in out and "02-Jan" in out and "Split" in out
    assert "03-Jan" in out and "Bonus" in out and "• —" in out
    sd.display_corporate_actions([], [])
    assert capsys.readouterr().out.count("No upcoming actions found.") == 2
    sd.display_board_meetings([])
    assert "No upcoming board meetings." in capsys.readouterr().out
    sd.display_board_meetings([{"meetingDate": "05-Jan", "purpose": "Results"}, {"bm_date": "06-Jan", "bm_purpose": "Div"}])
    out = capsys.readouterr().out
    assert "05-Jan" in out and "Results" in out and "06-Jan" in out and "Div" in out


def test_display_option_chain(sd, capsys):
    sd.display_option_chain({})
    assert "Not an F&O stock" in capsys.readouterr().out
    sd.display_option_chain({"maxpain": 2500, "pcr": 1.25})
    out = capsys.readouterr().out
    assert "₹2,500.00" in out and "1.25" in out and "Bullish" in out
    sd.display_option_chain({"maxpain": 0, "pcr": 1.0})
    out = capsys.readouterr().out
    assert "Bearish (PCR ≤ 1)" in out and "N/A" in out  # maxpain 0 falsy
    sd.display_option_chain({"maxpain": 1})
    out = capsys.readouterr().out
    assert "N/A" in out and "Sentiment" not in out


def test_display_historical_summary(sd, capsys):
    sd.display_historical_summary([])
    assert "No historical data returned." in capsys.readouterr().out
    sd.display_historical_summary([{"CH_CLOSING_PRICE": 10}, {"close": "12"}, {"junk": 1}, {"CLOSE": "x", "LTP": 14}])
    out = capsys.readouterr().out
    assert "Days of data" in out and " 3" in out
    assert "₹14.00" in out and "₹10.00" in out and "₹12.00" in out  # high/low/avg
    assert "▲ 40.00%" in out
    sd.display_historical_summary([{"foo": 1}])
    out = capsys.readouterr().out
    assert "Insufficient closing-price data." in out and "['foo']" in out
    sd.display_historical_summary([{"close": 10}, {"close": 8}])
    assert "▼ 20.00%" in capsys.readouterr().out


def test_display_bulk_deals(sd, capsys):
    deals = [
        {"symbol": " tcs ", "clientname": "ACME", "quantity": "100", "price": 3500, "buysell": " BUY "},
        {"symbol": "INFY", "clientname": "Z", "quantity": "1", "price": 1, "buysell": "SELL"},
        {"symbol": "TCS"},
    ]
    sd.display_bulk_deals(deals, "tcs")
    out = capsys.readouterr().out
    assert out.count("shares @") == 2 and "ACME" in out and "₹3,500.00" in out
    assert "INFY" not in out and "BUY" in out and "—" in out
    sd.display_bulk_deals(deals, "WIPRO")
    assert "No bulk deals for WIPRO today." in capsys.readouterr().out


# ------------------------------------------------------------------ run / batch
def _scan_patches(sd, monkeypatch):
    monkeypatch.setattr(sd, "fetch_ohlc_history", lambda s, days=90: _box_df(25.0))
    monkeypatch.setattr(sd, "compute_piotroski_score", lambda s: {"symbol": s, "f_score": 8, "interpretation": "STRONG",
                                                                  "component_scores": {}, "details": {}})
    monkeypatch.setattr(sd, "compute_coffee_can", lambda s: {"symbol": s, "qualifies": True, "score": "5/5",
                                                              "criteria": {}, "details": {}})


def test_run_text_output_with_scans_and_fno(sd, monkeypatch, capsys):
    monkeypatch.setattr(sd.nspy, "get_bulkdeals", _bulk_frame)
    _scan_patches(sd, monkeypatch)
    data = sd.run(" tcs ", show_fno=True, output_format="text", run_scans=True)
    out = capsys.readouterr().out
    assert data["symbol"] == "TCS" and "timestamp" in data
    for hdr in ("NSE — LIVE PRICE", "BSE — LIVE QUOTE", "UPCOMING CORPORATE ACTIONS", "BOARD MEETINGS",
                "F&O — OPTION CHAIN SUMMARY", "RECENT HISTORICAL DATA", "TODAY'S BULK DEALS",
                "DARVAS BOX SCAN", "PIOTROSKI F-SCORE", "COFFEE CAN PORTFOLIO SCREEN"):
        assert hdr in out
    assert "Running quantitative scans" in out and "DAILY STOCK REPORT — TCS" in out
    assert "• SELL" in out  # bulk deal for TCS shown


def test_run_text_output_without_scans_or_fno(sd, capsys):
    sd.run("TCS", output_format="text")
    out = capsys.readouterr().out
    assert "F&O — OPTION CHAIN" not in out and "DARVAS" not in out and "Running quantitative" not in out
    assert "NSE — LIVE PRICE" in out


def test_run_json_output(sd, monkeypatch, capsys):
    monkeypatch.setattr(sd.nspy, "get_bulkdeals", _bulk_frame)
    _scan_patches(sd, monkeypatch)
    data = sd.run("tcs", show_fno=True, output_format="json", run_scans=True)
    files = list(sd.DOWNLOAD_DIR.glob("TCS_report_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["symbol"] == "TCS"
    assert "raw" not in saved["nse"]["optionChain"] and saved["nse"]["optionChain"]["pcr"] == 1.25
    assert [d["symbol"] for d in saved["nse"]["bulkDeals"]] == ["TCS"]
    assert saved["scans"]["piotroski"]["f_score"] == 8 and saved["scans"]["darvas"]["signal"] == "BREAKOUT_BUY"
    assert saved["bse"]["scrip_code"] == "500325"
    assert saved["nse"]["actions"] and len(saved["nse"]["actions"]) == 5
    assert "JSON saved" in capsys.readouterr().out
    assert data["scans"]["coffee_can"]["qualifies"] is True
    # without scans the scans block is empty and the oc raw payload is not fetched
    data2 = sd.run("tcs", output_format="json", run_scans=False)
    assert data2["scans"] == {} and data2["nse"]["optionChain"] == {}
    # json mode prints no sections
    assert "NSE — LIVE PRICE" not in capsys.readouterr().out


def test_run_batch_isolation_and_summary(sd, monkeypatch, capsys):
    calls = []

    def fake_run(sym, show_fno=False, output_format="json", run_scans=False):
        calls.append((sym, show_fno, output_format, run_scans))
        if sym == "BAD":
            raise RuntimeError("explode")
        return {
            "symbol": sym, "timestamp": "t",
            "nse": {"equityQuote": {"priceInfo": {"lastPrice": 10, "pChange": 1.0,
                                                   "intraDayHighLow": {"max": 11, "min": 9},
                                                   "weekHighLow": {"max": 12, "min": 8}}}},
            "scans": {"darvas": {"signal": "IN_BOX", "box_top": 5, "box_bottom": 3},
                      "piotroski": {"f_score": 6}, "coffee_can": {"qualifies": sym == "OK1", "score": "4/5"}},
        }

    monkeypatch.setattr(sd, "run", fake_run)
    res = sd.run_nifty50_batch(show_fno=True, output_format="json", run_scans=True, symbols=["OK1", "BAD", "OK2"])
    assert [r["symbol"] for r in res] == ["OK1", "OK2"]
    assert calls[0] == ("OK1", True, "json", True)
    out = capsys.readouterr().out
    assert "2 OK, 1 failed" in out and "Failed: BAD" in out and "[02/3] BAD" in out
    csv = list(sd.DOWNLOAD_DIR.glob("nifty50_scan_summary_*.csv"))
    assert len(csv) == 1
    df = pd.read_csv(csv[0])
    assert list(df["Symbol"]) == ["OK1", "OK2"] and list(df["CoffeeCan"]) == ["YES", "NO"]
    assert list(df["Darvas_Signal"]) == ["IN_BOX", "IN_BOX"] and list(df["Piotroski_Score"]) == [6, 6]
    assert list(df["LTP"]) == [10, 10] and list(df["52wHigh"]) == [12, 12] and list(df["DayLow"]) == [9, 9]


def test_run_batch_defaults_to_nifty50_and_plain_summary(sd, monkeypatch):
    seen = []

    def fake_run(sym, show_fno=False, output_format="json", run_scans=False):
        seen.append(sym)
        return {"symbol": sym, "nse": {"equityQuote": {"priceInfo": {"ltp": 1, "dayHigh": 2, "dayLow": 0.5}}}}

    monkeypatch.setattr(sd, "run", fake_run)
    sd.run_nifty50_batch()
    assert seen == sd.NIFTY_50_SYMBOLS and len(seen) == 50
    csv = list(sd.DOWNLOAD_DIR.glob("nifty50_summary_*.csv"))
    assert len(csv) == 1
    df = pd.read_csv(csv[0])
    assert "Darvas_Signal" not in df.columns and df["LTP"].iloc[0] == 1 and df["DayHigh"].iloc[0] == 2


def test_summary_csv_not_written_for_empty(sd):
    sd._write_summary_csv([])
    assert list(sd.DOWNLOAD_DIR.glob("*.csv")) == []


def test_run_scans_only(sd, monkeypatch, capsys):
    _scan_patches(sd, monkeypatch)
    out = sd.run_scans_only(" tcs ")
    assert set(out) == {"darvas", "piotroski", "coffee_can"}
    assert out["darvas"]["signal"] == "BREAKOUT_BUY"
    text = capsys.readouterr().out
    assert "SCANS — TCS" in text and "DARVAS BOX SCAN" in text and "COFFEE CAN" in text
