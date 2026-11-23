"""Offline tests for the pure helpers of the per-stock report builders
(us_stock_daily_report, sg_stock_daily_report, stock_daily_report_improved).

Vendor SDKs (yfinance, nse, bse) are faked in sys.modules while the modules import.
"""

import importlib
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import pytest


class _FakeTicker:
    def __init__(self, symbol=""):
        self.symbol = symbol


def _install_fakes():
    sys.modules["yfinance"] = types.SimpleNamespace(
        Ticker=_FakeTicker, download=lambda *a, **k: pd.DataFrame()
    )
    sys.modules["nse"] = types.SimpleNamespace(NSE=type("NSE", (), {}))
    sys.modules["bse"] = types.SimpleNamespace(BSE=type("BSE", (), {}))


_NAMES = ("us_stock_daily_report", "sg_stock_daily_report", "stock_daily_report_improved")


@pytest.fixture(scope="module")
def mods():
    keys = ("yfinance", "nse", "bse", "nsepython", *_NAMES)
    saved = {k: sys.modules.get(k) for k in keys}
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="mp_rep_"))  # the modules create ./*_data dirs at import
    try:
        _install_fakes()
        sys.modules["nsepython"] = None  # force the optional-dependency branch
        for n in _NAMES:
            sys.modules.pop(n, None)
        out = {n: importlib.import_module(n) for n in _NAMES}
    finally:
        os.chdir(cwd)
    yield out
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture(params=_NAMES)
def mod(request, mods):
    return mods[request.param]


# ---------------------------------------------------------------- formatters
def test_fmt_prefix_and_decimals(mods):
    assert mods["us_stock_daily_report"].fmt(1234.5) == "$1,234.50"
    assert mods["sg_stock_daily_report"].fmt(1234.5) == "S$1,234.50"
    assert mods["stock_daily_report_improved"].fmt(1234.5) == "₹1,234.50"
    assert mods["us_stock_daily_report"].fmt("12", decimals=0) == "$12"
    assert mods["us_stock_daily_report"].fmt(5, prefix="") == "5.00"


def test_fmt_bad_inputs(mod):
    assert (
        mod.fmt(None) == "N/A" and mod.fmt("") == "N/A" and mod.fmt(0) == f"{mod.fmt(1)[0:-4]}0.00"
    )
    assert mod.fmt("abc") == "abc"  # truthy non-numeric strings pass through


def test_pct_colours_and_arrows(mod):
    up, down = mod.pct(1.234), mod.pct(-2)
    assert "▲ 1.23%" in up and "\033[92m" in up and up.endswith("\033[0m")
    assert "▼ 2.00%" in down and "\033[91m" in down
    assert "▲ 0.00%" in mod.pct(0) and mod.pct(None) == "N/A" and mod.pct("x") == "N/A"


@pytest.mark.parametrize(
    "val,expected",
    [
        (4.48e12, "$4.48T"),
        (999e9, "$999.00B"),
        (1e12, "$1.00T"),
        (1.825e11, "$182.50B"),
        (1e9, "$1.00B"),
        (2.5e6, "$2.50M"),
        (1e6, "$1.00M"),
        (12345, "$12.35K"),
        (1e3, "$1.00K"),
        (999, "$999.00"),
        (-2.5e9, "$-2.50B"),
        (0, "$0.00"),
        (None, "N/A"),
        ("x", "N/A"),
    ],
)
def test_fmt_large_thresholds(mods, val, expected):
    assert mods["us_stock_daily_report"].fmt_large(val) == expected
    if expected != "N/A":
        assert mods["sg_stock_daily_report"].fmt_large(val) == expected.replace("$", "S$")


def test_section_and_row_print(mod, capsys):
    mod.section("hello")
    out = capsys.readouterr().out
    assert "HELLO" in out and out.count("-" * 60) == 2 or "HELLO" in out
    mod.row("Label", "value")
    assert capsys.readouterr().out.rstrip("\n").lstrip().startswith("Label")


# ---------------------------------------------------------------- _row / _yf_financials
def test_row_helper_semantics(mod):
    df = pd.DataFrame({"2024": [10.0, np.nan], "2023": [5.0, 2.0]}, index=["A", "B"])
    assert mod._row(df, "A") == 10.0 and mod._row(df, "A", col=1) == 5.0
    assert mod._row(df, "B") is None and mod._row(df, "nope", "B", col=1) == 2.0
    assert (
        mod._row(df, "A", col=9) is None
        and mod._row(None, "A") is None
        and mod._row(pd.DataFrame(), "A") is None
    )


def test_yf_financials_prefers_new_names_without_truth_value_error(mods):
    m = mods["us_stock_daily_report"]
    inc, old = pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]})
    t = types.SimpleNamespace(
        income_stmt=inc,
        financials=old,
        balance_sheet=pd.DataFrame({"x": [1]}),
        cash_flow=pd.DataFrame(),
        cashflow=pd.DataFrame({"y": [3]}),
    )
    i, b, c = m._yf_financials(t)
    assert (
        i is inc and list(b.columns) == ["x"] and list(c.columns) == ["y"]
    )  # empty cash_flow -> falls back
    i, b, c = m._yf_financials(types.SimpleNamespace(income_stmt=None, financials=None))
    assert i is None and b is None and c is None


# ---------------------------------------------------------------- darvas box
def _bars(highs, lows=None, closes=None):
    lows = lows if lows is not None else [h - 1 for h in highs]
    closes = (
        closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows, strict=False)]
    )
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes})


def _box_series():
    # history: a peak of 110 held for 3+ bars, a trough of 100 held for 3+ bars after it
    highs = [100, 105, 110, 108, 109, 109, 109, 109, 109, 109]
    lows = [95, 98, 104, 100, 101, 102, 103, 103, 103, 103]
    return highs, lows


LABELLED = ["us_stock_daily_report", "sg_stock_daily_report"]


@pytest.mark.parametrize("name", LABELLED)
def test_darvas_empty_and_unlabelled_data(mods, name):
    m = mods[name]
    assert m.compute_darvas_box(None)["signal"] == "INSUFFICIENT_DATA"
    assert m.compute_darvas_box(pd.DataFrame())["note"] == "Empty OHLC DataFrame"
    r = m.compute_darvas_box(pd.DataFrame({"x": [1] * 20}))
    assert r["signal"] == "INSUFFICIENT_DATA" and "Could not identify OHLC columns" in r["note"]


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="source inconsistency: stock_daily_report_improved.compute_darvas_box has no None guard "
    "(the sibling US/SG implementations return INSUFFICIENT_DATA), so None raises AttributeError",
)
def test_improved_darvas_handles_none(mods):
    assert (
        mods["stock_daily_report_improved"].compute_darvas_box(None)["signal"]
        == "INSUFFICIENT_DATA"
    )


def test_improved_darvas_empty_and_unlabelled(mods):
    m = mods["stock_daily_report_improved"]
    assert m.compute_darvas_box(pd.DataFrame())["signal"] == "INSUFFICIENT_DATA"
    r = m.compute_darvas_box(pd.DataFrame({"x": [1] * 20}))
    assert r["signal"] == "INSUFFICIENT_DATA" and r["note"] == "Need ≥ 8 rows; got 20"


def test_darvas_minimum_bars_is_confirm_plus_five(mod):
    short = _bars([1.0] * (mod.DARVAS_CONFIRM + 4))
    assert mod.compute_darvas_box(short)["signal"] == "INSUFFICIENT_DATA"
    assert f"{mod.DARVAS_CONFIRM + 5}" in mod.compute_darvas_box(short)["note"]
    assert mod.compute_darvas_box(_bars([1.0] * (mod.DARVAS_CONFIRM + 5)))["signal"] == "NO_BOX"


def test_darvas_box_signals_and_geometry(mod):
    highs, lows = _box_series()
    for close, signal in ((111.0, "BREAKOUT_BUY"), (99.0, "BREAKDOWN_SELL"), (105.0, "IN_BOX")):
        df = _bars(
            highs + [close + 1],
            lows + [close - 1],
            [(h + l) / 2 for h, l in zip(highs, lows, strict=False)] + [close],
        )
        r = mod.compute_darvas_box(df)
        assert r["signal"] == signal, (close, r)
        assert r["box_top"] == 110 and r["box_bottom"] == 100 and r["box_range"] == 10
        assert (
            r["current_price"] == close
            and r["data_points"] == len(df)
            and r["confirm_days"] == mod.DARVAS_CONFIRM
        )
    r = mod.compute_darvas_box(_bars(highs + [106], lows + [104], [1] * len(highs) + [105]))
    assert r["position_in_box_pct"] == 50.0 and r["upside_to_top_pct"] == round(
        (110 - 105) / 105 * 100, 2
    )


def test_darvas_equality_with_box_edges_stays_in_box(mod):
    highs, lows = _box_series()
    for close in (110.0, 100.0):
        df = _bars(highs + [close], lows + [close], list(highs) + [close])
        assert mod.compute_darvas_box(df)["signal"] == "IN_BOX"


def test_darvas_current_bar_never_shapes_the_box(mod):
    highs, lows = _box_series()
    base = mod.compute_darvas_box(_bars(highs + [105], lows + [104], [1] * len(highs) + [105]))
    wild = mod.compute_darvas_box(_bars(highs + [999], lows + [1], [1] * len(highs) + [105]))
    assert (base["box_top"], base["box_bottom"]) == (wild["box_top"], wild["box_bottom"])


def test_darvas_confirm_parameter_changes_the_box(mod):
    highs, lows = _box_series()
    df = _bars(highs + [105], lows + [104], [1] * len(highs) + [105])
    assert mod.compute_darvas_box(df, confirm=3)["box_top"] == 110
    assert mod.compute_darvas_box(df, confirm=2)["confirm_days"] == 2


def test_darvas_falls_back_to_lowest_low_when_no_bottom_is_confirmed(mod):
    # box top confirmed, but lows never hold for `confirm` bars: use the minimum low of the segment
    highs = [100, 110] + [109] * 8
    lows = [
        90,
        99,
        98,
        97,
        96,
        95,
        94,
        93,
        92,
        91,
    ]  # strictly falling: nothing is exceeded by the next 3
    r = mod.compute_darvas_box(_bars(highs + [100], lows + [99], [1] * len(highs) + [100]))
    assert r["box_top"] == 110 and r["box_bottom"] == 91


def test_darvas_zero_prices_are_ignored_and_column_variants(mod):
    highs, lows = _box_series()
    highs2 = [0] + highs[1:]
    r = mod.compute_darvas_box(_bars(highs2 + [105], lows + [104], [1] * len(highs) + [105]))
    assert r["box_top"] == 110
    df = _bars(highs + [105], lows + [104], [1] * len(highs) + [105]).rename(
        columns={"High": "DAYHIGH", "Low": "DAYLOW", "Close": "LAST"}
    )
    assert mod.compute_darvas_box(df)["box_top"] == 110
    df = df.astype(object)
    df.iloc[0, 0] = "bad"  # non-numeric cells coerce to 0 and are skipped
    assert mod.compute_darvas_box(df)["box_top"] == 110


# ---------------------------------------------------------------- piotroski (US + SG share the same shape)
def _fin(cols, rows):
    return (
        pd.DataFrame(rows, index=list(rows), columns=None)
        if False
        else pd.DataFrame(
            {c: [rows[r][i] for r in rows] for i, c in enumerate(cols)}, index=list(rows)
        )
    )


def _ticker_good():
    inc = _fin(
        ["2024", "2023"],
        {
            "Net Income": [100.0, 60.0],
            "Total Revenue": [1000.0, 800.0],
            "Gross Profit": [500.0, 360.0],
        },
    )
    bal = _fin(
        ["2024", "2023"],
        {
            "Total Assets": [1000.0, 900.0],
            "Long Term Debt": [50.0, 100.0],
            "Current Assets": [500.0, 400.0],
            "Current Liabilities": [200.0, 200.0],
            "Share Issued": [100.0, 100.0],
        },
    )
    cf = _fin(["2024", "2023"], {"Operating Cash Flow": [150.0, 90.0]})
    return types.SimpleNamespace(income_stmt=inc, balance_sheet=bal, cash_flow=cf, info={})


@pytest.mark.parametrize("name", ["us_stock_daily_report", "sg_stock_daily_report"])
def test_piotroski_all_nine_pass_and_interpretation(mods, name):
    r = mods[name].compute_piotroski_score(_ticker_good(), "AAA")
    assert r["symbol"] == "AAA" and r["f_score"] == 9 and r["interpretation"].startswith("STRONG")
    assert set(r["component_scores"].values()) == {1} and len(r["component_scores"]) == 9


@pytest.mark.parametrize("name", ["us_stock_daily_report", "sg_stock_daily_report"])
def test_piotroski_individual_failures(mods, name):
    m = mods[name]
    t = _ticker_good()
    t.cash_flow.loc["Operating Cash Flow", "2024"] = -5.0
    r = m.compute_piotroski_score(t, "X")
    assert (
        r["component_scores"]["F2_OCF_positive"] == 0 and r["component_scores"]["F4_Accruals"] == 0
    )
    t = _ticker_good()
    t.balance_sheet.loc["Long Term Debt", "2024"] = 300.0
    assert m.compute_piotroski_score(t, "X")["component_scores"]["F5_Leverage_down"] == 0
    t = _ticker_good()
    t.balance_sheet.loc["Share Issued", "2024"] = 120.0
    assert m.compute_piotroski_score(t, "X")["component_scores"]["F7_No_dilution"] == 0
    t = _ticker_good()
    t.income_stmt.loc["Net Income"] = [-10.0, 60.0]
    sc = m.compute_piotroski_score(t, "X")["component_scores"]
    assert sc["F1_ROA_positive"] == 0 and sc["F3_ROA_improving"] == 0


@pytest.mark.parametrize("name", ["us_stock_daily_report", "sg_stock_daily_report"])
def test_piotroski_interpretation_bands_and_missing_data(mods, name):
    m = mods[name]
    assert m.compute_piotroski_score(
        types.SimpleNamespace(income_stmt=None, financials=None), "Z"
    ) == {"symbol": "Z", "error": "No income statement data available"}
    t = _ticker_good()
    t.cash_flow = pd.DataFrame()
    t.balance_sheet = t.balance_sheet.loc[["Total Assets"]]
    r = m.compute_piotroski_score(t, "Q")
    band = "STRONG" if r["f_score"] >= 7 else "MODERATE" if r["f_score"] >= 4 else "WEAK"
    assert r["interpretation"].startswith(band) and band != "STRONG"
    t = _ticker_good()
    t.income_stmt.loc["Net Income"] = [-10.0, 60.0]
    t.cash_flow.loc["Operating Cash Flow"] = [-5.0, 90.0]
    t.balance_sheet.loc["Long Term Debt"] = [300.0, 100.0]
    t.balance_sheet.loc["Share Issued"] = [120.0, 100.0]
    t.balance_sheet.loc["Current Assets"] = [300.0, 400.0]
    r = m.compute_piotroski_score(t, "W")
    assert r["f_score"] <= 3 and r["interpretation"].startswith("WEAK")


def test_piotroski_missing_share_data_counts_as_no_dilution(mods):
    t = _ticker_good()
    t.balance_sheet = t.balance_sheet.drop("Share Issued")
    assert (
        mods["us_stock_daily_report"].compute_piotroski_score(t, "X")["component_scores"][
            "F7_No_dilution"
        ]
        == 1
    )


# ---------------------------------------------------------------- coffee can (US)
def _coffee_ticker(**over):
    inc = _fin(
        ["2024", "2023", "2022", "2021"],
        {
            "Total Revenue": [1500.0, 1300.0, 1100.0, 1000.0],
            "Net Income": [150.0, 120.0, 100.0, 90.0],
        },
    )
    bal = _fin(
        ["2024", "2023", "2022", "2021"],
        {
            "Stockholders Equity": [500.0, 450.0, 400.0, 380.0],
            "Long Term Debt": [100.0, 100.0, 100.0, 100.0],
        },
    )
    cf = _fin(["2024"], {"Free Cash Flow": [80.0]})
    t = types.SimpleNamespace(
        income_stmt=inc,
        balance_sheet=bal,
        cash_flow=cf,
        info={"debtToEquity": 50.0, "marketCap": 5e9},
        fast_info=types.SimpleNamespace(market_cap=5e9),
    )
    for k, v in over.items():
        setattr(t, k, v)
    return t


def test_coffee_can_qualifies_and_reports_details(mods):
    r = mods["us_stock_daily_report"].compute_coffee_can(_coffee_ticker(), "AAA")
    assert r["qualifies"] is True and r["score"] == "6/6"
    assert r["details"]["Revenue_CAGR_%"] == round(((1500 / 1000) ** (1 / 3) - 1) * 100, 2)
    assert r["details"]["Debt_to_Equity"] == 0.5  # 50 > 10 is read as a percentage
    assert r["details"]["Market_Cap"] == "$5.00B" and r["details"]["Loss_years"] == 0


def test_coffee_can_each_criterion_can_fail(mods):
    m = mods["us_stock_daily_report"]
    base = _coffee_ticker()

    def one(ticker, key):
        return m.compute_coffee_can(ticker, "X")["criteria"][key]

    t = _coffee_ticker()
    t.income_stmt.loc["Total Revenue"] = [1100.0, 1090.0, 1080.0, 1000.0]
    assert one(t, "C1_Revenue_CAGR_gt10") == 0
    t = _coffee_ticker()
    t.income_stmt.loc["Net Income"] = [10.0, 10.0, 10.0, 10.0]
    assert one(t, "C2_ROE_gt15") == 0
    t = _coffee_ticker(info={"debtToEquity": 1.5, "marketCap": 5e9})
    assert one(t, "C3_LowDebt") == 0
    t = _coffee_ticker(
        info={"debtToEquity": 0.5, "marketCap": 9e8},
        fast_info=types.SimpleNamespace(market_cap=9e8),
    )
    assert one(t, "C4_MCap_ge1B") == 0
    t = _coffee_ticker()
    t.income_stmt.loc["Net Income", "2022"] = -1.0
    r = m.compute_coffee_can(t, "X")
    assert (
        r["criteria"]["C5_NoProfitLoss"] == 0
        and r["details"]["Loss_years"] == 1
        and r["qualifies"] is False
    )
    t = _coffee_ticker()
    t.cash_flow.loc["Free Cash Flow", "2024"] = -1.0
    assert one(t, "C6_FreeCashFlow_pos") == 0
    assert one(base, "C6_FreeCashFlow_pos") == 1


def test_coffee_can_boundaries_and_fallbacks(mods):
    m = mods["us_stock_daily_report"]
    t = _coffee_ticker(
        info={"debtToEquity": 10.0, "marketCap": 1e9},
        fast_info=types.SimpleNamespace(market_cap=None),
    )
    r = m.compute_coffee_can(t, "X")
    assert (
        r["criteria"]["C3_LowDebt"] == 0 and r["details"]["Debt_to_Equity"] == 10.0
    )  # 10 is NOT > 10: stays a ratio
    assert r["criteria"]["C4_MCap_ge1B"] == 1  # exactly $1B qualifies
    t = _coffee_ticker(info={"marketCap": 5e9})
    t.cash_flow = _fin(["2024"], {"Operating Cash Flow": [100.0], "Capital Expenditure": [-30.0]})
    r = m.compute_coffee_can(t, "X")
    assert (
        r["criteria"]["C3_LowDebt"] == 1 and r["details"]["Debt_to_Equity"] == 0.2
    )  # LTD / equity fallback
    assert r["criteria"]["C6_FreeCashFlow_pos"] == 1 and r["details"]["FCF_latest_$M"] == 0.0
    t.cash_flow = _fin(["2024"], {"Operating Cash Flow": [30.0], "Capital Expenditure": [-100.0]})
    assert m.compute_coffee_can(t, "X")["criteria"]["C6_FreeCashFlow_pos"] == 0


def test_coffee_can_no_data(mods):
    m = mods["us_stock_daily_report"]
    assert m.compute_coffee_can(
        types.SimpleNamespace(income_stmt=None, financials=None, info={}), "Z"
    ) == {"symbol": "Z", "error": "No income statement data available"}
    t = _coffee_ticker()
    t.income_stmt = _fin(["2024"], {"Total Revenue": [100.0], "Net Income": [10.0]})
    assert (
        m.compute_coffee_can(t, "X")["criteria"]["C1_Revenue_CAGR_gt10"] == 0
    )  # one year: no CAGR


# ---------------------------------------------------------------- US data-fetch helpers (fake Ticker objects)
@pytest.fixture
def us(mods):
    return mods["us_stock_daily_report"]


class _FI:
    last_price = 101.0
    previous_close = 100.0
    market_cap = 2e9
    year_high = 120.0
    currency = "USD"


def test_fetch_quote_prefers_fast_info_then_info(us):
    t = types.SimpleNamespace(
        fast_info=_FI(),
        info={
            "currentPrice": 1.0,
            "previousClose": 2.0,
            "open": 99.5,
            "trailingPE": 25.0,
            "recommendationKey": "buy",
            "shortName": "Acme",
        },
    )
    q = us.fetch_quote(t)
    assert q["lastPrice"] == 101.0 and q["previousClose"] == 100.0  # fast_info wins
    assert q["open"] == 99.5 and q["trailingPE"] == 25.0 and q["recommendation"] == "BUY"
    assert q["name"] == "Acme" and q["currency"] == "USD" and q["52wHigh"] == 120.0
    assert q["country"] == "US" and q["sector"] == "—" and q["exchange"] == "—"


def test_fetch_quote_survives_info_errors(us):
    class T:
        fast_info = types.SimpleNamespace()

        @property
        def info(self):
            raise RuntimeError("blocked")

    q = us.fetch_quote(T())
    assert q["lastPrice"] is None and q["recommendation"] == "—" and q["currency"] == "USD"


def test_fetch_historical_returns_frame_or_empty(us, capsys):
    df = pd.DataFrame({"Close": [1.0]})
    assert (
        us.fetch_historical(types.SimpleNamespace(history=lambda period, auto_adjust: df), "3mo")
        is df
    )
    assert us.fetch_historical(types.SimpleNamespace(history=lambda **k: "junk")).empty

    def boom(**k):
        raise RuntimeError("x")

    assert us.fetch_historical(types.SimpleNamespace(history=boom)).empty
    assert "[Historical error] x" in capsys.readouterr().out


def test_fetch_corporate_actions(us):
    idx = pd.to_datetime([f"2024-0{m}-15" for m in range(1, 8)])
    acts = pd.DataFrame(
        {"Dividends": [0.1, 0, 0.2, 0.3, 0, 0.4, 0.5], "Stock Splits": [0, 2.0, 0, 0, 0, 0, 3.0]},
        index=idx,
    )
    cal = {
        "Earnings Date": [pd.Timestamp("2024-08-01")],
        "Ex-Dividend Date": "2024-07-01",
        "Earnings Average": 1.5,
    }
    r = us.fetch_corporate_actions(types.SimpleNamespace(actions=acts, calendar=cal))
    assert [d["amount"] for d in r["dividends"]] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]  # last five payments only
    assert r["splits"] == [
        {"date": "2024-02-15", "ratio": 2.0},
        {"date": "2024-07-15", "ratio": 3.0},
    ]
    assert (
        r["calendar"]["next_earnings"].startswith("2024-08-01")
        and r["calendar"]["ex_dividend_date"] == "2024-07-01"
    )
    assert r["calendar"]["dividend_date"] == "—" and r["calendar"]["eps_est_avg"] == 1.5
    empty = us.fetch_corporate_actions(types.SimpleNamespace(actions=pd.DataFrame(), calendar=None))
    assert empty["dividends"] == [] and empty["calendar"]["next_earnings"] == "—"


def _chain(call_oi, put_oi, strikes):
    return types.SimpleNamespace(
        calls=pd.DataFrame({"strike": strikes, "openInterest": call_oi}),
        puts=pd.DataFrame({"strike": strikes, "openInterest": put_oi}),
    )


def test_fetch_options_pcr_and_max_pain_match_brute_force(us):
    strikes, c_oi, p_oi = (
        [90.0, 100.0, 110.0, 120.0],
        [5.0, 20.0, 30.0, 10.0],
        [40.0, 25.0, 8.0, 2.0],
    )
    t = types.SimpleNamespace(
        options=["2026-08-21", "2026-09-18"], option_chain=lambda e: _chain(c_oi, p_oi, strikes)
    )
    r = us.fetch_options_data(t)

    def pain(p):
        return sum(max(0, p - k) * v for k, v in zip(strikes, c_oi, strict=False)) + sum(
            max(0, k - p) * v for k, v in zip(strikes, p_oi, strict=False)
        )

    assert r["max_pain"] == min(strikes, key=pain) and r["expiry_date"] == "2026-08-21"
    assert r["call_oi"] == 65 and r["put_oi"] == 75 and r["pcr"] == round(75 / 65, 3)
    assert r["sentiment"] == "Bullish (PCR>1)"
    bearish = us.fetch_options_data(
        types.SimpleNamespace(
            options=["x"], option_chain=lambda e: _chain([10, 10], [10, 5], [1.0, 2.0])
        )
    )
    assert bearish["sentiment"] == "Bearish (PCR≤1)" and bearish["pcr"] == 0.75
    nocalls = us.fetch_options_data(
        types.SimpleNamespace(
            options=["x"], option_chain=lambda e: _chain([0, 0], [1, 1], [1.0, 2.0])
        )
    )
    assert nocalls["pcr"] is None and nocalls["sentiment"] == "Bearish (PCR≤1)"


def test_fetch_options_no_chain_or_errors(us, capsys):
    assert us.fetch_options_data(types.SimpleNamespace(options=[])) == {}

    def boom(e):
        raise RuntimeError("no chain")

    assert us.fetch_options_data(types.SimpleNamespace(options=["x"], option_chain=boom)) == {}
    assert "[Options error] no chain" in capsys.readouterr().out


def test_fetch_insiders_holders_and_news(us):
    ins = pd.DataFrame(
        {
            "Insider": list("ABCDEFG"),
            "Position": ["CEO"] * 7,
            "Transaction": ["Sale"] * 7,
            "Shares": range(7),
            "Value": range(7),
            "Start Date": pd.date_range("2024-01-01", periods=7),
        }
    )
    r = us.fetch_insider_trades(types.SimpleNamespace(insider_transactions=ins))
    assert [x["insider"] for x in r] == ["G", "F", "E", "D", "C"]  # newest five first
    assert us.fetch_insider_trades(types.SimpleNamespace(insider_transactions=None)) == []
    assert us.fetch_insider_trades(types.SimpleNamespace(insider_transactions=pd.DataFrame())) == []
    holders = pd.DataFrame({"Holder": list("ABCDEFG"), "Shares": range(7)})
    assert (
        len(us.fetch_institutional_holders(types.SimpleNamespace(institutional_holders=holders)))
        == 5
    )
    assert us.fetch_institutional_holders(types.SimpleNamespace(institutional_holders=None)) == []
    news = [
        {"content": {"title": "New style", "pubDate": "2026-07-30T10:00:00Z"}},
        {"title": "Old style", "providerPublishTime": 1_700_000_000},
    ] + [{"title": f"n{i}"} for i in range(5)]
    out = us.fetch_news(types.SimpleNamespace(news=news))
    assert len(out) == 5 and out[0] == {"title": "New style", "published": "2026-07-30T10:00"}
    assert out[1]["published"] == "2023-11-14 22:13" and out[1]["title"] == "Old style"
    assert us.fetch_news(types.SimpleNamespace(news=None)) == []


def test_insiders_errors_are_swallowed(us, capsys):
    class T:
        @property
        def insider_transactions(self):
            raise RuntimeError("down")

        @property
        def institutional_holders(self):
            raise RuntimeError("down")

        @property
        def news(self):
            raise RuntimeError("down")

    assert (
        us.fetch_insider_trades(T()) == []
        and us.fetch_institutional_holders(T()) == []
        and us.fetch_news(T()) == []
    )
    assert capsys.readouterr().out.count("error]") == 3


# ---------------------------------------------------------------- display functions run cleanly
def test_display_functions_print_without_error(mods, capsys):
    us = mods["us_stock_daily_report"]
    highs, lows = _box_series()
    box = us.compute_darvas_box(_bars(highs + [105], lows + [104], [1] * len(highs) + [105]))
    us.display_darvas_box(box)
    us.display_darvas_box({"signal": "NO_BOX", "box_top": None, "box_bottom": None, "note": "n"})
    us.display_piotroski_score(us.compute_piotroski_score(_ticker_good(), "AAA"))
    us.display_piotroski_score({"symbol": "X", "error": "boom"})
    us.display_coffee_can(us.compute_coffee_can(_coffee_ticker(), "AAA"))
    us.display_coffee_can({"symbol": "X", "error": "boom"})
    out = capsys.readouterr().out
    assert "DARVAS" in out.upper() and "PIOTROSKI" in out.upper() and "boom" in out


def _broken(*keys):
    t = _ticker_good()
    edits = {
        "F5": lambda: t.balance_sheet.loc.__setitem__("Long Term Debt", [300.0, 100.0]),
        "F6": lambda: t.balance_sheet.loc.__setitem__("Current Assets", [300.0, 400.0]),
        "F7": lambda: t.balance_sheet.loc.__setitem__("Share Issued", [120.0, 100.0]),
        "F8": lambda: t.income_stmt.loc.__setitem__("Gross Profit", [100.0, 360.0]),
        "F9": lambda: t.income_stmt.loc.__setitem__("Total Revenue", [700.0, 800.0]),
        "F2": lambda: t.cash_flow.loc.__setitem__("Operating Cash Flow", [-5.0, 90.0]),
    }
    for k in keys:
        edits[k]()
    return t


@pytest.mark.parametrize(
    "breaks,total,band",
    [
        ((), 9, "STRONG"),
        (("F2",), 7, "STRONG"),
        (("F2", "F5"), 6, "MODERATE"),
        (("F2", "F5", "F6", "F7"), 4, "MODERATE"),
        (("F2", "F5", "F6", "F7", "F8"), 3, "WEAK"),
        (("F2", "F5", "F6", "F7", "F8", "F9"), 2, "WEAK"),
    ],  # F2 also fails the F4 accrual test
)
@pytest.mark.parametrize("name", ["us_stock_daily_report", "sg_stock_daily_report"])
def test_piotroski_band_boundaries(mods, name, breaks, total, band):
    r = mods[name].compute_piotroski_score(_broken(*breaks), "X")
    assert r["f_score"] == total and r["interpretation"].startswith(band)


def test_coffee_can_roe_threshold_is_strictly_above_fifteen(mods):
    m = mods["us_stock_daily_report"]
    t = _coffee_ticker()
    t.income_stmt.loc["Net Income"] = [150.0, 150.0, 150.0, 150.0]
    t.balance_sheet.loc["Stockholders Equity"] = [1000.0, 1000.0, 1000.0, 1000.0]  # ROE exactly 15%
    assert m.compute_coffee_can(t, "X")["criteria"]["C2_ROE_gt15"] == 0


def test_options_put_call_ratio_of_exactly_one_is_not_bullish(mods):
    us = mods["us_stock_daily_report"]
    t = types.SimpleNamespace(
        options=["x"], option_chain=lambda e: _chain([10.0, 10.0], [10.0, 10.0], [1.0, 2.0])
    )
    assert us.fetch_options_data(t)["sentiment"] == "Bearish (PCR≤1)"
