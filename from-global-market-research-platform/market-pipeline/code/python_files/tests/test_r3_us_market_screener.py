"""Offline tests for us_market_screener (universe, Darvas batch, Piotroski, Coffee Can, SQLite, CSVs, main)."""

import importlib
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

NASDAQ_TXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc.|Q|N|N|100|N|N\n"
    "TSTX|Test Co|Q|Y|N|100|N|N\n"
    "ETFX|Some ETF|Q|N|N|100|Y|N\n"
    "BADF|Deficient Co|Q|N|D|100|N|N\n"
    "BRK-A|Berkshire|Q|N|N|100|N|N\n"
    "TOOLONG|Long Symbol|Q|N|N|100|N|N\n"
    "MSFT|Microsoft|Q|N|N|100|N|N\n"
    "File Creation Time: 1122202610:00||||||\n"
)
OTHER_TXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "IBM|IBM Corp|N|IBM|N|100|N|IBM\n"
    "SPY|SPDR|P|SPY|Y|100|N|SPY\n"
    "XYZ|Amex Co|A|XYZ|N|100|N|XYZ\n"
    "AAPL|Dup Apple|N|AAPL|N|100|N|AAPL\n"
    "QQQ|Weird Exchange|Q|QQQ|N|100|N|QQQ\n"
    "TST|Test|N|TST|N|100|Y|TST\n"
    "File Creation Time: 1122202610:00|||||||\n"
)


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def us(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    yf = types.ModuleType("yfinance")
    yf.download = lambda *a, **k: pd.DataFrame()
    yf.Ticker = lambda s: types.SimpleNamespace(info={})
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    monkeypatch.setitem(sys.modules, "tqdm", None)  # TQDM_OK False unless a test installs one
    sys.modules.pop("us_market_screener", None)
    mod = importlib.import_module("us_market_screener")
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "us_screener_output")
    monkeypatch.setattr(mod, "DB_FILE", tmp_path / "us_screener_output" / "screener.db")
    yield mod
    sys.modules.pop("us_market_screener", None)


def _frame(rows, ncols=None):
    n = len(next(iter(rows.values())))
    cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"][:n])
    return pd.DataFrame(rows).T.set_axis(cols, axis=1)


# ------------------------------------------------------------------ import-time tip
def test_tqdm_missing_prints_tip_and_flag(us):
    assert us.TQDM_OK is False


def test_tqdm_present_flag(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = types.ModuleType("tqdm")
    fake.tqdm = lambda **k: None
    monkeypatch.setitem(sys.modules, "tqdm", fake)
    yf = types.ModuleType("yfinance")
    yf.Ticker = object
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    sys.modules.pop("us_market_screener", None)
    mod = importlib.import_module("us_market_screener")
    assert mod.TQDM_OK is True
    sys.modules.pop("us_market_screener", None)


# ------------------------------------------------------------------ universe
def _install_get(us, monkeypatch, seen=None):
    def get(url, timeout=None):
        if seen is not None:
            seen.append((url, timeout))
        return _Resp(NASDAQ_TXT if url == us.NASDAQ_URL else OTHER_TXT)

    monkeypatch.setattr(us.requests, "get", get)


def test_universe_all_filters_and_dedupes(us, monkeypatch, capsys):
    seen = []
    _install_get(us, monkeypatch, seen)
    df = us.fetch_stock_universe("all")
    assert list(df.columns) == ["Symbol", "Name", "Exchange"]
    # test issues, ETFs, deficient, hyphenated and >5 char symbols removed
    assert list(df["Symbol"]) == ["AAPL", "MSFT", "IBM", "XYZ", "QQQ"]
    got = dict(zip(df["Symbol"], df["Exchange"]))
    assert got == {"AAPL": "NASDAQ", "MSFT": "NASDAQ", "IBM": "NYSE", "XYZ": "AMEX", "QQQ": "OTHER"}
    assert df.loc[df["Symbol"] == "AAPL", "Name"].iloc[0] == "Apple Inc."  # first (NASDAQ) copy wins
    assert [u for u, _ in seen] == [us.NASDAQ_URL, us.OTHER_URL] and all(t == 15 for _, t in seen)
    assert list(df.index) == list(range(5))
    assert "Fetching NASDAQ" in capsys.readouterr().out


def test_universe_exchange_filter(us, monkeypatch):
    seen = []
    _install_get(us, monkeypatch, seen)
    df = us.fetch_stock_universe("nasdaq")
    assert list(df["Symbol"]) == ["AAPL", "MSFT"] and [u for u, _ in seen] == [us.NASDAQ_URL]
    seen.clear()
    df = us.fetch_stock_universe("nyse")
    assert [u for u, _ in seen] == [us.OTHER_URL] and "AAPL" in set(df["Symbol"])
    for ex in ("amex", "other"):
        seen.clear()
        us.fetch_stock_universe(ex)
        assert [u for u, _ in seen] == [us.OTHER_URL]


# ------------------------------------------------------------------ darvas
def _ohlc(close_last, n=25, peak=10, lows=None):
    highs = [10.0] * n
    highs[min(peak, n - 5)] = 20.0
    lows_ = list(lows) if lows is not None else [8.0] * n
    if lows is None:
        lows_[min(12, n - 3)] = 5.0
    closes = [9.0] * n
    closes[-1] = close_last
    return pd.DataFrame({"High": highs, "Low": lows_, "Close": closes})


def test_darvas_signals_and_numbers(us):
    r = us._darvas_from_df(_ohlc(25.0))
    assert r == {"signal": "BREAKOUT_BUY", "box_top": 20.0, "box_bottom": 5.0, "current_price": 25.0,
                 "upside_pct": -20.0, "pos_in_box": 133.3}
    assert us._darvas_from_df(_ohlc(3.0))["signal"] == "BREAKDOWN_SELL"
    r = us._darvas_from_df(_ohlc(10.0))
    assert r["signal"] == "IN_BOX" and r["upside_pct"] == 100.0 and r["pos_in_box"] == 33.3
    assert us._darvas_from_df(_ohlc(20.0))["signal"] == "IN_BOX"
    assert us._darvas_from_df(_ohlc(5.0))["signal"] == "IN_BOX"
    assert us._darvas_from_df(_ohlc(20.01))["signal"] == "BREAKOUT_BUY"
    assert us._darvas_from_df(_ohlc(4.99))["signal"] == "BREAKDOWN_SELL"


def test_darvas_guards(us):
    assert us._darvas_from_df(None)["signal"] == "INSUFFICIENT_DATA"
    assert us._darvas_from_df(pd.DataFrame())["signal"] == "INSUFFICIENT_DATA"
    assert us._darvas_from_df(_ohlc(9, n=7))["signal"] == "INSUFFICIENT_DATA"  # needs confirm + 5 = 8
    assert us._darvas_from_df(_ohlc(9, n=8))["signal"] != "INSUFFICIENT_DATA"
    assert us._darvas_from_df(_ohlc(9, n=9), confirm=4)["signal"] != "INSUFFICIENT_DATA"
    assert us._darvas_from_df(_ohlc(9, n=8), confirm=4)["signal"] == "INSUFFICIENT_DATA"
    no_close = pd.DataFrame({"High": range(20), "Low": range(20)})
    assert us._darvas_from_df(no_close)["signal"] == "INSUFFICIENT_DATA"
    no_high = pd.DataFrame({"Low": range(20), "Close": range(20)})
    assert us._darvas_from_df(no_high)["signal"] == "INSUFFICIENT_DATA"
    all_nan = _ohlc(9)
    all_nan["Close"] = np.nan
    r = us._darvas_from_df(all_nan)
    assert r["signal"] == "NO_DATA" and r["current_price"] is None


def test_darvas_trims_trailing_nan_and_rejects_nonpositive_price(us):
    df = _ohlc(25.0, n=27)
    df.loc[25:, "Close"] = np.nan  # two stale trailing rows: last valid close (bar 24) is the "current"
    df.loc[24, "Close"] = 25.0
    r = us._darvas_from_df(df)
    assert r["current_price"] == 25.0
    df = _ohlc(0.0)
    assert us._darvas_from_df(df)["signal"] == "NO_DATA"
    df = _ohlc(-1.0)
    assert us._darvas_from_df(df)["signal"] == "NO_DATA"


def test_darvas_no_box_and_bottom_fallbacks(us):
    flat = pd.DataFrame({"High": [10.0] * 20, "Low": [8.0] * 20, "Close": [9.0] * 20})
    r = us._darvas_from_df(flat)
    assert r == {"signal": "NO_BOX", "box_top": None, "box_bottom": None}
    assert us._darvas_from_df(_ohlc(9.0, lows=[8.0] * 25))["box_bottom"] == 8.0  # min fallback
    r = us._darvas_from_df(_ohlc(9.0, lows=[0.0] * 25))
    assert r == {"signal": "NO_BOX", "box_top": 20.0, "box_bottom": None}
    df = _ohlc(9.0)
    df.loc[24, "High"] = 99.0
    assert us._darvas_from_df(df)["box_top"] == 20.0  # current bar excluded
    assert us._darvas_from_df(_ohlc(9.0), confirm=2)["box_top"] == 20.0


def _batch_frame(symbols, rows=25, close_last=25.0):
    base = _ohlc(close_last, n=rows)
    pieces = {s: base for s in symbols}
    return pd.concat(pieces, axis=1)


def test_run_darvas_batch_multi_and_missing(us, monkeypatch):
    calls = []

    def dl(**kw):
        calls.append(kw)
        return _batch_frame(kw["tickers"].split(" ")[:2])

    monkeypatch.setattr(us.yf, "download", dl)
    out = us.run_darvas_batch(["AAA", "BBB", "CCC"], batch_size=3)
    assert out["AAA"]["signal"] == "BREAKOUT_BUY" and out["BBB"]["signal"] == "BREAKOUT_BUY"
    assert out["CCC"] == {"signal": "NO_DATA", "box_top": None, "box_bottom": None}
    kw = calls[0]
    assert kw["tickers"] == "AAA BBB CCC" and kw["period"] == "6mo" and kw["group_by"] == "ticker"
    assert kw["auto_adjust"] is True and kw["progress"] is False


def test_run_darvas_batch_batching_progress_and_errors(us, monkeypatch):
    seen = []

    class Bar:
        n = 0

        def update(self, k):
            Bar.n += k

    def dl(**kw):
        batch = kw["tickers"].split(" ")
        seen.append(batch)
        if batch[0] == "S2":
            raise RuntimeError("batch failed")
        return _batch_frame(batch)

    monkeypatch.setattr(us.yf, "download", dl)
    syms = [f"S{i}" for i in range(5)]
    out = us.run_darvas_batch(syms, batch_size=2, pbar=Bar())
    assert seen == [["S0", "S1"], ["S2", "S3"], ["S4"]]
    assert set(out) == {"S0", "S1", "S4"}  # failed batch recorded nothing
    assert Bar.n == 5  # failed batch still advances the bar by its size


def test_run_darvas_batch_flat_frame_and_per_symbol_error(us, monkeypatch):
    flat = _ohlc(25.0)
    monkeypatch.setattr(us.yf, "download", lambda **kw: flat)
    out = us.run_darvas_batch(["ONE"])
    assert out["ONE"]["signal"] == "BREAKOUT_BUY"

    def boom(df, confirm=3):
        raise ValueError("bad frame")

    monkeypatch.setattr(us, "_darvas_from_df", boom)
    out = us.run_darvas_batch(["ONE"])
    assert out["ONE"] == {"signal": "ERROR", "box_top": None, "box_bottom": None}
    assert us.run_darvas_batch([]) == {}


# ------------------------------------------------------------------ helpers
def test_first_df_and_val(us):
    df = pd.DataFrame({"a": [1]})
    t = types.SimpleNamespace(x=None, y=pd.DataFrame(), z=df, w="notdf")
    assert us._first_df(t, "x", "y", "w", "z") is df
    assert us._first_df(t, "x", "y", "missing") is None
    d = pd.DataFrame({"2025": [10.0, np.nan], "2024": [5.0, 1.0]}, index=["A", "B"])
    assert us._val(d, "A") == 10.0 and us._val(d, "A", col=1) == 5.0
    assert us._val(d, "B") is None and us._val(d, "Z", "A") == 10.0
    assert us._val(d, "A", col=9) is None
    assert us._val(None, "A") is None and us._val(pd.DataFrame(), "A") is None


# ------------------------------------------------------------------ piotroski
def _good_t():
    inc = _frame({"Net Income": [40, 20], "Total Revenue": [300, 200], "Gross Profit": [150, 80]})
    bal = _frame({"Total Assets": [100, 100], "Long Term Debt": [10, 30], "Current Assets": [250, 190],
                  "Current Liabilities": [100, 100], "Share Issued": [90, 95]})
    cf = _frame({"Operating Cash Flow": [60, 30]})
    return types.SimpleNamespace(income_stmt=inc, balance_sheet=bal, cash_flow=cf)


def test_piotroski_perfect_and_criteria(us):
    r = us.run_piotroski(_good_t())
    assert r["f_score"] == 9 and r["error"] is None and set(r["components"]) == {f"F{i}" for i in range(1, 10)}

    def comp(rows):
        t = _good_t()
        for (frame, row), vals in rows.items():
            getattr(t, frame).loc[row] = vals
        return us.run_piotroski(t)["components"]

    assert comp({("income_stmt", "Net Income"): [-5, 20]})["F1"] == 0
    assert comp({("cash_flow", "Operating Cash Flow"): [-5, 30]})["F2"] == 0
    assert comp({("income_stmt", "Net Income"): [10, 20]})["F3"] == 0
    assert comp({("cash_flow", "Operating Cash Flow"): [20, 30]})["F4"] == 0
    assert comp({("balance_sheet", "Long Term Debt"): [40, 30]})["F5"] == 0
    assert comp({("balance_sheet", "Current Assets"): [100, 190]})["F6"] == 0
    assert comp({("balance_sheet", "Share Issued"): [100, 95]})["F7"] == 0
    assert comp({("balance_sheet", "Share Issued"): [95, 95]})["F7"] == 1  # equal is not dilution
    assert comp({("income_stmt", "Gross Profit"): [30, 80]})["F8"] == 0
    assert comp({("income_stmt", "Total Revenue"): [100, 200]})["F9"] == 0
    assert comp({("balance_sheet", "Long Term Debt"): [30, 30]})["F5"] == 0  # not strictly lower


def test_piotroski_negative_roa_zero_values(us):
    # ROA exactly zero is falsy: ni0 == 0 -> roa None; F3 needs both ROAs known
    t = _good_t()
    t.income_stmt.loc["Net Income"] = [0.0, 20]
    c = us.run_piotroski(t)["components"]
    assert c["F1"] == 0 and c["F3"] == 0 and c["F4"] == 0
    # improving but both negative: F3 = 1 (-0.05 > -0.2) while F1 = 0
    t = _good_t()
    t.income_stmt.loc["Net Income"] = [-5, -20]
    c = us.run_piotroski(t)["components"]
    assert c["F1"] == 0 and c["F3"] == 1


def test_piotroski_missing_data(us):
    r = us.run_piotroski(types.SimpleNamespace())
    assert r == {"f_score": None, "error": "no_income_data"}
    t = _good_t()
    only_inc = types.SimpleNamespace(income_stmt=t.income_stmt)
    r = us.run_piotroski(only_inc)
    assert r["f_score"] == 2 and r["components"]["F7"] == 1 and r["components"]["F8"] == 1  # shares assumed OK


# ------------------------------------------------------------------ coffee can
def _cc_t(info=None, rev=(400, 300, 200, 100), ni=(60, 45, 30, 20), eq=(200, 180, 160, 140),
          fcf=(50, 40, 30, 20), with_cf=True, ltd=(20, 20, 20, 20)):
    inc = _frame({"Total Revenue": list(rev), "Net Income": list(ni)})
    bal = _frame({"Stockholders Equity": list(eq), "Long Term Debt": list(ltd)})
    cf = _frame({"Free Cash Flow": list(fcf)}) if with_cf else None
    t = types.SimpleNamespace(income_stmt=inc, balance_sheet=bal, cash_flow=cf,
                              info=info if info is not None else {"debtToEquity": 45.0, "marketCap": 5e9})
    return t


def test_coffee_can_pass(us):
    r = us.run_coffee_can(_cc_t())
    assert r["qualifies"] is True and r["score"] == "6/6" and r["error"] is None
    assert r["criteria"] == {f"C{i}": 1 for i in range(1, 7)}
    expected_roe = np.mean([60 / 200, 45 / 180, 30 / 160, 20 / 140]) * 100
    assert r["roe_avg"] == round(expected_roe, 2)


def test_coffee_can_each_criterion(us):
    def crit(**kw):
        return us.run_coffee_can(_cc_t(**kw))["criteria"]

    assert crit(rev=(105, 104, 102, 100))["C1"] == 0
    assert crit(ni=(10, 8, 6, 5))["C2"] == 0  # ROE ~5%
    assert crit(info={"debtToEquity": 100, "marketCap": 5e9})["C3"] == 0  # 1.0 not < 1
    assert crit(info={"debtToEquity": 99, "marketCap": 5e9})["C3"] == 1
    assert crit(info={"debtToEquity": 10, "marketCap": 5e9})["C3"] == 0
    assert crit(info={"debtToEquity": 0.9, "marketCap": 5e9})["C3"] == 1
    assert crit(info={"debtToEquity": 50, "marketCap": 1e9})["C4"] == 1  # exactly $1B passes
    assert crit(info={"debtToEquity": 50, "marketCap": 9.99e8})["C4"] == 0
    assert crit(ni=(60, 45, 0, 20))["C5"] == 0  # zero profit is a loss year
    assert crit(fcf=(-1, 40, 30, 20))["C6"] == 0
    assert crit(fcf=(0, 40, 30, 20))["C6"] == 0  # exactly zero is not > 0
    r = us.run_coffee_can(_cc_t(ni=(60, 45, 0, 20)))
    assert r["qualifies"] is False and r["score"] == "5/6"


def test_coffee_can_roe_average_threshold(us):
    # equal equity, ROE 15% exactly is not > 15
    r = us.run_coffee_can(_cc_t(ni=(30, 30, 30, 30), eq=(200, 200, 200, 200)))
    assert r["criteria"]["C2"] == 0 and r["roe_avg"] == 15.0
    r = us.run_coffee_can(_cc_t(ni=(31, 31, 31, 31), eq=(200, 200, 200, 200)))
    assert r["criteria"]["C2"] == 1
    # non-positive equity years are skipped, not divided by
    r = us.run_coffee_can(_cc_t(ni=(60, 45, 30, 20), eq=(200, -50, 0, 140)))
    assert r["roe_avg"] == round(np.mean([60 / 200, 20 / 140]) * 100, 2)
    # every year has non-positive equity -> no average
    r = us.run_coffee_can(_cc_t(eq=(-1, -1, -1, -1)))
    assert r["roe_avg"] is None and r["criteria"]["C2"] == 0


def test_coffee_can_debt_and_mcap_fallbacks(us):
    # no info: D/E from the balance sheet, market cap unknown -> C4 fails
    t = _cc_t(info={})
    t.info = {}
    r = us.run_coffee_can(t)
    assert r["criteria"]["C3"] == 1 and r["criteria"]["C4"] == 0
    t = _cc_t(info={}, ltd=(300, 300, 300, 300))
    assert us.run_coffee_can(t)["criteria"]["C3"] == 0  # 300/200 = 1.5
    t = _cc_t(info={}, eq=(0, 0, 0, 0))
    assert us.run_coffee_can(t)["criteria"]["C3"] == 0  # zero equity fails closed
    # market cap from fast_info when info lacks it
    t = _cc_t(info={"debtToEquity": 50})
    t.fast_info = types.SimpleNamespace(market_cap=2e9)
    assert us.run_coffee_can(t)["criteria"]["C4"] == 1
    t = _cc_t(info={"debtToEquity": 50})  # no fast_info attribute at all
    assert us.run_coffee_can(t)["criteria"]["C4"] == 0


def test_coffee_can_fcf_derivation_and_missing(us):
    def fcf_flag(cf_rows):
        t = _cc_t(with_cf=False)
        t.cash_flow = _frame(cf_rows) if cf_rows else None
        return us.run_coffee_can(t)["criteria"]["C6"]

    assert fcf_flag({"Operating Cash Flow": [100, 90], "Capital Expenditure": [-60, -50]}) == 1
    assert fcf_flag({"Operating Cash Flow": [100, 90], "Capital Expenditures": [-100, -50]}) == 0  # equal -> 0
    assert fcf_flag({"Operating Cash Flow": [100, 90], "Capital Expenditure": [60, 50]}) == 1  # abs() of capex
    assert fcf_flag({"Operating Cash Flow": [100, 90]}) == 0  # no capex
    assert fcf_flag(None) == 0


def test_coffee_can_errors_and_info_failures(us):
    assert us.run_coffee_can(types.SimpleNamespace(info={})) == {
        "qualifies": False, "score": "0/6", "error": "no_income_data"}
    t = _cc_t()
    t.income_stmt = _frame({"Total Revenue": [100.0], "Net Income": [10.0]})  # one year only
    r = us.run_coffee_can(t)
    assert r["criteria"]["C1"] == 0

    class NoInfo:
        income_stmt = _cc_t().income_stmt

        @property
        def info(self):
            raise RuntimeError("blocked")

    r = us.run_coffee_can(NoInfo())
    assert r["criteria"]["C3"] == 0 and r["criteria"]["C4"] == 0 and r["error"] is None
    t = _cc_t(rev=(200, 100, 50, 0))  # non-positive oldest revenue -> no CAGR
    assert us.run_coffee_can(t)["criteria"]["C1"] == 0


# ------------------------------------------------------------------ sqlite
def test_init_db_idempotent_and_schema(us, tmp_path):
    p = tmp_path / "x.db"
    con = us.init_db(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(results)")}
    assert {"symbol", "darvas_signal", "f_score", "f9", "cc_c6", "cc_roe_avg", "scanned_at", "error"} <= cols
    assert {r[1] for r in con.execute("PRAGMA table_info(progress)")} == {"symbol", "phase", "done_at"}
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    con.close()
    con2 = us.init_db(p)  # re-opening an existing db must not fail
    con2.close()


def test_progress_and_upsert(us, tmp_path):
    con = us.init_db(tmp_path / "y.db")
    assert us.already_done(con, "darvas") == set()
    us.mark_done(con, "AAA", "darvas")
    us.mark_done(con, "BBB", "fundamentals")
    us.mark_done(con, "AAA", "darvas")  # INSERT OR REPLACE: no duplicate
    assert us.already_done(con, "darvas") == {"AAA"} and us.already_done(con, "fundamentals") == {"BBB"}
    us.mark_done(con, "AAA", "fundamentals")  # same symbol, new phase replaces the row (PK is symbol)
    assert us.already_done(con, "darvas") == set() and us.already_done(con, "fundamentals") == {"AAA", "BBB"}
    us.upsert_result(con, {"symbol": "AAA", "name": "A", "darvas_signal": "IN_BOX", "darvas_box_top": 10.0})
    us.upsert_result(con, {"symbol": "AAA", "f_score": 8, "darvas_signal": "BREAKOUT_BUY"})
    row = con.execute("SELECT name, darvas_signal, darvas_box_top, f_score FROM results").fetchall()
    assert row == [("A", "BREAKOUT_BUY", 10.0, 8)]  # partial update keeps untouched columns
    assert con.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


# ------------------------------------------------------------------ scan_fundamentals_one
def _scan_ticker(info=None, fast=None, **kw):
    t = _good_t()
    t.info = info if info is not None else {"sector": "Tech", "industry": "Software", "marketCap": 5e9,
                                            "debtToEquity": 40}
    if fast is not None:
        t.fast_info = fast
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_scan_fundamentals_one(us, monkeypatch):
    t = _scan_ticker(fast=types.SimpleNamespace(last_price=123.4))
    t.cash_flow = _frame({"Operating Cash Flow": [60, 30], "Free Cash Flow": [10, 5]})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)
    row = us.scan_fundamentals_one("AAA", "Alpha", "NASDAQ")
    assert row["symbol"] == "AAA" and row["name"] == "Alpha" and row["exchange"] == "NASDAQ"
    assert row["sector"] == "Tech" and row["industry"] == "Software" and row["market_cap"] == 5e9
    assert row["last_price"] == 123.4 and row["f_score"] == 9 and row["f1"] == 1 and row["f9"] == 1
    assert row["cc_score"] in {f"{i}/6" for i in range(7)}
    assert row["cc_qualifies"] in (0, 1) and all(f"cc_c{i}" in row for i in range(1, 7))
    assert "error" not in row and "scanned_at" in row


def test_scan_fundamentals_one_price_fallbacks_and_errors(us, monkeypatch):
    t = _scan_ticker(info={"currentPrice": 55.0})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)  # no fast_info -> currentPrice
    assert us.scan_fundamentals_one("A", "n", "e")["last_price"] == 55.0
    t = _scan_ticker(info={"regularMarketPrice": 44.0})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)
    assert us.scan_fundamentals_one("A", "n", "e")["last_price"] == 44.0
    # both statements missing -> the Piotroski error is recorded
    t = types.SimpleNamespace(info={})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: t)
    row = us.scan_fundamentals_one("A", "n", "e")
    assert row["error"] == "no_income_data" and row["f_score"] is None and row["cc_qualifies"] == 0
    assert row["sector"] == "" and row["cc_c1"] is None

    def boom(s):
        raise RuntimeError("x" * 300)

    monkeypatch.setattr(us.yf, "Ticker", boom)
    row = us.scan_fundamentals_one("A", "n", "e")
    assert len(row["error"]) == 200  # truncated

    class NoInfo:
        income_stmt = _good_t().income_stmt
        balance_sheet = _good_t().balance_sheet
        cash_flow = _good_t().cash_flow

        @property
        def info(self):
            raise RuntimeError("blocked")

    monkeypatch.setattr(us.yf, "Ticker", lambda s: NoInfo())
    row = us.scan_fundamentals_one("A", "n", "e")
    assert row["sector"] == "" and row["market_cap"] is None and row["f_score"] == 9 and "error" not in row


# ------------------------------------------------------------------ compile_outputs
def _fill(us, tmp_path, rows):
    us.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = us.init_db(tmp_path / "c.db")
    for r in rows:
        us.upsert_result(con, r)
    return con


def test_compile_outputs_empty(us, tmp_path, capsys):
    con = _fill(us, tmp_path, [])
    us.compile_outputs(con, "20260101")
    assert "No results in database yet." in capsys.readouterr().out
    assert list(us.OUTPUT_DIR.glob("*.csv")) == []


def test_compile_outputs_filters_and_sorts(us, tmp_path, capsys):
    rows = [
        {"symbol": "TRI", "name": "Triple", "darvas_signal": "BREAKOUT_BUY", "darvas_upside": -5.0,
         "f_score": 8, "cc_qualifies": 1, "cc_score": "6/6", "market_cap": 3e9},
        {"symbol": "TR2", "name": "Triple2", "darvas_signal": "BREAKOUT_BUY", "darvas_upside": -2.0,
         "f_score": 9, "cc_qualifies": 1, "cc_score": "6/6", "market_cap": 9e9},
        {"symbol": "F7", "darvas_signal": "BREAKOUT_BUY", "darvas_upside": -1.0, "f_score": 7,
         "cc_qualifies": 0, "market_cap": 1e9},
        {"symbol": "F6", "darvas_signal": "IN_BOX", "f_score": 6, "cc_qualifies": 1, "cc_score": "6/6",
         "market_cap": 2e9},
        {"symbol": "NOF", "darvas_signal": "BREAKOUT_BUY", "darvas_upside": -9.0, "cc_qualifies": 1},
    ]
    con = _fill(us, tmp_path, rows)
    us.compile_outputs(con, "20260101")
    out = capsys.readouterr().out
    d = us.OUTPUT_DIR
    summary = pd.read_csv(d / "scan_summary_20260101.csv")
    assert len(summary) == 5
    br = pd.read_csv(d / "darvas_breakouts_20260101.csv")
    assert list(br["symbol"]) == ["NOF", "TRI", "TR2", "F7"]  # sorted by upside ascending
    sp = pd.read_csv(d / "strong_piotroski_20260101.csv")
    assert list(sp["symbol"]) == ["TR2", "TRI", "F7"]  # f_score >= 7, descending; NaN excluded
    cc = pd.read_csv(d / "coffee_can_20260101.csv")
    assert set(cc["symbol"]) == {"TRI", "TR2", "F6", "NOF"}
    assert list(cc["symbol"])[:3] == ["TR2", "TRI", "F6"]  # by market cap descending
    tri = pd.read_csv(d / "triple_hits_20260101.csv")
    assert list(tri["symbol"]) == ["TR2", "TRI"]  # needs breakout + F>=7 + CC; F7 lacks CC, F6 not a breakout
    assert "Triple hits (2 stocks)" in out and "TR2" in out
    assert "Darvas breakouts (4)" in out and "Strong Piotroski (3)" in out and "Coffee Can qualifiers (4)" in out


def test_compile_outputs_no_triples(us, tmp_path, capsys):
    con = _fill(us, tmp_path, [{"symbol": "A", "darvas_signal": "IN_BOX", "f_score": 3, "cc_qualifies": 0}])
    us.compile_outputs(con, "20260102")
    out = capsys.readouterr().out
    assert "Triple hits (0 stocks)" in out and "Symbol" not in out.split("Triple hits")[1]
    assert pd.read_csv(us.OUTPUT_DIR / "triple_hits_20260102.csv").empty


# ------------------------------------------------------------------ main
def _patch_main(us, monkeypatch, symbols=("AAA", "BBB", "CCC")):
    uni = pd.DataFrame({"Symbol": list(symbols), "Name": [f"{s} Inc" for s in symbols],
                        "Exchange": ["NASDAQ"] * len(symbols)})
    calls = {"universe": 0, "darvas": [], "fund": []}

    def fetch(ex):
        calls["universe"] += 1
        calls["exchange"] = ex
        return uni

    def darvas(todo, batch_size=500, pbar=None):
        calls["darvas"].append((list(todo), batch_size))
        return {s: {"signal": "BREAKOUT_BUY" if s == "AAA" else "IN_BOX", "box_top": 10.0, "box_bottom": 5.0,
                    "current_price": 11.0, "upside_pct": -1.0, "pos_in_box": 120.0} for s in todo}

    def fund(sym, name, exch):
        calls["fund"].append(sym)
        return {"symbol": sym, "name": name, "exchange": exch, "f_score": 8, "cc_qualifies": 1 if sym == "AAA" else 0,
                "cc_score": "6/6"}

    monkeypatch.setattr(us, "fetch_stock_universe", fetch)
    monkeypatch.setattr(us, "run_darvas_batch", darvas)
    monkeypatch.setattr(us, "scan_fundamentals_one", fund)
    return calls


def test_main_full_pipeline(us, monkeypatch, capsys):
    calls = _patch_main(us, monkeypatch)
    us.main(exchange="nasdaq", workers=2, delay=0, batch_size=77)
    out = capsys.readouterr().out
    assert calls["universe"] == 1 and calls["exchange"] == "nasdaq"
    assert calls["darvas"] == [(["AAA", "BBB", "CCC"], 77)]
    assert sorted(calls["fund"]) == ["AAA", "BBB", "CCC"]
    assert (us.OUTPUT_DIR / "stock_universe.csv").exists()
    con = sqlite3.connect(us.DB_FILE)
    rows = con.execute("SELECT symbol, darvas_signal, f_score, cc_qualifies, name FROM results ORDER BY symbol").fetchall()
    assert rows == [("AAA", "BREAKOUT_BUY", 8, 1, "AAA Inc"), ("BBB", "IN_BOX", 8, 0, "BBB Inc"),
                    ("CCC", "IN_BOX", 8, 0, "CCC Inc")]
    assert con.execute("SELECT COUNT(*) FROM progress").fetchone()[0] == 3
    con.close()
    assert "Breakouts: 1  In-box: 2  Breakdown: 0" in out
    assert "Estimated time:" in out and "Phase 2" in out and "Database:" in out
    assert list(us.OUTPUT_DIR.glob("triple_hits_*.csv"))
    tri = pd.read_csv(next(us.OUTPUT_DIR.glob("triple_hits_*.csv")))
    assert list(tri["symbol"]) == ["AAA"]


def test_main_darvas_only_and_limit(us, monkeypatch, capsys):
    calls = _patch_main(us, monkeypatch)
    us.main(limit=2, darvas_only=True)
    out = capsys.readouterr().out
    assert calls["darvas"][0][0] == ["AAA", "BBB"] and calls["fund"] == []
    assert "Limited to 2 stocks (--limit 2)" in out and "--darvas-only set" in out
    assert "Phase 2" not in out


def test_main_resume_uses_cache_and_skips_done_fundamentals(us, monkeypatch, capsys):
    calls = _patch_main(us, monkeypatch)
    us.main(workers=1, delay=0)  # initial run populates the checkpoint
    calls["darvas"].clear()
    calls["fund"].clear()
    calls["universe"] = 0
    us.main(resume=True, workers=1, delay=0)
    out = capsys.readouterr().out
    assert calls["universe"] == 0 and "Loaded cached universe: 3 stocks" in out
    assert "skipping 0 already done" in out  # see the xfail below: markers overwrite each other
    # resume without a cached universe file refetches
    (us.OUTPUT_DIR / "stock_universe.csv").unlink()
    us.main(resume=True, darvas_only=True)
    assert calls["universe"] == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="progress has PRIMARY KEY(symbol), so each phase's mark_done overwrites the other phase's row: "
    "after a completed run --resume repeats Darvas and fundamentals for every symbol "
    "(us_market_screener.py init_db progress table / mark_done)",
)
def test_main_resume_skips_everything_after_complete_run(us, monkeypatch):
    calls = _patch_main(us, monkeypatch)
    us.main(workers=1, delay=0)
    calls["darvas"].clear()
    calls["fund"].clear()
    us.main(resume=True, workers=1, delay=0)
    assert calls["darvas"] == [([], 500)] and calls["fund"] == []


def test_main_progress_bars_when_tqdm_available(us, monkeypatch):
    calls = _patch_main(us, monkeypatch)
    bars = []

    class Bar:
        def __init__(self, **kw):
            self.kw = kw
            self.n = 0
            self.closed = False
            bars.append(self)

        def update(self, k=1):
            self.n += k

        def close(self):
            self.closed = True

    monkeypatch.setattr(us, "TQDM_OK", True)
    monkeypatch.setattr(us, "tqdm", Bar, raising=False)
    us.main(workers=2, delay=0)
    assert [b.kw["desc"] for b in bars] == ["Darvas", "Fundamentals"]
    assert all(b.closed for b in bars) and bars[0].kw["total"] == 3 and bars[1].n == 3
    assert sorted(calls["fund"]) == ["AAA", "BBB", "CCC"]


def test_piotroski_equal_roa_is_not_improving(us):
    t = _good_t()
    t.income_stmt.loc["Net Income"] = [20, 20]
    t.balance_sheet.loc["Total Assets"] = [100, 100]
    assert us.run_piotroski(t)["components"]["F3"] == 0
