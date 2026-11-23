"""Offline tests for full_us_market_scan (symbol universe, OHLC, fundamentals, Excel output, main)."""

import importlib
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def us(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # module creates ./us_full_scan at import
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: pd.DataFrame()
    fake_yf.Ticker = lambda t: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    for opt in ("market_data_cache", "ml_signal_engine", "nse_data_fetcher", "ncdssdk"):
        monkeypatch.setitem(sys.modules, opt, None)  # force ImportError so no real caches are touched
    sys.modules.pop("full_us_market_scan", None)
    mod = importlib.import_module("full_us_market_scan")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "us_full_scan")
    monkeypatch.setattr(mod, "_SYMBOL_CACHE", tmp_path / "us_full_scan" / ".symbols_cache.json")
    monkeypatch.setattr(mod, "_NCDS_CREDS_PATH", tmp_path / "ncds_auth.json")
    monkeypatch.setattr(mod, "SLEEP_BETWEEN", 0.0)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(mod, "_FX_CACHE", {})
    mod._ncds_symbol_cache.clear()
    yield mod
    sys.modules.pop("full_us_market_scan", None)


def _ohlc(n=10, start=10.0):
    c = np.linspace(start, start + n - 1, n)
    return pd.DataFrame({"Close": c, "High": c + 0.5, "Low": c - 0.5, "Open": c, "Volume": 1000.0})


def test_optional_modules_absent_flags(us):
    assert us._CACHE_OK is False and us._ML_OK is False and us._NSE_OK is False and us._NCDS_AVAILABLE is False
    assert us._bq_fields is not None


# ------------------------------------------------------------------ liquidity gate
def test_liq_gate_import_failure(us, monkeypatch):
    monkeypatch.setitem(sys.modules, "liquidity", None)
    assert us._liq_gate(_ohlc(), "AAPL") == (0.0, "UNKNOWN")


def test_liq_gate_fx_cached_and_failures(us, monkeypatch):
    liq = types.ModuleType("liquidity")
    calls = {"fx": 0}

    def fx():
        calls["fx"] += 1
        return {"USD": 1.0, "EUR": 1.1}
    liq.scan_fx = fx
    liq.scan_gate = lambda df, t, fxc: (123.0, "LIQUID") if fxc.get("EUR") else (0.0, "BAD")
    monkeypatch.setitem(sys.modules, "liquidity", liq)
    assert us._liq_gate(_ohlc(), "A") == (123.0, "LIQUID")
    us._liq_gate(_ohlc(), "B")
    assert calls["fx"] == 1  # FX fetched once per process

    us._FX_CACHE = {}

    def bad_fx():
        raise RuntimeError("fx down")
    liq.scan_fx = bad_fx
    liq.scan_gate = lambda df, t, fxc: (1.0, "OK") if fxc == {"USD": 1.0} else (0.0, "WRONG")
    assert us._liq_gate(_ohlc(), "A") == (1.0, "OK")  # falls back to USD-only table
    us._FX_CACHE = {}
    liq.scan_fx = lambda: None
    assert us._liq_gate(_ohlc(), "A") == (1.0, "OK")  # empty FX result also falls back

    def bad_gate(*a):
        raise RuntimeError("gate")
    liq.scan_gate = bad_gate
    assert us._liq_gate(_ohlc(), "A") == (0.0, "UNKNOWN")


# ------------------------------------------------------------------ caches and NCDS
def test_symbol_cache_roundtrip_ttl_corrupt(us):
    assert us._load_symbol_cache() == (None, None)
    us._save_symbol_cache(["A"], ["B"])
    assert us._load_symbol_cache() == (["A"], ["B"])
    us._SYMBOL_CACHE.write_text(json.dumps({"ts": time.time() - us.SYMBOL_CACHE_TTL - 1, "nasdaq": ["A"]}))
    assert us._load_symbol_cache() == (None, None)
    us._SYMBOL_CACHE.write_text("{")
    assert us._load_symbol_cache() == (None, None)
    us._SYMBOL_CACHE.write_text(json.dumps({"ts": time.time()}))
    assert us._load_symbol_cache() == ([], [])


def test_save_symbol_cache_swallows(us, monkeypatch, tmp_path):
    monkeypatch.setattr(us, "_SYMBOL_CACHE", tmp_path / "nodir" / "c.json")
    us._save_symbol_cache([], [])


def test_load_ncds_creds(us, monkeypatch):
    assert us._load_ncds_creds() is None  # SDK not available
    monkeypatch.setattr(us, "_NCDS_AVAILABLE", True)
    assert us._load_ncds_creds() is None  # no file
    us._NCDS_CREDS_PATH.write_text("not json")
    assert us._load_ncds_creds() is None
    us._NCDS_CREDS_PATH.write_text(json.dumps({"topic": "T"}))
    assert us._load_ncds_creds() == {"topic": "T"}


class _Msg:
    def __init__(self, val, callable_val=False):
        self._v = val
        self.value = (lambda: val) if callable_val else val


def test_fetch_symbols_from_ncds_classifies_and_dedups(us, monkeypatch):
    assert us.fetch_symbols_from_ncds() == ([], [])  # no creds
    monkeypatch.setattr(us, "_NCDS_AVAILABLE", True)
    us._NCDS_CREDS_PATH.write_text(json.dumps({"oauth": {"endpoint": "e"}, "kafka": {}}))
    d = lambda sym, mk, schema="SeqDirectoryMessage": {"schema_name": schema, "symbol": sym, "marketClass": mk}  # noqa: E731
    batches = [[_Msg(d("aapl", "Q"), True), _Msg(d("AAPL", "Q")), _Msg(d("IBM", "N")), _Msg(d("X.Y", "N")), _Msg(d("", "N")),
                _Msg(d("TSLA", "G")), _Msg(d("ZZZ", ""), True), _Msg(d("NOPE", "N", "Other")), _Msg("notadict")], []]
    seen = {}

    class Consumer:
        def __init__(self):
            self.closed = False

        def consume(self, num_messages, timeout):
            return batches.pop(0) if batches else []

        def close(self):
            self.closed = True

    cons = Consumer()

    class Client:
        def __init__(self, sec, kafka):
            seen.update(sec=sec, kafka=kafka)

        def ncds_kafka_consumer(self, topic):
            seen["topic"] = topic
            return cons
    monkeypatch.setattr(us, "_NCDSClient", Client, raising=False)
    nasdaq, nyse = us.fetch_symbols_from_ncds()
    assert nasdaq == ["AAPL", "TSLA", "ZZZ"] and nyse == ["IBM"]
    assert seen["topic"] == "NLSPLUS-CTA-V4" and cons.closed
    assert seen["sec"]["oauth.token.endpoint.uri"] == "e"


def test_fetch_symbols_from_ncds_error(us, monkeypatch, capsys):
    monkeypatch.setattr(us, "_NCDS_AVAILABLE", True)
    us._NCDS_CREDS_PATH.write_text('{"topic": "T"}')

    def boom(*a):
        raise RuntimeError("kafka down")
    monkeypatch.setattr(us, "_NCDSClient", boom, raising=False)
    assert us.fetch_symbols_from_ncds() == ([], [])
    assert "NCDS SDK error: kafka down" in capsys.readouterr().out


# ------------------------------------------------------------------ nasdaq files
NASDAQ_TXT = """Symbol|Security Name|Market Category|Test Issue|ETF
AAPL|Apple|Q|N|N
TESTX|Test|Q|Y|N
SPYX|Etf|Q|N|Y
BRK.B|Berk|N|N|N
TOOLONG|x|Q|N|N
MSFT|Micro|Q|N|N
File Creation Time: 1122|||
"""


def test_parse_nasdaq_file_filters(us):
    assert us._parse_nasdaq_file(NASDAQ_TXT) == ["AAPL", "MSFT"]
    assert us._parse_nasdaq_file("") == []


def test_parse_other_listed_with_exchange_filter(us):
    txt = "ACT Symbol|Security Name|Exchange|ETF|Test Issue\nIBM|IBM|N|N|N\nGE|GE|A|N|N\nBAD|x|N|N|Y\n"
    assert us._parse_nasdaq_file(txt, exchange_filter="n") == ["IBM"]
    assert us._parse_nasdaq_file(txt) == ["IBM", "GE"]
    short_row = "Symbol|Name\n\nAAPL|Apple\n"
    assert us._parse_nasdaq_file(short_row) == ["AAPL"]


def test_fetch_nasdaq_file_tries_https_then_http(us, monkeypatch):
    urls = []

    def get(url, timeout, headers):
        urls.append(url)
        if url.startswith("https"):
            raise RuntimeError("tls")
        return types.SimpleNamespace(text="BODY", raise_for_status=lambda: None)
    monkeypatch.setattr(us.requests, "get", get)
    assert us._fetch_nasdaq_file("x.txt") == "BODY"
    assert urls == [f"{us.NASDAQ_FTP_BASE}/x.txt", f"{us.NASDAQ_FTP_BASE_HTTP}/x.txt"]
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: types.SimpleNamespace(text="", raise_for_status=lambda: None))
    assert us._fetch_nasdaq_file("x.txt") == ""


def test_fetch_all_us_symbols_from_sec(us, monkeypatch, capsys):
    rows = [[1, "Apple", "AAPL", "Nasdaq"], [2, "IBM", "ibm", "NYSE"], [3, "Spy", "SPY", "Arca"], [4, "X", "BRK.B", "NYSE"],
            [5, "Y", "A B", "Nasdaq"], [6, "Z", "", "Nasdaq"], [7, "W", "OTC1", "OTC"], [8, "short"]]
    resp = types.SimpleNamespace(json=lambda: {"data": rows}, raise_for_status=lambda: None)
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: resp)
    assert us.fetch_all_us_symbols_from_sec() == (["AAPL"], ["IBM", "SPY"])

    def boom(*a, **k):
        raise RuntimeError("sec 403")
    monkeypatch.setattr(us.requests, "get", boom)
    assert us.fetch_all_us_symbols_from_sec() == ([], [])
    assert "SEC EDGAR fetch failed: sec 403" in capsys.readouterr().out


def test_fetch_nasdaq_symbols_ftp_then_sec_fallback(us, monkeypatch, capsys):
    monkeypatch.setattr(us, "_fetch_nasdaq_file", lambda f: NASDAQ_TXT)
    assert us.fetch_nasdaq_symbols() == ["AAPL", "MSFT"]
    monkeypatch.setattr(us, "_fetch_nasdaq_file", lambda f: "")
    monkeypatch.setattr(us, "fetch_all_us_symbols_from_sec", lambda: (["SECN"], ["SECY"]))
    assert us.fetch_nasdaq_symbols() == ["SECN"]
    assert "NASDAQ FTP unavailable" in capsys.readouterr().out
    assert us.fetch_nyse_symbols() == ["SECY"]
    monkeypatch.setattr(us, "_fetch_nasdaq_file", lambda f: "Symbol\nIBM\n")
    assert us.fetch_nyse_symbols() == ["IBM"]
    us._ncds_symbol_cache["nyse"] = ["NCDSY"]
    assert us.fetch_nyse_symbols() == ["NCDSY"]


def test_fetch_nasdaq_symbols_via_ncds(us, monkeypatch, capsys):
    monkeypatch.setattr(us, "_NCDS_AVAILABLE", True)
    us._NCDS_CREDS_PATH.write_text('{"topic": "T"}')
    monkeypatch.setattr(us, "fetch_symbols_from_ncds", lambda: (["N1", "N2"], ["Y1"]))
    assert us.fetch_nasdaq_symbols() == ["N1", "N2"]
    assert us._ncds_symbol_cache == {"nasdaq": ["N1", "N2"], "nyse": ["Y1"]}
    assert "NCDS: 2 NASDAQ symbols" in capsys.readouterr().out
    us._ncds_symbol_cache.clear()
    monkeypatch.setattr(us, "fetch_symbols_from_ncds", lambda: ([], []))
    monkeypatch.setattr(us, "_fetch_nasdaq_file", lambda f: NASDAQ_TXT)
    assert us.fetch_nasdaq_symbols() == ["AAPL", "MSFT"]
    assert "NCDS returned no NASDAQ symbols" in capsys.readouterr().out


# ------------------------------------------------------------------ OHLC
def test_bulk_download_uses_cache_when_available(us, monkeypatch):
    monkeypatch.setattr(us, "_CACHE_OK", True)
    got = {}
    monkeypatch.setattr(us, "_US_CACHE", types.SimpleNamespace(get_ohlc_bulk=lambda t, force: got.update(t=t, force=force) or {"A": _ohlc()}))
    assert list(us.bulk_download_ohlc(["A", "B"])) == ["A"] and got == {"t": ["A", "B"], "force": False}


def test_bulk_download_direct_batches_retry_and_error(us, monkeypatch, capsys):
    monkeypatch.setattr(us, "BATCH_SIZE", 2)
    sleeps = []
    monkeypatch.setattr(us.time, "sleep", lambda s: sleeps.append(s))
    cols = pd.MultiIndex.from_product([["Close", "High"], ["A", "B"]])
    good = pd.DataFrame(np.arange(40.0).reshape(10, 4), columns=cols)
    calls = {"n": 0}

    def dl(batch, **kw):
        calls["n"] += 1
        n = calls["n"]
        if n == 1:
            raise RuntimeError("429 Too Many Requests")
        if n == 2:
            return good
        if n == 3:
            return pd.DataFrame()
        if n == 4:
            raise RuntimeError("weird failure")
        return pd.DataFrame({"Close": [1.0, 2.0]})
    us.yf.download = dl
    out = us.bulk_download_ohlc(["A", "B", "C", "D", "E", "F", "G"], period="1y")
    assert set(out) == {"A", "B", "G"}
    text = capsys.readouterr().out
    assert "empty" in text and "ERROR — weird failure" in text and "OK (2 usable)" in text
    assert 30 in sleeps  # first rate-limit backoff = 30s * (attempt + 1)


def test_bulk_download_rate_limit_backoff_grows_and_gives_up(us, monkeypatch):
    sleeps = []
    monkeypatch.setattr(us.time, "sleep", lambda s: sleeps.append(s))

    def dl(*a, **k):
        raise RuntimeError("Rate limited")
    us.yf.download = dl
    assert us.bulk_download_ohlc(["A"]) == {}
    assert sleeps == [30, 60, 90]


# ------------------------------------------------------------------ golden crossover
def test_golden_crossover_states(us):
    assert us.compute_golden_crossover(None)["gc_signal"] is False
    assert us.compute_golden_crossover(pd.DataFrame())["dma50"] is None
    assert us.compute_golden_crossover(_ohlc(200))["dma200"] is None  # needs 201 bars
    rising = pd.DataFrame({"Close": np.linspace(10, 100, 260)})
    r = us.compute_golden_crossover(rising)
    assert r["gc_signal"] is False and r["dma50_above_200"] is True and r["dma_gap_%"] > 0
    # engineer a fresh cross on the last bar: flat, then a sharp jump in the final bars
    c = np.concatenate([np.full(240, 50.0), np.linspace(50.0, 50.0, 20)])
    base = pd.DataFrame({"Close": c})
    assert us.compute_golden_crossover(base)["dma50_above_200"] is False
    c2 = np.concatenate([np.linspace(100, 40, 230), np.linspace(40, 200, 31)])
    found = None
    for k in range(5, 31):
        res = us.compute_golden_crossover(pd.DataFrame({"Close": c2[:230 + k]}))
        if res["gc_signal"]:
            found = res
            break
    assert found is not None and found["dma50_above_200"] is True


# ------------------------------------------------------------------ fundamentals
def _fin(rows, ncols=3):
    cols = [f"y{i}" for i in range(ncols)]
    return pd.DataFrame({k: v + [v[-1]] * (ncols - len(v)) for k, v in rows.items()}, index=None).T.set_axis(cols, axis=1)


INC = _fin({"Net Income": [150, 100, 80], "Total Revenue": [1500, 1000, 800], "Gross Profit": [700, 400, 300], "EBIT": [300, 200, 150]})
BAL = _fin({"Total Assets": [1000, 1000, 900], "Long Term Debt": [50, 100, 100], "Current Assets": [500, 400, 300],
            "Current Liabilities": [200, 200, 200], "Share Issued": [100, 100, 100], "Stockholders Equity": [600, 500, 400]})
CF = _fin({"Operating Cash Flow": [200, 150, 100], "Free Cash Flow": [120, 100, 80]})
INC_Q = _fin({"Total Revenue": [130, 120, 115, 110, 100], "Net Income": [3e6, 2e6, 2e6, 1e6, 2e6]}, ncols=5)


class _Tk:
    def __init__(self, inc=INC, bal=BAL, cf=CF, inc_q=INC_Q, info=None, mcap=5e9, info_raises=False, fast_raises=False):
        self.income_stmt, self.balance_sheet, self.cash_flow, self.quarterly_income_stmt = inc, bal, cf, inc_q
        self._info, self._ir, self._fr, self._m = info if info is not None else {}, info_raises, fast_raises, mcap

    @property
    def fast_info(self):
        if self._fr:
            raise RuntimeError("fast")
        return types.SimpleNamespace(market_cap=self._m)

    @property
    def info(self):
        if self._ir:
            raise RuntimeError("info")
        return self._info


def test_fundamental_scan_all_screens(us, monkeypatch):
    info = {"shortName": "Apple", "sector": "Tech", "debtToEquity": 45.0, "totalDebt": 100, "totalCash": 50, "bookValue": 10}
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info=info))
    r = us.fundamental_scan("AAPL")
    assert r["name"] == "Apple" and r["sector"] == "Tech" and r["f_score"] == 9 and r["f_strong"]
    assert r["qualifies_cc"] and r["cc_score"] == "6/6"
    assert r["Revenue_CAGR_%"] == pytest.approx(((1500 / 800) ** 0.5 - 1) * 100, abs=0.01)
    assert r["qualifies_mf"] is False  # ROIC = 300/800 = 37.5% but EY = 300/(5e9+50) is tiny
    assert r["ROIC_%"] == 37.5 and r["Earnings_Yield_%"] == pytest.approx(0.0, abs=0.01)
    assert r["qualifies_bc"] is True  # sales +30% YoY (130 vs 100), profit +50%, net profit > $1M
    assert r["Sales_Growth_YoY_%"] == 30.0 and r["Profit_Growth_YoY_%"] == 50.0 and r["Net_Profit_$M"] == 3.0


def test_fundamental_scan_magic_formula_and_boundaries(us, monkeypatch):
    info = {"debtToEquity": 5.0, "totalDebt": 0, "totalCash": 0, "bookValue": 10}
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info=info, mcap=1000.0, inc_q=None))
    r = us.fundamental_scan("X")
    # mcap=1000 -> EV 1000, EBIT 300 => EY 30% > 15 and ROIC 37.5 > 25, but mcap must exceed 5e7 -> fails
    assert r["Earnings_Yield_%"] == 30.0 and r["qualifies_mf"] is False
    assert r["cc_score"] == "4/6"  # D/E of 5 and a market cap below $1B both fail
    assert r["qualifies_bc"] is False and r["Sales_Growth_YoY_%"] is None
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info=info, mcap=1e8, inc_q=None))
    r = us.fundamental_scan("X")
    assert r["qualifies_mf"] is False and r["Earnings_Yield_%"] == pytest.approx(0.0, abs=0.01)  # EV now large
    # debtToEquity <= 10 is a ratio; 10.01 is a percentage
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 10.0}, inc_q=None))
    assert us.fundamental_scan("X")["cc_score"] == "5/6"  # de=10 -> C3 fails
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info={"debtToEquity": 10.01}, inc_q=None))
    assert us.fundamental_scan("X")["cc_score"] == "6/6"  # de=0.1001 -> passes
    # no debtToEquity -> long-term debt / equity from the statements
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info={}, inc_q=None))
    assert us.fundamental_scan("X")["cc_score"] == "6/6"


def test_fundamental_scan_bull_cartel_thresholds(us, monkeypatch):
    def q(rev0, ni0):
        return _fin({"Total Revenue": [rev0, 0, 0, 0, 100], "Net Income": [ni0, 0, 0, 0, 2e6]}, ncols=5)
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc_q=q(115.0, 2.4e6)))
    assert us.fundamental_scan("X")["qualifies_bc"] is False  # sales growth exactly 15% is not > 15
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc_q=q(116.0, 2.4e6)))
    assert us.fundamental_scan("X")["qualifies_bc"] is False  # profit growth exactly 20% is not > 20
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc_q=q(116.0, 2.5e6)))
    assert us.fundamental_scan("X")["qualifies_bc"] is True
    # net profit must exceed $1M
    small = _fin({"Total Revenue": [116, 0, 0, 0, 100], "Net Income": [9e5, 0, 0, 0, 1e5]}, ncols=5)
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc_q=small))
    assert us.fundamental_scan("X")["qualifies_bc"] is False
    # fewer than five quarters -> bull cartel fields are None
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc_q=_fin({"Total Revenue": [1, 1]}, ncols=3)))
    r = us.fundamental_scan("X")
    assert r["qualifies_bc"] is False and r["Net_Profit_$M"] is None


def test_fundamental_scan_no_income_and_failures(us, monkeypatch):
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(inc=None, info={"longName": "LongCo"}))
    r = us.fundamental_scan("X")
    assert r["name"] == "LongCo" and r["f_score"] is None and r["cc_score"] == "N/A" and r["qualifies_mf"] is False
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(info_raises=True, fast_raises=True))
    r = us.fundamental_scan("X")
    assert r["name"] == "" and r["cc_score"] == "5/6"  # mcap unknown -> C4 fails

    def boom(s):
        raise RuntimeError("yf exploded")
    monkeypatch.setattr(us.yf, "Ticker", boom)
    r = us.fundamental_scan("X")
    assert r["error"] == "yf exploded" and r["f_score"] is None and r["cc_score"] == "N/A" and r["qualifies_bc"] is False


def test_fundamental_scan_cash_flow_fallbacks(us, monkeypatch):
    cf = _fin({"Operating Cash Flow": [200, 1, 1], "Capital Expenditure": [-300, -1, -1]})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(cf=cf, info={"debtToEquity": 20}))
    assert us.fundamental_scan("X")["cc_score"] == "5/6"
    cf2 = _fin({"Operating Cash Flow": [200, 1, 1], "Capital Expenditures": [-50, -1, -1]})
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(cf=cf2, info={"debtToEquity": 20}))
    assert us.fundamental_scan("X")["cc_score"] == "6/6"
    monkeypatch.setattr(us.yf, "Ticker", lambda s: _Tk(cf=None, info={"debtToEquity": 20}))
    assert us.fundamental_scan("X")["cc_score"] == "5/6"


# ------------------------------------------------------------------ excel
def test_save_excel_sheets_and_filters(us, tmp_path):
    allr = [{"Symbol": "A", "Change%": 1.0, "Turnover_USD": 5e6, "GC_Signal": "GOLDEN_CROSS", "DMA_Gap%": 3.0, "Upside_to_Top%": 2.0},
            {"Symbol": "B", "Change%": 2.0, "Turnover_USD": 4e6, "GC_Signal": "BELOW_200DMA", "DMA_Gap%": -1.0, "Upside_to_Top%": 1.0}]
    fund = [{"Symbol": "A", "Piotroski_Score": 8, "Piotroski_Strong": "YES", "CoffeeCan": "PASS", "MagicFormula": "PASS", "BullCartel": "PASS",
             "Revenue_CAGR_%": 12, "ROIC_%": 30, "Profit_Growth_YoY_%": 25, "Turnover_USD": 5e6},
            {"Symbol": "B", "Piotroski_Score": 3, "Piotroski_Strong": "NO", "CoffeeCan": "FAIL", "MagicFormula": "FAIL", "BullCartel": "FAIL",
             "Turnover_USD": 4e6}]
    ml = {"A": {"ML_Direction": "BULLISH", "ML_Pred_Ret%": 2.0}, "B": {"ML_Direction": "BEARISH", "ML_Pred_Ret%": -1.0}, "C": {"ML_Direction": "NEUTRAL", "ML_Pred_Ret%": 0.0}}
    path = us.save_excel(allr, allr, fund, fund[:1], [dict(fund[0], Screens_Passed=4)], tag="us", ml_signal_map=ml)
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets["All_Fundamentals"]["Symbol"]) == ["A", "B"]  # sorted by Piotroski desc
    for name in ("Piotroski_Strong", "Coffee_Can", "Magic_Formula", "Bull_Cartel", "Triple_Hits"):
        assert list(sheets[name]["Symbol"]) == ["A"], name
    assert list(sheets["Golden_Crossover"]["Symbol"]) == ["A"]
    assert list(sheets["ML_Bullish"]["Symbol"]) == ["A"] and list(sheets["ML_Bearish"]["Symbol"]) == ["B"]
    assert list(sheets["Multi_Screen_Hits"]["Screens_Passed"]) == [4]
    assert list(sheets["All_Stocks"]["Symbol"]) == ["B", "A"]  # sorted by Change% desc
    path2 = us.save_excel([], [], [], [], tag="t")
    assert "t_full_scan_" in path2.name and "ML_Bullish" not in pd.read_excel(path2, sheet_name=None)


# ------------------------------------------------------------------ main
def _install_main(us, monkeypatch, frames, fund=None, liq_none=()):
    monkeypatch.setattr(us, "_load_symbol_cache", lambda: (None, None))
    monkeypatch.setattr(us, "fetch_nasdaq_symbols", lambda: ["AAA", "BBB", "CCC", "PENNY", "DDD"])
    monkeypatch.setattr(us, "fetch_nyse_symbols", lambda: ["NYA", "AAA"])
    monkeypatch.setattr(us, "bulk_download_ohlc", lambda t, period="1y": {s: frames[s] for s in t if s in frames})
    monkeypatch.setattr(us, "_liq_gate", lambda df, t: (None if t in liq_none else 7e6, None if t in liq_none else "LIQUID") if True else None)
    sig = {"AAA": "BREAKOUT_BUY", "BBB": "BREAKDOWN_SELL", "CCC": "IN_BOX", "NYA": "BREAKOUT_BUY", "DDD": "BREAKOUT_BUY"}
    monkeypatch.setattr(us, "compute_darvas_box", lambda df, confirm=3: {"signal": sig[_who[float(df["Close"].iloc[0])]], "box_top": 20.0,
                                                                          "box_bottom": 5.0, "upside_to_top_pct": 4.0, "position_in_box_pct": 50.0, "data_points": 10})
    monkeypatch.setattr(us, "compute_golden_crossover", lambda df: {"gc_signal": _who[float(df["Close"].iloc[0])] == "AAA", "dma50_above_200": True,
                                                                    "dma50": 1.0, "dma200": 0.5, "dma_gap_%": 100.0})
    monkeypatch.setattr(us, "fundamental_scan", fund or (lambda s: {"name": s, "sector": "S", "f_score": 8, "f_strong": True, "qualifies_cc": True,
                                                                  "cc_score": "6/6", "qualifies_mf": s == "AAA", "qualifies_bc": False}))


_who = {}


def _frames():
    names = ["AAA", "BBB", "CCC", "PENNY", "DDD", "NYA"]
    _who.clear()
    out = {}
    for i, n in enumerate(names):
        start = 100.0 + 10 * i if n != "PENNY" else 0.1
        df = _ohlc(10, start=start)
        if n == "PENNY":
            df["Close"] = np.linspace(0.05, 0.5, 10)
        _who[float(df["Close"].iloc[0])] = n
        out[n] = df
    out["EMPTY"] = pd.DataFrame({"Close": [np.nan, np.nan]})
    return out


def test_main_full_scan_and_fundamentals_reuse(us, monkeypatch, tmp_path, capsys):
    frames = _frames()
    _install_main(us, monkeypatch, frames, liq_none=("DDD",))
    res = us.main(workers=1)
    out = capsys.readouterr().out
    assert "Total unique symbols: 6" in out  # AAA duplicated across lists
    assert res["total_scanned"] == 4          # PENNY (<$1) and DDD (gated) dropped, EMPTY ignored
    assert res["breakout_count"] == 2 and res["gc_count"] == 1
    assert len(res["triple_hits"]) == 2        # AAA + NYA: breakout + Piotroski + coffee can
    assert len(res["six_screen_hits"]) == 4    # every stock: above-200DMA + Piotroski + Coffee Can = 3 screens
    sheets = pd.read_excel(res["excel_path"], sheet_name=None)
    assert set(sheets["Triple_Hits"]["Symbol"]) == {"AAA", "NYA"}
    assert list(sheets["Magic_Formula"]["Symbol"]) == ["AAA"]
    assert "Darvas Breakout BUY:    2" in out and "Top Multi-Screen Hits:" in out
    # second run reuses the freshly written fundamentals workbook instead of refetching
    _install_main(us, monkeypatch, frames)
    fetched = []
    monkeypatch.setattr(us, "fundamental_scan", lambda s: fetched.append(s) or {"name": s, "f_score": 1, "f_strong": False})
    res2 = us.main(top=0, workers=1)
    out2 = capsys.readouterr().out
    assert fetched == ["DDD"]  # only the symbol missing from the previous workbook is fetched
    assert "4 reused, 1 to fetch" in out2 and "merged 4 reused fundamentals" in out2
    assert res2["total_scanned"] == 5  # DDD is no longer gated


def test_main_nasdaq_only_top_no_scans_and_symbol_cache(us, monkeypatch, capsys):
    frames = _frames()
    _install_main(us, monkeypatch, frames)
    monkeypatch.setattr(us, "_load_symbol_cache", lambda: (["AAA", "BBB", "CCC"], ["NYA"]))
    monkeypatch.setattr(us, "fundamental_scan", lambda s: pytest.fail("scans disabled"))
    res = us.main(nasdaq_only=True, top=2, run_scans=False, min_price=1.0)
    out = capsys.readouterr().out
    assert "Symbol cache hit: 3 NASDAQ + 1 NYSE/other" in out and "(limited to first 2 symbols)" in out
    assert "Stage 4 — Skipped (--no-scans)" in out
    assert res["total_scanned"] == 2 and res["triple_hits"] == []
    assert "Top Multi-Screen Hits" not in out


def test_main_ml_stage_and_market_regime(us, monkeypatch, capsys):
    frames = _frames()
    _install_main(us, monkeypatch, frames)
    monkeypatch.setattr(us, "_ML_OK", True)
    monkeypatch.setattr(us, "_NSE_OK", True)
    ml_df = pd.DataFrame({"symbol": ["AAA", "NYA", "BBB"], "direction": ["BULLISH", "BEARISH", "NEUTRAL"],
                          "predicted_ret%": [2.0, -1.0, 0.0], "confidence": [0.9, 0.8, 0.1], "train_rmse": [1, 1, 1]})
    monkeypatch.setattr(us, "_ML_ENGINE", types.SimpleNamespace(predict_batch=lambda data, max_workers: ml_df))
    closes = np.concatenate([np.full(200, 100.0), np.full(50, 110.0)])
    sp = pd.DataFrame({"Close": closes})
    vix = pd.DataFrame({"Close": [30.0]})
    us.yf.download = lambda sym, **k: sp if sym == "^GSPC" else vix
    res = us.main(workers=1)
    out = capsys.readouterr().out
    assert "Regime: BULL" in out and "[PANIC]" in out and "ML: BULLISH=1 | BEARISH=1 | NEUTRAL=1" in out
    sheets = pd.read_excel(res["excel_path"], sheet_name=None)
    assert list(sheets["ML_Bullish"]["Symbol"]) == ["AAA"] and list(sheets["ML_Bearish"]["Symbol"]) == ["NYA"]
    ms = sheets["Multi_Screen_Hits"].set_index("Symbol")
    assert ms.loc["AAA", "ML_Confirmed"] == "YES" and ms.loc["NYA", "ML_Confirmed"] == "NO"


def test_main_ml_failure_and_bear_regime(us, monkeypatch, capsys):
    frames = _frames()
    _install_main(us, monkeypatch, frames)
    monkeypatch.setattr(us, "_ML_OK", True)
    monkeypatch.setattr(us, "_NSE_OK", True)

    def boom(data, max_workers):
        raise RuntimeError("model blew up")
    monkeypatch.setattr(us, "_ML_ENGINE", types.SimpleNamespace(predict_batch=boom))
    mi = pd.MultiIndex.from_product([["Close"], ["^GSPC"]])
    sp = pd.DataFrame(np.concatenate([np.full(200, 100.0), np.full(50, 90.0)]).reshape(-1, 1), columns=mi)

    def dl(sym, **k):
        if sym == "^GSPC":
            return sp
        raise RuntimeError("no vix")
    us.yf.download = dl
    us.main(workers=1)
    out = capsys.readouterr().out
    assert "Regime: BEAR" in out and "ML signal error: model blew up" in out
