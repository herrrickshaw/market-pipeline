"""Offline tests for the edgar_* fundamentals collectors (EODHD, screener.in, Polygon, FinHub, AlphaVantage, yfinance batch).

Every hard-coded /Users/... path (credential files, reports, universe CSVs) is remapped into tmp_path so no
real file is ever opened; API keys used here are dummy strings and never leave the process.
"""

import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import requests

REAL_PATH = Path
HOME_PREFIX = "/Users/umashankar"


class Resp:
    def __init__(self, status=200, payload=None, raise_exc=None):
        self.status_code = status
        self._payload = payload
        self._raise = raise_exc

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self._raise:
            raise self._raise
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Sess:
    """Scripted requests.Session replacement."""

    script = []
    posts = []
    default = Resp(200, {})

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        item = Sess.script.pop(0) if Sess.script else Sess.default
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, json=None, timeout=None):
        Sess.posts.append((url, json))
        item = Sess.script.pop(0) if Sess.script else Sess.default
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _reset_sess():
    Sess.script, Sess.posts, Sess.default = [], [], Resp(200, {})
    yield


def _load(name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"

    def remap(*a, **k):
        if a and str(a[0]).startswith(HOME_PREFIX):
            return REAL_PATH(str(home) + str(a[0])[len(HOME_PREFIX):])
        return REAL_PATH(*a, **k)

    # module-level FileHandler paths (production_full) must not touch /Users
    monkeypatch.setattr(logging, "FileHandler", lambda *a, **k: logging.NullHandler())
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    if hasattr(mod, "Path"):
        monkeypatch.setattr(mod, "Path", remap)
    if hasattr(mod, "requests"):
        monkeypatch.setattr(mod.requests, "Session", Sess)
    if hasattr(mod, "time"):
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod.home = home
    (home / "market-pipeline" / "code" / "python_files" / "reports").mkdir(parents=True)
    return mod


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ================================================================== EODHD
@pytest.fixture
def eod(monkeypatch, tmp_path):
    mod = _load("edgar_eodhd_fast", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_eodhd_fast", None)


def test_eodhd_fetch_fundamentals(eod):
    x = eod.EODHDExtractor("dummy-key")
    assert x.session.headers["User-Agent"].startswith("Mozilla")
    Sess.script = [Resp(200, {"General": {"PERatio": 21.5, "PriceBookValueRatio": 3.1, "ROE": 0.2, "DebtToEquity": 0.5,
                                          "OperatingMarginTTM": 0.3, "MarketCapitalization": 1e9, "DividendYield": 0.01}})]
    r = x.fetch_fundamentals("AAPL")
    assert r["symbol"] == "AAPL" and r["pe"] == 21.5 and r["pb"] == 3.1 and r["debt_to_equity"] == 0.5
    assert r["opm"] == 0.3 and r["market_cap"] == 1e9 and r["dividend_yield"] == 0.01 and "fetched_at" in r
    url, params = x.session.calls[-1]
    assert url == "https://eodhd.com/api/fundamentals/AAPL.US" and params["api_token"] == "dummy-key"
    Sess.script = [Resp(200, {"Other": 1})]
    assert x.fetch_fundamentals("AAPL") is None and x.failed == []  # no General block is not a failure
    Sess.script = [Resp(500)]
    assert x.fetch_fundamentals("BAD") is None and x.failed == [{"symbol": "BAD", "error": "500"}]
    Sess.script = [requests.ConnectionError("net")]
    assert x.fetch_fundamentals("BAD2") is None and x.failed[-1]["error"] == "net"


def test_eodhd_extract_counts_and_report(eod, caplog):
    x = eod.EODHDExtractor("k")
    caplog.set_level(logging.INFO)
    payloads = {"A": 10.0, "B": None, "C": 5.0}

    def get(url, params=None, timeout=None):
        sym = url.split("/")[-1].split(".")[0]
        return Resp(200, {"General": {"PERatio": payloads[sym]}})

    x.session.get = get
    out = x.extract(["A", "B", "C"])
    assert [d["symbol"] for d in out["data"]] == ["A", "C"]  # symbols without a P/E are not kept
    m = out["metadata"]
    assert m["symbols_requested"] == 3 and m["symbols_extracted"] == 2
    assert m["success_rate_pct"] == pytest.approx(200 / 3) and m["failed"] == 0 and m["duration_seconds"] > 0
    assert "Successfully extracted: 2 (66.7%)" in caplog.text and "EODHD EXTRACTION - FINAL REPORT" in caplog.text
    # progress is logged every 50 symbols and at the end
    caplog.clear()
    x2 = eod.EODHDExtractor("k")
    x2.session.get = lambda url, params=None, timeout=None: Resp(200, {"General": {"PERatio": 1}})
    x2.start_time = 0
    x2.extract_batch([f"S{i}" for i in range(100)])
    assert caplog.text.count("Success:") == 2 and "[50/100]" in caplog.text and "[100/100]" in caplog.text


def test_eodhd_load_us_symbols(eod, tmp_path):
    syms = eod.load_us_symbols()  # no csv: hard-coded list, de-duplicated, placeholder SYMB* names removed
    assert syms[0] == "AAPL" and len(syms) == len(set(syms)) and not any(s.startswith("SYMB") for s in syms)
    assert "COIN" in syms and syms.count("COIN") == 1
    csv = eod.home / "market-pipeline" / "code" / "python_files" / "data" / "us_list.csv"
    _write(csv, "symbol,name\n" + "\n".join(f"S{i},n" for i in range(5)) + "\n")
    assert eod.load_us_symbols() == [f"S{i}" for i in range(5)]
    _write(csv, "ticker,name\nX1,a\nX2,b\n")  # no 'symbol' header -> first column
    assert eod.load_us_symbols() == ["X1", "X2"]
    _write(csv, "")  # unreadable csv falls back to the hard-coded list
    assert eod.load_us_symbols()[0] == "AAPL"
    _write(csv, "symbol\n" + "\n".join(f"T{i}" for i in range(2300)) + "\n")
    assert len(eod.load_us_symbols()) == 2200  # capped


def _eod_main_setup(eod, monkeypatch, key="dummy-env-key"):
    monkeypatch.setattr(eod, "load_us_symbols", lambda: ["AAA", "BBB"])
    sess_rows = {"AAA": 12.0, "BBB": None}
    monkeypatch.setattr(Sess, "default", None)

    def get(self, url, params=None, timeout=None):
        sym = url.split("/")[-1].split(".")[0]
        return Resp(200, {"General": {"PERatio": sess_rows[sym], "PriceBookValueRatio": 2.0, "MarketCapitalization": 5e9}})

    monkeypatch.setattr(Sess, "get", get)
    if key:
        monkeypatch.setenv("EODHD_KEY", key)
    else:
        monkeypatch.delenv("EODHD_KEY", raising=False)


def test_eodhd_main_writes_json_and_cql(eod, monkeypatch):
    _eod_main_setup(eod, monkeypatch)
    eod.main()
    reports = eod.home / "market-pipeline" / "code" / "python_files" / "reports"
    js = next(reports.glob("edgar_eodhd_2*.json"))
    data = json.loads(js.read_text())
    assert [d["symbol"] for d in data["data"]] == ["AAA"] and data["metadata"]["symbols_extracted"] == 1
    cql = next(reports.glob("edgar_eodhd_cassandra_*.cql")).read_text()
    assert "UPDATE herrrickshaw.stock_quotes" in cql and "pe = 12.0, pb = 2.0, roe = 0" in cql
    assert "market = 'us' AND yf_ticker = 'AAA'" in cql and "BBB" not in cql


def test_eodhd_main_without_key_exits(eod, monkeypatch):
    _eod_main_setup(eod, monkeypatch, key=None)
    with pytest.raises(SystemExit) as e:
        eod.main()
    assert e.value.code == 1
    monkeypatch.setenv("EODHD_KEY", "<your-key-here>")  # an unfilled placeholder counts as missing
    with pytest.raises(SystemExit):
        eod.main()


@pytest.mark.xfail(
    strict=True,
    raises=UnboundLocalError,
    reason="main() only assigns api_key when the credentials file holds an EODHD_KEY= line; a file with other "
    "keys but no EODHD_KEY raises UnboundLocalError instead of the friendly 'not configured' exit "
    "(edgar_eodhd_fast.py main)",
)
def test_eodhd_main_credentials_file_without_the_key(eod, monkeypatch):
    _eod_main_setup(eod, monkeypatch, key=None)
    _write(eod.home / ".config" / "market-secrets" / "credentials.env", "OTHER_KEY=dummy\n")
    with pytest.raises(SystemExit):
        eod.main()


# ================================================================== screener.in India
@pytest.fixture
def scr(monkeypatch, tmp_path):
    mod = _load("edgar_screener_india", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_screener_india", None)


def test_screener_login_paths(scr):
    Sess.script = [Resp(200, {})]
    x = scr.ScreenerInExtractor("a@example.com", "pw")
    assert x.token is None and x.session.headers["Referer"] == "https://www.screener.in"  # already authenticated
    Sess.script = [Resp(401), Resp(200, {"token": "tok123"})]
    x = scr.ScreenerInExtractor("a@example.com", "pw")
    assert x.token == "tok123" and x.session.headers["Authorization"] == "Token tok123"
    assert Sess.posts[-1][1] == {"email": "a@example.com", "password": "pw"}
    Sess.script = [Resp(401), Resp(200, {"nope": 1})]
    assert scr.ScreenerInExtractor("a", "b").token is None  # login answered but without a token
    Sess.script = [requests.ConnectionError("down")]
    assert scr.ScreenerInExtractor("a", "b").token is None  # a failing login does not abort the run
    Sess.script = [Resp(401), Resp(500)]
    assert scr.ScreenerInExtractor("a", "b").token is None


def _scr_ext(scr):
    Sess.script = [Resp(200, {})]
    return scr.ScreenerInExtractor("a", "b")


def test_screener_fetch_and_float_helper(scr):
    x = _scr_ext(scr)
    ok = {"company": {"pe": "21.5", "pb": 3, "roe": "x", "debt_to_equity": None, "opm": 12, "market_cap": "1e9",
                      "dividend_yield": 1.2}}
    Sess.script = [Resp(200, ok)]
    r = x.fetch_fundamentals("TCS")
    assert r["pe"] == 21.5 and r["pb"] == 3.0 and r["roe"] is None and r["debt_to_equity"] is None and r["market_cap"] == 1e9
    assert x.extracted == [r] and x.call_count == 1 and x.session.calls[-1][0].endswith("/company/TCS/financials/latest")
    Sess.script = [Resp(200, {"company": {"pe": None}})]
    assert x.fetch_fundamentals("NOPE") is None and len(x.extracted) == 1  # no P/E: not kept, not a failure
    Sess.script = [Resp(200, {"error": "x"})]
    assert x.fetch_fundamentals("NOPE") is None and x.failed[-1] == {"symbol": "NOPE", "error": "Not found or API error"}
    Sess.script = [Resp(200, {})]
    assert x.fetch_fundamentals("NOPE") is None and len(x.failed) == 2
    Sess.script = [Resp(404)]
    assert x.fetch_fundamentals("GONE") is None and x.failed[-1]["error"] == "404"
    assert scr.ScreenerInExtractor._float_or_none(None) is None and scr.ScreenerInExtractor._float_or_none("1.5") == 1.5
    assert scr.ScreenerInExtractor._float_or_none("bad") is None and scr.ScreenerInExtractor._float_or_none(0) == 0.0


def test_screener_extract_report(scr, caplog):
    x = _scr_ext(scr)
    caplog.set_level(logging.INFO)
    Sess.script = [Resp(200, {"company": {"pe": 10}}), Resp(200, {"error": "x"}), Resp(200, {"company": {"pe": 20}})]
    out = x.extract(["A", "B", "C"])
    assert [d["symbol"] for d in out["data"]] == ["A", "C"] and len(out["failed"]) == 1
    m = out["metadata"]
    assert m["symbols_requested"] == 3 and m["symbols_extracted"] == 2 and m["failed"] == 1 and m["api"] == "screener.in"
    assert m["success_rate_pct"] == pytest.approx(200 / 3)
    assert "[    3/    3]" in caplog.text and "SCREENER.IN EXTRACTION - FINAL REPORT" in caplog.text


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="extract([]) logs 100*len(extracted)/len(symbols) before the guarded metadata expression, so an "
    "empty symbol list divides by zero (edgar_screener_india.py extract, 'Successfully extracted' line)",
)
def test_screener_extract_empty_symbol_list(scr):
    out = _scr_ext(scr).extract([])
    assert out["metadata"]["success_rate_pct"] == 0


def test_screener_load_symbols(scr):
    syms = scr.load_india_symbols()  # no csv -> only the fallback list
    assert "RELIANCE" in syms and len(syms) == len(set(syms))
    csv = scr.home / "market-pipeline" / "code" / "python_files" / "data" / "nse_equity_list.csv"
    _write(csv, "SYMBOL,NAME\n" + "\n".join(f"N{i},x" for i in range(2500)) + "\n")
    syms = scr.load_india_symbols()
    assert len(syms) == 2500 and "RELIANCE" not in syms  # >= 2000 symbols: no fallback list added
    _write(csv, "SYMBOL,NAME\nONE,x\nTWO,y\nONE,z\n")
    syms = scr.load_india_symbols()
    assert {"ONE", "TWO", "RELIANCE"} <= set(syms) and len(syms) == len(set(syms))  # short list is topped up, de-duplicated
    _write(csv, "other,cols\n1,2\n")  # no symbol column -> falls back
    assert "RELIANCE" in scr.load_india_symbols()


def _scr_main_setup(scr, monkeypatch, creds):
    monkeypatch.setattr(scr, "load_india_symbols", lambda: ["AAA", "BBB"])
    if creds is not None:
        _write(scr.home / ".config" / "market-secrets" / "credentials.env", creds)

    def get(self, url, params=None, timeout=None):
        if url.endswith("/auth/user"):
            return Resp(200, {})
        sym = url.split("/")[-3]
        return Resp(200, {"company": {"pe": 15.0 if sym == "AAA" else None, "pb": 2.5}})

    monkeypatch.setattr(Sess, "get", get)


def test_screener_main_requires_credentials(scr, monkeypatch):
    _scr_main_setup(scr, monkeypatch, creds=None)
    with pytest.raises(SystemExit) as e:
        scr.main()
    assert e.value.code == 1
    _write(scr.home / ".config" / "market-secrets" / "credentials.env", "SCREENER_EMAIL=<your-email>\nSCREENER_PASSWORD=x\n")
    with pytest.raises(SystemExit):
        scr.main()  # placeholder email
    _write(scr.home / ".config" / "market-secrets" / "credentials.env", "SCREENER_EMAIL=a@example.com\n")
    with pytest.raises(SystemExit):
        scr.main()  # password missing


def test_screener_main_end_to_end(scr, monkeypatch):
    _scr_main_setup(scr, monkeypatch, 'SCREENER_EMAIL="a@example.com"\nSCREENER_PASSWORD="dummy-pw"\n')
    scr.main()
    reports = scr.home / "market-pipeline" / "code" / "python_files" / "reports"
    js = list(reports.glob("edgar_screener_india_2*.json"))
    assert len(js) == 1
    data = json.loads(js[0].read_text())
    assert [d["symbol"] for d in data["data"]] == ["AAA"]
    cql = next(reports.glob("*.cql")).read_text()
    assert "market = 'india' AND yf_ticker = 'AAA'" in cql and "pe = 15.0, pb = 2.5" in cql


# ================================================================== Polygon
@pytest.fixture
def poly(monkeypatch, tmp_path):
    mod = _load("edgar_polygon_universal", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_polygon_universal", None)


def test_polygon_fetch_and_extract(poly, caplog):
    x = poly.PolygonExtractor("dummy-key")
    Sess.script = [Resp(200, {"results": {"pe": 20, "pb": 3, "roe": 0.1, "debt_to_equity": 0.2, "market_cap": 5e9,
                                          "dividend_yield": 0.01}})]
    r = x.fetch_fundamentals("AAPL")
    assert r["pe"] == 20 and r["market_cap"] == 5e9 and x.extracted == [r]
    assert x.session.calls[-1] == ("https://api.polygon.io/v1/ticker/AAPL", {"apikey": "dummy-key"})
    Sess.script = [Resp(200, {"results": {"pe": None, "pb": 1}})]
    assert x.fetch_fundamentals("NOPE") is None and x.failed[-1] == {"symbol": "NOPE", "reason": "no_pe"}
    Sess.script = [Resp(200, {"results": {}})]
    assert x.fetch_fundamentals("EMPTY") is None and len(x.failed) == 1  # no result block: ignored
    Sess.script = [Resp(429)]
    assert x.fetch_fundamentals("LIM") is None and x.failed[-1] == {"symbol": "LIM", "error": "HTTP 429"}
    Sess.script = [Resp(200, ValueError("bad json"))]
    assert x.fetch_fundamentals("BADJ") is None and x.failed[-1]["error"] == "bad json"
    caplog.set_level(logging.INFO)
    waits = []
    poly.time.sleep = lambda s: waits.append(s)
    y = poly.PolygonExtractor("k")
    y.session.get = lambda url, params=None, timeout=None: Resp(200, {"results": {"pe": 5}})
    out = y.extract([f"S{i}" for i in range(11)])
    assert len(waits) == 2 and set(waits) == {12}  # a pause after every 5th call, none before the first
    assert out["metadata"]["symbols_extracted"] == 11 and out["metadata"]["market"] == "us"
    assert out["metadata"]["source"] == "polygon.io" and out["metadata"]["success_rate_pct"] == 100
    assert "[Rate limit] Waiting 12s" in caplog.text


def test_polygon_load_symbols(poly):
    assert poly.load_symbols("us")[0] == "AAPL" and len(poly.load_symbols("us")) == 10
    assert poly.load_symbols("eu") == poly.load_symbols("us")
    csv = poly.home / "market-pipeline" / "code" / "python_files" / "data" / "us_list.csv"
    _write(csv, "symbol,name\nX1,a\n\nX2,b\n")
    assert poly.load_symbols("us") == ["X1", "X2"]  # header skipped, blank rows skipped
    assert poly.load_symbols("eu")[0] == "AAPL"  # only the US universe is read from the csv


# ================================================================== FinHub
@pytest.fixture
def fh(monkeypatch, tmp_path):
    mod = _load("edgar_finhub_us_top500", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_finhub_us_top500", None)


def test_finhub_fetch_quote_and_extract(fh, caplog):
    x = fh.FinHubExtractor("dummy-key")
    Sess.script = [Resp(200, {"c": 150.0, "pe": 25.0, "marketCapitalization": 2e12})]
    r = x.fetch_quote("AAPL")
    assert r["price"] == 150.0 and r["pe"] == 25.0 and r["market_cap"] == 2e12 and x.call_count == 1
    assert x.session.calls[-1][1] == {"symbol": "AAPL", "token": "dummy-key"} and x.session.calls[-1][0].endswith("/quote")
    Sess.script = [Resp(200, {"c": 0})]
    assert x.fetch_quote("ZERO") is None and x.failed[-1] == {"symbol": "ZERO", "reason": "no_price"}
    Sess.script = [Resp(200, {"x": 1})]
    assert x.fetch_quote("MISSING") is None and x.failed[-1]["reason"] == "no_price"
    Sess.script = [Resp(200, {"c": 10, "pe": 0})]
    assert x.fetch_quote("NOPE") is None and x.failed[-1]["reason"] == "no_pe"  # P/E must be strictly positive
    Sess.script = [Resp(200, {"c": 10, "pe": -3})]
    assert x.fetch_quote("NEG") is None and x.failed[-1]["reason"] == "no_pe"
    Sess.script = [Resp(403)]
    assert x.fetch_quote("DENY") is None and x.failed[-1] == {"symbol": "DENY", "error": "HTTP 403"}
    Sess.script = [requests.Timeout("slow")]
    assert x.fetch_quote("SLOW") is None and x.failed[-1]["error"] == "slow"
    caplog.set_level(logging.INFO)
    sleeps = []
    fh.time.sleep = lambda s: sleeps.append(s)
    y = fh.FinHubExtractor("k")
    y.session.get = lambda url, params=None, timeout=None: Resp(200, {"c": 5, "pe": 8})
    out = y.extract(["A", "B", "C"])
    assert sleeps == [1.0, 1.0]  # no pause before the first call
    m = out["metadata"]
    assert m["symbols_extracted"] == 3 and m["api_calls"] == 3 and m["source"] == "finhub" and m["market"] == "us"
    assert "Symbols extracted: 3 / 3" in caplog.text


def test_finhub_symbol_list(fh):
    syms = fh.get_top_us_symbols()
    assert isinstance(syms, list) and syms[0] == "AAPL" and len(syms) > 100


# ================================================================== AlphaVantage (fast)
@pytest.fixture
def av(monkeypatch, tmp_path):
    mod = _load("edgar_alphavantage_fast", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_alphavantage_fast", None)


def test_alphavantage_fetch_and_extract(av, caplog):
    x = av.AlphaVantageExtractor("dummy-key")
    Sess.script = [Resp(200, {"PERatio": "25.5", "PriceToBookRatio": "None", "ReturnOnEquityTTM": "0.3",
                              "DebtToEquityRatio": "0.4", "OperatingMarginTTM": "0.2", "MarketCapitalization": "9e9",
                              "DividendYield": "0.01"})]
    r = x.fetch_fundamentals("AAPL")
    assert r["pe"] == 25.5 and r["pb"] is None and r["market_cap"] == 9e9 and x.call_count == 1
    assert x.session.calls[-1][1] == {"function": "OVERVIEW", "symbol": "AAPL", "apikey": "dummy-key"}
    Sess.script = [Resp(200, {"Note": "rate limited"})]
    assert x.fetch_fundamentals("A") is None and x.failed[-1]["error"] == "rate limited"
    Sess.script = [Resp(200, {"Error Message": "bad symbol"})]
    assert x.fetch_fundamentals("B") is None and x.failed[-1]["error"] == "bad symbol"
    Sess.script = [Resp(200, {"PERatio": "None"})]
    assert x.fetch_fundamentals("C") is None and len(x.failed) == 2  # no P/E: silently dropped
    Sess.script = [Resp(500)]
    assert x.fetch_fundamentals("D") is None and x.failed[-1]["error"] == "500"
    assert av.AlphaVantageExtractor._float_or_none("None") is None and av.AlphaVantageExtractor._float_or_none("2") == 2.0
    assert av.AlphaVantageExtractor._float_or_none(None) is None and av.AlphaVantageExtractor._float_or_none("x") is None
    caplog.set_level(logging.INFO)
    sleeps = []
    av.time.sleep = lambda s: sleeps.append(s)
    y = av.AlphaVantageExtractor("k")
    y.session.get = lambda url, params=None, timeout=None: Resp(200, {"PERatio": "9"})
    out = y.extract([f"S{i}" for i in range(11)])
    assert sleeps == [12, 12] and out["metadata"]["symbols_extracted"] == 11 and out["metadata"]["api"] == "alphavantage"
    assert len(av.load_top_us_symbols()) > 100 and av.load_top_us_symbols()[0] == "AAPL"


# ================================================================== production_full (yfinance)
@pytest.fixture
def pf(monkeypatch, tmp_path):
    mod = _load("edgar_production_full", monkeypatch, tmp_path)
    yield mod
    sys.modules.pop("edgar_production_full", None)


def _fake_yf(monkeypatch, info_by_symbol, fail=()):
    calls = []

    class T:
        def __init__(self, sym):
            self.sym = sym

        @property
        def info(self):
            calls.append(self.sym)
            if self.sym in fail:
                raise RuntimeError("yahoo down " * 10)
            return info_by_symbol.get(self.sym, {})

    m = types.ModuleType("yfinance")
    m.Ticker = T
    monkeypatch.setitem(sys.modules, "yfinance", m)
    return calls


def test_production_full_load_symbols(pf):
    assert pf.load_us_symbols()[:3] == ["AAPL", "MSFT", "GOOGL"]  # no csv -> fallback list
    csv = pf.home / "market-pipeline" / "data" / "us_list.csv"
    _write(csv, "AAA,Alpha\nBBB,Beta\n\nCCC\n")
    got = pf.load_us_symbols()
    assert got[:2] == ["AAA", "BBB"] and "CCC" in got


def test_production_full_extract_symbol(pf, monkeypatch):
    info = {"AAA": {"trailingPE": 12.345, "priceToBook": 1.5, "returnOnEquity": 0.2512, "debtToEquity": 55.5555,
                    "operatingMargins": 0.1, "marketCap": 1_000_000_000, "dividendYield": 0.0123}}
    calls = _fake_yf(monkeypatch, info, fail={"BAD"})
    r = pf.extract_symbol_with_yfinance("AAA")
    d = r["data"]
    assert r["status"] == "success" and d["pe"] == 12.35 and d["roe"] == 25.12 and d["opm"] == 10.0
    assert d["debt_to_equity"] == 55.56 and d["dividend_yield"] == 1.23 and d["market_cap"] == 1_000_000_000  # ints untouched
    sparse = pf.extract_symbol_with_yfinance("EMPTY")["data"]
    assert sparse["pe"] is None and sparse["roe"] is None and sparse["opm"] is None and sparse["dividend_yield"] is None
    err = pf.extract_symbol_with_yfinance("BAD", retries=3)
    assert err["status"] == "error" and len(err["error"]) == 50 and calls.count("BAD") == 3  # retried, error truncated
    assert pf.extract_symbol_with_yfinance("BAD", retries=1)["status"] == "error"
    assert pf.extract_symbol_with_yfinance("BAD", retries=0) is None  # no attempts at all
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert pf.extract_symbol_with_yfinance("AAA") is None  # yfinance not installed


def test_production_full_generate_cql(pf):
    ext = [{"status": "success", "data": {"symbol": "AAA", "pe": 10.0, "pb": None, "roe": 20.0, "market_cap": 5}},
           {"status": "error"}, {"status": "success", "data": None},
           {"status": "success", "data": {"symbol": "BBB", "pe": None}},
           {"status": "success", "data": {"symbol": "CCC", "debt_to_equity": 0.5, "dividend_yield": 1.5}}]
    cql = pf.generate_cql_batch(ext, batch_num=1)
    stmts = cql.split("\n\n")
    lines = cql.splitlines()
    assert sum(1 for ln in lines if ln.startswith("UPDATE")) == 2 and "BBB" not in cql  # nothing to set -> skipped
    assert "SET pe = 10.0, roe = 20.0, market_cap = 5" in cql and "pb =" not in cql
    assert "WHERE market = 'us' AND yf_ticker = 'AAA' AND price_date = '" in cql and "SET debt_to_equity = 0.5, dividend_yield = 1.5" in cql
    assert pf.generate_cql_batch([], 1) == "" and stmts


def test_production_full_main_end_to_end(pf, monkeypatch, capsys):
    info = {s: {"trailingPE": 10.0 + i, "marketCap": 10**9} for i, s in enumerate(["AAA", "BBB", "CCC"])}
    _fake_yf(monkeypatch, info, fail={"CCC"})
    monkeypatch.setattr(pf, "load_us_symbols", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(sys, "argv", ["x", "--batch-size", "1", "--workers", "2", "--symbols-limit", "3"])
    pf.main()
    reports = pf.home / "market-pipeline" / "code" / "python_files" / "reports"
    batches = sorted(reports.glob("edgar_cassandra_batch_*.cql"))
    assert len(batches) == 2  # 2 successful symbols at one statement per batch file
    body = batches[0].read_text()
    assert body.startswith("-- EDGAR Fundamentals Bulk Load - Production") and "-- Records: 1" in body and "-- Batch: 1/2" in body
    js = json.loads(next(reports.glob("edgar_production_results_*.json")).read_text())
    m = js["metadata"]
    assert m["symbols_requested"] == 3 and m["symbols_extracted"] == 2 and m["batch_count"] == 2 and m["total_records"] == 2
    assert m["success_rate_pct"] == pytest.approx(200 / 3) and sorted(d["symbol"] for d in js["data"]) == ["AAA", "BBB"]
