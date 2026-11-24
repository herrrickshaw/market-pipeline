"""Offline tests for the earnings_dates_* collectors (NSE, BSE, DART, TDnet, SEC 8-K, yahooquery)."""

import importlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests


class Resp:
    def __init__(self, status=200, payload=None, text="", content=b"", headers=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.content = content
        self.headers = headers or {}
        self._bad_json = bad_json
        self.encoding = None

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Replaces requests.Session; `script` is a list of Resp/Exception consumed in order per .get()."""

    instances = []

    def __init__(self):
        self.headers = {}
        self.cookies = types.SimpleNamespace(clear=lambda: FakeSession.cleared.append(1))
        self.calls = []
        FakeSession.instances.append(self)

    cleared = []
    script = []
    default = Resp(200, [])

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        item = FakeSession.script.pop(0) if FakeSession.script else FakeSession.default
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _reset_fake_session():
    FakeSession.instances = []
    FakeSession.cleared = []
    FakeSession.script = []
    FakeSession.default = Resp(200, [])
    yield


def _fresh(name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    real_mkdir, real_exists = Path.mkdir, Path.exists
    # a module that builds paths from __file__ must not create dirs in the repo or read any .env at import
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: real_mkdir(self, *a, **k) if str(self).startswith(str(tmp_path)) else None)
    monkeypatch.setattr(Path, "exists", lambda self: False if self.name == ".env" else real_exists(self))
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    monkeypatch.setattr(Path, "mkdir", real_mkdir)
    monkeypatch.setattr(Path, "exists", real_exists)
    monkeypatch.setattr(requests, "Session", FakeSession)
    if hasattr(mod, "time"):
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _cleanup(name):
    sys.modules.pop(name, None)


# ================================================================== NSE
@pytest.fixture
def nse(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_nse", monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "nse_cache")
    (tmp_path / "nse_cache").mkdir()
    monkeypatch.setattr(mod, "CACHE_PATH", tmp_path / "nse_cache" / "IN.parquet")
    yield mod
    _cleanup("earnings_dates_nse")


def test_nse_new_session_priming(nse, capsys):
    FakeSession.script = [Resp(403)]
    s = nse._new_session()
    assert s.headers["X-Requested-With"] == "XMLHttpRequest" and s.calls[0][0] == nse.BASE_URL + "/"
    assert "homepage priming GET returned 403" in capsys.readouterr().out
    FakeSession.script = [Resp(200)]
    nse._new_session()
    assert capsys.readouterr().out == ""
    FakeSession.script = [requests.ConnectionError("dns")]
    nse._new_session()
    assert "priming GET failed (dns)" in capsys.readouterr().out


def test_nse_get_json_paths(nse):
    s = FakeSession()
    FakeSession.script = [Resp(200, {"a": 1})]
    assert nse._get_json(s, "x", {"p": 1}) == ({"a": 1}, None) and s.calls[0][0] == f"{nse.API_BASE}/x"
    FakeSession.script = [Resp(200, bad_json=True)]
    assert nse._get_json(s, "x", {}) == (None, "invalid_json")
    FakeSession.script = [Resp(500)]
    assert nse._get_json(s, "x", {}) == (None, "http_500")
    # 403 re-primes (cookies cleared + homepage hit) and retries
    FakeSession.script = [Resp(403), Resp(200), Resp(200, [1])]
    n_before = len(s.calls)
    assert nse._get_json(s, "x", {}) == ([1], None)
    assert FakeSession.cleared == [1] and s.calls[n_before + 1][0] == nse.BASE_URL + "/"
    # the retry budget is exhausted: the final 401 is reported, not retried again
    FakeSession.script = [Resp(401), Resp(200), Resp(401), Resp(200), Resp(401)]
    assert nse._get_json(s, "x", {}) == (None, "http_401")
    # priming failure during a re-prime is swallowed
    FakeSession.script = [Resp(403), requests.ConnectionError("x"), Resp(200, {"ok": 1})]
    assert nse._get_json(s, "x", {}) == ({"ok": 1}, None)
    # request errors: retried, then reported
    FakeSession.script = [requests.Timeout("t1"), requests.Timeout("t2"), requests.Timeout("t3")]
    assert nse._get_json(s, "x", {}) == (None, "request_error:t3")
    FakeSession.script = [requests.Timeout("t1"), Resp(200, {"v": 2})]
    assert nse._get_json(s, "x", {}) == ({"v": 2}, None)
    assert nse._get_json(s, "x", {}, retries=0) is not None  # smoke: default response


def test_nse_fetchers_map_fields(nse, capsys):
    s = FakeSession()
    FakeSession.script = [Resp(200, [{"bm_date": "01-Jan-2025", "bm_purpose": "Results", "bm_desc": "d",
                                     "bm_timestamp": "t", "sm_isin": "INE1"}, {}])]
    rows = nse.fetch_board_meetings(s, "TCS")
    assert rows[0] == {"symbol": "TCS", "event_date": "01-Jan-2025", "event_type": "board_meeting", "source": "nse",
                       "purpose": "Results", "description": "d", "announced_at": "t", "isin": "INE1"}
    assert rows[1]["event_date"] is None and len(rows) == 2
    assert s.calls[0][1] == {"index": "equities", "symbol": "TCS"}
    FakeSession.script = [Resp(500)]
    assert nse.fetch_board_meetings(s, "TCS") == [] and "board-meetings error: http_500" in capsys.readouterr().out
    FakeSession.script = [Resp(200, None)]
    assert nse.fetch_board_meetings(s, "TCS") == []
    FakeSession.script = [Resp(200, [{"filingDate": "02-Feb", "period": "Q", "relatingTo": "Q3", "financialYear": "FY25",
                                     "consolidated": "C", "audited": "U", "broadCastDate": "b", "isin": "INE1"}])]
    r = nse.fetch_financial_results(s, "TCS", period="Annual")[0]
    assert r["event_type"] == "actual_result" and r["event_date"] == "02-Feb" and r["financial_year"] == "FY25"
    assert s.calls[-1][1]["period"] == "Annual"
    FakeSession.script = [Resp(404)]
    assert nse.fetch_financial_results(s, "TCS") == [] and "financial-results error" in capsys.readouterr().out
    FakeSession.script = [Resp(200, {"resCmpData": [{"re_create_dt": "d1", "re_from_dt": "f", "re_to_dt": "t",
                                                     "re_basic_eps_for_cont_dic_opr": 5.5, "re_con_pro_loss": 100}]})]
    e = nse.fetch_actual_eps(s, "TCS")
    assert e == [{"symbol": "TCS", "event_date": "d1", "event_type": "actual_eps", "source": "nse", "period_from": "f",
                  "period_to": "t", "basic_eps": 5.5, "net_profit_loss": 100}]
    FakeSession.script = [Resp(200, {"resCmpData": None})]
    assert nse.fetch_actual_eps(s, "TCS") == []
    FakeSession.script = [Resp(200, {})]
    assert nse.fetch_actual_eps(s, "TCS") == []
    FakeSession.script = [Resp(500)]
    assert nse.fetch_actual_eps(s, "TCS") == [] and "actual-eps error" in capsys.readouterr().out


def _stub_fetchers(nse, monkeypatch, fail=()):
    calls = []

    def bm(s, sym):
        calls.append(("bm", sym))
        if sym in fail:
            raise RuntimeError("boom")
        return [{"symbol": sym, "event_date": "d1", "event_type": "board_meeting", "source": "nse"}]

    monkeypatch.setattr(nse, "fetch_board_meetings", bm)
    monkeypatch.setattr(nse, "fetch_financial_results", lambda s, sym: calls.append(("fr", sym)) or [
        {"symbol": sym, "event_date": "d2", "event_type": "actual_result", "source": "nse"}])
    monkeypatch.setattr(nse, "fetch_actual_eps", lambda s, sym: calls.append(("eps", sym)) or [
        {"symbol": sym, "event_date": "d3", "event_type": "actual_eps", "source": "nse"}])
    return calls


def test_nse_fetch_and_cache_full_resume_force(nse, monkeypatch, capsys):
    calls = _stub_fetchers(nse, monkeypatch)
    df = nse.fetch_and_cache(["AAA", "BBB"])
    assert len(df) == 6 and set(df.event_type) == {"board_meeting", "actual_result", "actual_eps"}
    assert [c[1] for c in calls if c[0] == "bm"] == ["AAA", "BBB"]
    assert nse.CACHE_PATH.exists() and "2 board meetings" not in capsys.readouterr().out
    calls.clear()
    df2 = nse.fetch_and_cache(["AAA", "BBB", "CCC"])  # AAA, BBB cached -> only CCC fetched
    assert [c[1] for c in calls if c[0] == "bm"] == ["CCC"] and len(df2) == 9
    calls.clear()
    out = nse.fetch_and_cache(["AAA"])  # everything cached: nothing fetched
    assert calls == [] and len(out) == 9
    nse.fetch_and_cache(["AAA"], force=True)
    assert [c[1] for c in calls if c[0] == "bm"] == ["AAA"]
    assert len(nse._load_cached()) == 9  # duplicates collapse on (symbol, event_type, event_date)


def test_nse_fetch_and_cache_eps_only_and_errors(nse, monkeypatch, capsys):
    calls = _stub_fetchers(nse, monkeypatch, fail={"BAD"})
    nse.fetch_and_cache(["AAA", "BAD"])
    out = capsys.readouterr().out
    assert "BAD: UNEXPECTED ERROR (boom)" in out
    have = nse._load_cached()
    assert set(have.symbol) == {"AAA"}
    calls.clear()
    # eps_only back-fills only symbols with no actual_eps row, even when they have other rows
    old = pd.concat([nse._load_cached(), pd.DataFrame(
        [{"symbol": "OLD", "event_date": "x", "event_type": "board_meeting", "source": "nse"}])], ignore_index=True)
    old.to_parquet(nse.CACHE_PATH, index=False)
    nse.fetch_and_cache(["AAA", "OLD"], eps_only=True)
    assert calls == [("eps", "OLD")]  # AAA already has actual_eps
    assert set(nse._load_cached().query("event_type == 'actual_eps'").symbol) == {"AAA", "OLD"}
    calls.clear()
    nse.fetch_and_cache(["AAA"], eps_only=True, force=True)
    assert calls == [("eps", "AAA")]


def test_nse_checkpoint_cadence_and_all_error(nse, monkeypatch, capsys):
    _stub_fetchers(nse, monkeypatch)
    monkeypatch.setattr(nse, "CHECKPOINT_EVERY", 2)
    nse.fetch_and_cache(["A", "B", "C"])
    out = capsys.readouterr().out
    assert "checkpoint: 6 total rows saved (2/3 symbols done)" in out
    assert len(nse._load_cached()) == 9
    # eps-only checkpoints every 20 symbols
    nse.CACHE_PATH.unlink()
    monkeypatch.setattr(nse, "fetch_actual_eps", lambda s, sym: [{"symbol": sym, "event_date": "d", "event_type": "actual_eps",
                                                                  "source": "nse"}])
    syms = [f"S{i}" for i in range(21)]
    nse.fetch_and_cache(syms, eps_only=True)
    assert "checkpoint: 20 total rows saved (20/21 symbols done)" in capsys.readouterr().out
    # every symbol failing leaves the cache empty and says so
    nse.CACHE_PATH.unlink()
    monkeypatch.setattr(nse, "fetch_board_meetings", lambda s, sym: (_ for _ in ()).throw(RuntimeError("x")))
    df = nse.fetch_and_cache(["Z"])
    assert df.empty and "no new rows fetched" in capsys.readouterr().out


def test_nse_main_symbol_sources(nse, monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setattr(nse, "fetch_and_cache", lambda symbols, force=False, eps_only=False: seen.append((symbols, force, eps_only)) or pd.DataFrame(
        {"symbol": symbols, "event_type": "board_meeting", "event_date": "d"}))
    monkeypatch.setattr(sys, "argv", ["x"])
    nse.main()
    assert seen[-1][0] == nse.DEFAULT_SYMBOLS
    monkeypatch.setattr(sys, "argv", ["x", "--symbols", "AAA", "BBB", "--force", "--eps-only"])
    nse.main()
    assert seen[-1] == (["AAA", "BBB"], True, True)
    (tmp_path / "cache_seed").mkdir()
    (tmp_path / "cache_seed" / "sector_map_cache.json").write_text(json.dumps({"IN:X": "Tech", "US:Y": "t", "IN:Z": "t"}))
    monkeypatch.setattr(sys, "argv", ["x", "--from-sector-cache"])
    nse.main()
    assert seen[-1][0] == ["X", "Z"]
    (tmp_path / "cache_seed" / "full_universe_symbols.json").write_text(json.dumps({"IN": ["A", "B", "C"]}))
    monkeypatch.setattr(sys, "argv", ["x", "--full-universe", "--limit", "2"])
    nse.main()
    assert seen[-1][0] == ["A", "B"]


# ================================================================== BSE
@pytest.fixture
def bse(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_bse", monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "bse_cache")
    (tmp_path / "bse_cache").mkdir()
    monkeypatch.setattr(mod, "CACHE_PATH", tmp_path / "bse_cache" / "IN.parquet")
    monkeypatch.setattr(mod, "SCRIP_LIST_CACHE", tmp_path / "bse_cache" / "scrip_list.parquet")
    yield mod
    _cleanup("earnings_dates_bse")


def test_bse_session_and_get_json(bse, capsys):
    FakeSession.script = [Resp(403)]
    s = bse._new_session()
    assert "priming GET returned 403" in capsys.readouterr().out
    FakeSession.script = [requests.ConnectionError("x")]
    bse._new_session()
    assert "priming GET failed" in capsys.readouterr().out
    FakeSession.script = [Resp(200, {"a": 1})]
    assert bse._get_json(s, "p", {}) == ({"a": 1}, None) and s.calls[-1][0] == f"{bse.API_BASE}/p"
    FakeSession.script = [Resp(200, bad_json=True)]
    assert bse._get_json(s, "p", {}) == (None, "invalid_json")
    FakeSession.script = [Resp(403), Resp(200, [9])]
    assert bse._get_json(s, "p", {}) == ([9], None)  # no cookie clearing in the BSE variant
    FakeSession.script = [Resp(401), Resp(401), Resp(401)]
    assert bse._get_json(s, "p", {}) == (None, "http_401")
    FakeSession.script = [Resp(500)]
    assert bse._get_json(s, "p", {}) == (None, "http_500")
    FakeSession.script = [requests.Timeout("a"), requests.Timeout("b"), requests.Timeout("c")]
    assert bse._get_json(s, "p", {}) == (None, "request_error:c")


def test_bse_scrip_list_cache_and_error(bse):
    s = FakeSession()
    FakeSession.script = [Resp(200, [{"SCRIP_CD": 500325, "ISIN_NUMBER": "INE002A01018"}])]
    df = bse.fetch_scrip_list(s)
    assert list(df.SCRIP_CD) == [500325] and bse.SCRIP_LIST_CACHE.exists()
    n = len(s.calls)
    assert len(bse.fetch_scrip_list(s)) == 1 and len(s.calls) == n  # cached: no request
    FakeSession.script = [Resp(200, [{"SCRIP_CD": 1}, {"SCRIP_CD": 2}])]
    assert len(bse.fetch_scrip_list(s, force=True)) == 2
    bse.SCRIP_LIST_CACHE.unlink()
    FakeSession.script = [Resp(500)]
    with pytest.raises(RuntimeError, match="BSE scrip list fetch failed: http_500"):
        bse.fetch_scrip_list(s)


def test_bse_financial_results(bse, capsys):
    s = FakeSession()
    FakeSession.script = [Resp(200, {"Table": [{"NEWS_DT": "2025-01-01", "HEADLINE": "H", "CATEGORYNAME": "Result",
                                                "SUBCATNAME": "Results"}, {"DT_TM": "2025-02-02"}]})]
    rows, err = bse.fetch_financial_results(s, "500325")
    assert err is None and [r["event_date"] for r in rows] == ["2025-01-01", "2025-02-02"]
    assert rows[0]["scrip_cd"] == "500325" and rows[0]["source"] == "bse" and rows[1]["headline"] is None
    params = s.calls[-1][1]
    assert params["strScrip"] == "500325" and params["strPrevDate"] == "20160101" and len(params["strToDate"]) == 8
    FakeSession.script = [Resp(200, None)]
    assert bse.fetch_financial_results(s, "1") == ([], None)
    FakeSession.script = [Resp(200, bad_json=True)]
    assert bse.fetch_financial_results(s, "1") == ([], "invalid_json")
    assert "financial-results error: invalid_json" in capsys.readouterr().out


def test_bse_fetch_and_cache_resume_dedup(bse, monkeypatch):
    def fr(session, scrip):
        return [{"scrip_cd": scrip, "event_date": "d", "event_type": "actual_result", "source": "bse", "headline": "h"}], None

    monkeypatch.setattr(bse, "fetch_financial_results", fr)
    df = bse.fetch_and_cache(["1", 2])
    assert set(df.scrip_cd) == {"1", "2"} and bse.CACHE_PATH.exists()
    seen = []
    monkeypatch.setattr(bse, "fetch_financial_results", lambda s, c: seen.append(c) or fr(s, c))
    bse.fetch_and_cache(["1", "2", "3"])
    assert seen == ["3"]  # numeric/str codes compare as strings
    seen.clear()
    bse.fetch_and_cache(["1"], force=True)
    assert seen == ["1"] and len(bse._load_cached()) == 3
    assert len(bse.fetch_and_cache(["1", "2"])) == 3  # nothing missing -> cached frame returned


def test_bse_blocked_cooldown_and_checkpoint(bse, monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(bse.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(bse, "CONSECUTIVE_BLOCK_THRESHOLD", 3)
    monkeypatch.setattr(bse, "BLOCK_COOLDOWN_SECONDS", 99)

    def fr(session, scrip):
        if scrip in ("b1", "b2", "b3"):
            return [], "invalid_json"
        return [{"scrip_cd": scrip, "event_date": "d", "event_type": "actual_result", "source": "bse", "headline": scrip}], None

    monkeypatch.setattr(bse, "fetch_financial_results", fr)
    bse.fetch_and_cache(["ok1", "b1", "b2", "b3", "ok2"])
    out = capsys.readouterr().out
    assert "3 consecutive invalid_json responses -- treating as blocked, cooling down 99s" in out
    assert 99 in sleeps and len(FakeSession.instances) == 2  # a fresh session after the cooldown
    assert set(bse._load_cached().scrip_cd) == {"ok1", "ok2"}
    # a success in between resets the streak: two + ok + two never reaches the threshold of 3
    sleeps.clear()
    bse.CACHE_PATH.unlink()
    bse.fetch_and_cache(["b1", "b2", "ok3", "b1", "b2"])
    assert 99 not in sleeps
    # checkpoint cadence
    bse.CACHE_PATH.unlink()
    monkeypatch.setattr(bse, "CHECKPOINT_EVERY", 2)
    bse.fetch_and_cache(["c1", "c2", "c3"])
    assert "checkpoint: 2 total rows saved (2/3 scrips done)" in capsys.readouterr().out


def test_bse_all_errors_message(bse, monkeypatch, capsys):
    monkeypatch.setattr(bse, "fetch_financial_results", lambda s, c: ([], "http_500"))
    assert bse.fetch_and_cache(["x"]).empty
    assert "no new rows fetched" in capsys.readouterr().out


def test_bse_main_variants(bse, monkeypatch, tmp_path, capsys):
    scrips = pd.DataFrame({"SCRIP_CD": [1, 2, 3], "ISIN_NUMBER": ["I1", "I2", "I3"]})
    monkeypatch.setattr(bse, "fetch_scrip_list", lambda s, force=False: scrips)
    seen = []
    monkeypatch.setattr(bse, "fetch_and_cache", lambda codes, force=False: seen.append((codes, force)) or pd.DataFrame(
        {"scrip_cd": codes}))
    for argv, expected in [([], ["500325", "500002", "500003"]), (["--scrips", "9", "8"], ["9", "8"]),
                           (["--all-active"], [1, 2, 3]), (["--all-active", "--limit", "2"], [1, 2])]:
        monkeypatch.setattr(sys, "argv", ["x"] + argv)
        bse.main()
        assert seen[-1][0] == expected
    monkeypatch.setattr(sys, "argv", ["x", "--bse-only-vs-nse"])
    bse.main()  # no NSE cache: every scrip is BSE-exclusive
    assert seen[-1][0] == [1, 2, 3]
    nse_dir = tmp_path / "cache_seed" / "earnings_dates_nse"
    nse_dir.mkdir(parents=True)
    pd.DataFrame({"isin": ["I2", None]}).to_parquet(nse_dir / "IN.parquet")
    bse.main()
    assert seen[-1][0] == [1, 3] and "1 ISINs already in NSE collection, 2 BSE-exclusive scrips" in capsys.readouterr().out
    pd.DataFrame({"other": [1]}).to_parquet(nse_dir / "IN.parquet")  # no isin column
    bse.main()
    assert seen[-1][0] == [1, 2, 3]
    monkeypatch.setattr(sys, "argv", ["x", "--scrips", "1", "--force"])
    bse.main()
    assert seen[-1] == (["1"], True)


# ================================================================== DART
@pytest.fixture
def dart(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_dart", monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "cache_seed")
    (tmp_path / "cache_seed").mkdir()
    monkeypatch.setattr(mod, "CORP_CODE_CACHE", tmp_path / "cache_seed" / "dart_corp_code_map.parquet")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "cache_seed" / "earnings_dates_dart")
    mod.OUT_DIR.mkdir()
    monkeypatch.setattr(mod, "DART_KEY", "")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    _cleanup("earnings_dates_dart")


def test_dart_dotenv_loading(dart, monkeypatch, tmp_path):
    f1, f2 = tmp_path / "a.env", tmp_path / "b.env"
    f1.write_text("# comment\n\nNOEQUALS\nFAKE_ONE='v1'\nFAKE_TWO=\"v2\"\nEMPTY=\nFAKE_ONE=ignored\n")
    f2.write_text("FAKE_THREE=v3\nFAKE_TWO=other\n")
    monkeypatch.setattr(dart, "_ENV_PATHS", (tmp_path / "missing.env", f1, f2))
    env = {k: v for k, v in os.environ.items()}
    monkeypatch.setattr(os, "environ", env)  # isolate from the real process environment
    dart._load_dotenv()
    assert env["FAKE_ONE"] == "v1" and env["FAKE_TWO"] == "v2" and env["FAKE_THREE"] == "v3"
    assert "EMPTY" not in env and "NOEQUALS" not in env  # no value / no '=' lines are skipped
    # an unreadable file is skipped silently, and existing variables are never overwritten
    env["FAKE_THREE"] = "keep"
    monkeypatch.setattr(dart, "_ENV_PATHS", (f2,))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("denied")))
    dart._load_dotenv()
    assert env["FAKE_THREE"] == "keep"


def test_dart_helpers_and_check(dart):
    assert dart._bare_krx("005930.KS") == "005930" and dart._bare_krx("005930") == "005930"
    dart._check({"status": "000"}, "ctx")
    dart._check({"status": "013"}, "ctx")  # 013 = no data: not an error
    with pytest.raises(RuntimeError, match=r"during ctx: status=010 message=bad key"):
        dart._check({"status": "010", "message": "bad key"}, "ctx")
    with pytest.raises(RuntimeError):
        dart._check({}, "ctx")


def test_dart_self_test(dart, monkeypatch, capsys):
    seen = {}

    def get(url, params=None, timeout=None):
        seen["p"] = params
        return Resp(200, {"status": "010"})

    monkeypatch.setattr(dart.requests, "get", get)
    assert dart.self_test() is True and seen["p"]["crtfc_key"] == "0" * 40
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, {"status": "000"}))
    assert dart.self_test() is False and "UNEXPECTED status" in capsys.readouterr().out
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(500))
    assert dart.self_test() is False and "FAIL: HTTPError" in capsys.readouterr().out


def _corp_zip(rows):
    xml = "<result>" + "".join(
        f"<list><corp_code>{c}</corp_code><corp_name>{n}</corp_name><stock_code>{s}</stock_code>"
        f"<modify_date>20250101</modify_date></list>" for c, n, s in rows) + "</result>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CORPCODE.xml", xml)
    return buf.getvalue()


def test_dart_corp_code_map(dart, monkeypatch, capsys):
    with pytest.raises(RuntimeError, match="DART_KEY is not set"):
        dart.fetch_corp_code_map()
    monkeypatch.setattr(dart, "DART_KEY", "dummy")
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(url)
        return Resp(200, content=_corp_zip([("001", "Samsung", "005930"), ("002", "Private", ""), ("003", "SK", "000660")]),
                    headers={"Content-Type": "application/x-zip"})

    monkeypatch.setattr(dart.requests, "get", get)
    df = dart.fetch_corp_code_map()
    assert list(df.corp_code) == ["001", "002", "003"] and dart.CORP_CODE_CACHE.exists()
    assert "3 entities (2 with a KRX stock_code)" in capsys.readouterr().out
    assert calls == [f"{dart.BASE_URL}/corpCode.xml"]
    dart.fetch_corp_code_map()  # fresh cache (< 7 days): no download
    assert len(calls) == 1
    old = pd.Timestamp.now().timestamp() - 8 * 86400
    os.utime(dart.CORP_CODE_CACHE, (old, old))
    dart.fetch_corp_code_map()  # stale -> refetch
    assert len(calls) == 2
    dart.fetch_corp_code_map(force=True)
    assert len(calls) == 3
    # a non-zip answer (e.g. an error JSON) is reported with its body
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, payload={"status": "010"}, content=b"{}",
                                                                   headers={"Content-Type": "application/json"}))
    with pytest.raises(RuntimeError, match="did not return a zip"):
        dart.fetch_corp_code_map(force=True)
    # a zip served with a non-zip content type is still accepted via the PK magic bytes
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, content=_corp_zip([("9", "X", "123456")]),
                                                                   headers={"Content-Type": "text/plain"}))
    assert list(dart.fetch_corp_code_map(force=True).corp_code) == ["9"]


def test_dart_resolve_corp_codes(dart, monkeypatch):
    monkeypatch.setattr(dart, "fetch_corp_code_map", lambda force=False: pd.DataFrame(
        {"stock_code": ["005930", "000660"], "corp_code": ["001", "003"]}))
    assert dart.resolve_corp_codes(["005930.KS", "000660.KQ", "999999.KS"]) == {"005930.KS": "001", "000660.KQ": "003",
                                                                               "999999.KS": None}


def test_dart_fetch_filings_paginates(dart, monkeypatch):
    with pytest.raises(RuntimeError, match="DART_KEY is not set"):
        dart.fetch_filings("001", "20250101", "20260101")
    monkeypatch.setattr(dart, "DART_KEY", "dummy")
    pages = {1: {"status": "000", "total_page": 3, "list": [{"rcept_dt": "20250301"}]},
             2: {"status": "000", "total_page": "3", "list": [{"rcept_dt": "20250401"}, {"rcept_dt": "20250402"}]},
             3: {"status": "000", "total_page": 3, "list": [{"rcept_dt": "20250501"}]}}
    seen = []

    def get(url, params=None, timeout=None):
        seen.append(params["page_no"])
        return Resp(200, pages[params["page_no"]])

    monkeypatch.setattr(dart.requests, "get", get)
    out = dart.fetch_filings("001", "20250101", "20260101")
    assert [r["rcept_dt"] for r in out] == ["20250301", "20250401", "20250402", "20250501"] and seen == [1, 2, 3]
    # no data (013) and an empty page stop the loop immediately
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, {"status": "013", "message": "none"}))
    assert dart.fetch_filings("001", "a", "b") == []
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, {"status": "000", "total_page": 9, "list": []}))
    assert dart.fetch_filings("001", "a", "b") == []
    monkeypatch.setattr(dart.requests, "get", lambda *a, **k: Resp(200, {"status": "020", "message": "limit"}))
    with pytest.raises(RuntimeError, match="status=020"):
        dart.fetch_filings("001", "a", "b")


def test_dart_collect(dart, monkeypatch, capsys):
    df = dart.collect(["005930.KS"])
    assert df.empty and list(df.columns) == ["ticker", "corp_code", "filing_date", "report_name", "source"]
    assert "DART_KEY is not set" in capsys.readouterr().out
    monkeypatch.setattr(dart, "DART_KEY", "dummy")
    monkeypatch.setattr(dart, "resolve_corp_codes", lambda t: {"005930.KS": "001", "000660.KS": None, "035420.KS": "003"})
    got = {}

    def ff(code, bgn, end, pblntf_ty="A"):
        got["args"] = (code, bgn, end, pblntf_ty)
        if code == "003":
            raise RuntimeError("quota")
        return [{"rcept_dt": "20250315", "report_nm": "사업보고서"}, {"rcept_dt": "bad", "report_nm": "x"}]

    monkeypatch.setattr(dart, "fetch_filings", ff)
    df = dart.collect(["005930.KS", "000660.KS", "035420.KS"], years_back=3)
    out = capsys.readouterr().out
    assert "1/3 tickers had no DART corp_code match: ['000660.KS']" in out and "035420.KS (corp_code=003): FAILED — RuntimeError: quota" in out
    assert list(df.ticker) == ["005930.KS", "005930.KS"] and df.filing_date.iloc[0] == pd.Timestamp("2025-03-15")
    assert pd.isna(df.filing_date.iloc[1])  # unparseable dates become NaT
    assert got["args"][3] == "A" and int(got["args"][2]) > int(got["args"][1]) and df.source.iloc[0] == "dart"
    span_days = (pd.Timestamp(got["args"][2]) - pd.Timestamp(got["args"][1])).days
    assert 365 * 3 - 1 <= span_days <= 365 * 3 + 1
    monkeypatch.setattr(dart, "resolve_corp_codes", lambda t: {})
    assert dart.collect([]).empty


def test_dart_sample_tickers(dart, tmp_path):
    assert dart._sample_tickers(2) == ["005930.KS", "000660.KS"]  # no cache: known-good list
    (tmp_path / "cache_seed" / "sector_map_cache.json").write_text(json.dumps(
        {"KR:005930.KS": "x", "KR:000660.KS": "y", "KR:111111.KS": "z", "KR:035420.KS": "w", "US:AAPL": "t"}))
    assert dart._sample_tickers(10) == ["005930.KS", "000660.KS", "035420.KS", "111111.KS"]  # known-good first
    assert dart._sample_tickers(2) == ["005930.KS", "000660.KS"]


def test_dart_main(dart, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--self-test"])
    monkeypatch.setattr(dart, "self_test", lambda: True)
    with pytest.raises(SystemExit) as e:
        dart.main()
    assert e.value.code == 0
    monkeypatch.setattr(dart, "self_test", lambda: False)
    with pytest.raises(SystemExit) as e:
        dart.main()
    assert e.value.code == 1
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:  # no key
        dart.main()
    assert e.value.code == 1 and "DART_KEY is not set" in capsys.readouterr().out
    monkeypatch.setattr(dart, "DART_KEY", "dummy")
    monkeypatch.setattr(dart, "collect", lambda t: pd.DataFrame())
    with pytest.raises(SystemExit) as e:
        dart.main()
    assert e.value.code == 1 and "No rows fetched" in capsys.readouterr().out
    rec = {}
    monkeypatch.setattr(dart, "collect", lambda t: rec.setdefault("t", t) and pd.DataFrame(
        {"ticker": ["A", "B"], "filing_date": pd.to_datetime(["2025-02-01", "2025-01-01"])}))
    monkeypatch.setattr(sys, "argv", ["x", "--tickers", "A", "B"])
    dart.main()
    assert rec["t"] == ["A", "B"] and (dart.OUT_DIR / "KR.parquet").exists()
    assert "Saved 2 rows" in capsys.readouterr().out


# ================================================================== TDnet
@pytest.fixture
def td(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_tdnet", monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "tdnet_out")
    (tmp_path / "tdnet_out").mkdir()
    yield mod
    _cleanup("earnings_dates_tdnet")


def _row(code, name, title, time_="15:00", href="a.pdf"):
    return (f'<tr><td class="kjTime" x>{time_}</td><td class="kjCode" x>{code}</td><td class="kjName" x>{name}</td>'
            f'<td class="kjTitle" x><a href="{href}" t>{title}</a></td></tr>')


def test_tdnet_parse_rows_and_max_page(td):
    html = ("<table>" + _row("72030", "Toyota", "2025年3月期 決算短信") + _row("ABCDE", "Not a number", "x")
            + "<tr><td>no cells</td></tr>" + _row("67580", "Sony", "配当予想の修正", time_="09:30", href="b.pdf")
            + '<tr><td class="kjCode">11110</td><td class="kjName">NoTitle</td></tr></table>')
    rows = td._parse_rows(html)
    assert [r["ticker"] for r in rows] == ["7203.T", "6758.T"]
    assert rows[0] == {"code": "72030", "ticker": "7203.T", "company_name": "Toyota", "title": "2025年3月期 決算短信",
                       "pdf_url": td.BASE_URL + "a.pdf", "disclosure_time": "15:00"}
    assert rows[1]["disclosure_time"] == "09:30"
    html2 = _row("72030", "T", "x").replace('<td class="kjTime" x>15:00</td>', "")
    assert td._parse_rows(html2)[0]["disclosure_time"] is None
    assert td._max_page("I_list_001_20250101.html I_list_004_20250101.html I_list_002_20250101.html") == 4
    assert td._max_page("nothing") == 1


def test_tdnet_fetch_available_dates_and_pages(td, monkeypatch):
    monkeypatch.setattr(td.requests, "get", lambda url, headers=None, timeout=None: Resp(
        200, text="I_list_001_20250103.html I_list_001_20250101.html I_list_002_20250103.html"))
    assert td.fetch_available_dates() == ["20250101", "20250103"]
    seen = []

    def get(url, headers=None, timeout=None):
        seen.append(url)
        return Resp(404 if "99999999" in url else 200, text="404 Not Found" if "88888888" in url else "<html>ok</html>")

    monkeypatch.setattr(td.requests, "get", get)
    assert td._fetch_page("20250101", 2) == "<html>ok</html>" and seen[-1] == td.BASE_URL + "I_list_002_20250101.html"
    assert td._fetch_page("99999999", 1) is None  # 404 status
    assert td._fetch_page("88888888", 1) is None  # 404 body on a 200
    monkeypatch.setattr(td.requests, "get", lambda *a, **k: Resp(500, text="boom"))
    with pytest.raises(requests.HTTPError):
        td._fetch_page("20250101", 1)


def test_tdnet_collect_disclosures_filters(td, monkeypatch):
    pages = {
        ("20250110", 1): "<x>" + _row("72030", "Toyota", "決算短信 Q3") + _row("67580", "Sony", "お知らせ") + "I_list_002_20250110.html",
        ("20250110", 2): "<x>" + _row("72030", "Toyota", "決算短信 (page 2)") + _row("99840", "SoftBank", "決算短信"),
        ("20250111", 1): None,  # no disclosures that day
    }
    monkeypatch.setattr(td, "_fetch_page", lambda d, p: pages.get((d, p)))
    monkeypatch.setattr(td.time, "sleep", lambda s: None)
    df = td.collect_disclosures(tickers=["7203.T", "6758.T"], dates=["20250110", "20250111"])
    assert list(df.ticker) == ["7203.T", "7203.T"] and set(df.title) == {"決算短信 Q3", "決算短信 (page 2)"}
    assert df.event_date.iloc[0] == pd.Timestamp("2025-01-10").date() and set(df.source) == {"tdnet"}
    assert list(df.columns) == ["ticker", "event_date", "company_name", "title", "disclosure_time", "pdf_url", "source"]
    all_titles = td.collect_disclosures(tickers=["6758.T"], dates=["20250110"], keyword=None)
    assert list(all_titles.title) == ["お知らせ"]
    everything = td.collect_disclosures(dates=["20250110"], keyword=None)  # no ticker filter
    assert len(everything) == 4
    pages[("20250110", 2)] = None  # a missing continuation page is skipped
    assert len(td.collect_disclosures(tickers=["7203.T"], dates=["20250110"])) == 1
    monkeypatch.setattr(td, "fetch_available_dates", lambda: ["20250110"])
    assert len(td.collect_disclosures(tickers=["7203.T"])) == 1  # dates default to the accessible window
    assert td.collect_disclosures(tickers=["7203.T"], dates=[]).empty


def test_tdnet_sector_cache_and_main(td, monkeypatch, tmp_path, capsys):
    assert td._jp_tickers_from_sector_cache(str(tmp_path / "none.json")) == []
    p = tmp_path / "sc.json"
    p.write_text(json.dumps({"JP:7203.T": "x", "US:A": "y", "JP:6758.T": "z"}))
    assert td._jp_tickers_from_sector_cache(str(p)) == ["7203.T", "6758.T"]
    monkeypatch.setattr(td, "fetch_available_dates", lambda: ["20250101", "20250102", "20250103"])
    seen = {}
    frame = pd.DataFrame({"ticker": ["7203.T"], "event_date": [pd.Timestamp("2025-01-02").date()]})

    def collect(tickers=None, dates=None, keyword=None, sleep=0.5):
        seen.update(tickers=tickers, dates=dates, keyword=keyword)
        return frame if tickers != ["EMPTY"] else frame.iloc[0:0]

    monkeypatch.setattr(td, "collect_disclosures", collect)
    out = str(tmp_path / "out.parquet")
    monkeypatch.setattr(sys, "argv", ["x", "--out", out, "--days", "2"])
    td.main()
    assert seen["tickers"] == td.DEFAULT_SAMPLE and seen["dates"] == ["20250102", "20250103"] and seen["keyword"] == "決算短信"
    assert Path(out).exists() and "Found 1 matching disclosure(s)" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--tickers", "EMPTY", "--all-titles", "--out", out])
    td.main()
    assert seen["keyword"] is None and seen["dates"] == ["20250101", "20250102", "20250103"]
    assert "none of the requested tickers had a matching TDnet disclosure" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--from-sector-cache", "--out", out])
    monkeypatch.setattr(td, "_jp_tickers_from_sector_cache", lambda path="x": ["1111.T"])
    td.main()
    assert seen["tickers"] == ["1111.T"] and "Loaded 1 JP tickers" in capsys.readouterr().out


# ================================================================== SEC 8-K
@pytest.fixture
def s8(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_sec_8k", monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "sec_out")
    monkeypatch.setattr(mod, "OUT", tmp_path / "sec_out" / "US.parquet")
    monkeypatch.setattr(mod, "SECTOR_CACHE", tmp_path / "sector_map_cache.json")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    _cleanup("earnings_dates_sec_8k")


class S:
    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        for k, v in self.routes.items():
            if k in url:
                return v if not isinstance(v, Exception) else (_ for _ in ()).throw(v)
        return Resp(404)


def _subs(forms, dates, items, accns):
    return {"filings": {"recent": {"form": forms, "filingDate": dates, "items": items, "accessionNumber": accns}}}


def test_sec8k_ticker_cik_and_submissions(s8):
    sess = S({"company_tickers": Resp(200, {"0": {"ticker": "aapl", "cik_str": 320193}, "1": {"ticker": "MSFT", "cik_str": 789019}}),
              "CIK0000320193": Resp(200, {"x": 1}), "CIK0000999999": Resp(503)})
    assert s8._ticker_cik(sess) == {"AAPL": "0000320193", "MSFT": "0000789019"}
    assert s8._fetch_submissions(sess, "0000320193") == {"x": 1}
    assert s8._fetch_submissions(sess, "0000999999") is None
    bad = S({"company_tickers": Resp(500)})
    with pytest.raises(requests.HTTPError):
        s8._ticker_cik(bad)


def test_sec8k_collect_ticker_8ks(s8, monkeypatch):
    data = _subs(["8-K", "10-Q", "8-K", "8-K", "8-K"], ["2025-01-02", "2025-01-03", "2025-02-01", "2025-03-01", "2025-04-01"],
                 ["2.02,9.01", "", "5.02", "", " 2.02 "], ["a1", "a2", "a3", "a4", "a5"])
    monkeypatch.setattr(s8, "_fetch_submissions", lambda sess, cik: data)
    rows = s8.collect_ticker_8ks(None, "AAPL", "0000320193")
    assert [r["accession"] for r in rows] == ["a1", "a3", "a4", "a5"]  # 10-Q excluded
    assert [r["is_earnings_8k"] for r in rows[:2]] == [True, False] and pd.isna(rows[2]["is_earnings_8k"])
    assert rows[3]["is_earnings_8k"] is True  # whitespace around the item number is tolerated
    assert rows[2]["items"] is None and rows[0]["items"] == "2.02,9.01" and rows[0]["source"] == "sec_8k"
    # a '12.02'-style item must not match 2.02 (exact element match, not substring)
    data2 = _subs(["8-K"], ["2025-01-01"], ["12.02"], ["z"])
    monkeypatch.setattr(s8, "_fetch_submissions", lambda sess, cik: data2)
    assert s8.collect_ticker_8ks(None, "X", "1")[0]["is_earnings_8k"] is False
    # ragged arrays: missing items/date/accession fall back to blanks
    data3 = _subs(["8-K", "8-K"], ["2025-01-01"], [], [])
    monkeypatch.setattr(s8, "_fetch_submissions", lambda sess, cik: data3)
    r = s8.collect_ticker_8ks(None, "X", "1")
    assert r[1]["filing_date"] is None and r[1]["accession"] is None and pd.isna(r[1]["is_earnings_8k"])
    monkeypatch.setattr(s8, "_fetch_submissions", lambda sess, cik: None)
    assert s8.collect_ticker_8ks(None, "X", "1") == []
    monkeypatch.setattr(s8, "_fetch_submissions", lambda sess, cik: {})
    assert s8.collect_ticker_8ks(None, "X", "1") == []


def test_sec8k_run_save_and_main(s8, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(s8.requests, "Session", lambda: object())
    monkeypatch.setattr(s8, "_ticker_cik", lambda sess: {"AAPL": "1", "MSFT": "2"})

    def collect(sess, tk, cik):
        if tk == "MSFT":
            raise RuntimeError("sec down")
        return [{"ticker": tk, "filing_date": "2025-01-0%d" % d, "form_type": "8-K", "items": "2.02" if d == 1 else None,
                 "is_earnings_8k": True if d == 1 else pd.NA, "accession": f"{tk}-{d}", "source": "sec_8k"} for d in (2, 1)]

    monkeypatch.setattr(s8, "collect_ticker_8ks", collect)
    df = s8.run(["aapl", "MSFT", "ZZZZ"])
    out = capsys.readouterr().out
    assert "no CIK found (skipped): ['ZZZZ']" in out and "MSFT: FAILED (sec down)" in out and "aapl: 2 8-Ks total, 1 tagged Item 2.02" in out
    assert list(df.accession) == ["AAPL-1", "AAPL-2"]  # sorted by date
    assert pd.api.types.is_datetime64_any_dtype(df.filing_date)
    monkeypatch.setattr(s8, "collect_ticker_8ks", lambda *a: [])
    empty = s8.run(["AAPL"])
    assert empty.empty and list(empty.columns) == ["ticker", "filing_date", "form_type", "items", "is_earnings_8k", "accession", "source"]
    s8.save(df)
    assert s8.OUT.exists()
    newer = df.iloc[:1].copy()
    newer["items"] = "REPLACED"
    s8.save(newer)
    saved = pd.read_parquet(s8.OUT)
    assert len(saved) == 2 and saved.set_index("accession").loc["AAPL-1", "items"] == "REPLACED"  # keep='last'
    assert "-> " in capsys.readouterr().out

    # main: sample list from the sector cache, earnings-only view
    s8.SECTOR_CACHE.write_text(json.dumps({"US:AAPL": "t", "IN:X": "t", "US:MSFT": "t"}))
    assert s8._sample_us_tickers(1) == ["AAPL"] and s8._sample_us_tickers(5) == ["AAPL", "MSFT"]
    seen = {}
    monkeypatch.setattr(s8, "run", lambda tickers: seen.setdefault("t", tickers) and df)
    monkeypatch.setattr(sys, "argv", ["x", "--sample", "1", "--earnings-only"])
    s8.main()
    out = capsys.readouterr().out
    assert seen["t"] == ["AAPL"] and "AAPL — 1 filing(s) shown" in out and "[EARNINGS(2.02)]" in out
    monkeypatch.setattr(sys, "argv", ["x", "--tickers", "AAPL"])
    s8.main()
    out = capsys.readouterr().out
    assert "AAPL — 2 filing(s) shown" in out and "items-unknown" in out
    monkeypatch.setattr(s8, "run", lambda tickers: pd.DataFrame())
    s8.main()
    assert "no rows collected" in capsys.readouterr().out


# ================================================================== yahooquery
@pytest.fixture
def yq(monkeypatch, tmp_path):
    mod = _fresh("earnings_dates_yahooquery", monkeypatch, tmp_path)
    (tmp_path / "cache_seed").mkdir(exist_ok=True)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    _cleanup("earnings_dates_yahooquery")


def _install_yq(monkeypatch, earnings, cal, earnings_error=None, cal_error=None):
    created = []

    class Ticker:
        def __init__(self, syms, asynchronous=False, max_workers=None):
            created.append((list(syms), asynchronous, max_workers))

        @property
        def earnings(self):
            if earnings_error:
                raise earnings_error
            return earnings

        @property
        def calendar_events(self):
            if cal_error:
                raise cal_error
            return cal

    m = types.ModuleType("yahooquery")
    m.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yahooquery", m)
    return created


def test_yq_helpers(yq, tmp_path):
    assert yq._yf_ticker("IN", "TCS") == "TCS.NS" and yq._yf_ticker("US", "AAPL") == "AAPL"
    assert list(yq._chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]] and list(yq._chunks([], 3)) == []
    (tmp_path / "cache_seed" / "sector_map_cache.json").write_text(json.dumps(
        {"IN:A": "Tech", "IN:B": "Unknown", "US:C": "x", "IN:D": "Fin"}))
    assert yq._classified_symbols("IN") == ["A", "D"]
    (tmp_path / "cache_seed" / "full_universe_symbols.json").write_text(json.dumps({"IN": ["X", "Y"]}))
    assert yq._full_universe_symbols("IN") == ["X", "Y"]


def test_yq_fetch_batch_rows(yq, monkeypatch):
    earnings = {"TCS.NS": {"earningsChart": {"quarterly": [
        {"reportedDate": 1_700_000_000, "periodEndDate": 1_698_700_000, "actual": 1.5, "estimate": 1.4,
         "surprisePct": "7.1", "fiscalQuarter": "3Q2023", "calendarQuarter": "3Q2023"},
        {"actual": 2.0, "surprisePct": ""}]}},
        "INFY.NS": {"earningsChart": {"quarterly": []}}}
    cal = {"TCS.NS": {"earnings": {"earningsDate": ["2026-01-10", "x"], "isEarningsDateEstimate": True}},
           "INFY.NS": "no data string"}
    created = _install_yq(monkeypatch, earnings, cal)
    rows = yq.fetch_batch("IN", ["TCS", "INFY", "WIPRO"])
    assert created == [(["TCS.NS", "INFY.NS", "WIPRO.NS"], True, 10)]
    r0, r1, r2, r3 = rows
    assert r0["ticker"] == "TCS" and r0["reported_date"] == pd.Timestamp(1_700_000_000, unit="s") and r0["surprise_pct"] == 7.1
    assert r0["next_earnings_date"] == "2026-01-10" and r0["next_earnings_is_estimate"] is True and r0["source"] == "yahooquery"
    assert r1["reported_date"] is None and r1["period_end_date"] is None and r1["surprise_pct"] is None and r1["actual"] == 2.0
    assert r2["ticker"] == "INFY" and r2["reported_date"] is None and r2["next_earnings_date"] is None  # empty chart
    assert r3["ticker"] == "WIPRO" and r3["market"] == "IN"  # absent from both payloads


def test_yq_fetch_batch_survives_batch_failures(yq, monkeypatch, capsys):
    _install_yq(monkeypatch, None, None, earnings_error=RuntimeError("e"), cal_error=RuntimeError("c"))
    rows = yq.fetch_batch("US", ["AAPL"])
    out = capsys.readouterr().out
    assert ".earnings batch call failed: e" in out and ".calendar_events batch call failed: c" in out
    assert len(rows) == 1 and rows[0]["ticker"] == "AAPL" and rows[0]["reported_date"] is None
    _install_yq(monkeypatch, "not a dict", "not a dict")  # non-dict payloads are tolerated
    assert yq.fetch_batch("US", ["AAPL"])[0]["next_earnings_date"] is None


def _batch_rows(syms, history=True):
    return [{"ticker": s, "market": "US", "reported_date": pd.Timestamp("2025-01-01") if history else None,
             "period_end_date": pd.Timestamp("2024-12-31"), "actual": 1.0, "estimate": 1.0, "surprise_pct": 0.0,
             "fiscal_quarter": "q", "calendar_quarter": "q", "next_earnings_date": None, "next_earnings_is_estimate": None,
             "source": "yahooquery"} for s in syms]


def test_yq_run_market_batches_resume_and_merge(yq, monkeypatch, tmp_path, capsys):
    (tmp_path / "cache_seed" / "sector_map_cache.json").write_text(json.dumps({f"US:S{i}": "x" for i in range(5)}))
    calls = []
    monkeypatch.setattr(yq, "fetch_batch", lambda m, b: calls.append(list(b)) or _batch_rows(b))
    df = yq.run_market("US", batch_size=2)
    assert calls == [["S0", "S1"], ["S2", "S3"], ["S4"]] and len(df) == 5
    out = capsys.readouterr().out
    assert "5 symbols (classified), batching by 2" in out and "batch 3/3: 1 symbols" in out and "5 with a next" not in out
    calls.clear()
    df = yq.run_market("US", batch_size=2)  # resume: every ticker already has history
    assert calls == [] and len(df) == 5 and "5 tickers already have historical data cached, 0 still to fetch" in capsys.readouterr().out
    # tickers cached without history are refreshed; their stale rows are replaced, not duplicated
    prior = pd.read_parquet(tmp_path / "cache_seed" / "earnings_dates_yahooquery" / "US.parquet")
    prior.loc[prior.ticker == "S0", "reported_date"] = pd.NaT
    prior.to_parquet(tmp_path / "cache_seed" / "earnings_dates_yahooquery" / "US.parquet")
    df = yq.run_market("US", batch_size=2)
    assert calls == [["S0"]] and len(df) == 5 and df[df.ticker == "S0"].reported_date.notna().all()
    calls.clear()
    yq.run_market("US", batch_size=2, resume=False)
    assert len(calls) == 3  # resume off: everything fetched again


def test_yq_run_market_full_universe_and_delay(yq, monkeypatch, tmp_path):
    (tmp_path / "cache_seed" / "full_universe_symbols.json").write_text(json.dumps({"US": ["A", "B", "C"]}))
    sleeps = []
    monkeypatch.setattr(yq.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(yq, "fetch_batch", lambda m, b: _batch_rows(b, history=False))
    df = yq.run_market("US", batch_size=2, full_universe=True, batch_delay_seconds=7)
    assert sleeps == [7]  # a pause between batches, none after the last
    path = tmp_path / "cache_seed" / "earnings_dates_yahooquery_full" / "US.parquet"
    assert path.exists() and len(df) == 3
    # a run that fetches nothing new leaves the prior frame untouched
    monkeypatch.setattr(yq, "fetch_batch", lambda m, b: [])
    assert len(yq.run_market("US", batch_size=2, full_universe=True)) == 3


def test_yq_main(yq, monkeypatch):
    seen = []
    monkeypatch.setattr(yq, "run_market", lambda m, b, full_universe=False, batch_delay_seconds=0: seen.append((m, b, full_universe, batch_delay_seconds)))
    monkeypatch.setattr(sys, "argv", ["x"])
    yq.main()
    assert [s[0] for s in seen] == ["IN", "US", "JP", "KR"] and seen[0][1:] == (300, False, 0)
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--market", "JP", "--batch-size", "50", "--full-universe", "--batch-delay-seconds", "9"])
    yq.main()
    assert seen == [("JP", 50, True, 9)]


# ------------------------------------------------------------------ mutation follow-ups
def test_bse_cooldown_checkpoint_deduplicates_rows(bse, monkeypatch):
    monkeypatch.setattr(bse, "CONSECUTIVE_BLOCK_THRESHOLD", 2)
    monkeypatch.setattr(bse, "BLOCK_COOLDOWN_SECONDS", 1)
    row = {"scrip_cd": "ok1", "event_date": "d", "event_type": "actual_result", "source": "bse", "headline": "h"}
    pd.DataFrame([row]).to_parquet(bse.CACHE_PATH, index=False)  # already cached: --force re-fetches it

    def fr(session, scrip):
        return ([], "invalid_json") if scrip.startswith("b") else ([dict(row)], None)

    monkeypatch.setattr(bse, "fetch_financial_results", fr)
    bse.fetch_and_cache(["ok1", "b1", "b2"], force=True)
    assert len(bse._load_cached()) == 1  # the block checkpoint merges and de-duplicates before cooling down


def test_yahooquery_does_not_checkpoint_or_sleep_without_a_delay(yq, monkeypatch, tmp_path):
    (tmp_path / "cache_seed" / "sector_map_cache.json").write_text(json.dumps({f"US:S{i}": "x" for i in range(4)}))
    sleeps = []
    monkeypatch.setattr(yq.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(yq, "fetch_batch", lambda m, b: _batch_rows(b))
    yq.run_market("US", batch_size=2)
    assert sleeps == []  # batch_delay_seconds == 0 means no pause at all
