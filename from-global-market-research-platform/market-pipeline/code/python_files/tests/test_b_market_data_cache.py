"""Offline tests for market_data_cache (parquet OHLC cache, staleness, bulk/incremental fetch)."""

import json
import types
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import market_data_cache as mc


def _last_weekday(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _ohlc(end=None, n=60, seed=0, start_price=100.0):
    end = end or _last_weekday(date.today() - timedelta(days=1))
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    c = start_price * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "Open": c * 0.999,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Volume": np.full(n, 1e6),
        },
        index=idx,
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    for name in ("OHLC_DIR", "FUND_DIR", "INDEX_DIR", "META_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(mc, name, d)
    monkeypatch.setattr(mc, "META_FILE", tmp_path / "meta_dir" / "cache_index.json")
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    return mc.MarketCache(verbose=False)


class FakeYF:
    """yfinance stand-in: download() returns scripted frames/exceptions in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def download(self, tickers, **kw):
        self.calls.append((tickers, kw))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    def Ticker(self, t):
        return self.ticker_obj


def _yf(monkeypatch, *responses):
    f = FakeYF(*responses)
    monkeypatch.setattr(mc, "yf", f, raising=False)
    monkeypatch.setattr(mc, "_YF_OK", True)
    return f


def _multi(frames):
    """{ticker: ohlc} -> yfinance-style MultiIndex (field, ticker) frame."""
    return pd.concat(dict(frames.items()), axis=1).swaplevel(axis=1).sort_index(axis=1)


# ---------------------------------------------------------------- meta
def test_meta_roundtrip_and_corrupt(cache):
    cache._meta = {"k": {"rows": 1}}
    cache._save_meta()
    assert mc.MarketCache(verbose=False)._meta == {"k": {"rows": 1}}
    mc.META_FILE.write_text("{bad")
    assert mc.MarketCache(verbose=False)._meta == {}


def test_update_meta_records_rows_dates_and_empty(cache, tmp_path):
    df = _ohlc(n=10)
    cache._update_meta("ohlc:X", tmp_path / "x.parquet", df)
    m = cache._meta["ohlc:X"]
    assert (
        m["rows"] == 10
        and m["from"] == str(df.index.min().date())
        and m["to"] == str(df.index.max().date())
    )
    assert (
        m["file"].endswith("x.parquet")
        and json.loads(mc.META_FILE.read_text())["ohlc:X"]["rows"] == 10
    )
    cache._update_meta("ohlc:E", tmp_path / "e.parquet", df.iloc[0:0])
    assert cache._meta["ohlc:E"]["from"] is None and cache._meta["ohlc:E"]["to"] is None


# ---------------------------------------------------------------- _is_stale
def _entry(hours_old, to=None):
    return {"updated": (datetime.now() - timedelta(hours=hours_old)).isoformat(), "to": to}


def test_is_stale_unknown_and_garbage_entries(cache):
    assert cache._is_stale("nope", 20) is True
    cache._meta["bad"] = {"updated": "not a date"}
    assert cache._is_stale("bad", 20) is True
    cache._meta["noupd"] = {}
    assert cache._is_stale("noupd", 20) is True


def test_is_stale_by_age(cache):
    expected = str(_last_weekday(date.today() - timedelta(days=1)))
    cache._meta["a"] = _entry(19.9, expected)
    cache._meta["b"] = _entry(20.1, expected)
    assert cache._is_stale("a", 20) is False and cache._is_stale("b", 20) is True


def test_is_stale_judged_by_last_bar_not_fetch_time(cache):
    expected = _last_weekday(date.today() - timedelta(days=1))
    cache._meta["ok"] = _entry(1, str(expected))
    cache._meta["old_bar"] = _entry(1, str(expected - timedelta(days=1)))
    cache._meta["future"] = _entry(1, str(date.today()))
    cache._meta["nobar"] = _entry(1, None)
    assert cache._is_stale("ok", 20) is False
    assert cache._is_stale("old_bar", 20) is True
    assert cache._is_stale("future", 20) is False
    assert cache._is_stale("nobar", 20) is False


def test_last_cached_date(cache):
    cache._meta["a"] = {"to": "2024-03-05"}
    cache._meta["b"] = {"to": "garbage"}
    cache._meta["c"] = {}
    assert cache._last_cached_date("a") == date(2024, 3, 5)
    assert cache._last_cached_date("b") is None and cache._last_cached_date("c") is None
    assert cache._last_cached_date("zzz") is None


# ---------------------------------------------------------------- paths / helpers
def test_ohlc_path_sanitises_ticker(cache):
    assert cache._ohlc_path("A/B:C").name == "A_B_C.parquet"
    assert cache._ohlc_path("TCS.NS").name == "TCS.NS.parquet"


def test_clean_ohlc_delegates_and_handles_empty_and_bad_index():
    assert mc.MarketCache._clean_ohlc(pd.DataFrame()).empty
    df = _ohlc(n=20)
    bad = df.copy()
    bad.index = [str(d.date()) for d in df.index]
    bad.index = list(bad.index[:-1]) + ["not a date"]
    out = mc.MarketCache._clean_ohlc(bad)
    assert isinstance(out.index, pd.DatetimeIndex) and len(out) == 19
    junk = df.copy()
    junk["Close"] = -1.0
    assert mc.MarketCache._clean_ohlc(junk).empty  # nothing positive survives


def test_clean_ohlc_fallback_without_central_gate(monkeypatch):
    monkeypatch.setattr(mc, "_clean_ohlcv_central", None)
    df = _ohlc(n=5)
    df.iloc[2, 0] = np.nan
    dup = pd.concat([df, df.iloc[[-1]]])
    out = mc.MarketCache._clean_ohlc(dup)
    assert len(out) == 5 and not out.isna().any().any()


def test_write_parquet_atomic_and_cleans_tmp_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "x.parquet"
    mc.MarketCache._write_parquet(p, _ohlc(n=5))
    assert p.exists() and not p.with_suffix(".tmp").exists()
    real = pd.DataFrame.to_parquet

    def boom(self, path, **kw):
        real(self, path, **kw)  # leaves the tmp file behind...
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        mc.MarketCache._write_parquet(tmp_path / "y.parquet", _ohlc(n=5))
    assert not (tmp_path / "y.tmp").exists() and not (tmp_path / "y.parquet").exists()


def test_first_df_picks_first_nonempty_dataframe():
    good = pd.DataFrame({"a": [1]})
    t = types.SimpleNamespace(a=None, b=pd.DataFrame(), c="x", d=good)
    assert mc._first_df(t, "a", "b", "c", "d", "e") is good
    assert mc._first_df(t, "a", "b") is None


# ---------------------------------------------------------------- _yf_download_single
def test_download_single_without_yfinance(cache, monkeypatch):
    monkeypatch.setattr(mc, "_YF_OK", False)
    assert cache._yf_download_single("X", date(2024, 1, 1)).empty


def test_download_single_flat_and_multiindex(cache, monkeypatch):
    df = _ohlc(n=5)
    f = _yf(monkeypatch, df)
    out = cache._yf_download_single("X", date(2024, 1, 1))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"] and len(out) == 5
    assert f.calls[0][1]["start"] == date(2024, 1, 1) and f.calls[0][1]["auto_adjust"] is True
    _yf(monkeypatch, _multi({"X": df}))
    assert len(cache._yf_download_single("X", date(2024, 1, 1))) == 5


def test_download_single_retries_on_rate_limit_then_gives_up(cache, monkeypatch):
    sleeps = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: sleeps.append(s))
    f = _yf(monkeypatch, RuntimeError("429 Too Many"), RuntimeError("Rate limited"), _ohlc(n=3))
    assert len(cache._yf_download_single("X", date(2024, 1, 1))) == 3
    assert sleeps == [15, 30] and len(f.calls) == 3
    f = _yf(monkeypatch, RuntimeError("429"))
    assert cache._yf_download_single("X", date(2024, 1, 1)).empty
    assert len(f.calls) == 3
    f = _yf(monkeypatch, ValueError("other"))
    assert cache._yf_download_single("X", date(2024, 1, 1)).empty and len(f.calls) == 1


def test_download_single_empty_response_retries_without_sleep(cache, monkeypatch):
    f = _yf(monkeypatch, pd.DataFrame())
    assert cache._yf_download_single("X", date(2024, 1, 1)).empty and len(f.calls) == 3


# ---------------------------------------------------------------- get_ohlc
def test_get_ohlc_cold_start_creates_parquet_and_meta(cache, monkeypatch):
    f = _yf(monkeypatch, _ohlc(n=60))
    out = cache.get_ohlc("TCS.NS")
    assert len(out) == 60 and cache._ohlc_path("TCS.NS").exists()
    assert cache._meta["ohlc:TCS.NS"]["rows"] == 60
    start = f.calls[0][1]["start"]
    assert start == date.today() - timedelta(days=mc.OHLC_YEARS * 365 + 30)


def test_get_ohlc_cold_start_empty_download_writes_nothing(cache, monkeypatch):
    _yf(monkeypatch, pd.DataFrame())
    assert cache.get_ohlc("X").empty and not cache._ohlc_path("X").exists()


def test_get_ohlc_fresh_served_from_disk_then_memory(cache, monkeypatch):
    _yf(monkeypatch, _ohlc(n=60))
    cache.get_ohlc("X")
    cache._mem.clear()
    f = _yf(monkeypatch, RuntimeError("must not download"))
    a = cache.get_ohlc("X")
    b = cache.get_ohlc("X")
    assert a is b and f.calls == []  # second call comes from the in-memory layer


def test_get_ohlc_stale_incremental_appends_only_new_rows(cache, monkeypatch):
    old_end = _last_weekday(date.today() - timedelta(days=6))
    old = _ohlc(end=old_end, n=60)
    cache._write_parquet(cache._ohlc_path("X"), old)
    cache._update_meta("ohlc:X", cache._ohlc_path("X"), old)
    new_end = _last_weekday(date.today() - timedelta(days=1))
    new = _ohlc(end=new_end, n=4, seed=5)
    new = new[new.index > pd.Timestamp(old_end)]
    f = _yf(monkeypatch, new)
    out = cache.get_ohlc("X")
    assert f.calls[0][1]["start"] == old_end + timedelta(days=1)
    assert len(out) == 60 + len(new) and out.index.is_monotonic_increasing and out.index.is_unique
    assert cache._meta["ohlc:X"]["to"] == str(new.index.max().date())


def test_get_ohlc_stale_but_nothing_new_returns_existing(cache, monkeypatch):
    old = _ohlc(end=_last_weekday(date.today() - timedelta(days=6)), n=60)
    cache._write_parquet(cache._ohlc_path("X"), old)
    cache._update_meta("ohlc:X", cache._ohlc_path("X"), old)
    _yf(monkeypatch, pd.DataFrame())
    assert len(cache.get_ohlc("X")) == 60


def test_get_ohlc_start_not_before_today_skips_download(cache, monkeypatch):
    df = _ohlc(end=date.today(), n=30)
    cache._write_parquet(cache._ohlc_path("X"), df)
    cache._meta["ohlc:X"] = {
        "updated": (datetime.now() - timedelta(hours=30)).isoformat(),
        "to": str(date.today()),
        "rows": 30,
    }
    f = _yf(monkeypatch, RuntimeError("must not download"))
    assert len(cache.get_ohlc("X")) == 30 and f.calls == []


def test_get_ohlc_force_redownloads_full_history(cache, monkeypatch):
    _yf(monkeypatch, _ohlc(n=60))
    cache.get_ohlc("X")
    f = _yf(monkeypatch, _ohlc(n=80, seed=3))
    out = cache.get_ohlc("X", force=True)
    assert len(out) == 80 and f.calls[0][1]["start"] == date.today() - timedelta(
        days=mc.OHLC_YEARS * 365 + 30
    )


# ---------------------------------------------------------------- get_ohlc_bulk
def test_bulk_splits_cold_incremental_fresh_and_downloads(cache, monkeypatch, capsys):
    fresh = _ohlc(n=60, seed=1)
    cache._write_parquet(cache._ohlc_path("FRESH"), fresh)
    cache._update_meta("ohlc:FRESH", cache._ohlc_path("FRESH"), fresh)
    old = _ohlc(end=_last_weekday(date.today() - timedelta(days=8)), n=60, seed=2)
    cache._write_parquet(cache._ohlc_path("OLD"), old)
    cache._update_meta("ohlc:OLD", cache._ohlc_path("OLD"), old)
    new_end = _last_weekday(date.today() - timedelta(days=1))
    frames = {"COLD": _ohlc(n=60, seed=3), "OLD": _ohlc(end=new_end, n=3, seed=4)}
    f = _yf(monkeypatch, _multi(frames))
    cache.verbose = True
    out = cache.get_ohlc_bulk(["FRESH", "OLD", "COLD"])
    assert set(out) == {"FRESH", "OLD", "COLD"}
    assert len(out["OLD"]) > 60 and len(out["COLD"]) == 60
    starts = {tuple(c[0]): c[1]["start"] for c in f.calls}
    assert starts[("COLD",)] == date.today() - timedelta(days=mc.OHLC_YEARS * 365 + 30)
    assert starts[("OLD",)] == date.today() - timedelta(days=5)
    assert "1 fresh | 1 incremental | 1 cold-start" in capsys.readouterr().out
    assert cache._ohlc_path("COLD").exists() and "ohlc:COLD" in cache._meta


def test_bulk_force_treats_existing_as_incremental(cache, monkeypatch):
    df = _ohlc(n=60)
    cache._write_parquet(cache._ohlc_path("X"), df)
    cache._update_meta("ohlc:X", cache._ohlc_path("X"), df)
    f = _yf(monkeypatch, pd.DataFrame())
    out = cache.get_ohlc_bulk(["X"], force=True)
    assert len(f.calls) == 1 and len(out["X"]) == 60  # nothing new -> existing kept


def test_bulk_unreadable_fresh_file_falls_back_to_cold_download(cache, monkeypatch):
    df = _ohlc(n=60)
    cache._write_parquet(cache._ohlc_path("X"), df)
    cache._update_meta("ohlc:X", cache._ohlc_path("X"), df)
    cache._ohlc_path("X").write_text("corrupt")
    _yf(monkeypatch, _multi({"X": _ohlc(n=70, seed=9)}))
    # a single-ticker bulk call returns a flat frame in real yfinance; the multi frame covers the xs path
    out = cache.get_ohlc_bulk(["X"])
    assert len(out["X"]) == 70


def test_bulk_memory_layer_skips_disk(cache, monkeypatch):
    df = _ohlc(n=60)
    cache._write_parquet(cache._ohlc_path("X"), df)
    cache._update_meta("ohlc:X", cache._ohlc_path("X"), df)
    sentinel = df.iloc[:5]
    cache._mem["X"] = sentinel
    assert cache.get_ohlc_bulk(["X"])["X"] is sentinel


# ---------------------------------------------------------------- _bulk_yf_download
def test_bulk_download_batches_and_sleeps_between(cache, monkeypatch, capsys):
    monkeypatch.setattr(mc, "BATCH_SIZE", 2)
    sleeps = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: sleeps.append(s))
    frames = {t: _ohlc(n=5, seed=i) for i, t in enumerate("ABC")}
    f = _yf(monkeypatch, _multi({"A": frames["A"], "B": frames["B"]}), _multi({"C": frames["C"]}))
    out = cache._bulk_yf_download(list("ABC"), date(2024, 1, 1))
    assert set(out) == {"A", "B", "C"} and [c[0] for c in f.calls] == [["A", "B"], ["C"]]
    assert f.calls[0][1]["threads"] is True and sleeps == [mc.BATCH_SLEEP]
    assert "Bulk download complete: 3/3 tickers" in capsys.readouterr().out


def test_bulk_download_missing_ticker_and_flat_frame_and_empty(cache, monkeypatch, capsys):
    _yf(monkeypatch, _multi({"A": _ohlc(n=5)}))
    out = cache._bulk_yf_download(["A", "ZZ"], date(2024, 1, 1))
    assert list(out) == ["A"] and "OK (1/2)" in capsys.readouterr().out
    _yf(monkeypatch, _ohlc(n=5))  # flat frame -> assigned to the first ticker of the batch
    assert list(cache._bulk_yf_download(["Q"], date(2024, 1, 1))) == ["Q"]
    _yf(monkeypatch, pd.DataFrame())
    assert cache._bulk_yf_download(["A"], date(2024, 1, 1)) == {}
    assert "empty" in capsys.readouterr().out


def test_bulk_download_rate_limit_backoff_and_other_errors(cache, monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: sleeps.append(s))
    f = _yf(
        monkeypatch,
        RuntimeError("429"),
        RuntimeError("Too Many Requests"),
        _multi({"A": _ohlc(n=5)}),
    )
    assert list(cache._bulk_yf_download(["A"], date(2024, 1, 1))) == ["A"]
    assert sleeps == [30, 60] and len(f.calls) == 3
    sleeps.clear()
    f = _yf(monkeypatch, ValueError("boom"))
    assert (
        cache._bulk_yf_download(["A"], date(2024, 1, 1)) == {}
        and len(f.calls) == 1
        and sleeps == []
    )
    assert "ERROR: boom" in capsys.readouterr().out


def test_bulk_download_without_yfinance(cache, monkeypatch):
    monkeypatch.setattr(mc, "_YF_OK", False)
    assert cache._bulk_yf_download(["A"], date(2024, 1, 1)) == {}


# ---------------------------------------------------------------- financials
class _T:
    income_stmt = pd.DataFrame({pd.Timestamp("2024-03-31"): [100.0, 20.0]}, index=["Rev", "NI"])
    balance_sheet = pd.DataFrame({pd.Timestamp("2024-03-31"): [500.0]}, index=["TA"])
    cash_flow = pd.DataFrame()
    financials = None
    quarterly_income_stmt = pd.DataFrame({pd.Timestamp("2024-03-31"): [25.0]}, index=["Rev"])


def test_get_financials_fetch_cache_and_force(cache, monkeypatch):
    calls = []

    class Yf:
        @staticmethod
        def Ticker(t):
            calls.append(t)
            return _T()

    monkeypatch.setattr(mc, "yf", Yf, raising=False)
    monkeypatch.setattr(mc, "_YF_OK", True)
    out = cache.get_financials("TCS.NS")
    assert set(out) == {"income_stmt", "balance_sheet"}  # empty cash_flow omitted
    assert (mc.FUND_DIR / "TCS.NS_annual.parquet").exists() and "fund:TCS.NS:annual" in cache._meta
    again = cache.get_financials("TCS.NS")
    assert calls == ["TCS.NS"] and set(again) == {"income_stmt", "balance_sheet"}
    assert again["income_stmt"].columns[0] == pd.Timestamp("2024-03-31")
    cache.get_financials("TCS.NS", force=True)
    assert calls == ["TCS.NS", "TCS.NS"]
    q = cache.get_financials("TCS.NS", freq="quarterly")
    assert list(q) == ["quarterly_income"] and (mc.FUND_DIR / "TCS.NS_quarterly.parquet").exists()


def test_get_financials_failures(cache, monkeypatch, capsys):
    monkeypatch.setattr(mc, "_YF_OK", False)
    assert cache.get_financials("X") == {}

    class Boom:
        @staticmethod
        def Ticker(t):
            raise RuntimeError("down")

    monkeypatch.setattr(mc, "_YF_OK", True)
    monkeypatch.setattr(mc, "yf", Boom, raising=False)
    cache.verbose = True
    assert (
        cache.get_financials("X") == {}
        and "Financial fetch failed for X: down" in capsys.readouterr().out
    )
    (mc.FUND_DIR / "Y_annual.parquet").write_text("junk")  # corrupt cache file is ignored
    cache._meta["fund:Y:annual"] = {"updated": datetime.now().isoformat()}
    assert cache.get_financials("Y") == {}


# ---------------------------------------------------------------- get_index
def test_get_index_adds_indicators_and_caches(cache, monkeypatch):
    f = _yf(monkeypatch, _ohlc(n=300))
    out = cache.get_index("^NSEI")
    assert {"dma50", "dma200", "dma200_sl", "vol20"} <= set(out.columns)
    assert out["dma50"].iloc[-1] == pytest.approx(out["Close"].tail(50).mean())
    assert out["dma200_sl"].iloc[-1] == pytest.approx(
        out["dma200"].iloc[-1] - out["dma200"].iloc[-6]
    )
    assert (mc.INDEX_DIR / "NSEI.parquet").exists() and cache._meta["index:^NSEI"]["rows"] == 300
    assert f.calls[0][1]["start"] == date.today() - timedelta(days=mc.INDEX_YEARS * 365 + 30)
    f2 = _yf(monkeypatch, RuntimeError("must not download"))
    assert len(cache.get_index("^NSEI")) == 300 and f2.calls == []  # fresh -> disk


def test_get_index_empty_download(cache, monkeypatch):
    _yf(monkeypatch, pd.DataFrame())
    assert cache.get_index("^GSPC").empty and not (mc.INDEX_DIR / "GSPC.parquet").exists()


# ---------------------------------------------------------------- report / purge
def test_report_empty_and_populated(cache, capsys):
    assert cache.report().empty and "Cache is empty" in capsys.readouterr().out
    cache._meta = {
        "ohlc:A": {
            "rows": 5,
            "from": "2024-01-01",
            "to": "2024-02-01",
            "updated": (datetime.now() - timedelta(minutes=5)).isoformat(),
        },
        "fund:A:annual": {"updated": (datetime.now() - timedelta(hours=3)).isoformat()},
        "index:^X": {"updated": (datetime.now() - timedelta(days=2)).isoformat()},
        "ohlc:B": {"updated": "garbage"},
        "ohlc:C": {},
    }
    df = cache.report()
    assert df["Age"].tolist() == ["5m", "3h", "2d", "—", "—"]
    assert df["Rows"].tolist()[:2] == [5, "—"]
    assert "3 OHLC | 1 fundamental | 1 index" in capsys.readouterr().out


def test_purge_single_ticker_and_everything(cache, monkeypatch, capsys):
    _yf(monkeypatch, _ohlc(n=60))
    cache.get_ohlc("A")
    cache.get_ohlc("B")
    (mc.FUND_DIR / "A_annual.parquet").write_text("x")
    cache._meta["fund:A:annual"] = {"updated": "x"}
    cache.purge("A")
    assert not cache._ohlc_path("A").exists() and not (mc.FUND_DIR / "A_annual.parquet").exists()
    assert cache._ohlc_path("B").exists() and set(cache._meta) == {"ohlc:B"}
    cache.purge()
    assert not cache._ohlc_path("B").exists() and cache._meta == {}
    assert "Cache wiped" in capsys.readouterr().out


def test_get_cache_singleton_and_warm_cache(monkeypatch, cache, capsys):
    monkeypatch.setattr(mc, "_CACHE_INSTANCE", None)
    a = mc.get_cache()
    assert a is mc.get_cache()
    calls = []
    fake = types.SimpleNamespace(
        get_index=lambda s: calls.append(("idx", s)),
        get_ohlc_bulk=lambda syms: calls.append(("bulk", list(syms))),
        report=lambda: calls.append(("report",)),
    )
    monkeypatch.setattr(mc, "_CACHE_INSTANCE", fake)
    mc.warm_cache(["A.NS", "B.NS"])
    assert calls == [("idx", "^NSEI"), ("idx", "^GSPC"), ("bulk", ["A.NS", "B.NS"]), ("report",)]
    calls.clear()
    mc.warm_cache(["A"], index_tickers=["^X"])
    assert calls[0] == ("idx", "^X") and "Cache warm" in capsys.readouterr().out
