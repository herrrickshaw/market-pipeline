"""Offline tests for ohlcv_cache (incremental refresh, freshness, closes matrix)."""

import datetime as real_dt
import sys
import types

import pandas as pd
import pytest

import ohlcv_cache as oc


class _Clock:
    """Freeze oc._dt.datetime.now() / oc._dt.date.today()."""

    def __init__(self, monkeypatch, now):
        outer = now

        class FDT(real_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return outer

        class FDate(real_dt.date):
            @classmethod
            def today(cls):
                return outer.date()

        ns = types.SimpleNamespace(datetime=FDT, date=FDate, timedelta=real_dt.timedelta)
        monkeypatch.setattr(oc, "_dt", ns)


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(oc, "SLEEP_BETWEEN", 0.0)
    return tmp_path


# ---------------------------------------------------------------- tiny helpers
def test_path_uppercases_market(cache):
    assert oc._path("us") == cache / "ohlcv_US.parquet"


@pytest.mark.parametrize(
    "sym,sfx,expected",
    [
        ("TCS", ".NS", "TCS.NS"),
        ("TCS.NS", ".NS", "TCS.NS"),
        ("ADS.DE", ".NS", "ADS.DE"),  # already suffixed -> untouched
        ("TCS", "", "TCS"),
    ],
)
def test_yf_symbol(sym, sfx, expected):
    assert oc._yf(sym, sfx) == expected


def test_read_missing_and_corrupt(cache):
    assert oc._read("US").empty
    oc._path("US").write_text("garbage")
    assert oc._read("US").empty


def test_read_parses_dates(cache):
    pd.DataFrame({"Date": ["2024-01-02"], "Symbol": ["A"], "Close": [1.0]}).to_parquet(
        oc._path("US")
    )
    df = oc._read("US")
    assert str(df["Date"].dtype).startswith("datetime64")


def test_download_passes_period_or_start_and_swallows_errors(monkeypatch, capsys):
    seen = []

    def download(syms, **kw):
        seen.append((syms, kw))
        if syms == ["BAD"]:
            raise RuntimeError("rate limited " * 20)
        return "frame"

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    assert oc._download(["A", "B"], "1y", None) == "frame"
    assert oc._download(["A"], "1y", real_dt.date(2024, 1, 2)) == "frame"
    kw0, kw1 = seen[0][1], seen[1][1]
    assert kw0["period"] == "1y" and "start" not in kw0
    assert kw1["start"] == real_dt.date(2024, 1, 2) and "period" not in kw1
    assert kw0["auto_adjust"] is True and kw0["group_by"] == "ticker" and kw0["progress"] is False
    assert oc._download(["BAD"], "1y", None) is None
    assert "batch failed: rate limited" in capsys.readouterr().out


# ---------------------------------------------------------------- _last_expected_session
def _at(y, m, d, hh=0, mm=0):
    return real_dt.datetime(y, m, d, hh, mm)


@pytest.mark.parametrize(
    "now,market,expected",
    [
        # Wed 2026-07-29: India closes 15:30
        (_at(2026, 7, 29, 15, 29), "india", real_dt.date(2026, 7, 28)),  # before close
        (_at(2026, 7, 29, 15, 30), "india", real_dt.date(2026, 7, 29)),  # at close
        (_at(2026, 7, 29, 22, 43), "japan", real_dt.date(2026, 7, 29)),
        (_at(2026, 7, 29, 11, 29), "japan", real_dt.date(2026, 7, 28)),
        (_at(2026, 7, 29, 12, 0), "KR", real_dt.date(2026, 7, 29)),
        (_at(2026, 7, 29, 23, 0), "us", real_dt.date(2026, 7, 28)),  # US never closes "today"
        (_at(2026, 7, 29, 23, 0), "mars", real_dt.date(2026, 7, 28)),  # unknown -> yesterday
        # Monday: previous session is Friday
        (_at(2026, 7, 27, 9, 0), "india", real_dt.date(2026, 7, 24)),
        (_at(2026, 7, 27, 16, 0), "india", real_dt.date(2026, 7, 27)),
        # Saturday / Sunday roll back to Friday even after the close
        (_at(2026, 7, 25, 18, 0), "india", real_dt.date(2026, 7, 24)),
        (_at(2026, 7, 26, 18, 0), "india", real_dt.date(2026, 7, 24)),
        # Tuesday before open: Monday
        (_at(2026, 7, 28, 0, 30), "europe", real_dt.date(2026, 7, 27)),
        (_at(2026, 7, 28, 21, 0), "europe", real_dt.date(2026, 7, 28)),
    ],
)
def test_last_expected_session(monkeypatch, now, market, expected):
    _Clock(monkeypatch, now)
    assert oc._last_expected_session(market) == expected


# ---------------------------------------------------------------- refresh
def _bars(symbols, dates, base=100.0):
    """Long-format frame as stored in the cache."""
    rows = []
    for i, s in enumerate(symbols):
        for j, d in enumerate(dates):
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Open": base + i,
                    "High": base + i + 1,
                    "Low": base + i - 1,
                    "Close": base + i + j,
                    "Volume": 10 + j,
                    "Symbol": s,
                }
            )
    return pd.DataFrame(rows)


def _raw(ysyms, dates, close0=200.0):
    """What yf.download(group_by='ticker') returns."""
    idx = pd.DatetimeIndex(dates)
    if len(ysyms) == 1:
        return pd.DataFrame({c: [close0] * len(idx) for c in oc.COLS}, index=idx)
    cols = {}
    for i, y in enumerate(ysyms):
        for c in oc.COLS:
            cols[(y, c)] = [close0 + i] * len(idx)
    return pd.DataFrame(cols, index=idx)


class _Dl:
    def __init__(self, monkeypatch, responder):
        self.calls = []
        self.responder = responder
        monkeypatch.setattr(oc, "_download", self)

    def __call__(self, ysyms, period, start):
        self.calls.append((list(ysyms), period, start))
        return self.responder(ysyms, period, start)


def test_refresh_cold_seeds_and_persists(monkeypatch, cache, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    dl = _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-27", "2026-07-28"]))
    out = oc.refresh("IN", ["AAA", "BBB"], yf_suffix=".NS", period="6mo")
    assert dl.calls == [(["AAA.NS", "BBB.NS"], "6mo", None)]
    assert len(out) == 4 and set(out["Symbol"]) == {"AAA", "BBB"}  # keyed by bare symbol
    assert oc._path("IN").exists() and len(pd.read_parquet(oc._path("IN"))) == 4
    text = capsys.readouterr().out
    assert "cold — seeding 6mo" in text and "+4 rows -> 4 total" in text


def test_refresh_warm_cache_returns_without_download(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))  # Wed 10:00 -> expected session Tue 28th
    _bars(["A", "B"], ["2026-07-27", "2026-07-28"]).to_parquet(oc._path("IN"))
    dl = _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))
    out = oc.refresh("IN", ["A", "B"])
    assert dl.calls == [] and len(out) == 4
    assert "warm (4 rows, 2 symbols, through 2026-07-28) — no fetch" in capsys.readouterr().out


def test_refresh_quiet_mode_prints_nothing(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A"], ["2026-07-28"]).to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))
    oc.refresh("IN", ["A"], verbose=False)
    assert capsys.readouterr().out == ""


def _lagging_setup(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    cached = pd.concat(
        [_bars(["FRESH"], ["2026-07-28"]), _bars(["STUCK"], ["2026-07-24"])], ignore_index=True
    )
    cached.to_parquet(oc._path("IN"))
    return _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-28"], close0=999.0))


def test_refresh_refetches_only_lagging_symbols(monkeypatch, capsys):
    dl = _lagging_setup(monkeypatch)
    out = oc.refresh("IN", ["FRESH", "STUCK"], yf_suffix=".NS")
    assert [c[0] for c in dl.calls] == [["STUCK.NS"]]
    assert "1 symbol(s) lag the last trading day (2026-07-28)" in capsys.readouterr().out
    assert out[out["Symbol"] == "FRESH"]["Close"].tolist() == [100.0]  # untouched


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: lagged symbols are announced as 're-seeded over period' but the "
    "`else` branch still sets start = global cache max + 1 day, so the fetch starts after "
    "the symbol's gap and can never fill it",
)
def test_refresh_lagging_symbols_fetch_covers_their_gap(monkeypatch):
    dl = _lagging_setup(monkeypatch)
    oc.refresh("IN", ["FRESH", "STUCK"], yf_suffix=".NS")
    ((_, period, start),) = dl.calls
    # either a period-based reseed (start None) or a start no later than the symbol's last bar
    assert start is None or start <= real_dt.date(2026, 7, 24), (period, start)


def test_refresh_symbol_at_expected_session_is_not_lagged(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A"], ["2026-07-28"]).to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))
    oc.refresh("IN", ["A"])


def test_refresh_missing_symbols_seeded_when_cache_date_fresh(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A"], ["2026-07-28"]).to_parquet(oc._path("IN"))
    dl = _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-27", "2026-07-28"]))
    out = oc.refresh("IN", ["A", "NEW"], period="3mo")
    assert dl.calls == [(["NEW"], "3mo", None)]
    assert set(out["Symbol"]) == {"A", "NEW"}
    assert "1 of 2 symbols absent — seeding those over 3mo" in capsys.readouterr().out


def test_refresh_missing_symbols_and_stale_cache_refetches_everything(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A"], ["2026-07-20"]).to_parquet(oc._path("IN"))
    dl = _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-28"]))
    oc.refresh("IN", ["A", "NEW"])
    assert dl.calls == [(["A", "NEW"], "1y", None)]
    assert "cache also stale (through 2026-07-20)" in capsys.readouterr().out


def test_refresh_stale_cache_without_missing_is_incremental(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A", "B"], ["2026-07-20"]).to_parquet(oc._path("IN"))
    dl = _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-21", "2026-07-22"]))
    out = oc.refresh("IN", ["A", "B"])
    assert dl.calls == [(["A", "B"], "1y", real_dt.date(2026, 7, 21))]  # day after cache max
    assert len(out) == 2 + 4
    assert "incremental from 2026-07-21 (cache through 2026-07-20, 2 symbols)" in (
        capsys.readouterr().out
    )


def test_refresh_date_freshness_boundary(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    # cache max == today-1 counts as date-fresh: no incremental fetch (and not lagged)
    _bars(["A"], ["2026-07-28"]).to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))
    oc.refresh("IN", ["A"])


def test_refresh_dedupes_overlap_keeping_latest(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(["A"], ["2026-07-20", "2026-07-21"], base=100.0).to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-21", "2026-07-22"], close0=555.0))
    out = oc.refresh("IN", ["A"])
    row = out[out["Date"] == "2026-07-21"]
    assert len(row) == 1 and row["Close"].iloc[0] == 555.0
    assert len(out) == 3


def test_refresh_batches_by_batch_size(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    monkeypatch.setattr(oc, "BATCH_SIZE", 2)
    dl = _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-28"]))
    syms = ["A", "B", "C", "D", "E"]
    out = oc.refresh("IN", syms)
    assert [c[0] for c in dl.calls] == [["A", "B"], ["C", "D"], ["E"]]
    assert set(out["Symbol"]) == set(syms)
    text = capsys.readouterr().out
    assert "[1/3] 2 symbols" in text and "[3/3] 1 symbols" in text


def test_refresh_sleeps_between_batches_not_after_last(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    monkeypatch.setattr(oc, "BATCH_SIZE", 1)
    monkeypatch.setattr(oc, "SLEEP_BETWEEN", 0.5)
    sleeps = []
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-28"]))
    oc.refresh("IN", ["A", "B", "C"], verbose=False)
    assert sleeps == [0.5, 0.5]


def test_refresh_sleeps_on_failed_batch_too(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    monkeypatch.setattr(oc, "BATCH_SIZE", 1)
    monkeypatch.setattr(oc, "SLEEP_BETWEEN", 0.25)
    sleeps = []
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    _Dl(monkeypatch, lambda y, p, s: None)
    oc.refresh("IN", ["A", "B"], verbose=False)
    assert sleeps == [0.25, 0.25]


def test_refresh_failed_and_empty_batches_return_cached_unchanged(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _Dl(monkeypatch, lambda y, p, s: pd.DataFrame())
    assert oc.refresh("IN", ["A"], verbose=False).empty
    assert not oc._path("IN").exists()


def test_refresh_skips_symbols_missing_from_a_batch_or_all_nan(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))

    def resp(y, p, s):
        full = _raw(y, ["2026-07-28"])
        for c in oc.COLS:
            full[("C.NS", c)] = float("nan")
        return full.drop(columns=[("B.NS", c) for c in oc.COLS])

    _Dl(monkeypatch, resp)
    out = oc.refresh("IN", ["A", "B", "C"], yf_suffix=".NS", verbose=False)
    assert set(out["Symbol"]) == {"A"}  # B absent (KeyError), C all-NaN (dropped)


def test_refresh_single_symbol_flat_frame(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-27", "2026-07-28"]))
    out = oc.refresh("US", ["AAPL"], verbose=False)
    assert out["Symbol"].unique().tolist() == ["AAPL"] and len(out) == 2
    assert set(out.columns) == {"Date", "Open", "High", "Low", "Close", "Volume", "Symbol"}


def test_refresh_persist_failure_still_returns_data(monkeypatch, capsys):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _Dl(monkeypatch, lambda y, p, s: _raw(y, ["2026-07-28"]))
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet", lambda self, *a, **k: (_ for _ in ()).throw(OSError("full"))
    )
    out = oc.refresh("US", ["AAPL"])
    assert len(out) == 1
    assert "could not persist: full" in capsys.readouterr().out


def test_refresh_localizes_timezone_aware_index(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    raw = _raw(["AAPL"], ["2026-07-28"])
    raw.index = raw.index.tz_localize("America/New_York")
    _Dl(monkeypatch, lambda y, p, s: raw)
    out = oc.refresh("US", ["AAPL"], verbose=False)
    assert out["Date"].dt.tz is None


# ---------------------------------------------------------------- get / get_closes
def _warm(monkeypatch, symbols, dates):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _bars(symbols, dates).to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))


def test_get_returns_only_requested_symbols_sorted_and_min_history(monkeypatch):
    _warm(monkeypatch, ["A", "B", "C"], ["2026-07-28", "2026-07-27"])
    out = oc.get("IN", ["A", "B"], verbose=False)
    assert set(out) == {"A", "B"}
    assert list(out["A"].columns) == oc.COLS
    assert out["A"].index.is_monotonic_increasing
    assert oc.get("IN", ["A"], min_history=2, verbose=False).keys() == {"A"}
    assert oc.get("IN", ["A"], min_history=3, verbose=False) == {}


def test_get_empty_when_nothing_cached_or_fetched(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    _Dl(monkeypatch, lambda *a: None)
    assert oc.get("IN", ["A"], verbose=False) == {}
    assert oc.get_closes("IN", ["A"], verbose=False).empty


def test_get_closes_pivots_filters_thin_series_and_sorts(monkeypatch):
    _Clock(monkeypatch, _at(2026, 7, 29, 10, 0))
    cached = pd.concat(
        [
            _bars(["LONG"], ["2026-07-26", "2026-07-27", "2026-07-28"]),
            _bars(["THIN"], ["2026-07-28"]),
            _bars(["OTHER"], ["2026-07-28"]),
        ],
        ignore_index=True,
    )
    cached.to_parquet(oc._path("IN"))
    _Dl(monkeypatch, lambda *a: pytest.fail("downloaded"))
    wide = oc.get_closes("IN", ["LONG", "THIN"], min_history=2, verbose=False)
    assert list(wide.columns) == ["LONG"]  # THIN too short; OTHER not requested
    assert wide.index.is_monotonic_increasing and len(wide) == 3
    default = oc.get_closes("IN", ["LONG", "THIN"], verbose=False)  # min_history=100
    assert default.shape[1] == 0
    both = oc.get_closes("IN", ["LONG", "THIN"], min_history=1, verbose=False)
    assert list(both.columns) == ["LONG", "THIN"] and len(both) == 3
    assert both["THIN"].isna().sum() == 2
