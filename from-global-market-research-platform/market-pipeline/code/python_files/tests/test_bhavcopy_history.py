"""bhavcopy_history.py -- equity-only ISIN filter, store/cache staleness guards, per-day downloads
and the fetch_history() assembly (NSE precedence, dead-date cache, fast paths). All I/O is under
tmp_path; the NSE client, BSE session, market calendar and LMDB store are faked."""

import datetime as dt
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import bhavcopy_history as BH


@pytest.fixture
def env(tmp_path, monkeypatch):
    for n in ("nse", "bse"):
        (tmp_path / n).mkdir()
    monkeypatch.setattr(BH, "CACHE", tmp_path)
    monkeypatch.setattr(BH, "NSE_DIR", tmp_path / "nse")
    monkeypatch.setattr(BH, "BSE_DIR", tmp_path / "bse")
    monkeypatch.setattr(BH, "ASSEMBLED", tmp_path / "assembled_long.parquet")
    monkeypatch.setattr(BH, "CLEANED", tmp_path / "cleaned_long.parquet")
    monkeypatch.setattr(BH, "NODATA_FILE", tmp_path / "no_data_dates.json")
    monkeypatch.setitem(sys.modules, "market_calendar", None)  # weekday-only fallback
    return tmp_path


def weekdays(n_days):
    today = dt.date.today()
    return sorted(
        d for d in (today - dt.timedelta(days=i) for i in range(n_days)) if d.weekday() < 5
    )


def bhav_csv(rows):
    return pd.DataFrame(
        [
            {
                "TckrSymb": s,
                "ISIN": isin,
                "SctySrs": "EQ",
                "FinInstrmTp": "STK",
                "OpnPric": o,
                "HghPric": h,
                "LwPric": lo,
                "ClsPric": c,
                "TtlTradgVol": v,
            }
            for s, isin, o, h, lo, c, v in rows
        ]
    )


# ── _equity_only ──────────────────────────────────────────────────────────────
def test_equity_only_keeps_ine_isins():
    df = pd.DataFrame(
        {
            "ISIN": [
                "INE002A01018",
                "INF204K01K15",
                "IN0020200013",
                "IN9000000001",
                "IN1234",
                None,
            ],
            "x": range(6),
        }
    )
    assert list(BH._equity_only(df)["x"]) == [0]
    assert BH._equity_only(None) is None
    empty = pd.DataFrame({"ISIN": []})
    assert BH._equity_only(empty) is empty
    no_isin = pd.DataFrame({"a": [1]})
    assert BH._equity_only(no_isin) is no_isin  # nothing to judge by: unchanged


# ── cleaned vs assembled ──────────────────────────────────────────────────────
def test_cleaned_behind_assembled(env):
    d1, d2 = pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")
    cached = pd.DataFrame({"Date": [d1, d2]})
    want = {d1, d2}
    assert BH._cleaned_behind_assembled(None, want) is False
    assert BH._cleaned_behind_assembled(pd.DataFrame(), want) is False
    assert (
        BH._cleaned_behind_assembled(cached, want) is False
    )  # no cleaned file: nothing to compare
    pd.DataFrame({"Date": [d1]}).to_parquet(BH.CLEANED, index=False)
    assert BH._cleaned_behind_assembled(cached, want) is True
    assert BH._cleaned_behind_assembled(cached, {d1}) is False  # d2 not wanted: rolling window drop
    pd.DataFrame({"Date": [d1, d2]}).to_parquet(BH.CLEANED, index=False)
    assert BH._cleaned_behind_assembled(cached, want) is False
    BH.CLEANED.write_text("garbage")
    assert BH._cleaned_behind_assembled(cached, want) is True  # unreadable -> rebuild


# ── LMDB guards ───────────────────────────────────────────────────────────────
class FakeStore:
    def __init__(self, data):
        self.data = data

    def symbols(self):
        return list(self.data)

    def get(self, s):
        v = self.data[s]
        if isinstance(v, Exception):
            raise v
        return v


def closes_ending(date, n=3):
    idx = pd.date_range(end=date, periods=n)
    return pd.DataFrame({"Close": np.arange(n, dtype=float) + 1}, index=idx)


def install_store(monkeypatch, data):
    monkeypatch.setitem(sys.modules, "bhavcopy_store", FakeStore(data))


def test_lmdb_max_date_is_the_median_of_symbol_last_bars(monkeypatch):
    install_store(
        monkeypatch,
        {
            "A": closes_ending("2026-07-20"),
            "B": closes_ending("2026-07-20"),
            "C": closes_ending("2026-07-01"),
        },
    )
    assert BH._lmdb_max_date() == pd.Timestamp("2026-07-20")
    install_store(monkeypatch, {"A": closes_ending("2026-07-20"), "B": closes_ending("2026-07-10")})
    assert BH._lmdb_max_date() == pd.Timestamp("2026-07-15")  # median of two = midpoint


def test_lmdb_max_date_restricts_to_one_market_and_skips_bad_symbols(monkeypatch):
    install_store(
        monkeypatch,
        {
            "a": closes_ending("2026-07-20"),
            "US1": closes_ending("2025-01-01"),
            "ERR": OSError("x"),
            "EMPTY": pd.DataFrame({"Close": []}),
            "NOCLOSE": pd.DataFrame({"Open": [1.0]}, index=[pd.Timestamp("2026-01-01")]),
            "NONE": None,
        },
    )
    assert BH._lmdb_max_date(only={"A", "ERR", "EMPTY", "NOCLOSE", "NONE"}) == pd.Timestamp(
        "2026-07-20"
    )
    assert BH._lmdb_max_date(only={"ZZZ"}) is None
    assert BH._lmdb_max_date(only={"ERR"}) is None


def test_lmdb_max_date_without_symbols_or_store(monkeypatch):
    install_store(monkeypatch, {})
    assert BH._lmdb_max_date() is None
    monkeypatch.setitem(sys.modules, "bhavcopy_store", None)
    assert BH._lmdb_max_date() is None


def test_lmdb_behind_cleaned(env, monkeypatch):
    assert BH._lmdb_behind_cleaned() is False  # no cleaned parquet
    pd.DataFrame({"Date": ["2026-07-20", "2026-07-19"], "Symbol": ["a", "B"]}).to_parquet(
        BH.CLEANED, index=False
    )
    install_store(monkeypatch, {"A": closes_ending("2026-07-20"), "B": closes_ending("2026-07-20")})
    assert BH._lmdb_behind_cleaned() is False
    install_store(monkeypatch, {"A": closes_ending("2026-07-19"), "B": closes_ending("2026-07-19")})
    assert BH._lmdb_behind_cleaned() is True
    install_store(monkeypatch, {"X": closes_ending("2026-07-20")})
    assert BH._lmdb_behind_cleaned() is True  # none of India's symbols stored
    BH.CLEANED.write_text("junk")
    assert BH._lmdb_behind_cleaned() is False  # unreadable: do not rebuild


# ── per-day downloads ─────────────────────────────────────────────────────────
class FakeNSE:
    def __init__(self, tmp, frames):
        self.tmp, self.frames, self.calls = tmp, frames, []

    def equityBhavcopy(self, date):
        self.calls.append(date)
        key = date.date()
        if key not in self.frames:
            raise FileNotFoundError("holiday")
        p = self.tmp / f"raw_{key}.csv"
        self.frames[key].to_csv(p, index=False)
        return p


def test_nse_day_downloads_filters_equity_and_caches(env):
    day = dt.date(2026, 7, 20)
    raw = bhav_csv(
        [("TCS", "INE467B01029", 1, 2, 0.5, 1.5, 100), ("GOLDETF", "INF204K01K15", 1, 1, 1, 1, 5)]
    )
    raw.loc[1, "SctySrs"] = "BE"
    raw.columns = [f" {c} " for c in raw.columns]  # padded headers are stripped
    client = FakeNSE(env, {day: raw})
    out = BH._nse_day(day, client)
    assert list(out["TckrSymb"]) == ["TCS"]
    assert (env / "nse" / "20260720.csv").exists() and not (
        env / "raw_2026-07-20.csv"
    ).exists()  # temp removed
    assert client.calls == [dt.datetime(2026, 7, 20)]
    again = BH._nse_day(day, client)  # served from disk
    assert list(again["TckrSymb"]) == ["TCS"] and len(client.calls) == 1


def test_nse_day_cached_file_is_re_filtered_and_failures_are_none(env):
    day = dt.date(2026, 7, 20)
    pd.DataFrame({"TckrSymb": ["A", "B"], "ISIN": ["INE000000001", "INF000000002"]}).to_csv(
        env / "nse" / "20260720.csv", index=False
    )
    assert list(BH._nse_day(day, FakeNSE(env, {}))["TckrSymb"]) == ["A"]
    assert BH._nse_day(dt.date(2026, 7, 21), FakeNSE(env, {})) is None


class FakeSession:
    def __init__(self, status=200, body=b"", boom=False):
        self.status, self.body, self.boom, self.urls = status, body, boom, []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        if self.boom:
            raise OSError("down")
        return types.SimpleNamespace(status_code=self.status, content=self.body)


def test_bse_day_download_url_filter_and_cache(env):
    day = dt.date(2026, 7, 20)
    raw = bhav_csv(
        [("500325", "INE002A01018", 1, 2, 1, 2, 9), ("GSEC", "IN0020200013", 1, 1, 1, 1, 1)]
    )
    raw.loc[1, "FinInstrmTp"] = "STK"
    raw.loc[0, "FinInstrmTp"] = "STK"
    buf = raw.to_csv(index=False).encode() + b" " * 1000
    s = FakeSession(body=buf)
    out = BH._bse_day(day, s)
    assert list(out["TckrSymb"]) == ["500325"]
    assert s.urls[0][0].endswith("BhavCopy_BSE_CM_0_0_0_20260720_F_0000.CSV") and s.urls[0][1] == 25
    assert (env / "bse" / "20260720.csv").exists()
    assert BH._bse_day(day, FakeSession(boom=True)) is not None  # disk cache wins over network


def test_bse_day_failures(env):
    day = dt.date(2026, 7, 22)
    assert BH._bse_day(day, FakeSession(status=404, body=b"x" * 2000)) is None
    assert (
        BH._bse_day(day, FakeSession(status=200, body=b"x" * 999)) is None
    )  # too small to be real
    assert BH._bse_day(day, FakeSession(boom=True)) is None
    assert not list((env / "bse").iterdir())


# ── fetch_history ─────────────────────────────────────────────────────────────
def install_nse(monkeypatch, tmp, frames):
    client = FakeNSE(tmp, frames)
    client.folder = None

    def ctor(download_folder):
        client.folder = download_folder
        return client

    client.exit = lambda: setattr(client, "exited", True)
    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=ctor))
    return client


def no_store(monkeypatch):
    built = []
    monkeypatch.setitem(
        sys.modules,
        "bhavcopy_store",
        types.SimpleNamespace(
            build=lambda hist, verbose=True: built.append(sorted(hist)) or len(hist)
        ),
    )
    return built


def test_fetch_history_assembles_cleans_and_persists(env, monkeypatch):
    days = weekdays(4)
    frames = {
        d: bhav_csv(
            [
                ("AAA", "INE000000001", 10 + i, 12 + i, 9 + i, 11 + i, 1000),
                ("BBB", "INE000000002", 20, 21, 19, 20.5, 500),
            ]
        )
        for i, d in enumerate(days)
    }
    client = install_nse(monkeypatch, env, frames)
    built = no_store(monkeypatch)
    out = BH.fetch_history(n_days=4, exchanges=("NSE",), min_bars=1, verbose=False)
    assert sorted(out) == ["AAA", "BBB"]
    assert len(out["AAA"]) == len(days) and list(out["AAA"].columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert out["AAA"]["Close"].iloc[-1] == pytest.approx(11 + len(days) - 1)
    assert BH.ASSEMBLED.exists() and BH.CLEANED.exists() and built == [["AAA", "BBB"]]
    assert getattr(client, "exited", False) and client.folder == str(env / "nse")
    asm = pd.read_parquet(BH.ASSEMBLED)
    assert set(asm["_exch"]) == {"NSE"} and len(asm) == 2 * len(days)


def test_fetch_history_min_bars_filter(env, monkeypatch):
    days = weekdays(4)
    frames = {d: bhav_csv([("AAA", "INE000000001", 1, 2, 1, 2, 10)]) for d in days}
    frames[days[0]] = bhav_csv(
        [("AAA", "INE000000001", 1, 2, 1, 2, 10), ("RARE", "INE000000009", 1, 2, 1, 2, 10)]
    )
    install_nse(monkeypatch, env, frames)
    no_store(monkeypatch)
    out = BH.fetch_history(n_days=4, exchanges=("NSE",), min_bars=2, verbose=False)
    assert sorted(out) == ["AAA"]  # RARE has a single bar


def test_fetch_history_records_dead_dates_but_never_today(env, monkeypatch):
    days = weekdays(6)
    live = days[:-2]
    frames = {d: bhav_csv([("AAA", "INE000000001", 1, 2, 1, 2, 10)]) for d in live}
    install_nse(monkeypatch, env, frames)
    no_store(monkeypatch)
    BH.fetch_history(n_days=6, exchanges=("NSE",), min_bars=1, verbose=False)
    dead = json.loads(BH.NODATA_FILE.read_text())
    expect = {d.strftime("%Y-%m-%d") for d in days[-2:] if d != dt.date.today()}
    assert set(dead) == expect and dead == sorted(dead)
    if dt.date.today() in days[-2:]:
        assert dt.date.today().strftime("%Y-%m-%d") not in dead


def test_fetch_history_skips_known_dead_dates_on_the_next_run(env, monkeypatch, capsys):
    days = weekdays(5)
    frames = {d: bhav_csv([("AAA", "INE000000001", 1, 2, 1, 2, 10)]) for d in days[:-1]}
    client = install_nse(monkeypatch, env, frames)
    no_store(monkeypatch)
    BH.fetch_history(n_days=5, exchanges=("NSE",), min_bars=1, verbose=False)
    first_calls = len(client.calls)
    BH.fetch_history(n_days=5, exchanges=("NSE",), min_bars=1, verbose=True)
    out = capsys.readouterr().out
    assert len(client.calls) <= first_calls + 1  # at most today again
    assert "assembled cache:" in out and "dead dates skipped" in out


def test_fetch_history_fast_path_serves_cleaned_cache(env, monkeypatch, capsys):
    days = weekdays(5)
    rows = []
    for d in days:
        for s, px in (("AAA", 10.0), ("BBB", 20.0)):
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Symbol": s,
                    "Open": px,
                    "High": px,
                    "Low": px,
                    "Close": px,
                    "Volume": 5,
                    "_exch": "NSE",
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(BH.ASSEMBLED, index=False)
    df.drop(columns="_exch").to_parquet(BH.CLEANED, index=False)
    install_nse(monkeypatch, env, {})
    install_store(monkeypatch, {"AAA": closes_ending(days[-1]), "BBB": closes_ending(days[-1])})
    out = BH.fetch_history(n_days=5, exchanges=("NSE",), min_bars=3, verbose=True)
    assert sorted(out) == ["AAA", "BBB"] and len(out["AAA"]) == len(days)
    assert "fast path (cleaned cache)" in capsys.readouterr().out
    assert (
        BH.fetch_history(n_days=5, exchanges=("NSE",), min_bars=len(days) + 1, verbose=False) == {}
    )


def test_fetch_history_seeds_assembled_from_cleaned_and_syncs_a_lagging_store(
    env, monkeypatch, capsys
):
    days = weekdays(8)
    # the newest weekday has no file yet (unpublished): one date is "missing" but nothing downloads
    rows = [
        {
            "Date": pd.Timestamp(d),
            "Symbol": "AAA",
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Volume": 5,
        }
        for d in days[:-1]
    ]
    pd.DataFrame(rows).to_parquet(BH.CLEANED, index=False)
    install_nse(monkeypatch, env, {})
    built, verbosity = [], []
    old = closes_ending(pd.Timestamp(days[0]) - pd.Timedelta(days=10))
    monkeypatch.setitem(
        sys.modules,
        "bhavcopy_store",
        types.SimpleNamespace(
            symbols=lambda: ["AAA"],
            get=lambda s: old,
            build=lambda hist, verbose=True: (
                built.append(sorted(hist)),
                verbosity.append(verbose),
                len(hist),
            )[2],
        ),
    )
    out = BH.fetch_history(n_days=8, exchanges=("NSE",), min_bars=1, verbose=True)
    assert sorted(out) == ["AAA"] and BH.ASSEMBLED.exists()
    assert pd.read_parquet(BH.ASSEMBLED)["_exch"].eq("NSE").all()
    text = capsys.readouterr().out
    assert "seeded assembled cache" in text and built == [["AAA"]] and "LMDB was behind" in text
    assert verbosity == [False]  # the sync is quiet; fetch_history reports it itself


def test_fetch_history_nse_precedence_and_bse_only_dates(env, monkeypatch):
    days = weekdays(4)
    nse_frames = {days[-1]: bhav_csv([("AAA", "INE000000001", 10, 10, 10, 10, 1)])}
    client = install_nse(monkeypatch, env, nse_frames)
    no_store(monkeypatch)

    def bse(day, sess):
        if day == days[-1]:
            return bhav_csv([("AAA", "INE000000001", 99, 99, 99, 99, 1)])  # collides with NSE
        if day == days[-2]:
            return bhav_csv([("AAA", "INE000000001", 7, 7, 7, 7, 1)])  # BSE-only date
        return None

    monkeypatch.setattr(BH, "_bse_day", bse)
    out = BH.fetch_history(n_days=4, exchanges=("NSE", "BSE"), min_bars=1, verbose=False)
    closes = out["AAA"]["Close"]
    assert closes.iloc[-1] == 10.0  # NSE wins the colliding day
    assert closes.loc[pd.Timestamp(days[-2])] == 7.0  # BSE-only day retained
    assert len(client.calls) >= 1


def test_fetch_history_returns_empty_when_nothing_is_available(env, monkeypatch):
    install_nse(monkeypatch, env, {})
    no_store(monkeypatch)
    assert BH.fetch_history(n_days=3, exchanges=("NSE",), verbose=False) == {}


def test_fetch_history_ignores_corrupt_assembled_cache_and_nodata_file(env, monkeypatch):
    days = weekdays(3)
    BH.ASSEMBLED.write_text("garbage")
    BH.NODATA_FILE.write_text("{not json")
    install_nse(
        monkeypatch, env, {d: bhav_csv([("AAA", "INE000000001", 1, 2, 1, 2, 3)]) for d in days}
    )
    no_store(monkeypatch)
    out = BH.fetch_history(n_days=3, exchanges=("NSE",), min_bars=1, verbose=False)
    assert sorted(out) == ["AAA"]


def test_get_symbol_from_the_assembled_cache(env):
    assert BH.get_symbol("AAA") is None
    today = pd.Timestamp(dt.date.today())
    rows = []
    for off, exch, px in ((1, "BSE", 5.0), (1, "NSE", 6.0), (2, "BSE", 4.0), (900, "NSE", 1.0)):
        rows.append(
            {
                "Date": today - pd.Timedelta(days=off),
                "Symbol": "AAA",
                "Open": px,
                "High": px,
                "Low": px,
                "Close": px,
                "Volume": 3,
                "_exch": exch,
            }
        )
    rows.append(
        {
            "Date": today - pd.Timedelta(days=1),
            "Symbol": "OTHER",
            "Open": 9,
            "High": 9,
            "Low": 9,
            "Close": 9,
            "Volume": 1,
            "_exch": "NSE",
        }
    )
    pd.DataFrame(rows).to_parquet(BH.ASSEMBLED, index=False)
    g = BH.get_symbol("AAA")
    assert list(g["Close"]) == [4.0, 6.0]  # old bar cut, NSE wins the tie
    assert BH.get_symbol("NOPE") is None
    assert len(BH.get_symbol("AAA", n_days=1000)) == 3


def test_fetch_history_reports_new_trading_days_per_exchange(env, monkeypatch, capsys):
    days = weekdays(4)
    install_nse(
        monkeypatch, env, {days[-1]: bhav_csv([("AAA", "INE000000001", 10, 10, 10, 10, 1)])}
    )
    no_store(monkeypatch)

    def bse(day, sess):
        if day in (days[-1], days[-2]):
            return bhav_csv([("AAA", "INE000000001", 7, 7, 7, 7, 1)])
        return None

    monkeypatch.setattr(BH, "_bse_day", bse)
    BH.fetch_history(n_days=4, exchanges=("NSE", "BSE"), min_bars=1, verbose=True)
    text = capsys.readouterr().out
    assert "NSE: +1 new trading days" in text and "BSE: +2 new trading days" in text
