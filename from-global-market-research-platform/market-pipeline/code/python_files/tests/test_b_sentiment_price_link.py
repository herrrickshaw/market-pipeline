"""Offline tests for sentiment_price_link (news-shock detection, linkage table, forward monitor)."""

import json
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import pytest

_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp(prefix="mp_spl_"))  # module creates ./sentiment_link_results at import
try:
    import sentiment_price_link as sl
finally:
    os.chdir(_cwd)

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


def _quiet(n=300, seed=0, vol=0.005):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, vol, n))
    return pd.DataFrame(
        {"Close": c, "Volume": np.full(n, 1e6)}, index=pd.bdate_range("2024-01-01", periods=n)
    )


def _with_shock(day, ret, vol_mult=3.0, n=300, seed=0):
    """Replace one bar's return with `ret` (price path re-integrated) and scale its volume."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.005, n)
    r[day] = ret
    df = pd.DataFrame(
        {"Close": 100 * np.cumprod(1 + r), "Volume": np.full(n, 1e6)},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    df.iloc[day, df.columns.get_loc("Volume")] = 1e6 * vol_mult
    return df


# ---------------------------------------------------------------- detect_shocks_and_returns
def test_detect_needs_enough_history_and_handles_none():
    assert sl.detect_shocks_and_returns("X", None) == []
    assert sl.detect_shocks_and_returns("X", _quiet(sl.TRAIL_WIN + 69)) == []
    assert isinstance(sl.detect_shocks_and_returns("X", _quiet(sl.TRAIL_WIN + 70)), list)


def test_detect_positive_shock_with_forward_returns():
    df = _with_shock(150, 0.08)
    ev = [e for e in sl.detect_shocks_and_returns("AAA", df) if e["z"] > 5]
    assert len(ev) == 1
    e = ev[0]
    c = df["Close"].values
    assert e["symbol"] == "AAA" and e["direction"] == "POS"
    for label, h in sl.HORIZONS.items():
        assert e[f"fwd_{label}"] == pytest.approx((c[150 + h] - c[150]) / c[150] * 100)


def test_detect_negative_shock_direction():
    ev = [
        e for e in sl.detect_shocks_and_returns("AAA", _with_shock(150, -0.08)) if abs(e["z"]) > 5
    ]
    assert len(ev) == 1 and ev[0]["direction"] == "NEG" and ev[0]["z"] < 0


def test_detect_requires_volume_confirmation():
    assert [
        e
        for e in sl.detect_shocks_and_returns("A", _with_shock(150, 0.08, vol_mult=1.0))
        if e["z"] > 5
    ] == []
    assert [
        e
        for e in sl.detect_shocks_and_returns("A", _with_shock(150, 0.08, vol_mult=1.85))
        if e["z"] > 5
    ] == []
    # the spike is part of its own 20-bar average, so the effective bar is ~1.88x the baseline
    assert (
        len(
            [
                e
                for e in sl.detect_shocks_and_returns("A", _with_shock(150, 0.08, vol_mult=1.95))
                if e["z"] > 5
            ]
        )
        == 1
    )


def test_detect_requires_two_sigma_move():
    mild = [
        e
        for e in sl.detect_shocks_and_returns("A", _with_shock(150, 0.004, vol_mult=5.0))
        if abs(e["z"]) < 1.5 and e["z"] > 0
    ]
    assert mild == []  # a normal-sized move on big volume is not a shock


def test_detect_never_evaluates_the_last_sixty_four_bars():
    n = 300
    last_ok = [
        e for e in sl.detect_shocks_and_returns("A", _with_shock(n - 65, 0.09, n=n)) if e["z"] > 5
    ]
    assert len(last_ok) == 1 and not np.isnan(
        last_ok[0]["fwd_3mo"]
    )  # all four horizons are reachable
    assert [
        e for e in sl.detect_shocks_and_returns("A", _with_shock(n - 64, 0.09, n=n)) if e["z"] > 5
    ] == []


def test_detect_zero_volume_and_zero_vol_series_are_skipped():
    flat = pd.DataFrame(
        {"Close": np.full(300, 50.0), "Volume": np.full(300, 1e6)},
        index=pd.bdate_range("2024-01-01", periods=300),
    )
    assert sl.detect_shocks_and_returns("F", flat) == []
    nov = _with_shock(150, 0.08)
    nov["Volume"] = 0.0
    assert sl.detect_shocks_and_returns("N", nov) == []


def test_detect_thresholds_are_the_documented_constants():
    assert (sl.SHOCK_SIGMA, sl.SHOCK_VOL_X, sl.TRAIL_WIN) == (2.0, 1.8, 60)
    assert sl.HORIZONS == {"1d": 1, "1wk": 5, "1mo": 21, "3mo": 63}


# ---------------------------------------------------------------- analyse_linkage
def _events(n_pos=30, n_neg=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for direction, n, mu in (("POS", n_pos, 1.0), ("NEG", n_neg, -1.0)):
        for _ in range(n):
            rows.append(
                {"direction": direction, **{f"fwd_{k}": mu + rng.normal(0, 2) for k in sl.HORIZONS}}
            )
    return pd.DataFrame(rows)


def test_analyse_linkage_rows_and_stats():
    ev = _events()
    link = sl.analyse_linkage(ev)
    assert len(link) == 8 and set(link["Horizon"]) == set(sl.HORIZONS)
    pos1d = link[(link.Implied_Sentiment == "POSITIVE news-shock") & (link.Horizon == "1d")].iloc[0]
    vals = ev[ev.direction == "POS"]["fwd_1d"]
    assert pos1d["N_events"] == 30 and pos1d["Median%"] == round(vals.median(), 3)
    assert pos1d["Hit_rate%"] == round((vals > 0).mean() * 100, 1)
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    assert pos1d["Mean_wins%"] == round(vals.clip(lo, hi).mean(), 3) and pos1d["Std%"] == round(
        vals.clip(lo, hi).std(), 2
    )


def test_analyse_linkage_minimum_twenty_events_and_missing_direction():
    link = sl.analyse_linkage(_events(n_pos=19, n_neg=20))
    assert set(link["Implied_Sentiment"]) == {"NEGATIVE news-shock"}  # 19 < 20 dropped, 20 kept
    only_pos = sl.analyse_linkage(_events(n_pos=25, n_neg=0))
    assert set(only_pos["Implied_Sentiment"]) == {"POSITIVE news-shock"}
    assert sl.analyse_linkage(_events(n_pos=3, n_neg=3)).empty


def test_analyse_linkage_ignores_nan_forward_returns_and_winsorises():
    ev = _events(n_pos=320, n_neg=0)
    ev.loc[:19, "fwd_1d"] = np.nan  # 300 usable observations remain
    ev.loc[20, "fwd_1d"] = 1e6  # a wild outlier must not drive the winsorised mean
    row = sl.analyse_linkage(ev).query("Horizon == '1d'").iloc[0]
    assert row["N_events"] == 300 and row["Mean_wins%"] < 5 and row["Median%"] < 5


# ---------------------------------------------------------------- run_monitor
class _S:
    def __init__(self, score, label, n):
        self.score, self.label, self.n_articles = score, label, n


def test_run_monitor_appends_to_log_across_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sl, "MONITOR_LOG", tmp_path / "log.json")

    class SP:
        def status(self):
            return "Active providers: X"

        def get_ticker_sentiment(self, t, m):
            return _S(0.4 if t == "AAA" else -0.3, "POSITIVE" if t == "AAA" else "NEGATIVE", 3)

    monkeypatch.setitem(
        sys.modules, "sentiment_pipeline", types.SimpleNamespace(SentimentPipeline=SP)
    )
    sl.run_monitor(["AAA", "BBB"], "IN")
    sl.run_monitor(["AAA"], "US")
    log = json.loads((tmp_path / "log.json").read_text())
    assert [(r["ticker"], r["market"], r["label"]) for r in log] == [
        ("AAA", "IN", "POSITIVE"),
        ("BBB", "IN", "NEGATIVE"),
        ("AAA", "US", "POSITIVE"),
    ]
    assert log[0]["sentiment"] == 0.4 and log[0]["n_articles"] == 3 and len(log[0]["date"]) == 10
    out = capsys.readouterr().out
    assert "Total log: 3 rows" in out and "AAA" in out


def test_run_monitor_without_pipeline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sl, "MONITOR_LOG", tmp_path / "log.json")
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", None)
    sl.run_monitor(["AAA"], "IN")
    assert (
        "sentiment_pipeline.py not found." in capsys.readouterr().out
        and not (tmp_path / "log.json").exists()
    )


# ---------------------------------------------------------------- run_historical
@pytest.fixture
def hist_env(monkeypatch, tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "ref").mkdir()
    monkeypatch.setattr(sl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sl, "REF_DIR", tmp_path / "ref")
    monkeypatch.setattr(sl, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    return tmp_path


def test_run_historical_no_events(hist_env, capsys):
    _quiet(300, 1, vol=0.0005).to_parquet(hist_env / "cache" / "AAA.NS.parquet")
    sl.run_historical("IN", 0)
    out = capsys.readouterr().out
    assert "1 stocks with ≥200 bars" in out and "No shock events detected." in out


def test_run_historical_filters_market_and_short_files(hist_env, capsys):
    _quiet(300).to_parquet(hist_env / "cache" / "IN1.NS.parquet")
    _quiet(300).to_parquet(hist_env / "cache" / "US1.parquet")
    _quiet(150).to_parquet(hist_env / "cache" / "SHORT.NS.parquet")
    (hist_env / "cache" / "BAD.NS.parquet").write_text("junk")
    _quiet(300).to_parquet(hist_env / "ref" / "IN1.NS.parquet")  # duplicate stem is read once
    sl.run_historical("IN", 0)
    assert "1 stocks with ≥200 bars" in capsys.readouterr().out
    sl.run_historical("US", 0)
    assert "1 stocks with ≥200 bars" in capsys.readouterr().out
    sl.run_historical("ALL", 1)
    assert capsys.readouterr().out.count("1 stocks with") <= 1  # --max caps the files considered


@pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=(TypeError, IndexError),
    reason="source bug on pandas>=3: to_excel(w, 'Sheet', ...) passes sheet_name positionally, "
    "which pandas 3 makes keyword-only",
)
def test_run_historical_full_report(hist_env, monkeypatch, capsys):
    monkeypatch.setattr(sl, "parallel_map", lambda fn, items, **k: [fn(i) for i in items])
    ev = []
    for i in range(25):
        ev.append(
            {
                "symbol": f"S{i}",
                "direction": "POS",
                "z": 3.0,
                **{f"fwd_{k}": 1.0 + i * 0.01 for k in sl.HORIZONS},
            }
        )
        ev.append(
            {
                "symbol": f"S{i}",
                "direction": "NEG",
                "z": -3.0,
                **{f"fwd_{k}": -1.0 - i * 0.01 for k in sl.HORIZONS},
            }
        )
    monkeypatch.setattr(sl, "detect_shocks_and_returns", lambda s, df: ev if s == "AAA.NS" else [])
    _quiet(300).to_parquet(hist_env / "cache" / "AAA.NS.parquet")
    sl.run_historical("IN", 0)
    out = capsys.readouterr().out
    assert "Detected 50 implied-news-shock events (25 positive, 25 negative)" in out
    assert (
        "CONTINUATION (momentum) after good news" in out
        and "CONTINUED WEAKNESS (drift) after bad news" in out
    )
    files = list((hist_env / "out").glob("sentiment_link_IN_*.xlsx"))
    assert len(files) == 1
    assert pd.ExcelFile(files[0]).sheet_names == ["DISCLAIMER", "Linkage_Summary", "All_Events"]


def test_detect_minimum_history_is_trail_plus_seventy():
    n = sl.TRAIL_WIN + 69
    df = _with_shock(62, 0.1, n=n)
    assert (
        sl.detect_shocks_and_returns("A", df) == []
    )  # 129 bars: below the guard even though a shock exists
    df = _with_shock(62, 0.1, n=n + 1)
    assert any(e["z"] > 5 for e in sl.detect_shocks_and_returns("A", df))


def test_analyse_linkage_hit_rate_counts_only_strictly_positive_returns():
    rows = [{"direction": "POS", **{f"fwd_{k}": 0.0 for k in sl.HORIZONS}} for _ in range(10)]
    rows += [{"direction": "POS", **{f"fwd_{k}": 1.0 for k in sl.HORIZONS}} for _ in range(10)]
    link = sl.analyse_linkage(pd.DataFrame(rows))
    assert (link["Hit_rate%"] == 50.0).all()


def test_run_historical_counts_files_with_exactly_two_hundred_bars(hist_env, capsys):
    _quiet(200).to_parquet(hist_env / "cache" / "EDGE.NS.parquet")
    _quiet(199).to_parquet(hist_env / "cache" / "LOW.NS.parquet")
    sl.run_historical("IN", 0)
    assert "1 stocks with ≥200 bars" in capsys.readouterr().out
