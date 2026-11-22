"""entropic_yield_curve.py -- bear-onset detection, expanding-percentile signals,
event scoring and the least-squares curve fits."""

import math

import numpy as np
import pandas as pd
import pytest

import entropic_yield_curve as EYC

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


# ── _bear_onsets ──────────────────────────────────────────────────────────────
def series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="D"), dtype=float)


def test_bear_onset_requires_twenty_percent_drawdown():
    assert EYC._bear_onsets(series([100, 90, 81.5, 85])) == []  # -18.5%
    on = EYC._bear_onsets(series([100, 90, 79, 85]))  # -21%
    assert on == [pd.Timestamp("2020-01-03")]


def test_bear_episode_ends_only_within_five_percent_of_peak():
    s = series([100, 70, 94, 60, 96, 70])
    # onset day 2; 94 (-6%) does not end it, so the dip to 60 is not a new onset;
    # 96 (-4%) ends it; the next -30% is a second onset
    assert EYC._bear_onsets(s) == [s.index[1], s.index[5]]


def test_bear_recovery_threshold_is_five_percent_off_peak():
    assert len(EYC._bear_onsets(series([100, 70, 95.1, 70]))) == 2  # -4.9%: recovered
    assert len(EYC._bear_onsets(series([100, 70, 94.9, 70]))) == 1  # -5.1%: still in the bear


def test_bear_peak_is_running_max_not_start():
    s = series([100, 200, 150, 159])  # -25% from 200
    assert EYC._bear_onsets(s) == [s.index[2]]


# ── _signal_dates ─────────────────────────────────────────────────────────────
def test_signal_high_tail_fires_on_new_maxima_after_warmup(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    s = series([1, 2, 3, 4, 5, 6])
    f = EYC._signal_dates(s)
    assert list(f) == [False, False, True, True, True, True]


def test_signal_no_lookahead_future_values_do_not_change_past_fires(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    base = series([3, 1, 4, 1, 5, 9, 2, 6])
    extended = series([3, 1, 4, 1, 5, 9, 2, 6, 1000, -1000, 7])
    assert list(EYC._signal_dates(base)) == list(EYC._signal_dates(extended))[: len(base)]


def test_signal_low_tail_fires_on_new_minima(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    s = series([6, 5, 4, 3, 2, 1])
    assert list(EYC._signal_dates(s, low=True)) == [False, False, True, True, True, True]
    assert not EYC._signal_dates(series([1, 2, 3, 4, 5, 6]), low=True).any()
    assert not EYC._signal_dates(series([6, 5, 4, 3, 2, 1])).any()


def test_signal_low_tail_does_not_fire_for_mid_range_values(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    s = series([10, 20, 30, 25])
    assert not EYC._signal_dates(s, low=True).iloc[3]  # 25 is below the 95th pct but not the 5th


@pytest.mark.parametrize("low", [False, True])
def test_signal_ties_with_the_threshold_do_not_fire(monkeypatch, low):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    assert not EYC._signal_dates(series([5.0] * 10), low=low).any()


def test_signal_threshold_expands_over_all_history_not_a_trailing_window(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 3)
    s = series([100, 100, 100, 1, 2, 3, 4, 5, 6])
    assert not EYC._signal_dates(s).iloc[3:].any()  # early 100s keep the threshold high


def test_signal_threshold_uses_configured_percentile(monkeypatch):
    monkeypatch.setattr(EYC, "WARMUP", 5)
    monkeypatch.setattr(EYC, "PCTL", 0.5)
    s = series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    f = EYC._signal_dates(s)
    # median of 1..n is below the newest value, so every post-warmup point fires
    assert list(f[:4]) == [False] * 4 and f[4:].all()
    f2 = EYC._signal_dates(series([10, 9, 8, 7, 6, 5, 4]))
    assert not f2.any()


def test_signal_default_warmup_is_1000_days():
    s = series(np.arange(1200.0))
    f = EYC._signal_dates(s)
    assert not f.iloc[:999].any() and f.iloc[999:].all()


# ── _score ────────────────────────────────────────────────────────────────────
IDX = pd.date_range("2000-01-01", periods=1500, freq="D")


def fires_at(*offsets):
    s = pd.Series(False, index=IDX)
    for o in offsets:
        s.iloc[o] = True
    return s


def test_score_hit_precision_recall():
    onsets = [IDX[100], IDX[900]]
    out = EYC._score(fires_at(0, 500), onsets, IDX)
    # fire@0 -> bear at +100 (hit); fire@500 -> no bear within 252d (false positive);
    # bear@900 is 400d after the last fire: missed
    assert out == {
        "signals": 2,
        "hits": 1,
        "false_positives": 1,
        "bears": 2,
        "bears_caught": 1,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_score_lookahead_window_is_zero_to_252_days_inclusive():
    inside = EYC._score(fires_at(0), [IDX[252]], IDX)
    assert inside["hits"] == 1 and inside["bears_caught"] == 1
    outside = EYC._score(fires_at(0), [IDX[253]], IDX)
    assert outside["hits"] == 0 and outside["bears_caught"] == 0
    assert EYC._score(fires_at(10), [IDX[10]], IDX)["hits"] == 1  # same day counts
    assert EYC._score(fires_at(10), [IDX[9]], IDX)["hits"] == 0  # bear BEFORE the fire


def test_score_declusters_within_63_days_chaining():
    out = EYC._score(fires_at(0, 60, 120, 180), [], IDX)
    assert out["signals"] == 1  # each step <= 63
    out = EYC._score(fires_at(0, 63), [], IDX)
    assert out["signals"] == 1
    out = EYC._score(fires_at(0, 64), [], IDX)
    assert out["signals"] == 2


def test_score_nan_when_nothing_fires_or_no_bears():
    out = EYC._score(fires_at(), [IDX[10]], IDX)
    assert out["signals"] == 0 and math.isnan(out["precision"]) and out["recall"] == 0.0
    out = EYC._score(fires_at(5), [], IDX)
    assert out["signals"] == 1 and out["precision"] == 0.0 and math.isnan(out["recall"])


def test_score_aligned_fires_with_gaps():
    s = pd.Series([True, False, False], index=IDX[:3])
    assert EYC._score(s, [], IDX[:3])["signals"] == 1


@pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=IndexError,
    reason="fires.reindex(index) introduces NaN -> object dtype; pandas>=3 "
    "fillna(False) no longer downcasts to bool, so indexing with the "
    "object array raises",
)
def test_score_fires_covering_only_part_of_the_index():
    s = pd.Series([True, False, False], index=IDX[:3])
    out = EYC._score(s, [], IDX)  # dates beyond the series count as no-fire
    assert out["signals"] == 1
    other = pd.Series(True, index=pd.date_range("1990-01-01", periods=3))
    assert EYC._score(other, [], IDX)["signals"] == 0


def test_score_one_fire_catches_multiple_bears_but_counts_once():
    out = EYC._score(fires_at(0), [IDX[50], IDX[100]], IDX)
    assert out["hits"] == 1 and out["bears_caught"] == 2 and out["recall"] == 1.0


# ── fit ───────────────────────────────────────────────────────────────────────
@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(EYC, "YIELDS", tmp_path / "y.parquet")
    monkeypatch.setattr(EYC, "RC", tmp_path / "rc.parquet")
    return tmp_path


def yield_row(date, a, b0=5.0, maturities=EYC.FIT_MATURITIES, **extra):
    row = {"Date": pd.Timestamp(date), "30 Yr": b0, "20 Yr": np.nan}
    for c in EYC.FIT_MATURITIES:
        row[c] = np.nan
    for c, t in maturities.items():
        row[c] = b0 + a * math.log(t) / (2 * t)
    row.update(extra)
    return row


def write_yields(paths, rows):
    pd.DataFrame(rows).to_parquet(paths / "y.parquet", index=False)


def test_fit_without_cache_returns_1(paths, capsys):
    assert EYC.fit() == 1
    assert "run --fetch first" in capsys.readouterr().out


def test_fit_recovers_slope_and_rc(paths):
    write_yields(paths, [yield_row("2020-01-02", 0.5), yield_row("2020-01-03", -0.8)])
    assert EYC.fit() == 0
    d = pd.read_parquet(paths / "rc.parquet")
    assert list(d["n_mat"]) == [5, 5]
    assert d["a"].to_numpy() == pytest.approx([0.5, -0.8])
    assert d["rc"].to_numpy() == pytest.approx([1 + math.log(0.5), 1 + math.log(1.8)])
    assert d["rmse"].max() < 1e-12 and not d["b0_from_20y"].any()
    assert list(d["b0"]) == [5.0, 5.0]


def test_fit_rc_undefined_at_or_above_one(paths, capsys):
    write_yields(
        paths,
        [
            yield_row("2020-01-02", 1.05),
            yield_row("2020-01-03", 1.2),
            yield_row("2020-01-06", 0.99),
        ],
    )
    EYC.fit()
    d = pd.read_parquet(paths / "rc.parquet")
    assert d["rc"].isna().tolist() == [True, True, False]
    assert "R/C undefined (a >= 1): 2 days (66.7%)" in capsys.readouterr().out


def test_fit_flags_20y_fallback_only_when_the_30y_is_missing(paths):
    both = yield_row("2020-01-02", 0.3)
    both["20 Yr"] = 4.0  # 20Y present as well as 30Y
    write_yields(paths, [both])
    EYC.fit()
    assert not pd.read_parquet(paths / "rc.parquet")["b0_from_20y"].iloc[0]


def test_fit_slope_exactly_one_is_undefined_not_minus_infinity(paths):
    write_yields(paths, [yield_row("2020-01-02", 1.0, b0=0.0)])  # B0 = 0 => a is exactly 1.0
    EYC.fit()
    d = pd.read_parquet(paths / "rc.parquet")
    assert d["a"].iloc[0] == 1.0 and np.isnan(d["rc"].iloc[0])


def test_fit_uses_20y_when_30y_discontinued(paths):
    r1 = yield_row("2004-01-02", 0.3)
    r1["20 Yr"], r1["30 Yr"] = 5.0, np.nan
    r2 = yield_row("2004-01-05", 0.3)
    r3 = yield_row("2004-01-06", 0.3)
    r3["30 Yr"] = r3["20 Yr"] = np.nan  # neither: day skipped
    write_yields(paths, [r1, r2, r3])
    EYC.fit()
    d = pd.read_parquet(paths / "rc.parquet")
    assert list(d["b0_from_20y"]) == [True, False] and len(d) == 2
    assert d["a"].to_numpy() == pytest.approx([0.3, 0.3])


def test_fit_needs_three_maturities_per_day(paths):
    two = {c: EYC.FIT_MATURITIES[c] for c in ("3 Mo", "1 Yr")}
    three = {c: EYC.FIT_MATURITIES[c] for c in ("3 Mo", "1 Yr", "2 Yr")}
    write_yields(
        paths,
        [
            yield_row("2020-01-02", 0.4, maturities=two),
            yield_row("2020-01-03", 0.4, maturities=three),
        ],
    )
    EYC.fit()
    d = pd.read_parquet(paths / "rc.parquet")
    assert len(d) == 1 and d["n_mat"].iloc[0] == 3 and d["a"].iloc[0] == pytest.approx(0.4)


def test_fit_sorts_dates_and_reports_summary(paths, capsys):
    write_yields(paths, [yield_row("2020-01-06", 0.2), yield_row("2020-01-02", 0.6)])
    EYC.fit()
    out = capsys.readouterr().out
    assert "fitted 2 days, 2020-01-02 .. 2020-01-06" in out
    assert "a:    min 0.200  max 0.600  median 0.400" in out
    d = pd.read_parquet(paths / "rc.parquet")
    assert list(d["Date"]) == sorted(d["Date"])


# ── fit_2020 ──────────────────────────────────────────────────────────────────
def row2020(date, A, B, b0=4.0, drop=()):
    r = {"Date": pd.Timestamp(date), "30 Yr": b0, "20 Yr": np.nan}
    for c, t in EYC.MATURITIES_2020.items():
        if c in drop:
            r[c] = np.nan
        else:
            r[c] = b0 + A * math.log(t) / t + B / t
    return r


def test_fit_2020_joint_ols_recovers_parameters(paths):
    write = pd.DataFrame([row2020("2020-03-02", 0.4, 0.9), row2020("2020-03-03", -0.2, 0.1)])
    write.to_parquet(paths / "y.parquet", index=False)
    d = EYC.fit_2020()
    assert d["A"].to_numpy() == pytest.approx([0.4, -0.2])
    assert d["B"].to_numpy() == pytest.approx([0.9, 0.1])
    assert d["rc"].to_numpy() == pytest.approx([1 + math.log(0.6), 1 + math.log(1.2)])
    assert d["sigma"].to_numpy() == pytest.approx([math.exp(-0.9 / 0.6), math.exp(-0.1 / 1.2)])
    assert d["rmse"].max() < 1e-9 and set(d["n_mat"]) == {12}


def test_fit_2020_singularity_is_nan(paths):
    # A well above 1 gives p = 1 - A < 0 with a wide margin. A == 1.0 exactly sits ON the p == 0
    # boundary, where the sign of the least-squares float error (BLAS/numpy build dependent) decides
    # between NaN and a finite rc, which made this test fail on Python 3.13 in CI.
    pd.DataFrame([row2020("2020-03-02", 1.2, 0.5), row2020("2020-03-03", 1.5, 0.5)]).to_parquet(
        paths / "y.parquet", index=False
    )
    d = EYC.fit_2020()
    assert d["rc"].isna().all() and d["sigma"].isna().all()


def test_fit_2020_needs_five_maturities_and_a_long_bond(paths):
    rows = [
        row2020(
            "2020-03-02",
            0.3,
            0.2,
            drop=("1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr"),
        ),  # 4 left: skip
        row2020(
            "2020-03-03", 0.3, 0.2, drop=("1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr")
        ),  # exactly 5: keep
        row2020("2020-03-04", 0.3, 0.2),
    ]
    rows[2]["30 Yr"] = rows[2]["20 Yr"] = np.nan  # no B0: skip
    pd.DataFrame(rows).to_parquet(paths / "y.parquet", index=False)
    d = EYC.fit_2020()
    assert (
        len(d) == 1 and d["n_mat"].iloc[0] == 5 and d["Date"].iloc[0] == pd.Timestamp("2020-03-03")
    )


def test_fit_2020_pinned_sigma_identifies_from_three_month_point(paths):
    pd.DataFrame(
        [row2020("2020-03-02", 0.4, 0.0), row2020("2020-03-03", 0.4, 0.0, drop=("3 Mo",))]
    ).to_parquet(paths / "y.parquet", index=False)
    d = EYC.fit_2020(pin_sigma_at_3m=True)
    assert len(d) == 1  # no 3M point: skipped
    assert d["A"].iloc[0] == pytest.approx(0.4) and d["B"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert d["sigma"].iloc[0] == pytest.approx(1.0)


def test_fit_2020_pinned_sigma_with_nonzero_intercept_matches_closed_form(paths):
    A_true, B_true = 0.3, 0.6
    pd.DataFrame([row2020("2020-03-02", A_true, B_true)]).to_parquet(
        paths / "y.parquet", index=False
    )
    d = EYC.fit_2020(pin_sigma_at_3m=True)
    t = np.array([EYC.MATURITIES_2020[c] for c in EYC.MATURITIES_2020])
    y = 4.0 + A_true * np.log(t) / t + B_true / t
    yy = y - 4.0
    keep = t != 3.0
    x0 = np.log(t) / t
    A = (x0[keep] @ yy[keep]) / (x0[keep] @ x0[keep])
    B = (yy[t == 3.0][0] - A * np.log(3.0) / 3.0) * 3.0
    assert d["A"].iloc[0] == pytest.approx(A) and d["B"].iloc[0] == pytest.approx(B)
    assert d["sigma"].iloc[0] == pytest.approx(math.exp(-B / (1 - A)))


def test_parameters():
    assert (EYC.DD_THRESHOLD, EYC.LOOKAHEAD, EYC.ROLL, EYC.WARMUP, EYC.PCTL) == (
        0.20,
        252,
        60,
        1000,
        0.95,
    )
    assert EYC.MIN_POINTS == 3
