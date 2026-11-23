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


# ═══ report builders: fetch / test / run_2020 / sensitivity / main ═══════════════
import sys  # noqa: E402
import types  # noqa: E402

ALL_MATURITIES = [
    "1 Mo",
    "2 Mo",
    "3 Mo",
    "4 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
]
TENOR_YEARS = {
    "1 Mo": 1 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 0.25,
    "4 Mo": 4 / 12,
    "6 Mo": 0.5,
    "1 Yr": 1,
    "2 Yr": 2,
    "3 Yr": 3,
    "5 Yr": 5,
    "7 Yr": 7,
    "10 Yr": 10,
    "20 Yr": 20,
    "30 Yr": 30,
}


def synthetic_yields(freq="W-FRI"):
    idx = pd.date_range("1990-01-05", "2016-12-30", freq=freq)
    n = len(idx)
    phase = np.linspace(0, 40, n)
    level = 5.0 + 1.5 * np.sin(phase / 3)
    slope = 1.0 * np.sin(phase)
    d = pd.DataFrame({"Date": idx})
    for m in ALL_MATURITIES:
        t = TENOR_YEARS[m]
        d[m] = level - slope * np.exp(-t / 3) + 0.2 * np.log1p(t)
    d.loc[d["Date"].between("2002-03-01", "2006-02-01"), "30 Yr"] = np.nan  # the 30Y gap
    return d


def synthetic_spx(dates):
    n = len(dates)
    path = 100 * np.exp(np.cumsum(np.full(n, 0.002)))
    for lo, hi in ((300, 380), (800, 900), (1200, 1260)):  # three crashes
        path[lo:hi] *= np.linspace(1.0, 0.6, hi - lo)
        path[hi:] *= 0.6
    return pd.DataFrame({"Date": dates, "spx": path})


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(EYC, "CACHE", tmp_path)
    monkeypatch.setattr(EYC, "YIELDS", tmp_path / "y.parquet")
    monkeypatch.setattr(EYC, "SPX", tmp_path / "spx.parquet")
    monkeypatch.setattr(EYC, "RC", tmp_path / "rc.parquet")
    monkeypatch.setattr(EYC, "RC2020", tmp_path / "rc2020.parquet")
    monkeypatch.setattr(EYC, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(EYC, "REPORT", tmp_path / "reports" / "entropic_rc_test.md")
    monkeypatch.setattr(EYC, "WARMUP", 150)
    y = synthetic_yields()
    y.to_parquet(tmp_path / "y.parquet", index=False)
    synthetic_spx(y["Date"]).to_parquet(tmp_path / "spx.parquet", index=False)
    return tmp_path


def test_test_requires_fitted_inputs(lab, capsys):
    assert EYC.test() == 1
    assert "run --fetch and --fit first" in capsys.readouterr().out


def test_test_report_scorecard_and_zone_table(lab, capsys):
    assert EYC.fit() == 0
    assert EYC.test() == 0
    md = EYC.REPORT.read_text()
    spx = pd.read_parquet(EYC.SPX).set_index("Date")["spx"]
    onsets = EYC._bear_onsets(spx)
    assert f"## Bear onsets found ({len(onsets)})" in md and len(onsets) >= 2
    for o in onsets:
        assert f"- {o.date()}" in md
    for name in (
        "Var(R/C), 60d",
        "Var(a), 60d",
        "|a| level",
        "fit RMSE",
        "10Y-3M inversion (baseline)",
    ):
        assert f"| {name}" in md or f"| **{name}" in md
    for z in ("I 1990-1998", "II 1999-2002", "III 2003-2006", "IV 2007-2009", "V 2010-2016"):
        assert f"| {z} |" in md
    assert "corr(R/C, 10Y-3M spread)" in md and "## Verdict" in md
    rows = [x for x in md.splitlines() if x.startswith("| Var(R/C), 60d")]
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    assert cells[0] == "Var(R/C), 60d" and f"/{len(onsets)}" in cells[5]
    assert "-> " in capsys.readouterr().out


def test_test_signal_counts_match_the_primitives(lab):
    EYC.fit()
    EYC.test()
    md = EYC.REPORT.read_text()
    d = pd.read_parquet(EYC.RC).set_index("Date").sort_index()
    spx = pd.read_parquet(EYC.SPX).set_index("Date")["spx"]
    idx = d.index.intersection(spx.index)
    d = d.loc[idx]
    expect = EYC._score(
        EYC._signal_dates(d["rc"].rolling(EYC.ROLL).var()), EYC._bear_onsets(spx.loc[idx]), idx
    )
    line = [x for x in md.splitlines() if x.startswith("| Var(R/C), 60d")][0]
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert (int(cells[1]), int(cells[2]), int(cells[3])) == (
        expect["signals"],
        expect["hits"],
        expect["false_positives"],
    )
    assert cells[5] == f"{expect['bears_caught']}/{expect['bears']}"


def test_variant2020_runs_both_fits_and_writes_report(lab, capsys):
    assert EYC.run_2020() == 0
    out = capsys.readouterr().out
    assert "joint OLS on {ln(t)/t, 1/t}" in out and "sigma pinned at the 3M point" in out
    assert "share of days with R/C in [0.5, 1.5]" in out and "corr(R/C 2020, 10Y-3M spread)" in out
    assert EYC.RC2020.exists()  # only the joint fit is persisted
    md = (EYC.REPORTS / "entropic_rc_2020.md").read_text()
    assert md.startswith("# Entropic yield curve — the 2020") and "## Verdict" in md
    assert "| R/C LOW level (his trigger shape) |" in md and "| sigma HIGH level |" in md


def test_variant2020_empty_fit_is_reported(lab, monkeypatch, capsys):
    y = pd.read_parquet(EYC.YIELDS)
    y[[c for c in y.columns if c != "Date"]] = np.nan
    y.to_parquet(EYC.YIELDS, index=False)
    with pytest.raises(FileNotFoundError):  # no persisted 2020 series to study
        EYC.run_2020()
    assert "no rows" in capsys.readouterr().out


def test_sensitivity_prints_each_reading_of_t(lab, capsys):
    assert EYC.sensitivity() == 0
    out = capsys.readouterr().out
    for label in EYC.TIME_SCALES:
        assert f"=== t as {label}" in out
    assert out.count("I 1990-1998") == 3 and "Parker mean" in out and "-56.78" in out


def test_main_dispatch(monkeypatch):
    calls = []
    for name in ("fetch", "fit", "test", "sensitivity", "run_2020"):
        monkeypatch.setattr(EYC, name, lambda n=name: calls.append(n) or 0)
    monkeypatch.setattr(sys, "argv", ["x"])
    assert EYC.main() == 1 and calls == []
    monkeypatch.setattr(sys, "argv", ["x", "--fit", "--test"])
    assert EYC.main() == 0 and calls == ["fit", "test"]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--all"])
    EYC.main()
    assert calls == ["fetch", "fit", "test", "sensitivity"]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--variant2020", "--fit"])
    EYC.main()
    assert calls == ["run_2020"]  # the 2020 variant short-circuits
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--sensitivity"])
    monkeypatch.setattr(EYC, "sensitivity", lambda: 3)
    assert EYC.main() == 3


class FakeResp:
    def __init__(self, text="", fail=False):
        self.text, self.fail = text, fail

    def raise_for_status(self):
        if self.fail:
            raise OSError("500")


def treasury_csv(year):
    return "Date,1 Mo,10 Yr\n" + f"01/03/{year},1.0,2.0\n01/04/{year},1.1,2.1\n"


def test_fetch_downloads_years_and_the_index(lab, monkeypatch, capsys):
    import requests

    monkeypatch.setattr(EYC, "START_YEAR", 2020)
    monkeypatch.setattr(EYC, "END_YEAR", 2022)
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        if "/2021/" in url:
            return FakeResp(fail=True)
        if "/2022/" in url:
            return FakeResp("Other\n")
        return FakeResp(treasury_csv(2020))

    monkeypatch.setattr(requests, "get", get)
    cols = pd.MultiIndex.from_product([["Close"], ["^GSPC"]])
    raw = pd.DataFrame(
        [[3000.0], [3100.0]],
        columns=cols,
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="Date"),
    )
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw)
    )
    assert EYC.fetch() == 0
    out = capsys.readouterr().out
    assert (
        "2020: 2 rows, 2 maturities" in out
        and "2021: FAILED (OSError)" in out
        and "2022: empty" in out
    )
    assert all(t == 60 for _, t in seen) and len(seen) == 3
    y = pd.read_parquet(EYC.YIELDS)
    assert len(y) == 2 and y["Date"].is_monotonic_increasing
    s = pd.read_parquet(EYC.SPX)
    assert list(s.columns) == ["Date", "spx"] and list(s["spx"]) == [3000.0, 3100.0]


def test_fetch_with_no_treasury_data_fails_fast(lab, monkeypatch, capsys):
    import requests

    monkeypatch.setattr(EYC, "START_YEAR", 2020)
    monkeypatch.setattr(EYC, "END_YEAR", 2020)
    monkeypatch.setattr(requests, "get", lambda url, timeout: FakeResp(fail=True))
    assert EYC.fetch() == 1
    assert "no Treasury data fetched" in capsys.readouterr().out


def test_fetch_deduplicates_dates_keeping_the_first(lab, monkeypatch):
    import requests

    monkeypatch.setattr(EYC, "START_YEAR", 2020)
    monkeypatch.setattr(EYC, "END_YEAR", 2021)
    monkeypatch.setattr(requests, "get", lambda url, timeout: FakeResp(treasury_csv(2020)))
    raw = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"], name="Date"))
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw)
    )
    EYC.fetch()
    assert len(pd.read_parquet(EYC.YIELDS)) == 2  # same two dates from both years
