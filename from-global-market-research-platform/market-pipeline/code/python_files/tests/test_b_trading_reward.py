"""Offline tests for trading_reward (min-form aggregation, KL/turnover penalties, deflation)."""

import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

import trading_reward as tr


# ---------------------------------------------------------------- kl_to_equal
def test_kl_zero_at_equal_weight_and_scale_invariant():
    assert tr.kl_to_equal(np.ones(5)) == pytest.approx(0.0)
    assert tr.kl_to_equal([2, 2, 2, 2]) == pytest.approx(0.0)
    assert tr.kl_to_equal([1, 3]) == pytest.approx(tr.kl_to_equal([10, 30]))


def test_kl_maximal_at_one_hot_equals_log_n():
    assert tr.kl_to_equal([1, 0, 0, 0]) == pytest.approx(math.log(4))
    assert tr.kl_to_equal([0, 0, 7]) == pytest.approx(math.log(3))


def test_kl_known_value_and_monotone_in_concentration():
    w = np.array([0.75, 0.25])
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert tr.kl_to_equal(w) == pytest.approx(expected)
    assert (
        tr.kl_to_equal([0.5, 0.3, 0.2])
        < tr.kl_to_equal([0.8, 0.1, 0.1])
        < tr.kl_to_equal([1, 0, 0])
    )


def test_kl_is_nonnegative_and_ignores_tiny_weights():
    assert tr.kl_to_equal([0.5, 0.5, 1e-15]) >= 0
    assert tr.kl_to_equal([1.0, 1e-13]) == pytest.approx(tr.kl_to_equal([1.0, 0.0]), abs=1e-9)


# ---------------------------------------------------------------- turnover
def test_turnover_is_l1_distance():
    assert tr.turnover(np.array([0.5, 0.5]), np.array([0.2, 0.8])) == pytest.approx(0.6)
    assert tr.turnover([1, 0], [0, 1]) == pytest.approx(2.0)
    assert tr.turnover([0.3, 0.7], [0.3, 0.7]) == 0.0


def test_turnover_no_previous_or_length_mismatch_is_zero():
    assert tr.turnover(np.array([1.0, 0.0]), None) == 0.0
    assert tr.turnover(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.0


# ---------------------------------------------------------------- pessimistic
def test_pessimistic_empty_and_nonfinite_filtered():
    assert tr.pessimistic([]) == 0.0
    assert tr.pessimistic([float("nan"), float("inf"), -math.inf]) == 0.0
    assert tr.pessimistic([float("nan"), 0.5, float("inf")]) == 0.5


def test_pessimistic_q_zero_is_min_and_negative_q_too():
    s = [0.4, -0.2, 0.1]
    assert tr.pessimistic(s, 0) == -0.2
    assert tr.pessimistic(s, -1) == -0.2


def test_pessimistic_cvar_takes_mean_of_worst_k_rounded_up():
    s = [0.05, 0.04, 0.03, 0.02, 0.01, -0.10, 0.09, 0.08, 0.07, 0.06]
    # n=10, q=0.3 -> k=3 -> mean of the 3 worst
    assert tr.pessimistic(s, 0.3) == pytest.approx((-0.10 + 0.01 + 0.02) / 3)
    # n=6, q=0.3 -> ceil(1.8)=2
    assert tr.pessimistic([1, 2, 3, 4, 5, 6], 0.3) == pytest.approx(1.5)
    # n=10, q=0.1 -> ceil(1.0)=1 (floating point must not push k to 2)
    assert tr.pessimistic(list(range(10)), 0.1) == pytest.approx(0.0)


def test_pessimistic_never_exceeds_mean_and_default_q_is_cvar_q():
    s = [0.42, -0.03, -0.02, -0.04, -0.01, -0.02]
    assert tr.pessimistic(s) == tr.pessimistic(s, tr.CVAR_Q)
    assert tr.pessimistic(s) <= np.mean(s)
    assert tr.pessimistic(s) == pytest.approx((-0.04 + -0.03) / 2)  # k=ceil(6*0.3)=2


def test_pessimistic_q_one_is_mean():
    assert tr.pessimistic([1, 2, 3], 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------- deflate
def test_deflate_subtracts_expected_max_of_trials(monkeypatch):
    seen = {}

    def emax(trials):
        seen["t"] = trials
        return 0.25

    monkeypatch.setitem(sys.modules, "deflated_sharpe", types.SimpleNamespace(expected_max_sr=emax))
    assert tr.deflate(1.0, (x for x in [0.1, 0.2]), T=5) == pytest.approx(0.75)
    assert seen["t"] == [0.1, 0.2]  # iterables are materialised


def test_deflate_falls_back_to_best_on_import_or_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "deflated_sharpe", None)
    assert tr.deflate(1.0, [0.1, 0.2], 5) == 1.0

    def boom(t):
        raise ValueError("x")

    monkeypatch.setitem(sys.modules, "deflated_sharpe", types.SimpleNamespace(expected_max_sr=boom))
    assert tr.deflate(1.0, [0.1, 0.2], 5) == 1.0


def test_deflate_with_real_module_penalises_a_wide_search():
    rng = np.random.default_rng(0)
    trials = list(rng.normal(0.0, 0.05, 200))
    d = tr.deflate(0.10, trials, 6)
    assert d < 0.10 and math.isfinite(d)
    tight = list(rng.normal(0.0, 0.001, 200))
    assert tr.deflate(0.10, tight, 6) > d  # less dispersion -> smaller penalty


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: with a single trial expected_max_sr returns NaN and deflate() does not "
    "guard it (only exceptions fall back), so best - NaN poisons the reward with NaN",
)
def test_deflate_single_trial_stays_finite():
    assert math.isfinite(tr.deflate(0.10, [0.05], 6))


# ---------------------------------------------------------------- composite
W = np.repeat(0.25, 4)


def test_composite_equal_weight_no_penalties_is_pessimistic_base():
    folds = [0.06, 0.05, 0.05, 0.04, 0.05, 0.05]
    r, info = tr.composite(folds, W)
    assert r == pytest.approx(tr.pessimistic(folds))
    assert info["kl"] == pytest.approx(0.0) and info["turnover"] == 0.0
    assert info["deflated"] == info["pessimistic_base"] and info["reward"] == r
    assert info["mean_base"] == pytest.approx(np.mean(folds))
    assert info["scale"] == pytest.approx(0.05)


def test_composite_penalties_are_scaled_by_typical_fold_magnitude():
    folds = [0.10, 0.10, 0.10]
    w = np.array([1.0, 0.0, 0.0, 0.0])
    r, info = tr.composite(folds, w, beta=0.5)
    assert info["kl_cost"] == pytest.approx(0.5 * 0.10 * math.log(4))
    assert r == pytest.approx(0.10 - 0.5 * 0.10 * math.log(4))
    r2, _ = tr.composite([1.0, 1.0, 1.0], w, beta=0.5)
    assert r2 == pytest.approx(1.0 - 0.5 * 1.0 * math.log(4))  # penalty grows with the scale


def test_composite_turnover_penalty():
    folds = [0.10, 0.10]
    w, prev = np.array([0.5, 0.5]), np.array([0.2, 0.8])
    r, info = tr.composite(folds, w, w_prev=prev, beta=0.0, lam=0.5)
    assert info["turnover"] == pytest.approx(0.6)
    assert r == pytest.approx(0.10 - 0.5 * 0.10 * 0.6)


def test_composite_default_beta_and_lambda():
    assert (tr.BETA_KL, tr.LAMBDA_TURN, tr.CVAR_Q) == (0.30, 0.10, 0.30)
    folds = [0.1, 0.1]
    w = np.array([1.0, 0.0])
    r, info = tr.composite(folds, w, w_prev=np.array([0.0, 1.0]))
    assert r == pytest.approx(0.1 - 0.30 * 0.1 * math.log(2) - 0.10 * 0.1 * 2.0)


def test_composite_scale_floor_when_folds_are_zero_or_empty():
    r, info = tr.composite([0.0, 0.0], np.array([1.0, 0.0]), beta=1.0)
    assert info["scale"] == 1.0 and r == pytest.approx(-math.log(2))
    r, info = tr.composite([], W)
    assert info["scale"] == 1.0 and info["mean_base"] == 0.0 and r == 0.0
    r, info = tr.composite([float("nan")], W)
    assert info["scale"] == 1.0 and info["mean_base"] == 0.0


def test_composite_scale_just_above_floor_is_used():
    _, info = tr.composite([2e-9, 2e-9], W)
    assert info["scale"] == pytest.approx(2e-9)
    _, info = tr.composite([5e-10, 5e-10], W)
    assert info["scale"] == 1.0


def test_composite_uses_deflation_only_when_trials_given(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "deflated_sharpe", types.SimpleNamespace(expected_max_sr=lambda t: 0.02)
    )
    folds = [0.05, 0.05]
    plain, _ = tr.composite(folds, W)
    defl, info = tr.composite(folds, W, trial_scores=[0.1, 0.2], T=2)
    assert plain - defl == pytest.approx(0.02)
    assert info["deflated"] == pytest.approx(info["pessimistic_base"] - 0.02)
    assert tr.composite(folds, W, trial_scores=[])[0] == pytest.approx(plain - 0.02)


def test_composite_ranks_one_regime_hero_below_steady_book():
    hero = [0.42, -0.03, -0.02, -0.04, -0.01, -0.02]  # same mean, far worse worst-fold
    steady = [0.06, 0.05, 0.05, 0.04, 0.05, 0.05]
    assert np.mean(hero) == pytest.approx(np.mean(steady))
    assert min(hero) < min(steady)
    rh, _ = tr.composite(hero, W)
    rs, _ = tr.composite(steady, W)
    assert rs > rh


def test_composite_concentration_must_be_earned():
    folds = [0.06, 0.06, 0.06]
    spread, _ = tr.composite(folds, W)
    conc, _ = tr.composite(folds, np.array([0.97, 0.01, 0.01, 0.01]))
    assert conc < spread


# ---------------------------------------------------------------- benchmark / excess
def test_benchmark_series_us_uses_spy_and_strips_timezone(monkeypatch):
    seen = {}
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="America/New_York")

    class T:
        def __init__(self, t):
            seen["t"] = t

        def history(self, period, auto_adjust):
            seen["kw"] = (period, auto_adjust)
            return pd.DataFrame({"Close": [100.0, 101.0, None, 103.02]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T))
    b = tr.benchmark_series("us")
    assert seen["t"] == "SPY" and seen["kw"] == ("10y", True)
    assert b.index.tz is None
    assert b.iloc[0] == pytest.approx(0.01)


@pytest.mark.parametrize("m,tick", [("JP", "EWJ"), ("KR", "EWY"), ("EU", "VGK"), ("XX", "SPY")])
def test_benchmark_series_ticker_by_market(monkeypatch, m, tick):
    seen = {}

    class T:
        def __init__(self, t):
            seen["t"] = t

        def history(self, period, auto_adjust):
            return pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2024", periods=2))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T))
    tr.benchmark_series(m)
    assert seen["t"] == tick


def test_excess_vs_index_intersects_dates_and_never_fills(monkeypatch):
    bidx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    monkeypatch.setattr(tr, "benchmark_series", lambda m: pd.Series([0.01, 0.02, 0.03], index=bidx))
    sidx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    strat = pd.Series([0.5, 0.05, 0.01, 0.9], index=sidx)
    out = tr.excess_vs_index(strat, "IN")
    assert out.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert out.tolist() == pytest.approx([0.04, -0.01])


def test_excess_vs_index_drops_nan_strategy_days(monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    monkeypatch.setattr(tr, "benchmark_series", lambda m: pd.Series([0.01, 0.01], index=idx))
    out = tr.excess_vs_index(pd.Series([np.nan, 0.03], index=idx), "US")
    assert out.tolist() == pytest.approx([0.02])


def test_excess_vs_index_no_overlap_is_empty(monkeypatch):
    monkeypatch.setattr(
        tr, "benchmark_series", lambda m: pd.Series([0.01], index=pd.to_datetime(["2020-01-01"]))
    )
    assert tr.excess_vs_index(pd.Series([0.1], index=pd.to_datetime(["2024-01-01"])), "US").empty


# ---------------------------------------------------------------- main
def test_main_demo_ranks_hero_below_steady_book(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["trading_reward.py", "--trials", "50", "--seed", "1"])
    assert tr.main() == 0
    out = capsys.readouterr().out
    assert "- OLD reward picks: **concentrated, steady**" in out  # highest mean fold score
    assert "- NEW reward picks: **steady, spread**" in out  # KL penalty + worst-fold logic
    assert out.count("| one-regime hero") == 1 and "carried by ONE fold" in out


def test_main_beta_changes_ranking_weight(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["trading_reward.py", "--beta", "0", "--trials", "30"])
    tr.main()
    out0 = capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["trading_reward.py", "--beta", "5", "--trials", "30"])
    tr.main()
    out5 = capsys.readouterr().out
    assert out0 != out5


def test_composite_scale_is_the_median_absolute_fold_not_the_mean():
    _, info = tr.composite([0.1, 0.1, 0.1, 1.0], W)
    assert info["scale"] == pytest.approx(0.1)  # an outlier fold must not inflate the penalties
    _, info = tr.composite([-0.4, 0.1, 0.2], W)
    assert info["scale"] == pytest.approx(0.2)  # absolute values: median(0.4, 0.1, 0.2)
