"""E[max SR] and DSR in deflated_sharpe.py (Bailey & Lopez de Prado 2014)."""
import math

import numpy as np
import pytest
from scipy import stats

import deflated_sharpe as ds

GAMMA = 0.5772156649015329


def test_expected_max_sr_hand_computed():
    trials = [0.1, 0.2, 0.3, 0.4]            # std(ddof=1) = 0.1290994...
    v = np.std(trials, ddof=1)
    z = (1 - GAMMA) * stats.norm.ppf(1 - 1 / 4) + GAMMA * stats.norm.ppf(1 - 1 / (4 * math.e))
    assert ds.expected_max_sr(trials) == pytest.approx(v * z, rel=1e-12)


def test_expected_max_sr_ignores_non_finite_trials():
    assert ds.expected_max_sr([0.1, np.nan, 0.2, np.inf, 0.3, 0.4]) == \
        pytest.approx(ds.expected_max_sr([0.1, 0.2, 0.3, 0.4]))


@pytest.mark.parametrize("trials", [[], [0.3], [0.3, np.nan]])
def test_fewer_than_two_trials_is_undefined_not_a_pass(trials):
    assert math.isnan(ds.expected_max_sr(trials))
    assert math.isnan(ds.dsr(0.3, trials, 100))


def test_short_sample_is_undefined():
    assert math.isnan(ds.dsr(0.3, [0.1, 0.2, 0.3], T=2))


def test_dsr_hand_computed_and_bounded():
    trials = [0.02, 0.05, 0.08, 0.11]
    emax = ds.expected_max_sr(trials)
    T, sr = 260, 0.11
    expected = stats.norm.cdf((sr - emax) * math.sqrt(T - 1) / math.sqrt(1 + 0.5 * sr ** 2))
    got = ds.dsr(sr, trials, T)
    assert got == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= got <= 1.0


def test_identical_trials_give_zero_dispersion_benchmark():
    # no spread across trials -> E[max] = 0 -> DSR is P(SR > 0)
    assert ds.expected_max_sr([0.1, 0.1, 0.1]) == pytest.approx(0.0)


def test_negative_skew_and_fat_tails_lower_dsr_for_positive_sr():
    trials = [0.01, 0.03, 0.05, 0.07]
    base = ds.dsr(0.12, trials, 300)
    fat = ds.dsr(0.12, trials, 300, skew=-1.0, kurt=8.0)
    assert fat < base
