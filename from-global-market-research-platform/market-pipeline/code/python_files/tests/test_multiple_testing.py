"""Benjamini-Hochberg FDR and the deflated-Sharpe helper in multiple_testing.py.

Expected values are hand-computed from the BH step-up rule: sort the m p-values,
find the largest rank k with p_(k) <= k/m * q, reject the k smallest.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import multiple_testing as mt


class TestBhFdr:
    def test_textbook_case_rejects_three_of_five(self):
        # thresholds at q=.05, m=5: .01 .02 .03 .04 .05
        # sorted p: .001 .02 .03 .045 .2 -> ranks 1..3 pass, .045 > .04 fails
        p = [0.2, 0.001, 0.045, 0.03, 0.02]
        out = mt.bh_fdr(p, q=0.05)
        assert out.tolist() == [False, True, False, True, True]

    def test_step_up_rescues_a_smaller_p_that_fails_its_own_threshold(self):
        # .03 > 1/3*.05 on its own, but rank 3 (.032 <= .05) passes, so
        # every p-value at or below it is rejected (step-UP, not step-down).
        out = mt.bh_fdr([0.03, 0.031, 0.032], q=0.05)
        assert out.all()

    def test_nothing_survives_returns_all_false_with_input_index(self):
        s = pd.Series([0.5, 0.9, 0.7], index=["a", "b", "c"])
        out = mt.bh_fdr(s, q=0.10)
        assert list(out.index) == ["a", "b", "c"]
        assert not out.any()

    def test_nan_pvalues_are_excluded_from_m_and_never_survive(self):
        # with the NaN excluded m=2: thresholds .05, .10 at q=.10
        out = mt.bh_fdr([0.04, np.nan, 0.09], q=0.10)
        assert out.tolist() == [True, False, True]

    def test_ties_at_the_cutoff_all_survive(self):
        out = mt.bh_fdr([0.02, 0.02, 0.5], q=0.05)   # thresholds .0167 .0333 .05
        assert out.tolist() == [True, True, False]

    def test_preserves_non_default_index_alignment(self):
        s = pd.Series([0.001, 0.8], index=[10, 3])
        out = mt.bh_fdr(s, q=0.05)
        assert out.loc[10] and not out.loc[3]

    def test_single_hypothesis_reduces_to_plain_threshold(self):
        assert mt.bh_fdr([0.049], q=0.05).tolist() == [True]
        assert mt.bh_fdr([0.051], q=0.05).tolist() == [False]


class TestDeflatedSharpe:
    def test_matches_closed_form_for_hand_parameters(self):
        sr, n, T = 0.3, 10, 120
        v = (1 + 0.5 * sr ** 2) / (T - 1)            # normal returns, kurt=3
        g = 0.5772156649
        z = (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
        expected = stats.norm.cdf((sr - math.sqrt(v) * z) * math.sqrt(T - 1)
                                  / math.sqrt(1 + 0.5 * sr ** 2))
        assert mt.deflated_sharpe(sr, n, T) == pytest.approx(expected, rel=1e-9)

    def test_more_trials_deflate_more(self):
        a = mt.deflated_sharpe(0.15, 5, 250)
        b = mt.deflated_sharpe(0.15, 500, 250)
        assert 0 <= b < a <= 1

    def test_agrees_with_deflated_sharpe_module_given_the_same_trial_variance(self):
        import deflated_sharpe as ds
        trials = np.array([0.05, -0.02, 0.11, 0.08, 0.01, 0.03, -0.04, 0.09])
        T = 200
        v = trials.std(ddof=1) ** 2
        a = mt.deflated_sharpe(trials.max(), len(trials), T, sr_var=v)
        b = ds.dsr(trials.max(), trials, T)
        assert a == pytest.approx(b, rel=1e-9)

    def test_single_trial_has_no_selection_bias_so_dsr_is_plain_psr(self):
        # With ONE trial nothing was selected: the benchmark is SR*=0 and DSR
        # must equal the probabilistic Sharpe ratio. A losing strategy
        # (SR=-0.5 over 100 obs) must score near 0, never a certain pass.
        sr, T = -0.5, 100
        psr = stats.norm.cdf(sr * math.sqrt(T - 1) / math.sqrt(1 + 0.5 * sr ** 2))
        assert mt.deflated_sharpe(sr, 1, T) == pytest.approx(psr, rel=1e-9)
        assert mt.deflated_sharpe(sr, 1, T) < 0.05
