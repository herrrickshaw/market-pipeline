"""Offline tests for piotroski_plus: the 9-point F-score, the ROCE block, weights, verdict."""

import math

import pandas as pd
import pytest

import piotroski_plus as pp

YEARS = ["2024", "2023", "2022", "2021", "2020"]  # newest first


class _Ticker:
    def __init__(self, inc=None, bal=None, cfs=None):
        self.income_stmt, self.balance_sheet, self.cashflow = inc, bal, cfs


def _df(rows, years=YEARS):
    df = pd.DataFrame({k: pd.Series(v, index=years[: len(v)]) for k, v in rows.items()}).T
    return df.reindex(columns=[y for y in years if y in df.columns])  # newest first, like yfinance


def _good(**over):
    """Every Piotroski test and every ROCE test passes. `over` replaces a statement line's
    values by name (None removes the line)."""
    stmts = {
        "inc": {
            "EBIT": [150, 140, 130, 120, 110],
            "Net Income": [100, 80, 70, 60, 50],
            "Gross Profit": [500, 400],
            "Total Revenue": [1000, 900],
        },
        "bal": {
            "Total Assets": [1000] * 5,
            "Current Liabilities": [200] * 5,
            "Current Assets": [500, 400],
            "Long Term Debt": [100, 150],
            "Ordinary Shares Number": [100, 100],
            "Cash And Cash Equivalents": [100, 100],
        },
        "cfs": {"Operating Cash Flow": [120, 100]},
    }
    for line, values in over.items():
        owner = next(k for k, d in stmts.items() if line in d)
        if values is None:
            del stmts[owner][line]
        else:
            stmts[owner][line] = values
    return _Ticker(_df(stmts["inc"]), _df(stmts["bal"]), _df(stmts["cfs"]))


def _s(**over):
    return pp.score(_good(**over))


# ---------------------------------------------------------------- constants
def test_test_names_and_presets_are_consistent():
    assert len(pp.TEST_NAMES) == 12 and len(set(pp.TEST_NAMES)) == 12
    assert pp.PIOTROSKI_TESTS == pp.TEST_NAMES[:9] and pp.ROCE_TESTS == pp.TEST_NAMES[9:]
    for name, w in pp.PRESETS.items():
        assert set(w) == set(pp.TEST_NAMES), name
        assert all(v >= 0 for v in w.values())
    assert set(pp.PRESETS["canonical"].values()) == {1.0}
    assert (pp.ROCE_LEVEL_HURDLE, pp.ROCE_CV_HURDLE, pp.F_MAX, pp.PLUS_MAX) == (0.15, 0.30, 9, 3)


# ---------------------------------------------------------------- helpers _series/_at
def test_series_picks_first_present_nonempty_name_and_floats():
    df = _df({"A": [None, None], "B": [1, 2]}, years=["y1", "y2"])
    s = pp._series(df, "Missing", "A", "B")
    assert s.tolist() == [1.0, 2.0] and s.dtype == float  # "A" is all-NaN -> skipped
    assert pp._series(df, "nope") is None
    assert pp._series(None, "A") is None
    assert pp._series(pd.DataFrame(), "A") is None


def test_series_drops_nan_years():
    df = _df({"A": [1, None, 3]}, years=["y1", "y2", "y3"])
    assert pp._series(df, "A").index.tolist() == ["y1", "y3"]


def test_at():
    s = pd.Series([1.5], index=["2024"])
    assert pp._at(s, "2024") == 1.5 and pp._at(s, "2020") is None and pp._at(None, "2024") is None


# ---------------------------------------------------------------- score: happy path
def test_score_all_tests_pass():
    r = _s()
    assert r["f_score"] == 9 and r["f_tested"] == 9
    assert r["plus_score"] == 3 and r["plus_tested"] == 3 and r["total"] == 12
    for n in pp.TEST_NAMES:
        assert r[n] is True, n
    assert r["roce"] == pytest.approx(150 / 800)
    assert r["roce_ex_cash"] == pytest.approx(150 / 700)
    assert r["years"] == 5
    hist = [e / 800 for e in (150, 140, 130, 120, 110)]
    import numpy as np

    assert r["roce_cv"] == pytest.approx(np.std(hist) / np.mean(hist))


# ---------------------------------------------------------------- score: each test flips
@pytest.mark.parametrize(
    "test,over",
    [
        ("1_roa_positive", {"Net Income": [-10, -20, 0, 0, 0]}),
        ("2_cfo_positive", {"Operating Cash Flow": [-5, 100]}),
        ("3_roa_improving", {"Net Income": [70, 80, 70, 60, 50]}),
        ("4_accruals_cfo_gt_roa", {"Operating Cash Flow": [90, 100]}),
        ("5_leverage_falling", {"Long Term Debt": [200, 150]}),
        ("6_current_ratio_rising", {"Current Assets": [300, 400]}),
        ("7_no_dilution", {"Ordinary Shares Number": [110, 100]}),
        ("8_gross_margin_rising", {"Gross Profit": [400, 400]}),
        ("9_asset_turnover_rising", {"Total Revenue": [800, 900], "Gross Profit": [500, 400]}),
    ],
)
def test_each_piotroski_test_can_fail_independently(test, over):
    r = _s(**over)
    assert r[test] is False
    if test != "9_asset_turnover_rising":
        others = [n for n in pp.PIOTROSKI_TESTS if n != test]
        # a single input change must not flip unrelated tests (except the coupled ones)
        coupled = {"1_roa_positive": {"3_roa_improving", "4_accruals_cfo_gt_roa"},
                   "3_roa_improving": {"4_accruals_cfo_gt_roa"},
                   "2_cfo_positive": {"4_accruals_cfo_gt_roa"},
                   "8_gross_margin_rising": set()}.get(test, set())
        for n in others:
            if n not in coupled:
                assert r[n] is True, (test, n)
    assert r["f_tested"] == 9


def test_f_score_counts_passes():
    r = _s(**{"Long Term Debt": [200, 150], "Ordinary Shares Number": [110, 100]})
    assert r["f_score"] == 7 and r["f_tested"] == 9


def test_score_ties_are_not_passes():
    # strict inequalities for improvement tests
    r = _s(**{"Net Income": [80, 80, 70, 60, 50], "Long Term Debt": [150, 150],
              "Current Assets": [400, 400]})
    assert r["3_roa_improving"] is False
    assert r["5_leverage_falling"] is False
    assert r["6_current_ratio_rising"] is False


def test_dilution_tolerance_boundary_is_one_percent():
    assert _s(**{"Ordinary Shares Number": [101, 100]})["7_no_dilution"] is True
    assert _s(**{"Ordinary Shares Number": [101.5, 100]})["7_no_dilution"] is False
    assert _s(**{"Ordinary Shares Number": [99, 100]})["7_no_dilution"] is True


def test_accruals_uses_cfo_over_assets_vs_roa():
    # roa0 = 0.10; cfo0/ta0 must strictly exceed it
    assert _s(**{"Operating Cash Flow": [100, 100]})["4_accruals_cfo_gt_roa"] is False
    assert _s(**{"Operating Cash Flow": [101, 100]})["4_accruals_cfo_gt_roa"] is True


# ---------------------------------------------------------------- score: ROCE block
def test_roce_level_uses_ex_cash_capital_employed():
    # raw ROCE 150/800 = 18.75% (>15%) but ex-cash of 700 pushes it up; with 0 cash it equals raw
    r = _s(**{"Cash And Cash Equivalents": [0, 0]})
    assert r["roce_ex_cash"] == pytest.approx(150 / 800)
    # a LOW raw ROCE becomes a pass once cash is stripped out:
    # EBIT 110 / 800 = 13.75% (<15%); ex-cash (cash 200) = 110/600 = 18.3% (>15%)
    r = _s(EBIT=[110, 110, 110, 110, 110], **{"Cash And Cash Equivalents": [200, 200]})
    assert r["roce"] == pytest.approx(110 / 800) and r["10_roce_level"] is True
    r = _s(EBIT=[110, 110, 110, 110, 110], **{"Cash And Cash Equivalents": [0, 0]})
    assert r["10_roce_level"] is False


def test_roce_level_hurdle_is_strict():
    # ex-cash ROCE exactly 15%: EBIT 105 / (800 - 100) = 0.15
    r = _s(EBIT=[105, 105, 105, 105, 105])
    assert r["roce_ex_cash"] == pytest.approx(0.15)
    assert r["10_roce_level"] is False


def test_roce_stability_cv_threshold():
    r = _s(EBIT=[240, 80, 240, 80, 240])  # wildly swinging
    assert r["roce_cv"] > 0.30 and r["11_roce_stable"] is False
    assert _s()["11_roce_stable"] is True


def test_roce_trend_compares_latest_to_mean():
    declining = _s(EBIT=[100, 140, 150, 160, 170])
    assert declining["12_roce_not_deteriorating"] is False
    equalish = _s(EBIT=[130, 130, 130, 130, 130])  # latest == mean -> passes (>=)
    assert equalish["12_roce_not_deteriorating"] is True


def test_roce_needs_three_years_for_cv_and_trend():
    inc = _df({"EBIT": [150, 140], "Net Income": [100, 80]})
    bal = _df({"Total Assets": [1000, 1000], "Current Liabilities": [200, 200]})
    r = pp.score(_Ticker(inc, bal, _df({})))
    assert r["roce_cv"] is None and r["11_roce_stable"] is None and r["12_roce_not_deteriorating"] is None
    assert r["10_roce_level"] is True and r["plus_tested"] == 1 and r["plus_score"] == 1
    assert r["years"] == 2


def test_cv_undefined_when_mean_roce_near_zero():
    r = _s(EBIT=[1, -1, 1, -1, 1])  # mean ROCE ~0.0003 -> |mean| <= 0.01
    assert r["roce_cv"] is None and r["11_roce_stable"] is None


def test_non_positive_capital_employed_years_excluded_from_history():
    # one year with CL > TA is dropped from the history
    r = _s(**{"Current Liabilities": [200, 200, 1500, 200, 200]})
    assert r["years"] == 4


def test_missing_cash_line_treated_as_zero():
    r = _s(**{"Cash And Cash Equivalents": None})
    assert r["roce_ex_cash"] == pytest.approx(r["roce"])


# ---------------------------------------------------------------- score: missing data
def test_missing_lines_are_skipped_not_failed():
    r = _s(**{"Ordinary Shares Number": None, "Gross Profit": None})
    assert r["7_no_dilution"] is None and r["8_gross_margin_rising"] is None
    assert r["f_tested"] == 7 and r["f_score"] == 7  # nothing counted as a failure


def test_alias_line_names_are_accepted():
    inc = _df({"Operating Income": [150, 140, 130], "NetIncome": [100, 80, 70]})
    bal = _df({"Total Assets": [1000] * 3, "Total Current Liabilities": [200] * 3,
               "Total Current Assets": [500, 400], "LongTermDebt": [100, 150]})
    cfs = _df({"Total Cash From Operating Activities": [120, 100]})
    r = pp.score(_Ticker(inc, bal, cfs))
    assert r["1_roa_positive"] is True and r["6_current_ratio_rising"] is True
    assert r["5_leverage_falling"] is True and r["2_cfo_positive"] is True


def test_zero_total_assets_does_not_divide_by_zero():
    r = _s(**{"Total Assets": [0, 0, 0, 0, 0]})
    assert r["1_roa_positive"] is None and r["4_accruals_cfo_gt_roa"] is None
    assert r["roce"] is None and r["years"] == 0 and r["plus_tested"] == 0 and r["total"] is None
    assert r["f_tested"] == 4  # leverage/turnover need assets; the rest still run


def test_unusable_statements_return_empty_dict():
    assert pp.score(_Ticker(None, None, None)) == {}
    assert pp.score(_Ticker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())) == {}
    base = _good()
    assert pp.score(_Ticker(base.income_stmt.drop("EBIT"), base.balance_sheet, base.cashflow)) == {}
    assert pp.score(_Ticker(base.income_stmt, base.balance_sheet.drop("Total Assets"),
                            base.cashflow)) == {}
    assert pp.score(_Ticker(base.income_stmt, base.balance_sheet.drop("Current Liabilities"),
                            base.cashflow)) == {}


def test_needs_two_common_years():
    inc = _df({"EBIT": [150], "Net Income": [100]})
    bal = _df({"Total Assets": [1000], "Current Liabilities": [200]})
    assert pp.score(_Ticker(inc, bal, _df({}))) == {}
    # years only overlap on one date
    inc = _df({"EBIT": [150, 140]}, years=["2024", "2023"])
    bal = _df({"Total Assets": [1000, 1000], "Current Liabilities": [200, 200]},
              years=["2024", "2019"])
    assert pp.score(_Ticker(inc, bal, _df({}))) == {}


def test_attribute_access_error_returns_empty_dict():
    class Bad:
        @property
        def income_stmt(self):
            raise RuntimeError("yfinance")

    assert pp.score(Bad()) == {}


def test_total_is_none_unless_both_blocks_ran():
    r = _s(**{"Ordinary Shares Number": None})
    assert r["total"] == r["f_score"] + r["plus_score"]
    inc = _df({"EBIT": [150, 140]})
    bal = _df({"Total Assets": [1000, 1000], "Current Liabilities": [200, 200]})
    r = pp.score(_Ticker(inc, bal, _df({})))
    assert r["f_score"] is None and r["total"] is None and r["f_tested"] == 0


# ---------------------------------------------------------------- _ran / _passed
@pytest.mark.parametrize(
    "v,ran",
    [(None, False), (float("nan"), False), (True, True), (False, True), (0, True), (1, True),
     ("True", True), (" false ", True), ("1", True), ("0", True), ("nan", False), ("", False),
     ("maybe", False), (0.0, True), (math.inf, True)],
)
def test_ran(v, ran):
    assert pp._ran(v) is ran


@pytest.mark.parametrize(
    "v,passed",
    [(True, True), (False, False), ("True", True), ("true", True), ("1", True), (" TRUE ", True),
     ("False", False), ("0", False), ("no", False), (1, True), (0, False), (None, False)],
)
def test_passed(v, passed):
    assert pp._passed(v) is passed


# ---------------------------------------------------------------- weigh
def test_weigh_canonical_all_pass_and_all_fail():
    r = {n: True for n in pp.TEST_NAMES}
    w = pp.weigh(r)
    assert w == {"raw": 12.0, "possible": 12.0, "pct": 100.0, "coverage": 1.0}
    w = pp.weigh({n: False for n in pp.TEST_NAMES})
    assert w["raw"] == 0.0 and w["pct"] == 0.0 and w["coverage"] == 1.0


def test_weigh_skipped_tests_excluded_from_both_sides():
    r = {n: True for n in pp.TEST_NAMES}
    r["7_no_dilution"] = None
    r["8_gross_margin_rising"] = float("nan")
    w = pp.weigh(r)
    assert w["possible"] == 10.0 and w["raw"] == 10.0 and w["pct"] == 100.0
    assert w["coverage"] == pytest.approx(10 / 12)


def test_weigh_uses_weights_and_string_booleans():
    r = {n: "True" for n in pp.TEST_NAMES}
    r["11_roce_stable"] = "False"
    w = pp.weigh(r, "quality")
    total = sum(pp.PRESETS["quality"].values())
    assert w["raw"] == pytest.approx(total - 3.5)
    assert w["pct"] == pytest.approx((total - 3.5) / total * 100)


def test_weigh_custom_dict_unlisted_tests_weigh_zero():
    r = {n: True for n in pp.TEST_NAMES}
    r["2_cfo_positive"] = False
    w = pp.weigh(r, {"1_roa_positive": 1.0, "2_cfo_positive": 3.0})
    assert w["raw"] == 1.0 and w["possible"] == 4.0 and w["pct"] == 25.0 and w["coverage"] == 1.0


def test_weigh_zero_weight_tests_do_not_count_as_possible():
    r = {n: False for n in pp.TEST_NAMES}
    w = pp.weigh(r, "turnaround")
    assert w["possible"] == pytest.approx(sum(pp.PRESETS["turnaround"].values()))
    # zero-weight tests ran but contribute nothing; if only those ran, pct is undefined
    w = pp.weigh({"10_roce_level": True}, "turnaround")
    assert w == {"raw": 0.0, "possible": 0.0, "pct": None, "coverage": 0.0}


def test_weigh_empty_result_and_unknown_preset():
    assert pp.weigh({}) == {"raw": None, "possible": None, "pct": None, "coverage": None}
    assert pp.weigh(None) == {"raw": None, "possible": None, "pct": None, "coverage": None}
    with pytest.raises(KeyError, match="unknown preset 'nope'"):
        pp.weigh({"1_roa_positive": True}, "nope")


def test_weigh_empty_weight_vector_divides_safely():
    assert pp.weigh({"1_roa_positive": True}, {})["pct"] is None


def test_weigh_matches_score_output():
    r = _s()
    assert pp.weigh(r)["pct"] == 100.0
    r2 = _s(**{"Long Term Debt": [200, 150]})
    assert pp.weigh(r2)["raw"] == 11.0


def test_presets_rank_companies_differently():
    steady = {n: False for n in pp.TEST_NAMES}
    steady.update({"10_roce_level": False, "11_roce_stable": True, "7_no_dilution": True})
    improving = {n: False for n in pp.TEST_NAMES}
    improving.update({"3_roa_improving": True, "8_gross_margin_rising": True})
    assert pp.weigh(steady, "quality")["pct"] > pp.weigh(improving, "quality")["pct"]
    assert pp.weigh(improving, "turnaround")["pct"] > pp.weigh(steady, "turnaround")["pct"]


def test_sweep_all_and_subset():
    r = {n: True for n in pp.TEST_NAMES}
    assert set(pp.sweep(r)) == set(pp.PRESETS) and all(v == 100.0 for v in pp.sweep(r).values())
    assert list(pp.sweep(r, ["safety"])) == ["safety"]
    assert pp.sweep({}, ["quality"]) == {"quality": None}


# ---------------------------------------------------------------- verdict
def _v(f, ft, p, pt):
    return pp.verdict({"f_score": f, "f_tested": ft, "plus_score": p, "plus_tested": pt})


@pytest.mark.parametrize(
    "args,expected",
    [
        ((9, 9, 3, 3), "STRONG"),
        ((7, 9, 3, 3), "STRONG"),
        ((7, 9, 2, 3), "GOOD"),
        ((8, 9, 2, 2), "GOOD"),
        ((7, 9, 1, 3), "IMPROVING_WEAK_BUSINESS"),
        ((7, 9, 0, 3), "IMPROVING_WEAK_BUSINESS"),
        ((6, 9, 3, 3), "QUALITY_NOT_IMPROVING"),
        ((4, 9, 3, 3), "QUALITY_NOT_IMPROVING"),
        ((3, 9, 3, 3), "MIXED"),  # p==3 but fs < 4
        ((3, 9, 1, 3), "AVOID"),
        ((0, 9, 0, 3), "AVOID"),
        ((3, 9, 2, 3), "MIXED"),
        ((5, 9, 1, 3), "MIXED"),
        ((4, 9, 0, 3), "MIXED"),
    ],
)
def test_verdict_labels(args, expected):
    assert _v(*args) == expected


def test_verdict_scales_f_to_nine_point_frame():
    # 6/6 tested -> 9.0 -> STRONG ; 5/6 -> 7.5 -> counts as >= 7
    assert _v(6, 6, 3, 3) == "STRONG"
    assert _v(5, 6, 3, 3) == "STRONG"
    assert _v(4, 6, 3, 3) == "QUALITY_NOT_IMPROVING"  # 6.0


def test_verdict_insufficient_data_boundaries():
    assert _v(5, 5, 3, 3) == "INSUFFICIENT_DATA"  # f_tested < 6
    assert _v(6, 6, 3, 3) != "INSUFFICIENT_DATA"
    assert _v(9, 9, 1, 1) == "INSUFFICIENT_DATA"  # plus_tested < 2
    assert _v(9, 9, 2, 2) != "INSUFFICIENT_DATA"
    assert _v(None, 0, 3, 3) == "INSUFFICIENT_DATA"
    assert _v(9, 9, None, 0) == "INSUFFICIENT_DATA"
    assert pp.verdict({}) == "INSUFFICIENT_DATA" and pp.verdict(None) == "INSUFFICIENT_DATA"


def test_verdict_zero_scores_are_not_missing():
    assert _v(0, 9, 0, 3) == "AVOID"  # f_score 0 is data, not None


def test_verdict_end_to_end_from_score():
    assert pp.verdict(_s()) == "STRONG"
