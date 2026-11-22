"""adaptive_liquidity.py -- floors, capital gate, percentile tiers."""

import numpy as np
import pandas as pd
import pytest

import adaptive_liquidity as AL


def test_market_floor_only_india_and_normalises_input():
    assert AL.market_floor("IN") == 120_000
    assert AL.market_floor(" india ") == 120_000  # first two chars, stripped, upper
    assert AL.market_floor("in") == 120_000
    assert AL.market_floor("US") == 0.0
    assert AL.market_floor("") == 0.0
    assert AL.market_floor(None) == 0.0


def test_scan_floor_is_max_of_structural_and_policy():
    assert AL.scan_floor("US") == 10_000
    assert AL.scan_floor("IN") == 120_000
    assert AL.scan_floor("") == 10_000


def test_absolute_floor_alias_is_india_gate():
    assert AL.ABSOLUTE_FLOOR_USD == 120_000


def test_gate_position_and_min_adv():
    g = AL.Gate("US", capital_usd=100_000, positions=10)
    assert g.position_usd == 10_000
    assert g.min_adv_usd == pytest.approx(10_000 / 0.15)
    assert AL.Gate("US", 100.0, 0).position_usd == 100.0  # positions clamped to >=1
    assert AL.Gate("US", 100.0, -3).position_usd == 100.0


def test_min_adv_takes_market_floor_when_larger():
    small = AL.Gate("IN", capital_usd=10_000, positions=10)  # 1k/0.15 = 6.7k < 120k
    assert small.min_adv_usd == 120_000
    big = AL.Gate("IN", capital_usd=10_000_000, positions=10)  # 1M/0.15 > 120k
    assert big.min_adv_usd == pytest.approx(1_000_000 / 0.15)


def adv_frame(values):
    return pd.DataFrame({"sym": [f"S{i}" for i in range(len(values))], "adv_usd": values})


def test_classify_tradeable_and_adv_pct():
    g = AL.Gate("US", capital_usd=10_000, positions=10)  # position 1000, min_adv 6666.7
    df = adv_frame([0, -5, 6_000, 6_667, 10_000, 1e9])
    out = AL.classify(df, g)
    assert list(out["tradeable"]) == [False, False, False, True, True, True]
    assert out["adv_pct"].iloc[0] == float("inf") and out["adv_pct"].iloc[1] == float("inf")
    assert out["adv_pct"].iloc[4] == pytest.approx(10.0)
    assert "tradeable" not in df.columns  # input not mutated


def test_classify_execution_ceiling_is_inclusive_at_15_pct():
    # position 1500 on ADV 10000 = exactly 15% -> tradeable; 1501 -> not
    g = AL.Gate("US", capital_usd=1_500, positions=1)
    out = AL.classify(adv_frame([10_000]), g)
    assert out["adv_pct"].iloc[0] == pytest.approx(15.0) and bool(out["tradeable"].iloc[0])
    g2 = AL.Gate("US", capital_usd=1_501, positions=1)
    assert not bool(AL.classify(adv_frame([10_000]), g2)["tradeable"].iloc[0])


def test_classify_sub_dollar_adv_is_a_huge_finite_share_not_infinite():
    out = AL.classify(adv_frame([0.5, 2.0]), AL.Gate("US", 1000, 1))
    assert out["adv_pct"].iloc[0] == pytest.approx(1000 / 0.5 * 100)
    assert out["adv_pct"].iloc[1] == pytest.approx(1000 / 2.0 * 100)
    assert not out["tradeable"].any()


def test_classify_nan_adv_is_untradeable():
    out = AL.classify(adv_frame([np.nan, 1e6]), AL.Gate("US", 1000, 1))
    assert list(out["tradeable"]) == [False, True]


def test_classify_tiers_need_ten_tradeable_names():
    g = AL.Gate("US", 1000, 10)
    nine = AL.classify(adv_frame([1e6 * (i + 1) for i in range(9)]), g)
    assert nine["tier"].isna().all()
    ten = AL.classify(adv_frame([1e6 * (i + 1) for i in range(10)]), g)
    assert ten["tier"].notna().all()


def test_classify_percentile_tier_assignment_and_untradeable_excluded():
    g = AL.Gate("US", 1000, 10)  # min_adv 666.7
    vals = [1e6 * (i + 1) for i in range(10)] + [10.0, 0.0]  # 2 untradeable shells
    out = AL.classify(adv_frame(vals), g)
    # ranks are over the tradeable subset only (the shells do not shift boundaries)
    assert list(out["tier"].iloc[:10]) == [
        "T5_MOST_ILLIQUID",
        "T4_ILLIQUID",
        "T4_ILLIQUID",
        "T3_MID",
        "T3_MID",
        "T2_LIQUID",
        "T2_LIQUID",
        "T1_MOST_LIQUID",
        "T1_MOST_LIQUID",
        "T1_MOST_LIQUID",
    ]
    assert out["tier"].iloc[10:].isna().all()


def test_classify_custom_adv_column():
    df = pd.DataFrame({"turn": [1e6] * 3})
    out = AL.classify(df, AL.Gate("US", 1000, 10), adv_col="turn")
    assert out["tradeable"].all()


def test_capacity_quantile_points():
    df = adv_frame(list(range(1, 101)))
    out = AL.capacity(df, positions=4)
    assert list(out["universe_point"]) == ["median name", "bottom quartile", "bottom decile"]
    med = df["adv_usd"].quantile(0.5)
    assert out["adv_usd"].iloc[0] == med
    assert out["max_position_usd"].iloc[0] == pytest.approx(med * 0.15)
    assert out["max_portfolio_usd"].iloc[0] == pytest.approx(med * 0.15 * 4)
    assert out["adv_usd"].iloc[1] == df["adv_usd"].quantile(0.25)
    assert out["adv_usd"].iloc[2] == df["adv_usd"].quantile(0.10)
    assert AL.capacity(df)["max_portfolio_usd"].iloc[0] == pytest.approx(med * 0.15 * 10)


def scan_frame(n, tier="T4_SMALL"):
    return pd.DataFrame(
        {"Median_Turnover": [float(i + 1) for i in range(n)], "Liquidity_Tier": [tier] * n}
    )


def test_retier_does_not_mutate_input():
    df = scan_frame(20)
    out = AL.retier(df)
    assert (df["Liquidity_Tier"] == "T4_SMALL").all()
    assert out["Liquidity_Tier"].iloc[19] == "T1_MOST_LIQUID"


def test_retier_exact_cut_values():
    out = AL.retier(scan_frame(20))
    t = list(out["Liquidity_Tier"])
    # percentile of row i is 5*(i+1): >=80 T1 (rows 15+), >=60 T2 (11-14), >=40 T3 (7-10),
    # >=20 T4 (3-6), else T5 (0-2)
    assert t == (
        ["T5_MOST_ILLIQUID"] * 3
        + ["T4_ILLIQUID"] * 4
        + ["T3_MID"] * 4
        + ["T2_LIQUID"] * 4
        + ["T1_MOST_LIQUID"] * 5
    )


def test_retier_requires_20_rankable_rows():
    df = scan_frame(19)
    assert AL.retier(df) is df
    assert AL.retier(scan_frame(20)) is not None
    # 20 rows but one non-positive => only 19 rankable -> untouched
    d = scan_frame(20)
    d.loc[0, "Median_Turnover"] = 0.0
    assert AL.retier(d) is d


def test_retier_keeps_unknown_and_skips_nan_and_nonpositive():
    d = scan_frame(25)
    d.loc[0, "Liquidity_Tier"] = "UNKNOWN"
    d.loc[1, "Median_Turnover"] = np.nan
    d.loc[2, "Median_Turnover"] = -3.0
    out = AL.retier(d)
    assert out.loc[0, "Liquidity_Tier"] == "UNKNOWN"
    assert out.loc[1, "Liquidity_Tier"] == "T4_SMALL"
    assert out.loc[2, "Liquidity_Tier"] == "T4_SMALL"
    assert out["Liquidity_Tier"].iloc[3:].isin([n for _, n in AL.TIER_CUTS]).all()


def test_retier_missing_columns_returns_input():
    df = pd.DataFrame({"x": [1]})
    assert AL.retier(df) is df
    only_t = pd.DataFrame({"Median_Turnover": [1.0] * 30})
    assert AL.retier(only_t) is only_t


def test_retier_custom_columns():
    d = pd.DataFrame({"tv": np.arange(1.0, 21.0), "tr": ["x"] * 20})
    out = AL.retier(d, turnover_col="tv", tier_col="tr")
    assert out["tr"].iloc[19] == "T1_MOST_LIQUID"


def test_describe_line_and_empty():
    df = adv_frame([0.0, -1.0])
    assert AL.describe(df, "US") == "  US: no data"
    df = adv_frame([1e6, 3e6, 5e6])
    line = AL.describe(df, "IN")
    assert "IN" in line and "n=     3" in line
    assert "ADV median $    3.00M" in line
    assert "above Rs1cr floor:  100%" in line
    assert AL.describe(adv_frame([1e5, 5e6]), "IN").endswith("above Rs1cr floor:   50%")
