"""backtest_zone_rules.zone_signals — the rule definitions the digest's
per-market recommendation is chosen from. Weekly synthetic panel: six names
compounding at -2%..+3% per week, so every horizon ranks them in the same order.
"""
import numpy as np
import pandas as pd

import backtest_zone_rules as bz

GROWTH = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]


def panel(weeks=30):
    idx = pd.date_range("2025-01-03", periods=weeks, freq="W-FRI")
    t = np.arange(weeks)
    return pd.DataFrame({f"S{j}": 100 * (1 + g) ** t for j, g in enumerate(GROWTH)},
                        index=idx)


def test_momentum_rules_buy_the_top_tercile_and_sell_the_bottom():
    sig = bz.zone_signals(panel())
    expected = [-1, -1, 0, 1, 1, 1]          # pct ranks 1/6..1; <=1/3 sell, >=2/3 buy
    assert sig["mom126"].iloc[-1].tolist() == expected
    assert sig["mom_st"].iloc[-1].tolist() == expected


def test_mean_revert_is_the_mirror_of_short_term_tercile():
    sig = bz.zone_signals(panel())
    assert sig["revert"].iloc[-1].tolist() == [1, 1, 0, -1, -1, -1]


def test_trend_rule_buys_uptrends_sells_downtrends_holds_flat():
    sig = bz.zone_signals(panel())
    assert sig["trend"].iloc[-1].tolist() == [-1, -1, 0, 1, 1, 1]


def test_126d_momentum_needs_25_weeks_of_history():
    sig = bz.zone_signals(panel(weeks=30))
    # before week 25 the 25-week return is NaN -> rank NaN -> neither tercile
    assert (sig["mom126"].iloc[:25] == 0).all().all()
    assert (sig["mom126"].iloc[25] != 0).any()
