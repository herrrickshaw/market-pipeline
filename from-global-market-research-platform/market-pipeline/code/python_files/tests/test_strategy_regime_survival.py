"""strategy_regime_survival.signals — the rule set behind zone_regime.json, which
the digest prefers over the flat zone rule. Same synthetic panel as
test_backtest_zone_rules: six names compounding at -2%..+3% per week.
"""
import strategy_regime_survival as srs
from test_backtest_zone_rules import panel


def test_terciles_are_symmetric():
    sig = srs.signals(panel())
    assert sig["mom126"].iloc[-1].tolist() == [-1, -1, 0, 1, 1, 1]
    assert sig["mom_st"].iloc[-1].tolist() == [-1, -1, 0, 1, 1, 1]
    assert sig["revert"].iloc[-1].tolist() == [1, 1, 0, -1, -1, -1]


def test_trend_matches_backtest_zone_rules_definition():
    import backtest_zone_rules as bz
    a, b = srs.signals(panel())["trend"], bz.zone_signals(panel())["trend"]
    assert a.equals(b)


def test_breakout_is_near_52_week_high():
    sig = srs.signals(panel(weeks=60))
    # rising names sit at their high (BUY); -2%/wk is >20% off its high (SELL)
    assert sig["breakout"].iloc[-1].tolist()[-1] == 1
    assert sig["breakout"].iloc[-1].tolist()[0] == -1
