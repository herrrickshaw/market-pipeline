"""custom_screener.py -- metric computation from OHLCV/fundamentals, rule evaluation, ranking."""

import numpy as np
import pandas as pd
import pytest

import custom_screener as CS
from strategies.base import StockData


def ohlcv(close, high=None, low=None, volume=None):
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"Close": close})
    if high is not None:
        df["High"] = high if not np.isscalar(high) else close + high
    if low is not None:
        df["Low"] = low if not np.isscalar(low) else close - low
    if volume is not None:
        df["Volume"] = volume
    return df


def stock(df=None, sym="AAA", mkt="IN", **fund):
    return StockData(sym, mkt, ohlcv=df, fundamentals=dict(fund))


LINEAR = np.arange(100.0, 360.0)  # 260 closes: 100 .. 359


# ── _rsi ──────────────────────────────────────────────────────────────────────
def test_rsi_all_up_is_100():
    assert CS._rsi(pd.Series(np.arange(1.0, 20.0))) == 100.0


def test_rsi_balanced_is_50():
    close = pd.Series([10.0, 11.0] * 7 + [10.0])  # 14 diffs: 7 up, 7 down
    assert CS._rsi(close) == pytest.approx(50.0)


def test_rsi_hand_value():
    diffs = [1] * 10 + [-1] * 4
    close = pd.Series(100.0 + np.concatenate([[0], np.cumsum(diffs)]))
    assert CS._rsi(close) == pytest.approx(100 - 100 / (1 + 10 / 4))


def test_rsi_needs_n_plus_one_points():
    assert CS._rsi(pd.Series(np.arange(14.0))) is None  # 14 points
    assert CS._rsi(pd.Series(np.arange(15.0))) == 100.0  # 15 points
    assert CS._rsi(pd.Series(np.arange(6.0)), n=5) == 100.0
    assert CS._rsi(pd.Series(np.arange(5.0)), n=5) is None


def test_rsi_uses_only_last_window():
    close = pd.Series([50.0, 40.0, 30.0] + list(np.arange(100.0, 120.0)))
    assert CS._rsi(close) == 100.0  # early losses roll out


# ── _ret ──────────────────────────────────────────────────────────────────────
def test_ret_boundary_and_value():
    s = pd.Series([100.0, 110.0, 121.0])
    assert CS._ret(s, 2) == pytest.approx(21.0)  # len == n+1 is enough
    assert CS._ret(s, 1) == pytest.approx(10.0)
    assert CS._ret(s, 3) is None  # len <= n
    assert CS._ret(pd.Series([], dtype=float), 1) is None


# ── _ccc ──────────────────────────────────────────────────────────────────────
def test_ccc_formula():
    f = {"inventory": 100, "receivables": 50, "payables": 30, "cogs": 365, "revenue": 365}
    assert CS._ccc(f) == pytest.approx(100 + 50 - 30)
    f2 = {"inventory": 73, "receivables": 146, "payables": 36.5, "cogs": 730, "revenue": 365}
    assert CS._ccc(f2) == pytest.approx(36.5 + 146 - 18.25)  # cogs and revenue differ


def test_ccc_missing_denominators_contribute_zero():
    assert CS._ccc({"inventory": 100, "receivables": 50, "payables": 30}) == 0
    assert CS._ccc({"inventory": 100, "cogs": 0, "revenue": 365, "receivables": 365}) == 365
    assert CS._ccc({"inventory": 100, "cogs": 0, "receivables": 0, "revenue": 365}) == 0


def test_ccc_without_inputs_falls_back_to_reported_value():
    assert CS._ccc({}) is None
    assert CS._ccc({"ccc": 42}) == 42
    assert CS._ccc({"revenue": 1, "cogs": 1, "ccc": 7}) == 7


# ── compute_metrics ───────────────────────────────────────────────────────────
def test_metrics_fundamentals_only_when_no_ohlcv():
    for df in (None, pd.DataFrame()):
        s = stock(df, roe=18.5, inventory=73, cogs=365)
        m = CS.compute_metrics(s)
        assert m["roe"] == 18.5 and m["ccc"] == 73 / 365 * 365 and "ltp" not in m


def test_metrics_do_not_mutate_stock_fundamentals():
    s = stock(ohlcv(LINEAR), inventory=73, cogs=365)
    CS.compute_metrics(s)
    assert s.fundamentals == {"inventory": 73, "cogs": 365}


def test_metrics_ccc_rounding_and_absence():
    m = CS.compute_metrics(stock(None, inventory=100, cogs=300))
    assert m["ccc"] == round(100 / 300 * 365, 1) == 121.7
    assert "ccc" not in CS.compute_metrics(stock(None, roe=1))
    assert CS.compute_metrics(stock(None, ccc=12.34))["ccc"] == 12.3


def test_metrics_linear_series_hand_values():
    df = ohlcv(LINEAR, high=1, low=1, volume=[1000] * 259 + [3000])
    m = CS.compute_metrics(stock(df))
    assert m["ltp"] == 359.0
    assert m["ret_21"] == round((359 / 338 - 1) * 100, 2) == 6.21
    assert m["ret_63"] == round((359 / 296 - 1) * 100, 2)
    assert m["ret_126"] == round((359 / 233 - 1) * 100, 2)
    assert m["ret_252"] == round((359 / 107 - 1) * 100, 2)
    assert m["sma50"] == 334.5 and m["above_50dma"] is True
    assert m["sma200"] == 259.5 and m["above_200dma"] is True
    assert m["max_drawdown"] == 0.0
    dr = pd.Series(LINEAR).pct_change().dropna()
    assert m["vol_ann"] == round(float(dr.std() * np.sqrt(252) * 100), 2)
    assert m["rsi14"] == 100.0
    assert m["dist_52w_high"] == round((360 - 359) / 360 * 100, 2) == 0.28
    assert m["dist_52w_low"] == round((359 - 107) / 107 * 100, 2)  # window starts at 108-1
    assert m["avg_vol_20"] == 1000.0 and m["vol_ratio"] == 3.0
    assert m["atr14_pct"] == round(2 / 359 * 100, 2) == 0.56


def test_metrics_history_gates():
    m = CS.compute_metrics(stock(ohlcv(np.arange(100.0, 105.0))))  # 5 bars
    assert set(m) == {"ltp", "dist_52w_high", "dist_52w_low"}
    m = CS.compute_metrics(stock(ohlcv(np.arange(100.0, 106.0))))  # 6 bars
    assert "vol_ann" in m and "max_drawdown" in m and "ret_21" not in m
    m = CS.compute_metrics(stock(ohlcv(np.arange(100.0, 149.0))))  # 49 bars
    assert "sma50" not in m and "above_50dma" not in m and "rsi14" in m
    assert "sma50" in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 150.0))))
    assert "sma200" not in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 299.0))))
    assert "sma200" in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 300.0))))
    assert "rsi14" not in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 114.0))))


def test_metrics_return_horizon_boundaries():
    assert "ret_21" not in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 121.0))))  # 21 bars
    assert "ret_21" in CS.compute_metrics(stock(ohlcv(np.arange(100.0, 122.0))))  # 22 bars


def test_metrics_below_moving_average_flags_false():
    close = np.concatenate([np.full(60, 200.0), np.full(10, 50.0)])
    m = CS.compute_metrics(stock(ohlcv(close)))
    assert m["above_50dma"] is False and isinstance(m["above_50dma"], bool)


def test_metrics_price_equal_to_the_average_is_not_above_it():
    m = CS.compute_metrics(stock(ohlcv(np.full(250, 100.0))))
    assert m["sma50"] == 100.0 and m["above_50dma"] is False
    assert m["sma200"] == 100.0 and m["above_200dma"] is False


def test_metrics_zero_prices_do_not_produce_distance_metrics():
    m = CS.compute_metrics(stock(ohlcv(np.zeros(30), high=0.0, low=0.0)))
    assert "dist_52w_high" not in m and "dist_52w_low" not in m


def test_metrics_max_drawdown_is_worst_peak_to_trough():
    m = CS.compute_metrics(stock(ohlcv([100, 120, 60, 90, 100, 110])))
    assert m["max_drawdown"] == -50.0


def test_metrics_52w_window_only_last_252_bars():
    close = np.concatenate([[1000.0], np.full(252, 100.0)])  # spike falls out of window
    m = CS.compute_metrics(stock(ohlcv(close, high=close, low=close)))
    assert m["dist_52w_high"] == 0.0


def test_metrics_52w_without_high_low_uses_close_and_skips_nonpositive_low():
    m = CS.compute_metrics(stock(ohlcv([100.0, 200.0, 150.0])))
    assert m["dist_52w_high"] == 25.0 and m["dist_52w_low"] == 50.0
    z = CS.compute_metrics(stock(ohlcv([5.0, 6.0], high=[6.0, 7.0], low=[0.0, 5.0])))
    assert "dist_52w_low" not in z and "dist_52w_high" in z


def test_metrics_volume_rules():
    base = np.arange(100.0, 130.0)
    m = CS.compute_metrics(stock(ohlcv(base, volume=[500] * 29 + [1500])))
    assert m["avg_vol_20"] == 500.0 and m["vol_ratio"] == 3.0
    # today's bar is excluded from the average: a spike must not dilute itself
    m = CS.compute_metrics(stock(ohlcv(base, volume=[100] * 29 + [10_100])))
    assert m["avg_vol_20"] == 100.0
    z = CS.compute_metrics(stock(ohlcv(base, volume=[0] * 30)))
    assert z["avg_vol_20"] == 0.0 and "vol_ratio" not in z
    assert "avg_vol_20" not in CS.compute_metrics(
        stock(ohlcv(np.arange(100.0, 120.0), volume=[1] * 20))
    )  # <21 bars
    assert "avg_vol_20" in CS.compute_metrics(
        stock(ohlcv(np.arange(100.0, 121.0), volume=[1] * 21))
    )
    assert "avg_vol_20" not in CS.compute_metrics(stock(ohlcv(base)))  # no Volume


def test_metrics_atr_needs_hlc_and_15_bars():
    c = np.arange(100.0, 114.0)  # 14 bars
    assert "atr14_pct" not in CS.compute_metrics(stock(ohlcv(c, high=1, low=1)))
    c = np.arange(100.0, 115.0)  # 15 bars
    assert "atr14_pct" in CS.compute_metrics(stock(ohlcv(c, high=1, low=1)))
    assert "atr14_pct" not in CS.compute_metrics(stock(ohlcv(c, high=1)))


def test_metrics_atr_uses_gaps():
    c = np.full(20, 100.0)
    df = ohlcv(c, high=c + 1, low=c - 1)
    df.loc[19, ["High", "Low", "Close"]] = [111.0, 109.0, 110.0]  # gap up 10 on last bar
    m = CS.compute_metrics(stock(df))
    # last-14 TRs: 13 bars of 2 and one of max(2, |111-100|, |109-100|) = 11
    assert m["atr14_pct"] == round((13 * 2 + 11) / 14 / 110 * 100, 2)


# ── evaluate ──────────────────────────────────────────────────────────────────
S = stock(ohlcv(LINEAR, high=1, low=1), roe=20.0)


def test_evaluate_dict_rules_pass_and_fail():
    ok, m = CS.evaluate(S, {"roe": (">=", 20), "above_200dma": ("==", True)})
    assert ok and m["roe"] == 20.0
    assert CS.evaluate(S, {"roe": (">", 20)})[0] is False
    assert CS.evaluate(S, {"roe": ("<=", 20)})[0] is True
    assert CS.evaluate(S, {"roe": ("<", 20)})[0] is False
    assert CS.evaluate(S, {"roe": ("!=", 20)})[0] is False
    assert CS.evaluate(S, {"roe": ("==", 20)})[0] is True
    assert CS.evaluate(S, {"roe": (">=", 21), "above_200dma": ("==", True)})[0] is False


def test_evaluate_missing_metric_fails_even_before_bad_operator():
    assert CS.evaluate(S, {"nope": (">", 1)})[0] is False
    assert CS.evaluate(S, {"nope": ("~~", 1)})[0] is False  # missing wins over bad op


def test_evaluate_unknown_operator_raises_with_choices():
    with pytest.raises(ValueError, match=r"unknown operator '~~'.*>="):
        CS.evaluate(S, {"roe": ("~~", 1)})


def test_evaluate_type_mismatch_fails_quietly():
    assert CS.evaluate(S, {"roe": (">", "high")})[0] is False


def test_evaluate_zero_value_is_not_missing():
    z = stock(None, pe=0)
    assert CS.evaluate(z, {"pe": ("==", 0)})[0] is True


def test_evaluate_callable():
    assert CS.evaluate(S, lambda m: m["roe"] > 10)[0] is True
    ok, m = CS.evaluate(S, lambda m: m["roe"] > 99)
    assert ok is False and "roe" in m
    assert CS.evaluate(S, lambda m: m["missing"])[0] is False  # exception -> False
    assert CS.evaluate(S, lambda m: "truthy")[0] is True  # coerced to bool


def test_evaluate_empty_criteria_passes():
    assert CS.evaluate(S, {})[0] is True


# ── screen ────────────────────────────────────────────────────────────────────
def universe():
    return [
        stock(None, "A", "IN", roe=10, pe=30),
        stock(None, "B", "US", roe=30, pe=10),
        stock(None, "C", "IN", roe=20, pe=20),
        stock(None, "D", "IN", roe=5, pe=5),
    ]


def test_screen_filters_and_keeps_input_order_by_default():
    df = CS.screen(universe(), {"roe": (">=", 10)})
    assert list(df["Symbol"]) == ["A", "B", "C"]
    assert list(df["Market"]) == ["IN", "US", "IN"]
    assert list(df.index) == [0, 1, 2]


def test_screen_rank_top_ascending():
    u = universe()
    assert list(CS.screen(u, {"roe": (">", 0)}, rank_by="roe")["Symbol"]) == ["B", "C", "A", "D"]
    assert list(CS.screen(u, {"roe": (">", 0)}, rank_by="roe", ascending=True)["Symbol"]) == [
        "D",
        "A",
        "C",
        "B",
    ]
    top = CS.screen(u, {"roe": (">", 0)}, rank_by="roe", top=2)
    assert list(top["Symbol"]) == ["B", "C"] and list(top.index) == [0, 1]
    assert list(CS.screen(u, {"roe": (">", 0)}, top=1)["Symbol"]) == ["A"]


def test_screen_unknown_rank_column_keeps_order():
    assert list(CS.screen(universe(), {"roe": (">", 0)}, rank_by="zzz")["Symbol"]) == [
        "A",
        "B",
        "C",
        "D",
    ]


def test_screen_show_selects_existing_columns_only():
    df = CS.screen(universe(), {"roe": (">", 0)}, show=["pe", "nothing", "roe"])
    assert list(df.columns) == ["Symbol", "Market", "pe", "roe"]


def test_screen_empty_result_and_generators():
    assert CS.screen(universe(), {"roe": (">", 100)}).empty
    assert CS.screen([], {"roe": (">", 1)}).empty
    gen = (s for s in universe())
    assert len(CS.screen(gen, {"roe": (">", 0)})) == 4


def test_screen_with_callable_and_technicals():
    stocks = [stock(ohlcv(LINEAR), "UP"), stock(ohlcv(LINEAR[::-1].copy()), "DOWN")]
    df = CS.screen(stocks, {"above_200dma": ("==", True)})
    assert list(df["Symbol"]) == ["UP"]
    assert list(CS.screen(stocks, lambda m: m["ret_21"] < 0)["Symbol"]) == ["DOWN"]


def test_ops_table_and_metric_catalogue():
    assert set(CS.OPS) == {">", ">=", "<", "<=", "==", "!="}
    assert "rsi14" in CS.METRICS and "max_drawdown" in CS.METRICS
