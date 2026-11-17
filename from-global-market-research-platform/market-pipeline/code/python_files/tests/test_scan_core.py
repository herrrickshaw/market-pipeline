"""Darvas box logic in scan_core.compute_darvas_box (the shared implementation
behind all five full_*_market_scan.py modules).

Design rules under test:
  * the CURRENT bar is excluded from box formation
  * trailing NaN bars are dropped, never zero-filled
  * a missing / non-positive final close is NO_DATA, never IN_BOX

Fixture (confirm=3), bars 0..8 form the box, bar 9 is "today":
  highs 10 11 12 15 14.5 [14.8] 14.2 14.6 14.4   -> newest confirmed top = 14.8 (bar 5)
  lows  ...           [13.0] 13.5 13.8 13.9      -> bottom = 13.0 (next 3 lows all higher)
"""
import numpy as np
import pandas as pd
import pytest

import scan_core as sc

HIGHS = [10, 11, 12, 15, 14.5, 14.8, 14.2, 14.6, 14.4]
LOWS = [9, 10, 11, 13.5, 13.2, 13.0, 13.5, 13.8, 13.9]
CLOSES = [9.5, 10.5, 11.5, 14.0, 14.0, 14.0, 14.0, 14.2, 14.1]


def frame(highs, lows, closes, cols=("High", "Low", "Close")):
    return pd.DataFrame({cols[0]: highs, cols[1]: lows, cols[2]: closes},
                        index=pd.bdate_range("2026-01-05", periods=len(closes)))


def with_today(high, low, close, highs=HIGHS, lows=LOWS, closes=CLOSES):
    return frame(highs + [high], lows + [low], closes + [close])


def test_breakout_hand_computed():
    r = sc.compute_darvas_box(with_today(16, 15, 15.5))
    assert r["signal"] == "BREAKOUT_BUY"
    assert (r["box_top"], r["box_bottom"], r["current_price"]) == (14.8, 13.0, 15.5)
    assert r["box_range"] == 1.8
    assert r["upside_pct"] == r["upside_to_top_pct"] == round((14.8 - 15.5) / 15.5 * 100, 2)
    assert r["pos_in_box"] == r["position_in_box_pct"] == round(2.5 / 1.8 * 100, 1)
    assert r["data_points"] == 10


def test_in_box():
    r = sc.compute_darvas_box(with_today(14.5, 13.6, 14.0))
    assert r["signal"] == "IN_BOX"
    assert (r["box_top"], r["box_bottom"]) == (14.8, 13.0)


def test_current_bar_extremes_do_not_move_the_box():
    a = sc.compute_darvas_box(with_today(16, 15, 15.5))
    b = sc.compute_darvas_box(with_today(1000, 0.01, 15.5))
    assert (a["box_top"], a["box_bottom"]) == (b["box_top"], b["box_bottom"]) == (14.8, 13.0)


def test_breakdown_is_detected_because_current_low_is_excluded():
    # Lows after the top fall every bar, so no bottom confirms and the box
    # bottom falls back to the segment minimum (12.4). Today's low (11.0) must
    # NOT be part of that minimum, or a close of 11.5 would read IN_BOX.
    lows = LOWS[:5] + [13.0, 12.8, 12.6, 12.4]
    r = sc.compute_darvas_box(with_today(12.0, 11.0, 11.5, lows=lows))
    assert r["box_bottom"] == 12.4
    assert r["signal"] == "BREAKDOWN_SELL"


def test_trailing_nan_bar_is_dropped_not_zero_filled():
    df = with_today(16, 15, 15.5)
    df.loc[df.index[-1] + pd.offsets.BDay(1)] = [np.nan, np.nan, np.nan]
    r = sc.compute_darvas_box(df)
    assert r["signal"] == "BREAKOUT_BUY"
    assert r["current_price"] == 15.5


def test_all_nan_closes_is_no_data():
    df = with_today(16, 15, 15.5)
    df["Close"] = np.nan
    assert sc.compute_darvas_box(df)["signal"] == "NO_DATA"


def test_non_positive_final_close_is_no_data_never_in_box():
    r = sc.compute_darvas_box(with_today(16, 15, 0.0))
    assert r["signal"] == "NO_DATA"
    assert r["current_price"] is None


def test_rising_highs_never_confirm_a_top():
    n = 12
    r = sc.compute_darvas_box(frame(list(range(10, 10 + n)), list(range(9, 9 + n)),
                                    [x + 0.5 for x in range(9, 9 + n)]))
    assert r["signal"] == "NO_BOX"
    assert r["box_top"] is None


@pytest.mark.parametrize("df", [None, pd.DataFrame(),
                                frame([1, 2, 3], [1, 2, 3], [1, 2, 3])])
def test_insufficient_data(df):
    assert sc.compute_darvas_box(df)["signal"] == "INSUFFICIENT_DATA"


def test_close_is_not_resolved_to_adj_close_when_adj_comes_first():
    df = with_today(16, 15, 15.5)
    df.insert(0, "Adj Close", df["Close"] * 0.5)     # dividend-adjusted, lower
    r = sc.compute_darvas_box(df)
    assert r["current_price"] == 15.5
    assert r["signal"] == "BREAKOUT_BUY"


def test_whole_unit_rounding_for_yen_won():
    r = sc.compute_darvas_box(with_today(16, 15, 15.5), decimals=0)
    assert (r["box_top"], r["box_bottom"]) == (15.0, 13.0)


def test_bhavcopy_column_candidates():
    df = with_today(16, 15, 15.5)
    df.columns = ["CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE", "CH_CLOSING_PRICE"]
    r = sc.compute_darvas_box(df, col_candidates={
        "high": ["CH_TRADE_HIGH_PRICE"], "low": ["CH_TRADE_LOW_PRICE"],
        "close": ["CH_CLOSING_PRICE"]})
    assert (r["signal"], r["box_top"], r["box_bottom"]) == ("BREAKOUT_BUY", 14.8, 13.0)
