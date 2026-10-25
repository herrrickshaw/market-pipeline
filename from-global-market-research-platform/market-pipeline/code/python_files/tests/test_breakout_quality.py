"""breakout_quality.darvas_box — the box recomputation signal_tracker trusts
(the stored Darvas_Signal column was corrupt until 2026-07-21). It must agree
with scan_core on the same bars and honour the same current-bar-excluded rule.
"""
import numpy as np
import pandas as pd
import pytest

import breakout_quality as bq
import scan_core as sc
from test_scan_core import HIGHS, LOWS, with_today


@pytest.mark.parametrize("today", [(16, 15, 15.5), (14.5, 13.6, 14.0), (13.0, 12.0, 12.5)])
def test_agrees_with_scan_core(today):
    df = with_today(*today)
    a, b = bq.darvas_box(df), sc.compute_darvas_box(df)
    assert (a["signal"], a["box_top"], a["box_bottom"]) == \
        (b["signal"], b["box_top"], b["box_bottom"])


def test_hand_computed_box_and_age():
    r = bq.darvas_box(with_today(16, 15, 15.5))
    assert (r["box_top"], r["box_bottom"], r["signal"]) == (14.8, 13.0, "BREAKOUT_BUY")
    assert r["box_start_idx"] == 5
    assert r["box_bars"] == 4                       # bars 6..9 after the top
    assert r["box_width_pct"] == pytest.approx(1.8 / 13.0 * 100)


def test_breakdown_needs_current_low_excluded():
    lows = LOWS[:5] + [13.0, 12.8, 12.6, 12.4]
    r = bq.darvas_box(with_today(12.0, 11.0, 11.5, lows=lows))
    assert (r["box_bottom"], r["signal"]) == (12.4, "BREAKDOWN_SELL")


def test_box_ignores_current_bar_high():
    assert bq.darvas_box(with_today(1e6, 15, 15.5))["box_top"] == 14.8


def test_trailing_nan_close_is_trimmed():
    df = with_today(16, 15, 15.5)
    df.loc[df.index[-1] + pd.offsets.BDay(1)] = [np.nan, np.nan, np.nan]
    r = bq.darvas_box(df)
    assert (r["current"], r["signal"]) == (15.5, "BREAKOUT_BUY")


def test_no_close_or_short_history_returns_empty():
    df = with_today(16, 15, 15.5)
    assert bq.darvas_box(df.assign(Close=np.nan)) == {}
    assert bq.darvas_box(df.iloc[:6]) == {}
