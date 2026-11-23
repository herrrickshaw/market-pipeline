"""intraday_monitor.py -- market-hours helpers, intraday fetch, the six pattern detectors, the
combined scan, persistence, summary printing and the main() daemon loop. yfinance is faked; the
module does sys.exit() without it and creates ./intraday_results at import, so it is imported in a
sandbox cwd. Wall-clock time is pinned by replacing the module's datetime/date classes."""

import importlib
import json
import sys
import types
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

IST = timedelta(hours=5, minutes=30)


class StopLoop(Exception):
    pass


@pytest.fixture
def IM(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yf = types.SimpleNamespace(
        Ticker=lambda s: types.SimpleNamespace(history=lambda **k: pd.DataFrame())
    )
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    sys.modules.pop("intraday_monitor", None)
    mod = importlib.import_module("intraday_monitor")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    yield mod
    sys.modules.pop("intraday_monitor", None)


def pin_clock(monkeypatch, mod, ist):
    """Make the module believe the IST wall-clock is `ist` (datetime)."""
    utc = ist - IST

    class FakeDT(datetime):
        @classmethod
        def utcnow(cls):
            return utc

        @classmethod
        def now(cls, tz=None):
            return ist

    class FakeDate(date):
        @classmethod
        def today(cls):
            return ist.date()

    monkeypatch.setattr(mod, "datetime", FakeDT)
    monkeypatch.setattr(mod, "date", FakeDate)


def bars(rows, day="2026-11-23", start="09:15", freq="15min"):
    """rows: list of (high, low, close, volume); Open = Close."""
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq=freq)
    return pd.DataFrame(
        {
            "Open": [r[2] for r in rows],
            "High": [r[0] for r in rows],
            "Low": [r[1] for r in rows],
            "Close": [r[2] for r in rows],
            "Volume": [float(r[3]) for r in rows],
        },
        index=idx,
    )


def flat(n, price=100.0, vol=100, **kw):
    return bars([(price, price, price, vol)] * n, **kw)


# ── market hours ──────────────────────────────────────────────────────────────
MON = datetime(2026, 11, 23)  # a Monday


@pytest.mark.parametrize(
    "ist,expected",
    [
        (MON.replace(hour=9, minute=14), False),
        (MON.replace(hour=9, minute=15), True),
        (MON.replace(hour=12, minute=0), True),
        (MON.replace(hour=15, minute=30), True),
        (MON.replace(hour=15, minute=31), False),
        (MON.replace(hour=0, minute=0), False),
        (datetime(2026, 11, 28, 11, 0), False),  # Saturday
        (datetime(2026, 11, 29, 11, 0), False),  # Sunday
        (datetime(2026, 11, 27, 11, 0), True),  # Friday
    ],
)
def test_is_market_open(IM, monkeypatch, ist, expected):
    pin_clock(monkeypatch, IM, ist)
    assert IM.is_market_open_nse() is expected


@pytest.mark.parametrize(
    "ist,expected",
    [
        (MON.replace(hour=8, minute=0), 75),  # before today's open
        (MON.replace(hour=9, minute=14), 1),
        (MON.replace(hour=9, minute=15), 1440),  # already open: next day
        (MON.replace(hour=10, minute=0), 1395),
        (datetime(2026, 11, 27, 16, 0), 3915),  # Friday evening -> Monday
        (datetime(2026, 11, 28, 10, 0), 2835),  # Saturday -> Monday
        (datetime(2026, 11, 29, 8, 0), 1515),  # Sunday early -> Monday
    ],
)
def test_minutes_to_next_open(IM, monkeypatch, ist, expected):
    pin_clock(monkeypatch, IM, ist)
    assert IM.minutes_to_next_open() == expected


# ── fetch_intraday ────────────────────────────────────────────────────────────
def install_ticker(monkeypatch, mod, history):
    calls = []

    class T:
        def __init__(self, sym):
            calls.append(sym)

        def history(self, **kw):
            calls.append(kw)
            if isinstance(history, Exception):
                raise history
            return history

    monkeypatch.setattr(mod.yf, "Ticker", T, raising=False)
    return calls


def test_fetch_intraday_standardises(IM, monkeypatch):
    idx = pd.date_range("2026-11-23 03:45", periods=3, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, np.nan],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 20, 30],
            "Dividends": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    calls = install_ticker(monkeypatch, IM, raw)
    df = IM.fetch_intraday("TCS", interval=30, days_back=3)
    assert calls == ["TCS.NS", {"period": "3d", "interval": "30m"}]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2  # NaN row dropped
    assert str(df.index.tz) == "Asia/Kolkata"
    assert df.index[0].hour == 9 and df.index[0].minute == 15


def test_fetch_intraday_defaults_and_naive_index(IM, monkeypatch):
    raw = flat(2)
    calls = install_ticker(monkeypatch, IM, raw)
    df = IM.fetch_intraday("INFY")
    assert calls[1] == {"period": "5d", "interval": "15m"}
    assert df.index.tz is None
    assert len(df) == 2


def test_fetch_intraday_empty_error_and_no_yf(IM, monkeypatch):
    install_ticker(monkeypatch, IM, pd.DataFrame())
    assert IM.fetch_intraday("X").empty
    install_ticker(monkeypatch, IM, RuntimeError("net"))
    assert IM.fetch_intraday("X").empty
    install_ticker(monkeypatch, IM, pd.DataFrame({"Close": [1.0]}))  # missing columns
    assert IM.fetch_intraday("X").empty
    calls = install_ticker(monkeypatch, IM, flat(2))
    monkeypatch.setattr(IM, "_YF_OK", False)
    assert IM.fetch_intraday("X").empty
    assert calls == []


# ── compute_vwap ──────────────────────────────────────────────────────────────
def test_compute_vwap_resets_each_day(IM):
    d1 = bars([(12, 8, 10, 100), (13, 9, 11, 300)], day="2026-11-20")
    d2 = bars([(21, 19, 20, 50), (23, 21, 22, 50)], day="2026-11-23")
    df = pd.concat([d1, d2])
    v = IM.compute_vwap(df)
    assert list(v.index) == list(df.index)
    assert v.iloc[0] == pytest.approx(10.0)
    assert v.iloc[1] == pytest.approx((10 * 100 + 11 * 300) / 400)
    assert v.iloc[2] == pytest.approx(20.0)  # reset, not carried over
    assert v.iloc[3] == pytest.approx((20 * 50 + 22 * 50) / 100)


def test_compute_vwap_typical_price_and_zero_volume(IM):
    df = bars([(12, 6, 9, 0), (12, 6, 9, 10)])  # typical = 9
    v = IM.compute_vwap(df)
    assert np.isnan(v.iloc[0])  # zero cumulative volume -> NaN
    assert v.iloc[1] == pytest.approx(9.0)


# ── ORB ───────────────────────────────────────────────────────────────────────
ORB_BASE = [(102, 98, 100, 100), (104, 99, 101, 100)]  # range 98..104


def test_orb_too_short(IM):
    assert IM.detect_orb(pd.DataFrame()) == {"pattern": "ORB", "signal": None}
    assert IM.detect_orb(bars(ORB_BASE))["signal"] is None
    # exactly n_bars + 1 bars on the latest day works
    assert "orb_high" in IM.detect_orb(bars(ORB_BASE + [(103, 100, 101, 50)]))


def test_orb_ignores_previous_days(IM):
    yday = bars([(200, 190, 195, 10)] * 5, day="2026-11-20")
    today = bars(ORB_BASE, day="2026-11-23")  # only n_bars today
    assert IM.detect_orb(pd.concat([yday, today]))["signal"] is None
    today2 = bars(ORB_BASE + [(110, 103, 105, 60)], day="2026-11-23")
    r = IM.detect_orb(pd.concat([yday, today2]))
    assert r["signal"] == "BREAKOUT_LONG"
    assert r["orb_high"] == 104.0 and r["orb_low"] == 98.0 and r["orb_range"] == 6.0


def test_orb_breakout_breakdown_and_inside(IM):
    up = IM.detect_orb(bars(ORB_BASE + [(110, 103, 105, 60)]))
    assert up["signal"] == "BREAKOUT_LONG"
    assert up["current"] == 105.0 and up["range_pct"] == round(6 / 105 * 100, 2)
    assert up["time"].startswith("2026-11-23 09:45")
    down = IM.detect_orb(bars(ORB_BASE + [(99, 90, 95, 60)]))
    assert down["signal"] == "BREAKDOWN_SHORT"
    assert IM.detect_orb(bars(ORB_BASE + [(103, 99, 101, 60)]))["signal"] is None


def test_orb_boundaries_are_strict(IM):
    assert IM.detect_orb(bars(ORB_BASE + [(110, 100, 104, 60)]))["signal"] is None
    assert IM.detect_orb(bars(ORB_BASE + [(110, 90, 98, 60)]))["signal"] is None
    assert IM.detect_orb(bars(ORB_BASE + [(110, 90, 104.5, 60)]))["signal"] == "BREAKOUT_LONG"
    assert IM.detect_orb(bars(ORB_BASE + [(110, 90, 97.5, 60)]))["signal"] == "BREAKDOWN_SHORT"


def test_orb_n_bars_parameter(IM):
    df = bars([(102, 98, 100, 1), (130, 70, 100, 1), (105, 99, 103, 1), (106, 99, 105, 1)])
    r1 = IM.detect_orb(df, n_bars=1)  # range 98..102 -> 105 above
    assert r1["signal"] == "BREAKOUT_LONG" and r1["orb_high"] == 102.0
    r2 = IM.detect_orb(df, n_bars=2)  # range 70..130 -> inside
    assert r2["signal"] is None and r2["orb_high"] == 130.0 and r2["orb_low"] == 70.0


# ── VWAP deviation ────────────────────────────────────────────────────────────
def vwap_frame(last):
    return bars([(100, 100, 100, 100)] * 4 + [(last, last, last, 100)])


def test_vwap_short_input(IM):
    assert IM.detect_vwap_deviation(pd.DataFrame()) == {"pattern": "VWAP", "signal": None}
    assert IM.detect_vwap_deviation(flat(4)) == {"pattern": "VWAP", "signal": None}


def test_vwap_above_below_and_neutral(IM):
    up = IM.detect_vwap_deviation(vwap_frame(102.0))
    assert up["signal"] == "ABOVE_VWAP_REVERT"
    assert up["vwap"] == 100.4 and up["current"] == 102.0
    assert up["deviation%"] == round((102 - 100.4) / 100.4 * 100, 2)
    down = IM.detect_vwap_deviation(vwap_frame(98.0))
    assert down["signal"] == "BELOW_VWAP_REVERT"
    assert down["deviation%"] < 0
    near = IM.detect_vwap_deviation(vwap_frame(100.5))
    assert near["signal"] is None and near["deviation%"] > 0
    assert IM.detect_vwap_deviation(flat(6))["deviation%"] == 0.0


def test_vwap_threshold_boundary_is_inclusive(IM):
    df = vwap_frame(102.0)
    cv = float(IM.compute_vwap(df).iloc[-1])
    dev = (102.0 - cv) / cv * 100
    assert IM.detect_vwap_deviation(df, threshold_pct=dev)["signal"] == "ABOVE_VWAP_REVERT"
    assert IM.detect_vwap_deviation(df, threshold_pct=dev + 1e-6)["signal"] is None
    df2 = vwap_frame(98.0)
    cv2 = float(IM.compute_vwap(df2).iloc[-1])
    dev2 = (98.0 - cv2) / cv2 * 100
    assert IM.detect_vwap_deviation(df2, threshold_pct=-dev2)["signal"] == "BELOW_VWAP_REVERT"
    assert IM.detect_vwap_deviation(df2, threshold_pct=-dev2 + 1e-6)["signal"] is None


def test_vwap_bad_frame_is_swallowed(IM):
    bad = flat(6).drop(columns=["High"])
    assert IM.detect_vwap_deviation(bad) == {"pattern": "VWAP", "signal": None}


def test_vwap_nan_vwap_falls_back_to_current(IM):
    # zero volume throughout -> VWAP NaN; the deviation of a NaN is NaN -> no signal
    df = bars([(100, 100, 100, 0)] * 6)
    r = IM.detect_vwap_deviation(df)
    assert r["signal"] is None


# ── volume surge ──────────────────────────────────────────────────────────────
def surge_frame(last, n=10, base=100):
    return bars([(1, 1, 1, base)] * n + [(1, 1, 1, last)])


def test_volume_surge_short_and_zero_average(IM):
    assert IM.detect_volume_surge(pd.DataFrame()) == {"pattern": "VOL_SURGE", "signal": None}
    assert IM.detect_volume_surge(flat(10)) == {"pattern": "VOL_SURGE", "signal": None}
    assert IM.detect_volume_surge(surge_frame(500, base=0)) == {
        "pattern": "VOL_SURGE",
        "signal": None,
    }


def test_volume_surge_threshold(IM):
    hit = IM.detect_volume_surge(surge_frame(300))
    assert hit["signal"] == "SURGE" and hit["ratio"] == 3.0  # >= is inclusive
    assert hit["curr_vol"] == 300 and hit["avg_vol"] == 100
    assert hit["time"].startswith("2026-11-23")
    miss = IM.detect_volume_surge(surge_frame(299))
    assert miss["signal"] is None and miss["ratio"] == 2.99
    assert IM.detect_volume_surge(surge_frame(150), surge_factor=1.5)["signal"] == "SURGE"
    assert IM.detect_volume_surge(surge_frame(150), surge_factor=1.51)["signal"] is None


def test_volume_surge_window_is_previous_ten_bars(IM):
    rows = [(1, 1, 1, 1_000_000)] + [(1, 1, 1, 100)] * 10 + [(1, 1, 1, 300)]
    r = IM.detect_volume_surge(bars(rows))
    assert r["avg_vol"] == 100 and r["signal"] == "SURGE"  # old spike and current bar excluded


# ── momentum burst ────────────────────────────────────────────────────────────
def closes(*cs):
    return bars([(c, c, c, 10) for c in cs])


def test_momentum_short_input(IM):
    assert IM.detect_momentum_burst(pd.DataFrame()) == {"pattern": "MOMENTUM", "signal": None}
    assert IM.detect_momentum_burst(closes(1, 2, 3))["signal"] is None  # need n+1 bars
    assert IM.detect_momentum_burst(closes(1, 2, 3, 4))["signal"] == "BULLISH_BURST"


def test_momentum_directions(IM):
    up = IM.detect_momentum_burst(closes(9, 1, 2, 3))
    assert up["signal"] == "BULLISH_BURST"
    assert up["bars_checked"] == 3 and up["recent_closes"] == [1.0, 2.0, 3.0]
    assert IM.detect_momentum_burst(closes(1, 9, 8, 7))["signal"] == "BEARISH_BURST"
    assert IM.detect_momentum_burst(closes(1, 2, 3, 2))["signal"] is None
    assert IM.detect_momentum_burst(closes(1, 3, 2, 4))["signal"] is None
    assert IM.detect_momentum_burst(closes(1, 1, 2, 3))["signal"] == "BULLISH_BURST"
    assert IM.detect_momentum_burst(closes(1, 2, 2, 3))["signal"] is None  # flat step breaks it
    assert IM.detect_momentum_burst(closes(1, 3, 3, 2))["signal"] is None


def test_momentum_n_bars_parameter(IM):
    df = closes(5, 4, 1, 2, 3, 4)
    assert IM.detect_momentum_burst(df, n_bars=4)["signal"] == "BULLISH_BURST"
    assert IM.detect_momentum_burst(df, n_bars=5)["signal"] is None
    r = IM.detect_momentum_burst(df, n_bars=2)
    assert r["bars_checked"] == 2 and r["recent_closes"] == [3.0, 4.0]


# ── Bollinger squeeze ─────────────────────────────────────────────────────────
def squeeze_frame(last, n=25):
    return bars([(c, c, c, 10) for c in [100.0] * n + [last]])


def test_bb_too_short(IM):
    assert IM.detect_bb_squeeze(pd.DataFrame()) == {"pattern": "BB_SQUEEZE", "signal": None}
    assert IM.detect_bb_squeeze(flat(21)) == {"pattern": "BB_SQUEEZE", "signal": None}
    assert "bb_upper" in IM.detect_bb_squeeze(flat(22))


def test_bb_squeeze_breakouts(IM):
    up = IM.detect_bb_squeeze(squeeze_frame(110.0))
    assert up["signal"] == "SQUEEZE_BREAKOUT_UP" and up["in_squeeze"]
    assert up["current"] == 110.0
    assert up["bb_upper"] == round(100.5 + 2 * np.sqrt(5.0), 2)
    assert up["bb_lower"] == round(100.5 - 2 * np.sqrt(5.0), 2)
    assert up["bb_width%"] == round((4 * np.sqrt(5.0)) / 110.0 * 100, 2)
    down = IM.detect_bb_squeeze(squeeze_frame(90.0))
    assert down["signal"] == "SQUEEZE_BREAKOUT_DOWN"
    inside = IM.detect_bb_squeeze(squeeze_frame(100.0))
    assert inside["signal"] is None and inside["in_squeeze"]


def test_bb_no_squeeze_means_no_signal(IM):
    wide = [100.0 + (10 if i % 2 else -10) for i in range(25)] + [140.0]
    r = IM.detect_bb_squeeze(bars([(c, c, c, 10) for c in wide]))
    assert not r["in_squeeze"] and r["signal"] is None  # outside the band but no prior squeeze
    r2 = IM.detect_bb_squeeze(squeeze_frame(110.0), squeeze_factor=-1.0)
    assert not r2["in_squeeze"] and r2["signal"] is None


def test_bb_squeeze_only_looks_at_last_five_prior_widths(IM):
    # a squeeze 8 bars ago followed by wide bands: ignored
    vals = [100.0] * 15 + [100.0, 110.0, 90.0, 112.0, 88.0, 114.0, 86.0, 116.0, 84.0, 118.0]
    r = IM.detect_bb_squeeze(bars([(c, c, c, 10) for c in vals]), period=5)
    assert not r["in_squeeze"]


def test_bb_period_and_num_std_parameters(IM):
    df = squeeze_frame(110.0, n=8)
    r = IM.detect_bb_squeeze(df, period=4, num_std=1.0)
    # window [100,100,100,110]: mean 102.5, std 5 -> upper 107.5
    assert r["bb_upper"] == 107.5 and r["bb_lower"] == 97.5


# ── intraday Darvas ───────────────────────────────────────────────────────────
def darvas(
    closes_tail,
    spike_at=15,
    n=24,
    highs_const=110.0,
    spike=120.0,
    low_spike_at=None,
    lows_const=100.0,
):
    """n bars; box top = spike at index spike_at, box bottom = 95 at spike_at (lows)."""
    highs = [highs_const] * n
    lows = [lows_const] * n
    highs[spike_at] = spike
    lows[spike_at if low_spike_at is None else low_spike_at] = 95.0
    cl = [105.0] * n
    for k, v in enumerate(reversed(closes_tail)):
        cl[n - 1 - k] = v
    return bars([(h, lo, c, 10) for h, lo, c in zip(highs, lows, cl, strict=True)])


def test_darvas_too_short(IM):
    r = IM.detect_intraday_darvas(flat(19))
    assert r["signal"] is None and r["note"] == "need ≥20 bars, have 19"
    assert IM.detect_intraday_darvas(pd.DataFrame())["signal"] is None
    assert IM.detect_intraday_darvas(flat(20))["signal"] == "NO_BOX"


def test_darvas_no_box_when_highs_keep_rising(IM):
    rows = [(100 + i, 90 + i, 95 + i, 10) for i in range(24)]
    r = IM.detect_intraday_darvas(bars(rows))
    assert r == {"pattern": "INTRADAY_DARVAS", "signal": "NO_BOX", "current": 118.0}


def test_darvas_in_box_and_upside(IM):
    r = IM.detect_intraday_darvas(darvas([105.0]))
    assert r["signal"] == "IN_BOX"
    assert r["box_top"] == 120.0 and r["box_bottom"] == 95.0 and r["current"] == 105.0
    assert r["upside%"] == round((120 - 105) / 105 * 100, 2)
    assert r["time"].startswith("2026-11-23")


@pytest.mark.parametrize(
    "tail,signal",
    [
        ([115.0, 125.0], "INTRADAY_BREAKOUT"),
        ([120.0, 125.0], "INTRADAY_BREAKOUT"),  # prev == top still counts as a fresh breakout
        ([121.0, 125.0], "ABOVE_BOX"),
        ([120.0, 120.0], "IN_BOX"),  # close == top is not above it
        ([100.0, 90.0], "INTRADAY_BREAKDOWN"),
        ([95.0, 90.0], "INTRADAY_BREAKDOWN"),  # prev == bottom still counts
        ([94.0, 90.0], "BELOW_BOX"),
        ([100.0, 95.0], "IN_BOX"),  # close == bottom is not below it
    ],
)
def test_darvas_signals(IM, tail, signal):
    assert IM.detect_intraday_darvas(darvas(tail))["signal"] == signal


def test_darvas_search_reaches_first_bar(IM):
    r = IM.detect_intraday_darvas(darvas([105.0], spike_at=0, low_spike_at=0))
    assert r["signal"] == "IN_BOX" and r["box_top"] == 120.0


def test_darvas_search_window_is_30_bars(IM):
    # 40 bars: highs[:-1] has 39 entries, scan covers indices 9..35
    found = IM.detect_intraday_darvas(darvas([105.0], spike_at=9, n=40))
    assert found["box_top"] == 120.0
    missed = IM.detect_intraday_darvas(darvas([105.0], spike_at=8, n=40))
    assert missed["signal"] == "NO_BOX"


def test_darvas_current_bar_excluded_from_box(IM):
    df = darvas([105.0])
    df.iloc[-1, df.columns.get_loc("High")] = 500.0
    assert IM.detect_intraday_darvas(df)["box_top"] == 120.0


def test_darvas_bottom_fallback_to_lowest_valid_low(IM):
    n = 24
    highs = [110.0] * n
    highs[15] = 120.0
    lows = [100.0 - 0.5 * i for i in range(n)]  # strictly falling: no confirmed floor
    lows[22] = 0.0  # zero lows are ignored by the fallback
    df = bars([(h, lo, 105.0, 10) for h, lo in zip(highs, lows, strict=True)])
    r = IM.detect_intraday_darvas(df)
    assert r["box_bottom"] == lows[21]


def test_darvas_no_valid_lows(IM):
    n = 24
    highs = [110.0] * n
    highs[15] = 120.0
    lows = [100.0] * 15 + [0.0] * 9
    df = bars([(h, lo, 105.0, 10) for h, lo in zip(highs, lows, strict=True)])
    r = IM.detect_intraday_darvas(df)
    assert r["signal"] == "NO_BOX" and r["box_top"] == 120.0 and r["current"] == 105.0


def test_darvas_confirm_parameter(IM):
    highs = [100.0] * 15 + [120.0, 110.0] + [125.0 + k for k in range(7)]  # 24 bars
    df = bars([(h, 90.0, 105.0, 10) for h in highs])
    # one lower bar after the 120 peak confirms it for confirm=1 but not for confirm=2
    assert IM.detect_intraday_darvas(df, confirm=2)["signal"] == "NO_BOX"
    r = IM.detect_intraday_darvas(df, confirm=1)
    assert r["signal"] != "NO_BOX" and r["box_top"] == 120.0


# ── scan_one ──────────────────────────────────────────────────────────────────
def stub_detectors(monkeypatch, mod, signals):
    names = [
        "detect_orb",
        "detect_vwap_deviation",
        "detect_volume_surge",
        "detect_momentum_burst",
        "detect_bb_squeeze",
        "detect_intraday_darvas",
    ]
    for name, sig in zip(names, signals, strict=True):
        monkeypatch.setattr(mod, name, lambda df, _s=sig, _n=name: {"pattern": _n, "signal": _s})


def test_scan_one_no_data(IM, monkeypatch):
    monkeypatch.setattr(IM, "fetch_intraday", lambda *a, **k: pd.DataFrame())
    assert IM.scan_one("X") == {"symbol": "X", "error": "no_data", "signals": []}


def test_scan_one_fetch_arguments_and_fields(IM, monkeypatch):
    seen = {}

    def fake_fetch(sym, interval=15, days_back=5):
        seen.update(sym=sym, interval=interval, days_back=days_back)
        return bars([(10, 9, 9.876, 1234), (11, 9, 10.123, 4321)])

    monkeypatch.setattr(IM, "fetch_intraday", fake_fetch)
    stub_detectors(monkeypatch, IM, [None] * 6)
    r = IM.scan_one("ABC", interval=30)
    assert seen == {"sym": "ABC", "interval": 30, "days_back": 5}
    assert r["symbol"] == "ABC" and r["interval_m"] == 30
    assert r["price"] == 10.12 and r["volume"] == 4321
    assert r["time"].startswith("2026-11-23 09:30")
    assert r["confluence"] == 0 and r["direction"] == "NEUTRAL"
    assert len(r["patterns"]) == 6 and r["active"] == []


@pytest.mark.parametrize(
    "signals,confluence,bull,bear,direction",
    [
        (
            ["BREAKOUT_LONG", None, "SURGE", "BULLISH_BURST", None, "INTRADAY_BREAKOUT"],
            4,
            3,
            0,
            "BULLISH",
        ),
        (["BREAKDOWN_SHORT", "BELOW_VWAP_REVERT", None, None, None, None], 2, 0, 2, "BEARISH"),
        (["BREAKOUT_LONG", "BELOW_VWAP_REVERT", None, None, None, None], 2, 1, 1, "NEUTRAL"),
        ([None, None, "SURGE", None, None, "IN_BOX"], 1, 0, 0, "NEUTRAL"),  # IN_BOX is not active
        (
            [None, None, None, None, "SQUEEZE_BREAKOUT_DOWN", None],
            1,
            1,
            1,
            "NEUTRAL",
        ),  # BREAKOUT and DOWN
        ([None, "ABOVE_VWAP_REVERT", None, None, None, "ABOVE_BOX"], 2, 2, 1, "BULLISH"),
        ([None, None, None, "BEARISH_BURST", None, "BELOW_BOX"], 2, 1, 2, "BEARISH"),
        ([None, None, None, None, None, "INTRADAY_BREAKDOWN"], 1, 0, 1, "BEARISH"),
        ([None, None, None, None, "SQUEEZE_BREAKOUT_UP", None], 1, 1, 0, "BULLISH"),
    ],
)
def test_scan_one_aggregation(IM, monkeypatch, signals, confluence, bull, bear, direction):
    monkeypatch.setattr(IM, "fetch_intraday", lambda *a, **k: flat(3))
    stub_detectors(monkeypatch, IM, signals)
    r = IM.scan_one("X")
    assert (r["confluence"], r["bullish_n"], r["bearish_n"], r["direction"]) == (
        confluence,
        bull,
        bear,
        direction,
    )
    assert [p["signal"] for p in r["active"]] == [s for s in signals if s and s != "IN_BOX"]


def test_scan_one_with_real_detectors(IM, monkeypatch):
    df = bars(ORB_BASE + [(110, 103, 105, 60)] + [(110, 103, 105, 60)] * 4)
    monkeypatch.setattr(IM, "fetch_intraday", lambda *a, **k: df)
    r = IM.scan_one("X")
    assert [p["pattern"] for p in r["patterns"]] == [
        "ORB",
        "VWAP",
        "VOL_SURGE",
        "MOMENTUM",
        "BB_SQUEEZE",
        "INTRADAY_DARVAS",
    ]
    assert r["patterns"][0]["signal"] == "BREAKOUT_LONG"


# ── run_scan ──────────────────────────────────────────────────────────────────
def fake_result(sym, confluence, direction="BULLISH"):
    return {
        "symbol": sym,
        "time": "2026-11-23 10:00:00",
        "price": 100.0 + confluence,
        "volume": 1000,
        "direction": direction,
        "confluence": confluence,
        "bullish_n": confluence,
        "bearish_n": 0,
        "patterns": [
            {"pattern": "ORB", "signal": "BREAKOUT_LONG", "orb_high": 104.0, "orb_low": 98.0},
            {"pattern": "VWAP", "signal": None, "deviation%": 0.5},
            {"pattern": "VOL_SURGE", "signal": "SURGE", "ratio": 3.2},
            {
                "pattern": "INTRADAY_DARVAS",
                "signal": "IN_BOX",
                "box_top": 120.0,
                "box_bottom": 95.0,
            },
        ],
    }


def test_run_scan_flattens_sorts_and_skips_errors(IM, monkeypatch, capsys):
    def fake_scan(sym, interval):
        if sym == "BOOM":
            raise RuntimeError("explode")
        if sym == "NODATA":
            return {"symbol": sym, "error": "no_data", "signals": []}
        return fake_result(sym, {"A": 1, "B": 4, "C": 2}[sym])

    monkeypatch.setattr(IM, "scan_one", fake_scan)
    df = IM.run_scan(["A", "B", "BOOM", "C", "NODATA"], interval=30, workers=1)
    assert list(df["Symbol"]) == ["B", "C", "A"]
    assert list(df["Confluence"]) == [4, 2, 1]
    row = df.iloc[0]
    assert row["Price_₹"] == 104.0 and row["Volume"] == 1000 and row["Direction"] == "BULLISH"
    assert row["Bullish_N"] == 4 and row["Bearish_N"] == 0 and row["Time"].startswith("2026-11-23")
    assert row["ORB_signal"] == "BREAKOUT_LONG" and row["VOL_SURGE_signal"] == "SURGE"
    assert row["INTRADAY_DARVAS_signal"] == "IN_BOX" and pd.isna(row["VWAP_signal"])
    assert row["ORB_High"] == 104.0 and row["ORB_Low"] == 98.0
    assert row["VWAP_Dev%"] == 0.5 and row["Vol_Ratio"] == 3.2
    assert row["Box_Top"] == 120.0 and row["Box_Bottom"] == 95.0
    out = capsys.readouterr().out
    assert "5/5 scanned | 3 with signals" in out
    assert out.count("scanned") == 1


def test_run_scan_progress_every_ten(IM, monkeypatch, capsys):
    monkeypatch.setattr(IM, "scan_one", lambda s, i: fake_result(s, int(s[1:]) + 1))
    df = IM.run_scan([f"S{k}" for k in range(12)], workers=2)
    assert len(df) == 12 and df["Confluence"].is_monotonic_decreasing
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "scanned" in ln]
    assert len(lines) == 2
    assert lines[0].strip().startswith("10/12 scanned")
    assert lines[1].strip().startswith("12/12 scanned")


def test_run_scan_zero_confluence_not_counted_active(IM, monkeypatch, capsys):
    monkeypatch.setattr(IM, "scan_one", lambda s, i: fake_result(s, 0, "NEUTRAL"))
    df = IM.run_scan(["A", "B"], workers=1)
    assert len(df) == 2
    assert "2/2 scanned | 0 with signals" in capsys.readouterr().out


def test_run_scan_empty_results(IM, monkeypatch, capsys):
    monkeypatch.setattr(
        IM, "scan_one", lambda s, i: {"symbol": s, "error": "no_data", "signals": []}
    )
    df = IM.run_scan(["A"], workers=1)
    assert df.empty
    assert IM.run_scan([], workers=1).empty


def test_run_scan_default_worker_count(IM):
    assert IM.run_scan.__defaults__ == (15, IM.MAX_WORKERS)


# ── save_signals ──────────────────────────────────────────────────────────────
def signals_df(confluences):
    return pd.DataFrame(
        {
            "Symbol": [f"S{i}" for i in range(len(confluences))],
            "Confluence": confluences,
            "Direction": ["BULLISH"] * len(confluences),
        }
    )


def test_save_signals_writes_json_and_excel(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    jp, xp = IM.save_signals(signals_df([4, 2]), 15)
    assert jp.name == "intraday_20261123_15m.json" and xp.name == "intraday_20261123.xlsx"
    data = json.loads(jp.read_text())
    assert len(data) == 1 and data[0]["scan_time"] == "2026-11-23T10:05:00"
    assert [r["Symbol"] for r in data[0]["signals"]] == ["S0", "S1"]
    sheets = pd.read_excel(xp, sheet_name=None)
    assert set(sheets) == {"Scan_1005_15m", "HighConv_1005"}
    assert list(sheets["HighConv_1005"]["Symbol"]) == ["S0"]
    assert len(sheets["Scan_1005_15m"]) == 2


def test_save_signals_high_conviction_threshold(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    _, xp = IM.save_signals(signals_df([3, 2]), 15)
    assert list(pd.read_excel(xp, sheet_name="HighConv_1005")["Confluence"]) == [3]
    sub = IM.OUT_DIR / "sub"
    sub.mkdir()
    monkeypatch.setattr(IM, "OUT_DIR", sub)
    _, xp2 = IM.save_signals(signals_df([2, 1]), 15)
    assert set(pd.read_excel(xp2, sheet_name=None)) == {"Scan_1005_15m"}  # none reach 3


def test_save_signals_appends_json_and_sheets(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    IM.save_signals(signals_df([4]), 15)
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 20, 0))
    jp, xp = IM.save_signals(signals_df([5, 1]), 15)
    data = json.loads(jp.read_text())
    assert [d["scan_time"] for d in data] == ["2026-11-23T10:05:00", "2026-11-23T10:20:00"]
    assert len(data[1]["signals"]) == 2
    sheets = pd.read_excel(xp, sheet_name=None)
    assert set(sheets) == {"Scan_1005_15m", "HighConv_1005", "Scan_1020_15m", "HighConv_1020"}


def test_save_signals_replaces_same_named_sheet(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    IM.save_signals(signals_df([4]), 15)
    _, xp = IM.save_signals(signals_df([9, 8]), 15)  # same minute: sheet is overwritten
    sheets = pd.read_excel(xp, sheet_name=None)
    assert list(sheets["Scan_1005_15m"]["Confluence"]) == [9, 8]


def test_save_signals_separate_json_per_interval(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    j15, _ = IM.save_signals(signals_df([4]), 15)
    j30, _ = IM.save_signals(signals_df([4]), 30)
    assert j15 != j30 and j30.name.endswith("_30m.json")
    assert len(json.loads(j30.read_text())) == 1


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="empty df: openpyxl is given a workbook with no sheet, so the Excel write raises "
    "(intraday_monitor.py save_signals); main() guards against it, direct callers do not",
)
def test_save_signals_empty_frame(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5, 0))
    IM.save_signals(pd.DataFrame(), 15)


# ── print_summary ─────────────────────────────────────────────────────────────
def summary_df(rows):
    return pd.DataFrame(
        rows, columns=["Symbol", "Price_₹", "Confluence", "Direction", "ORB_signal", "VWAP_signal"]
    )


def test_print_summary_empty(IM, monkeypatch, capsys):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5))
    IM.print_summary(pd.DataFrame(), 15)
    out = capsys.readouterr().out
    assert "INTRADAY MONITOR — 23 Nov 2026  10:05 IST | 15-min bars" in out
    assert "No signals detected." in out
    assert "HIGH CONVICTION" not in out


def test_print_summary_lists(IM, monkeypatch, capsys):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5))
    rows = [("BULL1", 1234.5, 3, "BULLISH", "BREAKOUT_LONG", "ABOVE_VWAP_REVERT")]
    rows += [(f"BULL{k}", 100.0, 1, "BULLISH", None, None) for k in range(2, 13)]
    rows += [("BEAR1", 99.5, 2, "BEARISH", None, "BELOW_VWAP_REVERT")]
    rows += [(f"BEAR{k}", 50.0, 1, "BEARISH", None, None) for k in range(2, 8)]
    rows += [("NEUT", 10.0, 0, "NEUTRAL", None, None)]
    IM.print_summary(summary_df(rows), 30)
    out = capsys.readouterr().out
    assert "30-min bars" in out
    assert "Scanned: 20 stocks | BULLISH: 12 | BEARISH: 7" in out
    assert "TOP BULLISH SIGNALS" in out and "TOP BEARISH SIGNALS" in out
    assert "BULL10" in out and "BULL11" not in out  # head(10)
    assert "BEAR5" in out and "BEAR6" not in out  # head(5)
    assert "NEUT" not in out.replace("NEUTRAL", "")
    assert "₹1,234.50" in out
    assert "ORB, VWAP" in out  # patterns that fired, in column order
    assert "HIGH CONVICTION (3+ patterns): BULL1" in out
    assert IM.DISCLAIMER in out


def test_print_summary_only_bearish_and_no_high_conviction(IM, monkeypatch, capsys):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5))
    IM.print_summary(summary_df([("B", 10.0, 2, "BEARISH", None, None)]), 15)
    out = capsys.readouterr().out
    assert "TOP BEARISH SIGNALS" in out
    assert "TOP BULLISH SIGNALS" not in out
    assert "HIGH CONVICTION" not in out  # 2 < 3


# ── main ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def daemon(IM, monkeypatch):
    pin_clock(monkeypatch, IM, datetime(2026, 11, 23, 10, 5))
    calls = types.SimpleNamespace(sleeps=[], scans=[], saves=[], prints=[], opens=[], waits=[])
    calls.open_seq = []
    calls.df = signals_df([4])

    def is_open():
        calls.opens.append(1)
        return calls.open_seq.pop(0)

    def sleeper(s):
        calls.sleeps.append(s)
        if len(calls.sleeps) >= calls.max_sleeps:
            raise StopLoop

    calls.max_sleeps = 99
    calls.wait = 7
    monkeypatch.setattr(IM, "is_market_open_nse", is_open)
    monkeypatch.setattr(IM, "minutes_to_next_open", lambda: calls.wait)
    monkeypatch.setattr(IM.time, "sleep", sleeper)
    monkeypatch.setattr(
        IM,
        "run_scan",
        lambda syms, interval, workers: (calls.scans.append((syms, interval, workers)), calls.df)[
            1
        ],
    )
    monkeypatch.setattr(
        IM,
        "save_signals",
        lambda df, i: (calls.saves.append(i), (IM.OUT_DIR / "a.json", IM.OUT_DIR / "b.xlsx"))[1],
    )
    monkeypatch.setattr(IM, "print_summary", lambda df, i: calls.prints.append(i))
    return calls


def test_main_once_while_closed(IM, daemon, capsys):
    daemon.open_seq = [False]
    IM.main(once=True)
    out = capsys.readouterr().out
    assert "Market closed. Run with --once during market hours." in out
    assert daemon.scans == [] and daemon.sleeps == []
    assert "Stocks: 50 | Workers: 6" in out and "INTRADAY MONITOR — 15-min bars" in out
    assert IM.DISCLAIMER in out


def test_main_once_while_open(IM, daemon, capsys):
    daemon.open_seq = [True]
    IM.main(interval=30, symbols=["A", "B"], workers=3, once=True)
    out = capsys.readouterr().out
    assert daemon.scans == [(["A", "B"], 30, 3)]
    assert daemon.saves == [30] and daemon.prints == [30]
    assert daemon.sleeps == []
    assert "Stocks: 2 | Workers: 3" in out
    assert "[10:05 IST] Scan #1 — 2 stocks" in out
    assert "Saved: a.json | b.xlsx" in out


def test_main_defaults_to_nifty_50(IM, daemon):
    daemon.open_seq = [True]
    IM.main(once=True)
    assert daemon.scans == [(IM.NIFTY_50, 15, IM.MAX_WORKERS)]


def test_main_empty_scan_skips_persistence(IM, daemon, capsys):
    daemon.open_seq = [True]
    daemon.df = pd.DataFrame()
    IM.main(once=True)
    assert daemon.saves == [] and daemon.prints == []
    assert "Saved:" not in capsys.readouterr().out


@pytest.mark.parametrize("wait,expected", [(7, 420), (60, 3600), (61, 3600), (500, 3600)])
def test_main_sleeps_while_closed_capped_at_an_hour(IM, daemon, capsys, wait, expected):
    daemon.open_seq = [False]
    daemon.wait = wait
    daemon.max_sleeps = 1
    with pytest.raises(StopLoop):
        IM.main()
    assert daemon.sleeps == [expected]
    assert f"sleeping {wait} minutes until open" in capsys.readouterr().out


def test_main_loops_until_stopped(IM, daemon, capsys):
    daemon.open_seq = [False, True, True]
    daemon.max_sleeps = 3
    with pytest.raises(StopLoop):
        IM.main(interval=15, symbols=["A"], workers=2)
    assert daemon.sleeps == [420, 900, 900]  # closed wait, then two inter-scan sleeps
    assert len(daemon.scans) == 2 and daemon.saves == [15, 15]
    out = capsys.readouterr().out
    assert "Scan #1" in out and "Scan #2" in out
    assert "Next scan in 15 minutes" in out
