"""Offline tests for watchlist_digest helpers (cache lookups, zones, hygiene, rows, gaps)."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import watchlist_digest as wd


def _ohlc(n=80, drift=0.001, vol=0.01, seed=0, volume=1e6, dated=False):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2025-01-02", periods=n)
    df = pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, volume)}, index=idx
    )
    if dated:
        df = df.reset_index().rename(columns={"index": "Date"})
    return df


# ---------------------------------------------------------------- _wh_candidates / _yf_ticker
@pytest.mark.parametrize(
    "sym,mkt,expected",
    [
        ("5360", "KR", ["005360.KS", "005360.KQ"]),
        ("005930", "KR", ["005930.KS", "005930.KQ"]),
        ("7203", "JP", ["7203.T"]),
        ("ABC", "KR", ["ABC", "ABC"]),
        ("BRK.B", "US", ["BRK.B", "BRK-B"]),
        ("ADS.DE", "EU", ["ADS.DE", "ADS-DE"]),
    ],
)
def test_wh_candidates(sym, mkt, expected):
    assert wd._wh_candidates(sym, mkt) == expected


@pytest.mark.parametrize(
    "sym,mkt,expected",
    [
        ("TCS", "IN", "TCS.NS"),
        ("5930", "KR", "005930.KS"),
        ("ABC", "KR", "ABC"),
        ("7203", "JP", "7203.T"),
        ("AAPL", "US", "AAPL"),
        ("ADS.DE", "EU", "ADS.DE"),
    ],
)
def test_yf_ticker(sym, mkt, expected):
    assert wd._yf_ticker(sym, mkt) == expected


# ---------------------------------------------------------------- _load_ohlc / warehouse
@pytest.fixture
def dirs(monkeypatch, tmp_path):
    (tmp_path / "ohlc").mkdir()
    monkeypatch.setattr(wd.R, "OHLC_DIR", tmp_path / "ohlc")
    monkeypatch.setattr(wd, "WAREHOUSE", tmp_path / "wh")
    monkeypatch.setattr(wd, "_WH_FRAMES", {})
    return tmp_path


def test_load_ohlc_india_goes_through_bhavcopy_store(monkeypatch, dirs):
    got = []
    monkeypatch.setitem(
        sys.modules, "bhavcopy_store", types.SimpleNamespace(get=lambda s: got.append(s) or "FRAME")
    )
    for m in ("IN", "india", "NS"):
        assert wd._load_ohlc("TCS", m) == "FRAME"
    assert got == ["TCS"] * 3
    monkeypatch.setitem(sys.modules, "bhavcopy_store", None)
    assert wd._load_ohlc("TCS", "IN") is None


def test_load_ohlc_cache_file_spellings_and_corrupt(dirs):
    _ohlc(5).to_parquet(dirs / "ohlc" / "BRK-B.parquet")
    assert len(wd._load_ohlc("BRK.B", "US")) == 5  # dot spelling falls back to dash file
    (dirs / "ohlc" / "BAD.parquet").write_text("junk")
    assert wd._load_ohlc("BAD", "US") is None
    assert wd._load_ohlc("NOPE", "US") is None


def test_load_ohlc_warehouse_two_partitions_and_kr_padding(dirs):
    d = dirs / "wh" / "KR"
    d.mkdir(parents=True)
    for year, base in ((2024, 1.0), (2025, 2.0), (2026, 3.0)):
        pd.DataFrame(
            {
                "Symbol": ["005360.KS", "000001.KQ"],
                "Date": [f"{year}-01-02"] * 2,
                "Close": [base, base],
                "Volume": [1, 1],
            }
        ).to_parquet(d / f"year={year}.parquet")
    out = wd._load_ohlc("5360", "KR")
    assert out["Close"].tolist() == [2.0, 3.0]  # only the last two yearly partitions, date-sorted
    assert wd._load_ohlc("1", "KR")["Symbol"].iloc[0] == "000001.KQ"  # .KQ candidate
    assert wd._load_ohlc("999", "KR") is None
    assert "KR" in wd._WH_FRAMES  # memoised per market


def test_warehouse_frames_failure_is_empty_and_cached(dirs):
    assert wd._warehouse_frames("JP") == {} and wd._warehouse_frames("jp") == {}
    d = dirs / "wh" / "JP"
    d.mkdir(parents=True)
    (d / "year=2025.parquet").write_text("junk")
    wd._WH_FRAMES.clear()
    assert wd._warehouse_frames("JP") == {}


# ---------------------------------------------------------------- _pct_change
def test_pct_change_bars_and_guards():
    df = pd.DataFrame({"Close": [100.0, 110.0, 99.0]})
    assert wd._pct_change(df, 1) == pytest.approx(-10.0)
    assert wd._pct_change(df, 2) == pytest.approx(-1.0)
    assert wd._pct_change(df, 3) is None  # needs bars+1 observations
    assert wd._pct_change(pd.DataFrame({"Close": ["x", "y"]}), 1) is None
    assert wd._pct_change(pd.DataFrame({"Other": [1, 2]}), 1) is None
    assert wd._pct_change(pd.DataFrame({"Close": [1.0, None, 2.0]}), 1) == pytest.approx(
        100.0
    )  # NaNs dropped


# ---------------------------------------------------------------- liquidity labels
def test_liq_label_unknown_floor_tiers(monkeypatch):
    assert wd._liq_label(None, "US") == ("?", False)
    monkeypatch.setitem(
        sys.modules, "adaptive_liquidity", types.SimpleNamespace(scan_floor=lambda m: 1000.0)
    )
    monkeypatch.setitem(
        sys.modules,
        "liquidity",
        types.SimpleNamespace(
            SCAN_TIERS_USD=[(1e8, "T1_MEGA"), (1e7, "T2_LARGE"), (1e6, "T3_MID"), (1e3, "T4_SMALL")]
        ),
    )
    assert wd._liq_label(999.99, "US") == ("—", True)
    assert wd._liq_label(1000.0, "US") == ("T4", False)  # exactly at the floor is tradeable
    assert wd._liq_label(1e6, "US") == ("T3", False) and wd._liq_label(5e8, "US") == ("T1", False)


def test_liq_label_fallback_floors_when_modules_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "adaptive_liquidity", None)
    monkeypatch.setitem(sys.modules, "liquidity", None)
    assert wd._liq_label(119_999.0, "IN") == ("—", True)
    assert wd._liq_label(120_000.0, "IN") == ("T4", False)
    assert wd._liq_label(9_999.0, "US") == ("—", True)
    assert wd._liq_label(10_000.0, "US") == ("T4", False)
    assert wd._liq_label(50.0, "INDIA") == ("—", True)  # startswith("IN")


def test_turnover_usd_median_currency_and_units(monkeypatch):
    L = types.SimpleNamespace(
        currency_for=lambda s: "GBP", CCY={"US": "USD", "IN": "INR"}, SUBUNIT={"GBp": ("GBP", 100)}
    )
    monkeypatch.setitem(sys.modules, "liquidity", L)
    df = pd.DataFrame({"Close": [2.0] * 70, "Volume": [10.0] * 60 + [1e9] * 10})
    assert wd._turnover_usd(df, "ABC", "US", {"USD": 1.0}) == pytest.approx(
        20.0
    )  # median of last 60 bars...
    L.currency_for = lambda s: "GBp"
    assert wd._turnover_usd(None, "A", "US", {}) is None
    assert (
        wd._turnover_usd(pd.DataFrame({"Close": [1.0]}), "A", "US", {}) is None
    )  # no Volume column
    assert wd._turnover_usd(df.head(9), "A", "US", {"USD": 1.0}) is None  # <10 bars
    assert wd._turnover_usd(df, "A", "US", {}) is None  # no FX rate


# ---------------------------------------------------------------- sectors
@pytest.fixture
def sec(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "SECTOR_CACHE", tmp_path / "c" / "sector_map.json")
    return tmp_path


def test_sector_cache_roundtrip(sec):
    assert wd._load_sector_cache() == {}
    wd._save_sector_cache({"US:B": "IT", "US:A": None})
    assert wd._load_sector_cache() == {"US:A": None, "US:B": "IT"}
    assert list(json.loads(wd.SECTOR_CACHE.read_text())) == ["US:A", "US:B"]  # sorted keys


def _row(sym, mkt="US", missing=False):
    return {"symbol": sym, "market": mkt, "missing": missing}


def test_assign_sectors_cache_yfinance_cap_and_unclassified(sec, monkeypatch):
    wd._save_sector_cache({"US:CACHED": "Banks", "US:NULLCACHED": None})
    fetched = []

    class T:
        def __init__(self, t):
            fetched.append(t)
            self.info = {"sector": {"A": "IT", "B": None}.get(t)}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T))
    rows = [
        _row("CACHED"),
        _row("NULLCACHED"),
        _row("A"),
        _row("B"),
        _row("C"),
        _row("GONE", missing=True),
    ]
    wd.assign_sectors(rows, fetch_cap=2)
    got = {r["symbol"]: r["sector"] for r in rows}
    assert got == {
        "CACHED": "Banks",
        "NULLCACHED": "Unclassified",
        "A": "IT",
        "B": "Unclassified",
        "C": "Unclassified",
        "GONE": "Unclassified",
    }
    assert fetched == ["A", "B"]  # cap of 2 lookups; C and the missing row are never fetched
    cache = wd._load_sector_cache()
    assert cache["US:A"] == "IT" and cache["US:B"] is None and "US:GONE" not in cache


def test_assign_sectors_kr_tries_both_boards_and_in_suffix(sec, monkeypatch):
    fetched = []

    class T:
        def __init__(self, t):
            fetched.append(t)
            self.info = {"sector": "Tech" if t.endswith(".KQ") else None}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T))
    rows = [_row("5360", "KR"), _row("TCS", "IN")]
    wd.assign_sectors(rows)
    assert fetched[:2] == ["005360.KS", "005360.KQ"] and rows[0]["sector"] == "Tech"
    assert "TCS.NS" in fetched


def test_assign_sectors_jp_uses_workbook_mapping(sec, monkeypatch):
    monkeypatch.setattr(wd, "_jp_sectors_from_workbook", lambda: {"7203": "Transport Equipment"})
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(Ticker=lambda t: pytest.fail("no fetch"))
    )
    rows = [_row("7203", "JP")]
    wd.assign_sectors(rows)
    assert (
        rows[0]["sector"] == "Transport Equipment"
        and wd._load_sector_cache()["JP:7203"] == "Transport Equipment"
    )


def test_jp_sectors_from_workbook_maps_codes(monkeypatch, tmp_path):
    d = tmp_path / "japan_scan"
    d.mkdir()
    with pd.ExcelWriter(d / "japan_market_scan_20260101.xlsx") as w:
        pd.DataFrame({"Code": [7203.0, 6758.0, None], "Sector": [3700, 3650, 3700]}).to_excel(
            w, sheet_name="All_Stocks", index=False
        )
    monkeypatch.setattr(wd, "__file__", str(tmp_path / "watchlist_digest.py"))
    assert wd._jp_sectors_from_workbook() == {
        "7203": "Transport Equipment",
        "6758": "Electric Appliances",
    }
    monkeypatch.setattr(wd, "__file__", str(tmp_path / "nowhere" / "watchlist_digest.py"))
    assert wd._jp_sectors_from_workbook() == {}


# ---------------------------------------------------------------- render helpers
def test_median_ignores_none():
    assert (
        wd._median([1, None, 3, 2]) == 2 and wd._median([None]) is None and wd._median([]) is None
    )
    assert wd._median([1, 2]) == 1.5


def test_dist_bar_buckets_and_empty():
    assert wd._dist_bar([]) == "" and wd._dist_bar([None]) == ""
    html = wd._dist_bar([-6, -3, -1, 0, 1, 3, 6, 6], width=100)
    assert html.count("<td") == 7  # one cell per non-empty bucket
    assert (
        "#8b201f" in html and "#0c6b58" in html and "width:25px" in html
    )  # 2 of 8 in the top bucket
    only = wd._dist_bar([0.0, 0.5, -0.5])
    assert only.count("<td") == 1 and "#d9e2e8" in only


@pytest.mark.parametrize(
    "d,colour",
    [
        (-5.0, "#8b201f"),
        (-5.01, "#8b201f"),
        (-2.0, "#ca3433"),
        (-0.75, "#eebbba"),
        (0.75, "#d9e2e8"),
        (2.0, "#a3d9cd"),
        (5.0, "#16a085"),
        (5.01, "#0c6b58"),
    ],
)
def test_dist_bar_bucket_boundaries_are_upper_inclusive(d, colour):
    assert colour in wd._dist_bar([d])


def test_pctfmt():
    assert wd._pctfmt(None) == "—"
    assert "#16a085" in wd._pctfmt(1.234) and "+1.23%" in wd._pctfmt(1.234)
    assert "#ca3433" in wd._pctfmt(-0.5) and "#666" in wd._pctfmt(0.0)
    assert "font-weight:600" in wd._pctfmt(1.0, bold=True) and "font-weight" not in wd._pctfmt(1.0)


# ---------------------------------------------------------------- text / classify / fmt
def test_text_and_classify_and_fmt():
    for v in (None, float("nan"), "nan", " None ", "", "  "):
        assert wd._text(v) == ""
    assert wd._text(" hello ") == "hello" and wd._text(5) == "5"
    band = wd.FLAT_BAND_PCT
    assert wd.classify(None) == "⚪" and wd.classify(band) == "⚪" and wd.classify(-band) == "⚪"
    assert wd.classify(band + 0.01) == "🟢" and wd.classify(-band - 0.01) == "🔴"
    assert wd._fmt(None) == "—" and wd._fmt(1.5) == "+1.50%" and wd._fmt(-0.004) == "-0.00%"


# ---------------------------------------------------------------- why_listed / pick_category
@pytest.mark.parametrize(
    "status,note,expected",
    [
        ("held", "", "portfolio holding"),
        ("held", "core", "portfolio holding — core"),
        ("sold", "x", "exited position — x"),
        ("watch", "", "manually watched"),
        ("justified", "", "justified pick (evidence-backed screen)"),
        ("justified", "n", "justified pick — n"),
        ("signal", "recurring x3 +5%", "recurring mover x3 +5%"),
        (
            "signal",
            "value_rerating 2026-07-01 | extra",
            "screener signal: cheap vs sector + expanding PE (backtested) on 2026-07-01 extra",
        ),
        ("signal", "weird_filter 2026-07-01", "screener signal: weird_filter on 2026-07-01"),
        ("signal", "free text", "screener signal (source not recorded) — free text"),
        ("signal", "", "screener signal (source not recorded)"),
    ],
)
def test_why_listed(status, note, expected):
    assert wd.why_listed(status, note) == expected


@pytest.mark.parametrize(
    "status,note,cat",
    [
        ("justified", "value_rerating", "justified"),
        ("watch", "breakout", "watch"),
        ("signal", "value_rerating 2026", "vrr"),
        ("signal", "technical 2026", "breakout"),
        ("signal", "x breakout", "breakout"),
        ("signal", "golden_cross", "dma"),
        ("signal", "20 DMA cross", "dma"),
        ("signal", "piotroski+roce 2026", "proce"),
        ("signal", "piotroski+debt", "pdebt"),
        ("signal", "triple confluence", "triple"),
        ("signal", "recurring x2", "recurring"),
        ("signal", "mom @ 12%", "momentum"),
        ("signal", "6m mom", "momentum"),
        ("signal", "RSI 20", "rsi"),
        ("signal", "", "other"),
        ("signal", None, "other"),
    ],
)
def test_pick_category(status, note, cat):
    assert wd.pick_category({"status": status, "note": note}) == cat
    assert cat in wd.CATEGORY_TITLE


# ---------------------------------------------------------------- zones
def test_close_series_variants():
    assert wd._close_series(None) is None and wd._close_series(pd.DataFrame({"x": [1]})) is None
    assert wd._close_series(pd.DataFrame({"Close": ["a", None]})) is None
    df = pd.DataFrame(
        {"Date": ["2024-01-03", "2024-01-01", "bad", "2024-01-02"], "Close": [3.0, 1.0, 9.0, 2.0]}
    )
    s = wd._close_series(df)
    assert s.tolist() == [1.0, 2.0, 3.0] and isinstance(s.index, pd.DatetimeIndex)
    assert wd._close_series(
        pd.DataFrame({"price_date": ["2024-01-02", "2024-01-01"], "Close": [2.0, 1.0]})
    ).tolist() == [1.0, 2.0]


def test_zone_series_labels_and_minimum_bars():
    assert wd.zone_series(_ohlc(wd.MIN_BARS_FOR_ZONE - 1)) is None
    up = wd.zone_series(
        pd.DataFrame(
            {"Close": np.linspace(100, 200, 100)}, index=pd.bdate_range("2025-01-01", periods=100)
        )
    )
    assert up.iloc[-1] == "BUY" and set(up) <= {"BUY", "HOLD", "SELL"}
    down = wd.zone_series(
        pd.DataFrame(
            {"Close": np.linspace(200, 100, 100)}, index=pd.bdate_range("2025-01-01", periods=100)
        )
    )
    assert down.iloc[-1] == "SELL"
    assert wd.zone_series(None) is None


def test_zone_series_hold_between_averages():
    c = np.r_[
        np.linspace(100, 200, 80), np.linspace(200, 190, 20)
    ]  # off the highs but still above EMA50
    z = wd.zone_series(pd.DataFrame({"Close": c}, index=pd.bdate_range("2025-01-01", periods=100)))
    assert z.iloc[-1] in ("HOLD", "BUY")  # never SELL while above the 50-bar EMA


def test_sell_streak_counts_trailing_sells_only():
    s = pd.Series(["BUY", "SELL", "SELL", "HOLD", "SELL", "SELL", "SELL"])
    assert wd.sell_streak(s) == 3
    assert wd.sell_streak(pd.Series(["SELL", "HOLD"])) == 0
    assert wd.sell_streak(pd.Series(["SELL"] * 4)) == 4
    assert wd.sell_streak(None) == 0 and wd.sell_streak(pd.Series([], dtype=object)) == 0


# ---------------------------------------------------------------- guard_droppable
def test_guard_droppable_protects_statuses():
    wl = pd.DataFrame({"status": ["held", "watch", None, "SIGNAL", "sold"]})
    drop, prot = wd.guard_droppable(wl, [0, 1, 2, 3, 4])
    assert set(prot) == {
        i
        for i, s in enumerate(["held", "watch", "held", "signal", "sold"])
        if s in wd.PROTECTED_STATUSES
    }
    assert "held" in wd.PROTECTED_STATUSES  # blank status is treated as held -> protected
    assert 2 in prot and 3 in drop


# ---------------------------------------------------------------- assign_recommendations
@pytest.fixture
def rules(monkeypatch):
    def setup(rules=None, regime=None):
        monkeypatch.setattr(wd, "_ZONE_RULES", rules or {})
        monkeypatch.setattr(wd, "_ZONE_REGIME", regime or {})

    return setup


def _rrow(sym, mkt, zone="HOLD", d5=None, d21=None, d126=None, missing=False):
    return {
        "symbol": sym,
        "market": mkt,
        "zone": zone,
        "d5": d5,
        "d21": d21,
        "d126": d126,
        "missing": missing,
    }


def test_recommendations_default_to_zone_with_trend_rule(rules):
    rules()
    rows = [
        _rrow("A", "US", "BUY"),
        _rrow("B", "US", "SELL", missing=True),
        {"symbol": "C", "market": "US", "missing": False},
    ]
    wd.assign_recommendations(rows)
    assert [(r["rec"], r["rec_rule"]) for r in rows] == [
        ("BUY", "trend"),
        ("SELL", "trend"),
        ("?", "trend"),
    ]


def test_recommendations_revert_ranks_within_market_bottom_third_buys(rules):
    rules(rules={"US": {"rule": "revert"}})
    rows = [_rrow(f"S{i}", "US", d5=float(i)) for i in range(6)] + [
        _rrow("IN1", "IN", "BUY", d5=1.0)
    ]
    wd.assign_recommendations(rows)
    recs = [r["rec"] for r in rows[:6]]
    assert recs == [
        "BUY",
        "BUY",
        "HOLD",
        "SELL",
        "SELL",
        "SELL",
    ]  # pct ranks 1/6..6/6: <=1/3 buys, >=2/3 sells
    assert rows[6]["rec"] == "BUY" and rows[6]["rec_rule"] == "trend"  # other market unaffected


def test_recommendations_momentum_rules_buy_the_strong(rules):
    rules(rules={"US": {"rule": "mom_st"}})
    rows = [_rrow(f"S{i}", "US", d21=float(i)) for i in range(6)]
    wd.assign_recommendations(rows)
    assert [r["rec"] for r in rows] == ["SELL", "SELL", "HOLD", "BUY", "BUY", "BUY"]
    rules(rules={"US": {"rule": "mom126"}})
    rows = [_rrow(f"S{i}", "US", d126=float(i)) for i in range(6)]
    wd.assign_recommendations(rows)
    assert rows[5]["rec"] == "BUY" and rows[0]["rec"] == "SELL"


def test_recommendations_need_six_ranked_names_and_skip_missing_values(rules):
    rules(rules={"US": {"rule": "revert"}})
    rows = [_rrow(f"S{i}", "US", "BUY", d5=float(i)) for i in range(5)]
    wd.assign_recommendations(rows)
    assert all(r["rec"] == "BUY" for r in rows)  # 5 < 6 -> keeps the zone label
    rows = [_rrow(f"S{i}", "US", "BUY", d5=float(i)) for i in range(5)] + [
        _rrow("N", "US", "HOLD", d5=None)
    ]
    wd.assign_recommendations(rows)
    assert rows[-1]["rec"] == "HOLD"  # no d5 -> never ranked


def test_recommendations_regime_map_overrides_flat_rule(rules):
    rules(
        rules={"US": {"rule": "trend"}},
        regime={"US": {"active_rule": "revert", "current_regime": "BEAR"}},
    )
    rows = [_rrow(f"S{i}", "US", d5=float(i)) for i in range(6)]
    wd.assign_recommendations(rows)
    assert rows[0]["rec"] == "BUY" and all(
        r["rec_regime"] == "BEAR" and r["rec_rule"] == "revert" for r in rows
    )


def test_annotate_learned_is_shadow_only(monkeypatch):
    monkeypatch.setattr(wd, "_LEARNED_RECS", {("US", "AAA"): "BUY", ("IN", "BBB"): "SELL"})
    rows = [
        {"market": "US", "symbol": "aaa", "rec": "BUY"},
        {"market": "IN", "symbol": "BBB", "rec": "BUY"},
        {"market": "JP", "symbol": "CCC", "rec": "HOLD"},
    ]
    wd.annotate_learned(rows)
    assert rows[0]["rec_learned"] == "BUY" and rows[0]["rec_agree"] is True
    assert (
        rows[1]["rec_learned"] == "SELL"
        and rows[1]["rec_agree"] is False
        and rows[1]["rec"] == "BUY"
    )
    assert "rec_learned" not in rows[2]
    monkeypatch.setattr(wd, "_LEARNED_RECS", {})
    rows2 = [{"market": "US", "symbol": "A"}]
    wd.annotate_learned(rows2)
    assert rows2 == [{"market": "US", "symbol": "A"}]


# ---------------------------------------------------------------- maintain
@pytest.fixture
def mt(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "PURGED_ARCHIVE", tmp_path / "archive.csv")
    frames = {}
    monkeypatch.setattr(wd, "_load_ohlc", lambda s, m: frames.get(s))
    return frames


def _sell_frame(sessions_in_sell):
    c = np.r_[np.linspace(100, 200, 100), np.linspace(200, 100, sessions_in_sell)]
    return pd.DataFrame({"Close": c}, index=pd.bdate_range("2025-01-01", periods=len(c)))


def test_maintain_backfills_entry_columns_dates_and_prices(mt):
    df = pd.DataFrame(
        {"Close": np.linspace(10, 20, 80)}, index=pd.bdate_range("2025-01-01", periods=80)
    )
    mt["AAA"] = df
    wl = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "market": ["US"],
            "status": ["watch"],
            "note": ["found 2025-02-03 by x"],
        }
    )
    out, evicted, purged, changed = wd.maintain(wl)
    assert changed and out.at[0, "entry_date"] == "2025-02-03"
    assert out.at[0, "entry_price"] == round(
        float(df["Close"][df.index >= "2025-02-03"].iloc[0]), 4
    )
    assert evicted == [] and purged == []


def test_maintain_evicts_after_limit_and_purges_after_purge_limit(mt, tmp_path):
    # falling-bar counts chosen so the trailing SELL streaks are 6 (> 5), 23 (> 15) and 4
    mt.update({"EV": _sell_frame(8), "PU": _sell_frame(30), "OK": _sell_frame(6)})
    assert [wd.sell_streak(wd.zone_series(mt[k])) for k in ("EV", "PU", "OK")] == [6, 23, 4]
    wl = pd.DataFrame(
        {
            "symbol": ["EV", "PU", "OK"],
            "market": ["US"] * 3,
            "status": ["signal", "signal", "signal"],
            "note": ["", "", ""],
            "entry_date": ["2025-01-02"] * 3,
            "entry_price": [1.0] * 3,
        }
    )
    out, evicted, purged, changed = wd.maintain(wl)
    assert changed and evicted[0].startswith("EV (US, ") and purged == ["PU (US)"]
    assert out["symbol"].tolist() == ["EV", "OK"]
    assert (
        out.set_index("symbol").at["EV", "status"] == "evicted"
        and "evicted" in out.set_index("symbol").at["EV", "note"]
    )
    arch = pd.read_csv(tmp_path / "archive.csv")
    assert arch["symbol"].tolist() == ["PU"] and "purged_date" in arch.columns


def test_maintain_purges_old_evicted_rows_and_never_touches_held_or_sold(mt, tmp_path):
    mt.update({"OLD": _sell_frame(30), "H": _sell_frame(30), "S": _sell_frame(30)})
    wl = pd.DataFrame(
        {
            "symbol": ["OLD", "H", "S"],
            "market": ["US"] * 3,
            "status": ["evicted", "held", "sold"],
            "note": [""] * 3,
            "entry_date": [""] * 3,
            "entry_price": [None] * 3,
        }
    )
    out, evicted, purged, _ = wd.maintain(wl)
    assert purged == ["OLD (US)"] and out["symbol"].tolist() == ["H", "S"] and evicted == []


def test_maintain_missing_data_and_unchanged(mt):
    wl = pd.DataFrame(
        {
            "symbol": ["NODATA"],
            "market": ["US"],
            "status": ["watch"],
            "note": [""],
            "entry_date": ["2025-01-01"],
            "entry_price": [5.0],
        }
    )
    out, evicted, purged, changed = wd.maintain(wl)
    assert (evicted, purged, changed) == ([], [], False) and len(out) == 1


# ---------------------------------------------------------------- build_rows
@pytest.fixture
def rows_env(monkeypatch):
    monkeypatch.setattr(wd, "_fx_map", lambda: {"USD": 1.0})
    monkeypatch.setattr(wd, "_turnover_usd", lambda df, s, m, fx: 5e6 if s != "ILLIQ" else 100.0)
    monkeypatch.setattr(wd, "_liq_label", lambda tv, m: ("—", True) if tv < 1000 else ("T3", False))
    frames = {}
    monkeypatch.setattr(wd, "_load_ohlc", lambda s, m: frames.get(s))
    return frames


def test_build_rows_missing_priced_sorting_and_frames_out(rows_env):
    up = _ohlc(80, drift=0.01, vol=0.001, dated=True)
    rows_env.update(
        {"UP": up, "DOWN": _ohlc(80, drift=-0.01, vol=0.001, seed=1), "ILLIQ": _ohlc(80, seed=2)}
    )
    wl = pd.DataFrame(
        {
            "symbol": ["down", "gone", "up", "illiq"],
            "market": ["us", "us", None, "US"],
            "status": ["held", "watch", "WATCH", "signal"],
            "note": ["n1", float("nan"), "n3", None],
            "entry_date": ["2025-03-03", None, None, None],
            "entry_price": [50.0, None, None, None],
        }
    )
    frames = {}
    rows = wd.build_rows(wl, frames_out=frames)
    assert rows[0]["symbol"] == "UP" and rows[-1]["symbol"] in ("ILLIQ", "GONE")
    by = {r["symbol"]: r for r in rows}
    assert (
        by["UP"]["mark"] == "🟢" and by["DOWN"]["mark"] == "🔴" and by["UP"]["market"] == "US"
    )  # blank market -> US
    assert by["GONE"]["missing"] is True and by["GONE"]["mark"] == "❔" and by["GONE"]["note"] == ""
    assert by["ILLIQ"]["below_floor"] is True
    assert by["UP"]["last"] == str(up["Date"].max())[:10] and by["UP"]["status"] == "watch"
    assert by["DOWN"]["ret_entry"] == pytest.approx((by["DOWN"]["close"] / 50.0 - 1) * 100)
    assert by["DOWN"]["days_in"] > 0 and by["UP"]["days_in"] is None
    assert by["UP"]["d5"] is not None and by["UP"]["d126"] is None  # only 80 bars
    assert set(frames) == {("US", "UP"), ("US", "DOWN"), ("US", "ILLIQ")}


def test_build_rows_entry_return_needs_positive_entry_price(rows_env):
    rows_env["A"] = _ohlc(80)
    wl = pd.DataFrame(
        {"symbol": ["A"], "market": ["US"], "entry_price": [0.0], "entry_date": ["not a date"]}
    )
    r = wd.build_rows(wl)[0]
    assert r["ret_entry"] is None and r["days_in"] is None


# ---------------------------------------------------------------- data_gaps
def test_data_gaps_all_sections():
    rows = [{"symbol": f"M{i}", "market": "US", "missing": True} for i in range(8)] + [
        {
            "symbol": "U1",
            "market": "KR",
            "missing": False,
            "sector": "Unclassified",
            "liq": "T1",
            "last": "2026-07-30",
        },
        {
            "symbol": "L1",
            "market": "KR",
            "missing": False,
            "sector": "IT",
            "liq": "?",
            "last": "2026-07-30",
        },
        {
            "symbol": "OLD",
            "market": "KR",
            "missing": False,
            "sector": "IT",
            "liq": "T2",
            "last": "2026-07-20",
        },
        {
            "symbol": "S1",
            "market": "KR",
            "missing": False,
            "sector": "IT",
            "liq": "T2",
            "last": "2026-07-30",
            "status": "signal",
        },
        {
            "symbol": "S2",
            "market": "KR",
            "missing": False,
            "sector": "IT",
            "liq": "T2",
            "last": "2026-07-30",
            "status": "signal",
            "entry_date": "2026-01-01",
        },
        {
            "symbol": "MID",
            "market": "KR",
            "missing": False,
            "sector": "IT",
            "liq": "T2",
            "last": "2026-07-29",
        },
    ]
    gaps = wd.data_gaps(rows, "2026-07-30")
    text = " | ".join(gaps)
    assert (
        "8 names have no price data" in text
        and "US:8" in text
        and "M0, M1, M2, M3, M4, M5" in text
        and "M6" not in text
    )
    assert "1 priced names have no sector label" in text and "KR:1" in text
    assert "1 names have unmeasurable liquidity" in text
    assert "1 rows are &gt;1 session behind" in text and "OLD(07-20)" in text
    assert "1 signal rows have no entry date" in text  # only S1; S2 has one


def test_data_gaps_clean_and_previous_session_is_not_stale():
    rows = [
        {
            "symbol": "A",
            "market": "US",
            "missing": False,
            "sector": "IT",
            "liq": "T1",
            "last": "2026-07-30",
        },
        {
            "symbol": "B",
            "market": "US",
            "missing": False,
            "sector": "IT",
            "liq": "T1",
            "last": "2026-07-30",
        },
        {
            "symbol": "C",
            "market": "US",
            "missing": False,
            "sector": "IT",
            "liq": "T1",
            "last": "2026-07-29",
        },
    ]
    assert wd.data_gaps(rows, "2026-07-30") == []  # one session behind the modal date is tolerated
    rows[2]["last"] = (
        "2026-07-28"  # still: previous *distinct* date in the market is 07-28 itself -> not behind
    )
    assert wd.data_gaps(rows, "x") == []


# ---------------------------------------------------------------- render / dashboard (smoke + content)
@pytest.fixture
def rendered_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "_fx_map", lambda: {"USD": 1.0})
    monkeypatch.setattr(wd, "_turnover_usd", lambda df, s, m, fx: 100.0 if s == "ILLIQ" else 5e6)
    monkeypatch.setattr(wd, "_liq_label", lambda tv, m: ("—", True) if tv < 1000 else ("T3", False))
    monkeypatch.setattr(wd, "SECTOR_CACHE", tmp_path / "sector_map.json")
    monkeypatch.setattr(wd, "_ZONE_RULES", {})
    monkeypatch.setattr(wd, "_ZONE_REGIME", {})
    data = {
        s: _ohlc(80, drift=d, vol=0.004, seed=i, dated=(i % 2 == 0))
        for i, (s, d) in enumerate(
            [
                ("UP1", 0.01),
                ("UP2", 0.008),
                ("DN1", -0.01),
                ("DN2", -0.008),
                ("FLAT", 0.0),
                ("ILLIQ", 0.001),
                ("JPX", 0.002),
            ]
        )
    }
    monkeypatch.setattr(wd, "_load_ohlc", lambda s, m: data.get(s))
    wd._save_sector_cache(
        {
            f"US:{s}": sec
            for s, sec in [
                ("UP1", "Technology"),
                ("UP2", "Technology"),
                ("DN1", "Energy"),
                ("DN2", "Energy"),
                ("FLAT", "Utilities"),
                ("ILLIQ", "Utilities"),
            ]
        }
    )
    wl = pd.DataFrame(
        {
            "symbol": ["UP1", "UP2", "DN1", "DN2", "FLAT", "ILLIQ", "GONE", "HELD1", "EV1", "JPX"],
            "market": ["US"] * 9 + ["JP"],
            "status": [
                "signal",
                "justified",
                "watch",
                "signal",
                "signal",
                "signal",
                "signal",
                "held",
                "evicted",
                "signal",
            ],
            "note": [
                "value_rerating 2026-07-01",
                "triple x",
                "manual",
                "recurring x3 +4%",
                "technical 2026-07-02 A",
                "",
                "",
                "",
                "gone bad",
                "6m mom",
            ],
            "entry_date": ["2026-06-01"] + [""] * 9,
            "entry_price": [90.0] + [None] * 9,
        }
    )
    rows = wd.build_rows(wl)
    wd.assign_sectors(rows, fetch_cap=0)
    wd.assign_recommendations(rows)
    return rows


def test_render_dashboard_empty_and_populated(rendered_rows):
    assert wd.render_dashboard([]) == ""
    assert wd.render_dashboard([{"missing": True, "d1": None}]) == ""
    html = wd.render_dashboard(rendered_rows)
    assert "Market pulse" in html and "US" in html and "<table" in html
    full = wd.render_dashboard(rendered_rows, full=True)
    assert len(full) >= len(html) and "Market pulse" in full


def test_render_short_vs_full_and_row_filtering(rendered_rows):
    short = wd.render(rendered_rows, "2026-07-30")
    full = wd.render(rendered_rows, "2026-07-30", full=True)
    assert isinstance(short, str) and "2026-07-30" in short and len(full) >= len(short)
    for html in (short, full):
        assert "HELD1" not in html  # held/sold rows are dropped: research view only
    assert "UP1" in full and "DN1" in full


def test_render_includes_purged_images_and_bundles(rendered_rows):
    imgs = {"treemap": "cid:treemap", "rrg": "cid:rrg", "breadth": "cid:breadth"}
    html = wd.render(
        rendered_rows,
        "2026-07-30",
        purged=["OLD (US)", "OLD2 (IN)"],
        full=True,
        images=imgs,
        bundles=[],
    )
    assert "OLD (US)" in html and 'src="cid:treemap"' in html
    assert wd.render(rendered_rows, "2026-07-30", images=None).count("<img") == 0


def test_render_handles_empty_and_all_missing_rows():
    assert isinstance(wd.render([], "2026-07-30"), str)
    miss = [
        {
            "symbol": "X",
            "market": "US",
            "mark": "❔",
            "d1": None,
            "d5": None,
            "d21": None,
            "d126": None,
            "close": None,
            "last": None,
            "note": "",
            "status": "watch",
            "missing": True,
            "liq": "?",
            "below_floor": False,
            "why": "manually watched",
            "zone": "?",
            "streak": 0,
            "entry_date": "",
            "ret_entry": None,
            "days_in": None,
            "sector": "Unclassified",
            "rec": "?",
            "rec_rule": "trend",
        }
    ]
    html = wd.render(miss, "2026-07-30")
    assert "X" in html


def test_turnover_needs_ten_observations(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "liquidity",
        types.SimpleNamespace(currency_for=lambda s: "USD", CCY={"US": "USD"}, SUBUNIT={}),
    )
    ten = pd.DataFrame({"Close": [2.0] * 10, "Volume": [5.0] * 10})
    assert wd._turnover_usd(ten, "A", "US", {"USD": 1.0}) == 10.0
    assert wd._turnover_usd(ten.head(9), "A", "US", {"USD": 1.0}) is None


def test_zone_series_flat_prices_are_hold_not_sell():
    z = wd.zone_series(
        pd.DataFrame({"Close": np.full(60, 100.0)}, index=pd.bdate_range("2025-01-01", periods=60))
    )
    assert set(z) == {"HOLD"}  # close == EMA50 exactly: not below it


def _streak_n(target):
    for n in range(4, 60):
        c = np.r_[np.linspace(100, 200, 100), np.linspace(200, 100, n)]
        s = wd.sell_streak(
            wd.zone_series(
                pd.DataFrame({"Close": c}, index=pd.bdate_range("2025-01-01", periods=len(c)))
            )
        )
        if s == target:
            return n
    raise AssertionError(target)


@pytest.mark.parametrize(
    "streak,evicted,purged",
    [(5, False, False), (6, True, False), (15, True, False), (16, False, True)],
)
def test_maintain_streak_boundaries(mt, streak, evicted, purged):
    assert (wd.SELL_STREAK_LIMIT, wd.PURGE_SELL_SESSIONS) == (5, 15)
    n = _streak_n(streak)
    mt["X"] = _sell_frame(n)
    wl = pd.DataFrame(
        {
            "symbol": ["X"],
            "market": ["US"],
            "status": ["signal"],
            "note": [""],
            "entry_date": ["2025-01-02"],
            "entry_price": [1.0],
        }
    )
    out, ev, pu, _ = wd.maintain(wl)
    assert bool(ev) is evicted and bool(pu) is purged


def test_protected_statuses_include_value_hold():
    assert {"held", "value-hold", "sold"} == wd.PROTECTED_STATUSES
    wl = pd.DataFrame({"status": ["value-hold", "signal"]})
    assert wd.guard_droppable(wl, [0, 1]) == ([1], [0])


def test_why_listed_recurring_prefix_must_be_the_full_word():
    assert (
        wd.why_listed("signal", "recurrent thing")
        == "screener signal (source not recorded) — recurrent thing"
    )
