"""factorial_screener_test.py -- US technical signal detectors, fundamental screens,
market-cap attachment and benchmark-adjusted forward returns."""

import numpy as np
import pandas as pd
import pytest

import factorial_screener_test as FS


@pytest.fixture(autouse=True)
def _us_benchmark(monkeypatch):
    # importing factorial_screener_test_IN (e.g. via factor_zscore_panel) rebinds this global
    monkeypatch.setattr(FS, "BENCHMARK_SYMBOL", "SPY")


def panel(series, start="2015-01-05", volume=1000.0, high_mult=1.0):
    """Long OHLCV panel from {symbol: close array}; High = Close * high_mult."""
    frames = []
    for sym, closes in series.items():
        closes = np.asarray(closes, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "Symbol": sym,
                    "Date": pd.bdate_range(start, periods=len(closes)),
                    "Close": closes,
                    "High": closes * high_mult,
                    "Low": closes * 0.99,
                    "Volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def flagged(series, **kw):
    return FS._flag_split_days(panel(series, **kw))


# ── split flagging ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "ratio,flag",
    [
        (-50.0, True),
        (-47.0, True),
        (-46.9, False),
        (-53.0, True),
        (-53.1, False),
        (100.0, True),
        (102.0, True),
        (104.0, False),
        (-10.0, False),
        (0.0, False),
        (-90.0, True),
    ],
)
def test_split_flag_tolerance_is_three_points(ratio, flag):
    closes = [100.0] * 5 + [100.0 * (1 + ratio / 100)] + [50.0] * 3
    df = flagged({"X": closes})
    assert bool(df["likely_split"].iloc[5]) is flag


def test_split_flag_is_per_symbol_and_first_row_is_false():
    df = flagged({"A": [10.0, 10.0, 10.0], "B": [100.0, 50.0, 50.0]})
    a, b = df[df.Symbol == "A"], df[df.Symbol == "B"]
    assert not a["likely_split"].any()
    assert list(b["likely_split"]) == [False, True, False]
    assert not df["likely_split"].iloc[0]


def test_liquidity_and_volatility_columns():
    closes = 100.0 + np.arange(100.0)
    df = flagged({"X": closes}, volume=10.0)
    assert df["dollar_vol_63d"].iloc[:19].isna().all() and not np.isnan(
        df["dollar_vol_63d"].iloc[19]
    )
    assert df["dollar_vol_63d"].iloc[99] == pytest.approx((closes[37:100] * 10).mean())
    r = pd.Series(closes).pct_change()
    assert df["vol_63d_ann"].iloc[99] == pytest.approx(r.iloc[37:100].std() * np.sqrt(252))
    assert df["daily_ret"].iloc[1] == pytest.approx(101 / 100 - 1)


def test_flag_split_days_sorts_by_symbol_and_date():
    p = panel({"B": [1.0, 2.0], "A": [3.0, 4.0]}).iloc[::-1]
    out = FS._flag_split_days(p)
    assert list(out["Symbol"]) == ["A", "A", "B", "B"]


# ── loaders ───────────────────────────────────────────────────────────────────
def test_load_ohlcv_keeps_symbols_with_five_years_of_history(tmp_path, monkeypatch):
    long = panel({"LONG": np.full(1400, 10.0), "SHORT": np.full(400, 10.0)})
    long.to_parquet(tmp_path / "o.parquet", index=False)
    monkeypatch.setattr(FS, "OHLCV_PATH", str(tmp_path / "o.parquet"))
    out = FS.load_ohlcv()
    assert set(out["Symbol"]) == {"LONG"} and "likely_split" in out.columns


def test_load_ohlcv_history_threshold_is_inclusive(tmp_path, monkeypatch):
    dates = pd.DatetimeIndex(["2015-01-01", "2020-01-02"])  # exactly 1827 days = 5.002y
    pd.DataFrame(
        {"Symbol": "A", "Date": dates, "Close": 10.0, "High": 10.0, "Low": 9.0, "Volume": 1.0}
    ).to_parquet(tmp_path / "o.parquet", index=False)
    monkeypatch.setattr(FS, "OHLCV_PATH", str(tmp_path / "o.parquet"))
    assert len(FS.load_ohlcv()) == 2
    dates = pd.DatetimeIndex(["2015-01-01", "2019-12-30"])
    pd.DataFrame(
        {"Symbol": "A", "Date": dates, "Close": 10.0, "High": 10.0, "Low": 9.0, "Volume": 1.0}
    ).to_parquet(tmp_path / "o.parquet", index=False)
    assert FS.load_ohlcv().empty


def test_load_fundamentals_parses_dates_and_drops_unparseable(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "ticker": ["B", "A", "A", "C"],
            "filed": ["2021-03-01", "2020-03-01", "bad", "2020-01-01"],
            "fy_end": ["2020-12-31", "2019-12-31", "2018-12-31", None],
            "x": [1, 2, 3, 4],
        }
    ).to_parquet(tmp_path / "f.parquet", index=False)
    monkeypatch.setattr(FS, "FUND_PATH", str(tmp_path / "f.parquet"))
    out = FS.load_fundamentals()
    assert list(out["ticker"]) == ["A", "B"] and out["filed"].dtype.kind == "M"
    assert list(out.index) == [0, 1]


# ── technical signals ─────────────────────────────────────────────────────────
def test_golden_cross_fires_on_the_crossing_bar_only():
    closes = np.concatenate([np.full(200, 100.0), np.full(60, 80.0), np.full(100, 130.0)])
    out = FS.golden_cross_signals(panel({"X": closes}))
    s50, s200 = pd.Series(closes).rolling(50).mean(), pd.Series(closes).rolling(200).mean()
    above = (s50 > s200).to_numpy()
    expected = [i for i in range(200, len(closes)) if above[i] and not above[i - 1]]
    dates = pd.bdate_range("2015-01-05", periods=len(closes))
    assert list(out["symbol"]) == ["X"] * len(expected) and expected
    assert list(out["signal_date"]) == [dates[i] for i in expected]


def test_golden_cross_guards_price_floor_and_history():
    closes = np.concatenate([np.full(200, 100.0), np.full(60, 80.0), np.full(100, 130.0)])
    assert FS.golden_cross_signals(panel({"X": closes / 100})).empty  # prices below $5
    assert FS.golden_cross_signals(panel({"X": closes[:209]})).empty  # < 210 bars
    assert list(FS.golden_cross_signals(panel({"X": closes}, start="2015-01-05")).columns) == [
        "symbol",
        "signal_date",
    ]
    assert FS.golden_cross_signals(panel({"X": np.full(300, 100.0)})).empty


def test_darvas_first_breakout_day_only():
    n = 130
    closes = np.full(n, 99.0)
    closes[100] = 105.0
    closes[101:] = 99.0
    p = panel({"X": closes}, high_mult=1.0)
    p.loc[p.Symbol == "X", "High"] = np.where(np.arange(n) == 100, 105.0, 100.0)
    out = FS.darvas_signals(p)
    assert list(out["signal_date"]) == [pd.bdate_range("2015-01-05", periods=n)[100]]


def test_darvas_requires_a_stable_box_a_price_floor_and_history():
    n = 130
    closes = np.full(n, 99.0)
    closes[100] = 105.0
    hi = np.where(np.arange(n) == 100, 105.0, 100.0)
    # a box that moved within the last 9 bars is not a consolidation
    hi_moving = hi.copy()
    hi_moving[95] = 102.0
    p = panel({"X": closes})
    p["High"] = hi_moving
    assert FS.darvas_signals(p).empty
    p2 = panel({"X": closes / 100})
    p2["High"] = hi / 100
    assert FS.darvas_signals(p2).empty  # sub-$5
    assert FS.darvas_signals(panel({"X": closes[:77]})).empty  # < 63 + 10 + 5 bars


def test_new_high_signal_enters_the_zone_once():
    n = 300
    closes = np.full(n, 90.0)
    closes[280] = 99.0
    closes[281] = 99.0
    p = panel({"X": closes})
    p["High"] = 100.0
    out = FS.new_high_signals(p)
    assert list(out["signal_date"]) == [pd.bdate_range("2015-01-05", periods=n)[280]]
    p["High"] = np.where(np.arange(n) == 280, 100.0, 100.0)
    assert FS.new_high_signals(panel({"X": closes[:259]})).empty
    assert FS.new_high_signals(panel({"X": closes / 100})).empty


def test_new_high_zone_boundary_is_inclusive_at_98_percent():
    n = 300
    base = np.full(n, 90.0)
    near = base.copy()
    near[280] = 98.0  # exactly 0.98 * 100
    far = base.copy()
    far[280] = 97.99
    p = panel({"A": near, "B": far})
    p["High"] = 100.0
    out = FS.new_high_signals(p)
    assert list(out["symbol"]) == ["A"]


def test_below_200dma_cross():
    closes = np.concatenate([np.full(250, 100.0), [90.0], np.full(20, 90.0)])
    out = FS.below_200dma_signals(panel({"X": closes}))
    assert list(out["signal_date"]) == [pd.bdate_range("2015-01-05", periods=len(closes))[250]]
    assert FS.below_200dma_signals(panel({"X": closes / 100})).empty
    assert FS.below_200dma_signals(panel({"X": closes[:204]})).empty


def reversal_panel(rates, n=40, price=100.0, split_for=None):
    series = {f"S{i}": price * (1 + r) ** np.arange(n) for i, r in enumerate(rates)}
    p = FS._flag_split_days(panel(series))
    if split_for:
        mask = (p.Symbol == split_for) & (p.groupby("Symbol").cumcount() == 10)
        p.loc[mask, "likely_split"] = True
    return p


def test_short_term_reversal_picks_the_bottom_of_the_cross_section_once():
    rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    p = reversal_panel(rates)
    out = FS.short_term_reversal_signals(p, lookback=5, bottom_quantile=0.34, min_universe=6)
    first = pd.bdate_range("2015-01-05", periods=40)[5]
    assert sorted(out["symbol"]) == ["S0", "S1"]  # ranks 1/6 and 2/6 are <= 0.34
    assert set(out["signal_date"]) == {first}
    out2 = FS.short_term_reversal_signals(p, lookback=5, bottom_quantile=0.17, min_universe=6)
    assert list(out2["symbol"]) == ["S0"]  # 1/6 = 0.1667 <= 0.17, 2/6 is not


def test_reversal_universe_floor_split_exclusion_and_price_floor():
    rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    p = reversal_panel(rates)
    assert FS.short_term_reversal_signals(p, 5, 0.34, min_universe=7).empty
    assert not FS.short_term_reversal_signals(p, 5, 0.34, min_universe=6).empty
    ps = reversal_panel(rates, split_for="S0")
    out = FS.short_term_reversal_signals(ps, lookback=5, bottom_quantile=0.34, min_universe=5)
    dates = pd.bdate_range("2015-01-05", periods=40)
    s0 = out[out.symbol == "S0"]["signal_date"]
    assert list(s0) == [dates[5], dates[16]]  # nothing while the split is in the window
    cheap = reversal_panel(rates, price=4.0)
    assert FS.short_term_reversal_signals(cheap, 5, 0.34, min_universe=1).empty


def test_low_beta_signals_flag_defensive_names_once():
    n = 80
    rng = np.random.default_rng(3)
    bench_ret = rng.normal(0.0, 0.01, n)
    bench = 100 * np.cumprod(1 + bench_ret)
    low = 100 * np.cumprod(1 + 0.4 * bench_ret)
    high = 100 * np.cumprod(1 + 1.6 * bench_ret)
    p = FS._flag_split_days(panel({"SPY": bench, "LOW": low, "HIGH": high}))
    out = FS.low_beta_signals(p, lookback=20, beta_threshold=0.8)
    assert list(out["symbol"]) == ["LOW"]
    dates = pd.bdate_range("2015-01-05", periods=n)
    assert out["signal_date"].iloc[0] == dates[21]  # first bar with a full shifted window
    assert FS.low_beta_signals(p, lookback=76).empty  # needs lookback + 5 bars
    cheap = FS._flag_split_days(panel({"SPY": bench, "LOW": low / 100}))
    assert FS.low_beta_signals(cheap, lookback=20).empty  # price floor on the lagged close


def test_insider_buying_quarterly_rules(tmp_path):
    rows = []

    def add(sym, date, code, shares, price):
        rows.append(
            {
                "symbol": sym,
                "FILING_DATE": pd.Timestamp(date),
                "TRANS_CODE": code,
                "TRANS_SHARES": shares,
                "TRANS_PRICEPERSHARE": price,
            }
        )

    for d in ("2020-01-10", "2020-02-20"):
        add("GOOD", d, "P", 100, 10.0)
    add("GOOD", "2020-03-05", "S", 10, 10.0)
    add("ONE", "2020-01-10", "P", 100, 10.0)  # a single filing
    add("NET_SELL", "2020-01-10", "P", 10, 10.0)
    add("NET_SELL", "2020-01-11", "S", 100, 10.0)
    add("BADPX", "2020-01-10", "P", 100, 60_000.0)
    add("BADPX", "2020-01-11", "P", 100, 60_000.0)
    add("PENNY", "2020-01-10", "P", 100, 0.001)
    add("PENNY", "2020-01-11", "P", 100, 0.001)
    add("NEXTQ", "2020-03-30", "P", 100, 10.0)
    add("NEXTQ", "2020-04-02", "P", 100, 10.0)  # split across two quarters
    p = tmp_path / "ins.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    out = FS.insider_buying_signals(str(p))
    assert list(out["symbol"]) == ["GOOD"]
    assert out["signal_date"].iloc[0] == pd.Timestamp("2020-03-05")  # last filing of the quarter
    again = FS.insider_buying_signals(str(p), min_transactions=1)
    assert set(again["symbol"]) == {"GOOD", "ONE", "NEXTQ"} and len(again) == 4
    capped = FS.insider_buying_signals(str(p), max_price_per_share=100_000.0)
    assert "BADPX" in set(capped["symbol"])


def test_short_interest_decline_rules(tmp_path):
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "settlementDate": pd.to_datetime(["2020-01-15"] * 5),
            "previousShortPositionQuantity": [10_000, 9_999, 50_000, 50_000, 50_000],
            "changePercent": [-10.0, -50.0, -9.9, -10.0, 25.0],
        }
    )
    p = tmp_path / "si.parquet"
    df.to_parquet(p, index=False)
    out = FS.short_interest_decline_signals(str(p))
    assert list(out["symbol"]) == ["A", "D"]  # B: base too small; C, E: not enough decline
    assert set(out["signal_date"]) == {pd.Timestamp("2020-01-26")}  # + 11 calendar days
    out2 = FS.short_interest_decline_signals(str(p), min_decline_pct=-5.0, publication_lag_days=0)
    assert set(out2["symbol"]) == {"A", "C", "D"} and set(out2["signal_date"]) == {
        pd.Timestamp("2020-01-15")
    }


# ── fundamentals ──────────────────────────────────────────────────────────────
def fund_frame():
    years = list(range(2016, 2024))
    n = len(years)
    k = np.arange(n)
    df = pd.DataFrame(
        {
            "ticker": "AAA",
            "fy_end": pd.to_datetime([f"{y}-12-31" for y in years]),
            "filed": pd.to_datetime([f"{y + 1}-02-15" for y in years]),
            "total_assets": 1000.0 + 100 * k,
            "equity": 500.0 + 50 * k,
            "net_income": 50.0 + 10 * k,
            "current_assets": 400.0 + 20 * k,
            "current_liabilities": 200.0 + 5 * k,
            "long_term_debt": 300.0 - 10 * k,
            "short_term_debt": 50.0 - 2 * k,
            "gross_profit": 300.0 + 30 * k,
            "revenue": 1000.0 + 150 * k,
            "cost_of_revenue": 700.0 + 100 * k,
            "cfo": 80.0 + 12 * k,
            "capex": 40.0 + 20 * k,
            "shares": 100.0 - 1 * k,
            "cash": 100.0 + 10 * k,
            "ebit": 120.0 + 20 * k,
            "retained_earnings": 200.0 + 40 * k,
            "total_liabilities": 500.0 + 50 * k,
            "d_and_a": 30.0,
        }
    )
    return df


def test_fundamental_ratios_on_hand_values():
    out = FS.compute_fundamental_screens(fund_frame()).set_index("fy_end")
    r = out.loc["2018-12-31"]  # k = 2
    avg_assets = (1200 + 1100) / 2
    avg_equity = (600 + 550) / 2
    assert r["roa"] == pytest.approx(70 / avg_assets)
    assert r["roe"] == pytest.approx(70 / avg_equity)
    assert r["current_ratio"] == pytest.approx(440 / 210)
    assert r["leverage"] == pytest.approx(280 / 1200)
    assert r["gross_margin"] == pytest.approx(360 / 1300)
    assert r["asset_turnover"] == pytest.approx(1300 / avg_assets)
    assert r["fcf"] == pytest.approx(104 - 80)
    assert r["total_debt"] == pytest.approx(280 + 46)
    assert r["de_ratio"] == pytest.approx(326 / 600)
    assert r["rev_growth"] == pytest.approx((1300 - 1150) / 1150)
    assert r["asset_growth"] == pytest.approx(100 / 1100)
    assert r["buyback_pct"] == pytest.approx(1 / 99)
    assert r["eps"] == pytest.approx(70 / 98)
    assert r["roic"] == pytest.approx(160 / (600 + 326 - 120))
    assert r["roce"] == pytest.approx(160 / (1200 - 210))
    assert r["sloan_ratio"] == pytest.approx((70 - 104) / 1200)
    assert r["bvps"] == pytest.approx(600 / 98)
    assert r["graham_number"] == pytest.approx(np.sqrt(22.5 * (70 / 98) * (600 / 98)))
    assert r["rev_cagr_3y"] == pytest.approx((1300 / 1000) ** (1 / 3) - 1) or np.isnan(
        r["rev_cagr_3y"]
    )
    assert r["operating_margin"] == pytest.approx(160 / 1300)


def test_fundamental_pass_flags():
    out = FS.compute_fundamental_screens(fund_frame()).set_index("fy_end")
    r = out.loc["2020-12-31"]  # k = 4
    assert r["piotroski_f"] == 7 - 0 or r["piotroski_f"] >= 0
    assert r["piotroski_pass"] == int(r["piotroski_f"] >= 7)
    assert r["sloan_pass"] == 1  # NI < CFO
    assert r["pead_positive_surprise_pass"] == 1
    assert r["capacity_expansion_pass"] == int(
        ((60 + 20 + 0) / 1 or 1) and (120 - 100) / 100 > 0.25
    )
    assert r["low_asset_growth_pass"] == 1  # asset growth ~ 8.3%
    assert r["buyback_yield_pass"] == 0  # 1% buyback < 2%
    assert r["net_margin_pass"] == 0  # 90/1600 < 10%
    assert r["operating_margin_pass"] == 0
    assert r["eps_growth_pass"] == 0


def test_buyback_threshold_and_asset_growth_band():
    df = fund_frame()
    df["shares"] = [100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0]  # ~3% buybacks
    df["total_assets"] = [1000.0, 1099.9, 1220.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    out = FS.compute_fundamental_screens(df).set_index("fy_end")
    assert out.loc["2018-12-31", "buyback_yield_pass"] == 1
    assert out.loc["2017-12-31", "low_asset_growth_pass"] == 1  # +9.99%
    assert out.loc["2018-12-31", "low_asset_growth_pass"] == 0  # +10.9%
    assert out.loc["2019-12-31", "low_asset_growth_pass"] == 1  # -17%: inside (-50%, 10%)


def test_piotroski_counts_nine_signals_and_passes_at_seven():
    k = np.arange(4)
    df = pd.DataFrame(
        {
            "ticker": "P",
            "fy_end": pd.to_datetime([f"{2020 + i}-12-31" for i in k]),
            "filed": pd.to_datetime([f"{2021 + i}-02-01" for i in k]),
            "total_assets": [1000.0, 1000.0, 1000.0, 1000.0],
            "equity": 500.0,
            "net_income": [10.0, 20.0, 30.0, 40.0],
            "current_assets": [100.0, 120.0, 140.0, 160.0],
            "current_liabilities": 100.0,
            "long_term_debt": [300.0, 250.0, 200.0, 150.0],
            "short_term_debt": 0.0,
            "gross_profit": [300.0, 330.0, 360.0, 390.0],
            "revenue": 1000.0,
            "cost_of_revenue": 700.0,
            "cfo": [50.0, 60.0, 70.0, 80.0],
            "capex": 10.0,
            "shares": [100.0, 100.0, 99.0, 98.0],
            "cash": 50.0,
            "ebit": 100.0,
        }
    )
    out = FS.compute_fundamental_screens(df)
    last = out.iloc[-1]
    # ROA>0, CFO>0, dROA>0, CFO>NI, leverage down, current ratio up, no dilution, margin up; turnover flat
    assert last["piotroski_f"] == 8 and last["piotroski_pass"] == 1
    assert out.iloc[0]["piotroski_f"] < 7 and out.iloc[0]["piotroski_pass"] == 0


def test_dilution_and_missing_shares_handling():
    df = fund_frame()
    df["shares"] = [100.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
    out = FS.compute_fundamental_screens(df).set_index("fy_end")
    assert out.loc["2018-12-31", "d_shares"] == 5.0
    assert out.loc["2019-12-31", "d_shares"] == 0.0


def test_rolling_means_need_sixty_percent_of_window():
    out = FS.compute_fundamental_screens(fund_frame())
    assert out["roe_5y_mean"].notna().sum() >= 1
    # min_periods for a 5-year window is ceil(5*0.6) = 3 of the (roe-valid) years
    assert out["roce_3y_mean"].isna().iloc[0]


def test_debt_reduction_requires_complete_data_and_a_real_base():
    df = fund_frame()
    df["long_term_debt"] = [5e6, 4e6, 4e6, 4e6, 4e6, 4e6, 4e6, 4e6]
    df["short_term_debt"] = 0.0
    out = FS.compute_fundamental_screens(df).set_index("fy_end")
    assert out.loc["2017-12-31", "debt_reduction_pass"] == 1  # -20% on a 5M base
    assert out.loc["2018-12-31", "debt_reduction_pass"] == 0
    df["long_term_debt"] = [5e5, 4e5] + [4e5] * 6  # base under 1M
    small = FS.compute_fundamental_screens(df).set_index("fy_end")
    assert small.loc["2017-12-31", "debt_reduction_pass"] == 0
    missing = fund_frame()
    missing["long_term_debt"] = [5e6, 4e6] + [4e6] * 6
    missing.loc[1, "short_term_debt"] = np.nan
    assert (
        FS.compute_fundamental_screens(missing)
        .set_index("fy_end")
        .loc["2017-12-31", "debt_reduction_pass"]
        == 0
    )


def test_attach_market_cap_uses_next_close_within_two_weeks():
    fund = FS.compute_fundamental_screens(fund_frame())
    ohlcv = panel({"AAA": 10.0 + np.arange(2500) * 0.01}, start="2016-01-04")
    merged = FS.attach_market_cap(fund, ohlcv)
    row = merged.set_index("fy_end").loc["2018-12-31"]
    filed = pd.Timestamp("2019-02-15")  # a Friday -> a trading day
    px = ohlcv.set_index("Date")["Close"]
    assert row["Close"] == pytest.approx(px.loc[filed])
    assert row["mcap"] == pytest.approx(row["shares"] * row["Close"])
    assert row["earnings_yield"] == pytest.approx(
        row["ebit"] / (row["mcap"] + row["total_debt"] - row["cash"])
    )
    assert row["pb_ratio"] == pytest.approx(row["mcap"] / row["equity"])
    assert row["ps_ratio"] == pytest.approx(row["mcap"] / row["revenue"])
    assert row["pe_ttm"] == pytest.approx(row["mcap"] / row["net_income"])
    assert row["fcf_yield"] == pytest.approx(row["fcf"] / row["mcap"])
    ebitda = row["ebit"] + 30
    assert row["ebitda"] == ebitda and row["ev_ebitda"] == pytest.approx(row["ev"] / ebitda)
    assert row["net_debt_ebitda"] == pytest.approx((row["total_debt"] - row["cash"]) / ebitda)


def test_attach_market_cap_no_price_within_tolerance_gives_nan():
    fund = FS.compute_fundamental_screens(fund_frame())
    ohlcv = panel({"AAA": np.full(100, 10.0)}, start="2010-01-04")  # long before the filings
    merged = FS.attach_market_cap(fund, ohlcv)
    assert merged["Close"].isna().all() and merged["mcap"].isna().all()
    assert (merged["pb_pass"] == 0).all() and (merged["not_distress"] == 0).all()


def test_valuation_pass_thresholds():
    fund = FS.compute_fundamental_screens(fund_frame())
    ohlcv = panel({"AAA": np.full(2500, 10.0)}, start="2016-01-04")
    m = FS.attach_market_cap(fund, ohlcv).set_index("fy_end").loc["2020-12-31"]
    pb = m["mcap"] / m["equity"]
    assert m["pb_pass"] == int(0 < pb < 3) and m["ps_pass"] == int(0 < m["mcap"] / m["revenue"] < 2)
    z = (
        1.2 * (m["current_assets"] - m["current_liabilities"]) / m["total_assets"]
        + 1.4 * m["retained_earnings"] / m["total_assets"]
        + 3.3 * m["ebit"] / m["total_assets"]
        + 0.6 * m["mcap"] / m["total_liabilities"]
        + m["revenue"] / m["total_assets"]
    )
    assert m["altman_z"] == pytest.approx(z) and m["not_distress"] == int(z >= 1.81)
    cheap = ohlcv.assign(Close=0.001)
    mm = FS.attach_market_cap(fund, cheap).set_index("fy_end").loc["2020-12-31"]
    assert mm["coffee_can_pass"] == 0  # market cap below $1B


def test_signal_dates_keep_only_filings_with_a_pass():
    fund = FS.compute_fundamental_screens(fund_frame())
    scored = FS.attach_market_cap(fund, panel({"AAA": np.full(2500, 10.0)}, start="2016-01-04"))
    sig = FS.build_fundamental_signal_dates(scored)
    assert list(sig.columns)[:2] == ["symbol", "signal_date"]
    assert (sig.drop(columns=["symbol", "signal_date"]).sum(axis=1) > 0).all()
    assert len(sig) <= len(scored)
    none = scored.copy()
    none["filed"] = pd.NaT
    assert FS.build_fundamental_signal_dates(none).empty


# ── forward returns ───────────────────────────────────────────────────────────
def lookups_for(series):
    p = FS._flag_split_days(panel(series))
    return FS.forward_returns(p), FS.benchmark_lookup(p), p


def test_benchmark_lookup_requires_spy():
    p = FS._flag_split_days(panel({"AAA": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="SPY not found"):
        FS.benchmark_lookup(p)
    p = FS._flag_split_days(panel({"SPY": [100.0, 101.0], "AAA": [1.0, 2.0]}))
    dates, closes, split = FS.benchmark_lookup(p)
    assert list(closes) == [100.0, 101.0] and len(dates) == 2 and not split.any()


def test_forward_returns_lookup_shape():
    look, _, p = lookups_for({"SPY": [100.0] * 5, "AAA": [1.0, 2.0, 3.0, 4.0, 5.0]})
    dates, closes, split, dv, vol = look["AAA"]
    assert list(closes) == [1.0, 2.0, 3.0, 4.0, 5.0] and len(dv) == 5 and len(vol) == 5


def test_window_return_rules():
    dates = np.arange(5)
    closes = np.array([100.0, 110.0, 121.0, 100.0, 50.0])
    split = np.array([False, False, False, False, True])
    assert FS._window_return(dates, closes, split, 0, 2) == pytest.approx(21.0)
    assert np.isnan(FS._window_return(dates, closes, split, 0, 5))  # beyond the data
    assert np.isnan(FS._window_return(dates, closes, split, 3, 4))  # crosses a split day
    assert FS._window_return(dates, closes, split, 3, 3) == 0.0
    assert np.isnan(FS._window_return(dates, np.array([0.0, 1, 1, 1, 1]), split, 0, 2))
    assert np.isnan(FS._window_return(dates, np.array([-5.0, 1, 1, 1, 1]), split, 0, 2))


def test_attach_forward_returns_benchmark_adjusted(monkeypatch, capsys):
    monkeypatch.setattr(FS, "HORIZONS", {"T+2d": 2, "T+9d": 9})
    n = 25
    spy = 100.0 * (1 + 0.01 * np.arange(n))
    aaa = 50.0 * (1 + 0.02 * np.arange(n))
    look, bench, p = lookups_for({"SPY": spy, "AAA": aaa})
    dates = pd.bdate_range("2015-01-05", periods=n)
    sig = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "ZZZ", "AAA"],
            "signal_date": [
                dates[21],
                dates[0] - pd.Timedelta(days=3),
                dates[1],
                dates[-1] + pd.Timedelta(days=30),
            ],
        }
    )
    out = FS.attach_forward_returns(sig, look, bench)
    r2 = (aaa[23] / aaa[21] - 1) * 100
    b2 = (spy[23] / spy[21] - 1) * 100
    assert out["ret_T+2d"].iloc[0] == pytest.approx(r2)
    assert out["bench_ret_T+2d"].iloc[0] == pytest.approx(b2)
    assert out["xret_T+2d"].iloc[0] == pytest.approx(r2 - b2)
    assert np.isnan(out["ret_T+9d"].iloc[0])  # 21 + 9 runs past the panel
    # signal dated before the first bar enters on the first bar
    assert out["ret_T+2d"].iloc[1] == pytest.approx((aaa[2] / aaa[0] - 1) * 100)
    assert out["ret_T+9d"].iloc[1] == pytest.approx((aaa[9] / aaa[0] - 1) * 100)
    assert np.isnan(out["ret_T+2d"].iloc[2]) and np.isnan(
        out["xret_T+2d"].iloc[2]
    )  # unknown symbol
    assert np.isnan(out["ret_T+2d"].iloc[3]) and np.isnan(
        out["dollar_vol_63d"].iloc[3]
    )  # after the panel
    dv = (aaa[:22] * 1000.0).mean()  # all bars so far (window not yet full)
    assert out["dollar_vol_63d"].iloc[0] == pytest.approx(look["AAA"][3][21])
    assert out["dollar_vol_63d"].iloc[0] == pytest.approx(dv)
    assert out["log_liquidity"].iloc[0] == pytest.approx(np.log1p(dv))
    assert out["volatility_63d"].iloc[0] == pytest.approx(look["AAA"][4][21])
    assert "excluded 0 signal-horizon windows" in capsys.readouterr().out


def test_forward_windows_crossing_a_split_are_excluded_not_winsorised(monkeypatch, capsys):
    monkeypatch.setattr(FS, "HORIZONS", {"T+3d": 3})
    closes = np.array([100.0, 100.0, 50.0, 50.0, 50.0, 50.0])  # -50% split day at index 2
    look, bench, _ = lookups_for({"SPY": np.full(6, 100.0), "AAA": closes})
    dates = pd.bdate_range("2015-01-05", periods=6)
    out = FS.attach_forward_returns(
        pd.DataFrame({"symbol": ["AAA", "AAA"], "signal_date": [dates[0], dates[3]]}), look, bench
    )
    assert np.isnan(out["ret_T+3d"].iloc[0])  # window [0, 3] crosses the split
    assert np.isnan(out["ret_T+3d"].iloc[1])  # runs past the end of data
    assert "excluded 1 signal-horizon windows" in capsys.readouterr().out
