"""Offline tests for factorial_screener_test (signal generators, fundamental screens, forward returns)."""

import numpy as np
import pandas as pd
import pytest

import factorial_screener_test as fst


@pytest.fixture(autouse=True)
def _us_benchmark(monkeypatch):
    # the India/CN/JP variants rebind this module-level global at import; pin the US value for these tests
    monkeypatch.setattr(fst, "BENCHMARK_SYMBOL", "SPY")


def _px(sym, closes, start="2020-01-01", high=None, low=None, vol=1000.0):
    n = len(closes)
    d = pd.bdate_range(start, periods=n)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Symbol": sym, "Date": d, "Close": c,
        "High": c * 1.0 if high is None else np.asarray(high, dtype=float),
        "Low": c * 1.0 if low is None else np.asarray(low, dtype=float),
        "Volume": vol,
    })


# ------------------------------------------------------------------ split flag
def test_flag_split_days_flags_known_ratios_and_columns(capsys):
    closes = [100.0, 101.0, 50.5, 51.0, 102.0, 51.0] + [51.0] * 30
    df = fst._flag_split_days(_px("AAA", closes))
    flagged = df.loc[df["likely_split"], "Close"].tolist()
    # 101 -> 50.5 is -50% (a split); 51 -> 102 is +100% (reverse split); 102 -> 51 is -50%
    assert flagged == [50.5, 102.0, 51.0]
    assert {"daily_ret", "dollar_vol_63d", "vol_63d_ann"} <= set(df.columns)
    assert "flagged 3 likely-unadjusted-split days across 1 symbols" in capsys.readouterr().out
    # dollar volume needs >=20 observations
    assert df["dollar_vol_63d"].iloc[:19].isna().all()
    assert df["dollar_vol_63d"].iloc[19] == pytest.approx(df["Close"].iloc[:20].mean() * 1000.0)


def test_flag_split_tolerance_boundary():
    # -50% +/- 3.0 : -52.9% flagged, -53.5% not
    df = fst._flag_split_days(_px("A", [100.0, 47.1, 100.0, 46.5]))
    assert df["likely_split"].tolist()[1] is True or bool(df["likely_split"].iloc[1])
    assert not bool(df["likely_split"].iloc[3])


def test_load_ohlcv_filters_short_history(monkeypatch, tmp_path, capsys):
    long_ = _px("LONG", np.linspace(10, 20, 1600), start="2015-01-01")
    short = _px("SHORT", np.linspace(10, 20, 200), start="2020-01-01")
    p = tmp_path / "ohlcv.parquet"
    pd.concat([short, long_]).to_parquet(p)
    monkeypatch.setattr(fst, "OHLCV_PATH", str(p))
    out = fst.load_ohlcv()
    assert set(out["Symbol"]) == {"LONG"}
    assert "OHLCV: 1 symbols with >=5.0y history" in capsys.readouterr().out


def test_load_fundamentals_drops_bad_dates(monkeypatch, tmp_path, capsys):
    f = pd.DataFrame({
        "ticker": ["B", "A", "A", "C"],
        "filed": ["2021-03-01", "2021-03-01", "bad", None],
        "fy_end": ["2020-12-31", "2020-12-31", "2019-12-31", "2020-12-31"],
    })
    p = tmp_path / "f.parquet"
    f.to_parquet(p)
    monkeypatch.setattr(fst, "FUND_PATH", str(p))
    out = fst.load_fundamentals()
    assert out["ticker"].tolist() == ["A", "B"]
    assert "Fundamentals: 2 tickers, 2 rows" in capsys.readouterr().out


# ------------------------------------------------------------------ technicals
def _flag(df):
    df = df.copy()
    df["likely_split"] = False
    df["daily_ret"] = df.groupby("Symbol")["Close"].pct_change()
    return df


def test_golden_cross_fires_once_and_respects_min_length_and_price():
    # long decline then a long rally: 50dma crosses above 200dma once
    closes = np.concatenate([np.linspace(100, 20, 260), np.linspace(20, 200, 300)])
    df = _px("UP", closes)
    sig = fst.golden_cross_signals(df)
    assert len(sig) == 1 and sig["symbol"].iloc[0] == "UP"
    i = int(np.where(df["Date"] == sig["signal_date"].iloc[0])[0][0])
    c = pd.Series(closes)
    assert c.rolling(50).mean()[i] > c.rolling(200).mean()[i]
    assert c.rolling(50).mean()[i - 1] <= c.rolling(200).mean()[i - 1]
    # price floor: the same shape scaled to sub-$5 prices is ignored
    cheap = _px("CHEAP", closes / 100.0)
    assert fst.golden_cross_signals(cheap).empty
    # min_price override lets it through again
    assert len(fst.golden_cross_signals(cheap, min_price=0.0)) == 1
    # fewer than 210 bars is skipped
    assert fst.golden_cross_signals(_px("SHORT", closes[:209])).empty
    assert list(fst.golden_cross_signals(df).columns) == ["symbol", "signal_date"]


def test_below_200dma_cross_down_and_length_gate():
    closes = np.concatenate([np.linspace(20, 100, 260), np.linspace(100, 30, 100)])
    df = _px("DN", closes)
    sig = fst.below_200dma_signals(df)
    assert len(sig) == 1
    i = int(np.where(df["Date"] == sig["signal_date"].iloc[0])[0][0])
    c = pd.Series(closes)
    assert closes[i] < c.rolling(200).mean()[i] and closes[i - 1] >= c.rolling(200).mean()[i - 1]
    assert fst.below_200dma_signals(_px("S", closes[:204])).empty
    assert fst.below_200dma_signals(_px("P", closes / 100.0)).empty


def test_new_high_signals_first_day_of_run_only():
    closes = np.concatenate([np.linspace(50, 100, 270), np.linspace(70, 71, 10), np.linspace(100, 130, 30)])
    df = _px("NH", closes)
    sig = fst.new_high_signals(df)
    assert not sig.empty
    # fresh-run semantics: no two consecutive trading days both signal
    idx = [int(np.where(df["Date"] == d)[0][0]) for d in sig["signal_date"]]
    assert all(b - a > 1 for a, b in zip(idx, idx[1:]))
    assert all(i >= 252 for i in idx)
    high = df["High"].to_numpy()
    close = df["Close"].to_numpy()
    for i in idx:  # every signal really is within 2% of the prior 252-bar high (excluding the current bar)
        assert close[i] >= 0.98 * high[i - 252:i].max()
    assert fst.new_high_signals(_px("S", closes[:259])).empty
    assert fst.new_high_signals(_px("P", closes / 100.0)).empty


def test_darvas_signals_breakout_after_stable_box():
    base = np.full(100, 50.0)
    closes = np.concatenate([base, [55.0, 56.0, 57.0], np.full(5, 57.0)])
    high = closes.copy()
    high[:100] = 52.0
    df = _px("DV", closes, high=high)
    sig = fst.darvas_signals(df)
    assert len(sig) == 1
    assert sig["signal_date"].iloc[0] == df["Date"].iloc[100]
    # too little history
    assert fst.darvas_signals(_px("S", closes[:70], high=high[:70])).empty
    # price floor
    assert fst.darvas_signals(_px("P", closes / 100.0, high=high / 100.0)).empty
    # box that has not been stable for min_hold bars blocks the breakout
    assert fst.darvas_signals(df, min_hold=500).empty


def test_short_term_reversal_bottom_quantile_and_universe_gate():
    n = 40
    frames = []
    for k in range(10):
        slope = 0.007 if k == 0 else 0.005 + 0.0001 * k
        c = 100 * np.cumprod(np.full(n, 1 + slope))
        if k == 0:
            c[30:] = c[29] * 0.6  # crash on the last 10 days of symbol S0 -> bottom decile
        frames.append(_px(f"S{k}", c))
    df = _flag(pd.concat(frames, ignore_index=True))
    sig = fst.short_term_reversal_signals(df, lookback=5, bottom_quantile=0.10, min_universe=10)
    assert "S0" in set(sig["symbol"])
    first = sig[sig["symbol"] == "S0"]["signal_date"].min()
    assert first == df[df["Symbol"] == "S0"]["Date"].iloc[30]
    # universe too small -> nothing
    assert fst.short_term_reversal_signals(df, lookback=5, min_universe=11).empty
    # price floor removes everyone
    assert fst.short_term_reversal_signals(df, lookback=5, min_universe=1, min_price=1e6).empty
    # a split inside the trailing window disqualifies the loser
    d2 = df.copy()
    d2.loc[(d2["Symbol"] == "S0"), "likely_split"] = True
    sig2 = fst.short_term_reversal_signals(d2, lookback=5, min_universe=9)
    assert "S0" not in set(sig2["symbol"])


def test_low_beta_signals_flags_defensive_stock_not_benchmark():
    n = 320
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, n)
    spy = 100 * np.cumprod(1 + r)
    defensive = 50 * np.cumprod(1 + 0.1 * r + rng.normal(0, 0.0005, n))
    levered = 50 * np.cumprod(1 + 2.0 * r)
    df = _flag(pd.concat([_px("SPY", spy), _px("DEF", defensive), _px("LEV", levered)], ignore_index=True))
    sig = fst.low_beta_signals(df)
    assert set(sig["symbol"]) == {"DEF"}
    # first eligible day is no earlier than lookback bars in
    assert sig["signal_date"].min() >= df[df["Symbol"] == "DEF"]["Date"].iloc[252]
    # a higher threshold admits the 2x-levered name too? beta ~2 -> still excluded
    assert "LEV" not in set(fst.low_beta_signals(df, beta_threshold=1.5)["symbol"])
    assert "LEV" in set(fst.low_beta_signals(df, beta_threshold=2.5)["symbol"])
    # short history is skipped
    assert fst.low_beta_signals(df, lookback=400).empty
    # price floor blocks everything
    assert fst.low_beta_signals(df, min_price=1e6).empty


def test_insider_buying_signals_thresholds(tmp_path):
    f = pd.to_datetime
    rows = [
        # AAA Q1: two buys, one small sell -> pass
        ("AAA", "P", 100, 10.0, f("2021-01-10")), ("AAA", "P", 100, 10.0, f("2021-02-20")),
        ("AAA", "S", 10, 10.0, f("2021-03-01")),
        # BBB: single buy -> n_trans < 2
        ("BBB", "P", 1000, 10.0, f("2021-01-10")),
        # CCC: buys 2000 vs sells 1800 -> buy_share = 0.526 pass; DDD: buys 100 vs sells 1000 -> fails
        ("CCC", "P", 100, 10.0, f("2021-04-01")), ("CCC", "S", 90, 10.0, f("2021-05-01")),
        ("DDD", "P", 10, 10.0, f("2021-04-01")), ("DDD", "S", 100, 10.0, f("2021-05-01")),
        # EEE: absurd price filtered, leaving a lone row
        ("EEE", "P", 10, 1e9, f("2021-04-01")), ("EEE", "P", 10, 10.0, f("2021-04-02")),
        # FFF: sub-cent price filtered
        ("FFF", "P", 10, 0.001, f("2021-04-01")), ("FFF", "P", 10, 0.001, f("2021-04-02")),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "FILING_DATE"])
    p = tmp_path / "ins.parquet"
    df.to_parquet(p)
    out = fst.insider_buying_signals(str(p))
    got = dict(zip(out["symbol"], out["signal_date"]))
    assert set(got) == {"AAA", "CCC"}
    assert got["AAA"] == pd.Timestamp("2021-03-01")  # last filing within the quarter
    assert got["CCC"] == pd.Timestamp("2021-05-01")
    # buy share threshold
    out2 = fst.insider_buying_signals(str(p), min_buy_share=0.9)
    assert set(out2["symbol"]) == {"AAA"}
    assert set(fst.insider_buying_signals(str(p), min_transactions=1)["symbol"]) >= {"AAA", "BBB", "CCC"}


def test_short_interest_decline_signals(tmp_path):
    df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "previousShortPositionQuantity": [10_000, 9_999, 50_000, 50_000],
        "changePercent": [-10.0, -50.0, -9.99, -80.0],
        "settlementDate": pd.to_datetime(["2021-01-15"] * 4),
    })
    p = tmp_path / "si.parquet"
    df.to_parquet(p)
    out = fst.short_interest_decline_signals(str(p))
    assert out["symbol"].tolist() == ["A", "D"]
    assert (out["signal_date"] == pd.Timestamp("2021-01-26")).all()
    out2 = fst.short_interest_decline_signals(str(p), publication_lag_days=0, min_prior_position=0)
    assert out2["symbol"].tolist() == ["A", "B", "D"]
    assert (out2["signal_date"] == pd.Timestamp("2021-01-15")).all()


# ------------------------------------------------------------------ fundamentals
def _fund():
    rows = []
    for t, scale in (("AAA", 1.0), ("BBB", 0.5)):
        for k, yr in enumerate(range(2016, 2022)):
            g = 1.3 ** k
            rows.append(dict(
                ticker=t, fy_end=pd.Timestamp(f"{yr}-12-31"), filed=pd.Timestamp(f"{yr + 1}-02-15"),
                total_assets=1000 * g * scale, equity=500 * g * scale, net_income=100 * g * scale,
                current_assets=300 * g * scale, current_liabilities=150 * g * scale,
                long_term_debt=(200 - 10 * k) * scale * 1e6, short_term_debt=(50 - 5 * k) * scale * 1e6,
                gross_profit=np.nan if k == 0 else 400 * g * scale, revenue=1000 * g * scale,
                cost_of_revenue=600 * g * scale, cfo=150 * g * scale, capex=30 * g * scale,
                shares=100.0 - k, ebit=180 * g * scale, cash=50 * scale, d_and_a=20.0,
                retained_earnings=200 * g * scale, total_liabilities=500 * g * scale,
            ))
    return pd.DataFrame(rows)


def test_compute_fundamental_screens_ratios_and_flags():
    out = fst.compute_fundamental_screens(_fund())
    a = out[out["ticker"] == "AAA"].reset_index(drop=True)
    # first filing has no prior balance -> averaged ratios NaN
    assert pd.isna(a.loc[0, "roa"])
    assert a.loc[1, "avg_total_assets"] == pytest.approx((1000 + 1300) / 2)
    assert a.loc[1, "roa"] == pytest.approx(130 / 1150)
    assert a.loc[1, "roe"] == pytest.approx(130 / ((500 + 650) / 2))
    assert a.loc[1, "current_ratio"] == pytest.approx(2.0)
    # gross margin falls back to revenue - cost when gross_profit missing
    assert a.loc[0, "gross_margin"] == pytest.approx(0.4)
    assert a.loc[1, "fcf"] == pytest.approx(150 * 1.3 - 30 * 1.3)
    assert a.loc[1, "rev_growth"] == pytest.approx(0.3)
    assert a.loc[1, "capex_growth"] == pytest.approx(0.3)
    assert a.loc[1, "capacity_expansion_pass"] == 1  # 30% > 25%
    assert a.loc[0, "capacity_expansion_pass"] == 0
    # asset growth 30% is not < 10%
    assert a.loc[1, "low_asset_growth_pass"] == 0
    # share count fell 1% (<2%): no buyback pass; EPS growth positive
    assert a.loc[1, "buyback_yield_pass"] == 0
    assert a.loc[1, "pead_positive_surprise_pass"] == 1
    assert a.loc[1, "eps_growth_pass"] == 1
    # debt reduction: total debt fell from 250e6 to 235e6 = -6% -> not >=10%
    assert a.loc[1, "debt_reduction_pass"] == 0
    assert a.loc[1, "total_debt_strict"] == pytest.approx(235e6)
    assert a.loc[1, "net_margin_pass"] == 0  # net margin is exactly 10%: boundary is strict (>0.10)
    assert a.loc[1, "operating_margin_pass"] == 1  # 18% > 15%
    assert a.loc[1, "sloan_pass"] == 1  # NI < CFO
    assert a.loc[3, "rev_cagr_3y"] == pytest.approx(1.3 - 1)
    assert a.loc[3, "rev_3y_ago"] == pytest.approx(1000)
    assert a.loc[5, "shares_10y_back"] != a.loc[5, "shares_10y_back"]  # NaN: only 6 periods
    assert a.loc[5, "roe_5y_ago"] == pytest.approx(a.loc[0, "roe"], nan_ok=True)
    assert set(a["piotroski_pass"]) <= {0, 1}
    assert a["piotroski_f"].between(0, 9).all()
    assert (a["graham_number"].dropna() >= 0).all()
    # mean windows use min_periods = ceil(0.6 n) floored at 2
    assert a["roe_3y_mean"].notna().sum() == a["roe"].notna().sum() - 1


def test_compute_fundamental_screens_debt_reduction_and_missing_tags():
    f = _fund()
    # AAA: total debt halves in 2018; BBB: a missing short-term-debt tag must NOT look like a reduction
    f.loc[(f.ticker == "AAA") & (f.fy_end.dt.year == 2017), "long_term_debt"] = 250e6
    f.loc[(f.ticker == "AAA") & (f.fy_end.dt.year == 2018), ["long_term_debt", "short_term_debt"]] = [100e6, 10e6]
    f.loc[(f.ticker == "BBB") & (f.fy_end.dt.year == 2018), "short_term_debt"] = np.nan
    out = fst.compute_fundamental_screens(f)
    a = out[out.ticker == "AAA"].set_index(out[out.ticker == "AAA"]["fy_end"].dt.year)
    b = out[out.ticker == "BBB"].set_index(out[out.ticker == "BBB"]["fy_end"].dt.year)
    assert a.loc[2018, "debt_change_pct"] < -0.10
    assert a.loc[2018, "debt_reduction_pass"] == 1
    assert a.loc[2018, "debt_reduction_and_opgrowth_pass"] == a.loc[2018, "operating_profit_growth_pass"]
    assert pd.isna(b.loc[2018, "total_debt_strict"])
    assert b.loc[2018, "debt_reduction_pass"] == 0 and b.loc[2019, "debt_reduction_pass"] == 0


def test_attach_market_cap_prices_and_screens():
    fund = fst.compute_fundamental_screens(_fund())
    ohlcv = pd.concat([
        _px("AAA", np.full(2200, 40.0), start="2016-01-01"),
        _px("BBB", np.full(2200, 10.0), start="2016-01-01"),
    ])
    out = fst.attach_market_cap(fund, ohlcv)
    assert len(out) == len(fund)
    row = out[(out.ticker == "AAA") & (out.fy_end.dt.year == 2018)].iloc[0]
    assert row["Close"] == 40.0
    assert row["mcap"] == pytest.approx(row["shares"] * 40.0)
    assert row["ev"] == pytest.approx(row["mcap"] + row["total_debt"] - row["cash"])
    assert row["pb_ratio"] == pytest.approx(row["mcap"] / row["equity"])
    assert row["ev_ebitda"] == pytest.approx(row["ev"] / (row["ebit"] + 20.0))
    assert row["pe_ttm"] == pytest.approx(row["mcap"] / row["net_income"])
    # coffee can additionally needs mcap >= 1e9
    assert (out.loc[out["mcap"] < 1e9, "coffee_can_pass"] == 0).all()
    assert out["not_distress"].isin([0, 1]).all()
    # missing price within the 14-day tolerance -> NaN mcap, and NaN altman does not count as safe
    no_px = fst.attach_market_cap(fund, _px("ZZZ", np.full(50, 10.0), start="2016-01-01"))
    assert no_px["mcap"].isna().all() and (no_px["not_distress"] == 0).all()
    # N-years-back shifted columns exist and are NaN for the earliest filings
    assert out.loc[out.groupby("ticker")["filed"].idxmin(), "pe_3y_back"].isna().all()
    assert (out["pb_pass"][out["pb_ratio"] >= 3] == 0).all()


def test_build_fundamental_signal_dates_keeps_only_passes():
    cols = ["piotroski_pass", "coffee_can_pass", "magic_formula_pass", "bull_cartel_pass",
            "roce_plus_pass", "sloan_pass", "not_distress",
            "capacity_expansion_pass", "growth_stocks_pass", "graham_10y_pass", "small_cap_growth_pass",
            "pead_positive_surprise_pass", "debt_reduction_pass",
            "net_margin_pass", "operating_margin_pass", "pb_pass", "ps_pass",
            "ev_ebitda_pass", "peg_pass", "fcf_yield_pass",
            "eps_growth_pass", "roic_pass", "fcf_margin_pass", "net_debt_ebitda_pass", "ev_sales_pass",
            "low_asset_growth_pass", "buyback_yield_pass", "pe_pass",
            "operating_profit_growth_pass", "debt_reduction_and_opgrowth_pass"]
    df = pd.DataFrame(0, index=range(3), columns=cols)
    df["ticker"] = ["A", "B", "C"]
    df["filed"] = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01"), pd.NaT]
    df.loc[0, "pe_pass"] = 1
    df.loc[2, "pe_pass"] = 1  # NaT filing date is dropped
    out = fst.build_fundamental_signal_dates(df)
    assert out["symbol"].tolist() == ["A"] and out["signal_date"].iloc[0] == pd.Timestamp("2021-01-01")
    assert out["pe_pass"].iloc[0] == 1


# ------------------------------------------------------------------ forward returns
def _panel():
    stock = _flag(_px("STK", np.arange(100.0, 100.0 + 400)))
    spy = _flag(_px("SPY", np.full(400, 100.0)))
    for d in (stock, spy):
        d["dollar_vol_63d"] = 5000.0
        d["vol_63d_ann"] = 0.25
    return pd.concat([stock, spy], ignore_index=True)


def test_benchmark_lookup_and_missing_benchmark():
    p = _panel()
    d, c, s = fst.benchmark_lookup(p)
    assert len(d) == len(c) == len(s) == 400 and (c == 100.0).all()
    with pytest.raises(ValueError, match="SPY not found"):
        fst.benchmark_lookup(p[p["Symbol"] != "SPY"])


def test_window_return_bounds_split_and_bad_entry():
    closes = np.array([100.0, 110.0, 121.0, 0.0])
    flags = np.array([False, False, False, False])
    dates = np.arange(4)
    assert fst._window_return(dates, closes, flags, 0, 2) == pytest.approx(21.0)
    assert np.isnan(fst._window_return(dates, closes, flags, 0, 4))      # exits past the end
    assert fst._window_return(dates, closes, flags, 0, 3) == pytest.approx(-100.0)
    assert np.isnan(fst._window_return(dates, closes, flags, 3, 3))      # zero entry price
    assert np.isnan(fst._window_return(dates, np.array([-1.0, 1, 2, 3]), flags, 0, 2))
    flags2 = np.array([False, False, True, False])
    assert np.isnan(fst._window_return(dates, closes, flags2, 0, 2))     # split on exit day
    assert fst._window_return(dates, closes, flags2, 0, 1) == pytest.approx(10.0)  # window before split


def test_attach_forward_returns_values_nans_and_split_exclusion(capsys):
    p = _panel()
    lookups = fst.forward_returns(p)
    bench = fst.benchmark_lookup(p)
    d0 = p[p["Symbol"] == "STK"]["Date"].iloc[10]
    sigs = pd.DataFrame({
        "symbol": ["STK", "NOPE", "STK", "STK"],
        "signal_date": [d0, d0, p["Date"].max() + pd.Timedelta(days=30), p[p["Symbol"] == "STK"]["Date"].iloc[390]],
    })
    out = fst.attach_forward_returns(sigs, lookups, bench)
    r = out.iloc[0]
    assert r["ret_T+5d"] == pytest.approx((115 / 110 - 1) * 100)
    assert r["bench_ret_T+5d"] == pytest.approx(0.0)
    assert r["xret_T+5d"] == pytest.approx(r["ret_T+5d"])
    assert r["ret_T+252d"] == pytest.approx((362 / 110 - 1) * 100)
    assert r["dollar_vol_63d"] == 5000.0 and r["volatility_63d"] == 0.25
    assert r["log_liquidity"] == pytest.approx(np.log1p(5000.0))
    # unknown symbol and signal after the end are all-NaN
    assert out.iloc[1][[c for c in out.columns if c.startswith(("ret_", "xret_"))]].isna().all()
    assert out.iloc[2][[c for c in out.columns if c.startswith("ret_")]].isna().all()
    # a signal near the end only has the short horizons
    assert pd.notna(out.iloc[3]["ret_T+5d"]) and pd.isna(out.iloc[3]["ret_T+21d"])
    assert "excluded 0 signal-horizon windows" in capsys.readouterr().out
    # excluding a window crossing a split increments the exclusion count
    p2 = p.copy()
    p2.loc[(p2["Symbol"] == "STK") & (p2["Date"] == p[p["Symbol"] == "STK"]["Date"].iloc[13]), "likely_split"] = True
    out2 = fst.attach_forward_returns(sigs.iloc[[0]].copy(), fst.forward_returns(p2), fst.benchmark_lookup(p2))
    assert pd.isna(out2.iloc[0]["ret_T+5d"]) and pd.isna(out2.iloc[0]["xret_T+5d"])
    assert "excluded 5 signal-horizon windows" in capsys.readouterr().out


def test_main_wires_everything_and_saves(monkeypatch, tmp_path, capsys):
    ohlcv = _panel()
    ohlcv = pd.concat([ohlcv, _flag(_px("BBB", np.linspace(10, 11, 400))).assign(
        dollar_vol_63d=1.0, vol_63d_ann=0.1)], ignore_index=True)
    fund = fst.compute_fundamental_screens(_fund())
    fund = fund[fund["ticker"] == "AAA"].drop(columns=[c for c in fund.columns if c not in _fund().columns])
    saved = {}
    monkeypatch.setattr(fst, "load_ohlcv", lambda: ohlcv)
    monkeypatch.setattr(fst, "load_fundamentals", lambda: fund)
    monkeypatch.setattr(fst, "insider_buying_signals", lambda: pd.DataFrame(
        {"symbol": ["STK", "SPY"], "signal_date": [pd.Timestamp("2020-02-03")] * 2}))
    monkeypatch.setattr(fst, "short_interest_decline_signals", lambda: pd.DataFrame(
        {"symbol": ["BBB"], "signal_date": [pd.Timestamp("2020-02-03")]}))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=True: saved.update(df=self.copy(), path=path))
    fst.main()
    out = saved["df"]
    assert saved["path"].endswith("factorial_screener_signals_us.parquet")
    assert "SPY" not in set(out["symbol"])  # benchmark signals dropped
    assert {"insider_buying", "short_interest_decline"} <= set(out["screener"])
    assert {"year", "ret_T+5d", "xret_T+63d", "log_liquidity"} <= set(out.columns)
    txt = capsys.readouterr().out
    assert "dropped 2 technical-screener signals fired on SPY itself" in txt, txt  # new_highs on flat SPY + the insider row


def test_piotroski_f_values_and_pass_boundary():
    out = fst.compute_fundamental_screens(_fund())
    a = out[out["ticker"] == "AAA"]
    # first filing: only cfo>0, cfo>net income and the "no dilution" default (NaN share change counts as no dilution)
    assert a["piotroski_f"].tolist() == [3, 5, 7, 5, 6, 5]
    assert a["piotroski_pass"].tolist() == [0, 0, 1, 0, 0, 0]  # 7 is the pass threshold, 6 is not


def test_coffee_can_market_cap_floor_is_inclusive():
    f = _fund()
    f["long_term_debt"], f["short_term_debt"], f["shares"] = 10.0, 5.0, 100.0
    scored = fst.compute_fundamental_screens(f)
    assert (scored[scored["ticker"] == "AAA"]["coffee_can_pass"].iloc[3:] == 1).all()
    for close, expected in ((1e7, 1), (1e7 - 1, 0)):  # market cap exactly 1e9 passes; just under fails
        ohlcv = pd.concat([_px("AAA", np.full(2200, close), start="2016-01-01"), _px("BBB", np.full(2200, close), start="2016-01-01")])
        out = fst.attach_market_cap(scored, ohlcv)
        late = out[(out["ticker"] == "AAA") & (out["fy_end"].dt.year >= 2019)]
        assert (late["coffee_can_pass"] == expected).all(), close


def test_golden_cross_length_gate_and_warmup_suppression():
    # a genuine cross right after the 200-bar warm-up is only reported once the series reaches 210 bars
    step = np.concatenate([np.full(200, 50.0), np.full(10, 500.0)])
    assert fst.golden_cross_signals(_px("X", step[:205])).empty
    assert len(fst.golden_cross_signals(_px("X", step))) == 1
    # a steadily rising series is "above" from the first bar where the 200-DMA exists; that warm-up artefact is not a cross
    assert fst.golden_cross_signals(_px("UP", np.linspace(10, 100, 300))).empty
