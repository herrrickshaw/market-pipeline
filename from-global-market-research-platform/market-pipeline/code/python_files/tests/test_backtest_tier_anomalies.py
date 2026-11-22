"""backtest_tier_anomalies.py -- triple-barrier eviction events, re-entry leg and the report.

The Kalman filter and Markov state labeller are replaced with deterministic fakes so the
barrier logic is exercised on hand-built price paths.
"""
import types

import numpy as np
import pandas as pd
import pytest

import backtest_tier_anomalies as BTA


# ── rsi14_series ──────────────────────────────────────────────────────────────
def wilder_rsi_reference(close):
    up = dn = None
    out = []
    prev = None
    for c in close:
        if prev is None:
            out.append(np.nan)
        else:
            d = c - prev
            u, dd = max(d, 0.0), max(-d, 0.0)
            up = u if up is None else up + (u - up) / 14
            dn = dd if dn is None else dn + (dd - dn) / 14
            out.append(np.nan if dn == 0 else 100 - 100 / (1 + up / dn))
        prev = c
    return np.array(out)


def test_rsi_matches_wilder_recursion():
    rng = np.random.default_rng(3)
    c = pd.Series(100 + rng.normal(0, 1, 120).cumsum())
    got = BTA.rsi14_series(c).to_numpy()
    ref = wilder_rsi_reference(c.to_numpy())
    assert np.allclose(got[1:], ref[1:], equal_nan=True)
    assert np.isnan(got[0])
    assert ((got[1:] >= 0) & (got[1:] <= 100)).all()


def test_rsi_extremes():
    up = BTA.rsi14_series(pd.Series(np.arange(1.0, 40.0)))
    assert up.iloc[1:].isna().all()                          # no losses: 0 denominator -> NaN
    down = BTA.rsi14_series(pd.Series(np.arange(40.0, 1.0, -1.0)))
    assert (down.iloc[1:] == 0.0).all()
    bal = BTA.rsi14_series(pd.Series([10.0, 11.0] * 40))
    assert 40 < bal.iloc[-1] < 60


# ── load_market ───────────────────────────────────────────────────────────────
def test_load_market_concatenates_year_files_and_parses_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(BTA, "WH", tmp_path)
    d = tmp_path / "US"
    d.mkdir()
    for y, sym in ((2020, "A"), (2021, "B")):
        pd.DataFrame({"Date": [f"{y}-01-04"], "Symbol": [sym], "Close": [1.0], "Volume": [5],
                      "Extra": [9]}).to_parquet(d / f"year={y}.parquet", index=False)
    (d / "ignored.txt").write_text("x")
    df = BTA.load_market("US")
    assert list(df.columns) == ["Date", "Symbol", "Close", "Volume"]
    assert list(df["Symbol"]) == ["A", "B"] and df["Date"].dtype.kind == "M"
    assert list(df.index) == [0, 1]


# ── analyze_market ────────────────────────────────────────────────────────────
class FakeKalman:
    def filter(self, logp):
        slope = -0.001 if logp.name in ("AAA", "SHORT") else 0.001
        return {"slope": pd.Series(slope, index=logp.index)}


def fake_states(lr):
    return pd.Series(0 if lr.name in ("AAA", "SHORT") else 2, index=lr.index)


def bdays(start, end):
    return pd.bdate_range(start, end)


def build_prices():
    idx = bdays("2018-01-01", "2019-09-30")
    aaa = pd.Series(100.0, index=idx)
    aaa.iloc[:200] = 100.0 + 0.5 * (np.arange(200) % 2)           # early noise so RSI exists
    aaa.loc["2019-07-01":"2019-07-09"] = 105.0
    aaa.loc["2019-07-10":"2019-07-30"] = 112.0
    aaa.loc["2019-07-31"] = 117.6
    aaa.loc["2019-08-01":"2019-08-05"] = 110.0
    aaa.loc["2019-08-06":] = 99.0
    n = len(idx)
    bbb = pd.Series(100 * 1.0005 ** np.arange(n), index=idx)
    ccc = pd.Series(50 * 1.0003 ** np.arange(n), index=idx)
    ddd = pd.Series(80 * 1.0004 ** np.arange(n), index=idx)
    eee = pd.Series(60 * 1.0002 ** np.arange(n), index=idx)
    short = pd.Series(100 * 1.0004 ** np.arange(399), index=idx[-399:])
    return {"AAA": aaa, "BBB": bbb, "CCC": ccc, "DDD": ddd, "EEE": eee, "SHORT": short}


def to_long(prices):
    rows = []
    for sym, s in prices.items():
        rows.append(pd.DataFrame({"Date": s.index, "Symbol": sym, "Close": s.values,
                                  "Volume": 1000}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def setup(monkeypatch):
    prices = build_prices()
    monkeypatch.setattr(BTA, "load_market", lambda m: to_long(prices))
    monkeypatch.setattr(BTA, "KalmanTrendModel", FakeKalman)
    monkeypatch.setattr(BTA, "label_states", fake_states)
    monkeypatch.setattr(BTA, "rsi_zone", lambda m, v: "BUY" if m == "BUYM" else "NO")
    monkeypatch.setattr(BTA, "START", "2019-06-01")
    return prices


def events_by(ev, sym="AAA"):
    return [e for e in ev if e["symbol"] == sym]


def test_triple_barrier_outcomes_in_sequence(setup):
    ev = events_by(BTA.analyze_market("US"))
    assert [e["outcome"] for e in ev] == ["ANOMALY", "validated_decline", "ageout_validated"]
    e1, e2, e3 = ev
    assert str(e1["evict_date"]) == "2019-06-28"                  # month-end snapped to last bar
    assert (e1["days"], e1["exit_ret_pct"]) == (12, 12.0)
    assert str(e2["evict_date"]) == "2019-07-31"
    assert (e2["days"], e2["exit_ret_pct"]) == (6, round((99 / 117.6 - 1) * 100, 2))
    assert str(e3["evict_date"]) == "2019-08-30"
    assert (e3["days"], e3["exit_ret_pct"]) == (90, 0.0)
    for e in ev:
        assert e["market"] == "US" and e["drift_at_evict"] == -25.2


def test_only_bear_negative_drift_names_are_evicted(setup):
    ev = BTA.analyze_market("US")
    assert {e["symbol"] for e in ev} == {"AAA"}                   # rising names are never bear; SHORT < 400 bars


def test_history_floor_is_400_bars(setup, monkeypatch):
    prices = dict(setup)
    prices["SHORT"] = prices["AAA"].iloc[-400:].copy()
    monkeypatch.setattr(BTA, "load_market", lambda m: to_long(prices))
    assert {e["symbol"] for e in BTA.analyze_market("US")} == {"AAA", "SHORT"}      # 400 bars: kept
    prices["SHORT"] = prices["SHORT"].iloc[1:]
    assert {e["symbol"] for e in BTA.analyze_market("US")} == {"AAA"}               # 399: skipped


def test_reentry_leg_uses_calendar_horizons_and_censors(setup):
    e1 = events_by(BTA.analyze_market("US"))[0]
    assert str(e1["trigger_date"]) == "2019-07-10"
    assert e1["reentry_ret_21"] == pytest.approx(5.0)             # 117.6 / 112 on Jul 31
    assert e1["reentry_ret_63"] == pytest.approx(round((99 / 112 - 1) * 100, 2))
    assert e1["reentry_ret_126"] is None and e1["bench_ret_126"] is None
    assert e1["reentry_excess_126"] is None
    for h in BTA.REENTRY_HORIZONS[:2]:
        assert e1[f"reentry_excess_{h}"] == pytest.approx(
            round(e1[f"reentry_ret_{h}"] - e1[f"bench_ret_{h}"], 2), abs=0.011)


def test_benchmark_is_equal_weight_of_the_universe(setup):
    px = pd.DataFrame(setup).sort_index()
    eqw = (1 + px.pct_change().mean(axis=1).fillna(0)).cumprod()
    e1 = events_by(BTA.analyze_market("US"))[0]
    i, j = eqw.index.get_loc(pd.Timestamp("2019-07-10")), eqw.index.get_loc(pd.Timestamp("2019-07-31"))
    assert e1["bench_ret_21"] == pytest.approx(round((eqw.iloc[j] / eqw.iloc[i] - 1) * 100, 2))


def test_non_anomaly_events_have_no_reentry_fields(setup):
    for e in events_by(BTA.analyze_market("US"))[1:]:
        assert e["trigger_date"] is None and e["rsi_at_trigger"] is None
        for h in BTA.REENTRY_HORIZONS:
            assert e[f"reentry_ret_{h}"] is None and e[f"bench_ret_{h}"] is None
            assert e[f"reentry_excess_{h}"] is None


def test_rsi_recorded_at_evict_and_trigger(setup):
    ev = events_by(BTA.analyze_market("US"))
    rsi = BTA.rsi14_series(setup["AAA"])
    assert ev[0]["rsi_at_evict"] == round(rsi.loc["2019-06-28"], 1)
    assert ev[0]["rsi_at_trigger"] == round(rsi.loc["2019-07-10"], 1)
    assert ev[1]["rsi_at_evict"] == round(rsi.loc["2019-07-31"], 1)


def test_right_censored_final_month_produces_no_event(setup):
    ev = events_by(BTA.analyze_market("US"))
    assert all(str(e["evict_date"]) != "2019-09-30" for e in ev)  # nothing after the last bar


def flat_after_trigger(prices, monkeypatch, first, rest="2019-07-31"):
    """AAA: 100 at the Jun-28 eviction, `first` on Jul 1, then flat 100 (no later promotion)."""
    aaa = prices["AAA"].copy()
    aaa.loc["2019-07-01"] = first
    aaa.loc["2019-07-02":] = 100.0
    p = dict(prices)
    p["AAA"] = aaa
    monkeypatch.setattr(BTA, "load_market", lambda m: to_long(p))


def first_event(market="US"):
    return events_by(BTA.analyze_market(market))[0]


def test_rsi_buy_band_promotes_at_five_percent(setup, monkeypatch):
    flat_after_trigger(setup, monkeypatch, 106.0)                  # +6% from the Jun-28 close of 100
    assert first_event("US")["outcome"] == "ageout_validated"      # below the 10% bar, zone not BUY
    buy = first_event("BUYM")
    assert buy["outcome"] == "ANOMALY" and buy["exit_ret_pct"] == 6.0 and buy["days"] == 3
    assert str(buy["trigger_date"]) == "2019-07-01"


def test_promotion_barrier_is_inclusive(setup, monkeypatch):
    monkeypatch.setattr(BTA, "PROMOTE_RET", 12.5)                  # dyadic: 112.5/100 is exact
    flat_after_trigger(setup, monkeypatch, 112.5)
    assert first_event()["outcome"] == "ANOMALY"
    flat_after_trigger(setup, monkeypatch, 112.49)
    assert first_event()["outcome"] == "ageout_validated"


def test_rsi_promotion_barrier_is_inclusive(setup, monkeypatch):
    monkeypatch.setattr(BTA, "PROMOTE_RET_RSI", 6.25)
    flat_after_trigger(setup, monkeypatch, 106.25)
    assert first_event("BUYM")["outcome"] == "ANOMALY"
    assert first_event("US")["outcome"] == "ageout_validated"      # zone gate: not in BUY band
    flat_after_trigger(setup, monkeypatch, 106.24)
    assert first_event("BUYM")["outcome"] == "ageout_validated"


def test_decline_barrier_is_inclusive(setup, monkeypatch):
    monkeypatch.setattr(BTA, "AGEOUT_RET", -12.5)
    flat_after_trigger(setup, monkeypatch, 87.5)
    e = first_event()
    assert e["outcome"] == "validated_decline" and e["exit_ret_pct"] == -12.5 and e["days"] == 3
    flat_after_trigger(setup, monkeypatch, 87.51)
    assert first_event()["outcome"] == "ageout_validated"


def test_ageout_window_is_ninety_calendar_days(setup, monkeypatch):
    aaa = setup["AAA"].copy()
    aaa.loc["2019-07-01":] = 100.0
    aaa.loc["2019-09-26"] = 130.0                                  # Jun 28 + 90d = Sep 26: still inside
    p = dict(setup)
    p["AAA"] = aaa
    monkeypatch.setattr(BTA, "load_market", lambda m: to_long(p))
    e = first_event()
    assert e["outcome"] == "ANOMALY" and str(e["trigger_date"]) == "2019-09-26" and e["days"] == 90
    aaa.loc["2019-09-26"] = 100.0
    aaa.loc["2019-09-27"] = 130.0                                  # day 91: outside the window
    assert first_event()["outcome"] == "ageout_validated"


def test_kalman_failure_skips_the_name(setup, monkeypatch):
    class Boom:
        def filter(self, logp):
            raise RuntimeError("diverged")
    monkeypatch.setattr(BTA, "KalmanTrendModel", Boom)
    assert BTA.analyze_market("US") == []


def test_market_must_be_bull(setup, monkeypatch):
    prices = dict(setup)
    n = len(prices["BBB"])
    for k in ("BBB", "CCC", "DDD", "EEE"):
        prices[k] = pd.Series(100 * 0.9995 ** np.arange(n), index=prices[k].index)
    monkeypatch.setattr(BTA, "load_market", lambda m: to_long(prices))
    # falling peers: breadth (names above their 200DMA) is below half -> no evictions anywhere
    assert BTA.analyze_market("US") == []


def test_top_n_limits_universe_by_median_turnover(setup, monkeypatch):
    monkeypatch.setattr(BTA, "TOP_N", 2)
    prices = dict(setup)
    long = to_long(prices)
    long.loc[long.Symbol == "AAA", "Volume"] = 10**9               # most liquid names kept
    long.loc[long.Symbol == "BBB", "Volume"] = 10**8
    for k in ("CCC", "DDD", "EEE", "SHORT"):
        long.loc[long.Symbol == k, "Volume"] = 10
    monkeypatch.setattr(BTA, "load_market", lambda m: long)
    ev = BTA.analyze_market("US")
    assert {e["symbol"] for e in ev} == {"AAA"}
    long.loc[long.Symbol == "AAA", "Volume"] = 1                   # AAA drops out of the top 2
    assert BTA.analyze_market("US") == []


# ── main ──────────────────────────────────────────────────────────────────────
def fake_events(market):
    base = {"market": market, "symbol": "X", "rsi_at_evict": None, "drift_at_evict": -1.0}

    def ev(date, outcome, days, ret, rsi, r21=None, x21=None):
        e = dict(base, evict_date=date, outcome=outcome, days=days, exit_ret_pct=ret,
                 rsi_at_evict=rsi)
        for h in BTA.REENTRY_HORIZONS:
            e[f"reentry_ret_{h}"] = r21 if h == 21 else None
            e[f"reentry_excess_{h}"] = x21 if h == 21 else None
        return e
    return [ev("2019-01-31", "ANOMALY", 10, 12.0, 40.0, 3.0, 1.0),
            ev("2019-02-28", "ANOMALY", 20, 14.0, 50.0, 5.0, 3.0),
            ev("2020-01-31", "validated_decline", 5, -16.0, 30.0),
            ev("2020-02-28", "ageout_validated", 90, 1.0, 20.0)]


def test_main_writes_csv_and_markdown_summary(tmp_path, monkeypatch, capsys):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(BTA, "HERE", tmp_path)
    monkeypatch.setattr(BTA, "MARKETS", ["US", "JP"])

    def analyze(m):
        if m == "JP":
            raise RuntimeError("no data")
        return fake_events(m)
    monkeypatch.setattr(BTA, "analyze_market", analyze)
    BTA.main()
    out = capsys.readouterr().out
    assert "[JP] failed: no data" in out
    csv = pd.read_csv(tmp_path / "reports" / "tier_anomaly_events.csv")
    assert len(csv) == 4 and set(csv["market"]) == {"US"}
    md = (tmp_path / "reports" / "tier_anomaly_backtest.md").read_text()
    assert "## US — 4 evictions" in md
    assert "- ANOMALY 50.0% (median +13.0% in 15d) · validated-decline 25.0% · age-out 25.0%" in md
    assert "- RSI at eviction: anomalies median 45 vs validated 25" in md
    assert "- anomaly rate by year: 2019:100% · 2020:0%" in md
    assert ("- re-entry 21d: n=2 (censored 0) · raw +4.00% med / +4.00% mean · "
            "**excess +2.00% med / +2.00% mean** · win 100% · t=2.00") in md
    assert "- re-entry 63d: no uncensored events" in md
    assert "- re-entry 126d: no uncensored events" in md
    assert "Top 200 liquid names/market" in md and "(or +5.0% in RSI BUY band)" in md
    assert "validated at -15.0% or 90d" in md


def test_main_t_stat_nan_when_excess_has_no_dispersion(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(BTA, "HERE", tmp_path)
    monkeypatch.setattr(BTA, "MARKETS", ["US"])
    ev = fake_events("US")
    ev[0]["reentry_excess_21"] = ev[1]["reentry_excess_21"] = 2.0
    monkeypatch.setattr(BTA, "analyze_market", lambda m: ev)
    BTA.main()
    md = (tmp_path / "reports" / "tier_anomaly_backtest.md").read_text()
    assert "t=nan" in md
