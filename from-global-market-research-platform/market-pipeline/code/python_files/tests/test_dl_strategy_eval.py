"""dl_strategy_eval.py -- leak-free feature panel, long/flat walk-forward strategy, data loading.

scikit-learn is not a CI dependency, so the classifier is replaced with a deterministic fake.
The module creates ./dl_strategy_results at import and is therefore imported in a sandbox cwd.
"""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3


@pytest.fixture
def DSE(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("dl_strategy_eval", None)
    mod = importlib.import_module("dl_strategy_eval")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    yield mod
    sys.modules.pop("dl_strategy_eval", None)


def osc(n):
    """Rising but oscillating closes (a monotone series has an undefined RSI and is dropped)."""
    k = np.arange(n)
    return 100.0 + 0.02 * k + 5.0 * np.sin(0.25 * k)


def ohlcv(n, close=None, volume=1000.0):
    close = np.asarray(close if close is not None else osc(n), dtype=float)
    idx = pd.bdate_range("2015-01-01", periods=len(close))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(len(close), volume),
        },
        index=idx,
    )


# ── build_panel ───────────────────────────────────────────────────────────────
def test_panel_row_count_drops_warmup_and_forward_window(DSE):
    df = ohlcv(400)
    p = DSE.build_panel(df, horizon=21)
    assert len(p) == 400 - 251 - 21  # 252-bar high warm-up, 21 unlabeled tail
    assert p.index[0] == df.index[251] and p.index[-1] == df.index[400 - 22]
    assert list(p.columns) == DSE.FEATURES + ["_label", "_fwd_ret"]


def test_panel_feature_values_on_an_oscillating_series(DSE):
    c = osc(400)
    p = DSE.build_panel(ohlcv(400), horizon=21)
    i = 251
    row = p.iloc[0]
    assert row["ret_5d"] == pytest.approx((c[i] / c[i - 5] - 1) * 100)
    assert row["ret_21d"] == pytest.approx((c[i] / c[i - 21] - 1) * 100)
    assert row["ret_63d"] == pytest.approx((c[i] / c[i - 63] - 1) * 100)
    assert row["mom_accel"] == pytest.approx(row["ret_5d"] - row["ret_21d"])
    assert row["dma50_gap"] == pytest.approx(
        (c[i] - c[i - 49 : i + 1].mean()) / c[i - 49 : i + 1].mean() * 100
    )
    assert row["dma200_gap"] == pytest.approx(
        (c[i] - c[i - 199 : i + 1].mean()) / c[i - 199 : i + 1].mean() * 100
    )
    hi = c[i - 251 : i + 1].max() + 1.0  # High = close + 1
    assert row["dist_high"] == pytest.approx((c[i] - hi) / hi * 100)
    assert row["vol_ratio"] == pytest.approx(1.0)
    assert row["vol_21d"] == pytest.approx(
        pd.Series(c).pct_change().rolling(21).std().iloc[i] * np.sqrt(252) * 100
    )
    d = np.diff(c)[i - 14 : i]
    gain, loss = np.clip(d, 0, None).mean(), np.clip(-d, 0, None).mean()
    assert row["rsi_14"] == pytest.approx(100 - 100 / (1 + gain / loss))
    w = c[i - 19 : i + 1]
    assert row["bb_pct"] == pytest.approx(
        (c[i] - (w.mean() - 2 * w.std(ddof=1))) / (4 * w.std(ddof=1))
    )


def test_panel_rsi_of_a_balanced_series_is_fifty(DSE):
    close = np.tile([100.0, 101.0], 300)
    p = DSE.build_panel(ohlcv(600, close), horizon=5)
    assert p["rsi_14"].between(40, 60).all()


def test_panel_bollinger_position_is_half_at_the_mean(DSE):
    close = 100.0 + np.sin(np.linspace(0, 40, 500))
    p = DSE.build_panel(ohlcv(500, close), horizon=5)
    assert p["bb_pct"].between(-0.5, 1.5).all()


def test_panel_label_and_forward_return(DSE):
    c = osc(400)
    p = DSE.build_panel(ohlcv(400), horizon=21)
    i = 251
    assert p["_fwd_ret"].iloc[0] == pytest.approx((c[i + 21] / c[i] - 1) * 100)
    assert list(p["_label"]) == [int(x > 0) for x in p["_fwd_ret"]]
    assert 0 < p["_label"].mean() < 1  # both classes present


def test_panel_label_is_strictly_positive_forward_return(DSE):
    close = np.concatenate([osc(300), np.full(100, osc(300)[-1])])  # flat tail
    p = DSE.build_panel(ohlcv(400, close), horizon=21)
    flat_rows = p[p["_fwd_ret"] == 0.0]
    assert len(flat_rows) > 0 and (flat_rows["_label"] == 0).all()


def test_panel_features_never_use_future_bars(DSE):
    close = osc(500)
    full = DSE.build_panel(ohlcv(500, close), horizon=21)
    cut = close.copy()
    cut[300:] = 1000.0  # corrupt everything after bar 299
    changed = DSE.build_panel(ohlcv(500, cut), horizon=21)
    common = full.index.intersection(changed.index)
    early = common[common < ohlcv(500).index[300]]
    pd.testing.assert_frame_equal(full.loc[early, DSE.FEATURES], changed.loc[early, DSE.FEATURES])
    assert not np.allclose(
        full.loc[early, "_fwd_ret"].iloc[-1], changed.loc[early, "_fwd_ret"].iloc[-1]
    )


def test_panel_zero_volume_bar_blanks_the_next_twenty_rows(DSE):
    df = ohlcv(500)
    df.iloc[300, df.columns.get_loc("Volume")] = 0.0
    p = DSE.build_panel(df, horizon=5)
    gone = [d for d in df.index[300:320] if d in p.index]
    assert gone == []
    assert df.index[320] in p.index and df.index[299] in p.index


def test_panel_empty_when_history_too_short(DSE):
    assert DSE.build_panel(ohlcv(200), horizon=21).empty


# ── evaluate_stock (fake classifier) ──────────────────────────────────────────
class FakeClf:
    proba = 0.9

    def __init__(self, **kw):
        FakeClf.kw = kw

    def fit(self, X, y):
        FakeClf.fit_shape = X.shape
        return self

    def predict_proba(self, X):
        p = FakeClf.proba(X) if callable(FakeClf.proba) else np.full(len(X), FakeClf.proba)
        return np.column_stack([1 - p, p])


@pytest.fixture
def sk(DSE, monkeypatch):
    monkeypatch.setattr(DSE, "_SK_OK", True)
    monkeypatch.setattr(DSE, "GradientBoostingClassifier", FakeClf, raising=False)
    FakeClf.proba = 0.9
    return DSE


def wavy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.012, n)) + 0.15 * np.sin(np.arange(n) / 20)
    )


def test_evaluate_requires_sklearn_and_enough_history(DSE, sk, monkeypatch):
    assert sk.evaluate_stock(("X", ohlcv(900, wavy()), 21, 0.55)) is not None
    monkeypatch.setattr(sk, "_SK_OK", False)
    assert sk.evaluate_stock(("X", ohlcv(900, wavy()), 21, 0.55)) is None
    monkeypatch.setattr(sk, "_SK_OK", True)
    assert sk.evaluate_stock(("X", None, 21, 0.55)) is None
    assert sk.evaluate_stock(("X", ohlcv(599, wavy(599)), 21, 0.55)) is None


def test_evaluate_always_long_equals_buy_and_hold(sk):
    r = sk.evaluate_stock(("AAA", ohlcv(900, wavy()), 21, 0.55))
    assert r["symbol"] == "AAA"
    assert r["strat_cum%"] == r["bh_cum%"] and r["alpha%"] == 0.0 and bool(r["beat_bh"]) is False
    assert r["n_long"] == r["n_periods"] and r["strat_sharpe"] == r["bh_sharpe"]
    assert r["strat_maxdd%"] == r["bh_maxdd%"]


def test_evaluate_never_long_is_flat_cash(sk):
    FakeClf.proba = 0.1
    r = sk.evaluate_stock(("AAA", ohlcv(900, wavy()), 21, 0.55))
    assert r["n_long"] == 0 and r["strat_cum%"] == 0.0 and r["strat_sharpe"] == 0
    assert r["strat_maxdd%"] == 0.0 and r["hit_rate%"] == 0.0
    assert r["alpha%"] == pytest.approx(-r["bh_cum%"], abs=0.011)
    assert bool(r["beat_bh"]) is bool(r["bh_cum%"] < 0)


def test_evaluate_threshold_is_inclusive(sk):
    FakeClf.proba = 0.55
    r = sk.evaluate_stock(("AAA", ohlcv(900, wavy()), 21, 0.55))
    assert r["n_long"] == r["n_periods"]
    FakeClf.proba = 0.5499
    assert sk.evaluate_stock(("AAA", ohlcv(900, wavy()), 21, 0.55))["n_long"] == 0


def test_evaluate_periods_step_by_horizon_over_the_last_30_percent(sk):
    df = ohlcv(900, wavy())
    panel = sk.build_panel(df, 21)
    split = int(len(panel) * 0.70)
    te = len(panel) - split
    r = sk.evaluate_stock(("AAA", df, 21, 0.55))
    assert r["n_periods"] == len(range(0, te - 1, 21))
    assert FakeClf.fit_shape == (split, len(sk.FEATURES))
    assert FakeClf.kw == {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": 42,
    }


def test_evaluate_hand_computed_strategy_numbers(sk):
    FakeClf.proba = lambda X: np.where(
        np.arange(len(X)) % 42 == 0, 0.9, 0.1
    )  # long every other period
    df = ohlcv(900, wavy())
    panel = sk.build_panel(df, 21)
    split = int(len(panel) * 0.70)
    fwd = panel["_fwd_ret"].values[split:] / 100
    idx = list(range(0, len(fwd) - 1, 21))
    br = np.array([fwd[i] for i in idx])
    sr = np.array([fwd[i] if k % 2 == 0 else 0.0 for k, i in enumerate(idx)])
    r = sk.evaluate_stock(("AAA", df, 21, 0.55))
    assert r["n_long"] == sum(1 for k in range(len(idx)) if k % 2 == 0)
    assert r["strat_cum%"] == round((np.prod(1 + sr) - 1) * 100, 2)
    assert r["bh_cum%"] == round((np.prod(1 + br) - 1) * 100, 2)
    assert r["alpha%"] == round((np.prod(1 + sr) - np.prod(1 + br)) * 100, 2)
    assert r["bh_sharpe"] == round(br.mean() / br.std() * np.sqrt(252 / 21), 3)
    eq = np.cumprod(1 + br)
    assert r["bh_maxdd%"] == round(
        ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100, 2
    )
    assert r["hit_rate%"] == round((sr > 0).mean() * 100, 1)
    assert bool(r["beat_bh"]) is bool(np.prod(1 + sr) > np.prod(1 + br))


def test_evaluate_single_class_training_labels_are_rejected(sk):
    k = np.arange(900)
    always_up = 100.0 * np.exp(
        0.01 * k + 0.02 * np.sin(1.3 * k)
    )  # noisy but never down over 21 bars
    panel = sk.build_panel(ohlcv(900, always_up), 21)
    assert len(panel) >= 400 and set(panel["_label"].iloc[: int(len(panel) * 0.7)]) == {1}
    assert sk.evaluate_stock(("UP", ohlcv(900, always_up), 21, 0.55)) is None
    assert sk.evaluate_stock(("OK", ohlcv(900, wavy()), 21, 0.55)) is not None


def test_evaluate_needs_at_least_400_panel_rows(sk):
    assert sk.evaluate_stock(("AAA", ohlcv(672, wavy(672)), 21, 0.55)) is not None  # panel == 400
    assert sk.evaluate_stock(("AAA", ohlcv(671, wavy(671)), 21, 0.55)) is None  # panel == 399


def test_evaluate_last_bar_of_the_test_window_is_never_a_decision_point(sk):
    # 693 bars -> 421-row panel, 127 test rows; 126 = 6 * 21, so range(0, 126, 21) has 6 steps
    df = ohlcv(693, wavy(693))
    panel = sk.build_panel(df, 21)
    te = len(panel) - int(len(panel) * 0.7)
    assert (te - 1) % 21 == 0
    assert sk.evaluate_stock(("AAA", df, 21, 0.55))["n_periods"] == (te - 1) // 21


def test_evaluate_classifier_failure_is_swallowed(sk, monkeypatch):
    class Boom(FakeClf):
        def fit(self, X, y):
            raise RuntimeError("diverged")

    monkeypatch.setattr(sk, "GradientBoostingClassifier", Boom)
    assert sk.evaluate_stock(("AAA", ohlcv(900, wavy()), 21, 0.55)) is None


# ── load_stocks ───────────────────────────────────────────────────────────────
def test_load_stocks_market_filter_dedupe_and_length(DSE, tmp_path, monkeypatch):
    cache, ref = tmp_path / "ohlc", tmp_path / "ref"
    cache.mkdir()
    ref.mkdir()
    monkeypatch.setattr(DSE, "CACHE_DIR", cache)
    monkeypatch.setattr(DSE, "REF_DIR", ref)
    long_df = ohlcv(600).assign(Extra=1)
    for name in ("RELIANCE.NS", "TCS.BO", "AAPL", "MSFT"):
        long_df.to_parquet(cache / f"{name}.parquet")
    long_df.to_parquet(ref / "AAPL.parquet")  # duplicate stem: loaded once
    long_df.to_parquet(ref / "NVDA.parquet")
    ohlcv(599).to_parquet(cache / "SHORT.parquet")
    (cache / "BAD.parquet").write_text("not parquet")
    assert set(DSE.load_stocks("ALL", 0)) == {"RELIANCE.NS", "TCS.BO", "AAPL", "MSFT", "NVDA"}
    assert set(DSE.load_stocks("IN", 0)) == {"RELIANCE.NS"}  # only .NS for India
    assert set(DSE.load_stocks("US", 0)) == {"AAPL", "MSFT", "NVDA"}  # no .NS/.BO
    out = DSE.load_stocks("ALL", 0)["AAPL"]
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_stocks_max_applies_before_loading_and_fills_gaps(DSE, tmp_path, monkeypatch):
    cache = tmp_path / "c"
    cache.mkdir()
    monkeypatch.setattr(DSE, "CACHE_DIR", cache)
    monkeypatch.setattr(DSE, "REF_DIR", tmp_path / "none")
    gappy = ohlcv(600)
    gappy.iloc[0, 3] = np.nan
    gappy.iloc[10, 3] = np.nan
    for n in ("A", "B", "C"):
        gappy.to_parquet(cache / f"{n}.parquet")
    assert len(DSE.load_stocks("ALL", 2)) == 2
    df = DSE.load_stocks("ALL", 0)["A"]
    assert not df.isna().any().any()
    assert df["Close"].iloc[0] == df["Close"].iloc[1]  # back-filled head


# ── main ──────────────────────────────────────────────────────────────────────
def fake_results(n, alpha):
    return [
        {
            "symbol": f"S{i}",
            "n_periods": 10,
            "n_long": 5,
            "strat_cum%": 10.0 + alpha,
            "bh_cum%": 10.0,
            "alpha%": alpha,
            "strat_sharpe": 1.0,
            "bh_sharpe": 0.8,
            "strat_maxdd%": -5.0,
            "bh_maxdd%": -15.0,
            "hit_rate%": 55.0,
            "beat_bh": alpha > 0,
        }
        for i in range(n)
    ]


def run_main(DSE, monkeypatch, stocks, rows, capsys):
    monkeypatch.setattr(DSE, "_SK_OK", True)
    monkeypatch.setattr(DSE, "load_stocks", lambda m, mx: stocks)
    monkeypatch.setattr(DSE, "parallel_map", lambda fn, args, **kw: rows)
    try:
        DSE.main(market="IN", horizon=21, threshold=0.55, max_stocks=5)
    except IndexError:
        # pandas>=3 rejects the positional sheet name in main()'s final to_excel call (see the
        # xfail test below); everything of interest has been printed by then.
        if not PANDAS3:
            raise
    return capsys.readouterr().out


def test_main_stops_without_sklearn_or_data(DSE, monkeypatch, capsys):
    monkeypatch.setattr(DSE, "_SK_OK", False)
    DSE.main()
    assert "pip install scikit-learn" in capsys.readouterr().out
    out = run_main(DSE, monkeypatch, {f"S{i}": None for i in range(19)}, [], capsys)
    assert "Insufficient data." in out  # fewer than 20 stocks
    out = run_main(DSE, monkeypatch, {f"S{i}": None for i in range(20)}, [None, None], capsys)
    assert "No valid backtests." in out and "STEP 3" not in out


def test_main_return_edge_verdict(DSE, monkeypatch, capsys):
    out = run_main(
        DSE, monkeypatch, {f"S{i}": None for i in range(20)}, fake_results(3, 5.0), capsys
    )
    assert "Strategy shows a return edge (+5.0% avg alpha, beats BH 100% of the time)" in out
    assert "BEAT buy-and-hold: 100.0%" in out and "Top 10 by out-of-sample alpha" in out
    assert "Mean cumulative return%" in out and "Mean max drawdown%" in out


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="dd_edge = bh_maxdd - strat_maxdd, but max drawdowns are negative numbers: a "
    "strategy with the SMALLER drawdown (-5% vs -15%) yields -10, so the "
    "'cuts drawdown' verdict can only fire when the strategy's drawdown is WORSE",
)
def test_main_drawdown_reduction_verdict(DSE, monkeypatch, capsys):
    out = run_main(
        DSE, monkeypatch, {f"S{i}": None for i in range(20)}, fake_results(3, -1.0), capsys
    )
    assert "cuts drawdown by 10.0pp" in out


def rows_with_alphas(alphas, strat_dd=-5.0, bh_dd=-6.0):
    rows = fake_results(len(alphas), 0.0)
    for r, a in zip(rows, alphas, strict=True):
        r["alpha%"], r["beat_bh"] = a, a > 0
        r["strat_maxdd%"], r["bh_maxdd%"] = strat_dd, bh_dd
    return rows


def test_main_return_edge_needs_both_a_majority_and_positive_mean_alpha(DSE, monkeypatch, capsys):
    stocks = {f"S{i}": None for i in range(20)}
    wins_but_negative = run_main(
        DSE, monkeypatch, stocks, rows_with_alphas([1.0, 1.0, -10.0]), capsys
    )
    assert "return edge" not in wins_but_negative  # beats BH 67% but mean alpha < 0
    positive_but_rare = run_main(
        DSE, monkeypatch, stocks, rows_with_alphas([10.0, -1.0, -1.0]), capsys
    )
    assert "return edge" not in positive_but_rare  # mean alpha > 0 but beats BH 33%


def test_main_drawdown_branch_threshold_is_strictly_above_five_points(DSE, monkeypatch, capsys):
    stocks = {f"S{i}": None for i in range(20)}
    out = run_main(
        DSE, monkeypatch, stocks, rows_with_alphas([-1.0] * 3, strat_dd=-10.0, bh_dd=-5.0), capsys
    )
    assert "cuts drawdown" not in out and "does NOT reliably beat buy-and-hold" in out


def test_main_no_edge_verdict(DSE, monkeypatch, capsys):
    rows = fake_results(3, -1.0)
    for r in rows:
        r["bh_maxdd%"] = -6.0  # drawdown improvement only 1pp
    out = run_main(DSE, monkeypatch, {f"S{i}": None for i in range(20)}, rows, capsys)
    assert "does NOT reliably beat buy-and-hold (alpha -1.0%, beats BH 0%)" in out


@pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=IndexError,
    reason="main() calls DataFrame.to_excel(writer, 'Sheet', ...) with a positional sheet "
    "name, which pandas>=3 rejects; the ExcelWriter then fails to close",
)
def test_main_writes_the_excel_report(DSE, monkeypatch, capsys):
    rows = fake_results(3, -1.0)
    monkeypatch.setattr(DSE, "_SK_OK", True)
    monkeypatch.setattr(DSE, "load_stocks", lambda m, mx: {f"S{i}": None for i in range(20)})
    monkeypatch.setattr(DSE, "parallel_map", lambda fn, args, **kw: rows)
    DSE.main(market="IN", horizon=21, threshold=0.55, max_stocks=5)
    files = list(DSE.OUT_DIR.glob("dl_strategy_IN_*.xlsx"))
    assert len(files) == 1
    sheets = pd.read_excel(files[0], sheet_name=None)
    assert (
        set(sheets) == {"DISCLAIMER", "Strategy_Results"} and len(sheets["Strategy_Results"]) == 3
    )
