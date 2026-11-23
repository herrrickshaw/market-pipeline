"""Offline tests for ml_signal_engine (features, sequences, thresholds); sklearn is faked."""

import numpy as np
import pandas as pd
import pytest

import ml_signal_engine as ml


def _ohlc(n=400, drift=0.0005, vol=0.01, seed=0, volume=1e6):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Volume": np.full(n, volume)}, index=pd.bdate_range("2022-01-03", periods=n))


# ---- fakes replacing scikit-learn ------------------------------------------------
class FakeModel:
    """Predicts a constant (set via FakeModel.value), records the training matrix shape."""
    value = 0.0
    last_fit_shape = None

    def fit(self, X, y):
        FakeModel.last_fit_shape = X.shape
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), FakeModel.value) + np.arange(len(X)) * 0.0


def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.fixture
def fake_sk(monkeypatch):
    monkeypatch.setattr(ml, "_SKLEARN_OK", True)
    monkeypatch.setattr(ml, "mean_squared_error", _mse, raising=False)
    monkeypatch.setattr(ml, "mean_absolute_error", _mae, raising=False)
    monkeypatch.setattr(ml.MLSignalEngine, "_make_model", lambda self: FakeModel())
    FakeModel.value = 0.0


# ---------------------------------------------------------------- compute_features
def test_compute_features_columns_order_and_warmup():
    f = ml.compute_features(_ohlc(400))
    assert list(f.columns) == ml.FEATURE_NAMES
    assert not f.isna().any().any()
    assert len(f) == 400 - 199  # the 200-bar DMA is the longest warm-up


def test_compute_features_known_values():
    df = _ohlc(300, drift=0.001, vol=0.004)
    f = ml.compute_features(df)
    c = df["Close"]
    t = f.index[-1]
    assert f.at[t, "ret_1d"] == pytest.approx((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    assert f.at[t, "ret_5d"] == pytest.approx((c.iloc[-1] / c.iloc[-6] - 1) * 100)
    assert f.at[t, "dma50_gap"] == pytest.approx((c.iloc[-1] / c.tail(50).mean() - 1) * 100)
    assert f.at[t, "dma200_gap"] == pytest.approx((c.iloc[-1] / c.tail(200).mean() - 1) * 100)
    assert f.at[t, "hl_ratio"] == pytest.approx((c.iloc[-1] * 1.01 - c.iloc[-1] * 0.99) / c.iloc[-1])
    mid, sd = c.tail(20).mean(), c.tail(20).std()
    assert f.at[t, "close_norm"] == pytest.approx((c.iloc[-1] - mid) / sd)
    assert f.at[t, "bb_pct"] == pytest.approx((c.iloc[-1] - (mid - 2 * sd)) / (4 * sd))
    assert f.at[t, "vol_20d"] == pytest.approx(c.pct_change().tail(20).std() * np.sqrt(252) * 100)
    assert f.at[t, "vol_ratio"] == pytest.approx(1.0)  # constant volume
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    assert f.at[t, "macd"] == pytest.approx(ema12.iloc[-1] - ema26.iloc[-1])
    assert f.at[t, "macd_signal"] == pytest.approx((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])


def test_compute_features_rsi_range_and_direction():
    up = ml.compute_features(_ohlc(300, drift=0.004, vol=0.002))
    assert up["rsi_14"].between(0, 100).all()
    down = ml.compute_features(_ohlc(300, drift=-0.004, vol=0.004, seed=1))
    assert down["rsi_14"].mean() < up["rsi_14"].mean()


def test_compute_features_zero_volume_bars_become_missing_not_zero():
    df = _ohlc(300)
    df.iloc[-1, df.columns.get_loc("Volume")] = 0
    f = ml.compute_features(df)
    assert f.index[-1] != df.index[-1]  # that row has NaN vol_ratio and is dropped


def test_compute_features_does_not_use_future_data():
    df = _ohlc(300)
    a = ml.compute_features(df)
    b = ml.compute_features(df.iloc[:250])
    common = b.index
    pd.testing.assert_frame_equal(a.loc[common], b, check_exact=False, rtol=1e-9)


# ---------------------------------------------------------------- z_score_normalise
def test_z_score_normalise_columns_and_constant_column_guard():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z = ml.z_score_normalise(X)
    assert Z[:, 0].mean() == pytest.approx(0) and Z[:, 0].std() == pytest.approx(1)
    assert Z[:, 1].tolist() == [0, 0, 0]  # constant column: std 0 -> divided by 1, not NaN
    assert X[0, 0] == 1.0  # input untouched


# ---------------------------------------------------------------- sequences / model
def test_make_sequences_shapes_alignment_and_empty():
    n, k = 100, len(ml.FEATURE_NAMES)
    feats = pd.DataFrame(np.arange(n * k, dtype=float).reshape(n, k), columns=ml.FEATURE_NAMES)
    tgt = pd.Series(np.arange(n, dtype=float))
    X, y = ml.MLSignalEngine()._make_sequences(feats, tgt)
    assert X.shape == (n - ml.PREDICT_DAYS - ml.LOOKBACK, ml.LOOKBACK * k)
    assert y[0] == ml.LOOKBACK + ml.PREDICT_DAYS and y[-1] == n - 1
    first = ml.z_score_normalise(feats.values[: ml.LOOKBACK]).flatten()
    assert X[0] == pytest.approx(first)
    X, y = ml.MLSignalEngine()._make_sequences(feats.iloc[:65], tgt.iloc[:65])
    assert X.size == 0 and y.size == 0


def test_make_model_without_sklearn_is_none(monkeypatch):
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert ml.MLSignalEngine()._make_model() is None


def test_make_model_picks_ridge_or_linear():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, Ridge

    assert isinstance(ml.MLSignalEngine("ridge")._make_model(), Ridge)
    assert type(ml.MLSignalEngine("lr")._make_model()) is LinearRegression


def test_constructor_defaults():
    e = ml.MLSignalEngine()
    assert (e.model_type, e.workers, e._model_cache) == ("ridge", 4, {})


# ---------------------------------------------------------------- predict
def test_predict_insufficient_data_paths(fake_sk, monkeypatch):
    e = ml.MLSignalEngine()
    short = e.predict("X", _ohlc(ml.MIN_ROWS - 1))
    assert short["direction"] == "INSUFFICIENT_DATA" and short["train_rows"] == 0
    assert short["symbol"] == "X" and short["model"] == "ridge" and short["predicted_ret%"] == 0.0
    assert e.predict("X", None)["direction"] == "INSUFFICIENT_DATA"
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert e.predict("X", _ohlc(400))["direction"] == "INSUFFICIENT_DATA"


def test_predict_exact_min_rows_boundary_does_not_short_circuit(fake_sk):
    r = ml.MLSignalEngine().predict("X", _ohlc(ml.MIN_ROWS))
    assert r["direction"] in ("INSUFFICIENT_DATA", "NEUTRAL", "BULLISH", "BEARISH")
    assert "error" not in r


@pytest.mark.parametrize(
    "value,direction",
    [(0.5, "BULLISH"), (0.4999, "NEUTRAL"), (0.0, "NEUTRAL"), (-0.4999, "NEUTRAL"),
     (-0.5, "BEARISH"), (3.0, "BULLISH"), (-3.0, "BEARISH")],
)
def test_predict_threshold_classification(fake_sk, value, direction):
    FakeModel.value = value
    r = ml.MLSignalEngine().predict("AAA.NS", _ohlc(500))
    assert r["direction"] == direction and r["predicted_ret%"] == round(value, 3)
    assert r["features_used"] == len(ml.FEATURE_NAMES)
    assert r["train_rows"] == FakeModel.last_fit_shape[0]
    assert FakeModel.last_fit_shape[1] == ml.LOOKBACK * len(ml.FEATURE_NAMES)


def test_predict_confidence_capped_and_rmse_reported(fake_sk):
    FakeModel.value = 5.0  # constant prediction: std 0 -> pred_std falls back to 1.0
    r = ml.MLSignalEngine().predict("A", _ohlc(500))
    assert r["confidence"] == pytest.approx(min(1.0, 5.0 / (2 * 1.0 + 1e-9)), abs=1e-3) == 1.0
    FakeModel.value = 0.5
    r = ml.MLSignalEngine().predict("A", _ohlc(500))
    assert r["confidence"] == pytest.approx(0.25, abs=1e-3)  # |0.5| / (2*1)
    assert r["train_rmse"] > 0


def test_predict_trains_on_at_most_train_window_and_excludes_target_leak(fake_sk):
    r = ml.MLSignalEngine().predict("A", _ohlc(900))
    assert r["train_rows"] <= ml.TRAIN_WINDOW - ml.LOOKBACK - ml.PREDICT_DAYS


def test_predict_swallows_model_errors(fake_sk, monkeypatch):
    class Boom(FakeModel):
        def fit(self, X, y):
            raise RuntimeError("singular")

    monkeypatch.setattr(ml.MLSignalEngine, "_make_model", lambda self: Boom())
    r = ml.MLSignalEngine().predict("A", _ohlc(500))
    assert r["direction"] == "NEUTRAL" and r["error"] == "singular"


# ---------------------------------------------------------------- predict_batch
def test_predict_batch_sorted_by_prediction_with_summary(fake_sk, capsys, monkeypatch):
    preds = {"A": 1.0, "B": -2.0, "C": 0.0}

    def fake_predict(self, sym, df):
        return {"symbol": sym, "direction": "BULLISH" if preds[sym] > 0.5 else (
            "BEARISH" if preds[sym] < -0.5 else "NEUTRAL"), "predicted_ret%": preds[sym]}

    monkeypatch.setattr(ml.MLSignalEngine, "predict", fake_predict)
    df = ml.MLSignalEngine(model_type="lr").predict_batch({k: None for k in preds}, max_workers=2)
    assert df["symbol"].tolist() == ["A", "C", "B"]
    out = capsys.readouterr().out
    assert "3/3 done | BULLISH: 1 | BEARISH: 1" in out and "ML signals computed: 3 stocks" in out
    assert "BULLISH" in out and "33.3%" in out


def test_predict_batch_isolates_failures(fake_sk, monkeypatch):
    def fake_predict(self, sym, df):
        if sym == "BAD":
            raise ValueError("boom")
        return {"symbol": sym, "direction": "NEUTRAL", "predicted_ret%": 0.1}

    monkeypatch.setattr(ml.MLSignalEngine, "predict", fake_predict)
    df = ml.MLSignalEngine().predict_batch({"OK": None, "BAD": None})
    row = df[df["symbol"] == "BAD"].iloc[0]
    assert row["direction"] == "ERROR" and row["error"] == "boom" and row["predicted_ret%"] == 0.0


def test_predict_batch_empty(fake_sk):
    assert ml.MLSignalEngine().predict_batch({}).empty


# ---------------------------------------------------------------- evaluate_on_backtest
def test_backtest_no_predictions(fake_sk):
    r = ml.MLSignalEngine().evaluate_on_backtest({"S": _ohlc(100), "N": None}, "2023-01-01", "2023-06-01")
    assert r == {"error": "No test predictions generated"}


def test_backtest_metrics_hit_rate_and_period(fake_sk, capsys):
    FakeModel.value = 1.0  # always predicts up
    df = _ohlc(600, drift=0.002, vol=0.002)
    start, end = df.index[450], df.index[470]
    r = ml.MLSignalEngine().evaluate_on_backtest({"S": df}, str(start.date()), str(end.date()))
    assert r["n_predictions"] > 0 and r["test_period"] == f"{start.date()} – {end.date()}"
    assert r["directional_acc"] == pytest.approx(100.0, abs=0.1)  # strongly rising series
    assert r["rmse"] >= r["mae"] > 0
    out = capsys.readouterr().out
    assert "Directional Acc:" in out and "RMSE:" in out


def test_backtest_wrong_direction_scores_zero(fake_sk):
    FakeModel.value = -1.0
    df = _ohlc(600, drift=0.002, vol=0.002)
    r = ml.MLSignalEngine().evaluate_on_backtest({"S": df}, str(df.index[450].date()), str(df.index[460].date()))
    assert r["directional_acc"] == 0.0


# ---------------------------------------------------------------- compare_models
def test_compare_models_guards(monkeypatch):
    assert ml.compare_models(_ohlc(100)).empty
    monkeypatch.setattr(ml, "_SKLEARN_OK", False)
    assert ml.compare_models(_ohlc(500)).empty


def test_compare_models_table_and_split(fake_sk, monkeypatch, capsys):
    monkeypatch.setattr(ml, "LinearRegression", FakeModel, raising=False)
    monkeypatch.setattr(ml, "Ridge", lambda alpha: FakeModel(), raising=False)
    FakeModel.value = 0.0
    out = ml.compare_models(_ohlc(500), "SYM")
    assert out["Model"].tolist() == ["LinearRegression", "Ridge(alpha=1)", "Naive(last_ret)"]
    assert list(out.columns) == ["Model", "Train_RMSE", "Train_MAE", "Test_RMSE", "Test_MAE"]
    # constant-zero predictor vs the naive mean predictor on identical data
    assert (out["Train_RMSE"] >= 0).all()
    text = capsys.readouterr().out
    assert "Model comparison for SYM (train=" in text and "Paper benchmark" in text
    assert out.loc[2, "Train_RMSE"] <= out.loc[0, "Train_RMSE"] + 1e-9  # mean predictor is optimal for MSE
