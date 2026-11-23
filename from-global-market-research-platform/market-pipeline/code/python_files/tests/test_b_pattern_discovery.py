"""Offline tests for pattern_discovery (cleaning, feature engineering, archetypes, insights)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp(prefix="mp_pat_"))  # module creates ./pattern_results at import
try:
    import pattern_discovery as pdx
finally:
    os.chdir(_cwd)

N = 330


def _ohlcv(n=N, drift=0.001, vol=0.01, seed=0, volume=1e6, start=100.0):
    rng = np.random.default_rng(seed)
    c = start * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {
            "Open": c * 0.999,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Volume": np.full(n, volume),
        },
        index=idx,
    )


# ---------------------------------------------------------------- load_and_clean
@pytest.fixture
def dirs(monkeypatch, tmp_path):
    a, b = tmp_path / "cache", tmp_path / "ref"
    a.mkdir()
    b.mkdir()
    monkeypatch.setattr(pdx, "CACHE_DIR", a)
    monkeypatch.setattr(pdx, "REF_DIR", b)
    return a, b


def _save(d, name, df):
    df.to_parquet(d / f"{name}.parquet")


def test_load_filters_short_corrupt_nonpositive_and_illiquid(dirs):
    a, b = dirs
    _save(a, "GOOD.NS", _ohlcv())
    _save(a, "SHORT.NS", _ohlcv(n=251))
    _save(a, "EXACT.NS", _ohlcv(n=252))
    bad = _ohlcv()
    bad.iloc[5, bad.columns.get_loc("Close")] = 0.0
    _save(a, "ZERO.NS", bad)
    _save(a, "ILLIQ.NS", _ohlcv(volume=0.0))
    (a / "BROKEN.NS.parquet").write_text("not parquet")
    out = pdx.load_and_clean("ALL")
    assert sorted(out) == ["EXACT.NS", "GOOD.NS"]
    assert list(out["GOOD.NS"].columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_min_bars_override_and_dedup_sort_ffill(dirs):
    a, _ = dirs
    df = _ohlcv(n=300)
    dup = pd.concat([df, df.iloc[[-1]].assign(Close=555.0)]).iloc[::-1]  # unsorted + duplicate date
    dup.iloc[10, dup.columns.get_loc("Volume")] = np.nan
    _save(a, "X.NS", dup)
    out = pdx.load_and_clean("ALL", min_bars=100)["X.NS"]
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out["Volume"].notna().all()
    assert pdx.load_and_clean("ALL", min_bars=1000) == {}


def test_load_market_filter_cap_and_cross_dir_dedupe(dirs):
    a, b = dirs
    for n in ("A.NS", "B.BO", "C"):
        _save(a, n, _ohlcv())
    _save(b, "A.NS", _ohlcv(seed=9))
    assert sorted(pdx.load_and_clean("IN")) == ["A.NS"]  # only .NS; .BO excluded
    assert sorted(pdx.load_and_clean("US")) == ["C"]
    assert sorted(pdx.load_and_clean("ALL")) == ["A.NS", "B.BO", "C"]
    assert len(pdx.load_and_clean("ALL", max_stocks=2)) == 2
    first = pdx.load_and_clean("IN")["A.NS"]
    assert first["Close"].iloc[0] == pytest.approx(_ohlcv()["Close"].iloc[0])  # cache dir wins


# ---------------------------------------------------------------- extract_features
def test_extract_features_needs_enough_history():
    assert pdx.extract_features("X", _ohlcv(n=252 + pdx.FWD - 1)) == {}
    assert pdx.extract_features("X", _ohlcv(n=252 + pdx.FWD))["symbol"] == "X"


def test_extract_features_has_all_names_and_finite_values():
    f = pdx.extract_features("AAA", _ohlcv())
    assert set(pdx.FEATURE_NAMES) <= set(f) and "_target_fwd21" in f
    assert all(np.isfinite(f[k]) for k in pdx.FEATURE_NAMES)


def test_extract_features_target_is_forward_return_and_features_do_not_see_it():
    df = _ohlcv()
    f = pdx.extract_features("A", df)
    last, fut = df["Close"].iloc[-pdx.FWD - 1], df["Close"].iloc[-1]
    assert f["_target_fwd21"] == pytest.approx((fut - last) / last * 100)
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("Close")] *= 5  # future bar must not move any feature
    g = pdx.extract_features("A", spiked)
    for k in pdx.FEATURE_NAMES:
        assert g[k] == pytest.approx(f[k])
    assert g["_target_fwd21"] != pytest.approx(f["_target_fwd21"])


def test_extract_features_known_values_on_trend():
    n = 330
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = 100 * 1.002 ** np.arange(n)
    df = pd.DataFrame(
        {"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx
    )
    f = pdx.extract_features("T", df)
    cc = c[: n - pdx.FWD]
    assert f["ret_1m"] == pytest.approx((cc[-1] / cc[-21] - 1) * 100)
    assert f["ret_12m"] == pytest.approx((cc[-1] / cc[-252] - 1) * 100)
    assert f["max_drawdown"] == pytest.approx(0.0)
    assert f["rsi_14"] == 100  # no losses
    assert f["up_day_ratio"] == pytest.approx(100.0)
    assert f["dist_52w_high"] == pytest.approx(
        (cc[-1] - cc[-252:].max() * 1.01) / (cc[-252:].max() * 1.01) * 100
    )
    assert f["above_200dma_pct_days"] == pytest.approx(
        110 / 252 * 100
    )  # the 200DMA exists for 110 of the last 252 bars
    assert f["dma_trend"] > 0 and f["dma50_gap"] > 0 and f["dma200_gap"] > f["dma50_gap"]


def test_extract_features_downtrend_signs_and_rsi_bounds():
    f = pdx.extract_features("D", _ohlcv(drift=-0.003, vol=0.004))
    assert f["ret_12m"] < 0 and f["dma200_gap"] < 0 and f["max_drawdown"] < 0
    assert 0 <= f["rsi_14"] <= 100
    assert f["calmar_proxy"] == pytest.approx(f["ret_12m"] / abs(f["max_drawdown"]))


def test_extract_features_gap_frequency_counts_large_opens():
    df = _ohlcv(vol=0.0001, drift=0.0)
    df.iloc[100:110, df.columns.get_loc("Open")] *= 1.05
    base = pdx.extract_features("G", _ohlcv(vol=0.0001, drift=0.0))["gap_freq"]
    assert pdx.extract_features("G", df)["gap_freq"] > base


def test_extract_features_flat_prices_are_handled():
    n = 330
    c = np.full(n, 100.0)
    df = pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6},
        index=pd.bdate_range("2023-01-02", periods=n),
    )
    f = pdx.extract_features("F", df)
    assert f["vol_ratio"] == 1 and f["rsi_14"] == 100 and f["calmar_proxy"] == 0
    assert all(np.isfinite(f[k]) for k in pdx.FEATURE_NAMES)


# ---------------------------------------------------------------- build_feature_matrix
def test_build_feature_matrix_drops_unusable_and_keeps_columns(capsys):
    cleaned = {"A": _ohlcv(seed=1), "B": _ohlcv(seed=2), "SHORT": _ohlcv(n=100)}
    fm = pdx.build_feature_matrix(cleaned)
    assert sorted(fm["symbol"]) == ["A", "B"]
    assert set(pdx.FEATURE_NAMES) <= set(fm.columns)
    assert "Feature matrix: 2 stocks" in capsys.readouterr().out


# ---------------------------------------------------------------- name_archetype
def _prof(**kw):
    base = dict(ret_12m=0, vol_60d=30, dma200_gap=0, max_drawdown=-10, rsi_14=50, mean_reversion=0)
    base.update(kw)
    return pd.Series(base)


@pytest.mark.parametrize(
    "kw,label",
    [
        (dict(ret_12m=41, dma200_gap=11), "High-Momentum"),
        (dict(ret_12m=40, dma200_gap=11), "Balanced"),  # strict >
        (dict(ret_12m=41, dma200_gap=10), "Balanced"),
        (dict(ret_12m=-21, dma200_gap=-11), "Sustained Downtrenders"),
        (dict(ret_12m=-20, dma200_gap=-11), "Balanced"),
        (dict(vol_60d=51), "High-Volatility"),
        (dict(vol_60d=50), "Balanced"),
        (dict(vol_60d=24, dma200_gap=4.9), "Low-Vol Range-Bound"),
        (dict(vol_60d=25, dma200_gap=4.9), "Balanced"),
        (dict(vol_60d=24, dma200_gap=-4.9), "Low-Vol Range-Bound"),
        (dict(vol_60d=24, dma200_gap=5), "Balanced"),
        (dict(mean_reversion=10.1), "Mean-Reverters"),
        (dict(mean_reversion=10), "Balanced"),
        (dict(rsi_14=66, ret_12m=16), "Overbought Momentum"),
        (dict(rsi_14=65, ret_12m=16), "Balanced"),
        (dict(rsi_14=66, ret_12m=15), "Balanced"),
        (dict(max_drawdown=-51), "Deep-Drawdown Recovery"),
        (dict(max_drawdown=-50), "Balanced"),
        (dict(), "Balanced"),
    ],
)
def test_name_archetype(kw, label):
    assert label in pdx.name_archetype(_prof(**kw))


def test_name_archetype_priority_momentum_beats_volatility():
    assert "High-Momentum" in pdx.name_archetype(_prof(ret_12m=60, dma200_gap=20, vol_60d=80))
    assert "High-Volatility" in pdx.name_archetype(_prof(vol_60d=80, mean_reversion=50))


# ---------------------------------------------------------------- find_comoving_pairs
def test_comoving_pairs_finds_correlated_and_ignores_independent():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 260)
    idx = pd.bdate_range("2023-01-02", periods=261)

    def frame(r):
        c = 100 * np.cumprod(1 + np.r_[0, r])
        return pd.DataFrame({"Close": c}, index=idx)

    cleaned = {
        "A": frame(base),
        "B": frame(base + rng.normal(0, 0.001, 260)),
        "C": frame(rng.normal(0, 0.01, 260)),
        "SHORT": frame(base).iloc[:100],
    }
    fm = pd.DataFrame(
        {"symbol": ["A", "B", "C", "SHORT", "MISSING"], "avg_dollar_vol": [5, 4, 3, 2, 1]}
    )
    pairs = pdx.find_comoving_pairs(cleaned, fm)
    assert pairs[["Stock_A", "Stock_B"]].values.tolist() == [["A", "B"]]
    assert pairs["Correlation"].iloc[0] > 0.95


def test_comoving_pairs_too_few_or_top_n_cap():
    fm = pd.DataFrame({"symbol": ["A"], "avg_dollar_vol": [1]})
    assert pdx.find_comoving_pairs({"A": _ohlcv()}, fm).empty
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, 260)
    idx = pd.bdate_range("2023-01-02", periods=261)
    cleaned = {
        f"S{i}": pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + np.r_[0, base + rng.normal(0, 0.0005, 260)])}, index=idx
        )
        for i in range(6)
    }
    fm = pd.DataFrame({"symbol": list(cleaned), "avg_dollar_vol": range(6)})
    assert len(pdx.find_comoving_pairs(cleaned, fm, top_n=4)) == 4
    p = pdx.find_comoving_pairs(cleaned, fm, top_n=50)
    assert len(p) == 15 and p["Correlation"].is_monotonic_decreasing


# ---------------------------------------------------------------- insights / excel
def _fm():
    rows = []
    for i in range(6):
        r = {k: float(i) for k in pdx.FEATURE_NAMES}
        r.update(
            symbol=f"S{i}",
            cluster=i % 2,
            is_anomaly=(i == 5),
            ret_12m=50.0,
            dma200_gap=20.0,
            vol_60d=10.0 + i,
        )
        rows.append(r)
    return pd.DataFrame(rows)


def test_extract_insights_summarises_clusters_importances_anomalies_pairs(capsys):
    fm = _fm()
    imp = pd.DataFrame(
        {
            "Feature": pdx.FEATURE_NAMES[:3],
            "Importance": [0.5, 0.3, 0.2],
            "Importance%": [50.0, 30.0, 20.0],
        }
    )
    pairs = pd.DataFrame({"Stock_A": ["S0"], "Stock_B": ["S1"], "Correlation": [0.9]})
    ins = pdx.extract_insights(fm, imp, pairs, np.array([0.5, 0.2]))
    cl = ins["clusters"]
    assert cl["Cluster"].tolist() == [0, 1] and cl["Count"].tolist() == [3, 3]
    assert cl["Examples"].tolist() == ["S0, S2, S4", "S1, S3, S5"]
    assert all("High-Momentum" in a for a in cl["Archetype"])
    assert ins["importances"] is imp and ins["pairs"] is pairs
    assert ins["anomalies"]["symbol"].tolist() == ["S5"]
    out = capsys.readouterr().out
    assert "RETURN DRIVERS" in out and "ANOMALIES (1 stocks" in out and "CO-MOVING PAIRS" in out


def test_extract_insights_omits_empty_sections():
    fm = _fm().assign(is_anomaly=False)
    ins = pdx.extract_insights(fm, pd.DataFrame(), pd.DataFrame(), np.array([0.5, 0.2]))
    assert set(ins) == {"clusters"}


@pytest.mark.xfail(
    int(pd.__version__.split(".")[0]) >= 3,
    strict=True,
    raises=(TypeError, IndexError),
    reason="source bug on pandas>=3: to_excel(w, 'Sheet', ...) passes sheet_name positionally, "
    "which pandas 3 makes keyword-only",
)
def test_save_excel_writes_expected_sheets(monkeypatch, tmp_path):
    monkeypatch.setattr(pdx, "OUT_DIR", tmp_path)
    fm = _fm()
    ins = {"clusters": pd.DataFrame({"a": [1]}), "pairs": pd.DataFrame({"b": [1]})}
    path = pdx.save_excel(ins, fm, "IN")
    assert path.parent == tmp_path and path.name.startswith("patterns_IN_")
    sheets = pd.ExcelFile(path).sheet_names
    assert sheets == ["DISCLAIMER", "Archetypes", "CoMoving_Pairs", "All_Features"]


# ---------------------------------------------------------------- sklearn-backed steps


def _big_fm(n=120):
    rng = np.random.default_rng(0)
    d = {k: rng.normal(size=n) for k in pdx.FEATURE_NAMES}
    d["symbol"] = [f"S{i}" for i in range(n)]
    d["_target_fwd21"] = d["ret_1m"] * 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame(d)


def test_discover_clusters_labels_every_stock():
    pytest.importorskip("sklearn")
    fm, km, pca, evr = pdx.discover_clusters(_big_fm(), n_clusters=3)
    assert set(fm["cluster"]) <= {0, 1, 2} and len(fm) == 120
    assert {"pc1", "pc2", "is_anomaly"} <= set(fm.columns) and len(evr) == 5


def test_supervised_drivers_needs_hundred_rows_and_finds_signal():
    pytest.importorskip("sklearn")
    assert pdx.supervised_drivers(_big_fm(99)).empty
    imp = pdx.supervised_drivers(_big_fm(120))
    assert imp["Feature"].iloc[0] == "ret_1m" and imp["Importance"].is_monotonic_decreasing
    assert imp["Importance%"].sum() == pytest.approx(100, abs=2)


def test_main_flow_guards(monkeypatch, capsys):
    monkeypatch.setattr(pdx, "_SK_OK", True)
    monkeypatch.setattr(pdx, "load_and_clean", lambda m, mx: {"A": 1})
    pdx.main("ALL")
    assert "Insufficient data." in capsys.readouterr().out
    monkeypatch.setattr(pdx, "load_and_clean", lambda m, mx: {str(i): 1 for i in range(60)})
    monkeypatch.setattr(pdx, "build_feature_matrix", lambda c: pd.DataFrame({"a": [1]}))
    pdx.main("ALL")
    assert "Insufficient features." in capsys.readouterr().out
    monkeypatch.setattr(pdx, "_SK_OK", False)
    pdx.main("ALL")
    assert "pip install scikit-learn" in capsys.readouterr().out
