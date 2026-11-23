"""Offline tests for sector_analysis (sector cache, indices, fingerprints, archetypes)."""

import json
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pytest

_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp(prefix="mp_sec_"))  # module creates ./sector_results at import
try:
    import sector_analysis as sa
finally:
    os.chdir(_cwd)

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
IDX = pd.bdate_range("2022-01-03", periods=600)


@pytest.fixture
def env(monkeypatch, tmp_path):
    for n in ("cache", "ref", "out"):
        (tmp_path / n).mkdir()
    monkeypatch.setattr(sa, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sa, "REF_DIR", tmp_path / "ref")
    monkeypatch.setattr(sa, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(sa, "SECTOR_CACHE", tmp_path / "sector_map.json")
    return tmp_path


def _px(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))}, index=IDX[:n])


# ---------------------------------------------------------------- sector cache
def test_sector_cache_roundtrip_missing_and_corrupt(env):
    assert sa._load_sector_cache() == {}
    sa._save_sector_cache({"A": {"sector": "IT", "industry": "Software"}})
    assert sa._load_sector_cache() == {"A": {"sector": "IT", "industry": "Software"}}
    sa.SECTOR_CACHE.write_text("{bad")
    assert sa._load_sector_cache() == {}


# ---------------------------------------------------------------- fetch_sector
class _Tk:
    def __init__(self, info, boom=False):
        self._info, self._boom = info, boom

    @property
    def info(self):
        if self._boom:
            raise RuntimeError("rate limited")
        return self._info


def test_fetch_sector_variants(monkeypatch):
    infos = {"A": {"sector": "IT", "industry": "Software"}, "B": {"sector": None}, "C": None}
    monkeypatch.setattr(sa, "_YF_OK", True)
    monkeypatch.setattr(
        sa,
        "yf",
        types.SimpleNamespace(Ticker=lambda s: _Tk(infos[s]) if s in infos else _Tk({}, boom=True)),
        raising=False,
    )
    assert sa.fetch_sector("A") == ("A", "IT", "Software")
    assert sa.fetch_sector("B") == ("B", "Unknown", "Unknown")
    assert sa.fetch_sector("C") == ("C", "Unknown", "Unknown")
    assert sa.fetch_sector("Z") == ("Z", "Unknown", "Unknown")  # exception swallowed
    monkeypatch.setattr(sa, "_YF_OK", False)
    assert sa.fetch_sector("A") == ("A", "Unknown", "Unknown")


# ---------------------------------------------------------------- classify_sectors
def test_classify_fetches_only_missing_and_persists(env, monkeypatch, capsys):
    sa._save_sector_cache({"OLD": {"sector": "Banks", "industry": "Private"}})
    seen = []

    def fake_map(fn, items, workers, progress_every, label):
        seen.append((list(items), workers))
        return [(s, "IT", "Soft") for s in items]

    monkeypatch.setattr(sa, "parallel_map", fake_map)
    out = sa.classify_sectors(["OLD", "NEW"], workers=3)
    assert seen == [(["NEW"], 3)]
    assert out == {
        "OLD": {"sector": "Banks", "industry": "Private"},
        "NEW": {"sector": "IT", "industry": "Soft"},
    }
    assert set(json.loads(sa.SECTOR_CACHE.read_text())) == {"OLD", "NEW"}
    assert "2 symbols | 1 cached | 1 to fetch" in capsys.readouterr().out


def test_classify_refresh_ignores_cache_and_no_fetch_when_all_cached(env, monkeypatch):
    sa._save_sector_cache({"A": {"sector": "X", "industry": "Y"}})
    calls = []
    monkeypatch.setattr(
        sa,
        "parallel_map",
        lambda fn, items, **k: calls.append(list(items)) or [(s, "Z", "W") for s in items],
    )
    assert sa.classify_sectors(["A"])["A"]["sector"] == "X" and calls == []
    assert sa.classify_sectors(["A"], refresh=True)["A"]["sector"] == "Z" and calls == [["A"]]


def test_classify_returns_only_requested_symbols(env, monkeypatch):
    sa._save_sector_cache(
        {"A": {"sector": "X", "industry": "Y"}, "B": {"sector": "X", "industry": "Y"}}
    )
    assert list(sa.classify_sectors(["A"])) == ["A"]


# ---------------------------------------------------------------- load_returns / load_universe
def test_load_returns_min_bars_tail_and_directories(env):
    _px(600, 1).to_parquet(env / "cache" / "A.parquet")
    _px(251, 2).to_parquet(env / "cache" / "SHORT.parquet")
    _px(252, 3).to_parquet(env / "ref" / "EDGE.parquet")
    (env / "cache" / "BAD.parquet").write_text("junk")
    out = sa.load_returns(["A", "SHORT", "EDGE", "BAD", "NONE"])
    assert sorted(out) == ["A", "EDGE"]
    assert len(out["A"]) == 504 and len(out["EDGE"]) == 251
    assert out["A"].iloc[0] == pytest.approx(_px(600, 1)["Close"].pct_change().dropna().iloc[-504])
    assert sorted(sa.load_returns(["SHORT"], min_bars=100)) == ["SHORT"]


def test_load_universe_filters_by_market_and_cap(env):
    for n in ("A.NS", "B.BO", "C", "D"):
        _px(5).to_parquet(env / "cache" / f"{n}.parquet")
    _px(5).to_parquet(env / "ref" / "A.NS.parquet")
    assert sa.load_universe("IN", 0) == ["A.NS"]
    assert sa.load_universe("US", 0) == ["C", "D"]
    assert sa.load_universe("ALL", 0) == ["A.NS", "B.BO", "C", "D"]
    assert sa.load_universe("ALL", 2) == ["A.NS", "B.BO"]


# ---------------------------------------------------------------- build_sector_indices
def test_build_sector_indices_groups_filters_and_averages(capsys):
    r = {s: pd.Series([i * 0.01, i * 0.01], index=IDX[:2]) for i, s in enumerate("ABCDEF", start=1)}
    smap = {
        "A": {"sector": "IT"},
        "B": {"sector": "IT"},
        "C": {"sector": "IT"},
        "D": {"sector": "Banks"},
        "E": {"sector": "Banks"},
        "F": {"sector": "Unknown"},
        "G": {"sector": "IT"},
        "H": {"sector": None},
    }  # G has no returns
    idx, groups = sa.build_sector_indices(smap, r)
    assert groups == {
        "IT": ["A", "B", "C"]
    }  # Banks has 2 (<3); Unknown/None excluded; G missing data
    assert idx["IT"].tolist() == pytest.approx([0.02, 0.02])  # mean of 0.01, 0.02, 0.03
    assert "1 sectors with ≥3 liquid members" in capsys.readouterr().out


def test_build_sector_indices_min_members_boundary():
    r = {s: pd.Series([0.01, 0.02], index=IDX[:2]) for s in "AB"}
    smap = {"A": {"sector": "X"}, "B": {"sector": "X"}}
    _, groups = sa.build_sector_indices(smap, r)
    assert groups == {}


# ---------------------------------------------------------------- sector_fingerprint
def _series(n=400, seed=0, drift=0.0006, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, vol, n), index=IDX[:n])


def test_fingerprint_too_short():
    s = _series(99)
    assert sa.sector_fingerprint(s, s) == {}
    assert sa.sector_fingerprint(_series(100), _series(100)) != {}


def test_fingerprint_values():
    r = _series(400, 1)
    m = _series(400, 2)
    fp = sa.sector_fingerprint(r, m)
    assert set(fp) == {
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "beta_to_market",
        "trend_persistence",
        "best_month",
        "worst_month",
        "up_month_ratio",
        "recent_3m_mom",
    }
    ann = (1 + r).prod() ** (252 / 400) - 1
    assert fp["ann_return"] == pytest.approx(ann * 100, abs=0.01)
    assert fp["ann_vol"] == pytest.approx(r.std() * np.sqrt(252) * 100, abs=0.01)
    assert fp["sharpe"] == pytest.approx(ann / (r.std() * np.sqrt(252)), abs=1e-3)
    cum = (1 + r).cumprod()
    assert fp["max_drawdown"] == pytest.approx(
        ((cum - cum.cummax()) / cum.cummax()).min() * 100, abs=0.01
    )
    assert fp["recent_3m_mom"] == pytest.approx(((1 + r.tail(63)).prod() - 1) * 100, abs=0.01)
    assert fp["worst_month"] < 0 < fp["best_month"] and 0 < fp["up_month_ratio"] < 100


def test_fingerprint_beta_is_cov_over_variance_and_defaults_to_one():
    m = _series(300, 5)
    fp = sa.sector_fingerprint(m * 2, m)
    assert fp["beta_to_market"] == pytest.approx(2.0, abs=1e-3)
    assert sa.sector_fingerprint(m, m * 0.0)["beta_to_market"] == 1.0  # zero-variance market
    short = pd.concat([m.iloc[:20], pd.Series(np.nan, index=m.index[20:])])
    assert sa.sector_fingerprint(m, short)["beta_to_market"] == 1.0  # <=30 aligned observations


def test_fingerprint_zero_vol_gives_zero_sharpe():
    flat = pd.Series(0.0, index=IDX[:200])
    assert sa.sector_fingerprint(flat, flat)["sharpe"] == 0


def test_fingerprint_ignores_nans():
    r = _series(300, 3)
    noisy = r.copy()
    noisy.iloc[5] = np.nan
    assert sa.sector_fingerprint(noisy, r)["ann_vol"] == pytest.approx(
        sa.sector_fingerprint(r.drop(r.index[5]), r)["ann_vol"], abs=0.01
    )


# ---------------------------------------------------------------- name_sector_cluster
@pytest.mark.parametrize(
    "ret,vol,beta,label",
    [
        (26, 36, 1.0, "High-Growth"),
        (25, 36, 1.0, "Market-Tracking"),
        (26, 35, 1.0, "Market-Tracking"),
        (16, 29, 1.0, "Quality Compounders"),
        (15, 29, 1.0, "Market-Tracking"),
        (16, 30, 1.0, "Market-Tracking"),
        (5, 40, 1.21, "Aggressive"),
        (5, 40, 1.2, "Market-Tracking"),
        (5, 40, 0.69, "Defensive"),
        (5, 40, 0.7, "Market-Tracking"),
        (-1, 40, 1.0, "Laggards"),
        (0, 40, 1.0, "Market-Tracking"),
        (30, 40, 0.5, "High-Growth"),
    ],  # growth outranks defensive
)
def test_name_sector_cluster(ret, vol, beta, label):
    p = pd.Series({"ann_return": ret, "ann_vol": vol, "beta_to_market": beta})
    assert label in sa.name_sector_cluster(p)


# ---------------------------------------------------------------- extract_patterns
def test_extract_patterns_returns_correlation_and_prints_rankings(capsys):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    idx = pd.DataFrame(
        {f"S{i}": base * (1 if i < 3 else -1) + rng.normal(0, 0.002, 200) for i in range(6)},
        index=IDX[:200],
    )
    rows = [
        {
            "sector": c,
            "cluster": i % 2,
            "ann_return": 10.0 + i,
            "ann_vol": 20.0,
            "sharpe": 0.5 + i,
            "max_drawdown": -10.0,
            "beta_to_market": 1.0,
            "recent_3m_mom": float(i),
            **{
                f: 1.0
                for f in sa.SECTOR_FEATURES
                if f
                not in (
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                    "beta_to_market",
                    "recent_3m_mom",
                )
            },
        }
        for i, c in enumerate(idx.columns)
    ]
    fdf = pd.DataFrame(rows)
    corr = sa.extract_patterns(fdf, idx, idx.mean(axis=1))
    assert corr.shape == (6, 6) and corr.loc["S0", "S1"] > 0.9 and corr.loc["S0", "S5"] < -0.9
    out = capsys.readouterr().out
    assert "SECTOR RANKING" in out and "Leaders:  S5 (+5%), S4 (+4%), S3 (+3%)" in out
    assert "Laggards: S2 (+2%), S1 (+1%), S0 (+0%)" in out
    first = out.split("SECTOR RANKING")[1].splitlines()[3]
    assert first.strip().startswith("S5")  # highest Sharpe first


# ---------------------------------------------------------------- save_excel
@pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=(TypeError, IndexError),
    reason="source bug on pandas>=3: to_excel(w, 'Sheet', ...) passes sheet_name positionally, "
    "which pandas 3 makes keyword-only",
)
def test_save_excel_sheets(env):
    fdf = pd.DataFrame({"sector": ["A"], "sharpe": [1.0]})
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    path = sa.save_excel(fdf, None, corr, {"A": ["x", "y"]}, "IN")
    assert path.parent == env / "out" and path.name.startswith("sector_analysis_IN_")
    assert pd.ExcelFile(path).sheet_names == [
        "DISCLAIMER",
        "Sector_Fingerprints",
        "Sector_Correlation",
        "Sector_Members",
    ]
    mem = pd.read_excel(path, sheet_name="Sector_Members")
    assert mem.loc[0, "Members"] == "x, y" and mem.loc[0, "N"] == 2


# ---------------------------------------------------------------- sklearn-backed + main
def test_cluster_sectors_fingerprints_and_cluster_count():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    idx = pd.DataFrame(
        {f"S{i}": rng.normal(0.0004 * (i % 3), 0.005 * (1 + i % 3), 400) for i in range(8)},
        index=IDX[:400],
    )
    fdf, market = sa.cluster_sectors(idx, 3)
    assert sorted(fdf["sector"]) == sorted(idx.columns) and fdf["cluster"].nunique() <= 3
    assert market.equals(idx.mean(axis=1))
    fdf2, _ = sa.cluster_sectors(idx.iloc[:, :3], 5)  # fewer sectors than clusters -> max(2, n//2)
    assert fdf2["cluster"].nunique() <= 2


def test_main_guards(env, monkeypatch, capsys):
    monkeypatch.setattr(sa, "_SK_OK", False)
    sa.main("IN")
    assert "pip install scikit-learn yfinance" in capsys.readouterr().out
    monkeypatch.setattr(sa, "_SK_OK", True)
    monkeypatch.setattr(sa, "_YF_OK", True)
    monkeypatch.setattr(sa, "load_universe", lambda m, mx: ["A", "B"])
    monkeypatch.setattr(sa, "classify_sectors", lambda s, refresh=False: {"A": {"sector": "IT"}})
    monkeypatch.setattr(sa, "load_returns", lambda s: {"A": 1})
    monkeypatch.setattr(
        sa, "build_sector_indices", lambda m, r: (pd.DataFrame({"x": [1], "y": [1]}), {})
    )
    sa.main("IN")
    assert "Not enough sectors with data." in capsys.readouterr().out


def test_fingerprint_beta_needs_more_than_thirty_aligned_observations():
    m = _series(300, 5)
    sparse = pd.concat([m.iloc[:31], pd.Series(np.nan, index=m.index[31:])])
    assert sa.sector_fingerprint(m * 2, sparse)["beta_to_market"] == pytest.approx(
        2.0, abs=1e-3
    )  # 31 > 30
    sparse30 = pd.concat([m.iloc[:30], pd.Series(np.nan, index=m.index[30:])])
    assert sa.sector_fingerprint(m * 2, sparse30)["beta_to_market"] == 1.0  # 30 is not enough
