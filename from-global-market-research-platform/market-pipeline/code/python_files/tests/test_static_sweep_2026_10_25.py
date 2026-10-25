"""Proof tests for the 2026-10-25 static-analysis sweep (ruff/import-smoke/py_compile).

Each test failed on the code as found and passes with the matching fix. All
offline: the only network call site (MarketCache._yf_download_single) is
monkeypatched, and every path lives under the conftest sandbox or tmp_path.
"""
import importlib
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parent.parent


def _ohlc(n: int = 30, start: str = "2026-01-05") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0,
         "Close": close, "Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


# ruff RET503 market_data_cache.py:211 — get_ohlc fell off the end (returned
# None) when the Parquet file exists but the meta index has no "to" date, e.g.
# meta/cache_index.json lost or written by an older version. Callers do
# `cache.get_ohlc(t).empty` and crash with AttributeError.
def test_get_ohlc_returns_dataframe_when_parquet_exists_but_meta_has_no_date(monkeypatch):
    mdc = importlib.import_module("market_data_cache")
    cache = mdc.MarketCache(verbose=False)
    cache._meta = {}                       # no entry -> _last_cached_date() is None
    stale = _ohlc(20, "2025-06-02")
    path = mdc.OHLC_DIR / "SWEEPTEST.parquet"
    stale.to_parquet(path)

    fresh = _ohlc(30)
    calls = []

    def fake_download(ticker, start):
        calls.append((ticker, start))
        return fresh.copy()

    monkeypatch.setattr(cache, "_yf_download_single", fake_download)
    out = cache.get_ohlc("SWEEPTEST")

    assert isinstance(out, pd.DataFrame), "get_ohlc must never return None"
    assert not out.empty
    assert calls, "a cache entry without a last-bar date must trigger a (re)download"
    # and the cache is now self-healed: meta carries the last bar date
    assert cache._last_cached_date("ohlc:SWEEPTEST") == fresh.index.max().date()


# import_smoke: earnings_price_dataset.py imports CIRCUIT_BOUND_PCT from a
# module (regime_price_model) that is not in the repo, so the script cannot
# even be imported/run from a checkout.
def test_earnings_price_dataset_is_importable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)            # module mkdirs cache_seed/... relative to cwd
    mod = importlib.import_module("earnings_price_dataset")
    # the literature-backed regulatory single-day limits the module documents
    assert mod.CIRCUIT_BOUND_PCT["IN"] == pytest.approx(20.0)
    assert mod.CIRCUIT_BOUND_PCT["KR"] == pytest.approx(30.0)
    assert set(mod.CIRCUIT_BOUND_PCT) >= {"IN", "KR", "US", "JP"}


# py_compile / mypy [syntax]: deoldify.py is a Colab export whose `!pip ...`
# shell cells were left as bare lines, so the file is not valid Python and
# breaks any tool (compileall, coverage, packaging) that compiles the tree.
def test_deoldify_compiles():
    py_compile.compile(str(CODE / "deoldify.py"), doraise=True)
