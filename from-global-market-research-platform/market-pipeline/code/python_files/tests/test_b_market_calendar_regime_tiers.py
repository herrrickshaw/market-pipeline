"""Offline tests for market_calendar, market_regime and market_tiers."""

import datetime as dt
import json
import os
import sys
import tempfile
import types

# market_calendar creates its cache dir at import time; keep it out of $HOME.
os.environ.setdefault("BHAV_CACHE", tempfile.mkdtemp(prefix="mp_bhav_"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import market_calendar as mc  # noqa: E402
import market_regime as mr  # noqa: E402
import market_tiers as mt  # noqa: E402


# ================================================================ market_calendar
@pytest.mark.parametrize(
    "raw,iso",
    [
        ("15-Jan-2026", "2026-01-15"),
        ("2026-01-15", "2026-01-15"),
        ("15-January-2026", "2026-01-15"),
        ("  2026-01-15 ", "2026-01-15"),
        ("garbage ", "garbage"),
    ],
)
def test_parse(raw, iso):
    assert mc._parse(raw) == iso


@pytest.fixture
def cal(monkeypatch, tmp_path):
    cache = tmp_path / "nse_holidays.json"
    monkeypatch.setattr(mc, "HOLIDAY_CACHE", cache)
    return cache


def test_load_cache_missing(cal):
    assert mc._load_cache() == set()


def test_load_cache_fresh_and_stale_boundary(cal):
    today = dt.date.today()
    cal.write_text(json.dumps({"fetched": today.isoformat(), "dates": ["2030-01-01"]}))
    assert mc._load_cache() == {"2030-01-01"}
    edge = today - dt.timedelta(days=mc.REFRESH_AFTER_DAYS)  # exactly 14 days: still fresh
    cal.write_text(json.dumps({"fetched": edge.isoformat(), "dates": ["2030-01-01"]}))
    assert mc._load_cache() == {"2030-01-01"}
    old = today - dt.timedelta(days=mc.REFRESH_AFTER_DAYS + 1)
    cal.write_text(json.dumps({"fetched": old.isoformat(), "dates": ["2030-01-01"]}))
    assert mc._load_cache() == set()


def test_load_cache_corrupt_or_missing_keys(cal):
    cal.write_text("{not json")
    assert mc._load_cache() == set()
    cal.write_text(json.dumps({"dates": ["2030-01-01"]}))  # no 'fetched' -> epoch -> stale
    assert mc._load_cache() == set()


def _fake_nsepython(monkeypatch, payload=None, exc=None):
    def nse_holidays():
        if exc:
            raise exc
        return payload

    monkeypatch.setitem(sys.modules, "nsepython", types.SimpleNamespace(nse_holidays=nse_holidays))


def test_fetch_live_parses_cm_then_cbm(monkeypatch):
    _fake_nsepython(monkeypatch, {"CM": [{"tradingDate": "26-Jan-2031"}, {"tradingDate": ""}, {}]})
    assert mc._fetch_live() == {"2031-01-26"}
    _fake_nsepython(monkeypatch, {"CBM": [{"tradingDate": "2031-03-01"}]})
    assert mc._fetch_live() == {"2031-03-01"}
    _fake_nsepython(monkeypatch, {})
    assert mc._fetch_live() == set()


def test_fetch_live_failure_is_empty(monkeypatch):
    _fake_nsepython(monkeypatch, exc=RuntimeError("cookie"))
    assert mc._fetch_live() == set()
    monkeypatch.setitem(sys.modules, "nsepython", None)
    assert mc._fetch_live() == set()


def test_get_holidays_uses_fresh_cache_without_fetching(monkeypatch, cal):
    cal.write_text(json.dumps({"fetched": dt.date.today().isoformat(), "dates": ["2031-05-05"]}))
    monkeypatch.setattr(mc, "_fetch_live", lambda: pytest.fail("fetched"))
    h = mc.get_holidays()
    assert "2031-05-05" in h and h >= mc._FALLBACK


def test_get_holidays_refresh_ignores_cache_and_persists_live(monkeypatch, cal):
    cal.write_text(json.dumps({"fetched": dt.date.today().isoformat(), "dates": ["2031-05-05"]}))
    monkeypatch.setattr(mc, "_fetch_live", lambda: {"2031-06-06"})
    h = mc.get_holidays(refresh=True)
    assert "2031-06-06" in h and "2031-05-05" not in h and h >= mc._FALLBACK
    saved = json.loads(cal.read_text())
    assert saved == {"fetched": dt.date.today().isoformat(), "dates": ["2031-06-06"]}


def test_get_holidays_failed_live_returns_fallback_and_writes_nothing(monkeypatch, cal):
    monkeypatch.setattr(mc, "_fetch_live", lambda: set())
    assert mc.get_holidays() == mc._FALLBACK
    assert not cal.exists()


def test_get_holidays_unwritable_cache_is_tolerated(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "HOLIDAY_CACHE", tmp_path / "no_dir" / "h.json")
    monkeypatch.setattr(mc, "_fetch_live", lambda: {"2031-06-06"})
    assert "2031-06-06" in mc.get_holidays()


def test_is_trading_day_weekend_holiday_and_normal():
    h = {"2026-01-15"}
    assert mc.is_trading_day(dt.date(2026, 1, 14), h) is True  # Wednesday
    assert mc.is_trading_day(dt.date(2026, 1, 15), h) is False  # holiday
    assert mc.is_trading_day(dt.date(2026, 1, 17), set()) is False  # Saturday
    assert mc.is_trading_day(dt.date(2026, 1, 18), set()) is False  # Sunday
    assert mc.is_trading_day(dt.date(2026, 1, 16), set()) is True  # Friday
    # an explicit EMPTY holiday set must be honoured, not replaced by get_holidays()
    assert mc.is_trading_day(dt.date(2026, 1, 15), set()) is True


def test_is_trading_day_default_holidays(monkeypatch):
    monkeypatch.setattr(mc, "get_holidays", lambda refresh=False: {"2026-01-14"})
    assert mc.is_trading_day(dt.date(2026, 1, 14)) is False


def test_trading_days_single_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "get_holidays", lambda refresh=False: calls.append(1) or {"2026-01-15"})
    days = [dt.date(2026, 1, d) for d in range(12, 20)]  # Mon..Mon
    got = mc.trading_days(days)
    assert got == [
        dt.date(2026, 1, 12),
        dt.date(2026, 1, 13),
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 16),
        dt.date(2026, 1, 19),
    ]
    assert calls == [1]


def test_fallback_holidays_are_valid_iso_dates():
    for s in mc._FALLBACK:
        assert dt.date.fromisoformat(s).isoformat() == s


# ================================================================ market_tiers
def test_tiers_partition_and_cadence():
    assert mt.TIER1 == ["IN", "US"] and mt.TIER2 == ["EU", "JP", "KR"]
    assert not (set(mt.TIER3) & (set(mt.TIER1) | set(mt.TIER2)))
    assert sorted(mt.TIER3) == mt.TIER3 and mt.TIER3
    assert mt.CADENCE == {1: "daily", 2: "daily", 3: "weekly"}


@pytest.mark.parametrize(
    "m,tier,cad",
    [
        ("IN", 1, "daily"),
        ("us", 1, "daily"),
        ("Eu", 2, "daily"),
        ("jp", 2, "daily"),
        ("KR", 2, "daily"),
        ("HK", 3, "weekly"),
        ("ZZ", 3, "weekly"),
        ("", 3, "weekly"),
    ],
)
def test_tier_of_and_cadence_of(m, tier, cad):
    assert mt.tier_of(m) == tier
    assert mt.cadence_of(m) == cad


# ================================================================ market_regime
def _alt(amps):
    return [a * (1 if i % 2 else -1) for i, a in enumerate(amps)]


def _compute(monkeypatch, rets, market="us", n_cols=20):
    n = len(rets)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = {f"T{i}": 100 * np.cumprod(1 + np.asarray(rets)) for i in range(n_cols)}
    px = pd.DataFrame(cols, index=idx)
    seen = {}

    def download(tickers, **kw):
        seen["tickers"], seen["kw"] = tickers, kw
        return pd.concat({"Close": px}, axis=1)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    res = mr._compute(market)
    return res, seen


def test_compute_bull_and_bear(monkeypatch):
    assert _compute(monkeypatch, [0.002] * 300)[0]["trend"] == "bull"
    assert _compute(monkeypatch, [-0.002] * 300)[0]["trend"] == "bear"


def test_compute_chop(monkeypatch):
    res, _ = _compute(monkeypatch, [0.0] * 250 + [-0.02] * 4 + [0.1])
    assert res["trend"] == "chop"


def test_compute_requests_basket_two_years_daily(monkeypatch):
    _, seen = _compute(monkeypatch, [0.002] * 300, market="japan")
    assert seen["tickers"] == mr.BASKETS["japan"]
    assert seen["kw"]["period"] == "2y" and seen["kw"]["interval"] == "1d"
    assert seen["kw"]["auto_adjust"] is True and seen["kw"]["progress"] is False


def test_compute_too_short_history_unknown(monkeypatch):
    assert _compute(monkeypatch, [0.001] * 50)[0] == {"trend": "unknown", "vol": "unknown"}


def test_compute_too_few_names_unknown(monkeypatch):
    # n > 4 required per day: four columns can never form an index
    assert _compute(monkeypatch, [0.002] * 300, n_cols=4)[0] == {
        "trend": "unknown",
        "vol": "unknown",
    }


def test_compute_vol_low_high_mid(monkeypatch):
    low = _compute(monkeypatch, _alt([0.02] * 250 + [0.001] * 80))[0]["vol"]
    high = _compute(monkeypatch, _alt([0.001] * 250 + [0.02] * 80))[0]["vol"]
    assert (low, high) == ("LOW", "HIGH")
    amps = (
        [0.01] * 100 + list(np.linspace(0.002, 0.02, 100)) + list(np.linspace(0.02, 0.002, 26)[1:])
    )
    assert _compute(monkeypatch, _alt(amps))[0]["vol"] == "MID"


def test_compute_masks_impossible_returns(monkeypatch):
    # a 70% one-day print on every name is masked (>0.6) and must not move the index
    rets = [0.002] * 150 + [0.7] + [0.002] * 149
    res, _ = _compute(monkeypatch, rets)
    assert res["trend"] == "bull"


def test_get_regime_unknown_market_no_cache_io(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "CACHE", str(tmp_path / "r" / "c.json"))
    monkeypatch.setattr(mr, "_compute", lambda m: pytest.fail("computed"))
    r = mr.get_regime("mars")
    assert r["trend"] == "unknown" and r["vol"] == "unknown"
    assert r["asof"] == str(dt.date.today())
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize(
    "alias,market",
    [
        ("IN", "india"),
        ("eu", "europe"),
        ("JP", "japan"),
        ("kr", "korea"),
        ("CN", "china"),
        ("NSE", "india"),
        ("US", "us"),
        ("Japan", "japan"),
    ],
)
def test_get_regime_aliases(monkeypatch, tmp_path, alias, market):
    monkeypatch.setattr(mr, "CACHE", str(tmp_path / "c.json"))
    seen = []
    monkeypatch.setattr(mr, "_compute", lambda m: seen.append(m) or {"trend": "bull", "vol": "LOW"})
    mr.get_regime(alias)
    assert seen == [market]


def test_get_regime_computes_and_caches_for_the_day(monkeypatch, tmp_path):
    path = tmp_path / "sub" / "c.json"
    monkeypatch.setattr(mr, "CACHE", str(path))
    calls = []
    monkeypatch.setattr(
        mr, "_compute", lambda m: calls.append(m) or {"trend": "bull", "vol": "LOW"}
    )
    today = str(dt.date.today())
    assert mr.get_regime("us") == {"trend": "bull", "vol": "LOW", "asof": today}
    assert mr.get_regime("us")["trend"] == "bull"
    assert calls == ["us"]  # second call served from cache
    assert json.loads(path.read_text())["us"]["asof"] == today


def test_get_regime_stale_cache_recomputes_and_keeps_other_markets(monkeypatch, tmp_path):
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "us": {"trend": "bear", "vol": "HIGH", "asof": "2000-01-01"},
                "japan": {"trend": "chop", "vol": "MID", "asof": "2000-01-01"},
            }
        )
    )
    monkeypatch.setattr(mr, "CACHE", str(path))
    monkeypatch.setattr(mr, "_compute", lambda m: {"trend": "bull", "vol": "LOW"})
    assert mr.get_regime("us")["trend"] == "bull"
    saved = json.loads(path.read_text())
    assert saved["us"]["trend"] == "bull" and saved["japan"]["trend"] == "chop"


def test_get_regime_failure_returns_stale_hit_or_unknown(monkeypatch, tmp_path):
    path = tmp_path / "c.json"
    stale = {"trend": "bear", "vol": "HIGH", "asof": "2000-01-01"}
    path.write_text(json.dumps({"us": stale}))
    monkeypatch.setattr(mr, "CACHE", str(path))

    def boom(m):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(mr, "_compute", boom)
    assert mr.get_regime("us") == stale
    r = mr.get_regime("india")
    assert r["trend"] == "unknown" and r["asof"] == str(dt.date.today())
    assert json.loads(path.read_text()) == {"us": stale}  # failures are not cached


def test_get_regime_corrupt_cache_and_unwritable_dir(monkeypatch, tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{bad")
    monkeypatch.setattr(mr, "CACHE", str(path))
    monkeypatch.setattr(mr, "_compute", lambda m: {"trend": "chop", "vol": "MID"})
    assert mr.get_regime("us")["trend"] == "chop"
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(mr, "CACHE", str(blocker / "c.json"))  # makedirs under a file fails
    assert mr.get_regime("us")["trend"] == "chop"


def test_baskets_have_twenty_unique_names():
    assert set(mr.BASKETS) == {"us", "india", "japan", "korea", "china", "europe"}
    for m, b in mr.BASKETS.items():
        assert len(b) == 20 and len(set(b)) == 20, m
