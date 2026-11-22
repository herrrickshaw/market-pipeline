"""liquidity.py -- USD turnover, scan gate tiers, FX cache/fallback chain, index.

All FX/network paths are faked; cache files live under tmp_path.
"""
import datetime as dt
import io
import json
import sys
import types
import urllib.request

import numpy as np
import pandas as pd
import pytest

import liquidity as L


def frame(close, volume):
    return pd.DataFrame({"Close": close, "Volume": volume})


# ── currency_for ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ticker,ccy", [
    ("AAPL", "USD"), ("RELIANCE.NS", "INR"), ("VOD.L", "GBp"), ("7203.T", "JPY"),
    ("ADS.DE", "EUR"), ("X.UNKNOWN", "USD"), ("A.B.L", "GBp"), ("TEVA.TA", "ILA"),
])
def test_currency_for(ticker, ccy):
    assert L.currency_for(ticker) == ccy


# ── turnover_usd_for ──────────────────────────────────────────────────────────
def test_turnover_is_median_of_close_times_volume_over_rate():
    df = frame([10, 10, 10], [100, 200, 1_000_000])          # product median = 2000
    assert L.turnover_usd_for(df, "AAPL", {"USD": 1.0}) == 2000.0
    assert L.turnover_usd_for(df, "RELIANCE.NS", {"INR": 50.0}) == 40.0


def test_subunit_quote_divided_by_100_before_fx():
    df = frame([100.0] * 3, [10] * 3)                        # 1000 pence = 10 GBP
    assert L.turnover_usd_for(df, "VOD.L", {"GBP": 0.5}) == pytest.approx(20.0)
    # the GBp key itself is NOT an FX key
    assert L.turnover_usd_for(df, "VOD.L", {"GBp": 0.5}) == 0.0


@pytest.mark.parametrize("df", [
    None, "notaframe", pd.DataFrame({"Close": [1]}), pd.DataFrame({"Volume": [1]}),
    frame([np.nan, np.nan], [1, 1]), frame([0, 0], [5, 5]), frame([-1, -1], [5, 5]),
    frame(["a", "b"], ["c", "d"]), frame([], []),
])
def test_turnover_zero_for_unusable_frames(df):
    assert L.turnover_usd_for(df, "AAPL", {"USD": 1.0}) == 0.0


@pytest.mark.parametrize("rate", [0, -2.0, None])
def test_turnover_zero_for_bad_rate(rate):
    assert L.turnover_usd_for(frame([1, 1], [1, 1]), "AAPL", {"USD": rate}) == 0.0
    assert L.turnover_usd_for(frame([1, 1], [1, 1]), "AAPL", {}) == 0.0


# ── amihud_illiq ──────────────────────────────────────────────────────────────
def growth_frame(n, g=0.01, vol=1000.0):
    close = 100.0 * (1 + g) ** np.arange(n)
    return frame(close, [vol] * n)


def test_amihud_value_matches_hand_formula():
    df = growth_frame(61)
    close = df["Close"].to_numpy()
    expected = np.mean((close[1:] / close[:-1] - 1) / (close[1:] * 1000.0)) * 1e6
    assert L.amihud_illiq(df, "AAPL", {"USD": 1.0}) == pytest.approx(expected)


def test_amihud_min_days_boundary():
    assert L.amihud_illiq(growth_frame(61), "AAPL", {"USD": 1.0}) is not None   # 60 returns
    assert L.amihud_illiq(growth_frame(60), "AAPL", {"USD": 1.0}) is None       # 59 returns
    assert L.amihud_illiq(growth_frame(11), "AAPL", {"USD": 1.0}, min_days=10) is not None
    assert L.amihud_illiq(growth_frame(10), "AAPL", {"USD": 1.0}, min_days=10) is None


def test_amihud_drops_zero_volume_days_instead_of_counting_zero_impact():
    df = growth_frame(61)
    df.loc[df.index[-5:], "Volume"] = 0                     # 5 untraded days
    assert L.amihud_illiq(df, "AAPL", {"USD": 1.0}, min_days=60) is None
    assert L.amihud_illiq(df, "AAPL", {"USD": 1.0}, min_days=55) is not None


def test_amihud_fx_direction():
    df = growth_frame(61)
    base = L.amihud_illiq(df, "AAPL", {"USD": 1.0})
    assert L.amihud_illiq(df, "RELIANCE.NS", {"INR": 2.0}) == pytest.approx(base * 2)
    # pence: /100 then /rate  => 100x the per-USD impact of a USD-quoted equal series
    assert L.amihud_illiq(df, "VOD.L", {"GBP": 1.0}) == pytest.approx(base * 100)


@pytest.mark.parametrize("fx", [{}, {"USD": 0}, {"USD": -1.0}])
def test_amihud_none_on_bad_fx(fx):
    assert L.amihud_illiq(growth_frame(61), "AAPL", fx) is None


def test_amihud_none_when_unmeasurable_or_garbage():
    assert L.amihud_illiq(None, "AAPL", {"USD": 1.0}) is None
    assert L.amihud_illiq(pd.DataFrame({"Close": [1.0] * 70}), "AAPL", {"USD": 1.0}) is None
    df = frame(["x"] * 70, ["y"] * 70)
    assert L.amihud_illiq(df, "AAPL", {"USD": 1.0}) is None


def test_amihud_nan_mean_is_none():
    # min_days=0 lets an all-NaN return series through; the mean is NaN -> None
    assert L.amihud_illiq(frame([5.0], [5.0]), "AAPL", {"USD": 1.0}, min_days=0) is None


def test_amihud_flat_price_gives_zero_not_none():
    df = frame([5.0] * 70, [100.0] * 70)
    assert L.amihud_illiq(df, "AAPL", {"USD": 1.0}) == 0.0


# ── measurable / scan_gate ────────────────────────────────────────────────────
def test_measurable():
    assert L.measurable(frame([1], [1]))
    assert not L.measurable(None)
    assert not L.measurable(pd.DataFrame({"Close": [1]}))
    assert not L.measurable(pd.DataFrame({"Volume": [1]}))
    assert not L.measurable([1, 2])


def usd(tv):
    return frame([1.0] * 3, [tv] * 3)


@pytest.mark.parametrize("tv,tier", [
    (12_000_000, "T1_MEGA"), (11_999_999, "T2_LARGE"),
    (3_000_000, "T2_LARGE"), (2_999_999, "T3_MID"),
    (600_000, "T3_MID"), (599_999, "T4_SMALL"),
    (10_000, "T4_SMALL"),
])
def test_scan_gate_tier_boundaries(tv, tier):
    got_tv, got_tier = L.scan_gate(usd(tv), "AAPL", {"USD": 1.0})
    assert got_tv == tv and got_tier == tier


def test_scan_gate_structural_floor_is_exclusive_of_below():
    assert L.scan_gate(usd(9_999), "AAPL", {"USD": 1.0}) == (9_999.0, None)


def test_scan_gate_india_policy_floor():
    assert L.scan_gate(usd(119_999), "AAPL", {"USD": 1.0}, market_hint="IN") == (119_999.0, None)
    assert L.scan_gate(usd(120_000), "AAPL", {"USD": 1.0}, market_hint="IN")[1] == "T4_SMALL"
    assert L.scan_gate(usd(119_999), "AAPL", {"USD": 1.0}, market_hint="US")[1] == "T4_SMALL"


def test_scan_gate_fails_open_as_unknown():
    assert L.scan_gate(None, "AAPL", {"USD": 1.0}) == (0.0, "UNKNOWN")
    assert L.scan_gate(pd.DataFrame({"Close": [1]}), "AAPL", {"USD": 1.0}) == (0.0, "UNKNOWN")
    # currency without an FX rate is unmeasurable, not illiquid
    assert L.scan_gate(usd(1e9), "RELIANCE.NS", {"USD": 1.0}) == (0.0, "UNKNOWN")
    # sub-unit tickers need the BASE currency rate
    assert L.scan_gate(usd(1e9), "VOD.L", {"GBp": 1.0}) == (0.0, "UNKNOWN")


def test_scan_gate_subunit_uses_base_rate():
    # 1e6 pence-volume*close... close 1 * vol 1e8 = 1e8 pence = 1e6 GBP, rate 1 -> T3
    tv, tier = L.scan_gate(usd(1e8), "VOD.L", {"GBP": 1.0})
    assert tv == pytest.approx(1e6) and tier == "T3_MID"


def test_scan_floor_falls_back_when_adaptive_module_breaks(monkeypatch):
    import adaptive_liquidity as AL

    def boom(m):
        raise RuntimeError("x")
    monkeypatch.setattr(AL, "scan_floor", boom)
    assert L._scan_floor("US") == 120_000.0
    assert L.scan_gate(usd(119_999), "AAPL", {"USD": 1.0})[1] is None
    assert L.scan_gate(usd(120_000), "AAPL", {"USD": 1.0})[1] == "T4_SMALL"


# ── scan_fx / gate ────────────────────────────────────────────────────────────
def test_scan_fx_requests_base_currencies_and_drops_missing(monkeypatch):
    seen = {}

    def fake(need):
        seen["need"] = list(need)
        return {"USD": 1.0, "EUR": 0.9, "INR": 0, "GBP": 0.8}
    monkeypatch.setattr(L, "_fx_rates", fake)
    out = L.scan_fx()
    assert out == {"USD": 1.0, "EUR": 0.9, "GBP": 0.8}
    need = seen["need"]
    assert "GBp" not in need and "GBP" in need and "ILS" in need and "ZAR" in need
    assert "USD" in need and need == sorted(set(need), key=need.index)


def test_gate_caches_fx_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "_SCAN_FX", {})
    monkeypatch.setattr(L, "scan_fx", lambda: calls.append(1) or {"USD": 1.0})
    assert L.gate(usd(5_000_000), "AAPL") == (5_000_000.0, "T2_LARGE")
    assert L.gate(usd(100), "AAPL") == (100.0, None)
    assert len(calls) == 1


@pytest.mark.parametrize("ret", [{}, None])
def test_gate_empty_fx_degrades_to_usd_only(monkeypatch, ret):
    monkeypatch.setattr(L, "_SCAN_FX", {})
    monkeypatch.setattr(L, "scan_fx", lambda: ret)
    assert L.gate(usd(5_000_000), "AAPL")[1] == "T2_LARGE"
    assert L.gate(usd(5_000_000), "RELIANCE.NS") == (0.0, "UNKNOWN")


def test_gate_fx_exception_degrades_not_crashes(monkeypatch):
    def boom():
        raise RuntimeError("no net")
    monkeypatch.setattr(L, "_SCAN_FX", {})
    monkeypatch.setattr(L, "scan_fx", boom)
    assert L.gate(usd(5_000_000), "AAPL")[1] == "T2_LARGE"
    assert L._SCAN_FX == {"USD": 1.0}


def test_gate_scan_gate_exception_is_unknown(monkeypatch):
    monkeypatch.setattr(L, "_SCAN_FX", {"USD": 1.0})

    def boom(*a, **k):
        raise ValueError("x")
    monkeypatch.setattr(L, "scan_gate", boom)
    assert L.gate(usd(1), "AAPL") == (0.0, "UNKNOWN")


# ── FX cache ──────────────────────────────────────────────────────────────────
@pytest.fixture
def fxcache(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "fx_usd.json"
    monkeypatch.setattr(L, "FX_CACHE_PATH", p)
    return p


def test_fx_cache_roundtrip_filters_falsy(fxcache):
    L._fx_write_cache({"INR": 90.0, "JPY": 0, "EUR": None})
    assert L._fx_read_cache() == {"INR": 90.0}
    assert set(json.loads(fxcache.read_text())["rates"]) == {"INR"}


def test_fx_cache_write_failure_is_swallowed(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(L, "FX_CACHE_PATH", blocker / "fx.json")      # parent is a file
    L._fx_write_cache({"INR": 1.0})                                     # must not raise
    assert L._fx_read_cache() == {}


@pytest.mark.parametrize("days,fresh", [(0, True), (7, True), (8, False)])
def test_fx_cache_staleness_boundary(fxcache, days, fresh, capsys):
    fxcache.parent.mkdir(parents=True)
    ts = (dt.datetime.now() - dt.timedelta(days=days, minutes=1)).isoformat()
    fxcache.write_text(json.dumps({"as_of": ts, "rates": {"INR": "88.5"}}))
    got = L._fx_read_cache()
    assert got == ({"INR": 88.5} if fresh else {})
    assert ("refusing to use" in capsys.readouterr().out) is (not fresh)


def test_fx_cache_missing_or_corrupt_is_empty(fxcache):
    assert L._fx_read_cache() == {}
    fxcache.parent.mkdir(parents=True)
    fxcache.write_text("{not json")
    assert L._fx_read_cache() == {}


# ── fetchers (urllib faked) ───────────────────────────────────────────────────
def fake_urlopen(payload, seen=None, boom=False):
    def _open(req, timeout=None):
        if seen is not None:
            seen.append(getattr(req, "full_url", req))
        if boom:
            raise OSError("down")
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def test_erapi_returns_only_needed_truthy(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        fake_urlopen({"rates": {"INR": "90", "EUR": 0, "JPY": 150}}))
    assert L._fx_from_erapi(["INR", "EUR", "KRW"]) == {"INR": 90.0}


def test_erapi_failure_is_empty(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen({}, boom=True))
    assert L._fx_from_erapi(["INR"]) == {}


def test_frankfurter_url_and_filter(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        fake_urlopen({"rates": {"INR": 91, "EUR": 0}}, seen))
    assert L._fx_from_frankfurter(["INR", "EUR"]) == {"INR": 91.0}
    assert seen == ["https://api.frankfurter.app/latest?from=USD&to=INR,EUR"]


def test_frankfurter_failure_is_empty(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen({}, boom=True))
    assert L._fx_from_frankfurter(["INR"]) == {}


def test_fred_key_from_env_then_file(monkeypatch, tmp_path):
    monkeypatch.setattr(L.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("FRED_API_KEY", "envkey")
    assert L._fred_key() == "envkey"
    monkeypatch.delenv("FRED_API_KEY")
    (tmp_path / ".env.local").write_text("OTHER=1\nFRED_API_KEY='filekey' \n")
    assert L._fred_key() == "filekey"


def test_fred_without_key_makes_no_requests(monkeypatch):
    monkeypatch.setattr(L, "_fred_key", lambda: "")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen({}, boom=True))
    assert L._fx_from_fred(["INR"]) == {}


def test_fred_inverted_series_skips_and_failures(monkeypatch):
    monkeypatch.setattr(L, "_fred_key", lambda: "k")
    seen = []

    def _open(url, timeout=None):
        seen.append(url)
        if "DEXINUS" in url:
            return io.BytesIO(json.dumps({"observations": [
                {"value": "."}, {"value": "83.0"}]}).encode())
        if "DEXUSEU" in url:
            return io.BytesIO(json.dumps({"observations": [{"value": "1.25"}]}).encode())
        if "DEXJPUS" in url:
            return io.BytesIO(json.dumps({"observations": [{"value": "."}]}).encode())
        raise OSError("boom")                                  # KRW
    monkeypatch.setattr(urllib.request, "urlopen", _open)
    out = L._fx_from_fred(["INR", "EUR", "JPY", "KRW", "XXX"])
    assert out == {"INR": 83.0, "EUR": 1 / 1.25}
    assert len(seen) == 4 and "series_id=DEXINUS" in seen[0]      # XXX never requested


# ── _fx_rates chain ───────────────────────────────────────────────────────────
@pytest.fixture
def nofetch(monkeypatch, fxcache):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
        download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))))
    for n in ("_fx_from_erapi", "_fx_from_frankfurter", "_fx_from_fred"):
        monkeypatch.setattr(L, n, lambda need: {})


def test_fx_rates_usd_only_needs_no_fetch(nofetch, monkeypatch):
    monkeypatch.setattr(L, "_fx_from_erapi", lambda n: pytest.fail("called"))
    assert L._fx_rates(["USD"]) == {"USD": 1.0}


def test_fx_rates_stops_at_first_complete_source(nofetch, monkeypatch, capsys):
    monkeypatch.setattr(L, "_fx_from_erapi", lambda need: {"INR": 90.0, "EUR": 0.9})
    monkeypatch.setattr(L, "_fx_from_frankfurter", lambda n: pytest.fail("called"))
    assert L._fx_rates(["INR", "EUR", "USD"]) == {"USD": 1.0, "INR": 90.0, "EUR": 0.9}
    assert "fx: 2/2 rates from er-api:2" in capsys.readouterr().out


def test_fx_rates_falls_through_only_for_missing(nofetch, monkeypatch):
    got = {}
    monkeypatch.setattr(L, "_fx_from_erapi", lambda need: {"INR": 90.0})
    monkeypatch.setattr(L, "_fx_from_frankfurter", lambda need: got.setdefault("f", need) and {"EUR": 0.9})
    out = L._fx_rates(["INR", "EUR"])
    assert got["f"] == ["EUR"] and out["EUR"] == 0.9 and out["INR"] == 90.0


def test_fx_rates_persists_live_rates_and_merges_old(nofetch, monkeypatch, fxcache):
    L._fx_write_cache({"JPY": 150.0})
    monkeypatch.setattr(L, "_fx_from_erapi", lambda need: {"INR": 90.0})
    L._fx_rates(["INR"])
    assert json.loads(fxcache.read_text())["rates"] == {"JPY": 150.0, "INR": 90.0}


def test_fx_rates_serves_stale_failure_from_cache(nofetch, capsys):
    L._fx_write_cache({"INR": 95.0})
    out = L._fx_rates(["INR"])
    assert out == {"USD": 1.0, "INR": 95.0}
    assert "1 rate(s) served from cache (INR)" in capsys.readouterr().out


def test_fx_rates_warns_loudly_when_nothing_available(nofetch, capsys):
    out = L._fx_rates(["INR", "JPY"])
    assert out == {"USD": 1.0}
    assert "NO rate for INR, JPY" in capsys.readouterr().out


def test_fx_rates_yfinance_last_resort(nofetch, monkeypatch, capsys):
    close = pd.DataFrame({"INR=X": [89.0, 90.0, np.nan], "JPY=X": [0.0, 0.0, 0.0]})
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
        download=lambda tick, **k: {"Close": close}))
    out = L._fx_rates(["INR", "JPY", "EUR"])
    assert out["INR"] == 90.0 and "JPY" not in out and "EUR" not in out
    assert "yfinance:1" in capsys.readouterr().out


# ── build_index / index readers ───────────────────────────────────────────────
@pytest.fixture
def index_env(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "INDEX_PATH", tmp_path / "liq.parquet")
    monkeypatch.setattr(L, "_CACHE_MAP", {})
    return tmp_path


def seeds(n_ok=25):
    ok = frame(np.full(n_ok, 10.0), np.full(n_ok, 1000.0))
    return {
        "US": {"OK": ok, "SHORT": ok.head(19), "NOVOL": pd.DataFrame({"Close": [1.0] * 30}),
               "NONE": None, "ZERO": frame([1.0] * 30, [0.0] * 30)},
        "UK": {"PENCE": frame(np.full(25, 100.0), np.full(25, 100.0))},
        "IN": {"RELI": ok},
    }


def test_build_index(index_env, monkeypatch):
    data = seeds()
    kit = types.SimpleNamespace(MARKETS=["US", "UK", "IN", "XX"], load=lambda m: data.get(m, {}))
    monkeypatch.setitem(sys.modules, "screener_kit", kit)
    monkeypatch.setattr(L, "_fx_rates", lambda c: {"USD": 1.0, "GBP": 0.5, "INR": 0})
    idx = L.build_index(verbose=False)
    got = idx.set_index("Symbol")
    assert list(idx["Symbol"]) == ["OK", "PENCE"]                    # IN has no rate; others dropped
    assert got.loc["OK", "turnover_usd"] == 10_000.0 and got.loc["OK", "ltp"] == 10.0
    assert got.loc["PENCE", "turnover_usd"] == pytest.approx(10_000 / 100 / 0.5)
    assert got.loc["PENCE", "Market"] == "UK"
    assert pd.read_parquet(L.INDEX_PATH).shape[0] == 2


def test_build_index_tail20_median(index_env, monkeypatch):
    vol = np.concatenate([np.full(10, 1e9), np.full(20, 100.0)])
    data = {"US": {"A": frame(np.ones(30), vol)}}
    monkeypatch.setitem(sys.modules, "screener_kit",
                        types.SimpleNamespace(MARKETS=["US"], load=lambda m: data[m]))
    monkeypatch.setattr(L, "_fx_rates", lambda c: {"USD": 1.0})
    assert L.build_index(verbose=False)["turnover_usd"].iloc[0] == 100.0


def test_build_index_empty_writes_nothing_and_verbose_prints(index_env, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "screener_kit",
                        types.SimpleNamespace(MARKETS=["US"], load=lambda m: {}))
    monkeypatch.setattr(L, "_fx_rates", lambda c: {"USD": 1.0})
    assert L.build_index(verbose=True).empty
    assert not L.INDEX_PATH.exists()
    out = capsys.readouterr().out
    assert "US: indexed 0 stocks" in out and "liquidity index: 0 symbols" in out


def write_index(rows):
    pd.DataFrame(rows).to_parquet(L.INDEX_PATH, index=False)


ROWS = [
    {"Symbol": "A", "Market": "US", "turnover_usd": 50e6, "ltp": 1.0},
    {"Symbol": "B", "Market": "US", "turnover_usd": 5e6, "ltp": 1.0},
    {"Symbol": "C", "Market": "IN", "turnover_usd": 2e6, "ltp": 1.0},
    {"Symbol": "D", "Market": "IN", "turnover_usd": 1e5, "ltp": 1.0},
]


def test_load_index_missing_is_empty(index_env):
    assert L._load_index().empty and L.liquid_symbols() == []
    assert L._turnover_map() == {}


@pytest.mark.parametrize("v,mkt,expected", [
    (None, "US", "Unknown"),
    (20_000_000, "US", "High"), (19_999_999, "US", "Medium"),
    (2_000_000, "US", "Medium"), (1_999_999, "US", "Low"),
    (10_000_000, None, "High"), (9_999_999, None, "Medium"),
    (1_000_000, "ZZ", "Medium"), (999_999, "ZZ", "Low"),
    (5_000_000, "IN", "High"), (500_000, "IN", "Medium"), (499_999, "IN", "Low"),
])
def test_tier(v, mkt, expected):
    assert L.tier(v, mkt) == expected


def test_liquid_symbols_filters_sorts_and_limits(index_env):
    write_index(ROWS)
    assert L.liquid_symbols() == ["A", "B", "C"]                       # >= 1M default, desc
    assert L.liquid_symbols("IN") == ["C"]
    assert L.liquid_symbols(min_usd=5e6) == ["A", "B"]                 # inclusive bound
    assert L.liquid_symbols(min_usd=0, top=2) == ["A", "B"]
    assert L.liquid_symbols(market="IN", min_usd=0) == ["C", "D"]


def test_turnover_lookup(index_env):
    write_index(ROWS)
    assert L.turnover("B") == 5e6
    assert L.turnover("NOPE") is None


def test_annotate_with_and_without_market(index_env):
    write_index(ROWS)
    df = pd.DataFrame({"Symbol": ["A", "D", "Q"], "Market": ["US", "IN", "US"]})
    out = L.annotate(df)
    assert list(out["Liquidity"])[:2] == ["High", "Low"]
    assert out["Turnover_USD"].iloc[0] == 50e6 and pd.isna(out["Turnover_USD"].iloc[2])
    assert "Turnover_USD" not in df.columns                              # input not mutated
    nm = L.annotate(pd.DataFrame({"Symbol": ["B", "C"]}))
    assert list(nm["Liquidity"]) == ["Medium", "Medium"]


def test_annotate_passthrough_cases(index_env):
    write_index(ROWS)
    assert L.annotate(None) is None
    e = pd.DataFrame({"Symbol": []})
    assert L.annotate(e) is e
    nosym = pd.DataFrame({"X": [1]})
    assert L.annotate(nosym) is nosym
    named = L.annotate(pd.DataFrame({"Name": ["A"]}), symbol_col="Name")
    assert list(named["Liquidity"]) == ["High"]
