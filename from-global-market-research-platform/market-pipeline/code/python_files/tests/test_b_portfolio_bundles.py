"""Offline tests for portfolio_bundles (correlation clustering, inverse-vol weights, valuation)."""

import json

import numpy as np
import pandas as pd
import pytest

import portfolio_bundles as pb

N = 120
IDX = pd.bdate_range("2025-01-01", periods=200)


@pytest.fixture(autouse=True)
def stores(monkeypatch, tmp_path):
    monkeypatch.setattr(pb, "STORE", tmp_path / "cache_seed" / "portfolio_bundles.json")
    monkeypatch.setattr(pb, "MARK", tmp_path / "state" / "last_build")
    return tmp_path


def _prices(rets, start=100.0):
    return start * np.cumprod(1 + np.asarray(rets))


def _frame(prices, date_col=None):
    df = pd.DataFrame({"Close": prices}, index=IDX[: len(prices)])
    if date_col:
        df = df.reset_index(drop=True)
        df[date_col] = IDX[: len(prices)]
    return df


def _universe(mkt="IN", groups=((("A", 5), 0.010), (("B", 5), 0.010)), noise=0.002, seed=3):
    """Two independent factors -> two tight clusters of 5 names each."""
    rng = np.random.default_rng(seed)
    rows, frames = [], {}
    for (tag, n), fvol in groups:
        f = rng.normal(0.0005, fvol, N)
        for i in range(n):
            sym = f"{tag}{i}"
            r = f + rng.normal(0, noise * (1 + i), N)  # different idio vol per name
            frames[(mkt, sym)] = _frame(_prices(r))
            rows.append(
                {
                    "market": mkt,
                    "symbol": sym,
                    "zone": "BUY",
                    "sector": f"Sec{tag}",
                    "why": "screen " + sym,
                }
            )
    return rows, frames


# ---------------------------------------------------------------- _closes
def test_closes_requires_min_bars_and_datetime_index():
    fr = {("IN", "X"): _frame(_prices([0.001] * 59)), ("IN", "Y"): _frame(_prices([0.001] * 60))}
    assert pb._closes(fr, {"market": "IN", "symbol": "X"}) is None  # 59 < MIN_BARS
    assert len(pb._closes(fr, {"market": "IN", "symbol": "Y"})) == 60
    assert pb._closes(fr, {"market": "IN", "symbol": "NOPE"}) is None
    plain = pd.DataFrame({"Close": [1.0] * 100})  # RangeIndex, no date column
    assert pb._closes({("IN", "Z"): plain}, {"market": "IN", "symbol": "Z"}) is None


@pytest.mark.parametrize("col", ["Date", "date", "price_date"])
def test_closes_accepts_date_columns(col):
    fr = {("IN", "X"): _frame(_prices([0.001] * 80), date_col=col)}
    s = pb._closes(fr, {"market": "IN", "symbol": "X"})
    assert isinstance(s.index, pd.DatetimeIndex) and len(s) == 80


def test_closes_tail_dedupe_and_coercion():
    prices = list(_prices([0.001] * 150))
    df = _frame(prices)
    s = pb._closes({("IN", "X"): df}, {"market": "IN", "symbol": "X"})
    assert len(s) == pb.CORR_BARS + 10  # tail(bars)
    assert len(pb._closes({("IN", "X"): df}, {"market": "IN", "symbol": "X"}, bars=70)) == 70
    dup = _frame(prices[:80])
    dup = pd.concat([dup, dup.iloc[[-1]].assign(Close=999.0)])
    out = pb._closes({("IN", "X"): dup}, {"market": "IN", "symbol": "X"})
    assert out.iloc[-1] == 999.0 and out.index.is_unique  # keep last of duplicate dates
    bad = _frame(prices[:80]).astype(object)
    bad.iloc[0, 0] = "n/a"
    assert len(pb._closes({("IN", "X"): bad}, {"market": "IN", "symbol": "X"})) == 79


def test_closes_unparseable_dates_dropped():
    df = pd.DataFrame({"Date": ["bad"] * 3 + list(IDX[:70]), "Close": [1.0] * 73})
    s = pb._closes({("IN", "X"): df}, {"market": "IN", "symbol": "X"})
    assert len(s) == 70


# ---------------------------------------------------------------- _bundle_name
def test_bundle_name_top_two_sectors_and_defaults():
    mem = [{"sector": "Energy"}] * 3 + [{"sector": "Banks"}] * 2 + [{"sector": "IT"}]
    assert pb._bundle_name(mem, "IN") == "IN · Energy + Banks"
    assert pb._bundle_name([{"sector": "Unclassified"}, {"sector": ""}, {}], "US") == "US · Mixed"
    assert pb._bundle_name([{"sector": "IT"}, {"sector": None}], "JP") == "JP · IT"


# ---------------------------------------------------------------- build
def test_build_forms_two_tight_clusters_with_capped_inverse_vol_weights():
    rows, frames = _universe()
    store = pb.build(rows, frames)
    assert store["built"] == pd.Timestamp.today().strftime("%Y-%m-%d")
    assert len(store["bundles"]) == 2
    for b in store["bundles"]:
        syms = {m["symbol"] for m in b["members"]}
        assert syms in ({f"A{i}" for i in range(5)}, {f"B{i}" for i in range(5)})
        assert b["market"] == "IN" and b["formed"] == store["built"]
        assert b["intra_corr"] >= pb.MIN_INTRA_CORR and b["id"].startswith("IN-")
        ws = [m["weight"] for m in b["members"]]
        assert ws == sorted(ws, reverse=True)  # sorted by weight desc
        assert sum(ws) == pytest.approx(1.0, abs=1e-3)
        assert max(ws) <= 0.30  # similar vols -> near-equal weights
        assert b["name"] in ("IN · SecA", "IN · SecB")
        m0 = b["members"][0]
        assert m0["zone"] == "BUY" and m0["screen"].startswith("screen ") and m0["px_formation"] > 0
        assert "inverse-vol weight" in m0["rationale"]
    # the quietest name (lowest idiosyncratic vol) carries the largest weight
    a = next(b for b in store["bundles"] if b["name"].endswith("SecA"))
    assert a["members"][0]["symbol"] == "A0"


def test_build_persists_store_and_marker_and_load_roundtrips():
    rows, frames = _universe()
    store = pb.build(rows, frames)
    assert json.loads(pb.STORE.read_text()) == store
    assert pb.MARK.read_text() == store["built"][:7]
    assert pb.load() == store


def test_build_filters_universe_by_zone_floor_missing_and_bars():
    rows, frames = _universe(groups=((("A", 5), 0.010),))
    rows[0]["zone"] = "SELL"
    store = pb.build(rows, frames)  # 4 BUY/HOLD left -> exactly MIN_CLUSTER, still bundles
    assert len(store["bundles"]) == 1 and len(store["bundles"][0]["members"]) == 4
    rows[1]["missing"] = True
    assert pb.build(rows, frames)["bundles"] == []  # 3 left < MIN_CLUSTER
    rows, frames = _universe(groups=((("A", 5), 0.010),))
    rows[0]["below_floor"] = True
    assert len(pb.build(rows, frames)["bundles"][0]["members"]) == 4
    rows[1]["zone"] = "HOLD"  # HOLD is allowed
    assert len(pb.build(rows, frames)["bundles"][0]["members"]) == 4
    rows, frames = _universe(groups=((("A", 5), 0.010),))
    frames[("IN", "A0")] = _frame(_prices([0.001] * 30))  # too short
    assert len(pb.build(rows, frames)["bundles"][0]["members"]) == 4


def test_build_ignores_other_markets_and_unknown_markets():
    rows, frames = _universe(mkt="IN")
    rows2, frames2 = _universe(mkt="US", groups=((("C", 5), 0.010),))
    rows3, frames3 = _universe(mkt="ZZ", groups=((("D", 5), 0.010),))
    store = pb.build(rows + rows2 + rows3, {**frames, **frames2, **frames3})
    assert {b["market"] for b in store["bundles"]} == {"IN", "US"}  # ZZ is not a bundled market


def test_build_caps_oversized_clusters_at_max_cluster():
    rows, frames = _universe(groups=((("A", 13), 0.010),), noise=0.001)
    store = pb.build(rows, frames)
    assert pb.MAX_CLUSTER == 10
    assert [len(b["members"]) for b in store["bundles"]] == [10]


def test_build_drops_uncorrelated_names_and_small_clusters():
    rng = np.random.default_rng(7)
    rows, frames = [], {}
    for i in range(6):  # independent random walks -> no cluster reaches 4 tight names
        sym = f"R{i}"
        frames[("IN", sym)] = _frame(_prices(rng.normal(0, 0.01, N)))
        rows.append({"market": "IN", "symbol": sym, "zone": "BUY", "sector": "S", "why": ""})
    assert pb.build(rows, frames)["bundles"] == []


def test_build_weight_cap_is_applied_then_renormalised():
    rows, frames = _universe(groups=((("A", 4), 0.010),), noise=0.0005)
    f = np.random.default_rng(3).normal(0.0005, 0.010, N)
    # A0 almost riskless -> pure inverse-vol would hand it nearly everything
    frames[("IN", "A0")] = _frame(_prices(0.05 * f + np.random.default_rng(1).normal(0, 1e-5, N)))
    store = pb.build(rows, frames)
    got = {m["symbol"]: m["weight"] for m in store["bundles"][0]["members"]}
    px = pd.DataFrame({s: frames[("IN", s)]["Close"] for s in got}).tail(pb.CORR_BARS)
    vol = px.pct_change(fill_method=None).tail(60).std()
    iv = 1.0 / vol
    uncapped = iv / iv.sum()
    capped = uncapped.clip(upper=pb.WEIGHT_CAP)
    expected = capped / capped.sum()
    assert uncapped.max() > 0.5  # the premise: without the cap A0 would dominate
    for sym, w in got.items():
        assert w == pytest.approx(expected[sym], abs=1e-4)
    assert max(got.values()) < uncapped.max() - 0.05


# ---------------------------------------------------------------- maybe_build
def test_maybe_build_builds_when_no_store_then_reuses_within_month(monkeypatch):
    calls = []
    sentinel = {"built": "x", "bundles": [{"id": "b"}]}
    monkeypatch.setattr(pb, "build", lambda rows, frames: calls.append(1) or sentinel)
    assert pb.maybe_build([], {}) is sentinel
    pb.STORE.parent.mkdir(parents=True, exist_ok=True)
    pb.STORE.write_text(json.dumps({"bundles": [{"id": "old"}]}))
    pb.MARK.parent.mkdir(parents=True, exist_ok=True)
    pb.MARK.write_text(pd.Timestamp.today().strftime("%Y-%m"))
    assert pb.maybe_build([], {}) == {"bundles": [{"id": "old"}]}
    assert calls == [1]  # not rebuilt this month


def test_maybe_build_rebuilds_on_new_month_and_survives_failures(monkeypatch):
    pb.STORE.parent.mkdir(parents=True, exist_ok=True)
    pb.STORE.write_text(json.dumps({"bundles": [{"id": "old"}]}))
    pb.MARK.parent.mkdir(parents=True, exist_ok=True)
    pb.MARK.write_text("1999-01")
    monkeypatch.setattr(pb, "build", lambda r, f: {"bundles": [{"id": "new"}]})
    assert pb.maybe_build([], {}) == {"bundles": [{"id": "new"}]}

    def boom(r, f):
        raise RuntimeError("x")

    monkeypatch.setattr(pb, "build", boom)
    assert pb.maybe_build([], {}) == {
        "bundles": [{"id": "old"}]
    }  # falls back to the standing store
    pb.STORE.write_text("{bad")
    assert pb.maybe_build([], {}) == {"bundles": []}


def test_load_missing_or_corrupt():
    assert pb.load() is None
    pb.STORE.parent.mkdir(parents=True, exist_ok=True)
    pb.STORE.write_text("{bad")
    assert pb.load() is None


# ---------------------------------------------------------------- _bench_since / value
def _flat_universe(n=10, mkt="IN", end_mult=1.10):
    """n names that gain exactly end_mult-1 from the formation date to the last bar."""
    rows, frames = [], {}
    for i in range(n):
        p = np.full(N, 100.0)
        p[90:] = 100.0 * end_mult
        frames[(mkt, f"S{i}")] = _frame(p)
        rows.append({"market": mkt, "symbol": f"S{i}", "zone": "BUY"})
    return rows, frames


def test_bench_since_equal_weight_mean_and_minimum_names():
    rows, frames = _flat_universe(10)
    since = str(IDX[89].date())  # last close at/before formation is bar 89 (100.0)
    assert pb._bench_since(rows, frames, "IN", since) == pytest.approx(10.0)
    assert pb._bench_since(rows[:9], frames, "IN", since) is None  # needs >= 10 names


def test_bench_since_skips_missing_other_market_and_short_series_and_future_formation():
    rows, frames = _flat_universe(10)
    since = str(IDX[89].date())
    rows[0]["missing"] = True
    assert pb._bench_since(rows, frames, "IN", since) is None  # 9 usable
    rows[0].pop("missing")
    assert pb._bench_since(rows, frames, "US", since) is None
    # formation before the series starts -> no anchor bar -> None
    assert pb._bench_since(rows, frames, "IN", "2000-01-01") is None


def test_bench_since_anchor_is_last_close_at_or_before_formation():
    rows, frames = _flat_universe(10)
    on_gain_day = str(IDX[90].date())  # bar 90 already carries the jump -> anchor 110 -> 0%
    assert pb._bench_since(rows, frames, "IN", on_gain_day) == pytest.approx(0.0)


def _store_with(members, formed, mkt="IN", name="B"):
    return {
        "bundles": [{"id": "x", "name": name, "market": mkt, "formed": formed, "members": members}]
    }


def _members(syms, px_form, weights, mkt="IN"):
    return [
        {"symbol": s, "market": mkt, "weight": w, "px_formation": px_form}
        for s, w in zip(syms, weights)
    ]


def test_value_weighted_returns_drift_and_alpha():
    rows, frames = _flat_universe(10, end_mult=1.10)
    # replace two members' frames so their since-formation returns differ
    frames[("IN", "S0")] = _frame(np.r_[np.full(90, 100.0), np.full(30, 120.0)])  # +20% vs px 100
    frames[("IN", "S1")] = _frame(np.r_[np.full(90, 100.0), np.full(30, 80.0)])  # -20%
    members = _members(["S0", "S1", "S2", "S3"], 100.0, [0.4, 0.3, 0.2, 0.1])
    out = pb.value(_store_with(members, str(IDX[89].date())), rows, frames)
    assert len(out) == 1
    v = out[0]
    # since = 0.4*20 + 0.3*(-20) + 0.2*10 + 0.1*10
    assert v["since"] == pytest.approx(0.4 * 20 - 0.3 * 20 + 0.2 * 10 + 0.1 * 10)
    bench = np.mean([20, -20] + [10] * 8)
    assert v["alpha"] == pytest.approx(v["since"] - bench)
    assert v["d1"] == pytest.approx(0.0)  # last two bars equal
    cur = [0.4 * 1.2, 0.3 * 0.8, 0.2 * 1.1, 0.1 * 1.1]
    tot = sum(cur)
    assert v["drift"] == pytest.approx(
        max(abs(c / tot - w) for c, w in zip(cur, [0.4, 0.3, 0.2, 0.1])) * 100
    )
    assert v["sell_now"] == 0 and len(v["members"]) == 4


def test_value_counts_sell_zone_members_and_d1():
    rows, frames = _flat_universe(10)
    rows[0]["zone"] = "SELL"
    p = np.full(N, 100.0)
    p[90:] = 110.0
    p[-1] = 121.0  # last day +10%
    frames[("IN", "S2")] = _frame(p)
    members = _members(["S0", "S1", "S2", "S3"], 100.0, [0.25] * 4)
    v = pb.value(_store_with(members, str(IDX[89].date())), rows, frames)[0]
    assert v["sell_now"] == 1
    assert v["d1"] == pytest.approx(0.25 * 10.0)


def test_value_skips_members_without_data_and_small_bundles():
    rows, frames = _flat_universe(10)
    members = _members(["S0", "S1", "S2", "NOPE"], 100.0, [0.25] * 4)
    out = pb.value(_store_with(members, str(IDX[89].date())), rows, frames)
    assert len(out[0]["members"]) == 3  # MIN_CLUSTER - 1 = 3 is enough
    members = _members(["S0", "S1", "NOPE", "NOPE2"], 100.0, [0.25] * 4)
    assert pb.value(_store_with(members, str(IDX[89].date())), rows, frames) == []
    zero = _members(["S0", "S1", "S2"], 0, [0.3] * 3)
    assert pb.value(_store_with(zero, str(IDX[89].date())), rows, frames) == []  # no formation px


def test_value_uses_frames_for_members_no_longer_in_todays_rows():
    rows, frames = _flat_universe(10)
    members = _members(["S0", "S1", "S2", "S3"], 100.0, [0.25] * 4)
    out = pb.value(_store_with(members, str(IDX[89].date())), rows[4:], frames)
    assert out and out[0]["members"][0]["zone_now"] == "?"


def test_value_alpha_none_without_benchmark_and_ranking():
    rows, frames = _flat_universe(4)  # < 10 names -> no benchmark
    members = _members(["S0", "S1", "S2", "S3"], 100.0, [0.25] * 4)
    assert pb.value(_store_with(members, "2025-01-01"), rows, frames)[0]["alpha"] is None
    rows, frames = _flat_universe(10)
    frames[("IN", "S0")] = _frame(np.r_[np.full(90, 100.0), np.full(30, 130.0)])
    frames[("IN", "S1")] = _frame(np.r_[np.full(90, 100.0), np.full(30, 60.0)])
    st = {
        "bundles": [
            {
                "id": "lo",
                "name": "lo",
                "market": "IN",
                "formed": str(IDX[89].date()),
                "members": _members(["S1", "S2", "S3", "S4"], 100.0, [0.25] * 4),
            },
            {
                "id": "hi",
                "name": "hi",
                "market": "IN",
                "formed": str(IDX[89].date()),
                "members": _members(["S0", "S2", "S3", "S4"], 100.0, [0.25] * 4),
            },
        ]
    }
    out = pb.value(st, rows, frames)
    assert [o["bundle"]["id"] for o in out] == ["hi", "lo"]  # best alpha first


def test_value_empty_store():
    assert pb.value({}, [], {}) == []


def test_build_keeps_names_with_some_missing_history_but_drops_mostly_empty_ones():
    rows, frames = _universe(groups=((("A", 5), 0.010),))
    # A0 is missing its first 25 bars of the 90-bar window: 65/89 returns remain (>= 60%)
    f = frames[("IN", "A0")].copy()
    f.iloc[: N - pb.CORR_BARS + 25] = np.nan
    frames[("IN", "A0")] = f
    store = pb.build(rows, frames)
    assert "A0" in {m["symbol"] for m in store["bundles"][0]["members"]}
    # a name present for only the last 20 bars (< 60% of returns) is dropped from correlation
    g = frames[("IN", "A1")].copy()
    g.iloc[: N - 20] = np.nan
    frames[("IN", "A1")] = g
    store = pb.build(rows, frames)
    assert "A1" not in {m["symbol"] for m in store["bundles"][0]["members"]}
