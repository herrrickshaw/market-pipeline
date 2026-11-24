"""Offline tests for custom_indices (thematic / band / cluster / peer index construction, build and status)."""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

COST = 100.0 / 10_000


@pytest.fixture
def ci(monkeypatch):
    pgc = types.ModuleType("pg_client")
    pgc.up = True
    pgc.calls = []
    pgc.connect = lambda *a, **k: pgc.up
    pgc.ensure_schema = lambda stmts: pgc.calls.append(("ensure", list(stmts)))
    pgc.to_rows = lambda df, cols: [tuple(r) for r in df[list(cols)].itertuples(index=False, name=None)]
    pgc.delete_and_insert = lambda *a, **k: pgc.calls.append(("dai", a)) or len(a[4])
    monkeypatch.setitem(sys.modules, "pg_client", pgc)
    sys.modules.pop("custom_indices", None)
    mod = importlib.import_module("custom_indices")
    monkeypatch.setattr(mod, "SPECS", dict(mod.SPECS))  # builders add codes to the global; keep tests isolated
    mod.pgc = pgc
    yield mod
    sys.modules.pop("custom_indices", None)


def _panel(n_days=1500, n_syms=90, groups=9, seed=0, names=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    names = names or [f"S{i:03d}" for i in range(n_syms)]
    fac = rng.normal(0.0004, 0.01, (n_days, groups))
    rets = np.column_stack([0.95 * fac[:, i % groups] + 0.05 * rng.normal(0, 0.01, n_days)
                            for i in range(len(names))])
    px = pd.DataFrame(100.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    tv = pd.DataFrame(np.tile(np.linspace(1000, 100, len(names)), (n_days, 1)), index=idx, columns=names)
    return px, tv


def _inputs(px, tv):
    rets = px.pct_change(fill_method=None)
    return px, tv, rets, px.index


# ------------------------------------------------------------------ constants
def test_constants_and_ddl(ci):
    assert (ci.BAND_LO, ci.BAND_HI, ci.REBAL, ci.COST_BPS, ci.BASE, ci.WARMUP) == (200, 450, 126, 100.0, 1000.0, 260)
    assert set(ci.SPECS) == {"SMLSCR", "SMLEXC", "SMLEW"}
    assert 'CREATE TABLE IF NOT EXISTS "indices".custom_daily' in ci.DDL
    assert "custom_daily_pk" in ci.DDL and all(len(v[1]) >= 5 for v in ci.THEMES.values())
    assert ci.BANDS["INR1K"][1:] == (250, 1250)


# ------------------------------------------------------------------ build_themes
def test_build_themes_levels_costs_and_membership(ci, capsys):
    names = ["BALRAMCHIN", "TRIVENI", "PRAJIND", "DWARKESH", "BAJAJHIND", "HAL", "BEL", "BDL"]  # ethanol(5) + defence(3)
    px, tv = _panel(n_days=700, names=names, groups=2)
    px, tv, rets, dates = _inputs(px, tv)
    out = ci.build_themes(px, tv, rets, dates, COST)
    assert set(out) == {"ETHANOL"}  # defence has only 3 of its symbols -> skipped
    printed = capsys.readouterr().out
    assert "DEFENCE: only 3 symbols present — skipped" in printed and "ETHANOL" in printed and " 5/12 symbols" in printed
    pts = out["ETHANOL"]
    live = ["BALRAMCHIN", "TRIVENI", "PRAJIND", "DWARKESH", "BAJAJHIND"]
    i = ci.WARMUP
    first_d, first_lv, first_n = pts[0]
    assert first_d == dates[i + 1] and first_n == 5
    expected = 1000.0 * (1 - COST) * (1 + rets.loc[dates[i + 1], live].mean())
    assert first_lv == pytest.approx(expected)
    # the second rebalance block charges the cost again on the carried level
    j = i + ci.REBAL
    prev = [p for p in pts if p[0] == dates[j]][0][1]
    nxt = [p for p in pts if p[0] == dates[j + 1]][0]
    assert nxt[1] == pytest.approx(prev * (1 - COST) * (1 + rets.loc[dates[j + 1], live].mean()))
    assert pts[-1][0] == dates[-1]


def test_build_themes_requires_three_live_names_per_block(ci):
    names = ["BALRAMCHIN", "TRIVENI", "PRAJIND", "DWARKESH"]
    px, tv = _panel(n_days=700, names=names, groups=2)
    px.loc[px.index[:300], ["PRAJIND", "DWARKESH"]] = np.nan  # only 2 live at the first rebalance
    px, tv, rets, dates = _inputs(px, tv)
    pts = ci.build_themes(px, tv, rets, dates, COST)["ETHANOL"]
    j = ci.WARMUP + ci.REBAL
    assert pts[0][0] == dates[j + 1]  # the first block (2 live names) was skipped entirely
    assert all(n >= 3 for _, _, n in pts)
    # a theme with fewer than 4 names present is dropped; exactly 4 is kept
    px2, tv2 = _panel(n_days=700, names=names[:3], groups=2)
    assert ci.build_themes(*_inputs(px2, tv2), COST) == {}


# ------------------------------------------------------------------ build_bands
def test_build_bands_ranks_by_turnover(ci):
    px, tv = _panel(n_days=700, n_syms=60, groups=3)
    px, tv, rets, dates = _inputs(px, tv)
    out = ci.build_bands(px, tv, rets, dates, COST)
    assert set(out) == {"INL250", "INL350", "INL650"}  # INR1K (ranks 251+) has no members
    assert ci.SPECS["INL250"].startswith("India Liquid 250") and "INR1K" not in ci.SPECS
    d0, lv0, n0 = out["INL250"][0]
    assert d0 == dates[ci.WARMUP + 1] and n0 == 60
    assert lv0 == pytest.approx(1000.0 * (1 - COST) * (1 + rets.loc[dates[ci.WARMUP + 1]].mean()))


def test_build_bands_needs_thirty_members(ci):
    px, tv = _panel(n_days=700, n_syms=29, groups=3)
    assert ci.build_bands(*_inputs(px, tv), COST) == {}
    px, tv = _panel(n_days=700, n_syms=30, groups=3)
    assert set(ci.build_bands(*_inputs(px, tv), COST)) == {"INL250", "INL350", "INL650"}


def test_build_bands_member_order_follows_turnover(ci):
    px, tv = _panel(n_days=700, n_syms=40, groups=4)
    # make the lowest-numbered symbols the least liquid: reverse turnover
    tv = pd.DataFrame(np.tile(np.linspace(100, 1000, 40), (len(tv), 1)), index=tv.index, columns=tv.columns)
    px, tv, rets, dates = _inputs(px, tv)
    out = ci.build_bands(px, tv, rets, dates, COST)
    assert out["INL250"][0][2] == 40  # all names are in the top 250 regardless of order


# ------------------------------------------------------------------ build_clusters
def _install_sectors(monkeypatch, names, groups):
    mod = types.ModuleType("cluster_indices")
    ind = pd.Series({n: f"Grp{i % groups}" for i, n in enumerate(names)})
    mod.sectors = lambda: ind
    monkeypatch.setitem(sys.modules, "cluster_indices", mod)
    return ind


def test_build_clusters_groups_by_correlation(ci, monkeypatch):
    px, tv = _panel(n_days=1500, n_syms=90, groups=9)
    _install_sectors(monkeypatch, list(px.columns), 9)
    monkeypatch.setattr(ci, "CLUSTER_K", 9)
    out = ci.build_clusters(*_inputs(px, tv), COST)
    assert out and all(code.startswith("CL_GRP") for code in out)
    assert len(out) == 9
    for code, pts in out.items():
        assert len(pts) > 500  # series shorter than 500 observations are discarded
        assert all(n == 10 for _, _, n in pts)  # nine factor groups of ten names each
        assert ci.SPECS[code].startswith("[cluster] mostly Grp")
        dates_ = [d for d, _, _ in pts]
        assert dates_ == sorted(dates_)


def test_build_clusters_small_universe_gives_nothing(ci, monkeypatch):
    px, tv = _panel(n_days=1500, n_syms=60, groups=6)  # < 80 names after screening
    _install_sectors(monkeypatch, list(px.columns), 6)
    assert ci.build_clusters(*_inputs(px, tv), COST) == {}


def test_build_clusters_drops_symbols_without_sector(ci, monkeypatch):
    px, tv = _panel(n_days=1500, n_syms=90, groups=9)
    ind = _install_sectors(monkeypatch, list(px.columns), 9)
    monkeypatch.setattr(sys.modules["cluster_indices"], "sectors", lambda: ind.iloc[:70])  # 20 names unclassified
    assert ci.build_clusters(*_inputs(px, tv), COST) == {}  # 70 < 80


# ------------------------------------------------------------------ build_peers
def test_build_peers_recurring_groups(ci, monkeypatch):
    px, tv = _panel(n_days=1700, n_syms=90, groups=9)
    monkeypatch.setattr(ci, "CLUSTER_K", 9)
    out = ci.build_peers(*_inputs(px, tv), COST)
    assert out and all(code.startswith("PEER") for code in out)
    for code, pts in out.items():
        assert len(pts) > 500 and all(n >= ci.PEER_MIN_GROUP for _, _, n in pts)
        assert ci.SPECS[code].startswith("[recurring peers] stable group")
    first = min(d for pts in out.values() for d, _, _ in pts)
    # groups only start after >= 4 formations of co-occurrence history
    assert first > px.index[ci.WARMUP + 4 * ci.REBAL]


def test_build_peers_small_universe_gives_nothing(ci):
    px, tv = _panel(n_days=1700, n_syms=60, groups=6)
    assert ci.build_peers(*_inputs(px, tv), COST) == {}


# ------------------------------------------------------------------ build_global
def _long(prices):
    rows = [{"Date": d, "Symbol": s, "Close": v} for s, ser in prices.items() for d, v in ser.items()]
    return pd.DataFrame(rows)


def test_build_global_themes_per_market(ci, monkeypatch, capsys):
    idx = pd.bdate_range("2016-01-04", periods=700)
    rng = np.random.default_rng(1)

    def ser():
        return pd.Series(50 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx))), index=idx)

    us = _long({s: ser() for s in ["CCJ", "LEU", "BWXT", "SMR", "OKLO", "ETN", "PWR", "NVT", "HUBB", "AMAT", "LRCX"]})
    seen = []

    def load_market(m):
        seen.append(m)
        return us if m == "US" else pd.DataFrame({"Date": [], "Symbol": [], "Close": []})

    mod = types.ModuleType("backtest_tier_anomalies")
    mod.load_market = load_market
    monkeypatch.setitem(sys.modules, "backtest_tier_anomalies", mod)
    out = ci.build_global(COST)
    assert seen == ["US", "JP", "KR", "EU"]
    assert set(out) == {"NUCLEAR", "GRID"}  # SEMICAP has only 2 of 7 names; other markets are empty
    assert ci.SPECS["NUCLEAR"] == "[US] US Nuclear & SMR" and ci.SPECS["GRID"] == "[US] US Grid & Electrification"
    assert "SEMICAP" not in ci.SPECS
    pts = out["NUCLEAR"]
    assert all(n >= 3 for _, _, n in pts) and pts[0][0] > idx[ci.WARMUP]
    assert "[US] US Nuclear & SMR" in capsys.readouterr().out
    # data starting before 2016 is trimmed to 2016
    us_old = _long({s: pd.Series(np.linspace(10, 20, 900), index=pd.bdate_range("2014-01-01", periods=900))
                    for s in ["CCJ", "LEU", "BWXT"]})
    mod.load_market = lambda m: us_old if m == "US" else pd.DataFrame({"Date": [], "Symbol": [], "Close": []})
    out = ci.build_global(COST)
    assert out["NUCLEAR"][0][0] >= pd.Timestamp("2016-01-04")


# ------------------------------------------------------------------ build
class Flags:
    def __init__(self, excluded):
        self.excluded = excluded


def _install_build_deps(monkeypatch, px, tv, contaminated=()):
    P = types.ModuleType("pead_liquidity_study")
    P.load_prices = lambda: (px.copy(), tv.copy())
    P.contaminated = lambda p: set(contaminated)
    flags_calls = []

    def flags(p, t, i, band):
        flags_calls.append((i, list(band)))
        return Flags({c: (k % 3 == 0) for k, c in enumerate(band)})  # every third name excluded

    S = types.ModuleType("smallcap_screener")
    S.flags = flags
    monkeypatch.setitem(sys.modules, "pead_liquidity_study", P)
    monkeypatch.setitem(sys.modules, "smallcap_screener", S)
    return flags_calls


def _stub_builders(ci, monkeypatch):
    for name in ("build_themes", "build_bands", "build_clusters", "build_peers"):
        monkeypatch.setattr(ci, name, lambda *a, **k: {})
    monkeypatch.setattr(ci, "build_global", lambda cost: {})
    monkeypatch.setattr(ci, "status", lambda a: 0)


def test_build_core_indices_and_database_write(ci, monkeypatch, capsys):
    px, tv = _panel(n_days=700, n_syms=100, groups=5)
    flags_calls = _install_build_deps(monkeypatch, px, tv, contaminated=["S000", "S001", "NOT_A_COLUMN"])
    _stub_builders(ci, monkeypatch)
    monkeypatch.setattr(ci, "BAND_LO", 10)
    monkeypatch.setattr(ci, "BAND_HI", 90)
    rc = ci.build(types.SimpleNamespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "98 symbols × 700 bars (3 excluded for unexplained discontinuity)" in out  # message counts the set size
    (kind, args), = [c for c in ci.pgc.calls if c[0] == "dai"]
    schema, table, where, params, tuples, cols = args
    assert (schema, table, where) == ("indices", "custom_daily", "index_code = ANY(%s)")
    assert params == (["SMLEW", "SMLEXC", "SMLSCR"],) and cols == ["index_code", "index_name", "index_date", "level", "n_members"]
    df = pd.DataFrame(tuples, columns=cols)
    # three formations at i = 260, 386, 512, 638 -> flags consulted at each rebalance bar
    assert [i for i, _ in flags_calls] == [260, 386, 512, 638]
    assert all(len(b) == 80 and "S000" not in b and "S001" not in b for _, b in flags_calls)
    n = df.groupby("index_code").n_members.first().to_dict()
    assert n["SMLEW"] == 80 and n["SMLEXC"] == 27 and n["SMLSCR"] == 53  # every third of 80 names is excluded
    ew = df[df.index_code == "SMLEW"].sort_values("index_date")
    assert ew.index_date.iloc[0] == px.index[261].date() and ew.index_date.iloc[-1] == px.index[-1].date()
    band = [c for c in px.columns if c not in ("S000", "S001")]
    turn = tv[band].iloc[0:260].median().sort_values(ascending=False)
    members = list(turn.index[10:90])
    r1 = px[members].pct_change(fill_method=None).loc[px.index[261]].mean()
    assert ew.level.iloc[0] == pytest.approx(1000.0 * (1 - COST) * (1 + r1))
    assert "wrote" in out and "across 3 indices" in out
    ensured = [c for c in ci.pgc.calls if c[0] == "ensure"][0][1]
    assert any(s.startswith("CREATE SCHEMA") for s in ensured) and any("custom_daily_pk" in s for s in ensured)


def test_build_skips_bands_smaller_than_sixty_and_reports_no_postgres(ci, monkeypatch, capsys):
    px, tv = _panel(n_days=700, n_syms=100, groups=5)
    _install_build_deps(monkeypatch, px, tv)
    _stub_builders(ci, monkeypatch)
    monkeypatch.setattr(ci, "BAND_LO", 10)
    monkeypatch.setattr(ci, "BAND_HI", 69)  # 59 names: below the 60-name minimum
    ci.pgc.up = False
    # with no band the three core series are empty, and postgres is unavailable: the status is still printed
    monkeypatch.setattr(ci, "status", lambda a: 7)
    # empty frame -> df.index_code access fails; guard by supplying a themes series
    monkeypatch.setattr(ci, "build_themes", lambda *a, **k: {"ETHANOL": [(pd.Timestamp("2020-01-01"), 1000.0, 5)]})
    assert ci.build(types.SimpleNamespace()) == 7
    out = capsys.readouterr().out
    assert "postgres unavailable — custom indices not written" in out
    assert [c for c in ci.pgc.calls if c[0] == "dai"] == []


def test_build_merges_thematic_band_cluster_peer_and_global(ci, monkeypatch, capsys):
    px, tv = _panel(n_days=700, n_syms=100, groups=5)
    _install_build_deps(monkeypatch, px, tv)
    _stub_builders(ci, monkeypatch)
    monkeypatch.setattr(ci, "BAND_LO", 10)
    monkeypatch.setattr(ci, "BAND_HI", 90)
    d = pd.Timestamp("2020-01-02")
    ci.SPECS.update({"INL250": "India Liquid 250", "CL_X": "[cluster] x", "PEER1": "[recurring peers] g1",
                     "NUCLEAR": "[US] US Nuclear & SMR"})  # the real builders register their own names
    monkeypatch.setattr(ci, "build_themes", lambda *a: {"ETHANOL": [(d, 1010.0, 5)]})
    monkeypatch.setattr(ci, "build_bands", lambda *a: {"INL250": [(d, 1020.0, 90)]})
    monkeypatch.setattr(ci, "build_clusters", lambda *a: {"CL_X": [(d, 1030.0, 8)]})
    monkeypatch.setattr(ci, "build_peers", lambda *a: {"PEER1": [(d, 1040.0, 7)]})
    monkeypatch.setattr(ci, "build_global", lambda cost: {"NUCLEAR": [(d, 1050.0, 6)]})
    ci.build(types.SimpleNamespace())
    args = [c for c in ci.pgc.calls if c[0] == "dai"][0][1]
    df = pd.DataFrame(args[4], columns=args[5])
    assert set(df.index_code) == {"SMLEW", "SMLEXC", "SMLSCR", "ETHANOL", "INL250", "CL_X", "PEER1", "NUCLEAR"}
    et = df[df.index_code == "ETHANOL"].iloc[0]
    assert et.level == 1010.0 and et.n_members == 5 and et.index_name == ci.THEMES["ETHANOL"][0]  # name from THEMES
    assert df[df.index_code == "PEER1"].iloc[0].index_date == d.date()
    out = capsys.readouterr().out
    for hdr in ("thematic baskets", "breadth bands", "cluster-discovered", "recurring-peer", "cross-country thematic"):
        assert hdr in out
    assert df[df.index_code == "NUCLEAR"].iloc[0].index_name == "[US] US Nuclear & SMR"


# ------------------------------------------------------------------ status
class FakeDuck:
    def __init__(self, df):
        self.df_ = df
        self.executed = []

    def connect(self):
        return self

    def execute(self, sql):
        self.executed.append(sql)
        return types.SimpleNamespace(df=lambda: self.df_)


def _install_duck(monkeypatch, df):
    d = FakeDuck(df)
    monkeypatch.setitem(sys.modules, "duckdb", d)
    return d


def _series_rows(code, name, start, n, growth, members=50):
    idx = pd.bdate_range(start, periods=n)
    lv = 1000.0 * np.cumprod(np.full(n, 1 + growth))
    return pd.DataFrame({"index_code": code, "index_name": name, "index_date": idx.date, "level": lv, "n_members": members})


def test_status_prints_table_and_null_comparison(ci, monkeypatch, capsys):
    df = pd.concat([
        _series_rows("SMLEW", "Smallcap Band Equal-Weight", "2016-01-04", 800, 0.0004),
        _series_rows("SMLSCR", "Smallcap Screened", "2016-01-04", 800, 0.0006, members=40),
        _series_rows("SMLEXC", "Smallcap Excluded", "2016-01-04", 800, 0.0001, members=20),
        _series_rows("ETHANOL", "x", "2016-01-04", 800, 0.0002, members=9),
        _series_rows("CL_FOO", "x", "2016-01-04", 800, 0.0003),
        _series_rows("PEER1", "x", "2016-01-04", 800, 0.0003),
        _series_rows("MYSTERY", "ignored — not in the ordering", "2016-01-04", 800, 0.01),
    ], ignore_index=True)
    duck = _install_duck(monkeypatch, df)
    assert ci.status(types.SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert duck.executed[:2] == ["INSTALL postgres", "LOAD postgres"] and f"ATTACH '{ci.DSN}' AS pg (TYPE postgres)" in duck.executed[2]
    lines = out.splitlines()
    rows = {ln.split()[0]: ln for ln in lines if ln and not ln.startswith(" ") and ln.split()[0] in {"SMLEW", "SMLSCR", "SMLEXC", "ETHANOL", "CL_FOO", "PEER1", "MYSTERY"}}
    assert set(rows) == {"SMLEW", "SMLSCR", "SMLEXC", "ETHANOL", "CL_FOO", "PEER1"}  # unknown codes are not listed
    order = [lines.index(rows[k]) for k in ("SMLEW", "SMLSCR", "SMLEXC", "ETHANOL", "CL_FOO", "PEER1")]
    assert order == sorted(order)
    yrs = (pd.bdate_range("2016-01-04", periods=800)[-1] - pd.Timestamp("2016-01-04")).days / 365.25
    ew_cagr = (1.0004 ** 799) ** (1 / yrs) - 1
    sc_cagr = (1.0006 ** 799) ** (1 / yrs) - 1
    assert f"{ew_cagr:.2%}" in rows["SMLEW"]
    assert f"SMLSCR vs the null (SMLEW): {(sc_cagr - ew_cagr) * 100:+.2f}pp/yr" in out
    assert "SMLEXC vs the null (SMLEW): -" in out  # the excluded complement underperforms the null
    assert "rebalance 126 bars · 100bp charged to ALL series" in out and "span 2016-01-04" in out
    assert rows["SMLSCR"].rstrip().endswith("40")  # mean members
    assert "Smallcap Screened"[:20] in rows["SMLSCR"]


def test_status_without_null_and_empty(ci, monkeypatch, capsys):
    df = _series_rows("SMLSCR", "Smallcap Screened", "2016-01-04", 300, 0.0005)
    _install_duck(monkeypatch, df)
    assert ci.status(types.SimpleNamespace()) == 0
    assert "vs the null" not in capsys.readouterr().out  # no SMLEW baseline -> no comparison lines
    _install_duck(monkeypatch, pd.DataFrame(columns=["index_code", "index_name", "index_date", "level", "n_members"]))
    assert ci.status(types.SimpleNamespace()) == 1
    assert "no custom indices built" in capsys.readouterr().out


# ------------------------------------------------------------------ main
def test_main_dispatch(ci, monkeypatch):
    monkeypatch.setattr(ci, "build", lambda a: ("build", a.build))
    monkeypatch.setattr(ci, "status", lambda a: ("status", a.status))
    monkeypatch.setattr(sys, "argv", ["custom_indices.py", "--build"])
    assert ci.main() == ("build", True)
    monkeypatch.setattr(sys, "argv", ["custom_indices.py", "--status"])
    assert ci.main() == ("status", True)
    monkeypatch.setattr(sys, "argv", ["custom_indices.py"])
    assert ci.main() == ("status", False)  # default is the read-only status view


# ------------------------------------------------------------------ mutation follow-ups
def test_build_themes_keeps_blocks_with_exactly_three_live_names(ci):
    names = ["BALRAMCHIN", "TRIVENI", "PRAJIND", "DWARKESH", "BAJAJHIND"]
    px, tv = _panel(n_days=700, names=names, groups=2)
    px.loc[px.index[:300], ["DWARKESH", "BAJAJHIND"]] = np.nan  # three live names at the first rebalance
    pts = ci.build_themes(*_inputs(px, tv), COST)["ETHANOL"]
    assert pts[0][0] == px.index[ci.WARMUP + 1] and pts[0][2] == 3


def test_build_core_requires_sixty_band_members_exactly(ci, monkeypatch):
    px, tv = _panel(n_days=700, n_syms=100, groups=5)
    _install_build_deps(monkeypatch, px, tv)
    _stub_builders(ci, monkeypatch)
    monkeypatch.setattr(ci, "BAND_LO", 10)
    monkeypatch.setattr(ci, "BAND_HI", 70)  # exactly 60 names
    monkeypatch.setattr(ci, "build_themes", lambda *a: {"ETHANOL": [(pd.Timestamp("2020-01-02"), 1000.0, 5)]})
    ci.build(types.SimpleNamespace())
    args = [c for c in ci.pgc.calls if c[0] == "dai"][0][1]
    assert {"SMLEW", "SMLSCR", "SMLEXC"} <= set(pd.DataFrame(args[4], columns=args[5]).index_code)


def test_build_treats_unflagged_names_as_excluded(ci, monkeypatch):
    px, tv = _panel(n_days=700, n_syms=100, groups=5)
    _install_build_deps(monkeypatch, px, tv)
    _stub_builders(ci, monkeypatch)
    monkeypatch.setattr(ci, "BAND_LO", 10)
    monkeypatch.setattr(ci, "BAND_HI", 90)
    sys.modules["smallcap_screener"].flags = lambda p, t, i, band: Flags({c: False for c in band[:40]})  # 40 verdicts only
    ci.build(types.SimpleNamespace())
    args = [c for c in ci.pgc.calls if c[0] == "dai"][0][1]
    df = pd.DataFrame(args[4], columns=args[5])
    n = df.groupby("index_code").n_members.first().to_dict()
    assert n["SMLSCR"] == 40 and n["SMLEXC"] == 40  # a name with no verdict is excluded, not admitted


def test_build_clusters_needs_eighty_names_and_501_points(ci, monkeypatch):
    px, tv = _panel(n_days=762, n_syms=80, groups=8)  # exactly 80 names and 501 points per code
    _install_sectors(monkeypatch, list(px.columns), 8)
    monkeypatch.setattr(ci, "CLUSTER_K", 8)
    out = ci.build_clusters(*_inputs(px, tv), COST)
    assert out and all(len(v) == 501 for v in out.values())
    px, tv = _panel(n_days=761, n_syms=80, groups=8)  # 500 points: not more than 500, dropped
    assert ci.build_clusters(*_inputs(px, tv), COST) == {}
    px, tv = _panel(n_days=762, n_syms=79, groups=8)
    _install_sectors(monkeypatch, list(px.columns), 8)
    assert ci.build_clusters(*_inputs(px, tv), COST) == {}  # 79 names: below the minimum


def test_build_peers_first_group_appears_after_exactly_four_formations(ci, monkeypatch):
    px, tv = _panel(n_days=1700, n_syms=90, groups=9)
    monkeypatch.setattr(ci, "CLUSTER_K", 9)
    out = ci.build_peers(*_inputs(px, tv), COST)
    first = min(d for pts in out.values() for d, _, _ in pts)
    assert first == px.index[ci.WARMUP + 4 * ci.REBAL + 1]  # 4 formations of history, groups form at the 5th


def test_build_global_trims_history_to_2016_exactly(ci, monkeypatch):
    idx = pd.bdate_range("2015-06-01", periods=1000)
    rng = np.random.default_rng(2)
    frames = {s: pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx) for s in ["CCJ", "LEU", "BWXT"]}
    mod = types.ModuleType("backtest_tier_anomalies")
    mod.load_market = lambda m: _long(frames) if m == "US" else pd.DataFrame({"Date": [], "Symbol": [], "Close": []})
    monkeypatch.setitem(sys.modules, "backtest_tier_anomalies", mod)
    out = ci.build_global(COST)
    kept = pd.DatetimeIndex(sorted({d for d in idx if d.year >= 2016}))
    assert out["NUCLEAR"][0][0] == kept[ci.WARMUP + 1]  # warm-up is counted from the first 2016 bar


def test_build_peers_admits_pairs_seen_in_three_formations(ci, monkeypatch):
    # the first group only enters the universe at the 2nd formation, so its pairs are co-seen 3 times by the 5th
    px, tv = _panel(n_days=1700, n_syms=96, groups=8)
    late = [c for i, c in enumerate(px.columns) if i % 8 == 0]
    px.loc[px.index[:150], late] = np.nan  # < 220 observations in the first formation's window
    monkeypatch.setattr(ci, "CLUSTER_K", 8)
    out = ci.build_peers(*_inputs(px, tv), COST)
    assert len(out) == 8  # all eight groups form, including the late-joining one
    first = px.index[ci.WARMUP + 4 * ci.REBAL + 1]
    n_at_first = sum(n for pts in out.values() for d, _, n in pts if d == first)
    assert n_at_first == 96  # the late group (seen in 3 of 4 formations) is already included in the first peer formation
