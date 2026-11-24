"""Offline tests for pe_bands (TTM P/E bands and tiers) with fake pg_client / price panel."""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


class RecConn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def pb(monkeypatch, tmp_path):
    pgc = types.ModuleType("pg_client")
    pgc.conn = RecConn()
    pgc.connected = []
    pgc.writes = []
    pgc.connect = lambda *a, **k: pgc.connected.append(1) or True
    pgc.get_connection = lambda: pgc.conn
    pgc.replace_table_atomic = lambda schema, table, df, types_: pgc.writes.append((schema, table, df.copy(), types_)) or len(df)
    monkeypatch.setitem(sys.modules, "pg_client", pgc)
    P = types.ModuleType("pead_liquidity_study")
    P.px = pd.DataFrame()
    P.load_prices = lambda: (P.px, None)
    monkeypatch.setitem(sys.modules, "pead_liquidity_study", P)
    sys.modules.pop("pe_bands", None)
    mod = importlib.import_module("pe_bands")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    monkeypatch.setattr(mod, "SCREENER_DIR", str(tmp_path / "screener"))
    mod.pgc, mod.P = pgc, P
    yield mod
    sys.modules.pop("pe_bands", None)


def _prices(symbols, start="2018-01-01", end="2025-12-31", level=100.0):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({s: np.full(len(idx), level) for s in symbols}, index=idx)


def _write_screener(mod, rows):
    from pathlib import Path

    p = Path(mod.SCREENER_DIR)
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p / "IN_screener_only_backup.parquet")


def _annual(symbol, eps_list, start_year=2017, shares=1e7):
    """Rows whose eps (= net_income * 1e7 / shares) equals the given list, one per fiscal year-end."""
    return [{"ticker": symbol, "fy_end": f"{start_year + i}-03-31", "net_income": e * shares / 1e7, "shares": shares}
            for i, e in enumerate(eps_list)]


# ------------------------------------------------------------------ connect / screener eps
def test_connect_returns_connection(pb):
    assert pb.connect() is pb.pgc.conn and pb.pgc.connected == [1]


def test_load_screener_eps_missing_file_exits(pb):
    with pytest.raises(SystemExit) as e:
        pb.load_screener_eps()
    assert "missing" in str(e.value)


def test_load_screener_eps_cleaning(pb):
    rows = _annual("AAA", [10.0, 12.0])
    rows += [{"ticker": "BAD", "fy_end": "2020-03-31", "net_income": np.nan, "shares": 1e7},
             {"ticker": "ZSH", "fy_end": "2020-03-31", "net_income": 5.0, "shares": 0.0},
             {"ticker": "NEG", "fy_end": "2020-03-31", "net_income": 5.0, "shares": -1.0},
             {"ticker": "NS", "fy_end": "2020-03-31", "net_income": 5.0, "shares": np.nan}]
    rows += [{"ticker": "AAA", "fy_end": "2017-03-31", "net_income": 99.0, "shares": 1e7}]  # duplicate, later row wins
    _write_screener(pb, rows)
    d = pb.load_screener_eps()
    assert list(d.columns) == ["symbol", "fy_end", "avail", "eps"]
    assert set(d.symbol) == {"AAA"} and len(d) == 2
    a = d.sort_values("fy_end")
    assert list(a.eps) == [99.0, 12.0]  # last duplicate kept; eps = ni * 1e7 / shares
    assert list(a.avail) == [pd.Timestamp("2017-03-31") + pd.Timedelta(days=90),
                             pd.Timestamp("2018-03-31") + pd.Timedelta(days=90)]


# ------------------------------------------------------------------ build / _bands (screener path)
def _screener_world(pb):
    eps_a = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]  # 2017..2023
    eps_b = [5.0] * 7
    rows = _annual("AAA", eps_a) + _annual("BBB", eps_b) + _annual("LOSS", [-1.0] * 7) + _annual("FEW", [9.0] * 5)
    _write_screener(pb, rows)
    pb.P.px = _prices(["AAA", "BBB", "LOSS", "FEW", "NOPX_ELSEWHERE"])
    pb.P.px = pb.P.px.drop(columns=["NOPX_ELSEWHERE"])


def test_build_screener_path_values_bands_tiers_and_report(pb, capsys, tmp_path):
    _screener_world(pb)
    rc = pb.build(types.SimpleNamespace(source="screener"))
    assert rc == 0
    schema, table, d, ctypes = pb.pgc.writes[0]
    assert (schema, table) == ("fundamentals", "india_pe_daily") and ctypes is pb.COLUMN_TYPES
    assert list(d.columns) == list(pb.COLUMN_TYPES)
    # FEW has < 6 years, LOSS has no positive earnings -> neither survives
    assert set(d.symbol) == {"AAA", "BBB"}
    a = d[d.symbol == "AAA"].set_index("obs_date")
    # the first AAA filing is 2017-03-31 + 90d = 2017-06-29, before the price panel: ffilled at the panel start
    first = a.index.min()
    assert a.loc[first, "ttm_eps"] == 10.0 and a.loc[first, "pe"] == 10.0
    # eps steps to 20 once the FY2020 (31 Mar 2020) filing is public on 2020-06-29
    assert a.loc["2020-06-26", "pe"] == 10.0 and a.loc["2020-06-29", "pe"] == 5.0
    assert a.loc["2021-01-04", "pe"] == 5.0 and a.loc["2021-01-04", "close"] == 100.0
    # tier: AAA pays 10 vs BBB 20 (before 2020-06-29): the cheaper stock is tier 5 (rank 0.5), dearer tier 10
    day = d[d.obs_date == pd.Timestamp("2019-01-02")].set_index("symbol")
    assert day.loc["AAA", "pe"] == 10.0 and day.loc["BBB", "pe"] == 20.0
    assert day.loc["AAA", "tier"] == 5 and day.loc["BBB", "tier"] == 10
    assert set(map(str, d.band.unique())) <= {"<-2sd", "-2..-1sd", "normal", "+1..+2sd", ">+2sd", "nan"}  # NaN renders as nan
    assert (d.pe_sd.dropna() >= pb.SD_FLOOR_LOG).all()  # sigma floor on log P/E
    out = capsys.readouterr().out
    assert "screener.in annual:" in out and "4 symbols" not in out and "2 symbols" in out
    assert "band occupancy" in out and "wrote" in out
    rep = (tmp_path / "reports" / "pe_bands_screener.md").read_text()
    assert rep.startswith("# India P/E panel — bands, tiers and accuracy")
    assert "pe_bands.py --build --source screener" in rep and "## Cross-sectional tiers" in rep


def test_bands_normal_when_stable_and_flags_re_rating(pb):
    # stable eps -> P/E drifts only with price; a price jump after a stable history lands in the top band
    eps = [10.0] * 8
    _write_screener(pb, _annual("AAA", eps))
    px = _prices(["AAA"])
    px.loc[px.index >= "2025-06-01", "AAA"] = 400.0
    pb.P.px = px
    pb.build(types.SimpleNamespace(source="screener"))
    d = pb.pgc.writes[0][2].set_index("obs_date")
    assert d.loc["2024-12-31", "band"] == "normal"
    assert d.loc["2025-12-30", "pe_z"] > 2 and d.loc["2025-12-30", "band"] == ">+2sd"
    assert d.loc["2025-12-30", "pe"] == 40.0
    # z uses log P/E and the mean is reported back in P/E units
    assert d.loc["2024-12-31", "pe_mean"] == pytest.approx(10.0, rel=1e-6)


def test_bands_no_overlap_and_empty_series(pb, capsys):
    _write_screener(pb, _annual("AAA", [10.0] * 7))
    pb.P.px = _prices(["ZZZ"])  # panel lacks every symbol
    assert pb.build(types.SimpleNamespace(source="screener")) == 1
    assert "no overlap between parsed EPS and the price panel" in capsys.readouterr().out
    assert pb.pgc.writes == []
    px = _prices(["AAA"])
    px["AAA"] = np.nan  # present but empty
    pb.P.px = px
    assert pb.build(types.SimpleNamespace(source="screener")) == 1


def test_min_annual_boundary(pb):
    _write_screener(pb, _annual("SIX", [10.0] * 6) + _annual("FIVE", [10.0] * 5))
    pb.P.px = _prices(["SIX", "FIVE"])
    pb.build(types.SimpleNamespace(source="screener"))
    assert set(pb.pgc.writes[0][2].symbol) == {"SIX"}  # >= 6 years kept, 5 dropped


# ------------------------------------------------------------------ build (xbrl path)
def _quarters(symbol, n, eps=2.0, start="2017-03-31", filing_lag=45):
    pe_ = pd.date_range(start, periods=n, freq="QE-MAR")
    return pd.DataFrame({"symbol": symbol, "period_end": pe_, "filing_date": pe_ + pd.Timedelta(days=filing_lag),
                         "eps": eps if np.isscalar(eps) else list(eps)})


def _patch_read_sql(monkeypatch, frames):
    calls = []

    def fake(sql, con, params=None):
        calls.append(" ".join(str(sql).split()))
        for key, df in frames:
            if key in str(sql):
                if isinstance(df, Exception):
                    raise df
                return df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", fake)
    return calls


def test_build_xbrl_ttm_and_filters(pb, monkeypatch, capsys):
    q = pd.concat([_quarters("AAA", 16), _quarters("SHORT", 9), _quarters("DUP", 12)], ignore_index=True)
    q = pd.concat([q, q.iloc[[0]]], ignore_index=True)  # duplicate (symbol, period_end) is collapsed
    _patch_read_sql(monkeypatch, [("india_quarterly", q)])
    pb.P.px = _prices(["AAA", "SHORT", "DUP"])
    assert pb.build(types.SimpleNamespace(source="xbrl")) == 0
    d = pb.pgc.writes[0][2]
    # TTM = sum of 4 quarters = 8; SHORT yields only 6 TTM observations (< MIN_Q=8) and is dropped
    assert set(d.symbol) == {"AAA", "DUP"}
    assert set(d.ttm_eps.dropna()) == {8.0} and set(d.pe.dropna()) == {12.5}
    out = capsys.readouterr().out
    assert "TTM observations" in out and "symbols with >= 8 quarters" in out


def test_build_xbrl_span_filter_rejects_gappy_quarters(pb, monkeypatch, capsys):
    # one missing quarter stretches 4 consecutive rows over > 300 days, so they yield no TTM value
    q = _quarters("GAP", 16)
    q = q.drop(index=[5, 9, 13])
    _patch_read_sql(monkeypatch, [("india_quarterly", q)])
    pb.P.px = _prices(["GAP"])
    assert pb.build(types.SimpleNamespace(source="xbrl")) == 1
    assert "no symbol has enough history yet" in capsys.readouterr().out


def test_build_xbrl_empty_table(pb, monkeypatch, capsys):
    _patch_read_sql(monkeypatch, [("india_quarterly", pd.DataFrame(columns=["symbol", "period_end", "filing_date", "eps"]))])
    assert pb.build(types.SimpleNamespace(source="xbrl")) == 1
    assert "no parsed quarters" in capsys.readouterr().out


# ------------------------------------------------------------------ report
def _d_frame(n_syms=40, with_ratios=True):
    idx = pd.bdate_range("2025-01-01", periods=3)
    rows = []
    for s in range(n_syms):
        for i, dt in enumerate(idx):
            rows.append({"symbol": f"S{s}", "obs_date": dt, "close": 1.0, "ttm_eps": 1.0, "pe": 10.0 + s,
                         "pe_mean": 10.0, "pe_sd": 0.15, "pe_z": 0.0, "band": "normal" if s % 5 else "nan",
                         "tier": 1 + s // 4})
    return pd.DataFrame(rows)


def test_write_report_accuracy_section(pb, monkeypatch, tmp_path):
    d = _d_frame()
    ref = pd.DataFrame({"symbol": [f"S{s}" for s in range(40)], "pe_ref": [(10.0 + s) * (1.1 if s < 20 else 1.5) for s in range(40)]})
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", ref)])
    pb._write_report("xbrl", d, pb.pgc.conn)
    rep = (tmp_path / "reports" / "pe_bands_xbrl.md").read_text()
    assert "- **120** daily observations · **40** symbols" in rep
    assert "span **2025-01-01 → 2025-01-03**" in rep
    assert "## Accuracy vs `fundamentals.ratios`" in rep and "40 symbols present in both" in rep
    err = ((d[d.obs_date == d.obs_date.max()].set_index("symbol").pe - ref.set_index("symbol").pe_ref).abs()
           / ref.set_index("symbol").pe_ref * 100)
    assert f"**median {err.median():.1f}%**" in rep and f"| p50 | {err.quantile(0.5):.1f}% |" in rep
    assert "within 25%: 50%" in rep and "within 50%: 100%" in rep  # 20 of 40 are ~9% off, 20 are ~33% off
    assert "`nan` at 20.0%" in rep
    assert "| tier (latest bar) | symbols |" in rep and "| 1 | 4 |" in rep


def test_write_report_skips_accuracy_when_few_overlap_or_error(pb, monkeypatch, tmp_path):
    d = _d_frame()
    ref = pd.DataFrame({"symbol": [f"S{s}" for s in range(29)], "pe_ref": [10.0 + s for s in range(29)]})
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", ref)])
    pb._write_report("a", d, pb.pgc.conn)
    assert "Accuracy vs" not in (tmp_path / "reports" / "pe_bands_a.md").read_text()  # 29 < 30
    ref30 = pd.DataFrame({"symbol": [f"S{s}" for s in range(30)], "pe_ref": [10.0 + s for s in range(30)]})
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", ref30)])
    pb._write_report("b", d, pb.pgc.conn)
    assert "Accuracy vs" in (tmp_path / "reports" / "pe_bands_b.md").read_text()  # exactly 30 passes
    # non-positive reference P/Es are excluded from the comparison
    ref_bad = pd.DataFrame({"symbol": [f"S{s}" for s in range(40)], "pe_ref": [-1.0] * 15 + [10.0 + s for s in range(15, 40)]})
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", ref_bad)])
    pb._write_report("c", d, pb.pgc.conn)
    assert "25 symbols present in both" not in (tmp_path / "reports" / "pe_bands_c.md").read_text()
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", RuntimeError("relation missing " + "x" * 100))])
    before = pb.pgc.conn.rollbacks
    pb._write_report("d", d, pb.pgc.conn)
    rep = (tmp_path / "reports" / "pe_bands_d.md").read_text()
    assert "accuracy cross-check unavailable: relation missing" in rep and pb.pgc.conn.rollbacks == before + 1


# ------------------------------------------------------------------ extremes / show / main
def _stored(n=12):
    return pd.DataFrame({
        "symbol": [f"S{i:02d}" for i in range(n)], "obs_date": pd.Timestamp("2025-12-30"),
        "close": 100.0, "ttm_eps": 5.0, "pe": [10.0 + i for i in range(n)], "pe_mean": 12.0,
        "pe_sd": 0.2, "pe_z": [float(i) - 5 for i in range(n)], "band": "normal", "tier": 3,
    })


def test_extremes(pb, monkeypatch, capsys):
    _patch_read_sql(monkeypatch, [("india_pe_daily", _stored())])
    assert pb.extremes(types.SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "12 symbols with a band" in out
    cheap, rich = out.split("RICHEST")
    assert cheap.count("S0") == 8 and "S00" in cheap and "S07" in cheap and "S08" not in cheap  # 8 lowest z
    assert "S11" in rich and "S04" in rich and "S03" not in rich  # 8 highest z
    assert "z  -5.00" in cheap and "z  +6.00" in rich
    _patch_read_sql(monkeypatch, [("india_pe_daily", pd.DataFrame())])
    assert pb.extremes(types.SimpleNamespace()) == 1
    assert "nothing built" in capsys.readouterr().out


def test_show(pb, monkeypatch, capsys):
    calls = []

    def fake(sql, con, params=None):
        calls.append(params)
        return _stored(7)

    monkeypatch.setattr(pd, "read_sql", fake)
    assert pb.show(types.SimpleNamespace(show="RELIANCE")) == 0
    out = capsys.readouterr().out
    assert calls == [["RELIANCE"]] and out.startswith("RELIANCE: 7 days")
    assert out.count("normal") == 5  # tail(5)
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: pd.DataFrame())
    assert pb.show(types.SimpleNamespace(show="NOPE")) == 1
    assert "no P/E band for NOPE" in capsys.readouterr().out


def test_main_dispatch(pb, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(pb, "build", lambda a: seen.append(("build", a.source)) or 11)
    monkeypatch.setattr(pb, "extremes", lambda a: seen.append("extremes") or 12)
    monkeypatch.setattr(pb, "show", lambda a: seen.append(("show", a.show)) or 13)
    for argv, rc, call in [(["--build"], 11, ("build", "xbrl")), (["--build", "--source", "screener"], 11, ("build", "screener")),
                           (["--extremes"], 12, "extremes"), (["--show", "TCS"], 13, ("show", "TCS"))]:
        monkeypatch.setattr(sys, "argv", ["pe_bands.py"] + argv)
        assert pb.main() == rc and seen[-1] == call
    monkeypatch.setattr(sys, "argv", ["pe_bands.py"])
    assert pb.main() == 0 and "--build" in capsys.readouterr().out  # help
    # build wins over the other flags
    monkeypatch.setattr(sys, "argv", ["pe_bands.py", "--build", "--extremes"])
    assert pb.main() == 11


# ------------------------------------------------------------------ mutation follow-ups
def test_bands_assign_2sd_and_beyond(pb):
    # flat EPS keeps sigma at its 0.15 floor, so z = ln(price move) / 0.15
    _write_screener(pb, _annual("UP", [10.0] * 8) + _annual("DOWN", [10.0] * 8))
    px = _prices(["UP", "DOWN"])
    px.loc[px.index >= "2025-06-02", "UP"] = 140.0  # z = ln(1.4)/0.15 = +2.24
    px.loc[px.index >= "2025-06-02", "DOWN"] = 70.0  # z = ln(0.7)/0.15 = -2.38
    pb.P.px = px
    pb.build(types.SimpleNamespace(source="screener"))
    d = pb.pgc.writes[0][2].set_index(["symbol", "obs_date"])
    last = pd.Timestamp("2025-12-31")
    assert d.loc[("UP", last), "pe_z"] == pytest.approx(np.log(1.4) / 0.15, rel=1e-6) and d.loc[("UP", last), "band"] == ">+2sd"
    assert d.loc[("DOWN", last), "band"] == "<-2sd"


def test_tiers_round_up_and_three_stocks_split_into_deciles(pb):
    _write_screener(pb, _annual("A", [10.0] * 8) + _annual("B", [5.0] * 8) + _annual("C", [2.0] * 8))
    pb.P.px = _prices(["A", "B", "C"])
    pb.build(types.SimpleNamespace(source="screener"))
    d = pb.pgc.writes[0][2]
    day = d[d.obs_date == pd.Timestamp("2025-12-31")].set_index("symbol")
    assert (day.loc["A", "pe"], day.loc["B", "pe"], day.loc["C", "pe"]) == (10.0, 20.0, 50.0)
    assert (day.loc["A", "tier"], day.loc["B", "tier"], day.loc["C", "tier"]) == (4, 7, 10)  # ceil(3.33), ceil(6.67), 10


def test_xbrl_keeps_a_symbol_with_exactly_min_quarters_and_wider_quarter_spacing(pb, monkeypatch):
    # 11 quarters -> exactly 8 TTM observations (MIN_Q); spacing of 97 days keeps each 4-quarter span at 291 days
    pe_ = pd.to_datetime("2017-03-31") + pd.to_timedelta(np.arange(11) * 97, unit="D")
    q = pd.DataFrame({"symbol": "AAA", "period_end": pe_, "filing_date": pe_ + pd.Timedelta(days=45), "eps": 2.0})
    _patch_read_sql(monkeypatch, [("india_quarterly", q)])
    pb.P.px = _prices(["AAA"])
    assert pb.build(types.SimpleNamespace(source="xbrl")) == 0
    assert set(pb.pgc.writes[0][2].ttm_eps.dropna()) == {8.0}


def test_accuracy_section_ignores_non_positive_reference_pe(pb, monkeypatch, tmp_path):
    d = _d_frame()
    ref = pd.DataFrame({"symbol": [f"S{s}" for s in range(40)],
                        "pe_ref": [-1.0] * 15 + [10.0 + s for s in range(15, 40)]})  # only 25 usable references
    _patch_read_sql(monkeypatch, [("fundamentals.ratios", ref)])
    pb._write_report("e", d, pb.pgc.conn)
    assert "Accuracy vs" not in (tmp_path / "reports" / "pe_bands_e.md").read_text()
