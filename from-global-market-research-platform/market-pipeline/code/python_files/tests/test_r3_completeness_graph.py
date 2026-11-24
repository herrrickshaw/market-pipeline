"""Offline tests for completeness_graph (data, claim, gate, Cassandra and power audits + renderer)."""

import importlib
import json
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from r3_fakepg import FakeConn

REAL_PATH = Path


@pytest.fixture
def cg(monkeypatch, tmp_path):
    ss = types.ModuleType("system_state")
    ss.checks = []
    ss.sources = []
    ss.record_connection_check = lambda target, ok, **kw: ss.checks.append((target, ok, kw))
    ss.list_data_sources = lambda kind=None: ss.sources
    monkeypatch.setitem(sys.modules, "system_state", ss)
    sys.modules.pop("completeness_graph", None)
    mod = importlib.import_module("completeness_graph")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    monkeypatch.setattr(mod, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(mod, "OUT_MD", tmp_path / "reports" / "completeness_graph.md")
    (tmp_path / "reports").mkdir()

    # route the hard-coded /Users/... artefact paths into tmp_path
    def fake_path(*a, **k):
        if a and str(a[0]).startswith("/Users/umashankar/market-pipeline"):
            return REAL_PATH(str(tmp_path / "mp") + str(a[0])[len("/Users/umashankar/market-pipeline"):])
        return REAL_PATH(*a, **k)

    monkeypatch.setattr(mod, "Path", fake_path)
    mod.ss_stub = ss
    yield mod
    sys.modules.pop("completeness_graph", None)


def _install_pg(monkeypatch, conn, ok=True):
    pgc = types.ModuleType("pg_client")
    pgc.connect = lambda: ok
    pgc.get_connection = lambda: conn
    monkeypatch.setitem(sys.modules, "pg_client", pgc)
    return pgc


# ------------------------------------------------------------------ sources
def test_pg_sources_prefers_registry_then_falls_back(cg):
    assert cg._pg_sources() == cg.PG_SOURCES  # registry empty
    cg.ss_stub.sources = [{"source_key": "a.b", "date_column": "d", "critical": True, "cadence": "daily"}]
    assert cg._pg_sources() == [("a.b", "d", True, "daily")]

    def boom(kind=None):
        raise RuntimeError("db down")

    cg.ss_stub.list_data_sources = boom
    assert cg._pg_sources() == cg.PG_SOURCES


def test_con_records_check_and_raises_when_down(cg, monkeypatch):
    conn = FakeConn()
    _install_pg(monkeypatch, conn)
    assert cg._con() is conn
    assert cg.ss_stub.checks == [("postgres", True, {"checked_by": "completeness_graph.py"})]
    _install_pg(monkeypatch, conn, ok=False)
    with pytest.raises(RuntimeError, match="postgres unavailable"):
        cg._con()
    assert cg.ss_stub.checks[-1][1] is False


def _inventory_responder(counts, cols, distinct, spans, fail=()):
    def responder(sql, params):
        if sql.startswith("select count(*) from"):
            tbl = sql.split("from ")[1]
            if tbl in fail:
                return RuntimeError("relation does not exist")
            return (["c"], [(counts[tbl],)])
        if "information_schema.columns" in sql:
            return (["column_name"], [(c,) for c in cols[".".join(params)]])
        if sql.startswith("select count(distinct"):
            return (["c"], [(distinct[sql.split("from ")[1]],)])
        if sql.startswith("select min("):
            return (["lo", "hi"], [spans[sql.split("from ")[1]]])
        return None

    return responder


def test_inventory_measures_rows_symbols_span_and_staleness(cg, monkeypatch):
    today = date.today()
    cg.ss_stub.sources = [
        {"source_key": "t.daily", "date_column": "d", "critical": True, "cadence": "daily"},
        {"source_key": "t.static", "date_column": "d", "critical": False, "cadence": "static"},
        {"source_key": "t.future", "date_column": "d", "critical": False, "cadence": "daily"},
        {"source_key": "t.nodate", "date_column": None, "critical": False, "cadence": "weekly"},
        {"source_key": "t.broken", "date_column": "d", "critical": True, "cadence": "daily"},
        {"source_key": "t.stamp", "date_column": "d", "critical": False, "cadence": "weekly"},
    ]
    cols = {"t.daily": ["symbol", "d"], "t.static": ["index_name", "d"], "t.future": ["ticker", "d"],
            "t.nodate": ["x"], "t.stamp": ["scheme_code", "d"]}
    counts = {k: 10 * (i + 1) for i, k in enumerate(["t.daily", "t.static", "t.future", "t.nodate", "t.broken", "t.stamp"])}
    spans = {"t.daily": (today - timedelta(days=400), today - timedelta(days=3)),
             "t.static": (date(2016, 1, 4), date(2020, 1, 1)),
             "t.future": (today, today + timedelta(days=30)),
             "t.stamp": (datetime(2024, 1, 1), datetime.combine(today - timedelta(days=9), datetime.min.time()))}
    conn = FakeConn(_inventory_responder(counts, cols, {k: 7 for k in counts}, spans, fail=("t.broken",)))
    _install_pg(monkeypatch, conn)
    out = cg.inventory({"sources": []})
    by = {r["source"]: r for r in out["sources"]}
    d = by["t.daily"]
    assert d["rows"] == 10 and d["symbols"] == 7 and d["critical"] is True and d["tolerance"] == 5
    assert d["span"] == f"{today - timedelta(days=400)} -> {today - timedelta(days=3)}" and d["stale_days"] == 3
    assert by["t.static"]["tolerance"] is None and by["t.static"]["symbols"] == 7  # index_name is a symbol column
    assert by["t.future"]["stale_days"] == -30 and by["t.future"]["symbols"] == 7
    assert any("t.future: max d is" in n and "IN THE FUTURE" in n for n in out["notes"])
    assert "span" not in by["t.nodate"] and "symbols" not in by["t.nodate"] and by["t.nodate"]["rows"] == 40
    assert by["t.broken"]["rows"] is None and by["t.broken"]["error"].startswith("relation does not exist")
    assert conn.rollbacks == 1  # a failed count rolls back before the next table
    assert by["t.stamp"]["stale_days"] == 9 and by["t.stamp"]["span"].startswith("2024-01-01")
    # a max-date exactly one day in the future is within tolerance (strictly < -1 flags)
    cg.ss_stub.sources = [{"source_key": "t.one", "date_column": "d", "critical": False, "cadence": "daily"}]
    conn = FakeConn(_inventory_responder({"t.one": 1}, {"t.one": ["d"]}, {}, {"t.one": (today, today + timedelta(days=1))}))
    _install_pg(monkeypatch, conn)
    out = cg.inventory({"sources": []})
    assert out["notes"] == [] and out["sources"][0]["stale_days"] == -1


def test_inventory_unreachable_warehouse_and_local_files(cg, monkeypatch, tmp_path):
    _install_pg(monkeypatch, FakeConn(), ok=False)
    out = cg.inventory({"sources": []})
    assert out["sources"] == [] and out["notes"][0].startswith("warehouse unreachable: postgres unavailable")

    cg.ss_stub.sources = [{"source_key": "t.x", "date_column": None, "critical": False, "cadence": "static"}]
    conn = FakeConn(_inventory_responder({"t.x": 3}, {"t.x": ["a"]}, {}, {}))
    _install_pg(monkeypatch, conn)
    out = cg.inventory({"sources": []})
    files = [r for r in out["sources"] if r["kind"] == "file"]
    assert [f["exists"] for f in files] == [False, False] and files[0]["source"] == "india split factors"
    data = tmp_path / "mp" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"symbol": ["A", "A", "B"], "f": [1, 2, 3]}).to_parquet(data / "india_split_factors.parquet")
    pd.DataFrame({"symbol": ["X", "Y"]}).to_csv(data / "india_unverifiable_symbols.csv", index=False)
    out = cg.inventory({"sources": []})
    files = {f["source"]: f for f in out["sources"] if f["kind"] == "file"}
    assert files["india split factors"]["rows"] == 3 and files["india split factors"]["symbols"] == 2
    assert files["india unverifiable"]["rows"] == 2 and files["india unverifiable"]["exists"] is True
    pd.DataFrame({"other": [1]}).to_csv(data / "india_unverifiable_symbols.csv", index=False)
    out = cg.inventory({"sources": []})
    f = next(r for r in out["sources"] if r["source"] == "india unverifiable")
    assert f["rows"] == 1 and "symbols" not in f  # no symbol column -> no symbol count


# ------------------------------------------------------------------ claims
def test_audit_claims_verdicts(cg, tmp_path):
    cg.SCHEDULED[:] = [
        ("a_missing.py", "r.md"),
        ("b_nocite.py", None),
        ("c_unsourced.py", None),
        ("d_noreport.py", "gone.md"),
        ("e_consistent.py", "good.md"),
        ("f_stale.py", "good.md"),
        ("g_retracted.py", "good.md"),
    ]
    (tmp_path / "b_nocite.py").write_text('"""no numbers here"""\n')
    (tmp_path / "c_unsourced.py").write_text('"""edge +3.07% and t = 8.75"""\n')
    (tmp_path / "d_noreport.py").write_text('"""edge +1.50%"""\n')
    (tmp_path / "e_consistent.py").write_text('"""gets +12.90% with t 17.62"""\n')
    (tmp_path / "f_stale.py").write_text('"""gets +12.90% with t 99.99 and -2.01%"""\n')
    (tmp_path / "g_retracted.py").write_text(
        '"""\nThe old claim of +55.55% was wrong and is withdrawn.\n\nCurrent: +12.90%\n"""\n')
    (tmp_path / "reports" / "good.md").write_text("excess 12.90 and t = 17.62 and -2.01 more")
    out = {c["script"]: c for c in cg.audit_claims({})["claims"]}
    assert out["a_missing.py"] == {"script": "a_missing.py", "cited_report": "r.md", "exists": False}
    assert out["b_nocite.py"]["verdict"] == "no numeric claim cited"
    assert out["c_unsourced.py"]["verdict"] == "UNSOURCED" and out["c_unsourced.py"]["claims_in_docstring"] == ["+3.07%", "t = 8.75"]
    assert out["d_noreport.py"]["verdict"] == "CITED REPORT MISSING"
    assert out["e_consistent.py"]["verdict"] == "consistent" and out["e_consistent.py"]["missing_from_report"] == []
    assert out["e_consistent.py"]["report_mtime"] == date.today().strftime("%Y-%m-%d")
    f = out["f_stale.py"]
    assert f["verdict"] == "STALE — 1/3 quoted number(s) absent from good.md" and f["missing_from_report"] == ["t 99.99"]
    g = out["g_retracted.py"]  # the retraction paragraph (with the old figure) is excluded from claims
    assert g["claims_in_docstring"] == ["+12.90%"] and g["verdict"] == "consistent"


def test_audit_claims_only_reads_first_4000_chars(cg, tmp_path):
    cg.SCHEDULED[:] = [("long.py", None)]
    (tmp_path / "long.py").write_text("x = 1\n" + "#" * 4000 + "\n# late claim +9.99%\n")
    assert cg.audit_claims({})["claims"][0]["verdict"] == "no numeric claim cited"


# ------------------------------------------------------------------ gates
def test_audit_gates_classification(cg, tmp_path):
    cg.GATED_EXPECT[:] = ["missing.py", "noload.py", "gated.py", "ungated.py", "refuses.py",
                          "commented.py", "refuse_and_gate.py"]
    (tmp_path / "noload.py").write_text("print('no panel')\n")
    (tmp_path / "gated.py").write_text("px = load_prices()\nbad = contaminated(px)\n")
    (tmp_path / "ungated.py").write_text("px = load_prices()\n")
    (tmp_path / "refuses.py").write_text("px = load_market()\nprint('REFUSING unadjusted')\n")
    (tmp_path / "commented.py").write_text("px = load_prices()\n# contaminated(px) is deliberately not called\n")
    (tmp_path / "refuse_and_gate.py").write_text("px = load_prices()\nREFUSING\nbad = contaminated(px)\n")
    out = {g["script"]: g for g in cg.audit_gates({})["gates"]}
    assert out["missing.py"] == {"script": "missing.py", "exists": False}
    assert out["noload.py"]["verdict"].startswith("n/a")
    assert out["gated.py"]["verdict"] == "gated" and out["gated.py"]["calls_contaminated"] is True
    assert out["ungated.py"]["verdict"].startswith("UNGATED") and out["ungated.py"]["loads_panel"] is True
    assert out["refuses.py"]["verdict"].startswith("guarded by refusal")
    assert out["commented.py"]["calls_contaminated"] is False and out["commented.py"]["verdict"].startswith("UNGATED")
    assert out["refuse_and_gate.py"]["verdict"] == "gated"
    for pat in ("year=*.parquet", "cleaned_ohlcv", "ohlcv_adj", 'values="Close"', "values='Close'"):
        (tmp_path / "pat.py").write_text(f"x = '{pat}'\n")
        cg.GATED_EXPECT[:] = ["pat.py"]
        assert cg.audit_gates({})["gates"][0]["loads_panel"] is True


# ------------------------------------------------------------------ power
def test_audit_power_extracts_universe_and_counts(cg, tmp_path):
    r = tmp_path / "reports"
    (r / "short.md").write_text("tiny")
    (r / "none.md").write_text("x" * 300)  # nothing recognisable
    (r / "uni.md").write_text("# T\nUniverse: all NSE liquid names | with notes\n" + "text " * 60 +
                              "\nn = 1,234 events\nn=99\n12 monthly formations\n")
    (r / "topn.md").write_text("The top-200 liquid names were used. " + "x" * 200 + "\nevents: **5,000**\n")
    (r / "forms.md").write_text("padding " * 40 + "\n24 quarterly rebalances\n")
    out = {p["report"]: p for p in cg.audit_power({})["power"]}
    assert set(out) == {"uni.md", "topn.md", "forms.md"}
    u = out["uni.md"]
    assert u["universe"] == "all NSE liquid names" and u["sample_kind"] == "declared"
    assert u["n_max"] == 1234 and u["formations"] == 12
    t = out["topn.md"]
    assert t["universe"] == "top 200" and t["sample_kind"] == "SAMPLE (top-N liquid)" and t["n_max"] == 5000
    f = out["forms.md"]
    assert f["universe"] == "—" and f["sample_kind"] == "—" and f["n_max"] is None and f["formations"] == 24


# ------------------------------------------------------------------ cassandra
class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def one(self):
        return self.rows[0]

    def __iter__(self):
        return iter(self.rows)


class FakeSession:
    def __init__(self, data):
        self.data = data  # market -> dict
        self.queries = []

    def execute(self, q, params=None):
        self.queries.append((q, params))
        mkt = params[0] if params else None
        d = self.data.get(mkt)
        if d is None or d.get("explode"):
            raise RuntimeError("cassandra boom")
        if q.startswith("SELECT COUNT(*) FROM stock_quotes WHERE market=%s AND fundamentals_source"):
            if "imputed" not in d:
                raise RuntimeError("no such column")
            return FakeResult([(d["imputed"],)])
        if " > 0 ALLOW FILTERING" in q:
            col = q.split("AND ")[1].split(" >")[0]
            if col in d.get("bad_cols", ()):
                raise RuntimeError("bad col")
            return FakeResult([(d["cols"].get(col, 0),)])
        if "LIMIT 400" in q:
            col = q.split("SELECT ")[1].split(" FROM")[0]
            if col in d.get("bad_vals", ()):
                raise RuntimeError("bad")
            return FakeResult([(v,) for v in d["vals"].get(col, [])])
        if q.startswith("SELECT COUNT(*) FROM stock_quotes WHERE market=%s"):
            return FakeResult([(d["total"],)])
        raise AssertionError(q)


def _install_cassandra(monkeypatch, session, connect_error=None, shutdown_error=False):
    created = {}

    class Cluster:
        def __init__(self, hosts, connect_timeout=None):
            created["args"] = (hosts, connect_timeout)

        def connect(self, ks):
            if connect_error:
                raise connect_error
            created["ks"] = ks
            return session

        def shutdown(self):
            if shutdown_error:
                raise RuntimeError("shutdown failed")
            created["shutdown"] = True

    pkg, mod = types.ModuleType("cassandra"), types.ModuleType("cassandra.cluster")
    mod.Cluster = Cluster
    monkeypatch.setitem(sys.modules, "cassandra", pkg)
    monkeypatch.setitem(sys.modules, "cassandra.cluster", mod)
    return created


def test_audit_cassandra_without_driver_or_when_unreachable(cg, monkeypatch):
    monkeypatch.setitem(sys.modules, "cassandra", None)
    monkeypatch.setitem(sys.modules, "cassandra.cluster", None)
    out = cg.audit_cassandra({})
    assert out["cassandra"] == [] and "cassandra-driver not installed" in out["notes"][0]
    _install_cassandra(monkeypatch, None, connect_error=RuntimeError("no hosts " + "x" * 100))
    out = cg.audit_cassandra({})
    assert out["cassandra"] == [] and out["notes"][0].startswith("Cassandra unreachable: no hosts")
    assert len(out["notes"][0].split(": ", 1)[1]) == 70  # error truncated
    assert cg.ss_stub.checks[-1][0:2] == ("cassandra", False)


def _vals(n, uniq, whole=True):
    return [float(i % uniq) if whole else (i % uniq) + 0.37 for i in range(n)]


def test_audit_cassandra_hollow_imputed_and_suspect(cg, monkeypatch):
    data = {
        # healthy price + fundamentals, but almost everything imputed
        "us": {"total": 1000, "cols": {"cmp": 900, "pe": 800, "roe": 700}, "imputed": 960,
               "vals": {c: _vals(400, 300, whole=False) for c in cg.CASS_FUNDAMENTAL}},
        # prices present, fundamentals empty -> hollow
        "japan": {"total": 500, "cols": {"cmp": 500, "rsi": 480}, "imputed": 500, "vals": {}},
        # fundamentals populated with a 4-valued whole-number score in pb
        "china": {"total": 200, "cols": {"cmp": 200, "pb": 200}, "imputed": 0,
                  "vals": {"pb": _vals(200, 4), "roe": _vals(200, 150, whole=False)}},
        "india": {"total": 0},  # empty market is skipped
        "korea": {"explode": True},  # count failure skips the market
        "europe": {"total": 300, "cols": {"cmp": 300, "pe": 100}, "bad_cols": ("roe",),
                   "vals": {"pe": _vals(50, 50, whole=False)}},  # no imputed column -> measured keys absent
    }
    created = _install_cassandra(monkeypatch, FakeSession(data))
    out = cg.audit_cassandra({})
    recs = {r["market"]: r for r in out["cassandra"]}
    assert set(recs) == {"us", "japan", "china", "europe"} and created["args"] == (["127.0.0.1"], 8)
    assert created["ks"] == "herrrickshaw" and created["shutdown"] is True
    us = recs["us"]
    assert us["price"] == 900 and us["fund"] == 800 and us["fund_pct"] == 0.8 and us["hollow"] is False
    assert us["imputed"] == 960 and us["measured"] == 40 and us["measured_pct"] == 0.04
    assert us["suspect_cols"] == [] and us["suspect_scale"] is False
    jp = recs["japan"]
    assert jp["price"] == 500 and jp["fund"] == 0 and jp["hollow"] is True
    cn = recs["china"]
    assert cn["suspect_cols"] == ["pb(4 distinct)"] and cn["suspect_scale"] is True and cn["hollow"] is False
    eu = recs["europe"]
    assert "measured" not in eu and eu["fund"] == 100 and eu["fund_pct"] == pytest.approx(1 / 3)
    notes = "\n".join(out["notes"])
    assert "cassandra[us]: fund_pct reports 80% populated, but only 4.0% of that is measured" in notes
    assert "cassandra[japan]: 500 rows, prices populated (500) but fundamentals effectively EMPTY (0 = 0.0%)" in notes
    assert "cassandra[china]: pb(4 distinct) — whole numbers" in notes
    assert "europe" not in notes
    assert cg.ss_stub.checks[-1][0:2] == ("cassandra", True)


def test_audit_cassandra_thresholds(cg, monkeypatch):
    # hollow needs price > 50% of rows and fundamentals < 5%
    data = {"us": {"total": 100, "cols": {"cmp": 50, "pe": 4}, "imputed": 0, "vals": {}},
            "japan": {"total": 100, "cols": {"cmp": 51, "pe": 4}, "imputed": 0, "vals": {}},
            "china": {"total": 100, "cols": {"cmp": 51, "pe": 5}, "imputed": 0, "vals": {}},
            # exactly 100 values, 20 distinct whole numbers -> suspect; 21 distinct -> fine; 99 values -> skipped
            "korea": {"total": 10, "cols": {}, "imputed": 0, "vals": {"pe": _vals(100, 20)}},
            "europe": {"total": 10, "cols": {}, "imputed": 0, "vals": {"pe": _vals(100, 21)}},
            "hong_kong": {"total": 10, "cols": {}, "imputed": 0, "vals": {"pe": _vals(99, 5)}}}
    _install_cassandra(monkeypatch, FakeSession(data))
    recs = {r["market"]: r for r in cg.audit_cassandra({})["cassandra"]}
    assert recs["us"]["hollow"] is False  # 50 is not > 50% of 100
    assert recs["japan"]["hollow"] is True
    assert recs["china"]["hollow"] is False  # 5% fundamentals is not < 5%
    assert recs["korea"]["suspect_cols"] == ["pe(20 distinct)"]
    assert recs["europe"]["suspect_cols"] == [] and recs["hong_kong"]["suspect_cols"] == []


def test_audit_cassandra_shutdown_error_is_ignored(cg, monkeypatch):
    _install_cassandra(monkeypatch, FakeSession({}), shutdown_error=True)
    assert cg.audit_cassandra({})["cassandra"] == []


# ------------------------------------------------------------------ synthesize / build / render
def _state(**kw):
    s = {"sources": [], "claims": [], "gates": [], "power": [], "cassandra": [], "notes": [], "summary": ""}
    s.update(kw)
    return s


def test_synthesize_counts(cg):
    s = _state(
        sources=[{"source": "a", "tolerance": 5, "stale_days": 6},      # stale
                 {"source": "b", "tolerance": 5, "stale_days": 5},      # at tolerance: fine
                 {"source": "c", "tolerance": None, "stale_days": 900},  # static
                 {"source": "d", "error": "x"}, {"source": "e", "exists": False}, {"source": "f", "exists": True}],
        claims=[{"verdict": "consistent"}, {"verdict": "STALE — 1/2"}, {"verdict": "UNSOURCED"},
                {"verdict": "CITED REPORT MISSING"}, {"verdict": "no numeric claim cited"}],
        gates=[{"verdict": "gated"}, {"verdict": "UNGATED — x"}],
        power=[{"sample_kind": "SAMPLE (top-N liquid)"}, {"sample_kind": "declared"}],
        cassandra=[{"hollow": True}, {"hollow": False}],
    )
    summary = cg.synthesize(s)["summary"]
    assert summary.startswith("6 sources inventoried, 2 missing/erroring, 1 source(s) past their expected refresh cadence")
    assert "5 scheduled scripts checked, 3 quoting numbers" in summary
    assert "2 analyses checked for the contamination gate, 1 ungated" in summary
    assert "2 conclusions have a declared universe, 1 rest on a top-N liquid SAMPLE" in summary
    assert "2 Cassandra market(s) audited, 1 present but hollow" in summary
    assert " · " in summary


def _fake_langgraph(monkeypatch):
    START, END = "__start__", "__end__"

    class StateGraph:
        def __init__(self, schema):
            self.nodes, self.edges = {}, []

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, a, b):
            self.edges.append((a, b))

        def compile(self):
            return self

        def invoke(self, state):
            state = dict(state)
            done, order = {START}, []
            pending = dict(self.nodes)
            while pending:  # run any node whose upstream nodes all completed
                for name in list(pending):
                    ups = [a for a, b in self.edges if b == name]
                    if all(u in done for u in ups):
                        upd = pending.pop(name)(state)
                        for k, v in upd.items():
                            state[k] = (state[k] + v) if isinstance(v, list) else v
                        done.add(name)
                        order.append(name)
            state["_order"] = order
            return state

    lg, gr = types.ModuleType("langgraph"), types.ModuleType("langgraph.graph")
    gr.END, gr.START, gr.StateGraph = END, START, StateGraph
    monkeypatch.setitem(sys.modules, "langgraph", lg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", gr)


def test_build_wires_fan_out_and_join(cg, monkeypatch):
    _fake_langgraph(monkeypatch)
    app = cg.build()
    assert set(app.nodes) == {"inventory", "audit_claims", "audit_gates", "audit_power", "audit_cassandra", "synthesize"}
    assert ("__start__", "inventory") in app.edges and ("synthesize", "__end__") in app.edges
    for n in ("audit_claims", "audit_gates", "audit_power", "audit_cassandra"):
        assert ("inventory", n) in app.edges and (n, "synthesize") in app.edges
    assert len(app.edges) == 10


def _full_state():
    return _state(
        summary="S",
        sources=[{"source": "t.a", "rows": 1234, "symbols": 5, "span": "2020 -> 2026", "stale_days": 9, "tolerance": 5,
                  "cadence": "daily", "critical": True},
                 {"source": "t.b", "rows": None, "error": "boom", "cadence": "weekly", "critical": False},
                 {"source": "f.c", "kind": "file", "exists": False, "critical": True},
                 {"source": "t.d", "rows": 5, "stale_days": 1, "tolerance": 5, "critical": False}],
        claims=[{"script": "x.py", "cited_report": "r.md", "claims_in_docstring": ["+1.00%", "t 2.00", "a", "b", "c"],
                 "verdict": "STALE — 1/2"},
                {"script": "y.py", "cited_report": None, "verdict": "no numeric claim cited"},
                {"script": "z.py", "cited_report": "r.md"}],
        gates=[{"script": "m.py", "exists": False},
               {"script": "g.py", "exists": True, "calls_contaminated": True, "verdict": "gated"},
               {"script": "u.py", "exists": True, "calls_contaminated": False, "loads_panel": True,
                "verdict": "UNGATED — loads a raw price panel"},
               {"script": "n.py", "exists": True, "calls_contaminated": False, "loads_panel": False,
                "verdict": "n/a — no raw price panel loaded"}],
        power=[{"report": "small.md", "universe": "U" * 80, "sample_kind": "declared", "n_max": 10, "formations": None},
               {"report": "big.md", "universe": "top 200", "sample_kind": "SAMPLE (top-N liquid)", "n_max": 5000,
                "formations": 12}],
        cassandra=[{"market": "japan", "rows": 500, "price": 500, "fund": 0, "fund_pct": 0.0, "hollow": True},
                   {"market": "china", "rows": 200, "price": 200, "fund": 200, "fund_pct": 1.0, "suspect_scale": True,
                    "measured": 0, "measured_pct": 0.0},
                   {"market": "us", "rows": 1000, "price": 900, "fund": 800, "fund_pct": 0.8, "measured": 40,
                    "measured_pct": 0.04},
                   {"market": "india", "rows": 300, "price": 300, "fund": 290, "fund_pct": 0.97, "measured": 290,
                    "measured_pct": 0.97},
                   {"market": "korea", "rows": 100, "price": 100, "fund": 1, "fund_pct": 0.01, "measured": 1,
                    "measured_pct": 0.01}],
        notes=["note one", "note two"],
    )


def test_render_sections(cg):
    md = cg.render(_full_state())
    assert md.startswith("# Completeness audit — data, claims, and gates\n")
    assert "**S**" in md and md.endswith("\n")
    assert "| `t.a` | 1,234 | 5 | 2020 -> 2026 | 🔴 9 | daily | ✅ |" in md  # stale past tolerance flagged
    assert "| `t.b` | — | — | — | — | weekly | 🔴 |" in md  # error row
    assert "| `f.c` | — | — | — | — | — | 🔴 |" in md  # missing file
    assert "| `t.d` | 5 | — | — | 1 | — |  |" in md  # not stale, not critical
    assert "| `x.py` | r.md | +1.00%, t 2.00, a, b | 🔴 STALE — 1/2 |" in md  # only 4 claims shown
    assert "| `y.py` | — | — | no numeric claim cited |" in md and "| `z.py` | r.md | — | — |" in md
    assert "| `m.py` | — | missing |" in md and "| `g.py` | yes | gated |" in md
    assert "| `u.py` | NO | 🔴 UNGATED — loads a raw price panel |" in md and "| `n.py` | — | n/a" in md
    assert md.index("big.md") < md.index("small.md")  # sorted by n descending
    assert "| `small.md` | " + "U" * 58 + " |" in md and "U" * 59 not in md  # universe cut at 58 chars
    assert "🔶 SAMPLE (top-N liquid) | 5000 | 12 |" in md and "| 10 | — |" in md
    assert "## 5. Cassandra" in md
    row = {m: next(line for line in md.splitlines() if line.startswith(f"| {m} |"))
           for m in ("japan", "china", "us", "india", "korea")}
    assert "🔴 HOLLOW" in row["japan"] and "n/a |" in row["japan"]
    assert "🔴 SCORE IN A RATIO COLUMN" in row["china"]
    assert "🔴 ALMOST ALL IMPUTED" in row["us"] and "40 (4.0%)" in row["us"]
    assert row["india"].endswith("fundamentals present |") and "290 (97.0%)" in row["india"]
    assert row["korea"].endswith("| — |")
    order = [md.index(f"\n| {m} |") for m in ("us", "japan", "india", "china", "korea")]
    assert order == sorted(order)  # markets are listed by row count, largest first
    assert "## Notes" in md and "- 🔴 note one" in md and "- 🔴 note two" in md
    assert md.rstrip().endswith("while looking like a clean pass.")


def test_render_omits_empty_sections(cg):
    md = cg.render(_state(summary="S"))
    assert "## 5. Cassandra" not in md and "## Notes" not in md
    for h in ("## 1. Data completeness", "## 2. Claim consistency", "## 3. Contamination gate", "## 4. Breadth and power"):
        assert h in md


def test_main_json_and_markdown(cg, monkeypatch, capsys, tmp_path):
    _fake_langgraph(monkeypatch)
    for name in ("inventory", "audit_claims", "audit_gates", "audit_power", "audit_cassandra"):
        monkeypatch.setattr(cg, name, (lambda n: lambda s: {"notes": [f"from {n}"]} if n == "inventory" else {})(name))
    monkeypatch.setattr(sys, "argv", ["completeness_graph.py", "--json"])
    assert cg.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data["notes"] == ["from inventory"] and "summary" in data and data["_order"][0] == "inventory"
    assert not cg.OUT_MD.exists()  # --json does not write the report
    monkeypatch.setattr(sys, "argv", ["completeness_graph.py"])
    assert cg.main() == 0
    out = capsys.readouterr().out
    assert f"wrote {cg.OUT_MD}" in out and "# Completeness audit" in out
    text = cg.OUT_MD.read_text()
    assert "- 🔴 from inventory" in text and text.startswith("# Completeness audit")
