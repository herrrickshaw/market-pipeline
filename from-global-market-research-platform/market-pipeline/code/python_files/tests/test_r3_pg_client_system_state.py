"""Offline tests for pg_client (connection + bulk-write primitives) and system_state (state registry)."""

import importlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from r3_fakepg import FakeConn

# ====================================================================== pg_client


@pytest.fixture
def pg(monkeypatch):
    ps = types.ModuleType("psycopg2")
    extras = types.ModuleType("psycopg2.extras")
    ps.extras = extras
    ps.connect_calls = []
    ps.next_conn = FakeConn()
    ps.fail_connect = None
    inserts = []

    def connect(dsn):
        ps.connect_calls.append(dsn)
        if ps.fail_connect:
            raise ps.fail_connect
        return ps.next_conn

    def execute_values(cur, sql, rows):
        inserts.append((" ".join(sql.split()), list(rows)))
        if ps.next_conn.responder("EXECUTE_VALUES", None) is not None:
            raise ps.next_conn.responder("EXECUTE_VALUES", None)

    ps.connect = connect
    extras.execute_values = execute_values
    ps.inserts = inserts
    monkeypatch.setitem(sys.modules, "psycopg2", ps)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    sys.modules.pop("pg_client", None)
    mod = importlib.import_module("pg_client")
    yield mod, ps
    mod.close()
    sys.modules.pop("pg_client", None)


def test_connect_idempotent_and_state(pg):
    mod, ps = pg
    assert mod.is_available() is False and mod.get_connection() is None
    assert mod.connect("dsn-1") is True
    assert mod.is_available() is True and mod.get_connection() is ps.next_conn
    assert ps.next_conn.sqls() == ["SELECT 1"]
    assert mod.connect("dsn-2") is True  # already available: no second connection
    assert ps.connect_calls == ["dsn-1"]


def test_connect_uses_default_dsn_and_failure_is_swallowed(pg):
    mod, ps = pg
    ps.fail_connect = RuntimeError("no server")
    assert mod.connect() is False
    assert ps.connect_calls == [mod.DSN] and mod.get_connection() is None and mod.is_available() is False
    ps.fail_connect = None
    ps.next_conn = FakeConn(lambda sql, params: RuntimeError("select failed") if sql == "SELECT 1" else None)
    assert mod.connect() is False  # SELECT 1 failing also reads as unavailable
    assert mod.get_connection() is None


def test_close_tolerates_errors(pg):
    mod, ps = pg
    mod.connect()
    conn = ps.next_conn
    mod.close()
    assert conn.closed and mod.get_connection() is None and not mod.is_available()
    mod.close()  # second close is a no-op

    class Bad(FakeConn):
        def close(self):
            raise RuntimeError("already dead")

    ps.next_conn = Bad()
    mod.connect()
    mod.close()  # must not raise
    assert mod.get_connection() is None


def test_pg_type_mapping(pg):
    mod, _ = pg
    assert mod.pg_type("DOUBLE") == "double precision" and mod.pg_type("double") == "double precision"
    assert mod.pg_type("FLOAT") == "real" and mod.pg_type("HUGEINT") == "numeric" and mod.pg_type("BLOB") == "bytea"
    assert mod.pg_type("VARCHAR") == "VARCHAR" and mod.pg_type("DECIMAL(10,2)") == "DECIMAL(10,2)"


def test_to_rows_converts_numpy_and_missing(pg):
    mod, _ = pg
    df = pd.DataFrame({"i": np.array([1, 2], dtype=np.int64), "f": [1.5, np.nan], "b": np.array([True, False]),
                       "s": ["x", None], "n": pd.array([1, None], dtype="Int64")})
    rows = mod.to_rows(df, ["i", "f", "b", "s", "n"])
    assert rows == [(1, 1.5, True, "x", 1), (2, None, False, None, None)]
    assert type(rows[0][0]) is int and type(rows[0][1]) is float and type(rows[0][2]) is bool
    assert mod.to_rows(df, ["s", "i"])[0] == ("x", 1)  # column order is the requested one
    assert mod.to_rows(df.iloc[0:0], ["i"]) == []


def test_ensure_schema(pg):
    mod, ps = pg
    mod.ensure_schema(["CREATE X"])  # not connected: no-op, must not raise
    def responder(sql, params):
        if sql.startswith("BAD"):
            return RuntimeError("exists")

    ps.next_conn = FakeConn(responder)
    mod.connect()
    ps.next_conn.log.clear()
    mod.ensure_schema(["  CREATE A  ", "", "   ", "BAD STMT", "CREATE B"])
    assert ps.next_conn.sqls() == ["CREATE A", "BAD STMT", "CREATE B"]
    assert ps.next_conn.commits == 2 and ps.next_conn.rollbacks == 1  # the failing one rolled back, rest continued


def test_replace_table_atomic(pg):
    mod, ps = pg
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert mod.replace_table_atomic("s", "t", df, {"a": "int", "b": "text"}) == 0  # no connection
    mod.connect()
    ps.next_conn.log.clear()
    assert mod.replace_table_atomic("s", "t", pd.DataFrame(), {"a": "int"}) == 0
    assert mod.replace_table_atomic("s", "t", None, {"a": "int"}) == 0
    assert ps.next_conn.log == []
    n = mod.replace_table_atomic("s", "t", df, {"a": "integer", "b": "text"})
    assert n == 2
    assert ps.next_conn.sqls() == [
        'CREATE SCHEMA IF NOT EXISTS "s"',
        'DROP TABLE IF EXISTS "s"."t__incoming"',
        'CREATE TABLE "s"."t__incoming" ("a" integer, "b" text)',
        'DROP TABLE IF EXISTS "s"."t"',
        'ALTER TABLE "s"."t__incoming" RENAME TO "t"',
    ]
    assert ps.inserts[-1] == ('INSERT INTO "s"."t__incoming" ("a", "b") VALUES %s', [(1, "x"), (2, "y")])
    assert ps.next_conn.commits == 1 and ps.next_conn.rollbacks == 0


def test_replace_table_atomic_rolls_back_and_raises(pg):
    mod, ps = pg
    ps.next_conn = FakeConn(lambda sql, params: RuntimeError("ddl failed") if sql.startswith("DROP TABLE IF EXISTS \"s\".\"t\"") and "__incoming" not in sql else None)
    mod.connect()
    with pytest.raises(RuntimeError, match="ddl failed"):
        mod.replace_table_atomic("s", "t", pd.DataFrame({"a": [1]}), {"a": "int"})
    assert ps.next_conn.rollbacks == 1 and ps.next_conn.commits == 0


def test_delete_and_insert(pg):
    mod, ps = pg
    assert mod.delete_and_insert("s", "t", "k = %s", (1,), [(1,)], ["k"]) == 0  # no connection
    mod.connect()
    ps.next_conn.log.clear()
    n = mod.delete_and_insert("s", "t", "k = %s", (7,), [(1, "a"), (2, "b")], ["k", "v"])
    assert n == 2
    assert ps.next_conn.log == [('DELETE FROM "s"."t" WHERE k = %s', (7,))]
    assert ps.inserts[-1] == ('INSERT INTO "s"."t" ("k", "v") VALUES %s', [(1, "a"), (2, "b")])
    assert ps.next_conn.commits == 1
    ps.next_conn.log.clear()
    n = mod.delete_and_insert("s", "t", "ignored", (), [(1,)], ["k"], truncate=True)
    assert n == 1 and ps.next_conn.sqls() == ['TRUNCATE "s"."t"']
    # empty rows: the delete still runs, nothing inserted, still committed
    ps.next_conn.log.clear()
    ps.inserts.clear()
    assert mod.delete_and_insert("s", "t", "k = 1", (), [], ["k"]) == 0
    assert ps.next_conn.sqls() == ['DELETE FROM "s"."t" WHERE k = 1'] and ps.inserts == []
    assert ps.next_conn.commits == 3


def test_delete_and_insert_failure_rolls_back_delete(pg):
    mod, ps = pg
    mod.connect()
    ps.next_conn.responder = lambda sql, params: RuntimeError("insert failed") if sql == "EXECUTE_VALUES" else None
    with pytest.raises(RuntimeError, match="insert failed"):
        mod.delete_and_insert("s", "t", "k = 1", (), [(1,)], ["k"])
    assert ps.next_conn.rollbacks == 1 and ps.next_conn.commits == 0


def test_append_rows(pg):
    mod, ps = pg
    assert mod.append_rows("s", "t", ["a"], [(1,)]) == 0
    mod.connect()
    assert mod.append_rows("s", "t", ["a"], []) == 0 and ps.inserts == []
    assert mod.append_rows("s", "t", ["a", "b"], [(1, 2), (3, 4)]) == 2
    assert ps.inserts[-1] == ('INSERT INTO "s"."t" ("a", "b") VALUES %s', [(1, 2), (3, 4)])
    assert ps.next_conn.commits == 1
    ps.next_conn.responder = lambda sql, params: RuntimeError("x") if sql == "EXECUTE_VALUES" else None
    with pytest.raises(RuntimeError):
        mod.append_rows("s", "t", ["a"], [(1,)])
    assert ps.next_conn.rollbacks == 1


def test_upsert_rows(pg):
    mod, ps = pg
    assert mod.upsert_rows("s", "t", ["k", "v"], [(1, 2)], ["k"]) == 0
    mod.connect()
    assert mod.upsert_rows("s", "t", ["k", "v"], [], ["k"]) == 0
    assert mod.upsert_rows("s", "t", ["k", "v", "w"], [(1, 2, 3)], ["k"]) == 1
    assert ps.inserts[-1][0] == ('INSERT INTO "s"."t" ("k", "v", "w") VALUES %s ON CONFLICT ("k") '
                                 'DO UPDATE SET "v" = EXCLUDED."v", "w" = EXCLUDED."w"')
    mod.upsert_rows("s", "t", ["k", "v", "w"], [(1, 2, 3)], ["k"], update_cols=["w"])
    assert ps.inserts[-1][0].endswith('DO UPDATE SET "w" = EXCLUDED."w"')
    mod.upsert_rows("s", "t", ["k", "v"], [(1, 2)], ["k"], update_cols=[])
    assert ps.inserts[-1][0].endswith('ON CONFLICT ("k") DO NOTHING')
    mod.upsert_rows("s", "t", ["a", "b", "v"], [(1, 2, 3)], ["a", "b"])
    assert 'ON CONFLICT ("a", "b")' in ps.inserts[-1][0]
    ps.next_conn.responder = lambda sql, params: RuntimeError("x") if sql == "EXECUTE_VALUES" else None
    with pytest.raises(RuntimeError):
        mod.upsert_rows("s", "t", ["k", "v"], [(1, 2)], ["k"])
    assert ps.next_conn.rollbacks == 1


# ==================================================================== system_state


class StubPG:
    """pg_client stand-in injected into system_state."""

    def __init__(self, conn=None, ok=True):
        self.conn = conn or FakeConn()
        self.ok = ok
        self.ensured = []

    def connect(self):
        return self.ok

    def ensure_schema(self, stmts):
        self.ensured.append(list(stmts))

    def get_connection(self):
        return self.conn


@pytest.fixture
def ss(monkeypatch):
    stub = types.ModuleType("pg_client")
    stub.connect = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, "pg_client", stub)
    sys.modules.pop("system_state", None)
    mod = importlib.import_module("system_state")
    monkeypatch.setattr(mod, "pg_client", StubPG())
    yield mod
    sys.modules.pop("system_state", None)


def _use(ss, monkeypatch, responder=None, ok=True):
    stub = StubPG(FakeConn(responder), ok=ok)
    monkeypatch.setattr(ss, "pg_client", stub)
    return stub


def test_ensure_bootstraps_schema_once_per_call(ss, monkeypatch):
    stub = _use(ss, monkeypatch)
    assert ss._ensure() is True
    assert len(stub.ensured) == 2
    stmts = stub.ensured[0]
    assert all(s and not s.endswith(";") for s in stmts) and any("CREATE SCHEMA" in s for s in stmts)
    assert len([s for s in stmts if s.startswith("CREATE TABLE")]) == 4
    assert stub.ensured[1][0].startswith("CREATE OR REPLACE VIEW")
    stub2 = _use(ss, monkeypatch, ok=False)
    assert ss._ensure() is False and stub2.ensured == []


def test_every_reader_and_writer_noops_without_postgres(ss, monkeypatch):
    stub = _use(ss, monkeypatch, ok=False)
    assert ss.list_data_sources() == [] and ss.all_data_snapshots() == []
    assert ss.latest_data_snapshot("k") is None and ss.latest_analysis_runs() == []
    assert ss.connection_status() == []
    ss.register_data_source("k", "t", "loc")
    ss.record_data_snapshot("k", "ok")
    ss.record_analysis_run("a", "ok")
    ss.record_connection_check("postgres", True)
    assert stub.conn.log == []


def test_list_data_sources_filters_by_kind(ss, monkeypatch):
    cols = ["source_key", "kind", "location", "date_column", "critical", "cadence", "owner_script", "notes"]
    rows = [("a", "postgres_table", "s.a", "d", True, "daily", None, None)]
    stub = _use(ss, monkeypatch, lambda sql, p: (cols, rows))
    out = ss.list_data_sources(kind="postgres_table")
    assert out == [dict(zip(cols, rows[0]))]
    sql, params = stub.conn.log[-1]
    assert "WHERE kind = %s" in sql and params == ("postgres_table",)
    ss.list_data_sources()
    sql, params = stub.conn.log[-1]
    assert "WHERE" not in sql and "ORDER BY source_key" in sql and params is None


def test_register_data_source(ss, monkeypatch):
    stub = _use(ss, monkeypatch)
    ss.register_data_source("k", "local_file", "/x", date_column="d", critical=True, cadence="daily",
                            owner_script="s.py", notes="n")
    sql, params = stub.conn.log[-1]
    assert "ON CONFLICT (source_key) DO UPDATE" in sql
    assert params == ("k", "local_file", "/x", "d", True, "daily", "s.py", "n")
    assert stub.conn.commits == 1
    ss.register_data_source("k2", "t", "l")
    assert stub.conn.log[-1][1] == ("k2", "t", "l", None, False, None, None, None)  # defaults
    _use(ss, monkeypatch, lambda sql, p: RuntimeError("db down") if "INSERT" in sql else None)
    stub = ss.pg_client
    ss.register_data_source("k", "t", "l")  # swallowed
    assert stub.conn.rollbacks == 1 and stub.conn.commits == 0


def test_record_data_snapshot_json_and_rollback(ss, monkeypatch):
    stub = _use(ss, monkeypatch)
    ss.record_data_snapshot("ds", "ok", rows=5, coverage_pct=99.5, stale_days=2, detail={"a": 1})
    sql, params = stub.conn.log[-1]
    assert "INSERT INTO \"system_state\".data_snapshots" in sql
    assert params == ("ds", "ok", 5, 99.5, 2, '{"a": 1}')
    ss.record_data_snapshot("ds", "ok")
    assert stub.conn.log[-1][1] == ("ds", "ok", None, None, None, None)
    stub = _use(ss, monkeypatch, lambda sql, p: RuntimeError("x") if "INSERT" in sql else None)
    ss.record_data_snapshot("ds", "ok")
    assert stub.conn.rollbacks == 1


def test_latest_and_all_data_snapshots(ss, monkeypatch):
    cols = ["dataset_key", "run_at", "status", "rows", "coverage_pct", "stale_days", "detail"]
    row = ("ds", "2026-01-01", "ok", 5, 1.0, 0, None)
    stub = _use(ss, monkeypatch, lambda sql, p: (cols, [row]))
    assert ss.latest_data_snapshot("ds") == dict(zip(cols, row))
    assert stub.conn.log[-1][1] == ("ds",) and "ORDER BY run_at DESC LIMIT 1" in stub.conn.log[-1][0]
    _use(ss, monkeypatch, lambda sql, p: (cols, []))
    assert ss.latest_data_snapshot("nope") is None
    _use(ss, monkeypatch, lambda sql, p: (cols, [row, row]))
    assert len(ss.all_data_snapshots()) == 2


def test_record_and_read_analysis_runs(ss, monkeypatch):
    stub = _use(ss, monkeypatch)
    ss.record_analysis_run("a", "ok", summary={"n": 1, "d": Path("/x")}, output_path="/o", duration_s=1.5, error="e")
    sql, params = stub.conn.log[-1]
    assert params[0:2] == ("a", "ok") and json.loads(params[2]) == {"n": 1, "d": "/x"}  # default=str
    assert params[3:] == ("/o", 1.5, "e")
    ss.record_analysis_run("a", "ok")
    assert stub.conn.log[-1][1] == ("a", "ok", None, None, None, None)
    cols = ["analysis_key", "run_at", "status", "summary", "output_path", "duration_s", "error"]
    stub = _use(ss, monkeypatch, lambda sql, p: (cols, [("a", 1, "ok", None, None, None, None)]))
    assert ss.latest_analysis_runs("a", limit=3)[0]["analysis_key"] == "a"
    assert stub.conn.log[-1][1] == ("a", 3) and "WHERE analysis_key = %s" in stub.conn.log[-1][0]
    ss.latest_analysis_runs()
    assert stub.conn.log[-1][1] == (20,) and "WHERE" not in stub.conn.log[-1][0]
    stub = _use(ss, monkeypatch, lambda sql, p: RuntimeError("x") if "INSERT" in sql else None)
    ss.record_analysis_run("a", "ok")
    assert stub.conn.rollbacks == 1


def test_connection_checks(ss, monkeypatch):
    stub = _use(ss, monkeypatch)
    ss.record_connection_check("postgres", True, latency_ms=3.2, error=None, checked_by="x.py")
    assert stub.conn.log[-1][1] == ("postgres", True, 3.2, None, "x.py")
    cols = ["target", "checked_at", "available", "latency_ms", "error", "checked_by"]
    stub = _use(ss, monkeypatch, lambda sql, p: (cols, [("postgres", 1, True, 1.0, None, "x")]))
    assert ss.connection_status("postgres") == [dict(zip(cols, ("postgres", 1, True, 1.0, None, "x")))]
    assert stub.conn.log[-1][1] == ("postgres",) and "LIMIT 1" in stub.conn.log[-1][0]
    ss.connection_status()
    assert "DISTINCT ON (target)" in stub.conn.log[-1][0]
    stub = _use(ss, monkeypatch, lambda sql, p: RuntimeError("x") if "INSERT" in sql else None)
    ss.record_connection_check("postgres", False, error="e")
    assert stub.conn.rollbacks == 1


def test_check_postgres_records_latency_and_failure(ss, monkeypatch):
    recorded = []
    monkeypatch.setattr(ss, "record_connection_check", lambda *a, **k: recorded.append((a, k)))
    stub = _use(ss, monkeypatch, ok=True)
    assert ss.check_postgres() == (True, None)
    assert recorded[0][0] == ("postgres", True) and recorded[0][1]["checked_by"] == "system_state.py"
    assert recorded[0][1]["latency_ms"] >= 0
    stub.ok = False
    assert ss.check_postgres() == (False, None)  # unreachable, no exception: ok False, no error text
    stub.connect = lambda: (_ for _ in ()).throw(RuntimeError("driver exploded"))
    assert ss.check_postgres() == (False, "driver exploded")
    assert recorded[-1][0] == ("postgres", False) and recorded[-1][1]["error"] == "driver exploded"


def test_check_cassandra(ss, monkeypatch):
    recorded = []
    monkeypatch.setattr(ss, "record_connection_check", lambda *a, **k: recorded.append((a, k)))
    mods = {}
    for n in ("cassandra", "cassandra.cluster"):
        mods[n] = types.ModuleType(n)
    seen = {}

    class Cluster:
        def __init__(self, hosts, connect_timeout=None):
            seen["args"] = (hosts, connect_timeout)

        def connect(self, ks):
            seen["ks"] = ks

    mods["cassandra.cluster"].Cluster = Cluster
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    assert ss.check_cassandra() == (True, None)
    assert seen == {"args": (["127.0.0.1"], 8), "ks": "herrrickshaw"}
    assert recorded[-1][0] == ("cassandra", True)

    class Bad(Cluster):
        def connect(self, ks):
            raise RuntimeError("no nodes")

    mods["cassandra.cluster"].Cluster = Bad
    assert ss.check_cassandra() == (False, "no nodes")
    assert recorded[-1][0] == ("cassandra", False) and recorded[-1][1]["error"] == "no nodes"
    monkeypatch.setitem(sys.modules, "cassandra.cluster", None)  # driver missing -> ImportError path
    ok, err = ss.check_cassandra()
    assert ok is False and err


def test_list_locations_merges_registries_and_sources(ss, monkeypatch):
    dr = types.ModuleType("data_registry")
    dr.DATASETS = [types.SimpleNamespace(key="k1", path=Path("/d/1"), writer="w.py", section="S")]
    rr = types.ModuleType("repo_registry")
    for n in ("MARKET_PIPELINE", "GLOBAL_MARKET_DATA", "GLOBAL_STOCK_SCREENER", "OHLCV_ADJ_IN"):
        setattr(rr, n, Path("/r") / n)
    monkeypatch.setitem(sys.modules, "data_registry", dr)
    monkeypatch.setitem(sys.modules, "repo_registry", rr)
    cols = ["source_key", "kind", "location", "date_column", "critical", "cadence", "owner_script", "notes"]
    rows = [("f1", "local_file", "/f1", None, False, None, "own.py", None),
            ("t1", "postgres_table", "s.t", None, False, None, None, None),
            ("d1", "local_dir", "/d1", None, False, None, None, None),
            ("r1", "sibling_repo_sqlite", "/r1.sqlite", None, False, None, None, None)]
    _use(ss, monkeypatch, lambda sql, p: (cols, rows))
    out = ss.list_locations()
    keys = [o["key"] for o in out]
    assert keys[:1] == ["k1"] and "repo.market_pipeline" in keys and "repo.ohlcv_adj_in" in keys
    assert "t1" not in keys and {"f1", "d1", "r1"} <= set(keys)  # only file-like source kinds are merged
    f1 = next(o for o in out if o["key"] == "f1")
    assert f1["writer"] == "own.py" and f1["kind"] == "local_file"
    assert out[0] == {"key": "k1", "path": "/d/1", "kind": "local", "writer": "w.py", "section": "S"}
    # missing registries are tolerated
    monkeypatch.setitem(sys.modules, "data_registry", None)
    monkeypatch.setitem(sys.modules, "repo_registry", None)
    assert [o["key"] for o in ss.list_locations()] == ["f1", "d1", "r1"]


def test_etl_registry_summary(ss, monkeypatch, tmp_path):
    monkeypatch.setattr(ss.Path, "home", classmethod(lambda cls: tmp_path))
    assert ss.etl_registry_summary() is None  # file missing
    d = tmp_path / "repos" / "global-market-data" / "warehouse"
    d.mkdir(parents=True)
    db = d / "etl_registry.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table audits (run_at text, files_checked int, missing_local int, sha_mismatch int, "
                "cloud_missing int, dupe_groups int, wasted_bytes int, stale_sources int, result text)")
    con.commit()
    con.close()
    assert ss.etl_registry_summary() is None  # no rows
    con = sqlite3.connect(db)
    con.execute("insert into audits values ('2026-01-01', 10, 1, 0, 0, 2, 100, 0, 'old')")
    con.execute("insert into audits values ('2026-02-01', 20, 0, 1, 0, 3, 200, 1, 'new')")
    con.commit()
    con.close()
    r = ss.etl_registry_summary()
    assert r["run_at"] == "2026-02-01" and r["files_checked"] == 20 and r["result"] == "new"  # latest row
    db.write_bytes(b"not a database")
    assert ss.etl_registry_summary() is None  # unreadable -> None


def test_get_state_dispatch(ss, monkeypatch):
    monkeypatch.setattr(ss, "latest_data_snapshot", lambda k: ("one", k))
    monkeypatch.setattr(ss, "all_data_snapshots", lambda: "all")
    monkeypatch.setattr(ss, "latest_analysis_runs", lambda k=None: ("runs", k))
    monkeypatch.setattr(ss, "list_data_sources", lambda kind=None: ("src", kind))
    monkeypatch.setattr(ss, "list_locations", lambda: "locs")
    monkeypatch.setattr(ss, "connection_status", lambda target=None: ("conn", target))
    assert ss.get_state("data", "k") == ("one", "k") and ss.get_state("data") == "all"
    assert ss.get_state("analysis", "a") == ("runs", "a") and ss.get_state("sources", "x") == ("src", "x")
    assert ss.get_state("locations") == "locs" and ss.get_state("connections", "pg") == ("conn", "pg")
    with pytest.raises(ValueError, match="unknown category: nope"):
        ss.get_state("nope")


def _stub_state(ss, monkeypatch):
    monkeypatch.setattr(ss, "get_state", lambda c, k=None: [{"c": c, "i": i} for i in range(25)] if c == "sources" else [{"c": c}])
    monkeypatch.setattr(ss, "etl_registry_summary", lambda: {"result": "ok"})


def test_print_status_text_and_json(ss, monkeypatch, capsys):
    _stub_state(ss, monkeypatch)
    assert ss._print_status(None, False) == 0
    out = capsys.readouterr().out
    assert "=== SOURCES (25) ===" in out and "… 5 more" in out and out.count("{'c': 'sources'") == 20
    assert "=== LOCATIONS (1) ===" in out and "SIBLING ETL_REGISTRY" in out and "{'result': 'ok'}" in out
    assert ss._print_status("data", False) == 0
    out = capsys.readouterr().out
    assert "=== DATA (1) ===" in out and "SIBLING ETL_REGISTRY" not in out  # only when no category given
    assert ss._print_status("locations", True) == 0
    assert json.loads(capsys.readouterr().out) == {"locations": [{"c": "locations"}]}
    monkeypatch.setattr(ss, "etl_registry_summary", lambda: None)
    ss._print_status(None, False)
    assert "unavailable (file missing or unreadable)" in capsys.readouterr().out


def test_main_cli(ss, monkeypatch, capsys):
    _stub_state(ss, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["system_state.py", "--status", "--category", "data", "--json"])
    assert ss.main() == 0
    assert json.loads(capsys.readouterr().out) == {"data": [{"c": "data"}]}
    monkeypatch.setattr(ss, "check_postgres", lambda: (True, None))
    monkeypatch.setattr(ss, "check_cassandra", lambda: (False, "no nodes"))
    monkeypatch.setattr(sys, "argv", ["system_state.py", "--check-connections"])
    assert ss.main() == 0
    out = capsys.readouterr().out
    assert "postgres:  OK" in out and "cassandra: UNAVAILABLE — no nodes" in out
    monkeypatch.setattr(sys, "argv", ["system_state.py", "--check-connections", "--json"])
    ss.main()
    assert json.loads(capsys.readouterr().out) == {"postgres": {"available": True, "error": None},
                                                    "cassandra": {"available": False, "error": "no nodes"}}
    monkeypatch.setattr(ss, "check_postgres", lambda: (False, "down"))
    monkeypatch.setattr(sys, "argv", ["system_state.py", "--check-connections"])
    ss.main()
    assert "postgres:  UNAVAILABLE — down" in capsys.readouterr().out
