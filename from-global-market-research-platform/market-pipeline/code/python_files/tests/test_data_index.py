"""data_index.py -- the freshness gate: file coverage, OHLC data-date coverage, status, report/require."""

import datetime as dt
import json
import os
import sys
import types
from pathlib import Path

import pytest

import data_index as DI
import data_registry as R

NOW = 1_800_000_000.0  # frozen "time.time()" for the file-age maths
DAY = 86400.0


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(DI.time, "time", lambda: NOW)


def touch(path, age_days):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    m = NOW - age_days * DAY
    os.utime(path, (m, m))
    return path


def ds(path, key="k", max_age=1.0, section="ingest", writer="w.py", note="", **kw):
    return R.Dataset(key, Path(path), writer, section, "daily", max_age, note, **kw)


# ── mtime helpers ─────────────────────────────────────────────────────────────
def test_mtimes_for_missing_file_and_directory(tmp_path, frozen):
    assert DI._mtimes(tmp_path / "nope") == []
    f = touch(tmp_path / "a.txt", 2)
    assert DI._mtimes(f) == [NOW - 2 * DAY]
    touch(tmp_path / "d" / "x.txt", 1)
    touch(tmp_path / "d" / "sub" / "y.txt", 3)
    (tmp_path / "d" / "empty").mkdir()
    assert sorted(DI._mtimes(tmp_path / "d")) == [NOW - 3 * DAY, NOW - 1 * DAY]


def test_newest_mtime_is_max_or_none(tmp_path, frozen):
    touch(tmp_path / "d" / "x", 1)
    touch(tmp_path / "d" / "y", 5)
    assert DI._newest_mtime(tmp_path / "d") == NOW - 1 * DAY
    assert DI._newest_mtime(tmp_path / "missing") is None
    (tmp_path / "emptydir").mkdir()
    assert DI._newest_mtime(tmp_path / "emptydir") is None


def test_age_days(frozen):
    assert DI._age_days(None) is None
    assert DI._age_days(NOW - 36 * 3600) == pytest.approx(1.5)
    assert DI._age_days(NOW) == 0.0


def test_coverage_fraction_with_inclusive_cutoff(tmp_path, frozen):
    for i, age in enumerate([0.5, 1.0, 1.5, 3.0]):
        touch(tmp_path / "d" / f"f{i}", age)
    # max_age 1.0: the file exactly 1 day old counts as fresh (>= cutoff)
    assert DI._coverage(tmp_path / "d", 1.0) == pytest.approx(0.5)
    assert DI._coverage(tmp_path / "d", 3.0) == 1.0
    assert DI._coverage(tmp_path / "d", 0.4) == 0.0
    assert DI._coverage(tmp_path / "none", 1.0) is None


def test_coverage_single_file_is_zero_or_one(tmp_path, frozen):
    f = touch(tmp_path / "f", 2)
    assert DI._coverage(f, 3) == 1.0 and DI._coverage(f, 1) == 0.0


# ── postgres freshness ────────────────────────────────────────────────────────
def fake_pg(monkeypatch, rows=None, boom=False, name="psycopg2"):
    class Cur:
        def execute(self, q):
            if boom:
                raise RuntimeError("down")
            self.q = q

        def fetchall(self):
            return rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn(Cur):
        def cursor(self):
            return Cur()

    mod = types.SimpleNamespace(connect=lambda dsn: Conn())
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.setitem(sys.modules, name, mod)


def test_pg_freshness_without_driver_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    assert DI._pg_freshness() == {}


@pytest.mark.parametrize("driver", ["psycopg2", "psycopg"])
def test_pg_freshness_maps_market_to_latest_date(monkeypatch, driver):
    fake_pg(monkeypatch, rows=[("IN", dt.date(2026, 7, 17)), ("US", "2026-07-16")], name=driver)
    assert DI._pg_freshness() == {"IN": "2026-07-17", "US": "2026-07-16"}


def test_pg_freshness_query_failure_is_empty(monkeypatch):
    fake_pg(monkeypatch, boom=True)
    assert DI._pg_freshness() == {}


# ── OHLC data-date coverage ───────────────────────────────────────────────────
def write_index(tmp_path, monkeypatch, entries):
    d = tmp_path / "meta"
    d.mkdir(exist_ok=True)
    (d / "cache_index.json").write_text(json.dumps(entries))
    monkeypatch.setattr(R, "META_DIR", d)


def days_ago(n):
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


def test_ohlc_coverage_uniform_cache(tmp_path, monkeypatch):
    write_index(tmp_path, monkeypatch, {f"ohlc:T{i}": {"to": days_ago(1)} for i in range(10)})
    out = DI._ohlc_data_coverage(2.0)
    assert out == {
        "coverage": 1.0,
        "total": 10,
        "fresh": 10,
        "newest": days_ago(1),
        "max_bar": days_ago(1),
        "lag_days": 1,
        "edge_stale": False,
    }


def test_ohlc_coverage_anchors_on_median_not_max(tmp_path, monkeypatch):
    # 97 tickers at the healthy edge (3 days ago), 3 stragglers carrying a newer bar
    entries = {f"ohlc:T{i}": {"to": days_ago(3)} for i in range(97)}
    entries.update({f"ohlc:N{i}": {"to": days_ago(0)} for i in range(3)})
    write_index(tmp_path, monkeypatch, entries)
    out = DI._ohlc_data_coverage(1.0)
    assert out["newest"] == days_ago(3) and out["max_bar"] == days_ago(0)
    assert out["coverage"] == 1.0 and out["fresh"] == 100
    assert out["lag_days"] == 3 and out["edge_stale"] is False  # lag measured from the median edge


def test_ohlc_coverage_counts_behind_and_unparseable_tickers(tmp_path, monkeypatch):
    entries = {f"ohlc:A{i}": {"to": days_ago(1)} for i in range(6)}
    entries.update(
        {
            "ohlc:B0": {"to": days_ago(8)},
            "ohlc:B1": {"to": days_ago(8)},
            "ohlc:BAD": {"to": "garbage"},
            "ohlc:NONE": {},
            "other:x": {"to": days_ago(30)},
        }
    )
    write_index(tmp_path, monkeypatch, entries)
    out = DI._ohlc_data_coverage(2.0)
    assert out["total"] == 10  # non-ohlc key ignored, bad ones counted
    assert out["fresh"] == 6 and out["coverage"] == pytest.approx(0.6)
    assert out["newest"] == days_ago(1)


def test_assess_ohlc_exactly_eighty_percent_uniform_is_healthy(tmp_path, monkeypatch, frozen):
    f = touch(tmp_path / "cache", 0.01)
    entries = {f"ohlc:A{i}": {"to": days_ago(1)} for i in range(8)}
    entries.update({f"ohlc:B{i}": {"to": days_ago(9)} for i in range(2)})
    write_index(tmp_path, monkeypatch, entries)
    a = DI.assess(ds(f, key="cache.ohlc", max_age=2.0))
    assert a["coverage"] == pytest.approx(0.8) and a["status"] == "OK"


def test_ohlc_coverage_cutoff_is_inclusive(tmp_path, monkeypatch):
    entries = {f"ohlc:A{i}": {"to": days_ago(1)} for i in range(5)}
    entries["ohlc:EDGE"] = {"to": days_ago(3)}  # exactly newest - 2d
    entries["ohlc:OUT"] = {"to": days_ago(4)}
    write_index(tmp_path, monkeypatch, entries)
    out = DI._ohlc_data_coverage(2.0)
    assert out["fresh"] == 6 and out["total"] == 7


def test_ohlc_edge_stale_boundary_is_max_age_plus_four(tmp_path, monkeypatch):
    write_index(tmp_path, monkeypatch, {"ohlc:A": {"to": days_ago(6)}})
    assert DI._ohlc_data_coverage(2.0)["edge_stale"] is False  # lag 6 == 2 + 4
    write_index(tmp_path, monkeypatch, {"ohlc:A": {"to": days_ago(7)}})
    assert DI._ohlc_data_coverage(2.0)["edge_stale"] is True


def test_ohlc_coverage_accepts_timestamps_and_degenerate_files(tmp_path, monkeypatch):
    write_index(tmp_path, monkeypatch, {"ohlc:A": {"to": days_ago(1) + "T10:00:00"}})
    assert DI._ohlc_data_coverage(2.0)["newest"] == days_ago(1)
    write_index(tmp_path, monkeypatch, {"other": {}})
    assert DI._ohlc_data_coverage(2.0) is None  # no ohlc keys
    write_index(tmp_path, monkeypatch, {"ohlc:A": {"to": "nope"}, "ohlc:B": {}})
    assert DI._ohlc_data_coverage(2.0) is None  # nothing parseable
    monkeypatch.setattr(R, "META_DIR", tmp_path / "missing")
    assert DI._ohlc_data_coverage(2.0) is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "cache_index.json").write_text("{bad")
    monkeypatch.setattr(R, "META_DIR", tmp_path / "bad")
    assert DI._ohlc_data_coverage(2.0) is None


# ── assess ────────────────────────────────────────────────────────────────────
def test_assess_missing(tmp_path, frozen):
    a = DI.assess(ds(tmp_path / "nope"))
    assert a == {
        "key": "k",
        "status": "MISSING",
        "age_days": None,
        "detail": str(tmp_path / "nope"),
    }


def test_assess_no_max_age_is_ok_with_age(tmp_path, frozen):
    f = touch(tmp_path / "log", 40)
    a = DI.assess(ds(f, max_age=None))
    assert a["status"] == "OK" and a["age_days"] == pytest.approx(40.0) and a["detail"] == ""


def test_assess_single_file_fresh_and_stale(tmp_path, frozen):
    f = touch(tmp_path / "f", 0.5)
    ok = DI.assess(ds(f, max_age=1.0))
    assert ok["status"] == "OK" and ok["coverage"] == 1.0 and ok["detail"] == ""
    g = touch(tmp_path / "g", 3)
    st = DI.assess(ds(g, max_age=1.0))
    assert (
        st["status"] == "STALE"
        and st["coverage"] == 0.0
        and "0% of 1 files within 1d" in st["detail"]
    )


def test_assess_directory_needs_eighty_percent_coverage(tmp_path, frozen):
    for i in range(5):
        touch(tmp_path / "d4" / f"f{i}", 0.5 if i < 4 else 5)  # 4/5 = 80% -> OK
    a = DI.assess(ds(tmp_path / "d4", max_age=1.0))
    assert a["status"] == "OK" and a["coverage"] == pytest.approx(0.8)
    assert a["detail"] == "80% of 5 files within 1d (need 80%) — newest is 0.5d old"
    for i in range(5):
        touch(tmp_path / "d3" / f"f{i}", 0.5 if i < 3 else 5)  # 60% -> STALE
    b = DI.assess(ds(tmp_path / "d3", max_age=1.0))
    assert b["status"] == "STALE" and b["coverage"] == pytest.approx(0.6)
    assert "60% of 5 files within 1d" in b["detail"]


def test_assess_newest_file_cannot_mask_a_stale_directory(tmp_path, frozen):
    touch(tmp_path / "d" / "fresh", 0.01)
    for i in range(9):
        touch(tmp_path / "d" / f"old{i}", 10)
    a = DI.assess(ds(tmp_path / "d", max_age=1.0))
    assert a["status"] == "STALE" and a["age_days"] == pytest.approx(0.01)


def test_assess_ohlc_uses_data_dates_not_mtimes(tmp_path, monkeypatch, frozen):
    f = touch(tmp_path / "cache", 0.01)  # fresh mtime, stale content
    write_index(
        tmp_path,
        monkeypatch,
        {f"ohlc:A{i}": {"to": days_ago(1)} for i in range(9)}
        | {f"ohlc:B{i}": {"to": days_ago(9)} for i in range(9)},
    )
    a = DI.assess(ds(f, key="cache.ohlc", max_age=2.0))
    assert a["status"] == "STALE" and a["coverage"] == pytest.approx(0.5)
    assert "only 50% of 18 tickers are within 2d of the leading edge (need 80%)" in a["detail"]
    assert "refresh was NOT uniform" in a["detail"]
    assert a["age_days"] == pytest.approx(0.01)


def test_assess_ohlc_healthy_and_edge_stale(tmp_path, monkeypatch, frozen):
    f = touch(tmp_path / "cache", 0.01)
    write_index(tmp_path, monkeypatch, {f"ohlc:A{i}": {"to": days_ago(1)} for i in range(10)})
    ok = DI.assess(ds(f, key="cache.ohlc", max_age=2.0))
    assert ok["status"] == "OK" and ok["detail"] == f"latest bar {days_ago(1)}, 100% uniform"
    write_index(tmp_path, monkeypatch, {f"ohlc:A{i}": {"to": days_ago(20)} for i in range(10)})
    old = DI.assess(ds(f, key="cache.ohlc", max_age=2.0))
    assert old["status"] == "STALE"
    assert old["detail"] == (
        f"latest bar {days_ago(20)} (20d ago); leading edge itself is 20d behind today"
    )


def test_assess_ohlc_without_index_falls_back_to_file_coverage(tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(R, "META_DIR", tmp_path / "missing")
    f = touch(tmp_path / "cache", 5)
    assert DI.assess(ds(f, key="cache.ohlc", max_age=1.0))["status"] == "STALE"
    g = touch(tmp_path / "cache2", 0.5)
    assert DI.assess(ds(g, key="cache.ohlc", max_age=1.0))["status"] == "OK"


def test_assess_warehouse_dataset(monkeypatch):
    wh = ds("/dev/null", key="wh", max_age=3.0, section="unowned")
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {})
    a = DI.assess(wh)
    assert a["status"] == "UNKNOWN" and a["age_days"] is None and "unreachable" in a["detail"]
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {"IN": days_ago(1), "US": days_ago(3)})
    b = DI.assess(wh)
    assert b["status"] == "OK" and b["age_days"] == 3  # oldest market, boundary == max_age
    assert b["detail"] == f"IN:{days_ago(1)}, US:{days_ago(3)}"
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {"IN": days_ago(1), "US": days_ago(4)})
    assert DI.assess(wh)["status"] == "STALE"
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {"IN": "not-a-date"})
    assert DI.assess(wh)["status"] == "OK" and DI.assess(wh)["age_days"] is None
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {"IN": days_ago(40)})
    assert DI.assess(ds("/dev/null", key="wh", max_age=None, section="unowned"))["status"] == "OK"


def test_assess_only_unowned_dev_null_is_the_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr(DI, "_pg_freshness", lambda: pytest.fail("not a warehouse"))
    # /dev/null in a non-"unowned" section is an ordinary (empty) path, not the warehouse
    assert DI.assess(ds("/dev/null", section="ingest"))["status"] == "MISSING"
    assert DI.assess(ds(tmp_path / "x", section="unowned"))["status"] == "MISSING"


# ── report / require / main ───────────────────────────────────────────────────
@pytest.fixture
def registry(tmp_path, frozen, monkeypatch):
    fresh = touch(tmp_path / "fresh", 0.1)
    stale = touch(tmp_path / "stale", 9)
    sets = [
        ds(fresh, "a.fresh", 1.0, "ingest", "w1.py"),
        ds(stale, "a.stale", 1.0, "ingest", "w2.py", note="🔴 do not trust this"),
        ds(tmp_path / "gone", "b.missing", 1.0, "mailer", "w3.py"),
        ds(fresh, "c.ok", 1.0, "modelling", "w4.py"),
    ]
    monkeypatch.setattr(R, "DATASETS", sets)
    monkeypatch.setattr(R, "for_section", lambda s: [d for d in sets if d.section == s])
    return sets


def test_report_text_marks_and_summary(registry, capsys):
    assert DI.report() == 0
    out = capsys.readouterr().out
    assert "DATA INDEX\n" in out and "[ingest]" in out and "[mailer]" in out
    assert "a.fresh" in out and "0.1d  OK" in out
    assert "⚠️  a.stale" in out and "9.0d  STALE" in out and "← w2.py" in out
    line = [x for x in out.splitlines() if "b.missing" in x and "❌" in x][0]
    assert line == "   ❌ b.missing                    —    MISSING  ← w3.py"
    assert "🔴 do not trust this" in out
    assert "2 dataset(s) stale or missing:" in out
    assert "STALE    a.stale" in out and "MISSING  b.missing" in out and "writer: w3.py" in out


def test_report_section_filter_and_all_ok_message(registry, capsys):
    assert DI.report(section="modelling") == 0
    out = capsys.readouterr().out
    assert "DATA INDEX — modelling" in out and "a.stale" not in out
    assert "all tracked datasets within tolerance" in out


def test_report_json_includes_registry_fields(registry, capsys):
    DI.report(as_json=True)
    rows = json.loads(capsys.readouterr().out)
    assert [r["key"] for r in rows] == ["a.fresh", "a.stale", "b.missing", "c.ok"]
    r = rows[1]
    assert r["status"] == "STALE" and r["writer"] == "w2.py" and r["section"] == "ingest"
    assert r["max_age_days"] == 1.0 and r["cadence"] == "daily" and r["note"].startswith("🔴")
    assert DI.report(section="mailer", as_json=True) == 0
    assert [x["key"] for x in json.loads(capsys.readouterr().out)] == ["b.missing"]


def test_report_age_label_switches_to_old_at_999_days(tmp_path, frozen, monkeypatch, capsys):
    just = touch(tmp_path / "a", 998.5)
    edge = touch(tmp_path / "b", 999.0)
    monkeypatch.setattr(R, "DATASETS", [ds(just, "x.just", None), ds(edge, "x.edge", None)])
    DI.report()
    lines = capsys.readouterr().out.splitlines()
    assert "998.5d" in [x for x in lines if "x.just" in x][0]
    assert "  old" in [x for x in lines if "x.edge" in x][0]


def test_report_old_datasets_show_old_label(tmp_path, frozen, monkeypatch, capsys):
    f = touch(tmp_path / "ancient", 2000)
    monkeypatch.setattr(R, "DATASETS", [ds(f, "x", None)])
    DI.report()
    assert "  old" in capsys.readouterr().out


def test_report_persist_records_every_dataset_and_survives_failures(registry, monkeypatch, capsys):
    calls = []

    def record(key, status, coverage_pct=None, stale_days=None, detail=None):
        calls.append((key, status, coverage_pct))
        if key == "a.stale":
            raise RuntimeError("db down")

    monkeypatch.setitem(
        sys.modules, "system_state", types.SimpleNamespace(record_data_snapshot=record)
    )
    assert DI.report(persist=True) == 0
    assert [c[0] for c in calls] == ["a.fresh", "a.stale", "b.missing", "c.ok"]
    assert calls[0] == ("a.fresh", "OK", 1.0)


def test_report_persist_unavailable_is_nonfatal(registry, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "system_state", None)
    assert DI.report(persist=True) == 0
    assert "--persist unavailable" in capsys.readouterr().err


def test_require_gate(registry, capsys):
    assert DI.require("modelling") == 0
    assert "[gate] modelling: all 1 datasets fresh" in capsys.readouterr().out
    assert DI.require("ingest") == 1
    out = capsys.readouterr().out
    assert "❌ ingest: 1 dataset(s) not usable" in out and "STALE    a.stale" in out
    assert "(9.0d old)" in out and "writer: w2.py" in out
    assert DI.require("mailer") == 1
    assert "(missing)" in capsys.readouterr().out


def test_require_ignores_unknown_status(monkeypatch, capsys):
    wh = ds("/dev/null", "wh", 3.0, "unowned")
    monkeypatch.setattr(R, "for_section", lambda s: [wh])
    monkeypatch.setattr(DI, "_pg_freshness", lambda: {})
    assert DI.require("unowned") == 0


def test_main_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(DI, "report", lambda **k: calls.append(("report", k)) or 0)
    monkeypatch.setattr(DI, "require", lambda s: calls.append(("require", s)) or 1)
    monkeypatch.setattr(sys, "argv", ["x", "--require", "ingest", "--section", "mailer"])
    assert DI.main() == 1 and calls == [("require", "ingest")]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--section", "mailer", "--json", "--persist"])
    assert DI.main() == 0
    assert calls == [("report", {"section": "mailer", "as_json": True, "persist": True})]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x"])
    DI.main()
    assert calls == [("report", {"section": None, "as_json": False, "persist": False})]
    monkeypatch.setattr(sys, "argv", ["x", "--section", "bogus"])
    with pytest.raises(SystemExit):
        DI.main()
