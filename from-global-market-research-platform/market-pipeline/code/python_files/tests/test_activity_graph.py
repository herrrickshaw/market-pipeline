"""activity_graph.py -- overnight-run log mining: run splitting, failure extraction, mailer gate,
output freshness and the ranked task list. Log files live under tmp_path; langgraph is faked."""

import json
import os
import sys
import types
from datetime import date, datetime

import pytest

import activity_graph as AG

TAG = "20260721"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(AG, "HERE", tmp_path)
    monkeypatch.setattr(AG, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(AG, "OUT_MD", tmp_path / "reports" / "activity_graph.md")
    monkeypatch.setattr(AG, "COLLECT_LOGS", tmp_path / "collect")
    monkeypatch.setattr(AG, "BASEBRIEF", tmp_path / "base.log")
    monkeypatch.setattr(AG, "N8N_LOG", tmp_path / "n8n_out.log")
    monkeypatch.setattr(AG, "BRIEF", tmp_path / "brief_today.html")
    monkeypatch.setattr(AG, "CORR", tmp_path / "correlation_scan")
    return tmp_path


def state(**kw):
    s = {
        "runs": [],
        "steps": [],
        "gates": [],
        "outputs": [],
        "tasks": [],
        "notes": [],
        "summary": "",
    }
    s.update(kw)
    return s


def set_mtime(path, dt):
    t = dt.timestamp()
    os.utime(path, (t, t))


# ── regexes ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "line",
    [
        "x failed (continuing)",
        "scan failed",
        "ModuleNotFoundError: No module named 'x'",
        "Traceback (most...",
        "Operation not permitted",
        "MISSING REQUIRED key",
        "cqlsh not found",
        "mail not configured",
        "step FAIL now",
    ],
)
def test_failure_pattern_matches(line):
    assert AG.FAIL_PAT.search(line)


@pytest.mark.parametrize("line", ["all good", "FAILURE rate 0", "unfailed", "config ok"])
def test_failure_pattern_ignores_normal_lines(line):
    assert not AG.FAIL_PAT.search(line)


def test_step_alert_and_recon_patterns():
    assert AG.STEP.match("[3a/14] scanning").group(1) == "3a"
    assert AG.STEP.match("[10/14]   run it").group(3) == "run it"
    assert not AG.STEP.match("[x/14] nope")
    m = AG.ALERT.match("[ALERT] 3 step(s) failed (continuing): scan, mail")
    assert m.group(1) == "3" and m.group(2) == "scan, mail"
    r = AG.RECON_FAIL.search(
        "[IN] indian_full_scan.xlsx: 40 checked · FAIL — 12/40 stale (worst TCS 4.25%)"
    )
    assert r.groups() == ("IN", "indian_full_scan.xlsx", "40", "12", "40", "TCS", "4.25")
    assert AG.SENT.search("sent 'Daily brief'").group(1) == "Daily brief"
    assert AG.SUPPRESS.search("sending SUPPRESSED") and AG.SUPPRESS.search("send SUPPRESSED")
    assert (
        AG.MODNOTFOUND.search("ModuleNotFoundError: No module named 'statsmodels'").group(1)
        == "statsmodels"
    )


# ── reading helpers ───────────────────────────────────────────────────────────
def test_read_and_mtime_tolerate_missing_files(tmp_path):
    assert AG._read(tmp_path / "nope") == []
    f = tmp_path / "f.txt"
    f.write_bytes(b"a\nb\xff\nc")
    assert AG._read(f) == ["a", "b�", "c"]
    assert AG._mtime(tmp_path / "nope") is None
    set_mtime(f, datetime(2026, 7, 21, 1, 2, 3))
    assert AG._mtime(f) == datetime(2026, 7, 21, 1, 2, 3)


def test_split_runs_on_the_banner():
    lines = [
        "preamble ignored",
        "=== Daily pipeline 2026-07-21 00:30 ===",
        "[1/3] a",
        "=== done 2026-07-21 00:45 ===",
        "=== Daily pipeline 2026-07-21 06:00 ===",
        "[1/3] b",
    ]
    runs = AG._split_runs(lines)
    assert len(runs) == 2
    assert (
        runs[0]["start"].strip() == "2026-07-21 00:30"
        and runs[0]["end"].strip() == "2026-07-21 00:45"
    )
    assert runs[0]["lines"][0].startswith("=== Daily pipeline") and len(runs[0]["lines"]) == 3
    assert runs[1]["end"] is None and runs[1]["lines"][-1] == "[1/3] b"
    assert AG._split_runs([]) == [] and AG._split_runs(["no banner"]) == []
    assert AG._split_runs(["=== done early ==="]) == []  # trailer without a run


# ── discover ──────────────────────────────────────────────────────────────────
def write_logs(env):
    (env / f"daily_pipeline_{TAG}.log").write_text(
        "=== Daily pipeline 00:30 ===\n[1/2] x\n=== done 00:50 ===\n=== Daily pipeline 06:00 ===\n[1/2] y\n"
    )
    (env / f"ingest_pipeline_{TAG}.log").write_text("ingest\n=== done ingest ok ===\ntrail\n")
    (env / "collect").mkdir()
    (env / "collect" / "collection_2026-07-21_a.log").write_text("c1\ncqlsh not found\n")
    (env / "collect" / "collection_2026-07-20_b.log").write_text("old\n")


def test_discover_inventory(env):
    write_logs(env)
    (env / "base.log").write_text("\n".join(f"l{i}" for i in range(60)))
    set_mtime(env / "base.log", datetime(2026, 7, 21, 8, 0))
    out = AG.discover(state(), TAG)
    jobs = [(r["job"], r["label"], r["complete"]) for r in out["runs"]]
    assert jobs == [
        ("dailybrief", "daily_pipeline #1", True),
        ("dailybrief", "daily_pipeline #2", False),
        ("ingest", "ingest_pipeline", True),
        ("collection", "collection_2026-07-21_a", True),
        ("basebrief", "market-brief-base", True),
    ]
    ing = out["runs"][2]
    assert ing["end"] == "=== done ingest ok ===" and ing["lines"][0] == "ingest"
    base = out["runs"][-1]
    assert len(base["lines"]) == 40 and base["lines"][-1] == "l59"  # tail of the log only
    assert out["notes"] == []


def test_discover_notes_for_missing_inputs(env):
    out = AG.discover(state(), TAG)
    assert out["runs"] == []
    assert out["notes"][0].startswith(f"NO daily_pipeline_{TAG}.log")
    assert f"no ingest_pipeline_{TAG}.log" in out["notes"]
    (env / "collect").mkdir()
    assert "no collection_*.log for this date" in AG.discover(state(), TAG)["notes"]
    (env / "base.log").write_text("x")
    set_mtime(env / "base.log", datetime(2026, 7, 19, 8, 0))
    notes = AG.discover(state(), TAG)["notes"]
    assert any(n.startswith("basebrief run.log last written") and TAG in n for n in notes)


# ── activity ──────────────────────────────────────────────────────────────────
def run(job="dailybrief", label="r", lines=(), complete=True, start="s", end="e"):
    return {
        "job": job,
        "label": label,
        "log": "l",
        "start": start,
        "end": end,
        "lines": list(lines),
        "complete": complete,
    }


def test_activity_counts_steps_failures_and_modules():
    lines = [
        "[1/14] first",
        "  [2/14]   second",
        "noise",
        "Traceback (most recent call last):",
        "ModuleNotFoundError: No module named 'scipy'",
        "ModuleNotFoundError: No module named 'scipy'",
        "[ALERT] 2 step(s) failed (continuing): a, b",
        "[ALERT] 1 step(s) failed (continuing): later",
    ]
    out = AG.activity(state(runs=[run(lines=lines)]))["steps"][0]
    assert out["n_steps"] == 2 and out["last_step"] == "[2/14]   second".strip()
    assert out["n_fail_lines"] == 4  # Traceback, ModuleNotFound (once), 2 alerts
    assert out["fails"] == sorted(out["fails"]) and out["missing_modules"] == ["scipy"]
    assert out["alert_count"] == 1 and out["alert_detail"] == "later"  # last alert wins
    assert out["job"] == "dailybrief" and out["complete"] is True


def test_activity_clean_and_failure_cap():
    clean = AG.activity(state(runs=[run(lines=["ok"])]))["steps"][0]
    assert clean["n_steps"] == 0 and clean["last_step"] is None and clean["alert_count"] == 0
    assert clean["alert_detail"] is None and clean["fails"] == []
    many = [f"Traceback {i}" for i in range(20)]
    out = AG.activity(state(runs=[run(lines=many)]))["steps"][0]
    assert out["n_fail_lines"] == 20 and len(out["fails"]) == 12


# ── gates ─────────────────────────────────────────────────────────────────────
def test_gates_extract_reconcile_failures_and_sends():
    lines = [
        "[IN] indian_full_scan.xlsx: 40 checked · FAIL — 12/40 stale (worst TCS 4.25%)",
        "✓ validated screener.in",
        "send SUPPRESSED — prices 4.25% stale",
        "draft saved",
        "sent 'Brief A'",
        "sent 'Brief B'",
    ]
    g = AG.gates(
        state(
            runs=[
                run(lines=lines),
                run(job="collection", lines=lines),
                run(job="ingest", lines=["quiet"]),
            ]
        )
    )["gates"]
    assert len(g) == 1  # collection ignored, quiet ingest skipped
    g = g[0]
    assert g["screener_validated"] and g["suppressed"] and g["draft_saved"]
    assert g["reason"] == "prices 4.25% stale"
    assert g["sent_subjects"] == ["Brief A", "Brief B"]
    assert g["reconcile_failures"] == [
        {
            "market": "IN",
            "workbook": "indian_full_scan.xlsx",
            "checked": 40,
            "stale": 12,
            "of": 40,
            "worst_sym": "TCS",
            "worst_pct": 4.25,
        }
    ]


def test_gates_suppressed_without_dash_uses_the_whole_line():
    g = AG.gates(state(runs=[run(lines=["  sending SUPPRESSED  "])]))["gates"][0]
    assert g["reason"] == "sending SUPPRESSED" and not g["screener_validated"]


# ── outputs ───────────────────────────────────────────────────────────────────
def test_outputs_freshness(env):
    today = datetime.combine(date.today(), datetime.min.time()).replace(hour=1)
    out = AG.outputs(state())["outputs"]
    assert out == [
        {"artefact": "brief_today.html", "mtime": None, "fresh": False, "detail": "absent"}
    ]
    (env / "brief_today.html").write_text("x" * 1234)
    set_mtime(env / "brief_today.html", today)
    first = AG.outputs(state())["outputs"][0]
    assert first["fresh"] is True and first["detail"] == "1,234 bytes"
    set_mtime(env / "brief_today.html", datetime(2020, 1, 1))
    assert AG.outputs(state())["outputs"][0]["fresh"] is False


def test_outputs_correlation_freshness_and_n8n(env):
    env_corr = env / "correlation_scan"
    env_corr.mkdir()
    (env_corr / "nse_correlation_clusters.txt").write_text("x")
    (env_corr / "us_correlation_clusters.txt").write_text("x")
    set_mtime(env_corr / "nse_correlation_clusters.txt", datetime(2020, 1, 1))
    recent = datetime.now()
    set_mtime(env_corr / "us_correlation_clusters.txt", recent)
    (env / "n8n_out.log").write_text('noise\nActivated workflow "Watchlist Digest 07:00" ok\n')
    out = {o["artefact"]: o for o in AG.outputs(state())["outputs"]}
    assert out["correlation clusters"]["fresh"] is True
    assert out["correlation clusters"]["detail"] == "2 file(s), newest us_correlation_clusters.txt"
    assert out["correlation matrix"]["fresh"] is False
    assert out["correlation matrix"]["detail"] == "0 file(s) — NEVER PRODUCED"
    n8n = out["n8n Watchlist Digest 07:00"]
    assert n8n["fresh"] is True and n8n["detail"].startswith("ACTIVE — Activated workflow")


def test_outputs_correlation_one_day_old_is_still_fresh(env):
    d = env / "correlation_scan"
    d.mkdir()
    (d / "a_correlation_clusters.txt").write_text("x")
    set_mtime(
        d / "a_correlation_clusters.txt", datetime.now() - __import__("datetime").timedelta(days=1)
    )
    assert {o["artefact"]: o for o in AG.outputs(state())["outputs"]}["correlation clusters"][
        "fresh"
    ] is True
    set_mtime(
        d / "a_correlation_clusters.txt", datetime.now() - __import__("datetime").timedelta(days=3)
    )
    assert {o["artefact"]: o for o in AG.outputs(state())["outputs"]}["correlation clusters"][
        "fresh"
    ] is False


# ── synthesize ────────────────────────────────────────────────────────────────
def titles(res):
    return [(t["severity"], t["title"]) for t in res["tasks"]]


def test_synthesize_missing_log_is_a_blocker():
    res = AG.synthesize(
        state(notes=["NO daily_pipeline_X.log — the run produced none", "other note"])
    )
    assert titles(res) == [("BLOCKER", "Scheduled 00:30 run produced no log")]


def gate(suppressed=True, validated=True, recon=()):
    return {
        "job": "dailybrief",
        "label": "run#1",
        "screener_validated": validated,
        "suppressed": suppressed,
        "reason": "stale prices",
        "reconcile_failures": list(recon),
        "sent_subjects": [],
        "draft_saved": False,
    }


def test_synthesize_gate_severities():
    res = AG.synthesize(state(gates=[gate(validated=True)]))
    assert titles(res) == [("HIGH", "Mailer send suppressed — validation PASSED")]
    res = AG.synthesize(state(gates=[gate(validated=False)]))
    assert titles(res) == [("BLOCKER", "Mailer send suppressed — screener.in MISMATCH")]
    assert AG.synthesize(state(gates=[gate(suppressed=False)]))["tasks"] == []


def test_synthesize_reconcile_failures_whole_vs_partial_sample():
    whole = {"market": "IN", "stale": 40, "of": 40, "worst_sym": "TCS", "worst_pct": 4.25}
    part = {"market": "US", "stale": 3, "of": 40, "worst_sym": "AAPL", "worst_pct": 1.5}
    res = AG.synthesize(state(gates=[gate(suppressed=False, recon=[whole, part])]))
    t = {x["title"]: x for x in res["tasks"]}
    hi = t["Price reconcile FAIL — IN (40/40 stale)"]
    assert hi["severity"] == "HIGH" and "whole sample stale, not intraday drift" in hi["why"]
    assert "scan's price source" in hi["fix"] and "worst TCS 4.2% vs fresh quote" in hi["why"]
    md = t["Price reconcile FAIL — US (3/40 stale)"]
    assert (
        md["severity"] == "MEDIUM" and "whole sample" not in md["why"] and "Spot-check" in md["fix"]
    )


def step(**kw):
    base = {
        "job": "dailybrief",
        "label": "run#1",
        "start": "s",
        "end": "e",
        "complete": True,
        "n_steps": 3,
        "last_step": "[3/14] z",
        "n_fail_lines": 0,
        "fails": [],
        "missing_modules": [],
        "alert_count": 0,
        "alert_detail": None,
    }
    base.update(kw)
    return base


def test_synthesize_step_findings():
    res = AG.synthesize(
        state(
            steps=[
                step(missing_modules=["scipy"], alert_count=2, alert_detail="a, b", complete=False)
            ]
        )
    )
    assert titles(res) == [
        ("BLOCKER", "run#1 never reached its done trailer"),
        ("HIGH", "Missing dependency: scipy"),
        ("MEDIUM", "run#1: 2 step(s) failed"),
    ]
    blocker = res["tasks"][0]
    assert blocker["why"] == "last step: [3/14] z"
    # incomplete non-dailybrief jobs are not blockers
    other = AG.synthesize(state(steps=[step(job="ingest", complete=False)]))
    assert other["tasks"] == []


def test_synthesize_outputs_and_runs():
    outputs = [
        {"artefact": "brief_today.html", "mtime": "old", "fresh": False, "detail": "x"},
        {
            "artefact": "correlation matrix",
            "mtime": None,
            "fresh": False,
            "detail": "0 file(s) — NEVER PRODUCED",
        },
        {
            "artefact": "n8n Watchlist Digest 07:00",
            "mtime": "m",
            "fresh": True,
            "detail": "ACTIVE — x",
        },
    ]
    runs = [
        {"job": "basebrief", "label": "b", "lines": ["email: not configured"]},
        {"job": "collection", "label": "c", "lines": ["cqlsh not found"]},
    ]
    res = AG.synthesize(state(outputs=outputs, runs=runs))
    assert titles(res) == [
        ("BLOCKER", "brief_today.html is not from today"),
        ("HIGH", "n8n 07:00 Watchlist Digest is ACTIVE"),
        ("MEDIUM", "correlation matrix has never produced output"),
        ("LOW", "basebrief cannot send (SMTP unset)"),
        ("LOW", "collection job cannot load Cassandra"),
    ]
    assert res["summary"] == "2 run(s) across 2 scheduler(s); 5 task(s), 1 blocker(s)"


def test_synthesize_deduplicates_and_orders_by_severity():
    res = AG.synthesize(
        state(
            steps=[step(label="a", missing_modules=["x"]), step(label="b", missing_modules=["x"])]
        )
    )
    assert len(res["tasks"]) == 1  # same (severity, title)
    res = AG.synthesize(
        state(
            steps=[step(missing_modules=["x"], alert_count=1, alert_detail="d")],
            notes=["NO daily_pipeline_X.log"],
        )
    )
    sev = [t["severity"] for t in res["tasks"]]
    assert sev == sorted(sev, key=["BLOCKER", "HIGH", "MEDIUM", "LOW"].index)


def test_synthesize_empty_state():
    res = AG.synthesize(state())
    assert (
        res["tasks"] == []
        and res["summary"] == "0 run(s) across 0 scheduler(s); 0 task(s), 0 blocker(s)"
    )


# ── render ────────────────────────────────────────────────────────────────────
def full_state():
    return state(
        summary="S",
        steps=[
            step(complete=False, n_steps=4, n_fail_lines=2, alert_count=3),
            step(label="ok", complete=True),
        ],
        gates=[
            dict(
                gate(),
                reconcile_failures=[
                    {"market": "IN", "stale": 4, "of": 10, "worst_sym": "X", "worst_pct": 2.0}
                ],
                sent_subjects=["Subj"],
                draft_saved=True,
            )
        ],
        outputs=[{"artefact": "brief_today.html", "fresh": False, "mtime": "m", "detail": "d"}],
        tasks=[{"severity": "HIGH", "title": "T", "why": "W", "fix": "F"}],
        notes=["N1"],
    )


def test_render_sections():
    md = AG.render(full_state(), TAG)
    assert md.startswith(f"# Overnight activity — {TAG}\n\nS\n")
    assert (
        "| dailybrief | run#1 | **NO** | 4 | 2 | 3 |" in md
        and "| dailybrief | ok | yes | 3 | 0 | — |" in md
    )
    assert "**run#1** — screener.in validated: yes · suppressed: yes · draft: yes" in md
    assert (
        "  - blocker: stale prices" in md
        and "  - [IN] 4/10 stale, worst X 2.0%" in md
        and "  - sent: Subj" in md
    )
    assert "| brief_today.html | **no** | m | d |" in md
    assert "1. **[HIGH] T**\n   - why: W\n   - fix: F" in md
    assert "## Notes\n\n- N1" in md


def test_render_empty_states():
    md = AG.render(state(summary="S"), TAG)
    assert "## Mailer gate" not in md and "## Notes" not in md
    assert "None — nothing blocking the send." in md


# ── build / main (langgraph faked) ────────────────────────────────────────────
class FakeStateGraph:
    def __init__(self, schema):
        self.nodes = []

    def add_node(self, name, fn):
        self.nodes.append((name, fn))

    def add_edge(self, a, b):
        pass

    def compile(self):
        nodes = self.nodes

        class App:
            def invoke(self, s):
                s = dict(s)
                for _, fn in nodes:
                    for k, v in fn(s).items():
                        s[k] = s[k] + v if isinstance(v, list) else v
                return s

        return App()


@pytest.fixture
def fake_langgraph(monkeypatch):
    mod = types.SimpleNamespace(END="END", START="START", StateGraph=FakeStateGraph)
    monkeypatch.setitem(sys.modules, "langgraph", types.SimpleNamespace(graph=mod))
    monkeypatch.setitem(sys.modules, "langgraph.graph", mod)


def test_main_writes_markdown_report(env, fake_langgraph, monkeypatch, capsys):
    write_logs(env)
    monkeypatch.setattr(sys, "argv", ["x", "--date", TAG])
    assert AG.main() == 0
    out = capsys.readouterr().out
    assert f"# Overnight activity — {TAG}" in out and f"wrote {AG.OUT_MD}" in out
    assert (env / "reports" / "activity_graph.md").read_text().startswith("# Overnight activity")


def test_main_json_output_strips_log_lines(env, fake_langgraph, monkeypatch, capsys):
    write_logs(env)
    monkeypatch.setattr(sys, "argv", ["x", "--json", "--date", TAG])
    assert AG.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"] and all("lines" not in r for r in data["runs"])
    assert not (env / "reports").exists()  # JSON mode writes nothing
    assert data["summary"].endswith("blocker(s)")


# ── mutation follow-ups ───────────────────────────────────────────────────────
def test_synthesize_ignores_basebrief_lines_that_are_not_the_smtp_warning():
    runs = [{"job": "basebrief", "label": "b", "lines": ["email: configured, sent 1"]}]
    assert titles(AG.synthesize(state(runs=runs))) == []


def test_main_works_when_reports_dir_already_exists(env, fake_langgraph, monkeypatch, capsys):
    write_logs(env)
    (env / "reports").mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["x", "--date", TAG])
    assert AG.main() == 0
    assert (env / "reports" / "activity_graph.md").exists()
