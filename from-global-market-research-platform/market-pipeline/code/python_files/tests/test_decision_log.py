"""decision_log.py -- secret redaction, source extraction and markdown rendering.

Everything is local: subprocess is faked, transcripts and logs live under tmp_path.
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import decision_log as DL


# ── _redact ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("password=hunter2", "password=<redacted>"),
    ("PASSWORD: hunter2", "PASSWORD=<redacted>"),
    ("SCREENER_PASSWORD=abc123", "SCREENER_PASSWORD=<redacted>"),
    ("my-api-key = zzz", "my-api-key =<redacted>"),
    ("crtfc_key=0123", "crtfc_key=<redacted>"),
    ("token: abc", "token=<redacted>"),
    ("x passwd:  y", "x passwd=<redacted>"),
])
def test_redact_key_value(text, expected):
    assert DL._redact(text) == expected


def test_redact_consumes_rest_of_line_not_just_first_word():
    # a Gmail app password is four space-separated groups -- all of it must go
    out = DL._redact("GMAIL_APP_PASSWORD=abcd efgh ijkl mnop")
    assert out == "GMAIL_APP_PASSWORD=<redacted>"
    assert "efgh" not in out and "mnop" not in out


def test_redact_stops_at_newline():
    out = DL._redact("secret=abc def\nnext line stays")
    assert out == "secret=<redacted>\nnext line stays"


@pytest.mark.parametrize("secret", [
    "alice@example.com",
    "ABCDE1234F",                                  # PAN
    "abcd efgh ijkl mnop",                         # bare gmail app password
    "ghp_abcdefghijklmnop1234",
    "sk-abcdefghijklmnopqrst",
    "xoxb-1234567890abc",
    "0123456789abcdef0123456789abcdef",            # 32 hex
])
def test_redact_bare_secret_shapes(secret):
    out = DL._redact(f"see {secret} here")
    assert secret not in out and "<redacted>" in out
    assert out.startswith("see ") and out.endswith(" here")


def test_redact_hex_length_boundary():
    assert DL._redact("a" * 31) == "a" * 31                         # 31 hex chars: kept
    assert DL._redact("a" * 32) == "<redacted>"


def test_redact_leaves_ordinary_text_alone():
    s = "Fixed the liquidity floor in adaptive_liquidity.py (PR #12)"
    assert DL._redact(s) == s


@pytest.mark.parametrize("v", ["", None])
def test_redact_empty(v):
    assert DL._redact(v) == ""


# ── _run ──────────────────────────────────────────────────────────────────────
def test_run_returns_stdout_and_swallows_errors(monkeypatch, tmp_path):
    seen = {}

    def ok(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="hello", stderr="")
    monkeypatch.setattr(subprocess, "run", ok)
    assert DL._run(["x", "y"]) == "hello"
    assert seen["cwd"] == DL.HERE and seen["timeout"] == 60 and seen["capture_output"]
    DL._run(["x"], cwd=tmp_path)
    assert seen["cwd"] == tmp_path

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("x", 60)
    monkeypatch.setattr(subprocess, "run", boom)
    assert DL._run(["x"]) == ""


# ── git_changes ───────────────────────────────────────────────────────────────
def test_git_changes_parses_commits_and_files(monkeypatch):
    seen = {}

    def fake(cmd, cwd=None):
        seen["cmd"] = cmd
        return ("abc123|Alice|fix: floor | the pipe\n"
                "a.py\n"
                "dir/b.py\n"
                "\n"
                "def456|Bob|api_key=SECRETVALUE tweak\n"
                "c.py\n")
    monkeypatch.setattr(DL, "_run", fake)
    out = DL.git_changes("2026-07-20")
    assert [c["hash"] for c in out] == ["abc123", "def456"]
    assert out[0]["author"] == "Alice" and out[0]["subject"] == "fix: floor | the pipe"
    assert out[0]["files"] == ["a.py", "dir/b.py"] and out[1]["files"] == ["c.py"]
    assert "SECRETVALUE" not in out[1]["subject"]
    assert "--since=2026-07-20 00:00" in seen["cmd"] and "--until=2026-07-20 23:59" in seen["cmd"]


def test_git_changes_ignores_files_before_first_commit_and_two_part_lines(monkeypatch):
    monkeypatch.setattr(DL, "_run", lambda cmd, cwd=None: "orphan.py\nh|only-two\nabc|A|s\nf.py")
    out = DL.git_changes("2026-07-20")
    assert len(out) == 1 and out[0]["files"] == ["f.py"]
    monkeypatch.setattr(DL, "_run", lambda cmd, cwd=None: "")
    assert DL.git_changes("2026-07-20") == []


# ── pipeline_runs ─────────────────────────────────────────────────────────────
def test_pipeline_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "HERE", tmp_path)
    (tmp_path / "india_pipeline_20260720.log").write_text(
        "[STEP] 1 00:01 fetch prices\n[STEP] 2 00:02 build\nnoise\n"
        "[ALERT] token=abc leaked\n=== done india 2026-07-20 — 12m 3s ===\n")
    (tmp_path / "us_pipeline_20260720.log").write_text("[STEP] 1 x y\n")
    (tmp_path / "eu_pipeline_20260719.log").write_text("[STEP] 1 x y\n")     # other day
    runs = DL.pipeline_runs("2026-07-20")
    assert [r["section"] for r in runs] == ["india", "us"]                    # sorted by path
    india, us = runs
    assert india["steps"] == 2 and india["duration"] == "12m 3s"
    assert india["failures"] == ["token=<redacted>"]
    assert us["steps"] == 1 and us["duration"] == "did not finish" and us["failures"] == []
    assert DL.pipeline_runs("2026-01-01") == []


# ── data_state ────────────────────────────────────────────────────────────────
def test_data_state_maps_rows_and_tolerates_garbage(monkeypatch):
    rows = [{"key": "k1", "status": "OK", "writer": "w1", "age_days": 1.5, "extra": 1},
            {"key": "k2", "status": "STALE", "writer": "w2"}]
    seen = {}

    def fake(cmd, cwd=None):
        seen["cmd"] = cmd
        return json.dumps(rows)
    monkeypatch.setattr(DL, "_run", fake)
    out = DL.data_state()
    assert out == [{"key": "k1", "status": "OK", "writer": "w1", "age_days": 1.5},
                   {"key": "k2", "status": "STALE", "writer": "w2", "age_days": None}]
    assert seen["cmd"][1].endswith("data_index.py") and seen["cmd"][-1] == "--json"
    monkeypatch.setattr(DL, "_run", lambda cmd, cwd=None: "not json")
    assert DL.data_state() == []
    monkeypatch.setattr(DL, "_run", lambda cmd, cwd=None: "")
    assert DL.data_state() == []


# ── transcript_activity ───────────────────────────────────────────────────────
DAY = "2026-07-20"


def day_start():
    return dt.datetime.fromisoformat(DAY).timestamp()


def iso_z(offset_s=0.0):
    t = dt.datetime.fromtimestamp(day_start() + offset_s, tz=dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def write_transcript(root, records, name="s1.jsonl", raw=None, age_days=0):
    d = root / "proj"
    d.mkdir(exist_ok=True)
    f = d / name
    lines = [json.dumps(r) for r in records] + (raw or [])
    f.write_text("\n".join(lines) + "\n")
    mt = day_start() + 3600 - age_days * 86400
    os.utime(f, (mt, mt))
    return f


def user(content, ts=None):
    r = {"type": "user", "message": {"content": content}}
    if ts:
        r["timestamp"] = ts
    return r


def tool(name, inp, ts=None):
    r = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name,
                                                       "input": inp}]}}
    if ts:
        r["timestamp"] = ts
    return r


@pytest.fixture
def tdir(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "TRANSCRIPTS", tmp_path)
    return tmp_path


def test_transcript_prompts_filtering_redaction_and_truncation(tdir):
    write_transcript(tdir, [
        user("  fix the floor  "),
        user("<system-reminder>x"),
        user("   "),
        user(["not", "a", "string"]),
        user("deploy with password=hunter2"),
        user("z" * 400),
    ])
    p = DL.transcript_activity(DAY)["prompts"]
    assert p[0] == "fix the floor"
    assert p[1] == "deploy with password=<redacted>"
    assert p[2] == "z" * 300 and len(p) == 3


def test_transcript_day_window_boundaries(tdir):
    write_transcript(tdir, [
        user("before", iso_z(-1)), user("start", iso_z(0)),
        user("mid", iso_z(43200)), user("last", iso_z(86399)),
        user("end", iso_z(86400)),
        user("notime"), user("badtime", "not-a-date"),
    ])
    assert DL.transcript_activity(DAY)["prompts"] == ["start", "mid", "last", "notime", "badtime"]


def test_transcript_skips_stale_files_but_keeps_yesterdays(tdir):
    write_transcript(tdir, [user("old")], name="old.jsonl", age_days=3)
    write_transcript(tdir, [user("recent")], name="recent.jsonl", age_days=0)
    assert DL.transcript_activity(DAY)["prompts"] == ["recent"]
    # mtime exactly one day before the window start is still read (strict <)
    f = write_transcript(tdir, [user("edge")], name="edge.jsonl")
    os.utime(f, (day_start() - 86400, day_start() - 86400))
    assert "edge" in DL.transcript_activity(DAY)["prompts"]
    os.utime(f, (day_start() - 86401, day_start() - 86401))
    assert "edge" not in DL.transcript_activity(DAY)["prompts"]


def test_transcript_skips_malformed_lines(tdir):
    write_transcript(tdir, [user("ok")], raw=["{broken", "", "[1,2]x"])
    assert DL.transcript_activity(DAY)["prompts"] == ["ok"]


def test_transcript_tools_files_decisions_and_hosts(tdir):
    write_transcript(tdir, [
        tool("Read", {"file_path": "/a/b.py"}),
        tool("Read", {"file_path": "/a/b.py"}),
        tool("NotebookEdit", {"notebook_path": "/n.ipynb"}),
        tool("AskUserQuestion", {"questions": [
            {"question": "Use token=abc123 ?", "options": [{"label": "Yes"}, {"label": "No"}]},
            {"question": "Q2"}]}),
        tool("Bash", {"command": "curl https://api.example.com/x && wget http://b.org:80/y"}),
        {"type": "assistant", "message": {"content": [
            "stray string", {"type": "text", "text": "hi"}, {"type": "tool_use"}]}},
        {"type": "assistant", "message": {"content": None}},
    ])
    a = DL.transcript_activity(DAY)
    assert a["tools"] == {"Read": 2, "NotebookEdit": 1, "AskUserQuestion": 1, "Bash": 1, "?": 1}
    assert a["decisions"] == [
        {"question": "Use token=<redacted>", "options": ["Yes", "No"]},
        {"question": "Q2", "options": []}]
    assert a["touched"] == sorted(["/a/b.py", "/n.ipynb", "net:api.example.com", "net:b.org"])


def test_transcript_touched_capped_at_60_sorted(tdir):
    write_transcript(tdir, [tool("Read", {"file_path": f"/f{i:03d}"}) for i in range(80)])
    t = DL.transcript_activity(DAY)["touched"]
    assert len(t) == 60 and t == sorted(t) and t[0] == "/f000"


def test_transcript_empty_when_nothing(tdir):
    assert DL.transcript_activity(DAY) == {"prompts": [], "decisions": [], "tools": {},
                                           "touched": []}


# ── render ────────────────────────────────────────────────────────────────────
def patch_sources(monkeypatch, git=(), runs=(), data=(), act=None):
    monkeypatch.setattr(DL, "git_changes", lambda d: list(git))
    monkeypatch.setattr(DL, "pipeline_runs", lambda d: list(runs))
    monkeypatch.setattr(DL, "data_state", lambda: list(data))
    monkeypatch.setattr(DL, "transcript_activity", lambda d: act or
                        {"prompts": [], "decisions": [], "tools": {}, "touched": []})


def test_render_empty_day(monkeypatch):
    patch_sources(monkeypatch)
    md = DL.render("2026-07-20")
    assert md.startswith("# Decision log — 2026-07-20")
    for line in ("- data_index unavailable", "- no section logs for this date", "- no commits",
                 "- none recorded"):
        assert line in md
    assert md.count("- none recorded") == 2                       # decisions + requests
    assert md.endswith("_Secrets are redacted at extraction. Do not commit this file._")
    assert "- network:" not in md and "- tools:" not in md and "- files:" not in md


def test_render_populated(monkeypatch):
    act = {"prompts": [f"req {i}\nsecond line" for i in range(30)],
           "decisions": [{"question": "Q?", "options": ["A", "B"]}],
           "tools": {f"T{i}": i for i in range(1, 12)},
           "touched": ["/f1", "/f2", "net:b.org", "net:a.com", "net:b.org"]}
    patch_sources(
        monkeypatch,
        git=[{"hash": "abc", "subject": "fix it", "files": ["a", "b"]}],
        runs=[{"section": "india", "steps": 3, "duration": "5m", "failures": ["boom"]},
              {"section": "us", "steps": 1, "duration": "1m", "failures": []}],
        data=[{"key": "k1", "status": "OK", "writer": "w", "age_days": 0.5},
              {"key": "k2", "status": "STALE", "writer": "w2", "age_days": 3.25},
              {"key": "k3", "status": "MISSING", "writer": "w3", "age_days": None}],
        act=act)
    md = DL.render("2026-07-20")
    assert "- 1/3 datasets OK" in md
    assert "  - ⚠️ `k2` **STALE** (3.2d) ← w2" in md
    assert "`k3` **MISSING** (—) ← w3" in md
    assert "- `india` — 3 steps, 5m · **1 failure(s)**" in md and "  - boom" in md
    assert "- `us` — 1 steps, 1m\n" in md
    assert "- `abc` fix it  (2 files)" in md
    assert "- **Q?**\n  - options: A, B" in md
    assert md.count("- req ") == 25 and "req 24" in md and "req 25" not in md
    assert "second line" not in md                                  # first line only
    assert "- network: a.com, b.org" in md and "- files: 2 touched" in md
    tools_line = [x for x in md.splitlines() if x.startswith("- tools:")][0]
    assert tools_line == "- tools: T11×11, T10×10, T9×9, T8×8, T7×7, T6×6, T5×5, T4×4"


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_writes_one_file_per_day(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(DL, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(DL, "render", lambda day: f"md for {day}")
    monkeypatch.setattr(sys, "argv", ["x", "--date", "2026-03-01", "--days", "3"])
    assert DL.main() == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["2026-02-27.md", "2026-02-28.md",
                                                     "2026-03-01.md"]
    assert (out / "2026-02-28.md").read_text() == "md for 2026-02-28"
    assert capsys.readouterr().out.count("→") == 3


def test_main_default_is_today_single_day(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(DL, "render", lambda day: day)
    monkeypatch.setattr(sys, "argv", ["x"])
    DL.main()
    assert [p.name for p in (tmp_path / "out").iterdir()] == [f"{dt.date.today().isoformat()}.md"]


def test_main_install_writes_plist_with_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(DL, "HERE", tmp_path)
    monkeypatch.setenv("MARKET_CACHE", "/mc")
    monkeypatch.delenv("BHAV_CACHE", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--install"])
    assert DL.main() == 0
    plist = (tmp_path / "launchd" / "com.umashankar.decisionlog.plist").read_text()
    assert "<key>MARKET_CACHE</key><string>/mc</string>" in plist
    assert "<string>/Users/umashankar/market-pipeline/data/bhavcopy_cache</string>" in plist
    assert f"<string>{tmp_path}/.venv/bin/python3</string>" in plist
    assert "<key>Hour</key><integer>23</integer><key>Minute</key><integer>45</integer>" in plist
    assert "launchctl bootstrap" in capsys.readouterr().out
    assert not (tmp_path / "decision_log").exists()                # --install does not run
