"""Offline tests for repo_tracker (path-drift audit, section wiring, orphan detection)."""

import json
import subprocess
import sys
import types

import pytest

import repo_tracker as rt


@pytest.fixture
def repo(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "HERE", tmp_path)
    return tmp_path


def _w(repo, name, text):
    (repo / name).write_text(text)


# ---------------------------------------------------------------- _code_lines
def test_code_lines_drops_comments_blanks_and_docstrings_keeping_original_numbers():
    text = "\n".join(
        [
            '"""module doc',  # 1
            'with "/Users/x/Downloads/a" in prose',  # 2
            '"""',  # 3
            "# comment /Users/x/Downloads/b",  # 4
            "",  # 5
            "x = 1  # trailing",  # 6
            "y = 'a'  # note",  # 7
            "'''one-line doc'''",  # 8
            "def f():",  # 9
            '    """inline doc"""',  # 10
            '    return "/Users/x/real"',  # 11
        ]
    )
    assert rt._code_lines(text) == [
        (6, ["x = 1", " trailing"][0]),
        (7, "y = 'a'"),
        (9, "def f():"),
        (11, '    return "/Users/x/real"'),
    ]


def test_code_lines_strips_inline_comment_only_after_two_spaces():
    out = rt._code_lines('a = "p"  # c\nb = "q # not a comment"\n')
    assert out == [(1, 'a = "p"'), (2, 'b = "q # not a comment"')]


def test_code_lines_docstring_end_delimiter_on_same_line_as_text():
    text = '"""start\nmiddle end"""\nreal = 1\n'
    assert rt._code_lines(text) == [(3, "real = 1")]


# ---------------------------------------------------------------- regexes
@pytest.mark.parametrize(
    "line,fatal,drift",
    [
        ('p = Path.home() / "Downloads" / "x"', True, False),
        ('p = Path.home()/"Downloads"', True, False),
        ('p = "/Users/bob/Downloads/code"', True, True),
        ('p = "/Users/bob/repos/x"', False, True),
        ('p = "/tmp/x"', False, False),
    ],
)
def test_path_regexes(line, fatal, drift):
    assert bool(rt.TCC_FATAL_RE.search(line)) is fatal
    assert bool(rt.ABS_DRIFT_RE.search(line)) is drift


def test_env_idiom_and_invoke_regexes():
    assert rt.ENV_IDIOM_RE.search('os.environ.get("MARKET_CACHE", x)')
    assert rt.ENV_IDIOM_RE.search('environ.get( "PGDSN")')
    assert not rt.ENV_IDIOM_RE.search('os.environ.get("HOME")')
    assert rt.INVOKE_RE.findall("$PY -u scan.py --x\n$PY mail.py\n$PY -u -O other.py") == [
        "scan.py",
        "mail.py",
        "other.py",
    ]
    assert not rt.INVOKE_RE.findall("python scan.py")


# ---------------------------------------------------------------- section_wiring
def test_section_wiring_collects_invocations_and_local_inline_imports(repo):
    _w(repo, "ingest.sh", "$PY a.py\n$PY -u b.py arg\npython -c 'import helper, nothere'\n")
    _w(repo, "helper.py", "x=1")
    (repo / "mailer.sh").write_text("$PY a.py\n")
    wiring = rt.section_wiring()
    assert wiring["ingest.sh"] == {"a.py", "b.py", "helper.py"}  # only imports that exist locally
    assert wiring["mailer.sh"] == {"a.py"} and "modelling.sh" not in wiring


# ---------------------------------------------------------------- path_audit
def test_path_audit_severities_and_ordering(repo):
    _w(repo, "fatal.py", 'p = Path.home() / "Downloads" / "x"\n')
    _w(
        repo,
        "ok.py",
        'import os\nroot = os.environ.get("MARKET_CACHE", Path.home() / "Downloads")\n',
    )
    _w(repo, "drift.py", 'p = "/Users/me/repos/thing"\nq = "/Users/me/repos/other"\n')
    _w(repo, "clean.py", 'p = "/tmp/x"\n')
    _w(repo, "prose.py", '"""uses /Users/me/Downloads/x"""\n# "/Users/me/y"\n')
    _w(repo, ".hidden.py", 'p = "/Users/me/z"\n')
    res = rt.path_audit()
    assert [(f["file"], f["severity"]) for f in res] == [
        ("fatal.py", "TCC-FATAL"),
        ("drift.py", "DRIFT"),
        ("ok.py", "OK-ish"),
    ]
    drift = res[1]
    assert (
        drift["drift_lines"] == [1, 2]
        and drift["fatal_lines"] == []
        and drift["has_env_escape"] is False
    )
    assert res[0]["lines"] == [1] and res[2]["has_env_escape"] is True


def test_path_audit_downloads_literal_counts_as_fatal_not_drift(repo):
    _w(repo, "m.py", 'a = "/Users/bob/Downloads/x"\nb = "/Users/bob/other"\n')
    f = rt.path_audit()[0]
    assert f["severity"] == "TCC-FATAL" and f["fatal_lines"] == [1] and f["drift_lines"] == [2]
    assert f["lines"] == [1]  # fatal lines take precedence in the headline list


def test_path_audit_caps_reported_lines_at_six(repo):
    _w(repo, "m.py", "\n".join(f'a{i} = "/Users/bob/x{i}"' for i in range(10)) + "\n")
    assert rt.path_audit()[0]["drift_lines"] == [1, 2, 3, 4, 5, 6]


def test_path_audit_sorts_by_filename_within_severity(repo):
    for n in ("b.py", "a.py"):
        _w(repo, n, 'p = "/Users/me/x"\n')
    assert [f["file"] for f in rt.path_audit()] == ["a.py", "b.py"]


# ---------------------------------------------------------------- orphans
def test_orphans_wired_imported_and_infra_are_excluded(repo):
    _w(repo, "ingest.sh", "$PY wired.py\n")
    _w(repo, "wired.py", "import imported\nfrom data_registry import X\n")
    for n in ("imported.py", "orphan_a.py", "orphan_b.py", "send_alert.py", "data_registry.py"):
        _w(repo, n, "x = 1\n")
    assert rt.orphans() == ["orphan_a.py", "orphan_b.py"]


def test_orphans_import_reachability_is_one_level_deep(repo):
    _w(repo, "mailer.sh", "$PY a.py\n")
    _w(repo, "a.py", "import b\n")
    _w(repo, "b.py", "import c\n")
    _w(repo, "c.py", "x = 1\n")
    assert rt.orphans() == ["c.py"]


def test_orphans_ignores_wired_names_that_do_not_exist(repo):
    _w(repo, "ingest.sh", "$PY ghost.py\n")
    _w(repo, "lone.py", "x=1\n")
    assert rt.orphans() == ["lone.py"]


# ---------------------------------------------------------------- git_state
def test_git_state_parses_porcelain(monkeypatch):
    def run(cmd, cwd, capture_output, text, timeout):
        if cmd[1] == "status":
            return types.SimpleNamespace(
                stdout=" M a.py\n M b.py\nM  staged.py\n?? new.py\n?? other\n"
            )
        return types.SimpleNamespace(stdout="feature\n")

    monkeypatch.setattr(rt.subprocess, "run", run)
    assert rt.git_state() == {"branch": "feature", "modified": 2, "untracked": 2}


def test_git_state_error_is_truncated(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("git", 20)

    monkeypatch.setattr(rt.subprocess, "run", boom)
    assert "error" in rt.git_state() and len(rt.git_state()["error"]) <= 60


# ---------------------------------------------------------------- main
def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["repo_tracker.py", *args])
    return rt.main()


@pytest.fixture
def populated(repo, monkeypatch):
    _w(repo, "fatal.py", 'p = Path.home() / "Downloads"\n')
    _w(repo, "drift.py", 'p = "/Users/me/x"\n')
    _w(repo, "ingest.sh", "$PY drift.py\n$PY ghost.py\n")
    monkeypatch.setattr(rt, "git_state", lambda: {"branch": "main", "modified": 3, "untracked": 1})
    return repo


def test_main_json_output(populated, monkeypatch, capsys):
    assert _main(monkeypatch, "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert [f["file"] for f in out["path_audit"]] == ["fatal.py", "drift.py"]
    assert out["wiring"] == {"ingest.sh": ["drift.py", "ghost.py"]} and out["orphans"] == [
        "fatal.py"
    ]
    assert out["git"]["branch"] == "main"


def test_main_default_report_and_exit_code(populated, monkeypatch, capsys):
    assert _main(monkeypatch) == 1  # a TCC-FATAL violation fails the run
    out = capsys.readouterr().out
    assert "❌ fatal.py" in out and "drift.py" in out and "← wired into ingest.sh" in out
    assert "SECTION WIRING" in out and "❌ ghost.py" in out  # wired script that does not exist
    assert "UNWIRED SCRIPTS (1)" in out and "GIT: branch main · 3 modified · 1 untracked" in out


def test_main_paths_only_and_orphans_only(populated, monkeypatch, capsys):
    assert _main(monkeypatch, "--paths") == 1
    out = capsys.readouterr().out
    assert "HARDCODED-PATH AUDIT" in out and "SECTION WIRING" not in out and "GIT:" not in out
    assert _main(monkeypatch, "--orphans") == 0  # orphans-only never fails on path violations
    out = capsys.readouterr().out
    assert "SECTION WIRING" in out and "HARDCODED-PATH AUDIT" not in out


def test_main_clean_repo_exits_zero(repo, monkeypatch, capsys):
    _w(repo, "ok.py", "x = 1\n")
    monkeypatch.setattr(rt, "git_state", lambda: {})
    assert _main(monkeypatch) == 0
    out = capsys.readouterr().out
    assert "none — every ~/Downloads literal has" in out and "GIT: branch ? · ? modified" in out


def test_main_truncates_long_lists(repo, monkeypatch, capsys):
    for i in range(25):
        _w(repo, f"d{i:02d}.py", 'p = "/Users/me/x"\n')
    monkeypatch.setattr(rt, "git_state", lambda: {})
    _main(monkeypatch)
    out = capsys.readouterr().out
    assert "... and 5 more" in out  # drift list capped at 20
