"""Offline tests for scan_timings (log parsing/backfill/report) and risk_management overlay math."""

import datetime as dt
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import risk_management as rk
import scan_timings as st


# ================================================================ scan_timings
def _log(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_step_regex_requires_epoch_iso_and_bracket_label():
    m = st.STEP_RE.match("[STEP] 1700000000 2023-11-14T22:13:20 [2/14] India scan")
    assert m.group(1) == "1700000000" and m.group(3) == "[2/14] India scan"
    assert st.STEP_RE.match("[STEP] abc 2023 [2/14] x") is None
    assert st.STEP_RE.match("noise [STEP] 1 2 [x] y") is None  # anchored at line start
    assert st.STEP_RE.match("[STEP] 1 2 no-bracket") is None


def test_measured_durations_are_next_marker_minus_this(tmp_path):
    p = _log(
        tmp_path,
        "daily_pipeline_20260715.log",
        "\n".join(
            [
                "=== Daily pipeline x ===",
                "[STEP] 1000 t [1/14] Alpha",
                "some output",
                "[STEP] 1060 t [2/14] Beta  ",
                "[STEP] 1500 t [3/14] Gamma",
                "[STEP] 1530 t [__end__] done",
            ]
        ),
    )
    rows = st.measured(p)
    assert rows == [
        {"step": "[1/14] Alpha", "seconds": 60, "source": "measured"},
        {"step": "[2/14] Beta", "seconds": 440, "source": "measured"},
        {"step": "[3/14] Gamma", "seconds": 30, "source": "measured"},
    ]


def test_measured_last_marker_without_successor_is_dropped_and_end_never_reported(tmp_path):
    p = _log(tmp_path, "a.log", "[STEP] 1 t [1/14] A\n[STEP] 9 t [2/14] B\n")
    assert [r["step"] for r in st.measured(p)] == ["[1/14] A"]
    p = _log(tmp_path, "b.log", "[STEP] 1 t [__end__] only\n")
    assert st.measured(p) == []
    assert st.measured(_log(tmp_path, "c.log", "no markers\n")) == []


def test_measured_tolerates_undecodable_bytes(tmp_path):
    p = tmp_path / "d.log"
    p.write_bytes(b"[STEP] 1 t [1/14] A\n\xff\xfe junk\n[STEP] 5 t [2/14] B\n")
    assert st.measured(p)[0]["seconds"] == 4


def test_parse_banner_formats():
    assert st._parse_banner("Tue Jul 14 00:15:01 UTC 2026") == dt.datetime(2026, 7, 14, 0, 15, 1)
    assert st._parse_banner(" Tue Jul  7 00:15:01 UTC 2026 ") == dt.datetime(2026, 7, 7, 0, 15, 1)
    assert st._parse_banner("garbage") is None


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "HERE", tmp_path)
    monkeypatch.setattr(st.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path


def _artifact(home, pattern_name, when):
    sub, fname = pattern_name.split("/")
    d = home / sub
    d.mkdir(exist_ok=True)
    f = d / fname
    f.write_text("x")
    ts = when.timestamp()
    os.utime(f, (ts, ts))
    return f


def test_backfill_reconstructs_steps_from_artifact_mtimes(home):
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    log = _log(
        home,
        "daily_pipeline_20260714.log",
        "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\nstuff\n"
        "=== done Tue Jul 14 01:00:00 UTC 2026 ===\n",
    )
    _artifact(
        home, "indian_full_scan/indian_full_scan_20260714.xlsx", start + dt.timedelta(minutes=10)
    )
    _artifact(home, "korea_scan/korea_market_scan_20260714.xlsx", start + dt.timedelta(minutes=40))
    _artifact(home, "us_full_scan/us_full_scan_20260714.xlsx", start + dt.timedelta(minutes=25))
    rows = st.backfill(log)
    assert [(r["step"], r["seconds"], r["source"]) for r in rows] == [
        ("[2/14]", 600, "est"),
        ("[4/14]", 900, "est"),
        ("[8/14]", 900, "est"),
    ]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: DONE_RE starts with '^' but is used with .search() and no re.M, so it "
    "can only match on the first line of the log; the '=== done ===' line never matches and the "
    "'[9-14] correlations+mailer' tail row is never produced",
)
def test_backfill_adds_tail_row_from_done_line(home):
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    log = _log(
        home,
        "daily_pipeline_20260714.log",
        "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\nstuff\n"
        "=== done Tue Jul 14 01:00:00 UTC 2026 ===\n",
    )
    _artifact(home, "korea_scan/korea_market_scan_20260714.xlsx", start + dt.timedelta(minutes=40))
    rows = st.backfill(log)
    assert rows[-1] == {"step": "[9-14] correlations+mailer", "source": "est", "seconds": 1200}


def test_backfill_ignores_other_days_and_files_older_than_start(home):
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    log = _log(
        home, "daily_pipeline_20260714.log", "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\n"
    )
    _artifact(home, "indian_full_scan/indian_full_scan_a.xlsx", start - dt.timedelta(minutes=1))
    _artifact(home, "us_full_scan/us_full_scan_b.xlsx", start + dt.timedelta(days=1))
    assert st.backfill(log) == []


def test_backfill_without_done_line_has_no_tail_row_and_missing_banner_is_empty(home):
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    log = _log(
        home, "daily_pipeline_20260714.log", "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\n"
    )
    _artifact(home, "japan_scan/japan_market_scan_x.xlsx", start + dt.timedelta(minutes=5))
    rows = st.backfill(log)
    assert [r["step"] for r in rows] == ["[7/14]"] and rows[0]["seconds"] == 300
    assert st.backfill(_log(home, "nobanner.log", "nothing\n")) == []
    bad = _log(home, "badbanner.log", "=== Daily pipeline not a date ===\n")
    assert st.backfill(bad) == []


def test_backfill_uses_newest_artifact_per_step(home):
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    log = _log(
        home, "daily_pipeline_20260714.log", "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\n"
    )
    _artifact(home, "indian_full_scan/indian_full_scan_a.xlsx", start + dt.timedelta(minutes=5))
    _artifact(home, "indian_full_scan/indian_full_scan_b.xlsx", start + dt.timedelta(minutes=9))
    _artifact(home, "indian_full_scan/indian_full_scan_c.xlsx", start + dt.timedelta(minutes=7))
    assert [(r["step"], r["seconds"]) for r in st.backfill(log)] == [("[2/14]", 540)]


def test_measured_end_marker_in_the_middle_is_skipped_but_still_bounds_previous_step(tmp_path):
    p = _log(
        tmp_path,
        "m.log",
        "[STEP] 1 t [1/14] A\n[STEP] 5 t [__end__] x\n[STEP] 9 t [2/14] B\n[STEP] 20 t [3/14] C\n",
    )
    assert [(r["step"], r["seconds"]) for r in st.measured(p)] == [
        ("[1/14] A", 4),
        ("[2/14] B", 11),
    ]


def test_collect_does_not_double_count_files_matched_by_two_globs(home, monkeypatch):
    _log(home, "daily_pipeline_20260715.log", "[STEP] 10 t [1/14] A\n[STEP] 70 t [2/14] B\n")
    monkeypatch.setattr(st, "LOG_GLOBS", [str(home / "daily_pipeline_*.log"), str(home / "*.log")])
    assert len(st.collect()) == 1


def test_collect_prefers_measured_falls_back_to_backfill_and_tags_rows(home, monkeypatch):
    monkeypatch.setattr(st, "LOG_GLOBS", [str(home / "daily_pipeline_*.log")])
    _log(home, "daily_pipeline_20260715.log", "[STEP] 10 t [1/14] A\n[STEP] 70 t [2/14] B\n")
    start = dt.datetime(2026, 7, 14, 0, 15, 0)
    _log(
        home, "daily_pipeline_20260714.log", "=== Daily pipeline Tue Jul 14 00:15:00 UTC 2026 ===\n"
    )
    _artifact(home, "indian_full_scan/indian_full_scan_x.xlsx", start + dt.timedelta(minutes=2))
    rows = st.collect()
    by_log = {}
    for r in rows:
        by_log.setdefault(r["log"], []).append(r)
    assert by_log["daily_pipeline_20260715.log"][0]["source"] == "measured"
    assert by_log["daily_pipeline_20260715.log"][0]["run_date"] == "2026-07-15"
    assert by_log["daily_pipeline_20260714.log"][0] == {
        "step": "[2/14]",
        "seconds": 120,
        "source": "est",
        "run_date": "2026-07-14",
        "log": "daily_pipeline_20260714.log",
    }
    assert [r["log"] for r in rows] == sorted(r["log"] for r in rows)  # files processed sorted


def test_collect_unparseable_date_in_name(home, monkeypatch):
    monkeypatch.setattr(st, "LOG_GLOBS", [str(home / "*.log")])
    _log(home, "weird.log", "[STEP] 1 t [1/14] A\n[STEP] 3 t [2/14] B\n")
    r = st.collect()[0]
    assert r["run_date"].startswith("?") and r["log"] == "weird.log"


def test_collect_no_logs(home, monkeypatch):
    monkeypatch.setattr(st, "LOG_GLOBS", [str(home / "none_*.log")])
    assert st.collect() == []


class _Con:
    def __init__(self, existing):
        self.existing, self.log = existing, []

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        if sql.lstrip().startswith("SELECT count"):
            key = tuple(params)
            return types.SimpleNamespace(fetchone=lambda: (1 if key in self.existing else 0,))
        return None

    def close(self):
        self.log.append(("close", None))


def test_store_inserts_only_new_rows(monkeypatch):
    con = _Con(existing={("2026-07-15", "[1/14] A")})
    monkeypatch.setitem(sys.modules, "duckdb", types.SimpleNamespace(connect=lambda: con))
    rows = [
        {
            "run_date": "2026-07-15",
            "step": "[1/14] A",
            "seconds": 5,
            "source": "measured",
            "log": "l",
        },
        {
            "run_date": "2026-07-15",
            "step": "[2/14] B",
            "seconds": 7,
            "source": "measured",
            "log": "l",
        },
    ]
    msg = st.store(rows)
    assert msg == "stored 1 new rows (idempotent per run_date+step)"
    inserts = [p for s, p in con.log if s.lstrip().startswith("INSERT")]
    assert inserts == [["2026-07-15", "[2/14] B", 7, "measured", "l"]]
    assert ("close", None) in con.log


def test_store_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    assert st.store([]).startswith("not stored (")


def _rows():
    return [
        {"run_date": "2026-07-14", "step": "[1/14] A", "seconds": 59, "source": "measured"},
        {"run_date": "2026-07-14", "step": "[2/14] B", "seconds": 3725, "source": "est"},
        {"run_date": "2026-07-15", "step": "[1/14] A", "seconds": 60, "source": "measured"},
    ]


def test_report_formats_times_orders_steps_and_totals(capsys):
    st.report(_rows(), slowest=2)
    out = capsys.readouterr().out
    assert "PER-STEP RUN TIMES (2 runs: 2026-07-14, 2026-07-15)" in out
    assert "59s" in out and "1m 00s" in out and "62m 05s" in out  # 59s stays seconds; 60s -> 1m 00s
    lines = [ln for ln in out.splitlines() if "2026-07-14" in ln and "/14]" in ln]
    assert "[2/14] B" in lines[0] and "[1/14] A" in lines[1]  # slowest first within a run
    assert "TOTAL" in out and "63m 04s" in out
    slow = out.split("SLOWEST STEPS")[1]
    assert slow.index("[2/14] B") < slow.index("[1/14] A")
    assert "max 62m 05s   runs=1" in slow and "max 1m 00s   runs=2" in slow


def test_report_exactly_sixty_seconds_is_shown_as_a_minute(capsys):
    st.report(
        [{"run_date": "2026-07-15", "step": "[1/14] A", "seconds": 60, "source": "measured"}], 3
    )
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "[1/14] A" in ln)
    assert "1m 00s" in line and " 60s" not in line


def test_report_slowest_ranks_by_worst_run_not_best(capsys):
    rows = [
        {"run_date": "2026-07-14", "step": "[1/14] SPIKY", "seconds": 1, "source": "measured"},
        {"run_date": "2026-07-15", "step": "[1/14] SPIKY", "seconds": 500, "source": "measured"},
        {"run_date": "2026-07-14", "step": "[2/14] STEADY", "seconds": 100, "source": "measured"},
        {"run_date": "2026-07-15", "step": "[2/14] STEADY", "seconds": 110, "source": "measured"},
    ]
    st.report(rows, 1)
    slow = capsys.readouterr().out.split("SLOWEST STEPS")[1]
    assert "SPIKY" in slow and "STEADY" not in slow


def test_report_slowest_limit_and_empty(capsys):
    st.report(_rows(), slowest=1)
    slow = capsys.readouterr().out.split("SLOWEST STEPS")[1]
    assert "[2/14] B" in slow and "[1/14] A" not in slow
    st.report([], 5)
    assert "no timings found" in capsys.readouterr().out


# ================================================================ risk_management
def test_module_constants():
    assert (rk.TARGET_VOL, rk.DD_HALT, rk.DD_RESUME) == (0.10, 0.15, 0.07)
    assert pytest.approx(np.sqrt(rk.S.ANN)) == rk.ANN


def test_stats_empty_and_constant_series_are_nan():
    for s in (pd.Series(dtype=float), pd.Series([0.01] * 5)):
        out = rk.stats(s)
        assert set(out) == {"ann_ret", "ann_vol", "sharpe", "maxdd"}
        assert all(np.isnan(v) for v in out.values())


def test_stats_known_values():
    r = pd.Series([0.10, -0.05, 0.02, 0.03])
    out = rk.stats(r)
    A = rk.S.ANN
    assert out["ann_ret"] == pytest.approx(r.mean() * A * 100)
    assert out["ann_vol"] == pytest.approx(r.std() * np.sqrt(A) * 100)
    assert out["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(A))
    # equity 1.10, 1.045, ... -> deepest drawdown is -5% from the 1.10 peak
    assert out["maxdd"] == pytest.approx(-5.0)


def test_stats_maxdd_is_zero_for_monotone_gains_and_negative_otherwise():
    assert rk.stats(pd.Series([0.01, 0.02, 0.03]))["maxdd"] == 0.0
    assert rk.stats(pd.Series([0.1, -0.5, 0.1]))["maxdd"] == pytest.approx(-50.0)


def _iv(vals):
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals), freq="W"))


def test_overlay_scale_is_capped_at_three_and_defaults_to_one_early():
    iv = _iv([0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001])
    out, halts = rk.apply_overlay(iv)
    # first 5 points have <6 observations -> NaN vol -> scale 1.0 (returns unchanged)
    assert out.iloc[:5].tolist() == pytest.approx(iv.iloc[:5].tolist())
    # tiny realised vol => scale capped at 3.0, never more
    assert (out.iloc[5:].abs() <= iv.iloc[5:].abs() * 3.0 + 1e-12).all()
    assert out.iloc[5] == pytest.approx(iv.iloc[5] * 3.0)
    assert halts == 0 and out.index.equals(iv.index)


def test_overlay_scales_down_when_realised_vol_exceeds_target():
    iv = _iv([0.10, -0.10] * 6)
    out, halts = rk.apply_overlay(iv)
    realised = iv.rolling(13, min_periods=6).std() * rk.ANN
    expected_scale = rk.TARGET_VOL / realised.iloc[-1]
    assert expected_scale < 1 and halts == 0
    assert out.iloc[-1] == pytest.approx(iv.iloc[-1] * expected_scale)


@pytest.fixture
def triple(monkeypatch):
    """TARGET_VOL huge => the vol scale is always capped at 3.0 once 6 observations exist
    (earlier points keep scale 1.0), which makes the kill-switch path hand-computable."""
    monkeypatch.setattr(rk, "TARGET_VOL", 1e6)


def test_overlay_kill_switch_halves_then_restores_exposure(triple):
    iv = _iv([0.0] * 5 + [-0.06, 0.1, 0.1])
    out, halts = rk.apply_overlay(iv)
    # crash bar at full exposure: -0.06*3 = -0.18 (-18% > 15% -> halt)
    # next bar halved: 0.1*3*0.5 = 0.15 -> equity 0.943 (dd -5.7% > -7%) -> restored
    # last bar at full exposure: 0.1*3
    assert out.tolist() == pytest.approx([0, 0, 0, 0, 0, -0.18, 0.15, 0.30])
    assert halts == 1


def test_overlay_stays_halved_until_drawdown_recovers_under_resume_level(triple):
    # after the halt, +0.02 scaled to 0.06*0.5=0.03 -> equity 0.8446 (dd -15.5%): still halved
    iv = _iv([0.0] * 5 + [-0.06, 0.02, 0.02])
    out, halts = rk.apply_overlay(iv)
    assert out.iloc[6] == pytest.approx(0.03)
    assert out.iloc[7] == pytest.approx(0.03)  # still at half exposure
    assert halts == 1


def test_overlay_no_halt_below_threshold_and_two_episodes_counted_twice(triple):
    _, halts = rk.apply_overlay(_iv([0.0] * 5 + [-0.04, 0.0, 0.0]))  # -12% drawdown only
    assert halts == 0
    out, halts = rk.apply_overlay(_iv([0.0] * 5 + [-0.06, 0.1, 0.1, -0.06, 0.0]))
    assert halts == 2
    assert out.iloc[7] == pytest.approx(0.30) and out.iloc[8] == pytest.approx(-0.18)


def test_overlay_empty_series():
    out, halts = rk.apply_overlay(pd.Series(dtype=float))
    assert out.empty and halts == 0
