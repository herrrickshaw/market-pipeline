"""Offline tests for consistency_audit (per-market scan-workbook checks and cross-market comparison)."""

import importlib
import json
import os
import sys
import time
import types

import pandas as pd
import pytest


@pytest.fixture
def ca(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the scan-workbook globs are relative to the working directory
    fake = types.ModuleType("scan_bhavcopy")
    fake.LIQ_FLOOR = 25_000_000.0  # the real module drags in nsepython (and sys.exit()s without it)
    monkeypatch.setitem(sys.modules, "scan_bhavcopy", fake)
    sys.modules.pop("consistency_audit", None)
    mod = importlib.import_module("consistency_audit")
    yield mod
    sys.modules.pop("consistency_audit", None)


def _workbook(path, rows=100, tiers=None, turnover=None, turnover_col="Turnover_USD", bars=250, dma=None,
              fundamentals=None, extra_sheet=None, drop=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame({"Symbol": [f"S{i}" for i in range(rows)]})
    d["Liquidity_Tier"] = tiers if tiers is not None else ["LIQUID"] * rows
    d[turnover_col] = turnover if turnover is not None else [1e6] * rows
    if bars is not None:
        d["Data_Points"] = [bars] * rows
    if dma is not None:
        d["200_Day_MA"] = dma
    for c in drop:
        d = d.drop(columns=c)
    with pd.ExcelWriter(path) as w:
        d.to_excel(w, sheet_name="All_Stocks", index=False)
        if fundamentals is not None:
            pd.DataFrame({"Piotroski_Score": fundamentals}).to_excel(w, sheet_name=extra_sheet or "Fundamentals", index=False)


# ------------------------------------------------------------------ floors
def test_floors_have_per_market_currency(ca):
    assert set(ca.FLOORS) == {"India", "US", "Europe", "Japan", "Korea"}
    assert ca.FLOORS["India"].currency == "INR" and ca.FLOORS["US"].currency == "USD"
    assert ca.FLOORS["India"].amount == 25_000_000.0  # taken from scan_bhavcopy.LIQ_FLOOR
    assert ca.COLUMN_CURRENCY == {"India": "INR", "US": "USD", "Europe": "USD", "Japan": "USD", "Korea": "USD"}
    assert ca.FLOORS["India"].amount > 0 and ca.FLOORS["Japan"].amount == ca.FLOORS["Europe"].amount == ca.FLOORS["US"].amount


def test_floors_fall_back_to_defaults_when_imports_fail(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "adaptive_liquidity", None)
    monkeypatch.setitem(sys.modules, "scan_bhavcopy", None)
    sys.modules.pop("consistency_audit", None)
    mod = importlib.import_module("consistency_audit")
    assert mod.FLOORS["US"].amount == 10000.0 and mod.FLOORS["India"].amount == 10000000.0
    sys.modules.pop("consistency_audit", None)


@pytest.mark.xfail(
    strict=True,
    raises=SystemExit,
    reason="_floors() imports scan_bhavcopy inside `except Exception`, but that module pulls in "
    "full_indian_market_scan which calls sys.exit() when nsepython is missing; SystemExit is not an "
    "Exception, so importing consistency_audit itself dies (consistency_audit.py _floors)",
)
def test_import_survives_a_missing_nsepython(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "nsepython", None)
    for name in ("scan_bhavcopy", "full_indian_market_scan", "consistency_audit"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module("consistency_audit")
    assert mod.FLOORS["India"].amount == 10000000.0  # falls back to the default floor
    for name in ("scan_bhavcopy", "full_indian_market_scan", "consistency_audit"):
        sys.modules.pop(name, None)


def test_latest_picks_last_sorted_match(ca, tmp_path):
    assert ca._latest("nothing_*.xlsx") is None
    for n in ("b", "a", "c"):
        (tmp_path / f"scan_{n}.xlsx").write_text("x")
    assert ca._latest("scan_*.xlsx") == "scan_c.xlsx"


# ------------------------------------------------------------------ audit_market
def _audit(ca, tmp_path, name="US", **kw):
    f = tmp_path / "us" / "scan_2026.xlsx"
    _workbook(f, **kw)
    return ca.audit_market(name, "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")


def test_audit_market_clean(ca, tmp_path):
    r = _audit(ca, tmp_path, rows=100, dma=[10.0] * 100, fundamentals=[5, 6, 7])
    assert r["issues"] == [] and r["rows"] == 100 and r["bars"] == 250.0 and r["tiers"] == {"LIQUID": 100}
    assert r["below_floor"] == 0 and r["dma200_pct"] == 100 and r["piotroski"] == 3 and r["file"] == "scan_2026.xlsx"
    assert r["floor"] == f"{ca.FLOORS['US'].amount:,.0f} USD" and r["age_min"] >= 0


def test_audit_market_missing_or_unreadable(ca, tmp_path):
    r = ca.audit_market("US", "nothing/*.xlsx", "Symbol", "LTP", "Turnover_USD")
    assert r["issues"] == [("ERROR", "no scan workbook found")] and r["file"] is None
    bad = tmp_path / "us" / "scan_bad.xlsx"
    bad.parent.mkdir()
    bad.write_text("not an excel file")
    r = ca.audit_market("US", "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")
    assert r["issues"][0][0] == "ERROR" and r["issues"][0][1].startswith("unreadable:") and "rows" not in r


def test_audit_market_age_is_in_minutes(ca, tmp_path):
    f = tmp_path / "us" / "scan_2026.xlsx"
    _workbook(f)
    old = time.time() - 125 * 60
    os.utime(f, (old, old))
    assert ca.audit_market("US", "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")["age_min"] in (125, 126)


def test_liquidity_gate_checks(ca, tmp_path):
    r = _audit(ca, tmp_path, drop=("Liquidity_Tier",))
    assert ("ERROR", "no liquidity gate (Liquidity_Tier absent)") in r["issues"] and r["tiers"] == 0
    r = _audit(ca, tmp_path, rows=10, tiers=["UNKNOWN"] * 10)
    assert r["issues"][0] == ("ERROR", "gate INACTIVE — all 10 rows UNKNOWN (FX unavailable at scan time?)")
    # 5% UNKNOWN is tolerated, above 5% warns
    r = _audit(ca, tmp_path, rows=100, tiers=["UNKNOWN"] * 5 + ["LIQUID"] * 95)
    assert not any("UNKNOWN" in m for _, m in r["issues"])
    r = _audit(ca, tmp_path, rows=100, tiers=["UNKNOWN"] * 6 + ["LIQUID"] * 94)
    assert ("WARN", "6 rows UNKNOWN (6%)") in r["issues"]


def test_turnover_floor_checks(ca, tmp_path):
    floor = ca.FLOORS["US"].amount
    r = _audit(ca, tmp_path, rows=10, turnover=[floor * 10] * 10)
    assert r["below_floor"] == 0 and r["issues"] == []
    r = _audit(ca, tmp_path, rows=10, turnover=[floor * 10] * 8 + [floor - 1, floor])
    assert r["below_floor"] == 1  # a turnover exactly at the floor is not below it
    assert r["issues"] == [("WARN", f"1 rows below the {floor:,.0f} USD floor")]
    r = _audit(ca, tmp_path, rows=3, turnover=[1.0, 2.0, 3.0])
    assert r["issues"] == [("ERROR", "every row below the liquidity floor — gate not applied")]
    r = _audit(ca, tmp_path, rows=5, turnover_col="Other")
    assert ("WARN", "no turnover column (Turnover_USD)") in r["issues"] and "below_floor" not in r


def test_turnover_unknown_market_and_bad_sample(ca, tmp_path):
    r = _audit(ca, tmp_path, name="Mars")
    assert r["issues"] == [("WARN", "no floor configured for Mars")]
    r = _audit(ca, tmp_path, rows=3, turnover=[float("nan")] * 3)  # no numeric sample at all: no currency check, all NaN
    assert r["below_floor"] == 0


def test_turnover_currency_and_unusable_sample_guards(ca, tmp_path, monkeypatch):
    f = tmp_path / "us" / "scan_2026.xlsx"
    _workbook(f)

    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def __lt__(self, other):
            raise self.exc

    monkeypatch.setattr(ca, "Money", lambda a, c: Boom(TypeError("INR vs USD")))
    r = ca.audit_market("US", "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")
    assert r["issues"][-1] == ("ERROR", "currency mismatch: INR vs USD") and "below_floor" not in r
    monkeypatch.setattr(ca, "Money", lambda a, c: Boom(ca.UnsafePrice("NaN money")))
    r = ca.audit_market("US", "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")
    assert r["issues"][-1] == ("WARN", "unusable turnover sample: NaN money")


def test_dma_and_window_checks(ca, tmp_path):
    r = _audit(ca, tmp_path, rows=10, dma=[1.0] * 5 + [float("nan")] * 5)
    assert r["dma200_pct"] == 50 and not any("200-DMA" in m for _, m in r["issues"])  # exactly half is accepted
    r = _audit(ca, tmp_path, rows=10, dma=[1.0] * 4 + [float("nan")] * 6)
    assert r["dma200_pct"] == 40
    assert ("ERROR", "reports a 200-DMA but only 40% populated (median bars=250.0, needs 200)") in r["issues"]
    r = _audit(ca, tmp_path, rows=10, bars=199, dma=[1.0] * 10)
    assert ("ERROR", "window too short: 199 bars < 200") in r["issues"]
    r = _audit(ca, tmp_path, rows=10, bars=200, dma=[1.0] * 10)
    assert not any("window too short" in m for _, m in r["issues"])  # 200 bars is enough
    r = _audit(ca, tmp_path, rows=10, bars=50)  # a short window is fine when no 200-DMA is reported
    assert r["issues"] == []
    r = _audit(ca, tmp_path, rows=10, bars=None, dma=[1.0] * 10)
    assert r["bars"] is None and not any("window" in m for _, m in r["issues"])


def test_fundamentals_sheet_checks(ca, tmp_path):
    r = _audit(ca, tmp_path, rows=5, fundamentals=[1.0, None, 3.0])
    assert r["piotroski"] == 2 and r["issues"] == []
    r = _audit(ca, tmp_path, rows=5, fundamentals=[None, None])
    assert r["piotroski"] == 0 and r["issues"] == [("WARN", "'Fundamentals' sheet present but 0 Piotroski scores — fundamentals unavailable this run")]
    r = _audit(ca, tmp_path, rows=5)
    assert r["piotroski"] == 0 and r["issues"] == []  # no fundamentals sheet at all is not flagged here
    r = _audit(ca, tmp_path, rows=5, fundamentals=[1.0], extra_sheet="Fundamental_Scores")
    assert r["piotroski"] == 1  # any sheet name containing 'undamental'


def test_fundamentals_sheet_without_score_column(ca, tmp_path):
    f = tmp_path / "us" / "scan_2026.xlsx"
    _workbook(f, rows=5)
    with pd.ExcelWriter(f, mode="a") as w:
        pd.DataFrame({"Other": [1]}).to_excel(w, sheet_name="Fundamentals", index=False)
    r = ca.audit_market("US", "us/scan_*.xlsx", "Symbol", "LTP", "Turnover_USD")
    assert r["piotroski"] == 0 and r["issues"][0][0] == "WARN"


# ------------------------------------------------------------------ main
def _patch_markets(ca, monkeypatch, results):
    monkeypatch.setattr(ca, "MARKETS", [(r["market"],) for r in results])
    by = {r["market"]: r for r in results}
    monkeypatch.setattr(ca, "audit_market", lambda name: by[name])


def _res(market, rows=100, bars=250.0, piotroski=3, issues=()):
    return {"market": market, "file": f"{market}.xlsx", "rows": rows, "bars": bars, "piotroski": piotroski,
            "issues": list(issues), "age_min": 5, "dma200_pct": 100, "below_floor": 0}


def test_main_clean_text_report(ca, monkeypatch, capsys):
    _patch_markets(ca, monkeypatch, [_res("India"), _res("US")])
    monkeypatch.setattr(sys, "argv", ["x"])
    assert ca.main() == 0
    out = capsys.readouterr().out
    assert "CROSS-MARKET CONSISTENCY AUDIT" in out and "✓ all markets consistent" in out
    assert "India" in out and "5m" in out and "ANOMALY" not in out


def test_main_errors_and_warnings_set_exit_code(ca, monkeypatch, capsys):
    _patch_markets(ca, monkeypatch, [_res("India", issues=[("WARN", "w1")]), _res("US", issues=[("ERROR", "e1")])])
    monkeypatch.setattr(sys, "argv", ["x"])
    assert ca.main() == 1
    out = capsys.readouterr().out
    assert "1 ANOMALY(S)" in out and "US         e1" in out and "1 warning(s)" in out and "India      w1" in out
    _patch_markets(ca, monkeypatch, [_res("India", issues=[("WARN", "w1")])])
    assert ca.main() == 0  # warnings alone do not fail the run


def test_main_cross_market_findings(ca, monkeypatch, capsys):
    res = [_res("India", bars=1000.0, piotroski=0), _res("US", bars=250.0, piotroski=5), _res("Japan", bars=90.0, piotroski=5)]
    _patch_markets(ca, monkeypatch, res)
    monkeypatch.setattr(sys, "argv", ["x", "--json"])
    assert ca.main() == 1  # the missing-fundamentals finding is an ERROR
    out = json.loads(capsys.readouterr().out)
    cross = next(r for r in out if r["market"] == "cross-market")["issues"]
    assert ("WARN", "history windows differ 11x (90-1000 bars; India widest) — same screeners, different lookbacks") in [tuple(c) for c in cross]
    assert any(lv == "ERROR" and msg.startswith("India has NO fundamentals while US, Japan do") for lv, msg in cross)


def test_main_cross_market_thresholds(ca, monkeypatch, capsys):
    # windows differ exactly 3x: not flagged; just above 3x: flagged
    _patch_markets(ca, monkeypatch, [_res("India", bars=300.0), _res("US", bars=100.0)])
    monkeypatch.setattr(sys, "argv", ["x", "--json"])
    assert ca.main() == 0
    assert not any(r["market"] == "cross-market" for r in json.loads(capsys.readouterr().out))
    _patch_markets(ca, monkeypatch, [_res("India", bars=301.0), _res("US", bars=100.0)])
    ca.main()
    assert any(r["market"] == "cross-market" for r in json.loads(capsys.readouterr().out))
    # a market without Data_Points among several is called out; when every market lacks it, nothing is
    _patch_markets(ca, monkeypatch, [_res("India", bars=None), _res("US", bars=250.0), _res("Japan", bars=250.0)])
    ca.main()
    cross = next(r for r in json.loads(capsys.readouterr().out) if r["market"] == "cross-market")["issues"]
    assert cross == [["WARN", "India lacks Data_Points — history depth unverifiable, so the 200-DMA check cannot be applied there"]]
    _patch_markets(ca, monkeypatch, [_res("India", bars=None), _res("US", bars=None)])
    ca.main()
    assert not any(r["market"] == "cross-market" for r in json.loads(capsys.readouterr().out))


def test_main_real_workbooks_end_to_end(ca, monkeypatch, tmp_path, capsys):
    paths = {"India": "indian_full_scan/indian_full_scan_1.xlsx", "US": "us_full_scan/us_full_scan_1.xlsx"}
    _workbook(tmp_path / paths["India"], turnover_col="Median_Turnover", turnover=[5e7] * 100, bars=250)
    _workbook(tmp_path / paths["US"], bars=250)
    monkeypatch.setattr(sys, "argv", ["x", "--json"])
    rc = ca.main()
    res = json.loads(capsys.readouterr().out)
    by = {r["market"]: r for r in res}
    assert by["India"]["issues"] == [] and by["US"]["issues"] == []
    assert by["Europe"]["issues"] == [["ERROR", "no scan workbook found"]] and rc == 1
