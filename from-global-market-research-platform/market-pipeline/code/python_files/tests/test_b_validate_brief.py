"""Offline tests for validate_brief (external cross-check of the India scan)."""

import datetime as dt
import json
import sys
import types

import pandas as pd
import pytest

import validate_brief as vb


# ---------------------------------------------------------------- _latest_scan
def test_latest_scan_is_cwd_relative_and_sorted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert vb._latest_scan() == ""
    d = tmp_path / "indian_full_scan"
    d.mkdir()
    for n in ("indian_full_scan_20260101_0300.xlsx", "indian_full_scan_20260201_0300.xlsx"):
        (d / n).write_text("x")
    assert vb._latest_scan().endswith("indian_full_scan_20260201_0300.xlsx")


# ---------------------------------------------------------------- _scan_staleness_note
def _name(days_old):
    d = dt.date.today() - dt.timedelta(days=days_old)
    return f"indian_full_scan/indian_full_scan_{d:%Y%m%d}_0300.xlsx"


def test_staleness_note_empty_for_today_and_future():
    assert vb._scan_staleness_note(_name(0)) == ""
    assert vb._scan_staleness_note(_name(-1)) == ""


def test_staleness_note_names_age_and_file():
    note = vb._scan_staleness_note(_name(3))
    assert "SCAN IS 3 DAY(S) OLD" in note
    assert f"expected {dt.date.today():%Y-%m-%d}" in note
    assert note.startswith("  ")
    assert "indian_full_scan_" in note and "stale data, not a bad pick" in note
    assert "1 DAY(S)" in vb._scan_staleness_note(_name(1))


@pytest.mark.parametrize(
    "path",
    [
        "indian_full_scan/other.xlsx",
        "x_20260101.xlsx",
        "indian_full_scan/indian_full_scan_20261399_0300.xlsx",
        "",
    ],
)
def test_staleness_note_unparseable_names_are_silent(path):
    assert vb._scan_staleness_note(path) == ""


# ---------------------------------------------------------------- _fetch
class _Resp:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text


def _fake_bs4(monkeypatch, top_text="", has_top=True, boom=False):
    class Node:
        def get_text(self, sep, strip):
            return top_text

    class Soup:
        def __init__(self, html, parser):
            if boom:
                raise RuntimeError("parser exploded " * 10)
            self.parser = parser

        def select_one(self, sel):
            assert sel == "#top"
            return Node() if has_top else None

        def get_text(self, sep, strip):
            return top_text

    monkeypatch.setitem(sys.modules, "bs4", types.SimpleNamespace(BeautifulSoup=Soup))


def _get(monkeypatch, resp):
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(vb.requests, "get", get)
    return seen


def test_fetch_parses_price_change_and_close_date(monkeypatch):
    seen = _get(monkeypatch, _Resp(200, "<html/>"))
    _fake_bs4(monkeypatch, "Infosys Ltd ₹ 1,099.50 -1.72% 14 Jul - close price 52w high")
    out = vb._fetch("INFY")
    assert out == {"price": 1099.5, "change": -1.72, "date": "14 Jul"}
    assert seen["url"] == "https://www.screener.in/company/INFY/" and seen["timeout"] == 25


def test_fetch_optional_fields(monkeypatch):
    _get(monkeypatch, _Resp(200, ""))
    _fake_bs4(monkeypatch, "Foo ₹ 289 live")
    assert vb._fetch("X") == {"price": 289.0, "change": None, "date": None}
    _fake_bs4(monkeypatch, "Foo ₹ 5 +0.5% 3 Jan close price")
    assert vb._fetch("X") == {"price": 5.0, "change": 0.5, "date": "3 Jan"}


def test_fetch_falls_back_to_whole_soup_without_top(monkeypatch):
    _get(monkeypatch, _Resp(200, ""))
    _fake_bs4(monkeypatch, "Foo ₹ 77", has_top=False)
    assert vb._fetch("X")["price"] == 77.0


def test_fetch_reads_price_deep_in_the_header_text(monkeypatch):
    _get(monkeypatch, _Resp(200, ""))
    _fake_bs4(
        monkeypatch, "word " * 60 + "₹ 321 +1.0% 14 Jul - close price"
    )  # price after char 300
    assert vb._fetch("X")["price"] == 321.0


def test_fetch_truncates_header_text_to_400_chars(monkeypatch):
    _get(monkeypatch, _Resp(200, ""))
    _fake_bs4(monkeypatch, "x" * 400 + " ₹ 99")
    assert vb._fetch("X") == {"error": "no price in header"}


def test_fetch_error_paths(monkeypatch):
    _get(monkeypatch, RuntimeError("conn refused " * 10))
    out = vb._fetch("X")
    assert (
        out["error"].startswith("fetch: conn refused") and len(out["error"]) <= len("fetch: ") + 60
    )
    _get(monkeypatch, _Resp(404))
    assert vb._fetch("X") == {"error": "404"}
    _get(monkeypatch, _Resp(503))
    assert vb._fetch("X") == {"error": "http 503"}
    _get(monkeypatch, _Resp(200, "<html/>"))
    _fake_bs4(monkeypatch, boom=True)
    assert vb._fetch("X")["error"].startswith("parse: parser exploded")
    _fake_bs4(monkeypatch, "no price here")
    assert vb._fetch("X") == {"error": "no price in header"}


# ---------------------------------------------------------------- validate
def _workbook(tmp_path, monkeypatch, rows, name=None):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "indian_full_scan"
    d.mkdir(exist_ok=True)
    name = name or f"indian_full_scan_{dt.date.today():%Y%m%d}_0300.xlsx"
    with pd.ExcelWriter(d / name) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="All_Stocks", index=False)
    return name


def _screener(monkeypatch, table, calls=None):
    def fetch(sym):
        if calls is not None:
            calls.append(sym)
        return table[sym]

    monkeypatch.setattr(vb, "_fetch", fetch)
    sleeps = []
    monkeypatch.setattr(vb.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _stocks(n, price=100.0, start_turnover=100):
    return [
        {"Symbol": f"S{i}", "LTP": price, "Median_Turnover": start_turnover - i} for i in range(n)
    ]


def _site(price=100.0, date="14 Jul"):
    return {"price": price, "change": 0.1, "date": date}


def test_validate_no_scan(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert vb.validate(6, 2.0) == {
        "ok": False,
        "reason": "no India scan workbook found",
        "checks": [],
    }


def test_validate_happy_path(monkeypatch, tmp_path):
    name = _workbook(tmp_path, monkeypatch, _stocks(10))
    calls = []
    sleeps = _screener(monkeypatch, {f"S{i}": _site() for i in range(10)}, calls)
    r = vb.validate(4, 2.0)
    assert r["ok"] is True and r["reason"] == "" and r["scan"] == name
    assert r["verified"] == 4 and r["dated"] == 4 and r["close_date"] == "14 Jul"
    assert calls == ["S0", "S1", "S2", "S3"]  # stops once the sample is verified
    assert sleeps == [0.6] * 4
    assert all(c["status"] == "ok" and c["diff_pct"] == 0.0 for c in r["checks"])


def test_validate_orders_by_turnover_and_oversamples_three_x(monkeypatch, tmp_path):
    rows = [{"Symbol": f"S{i}", "LTP": 100.0, "Median_Turnover": i} for i in range(20)]
    _workbook(tmp_path, monkeypatch, rows)
    calls = []
    _screener(monkeypatch, {f"S{i}": {"error": "404"} for i in range(20)}, calls)
    r = vb.validate(2, 2.0)
    assert calls == [f"S{i}" for i in range(19, 13, -1)]  # head(sample*3)=6, highest turnover first
    assert r["verified"] == 0 and r["ok"] is False
    assert r["reason"] == "only 0 of 2 verifiable (need 3) — cannot confirm"
    assert all(c == {"symbol": c["symbol"], "status": "skip", "why": "404"} for c in r["checks"])


def test_validate_without_turnover_column_keeps_file_order(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, [{"Symbol": s, "LTP": 100.0} for s in ("B", "A", "C", "D")])
    calls = []
    _screener(monkeypatch, {s: _site() for s in "ABCD"}, calls)
    vb.validate(3, 2.0)
    assert calls == ["B", "A", "C"]


def test_validate_skips_unfetchable_names_and_keeps_going(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(8))
    table = {f"S{i}": _site() for i in range(8)}
    table["S0"] = {"error": "404"}
    table["S2"] = {"error": "http 503"}
    _screener(monkeypatch, table)
    r = vb.validate(3, 2.0)
    assert r["ok"] is True and r["verified"] == 3
    assert [c["status"] for c in r["checks"]] == ["skip", "ok", "skip", "ok", "ok"]


def test_validate_price_tolerance_boundary_and_mismatch(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(6, price=102.0))
    _screener(monkeypatch, {f"S{i}": _site(100.0) for i in range(6)})
    r = vb.validate(3, 2.0)  # |102-100|/100 = 2.0% == tolerance -> still ok
    assert r["ok"] is True and r["checks"][0]["diff_pct"] == 2.0
    r = vb.validate(3, 1.99)
    assert r["ok"] is False and r["reason"] == "3 price mismatch(es) beyond 1.99%"
    assert {c["status"] for c in r["checks"]} == {"MISMATCH"}


def test_validate_diff_is_relative_to_their_price(monkeypatch, tmp_path):
    _workbook(
        tmp_path,
        monkeypatch,
        [
            {"Symbol": "A", "LTP": 110.0, "Median_Turnover": 3},
            {"Symbol": "B", "LTP": 100.0, "Median_Turnover": 2},
            {"Symbol": "C", "LTP": 100.0, "Median_Turnover": 1},
        ],
    )
    _screener(monkeypatch, {"A": _site(100.0), "B": _site(100.0), "C": _site(100.0)})
    r = vb.validate(3, 20.0)
    assert r["checks"][0]["diff_pct"] == 10.0 and r["ok"] is True


def test_validate_zero_screener_price_is_a_mismatch(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(4))
    _screener(monkeypatch, {f"S{i}": _site(0.0) for i in range(4)})
    r = vb.validate(3, 2.0)
    assert r["checks"][0]["diff_pct"] == 999 and r["ok"] is False


def test_validate_requires_min_verified(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(8))
    table = {f"S{i}": _site() for i in range(2)}
    table.update({f"S{i}": {"error": "404"} for i in range(2, 8)})
    _screener(monkeypatch, table)
    r = vb.validate(3, 2.0)
    assert r["verified"] == 2 and r["ok"] is False
    assert r["reason"] == "only 2 of 3 verifiable (need 3) — cannot confirm"


def test_validate_date_disagreement_fails_even_for_one_name(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(8))
    table = {f"S{i}": _site(date="14 Jul") for i in range(8)}
    table["S1"] = _site(date="29 May")  # one stale close among the majority
    _screener(monkeypatch, table)
    r = vb.validate(4, 2.0)
    assert r["ok"] is False and r["close_date"] == "14 Jul"
    assert r["reason"].startswith("close dates disagree — ['S1'] vs majority 14 Jul")


def test_validate_live_quotes_make_dates_unverifiable(monkeypatch, tmp_path):
    _workbook(tmp_path, monkeypatch, _stocks(8))
    table = {f"S{i}": _site(date=None) for i in range(8)}
    _screener(monkeypatch, table)
    r = vb.validate(4, 2.0)
    assert r["ok"] is False and r["dated"] == 0 and r["close_date"] is None
    assert r["reason"].startswith("close date unverifiable — only 0/4 names showed")


@pytest.mark.parametrize(
    "verified,dated,unverifiable",
    [
        (3, 2, False),
        (3, 1, True),
        (4, 2, False),
        (4, 1, True),
        (5, 3, False),
        (5, 2, True),
        (6, 3, False),
    ],
)
def test_validate_dated_threshold_is_half_rounded_up_min_two(
    monkeypatch, tmp_path, verified, dated, unverifiable
):
    _workbook(tmp_path, monkeypatch, _stocks(verified * 3))
    table = {f"S{i}": _site(date="14 Jul" if i < dated else None) for i in range(verified * 3)}
    _screener(monkeypatch, table)
    r = vb.validate(verified, 2.0)
    assert r["verified"] == verified and r["dated"] == dated
    assert (r["reason"].startswith("close date unverifiable")) is unverifiable


def test_validate_reason_precedence(monkeypatch, tmp_path):
    # not enough verified beats everything; mismatch beats date problems
    _workbook(tmp_path, monkeypatch, _stocks(6, price=150.0))
    _screener(monkeypatch, {f"S{i}": _site(date=None) for i in range(6)})
    r = vb.validate(3, 2.0)
    assert "price mismatch" in r["reason"]  # mismatch reported before 'date unverifiable'


def test_validate_prints_staleness_note(monkeypatch, tmp_path, capsys):
    _workbook(
        tmp_path,
        monkeypatch,
        _stocks(4),
        name=f"indian_full_scan_{(dt.date.today() - dt.timedelta(days=2)):%Y%m%d}_0300.xlsx",
    )
    _screener(monkeypatch, {f"S{i}": _site() for i in range(4)})
    vb.validate(3, 2.0)
    assert "SCAN IS 2 DAY(S) OLD" in capsys.readouterr().out


# ---------------------------------------------------------------- main
def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["validate_brief.py", *args])
    return vb.main()


def test_main_json_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(vb, "validate", lambda n, t: {"ok": True, "checks": [], "x": n, "t": t})
    assert _main(monkeypatch, "--json", "--sample", "9", "--tolerance", "1.5") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["x"] == 9 and out["t"] == 1.5
    monkeypatch.setattr(vb, "validate", lambda n, t: {"ok": False, "checks": []})
    assert _main(monkeypatch, "--json") == 1


def test_main_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(
        vb,
        "validate",
        lambda n, t: (
            seen.append((n, t)) or {"ok": True, "checks": [], "verified": 0, "close_date": None}
        ),
    )
    _main(monkeypatch)
    assert seen == [(6, 2.0)]


def test_main_text_output_pass_and_fail(monkeypatch, capsys):
    checks = [
        {
            "symbol": "AAA",
            "status": "ok",
            "ours": 100.0,
            "screener": 100.5,
            "diff_pct": 0.5,
            "date": "14 Jul",
        },
        {
            "symbol": "BBB",
            "status": "MISMATCH",
            "ours": 100.0,
            "screener": 90.0,
            "diff_pct": 11.11,
            "date": None,
        },
        {"symbol": "CCC", "status": "skip", "why": "404"},
    ]
    monkeypatch.setattr(
        vb,
        "validate",
        lambda n, t: {
            "ok": True,
            "scan": "scan.xlsx",
            "verified": 3,
            "close_date": "14 Jul",
            "checks": checks,
            "reason": "",
        },
    )
    assert _main(monkeypatch) == 0
    out = capsys.readouterr().out
    assert "validating scan.xlsx against screener.in (tolerance 2.0%)" in out
    assert (
        "ok AAA" in out
        and "!! BBB" in out
        and "- " in out
        and "CCC" in out
        and "skipped (404)" in out
    )
    assert "✓ validated: 3 names agree, close date 14 Jul" in out
    monkeypatch.setattr(
        vb,
        "validate",
        lambda n, t: {"ok": False, "scan": "s.xlsx", "checks": [], "reason": "stale"},
    )
    assert _main(monkeypatch) == 1
    out = capsys.readouterr().out
    assert "BRIEF FAILED EXTERNAL VALIDATION — DO NOT SEND" in out and "stale" in out
