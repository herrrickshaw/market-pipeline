"""Offline tests for validate_convergence (two-source price check, verdict logic)."""

import json
import sys

import pandas as pd
import pytest

import validate_convergence as vc


# ---------------------------------------------------------------- _get
class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


def test_get_decodes_with_replacement_and_sends_ua(monkeypatch):
    seen = {}

    def urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        seen["timeout"] = timeout
        return _Resp(b"ok \xff end")

    monkeypatch.setattr(vc.urllib.request, "urlopen", urlopen)
    assert vc._get("http://x") == "ok � end"
    assert seen["ua"].startswith("Mozilla/5.0") and seen["timeout"] == 20
    vc._get("http://x", timeout=5)
    assert seen["timeout"] == 5


def test_get_returns_none_on_any_error(monkeypatch):
    def boom(req, timeout):
        raise OSError("refused")

    monkeypatch.setattr(vc.urllib.request, "urlopen", boom)
    assert vc._get("http://x") is None


# ---------------------------------------------------------------- screener_price
def _page(body):
    return f"<html><body>{body}</body></html>"


def test_screener_price_parses_rupee_price_and_close_date(monkeypatch):
    html = _page('<span class="a">₹ 1,234.50</span> <span>12 Jul - close price</span>')
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: html)
    assert vc.screener_price("TCS") == {"ok": True, "price": 1234.5, "close_date": "12 Jul"}


def test_screener_price_falls_back_to_rs_and_tolerates_missing_date(monkeypatch):
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: _page("Rs. 99"))
    assert vc.screener_price("X") == {"ok": True, "price": 99.0, "close_date": None}
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: _page("Rs 5,000"))
    assert vc.screener_price("X")["price"] == 5000.0


def test_screener_price_prefers_rupee_symbol_over_rs(monkeypatch):
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: _page("Rs. 1 and ₹ 2"))
    assert vc.screener_price("X")["price"] == 2.0


def test_screener_price_failures(monkeypatch):
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: None)
    assert vc.screener_price("X") == {"ok": False, "reason": "unreachable"}
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: _page("nothing here"))
    assert vc.screener_price("X") == {"ok": False, "reason": "no price on page"}


def test_screener_price_requests_symbol_url(monkeypatch):
    urls = []
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: urls.append(url))
    vc.screener_price("RELIANCE")
    assert urls == ["https://www.screener.in/company/RELIANCE/"]


# ---------------------------------------------------------------- mc_code
def _suggest(*rows):
    return json.dumps([{"pdt_dis_nm": d, "link_src": l} for d, l in rows])


def test_mc_code_requires_exact_symbol_match(monkeypatch):
    body = _suggest(
        ("<b>Reliance Power</b>&nbsp;INE614G01033, RELIANCEPOWER, 532939", "https://x/y/RPOW/"),
        ("<span>Reliance Industries</span>&nbsp;INE002A01018, reliance, 500325", "https://x/y/RI"),
    )
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: body)
    assert vc.mc_code("RELIANCE") == "RI"  # case-insensitive exact, skips RELIANCEPOWER
    assert vc.mc_code("RELIANCEPOWER") == "RPOW"  # trailing slash stripped


def test_mc_code_no_match_empty_or_bad_json(monkeypatch):
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: _suggest(("Foo, BAR", "x/y/Z")))
    assert vc.mc_code("RELIANCE") is None
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: None)
    assert vc.mc_code("X") is None
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: "")
    assert vc.mc_code("X") is None
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: "not json")
    assert vc.mc_code("X") is None
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: json.dumps({"not": "a list"}))
    assert vc.mc_code("X") is None


def test_mc_code_quotes_symbol_in_url(monkeypatch):
    urls = []
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: urls.append(url))
    vc.mc_code("M&M")
    assert "query=M%26M&" in urls[0]


# ---------------------------------------------------------------- mc_price
def test_mc_price_paths(monkeypatch):
    monkeypatch.setattr(vc, "mc_code", lambda s: None)
    assert vc.mc_price("X") == {"ok": False, "reason": "symbol not resolved"}
    monkeypatch.setattr(vc, "mc_code", lambda s: "RI")
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: None)
    assert vc.mc_price("X") == {"ok": False, "reason": "unreachable"}
    monkeypatch.setattr(
        vc, "_get", lambda url, timeout=20: json.dumps({"data": {"pricecurrent": "101.5"}})
    )
    assert vc.mc_price("X") == {"ok": True, "price": 101.5, "code": "RI"}
    both = json.dumps({"data": {"pricecurrent": 101.5, "HP": 999}})
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: both)
    assert vc.mc_price("X")["price"] == 101.5  # pricecurrent wins over the high-price field
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: json.dumps({"data": {"HP": 88}}))
    assert vc.mc_price("X")["price"] == 88.0  # falls back to HP
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: json.dumps({"data": {}}))
    assert vc.mc_price("X") == {"ok": False, "reason": "no price field"}
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: json.dumps({}))
    assert vc.mc_price("X") == {"ok": False, "reason": "no price field"}
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: "garbage")
    assert vc.mc_price("X") == {"ok": False, "reason": "parse: JSONDecodeError"}
    monkeypatch.setattr(
        vc, "_get", lambda url, timeout=20: json.dumps({"data": {"pricecurrent": "n/a"}})
    )
    assert vc.mc_price("X") == {"ok": False, "reason": "parse: ValueError"}


def test_mc_price_uses_resolved_code_in_url(monkeypatch):
    urls = []
    monkeypatch.setattr(vc, "mc_code", lambda s: "RI")
    monkeypatch.setattr(vc, "_get", lambda url, timeout=20: urls.append(url))
    vc.mc_price("RELIANCE")
    assert urls == ["https://priceapi.moneycontrol.com/pricefeed/nse/equitycash/RI"]


# ---------------------------------------------------------------- _latest_scan
def test_latest_scan_picks_lexicographically_last(monkeypatch, tmp_path):
    monkeypatch.setattr(vc, "HERE", tmp_path)
    assert vc._latest_scan() is None
    d = tmp_path / "indian_full_scan"
    d.mkdir()
    for n in ("indian_full_scan_20260101.xlsx", "indian_full_scan_20260301.xlsx", "other.xlsx"):
        (d / n).write_text("x")
    assert vc._latest_scan().name == "indian_full_scan_20260301.xlsx"


# ---------------------------------------------------------------- run
def _scan(tmp_path, rows):
    p = tmp_path / "scan.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="All_Stocks", index=False)
    return p


def _setup(monkeypatch, tmp_path, rows, screener, mc):
    monkeypatch.setattr(vc, "_latest_scan", lambda: _scan(tmp_path, rows))
    monkeypatch.setattr(vc, "screener_price", lambda s: screener[s])
    monkeypatch.setattr(vc, "mc_price", lambda s: mc[s])
    sleeps = []
    monkeypatch.setattr(vc.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _run_json(capsys, sample=10):
    assert vc.run(sample, True) in (0, 1)
    return json.loads(capsys.readouterr().out)


OK = lambda p: {"ok": True, "price": p}  # noqa: E731
BAD = {"ok": False, "reason": "unreachable"}


def test_run_without_scan_returns_one(monkeypatch, capsys):
    monkeypatch.setattr(vc, "_latest_scan", lambda: None)
    assert vc.run(5, False) == 1
    assert "no India scan workbook found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "s,m,verdict",
    [
        (OK(100.0), OK(100.0), "CONFIRMED"),
        (OK(101.99), OK(98.01), "CONFIRMED"),  # both within 2%
        (OK(102.0), BAD, "DEGRADED"),  # exactly 2.0% counts as agreeing
        (BAD, OK(100.5), "DEGRADED"),
        (OK(100.0), OK(110.0), "SPLIT"),
        (OK(110.0), OK(100.0), "SPLIT"),
        (OK(110.0), OK(120.0), "REJECTED"),
        (OK(110.0), BAD, "REJECTED"),
        (BAD, OK(90.0), "REJECTED"),
        (BAD, BAD, "NO_SOURCE"),
        (OK(102.01), BAD, "REJECTED"),  # just outside tolerance
    ],
)
def test_run_verdict_matrix(monkeypatch, tmp_path, capsys, s, m, verdict):
    _setup(
        monkeypatch,
        tmp_path,
        [{"Symbol": "AAA", "LTP": 100.0, "Median_Turnover": 5}],
        {"AAA": s},
        {"AAA": m},
    )
    rows = _run_json(capsys)
    assert [r["verdict"] for r in rows] == [verdict]


def test_run_record_fields_and_diff_pct(monkeypatch, tmp_path, capsys):
    _setup(
        monkeypatch,
        tmp_path,
        [{"Symbol": " aaa ", "LTP": 100.456, "Median_Turnover": 5}],
        {"AAA": {"ok": True, "price": 103.0, "close_date": "12 Jul"}},
        {"AAA": {"ok": False, "reason": "symbol not resolved"}},
    )
    r = _run_json(capsys)[0]
    assert r["symbol"] == "AAA" and r["ours"] == 100.46
    assert r["screener"] == 103.0 and r["moneycontrol"] is None
    assert r["screener_diff_pct"] == pytest.approx(abs(103.0 - 100.456) / 100.456 * 100)
    assert r["moneycontrol_diff_pct"] is None
    assert r["screener_err"] is None and r["mc_err"] == "symbol not resolved"
    assert r["close_date"] == "12 Jul"


def test_run_samples_top_turnover_skips_bad_rows_and_rate_limits(monkeypatch, tmp_path, capsys):
    rows = [
        {"Symbol": "LOW", "LTP": 10.0, "Median_Turnover": 1},
        {"Symbol": "NOLTP", "LTP": None, "Median_Turnover": 99},
        {"Symbol": "HIGH", "LTP": 10.0, "Median_Turnover": 50},
        {"Symbol": "MID", "LTP": 10.0, "Median_Turnover": 20},
        {"Symbol": "BADLTP", "LTP": "n/a", "Median_Turnover": 98},
    ]
    src = {s: OK(10.0) for s in ("LOW", "HIGH", "MID")}
    sleeps = _setup(monkeypatch, tmp_path, rows, src, src)
    out = _run_json(capsys, sample=2)
    assert [r["symbol"] for r in out] == ["HIGH", "MID"]
    assert sleeps == [vc.RATE, vc.RATE]  # one pause per checked name
    assert vc.RATE > 0  # the pause is real: this is a polite scraper


def test_run_exit_code_only_one_when_rejected(monkeypatch, tmp_path, capsys):
    rows = [{"Symbol": "A", "LTP": 100.0, "Median_Turnover": 2}]
    _setup(monkeypatch, tmp_path, rows, {"A": OK(150.0)}, {"A": BAD})
    assert vc.run(5, False) == 1
    capsys.readouterr()
    _setup(monkeypatch, tmp_path, rows, {"A": BAD}, {"A": BAD})
    assert vc.run(5, False) == 0  # NO_SOURCE must not suppress the brief
    capsys.readouterr()
    _setup(monkeypatch, tmp_path, rows, {"A": OK(100.0)}, {"A": OK(150.0)})
    assert vc.run(5, False) == 0  # SPLIT is flagged, not rejected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: `if as_json: print(...); return 0` returns before the REJECTED exit-code "
    "logic, so `--json` runs exit 0 even when a reachable source contradicts our price, "
    "defeating the documented non-zero exit that suppresses the brief",
)
def test_run_json_mode_still_signals_rejected_via_exit_code(monkeypatch, tmp_path, capsys):
    rows = [{"Symbol": "A", "LTP": 100.0, "Median_Turnover": 2}]
    _setup(monkeypatch, tmp_path, rows, {"A": OK(150.0)}, {"A": BAD})
    assert vc.run(5, True) == 1


def test_run_table_output(monkeypatch, tmp_path, capsys):
    rows = [
        {"Symbol": "AAA", "LTP": 100.0, "Median_Turnover": 3},
        {"Symbol": "BBB", "LTP": 50.0, "Median_Turnover": 2},
    ]
    _setup(
        monkeypatch,
        tmp_path,
        rows,
        {"AAA": OK(100.0), "BBB": {"ok": False, "reason": "no price on page"}},
        {"AAA": OK(101.0), "BBB": {"ok": False, "reason": "symbol not resolved"}},
    )
    assert vc.run(5, False) == 0
    out = capsys.readouterr().out
    assert "CONVERGENCE CHECK — scan.xlsx   (2 picks, tolerance 2.0%)" in out
    aaa = next(ln for ln in out.splitlines() if ln.strip().startswith("AAA"))
    assert "100.00" in aaa and "101.00" in aaa and aaa.rstrip().endswith("CONFIRMED")
    bbb = next(ln for ln in out.splitlines() if ln.strip().startswith("BBB"))
    assert "no price o" in bbb and "symbol not" in bbb and bbb.rstrip().endswith("NO_SOURCE")
    assert "{'CONFIRMED': 1, 'NO_SOURCE': 1}" in out


def test_main_passes_args(monkeypatch):
    seen = []
    monkeypatch.setattr(vc, "run", lambda n, j: seen.append((n, j)) or 0)
    monkeypatch.setattr(sys, "argv", ["validate_convergence.py"])
    assert vc.main() == 0
    monkeypatch.setattr(sys, "argv", ["validate_convergence.py", "--sample", "3", "--json"])
    vc.main()
    assert seen == [(6, False), (3, True)]
