"""equity_ownership.py -- AMFI SIP-note parsing, FRED fetch and screener.in shareholding scrape (all faked)."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import equity_ownership as EO


# ── _parse_sip ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,expected",
    [
        ("SIP contributions reached Rs 28,265 crore in the month.", 28265.0),
        ("SIP contribution amount reaching Rs. 26,459 crore", 26459.0),
        ("The industry SIP flows touched a new high of Rs 29,361 crore", 29361.0),
        ("sip contributions of rs 20,000 CRORE", 20000.0),
    ],
)
def test_parse_sip_phrasings(text, expected):
    assert EO._parse_sip(text) == expected


def test_parse_sip_ignores_the_equity_fund_inflow_sentence():
    flat = (
        "Equity funds saw positive inflows during the month totalling Rs 33,430 crore. "
        "Monthly SIP contributions stood steady at Rs 28,265 crore."
    )
    assert EO._parse_sip(flat) == 28265.0


@pytest.mark.parametrize(
    "text",
    [
        "SIP contributions rising from Rs 20,000 crore last year",
        "SIP contributions declined to Rs 20,000 crore",
        "SIP contributions saw an increase of Rs 20,000 crore",
        "SIP assets under management Rs 20,000 crore",
        "SIP AUM contributions Rs 20,000 crore",
        "SIP flows stood at Rs 20,000 crore",
    ],
)
def test_parse_sip_rejects_comparison_and_aum_sentences(text):
    assert EO._parse_sip(text) is None


def test_parse_sip_requires_a_flow_noun_and_rupees():
    assert EO._parse_sip("SIP accounts reached 9.11 crore") is None
    assert EO._parse_sip("SIP registrations Rs 28,000 crore") is None
    assert EO._parse_sip("") is None


@pytest.mark.parametrize(
    "value,ok", [("2,999", False), ("3,000", True), ("60,000", True), ("60,001", False)]
)
def test_parse_sip_plausibility_range_is_inclusive(value, ok):
    out = EO._parse_sip(f"SIP contributions of Rs {value} crore")
    assert (out is not None) is ok


def test_parse_sip_repeated_agreeing_figure_and_majority_vote():
    same = "SIP contributions of Rs 25,000 crore. Overall SIP flows at Rs 25,000 crore."
    assert EO._parse_sip(same) == 25000.0
    majority = (
        "SIP contributions Rs 25,000 crore. SIP flows Rs 25,000 crore. "
        "SIP contribution Rs 30,000 crore."
    )
    assert EO._parse_sip(majority) == 25000.0


def test_parse_sip_conflicting_figures_without_majority_is_none():
    assert EO._parse_sip("SIP contributions Rs 25,000 crore. SIP flows Rs 30,000 crore.") is None
    four = (
        "SIP flows Rs 25,000 crore. SIP flows Rs 25,000 crore. "
        "SIP flows Rs 30,000 crore. SIP flows Rs 30,000 crore."
    )
    assert EO._parse_sip(four) is None  # 2 of 4 is not > half


def test_parse_sip_a_rejecting_phrase_shortly_before_the_figure_poisons_it():
    # "AUM" sits ~35 characters before the SIP sentence: within the 60-char look-behind -> refuse
    near = "AUM surged. " + "x" * 20 + " SIP contributions reached Rs 28,265 crore."
    assert EO._parse_sip(near) is None
    far = "AUM surged. " + "x" * 100 + " SIP contributions reached Rs 28,265 crore."
    assert EO._parse_sip(far) == 28265.0


def test_parse_sip_keeps_the_good_match_when_one_is_rejected():
    filler = " The remaining sections of the note discuss other categories of schemes at length. "
    flat = (
        "SIP contributions rising from Rs 20,000 crore a year ago."
        + filler
        + "The month's SIP contributions reached Rs 28,265 crore."
    )
    assert EO._parse_sip(flat) == 28265.0


# ── _fred ─────────────────────────────────────────────────────────────────────
def test_fred_parses_observations_and_drops_missing(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self):
            seen["checked"] = True

        def json(self):
            return {
                "observations": [
                    {"date": "2020-01-02", "value": "1.5"},
                    {"date": "2020-01-03", "value": "."},
                    {"date": "2020-01-06", "value": "2.5"},
                ]
            }

    def fake_get(url, params, timeout):
        seen.update(url=url, params=params, timeout=timeout)
        return Resp()

    monkeypatch.setattr(EO.requests, "get", fake_get)
    s = EO._fred("DGS10", "KEY")
    assert list(s.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert list(s.values) == [1.5, 2.5] and s.index.name == "Date"
    assert seen["params"] == {
        "series_id": "DGS10",
        "api_key": "KEY",
        "file_type": "json",
        "observation_start": "1990-01-01",
    }
    assert seen["checked"] and seen["timeout"] == 60 and "stlouisfed" in seen["url"]


def test_key_delegates_to_yen_carry_test(monkeypatch):
    monkeypatch.setitem(sys.modules, "yen_carry_test", types.SimpleNamespace(_fred_key=lambda: "K"))
    assert EO._key() == "K"


# ── _screener_company ─────────────────────────────────────────────────────────
HTML = '<li>Market Cap <span class="x">₹</span> <span class="number">17,69,785</span> Cr.</li>'


def shareholding_table():
    return pd.DataFrame({"": ["Promoters", "FIIs", "DIIs"], "Mar 2024": [50.0, 20.0, 10.0]})


def fake_site(monkeypatch, responses, tables=None):
    calls = []

    class Resp:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

    def get(url, headers, timeout):
        calls.append(url)
        r = responses.get(url)
        if isinstance(r, Exception):
            raise r
        return Resp(*r) if r else Resp(404, "")

    monkeypatch.setattr(EO.requests, "get", get)
    monkeypatch.setattr(
        EO.pd,
        "read_html",
        lambda buf: (
            tables if tables is not None else [pd.DataFrame({"x": [1]}), shareholding_table()]
        ),
    )
    return calls


def test_screener_company_prefers_consolidated_and_parses_indian_grouping(monkeypatch):
    base = EO.SCREENER.format(sym="TCS")
    calls = fake_site(monkeypatch, {base + "consolidated/": (200, HTML)})
    table, mc = EO._screener_company("TCS")
    assert mc == 1769785.0 and list(table.iloc[:, 0]) == ["Promoters", "FIIs", "DIIs"]
    assert calls == [base + "consolidated/"]


def test_screener_company_falls_back_to_standalone_page(monkeypatch):
    base = EO.SCREENER.format(sym="X")
    calls = fake_site(monkeypatch, {base + "consolidated/": (404, ""), base: (200, "no mc here")})
    table, mc = EO._screener_company("X")
    assert mc is None and table is not None
    assert calls == [base + "consolidated/", base]


def test_screener_company_network_error_then_success(monkeypatch):
    base = EO.SCREENER.format(sym="X")
    fake_site(monkeypatch, {base + "consolidated/": OSError("reset"), base: (200, HTML)})
    table, mc = EO._screener_company("X")
    assert mc == 1769785.0 and table is not None


def test_screener_company_requires_promoter_and_fii_rows(monkeypatch):
    base = EO.SCREENER.format(sym="X")
    no_fii = pd.DataFrame({"": ["Promoters", "Public"], "v": [1, 2]})
    fake_site(
        monkeypatch, {base + "consolidated/": (200, HTML), base: (200, HTML)}, tables=[no_fii]
    )
    assert EO._screener_company("X") == (None, None)


def test_screener_company_unparseable_tables_are_skipped(monkeypatch):
    base = EO.SCREENER.format(sym="X")
    fake_site(monkeypatch, {base + "consolidated/": (200, HTML), base: (200, HTML)})
    monkeypatch.setattr(
        EO.pd, "read_html", lambda buf: (_ for _ in ()).throw(ValueError("no tables"))
    )
    assert EO._screener_company("X") == (None, None)


def test_screener_company_all_pages_missing(monkeypatch):
    fake_site(monkeypatch, {})
    assert EO._screener_company("X") == (None, None)


# ═══ FRED ownership table, NSE FII/DII append, AMFI SIP series ═══════════════════
import subprocess  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(EO, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(EO, "US_OWN", tmp_path / "cache" / "us.parquet")
    monkeypatch.setattr(EO, "IN_FLOW", tmp_path / "cache" / "flow.parquet")
    monkeypatch.setattr(EO, "SIP", tmp_path / "cache" / "sip.parquet")
    return tmp_path


def fred_series(level):
    idx = pd.DatetimeIndex(["2023-12-31", "2024-03-31"], name="Date")
    return pd.Series([level, level * 1.1], index=idx, name="v")


def test_run_us_builds_ownership_with_household_residual(cache, monkeypatch, capsys):
    monkeypatch.setattr(EO, "_key", lambda: "KEY")
    levels = {sid: float(i + 1) * 10 for i, sid in enumerate(EO.Z1.values())}
    levels[EO.Z1_TOTAL] = 1000.0
    monkeypatch.setattr(EO, "_fred", lambda sid, key: fred_series(levels[sid]))
    d = EO.run_us()
    inst = sum(levels[s] for s in EO.Z1.values())
    assert d["TOTAL"].iloc[0] == 1000.0 and d["Households (residual)"].iloc[0] == pytest.approx(
        1000.0 - inst
    )
    assert d["Households (residual)"].iloc[1] == pytest.approx(1100.0 - inst * 1.1)
    assert list(d.columns[: len(EO.Z1)]) == list(EO.Z1)
    saved = pd.read_parquet(EO.US_OWN)
    assert len(saved) == 2 and "Date" in saved.columns
    out = capsys.readouterr().out
    assert "2 quarters, 2023-12-31 .. 2024-03-31" in out and out.count(" ok") == len(EO.Z1)


def test_run_us_without_a_key_does_nothing(cache, monkeypatch, capsys):
    monkeypatch.setattr(EO, "_key", lambda: "")
    assert EO.run_us().empty and "no FRED key" in capsys.readouterr().out
    assert not EO.US_OWN.exists()


def test_run_us_tolerates_one_failing_series(cache, monkeypatch, capsys):
    monkeypatch.setattr(EO, "_key", lambda: "KEY")
    bad = EO.Z1["ETFs"]

    def fred(sid, key):
        if sid == bad:
            raise OSError("down")
        return fred_series(10.0 if sid != EO.Z1_TOTAL else 1000.0)

    monkeypatch.setattr(EO, "_fred", fred)
    d = EO.run_us()
    assert "ETFs" not in d.columns and "Households (residual)" in d.columns
    assert f"{bad} FAILED (OSError)" in capsys.readouterr().out


def test_run_us_drops_quarters_without_a_total(cache, monkeypatch):
    monkeypatch.setattr(EO, "_key", lambda: "KEY")
    short_total = pd.Series([1000.0], index=pd.DatetimeIndex(["2024-03-31"], name="Date"))
    monkeypatch.setattr(
        EO, "_fred", lambda sid, key: short_total if sid == EO.Z1_TOTAL else fred_series(10.0)
    )
    d = EO.run_us()
    assert list(d.index) == [pd.Timestamp("2024-03-31")]


def test_us_shares_orders_holders_by_share_of_total():
    d = pd.DataFrame({"A": [1.0, 20.0], "B": [1.0, 50.0], "TOTAL": [10.0, 100.0]})
    s = EO.us_shares(d)
    assert list(s.index) == ["B", "A"] and list(s.values) == [50.0, 20.0]


class NseSession:
    def __init__(self, rows=None, fail_on=None):
        self.rows, self.fail_on, self.headers, self.calls = rows, fail_on, {}, []

    def get(self, url, timeout):
        self.calls.append(url)
        if self.fail_on and self.fail_on in url:
            raise OSError("blocked")
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.rows)


def flow_rows(day="21-Jul-2026", net="100.5"):
    return [
        {"date": day, "category": "FII/FPI", "buyValue": "1,0", "sellValue": "50", "netValue": net},
        {"date": day, "category": "DII", "buyValue": "70", "sellValue": "20", "netValue": "50"},
    ]


def test_run_india_appends_without_duplicating(cache, monkeypatch, capsys):
    s = NseSession(flow_rows())
    monkeypatch.setattr(EO.requests, "Session", lambda: s)
    EO.run_india()
    d = pd.read_parquet(EO.IN_FLOW)
    assert len(d) == 2 and set(d["category"]) == {"FII/FPI", "DII"}
    assert d.loc[d.category == "FII/FPI", "netValue"].iloc[0] == 100.5
    assert pd.isna(d.loc[d.category == "FII/FPI", "buyValue"].iloc[0])  # "1,0" is not numeric
    assert s.headers["User-Agent"] == EO.UA and s.calls == [
        "https://www.nseindia.com",
        EO.NSE_FIIDII,
    ]
    assert "2 rows, 2026-07-21 .. 2026-07-21 (append-only, no backfill)" in capsys.readouterr().out
    s2 = NseSession(flow_rows(net="200") + flow_rows(day="22-Jul-2026"))
    monkeypatch.setattr(EO.requests, "Session", lambda: s2)
    EO.run_india()
    d = pd.read_parquet(EO.IN_FLOW)
    assert len(d) == 4  # (Date, category) is the key
    again = d[(d.Date == "2026-07-21") & (d.category == "FII/FPI")]
    assert again["netValue"].iloc[0] == 200.0  # newest value wins
    assert list(d["Date"]) == sorted(d["Date"])


def test_run_india_failure_and_bad_dates(cache, monkeypatch, capsys):
    monkeypatch.setattr(EO.requests, "Session", lambda: NseSession(fail_on="/api/"))
    EO.run_india()
    assert "NSE fetch failed (OSError)" in capsys.readouterr().out and not EO.IN_FLOW.exists()
    rows = flow_rows(day="not-a-date") + flow_rows(day="21-Jul-2026")[:1]
    monkeypatch.setattr(EO.requests, "Session", lambda: NseSession(rows))
    EO.run_india()
    assert len(pd.read_parquet(EO.IN_FLOW)) == 1  # unparseable date dropped


def pdf_response(text_marker="x"):
    return types.SimpleNamespace(status_code=200, content=b"%PDF-1.4 " + text_marker.encode())


def test_run_sip_parses_notes_and_writes_series(cache, monkeypatch, capsys):
    texts = {
        "January2024": "SIP contributions reached Rs 18,838 crore. SIP assets account for 20.5% of AUM.",
        "February2024": "SIP flows were Rs 19,187 crore.",
    }

    def get(url, headers, timeout):
        for key in texts:
            if url.endswith(f"AMFIMonthlyNote_{key}.pdf"):
                return pdf_response(key)
        return types.SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(EO.requests, "get", get)

    def run(cmd, capture_output, text, timeout):
        marker = open(cmd[1], "rb").read().decode().split("%PDF-1.4 ")[1]
        return types.SimpleNamespace(stdout=texts[marker])

    monkeypatch.setattr(subprocess, "run", run)
    EO.run_sip(start_year=2024)
    d = pd.read_parquet(EO.SIP)
    assert list(d["sip_crore"]) == [18838.0, 19187.0]
    assert list(d["month"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert d["sip_pct_of_aum"].iloc[0] == 20.5 and pd.isna(d["sip_pct_of_aum"].iloc[1])
    assert d["source"].iloc[0] == "AMFIMonthlyNote_January2024.pdf"
    out = capsys.readouterr().out
    assert (
        "Jan 2024: SIP Rs 18,838 crore, 20.5% of AUM" in out
        and "Feb 2024: SIP Rs 19,187 crore" in out
    )
    assert "-> sip.parquet: 2 months" in out


def test_run_sip_skips_ambiguous_notes_and_non_pdf_responses(cache, monkeypatch, capsys):
    def get(url, headers, timeout):
        if "January2025" in url:
            return pdf_response("a")
        if "February2025" in url:
            return types.SimpleNamespace(status_code=200, content=b"<html>not a pdf")
        if "March2025" in url:
            raise OSError("down")
        return types.SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(EO.requests, "get", get)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="no figure here")
    )
    EO.run_sip(start_year=2025)
    out = capsys.readouterr().out
    assert "Jan 2025: no unambiguous SIP figure — SKIPPED" in out and "no SIP figures parsed" in out
    assert not EO.SIP.exists()


def test_run_sip_survives_a_missing_pdftotext(cache, monkeypatch, capsys):
    monkeypatch.setattr(
        EO.requests,
        "get",
        lambda url, headers, timeout: (
            pdf_response("a")
            if "January2026" in url
            else types.SimpleNamespace(status_code=404, content=b"")
        ),
    )

    def boom(*a, **k):
        raise FileNotFoundError("pdftotext")

    monkeypatch.setattr(subprocess, "run", boom)
    EO.run_sip(start_year=2026)
    assert "no SIP figures parsed" in capsys.readouterr().out


# ═══ India ownership aggregation, implied flow proxy, report table, CLI ═══════════
def screener_table(pro, fii, dii, gov, pub, quarters=("Mar 2024", "Jun 2024")):
    data = {
        "": ["Promoters +", "FIIs +", "DIIs +", "Government +", "Public +", "No. of Shareholders"]
    }
    for q, k in zip(quarters, range(len(quarters)), strict=True):
        data[q] = [f"{pro + k}%", f"{fii}%", f"{dii}%", f"{gov}%", f"{pub}%", "1,234"]
    return pd.DataFrame(data)


@pytest.fixture
def india_env(cache, monkeypatch):
    monkeypatch.setattr(EO, "IN_OWN", cache / "cache" / "in_own.parquet")
    monkeypatch.setattr(EO, "IN_OWN_LONG", cache / "cache" / "in_own_long.parquet")
    monkeypatch.setattr(EO, "IN_PROXY", cache / "cache" / "in_proxy.parquet")
    monkeypatch.setattr(EO, "REPORTS", cache / "reports")
    monkeypatch.setattr(EO, "REPORT", cache / "reports" / "table.md")
    monkeypatch.setattr(sys.modules["time"], "sleep", lambda s: None)
    return cache


def install_kit(monkeypatch, symbols, seen=None):
    def load(market, min_turnover_usd):
        if seen is not None:
            seen.update(market=market, min_turnover_usd=min_turnover_usd)
        return dict.fromkeys(symbols)

    monkeypatch.setitem(sys.modules, "screener_kit", types.SimpleNamespace(load=load))


def test_india_ownership_cap_weighted_aggregate(india_env, monkeypatch, capsys):
    syms = [f"S{i:02d}" for i in range(34)]
    seen = {}
    install_kit(monkeypatch, reversed(syms), seen)
    caps = {s: 1000.0 * (i + 1) for i, s in enumerate(syms)}

    def company(sym):
        i = int(sym[1:])
        if i >= 32:  # two names fail to scrape
            return (None, None) if i == 32 else (screener_table(50, 10, 10, 1, 29), None)
        quarters = ("Mar 2024", "Jun 2024") if i < 30 else ("Mar 2024",)  # only 30 report June
        return screener_table(40 + (i % 2) * 20, 10, 20, 1, 29 - (i % 2) * 20, quarters), caps[sym]

    monkeypatch.setattr(EO, "_screener_company", company)
    EO.run_india_ownership(limit=100, sleep=0)
    d = pd.read_parquet(EO.IN_OWN)
    assert seen == {"market": "IN", "min_turnover_usd": 20_000_000}
    assert list(d["quarter"]) == ["Mar 2024", "Jun 2024"] and list(d["n_companies"]) == [32, 30]
    w = np.array([caps[s] for s in syms[:32]])
    pro = np.array([40 + (i % 2) * 20 for i in range(32)], dtype=float)
    assert d.loc[0, "Promoters"] == pytest.approx((pro * w).sum() / w.sum())
    assert d.loc[1, "Promoters"] == pytest.approx(((pro + 1)[:30] * w[:30]).sum() / w[:30].sum())
    assert d.loc[0, "FIIs"] == pytest.approx(10.0) and d.loc[0, "DIIs"] == pytest.approx(20.0)
    long = pd.read_parquet(EO.IN_OWN_LONG)
    assert set(long.columns) == {"symbol", "quarter", "holder", "pct", "mcap_now_cr"}
    assert "Promoters" in set(long["holder"]) and "No. of Shareholders" not in set(long["holder"])
    out = capsys.readouterr().out
    assert "universe: 34 liquid India names" in out and "2 failed" in out


def test_india_ownership_universe_limit_and_empty_run(india_env, monkeypatch, capsys):
    install_kit(monkeypatch, [f"S{i:02d}" for i in range(10)])
    monkeypatch.setattr(EO, "_screener_company", lambda s: (None, None))
    EO.run_india_ownership(limit=4, sleep=0)
    out = capsys.readouterr().out
    assert "universe: 4 liquid India names" in out and "nothing collected" in out
    assert not EO.IN_OWN.exists()


def test_india_ownership_thin_quarters_are_dropped(india_env, monkeypatch):
    syms = [f"S{i:02d}" for i in range(31)]
    install_kit(monkeypatch, syms)
    monkeypatch.setattr(
        EO,
        "_screener_company",
        lambda s: (
            screener_table(
                50, 10, 10, 1, 29, ("Mar 2024",) if s != "S00" else ("Mar 2024", "Jul 2026")
            ),
            1000.0,
        ),
    )
    EO.run_india_ownership(limit=100, sleep=0)
    assert list(pd.read_parquet(EO.IN_OWN)["quarter"]) == ["Mar 2024"]  # Jul 2026 has n=1


def flow_fixture(india_env, monkeypatch, n=32, quarters=("Mar 2024", "Jun 2024", "Sep 2024")):
    syms = [f"S{i:02d}" for i in range(n)]
    rows = []
    for i, s in enumerate(syms):
        for k, q in enumerate(quarters):
            rows.append(
                {
                    "symbol": s,
                    "quarter": q,
                    "holder": "FIIs",
                    "pct": 10.0 + k,
                    "mcap_now_cr": 1000.0 * (i + 1),
                }
            )
            rows.append(
                {
                    "symbol": s,
                    "quarter": q,
                    "holder": "DIIs",
                    "pct": 20.0 - k,
                    "mcap_now_cr": 1000.0 * (i + 1),
                }
            )
    EO.CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(EO.IN_OWN_LONG, index=False)
    wh = india_env / "wh"
    wh.mkdir()
    ends = pd.to_datetime(["2024-03-28", "2024-06-28", "2024-09-30"])
    px = []
    for s in syms:
        for k, e in enumerate(ends):
            px.append({"Date": e, "Symbol": s.lower(), "Close": 100.0 * 1.1**k})
    pd.DataFrame(px).to_parquet(wh / "year=2024.parquet", index=False)
    monkeypatch.setattr(EO, "WAREHOUSE_IN", wh)
    return syms


def test_flow_proxy_isolates_trading_from_price_moves(india_env, monkeypatch, capsys):
    syms = flow_fixture(india_env, monkeypatch)
    EO.run_flow_proxy()
    f = pd.read_parquet(EO.IN_PROXY)
    assert set(f["quarter"]) == {"Jun 2024", "Sep 2024"} and set(f["holder"]) == {"FIIs", "DIIs"}
    caps_now = np.array([1000.0 * (i + 1) for i in range(32)])
    mcap_jun = caps_now * (1.1**1) / (1.1**2)
    fii_jun = f[(f.quarter == "Jun 2024") & (f.holder == "FIIs")].iloc[0]
    assert fii_jun["implied_flow_cr"] == pytest.approx(
        (0.01 * mcap_jun).sum()
    )  # +1pp of that quarter's cap
    dii_jun = f[(f.quarter == "Jun 2024") & (f.holder == "DIIs")].iloc[0]
    assert dii_jun["implied_flow_cr"] == pytest.approx(-(0.01 * mcap_jun).sum())
    assert fii_jun["n"] == 32 and fii_jun["mkt_return_pct"] == pytest.approx(10.0)
    assert fii_jun["value_cr"] == pytest.approx((0.11 * mcap_jun).sum())
    out = capsys.readouterr().out
    assert "2 quarter pairs" in out and "cumulative:" in out and "FIIs +" in out
    assert (
        "FIIs       positive in  2/2  quarters" in out
        and "DIIs       positive in  0/2  quarters" in out
    )
    assert "corr(flow, mkt return)" in out
    assert syms


def test_flow_proxy_guards(india_env, monkeypatch, capsys):
    EO.run_flow_proxy()
    assert "run --inown first" in capsys.readouterr().out
    flow_fixture(india_env, monkeypatch)
    monkeypatch.setattr(EO, "WAREHOUSE_IN", india_env / "empty")
    (india_env / "empty").mkdir()
    EO.run_flow_proxy()
    assert "India price panel not found" in capsys.readouterr().out


def test_flow_proxy_needs_thirty_common_companies(india_env, monkeypatch, capsys):
    flow_fixture(india_env, monkeypatch, n=29)
    EO.run_flow_proxy()
    assert "no comparable quarter pairs" in capsys.readouterr().out and not EO.IN_PROXY.exists()


def seed_report_inputs(india_env):
    EO.CACHE.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2010-03-31", periods=60, freq="QE", name="Date")
    d = pd.DataFrame(
        {
            "Mutual funds": np.linspace(10e6, 20e6, 60),
            "Households (residual)": np.linspace(50e6, 30e6, 60),
            "TOTAL": 100.0e6,
        },
        index=idx,
    )
    d.reset_index().to_parquet(EO.US_OWN, index=False)
    pd.DataFrame(
        {
            "quarter": ["Mar 2024", "Jun 2024", "Sep 2024", "Dec 2024", "Mar 2025"],
            "n_companies": [40] * 5,
            "Promoters": 50.0,
            "FIIs": [20, 19, 18, 17, 16.0],
            "DIIs": [14, 15, 16, 17, 18.0],
            "Government": 1.0,
            "Public": 15.0,
        }
    ).to_parquet(EO.IN_OWN, index=False)
    pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-20", "2026-07-21"] * 2),
            "category": ["DII"] * 2 + ["FII/FPI"] * 2,
            "buyValue": 1.0,
            "sellValue": 1.0,
            "netValue": [100.0, 120.0, -80.0, -90.0],
        }
    ).to_parquet(EO.IN_FLOW, index=False)
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-07-01", "2025-08-01"]),
            "sip_crore": [28000.0, 28265.0],
            "sip_pct_of_aum": [20.0, None],
            "source": "x",
        }
    ).to_parquet(EO.SIP, index=False)


def test_run_table_with_all_inputs(india_env, capsys):
    seed_report_inputs(india_env)
    EO.run_table()
    md = EO.REPORT.read_text()
    assert "Total corporate equity outstanding **$100.0 trillion**" in md
    assert "| Mutual funds | **20.0%** |" in md and "| Households (residual) | **30.0%** |" in md
    assert "Ten-year drift" in md and "| Mutual funds |" in md
    assert (
        "India ownership — COMPUTED" in md
        and "**5 quarters**" not in md
        and "across **40 liquid" in md
    )
    assert "FIIs 20.0% → 16.0% (-4.0pp), DIIs 14.0% → 18.0% (+4.0pp)" in md
    assert "India FLOW — COMPUTED" in md and "| 2026-07-21 | +120 | -90 |" in md
    assert "India SIP contribution — COMPUTED" in md and "| Jul 2025 | 28,000 | 20.0% |" in md
    assert "| Aug 2025 | 28,265 | — |" in md and "+0.9% over the window" in md
    assert (
        "## 🟡 Japan — CITED, not computed" in md and "## 🟡 India" not in md
    )  # superseded by computed
    assert "Stock vs flow" in md and f"-> {EO.REPORT}" in capsys.readouterr().out


def test_run_table_without_inputs_falls_back_to_citations(india_env):
    EO.run_table()
    md = EO.REPORT.read_text()
    assert "Run `--us` first." in md and "## 🟡 India — CITED, not computed" in md
    assert "COMPUTED" not in md.replace("COMPUTED here", "")


def test_run_table_short_us_history_skips_the_drift_section(india_env):
    EO.CACHE.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2023-03-31", periods=4, freq="QE", name="Date")
    pd.DataFrame({"A": [10.0] * 4, "TOTAL": [100.0] * 4}, index=idx).reset_index().to_parquet(
        EO.US_OWN, index=False
    )
    EO.run_table()
    assert "Ten-year drift" not in EO.REPORT.read_text()


def test_run_sources_and_main_dispatch(india_env, monkeypatch, capsys):
    EO.run_sources()
    out = capsys.readouterr().out
    assert out.startswith("Data access, tested 2026-08-01") and "OPEN: AMFI SIP series" in out
    calls = []
    for name in (
        "run_us",
        "run_india",
        "run_sip",
        "run_india_ownership",
        "run_flow_proxy",
        "run_table",
        "run_sources",
    ):
        monkeypatch.setattr(EO, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(sys, "argv", ["x"])
    assert EO.main() == 1 and calls == []
    monkeypatch.setattr(sys, "argv", ["x", "--us", "--table"])
    assert EO.main() == 0 and calls == ["run_us", "run_table"]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--all"])
    EO.main()
    assert calls == [
        "run_us",
        "run_india",
        "run_sip",
        "run_india_ownership",
        "run_flow_proxy",
        "run_table",
    ]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--sources", "--us"])
    EO.main()
    assert calls == ["run_sources"]
