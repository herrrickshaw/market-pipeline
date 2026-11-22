"""equity_ownership.py -- AMFI SIP-note parsing, FRED fetch and screener.in shareholding scrape (all faked)."""

import sys
import types

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
