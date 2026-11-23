"""Offline tests for stock_enricher (sector PE zones, enrichment, HTML tables)."""

import types

import numpy as np
import pandas as pd
import pytest

import stock_enricher as se


# ---------------------------------------------------------------- _classify_pe
@pytest.mark.parametrize("pe", [None, float("nan"), 0, -3.0])
def test_classify_pe_not_applicable(pe):
    r = se._classify_pe(pe, "bank")
    assert r == {
        "zone": "N/A",
        "emoji": "⚪",
        "pe_str": "N/A",
        "buy": None,
        "fair": None,
        "caution": None,
    }


@pytest.mark.parametrize(
    "pe,zone",
    [
        (12, "BUY ZONE"),
        (12.01, "FAIR VALUE"),
        (18, "FAIR VALUE"),
        (18.01, "CAUTION"),
        (22, "CAUTION"),
        (22.01, "SELL ZONE"),
        (1, "BUY ZONE"),
        (500, "SELL ZONE"),
    ],
)
def test_classify_pe_boundaries_are_inclusive_upper(pe, zone):
    r = se._classify_pe(pe, "Private Bank", "IN")  # bank: (12, 18, 22)
    assert r["zone"] == zone and (r["buy"], r["fair"], r["caution"]) == (12, 18, 22)
    assert r["pe_str"] == f"{pe:.1f}x"


def test_classify_pe_emojis():
    emo = [se._classify_pe(p, "bank")["emoji"] for p in (10, 15, 20, 30)]
    assert emo == ["🟢", "🟡", "🟠", "🔴"]


def test_classify_pe_default_and_case_insensitive_substring_matching():
    assert se._classify_pe(20, "")["buy"] == 15  # default thresholds (15, 25, 35)
    assert se._classify_pe(20, None)["fair"] == 25
    assert se._classify_pe(20, "Unknown Sector")["caution"] == 35
    assert (
        se._classify_pe(20, "SOFTWARE - Infrastructure")["buy"] == 18
    )  # 'software' (and not 'infrastructure')
    assert se._classify_pe(20, "Oil & Gas Refining")["buy"] == 8


def test_classify_pe_market_switch_uses_us_table():
    assert (
        se._classify_pe(20, "technology", "IN")["buy"] == 15
    )  # not in the Indian table -> default
    assert se._classify_pe(20, "technology", "US")["buy"] == 22
    assert se._classify_pe(20, "mystery", "US")["fair"] == 22  # US default


def test_classify_pe_first_matching_keyword_wins_in_table_order():
    # 'bank' precedes 'finance' in the Indian table
    assert se._classify_pe(20, "Bank and Finance")["buy"] == 12
    # 'real estate' (20,35,50) vs 'estate'-free keyword order check: 'realty' also exists
    assert se._classify_pe(20, "Realty Developers")["buy"] == 20


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="yfinance's sector label is 'Financial Services', which does not contain the keyword "
    "'finance' (finan-cial vs finan-ce), so Indian financials fall through to the generic "
    "(15, 25, 35) PE band instead of the finance band (12, 20, 25)",
)
def test_classify_pe_matches_yfinance_financial_services_label():
    assert se._classify_pe(20, "Financial Services", "IN")["buy"] == 12


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: dict order makes the 'technology' key match 'biotechnology' first, so the "
    "dedicated biotechnology band (25, 45, 80) can never be selected",
)
def test_classify_pe_biotech_has_its_own_band():
    assert se._classify_pe(20, "Biotechnology", "US")["caution"] == 80


def test_threshold_tables_are_monotonic_and_have_defaults():
    for table in (se._IN_SECTOR_PE, se._US_SECTOR_PE):
        assert "default" in table
        assert all(b <= f <= c for b, f, c in table.values())


# ---------------------------------------------------------------- _fetch_one
class _Tk:
    def __init__(self, info):
        self.info = info


def _yf(monkeypatch, infos):
    """infos: list of info dicts/exceptions returned by successive Ticker(...).info accesses."""
    seq = list(infos)
    calls = []

    class _T:
        def __init__(self, sym):
            calls.append(sym)

        @property
        def info(self):
            v = seq.pop(0) if len(seq) > 1 else seq[0]
            if isinstance(v, Exception):
                raise v
            return v

    monkeypatch.setattr(se, "yf", types.SimpleNamespace(Ticker=_T), raising=False)
    monkeypatch.setattr(se, "_YF_OK", True)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


_INFO = {
    "shortName": "  Tata Consultancy Services  ",
    "sector": "Information Technology",
    "exchange": "NSI",
    "trailingPE": 27.349,
    "forwardPE": 24.0,
    "a": 1,
    "b": 2,
    "c": 3,
}


def test_fetch_one_without_yfinance_returns_defaults(monkeypatch):
    monkeypatch.setattr(se, "_YF_OK", False)
    r = se._fetch_one("TCS", ".NS")
    assert r["company_name"] == "TCS" and r["exchange"] == "NSE" and r["pe_zone"] == "⚪ N/A"
    assert se._fetch_one("X", ".BO")["exchange"] == "BSE"
    assert se._fetch_one("X", "", "US")["exchange"] == "NASDAQ"
    assert se._fetch_one("X", "", "IN")["exchange"] == "NYSE"


def test_fetch_one_full_enrichment(monkeypatch):
    calls = _yf(monkeypatch, [_INFO])
    r = se._fetch_one("TCS", ".NS")
    assert calls == ["TCS.NS"]
    assert (
        r["company_name"] == "Tata Consultancy Services" and r["sector"] == "Information Technology"
    )
    assert r["trailing_pe"] == 27.3 and r["forward_pe"] == 24.0
    assert r["pe_zone"] == "🟡 FAIR VALUE" and r["pe_label"] == "FAIR VALUE"
    assert r["pe_thresholds"] == "Buy<18 Fair<28 Caution<38"


def test_fetch_one_long_names_truncated_and_sector_fallback(monkeypatch):
    info = dict(_INFO, shortName=None, longName="X" * 60, sector=None, industry="Software")
    _yf(monkeypatch, [info])
    r = se._fetch_one("T", ".NS")
    assert r["company_name"] == "X" * 38 + "…" and r["sector"] == "Software"
    info2 = dict(_INFO, shortName="Y" * 40)
    _yf(monkeypatch, [info2])
    assert se._fetch_one("T", ".NS")["company_name"] == "Y" * 40  # exactly 40: untouched


def test_fetch_one_forward_pe_used_when_trailing_missing(monkeypatch):
    _yf(monkeypatch, [dict(_INFO, trailingPE=None, forwardPE=10.0)])
    r = se._fetch_one("T", ".NS")
    assert r["trailing_pe"] is None and r["forward_pe"] == 10.0 and r["pe_label"] == "BUY ZONE"
    _yf(monkeypatch, [dict(_INFO, trailingPE=None, forwardPE=None)])
    assert se._fetch_one("T", ".NS")["pe_label"] == "N/A"


@pytest.mark.parametrize(
    "suffix,exch,expected",
    [
        (".NS", "NMS", "NSE"),
        (".BO", "NMS", "BSE"),
        ("", "NMS", "NASDAQ"),
        ("", "NGM", "NASDAQ"),
        ("", "NCM", "NASDAQ"),
        ("", "NYS", "NYSE"),
        ("", "OTHER", "NASDAQ"),
    ],
)
def test_fetch_one_exchange_mapping(monkeypatch, suffix, exch, expected):
    _yf(monkeypatch, [dict(_INFO, exchange=exch)])
    assert se._fetch_one("T", suffix, "US")["exchange"] == expected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: Yahoo reports NYSE as 'NYQ', which does not contain 'NYS', so NYSE stocks "
    "keep the NASDAQ default for the US market",
)
def test_fetch_one_recognises_yahoo_nyse_code(monkeypatch):
    _yf(monkeypatch, [dict(_INFO, exchange="NYQ")])
    assert se._fetch_one("T", "", "US")["exchange"] == "NYSE"


def test_fetch_one_sparse_info_retries_then_gives_up(monkeypatch):
    sleeps = []
    calls = _yf(monkeypatch, [{"a": 1}])
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    r = se._fetch_one("T", ".NS", retries=3)
    assert len(calls) == 3 and sleeps == [3, 6, 9] and r["company_name"] == "T"


def test_fetch_one_recovers_after_sparse_attempt(monkeypatch):
    calls = _yf(monkeypatch, [{}, _INFO])
    assert (
        se._fetch_one("T", ".NS")["company_name"] == "Tata Consultancy Services" and len(calls) == 2
    )


def test_fetch_one_rate_limit_retries_other_errors_abort(monkeypatch):
    sleeps = []
    calls = _yf(
        monkeypatch, [RuntimeError("429 Too Many Requests"), RuntimeError("Rate limited"), _INFO]
    )
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    assert (
        se._fetch_one("T", ".NS")["pe_label"] == "FAIR VALUE"
        and sleeps == [5, 10]
        and len(calls) == 3
    )
    calls = _yf(monkeypatch, [ValueError("boom")])
    assert se._fetch_one("T", ".NS")["company_name"] == "T" and len(calls) == 1


def test_fetch_one_parse_errors_keep_partial_result(monkeypatch):
    _yf(monkeypatch, [dict(_INFO, trailingPE="not a number")])
    r = se._fetch_one("T", ".NS")
    assert (
        r["company_name"] == "Tata Consultancy Services" and r["trailing_pe"] is None
    )  # failed at float()


# ---------------------------------------------------------------- enrich_stocks
def test_enrich_stocks_maps_columns_and_defaults(monkeypatch, capsys):
    def fake_fetch(sym, sfx, market):
        if sym == "BAD":
            raise RuntimeError("x")
        return {
            "symbol": sym,
            "company_name": f"{sym} Corp",
            "exchange": "BSE" if sfx == ".BO" else "NSE",
            "sector": "IT",
            "trailing_pe": 20.0,
            "forward_pe": 18.0,
            "pe_zone": "🟡 FAIR VALUE",
            "pe_label": "FAIR VALUE",
        }

    monkeypatch.setattr(se, "_fetch_one", fake_fetch)
    df = pd.DataFrame(
        {"Symbol": ["AAA", "BBB", "BAD"], "Suffix": [".NS", ".BO", ".NS"], "LTP": [1, 2, 3]}
    )
    out = se.enrich_stocks(df, workers=2)
    assert out.columns.tolist()[-7:] == [
        "Company_Name",
        "Exchange",
        "Sector",
        "Trailing_PE",
        "Forward_PE",
        "PE_Zone",
        "PE_Label",
    ]
    assert out["Company_Name"].tolist() == [
        "AAA Corp",
        "BBB Corp",
        "BAD",
    ]  # failed fetch -> symbol fallback
    assert out["Exchange"].tolist() == ["NSE", "BSE", "NSE"] and out["PE_Zone"].iloc[2] == "⚪ N/A"
    assert out["PE_Label"].iloc[2] == "N/A" and pd.isna(out["Trailing_PE"].iloc[2])
    assert "Suffix" in df.columns and "Company_Name" not in df.columns  # input untouched
    assert "3/3 enriched" in capsys.readouterr().out


def test_enrich_stocks_default_suffix_by_market(monkeypatch):
    seen = []
    monkeypatch.setattr(
        se, "_fetch_one", lambda s, sfx, m: seen.append((s, sfx, m)) or {"symbol": s}
    )
    se.enrich_stocks(pd.DataFrame({"Symbol": ["A"]}), market="IN")
    se.enrich_stocks(pd.DataFrame({"Symbol": ["B"]}), market="US")
    se.enrich_stocks(
        pd.DataFrame({"Sym": ["C"], "Sfx": [".BO"]}), symbol_col="Sym", suffix_col="Sfx"
    )
    assert sorted(seen) == [("A", ".NS", "IN"), ("B", "", "US"), ("C", ".BO", "IN")]


# ---------------------------------------------------------------- display / groups / html
def test_format_stock_display():
    assert (
        se.format_stock_display(
            {"Company_Name": "Tata Motors", "Symbol": "TATAMOTORS", "Exchange": "BSE"}
        )
        == "TATA MOTORS [TATAMOTORS · BSE]"
    )
    assert se.format_stock_display({"Symbol": "X"}) == "X [X · NSE]"


def test_build_exchange_groups():
    df = pd.DataFrame({"Symbol": list("ABCDE"), "Exchange": ["NSE", "BSE", "NASDAQ", "TSE", "NSE"]})
    g = se.build_exchange_groups(df)
    assert list(g) == ["NSE", "BSE", "NASDAQ", "OTHER"]  # NYSE absent -> omitted; order fixed
    assert g["NSE"]["Symbol"].tolist() == ["A", "E"] and g["NSE"].index.tolist() == [0, 1]
    assert g["OTHER"]["Symbol"].tolist() == ["D"]
    nodata = pd.DataFrame({"Symbol": ["A"]})
    assert se.build_exchange_groups(nodata)["ALL"] is nodata


def test_generate_stock_table_html_rows_formatting_and_caps():
    df = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB", "CCC"],
            "Company_Name": ["Alpha", "Beta", "Gamma"],
            "Exchange": ["NSE", "BSE", "NSE"],
            "LTP": [1234.5, 20.0, "—"],
            "Change%": [1.5, -2.25, None],
            "Upside_to_Top%": [10.0, "—", None],
            "Trailing_PE": [18.04, None, np.nan],
            "PE_Zone": ["🟢 BUY ZONE", "🟠 CAUTION", "⚪ N/A"],
            "PE_Label": ["BUY ZONE", "CAUTION", "N/A"],
            "Screens_Passed": ["3/6", "2/6", "—"],
        }
    )
    html = se.generate_stock_table_html(df, "IN", max_rows=2)
    assert html.count("<tr>") == 3 and "CCC" not in html  # header + 2 rows
    assert "<strong>ALPHA</strong> [AAA · NSE]" in html and "₹1,234.50" in html
    assert (
        '<span class="pass">▲ +1.50%</span>' in html
        and '<span class="fail">▼ -2.25%</span>' in html
    )
    assert "+10.0%" in html and "18.0x" in html and "N/A" in html
    assert (
        '<span class="pass">🟢 BUY ZONE</span>' in html
        and '<span class="warn">🟠 CAUTION</span>' in html
    )
    assert "3/6" in html


def test_generate_stock_table_html_us_currency_and_edge_values():
    df = pd.DataFrame(
        {
            "Symbol": ["X"],
            "LTP": [10.0],
            "Change%": [0.0],
            "Upside%": [None],
            "PE_Label": ["SELL ZONE"],
            "PE_Zone": ["🔴 SELL ZONE"],
            "Darvas_Signal": ["BUY"],
        }
    )
    html = se.generate_stock_table_html(df, "US")
    assert "$10.00" in html and "▲ +0.00%" in html  # zero change counts as up
    assert '<span class="fail">🔴 SELL ZONE</span>' in html and "<td>BUY</td>" in html
    assert "[X · NSE]" in html  # defaults when Exchange / Company_Name absent


def test_generate_stock_table_html_empty():
    assert "No qualifying stocks" in se.generate_stock_table_html(pd.DataFrame())
