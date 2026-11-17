"""nse_xbrl_results: the parser must bind every number to the filing's OWN
fiscal quarter (from the index record), prefer plain over dimensional contexts,
and fall through field aliases in order. All XBRL here is synthetic; no network.
"""
import datetime as real_dt
import types

import pandas as pd
import pytest

import nse_xbrl_results as nx

Q_START, Q_END = "2024-07-01", "2024-09-30"


def _ctx(cid, start=None, end=None, instant=None, member=False):
    period = (f"<xbrli:instant>{instant}</xbrli:instant>" if instant else
              f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>")
    seg = ('<xbrli:segment><xbrldi:explicitMember dimension="in-bse-fin:X">'
           'in-bse-fin:ConsolidatedMember</xbrldi:explicitMember></xbrli:segment>'
           if member else "")
    return (f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="s">'
            f'500000</xbrli:identifier>{seg}</xbrli:entity><xbrli:period>{period}'
            f'</xbrli:period></xbrli:context>')


def _fact(el, cid, val, attrs_first=False):
    a = f'contextRef="{cid}" unitRef="INR" decimals="-5"'
    if attrs_first:
        a = f'unitRef="INR" decimals="-5" contextRef="{cid}"'
    return f"<in-bse-fin:{el} {a}>{val}</in-bse-fin:{el}>"


FILING = "<xbrli:xbrl>" + "".join([
    _ctx("OneD", Q_START, Q_END),
    _ctx("YTD", "2024-04-01", Q_END),
    _ctx("PrevQ", "2023-07-01", "2023-09-30"),
    _ctx("OneD_Seg", Q_START, Q_END, member=True),
    _ctx("Inst", instant=Q_END),
    # revenue: YTD and dimensional appear BEFORE the plain quarter value
    _fact("RevenueFromOperations", "YTD", 2000),
    _fact("RevenueFromOperations", "OneD_Seg", 900),
    _fact("RevenueFromOperations", "OneD", 1000),
    _fact("RevenueFromOperations", "PrevQ", 800),
    _fact("IncomeFromOperations", "OneD", 999),          # later alias: ignored
    # pat only via the continuing-operations alias, negative
    _fact("ProfitLossForPeriodFromContinuingOperations", "OneD", "-50.5"),
    _fact("FinanceCosts", "OneD", 12, attrs_first=True),
    _fact("OtherIncome", "OneD_Seg", 7),                 # only dimensional exists
    _fact("Income", "YTD", 2100),                        # only YTD exists
    _fact("TaxExpense", "Inst", 5),                      # instant context: not a period
]) + "</xbrli:xbrl>"


@pytest.fixture
def filing(tmp_path):
    p = tmp_path / "INDAS_1.xml"
    p.write_text(FILING)
    return p


class TestParseOne:
    def test_value_is_bound_to_the_filing_quarter_and_plain_beats_dimensional(self, filing):
        d = nx._parse_one(filing, Q_START, Q_END)
        assert d["revenue"] == 1000.0

    def test_alias_fallback_and_negative_values(self, filing):
        assert nx._parse_one(filing, Q_START, Q_END)["pat"] == -50.5

    def test_attribute_order_does_not_matter(self, filing):
        assert nx._parse_one(filing, Q_START, Q_END)["finance_costs"] == 12.0

    def test_dimensional_value_used_when_it_is_the_only_one_for_the_quarter(self, filing):
        assert nx._parse_one(filing, Q_START, Q_END)["other_income"] == 7.0

    def test_other_period_values_are_never_borrowed(self, filing):
        d = nx._parse_one(filing, Q_START, Q_END)
        assert d["total_income"] is None          # only a YTD value exists
        assert d["tax"] is None                   # only an instant context
        assert d["eps_basic"] is None

    def test_period_columns_come_from_the_bound_context(self, filing):
        d = nx._parse_one(filing, Q_START, Q_END)
        assert (d["period_start"], d["period_end"]) == (Q_START, Q_END)

    def test_prior_year_quarter_binds_to_its_own_values(self, filing):
        d = nx._parse_one(filing, "2023-07-01", "2023-09-30")
        assert d["revenue"] == 800.0
        assert d["pat"] is None

    def test_undefined_conventional_oned_context_is_synthesised(self, tmp_path):
        # NSE "WEB" files reference OneD without defining it.
        x = ("<xbrli:xbrl>" + _ctx("FourD", "2024-04-01", Q_END)
             + _fact("RevenueFromOperations", "FourD", 2000)
             + _fact("RevenueFromOperations", "OneD", 1000) + "</xbrli:xbrl>")
        p = tmp_path / "web.xml"
        p.write_text(x)
        assert nx._parse_one(p, Q_START, Q_END)["revenue"] == 1000.0

    def test_file_without_contexts_returns_empty(self, tmp_path):
        p = tmp_path / "empty.xml"
        p.write_text("<xbrli:xbrl></xbrli:xbrl>")
        assert nx._parse_one(p) == {}


def test_parse_all_joins_index_record_and_keeps_filing_date(tmp_path, monkeypatch, filing):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "INDAS_1.xml").write_text(FILING)
    (xml_dir / "INDAS_2.xml").write_text("<xbrli:xbrl>" + _ctx("OneD", Q_START, Q_END)
                                         + "</xbrli:xbrl>")       # no P&L: unparseable
    (xml_dir / "orphan.xml").write_text(FILING)                   # not in index: skipped
    idx = pd.DataFrame([
        {"symbol": "abc", "xbrl": "https://x/corporate/xbrl/INDAS_1.xml",
         "fromDate": "01-Jul-2024", "toDate": "30-Sep-2024",
         "filingDate": "08-Nov-2024 18:05", "audited": "Un-Audited",
         "consolidated": "Standalone", "relatingTo": "Second Quarter", "seqNumber": "1"},
        {"symbol": "xyz", "xbrl": "https://x/corporate/xbrl/INDAS_2.xml",
         "fromDate": "01-Jul-2024", "toDate": "30-Sep-2024",
         "filingDate": "09-Nov-2024 10:00", "audited": "Un-Audited",
         "consolidated": "Standalone", "relatingTo": "Second Quarter", "seqNumber": "2"},
    ])
    idx.to_parquet(tmp_path / "results_index.parquet")
    monkeypatch.setattr(nx, "XML_DIR", xml_dir)
    monkeypatch.setattr(nx, "INDEX_PQ", tmp_path / "results_index.parquet")
    monkeypatch.setattr(nx, "PANEL_PQ", tmp_path / "pit_quarterly.parquet")
    nx.parse_all()
    out = pd.read_parquet(tmp_path / "pit_quarterly.parquet")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["symbol"] == "ABC"
    assert r["filing_date"] == "08-Nov-2024 18:05"
    assert (r["revenue"], r["pat"]) == (1000.0, -50.5)
    assert (r["period_start"], r["period_end"]) == (Q_START, Q_END)


def _freeze_today(monkeypatch, y, m, d):
    class FakeDate(real_dt.date):
        @classmethod
        def today(cls):
            return cls(y, m, d)
    monkeypatch.setattr(nx, "_dt", types.SimpleNamespace(
        date=FakeDate, timedelta=real_dt.timedelta, datetime=real_dt.datetime))


def test_windows_are_exact_calendar_quarters_newest_first(monkeypatch):
    _freeze_today(monkeypatch, 2026, 10, 25)
    got = [(s.isoformat(), e.isoformat()) for s, e in nx._windows(3)]
    assert got == [("2026-10-01", "2026-10-25"), ("2026-07-01", "2026-09-30"),
                   ("2026-04-01", "2026-06-30")]


def test_windows_cross_the_year_boundary(monkeypatch):
    _freeze_today(monkeypatch, 2026, 2, 10)
    got = [(s.isoformat(), e.isoformat()) for s, e in nx._windows(2)]
    assert got == [("2026-01-01", "2026-02-10"), ("2025-10-01", "2025-12-31")]


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_integrated_index_maps_quarter_end_to_quarter_start(tmp_path, monkeypatch):
    rows = [{"symbol": "ABC", "cmName": "Abc Ltd", "broadcast_Date": "12-Feb-2026 17:00",
             "creation_Date": "12-Feb-2026 16:58", "qe_Date": "31-Dec-2025",
             "xbrl": "https://x/a.xml", "consolidated": "Standalone",
             "audited": "Un-Audited", "type_Sub": "Q3", "seq_Id": 77},
            {"symbol": "DEF", "cmName": "Def Ltd", "broadcast_Date": "10-Aug-2025 12:00",
             "creation_Date": "10-Aug-2025 11:59", "qe_Date": "30-Jun-2025",
             "xbrl": "https://x/d.xml", "consolidated": "Consolidated",
             "audited": "Un-Audited", "type_Sub": "Q1", "seq_Id": 78}]
    calls = []

    class FakeSession:
        headers = {}

        def get(self, url, timeout=None):
            calls.append(url)
            return _Resp({"data": rows, "totalCount": 2} if "integrated" in url else {})

    monkeypatch.setattr(nx.requests, "Session", FakeSession)
    monkeypatch.setattr(nx.time, "sleep", lambda s: None)
    monkeypatch.setattr(nx, "ROOT", tmp_path)
    monkeypatch.setattr(nx, "INDEX_PQ", tmp_path / "results_index.parquet")
    _freeze_today(monkeypatch, 2026, 2, 20)
    out = nx.fetch_index_integrated(1)
    got = out.set_index("symbol")
    assert got.loc["ABC", "fromDate"] == "01-Oct-2025"
    assert got.loc["ABC", "toDate"] == "31-Dec-2025"
    assert got.loc["DEF", "fromDate"] == "01-Apr-2025"
    assert got.loc["ABC", "filingDate"] == "12-Feb-2026 16:58"
    assert got.loc["ABC", "seqNumber"] == "IF77"
    assert sum("integrated" in c for c in calls) == 1       # one page was enough
    # re-running is idempotent on seqNumber
    assert len(nx.fetch_index_integrated(1)) == 2
