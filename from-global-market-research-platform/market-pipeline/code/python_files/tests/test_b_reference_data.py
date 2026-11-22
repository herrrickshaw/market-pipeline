"""Offline tests for reference_data (Damodaran/French parsing, cache, lookups)."""

import io
import zipfile

import pandas as pd
import pytest

import reference_data as rd


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "CACHE", tmp_path)
    return tmp_path


def _xlsx(sheets):
    """sheets: {name: list-of-rows} -> xlsx bytes (header=None layout)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)
    return buf.getvalue()


def _industry_sheet():
    pre = [["Damodaran preamble"], [None], ["Date: 2026"], [None], [None]]
    hdr = ["Industry Name", "Number of firms", "Current PE", "Trailing PE", "ROE", "Beta"]
    body = [
        ["Software (System & Application)", 300, 35.5, 40.0, 0.2, 1.2],
        ["Bank (Money Center)", 20, 9.1, 10.0, 0.11, 0.9],
        [None, 1, 1, 1, 1, 1],
    ]
    return pre + [hdr] + body


# ---------------------------------------------------------------- _cached_get
class _Resp:
    def __init__(self, content=b"payload", ok=True):
        self.content, self._ok = content, ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("404")


def test_cached_get_downloads_once_and_reuses_file(monkeypatch, cache):
    calls = []

    def get(url, headers, timeout):
        calls.append((url, headers, timeout))
        return _Resp(b"abc")

    monkeypatch.setattr(rd.requests, "get", get)
    assert rd._cached_get("http://x/y", "f.bin") == b"abc"
    assert (cache / "f.bin").read_bytes() == b"abc"
    assert rd._cached_get("http://x/y", "f.bin") == b"abc"
    assert len(calls) == 1 and calls[0][2] == 45 and "Mozilla" in calls[0][1]["User-Agent"]


def test_cached_get_http_error_not_cached(monkeypatch, cache):
    monkeypatch.setattr(rd.requests, "get", lambda *a, **k: _Resp(b"err", ok=False))
    with pytest.raises(RuntimeError):
        rd._cached_get("http://x", "bad.bin")
    assert not (cache / "bad.bin").exists()


# ---------------------------------------------------------------- _read_damodaran
def test_read_damodaran_finds_header_after_preamble():
    df = rd._read_damodaran(_xlsx({"Industry Averages": _industry_sheet()}))
    assert list(df["Industry"]) == ["Software (System & Application)", "Bank (Money Center)"]
    assert "_key" in df.columns
    assert df["_key"].tolist() == ["software (system & application)", "bank (money center)"]
    assert df.loc[df.Industry == "Bank (Money Center)", "Current PE"].iloc[0] == 9.1


def test_read_damodaran_skips_sheets_without_header():
    sheets = {"Readme": [["nothing", "here"]], "Data": _industry_sheet()}
    df = rd._read_damodaran(_xlsx(sheets))
    assert df is not None and len(df) == 2


def test_read_damodaran_header_case_insensitive_and_stripped():
    rows = [["  INDUSTRY NAME ", "ROE"], ["Steel", 0.1]]
    df = rd._read_damodaran(_xlsx({"S": rows}))
    assert df["Industry"].tolist() == ["Steel"] and df["_key"].tolist() == ["steel"]


def test_read_damodaran_header_inside_probe_window_is_found():
    rows = [["x"]] * 13 + [["Industry Name", "ROE"], ["Steel", 0.1]]  # header on the 14th row
    assert rd._read_damodaran(_xlsx({"S": rows}))["Industry"].tolist() == ["Steel"]


def test_read_damodaran_header_beyond_probe_window_is_missed():
    rows = [["x"]] * 16 + [["Industry Name", "ROE"], ["Steel", 0.1]]
    assert rd._read_damodaran(_xlsx({"S": rows})) is None  # only 15 rows are probed


def test_read_damodaran_garbage_or_no_header():
    assert rd._read_damodaran(b"not an excel file") is None
    assert rd._read_damodaran(_xlsx({"S": [["a", "b"], [1, 2]]})) is None


# ---------------------------------------------------------------- damodaran()
def _patch_get(monkeypatch, files):
    seen = []

    def cached(url, fname):
        seen.append((url, fname))
        v = files.get(url.rsplit("/", 1)[1])
        if v is None:
            raise RuntimeError("404")
        return v

    monkeypatch.setattr(rd, "_cached_get", cached)
    return seen


def test_damodaran_unknown_metric():
    assert rd.damodaran("nope") is None


def test_damodaran_region_file_then_us_fallback(monkeypatch):
    seen = _patch_get(monkeypatch, {"pedata.xls": _xlsx({"S": _industry_sheet()})})
    df = rd.damodaran("pe", "emerg")
    assert len(df) == 2
    assert [f for _, f in seen] == ["damodaran_pe_emerg.xls", "damodaran_pe_US.xls"]
    assert seen[0][0].endswith("/pedataemerg.xls")


def test_damodaran_region_hit_does_not_fall_back(monkeypatch):
    seen = _patch_get(monkeypatch, {"roeeurope.xls": _xlsx({"S": _industry_sheet()})})
    assert rd.damodaran("roe", "europe") is not None
    assert len(seen) == 1


def test_damodaran_empty_region_means_us_only(monkeypatch):
    seen = _patch_get(monkeypatch, {})
    assert rd.damodaran("beta", "") is None
    assert [f for _, f in seen] == ["damodaran_beta_US.xls"]


def test_damodaran_unparseable_region_file_tries_us(monkeypatch):
    _patch_get(monkeypatch, {"betaemerg.xls": b"junk", "beta.xls": _xlsx({"S": _industry_sheet()})})
    assert len(rd.damodaran("beta", "emerg")) == 2


# ---------------------------------------------------------------- industry_metric
@pytest.fixture
def table(monkeypatch):
    df = rd._read_damodaran(_xlsx({"S": _industry_sheet()}))
    monkeypatch.setattr(rd, "damodaran", lambda metric="pe", region="emerg": df)
    return df


def test_industry_metric_exact_match_prefers_current_pe(table):
    assert rd.industry_metric("Bank (Money Center)") == 9.1
    assert rd.industry_metric("  bank (money center) ") == 9.1


def test_industry_metric_loose_prefix_match(table):
    assert rd.industry_metric("Software (Something Else)") == 35.5  # first 8 chars "software"


def test_industry_metric_no_match_or_empty(table):
    assert rd.industry_metric("Zzzzzzzz") is None
    assert rd.industry_metric("") is None
    assert rd.industry_metric(None) is None


def test_industry_metric_column_hint_wins_and_other_metrics(table):
    assert rd.industry_metric("Bank (Money Center)", column_hint="trailing pe") == 10.0
    assert rd.industry_metric("Bank (Money Center)", "roe") == 0.11
    assert rd.industry_metric("Bank (Money Center)", "beta") == 0.9
    assert rd.industry_metric("Bank (Money Center)", "unknownmetric") is None


def test_industry_metric_non_numeric_cell_skipped(monkeypatch):
    rows = [["Industry Name", "Current PE", "PE"], ["Steel", "abc", 12.0]]
    df = rd._read_damodaran(_xlsx({"S": rows}))
    monkeypatch.setattr(rd, "damodaran", lambda metric="pe", region="emerg": df)
    assert rd.industry_metric("Steel") == 12.0  # falls through to the next numeric column


def test_industry_metric_no_table(monkeypatch):
    monkeypatch.setattr(rd, "damodaran", lambda *a, **k: None)
    assert rd.industry_metric("Steel") is None


# ---------------------------------------------------------------- french_factors
def _ff_zip(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors.CSV", text)
    return buf.getvalue()


FF_TEXT = """This file was created by CMPT_ME_BEME_RETS using the 202603 CRSP database.

,Mkt-RF,SMB,HML,RF
192607,   2.96,  -2.56,  -2.43,   0.22
192608,   2.64,  -1.17,   3.82,   0.25
192609,bad,1,2,3

 Annual Factors: January-December
1927,29.47,-2.46,-3.68,3.12
"""


def test_french_factors_parses_monthly_block_only(monkeypatch):
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: _ff_zip(FF_TEXT))
    df = rd.french_factors()
    assert list(df.columns) == ["YYYYMM", "Mkt-RF", "SMB", "HML", "RF"]
    assert df["YYYYMM"].tolist() == ["192607", "192608", "192609"]
    assert df.loc[0, "Mkt-RF"] == 2.96 and df.loc[1, "HML"] == 3.82
    assert pd.isna(df.loc[2, "Mkt-RF"])  # coerced
    assert "1927" not in df["YYYYMM"].tolist()  # annual block excluded


def test_french_factors_failures(monkeypatch):
    def boom(url, f):
        raise RuntimeError("down")

    monkeypatch.setattr(rd, "_cached_get", boom)
    assert rd.french_factors() is None
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: b"not a zip")
    assert rd.french_factors() is None
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: _ff_zip("no data rows here\n"))
    assert rd.french_factors() is None


# ---------------------------------------------------------------- companies
def _companies_xls():
    df = pd.DataFrame(
        {
            "Company Name": ["Apple Inc.", "Reliance Industries", None],
            "Exchange:Ticker": ["NasdaqGS:AAPL", "NSEI:RELIANCE", "X:Y"],
            "Industry Group": ["Computers/Peripherals", "Oil/Gas (Integrated)", "Z"],
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="By company name", index=False)
    return buf.getvalue()


def test_damodaran_companies_splits_ticker_and_caches_parquet(monkeypatch, cache):
    calls = []
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: calls.append(f) or _companies_xls())
    df = rd.damodaran_companies()
    assert len(df) == 2  # row without a company name dropped
    assert df["Exchange"].tolist() == ["NasdaqGS", "NSEI"]
    assert df["Ticker"].tolist() == ["AAPL", "RELIANCE"]
    assert (cache / "damodaran_companies.parquet").exists()
    monkeypatch.setattr(rd, "_cached_get", lambda *a: pytest.fail("refetched"))
    assert len(rd.damodaran_companies()) == 2  # served from parquet
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: calls.append(f) or _companies_xls())
    rd.damodaran_companies(refresh=True)
    assert len(calls) == 2


def test_damodaran_companies_failure(monkeypatch):
    monkeypatch.setattr(rd, "_cached_get", lambda *a: b"junk")
    assert rd.damodaran_companies() is None

    def boom(*a):
        raise RuntimeError("x")

    monkeypatch.setattr(rd, "_cached_get", boom)
    assert rd.damodaran_companies() is None


def test_company_industry_by_ticker_then_name(monkeypatch):
    monkeypatch.setattr(rd, "_cached_get", lambda url, f: _companies_xls())
    assert rd.company_industry("aapl") == "Computers/Peripherals"
    assert rd.company_industry("  RELIANCE INDUSTRIES ") == "Oil/Gas (Integrated)"
    assert rd.company_industry("unknown co") is None


def test_company_industry_no_universe(monkeypatch):
    monkeypatch.setattr(rd, "damodaran_companies", lambda refresh=False: None)
    assert rd.company_industry("AAPL") is None


# ---------------------------------------------------------------- aqr_links / info
def test_aqr_links_dedup_sort_and_absolutize(monkeypatch):
    html = (
        '<a href="/-/media/a.xlsx">a</a><a href="https://x.com/b.XLS">b</a>'
        '<a href="/-/media/a.xlsx">dup</a><a href="/doc.pdf">p</a>'
    )
    monkeypatch.setattr(rd.requests, "get", lambda *a, **k: type("R", (), {"text": html})())
    assert rd.aqr_links() == ["https://www.aqr.com/-/media/a.xlsx", "https://x.com/b.XLS"]


def test_aqr_links_failure_is_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(rd.requests, "get", boom)
    assert rd.aqr_links() == []


def test_info_lists_cached_files_sorted(cache):
    (cache / "b.xls").write_bytes(b"1")
    (cache / "a.xls").write_bytes(b"1")
    assert rd.info() == {"cache_dir": str(cache), "cached_files": ["a.xls", "b.xls"]}


def test_industry_metric_loose_match_uses_first_eight_characters(monkeypatch):
    rows = [["Industry Name", "Current PE"], ["Soft Drinks", 11.0], ["Software", 22.0]]
    df = rd._read_damodaran(_xlsx({"S": rows}))
    monkeypatch.setattr(rd, "damodaran", lambda metric="pe", region="emerg": df)
    # "software (zzz)" has no exact row; its first 8 chars pick Software, not Soft Drinks
    assert rd.industry_metric("Software (Zzz)") == 22.0
