"""Offline tests for universe_sources (official-feed parsers, Damodaran suffix mapping)."""

import sys
import types

import pandas as pd
import pytest

import universe_sources as us


class _R:
    def __init__(self, payload=None, content=b"", status_ok=True):
        self._p, self.content, self._ok = payload, content, status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


def _get(monkeypatch, handler):
    calls = []

    def get(url, **kw):
        calls.append((url, kw))
        return handler(url, kw)

    monkeypatch.setattr(us.requests, "get", get)
    return calls


# ---------------------------------------------------------------- US / SG
def test_us_sec_normalises_dedups_sorts_and_sends_contact_header(monkeypatch):
    payload = {
        "0": {"ticker": "aapl"},
        "1": {"ticker": "BRK.B"},
        "2": {"ticker": "msft"},
        "3": {"ticker": "AAPL"},
        "4": {"ticker": ""},
        "5": {"cik": 1},
    }
    calls = _get(monkeypatch, lambda u, k: _R(payload))
    assert us.us_sec() == ["AAPL", "BRK-B", "MSFT"]
    url, kw = calls[0]
    assert "sec.gov" in url and kw["timeout"] == 30 and "@" in kw["headers"]["User-Agent"]


def test_us_sec_http_error_propagates(monkeypatch):
    _get(monkeypatch, lambda u, k: _R({}, status_ok=False))
    with pytest.raises(RuntimeError):
        us.us_sec()


def test_sg_sgx_keeps_only_equity_like_types(monkeypatch):
    rows = [
        {"nc": "D05", "type": "stocks"},
        {"nc": "A17U", "type": "reits"},
        {"nc": "K71U", "type": "businesstrusts"},
        {"nc": "ETF1", "type": "etfs"},
        {"nc": "", "type": "stocks"},
        {"type": "stocks"},
        {"nc": "D05", "type": "stocks"},
    ]
    calls = _get(monkeypatch, lambda u, k: _R({"data": {"prices": rows}}))
    assert us.sg_sgx() == ["A17U.SI", "D05.SI", "K71U.SI"]
    assert "excludetypes=bonds" in calls[0][0]


# ---------------------------------------------------------------- CN
def test_cn_eastmoney_paginates_until_total(monkeypatch):
    pages = {
        1: {"total": 3, "diff": [{"f12": "600000", "f13": 1}, {"f12": "000001", "f13": 0}]},
        2: {"total": 3, "diff": {"0": {"f12": "300750", "f13": 0}, "1": {"f12": None, "f13": 0}}},
    }
    calls = _get(monkeypatch, lambda u, k: _R({"data": pages[k["params"]["pn"]]}))
    assert us.cn_eastmoney() == ["000001.SZ", "300750.SZ", "600000.SS"]
    assert [c[1]["params"]["pn"] for c in calls] == [1, 2]  # stops once total reached
    assert calls[0][1]["params"]["pz"] == 200


def test_cn_eastmoney_falls_over_to_next_host(monkeypatch):
    seen = []

    def handler(url, kw):
        seen.append(url)
        if url.startswith("https://push2.eastmoney.com"):
            raise OSError("blocked")
        return _R({"data": {"total": 1, "diff": [{"f12": "600519", "f13": 1}]}})

    _get(monkeypatch, handler)
    assert us.cn_eastmoney() == ["600519.SS"]
    assert seen[0].startswith("https://push2.") and seen[1].startswith("https://82.push2.")


def test_cn_eastmoney_all_hosts_fail_or_empty(monkeypatch):
    def boom(u, k):
        raise OSError("down")

    _get(monkeypatch, boom)
    assert us.cn_eastmoney() == []
    _get(monkeypatch, lambda u, k: _R({"data": None}))
    assert us.cn_eastmoney() == []
    _get(monkeypatch, lambda u, k: _R({"data": {"total": 5, "diff": []}}))
    assert us.cn_eastmoney() == []


def test_cn_eastmoney_stops_on_empty_page_even_if_total_unmet(monkeypatch):
    pages = {1: {"total": 99, "diff": [{"f12": "600000", "f13": 1}]}, 2: {"total": 99, "diff": []}}
    _get(monkeypatch, lambda u, k: _R({"data": pages[k["params"]["pn"]]}))
    assert us.cn_eastmoney() == ["600000.SS"]


# ---------------------------------------------------------------- JP / KR / IN
def _fake_mod(monkeypatch, name, **attrs):
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))


def test_jp_jpx_prefers_official_then_falls_back(monkeypatch):
    def jpx():
        return [{"code": "7203"}, {"yf_ticker": "6758.T"}, {"code": "7203"}, {}]

    def kab():
        return [{"code": "9984"}]

    _fake_mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=jpx,
        fetch_tse_universe_kabupy=kab,
    )
    assert us.jp_jpx() == ["6758.T", "7203.T"]

    def jpx_boom():
        raise RuntimeError("x")

    _fake_mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=jpx_boom,
        fetch_tse_universe_kabupy=kab,
    )
    assert us.jp_jpx() == ["9984.T"]
    _fake_mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=lambda: [],
        fetch_tse_universe_kabupy=kab,
    )
    assert us.jp_jpx() == ["9984.T"]  # empty official list also falls back


def test_kr_krx_default_suffix_and_missing_codes(monkeypatch):
    _fake_mod(
        monkeypatch,
        "full_korea_market_scan",
        build_krx_universe=lambda: [
            {"code": "005930"},
            {"code": "035720", "yf_suffix": ".KQ"},
            {"name": "nocode"},
            {"code": "005930"},
        ],
    )
    assert us.kr_krx() == ["005930.KS", "035720.KQ"]


def test_in_bhavcopy_store_then_history_fallback(monkeypatch):
    _fake_mod(monkeypatch, "bhavcopy_store", symbols=lambda: {"TCS", "INFY"})
    _fake_mod(monkeypatch, "bhavcopy_history", fetch_history=lambda verbose: pytest.fail("used"))
    assert us.in_bhavcopy() == ["INFY", "TCS"]

    _fake_mod(monkeypatch, "bhavcopy_store", symbols=lambda: set())
    _fake_mod(monkeypatch, "bhavcopy_history", fetch_history=lambda verbose: {"WIPRO": 1, "ITC": 2})
    assert us.in_bhavcopy() == ["ITC", "WIPRO"]

    def boom():
        raise RuntimeError("no store")

    _fake_mod(monkeypatch, "bhavcopy_store", symbols=boom)
    assert us.in_bhavcopy() == ["ITC", "WIPRO"]


# ---------------------------------------------------------------- Euronext / EU
def _euronext_csv():
    text = (
        "Name;ISIN;Symbol;Market;Currency\n"
        "LVMH;FR0000121014;MC;Euronext Paris, Euronext Brussels;EUR\n"
        "ASML;NL0010273215;ASML;Euronext Amsterdam;EUR\n"
        "Equinor;NO0010096985;EQNR;Oslo Børs;NOK\n"
        "Unknown Venue;XX0000000000;UNK;Somewhere;EUR\n"
        "No ISIN;;NOISIN;Euronext Paris;EUR\n"
        "No Symbol;FR0000000001;;Euronext Paris;EUR\n"
        "Intesa;IT0000072618;ISP;Borsa Italiana;EUR\n"
        "LVMH dup;FR0000121014;MC;Euronext Paris;EUR\n"
    )
    return text.encode("utf-8-sig")


def test_euronext_equities_maps_primary_venue_suffix(monkeypatch):
    calls = _get(monkeypatch, lambda u, k: _R(content=_euronext_csv()))
    out = us.euronext_equities("XPAR,XAMS")
    assert out == ["ASML.AS", "EQNR.OL", "ISP.MI", "MC.PA"]
    url, kw = calls[0]
    assert url.endswith("mics=XPAR,XAMS") and kw["timeout"] == 40
    assert "Referer" in kw["headers"]


def test_euronext_equities_http_error(monkeypatch):
    _get(monkeypatch, lambda u, k: _R(status_ok=False))
    with pytest.raises(RuntimeError):
        us.euronext_equities()


def test_eu_curated_unions_stoxx_meta_and_extras(monkeypatch):
    _fake_mod(
        monkeypatch, "full_european_market_scan", EURO_STOXX_50_META={"SAP.DE": 1, "ZZZ.PA": 2}
    )
    out = us.eu_curated()
    assert out == sorted(set(out))
    assert "ZZZ.PA" in out and "HSBA.L" in out and "NESN.SW" in out and "SAP.DE" in out
    assert out.count("SAP.DE") == 1


def test_eu_adds_live_euronext_and_tolerates_failure(monkeypatch):
    monkeypatch.setattr(us, "eu_curated", lambda: ["B.L", "A.DE"])
    monkeypatch.setattr(us, "euronext_equities", lambda: ["MC.PA", "A.DE"])
    assert us.eu() == ["A.DE", "B.L", "MC.PA"]

    def boom():
        raise OSError("net")

    monkeypatch.setattr(us, "euronext_equities", boom)
    assert us.eu() == ["A.DE", "B.L"]


def test_investpy_universe_concats_and_skips_failures(monkeypatch):
    def get_stocks(country):
        if country == "norway":
            raise RuntimeError("blocked")
        return pd.DataFrame({"symbol": [country[:2].upper()], "country": [country]})

    _fake_mod(monkeypatch, "investpy", get_stocks=get_stocks)
    df = us.investpy_universe(("france", "norway", "italy"))
    assert df["country"].tolist() == ["france", "italy"] and df.index.tolist() == [0, 1]


def test_investpy_universe_missing_lib_or_all_failing(monkeypatch):
    monkeypatch.setitem(sys.modules, "investpy", None)
    assert us.investpy_universe() is None

    def boom(country):
        raise RuntimeError("x")

    _fake_mod(monkeypatch, "investpy", get_stocks=boom)
    assert us.investpy_universe(("france",)) is None


# ---------------------------------------------------------------- Damodaran-derived
def _damo(rows):
    return pd.DataFrame(rows, columns=["Country", "Exchange", "Ticker"])


def _patch_damo(monkeypatch, df):
    _fake_mod(monkeypatch, "reference_data", damodaran_companies=lambda: df)


def test_from_damodaran_suffix_by_exchange_and_hk_padding(monkeypatch):
    _patch_damo(
        monkeypatch,
        _damo(
            [
                ("Hong Kong", "SEHK", "700"),
                ("Hong Kong", "SEHK", "9988"),
                ("Hong Kong", "XYZ", "1"),
                ("Japan", "TSE", "7203"),
                ("Hong Kong", "SEHK", None),
                ("Hong Kong", "SEHK", ""),
                ("Hong Kong", "SEHK", " 5 "),
            ]
        ),
    )
    assert us.from_damodaran("Hong Kong") == ["0005.HK", "0700.HK", "9988.HK"]


def test_from_damodaran_allowed_exchange_filter(monkeypatch):
    _patch_damo(
        monkeypatch,
        _damo([("Canada", "TSX", "RY"), ("Canada", "TSXV", "ABC"), ("Canada", "CNSX", "XYZ")]),
    )
    assert us.from_damodaran("Canada", {"TSX"}) == ["RY.TO"]
    assert us.from_damodaran("Canada", {"TSX", "TSXV"}) == ["ABC.V", "RY.TO"]
    assert us.from_damodaran("Canada") == ["ABC.V", "RY.TO", "XYZ.CN"]
    assert us.from_damodaran("Canada", set()) == [
        "ABC.V",
        "RY.TO",
        "XYZ.CN",
    ]  # empty set = no filter


def test_from_damodaran_nordic_country_suffix_overrides_exchange(monkeypatch):
    _patch_damo(
        monkeypatch,
        _damo(
            [
                ("Sweden", "OM", "VOLV-B"),
                ("Denmark", "OM", "NOVO-B"),
                ("Finland", "OM", "NOKIA"),
                ("Norway", "OSE", "EQNR"),
            ]
        ),
    )
    assert us.from_damodaran("Sweden") == ["VOLV-B.ST"]
    assert us.from_damodaran("Denmark") == ["NOVO-B.CO"]
    assert us.from_damodaran("Finland") == ["NOKIA.HE"]
    assert us.from_damodaran("Norway") == ["EQNR.OL"]


def test_from_damodaran_no_universe_or_no_match(monkeypatch):
    _patch_damo(monkeypatch, None)
    assert us.from_damodaran("Canada") == []
    _patch_damo(monkeypatch, _damo([("Canada", "TSX", "RY")]))
    assert us.from_damodaran("Atlantis") == []


def test_damodaran_suffix_table_covers_provider_exchanges():
    needed = {
        "SEHK",
        "TWSE",
        "TPEX",
        "TSX",
        "TSXV",
        "CNSX",
        "ASX",
        "BOVESPA",
        "JSE",
        "SASE",
        "LSE",
        "AIM",
        "XTRA",
        "DB",
        "HMSE",
        "SWX",
    }
    assert needed <= set(us._DAMO_EXCH_SUFFIX)


# ---------------------------------------------------------------- registry
def test_get_universe_dispatches_case_insensitively(monkeypatch):
    monkeypatch.setitem(us.PROVIDERS, "US", lambda: ["AAPL"])
    assert us.get_universe("us") == ["AAPL"]
    with pytest.raises(KeyError):
        us.get_universe("zz")


def test_providers_lambdas_pass_country_and_exchanges(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        us, "from_damodaran", lambda country, allowed=None: seen.update({country: allowed}) or []
    )
    for m in ("HK", "TW", "CA", "AU", "BR", "ZA", "SA", "SE", "DK", "FI", "UK", "DE", "CH"):
        us.get_universe(m)
    assert seen["Hong Kong"] == {"SEHK"} and seen["Taiwan"] == {"TWSE", "TPEX"}
    assert seen["United Kingdom"] == {"LSE", "AIM"} and seen["Germany"] == {"XTRA", "DB", "HMSE"}
    assert seen["Sweden"] is None and seen["Canada"] == {"TSX", "TSXV", "CNSX"}
    assert set(us.PROVIDERS) >= {"US", "IN", "JP", "KR", "SG", "CN", "EU"}
