"""Offline tests for ticker_exchange_map (suffix->venue, US spelling variants, build/main)."""

import pandas as pd
import pytest

import ticker_exchange_map as tem


# ---------------------------------------------------------------- _suffix
@pytest.mark.parametrize(
    "sym,expected",
    [
        ("RELIANCE.NS", "NS"),
        ("0700.hk", "HK"),
        ("1513.TWO", "TWO"),
        ("AAPL", None),
        ("BRK-B", None),
        ("ABCD.WXYZ", None),
        ("A.B.DE", "DE"),
        ("", None),
        (".T", "T"),
    ],
)
def test_suffix(sym, expected):
    assert tem._suffix(sym) == expected


def test_suffix_map_is_consistent():
    for sfx, (name, mic, cc) in tem.SUFFIX_MAP.items():
        assert sfx == sfx.upper() and 1 <= len(sfx) <= 3
        assert len(mic) == 4 and len(cc) == 2 and name
    # the .NS / .BO distinction the module exists for
    assert tem.SUFFIX_MAP["NS"][1] == "XNSE" and tem.SUFFIX_MAP["BO"][1] == "XBOM"
    assert set(tem.US_EXCHANGE) == {"N", "Q", "A", "P", "Z", "V"}


# ---------------------------------------------------------------- _us_candidates
def test_us_candidates_plain_and_uppercased():
    assert tem._us_candidates("aapl") == ["AAPL"]


def test_us_candidates_share_class():
    assert tem._us_candidates("BRK-B") == ["BRK-B", "BRK.B"]


def test_us_candidates_preferred():
    assert tem._us_candidates("ABR-PD") == ["ABR-PD", "ABR$D", "ABR-D", "ABR.PD"]


def test_us_candidates_warrant():
    assert tem._us_candidates("ACHR-WT") == ["ACHR-WT", "ACHR.W", "ACHR+", "ACHR.WT"]
    assert tem._us_candidates("XYZ-WS")[1:3] == ["XYZ.W", "XYZ+"]


def test_us_candidates_preferred_needs_single_letter_at_end():
    assert tem._us_candidates("ABR-PDX") == ["ABR-PDX", "ABR.PDX"]


# ---------------------------------------------------------------- _us_lookup
def _table():
    return pd.DataFrame(
        {"Listing Exchange": ["N", "Q", "A"], "ETF": ["N", "Y", "N"]},
        index=pd.Index(["ABR$D", "AAPL", "BRK.B"], name="_key"),
    )


def test_us_lookup_first_matching_candidate_and_none():
    t = _table()
    assert tem._us_lookup(["ABR-PD", "aapl", "BRK-B", "ZZZ"], t, "Listing Exchange") == [
        "N",
        "Q",
        "A",
        None,
    ]
    assert tem._us_lookup(["AAPL"], t, "ETF") == ["Y"]
    assert tem._us_lookup([], t, "ETF") == []


def test_us_lookup_prefers_earlier_candidate():
    t = pd.DataFrame({"v": ["exact", "dotted"]}, index=["BRK-B", "BRK.B"])
    assert tem._us_lookup(["BRK-B"], t, "v") == ["exact"]


# ---------------------------------------------------------------- _nasdaq
def _nasdaq_file(tmp_path):
    p = tmp_path / "nasdaqtraded.txt"
    p.write_text(
        "Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|Test Issue|"
        "NASDAQ Symbol|CQS Symbol\n"
        "Y|AAPL|Apple Inc.|Q|G|N|N|AAPL|\n"
        "Y|BRK.B|Berkshire B|N||N|N|BRK-B|BRK.B\n"
        "Y|TEST|Test Co|Q|G|N|Y|TEST|\n"
        "Y|spy |SPDR|P||Y|N|SPY|SPY\n"
        "Y||NoSymbol|Q|G|N|N||\n"
    )
    return p


def test_nasdaq_indexes_all_spellings_and_drops_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(tem, "NASDAQ_URL", str(_nasdaq_file(tmp_path)))
    n = tem._nasdaq()
    assert "TEST" not in n.index
    assert {"AAPL", "BRK.B", "BRK-B", "SPY"} <= set(n.index)
    assert n.index.is_unique
    assert n.at["SPY", "Listing Exchange"] == "P"  # stripped + upper-cased
    assert n.at["BRK-B", "Security Name"] == "Berkshire B"


# ---------------------------------------------------------------- build / main
def _universe(tmp_path):
    u = tmp_path / "uni"
    u.mkdir()
    pd.DataFrame(
        {
            "yf_symbol": ["RELIANCE.NS", "AAPL", "BRK-B"],
            "exchange": ["NSE/BSE", "NYSE/NASDAQ/AMEX", "NYSE/NASDAQ/AMEX"],
            "market_code": ["IN", "US", "US"],
            "market_name": ["India", "US", "US"],
        }
    ).to_csv(u / "validated_universe_flat.csv", index=False)
    pd.DataFrame(
        {
            "yf_symbol": ["RELIANCE.NS", "7203.T", "SPY", "ZZZ.UNKNOWN", "QQQ"],
            "exchange": ["DUP", "TSE", "NYSE/NASDAQ/AMEX", "?", "NYSE/NASDAQ/AMEX"],
            "market_code": ["XX", "JP", "US", "ZZ", "US"],
            "market_name": ["dup", "Japan", "US", "?", "US"],
        }
    ).to_csv(u / "global_universe_flat.csv", index=False)
    return u


@pytest.fixture
def built(monkeypatch, tmp_path):
    monkeypatch.setattr(tem, "UNIVERSE", _universe(tmp_path))
    monkeypatch.setattr(tem, "NASDAQ_URL", str(_nasdaq_file(tmp_path)))
    monkeypatch.setattr(
        tem,
        "_pg_reference",
        lambda: pd.DataFrame(
            {"name": ["Reliance Industries"], "liquidity_tier": ["T1"], "turnover_usd": [1.5e6]},
            index=pd.Index(["RELIANCE.NS"], name="ticker"),
        ),
    )
    return tmp_path


def test_build_resolves_venues_and_joins_reference(built, capsys):
    d = tem.build()
    assert list(d.columns) == tem.COLS
    assert d["yf_symbol"].is_unique and len(d) == 7
    by = d.set_index("yf_symbol")
    # validated wins over the global duplicate
    assert bool(by.at["RELIANCE.NS", "validated"]) is True
    assert by.at["RELIANCE.NS", "exchange_group"] == "NSE/BSE"
    assert by.at["RELIANCE.NS", "exchange_mic"] == "XNSE"
    assert by.at["RELIANCE.NS", "country"] == "IN"
    assert by.at["RELIANCE.NS", "base_ticker"] == "RELIANCE"
    assert by.at["RELIANCE.NS", "name"] == "Reliance Industries"
    assert by.at["RELIANCE.NS", "liquidity_tier"] == "T1"
    assert by.at["7203.T", "exchange"] == "Tokyo Stock Exchange"
    # US: venue from the NASDAQ directory, incl. dotted/dashed share class and ETF flag
    assert by.at["AAPL", "exchange_mic"] == "XNAS" and by.at["AAPL", "country"] == "US"
    assert by.at["BRK-B", "exchange_mic"] == "XNYS"
    assert bool(by.at["SPY", "is_etf"]) is True and by.at["SPY", "exchange_mic"] == "ARCX"
    assert bool(by.at["AAPL", "is_etf"]) is False
    assert by.at["AAPL", "name"] == "Apple Inc."
    # unknown US symbol stays blank; unknown suffix is not treated as US
    assert pd.isna(by.at["QQQ", "exchange"])
    assert pd.isna(by.at["ZZZ.UNKNOWN", "exchange"])
    assert by.at["BRK-B", "base_ticker"] == "BRK-B"
    out = capsys.readouterr().out
    assert "universe: 7 symbols (3 validated)" in out
    assert "liquidity tiers joined" in out


def test_build_sorted_by_country_exchange_symbol(built):
    d = tem.build()
    countries = [c for c in d["country"] if isinstance(c, str)]
    assert countries == sorted(countries)


def test_build_survives_nasdaq_and_pg_failures(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tem, "UNIVERSE", _universe(tmp_path))
    monkeypatch.setattr(tem, "NASDAQ_URL", str(tmp_path / "missing.txt"))

    def boom():
        raise RuntimeError("no psql")

    monkeypatch.setattr(tem, "_pg_reference", boom)
    d = tem.build()
    out = capsys.readouterr().out
    assert "NASDAQ directory unavailable (FileNotFoundError)" in out
    assert "Postgres enrich skipped (RuntimeError)" in out
    assert d["liquidity_tier"].isna().all() and d["turnover_usd"].isna().all()
    assert d.set_index("yf_symbol").at["7203.T", "exchange_mic"] == "XTKS"


def _args(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["ticker_exchange_map.py", *argv])


def test_main_lookup_requires_built_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tem, "OUT", tmp_path / "nope.parquet")
    _args(monkeypatch, "--lookup", "AAPL")
    assert tem.main() == 1
    assert "build it first" in capsys.readouterr().out


def test_main_writes_then_lookup_is_case_insensitive(built, monkeypatch, capsys):
    out = built / "cache" / "ref.parquet"
    monkeypatch.setattr(tem, "OUT", out)
    _args(monkeypatch)
    assert tem.main() == 0
    assert out.exists() and len(pd.read_parquet(out)) == 7
    text = capsys.readouterr().out
    assert "rows 7 | exchanges" in text
    assert "duplicate yf_symbol: 0" in text
    assert "unresolved venue 2 (28.6%)" in text
    _args(monkeypatch, "--lookup", "reliance.ns", "nope")
    assert tem.main() == 0
    text = capsys.readouterr().out
    assert "RELIANCE.NS" in text and "XNSE" in text
    _args(monkeypatch, "--lookup", "nope")
    assert tem.main() == 0
    assert "no match" in capsys.readouterr().out


def test_main_verify_writes_nothing(built, monkeypatch, capsys):
    out = built / "cache" / "ref.parquet"
    monkeypatch.setattr(tem, "OUT", out)
    _args(monkeypatch, "--verify")
    assert tem.main() == 0
    assert not out.exists()
    assert "--verify: not written" in capsys.readouterr().out
