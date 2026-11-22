"""Offline tests for symbol_master (name cleaning, deadline guard, builders, cache)."""

import signal
import sys
import time
import types
from datetime import datetime

import pandas as pd
import pytest

import symbol_master as sm


# ---------------------------------------------------------------- clean_name
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ADANI ENTERPRISES LIMITED", "ADANI ENTERPRISES"),
        ("MARUTI SUZUKI INDIA LTD.", "MARUTI SUZUKI INDIA"),
        ("Amara Raja Energy & Mobility L", "AMARA RAJA ENERGY"),
        ("Tata Consultancy Services Limited Extra Words", "TATA CONSULTANCY SERVICES"),
        ("The Bank of India", "BANK INDIA"),  # THE/OF filler dropped
        ("3M Co", ""),  # digits removed -> "M" (len<3) dropped
        ("AB CD", ""),  # tokens shorter than 3 chars dropped
        ("ABC", "ABC"),  # exactly 3 chars kept
        ("", ""),
        (None, ""),
        (123, ""),
        ("LIMITED", ""),
        ("Reliance-Industries Ltd", "RELIANCE INDUSTRIES"),
    ],
)
def test_clean_name(raw, expected):
    assert sm.clean_name(raw) == expected


def test_clean_name_keeps_at_most_three_tokens():
    assert sm.clean_name("ONE TWO THR FOU FIV") == "ONE TWO THR"


# ---------------------------------------------------------------- _deadline
def test_deadline_restores_socket_timeout_and_handler():
    import socket

    socket.setdefaulttimeout(None)
    prev = signal.getsignal(signal.SIGALRM)
    with sm._deadline(5, "lbl"):
        assert socket.getdefaulttimeout() == 5  # min(seconds, 30)
    assert socket.getdefaulttimeout() is None
    assert signal.getsignal(signal.SIGALRM) == prev


def test_deadline_socket_timeout_capped_at_30():
    import socket

    socket.setdefaulttimeout(None)
    with sm._deadline(60, "x"):
        assert socket.getdefaulttimeout() == 30
    assert socket.getdefaulttimeout() is None


def test_deadline_raises_source_timeout():
    with (
        pytest.raises(sm.SourceTimeout, match="slowsrc: exceeded 1s wall-clock deadline"),
        sm._deadline(1, "slowsrc"),
    ):
        time.sleep(3)


def test_deadline_prints_when_over_half(monkeypatch, capsys):
    clock = iter([100.0, 100.8])
    monkeypatch.setattr(sm.time, "time", lambda: next(clock))
    with sm._deadline(1, "lbl"):
        pass
    assert "lbl: 1s (deadline 1s)" in capsys.readouterr().out


def test_deadline_quiet_when_fast(monkeypatch, capsys):
    clock = iter([100.0, 100.4])
    monkeypatch.setattr(sm.time, "time", lambda: next(clock))
    with sm._deadline(1, "lbl"):
        pass
    assert capsys.readouterr().out == ""


def test_deadline_non_main_thread_falls_back_without_alarm():
    import socket
    import threading

    seen = {}

    def work():
        socket.setdefaulttimeout(None)
        with sm._deadline(2, "t"):
            seen["during"] = socket.getdefaulttimeout()
            seen["handler"] = signal.getsignal(signal.SIGALRM)
        seen["after"] = socket.getdefaulttimeout()

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert seen["during"] == 2
    assert seen["after"] is None


# ---------------------------------------------------------------- _us_names
class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _fake_requests(monkeypatch, payload=None, exc=None):
    calls = []

    def get(url, timeout=None, headers=None):
        calls.append((url, timeout, headers))
        if exc:
            raise exc
        return _Resp(payload)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    return calls


def test_us_names_filters_and_classifies(monkeypatch):
    data = {
        "data": [
            [1, "Apple Inc", "aapl", "Nasdaq"],
            [2, "Ford", "F", "NYSE"],
            [3, "Amex Co", "AMX", "NYSE American AMEX"],
            [4, "OTC Co", "OTCX", "OTC"],  # unknown exchange -> skipped
            [5, "Dotted", "BRK.B", "NYSE"],  # '.' skipped
            [6, "Too Long", "ABCDEF", "NYSE"],  # >5 chars skipped
            [7, "Spaced", "A B", "NYSE"],
            [8, "Empty", "", "NYSE"],
            [9, "Five", "ABCDE", "Nasdaq"],  # exactly 5 allowed
            [10, "Short"],  # too few columns
        ]
    }
    calls = _fake_requests(monkeypatch, data)
    rows = sm._us_names()
    assert rows == [
        {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "suffix": ""},
        {"symbol": "F", "name": "Ford", "exchange": "NYSE", "suffix": ""},
        {"symbol": "AMX", "name": "Amex Co", "exchange": "NYSE", "suffix": ""},
        {"symbol": "ABCDE", "name": "Five", "exchange": "NASDAQ", "suffix": ""},
    ]
    assert calls[0][1] == 30
    assert "sec.gov" in calls[0][0]


def test_us_names_network_error_returns_empty(monkeypatch):
    _fake_requests(monkeypatch, exc=RuntimeError("down"))
    assert sm._us_names() == []


def test_us_names_missing_data_key(monkeypatch):
    _fake_requests(monkeypatch, {})
    assert sm._us_names() == []


# ---------------------------------------------------------------- _europe_names (CSV fallback)
def _fake_root(monkeypatch, tmp_path):
    root = tmp_path / "a" / "b"
    (root / "data").mkdir(parents=True)
    # Path(__file__).resolve().parents[3] of <root>/c/x/y/z.py -> root
    monkeypatch.setattr(sm, "__file__", str(root / "c" / "x" / "y" / "z.py"))
    # make sure duckdb cannot be imported -> fallback path
    monkeypatch.setitem(sys.modules, "duckdb", None)
    return root


def test_europe_names_csv_fallback(monkeypatch, tmp_path):
    root = _fake_root(monkeypatch, tmp_path)
    pd.DataFrame(
        {
            "yf_ticker": ["ADS.DE", " RIO.L "],
            "name": ["Adidas", "Rio Tinto"],
            "exchange": ["Xetra", "LSE"],
        }
    ).to_csv(root / "data" / "europe_all_list.csv", index=False)
    rows = sm._europe_names()
    assert rows == [
        {"symbol": "ADS.DE", "name": "Adidas", "exchange": "Xetra", "suffix": ""},
        {"symbol": "RIO.L", "name": "Rio Tinto", "exchange": "LSE", "suffix": ""},
    ]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: str(NaN) == 'nan' is truthy, so the `if yf_ticker and name` guard in "
    "_europe_names never rejects blank CSV cells; a blank yf_ticker becomes symbol 'nan'",
)
def test_europe_names_blank_cells_are_skipped(monkeypatch, tmp_path):
    root = _fake_root(monkeypatch, tmp_path)
    (root / "data" / "europe_all_list.csv").write_text(
        "yf_ticker,name,exchange\n,Ghost,X\nNONAME.PA,,Euronext Paris\nADS.DE,Adidas,Xetra\n"
    )
    assert [r["symbol"] for r in sm._europe_names()] == ["ADS.DE"]


def test_europe_names_no_source_is_empty(monkeypatch, tmp_path):
    _fake_root(monkeypatch, tmp_path)
    assert sm._europe_names() == []


def test_europe_names_bad_csv_is_empty(monkeypatch, tmp_path):
    root = _fake_root(monkeypatch, tmp_path)
    (root / "data" / "europe_all_list.csv").write_text("a,b\n1,2\n")  # no yf_ticker column
    assert sm._europe_names() == []


def test_europe_names_duckdb_preferred(monkeypatch, tmp_path):
    _fake_root(monkeypatch, tmp_path)
    closed = []

    class Con:
        def execute(self, q):
            assert "europe_all_list" in q
            return self

        def fetchdf(self):
            return pd.DataFrame({"yf_ticker": ["SAP.DE"], "name": ["SAP"], "exchange": ["Xetra"]})

        def close(self):
            closed.append(1)

    def connect(path, read_only):
        assert read_only is True and path.endswith("market_data.duckdb")
        return Con()

    monkeypatch.setitem(sys.modules, "duckdb", types.SimpleNamespace(connect=connect))
    assert sm._europe_names() == [
        {"symbol": "SAP.DE", "name": "SAP", "exchange": "Xetra", "suffix": ""}
    ]
    assert closed == [1]


# ---------------------------------------------------------------- build_master / load_master
@pytest.fixture
def master(monkeypatch, tmp_path):
    path = tmp_path / "sm" / "symbol_master.parquet"
    path.parent.mkdir()
    monkeypatch.setattr(sm, "MASTER", path)
    monkeypatch.setattr(sm, "_LOOKUP", {})
    for fn in (
        "_nse_names",
        "_bse_names",
        "_us_names",
        "_japan_names",
        "_korea_names",
        "_china_names",
        "_hongkong_names",
        "_europe_names",
    ):
        monkeypatch.setattr(sm, fn, lambda *a, **k: [])
    return path


def test_build_master_empty_returns_empty_frame(master, capsys):
    df = sm.build_master()
    assert df.empty
    assert not master.exists()
    assert "No symbols fetched" in capsys.readouterr().out


def test_build_master_combines_dedups_and_names(master, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        sm,
        "_nse_names",
        lambda: [
            {
                "symbol": "ADANIENT",
                "name": "ADANI ENTERPRISES LIMITED",
                "exchange": "NSE",
                "suffix": ".NS",
            },
            {"symbol": "ADANIENT", "name": "DUP", "exchange": "NSE", "suffix": ".NS"},
        ],
    )

    def bse(nse_syms):
        seen["nse"] = nse_syms
        return [
            {"symbol": "500325", "name": "RELIANCE IND LTD", "exchange": "BSE", "suffix": ".BO"}
        ]

    monkeypatch.setattr(sm, "_bse_names", bse)
    monkeypatch.setattr(
        sm,
        "_us_names",
        lambda: [{"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "suffix": ""}],
    )
    df = sm.build_master()
    assert seen["nse"] == {"ADANIENT"}
    assert list(df.columns) == ["symbol", "name", "name_clean", "exchange", "suffix", "yf_symbol"]
    assert len(df) == 3  # duplicate (symbol, exchange) dropped
    r = df[df.symbol == "ADANIENT"].iloc[0]
    assert r["name"] == "ADANI ENTERPRISES LIMITED"  # first kept
    assert r["name_clean"] == "ADANI ENTERPRISES"
    assert r["yf_symbol"] == "ADANIENT.NS"
    assert df[df.symbol == "AAPL"].iloc[0]["yf_symbol"] == "AAPL"
    assert master.exists()
    assert len(pd.read_parquet(master)) == 3


def test_build_master_include_us_false_skips_us(master, monkeypatch):
    def boom():
        raise AssertionError("US must not be called")

    monkeypatch.setattr(sm, "_us_names", boom)
    monkeypatch.setattr(
        sm,
        "_japan_names",
        lambda: [{"symbol": "7203.T", "name": "Toyota Motor", "exchange": "TSE", "suffix": ""}],
    )
    df = sm.build_master(include_us=False)
    assert list(df.symbol) == ["7203.T"]


def test_same_symbol_two_exchanges_kept_in_build(master, monkeypatch):
    monkeypatch.setattr(
        sm,
        "_nse_names",
        lambda: [{"symbol": "X", "name": "X CORP", "exchange": "NSE", "suffix": ".NS"}],
    )
    monkeypatch.setattr(
        sm,
        "_us_names",
        lambda: [{"symbol": "X", "name": "X INC", "exchange": "NYSE", "suffix": ""}],
    )
    assert len(sm.build_master()) == 2


def _write_master(path, name="CO ONE"):
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "name": [name],
            "name_clean": ["ONE"],
            "exchange": ["NSE"],
            "suffix": [".NS"],
            "yf_symbol": ["AAA.NS"],
        }
    ).to_parquet(path, index=False)


def test_load_master_fresh_file_is_read_not_rebuilt(master, monkeypatch):
    _write_master(master)
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: pytest.fail("rebuilt"))
    assert list(sm.load_master()["symbol"]) == ["AAA"]


def test_stale_window_is_twenty_four_hours():
    assert sm.STALE_HOURS == 24


def test_load_master_stale_file_rebuilds(master, monkeypatch):
    _write_master(master)
    old = datetime.now().timestamp() - (sm.STALE_HOURS + 1) * 3600
    import os

    os.utime(master, (old, old))
    sentinel = pd.DataFrame({"symbol": ["REBUILT"]})
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: sentinel)
    assert sm.load_master() is sentinel


def test_load_master_stale_but_auto_refresh_off_uses_file(master, monkeypatch):
    _write_master(master)
    old = datetime.now().timestamp() - (sm.STALE_HOURS + 1) * 3600
    import os

    os.utime(master, (old, old))
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: pytest.fail("rebuilt"))
    assert list(sm.load_master(auto_refresh=False)["symbol"]) == ["AAA"]


def test_load_master_just_inside_stale_window_not_rebuilt(master, monkeypatch):
    _write_master(master)
    ts = datetime.now().timestamp() - (sm.STALE_HOURS - 1) * 3600
    import os

    os.utime(master, (ts, ts))
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: pytest.fail("rebuilt"))
    sm.load_master()


def test_load_master_missing_file_builds(master, monkeypatch):
    sentinel = pd.DataFrame({"symbol": ["B"]})
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: sentinel)
    assert sm.load_master() is sentinel


def test_load_master_corrupt_file_builds(master, monkeypatch):
    master.write_text("not parquet")
    sentinel = pd.DataFrame({"symbol": ["B"]})
    monkeypatch.setattr(sm, "build_master", lambda *a, **k: sentinel)
    assert sm.load_master() is sentinel


# ---------------------------------------------------------------- lookups
def test_name_lookups_first_duplicate_wins_and_unknown_blank(master, monkeypatch):
    m = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "name": ["First", "Second", "Bee"],
            "name_clean": ["FIRST", "SECOND", "BEE"],
            "exchange": ["NSE", "NYSE", "NSE"],
            "suffix": ["", "", ""],
            "yf_symbol": ["AAA", "AAA", "BBB"],
        }
    )
    monkeypatch.setattr(sm, "load_master", lambda *a, **k: m)
    assert sm.name_for("AAA") == "First"
    assert sm.clean_name_for("AAA") == "FIRST"
    assert sm.name_for("BBB") == "Bee"
    assert sm.name_for("NOPE") == ""
    assert sm.clean_name_for("NOPE") == ""


def test_lookup_is_built_once(master, monkeypatch):
    calls = []
    m = pd.DataFrame(
        {
            "symbol": ["A"],
            "name": ["N"],
            "name_clean": ["NN"],
            "exchange": ["NSE"],
            "suffix": [""],
            "yf_symbol": ["A"],
        }
    )
    monkeypatch.setattr(sm, "load_master", lambda *a, **k: calls.append(1) or m)
    sm.name_for("A")
    sm.name_for("A")
    sm.clean_name_for("A")
    assert calls == [1]


def test_lookup_empty_master_stays_empty_and_retries(master, monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "load_master", lambda *a, **k: calls.append(1) or pd.DataFrame())
    assert sm.name_for("A") == ""
    assert sm.name_for("A") == ""
    assert calls == [1, 1]  # an empty master is not cached


# ---------------------------------------------------------------- exchange builders (fake vendor libs)
def _csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


class _FakeNSE:
    files = {}
    asked = []

    def __init__(self, download_folder, server):
        assert server is False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def equityBhavcopy(self, d):
        _FakeNSE.asked.append(d)
        v = _FakeNSE.files.get(len(_FakeNSE.asked) - 1)
        if isinstance(v, Exception):
            raise v
        return v


def test_nse_names_walks_back_to_first_available_day(monkeypatch, tmp_path):
    _FakeNSE.asked = []
    good = _csv(
        tmp_path,
        "n.csv",
        pd.DataFrame(
            {
                "SctySrs": ["EQ", "BE", "EQ"],
                "TckrSymb": [" TCS ", "JUNK", "INFY"],
                "FinInstrmNm": ["TATA CONSULTANCY", "J", "INFOSYS LTD"],
            }
        ),
    )
    _FakeNSE.files = {0: None, 1: RuntimeError("x"), 2: good}
    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=_FakeNSE))
    rows = sm._nse_names()
    assert rows == [
        {"symbol": "TCS", "name": "TATA CONSULTANCY", "exchange": "NSE", "suffix": ".NS"},
        {"symbol": "INFY", "name": "INFOSYS LTD", "exchange": "NSE", "suffix": ".NS"},
    ]
    assert len(_FakeNSE.asked) == 3  # stops at the first good day
    assert _FakeNSE.asked[0] > _FakeNSE.asked[1] > _FakeNSE.asked[2]


def test_nse_names_gives_up_after_seven_days(monkeypatch):
    _FakeNSE.asked = []
    _FakeNSE.files = {}
    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=_FakeNSE))
    assert sm._nse_names() == []
    assert len(_FakeNSE.asked) == 7


def test_nse_names_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "nse", None)
    assert sm._nse_names() == []


class _FakeBSE:
    paths = []
    asked = 0

    def __init__(self, download_folder):
        pass

    def bhavcopyReport(self, d):
        _FakeBSE.asked += 1
        return _FakeBSE.paths.pop(0)


def test_bse_names_skips_dual_listed_and_non_equity_groups(monkeypatch, tmp_path):
    p = _csv(
        tmp_path,
        "b.csv",
        pd.DataFrame(
            {
                "SctySrs": ["A", "Z", "XT", "B"],
                "TckrSymb": ["500325", "999999", "500112", "RELIANCE"],
                "FinInstrmNm": ["RIL BSE", "ZGROUP", "SBI BSE", "DUAL"],
            }
        ),
    )
    _FakeBSE.paths = [p]
    _FakeBSE.asked = 0
    monkeypatch.setitem(sys.modules, "bse", types.SimpleNamespace(BSE=_FakeBSE))
    rows = sm._bse_names({"RELIANCE"})
    assert [r["symbol"] for r in rows] == ["500325", "500112"]
    assert all(r["exchange"] == "BSE" and r["suffix"] == ".BO" for r in rows)


def test_bse_names_no_group_column_keeps_all(monkeypatch, tmp_path):
    p = _csv(tmp_path, "b.csv", pd.DataFrame({"TckrSymb": ["1", "2"], "FinInstrmNm": ["a", "b"]}))
    _FakeBSE.paths = [None, p]
    _FakeBSE.asked = 0
    monkeypatch.setitem(sys.modules, "bse", types.SimpleNamespace(BSE=_FakeBSE))
    assert [r["symbol"] for r in sm._bse_names(set())] == ["1", "2"]
    assert _FakeBSE.asked == 2


def test_bse_names_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "bse", None)
    assert sm._bse_names(set()) == []


def test_japan_names_include_list_and_code_padding(monkeypatch):
    prime = "プライム（内国株式）"
    cells = [
        ["hdr"] * 4,
        ["d", 7203.0, "Toyota", prime],
        ["d", 33.0, "ShortCode", "グロース（内国株式）"],  # zero-padded to 0033
        ["d", 1306.0, "Topix ETF", "ETF・ETN"],  # not an include category
        ["d", "130A", "AlphaCode", "スタンダード（内国株式）"],  # non-digit code dropped
        ["d", 0, "ZeroCode", prime],  # falsy code skipped
        ["d", 9999.0, "", prime],  # empty name skipped
    ]

    class WS:
        nrows = len(cells)

        def cell_value(self, i, j):
            return cells[i][j]

    class WB:
        def sheet_by_index(self, i):
            return WS()

    xlrd = types.SimpleNamespace(open_workbook=lambda file_contents: WB())
    req = types.SimpleNamespace(
        get=lambda url, timeout: types.SimpleNamespace(ok=True, content=b"x")
    )
    monkeypatch.setitem(sys.modules, "xlrd", xlrd)
    monkeypatch.setitem(sys.modules, "requests", req)
    rows = sm._japan_names()
    assert rows == [
        {"symbol": "7203.T", "name": "Toyota", "exchange": "JAPAN", "suffix": ""},
        {"symbol": "0033.T", "name": "ShortCode", "exchange": "JAPAN", "suffix": ""},
    ]


def test_japan_names_http_not_ok(monkeypatch):
    req = types.SimpleNamespace(
        get=lambda url, timeout: types.SimpleNamespace(ok=False, content=b"")
    )
    monkeypatch.setitem(sys.modules, "xlrd", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "requests", req)
    assert sm._japan_names() == []


def test_japan_names_errors_swallowed(monkeypatch):
    monkeypatch.setitem(sys.modules, "xlrd", None)
    assert sm._japan_names() == []
    monkeypatch.setitem(sys.modules, "xlrd", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net"))),
    )
    assert sm._japan_names() == []


def test_korea_names_pads_codes_and_suffixes_by_board(monkeypatch):
    listings = {
        "KOSPI": pd.DataFrame(
            {"Code": ["5930", "000660", "000001"], "Name": ["Samsung", "Hynix", ""]}
        ),
        "KOSDAQ": pd.DataFrame({"Symbol": ["35720"], "Name": ["Kakao"]}),
    }
    fdr = types.SimpleNamespace(StockListing=lambda m: listings[m])
    monkeypatch.setitem(sys.modules, "FinanceDataReader", fdr)
    rows = sm._korea_names()
    assert [(r["symbol"], r["exchange"]) for r in rows] == [
        ("005930.KS", "KOSPI"),
        ("000660.KS", "KOSPI"),
        ("035720.KQ", "KOSDAQ"),
    ]
    assert all(r["suffix"] == "" for r in rows)


def test_korea_names_timeout_skips_that_board(monkeypatch, capsys):
    def listing(m):
        if m == "KOSPI":
            raise sm.SourceTimeout("slow")
        return pd.DataFrame({"Code": ["35720"], "Name": ["Kakao"]})

    monkeypatch.setitem(
        sys.modules, "FinanceDataReader", types.SimpleNamespace(StockListing=listing)
    )
    rows = sm._korea_names()
    assert [r["symbol"] for r in rows] == ["035720.KQ"]
    assert "skipping KOSPI" in capsys.readouterr().out


def test_korea_names_other_error_skips_silently_and_missing_lib(monkeypatch):
    def listing(m):
        raise ValueError("boom")

    monkeypatch.setitem(
        sys.modules, "FinanceDataReader", types.SimpleNamespace(StockListing=listing)
    )
    assert sm._korea_names() == []
    monkeypatch.setitem(sys.modules, "FinanceDataReader", None)
    assert sm._korea_names() == []


def test_china_names_prefix_routing_and_bjse_excluded(monkeypatch):
    df = pd.DataFrame(
        {
            "code": ["600519", "000001", "300750", "920000", "1", "ABCDEF", "688981"],
            "name": ["Moutai", "PingAn", "CATL", "BJ", "ShortCode", "Alpha", "SMIC"],
        }
    )
    monkeypatch.setitem(
        sys.modules, "akshare", types.SimpleNamespace(stock_info_a_code_name=lambda: df)
    )
    rows = sm._china_names()
    got = {r["symbol"]: r["exchange"] for r in rows}
    assert got == {"600519.SS": "SSE", "000001.SZ": "SZSE", "300750.SZ": "SZSE", "688981.SS": "SSE"}
    # "1" pads to 000001 -> duplicate of PingAn code but a separate row
    assert len(rows) == 5


def test_china_names_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", None)
    assert sm._china_names() == []

    def boom():
        raise RuntimeError("x")

    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(stock_info_a_code_name=boom))
    assert sm._china_names() == []


def test_hongkong_names_filters_and_4_digit_codes(monkeypatch):
    df = pd.DataFrame(
        {
            "Symbol": ["00700", "80700", "04332", "09988", "ABC", "", "00005", "04"],
            "Name": [
                "Tencent",
                "Tencent RMB",
                "MSFT HDR",
                "Alibaba",
                "Bad",
                "Empty",
                "HSBC",
                "Edge",
            ],
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "FinanceDataReader",
        types.SimpleNamespace(StockListing=lambda m: df if m == "HKEX" else None),
    )
    rows = sm._hongkong_names()
    assert [r["symbol"] for r in rows] == ["0700.HK", "9988.HK", "0005.HK"]
    assert all(r["exchange"] == "HKEX" and r["suffix"] == "" for r in rows)


def test_hongkong_names_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "FinanceDataReader", None)
    assert sm._hongkong_names() == []
