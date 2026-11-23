"""Offline tests for us_fundamentals_edgar (companyfacts parsing, CIK map, store, main loop)."""

import datetime as dt
import json
import os
import sys
import types

import pandas as pd
import pytest

import us_fundamentals_edgar as ed


@pytest.fixture
def paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ed, "STORE", tmp_path / "fund" / "US_current.parquet")
    monkeypatch.setattr(ed, "CIK_CACHE", tmp_path / "fund" / "company_tickers.json")
    monkeypatch.setattr(ed, "UNIVERSE_PARQUET", tmp_path / "ohlcv_US.parquet")
    monkeypatch.setattr(ed.time, "sleep", lambda s: None)
    return tmp_path


def _e(start, end, val, form="10-K", filed="2025-02-01", **kw):
    d = {"end": end, "val": val, "form": form, "filed": filed}
    if start:
        d["start"] = start
    d.update(kw)
    return d


# ---------------------------------------------------------------- _hash_order
def test_hash_order_dedupes_and_is_deterministic_non_alphabetical():
    syms = [f"S{i:03d}" for i in range(50)] + ["S000"]
    a = ed._hash_order(syms)
    assert len(a) == 50 and a == ed._hash_order(reversed(syms))
    assert a != sorted(a) and set(a) == set(syms)


# ---------------------------------------------------------------- _annual_series
def test_annual_series_flow_span_window_inclusive():
    units = {
        "USD": [
            _e("2024-01-01", "2024-11-26", 1),  # 330 days -> kept
            _e("2023-01-01", "2023-11-25", 2),  # 328 days -> dropped
            _e("2022-01-01", "2023-01-16", 3),  # 380 days -> kept
            _e("2021-01-01", "2022-01-17", 4),
        ]
    }  # 381 days -> dropped
    assert ed._annual_series(units, instant=False) == {"2024-11-26": 1, "2023-01-16": 3}


def test_annual_series_latest_filing_wins_and_ordering_independent():
    rows = [
        _e("2023-01-01", "2023-12-31", 101, filed="2025-02-01"),
        _e("2023-01-01", "2023-12-31", 100, filed="2024-02-01"),
    ]
    assert ed._annual_series({"USD": rows}, False) == {"2023-12-31": 101}
    assert ed._annual_series({"USD": rows[::-1]}, False) == {"2023-12-31": 101}


def test_annual_series_forms_units_and_missing_fields():
    units = {
        "USD": [
            _e(None, "2023-12-31", 1, form="10-K"),
            _e(None, "2022-12-31", 2, form="20-F"),
            _e(None, "2021-12-31", 3, form="40-F/A"),
            _e(None, "2020-12-31", 4, form="10-Q"),
            _e(None, "2019-12-31", 5, form="8-K"),
            _e(None, "", 6),
            {"form": "10-K", "val": 7},
        ],
        "EUR": [_e(None, "2018-12-31", 8)],
        "shares": [_e(None, "2017-12-31", 9)],
    }
    assert ed._annual_series(units, instant=True) == {
        "2017-12-31": 9,
        "2021-12-31": 3,
        "2022-12-31": 2,
        "2023-12-31": 1,
    }
    assert (
        ed._annual_series({"USD": [_e(None, "2023-12-31", 1)]}, instant=False) == {}
    )  # flow needs a start


# ---------------------------------------------------------------- parse_companyfacts
def _doc():
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _e("2023-01-01", "2023-12-31", 100, filed="2024-02-01"),
                            _e("2023-01-01", "2023-12-31", 101, filed="2025-02-01"),
                            _e("2024-01-01", "2024-12-31", 200),
                            _e("2024-10-01", "2024-12-31", 999),
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            _e(None, "2023-12-31", 1000, filed="2024-02-01"),
                            _e(None, "2024-12-31", 1100),
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {"units": {"USD": [_e(None, "2999-12-31", 555)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_e("2024-01-01", "2024-12-31", 300)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_e("2024-01-01", "2024-12-31", 50)]}
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2025-01-15", "val": 10},
                            {"end": "2024-01-15", "val": 8},
                        ]
                    }
                }
            },
        }
    }


def test_parse_companyfacts_rows_values_and_derived_fields():
    rows = ed.parse_companyfacts(_doc())
    by = {r["fy_end"]: r for r in rows}
    assert sorted(by) == [
        "2023-12-31",
        "2024-12-31",
    ]  # future-dated debt maturity never mints a year
    assert by["2023-12-31"]["net_income"] == 101  # restated comparative
    assert by["2024-12-31"]["net_income"] == 200  # quarterly 999 excluded by span
    r = by["2024-12-31"]
    assert r["capex"] == -50 and r["free_cash_flow"] == 250 and r["cfo"] == 300
    assert r["total_assets"] == 1100 and r["source"] == "edgar"
    assert by["2023-12-31"]["capex"] is None and by["2023-12-31"]["free_cash_flow"] is None
    assert set(ed.GAAP_MAP) <= set(r) and "free_cash_flow" in r and "shares" in r


def test_parse_capex_stored_negative_regardless_of_sign():
    d = _doc()
    d["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"] = [
        _e("2024-01-01", "2024-12-31", -50)
    ]
    assert {r["fy_end"]: r for r in ed.parse_companyfacts(d)}["2024-12-31"]["capex"] == -50


def test_parse_shares_taken_from_nearest_dei_date():
    by = {r["fy_end"]: r for r in ed.parse_companyfacts(_doc())}
    assert by["2024-12-31"]["shares"] == 10  # 2025-01-15 is nearest
    assert by["2023-12-31"]["shares"] == 8  # 2024-01-15 is nearest
    d = _doc()
    del d["facts"]["dei"]
    assert all(r["shares"] is None for r in ed.parse_companyfacts(d))


def test_parse_tag_priority_first_present_with_data_wins():
    d = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [_e("2024-01-01", "2024-12-31", 1)]}},
                "ProfitLoss": {"units": {"USD": [_e("2024-01-01", "2024-12-31", 2)]}},
                "Revenues": {
                    "units": {"USD": []}
                },  # present but empty -> falls through to next tag
                "SalesRevenueNet": {"units": {"USD": [_e("2024-01-01", "2024-12-31", 9)]}},
            }
        }
    }
    r = ed.parse_companyfacts(d)[0]
    assert r["net_income"] == 1 and r["revenue"] == 9


def test_parse_keeps_last_five_years_only_and_handles_empty():
    entries = [_e(f"{y}-01-01", f"{y}-12-31", y) for y in range(2015, 2025)]
    rows = ed.parse_companyfacts(
        {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": entries}}}}}
    )
    assert [r["fy_end"][:4] for r in rows] == ["2020", "2021", "2022", "2023", "2024"]
    assert ed.parse_companyfacts({}) == [] and ed.parse_companyfacts({"facts": {}}) == []


def test_parse_future_year_end_excluded_even_for_flows():
    nxt = str(dt.date.today().year + 1)
    d = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _e(f"{nxt}-01-01", f"{nxt}-12-31", 1),
                            _e("2023-01-01", "2023-12-31", 2),
                        ]
                    }
                }
            }
        }
    }
    assert [r["fy_end"] for r in ed.parse_companyfacts(d)] == ["2023-12-31"]


def test_self_test_passes(capsys):
    assert ed.self_test() == 0 and "self-test OK" in capsys.readouterr().out


# ---------------------------------------------------------------- load_cik_map / fetch_companyfacts
class _Resp:
    def __init__(self, payload=None, status=200, text=None):
        self._p, self.status_code = payload, status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_RAW = {"0": {"ticker": "aapl", "cik_str": 320193}, "1": {"ticker": "MSFT", "cik_str": "789019"}}


def test_load_cik_map_downloads_caches_and_uppercases(paths, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ed.requests,
        "get",
        lambda url, headers, timeout: calls.append((url, headers, timeout)) or _Resp(_RAW),
    )
    assert ed.load_cik_map() == {"AAPL": 320193, "MSFT": 789019}
    assert ed.CIK_CACHE.exists() and calls[0][2] == 30 and "@" in calls[0][1]["User-Agent"]
    monkeypatch.setattr(ed.requests, "get", lambda *a, **k: pytest.fail("cache should be used"))
    assert ed.load_cik_map() == {"AAPL": 320193, "MSFT": 789019}


def test_load_cik_map_refetches_when_cache_is_old(paths, monkeypatch):
    ed.CIK_CACHE.parent.mkdir(parents=True)
    ed.CIK_CACHE.write_text(json.dumps({"0": {"ticker": "OLD", "cik_str": 1}}))
    old = ed.time.time() - 8 * 86400
    os.utime(ed.CIK_CACHE, (old, old))
    monkeypatch.setattr(ed.requests, "get", lambda *a, **k: _Resp(_RAW))
    assert "OLD" not in ed.load_cik_map()
    young = ed.time.time() - 6 * 86400
    os.utime(ed.CIK_CACHE, (young, young))
    monkeypatch.setattr(ed.requests, "get", lambda *a, **k: pytest.fail("fresh enough"))
    assert "AAPL" in ed.load_cik_map()


def test_load_cik_map_http_error(paths, monkeypatch):
    monkeypatch.setattr(ed.requests, "get", lambda *a, **k: _Resp(status=500))
    with pytest.raises(RuntimeError):
        ed.load_cik_map()


def test_fetch_companyfacts_url_404_and_errors():
    seen = []

    class S:
        def __init__(self, resp):
            self.resp = resp

        def get(self, url, headers, timeout):
            seen.append(url)
            return self.resp

    assert ed.fetch_companyfacts(320193, S(_Resp({"x": 1}))) == {"x": 1}
    assert seen == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]
    assert ed.fetch_companyfacts(1, S(_Resp(status=404))) is None
    with pytest.raises(RuntimeError):
        ed.fetch_companyfacts(1, S(_Resp(status=503)))


# ---------------------------------------------------------------- store
def _row(t, **kw):
    return {"ticker": t, "fy_end": "2024-12-31", "cfo": 1.0, "total_assets": 2.0, **kw}


def test_load_store_missing(paths):
    assert ed.load_store().empty


def test_save_store_merges_replaces_case_insensitively_and_backs_up(paths):
    s = ed.save_store(pd.DataFrame(), [_row("AAPL"), _row("MSFT")], "t1")
    assert (
        len(s) == 2
        and (s["collected_at"] == "t1").all()
        and not ed.STORE.with_suffix(".parquet.bak").exists()
    )
    s2 = ed.save_store(s, [_row("aapl", cfo=9.0)], "t2")
    assert (
        sorted(s2["ticker"]) == ["MSFT", "aapl"] and s2.set_index("ticker").at["aapl", "cfo"] == 9.0
    )
    assert (
        ed.STORE.with_suffix(".parquet.bak").exists()
        and not ed.STORE.with_suffix(".parquet.tmp").exists()
    )
    assert len(ed.load_store()) == 2


def test_save_store_no_rows_is_noop(paths):
    store = pd.DataFrame({"ticker": ["A"]})
    assert ed.save_store(store, [], "t") is store and not ed.STORE.exists()


# ---------------------------------------------------------------- universe / coverage
def test_universe_fallback_reads_unique_sorted_symbols(paths):
    pd.DataFrame({"Symbol": ["B", "A", "B", None]}).to_parquet(ed.UNIVERSE_PARQUET)
    assert ed._universe_fallback() == ["A", "B"]


def test_universe_swallows_errors(paths, monkeypatch):
    fake = types.SimpleNamespace(connect=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setitem(sys.modules, "duckdb", fake)
    assert ed.universe() == []


def test_universe_reads_symbols_via_duckdb(paths, monkeypatch):
    class Con:
        def execute(self, q):
            assert "read_parquet" in q
            return types.SimpleNamespace(fetchall=lambda: [("A",), ("",), (None,), ("B",)])

    monkeypatch.setitem(sys.modules, "duckdb", types.SimpleNamespace(connect=lambda: Con()))
    assert ed.universe() == ["A", "B"]


def test_coverage_report(capsys):
    ed.coverage(pd.DataFrame())
    assert "store empty" in capsys.readouterr().out
    store = pd.DataFrame(
        {
            "ticker": ["AAPL", "aapl", "AMZN", "MSFT", "ZZZ"],
            "cfo": [1, 1, 1, 1, None],
            "total_assets": [1, 1, 1, 1, 1],
        }
    )
    ed.coverage(store)
    out = capsys.readouterr().out
    assert "usable tickers (cfo+total_assets): 3" in out  # AAPL, AMZN, MSFT; ZZZ lacks cfo
    assert "A-share: 66.7%" in out and "{'A': 2, 'M': 1}" in out


# ---------------------------------------------------------------- main
def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["us_fundamentals_edgar.py", *args])
    return ed.main()


def test_main_self_test_flag(monkeypatch, capsys):
    assert _main(monkeypatch, "--self-test") == 0


def test_main_universe_failure_returns_one(paths, monkeypatch, capsys):
    monkeypatch.setattr(ed, "universe", lambda: [])
    assert _main(monkeypatch) == 1  # fallback parquet missing -> error path
    assert "cannot load universe" in capsys.readouterr().out


def test_main_collects_skips_fresh_and_no_cik_and_limits(paths, monkeypatch, capsys):
    monkeypatch.setattr(ed, "universe", lambda: ["AAA", "BBB", "CCC", "NOCIK"])
    monkeypatch.setattr(ed, "load_cik_map", lambda: {"AAA": 1, "BBB": 2, "CCC": 3})
    fresh = pd.DataFrame([_row("BBB", collected_at=dt.datetime.now(dt.timezone.utc).isoformat())])
    ed.STORE.parent.mkdir(parents=True)
    fresh.to_parquet(ed.STORE)
    fetched = []

    def fetch(cik, sess):
        fetched.append(cik)
        if cik == 3:
            raise RuntimeError("boom")
        return _doc()

    monkeypatch.setattr(ed, "fetch_companyfacts", fetch)
    assert _main(monkeypatch) == 0
    assert sorted(fetched) == [1, 3]  # BBB fresh, NOCIK has no CIK
    out = capsys.readouterr().out
    assert "universe 4 | no CIK 1 | fresh-skip 1 | fetching 2" in out
    assert "! CCC: boom" in out and "done: 1 fetched, 1 failed" in out
    store = ed.load_store()
    assert (
        set(store["ticker"]) == {"BBB", "AAA"}
        and (store[store.ticker == "AAA"]["net_income"].dropna() > 0).all()
    )


def test_main_limit_stale_rows_and_checkpoint(paths, monkeypatch, capsys):
    monkeypatch.setattr(ed, "universe", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(ed, "load_cik_map", lambda: {"AAA": 1, "BBB": 2})
    monkeypatch.setattr(ed, "CHECKPOINT_EVERY", 1)
    stale = pd.DataFrame([_row("AAA", collected_at="2000-01-01T00:00:00+00:00")])
    ed.STORE.parent.mkdir(parents=True)
    stale.to_parquet(ed.STORE)
    fetched = []
    monkeypatch.setattr(ed, "fetch_companyfacts", lambda cik, s: fetched.append(cik) or _doc())
    assert _main(monkeypatch, "--limit", "1") == 0
    assert len(fetched) == 1  # --limit caps the run; stale AAA is eligible again
    assert "1/1 checkpointed" in capsys.readouterr().out


def test_main_interrupt_checkpoints_and_stops(paths, monkeypatch, capsys):
    monkeypatch.setattr(ed, "universe", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(ed, "load_cik_map", lambda: {"AAA": 1, "BBB": 2})
    n = []

    def fetch(cik, s):
        n.append(cik)
        if len(n) == 2:
            raise KeyboardInterrupt
        return _doc()

    monkeypatch.setattr(ed, "fetch_companyfacts", fetch)
    assert _main(monkeypatch) == 0
    assert "interrupted — checkpointing" in capsys.readouterr().out and len(n) == 2
    assert len(ed.load_store()) > 0  # the first ticker's rows were saved


def test_parse_includes_a_fiscal_year_ending_today():
    today = dt.date.today()
    start = today - dt.timedelta(days=364)
    d = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [_e(str(start), str(today), 5)]}}}}
    }
    assert [r["fy_end"] for r in ed.parse_companyfacts(d)] == [str(today)]


def test_main_does_not_checkpoint_before_fifty_tickers(paths, monkeypatch, capsys):
    syms = [f"S{i:02d}" for i in range(26)]
    monkeypatch.setattr(ed, "universe", lambda: syms)
    monkeypatch.setattr(ed, "load_cik_map", lambda: {s: i + 1 for i, s in enumerate(syms)})
    monkeypatch.setattr(ed, "fetch_companyfacts", lambda cik, s: None)
    assert ed.CHECKPOINT_EVERY == 50
    assert _main(monkeypatch) == 0
    assert "checkpointed" not in capsys.readouterr().out
