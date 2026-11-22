"""Offline tests for recurring_movers (ledger recurrence x price movement)."""

import sys
from datetime import date, timedelta

import pandas as pd
import pytest

import recurring_movers as rm

TODAY = date.today()


def _d(n):
    return TODAY - timedelta(days=n)


@pytest.fixture
def env(monkeypatch, tmp_path):
    (tmp_path / "cache_seed").mkdir()
    monkeypatch.setattr(rm, "HERE", tmp_path)
    monkeypatch.setattr(rm, "LEDGER", tmp_path / "cache_seed" / "signal_ledger.parquet")
    monkeypatch.setattr(rm, "WATCHLIST", tmp_path / "watchlist.csv")
    monkeypatch.setattr(rm, "OUT", tmp_path / "reports" / "recurring_movers.csv")
    return tmp_path


def _ledger(env, rows):
    cols = ["symbol", "market", "signal_date", "source", "filter", "detail", "price_at_signal"]
    df = pd.DataFrame(rows, columns=cols)
    df["signal_date"] = df["signal_date"].astype(str)
    df.to_parquet(rm.LEDGER, index=False)


def _row(sym, mkt, days_ago, price, source="mailer", filt="darvas", detail="d"):
    return (sym, mkt, _d(days_ago), source, filt, detail, price)


def _workbook(env, market, rows, sym_col="Symbol", px_col="LTP", extra=None):
    glob, _, _ = rm.WORKBOOKS[market]
    path = env / glob.replace("*", "20260101")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if extra:
        for k, v in extra.items():
            df[k] = v
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="All_Stocks", index=False)
    return path


# ---------------------------------------------------------------- _norm
@pytest.mark.parametrize(
    "sym,mkt,expected",
    [
        ("7203.T", "JP", "7203"),
        ("005930.KS", "KR", "5930"),
        ("035720.kq", "KR", "35720"),
        ("400", "KR", "400"),
        ("000400", "KR", "400"),
        ("0", "KR", "0"),
        ("000", "KR", "0"),
        ("  aapl ", "US", "AAPL"),
        ("ADS.DE", "EU", "ADS.DE"),
        ("007", "US", "007"),
        ("X.NS", "IN", "X.NS"),
    ],
)
def test_norm(sym, mkt, expected):
    assert rm._norm(sym, mkt) == expected


# ---------------------------------------------------------------- current_prices
def test_current_prices_uses_newest_workbook_and_normalises(env):
    old = _workbook(
        env,
        "KR",
        [{"YF_Ticker": "005930.KS", "LTP_KRW": 1.0, "Name": "old"}],
    )
    newest = env / "korea_scan" / "korea_market_scan_99999999.xlsx"
    with pd.ExcelWriter(newest) as w:
        pd.DataFrame(
            {
                "YF_Ticker": ["005930.KS", "000660.KS", "000660.KS", "BAD.KS"],
                "LTP_KRW": [70000, 150000, 151000, "n/a"],
                "Name": ["Samsung", "Hynix", "Hynix2", "Bad"],
                "Liquidity_Tier": ["T1", "T5_MOST_ILLIQUID", "T2", "T3"],
            }
        ).to_excel(w, sheet_name="All_Stocks", index=False)
    assert old.exists()
    px = rm.current_prices("KR")
    assert sorted(px["symbol_norm"]) == ["5930", "660"]  # non-numeric price dropped, dup collapsed
    row = px.set_index("symbol_norm")
    assert row.loc["5930", "price"] == 70000 and row.loc["5930", "tier"] == "T1"
    assert row.loc["660", "price"] == 150000  # first duplicate kept
    assert row.loc["5930", "name"] == "Samsung"


def test_current_prices_no_files_and_missing_optional_columns(env):
    empty = rm.current_prices("US")
    assert empty.empty and list(empty.columns) == ["symbol_norm", "price", "tier"]
    _workbook(env, "US", [{"Symbol": "aapl", "LTP": 10.0}])
    px = rm.current_prices("US")
    assert px["symbol_norm"].tolist() == ["AAPL"] and px["tier"].tolist() == [""]
    assert px["name"].tolist() == [""]


# ---------------------------------------------------------------- build
def test_build_requires_recurrence_and_movement(env):
    _ledger(
        env,
        [
            _row("UP", "US", 10, 100.0),
            _row("UP", "US", 3, 101.0),  # 2 dates, +10% -> keep
            _row("FLAT", "US", 10, 100.0),
            _row("FLAT", "US", 3, 100.0),  # not moving
            _row("ONCE", "US", 10, 100.0),  # one date
            _row("DOWN", "US", 10, 100.0),
            _row("DOWN", "US", 3, 100.0),  # negative move
        ],
    )
    _workbook(
        env,
        "US",
        [
            {"Symbol": "UP", "LTP": 110.0},
            {"Symbol": "FLAT", "LTP": 101.0},
            {"Symbol": "ONCE", "LTP": 200.0},
            {"Symbol": "DOWN", "LTP": 80.0},
        ],
    )
    df = rm.build(30, 2, 2.0)
    assert df["symbol"].tolist() == ["UP"]
    r = df.iloc[0]
    assert r["appearances"] == 2 and r["entry_price"] == 100.0 and r["price"] == 110.0
    assert r["move_pct"] == pytest.approx(10.0)
    assert r["first_signal"] == _d(10) and r["last_signal"] == _d(3)


def test_build_thresholds_are_inclusive(env):
    _ledger(env, [_row("EDGE", "US", 10, 100.0), _row("EDGE", "US", 3, 100.0)])
    _workbook(env, "US", [{"Symbol": "EDGE", "LTP": 102.0}])
    assert len(rm.build(30, 2, 2.0)) == 1  # move == min_move and dates == min_dates
    assert len(rm.build(30, 2, 2.0001)) == 0


def test_build_window_boundary(env):
    _ledger(env, [_row("A", "US", 30, 100.0), _row("A", "US", 31, 50.0), _row("A", "US", 5, 100.0)])
    _workbook(env, "US", [{"Symbol": "A", "LTP": 120.0}])
    df = rm.build(30, 2, 2.0)
    # the 31-day-old row is outside the window; the 30-day-old row is inside
    assert df.iloc[0]["first_signal"] == _d(30) and df.iloc[0]["entry_price"] == 100.0


def test_build_excludes_biased_source(env, capsys):
    _ledger(
        env,
        [
            _row("BIAS", "US", 10, 100.0, source="india_factor_panel"),
            _row("BIAS", "US", 3, 100.0, source="india_factor_panel"),
            _row("OK", "US", 10, 100.0),
            _row("OK", "US", 3, 100.0),
        ],
    )
    _workbook(env, "US", [{"Symbol": "BIAS", "LTP": 150.0}, {"Symbol": "OK", "LTP": 150.0}])
    df = rm.build(30, 2, 2.0)
    assert df["symbol"].tolist() == ["OK"]
    assert (
        "excluded 2 ledger rows from biased source(s): india_factor_panel"
        in capsys.readouterr().out
    )


def test_build_drops_illiquid_tier_and_missing_prices(env):
    _ledger(
        env,
        [
            _row("ILL", "US", 10, 100.0),
            _row("ILL", "US", 3, 100.0),
            _row("GONE", "US", 10, 100.0),
            _row("GONE", "US", 3, 100.0),
            _row("OK", "US", 10, 100.0),
            _row("OK", "US", 3, 100.0),
        ],
    )
    _workbook(
        env,
        "US",
        [{"Symbol": "ILL", "LTP": 150.0}, {"Symbol": "OK", "LTP": 150.0}],
        extra={"Liquidity_Tier": ["T5_MOST_ILLIQUID", "T2"]},
    )
    df = rm.build(30, 2, 2.0)
    assert df["symbol"].tolist() == ["OK"] and df["tier"].tolist() == ["T2"]


def test_build_ranks_by_appearances_then_move_and_merges_filters(env):
    _ledger(
        env,
        [
            _row("THREE", "US", 12, 100.0, filt="darvas"),
            _row("THREE", "US", 8, 100.0, filt="piotroski"),
            _row("THREE", "US", 2, 100.0, filt="darvas", detail="latest"),
            _row("BIG2", "US", 10, 100.0),
            _row("BIG2", "US", 3, 100.0),
            _row("SMALL2", "US", 10, 100.0),
            _row("SMALL2", "US", 3, 100.0),
        ],
    )
    _workbook(
        env,
        "US",
        [
            {"Symbol": "THREE", "LTP": 105.0},
            {"Symbol": "BIG2", "LTP": 150.0},
            {"Symbol": "SMALL2", "LTP": 110.0},
        ],
    )
    df = rm.build(30, 2, 2.0)
    assert df["symbol"].tolist() == ["THREE", "BIG2", "SMALL2"]
    three = df.iloc[0]
    assert three["filters"] == "darvas+piotroski" and three["best_detail"] == "latest"
    assert list(df.columns) == [
        "symbol",
        "name",
        "market",
        "appearances",
        "move_pct",
        "entry_price",
        "price",
        "first_signal",
        "last_signal",
        "filters",
        "best_detail",
        "tier",
    ]


def test_build_counts_distinct_dates_not_rows(env):
    _ledger(
        env,
        [
            _row("A", "US", 5, 100.0, filt="darvas"),
            _row("A", "US", 5, 100.0, filt="piotroski"),
            _row("B", "US", 5, 100.0),
            _row("B", "US", 4, 100.0),
        ],
    )
    _workbook(env, "US", [{"Symbol": "A", "LTP": 200.0}, {"Symbol": "B", "LTP": 200.0}])
    assert rm.build(30, 2, 2.0)["symbol"].tolist() == ["B"]  # A: two rows, one date


def test_build_matches_kr_codes_across_zero_padding(env):
    _ledger(env, [_row("400", "KR", 10, 100.0), _row("400", "KR", 3, 100.0)])
    _workbook(
        env,
        "KR",
        [{"YF_Ticker": "000400.KS", "LTP_KRW": 120.0}],
    )
    df = rm.build(30, 2, 2.0)
    assert df["move_pct"].tolist() == [pytest.approx(20.0)]


def test_build_market_without_workbook_is_skipped_when_others_have_one(env):
    _ledger(
        env,
        [
            _row("A", "US", 10, 100.0),
            _row("A", "US", 3, 100.0),
            _row("B", "IN", 10, 100.0),
            _row("B", "IN", 3, 100.0),
        ],
    )
    _workbook(env, "US", [{"Symbol": "A", "LTP": 120.0}])
    # IN has no workbook -> its rows have no current price -> dropped, US still reported
    assert rm.build(30, 2, 2.0)["symbol"].tolist() == ["A"]


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="source bug: when no symbol recurs (or the window is empty) build() falls back to "
    "`grouped.assign(price=None, tier='')`, which has no 'name' column, so df[cols] raises "
    "KeyError instead of returning an empty table (main()'s 'none qualify today' branch)",
)
def test_build_nothing_recurs_returns_empty_frame(env):
    _ledger(env, [_row("A", "US", 100, 100.0), _row("A", "US", 99, 100.0)])
    assert rm.build(30, 2, 2.0).empty


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="source bug: current_prices() returns an empty frame WITHOUT a 'name' column when "
    "no workbook exists, so a market with recurring signals but no scan workbook makes "
    "build() raise KeyError('name') instead of yielding no rows",
)
def test_build_only_market_without_workbook_returns_empty_frame(env):
    _ledger(env, [_row("A", "US", 10, 100.0), _row("A", "US", 3, 100.0)])
    assert rm.build(30, 2, 2.0).empty


# ---------------------------------------------------------------- sync_watchlist
def _qual(rows):
    cols = ["symbol", "market", "appearances", "move_pct", "first_signal", "entry_price"]
    return pd.DataFrame(rows, columns=cols)


def _wl(env, rows):
    pd.DataFrame(rows, columns=["symbol", "market", "status", "note"]).to_csv(
        rm.WATCHLIST, index=False
    )


def test_sync_adds_new_names_without_touching_existing(env):
    _wl(env, [["HELD", "US", "held", "keep me"]])
    df = _qual(
        [
            ["held", "us", 3, 5.0, _d(9), 10.0],  # already present (case-insens.)
            ["NEW", "US", 2, 7.123, _d(9), 12.34567],
        ]
    )
    assert rm.sync_watchlist(df, 25) == 1
    wl = pd.read_csv(rm.WATCHLIST)
    assert wl["symbol"].tolist() == ["HELD", "NEW"]
    assert wl.loc[0, "status"] == "held" and wl.loc[0, "note"] == "keep me"
    new = wl.iloc[1]
    assert new["status"] == "signal" and new["entry_price"] == pytest.approx(12.3457)
    assert new["note"] == f"recurring x2 +7.1% since {_d(9)}"
    assert new["entry_date"] == str(_d(9))


def test_sync_top_cap_applies_before_dedupe(env):
    _wl(env, [["A", "US", "held", ""]])
    df = _qual(
        [
            ["A", "US", 5, 9.0, _d(5), 1.0],
            ["B", "US", 4, 8.0, _d(5), 1.0],
            ["C", "US", 3, 7.0, _d(5), 1.0],
        ]
    )
    assert rm.sync_watchlist(df, 2) == 1  # head(2) = A (exists), B -> only B added
    assert pd.read_csv(rm.WATCHLIST)["symbol"].tolist() == ["A", "B"]


def test_sync_noop_cases(env):
    df = _qual([["NEW", "US", 2, 7.0, _d(5), 1.0]])
    assert rm.sync_watchlist(df, 25) == 0  # no watchlist file
    _wl(env, [["A", "US", "held", ""]])
    assert rm.sync_watchlist(_qual([]), 25) == 0
    assert rm.sync_watchlist(df, 0) == 0
    assert pd.read_csv(rm.WATCHLIST)["symbol"].tolist() == ["A"]


def test_sync_same_symbol_other_market_is_distinct_and_dedupes_within_run(env):
    _wl(env, [["A", "US", "held", ""]])
    df = _qual([["A", "IN", 2, 5.0, _d(5), 1.0], ["A", "IN", 2, 5.0, _d(5), 1.0]])
    assert rm.sync_watchlist(df, 25) == 1


def test_sync_missing_entry_price_is_blank(env):
    _wl(env, [["A", "US", "held", ""]])
    df = _qual([["NEW", "US", 2, 7.0, _d(5), float("nan")]])
    rm.sync_watchlist(df, 25)
    assert pd.isna(pd.read_csv(rm.WATCHLIST).iloc[1]["entry_price"])


# ---------------------------------------------------------------- main
def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["recurring_movers.py", *args])
    return rm.main()


def test_main_builds_csv_and_syncs(env, monkeypatch, capsys):
    _ledger(env, [_row("UP", "US", 10, 100.0), _row("UP", "US", 3, 100.0)])
    _workbook(env, "US", [{"Symbol": "UP", "LTP": 110.0}])
    _wl(env, [["OTHER", "US", "watch", ""]])
    assert _main(monkeypatch) == 0
    out = pd.read_csv(rm.OUT)
    assert out["symbol"].tolist() == ["UP"]
    text = capsys.readouterr().out
    assert (
        "RECURRING MOVERS — 1 names (>= 2 dates, >= +2.0% since first signal, window 30d)" in text
    )
    assert "+10.0%" in text and "watchlist.csv: +1 new `signal` names (top 25 only" in text
    assert pd.read_csv(rm.WATCHLIST)["symbol"].tolist() == ["OTHER", "UP"]


def test_main_no_sync_and_custom_thresholds_and_empty(env, monkeypatch, capsys):
    _ledger(env, [_row("UP", "US", 10, 100.0), _row("UP", "US", 3, 100.0)])
    _workbook(env, "US", [{"Symbol": "UP", "LTP": 110.0}])
    _wl(env, [["OTHER", "US", "watch", ""]])
    assert _main(monkeypatch, "--no-sync", "--min-move", "50") == 0
    text = capsys.readouterr().out
    assert "none qualify today" in text and "window 30d" in text and ">= +50.0%" in text
    assert "watchlist.csv" not in text.replace("-> ", "")
    assert pd.read_csv(rm.WATCHLIST)["symbol"].tolist() == ["OTHER"]
    assert pd.read_csv(rm.OUT).empty


def test_main_truncates_display_at_25(env, monkeypatch, capsys):
    syms = [f"S{i:02d}" for i in range(27)]
    rows = []
    for s in syms:
        rows += [_row(s, "US", 10, 100.0), _row(s, "US", 3, 100.0)]
    _ledger(env, rows)
    _workbook(env, "US", [{"Symbol": s, "LTP": 110.0} for s in syms])
    _main(monkeypatch, "--no-sync")
    text = capsys.readouterr().out
    assert "… 2 more in recurring_movers.csv" in text
    assert len(pd.read_csv(rm.OUT)) == 27
