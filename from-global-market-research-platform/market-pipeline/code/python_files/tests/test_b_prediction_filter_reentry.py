"""Offline tests for prediction_filter (RSI bands, purge flow) and reentry_engine."""

import json
import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import prediction_filter as pf
import reentry_engine as re_


# ================================================================ rsi14 / rsi_zone
def test_rsi14_short_history_is_none():
    assert pf.rsi14(pd.Series(range(1, 20))) is None  # 19 points < 20
    assert pf.rsi14(pd.Series([1.0] * 5 + [None] * 30)) is None


def test_rsi14_all_losses_is_zero_and_flat_is_none():
    down = pd.Series(np.linspace(100, 50, 40))
    assert pf.rsi14(down) == 0.0
    assert pf.rsi14(pd.Series([10.0] * 40)) is None  # 0/0


def test_rsi14_all_gains_is_undefined_none():
    # dn == 0 is replaced by NA -> rs NA -> no value (documented current behaviour)
    assert pf.rsi14(pd.Series(np.linspace(50, 100, 40))) is None


def test_rsi14_mixed_in_range_rounded_and_coerces_strings():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
    v = pf.rsi14(close)
    assert 0 < v < 100 and v == round(v, 1)
    assert pf.rsi14(close.astype(str)) == v
    junk = close.astype(object).copy()
    junk.iloc[3] = "bad"
    assert pf.rsi14(junk) is not None


def test_rsi14_direction_sensitive():
    base = np.concatenate([np.linspace(100, 120, 30), np.linspace(120, 118, 5)])
    up_bias = pf.rsi14(pd.Series(base))
    down_bias = pf.rsi14(pd.Series(base[::-1]))
    assert up_bias > 50 > down_bias


@pytest.mark.parametrize(
    "mkt,r,zone",
    [
        ("IN", 39.9, "SELL"),
        ("IN", 40, "HOLD"),
        ("IN", 49.9, "HOLD"),
        ("IN", 50, "BUY"),
        ("IN", 75, "BUY"),
        ("IN", 75.1, "HOLD"),
        ("IN", 80, "HOLD"),
        ("IN", 80.1, "SELL"),
        ("US", 35, "BUY"),
        ("US", 35.1, "HOLD"),
        ("US", 69.9, "HOLD"),
        ("US", 70, "SELL"),
        ("JP", 20, "BUY"),
        ("KR", 80, "SELL"),
        ("EU", 50, "HOLD"),
        ("XX", 30, "BUY"),
        ("XX", 90, "SELL"),  # unknown market -> US band
        ("US", None, "?"),
    ],
)
def test_rsi_zone(mkt, r, zone):
    assert pf.rsi_zone(mkt, r) == zone


# ================================================================ annotate_rsi
def _scores(rows):
    return pd.DataFrame(rows, columns=["market", "symbol", "verdict"])


def _fake_digest(monkeypatch, loader):
    monkeypatch.setitem(sys.modules, "watchlist_digest", types.SimpleNamespace(_load_ohlc=loader))


def _closes(kind):
    if kind == "hot":
        return pd.Series(list(np.linspace(100, 90, 30)) + list(np.linspace(90, 130, 14)))
    return pd.Series(np.linspace(100, 50, 40))


def test_annotate_rsi_downgrades_enter_ok_only_in_sell_zone(monkeypatch, capsys):
    frames = {
        "HOT": pd.DataFrame(
            {
                "Close": [
                    100,
                    101,
                    103,
                    102,
                    104,
                    106,
                    105,
                    108,
                    107,
                    109,
                    111,
                    110,
                    112,
                    114,
                    113,
                    115,
                    117,
                    116,
                    118,
                    120,
                    119,
                    121,
                    123,
                    122,
                    124,
                    126,
                    125,
                    127,
                    129,
                    128,
                ]
            }
        ),
        "COLD": pd.DataFrame({"Close": _closes("cold")}),
    }
    _fake_digest(monkeypatch, lambda sym, mkt: frames.get(sym))
    s = _scores(
        [
            ("US", "HOT", "ENTER-OK"),
            ("US", "COLD", "ENTER-OK"),
            ("US", "HOT", "WEAK"),
            ("US", "NONE", "ENTER-OK"),
        ]
    )
    # force deterministic zones regardless of data shape
    monkeypatch.setattr(pf, "rsi14", lambda c: 90.0 if c.iloc[-1] > c.iloc[0] else 10.0)
    out = pf.annotate_rsi(s)
    assert out["rsi_zone"].tolist() == ["SELL", "BUY", "SELL", "?"]
    assert out["verdict"].tolist() == ["ENTER-WAIT (RSI)", "ENTER-OK", "WEAK", "ENTER-OK"]
    text = capsys.readouterr().out
    assert "RSI layer: 3/4 priced" in text and "1 ENTER-OK downgraded" in text


def test_annotate_rsi_loader_errors_and_missing_close(monkeypatch):
    def loader(sym, mkt):
        if sym == "ERR":
            raise RuntimeError("no data")
        return pd.DataFrame({"Open": [1, 2, 3]})  # no Close column

    _fake_digest(monkeypatch, loader)
    s = _scores([("US", "ERR", "ENTER-OK"), ("US", "NOCLOSE", "ENTER-OK")])
    out = pf.annotate_rsi(s)
    assert out["rsi"].isna().all() and out["rsi_zone"].tolist() == ["?", "?"]
    assert out["verdict"].tolist() == ["ENTER-OK", "ENTER-OK"]


def test_annotate_rsi_without_digest_module(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "watchlist_digest", None)
    out = pf.annotate_rsi(_scores([("US", "A", "ENTER-OK")]))
    assert out["rsi_zone"].tolist() == ["?"] and out["rsi"].isna().all()
    assert "RSI layer skipped" in capsys.readouterr().out


# ================================================================ refresh_scores
def test_refresh_scores_writes_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "SCORES", tmp_path / "scores.csv")
    mod = types.SimpleNamespace(part_b=lambda: (None, pd.DataFrame({"symbol": ["A", "B"]})))
    monkeypatch.setitem(sys.modules, "mailer_prediction_audit", mod)
    assert pf.refresh_scores() is True
    assert pd.read_csv(tmp_path / "scores.csv")["symbol"].tolist() == ["A", "B"]


def test_refresh_scores_failure_depends_on_existing_csv(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pf, "SCORES", tmp_path / "scores.csv")

    def boom():
        raise RuntimeError("x" * 200)

    monkeypatch.setitem(sys.modules, "mailer_prediction_audit", types.SimpleNamespace(part_b=boom))
    assert pf.refresh_scores() is False
    (tmp_path / "scores.csv").write_text("a\n1\n")
    assert pf.refresh_scores() is True
    assert "score refresh failed" in capsys.readouterr().out


# ================================================================ prediction_filter.main
TODAY = date.today()


def _score_row(market, symbol, status, verdict, age_days, **kw):
    return {
        "market": market,
        "symbol": symbol,
        "status": status,
        "verdict": verdict,
        "asof": (TODAY - timedelta(days=age_days)).isoformat(),
        "stock_state": kw.get("state", "bear"),
        "kalman_drift_ann_pct": kw.get("drift", -12.5),
        "markov_e21d_pct": kw.get("mk", -3.1),
    }


@pytest.fixture
def env(monkeypatch, tmp_path):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(pf, "SCORES", tmp_path / "reports" / "scores.csv")
    monkeypatch.setattr(pf, "SELL_REVIEW", tmp_path / "reports" / "sell.csv")
    monkeypatch.setattr(pf, "WATCHLIST", tmp_path / "watchlist.csv")
    monkeypatch.setattr(pf, "PURGED", tmp_path / "purged.csv")
    monkeypatch.setattr(pf, "annotate_rsi", lambda s: s)  # RSI layer tested separately
    return tmp_path


def _write(env, rows, wl=None):
    pd.DataFrame(rows).to_csv(env / "reports" / "scores.csv", index=False)
    wl = (
        wl
        if wl is not None
        else [
            {"market": r["market"], "symbol": r["symbol"], "status": r["status"], "note": ""}
            for r in rows
        ]
    )
    pd.DataFrame(wl).to_csv(env / "watchlist.csv", index=False)


def _argv(monkeypatch, *a):
    monkeypatch.setattr(sys, "argv", ["prediction_filter.py", "--no-refresh", *a])


def test_main_no_weak_names(env, monkeypatch, capsys):
    _write(env, [_score_row("US", "AAA", "watch", "ENTER-OK", 1)])
    _argv(monkeypatch)
    assert pf.main() == 0
    assert "no WEAK names" in capsys.readouterr().out
    assert not (env / "purged.csv").exists()


def test_main_purges_non_held_archives_and_protects_held(env, monkeypatch, capsys):
    rows = [
        _score_row("US", "BAD", "watch", "WEAK (bear)", 1),
        _score_row("US", "HELD", "held", "WEAK (bear)", 1),
        _score_row("IN", "OK", "watch", "ENTER-OK", 1),
    ]
    _write(env, rows)
    _argv(monkeypatch)
    assert pf.main() == 0
    wl = pd.read_csv(env / "watchlist.csv")
    assert sorted(wl["symbol"]) == ["HELD", "OK"]
    purged = pd.read_csv(env / "purged.csv")
    assert purged["symbol"].tolist() == ["BAD"] and purged["status"].tolist() == ["evicted"]
    assert "prediction-filter" in purged["note"].iloc[0]
    sell = pd.read_csv(env / "reports" / "sell.csv")
    assert sell["symbol"].tolist() == ["HELD"]
    assert list(sell.columns) == [
        "market",
        "symbol",
        "status",
        "stock_state",
        "kalman_drift_ann_pct",
        "markov_e21d_pct",
        "verdict",
    ]
    out = capsys.readouterr().out
    assert "purging 1 non-held WEAK name(s)" in out and "watchlist now 2 rows" in out


def test_main_archive_is_append_only_with_single_header(env, monkeypatch):
    _write(env, [_score_row("US", "BAD1", "watch", "WEAK", 1)])
    _argv(monkeypatch)
    pf.main()
    _write(env, [_score_row("US", "BAD2", "watch", "WEAK", 1)])
    pf.main()
    text = (env / "purged.csv").read_text().splitlines()
    assert sum(1 for ln in text if ln.startswith("market,")) == 1
    assert pd.read_csv(env / "purged.csv")["symbol"].tolist() == ["BAD1", "BAD2"]


def test_main_dry_run_touches_nothing(env, monkeypatch, capsys):
    _write(env, [_score_row("US", "BAD", "watch", "WEAK", 1)])
    before = (env / "watchlist.csv").read_text()
    _argv(monkeypatch, "--dry-run")
    assert pf.main() == 0
    assert (env / "watchlist.csv").read_text() == before
    assert not (env / "purged.csv").exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_main_stale_panel_boundary(env, monkeypatch, capsys):
    rows = [
        _score_row("US", "FRESH7", "watch", "WEAK", 7),
        _score_row("IN", "STALE8", "watch", "WEAK", 8),
    ]
    _write(env, rows)
    _argv(monkeypatch)
    pf.main()
    assert pd.read_csv(env / "purged.csv")["symbol"].tolist() == ["FRESH7"]
    assert "1 WEAK skipped" in capsys.readouterr().out
    assert "STALE8" in pd.read_csv(env / "watchlist.csv")["symbol"].tolist()


def test_main_max_age_override(env, monkeypatch):
    _write(env, [_score_row("IN", "OLD", "watch", "WEAK", 30)])
    _argv(monkeypatch, "--max-age", "60")
    pf.main()
    assert pd.read_csv(env / "purged.csv")["symbol"].tolist() == ["OLD"]


def test_main_only_held_weak_writes_review_but_no_purge(env, monkeypatch, capsys):
    _write(env, [_score_row("US", "HELD", "held", "WEAK", 1)])
    _argv(monkeypatch)
    assert pf.main() == 0
    assert (env / "reports" / "sell.csv").exists() and not (env / "purged.csv").exists()
    assert "no purgeable" in capsys.readouterr().out


def test_main_weak_not_in_watchlist_is_not_archived(env, monkeypatch, capsys):
    rows = [_score_row("US", "GHOST", "watch", "WEAK", 1)]
    _write(env, rows, wl=[{"market": "US", "symbol": "OTHER", "status": "watch", "note": ""}])
    _argv(monkeypatch)
    assert pf.main() == 0
    assert "purging 0 non-held" in capsys.readouterr().out
    assert pd.read_csv(env / "watchlist.csv")["symbol"].tolist() == ["OTHER"]


def test_main_key_is_market_and_symbol(env, monkeypatch):
    rows = [_score_row("US", "DUP", "watch", "WEAK", 1)]
    wl = [
        {"market": "US", "symbol": "DUP", "status": "watch", "note": ""},
        {"market": "IN", "symbol": "DUP", "status": "watch", "note": ""},
    ]
    _write(env, rows, wl=wl)
    _argv(monkeypatch)
    pf.main()
    assert pd.read_csv(env / "watchlist.csv")["market"].tolist() == ["IN"]


def test_main_refresh_failure_returns_one(env, monkeypatch):
    monkeypatch.setattr(pf, "refresh_scores", lambda: False)
    monkeypatch.setattr(sys, "argv", ["prediction_filter.py"])
    assert pf.main() == 1


def test_main_refresh_called_unless_no_refresh(env, monkeypatch):
    calls = []
    monkeypatch.setattr(pf, "refresh_scores", lambda: calls.append(1) or True)
    _write(env, [_score_row("US", "A", "watch", "ENTER-OK", 1)])
    monkeypatch.setattr(sys, "argv", ["prediction_filter.py"])
    pf.main()
    assert calls == [1]


# ================================================================ reentry_engine
def test_market_regime_reads_json_or_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(re_, "HERE", tmp_path)
    assert re_.market_regime() == {}
    (tmp_path / "cache_seed").mkdir()
    (tmp_path / "cache_seed" / "zone_regime.json").write_text(
        json.dumps({"US": {"current_regime": "bear"}})
    )
    assert re_.market_regime() == {"US": {"current_regime": "bear"}}
    (tmp_path / "cache_seed" / "zone_regime.json").write_text("{bad")
    assert re_.market_regime() == {}


@pytest.fixture
def renv(monkeypatch, tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "cache_seed").mkdir()
    monkeypatch.setattr(re_, "HERE", tmp_path)
    for name, rel in [
        ("W3", "watchlist3.csv"),
        ("WL", "watchlist.csv"),
        ("OUT", "reports/reentry_candidates.csv"),
        ("LOG", "reports/watchlist_tiers_log.csv"),
        ("RECENT", "reports/reentry_recent.csv"),
        ("BACKTEST_EVENTS", "reports/tier_anomaly_events.csv"),
    ]:
        monkeypatch.setattr(re_, name, tmp_path / rel)
    return tmp_path


def _w3(rows):
    pd.DataFrame(rows).to_csv(re_.W3, index=False)


def _w3row(sym, mkt="US", promoted_days=1, enrolled_days=10, ret=5.0, trig="bounce"):
    return {
        "market": mkt,
        "symbol": sym,
        "trigger": trig,
        "ret_pct": ret,
        "promoted": (TODAY - timedelta(days=promoted_days)).isoformat(),
        "enrolled": (TODAY - timedelta(days=enrolled_days)).isoformat(),
    }


def _prices(closes):
    return pd.DataFrame({"Close": closes})


def _digest(monkeypatch, mapping):
    def loader(sym, mkt):
        v = mapping[sym]
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setitem(sys.modules, "watchlist_digest", types.SimpleNamespace(_load_ohlc=loader))


def _rsi(monkeypatch, value=50.0):
    monkeypatch.setattr(pf, "rsi14", lambda c: value)


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["reentry_engine.py", *args])
    return re_.main()


def _wl(rows=None):
    pd.DataFrame(rows or [{"symbol": "ZZZ", "market": "US", "status": "watch"}]).to_csv(
        re_.WL, index=False
    )


def test_main_no_watchlist3(renv, monkeypatch, capsys):
    assert _run(monkeypatch) == 0
    assert "no watchlist3.csv" in capsys.readouterr().out


def test_main_empty_watchlist3(renv, monkeypatch, capsys):
    pd.DataFrame(columns=["market", "symbol", "promoted"]).to_csv(re_.W3, index=False)
    assert _run(monkeypatch) == 0
    assert "tier 3 empty" in capsys.readouterr().out


def test_main_ages_out_stale_rows_and_reports_no_fresh(renv, monkeypatch, capsys):
    _w3(
        [
            _w3row("OLD", promoted_days=31),
            _w3row("MID", promoted_days=30),
            _w3row("MID2", promoted_days=11),
        ]
    )
    _wl()
    assert _run(monkeypatch) == 0
    kept = pd.read_csv(re_.W3)["symbol"].tolist()
    assert kept == ["MID", "MID2"]  # 30d is kept (> STALE_DAYS drops), 31d dropped
    out = capsys.readouterr().out
    assert "aged out 1 stale" in out and "no fresh anomalies" in out


def test_main_dry_run_does_not_age_out(renv, monkeypatch):
    _w3([_w3row("OLD", promoted_days=40)])
    _wl()
    _run(monkeypatch, "--dry-run")
    assert pd.read_csv(re_.W3)["symbol"].tolist() == ["OLD"]


def test_fresh_boundary_is_ten_days(renv, monkeypatch, capsys):
    _w3([_w3row("EDGE", promoted_days=10), _w3row("LATE", promoted_days=11)])
    _wl()
    _digest(
        monkeypatch,
        {"EDGE": _prices(np.linspace(80, 100, 60)), "LATE": _prices(np.linspace(80, 100, 60))},
    )
    _rsi(monkeypatch)
    _run(monkeypatch)
    assert "1 fresh anomalies" in capsys.readouterr().out
    assert pd.read_csv(re_.OUT)["symbol"].tolist() == ["EDGE"]


def test_candidates_filters_and_scoring(renv, monkeypatch):
    rows = [
        _w3row("GOOD", enrolled_days=10, promoted_days=1),  # fast (9d)
        _w3row("SLOW", enrolled_days=40, promoted_days=1),  # 39d
        _w3row("BEARMKT", mkt="IN"),
        _w3row("HELD"),
        _w3row("SHORT"),
        _w3row("NODATA"),
        _w3row("NOCLOSE"),
        _w3row("RAISES"),
        _w3row("HOT"),
    ]
    _w3(rows)
    _wl([{"symbol": "HELD", "market": "US", "status": "held"}])
    (renv / "cache_seed" / "zone_regime.json").write_text(
        json.dumps({"IN": {"current_regime": "BEAR"}, "US": {"current_regime": "bull"}})
    )
    deep = np.concatenate([np.full(100, 100.0), np.linspace(100, 70, 30)])  # -30% from the high
    flat = np.full(130, 100.0)
    hot = np.concatenate([np.full(100, 100.0), np.linspace(100, 71, 30)])  # RSI vetoed below
    _digest(
        monkeypatch,
        {
            "GOOD": _prices(deep),
            "SLOW": _prices(flat),
            "BEARMKT": _prices(deep),
            "HELD": _prices(deep),
            "SHORT": _prices(np.linspace(1, 2, 29)),
            "NODATA": None,
            "NOCLOSE": pd.DataFrame({"Open": [1] * 60}),
            "RAISES": RuntimeError("x"),
            "HOT": _prices(hot),
        },
    )
    monkeypatch.setattr(pf, "rsi14", lambda c: 90.0 if abs(c.iloc[-1] - 71.0) < 1e-9 else 50.0)
    assert _run(monkeypatch) == 0
    out = pd.read_csv(re_.OUT)
    assert out["symbol"].tolist() == ["GOOD", "SLOW"]  # ranked by marker score
    good = out.set_index("symbol").loc["GOOD"]
    assert good["marker_score"] == 2 and good["days_to_trigger"] == 9
    assert good["drawdown_pct"] == pytest.approx(-30.0, abs=0.1)
    slow = out.set_index("symbol").loc["SLOW"]
    assert slow["marker_score"] == 0 and slow["days_to_trigger"] == 39
    assert "_rank" not in out.columns
    # default (no --commit): the watchlist is not modified
    assert pd.read_csv(re_.WL)["symbol"].tolist() == ["HELD"]


def test_commit_adds_rows_logs_and_caps_per_market(renv, monkeypatch, capsys):
    syms = ["A1", "A2", "A3", "A4"]
    _w3([_w3row(s, promoted_days=1 + i) for i, s in enumerate(syms)] + [_w3row("B1", mkt="IN")])
    _wl()
    _digest(monkeypatch, {s: _prices(np.linspace(80, 100, 60)) for s in syms + ["B1"]})
    _rsi(monkeypatch)
    assert _run(monkeypatch, "--commit") == 0
    wl = pd.read_csv(re_.WL)
    added = wl[wl["status"] == "reentry"]
    assert sorted(added["symbol"]) == ["A1", "A2", "A3", "B1"]  # A4 (oldest) capped out
    assert (added["entry_date"] == str(TODAY)).all()
    assert added["entry_price"].iloc[0] == pytest.approx(100.0)
    log = pd.read_csv(re_.LOG)
    assert len(log) == 4 and set(log["event"]) == {"reentry→tier1"}
    assert re_.LOG.read_text().count("date,event") == 1  # header once
    assert "(cap 3/market)" in capsys.readouterr().out


def test_commit_skips_existing_watchlist_names_and_dry_run_blocks_commit(renv, monkeypatch, capsys):
    _w3([_w3row("A1")])
    _wl()
    _digest(monkeypatch, {"A1": _prices(np.linspace(80, 100, 60))})
    _rsi(monkeypatch)
    _run(monkeypatch, "--commit", "--dry-run")
    assert pd.read_csv(re_.WL)["symbol"].tolist() == ["ZZZ"]
    assert "DRY RUN" in capsys.readouterr().out


def test_not_committed_message_and_overbought_veto(renv, monkeypatch, capsys):
    _w3([_w3row("A1"), _w3row("A2")])
    _wl()
    _digest(
        monkeypatch,
        {"A1": _prices(np.linspace(80, 100, 60)), "A2": _prices(np.linspace(80, 101, 60))},
    )
    monkeypatch.setattr(pf, "rsi14", lambda c: 75.0 if c.iloc[-1] > 100.5 else 50.0)
    _run(monkeypatch)
    out = capsys.readouterr().out
    assert pd.read_csv(re_.OUT)["symbol"].tolist() == ["A1"]  # A2 vetoed (US 75 >= 70)
    assert "NOT written" in out


def test_write_recent_reentries_merges_live_and_backtest(renv, monkeypatch, capsys):
    _w3([_w3row("LIVE", promoted_days=2, ret=4.0), _w3row("OLDLIVE", promoted_days=60)])
    ev = pd.DataFrame(
        {
            "market": ["US", "US", "IN", "IN"],
            "symbol": ["LIVE", "BT1", "BT2", "BT3"],
            "outcome": ["ANOMALY", "ANOMALY", "ANOMALY", "OTHER"],
            "evict_date": [str(TODAY - timedelta(days=50))] * 2
            + [str(TODAY - timedelta(days=40))] * 2,
            "days": [30, 3, 10, 5],
            "exit_ret_pct": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ev.to_csv(re_.BACKTEST_EVENTS, index=False)
    n = re_.write_recent_reentries()
    r = pd.read_csv(re_.RECENT).set_index("symbol")
    # LIVE appears in both: the live row (newest trigger) wins the dedupe
    assert r.loc["LIVE", "source"] == "live"
    assert "OLDLIVE" not in r.index  # 60d old, outside the 45d window
    assert "BT3" not in r.index  # not an ANOMALY outcome
    # BT1 trigger = evict+3d = 47d ago -> outside; BT2 = 30d ago -> inside
    assert "BT1" not in r.index and r.loc["BT2", "source"] == "backtest"
    assert n == 2 == len(r)


def test_write_recent_reentries_nothing(renv):
    assert re_.write_recent_reentries() == 0
    assert not re_.RECENT.exists()
