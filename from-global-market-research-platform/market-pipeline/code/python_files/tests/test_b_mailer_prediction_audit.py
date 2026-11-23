"""Offline tests for mailer_prediction_audit (regime series, verdicts, protected purge).

The external `price_prediction` package is not part of this repo, so it is faked.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mpa():
    saved = {
        k: sys.modules.get(k)
        for k in (
            "price_prediction",
            "price_prediction.kalman_trend",
            "price_prediction.markov_regime",
            "mailer_prediction_audit",
        )
    }

    class KalmanTrendModel:
        slope_value = 0.001

        def filter(self, logp):
            return pd.DataFrame(
                {"slope": np.full(len(logp), KalmanTrendModel.slope_value)}, index=logp.index
            )

    class MarkovRegimeModel:
        e_ret = 0.0005

        def fit(self, states, lr):
            return self

        def predict_return(self, st):
            return MarkovRegimeModel.e_ret

    def label_states(lr):
        return pd.Series(np.full(len(lr), label_states.state), index=lr.index)

    label_states.state = 2
    pkg = types.ModuleType("price_prediction")
    kt = types.SimpleNamespace(KalmanTrendModel=KalmanTrendModel)
    mr = types.SimpleNamespace(MarkovRegimeModel=MarkovRegimeModel, label_states=label_states)
    sys.modules.update(
        {
            "price_prediction": pkg,
            "price_prediction.kalman_trend": kt,
            "price_prediction.markov_regime": mr,
        }
    )
    sys.modules.pop("mailer_prediction_audit", None)
    mod = importlib.import_module("mailer_prediction_audit")
    mod._fakes = types.SimpleNamespace(K=KalmanTrendModel, M=MarkovRegimeModel, label=label_states)
    yield mod
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture
def audit(mpa, monkeypatch, tmp_path):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(mpa, "HERE", tmp_path)
    mpa._fakes.K.slope_value = 0.001
    mpa._fakes.M.e_ret = 0.0005
    mpa._fakes.label.state = 2
    return mpa


# ---------------------------------------------------------------- _basket_symbols
def test_basket_symbols_maps_conventions(audit):
    cols = {"RELIANCE", "TCS", "HDFCBANK.NS", "OTHER"}
    got = audit._basket_symbols("IN", cols)
    assert got == ["RELIANCE", "HDFCBANK.NS", "TCS"] or set(got) == {
        "RELIANCE",
        "HDFCBANK.NS",
        "TCS",
    }
    assert audit._basket_symbols("us", {"SPY", "QQQ", "NOPE"}) == [
        "SPY",
        "QQQ",
    ]  # basket order preserved
    assert audit._basket_symbols("EU", set()) == []
    assert audit._basket_symbols("KR", {"005930.KS"}) == ["005930.KS"]


# ---------------------------------------------------------------- trend_series
def _close(n, drift, cols=None, seed=0):
    rng = np.random.default_rng(seed)
    cols = cols or [f"T{i}" for i in range(6)]
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {c: 100 * np.cumprod(1 + drift + rng.normal(0, 0.001, n)) for c in cols}, index=idx
    )


def test_trend_series_labels_bull_bear_and_warmup(audit, monkeypatch):
    monkeypatch.setitem(audit.BASKETS, "us", [f"T{i}" for i in range(6)])
    bull = audit.trend_series("US", _close(300, 0.002))
    assert set(bull.unique()) <= {"bull", "chop", "warmup"} and bull.iloc[-1] == "bull"
    assert (bull.iloc[:50] == "warmup").all()  # EMA100 needs 100 observations
    bear = audit.trend_series("US", _close(300, -0.002))
    assert bear.iloc[-1] == "bear"


def test_trend_series_needs_five_names_per_day_and_masks_bad_ticks(audit, monkeypatch):
    monkeypatch.setitem(audit.BASKETS, "us", [f"T{i}" for i in range(6)])
    few = _close(300, 0.002, cols=[f"T{i}" for i in range(6)]).iloc[:, :4]
    assert audit.trend_series("US", few).empty  # four names never reach the n > 4 requirement
    spiked = _close(300, 0.002)
    spiked.iloc[200] *= 5  # +400% print on every name is masked, not followed
    assert audit.trend_series("US", spiked).iloc[-1] == "bull"


# ---------------------------------------------------------------- part_a
def _fake_md(monkeypatch):
    monkeypatch.setattr(
        pd.DataFrame,
        "to_markdown",
        lambda self, index=False: "|MD|" + ",".join(map(str, self.columns)),
    )


def test_part_a_gated_vs_ungated_excess(audit, monkeypatch, tmp_path):
    _fake_md(monkeypatch)
    for m in ("IN", "US"):
        monkeypatch.setitem(
            audit.BASKETS, {"IN": "india", "US": "us"}[m], [f"T{i}" for i in range(6)]
        )
    idx = pd.bdate_range("2023-01-02", periods=300)
    long = (
        _close(300, 0.002)
        .reset_index()
        .melt(id_vars="index", var_name="Symbol", value_name="Close")
        .rename(columns={"index": "Date"})
    )
    monkeypatch.setattr(audit, "_panel", lambda mkt: long)
    sig_dates = [idx[250], idx[280], idx[40]]  # two in the bull stretch, one in warm-up
    so = pd.DataFrame(
        {
            "status": ["scored"] * 6 + ["pending"],
            "market": ["IN"] * 6 + ["IN"],
            "h": [5.0, 5.0, 5.0, 21.0, 21.0, 21.0, 5.0],
            "signal_date": sig_dates * 2 + [idx[250]],
            "excess_ret": [0.02, 0.04, -0.10, 0.01, 0.03, -0.20, 0.5],
        }
    )
    (tmp_path / "reports" / "signal_outcomes.parquet").parent.mkdir(exist_ok=True)
    so.to_parquet(tmp_path / "reports" / "signal_outcomes.parquet")
    text, t = audit.part_a()
    row5 = t[(t.market == "IN") & (t.h == 5)].iloc[0]
    assert row5.n_all == 3 and row5.n_bull == 2 and row5.n_blocked == 1
    assert (
        row5.ungated_excess == round((0.02 + 0.04 - 0.10) / 3 * 100, 2) and row5.gated_excess == 3.0
    )
    assert row5.blocked_excess == -10.0  # the warm-up signal is blocked, not gated in
    assert set(t.market) == {"IN"} and "## A — Retro regime-gate" in text


# ---------------------------------------------------------------- part_b verdicts
def _watch(tmp_path, rows):
    pd.DataFrame(rows, columns=["symbol", "market", "status", "note"]).to_csv(
        tmp_path / "watchlist.csv", index=False
    )


def _setup_b(
    audit, monkeypatch, tmp_path, trend="bull", vol="LOW", state=2, drift=0.001, n=400, rows=None
):
    _fake_md(monkeypatch)
    _watch(
        tmp_path,
        rows
        or [
            ("AAA", "US", "watch", ""),
            ("MISSING", "US", "signal", ""),
            ("SHORT", "US", "watch", ""),
            ("OLD", "US", "sold", ""),
            ("JP1", "JP", "watch", ""),
        ],
    )
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=n)
    panel = {
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        "JP1": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        "SHORT": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
    }
    df = pd.DataFrame(panel, index=idx)
    df.loc[df.index[: n - 100], "SHORT"] = np.nan  # < 300 observations
    long = (
        df.reset_index()
        .melt(id_vars="index", var_name="Symbol", value_name="Close")
        .rename(columns={"index": "Date"})
    )
    monkeypatch.setattr(audit, "_panel", lambda mkt: long)
    monkeypatch.setitem(
        sys.modules,
        "market_regime",
        types.SimpleNamespace(
            get_regime=lambda m: {"trend": trend, "vol": vol},
            BASKETS=audit.BASKETS,
            _ALIAS=audit._ALIAS,
        ),
    )
    audit._fakes.K.slope_value = drift
    audit._fakes.label.state = state


@pytest.mark.parametrize(
    "trend,vol,state,drift,verdict",
    [
        ("bear", "LOW", 2, 0.001, "DEMOTE (market bear)"),
        ("chop", "HIGH", 2, 0.001, "DEMOTE (chop+HIGH vol)"),
        ("chop", "LOW", 2, 0.001, "ENTER-OK"),
        ("bull", "HIGH", 2, 0.001, "ENTER-OK"),
        ("bull", "LOW", 2, -0.001, "HOLD-OK"),  # bull state but negative drift
        ("bull", "LOW", 1, 0.001, "HOLD-OK"),  # chop state, positive drift
        ("bull", "LOW", 1, -0.001, "HOLD-OK"),  # not a bear state
        ("bull", "LOW", 0, 0.001, "HOLD-OK"),  # bear state but positive drift
        ("bull", "LOW", 0, -0.001, "WEAK (stock bear-state, drift<0)"),
    ],
)
def test_part_b_verdict_matrix(audit, monkeypatch, tmp_path, trend, vol, state, drift, verdict):
    _setup_b(audit, monkeypatch, tmp_path, trend, vol, state, drift)
    text, df = audit.part_b()
    assert df[df.symbol == "AAA"].verdict.iloc[0] == verdict


def test_part_b_filters_names_and_computes_fields(audit, monkeypatch, tmp_path):
    _setup_b(audit, monkeypatch, tmp_path)
    _, df = audit.part_b()
    assert sorted(df.symbol) == ["AAA", "JP1"]  # sold / missing / short-history names excluded
    r = df[df.symbol == "AAA"].iloc[0]
    assert r.kalman_drift_ann_pct == round(0.001 * 252 * 100, 1)
    assert r.markov_e21d_pct == round(0.0005 * 21 * 100, 2) and r.stock_state == "bull"
    assert r.mkt_trend == "bull" and r.status == "watch" and len(r["asof"]) == 10


def test_part_b_state_names_and_warmup(audit, monkeypatch, tmp_path):
    _setup_b(audit, monkeypatch, tmp_path, state=7)
    assert audit.part_b()[1].stock_state.iloc[0] == "warmup"
    _setup_b(audit, monkeypatch, tmp_path, state=1)
    assert audit.part_b()[1].stock_state.iloc[0] == "chop"


def test_part_b_skips_markets_with_no_active_names(audit, monkeypatch, tmp_path):
    _setup_b(audit, monkeypatch, tmp_path, rows=[("AAA", "US", "watch", "")])
    _, df = audit.part_b()
    assert set(df.market) == {"US"}


# ---------------------------------------------------------------- purge_weak
def test_purge_weak_protects_held_and_archives_the_rest(audit, monkeypatch, tmp_path, capsys):
    wl = pd.DataFrame(
        {
            "symbol": ["W1", "W2", "OK", "H1"],
            "market": ["US", "US", "US", "US"],
            "status": ["signal", "watch", "signal", "held"],
            "note": ["n1", "", "fine", "mine"],
        }
    )
    wl.to_csv(tmp_path / "watchlist.csv", index=False)
    scores = pd.DataFrame(
        {
            "symbol": ["W1", "W2", "OK", "H1", "W1"],
            "market": ["US", "US", "US", "US", "IN"],
            "verdict": ["WEAK (x)", "WEAK (x)", "ENTER-OK", "WEAK (x)", "WEAK (x)"],
        }
    )
    audit.purge_weak(scores)
    out = pd.read_csv(tmp_path / "watchlist.csv")
    assert out["symbol"].tolist() == ["OK", "H1"]
    held = out.set_index("symbol").at["H1", "note"]
    assert held.startswith("mine | weak ") and "protected status, not purged" in held
    arch = pd.read_csv(tmp_path / "watchlist_purged.csv", header=None)
    assert arch[0].tolist() == ["W1", "W2"] and set(arch[2]) == {"purged"}
    assert "purged 2; protected (flagged only): 1" in capsys.readouterr().out


def test_purge_weak_flag_is_idempotent_and_blank_note_has_no_separator(audit, tmp_path):
    pd.DataFrame(
        {
            "symbol": ["H1", "H2"],
            "market": ["US", "US"],
            "status": ["held", "value-hold"],
            "note": [float("nan"), "x"],
        }
    ).to_csv(tmp_path / "watchlist.csv", index=False)
    scores = pd.DataFrame({"symbol": ["H1", "H2"], "market": ["US", "US"], "verdict": ["WEAK"] * 2})
    audit.purge_weak(scores)
    audit.purge_weak(scores)
    out = pd.read_csv(tmp_path / "watchlist.csv").set_index("symbol")
    assert out.at["H1", "note"].startswith("weak ") and out.at["H1", "note"].count("weak ") == 1
    assert out.at["H2", "note"].count("weak ") == 1 and out.at["H2", "note"].startswith("x | weak ")
    assert not (tmp_path / "watchlist_purged.csv").exists()


# ---------------------------------------------------------------- main
def test_main_writes_report_and_scores_and_honours_purge_flag(audit, monkeypatch, tmp_path, capsys):
    df = pd.DataFrame({"symbol": ["A"], "verdict": ["ENTER-OK"]})
    monkeypatch.setattr(audit, "part_a", lambda: ("A-TEXT", pd.DataFrame()))
    monkeypatch.setattr(audit, "part_b", lambda: ("B-TEXT", df))
    purged = []
    monkeypatch.setattr(audit, "purge_weak", lambda s: purged.append(s))
    monkeypatch.setattr(sys, "argv", ["mailer_prediction_audit.py"])
    audit.main()
    md = (tmp_path / "reports" / "mailer_prediction_audit.md").read_text()
    assert (
        md.startswith("# Mailer picks × prediction strategy — audit (")
        and "A-TEXT" in md
        and "B-TEXT" in md
    )
    assert pd.read_csv(tmp_path / "reports" / "watchlist_prediction_scores.csv")[
        "symbol"
    ].tolist() == ["A"]
    assert purged == []
    monkeypatch.setattr(sys, "argv", ["mailer_prediction_audit.py", "--purge-weak"])
    audit.main()
    assert purged == [df]


def test_trend_series_keeps_moves_up_to_sixty_percent(audit, monkeypatch):
    monkeypatch.setitem(audit.BASKETS, "us", [f"T{i}" for i in range(6)])
    close = _close(300, 0.002)
    close.iloc[-1] = (
        close.iloc[-2] * 0.45
    )  # a -55% last print is real (<= 60%) and breaks the trend
    assert audit.trend_series("US", close).iloc[-1] == "bear"


def test_part_b_history_threshold_and_return_filter(audit, monkeypatch, tmp_path):
    seen = []
    orig = audit._fakes.K.filter

    def spy(self, logp):
        seen.append(len(logp))
        return orig(self, logp)

    monkeypatch.setattr(audit._fakes.K, "filter", spy)
    _setup_b(
        audit,
        monkeypatch,
        tmp_path,
        rows=[("EXACT", "US", "watch", ""), ("JUMP", "US", "watch", "")],
    )
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(3)
    exact = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)), index=idx)
    jump = exact.copy()
    jump.iloc[150:] *= 2.0  # a +69% log return: filtered out of the Kalman input
    long = (
        pd.concat({"EXACT": exact, "JUMP": jump}, axis=1)
        .reset_index()
        .melt(id_vars="index", var_name="Symbol", value_name="Close")
        .rename(columns={"index": "Date"})
    )
    monkeypatch.setattr(audit, "_panel", lambda mkt: long)
    _, df = audit.part_b()
    assert sorted(df.symbol) == ["EXACT", "JUMP"]  # exactly 300 observations is enough
    assert sorted(seen) == [
        298,
        299,
    ]  # EXACT: 299 returns; JUMP: one return dropped by the |r| < 0.5 filter


def test_part_b_zero_drift_bull_state_is_hold_not_enter(audit, monkeypatch, tmp_path):
    _setup_b(audit, monkeypatch, tmp_path, "bull", "LOW", 2, 0.0)
    assert audit.part_b()[1].query("symbol == 'AAA'").verdict.iloc[0] == "HOLD-OK"
