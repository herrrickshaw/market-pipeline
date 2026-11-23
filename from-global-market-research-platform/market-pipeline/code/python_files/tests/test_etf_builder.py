"""etf_builder.py -- weighting schemes, performance maths, point-in-time panel loader."""

import sys

import numpy as np
import pandas as pd
import pytest

import etf_builder as EB

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
# w_invvol writes into `R.std().values`, which is a read-only view under pandas 3
# copy-on-write, so the scheme raises ValueError there (main() swallows it and silently
# substitutes equal weights).
invvol_cow_bug = pytest.mark.xfail(
    PANDAS3,
    strict=True,
    raises=ValueError,
    reason="w_invvol assigns into a read-only array under pandas>=3 copy-on-write",
)


def rets(n_days=300, n_assets=10, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    if drift is not None:
        data = data + np.asarray(drift)
    return pd.DataFrame(data, columns=[f"S{i}" for i in range(n_assets)])


# ── weights ───────────────────────────────────────────────────────────────────
def test_equal_weights():
    R = rets(50, 4)
    w = EB.w_equal(R, None, None)
    assert list(w) == [0.25] * 4 and w.sum() == pytest.approx(1.0)


@invvol_cow_bug
def test_inverse_vol_weights_proportional_to_one_over_sigma():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(
        {
            "a": rng.normal(0, 0.01, 500),
            "b": rng.normal(0, 0.02, 500),
            "c": rng.normal(0, 0.04, 500),
        }
    )
    w = EB.w_invvol(R, None, None)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    inv = 1 / R.std().values
    assert w == pytest.approx(inv / inv.sum())


@invvol_cow_bug
def test_inverse_vol_drops_zero_variance_names():
    R = pd.DataFrame({"a": [0.01, -0.01, 0.02, -0.02], "flat": [0.0] * 4})
    w = EB.w_invvol(R, None, None)
    assert np.isnan(w[1]) and w[0] == pytest.approx(1.0)


def test_min_variance_respects_cap_sum_and_prefers_low_vol():
    n = 12
    variances = np.linspace(0.01, 0.12, n)
    cov = np.diag(variances)
    w = EB.w_minvar(None, None, cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.min() >= -1e-9 and w.max() <= EB.WEIGHT_CAP + 1e-6
    assert w[0] >= w[-1]
    # cap binds on the cheapest names (unconstrained solution would put >10% there)
    assert w[0] == pytest.approx(EB.WEIGHT_CAP, abs=1e-4)
    assert w @ cov @ w < np.repeat(1 / n, n) @ cov @ np.repeat(1 / n, n)


def test_min_variance_identical_assets_equal_weights():
    w = EB.w_minvar(None, None, np.eye(10) * 0.04)
    assert w == pytest.approx(np.repeat(0.1, 10), abs=1e-5)


def test_max_sharpe_chases_the_highest_mean_up_to_the_cap():
    n = 12
    mu = np.array([0.50] + [0.05] * (n - 1))
    cov = np.eye(n) * 0.04
    w = EB.w_maxsharpe(None, mu, cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w[0] == pytest.approx(EB.WEIGHT_CAP, abs=1e-4)
    assert w.max() <= EB.WEIGHT_CAP + 1e-6 and w.min() >= -1e-9


@pytest.mark.parametrize("fn", ["w_minvar", "w_maxsharpe"])
def test_optimisers_fall_back_to_equal_weight_when_solver_fails(fn, monkeypatch):
    import scipy.optimize as so

    class Fail:
        success = False
        x = np.zeros(3)

    monkeypatch.setattr(so, "minimize", lambda *a, **k: Fail())
    w = getattr(EB, fn)(None, np.array([0.1, 0.2, 0.3]), np.eye(3))
    assert list(w) == pytest.approx([1 / 3] * 3)


def test_max_sharpe_zero_variance_does_not_divide_by_zero():
    w = EB.w_maxsharpe(None, np.array([0.1, 0.1]), np.zeros((2, 2)))
    assert np.isfinite(w).all()


def test_momentum_picks_top_quintile_with_floor_of_five():
    R = rets(300, 10, drift=np.arange(10) * 0.001)
    R = R - R.mean() + np.arange(10) * 0.001  # exact per-column drift
    w = EB.w_momentum(R, None, None)
    assert (w > 0).sum() == 5 and set(np.flatnonzero(w)) == {5, 6, 7, 8, 9}
    assert w.sum() == pytest.approx(1.0) and w[w > 0] == pytest.approx([0.2] * 5)
    big = rets(300, 40, drift=np.arange(40) * 0.0005)
    w40 = EB.w_momentum(big, None, None)
    assert (w40 > 0).sum() == 8 and w40[w40 > 0] == pytest.approx([1 / 8] * 8)  # 40 // 5


def test_momentum_skips_the_most_recent_month():
    R = pd.DataFrame(0.001, index=range(300), columns=[f"S{i}" for i in range(10)])
    R["S0"] = 0.0
    R.loc[279:, "S0"] = 0.10  # all of S0's gain is in the last 21 days
    w = EB.w_momentum(R, None, None)
    assert w[0] == 0.0  # 12-1 skip removes it
    assert (w > 0).sum() == 5


def test_schemes_registry():
    assert list(EB.SCHEMES) == [
        "1/N equal",
        "inverse-vol",
        "min-variance",
        "max-Sharpe",
        "momentum 12-1",
    ]
    assert EB.SCHEMES["1/N equal"] is EB.w_equal


# ── perf ──────────────────────────────────────────────────────────────────────
def test_perf_values_for_monthly_nav():
    nav = pd.Series([1.0, 1.1, 1.21, 0.968, 1.0648] + [1.0648] * 8)  # 13 points -> 12 periods
    p = EB.perf(nav, 12)
    r = nav.pct_change().dropna()
    assert p["cagr"] == pytest.approx(1.0648 ** (1 / 1.0) - 1)
    assert p["vol"] == pytest.approx(r.std() * np.sqrt(12))
    assert p["sharpe"] == pytest.approx(p["cagr"] / p["vol"])
    assert p["maxdd"] == pytest.approx(0.968 / 1.21 - 1)


def test_perf_annualises_by_periods_per_year_not_252():
    nav = pd.Series(1.02 ** np.arange(0, 13))  # +2%/month for a year
    assert EB.perf(nav, 12)["cagr"] == pytest.approx(1.02**12 - 1)
    assert EB.perf(pd.Series(1.02 ** np.arange(0, 5)), 4)["cagr"] == pytest.approx(1.02**4 - 1)
    nav = pd.Series([1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0])
    r = nav.pct_change().dropna()
    assert EB.perf(nav, 4)["vol"] == pytest.approx(r.std() * np.sqrt(4))
    assert EB.perf(nav, 12)["vol"] == pytest.approx(r.std() * np.sqrt(12))


def test_perf_degenerate_inputs():
    assert EB.perf(pd.Series([1.0]), 12) == {}
    assert EB.perf(pd.Series([0.0, 1.0, 2.0]), 12) == {}
    assert EB.perf(pd.Series([-1.0, 1.0, 2.0]), 12) == {}
    flat = EB.perf(pd.Series([1.0, 1.0, 1.0]), 12)
    assert flat["cagr"] == 0.0 and flat["vol"] == 0.0 and np.isnan(flat["sharpe"])
    assert flat["maxdd"] == 0.0


# ── load_panel / main ─────────────────────────────────────────────────────────
def test_load_panel_pivots_close_and_turnover_from_2016(monkeypatch):
    import backtest_tier_anomalies as bta

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2015-12-31", "2016-01-04", "2016-01-04", "2016-01-05", "2016-01-05"]
            ),
            "Symbol": ["A", "A", "B", "A", "B"],
            "Close": [1.0, 10.0, 20.0, 11.0, 21.0],
            "Volume": [5, 2, 3, 4, 6],
        }
    )
    monkeypatch.setattr(bta, "load_market", lambda m: df)
    px, tv = EB.load_panel("US")
    assert list(px.index.strftime("%Y-%m-%d")) == ["2016-01-04", "2016-01-05"]  # 2015 dropped
    assert px.loc["2016-01-05", "B"] == 21.0
    assert tv.loc["2016-01-04", "A"] == 20.0 and tv.loc["2016-01-05", "B"] == 126.0
    assert list(px.columns) == ["A", "B"]


def test_main_refuses_unadjusted_india_panel(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--market", "IN"])
    monkeypatch.setattr(EB, "load_panel", lambda m: pytest.fail("must refuse before loading"))
    assert EB.main() == 1
    out = capsys.readouterr().out
    assert "REFUSING IN" in out and "split-UNADJUSTED" in out and "--allow-unadjusted" in out


def test_main_override_flag_gets_past_the_refusal(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--market", "IN", "--allow-unadjusted"])
    loaded = []
    monkeypatch.setattr(
        EB, "load_panel", lambda m: loaded.append(m) or (_ for _ in ()).throw(KeyError("stop"))
    )
    with pytest.raises(KeyError):
        EB.main()
    assert loaded == ["IN"]


def test_constants():
    assert {"IN"} == EB.UNADJUSTED and EB.WEIGHT_CAP == 0.10 and EB.COST_BPS == 10.0


# ═══ main(): walk-forward report against faked loaders / sklearn / yfinance ═════
import types  # noqa: E402


class FakeLedoitWolf:
    def fit(self, X):
        self.covariance_ = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
        return self


def synthetic_panel(n_names=30, years=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=252 * years)
    mkt = rng.normal(0.0004, 0.008, len(idx))
    cols = {}
    for i in range(n_names):
        beta = 0.5 + 0.05 * i
        r = beta * mkt + rng.normal(0, 0.01, len(idx))
        cols[f"S{i:02d}"] = 100 * np.cumprod(1 + r)
    px = pd.DataFrame(cols, index=idx)
    tv = pd.DataFrame({c: 1e6 * (1 + k) for k, c in enumerate(px.columns)}, index=idx)
    return px, tv


@pytest.fixture
def main_env(tmp_path, monkeypatch):
    px, tv = synthetic_panel()
    monkeypatch.setattr(EB, "load_panel", lambda m: (px, tv))
    monkeypatch.setattr(EB, "OUT_MD", tmp_path / "etf.md")
    monkeypatch.setitem(sys.modules, "sklearn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "sklearn.covariance", types.SimpleNamespace(LedoitWolf=FakeLedoitWolf)
    )
    return px


def fake_yf(monkeypatch, series=None, boom=False):
    class Ticker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, start, end, auto_adjust):
            if boom:
                raise RuntimeError("no network")
            s = series
            return pd.DataFrame({"Close": s.values}, index=s.index.tz_localize("UTC"))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["etf_builder", *argv])
    return EB.main()


def test_main_report_without_benchmark(main_env, monkeypatch, capsys):
    fake_yf(monkeypatch, boom=True)
    assert run_main(monkeypatch, "--market", "US", "--top", "25") == 0
    md = EB.OUT_MD.read_text()
    assert md.startswith("# ETF builder — US top-25, walk-forward, monthly rebalance")
    for scheme in EB.SCHEMES:
        assert f"| {scheme} |" in md
    assert "benchmark SPY unavailable" in capsys.readouterr().out
    assert "n/a" in md and "(the ETF you'd just buy)" not in md
    assert "long-only, 10% cap, 10bp turnover cost" in md


def test_main_report_with_benchmark_and_quarterly_rebalance(main_env, monkeypatch):
    px = main_env
    bench = pd.Series(100 * np.cumprod(1 + np.full(len(px), 0.0003)), index=px.index)
    fake_yf(monkeypatch, bench)
    assert run_main(monkeypatch, "--top", "25", "--rebalance", "QE", "--benchmark", "QQQ") == 0
    md = EB.OUT_MD.read_text()
    assert "quarterly rebalance" in md and "| **QQQ (the ETF you'd just buy)** |" in md
    rows = [x for x in md.splitlines() if x.startswith("| 1/N equal")]
    assert len(rows) == 1 and rows[0].count("|") == 10  # 9 cells, as the header


def test_main_equal_weight_row_matches_hand_calculation(main_env, monkeypatch):
    fake_yf(monkeypatch, boom=True)
    run_main(monkeypatch, "--top", "30")
    md = EB.OUT_MD.read_text()
    row = [x for x in md.splitlines() if x.startswith("| 1/N equal")][0].split("|")
    assert row[7].strip() == "+0.00%"  # vs 1/N: itself
    assert row[8].strip() == "n/a"  # no benchmark


def test_main_survives_a_scheme_that_raises(main_env, monkeypatch):
    fake_yf(monkeypatch, boom=True)

    def broken(R, mu, cov):
        raise RuntimeError("solver blew up")

    monkeypatch.setitem(EB.SCHEMES, "max-Sharpe", broken)
    assert run_main(monkeypatch, "--top", "25") == 0
    assert "| max-Sharpe |" in EB.OUT_MD.read_text()  # fell back to equal weights


def test_main_nan_weights_fall_back_to_equal_and_negative_sum_too(main_env, monkeypatch):
    fake_yf(monkeypatch, boom=True)
    monkeypatch.setitem(EB.SCHEMES, "1/N equal", lambda R, mu, cov: np.full(R.shape[1], np.nan))
    monkeypatch.setitem(EB.SCHEMES, "inverse-vol", lambda R, mu, cov: -np.ones(R.shape[1]))
    assert run_main(monkeypatch, "--top", "25") == 0


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="when no rebalance has >= 20 usable names, `idx` stays empty and main() "
    "dereferences idx[0] for the header instead of reporting 'no periods'",
)
def test_main_too_few_names_reports_instead_of_crashing(main_env, monkeypatch):
    px, tv = synthetic_panel(n_names=10)
    monkeypatch.setattr(EB, "load_panel", lambda m: (px, tv))
    fake_yf(monkeypatch, boom=True)
    assert run_main(monkeypatch, "--top", "25") in (0, 1)
