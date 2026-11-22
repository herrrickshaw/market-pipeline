"""bond_equity_entropy.py -- sample entropy, transfer entropy, report sections.

Hand-computable cases pin the estimators; the report builders run on small
synthetic frames. The module-level RNG is replaced per test so results never
depend on test order.
"""

import math
import sys

import numpy as np
import pandas as pd
import pytest

import bond_equity_entropy as BE


@pytest.fixture(autouse=True)
def fixed_rng(monkeypatch):
    monkeypatch.setattr(BE, "RNG", np.random.default_rng(12345))


# ── sampen ────────────────────────────────────────────────────────────────────
def test_sampen_alternating_series_hand_value():
    x = np.array([0.0, 1.0] * 20)
    # m=2: 20 vectors [0,1] + 19 vectors [1,0]  -> B = 20*19 + 19*18 = 722
    # m=3: 19 of each type                       -> A = 2 * 19*18   = 684
    assert BE.sampen(x) == pytest.approx(-math.log(684 / 722))


def test_sampen_all_vectors_match_when_r_huge():
    x = np.arange(30, dtype=float)
    # r = 100*std swallows every distance: B = 29*28, A = 28*27
    assert BE.sampen(x, r_mult=100.0) == pytest.approx(-math.log((28 * 27) / (29 * 28)))


def test_sampen_tolerance_is_inclusive():
    x = np.array([0.0, 1.0] * 20)
    # std 0.5, r_mult 2 -> r = 1.0 == the largest pairwise distance: everything matches
    assert BE.sampen(x, r_mult=2.0) == pytest.approx(-math.log((38 * 37) / (39 * 38)))


def test_sampen_ignores_nonfinite_values():
    x = np.array([0.0, 1.0] * 20)
    y = np.concatenate([x, [np.nan, np.inf, -np.inf]])
    assert BE.sampen(y) == pytest.approx(BE.sampen(x))


def test_sampen_min_length_boundary():
    assert np.isnan(BE.sampen(np.array([0.0, 1.0] * 10 + [0.0])))  # N=21 < m+20
    assert np.isfinite(BE.sampen(np.array([0.0, 1.0] * 11)))  # N=22


def test_sampen_m_changes_the_boundary():
    x = np.tile([0.0, 1.0], 12)[:23]
    assert np.isfinite(BE.sampen(x, m=3))
    assert np.isnan(BE.sampen(x[:22], m=3))


def test_sampen_constant_or_no_matches_is_nan():
    assert np.isnan(BE.sampen(np.ones(50)))
    rng = np.random.default_rng(0)
    # strictly increasing distinct values, tiny r -> no length-m matches
    x = np.cumsum(rng.uniform(1, 2, 60))
    assert np.isnan(BE.sampen(x, r_mult=1e-6))


def test_sampen_random_noise_is_higher_than_periodic():
    rng = np.random.default_rng(7)
    noise = rng.normal(size=300)
    periodic = np.tile([0.0, 1.0, 0.5], 100)
    assert BE.sampen(noise) > BE.sampen(periodic)


# ── _disc / transfer entropy ──────────────────────────────────────────────────
def test_disc_equiprobable_terciles():
    assert list(BE._disc(np.arange(9.0), 3)) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert list(BE._disc(np.array([1.0, 1000.0, -5.0, 3.0]), 2)) == [0, 1, 0, 1]


def test_disc_robust_to_outliers():
    a = BE._disc(np.array([1, 2, 3, 4, 5, 6.0]), 3)
    b = BE._disc(np.array([1, 2, 3, 4, 5, 6e9]), 3)
    assert list(a) == list(b)


def lagged_pair(n=3000, lag=1, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.normal(size=n)
    return src, np.roll(src, lag)


def test_te_detects_directional_flow():
    src, dst = lagged_pair()
    fwd = BE.transfer_entropy(src, dst)
    back = BE.transfer_entropy(dst, src)
    assert fwd == pytest.approx(math.log(3), abs=0.03)  # dst[t+1] == src[t] exactly
    assert back < 0.02 and fwd > 50 * back


def test_te_lag_parameter():
    src, dst = lagged_pair(lag=2, seed=3)
    assert BE.transfer_entropy(src, dst, lag=2) > 1.0
    assert BE.transfer_entropy(src, dst, lag=1) < 0.05


def test_te_bins_parameter():
    src, dst = lagged_pair(n=4000, seed=5)
    assert BE.transfer_entropy(src, dst, bins=2) == pytest.approx(math.log(2), abs=0.03)


def test_te_is_nonnegative_for_independent_series():
    rng = np.random.default_rng(11)
    te = BE.transfer_entropy(rng.normal(size=2000), rng.normal(size=2000))
    assert 0.0 <= te < 0.02


def test_te_minimum_sample_boundary():
    rng = np.random.default_rng(1)
    assert np.isnan(BE.transfer_entropy(rng.normal(size=50), rng.normal(size=50)))  # n=49
    assert np.isfinite(BE.transfer_entropy(rng.normal(size=51), rng.normal(size=51)))  # n=50


# ── effective_te ──────────────────────────────────────────────────────────────
def test_effective_te_flow_is_significant_and_p_floor():
    src, dst = lagged_pair(n=1500)
    raw, eff, p = BE.effective_te(src, dst, n_sur=19)
    assert raw == pytest.approx(BE.transfer_entropy(src, dst))
    assert eff < raw and eff > 1.0
    assert p == pytest.approx(1 / 20)  # (0 + 1) / (19 + 1)


def test_effective_te_independent_is_not_significant():
    rng = np.random.default_rng(21)
    src, dst = rng.normal(size=1500), rng.normal(size=1500)
    raw, eff, p = BE.effective_te(src, dst, n_sur=99)
    assert eff <= raw  # surrogate mean of TE is >= 0
    assert 1 / 100 <= p <= 1.0 and p > 0.02
    assert abs(eff) < 0.01


def test_effective_te_does_not_mutate_source():
    src, dst = lagged_pair(n=800)
    before = src.copy()
    BE.effective_te(src, dst, n_sur=5)
    assert np.array_equal(src, before)


def test_effective_te_short_series_all_nan():
    out = BE.effective_te(np.arange(10.0), np.arange(10.0), n_sur=3)
    assert all(np.isnan(v) for v in out)


# ── load ──────────────────────────────────────────────────────────────────────
def test_load_derivations(tmp_path, monkeypatch):
    dates = pd.to_datetime(
        ["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-06", "2020-01-07", "2020-01-08"]
    )
    y = pd.DataFrame(
        {
            "Date": dates,
            "3 Mo": [1.0, 1.5, 1.2, 1.1, 1.3, 1.4],
            "2 Yr": 2.0,
            "10 Yr": [3.0, 2.0, 2.5, np.nan, 3.5, 4.0],
            "30 Yr": 4.0,
        }
    )
    spx_dates = list(dates) + [pd.Timestamp("2020-01-02")]  # duplicate date
    spx = pd.DataFrame(
        {"Date": spx_dates, "spx": [110.0, 100.0, 105.0, 120.0, np.nan, 140.0, 999.0]}
    )
    y.to_parquet(tmp_path / "y.parquet")
    spx.to_parquet(tmp_path / "s.parquet")
    monkeypatch.setattr(BE, "YIELDS", tmp_path / "y.parquet")
    monkeypatch.setattr(BE, "SPX", tmp_path / "s.parquet")
    d = BE.load()
    # sorted; rows with NaN spx (01-07) or NaN y10y (01-06) dropped; first diff row dropped
    assert list(d.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03", "2020-01-08"]
    assert d.loc["2020-01-02", "spx"] == 105.0  # duplicate: first kept
    assert d.loc["2020-01-02", "ret"] == pytest.approx(math.log(105 / 100))
    assert d.loc["2020-01-03", "dy10"] == pytest.approx(3.0 - 2.5)
    assert d.loc["2020-01-08", "dy10"] == pytest.approx(4.0 - 3.0)  # diff over remaining rows
    assert d.loc["2020-01-03", "bond_ret"] == pytest.approx(-7.0 * 0.5 / 100.0)
    assert d.loc["2020-01-03", "spread"] == pytest.approx(3.0 - 1.0)
    assert d.loc["2020-01-02", "spread"] == pytest.approx(2.5 - 1.2)
    assert (d["spread"] == d["real_proxy"]).all()
    assert d.loc["2020-01-02", "y2y"] == 2.0 and d.loc["2020-01-02", "y30y"] == 4.0


# ── report builders ───────────────────────────────────────────────────────────
def make_frame(start, n, seed=0, jumps=()):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = rng.normal(0.0004, 0.01, n)
    y10 = np.full(n, 3.0)
    for j in jumps:
        y10[j:] += 1.0
    bond = -0.5 * ret + rng.normal(0, 0.005, n)
    d = pd.DataFrame(
        {
            "ret": ret,
            "bond_ret": bond,
            "y10y": y10,
            "y3m": 2.0 + 2.5 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 0.05, n),
        },
        index=idx,
    )
    d["dy10"] = d["y10y"].diff()
    d["spread"] = d["y10y"] - d["y3m"]
    return d


def test_run_sampen_report_and_return(capsys):
    d = make_frame("2007-01-01", 1100, seed=3)  # spans GFC window only
    out = []
    s = BE.run_sampen(d, out)
    text = "\n".join(out)
    assert list(s.columns) == ["sampen", "vol"] and not s.empty
    assert s.index[0] == d.index[251]  # first window ends at row 252
    assert np.diff(s.index.values).min() >= np.timedelta64(21, "D")
    # vol is annualised std of the window
    assert s["vol"].iloc[0] == pytest.approx(d["ret"].iloc[:252].std(ddof=0) * math.sqrt(252))
    assert "GFC 2007-09 to 2009-06" in text
    assert "dot-com" not in text and "COVID" not in text and "2022 bear" not in text
    gfc = s.loc["2007-09-01":"2009-06-30"]
    calm = s.drop(gfc.index)
    delta = gfc.sampen.mean() - calm.sampen.mean()
    assert f"**SampEn {calm.sampen.mean():.3f}** (n={len(calm)} windows)" in text
    assert f"| {gfc.sampen.mean():.3f} | {delta:+.3f} |" in text
    assert ("DOWN (more regular)" in text) is bool(delta < 0)
    c = s.sampen.corr(s.vol)
    assert f"{c:+.3f}" in text
    assert ("largely the inverse of vol" in text) is bool(abs(c) > 0.6)


def test_run_te_table_structure_and_skips_short_segments(monkeypatch):
    d = make_frame("2010-01-01", 2100, seed=5)  # only 2010-2019 segment (+full)
    monkeypatch.setattr(BE.effective_te, "__defaults__", (20,))
    out = []
    BE.run_te(d, out)
    text = "\n".join(out)
    assert any(x.startswith("| full sample 1990-2026 | 2,100 |") for x in out)
    assert any(x.startswith("| 2010-2019 |") for x in out)
    for gone in ("| 1990-1999", "| 2000-2009", "| 2020-2026"):
        assert gone not in text
    assert "### State dependence" in text and "calm (bottom 80% vol)" in text
    assert "turbulent (top 20% vol)" in text
    c = d["bond_ret"].corr(d["ret"])
    assert f"= **{c:+.3f}**" in text
    assert f"(bond ret t-1, equity ret t) = {d['bond_ret'].shift(1).corr(d['ret']):+.3f}" in text
    assert f"(equity ret t-1, bond ret t) = {d['ret'].shift(1).corr(d['bond_ret']):+.3f}" in text
    assert "20 shuffled surrogates" not in text  # header uses N_SURROGATE const
    assert f"{BE.N_SURROGATE} shuffled surrogates" in text


def test_run_te_lead_label_follows_net_sign(monkeypatch):
    d = make_frame("2010-01-01", 500, seed=9)
    vals = iter([(0, 0.5, 0.01), (0, 0.2, 0.2)] * 3)  # bond->eq 0.5, eq->bond 0.2
    monkeypatch.setattr(BE, "effective_te", lambda s, t: next(vals))
    out = []
    BE.run_te(d, out)
    full = [x for x in out if x.startswith("| full sample")][0]
    assert (
        full == "| full sample 1990-2026 | 500 | 0.50000 | 0.010 | 0.20000 | 0.200 | bond leads |"
    )
    vals2 = iter([(0, 0.1, 0.5), (0, 0.4, 0.5)] * 3)
    monkeypatch.setattr(BE, "effective_te", lambda s, t: next(vals2))
    out = []
    BE.run_te(d, out)
    assert [x for x in out if x.startswith("| full sample")][0].endswith("| equity leads |")


def selloff_frame():
    n = 4200
    d = make_frame("2010-01-01", n, seed=2, jumps=(400, 800, 1200, 1600, 2000, 2400))
    ret = d["ret"].to_numpy().copy()
    ret[401:422] = 0.01  # fwd21 after first onset = 0.21
    ret[801:822] = 0.001
    d["ret"] = ret
    return d


def test_run_selloff_episodes_and_event_table():
    d = selloff_frame()
    out = []
    BE.run_selloff(d, out)
    text = "\n".join(out)
    assert "(**+0 bp**)" in text and "**6 episodes**" in text
    row21 = [x for x in out if x.startswith("| 21d")][0]
    fwd = d["ret"].rolling(21).sum().shift(-21)
    onsets = d.index[[400, 800, 1200, 1600, 2000, 2400]]
    ev = fwd.reindex(onsets)
    assert row21.startswith(
        f"| 21d | {ev.mean() * 100:+.2f}% | {ev.median() * 100:+.2f}% | "
        f"{(ev > 0).mean() * 100:.0f}% | {fwd.dropna().mean() * 100:+.2f}% | "
        f"{(ev.mean() - fwd.dropna().mean()) * 100:+.2f}pp |"
    )
    assert "| 63d |" in text and "| 126d |" in text and "| 252d |" in text


def test_run_selloff_declusters_within_60_days():
    n = 1500
    d = make_frame("2010-01-01", n, seed=4, jumps=(400, 430, 900))
    # jumps at 400 and 430 are one cluster (<=60 days apart, contiguous elevated dy20 days);
    # 900 starts a new episode
    out = []
    BE.run_selloff(d, out)
    assert "**2 episodes**" in "\n".join(out)


def test_run_selloff_regime_table_needs_more_than_four_episodes():
    out = []
    BE.run_selloff(selloff_frame(), out)
    text = "\n".join(out)
    assert "| negative (bonds hedging) |" in text and "| positive (bonds NOT hedging) |" in text
    out2 = []
    BE.run_selloff(make_frame("2010-01-01", 1500, seed=4, jumps=(400, 800, 1200)), out2)
    assert "negative (bonds hedging)" not in "\n".join(out2)


def test_run_selloff_era_labels():
    out = []
    BE.run_selloff(selloff_frame(), out)
    text = "\n".join(out)
    roll = selloff_frame()["bond_ret"].rolling(252).corr(selloff_frame()["ret"])
    c = roll.loc["2010-01-01":"2019-12-31"].mean()
    label = (
        "YES — negative corr"
        if c < -0.05
        else "NO — positive corr, they fall together"
        if c > 0.05
        else "neither, ~zero"
    )
    assert f"| 2010-2019 | {c:+.3f} | {label} |" in text
    assert "| 1990-1999 | +nan |" in text


def test_run_carry_tables_match_independent_computation():
    d = make_frame("2010-01-01", 1600, seed=8)
    d["spread"] = d["y10y"] - d["y3m"]
    out = []
    BE.run_carry(d, out)
    text = "\n".join(out)
    fwd = d["ret"].rolling(252).sum().shift(-252)
    ok = fwd.notna()
    corr = np.corrcoef(d["y3m"][ok], fwd[ok])[0, 1]
    assert f"| 3m bill LEVEL (funding cost) | {corr:+.3f} |" in text
    corr_sp = np.corrcoef(d["spread"][ok], fwd[ok])[0, 1]
    assert f"| term spread 10y-3m (carry) | {corr_sp:+.3f} |" in text
    qrows = [x for x in out if x.startswith(("| Q1", "| Q2", "| Q3", "| Q4", "| Q5"))]
    assert len(qrows) == 5 and qrows[0].startswith("| Q1 cheapest |")
    assert qrows[4].startswith("| Q5 dearest |")
    counts = [int(r.rsplit("|", 2)[1].replace(",", "")) for r in qrows]
    assert sum(counts) == int(ok.sum())
    assert f"unconditional: {fwd[ok].mean() * 100:+.2f}%" in text
    assert "INVERTED" in text  # sin-shaped 3m rate crosses the 10y


def test_run_carry_no_inversion_line_when_curve_never_inverts():
    d = make_frame("2010-01-01", 1600, seed=8)
    d["y3m"] = np.linspace(0.5, 2.5, len(d)) + np.random.default_rng(1).normal(0, 0.01, len(d))
    d["spread"] = d["y10y"] - d["y3m"]
    out = []
    BE.run_carry(d, out)
    assert "INVERTED" not in "\n".join(out)


def test_run_synthesis_effective_sample_size_numbers():
    idx = pd.bdate_range("1990-01-02", periods=9000)
    d = pd.DataFrame({"x": 0.0}, index=idx)
    yrs = (idx.max() - idx.min()).days / 365.25
    out = []
    BE.run_synthesis(d, out)
    text = "\n".join(out)
    assert f"With 9,000 daily rows spanning {yrs:.0f} years" in text
    assert f"roughly **{yrs:.0f} independent observations**" in text
    assert f"holds about {yrs / 5:.0f} independent years" in text
    assert text.rstrip().endswith("Not investment advice.*")


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_no_flags_prints_help_and_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert BE.main() == 1
    assert "--sampen" in capsys.readouterr().out


def test_main_requires_cached_inputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(BE, "YIELDS", tmp_path / "missing.parquet")
    monkeypatch.setattr(BE, "SPX", tmp_path / "missing2.parquet")
    monkeypatch.setattr(sys, "argv", ["x", "--carry"])
    assert BE.main() == 1
    assert "entropic_yield_curve.py --fetch" in capsys.readouterr().out


@pytest.fixture
def main_env(tmp_path, monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=1400)
    rng = np.random.default_rng(0)
    y = pd.DataFrame(
        {
            "Date": idx,
            "3 Mo": 1 + np.sin(np.linspace(0, 6, len(idx))) + 1.5,
            "2 Yr": 2.0,
            "10 Yr": 3 + rng.normal(0, 0.02, len(idx)).cumsum(),
            "30 Yr": 4.0,
        }
    )
    s = pd.DataFrame(
        {"Date": idx, "spx": 1000 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum())}
    )
    y.to_parquet(tmp_path / "y.parquet")
    s.to_parquet(tmp_path / "s.parquet")
    monkeypatch.setattr(BE, "YIELDS", tmp_path / "y.parquet")
    monkeypatch.setattr(BE, "SPX", tmp_path / "s.parquet")
    monkeypatch.setattr(BE, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(BE, "REPORT", tmp_path / "reports" / "r.md")
    return tmp_path


def test_main_writes_selected_sections_only(main_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--carry"])
    assert BE.main() == 0
    text = (main_env / "reports" / "r.md").read_text()
    assert text.startswith("# Bond yields and equities")
    assert "## 4. Is there a borrow-and-trade" in text
    for absent in ("## 1.", "## 2.", "## 3.", "## 5."):
        assert absent not in text
    assert "(1,399 trading days)" in text
    assert text.endswith("\n") and "-> " in capsys.readouterr().out


def test_main_all_runs_every_section(main_env, monkeypatch):
    monkeypatch.setattr(BE.effective_te, "__defaults__", (10,))
    monkeypatch.setattr(sys, "argv", ["x", "--all"])
    assert BE.main() == 0
    text = (main_env / "reports" / "r.md").read_text()
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
        assert h in text


# ═══ differential / boundary tests added after mutation analysis ═════════════
def sampen_reference(x, m=2, r_mult=0.2):
    """Straightforward O(N^2) sample entropy with population std, <= tolerance."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = len(x)
    r = r_mult * x.std()

    def matches(mm):
        vecs = [x[i : i + mm] for i in range(N - mm + 1)]
        c = 0
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                if i != j and np.max(np.abs(vecs[i] - vecs[j])) <= r:
                    c += 1
        return c

    B, A = matches(m), matches(m + 1)
    return -math.log(A / B)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_sampen_matches_reference_on_noise(seed):
    x = np.random.default_rng(seed).normal(size=60)
    assert BE.sampen(x) == pytest.approx(sampen_reference(x))


def test_sampen_tolerance_uses_population_std():
    # one pair distance sits between 0.2*std(ddof=0) and 0.2*std(ddof=1)
    base = np.array([0.0, 1.0] * 19 + [0.0, 0.0])
    for v in np.linspace(0.080, 0.125, 400):
        x = base.copy()
        x[-1] = v
        r0 = 0.2 * x.std()
        r1 = 0.2 * x.std(ddof=1)
        if r0 < v <= r1:
            assert BE.sampen(x) == pytest.approx(sampen_reference(x))
            return
    pytest.fail("could not build a tolerance-sensitive series")


def test_effective_te_p_value_counts_ties_as_exceedances(monkeypatch):
    calls = iter([1.0] + [1.0] * 9 + [0.5] * 10)
    monkeypatch.setattr(BE, "transfer_entropy", lambda s, d: next(calls))
    raw, eff, p = BE.effective_te(np.arange(5.0), np.arange(5.0), n_sur=10)
    # surrogates: 9 + 1 ... first call is raw; then 10 surrogates -> [1.0]*9 then 0.5
    assert raw == 1.0
    assert p == pytest.approx((9 + 1) / 11)  # ties (>=) count
    assert eff == pytest.approx(1.0 - (9 * 1.0 + 0.5) / 10)


def test_effective_te_defaults_and_constants():
    import inspect

    assert BE.N_SURROGATE == 300 and BE.BINS == 3
    assert inspect.signature(BE.effective_te).parameters["n_sur"].default == 300
    assert inspect.signature(BE.transfer_entropy).parameters["bins"].default == 3


def test_sampen_windows_step_exactly_21_rows():
    d = make_frame("2007-01-01", 900, seed=3)
    s = BE.run_sampen(d, [])
    pos = d.index.get_indexer(s.index)
    assert pos[0] == 251 and set(np.diff(pos)) == {21}


def reference_selloff(d):
    dy20 = d["y10y"].diff(20)
    thr = dy20.quantile(0.95)
    onsets, last = [], None
    for dt in d.index[(dy20 > thr).fillna(False)]:
        if last is None or (dt - last).days > 60:
            onsets.append(dt)
        last = dt
    return thr, onsets


def calendar_frame(n, shifts, seed=0):
    """Daily calendar index; y10y = 3 + cumulative level shifts {day: size}."""
    idx = pd.date_range("2012-01-01", periods=n, freq="D")
    y = np.full(n, 3.0)
    for day, size in shifts.items():
        y[day:] += size
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.01, n)
    d = pd.DataFrame(
        {"ret": ret, "bond_ret": -0.4 * ret + rng.normal(0, 0.004, n), "y10y": y}, index=idx
    )
    d["dy10"] = d["y10y"].diff()
    return d


def episodes(out):
    line = [x for x in out if "episodes" in x][0]
    return int(line.split("**")[-2].split()[0]), line


@pytest.mark.parametrize("second,expected", [(179, 1), (180, 2), (178, 1), (181, 2)])
def test_selloff_declustering_gap_is_sixty_calendar_days(second, expected):
    d = calendar_frame(1500, {100: 1.0, second: 1.0})
    out = []
    BE.run_selloff(d, out)
    assert episodes(out)[0] == expected == len(reference_selloff(d)[1])


def test_selloff_threshold_is_the_95th_percentile_of_20_day_changes():
    sizes = [1.0 + 0.05 * k for k in range(12)]
    shifts = {200 + 100 * k: s for k, s in enumerate(sizes)}
    d = calendar_frame(3000, shifts)
    thr, onsets = reference_selloff(d)
    thr90 = d["y10y"].diff(20).quantile(0.90)
    assert thr > 0 and thr90 == 0 and len(onsets) == 7  # config really separates q95 from q90
    out = []
    BE.run_selloff(d, out)
    n, line = episodes(out)
    assert n == 7 and f"(**+{thr * 100:.0f} bp**)" in line


def test_selloff_regime_table_appears_from_five_episodes():
    four = calendar_frame(2600, dict.fromkeys((400, 800, 1200, 1600), 1.0))
    five = calendar_frame(2600, dict.fromkeys((400, 800, 1200, 1600, 2000), 1.0))
    out4, out5 = [], []
    BE.run_selloff(four, out4)
    BE.run_selloff(five, out5)
    assert "negative (bonds hedging)" not in "\n".join(out4)
    assert "negative (bonds hedging)" in "\n".join(out5)
    assert episodes(out5)[0] == 5


def era_frame():
    idx = pd.bdate_range("2010-01-01", "2026-12-31")
    n = len(idx)
    rng = np.random.default_rng(11)
    ret = rng.normal(0, 0.01, n)
    z = rng.normal(0, 0.01, n)
    rho = np.where(idx < "2020-01-01", 0.03, np.where(idx < "2022-01-01", -0.30, 0.30))
    bond = rho * ret + np.sqrt(1 - rho**2) * z
    d = pd.DataFrame(
        {"ret": ret, "bond_ret": bond, "y10y": 3.0 + 0.001 * rng.normal(size=n).cumsum()}, index=idx
    )
    d["dy10"] = d["y10y"].diff()
    return d


def test_selloff_era_labels_use_plus_minus_five_percent_band():
    d = era_frame()
    out = []
    BE.run_selloff(d, out)
    roll = d["bond_ret"].rolling(252).corr(d["ret"])
    c1, c2, c3 = (
        roll.loc["2010-01-01":"2019-12-31"].mean(),
        roll.loc["2020-01-01":"2021-12-31"].mean(),
        roll.loc["2022-01-01":"2026-12-31"].mean(),
    )
    assert 0.0 < c1 < 0.05 and c2 < -0.05 and c3 > 0.05  # config sits inside every band
    text = "\n".join(out)
    assert f"| 2010-2019 | {c1:+.3f} | neither, ~zero |" in text
    assert f"| 2020-2021 | {c2:+.3f} | YES — negative corr |" in text
    assert f"| 2022-2026 | {c3:+.3f} | NO — positive corr, they fall together |" in text


def sampen_with_target_corr(monkeypatch, d, rho):
    win, step = 252, 21
    r = d["ret"].values
    ends = list(range(win, len(r) + 1, step))
    vols = np.array([np.std(r[e - win : e]) * np.sqrt(252) for e in ends])
    zv = (vols - vols.mean()) / vols.std(ddof=1)
    noise = np.random.default_rng(2).normal(size=len(vols))
    noise = noise - np.polyval(np.polyfit(zv, noise, 1), zv)  # orthogonal to zv
    zn = (noise - noise.mean()) / noise.std(ddof=1)
    target = list(2.0 + rho * zv + math.sqrt(1 - rho**2) * zn)
    it = iter(target)
    monkeypatch.setattr(BE, "sampen", lambda x: next(it))
    return target


@pytest.mark.parametrize(
    "rho,label",
    [
        (0.55, "NOT just inverted volatility"),
        (-0.55, "NOT just inverted volatility"),
        (0.65, "largely the inverse of vol"),
        (-0.65, "largely the inverse of vol"),
    ],
)
def test_sampen_vol_correlation_label_threshold(monkeypatch, rho, label):
    d = make_frame("2007-01-01", 1100, seed=3)
    sampen_with_target_corr(monkeypatch, d, rho)
    out = []
    s = BE.run_sampen(d, out)
    assert s.sampen.corr(s.vol) == pytest.approx(rho, abs=1e-6)
    assert f"{rho:+.3f}" in "\n".join(out) and label in "\n".join(out)


def test_te_state_split_uses_top_20_percent_vol_and_400_row_floor(monkeypatch):
    d = make_frame("2010-01-01", 2100, seed=5)
    monkeypatch.setattr(BE, "effective_te", lambda s, t: (0.0, 0.1, 0.5))
    out = []
    BE.run_te(d, out)
    vol = d["ret"].rolling(63).std()
    hi = vol > vol.quantile(0.80)
    n_turb, n_calm = int(hi.sum()), int((~hi).sum())
    text = "\n".join(out)
    assert f"| turbulent (top 20% vol) | {n_turb:,} |" in text
    assert f"| calm (bottom 80% vol) | {n_calm:,} |" in text


def test_te_full_sample_row_requires_400_rows(monkeypatch):
    monkeypatch.setattr(BE, "effective_te", lambda s, t: (0.0, 0.1, 0.5))
    for n, present in ((400, True), (399, False)):
        out = []
        BE.run_te(make_frame("2010-01-01", n, seed=5), out)
        assert any(x.startswith(f"| full sample 1990-2026 | {n:,} |") for x in out) is present


def test_te_equal_effective_flows_label_equity_leads(monkeypatch):
    monkeypatch.setattr(BE, "effective_te", lambda s, t: (0.0, 0.3, 0.5))
    out = []
    BE.run_te(make_frame("2010-01-01", 500, seed=9), out)
    assert [x for x in out if x.startswith("| full sample")][0].endswith("| equity leads |")


def test_carry_funding_shock_line_matches_independent_computation():
    d = make_frame("2010-01-01", 1600, seed=8)
    out = []
    BE.run_carry(d, out)
    fwd = d["ret"].rolling(252).sum().shift(-252)
    d2 = d.assign(fwd1y=fwd).dropna(subset=["fwd1y"])
    shock = d2["y3m"].diff(63)
    big = shock > shock.quantile(0.90)
    line = [x for x in out if "top-decile" in x][0]
    assert line == (
        f"- days after a top-decile 3m-rate rise (>{shock.quantile(0.90) * 100:.0f}bp/63d): "
        f"mean fwd 1y equity **{d2.loc[big, 'fwd1y'].mean() * 100:+.2f}%** "
        f"(n={int(big.sum()):,})"
    )
    inv_line = [x for x in out if "INVERTED" in x][0]
    inv = d2["spread"] < 0
    assert f"**{d2.loc[inv, 'fwd1y'].mean() * 100:+.2f}%** (n={int(inv.sum()):,})" in inv_line


def test_main_requires_both_inputs(tmp_path, monkeypatch, capsys):
    (tmp_path / "y.parquet").write_text("x")
    monkeypatch.setattr(BE, "YIELDS", tmp_path / "y.parquet")
    monkeypatch.setattr(BE, "SPX", tmp_path / "missing.parquet")
    monkeypatch.setattr(sys, "argv", ["x", "--carry"])
    assert BE.main() == 1
    assert "entropic_yield_curve.py --fetch" in capsys.readouterr().out
    monkeypatch.setattr(BE, "YIELDS", tmp_path / "missing2.parquet")
    monkeypatch.setattr(BE, "SPX", tmp_path / "y.parquet")
    assert BE.main() == 1
