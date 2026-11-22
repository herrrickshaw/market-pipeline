"""factor_zscore_panel.py -- winsorising, per-market and sector-relative z-scores, coverage report."""
import numpy as np
import pandas as pd
import pytest

import factor_zscore_panel as FZ


@pytest.fixture(autouse=True)
def _restore_fund_path(monkeypatch):
    # the module (and its imports) rebind fst.FUND_PATH as a side effect; restore afterwards
    monkeypatch.setattr(FZ.fst, "FUND_PATH", FZ.fst.FUND_PATH)


# ── _winsorize ────────────────────────────────────────────────────────────────
def test_winsorize_clips_to_1st_and_99th_percentile():
    s = pd.Series(np.arange(1.0, 101.0))
    out = FZ._winsorize(s)
    assert out.min() == pytest.approx(s.quantile(0.01)) and out.max() == pytest.approx(s.quantile(0.99))
    assert out.iloc[49] == 50.0                                  # interior untouched
    assert s.iloc[0] == 1.0                                      # input untouched


def test_winsorize_needs_20_non_null_values():
    vals = list(np.arange(1.0, 19.0)) + [10_000.0]               # 19 valid
    assert FZ._winsorize(pd.Series(vals)).max() == 10_000.0
    vals20 = list(np.arange(1.0, 20.0)) + [10_000.0]             # 20 valid
    assert FZ._winsorize(pd.Series(vals20)).max() < 10_000.0


def test_winsorize_quantiles_ignore_nan_and_nan_stays():
    s = pd.Series([np.nan] * 30 + list(np.arange(1.0, 21.0)))
    out = FZ._winsorize(s)
    assert out.isna().sum() == 30
    assert out.max() == pytest.approx(pd.Series(np.arange(1.0, 21.0)).quantile(0.99))


def test_winsorize_custom_bounds():
    s = pd.Series(np.arange(1.0, 101.0))
    out = FZ._winsorize(s, lower=0.10, upper=0.90)
    assert out.min() == pytest.approx(s.quantile(0.10)) and out.max() == pytest.approx(s.quantile(0.90))


# ── add_zscores ───────────────────────────────────────────────────────────────
def panel(n_us=30, n_in=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mkt, n, scale in (("us", n_us, 1.0), ("india", n_in, 100.0)):
        for i in range(n):
            rows.append({"market": mkt, "ticker": f"{mkt}{i}",
                         "fy_end": pd.Timestamp("2020-12-31") + pd.DateOffset(years=i % 3),
                         "roe": float(rng.normal(10, 3)) * scale})
    return pd.DataFrame(rows)


def test_zscore_is_per_market_winsorised_standardisation():
    df = panel()
    out = FZ.add_zscores(df, ["roe"])
    for mkt in ("us", "india"):
        sub = out[out["market"] == mkt]
        raw = df.loc[sub.index, "roe"]
        w = raw.clip(raw.quantile(0.01), raw.quantile(0.99))
        expected = (w - w.mean()) / w.std()
        assert sub["z_roe"].to_numpy() == pytest.approx(expected.to_numpy())
        assert sub["z_roe"].mean() == pytest.approx(0.0, abs=1e-9)
        assert sub["z_roe"].std() == pytest.approx(1.0)               # ddof=1
    assert out["has_roe"].all()


def test_zscore_adds_fy_year_and_does_not_mutate_input():
    df = panel()
    out = FZ.add_zscores(df, ["roe"])
    assert "fy_year" not in df.columns and "z_roe" not in df.columns
    assert set(out["fy_year"]) == {2020, 2021, 2022}


def test_non_finite_values_have_no_zscore_and_do_not_skew_stats():
    df = panel(30, 0)
    clean = FZ.add_zscores(df.copy(), ["roe"])
    df2 = df.copy()
    df2.loc[0, "roe"] = np.inf
    df2.loc[1, "roe"] = -np.inf
    df2.loc[2, "roe"] = np.nan
    out = FZ.add_zscores(df2, ["roe"])
    assert list(out["has_roe"].iloc[:4]) == [False, False, False, True]
    assert out["z_roe"].iloc[:3].isna().all() and out["z_roe"].iloc[3:].notna().all()
    assert out["z_roe"].iloc[3:].mean() == pytest.approx(0.0, abs=1e-9)
    assert clean["z_roe"].notna().all()


def test_constant_factor_gets_nan_zscore():
    df = panel(10, 10)
    df["flat"] = 5.0
    out = FZ.add_zscores(df, ["flat"])
    assert out["has_flat"].all() and out["z_flat"].isna().all()


def test_missing_factor_column_is_skipped_with_message(capsys):
    out = FZ.add_zscores(panel(5, 5), ["roe", "ghost"])
    assert "[skip] ghost: not present in panel for this market" in capsys.readouterr().out
    assert "z_ghost" not in out.columns and "z_roe" in out.columns


def test_markets_are_standardised_independently():
    df = panel()
    out = FZ.add_zscores(df, ["roe"])
    # india raw values are ~100x larger, yet both markets centre on zero
    assert out.groupby("market")["z_roe"].mean().abs().max() < 1e-9


def sector_panel(cell_sizes):
    rows = []
    rng = np.random.default_rng(3)
    for (ind, yr), n in cell_sizes.items():
        for i in range(n):
            rows.append({"market": "us", "ticker": f"{ind}{yr}{i}", "industry_group": ind,
                         "fy_end": pd.Timestamp(f"{yr}-12-31"),
                         "pb_ratio": float(rng.normal(3, 1)), "roe": float(rng.normal(10, 2))})
    return pd.DataFrame(rows)


def test_sector_z_requires_min_cell_of_five_and_only_for_relative_factors():
    df = sector_panel({("Tech", 2020): 5, ("Banks", 2020): 4, ("Tech", 2021): 6})
    out = FZ.add_zscores(df, ["pb_ratio", "roe"])
    assert "sector_z_roe" not in out.columns                     # not a sector-relative factor
    tech20 = out[(out["industry_group"] == "Tech") & (out["fy_year"] == 2020)]
    banks = out[out["industry_group"] == "Banks"]
    assert tech20["sector_z_pb_ratio"].notna().all()             # n == 5 qualifies
    assert banks["sector_z_pb_ratio"].isna().all()               # n == 4 does not
    w = df.loc[tech20.index, "pb_ratio"]
    # winsorising 15 values (<20 valid) is a no-op, so the cell z-score is plain
    assert tech20["sector_z_pb_ratio"].to_numpy() == pytest.approx(
        ((w - w.mean()) / w.std()).to_numpy())


def test_sector_z_absent_without_industry_group():
    df = sector_panel({("Tech", 2020): 8}).drop(columns="industry_group")
    assert "sector_z_pb_ratio" not in FZ.add_zscores(df, ["pb_ratio"]).columns


def test_sector_cells_split_by_year():
    df = sector_panel({("Tech", 2020): 5, ("Tech", 2021): 4})
    out = FZ.add_zscores(df, ["pb_ratio"])
    assert out.loc[out["fy_year"] == 2020, "sector_z_pb_ratio"].notna().all()
    assert out.loc[out["fy_year"] == 2021, "sector_z_pb_ratio"].isna().all()


# ── report_coverage ───────────────────────────────────────────────────────────
def test_report_coverage_flags_below_15_percent(capsys):
    df = pd.DataFrame({"market": ["us"] * 20 + ["india"] * 20,
                       "ticker": [f"t{i}" for i in range(40)],
                       "has_roe": [True] * 20 + [True] * 2 + [False] * 18,
                       "has_pe_ttm": [True] * 3 + [False] * 17 + [True] * 20})
    FZ.report_coverage(df, ["roe", "pe_ttm", "absent"])
    lines = capsys.readouterr().out.splitlines()
    us_roe = [x for x in lines if "roe" in x][0]
    assert "100.0% populated" in us_roe and "BELOW" not in us_roe
    in_roe = [x for x in lines if "roe" in x][1]
    assert "10.0% populated  <-- BELOW 15% FLOOR" in in_roe
    us_pe = [x for x in lines if "pe_ttm" in x][0]
    assert "15.0% populated" in us_pe and "BELOW" not in us_pe     # exactly 15% passes
    assert not any("absent" in x for x in lines)
    assert "us (n=20 filings, 20 tickers)" in "\n".join(lines)
    assert "india (n=20 filings, 20 tickers)" in "\n".join(lines)


# ── attach_industry ───────────────────────────────────────────────────────────
def test_attach_industry_joins_by_market_specific_filter(monkeypatch, capsys):
    damo = pd.DataFrame({
        "Ticker": ["AAPL", "AAPL", "RELIANCE", "RELIANCE", "MSFT", "TCS"],
        "Exchange": ["NasdaqGS", "NYSE", "NSEI", "BSE", "NasdaqGS", "NSEI"],
        "Country": ["United States", "United States", "India", "India", "United States", "India"],
        "Industry Group": ["Hardware", "Hardware-dup", "Oil", "Oil-BSE", "Software", "IT"],
    })
    monkeypatch.setattr(FZ.pd, "read_parquet", lambda p: damo)
    panel = pd.DataFrame({"market": ["us", "us", "india", "india", "india"],
                          "ticker": ["AAPL", "ZZZ", "RELIANCE", "TCS", "NOPE"]})
    out = FZ.attach_industry(panel)
    got = dict(zip(out["ticker"], out["industry_group"]))
    assert got["AAPL"] == "Hardware"                      # drop_duplicates keeps first
    assert got["RELIANCE"] == "Oil"                       # NSEI rows only, not BSE
    assert got["TCS"] == "IT" and pd.isna(got["ZZZ"]) and pd.isna(got["NOPE"])
    assert len(out) == 5 and "Industry Group" not in out.columns
    text = capsys.readouterr().out
    assert "us: 1/2 tickers matched to a Damodaran Industry Group (50.0%)" in text
    assert "india: 2/3 tickers matched to a Damodaran Industry Group (66.7%)" in text


def test_attach_industry_us_rows_never_take_indian_industry(monkeypatch):
    damo = pd.DataFrame({"Ticker": ["X"], "Exchange": ["NSEI"], "Country": ["India"],
                         "Industry Group": ["Steel"]})
    monkeypatch.setattr(FZ.pd, "read_parquet", lambda p: damo)
    out = FZ.attach_industry(pd.DataFrame({"market": ["us", "india"], "ticker": ["X", "X"]}))
    assert out.set_index("market")["industry_group"].to_dict()["india"] == "Steel"
    assert pd.isna(out.set_index("market")["industry_group"].to_dict()["us"])


# ── panel builders (heavy loaders faked) ──────────────────────────────────────
def test_build_us_panel_wiring(monkeypatch):
    calls = []
    monkeypatch.setattr(FZ.fst, "load_fundamentals", lambda: calls.append(("fund", FZ.fst.FUND_PATH)) or "F")
    monkeypatch.setattr(FZ.fst, "load_ohlcv", lambda: "O")
    monkeypatch.setattr(FZ.fst, "compute_fundamental_screens", lambda f: calls.append(("score", f)) or "S")
    monkeypatch.setattr(FZ.fst, "attach_market_cap",
                        lambda s, o: calls.append(("mc", s, o)) or pd.DataFrame({"t": [1]}))
    out = FZ.build_us_panel()
    assert calls == [("fund", FZ.US_FUND_PATH), ("score", "F"), ("mc", "S", "O")]
    assert list(out["market"]) == ["us"]


def test_build_india_panel_wiring(monkeypatch):
    order = []
    monkeypatch.setattr(FZ.fst, "load_fundamentals", lambda: order.append(FZ.fst.FUND_PATH) or "F")
    monkeypatch.setattr(FZ.fst_in, "load_ohlcv_in", lambda: "O")
    monkeypatch.setattr(FZ.fst_in_full, "merge_backup_fields", lambda f: order.append("merge") or "F2")
    monkeypatch.setattr(FZ.fst, "compute_fundamental_screens", lambda f: order.append(("score", f)) or "S")
    monkeypatch.setattr(FZ.fst, "attach_market_cap", lambda s, o: pd.DataFrame({"t": [1]}))
    monkeypatch.setattr(FZ.fst_in_full, "fix_currency_scaled_thresholds",
                        lambda m: order.append("fix") or m)
    monkeypatch.setattr(FZ.fst_in_full, "add_working_capital_ratios",
                        lambda m: order.append("wc") or m)
    out = FZ.build_india_panel()
    assert order == [FZ.fst_in_full.FUND_PATH_IN, "merge", ("score", "F2"), "fix", "wc"]
    assert list(out["market"]) == ["india"]
