"""factor_combo_test.py -- year-clustered excess-return stats and the factor-combination table."""
import math
import sys

import numpy as np
import pandas as pd
import pytest

import factor_combo_test as FC


# ── _top1_share ───────────────────────────────────────────────────────────────
def test_top1_share_is_best_over_total():
    assert FC._top1_share(pd.Series([1.0, 2.0, 7.0])) == pytest.approx(0.7)
    assert FC._top1_share(pd.Series([5.0, 5.0])) == pytest.approx(0.5)
    assert FC._top1_share(pd.Series([1.0, np.nan, 3.0])) == pytest.approx(0.75)


@pytest.mark.parametrize("vals", [[], [4.0], [np.nan, 4.0], [-1.0, -2.0], [1.0, -1.0], [3.0, -5.0]])
def test_top1_share_undefined_cases(vals):
    assert math.isnan(FC._top1_share(pd.Series(vals, dtype=float)))


# ── year_clustered ────────────────────────────────────────────────────────────
def yearly(medians, per_year=5):
    rows = []
    for k, m in enumerate(medians):
        for j in range(per_year):
            rows.append({"year": 2016 + k, "x": m + (j - per_year // 2) * 0.1})
    return pd.DataFrame(rows)


def test_year_clustered_hand_values():
    out = FC.year_clustered(yearly([1.0, 2.0, 3.0, 4.0]), "x")
    sd = np.std([1, 2, 3, 4], ddof=1)
    assert out["mean"] == pytest.approx(2.5) and out["n_years"] == 4
    assert out["se"] == pytest.approx(sd / 2) and out["t"] == pytest.approx(2.5 / (sd / 2))
    assert out["years_positive"] == 4
    assert out["pooled_mean"] == pytest.approx(2.5) and out["pooled_median"] == pytest.approx(2.5)
    assert out["top1_share"] == pytest.approx(4.2 / 50.0)                 # best 4.2 over total 50


def test_year_clustered_uses_median_within_year_not_mean():
    df = pd.DataFrame({"year": [2020] * 5 + [2021] * 5,
                       "x": [1, 1, 1, 1, 1000.0] + [2, 2, 2, 2, 2]})
    out = FC.year_clustered(df, "x")
    assert out["mean"] == pytest.approx(1.5)                              # medians 1 and 2
    assert out["pooled_mean"] > 100


def test_year_clustered_counts_positive_years_strictly():
    df = pd.DataFrame({"year": [2016, 2017, 2018, 2019], "x": [-1.0, 0.0, 1.0, 2.0]})
    assert FC.year_clustered(df, "x")["years_positive"] == 2               # 0.0 is not positive


def test_year_clustered_degenerate_inputs():
    assert FC.year_clustered(pd.DataFrame(), "x") == {}
    assert FC.year_clustered(pd.DataFrame({"year": [1]}), "x") == {}
    assert FC.year_clustered(yearly([1.0]), "x") == {"n_years": 1}
    assert FC.year_clustered(pd.DataFrame({"year": [2020, 2021], "x": [np.nan, np.nan]}), "x") == {"n_years": 0}


def test_year_clustered_zero_dispersion_has_no_t_stat():
    flat = pd.DataFrame({"year": [2016, 2017, 2018], "x": [2.0, 2.0, 2.0]})
    out = FC.year_clustered(flat, "x")
    assert out["mean"] == 2.0 and math.isnan(out["se"]) and math.isnan(out["t"])
    assert out["years_positive"] == 3


def test_year_clustered_two_years_is_enough():
    out = FC.year_clustered(pd.DataFrame({"year": [1, 2], "x": [1.0, 3.0]}), "x")
    # medians 1 and 3: mean 2, sd sqrt(2), se 1 -> t 2
    assert out["n_years"] == 2 and out["mean"] == 2.0 and out["t"] == pytest.approx(2.0)


# ── run ───────────────────────────────────────────────────────────────────────
def make_panel(tmp_path, n_per_year=80, years=range(2016, 2022), horizon=252, with_liq=True):
    rows = []
    for y in years:
        off = y - 2015
        for i in range(n_per_year):
            half = i % 2 == 0
            rows.append({"year": y, f"xret_T+{horizon}d": off + (1.0 if half else -1.0),
                         "piotroski": half, "roce_plus": i % 4 == 0,
                         "debt_reduction": None, "not_distress": True,
                         "low_asset_growth": half,
                         "log_liquidity": float(i % 10)})
    d = pd.DataFrame(rows)
    d.loc[d.year.isin([2016, 2017, 2018, 2019]), "debt_reduction"] = True
    d["debt_reduction"] = d["debt_reduction"].astype(object)
    if not with_liq:
        d = d.drop(columns="log_liquidity")
    p = tmp_path / "panel.parquet"
    d.to_parquet(p, index=False)
    return p


def test_run_missing_panel_returns_1(tmp_path, capsys):
    assert FC.run(tmp_path / "nope.parquet", 252, None) == 1
    assert "panel not found" in capsys.readouterr().err


def test_run_missing_horizon_column_returns_1(tmp_path, capsys):
    p = make_panel(tmp_path, horizon=252)
    assert FC.run(p, 21, None) == 1
    assert "no column xret_T+21d; have ['xret_T+252d']" in capsys.readouterr().out


def test_run_table_values(tmp_path, capsys):
    p = make_panel(tmp_path)
    rc, tbl = FC.run(p, 252, None)
    assert rc == 0
    by = tbl.set_index("combo")
    assert list(tbl["combo"]) == [n for n, _ in FC.COMBOS]
    base = by.loc["ALL (baseline)"]
    assert base["n"] == 480 and base["mean"] == pytest.approx(3.5) and base["edge"] == pytest.approx(0.0)
    pio = by.loc["piotroski"]
    assert pio["n"] == 240 and pio["mean"] == pytest.approx(4.5) and pio["edge"] == pytest.approx(1.0)
    assert pio["years_positive"] == 6 and bool(pio["testable"])
    sd = np.std(np.arange(2, 8), ddof=1)
    assert pio["t"] == pytest.approx(4.5 / (sd / math.sqrt(6)))
    both = by.loc["piotroski + roce_plus"]
    assert both["n"] == 120 and both["mean"] == pytest.approx(4.5)
    out = capsys.readouterr().out
    assert "universe: 480 symbol-years across 6 years (2016-2021)" in out
    assert "baseline median xret: +3.50pp" in out and "FACTOR COMBINATION TEST — panel.parquet" in out


def test_run_testability_floor_n_and_years(tmp_path, capsys):
    p = make_panel(tmp_path)
    _, tbl = FC.run(p, 252, None)
    by = tbl.set_index("combo")
    # debt_reduction exists in only 4 of 6 years (4*80=320 rows): testable; combined with a
    # 3-year slice it would not be
    assert bool(by.loc["debt_reduction", "testable"]) and by.loc["debt_reduction", "n_years"] == 4
    out = capsys.readouterr().out
    d = pd.read_parquet(p)
    d.loc[d.year.isin([2018, 2019]), "debt_reduction"] = None
    d.to_parquet(p, index=False)
    _, tbl = FC.run(p, 252, None)
    row = tbl.set_index("combo").loc["debt_reduction"]
    assert not bool(row["testable"]) and row["n"] == 160
    assert "UNTESTABLE — below MIN_N/MIN_YEARS" in capsys.readouterr().out


def test_run_min_n_boundary(tmp_path):
    rows = []
    for y in range(2016, 2020):                                         # exactly 4 years
        for i in range(10):                                             # 10 flagged per year -> n=40
            rows.append({"year": y, "xret_T+252d": float(i), "piotroski": True})
        rows.append({"year": y, "xret_T+252d": 0.0, "piotroski": False})
    p = tmp_path / "p.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    _, tbl = FC.run(p, 252, None)
    assert bool(tbl.set_index("combo").loc["piotroski", "testable"])    # n == 40 is enough
    d = pd.DataFrame(rows)
    d = d.drop(d[(d.year == 2019) & (d.piotroski) & (d["xret_T+252d"] == 9.0)].index)
    d.to_parquet(p, index=False)                                        # n == 39
    _, tbl = FC.run(p, 252, None)
    assert not bool(tbl.set_index("combo").loc["piotroski", "testable"])


def test_run_missing_flag_column_is_untestable_not_an_error(tmp_path):
    p = make_panel(tmp_path)
    d = pd.read_parquet(p).drop(columns=["roce_plus"])
    d.to_parquet(p, index=False)
    _, tbl = FC.run(p, 252, None)
    by = tbl.set_index("combo")
    assert by.loc["roce_plus", "n"] == 0 and not bool(by.loc["roce_plus", "testable"])
    assert by.loc["piotroski + roce_plus", "n"] == 0


def test_run_liquidity_gate_filters_rows(tmp_path, capsys):
    p = make_panel(tmp_path)
    _, tbl = FC.run(p, 252, 5.0)
    assert tbl.set_index("combo").loc["ALL (baseline)", "n"] == 240
    assert "liquidity gate: log_liquidity >= 5.0 kept 240/480" in capsys.readouterr().out
    p2 = make_panel(tmp_path, with_liq=False)
    capsys.readouterr()
    _, tbl2 = FC.run(p2, 252, 5.0)                                      # no column: gate silently off
    assert tbl2.set_index("combo").loc["ALL (baseline)", "n"] == 480
    assert "liquidity gate" not in capsys.readouterr().out


def test_run_drops_rows_without_a_return(tmp_path):
    p = make_panel(tmp_path)
    d = pd.read_parquet(p)
    d.loc[d.index[:100], "xret_T+252d"] = np.nan
    d.to_parquet(p, index=False)
    _, tbl = FC.run(p, 252, None)
    assert tbl.set_index("combo").loc["ALL (baseline)", "n"] == 380


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_writes_csv(tmp_path, monkeypatch, capsys):
    p = make_panel(tmp_path)
    out = tmp_path / "sub" / "t.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--panel", str(p), "--out", str(out)])
    assert FC.main() == 0
    assert list(pd.read_csv(out)["combo"])[0] == "ALL (baseline)"
    assert f"→ {out}" in capsys.readouterr().out


def test_main_rejects_unsupported_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--horizon", "30"])
    with pytest.raises(SystemExit):
        FC.main()


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="run() returns a bare 1 on error paths but main() unpacks `rc, tbl = run(...)`, "
                          "so a missing panel crashes with TypeError instead of exiting 1")
def test_main_missing_panel_exits_one_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--panel", str(tmp_path / "missing.parquet")])
    assert FC.main() == 1
