"""darvas_breakouts.py -- fresh-breakout tiering, ranking, HTML fragment, CSV output."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

import darvas_breakouts as DB


# ── scalar helpers ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v,default,expected", [
    (None, None, None), (None, 5, 5), (float("nan"), 7, 7), ("3.5", None, 3.5),
    (2, None, 2.0), ("abc", -1, -1), (np.nan, None, None), ([], 9, 9),
])
def test_num(v, default, expected):
    assert DB._num(v, default) == expected


@pytest.mark.parametrize("v,exp", [
    (True, True), ("TRUE", True), (" yes ", True), ("Y", True), (1, True), ("1", True),
    (False, False), ("no", False), (0, False), (None, False), ("", False), ("nan", False),
])
def test_truthy(v, exp):
    assert DB._truthy(v) is exp


def test_exchange_for():
    assert DB._exchange_for("IN", {"Suffix": ".BO"}) == "BSE"
    assert DB._exchange_for("IN", {"Suffix": "bo"}) == "BSE"
    assert DB._exchange_for("IN", {"Suffix": ".bse"}) == "BSE"
    assert DB._exchange_for("IN", {"Suffix": ".NS"}) == "NSE"
    assert DB._exchange_for("IN", {}) == "NSE"
    assert DB._exchange_for("IN", {"Suffix": None}) == "NSE"
    assert DB._exchange_for("EU", {"Symbol": "SAP.DE"}) == "XETRA"
    assert DB._exchange_for("EU", {"Symbol": "vod.l"}) == "London"
    assert DB._exchange_for("EU", {"Symbol": "XYZ.QQ"}) == "Europe"
    assert DB._exchange_for("EU", {"Symbol": "NODOT"}) == "Europe"
    assert DB._exchange_for("US", {"Symbol": "AAPL"}) == "US"


def test_name_for(monkeypatch):
    assert DB._name_for("US", "AAPL", {"Name": "Apple Inc,US0378331005,x"}) == "Apple Inc"
    assert DB._name_for("US", "AAPL", {"Name": "  "}) == "Apple"
    assert DB._name_for("US", "AAPL", {"Name": "nan"}) == "Apple"
    assert DB._name_for("US", "AAPL", {"Name": "None"}) == "Apple"
    assert DB._name_for("US", "ZZZZ", {}) == "ZZZZ"
    assert DB._name_for("IN", "RELIANCE", {"Name": None}) == "Reliance Industries"
    monkeypatch.setattr(DB, "_eu_meta", lambda: {"SAP.DE": ("SAP SE", "Tech")})
    assert DB._name_for("EU", "SAP.DE", {}) == "SAP SE"
    assert DB._name_for("EU", "OTHER.DE", {}) == "OTHER.DE"
    assert DB._name_for("US", "SAP.DE", {}) == "SAP.DE"          # EU meta only for EU


def test_eu_meta_uses_scan_module_or_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "full_european_market_scan",
                        types.SimpleNamespace(EURO_STOXX_50_META={"A": ("a", "s")}))
    assert DB._eu_meta() == {"A": ("a", "s")}
    monkeypatch.setitem(sys.modules, "full_european_market_scan", None)   # import fails
    assert DB._eu_meta() == {}


def test_latest_scan_picks_last_sorted_and_falls_through(tmp_path, monkeypatch):
    (tmp_path / "b").mkdir()
    for n in ("us_full_scan_20260101.xlsx", "us_full_scan_20260301.xlsx",
              "us_full_scan_20260201.xlsx"):
        (tmp_path / n).write_text("x")
    monkeypatch.setattr(DB, "SCAN_GLOBS", {"US": [str(tmp_path / "b" / "*.xlsx"),
                                                  str(tmp_path / "us_full_scan_*.xlsx")]})
    assert DB._latest_scan("US").endswith("us_full_scan_20260301.xlsx")
    monkeypatch.setattr(DB, "SCAN_GLOBS", {"US": [str(tmp_path / "nope*.xlsx")]})
    assert DB._latest_scan("US") is None


# ── extract ───────────────────────────────────────────────────────────────────
def us_row(sym, pos, chg=1.0, signal="BREAKOUT_BUY", gc="GOLDEN_CROSS", dp=300, gap=5.0,
           ltp=100.0):
    return {"Symbol": sym, "LTP": ltp, "Change%": chg, "Darvas_Signal": signal,
            "Box_Top": 95.0, "Box_Bottom": 90.0, "Upside_to_Top%": 0.0,
            "Position_in_Box%": pos, "Data_Points": dp, "GC_Signal": gc, "DMA_Gap%": gap}


@pytest.fixture
def scan_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(DB, "HERE", tmp_path)
    monkeypatch.setattr(DB, "SCAN_GLOBS", {
        "US": [str(tmp_path / "us_*.xlsx")], "IN": [str(tmp_path / "in_*.xlsx")],
        "EU": [str(tmp_path / "eu_*.xlsx")]})
    return tmp_path


def write_scan(path, rows, sheet="Darvas_Signals"):
    pd.DataFrame(rows).to_excel(path, sheet_name=sheet, index=False)


def test_extract_no_workbook_or_sheet_or_empty(scan_dir, capsys):
    assert DB.extract("US").empty
    assert "no scan workbook found" in capsys.readouterr().out
    write_scan(scan_dir / "us_1.xlsx", [us_row("A", 110)], sheet="Other")
    assert DB.extract("US").empty
    assert "has no Darvas_Signals sheet" in capsys.readouterr().out
    write_scan(scan_dir / "us_1.xlsx", [{"Symbol": "A", "Other": 1}])
    assert DB.extract("US").empty                                # no Darvas_Signal column
    write_scan(scan_dir / "us_1.xlsx", [us_row("A", 110, signal="HOLD")])
    assert DB.extract("US").empty                                # no breakouts at all


def test_tier1_golden_cross_boundary_130(scan_dir):
    write_scan(scan_dir / "us_1.xlsx", [us_row("EDGE", 130), us_row("OVER", 130.01)])
    out = DB.extract("US").set_index("Symbol")
    assert out.loc["EDGE", "Tier"] == 1
    assert "OVER" not in out.index             # tier 3 is dropped once any fresh name exists
    assert out.loc["EDGE", "Golden_Cross"] == "YES"


def test_tier2_conditions_each_gate(scan_dir):
    rows = [
        us_row("T2", 110, gc="ABOVE_200DMA"),
        us_row("LOW", 99.9, gc="ABOVE_200DMA"),             # below 100 -> tier 3
        us_row("LOWOK", 100, gc="ABOVE_200DMA"),             # inclusive 100
        us_row("HIGHOK", 120, gc="ABOVE_200DMA"),            # inclusive 120
        us_row("HIGH", 120.01, gc="ABOVE_200DMA"),           # tier 3
        us_row("DOWN", 110, gc="ABOVE_200DMA", chg=0.0),     # needs chg > 0
        us_row("SHORT", 110, gc="ABOVE_200DMA", dp=249),     # needs >= 250 bars
        us_row("BARS", 110, gc="ABOVE_200DMA", dp=250),
        us_row("NOTREND", 110, gc="BELOW"),                  # not an uptrend
    ]
    write_scan(scan_dir / "us_1.xlsx", rows)
    out = DB.extract("US").set_index("Symbol")
    tiers = out["Tier"].to_dict()
    for s in ("T2", "LOWOK", "HIGHOK", "BARS"):
        assert tiers[s] == 2, s
    for s in ("LOW", "HIGH", "DOWN", "SHORT", "NOTREND"):
        assert s not in tiers, s                            # only tier 1/2 survive when any exist


def test_fallback_when_no_fresh_names_is_closest_to_box_and_capped(scan_dir):
    rows = [us_row(f"S{i}", 140 + i * 10, gc="BELOW") for i in range(15)]
    rows.append(us_row("NEAR", 101.0, gc="BELOW", chg=-1.0))     # tier 3 but nearest the box
    write_scan(scan_dir / "us_1.xlsx", rows)
    out = DB.extract("US")
    assert len(out) == 12                                         # head(12)
    assert out["Tier"].eq(3).all()
    assert out["Symbol"].iloc[0] == "NEAR"
    assert "_ext" not in out.columns


def test_skips_blank_symbols_and_case_insensitive_signal(scan_dir):
    write_scan(scan_dir / "us_1.xlsx", [
        us_row("   ", 110), us_row("ok", 110, signal="breakout_buy")])
    out = DB.extract("US")
    assert list(out["Symbol"]) == ["ok"]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="a blank Symbol cell reads back from Excel as NaN; "
                          "str(NaN or '') is 'nan', so the row is kept as symbol 'nan'")
def test_blank_symbol_cell_is_skipped(scan_dir):
    write_scan(scan_dir / "us_1.xlsx", [us_row("", 110), us_row("ok", 110)])
    assert list(DB.extract("US")["Symbol"]) == ["ok"]


def test_ordering_tier_then_score_and_score_formula(scan_dir):
    rows = [us_row("T1LOW", 105, chg=1.0, gap=0.0),
            us_row("T1HIGH", 105, chg=5.0, gap=10.0),
            us_row("T2", 110, gc="ABOVE_200DMA", chg=1.0, gap=0.0)]
    write_scan(scan_dir / "us_1.xlsx", rows)
    out = DB.extract("US")
    assert list(out["Symbol"]) == ["T1HIGH", "T1LOW", "T2"]
    # prox 100 + chg + 0.5*gap + 8 golden + 4 uptrend
    assert out["_score"].iloc[0] == pytest.approx(100 + 5 + 5 + 8 + 4)
    assert out["_score"].iloc[1] == pytest.approx(100 + 1 + 0 + 8 + 4)
    # T2: prox = 100 - |110-105| = 95, uptrend but not golden
    assert out["_score"].iloc[2] == pytest.approx(95 + 1 + 0 + 4)


def test_score_proximity_peaks_at_105_and_floors_at_zero(scan_dir):
    rows = [us_row("A", 105, gc="BELOW", chg=0, gap=0), us_row("B", 400, gc="BELOW", chg=0, gap=0)]
    rows[0]["Position_in_Box%"] = 105
    write_scan(scan_dir / "us_1.xlsx", rows)
    out = DB.extract("US").set_index("Symbol")
    assert out.loc["A", "_score"] == 100.0 and out.loc["B", "_score"] == 0.0


def test_missing_position_is_not_fresh(scan_dir):
    r = us_row("NOPOS", 0)
    r["Position_in_Box%"] = np.nan
    write_scan(scan_dir / "us_1.xlsx", [r])
    out = DB.extract("US")
    assert out["Tier"].iloc[0] == 3 and pd.isna(out["Position_in_Box%"].iloc[0])


def test_india_dma_flag_encoding_and_suffix(scan_dir):
    rows = [{"Symbol": "RELIANCE", "Suffix": ".BO", "LTP": 100.0, "Change%": 1.0,
             "Darvas_Signal": "BREAKOUT_BUY", "Box_Top": 1, "Box_Bottom": 1,
             "Upside_to_Top%": 0.0, "Position_in_Box%": 110, "Data_Points": 300,
             "DMA50_above_200": "True"},
            {"Symbol": "TCS", "Suffix": ".NS", "LTP": 100.0, "Change%": 1.0,
             "Darvas_Signal": "BREAKOUT_BUY", "Box_Top": 1, "Box_Bottom": 1,
             "Upside_to_Top%": 0.0, "Position_in_Box%": 110, "Data_Points": 300,
             "DMA50_above_200": "False"}]
    write_scan(scan_dir / "in_1.xlsx", rows)
    out = DB.extract("IN")
    assert list(out["Symbol"]) == ["RELIANCE"]                    # TCS: no uptrend -> tier 3, dropped
    r = out.iloc[0]
    assert r["Exchange"] == "BSE" and r["Uptrend"] == "YES" and r["Golden_Cross"] == ""
    assert r["Name"] == "Reliance Industries" and pd.isna(r["50/200_Gap%"])
    assert r["Tier"] == 2


def test_eu_distance_and_trend_encodings(scan_dir):
    base = {"LTP": 10.0, "Change%": 1.0, "Darvas_Signal": "BREAKOUT_BUY", "Box_Top": 1,
            "Box_Bottom": 1, "Upside_to_Top%": 0.0, "Position_in_Box%": 110}
    rows = [dict(base, Symbol="DIST.DE", **{"Distance_to_200MA%": 4.0, "Trend_Signal": "down"}),
            dict(base, Symbol="NEG.DE", **{"Distance_to_200MA%": -4.0, "Trend_Signal": "down"}),
            dict(base, Symbol="ZERO.DE", **{"Distance_to_200MA%": 0.0, "Trend_Signal": "down"}),
            dict(base, Symbol="TREND.PA", **{"Distance_to_200MA%": -1.0, "Trend_Signal": "Uptrend"}),
            dict(base, Symbol="ABOVE.L", **{"Distance_to_200MA%": -1.0, "Trend_Signal": "Above 200"})]
    write_scan(scan_dir / "eu_1.xlsx", rows)
    out = DB.extract("EU").set_index("Symbol")
    # no Data_Points column -> 9999 => the >=250 bars gate passes
    assert set(out.index) == {"DIST.DE", "TREND.PA", "ABOVE.L"}
    assert out.loc["DIST.DE", "50/200_Gap%"] == 4.0                # gap read from distance col
    assert out.loc["DIST.DE", "Exchange"] == "XETRA" and out.loc["ABOVE.L", "Exchange"] == "London"
    assert (out["Tier"] == 2).all()


# ── HTML ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v,nd,exp", [(None, 2, "n/a"), (float("nan"), 2, "n/a"),
                                      (1234.567, 2, "1,234.57"), (1234.567, 0, "1,235")])
def test_fmt(v, nd, exp):
    assert DB._fmt(v, nd) == exp


def sample_df():
    return pd.DataFrame([
        {"Tier": 1, "Symbol": "AAA", "Name": "Alpha", "Exchange": "US", "LTP": 1234.5,
         "Change%": 2.5, "Position_in_Box%": 104.4, "Upside_to_Top%": 1.26,
         "50/200_Gap%": 3.14159, "Golden_Cross": "YES"},
        {"Tier": 2, "Symbol": "BBB", "Name": "Beta", "Exchange": "US", "LTP": None,
         "Change%": -1.0, "Position_in_Box%": 110.0, "Upside_to_Top%": 0.0,
         "50/200_Gap%": None, "Golden_Cross": ""},
        {"Tier": 3, "Symbol": "CCC", "Name": "Gamma", "Exchange": "US", "LTP": 5.0,
         "Change%": 0.0, "Position_in_Box%": 200.0, "Upside_to_Top%": 0.0,
         "50/200_Gap%": -2.0, "Golden_Cross": ""},
    ])


def test_rows_html_formatting():
    html = DB.rows_html(sample_df(), "IN")
    rows = html.split("</tr>")[:-1]
    assert len(rows) == 3
    assert "🥇 T1" in rows[0] and "🥈 T2" in rows[1] and "<td>•</td>" in rows[2]
    assert "₹1,234.50" in rows[0] and "+2.50%" in rows[0] and "#2e7d32" in rows[0]
    assert "104%" in rows[0] and "1.3%" in rows[0] and "+3.1%" in rows[0] and "YES" in rows[0]
    assert "₹n/a" in rows[1] and "-1.00%" in rows[1] and "#c62828" in rows[1]
    assert "+0.00%" in rows[2] and "#2e7d32" in rows[2] and "-2.0%" in rows[2]   # 0 counts as green
    assert "$" in DB.rows_html(sample_df(), "US") and "€" in DB.rows_html(sample_df(), "EU")


def test_rows_html_missing_gap_all_none_shows_na():
    df = sample_df()
    df["50/200_Gap%"] = [None, None, None]
    assert DB.rows_html(df, "US").count(">n/a<") == 3


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="a missing gap in a column that also has numbers becomes NaN, "
                          "and `gap is None` misses it, rendering '+nan%'")
def test_rows_html_missing_gap_mixed_column_shows_na():
    html = DB.rows_html(sample_df(), "US")           # BBB has a missing gap next to numeric ones
    assert "nan" not in html


def test_to_html_empty_and_counts():
    empty = DB.to_html(pd.DataFrame(), "IN")
    assert "India" in empty and "No fresh breakouts" in empty and "<table" not in empty
    full = DB.to_html(sample_df(), "EU")
    assert "Europe" in full and "(1 Tier-1 · 1 Tier-2)" in full
    assert full.count("<tr><td>") == 3 and "<th>Tier</th>" in full
    assert "US" in DB.to_html(sample_df(), "US")


# ── build / main ──────────────────────────────────────────────────────────────
def test_build_writes_csv_without_score_and_lowercases_market(scan_dir, capsys):
    write_scan(scan_dir / "us_1.xlsx", [us_row("AAPL", 105)])
    html, df = DB.build("us")
    assert "AAPL" in html and not df.empty
    files = list((scan_dir / "darvas_breakouts").glob("us_darvas_breakouts_*.csv"))
    assert len(files) == 1
    saved = pd.read_csv(files[0])
    assert "_score" not in saved.columns and list(saved["Symbol"]) == ["AAPL"]
    assert "1 fresh breakouts" in capsys.readouterr().out


def test_build_write_csv_false_or_empty_writes_nothing(scan_dir):
    write_scan(scan_dir / "us_1.xlsx", [us_row("AAPL", 105)])
    DB.build("US", write_csv=False)
    assert not (scan_dir / "darvas_breakouts").exists()
    html, df = DB.build("IN")                                   # no IN workbook
    assert df.empty and "No fresh breakouts" in html
    assert not (scan_dir / "darvas_breakouts").exists()


def test_main_html_only_and_summary(scan_dir, monkeypatch, capsys):
    write_scan(scan_dir / "us_1.xlsx", [us_row("AAPL", 105)])
    monkeypatch.setattr(sys, "argv", ["x", "--market", "US", "--html-only"])
    DB.main()
    out = capsys.readouterr().out
    assert out.startswith("<h3>") and "AAPL" in out
    assert not (scan_dir / "darvas_breakouts").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--market", "US"])
    DB.main()
    out = capsys.readouterr().out
    assert "US DARVAS BREAKOUTS" in out and "1 fresh (1 Tier-1, 0 Tier-2)" in out
    monkeypatch.setattr(sys, "argv", ["x", "--market", "EU"])
    DB.main()
    assert "No fresh breakouts." in capsys.readouterr().out
