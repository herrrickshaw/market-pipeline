"""build_business_valuation_report.py -- P&L partitioning and the workbook it writes
(sheet inventory, assumption cells, DCF formulas wired to the tax sheet)."""


import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

import build_business_valuation_report as BV


def pnl_frame():
    return pd.DataFrame(
        {
            "symbol": ["H1", "H2", "H3", "S1", "S2", "S3", "NA"],
            "market": ["IN", "IN", "US", "US", "US", "IN", "US"],
            "status": ["held", "held", "held", "sold", "sold", "sold", "held"],
            "basis": [
                "live",
                "live",
                "live",
                "realized-tax-filing",
                "realized-tax-filing",
                "legacy",
                "live",
            ],
            "entry_price": [100.0, 50.0, 10.0, 20.0, 30.0, 5.0, 1.0],
            "ltp": [120.0, 40.0, 11.0, 10.0, 33.0, 5.0, 1.0],
            "pct_since": [20.0, -20.0, 10.0, -50.0, 10.0, 0.0, None],
        }
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(BV.WP, "build", lambda market=None, status=None: pnl_frame())
    monkeypatch.setattr(BV, "OUT", tmp_path / "out" / "report.xlsx")
    monkeypatch.setattr(BV, "__file__", str(tmp_path / "build_business_valuation_report.py"))
    (tmp_path / "reports").mkdir()
    pd.DataFrame({"desk": ["IN", "US"], "aum_musd": [1, 10], "net_edge_bps": [40, 5]}).to_csv(
        tmp_path / "reports" / "execution_capacity.csv", index=False
    )
    return tmp_path


# ── load_pnl ──────────────────────────────────────────────────────────────────
def test_load_pnl_partitions_and_scales(env):
    d, held, sold_real, sold_unv = BV.load_pnl()
    assert len(d) == 6 and "NA" not in set(d["symbol"])  # NaN return dropped
    assert list(held["symbol"]) == ["H1", "H3", "H2"]  # best first
    assert list(held["pct_since"]) == [0.2, 0.1, -0.2]  # percent -> fraction
    assert list(sold_real["symbol"]) == ["S2", "S1"] and list(sold_unv["symbol"]) == ["S3"]
    assert pnl_frame()["pct_since"].iloc[0] == 20.0  # source frame untouched


# ── styling helpers ───────────────────────────────────────────────────────────
def test_write_df_formats_columns_and_returns_next_row():
    wb = Workbook()
    ws = wb.active
    df = pd.DataFrame({"a": ["x", "y"], "pct": [0.1, -0.2], "usd": [1234.5, 2.0]})
    nxt = BV._write_df(ws, df, 3, pct_cols=("pct",), money_cols=("usd",))
    assert nxt == 3 + 1 + 2
    assert [ws.cell(row=3, column=c).value for c in (1, 2, 3)] == ["a", "pct", "usd"]
    assert ws.cell(row=3, column=1).fill.fgColor.rgb.endswith(BV.NAVY)
    assert ws.cell(row=4, column=2).number_format == "+0.0%;-0.0%"
    assert ws.cell(row=4, column=3).number_format == "#,##0.00"
    assert (
        ws.cell(row=5, column=1).value == "y"
        and ws.cell(row=5, column=1).number_format == "General"
    )
    assert BV._write_df(ws, df.iloc[0:0], 20) == 21  # header only


def test_note_merges_and_sizes_rows():
    ws = Workbook().active
    nxt = BV._note(ws, "n" * 250, 2, span=6)
    assert nxt == 3 and ws.row_dimensions[2].height == 15 * (1 + 250 // 110)
    assert "A2:F2" in {str(r) for r in ws.merged_cells.ranges}
    BV._note(ws, "short", 5, col=3, span=2, font=BV.WARN_FONT)
    assert ws.cell(row=5, column=3).font.b is True and "C5:D5" in {
        str(r) for r in ws.merged_cells.ranges
    }


def test_title_assume_and_autofit():
    ws = Workbook().active
    assert BV._title(ws, "T", row=4) == 5 and ws.cell(row=4, column=1).font.sz == 16
    assert BV._assume(ws, 7, "label", 0.2, "0.0%") == 8
    c = ws.cell(row=7, column=3)
    assert c.value == 0.2 and c.number_format == "0.0%" and c.fill.fgColor.rgb.endswith("FFF6D8")
    BV._assume(ws, 8, "plain", 5)
    assert ws.cell(row=8, column=3).number_format == "General"
    BV._autofit(ws, [10, 20])
    assert ws.column_dimensions["A"].width == 10 and ws.column_dimensions["B"].width == 20


# ── sheets ────────────────────────────────────────────────────────────────────
def test_cover_snapshot_numbers(env):
    d, held, real, unv = BV.load_pnl()
    wb = Workbook()
    ws = BV.sheet_cover(wb, held, real, unv, d)
    assert ws.title == "Cover & Methodology"
    rows = {
        ws.cell(row=r, column=1).value: ws.cell(row=r, column=3).value
        for r in range(1, ws.max_row + 1)
    }
    assert rows["Total watchlist positions (with computable P&L)"] == 6
    assert rows["Held positions (actual capital at risk, unrealized)"] == 3
    assert rows["Held — mean / median return"] == "+3.3% / +10.0%"
    assert rows["Held — win rate"] == "66.7%"
    assert rows["Closed positions — REAL exit price (Schedule FA)"] == 2
    assert rows["Closed, real exit — mean / median return"] == "-20.0% / -20.0%"
    assert rows["Closed, real exit — win rate"] == "50.0%"
    assert rows["Closed positions — no exit price on file (legacy, unverified)"] == 1


def test_cover_without_real_exits_says_na(env):
    d, held, real, unv = BV.load_pnl()
    ws = BV.sheet_cover(Workbook(), held, real.iloc[0:0], unv, d)
    vals = [ws.cell(row=r, column=3).value for r in range(1, ws.max_row + 1)]
    assert vals.count("n/a") == 2


def test_actual_pnl_sheet_layout(env):
    d, held, real, unv = BV.load_pnl()
    ws = BV.sheet_actual_pnl(Workbook(), held, real, unv, d)
    cells = {ws.cell(row=r, column=1).value for r in range(1, ws.max_row + 1)}
    assert {"Held — by market", "Held — full detail, sorted by return"} <= cells
    assert "Closed — REAL exit price (Schedule FA tax filing)" in cells
    assert any(str(v).startswith("Closed — NO exit price") for v in cells if v)
    r_head = next(r for r in range(1, ws.max_row + 1) if ws.cell(row=r, column=2).value == "n")
    assert [ws.cell(row=r_head, column=c).value for c in (1, 2, 3, 4)] == [
        "market",
        "n",
        "mean_return",
        "median_return",
    ]
    assert [ws.cell(row=r_head + 1, column=c).value for c in (1, 2)] == ["IN", 2]
    assert ws.cell(row=r_head + 1, column=3).value == 0.0  # (0.2 - 0.2) / 2


def test_signal_quality_and_capacity_sheets(env):
    wb = Workbook()
    ws = BV.sheet_signal_quality(wb)
    vals = [[c.value for c in row] for row in ws.iter_rows()]
    assert any(row[:3] == ["Golden Cross", "India", 35376] for row in vals)
    assert any(row[:3] == ["Golden Cross", "USA", 29941] for row in vals)
    cap = BV.sheet_capacity(wb)
    flat = [c.value for row in cap.iter_rows() for c in row if c.value is not None]
    assert "net_edge_bps" in flat and 40 in flat  # from the CSV
    assert "capacity_$M_AUM" in flat and "~0 (already dead at $1M)" in flat


def test_tax_sheet_assumptions_and_row_map(env):
    ws = BV.sheet_tax_and_costs(Workbook())
    t = ws._tax_rows
    assert set(t) == {
        "in_stcg",
        "in_ltcg",
        "in_exempt",
        "in_stt",
        "us_stcg",
        "us_ltcg",
        "lev",
        "borrow",
    }
    assert ws.cell(row=t["in_stcg"], column=3).value == 0.20
    assert ws.cell(row=t["in_ltcg"], column=3).value == 0.125
    assert ws.cell(row=t["in_exempt"], column=3).value == 125_000
    assert ws.cell(row=t["in_stt"], column=3).value == 0.002
    assert (
        ws.cell(row=t["us_stcg"], column=3).value == 0.30
        and ws.cell(row=t["us_ltcg"], column=3).value == 0.15
    )
    assert (
        ws.cell(row=t["lev"], column=3).value == 0.0
        and ws.cell(row=t["borrow"], column=3).value == 0.09
    )
    # each map entry points at the row carrying the matching label
    assert "LTCG rate (equity" in ws.cell(row=t["in_ltcg"], column=1).value
    assert "Leverage ratio" in ws.cell(row=t["lev"], column=1).value
    assert "Cost of borrowed capital" in ws.cell(row=t["borrow"], column=1).value


def test_dcf_formulas_reference_assumptions_and_tax_sheet(env):
    d, held, real, unv = BV.load_pnl()
    wb = Workbook()
    tax = BV.sheet_tax_and_costs(wb)
    ws = BV.sheet_dcf(wb, held, tax)
    t = tax._tax_rows
    cells = {
        (c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value is not None
    }
    labels = {v: r for (r, col), v in cells.items() if col == 1 and isinstance(v, str)}
    r_cap = next(
        r for (r, c), v in cells.items() if c == 1 and str(v).startswith("Deployable capital")
    )
    assert (
        cells[(r_cap, 3)] == 1_000_000
        and cells[(r_cap + 1, 3)] == 0.02
        and cells[(r_cap + 3, 3)] == 0.25
    )
    pre = next(
        r for (r, c), v in cells.items() if c == 1 and str(v).startswith("Pre-tax annual FCF")
    )
    assert cells[(pre, 3)] == f"=C{r_cap}*(C{r_cap + 1}-C{r_cap + 2})"
    assert cells[(pre + 1, 3)] == f"=C{pre}*'Taxation & Borrowing Costs'!C{t['in_ltcg']}"
    assert (
        cells[(pre + 2, 3)]
        == f"=C{r_cap}*'Taxation & Borrowing Costs'!C{t['lev']}*'Taxation & Borrowing Costs'!C{t['borrow']}"
    )
    assert cells[(pre + 3, 3)] == f"=C{pre}-C{pre + 1}-C{pre + 2}"
    pv_line = next(r for (r, c), v in cells.items() if c == 1 and v == "PV of FCF")
    assert cells[(pv_line, 2)] == f"=B{pv_line - 1}/(1+$C${r_cap + 3})^1"
    assert cells[(pv_line, 6)] == f"=F{pv_line - 1}/(1+$C${r_cap + 3})^5"
    tv = next(r for (r, c), v in cells.items() if c == 1 and str(v).startswith("Terminal value"))
    assert cells[(tv, 3)] == f"=(F{pv_line - 1}*(1+C{r_cap + 5}))/(C{r_cap + 3}-C{r_cap + 5})"
    assert cells[(tv + 1, 3)] == f"=C{tv}/(1+C{r_cap + 3})^C{r_cap + 4}"
    npv = next(
        r
        for (r, c), v in cells.items()
        if c == 1 and str(v).startswith("Estimated operation value")
    )
    assert cells[(npv, 3)] == f"=SUM(B{pv_line}:F{pv_line})+C{tv + 1}"
    assert labels  # sheet has labelled rows


def test_benchmark_sheet_rows_and_median_pe(env):
    ws = BV.sheet_benchmark(Workbook())
    vals = [[c.value for c in row] for row in ws.iter_rows()]
    head = next(r for r in vals if r[0] == "ticker")
    assert head[:6] == ["ticker", "company", "market", "business", "market_cap", "P/E"]
    tickers = [r[0] for r in vals if r[0] in BV.COMPARABLES]
    assert tickers == list(BV.COMPARABLES)
    text = " ".join(str(c) for r in vals for c in r if c)
    median = pd.Series([c["pe"] for c in BV.COMPARABLES.values()]).median()
    assert f"Median P/E across these six: ~{median:.1f}x" in text


def test_build_writes_a_readable_workbook(env, capsys):
    out = BV.build()
    assert out == BV.OUT and out.exists() and "wrote" in capsys.readouterr().out
    wb = load_workbook(out)
    assert wb.sheetnames == [
        "Cover & Methodology",
        "Actual P&L",
        "Strategy Signal Quality",
        "Capacity & Cost Model",
        "Taxation & Borrowing Costs",
        "DCF — Operation Valuation",
        "Comparable Listed Firms",
    ]


def test_output_filename_embeds_the_date():
    assert BV.OUT.name.startswith("strategy_earnings_and_valuation_")
    assert BV.OUT.suffix == ".xlsx" and BV.OUT.parent.name == "business_valuation"
