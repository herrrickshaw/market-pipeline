"""daily_combined_report.py -- pick aggregation across screener sheets, convergence, rendering.

The module creates ./combined_report_results at import, so it is imported inside a
sandboxed working directory (and removed from sys.modules afterwards).
"""
import importlib
import json
import subprocess
import sys
import types

import pandas as pd
import pytest


@pytest.fixture
def DCR(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("daily_combined_report", None)
    mod = importlib.import_module("daily_combined_report")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    yield mod
    sys.modules.pop("daily_combined_report", None)


def write_xlsx(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xw:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(xw, sheet_name=name, index=False)


def by_symbol(df):
    return {r["Symbol"]: r for _, r in df.iterrows()}


# ── get_fundamental_picks ─────────────────────────────────────────────────────
def test_picks_layout_a_tiering_sorting_and_piotroski(DCR, tmp_path):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_20260101.xlsx", {
        "Triple_Hits": [{"Symbol": "TTT", "Piotroski_Score": 9, "LTP": 10.0}],
        "Multi_Screen_Hits": [{"Symbol": "MMM", "Piotroski_Score": 7, "LTP": 20.0}],
        "Piotroski_Strong": [{"Symbol": "AAA", "Piotroski_Score": 8, "LTP": 30.0},
                             {"Symbol": "MMM", "Piotroski_Score": 8, "LTP": 21.0},
                             {"Symbol": "SSS", "Piotroski_Score": 8, "LTP": 33.0}],
        "Coffee_Can": [{"Symbol": "AAA"}, {"Symbol": "SSS"}],
        "Magic_Formula": [{"Symbol": "AAA"}],
        "Bull_Cartel": [{"Symbol": "AAA"}, {"Symbol": "ONE"}]})
    df = DCR.get_fundamental_picks("US")
    rows = by_symbol(df)
    assert rows["TTT"]["Tier"] == "Triple Hit" and rows["TTT"]["Screens"] == "—"
    assert rows["MMM"]["Tier"] == "Multi-Screen" and rows["MMM"]["Screens"] == "Piotroski"
    assert rows["AAA"]["Tier"] == "Multi-Screen"                  # 4 screeners >= 3
    assert rows["AAA"]["Screens"] == "BullCartel+CoffeeCan+MagicFormula+Piotroski"
    assert rows["AAA"]["N_Screens"] == 4
    assert rows["SSS"]["Tier"] == "Single-Screen" and rows["SSS"]["N_Screens"] == 2
    assert rows["ONE"]["Tier"] == "Single-Screen" and rows["ONE"]["N_Screens"] == 1
    assert list(df["Symbol"])[:2] == ["TTT", "AAA"]
    assert list(df["Tier"]) == ["Triple Hit", "Multi-Screen", "Multi-Screen",
                                "Single-Screen", "Single-Screen"]
    assert list(df["N_Screens"])[:2] == [0, 4] and list(df["N_Screens"])[3:] == [2, 1]
    assert rows["MMM"]["Piotroski"] == 8 and rows["MMM"]["LTP"] == 21.0   # later sheet wins
    assert rows["TTT"]["Piotroski"] == 9
    assert "_r" not in df.columns


def test_picks_exactly_three_screeners_is_multi_screen(DCR, tmp_path):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {
        "Piotroski_Strong": [{"Symbol": "X3"}, {"Symbol": "X2"}],
        "Coffee_Can": [{"Symbol": "X3"}, {"Symbol": "X2"}],
        "Magic_Formula": [{"Symbol": "X3"}]})
    rows = by_symbol(DCR.get_fundamental_picks("US"))
    assert rows["X3"]["Tier"] == "Multi-Screen" and rows["X2"]["Tier"] == "Single-Screen"


def test_picks_darvas_sheet_keeps_only_breakouts(DCR, tmp_path):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {
        "Darvas_Signals": [{"Symbol": "BRK", "Darvas_Signal": "breakout_buy"},
                           {"Symbol": "HOLD", "Darvas_Signal": "HOLD"}]})
    rows = by_symbol(DCR.get_fundamental_picks("US"))
    assert list(rows) == ["BRK"] and rows["BRK"]["Screens"] == "Darvas"


def test_picks_layout_b_fundamentals_and_golden_cross(DCR, tmp_path):
    write_xlsx(tmp_path / "indian_full_scan" / "indian_full_scan_1.xlsx", {
        "Fundamentals": [
            {"Symbol": "RELIANCE", "LTP": 100.0, "Piotroski_Score": 8, "Piotroski_Strong": "yes",
             "CoffeeCan": "pass", "MagicFormula": "PASS", "BullCartel": "FAIL",
             "Darvas_Signal": "BREAKOUT_BUY"},
            {"Symbol": "TCS", "LTP": 50.0, "Piotroski_Score": 3, "Piotroski_Strong": "NO",
             "CoffeeCan": "FAIL", "MagicFormula": "FAIL", "BullCartel": "PASS",
             "Darvas_Signal": "HOLD"}],
        "All_Stocks": [{"Symbol": "TCS", "GC_Signal": "golden_cross", "Piotroski_Score": 4,
                        "LTP": 51.0},
                       {"Symbol": "INFY", "GC_Signal": "ABOVE_200DMA"}]})
    rows = by_symbol(DCR.get_fundamental_picks("IN"))
    assert set(rows) == {"RELIANCE", "TCS"}
    assert rows["RELIANCE"]["Screens"] == "CoffeeCan+Darvas+MagicFormula+Piotroski"
    assert rows["RELIANCE"]["Tier"] == "Multi-Screen" and rows["RELIANCE"]["N_Screens"] == 4
    assert rows["TCS"]["Screens"] == "BullCartel+GoldenCross" and rows["TCS"]["Tier"] == "Single-Screen"
    assert rows["TCS"]["Piotroski"] == 4 and rows["TCS"]["LTP"] == 51.0


def test_picks_all_stocks_without_gc_column_is_ignored(DCR, tmp_path):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {
        "All_Stocks": [{"Symbol": "Q"}], "Coffee_Can": [{"Symbol": "Z"}]})
    assert list(DCR.get_fundamental_picks("US")["Symbol"]) == ["Z"]


def test_picks_uses_newest_workbook(DCR, tmp_path):
    for n, sym in (("1", "OLD"), ("2", "NEW")):
        write_xlsx(tmp_path / "us_full_scan" / f"us_full_scan_{n}.xlsx", {"Coffee_Can": [{"Symbol": sym}]})
    assert list(DCR.get_fundamental_picks("US")["Symbol"]) == ["NEW"]


def test_picks_prints_summary_counts(DCR, tmp_path, capsys):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {
        "Triple_Hits": [{"Symbol": "T"}], "Coffee_Can": [{"Symbol": "S"}]})
    DCR.get_fundamental_picks("US")
    assert "2 UNIQUE fundamental picks across all screeners (1 triple, 0 multi-screen, 1 single-screen)" \
        in capsys.readouterr().out


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="a blank Symbol cell reads from Excel as NaN; str(NaN or '') is 'nan', so "
                          "the blank row is aggregated as a pick named 'nan'")
def test_picks_blank_symbol_cells_are_skipped(DCR, tmp_path):
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {
        "Coffee_Can": [{"Symbol": None, "LTP": 1.0}, {"Symbol": "OK", "LTP": 2.0}]})
    assert list(DCR.get_fundamental_picks("US")["Symbol"]) == ["OK"]


def test_picks_runs_scan_when_missing_or_forced(DCR, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, timeout: calls.append((cmd, timeout)))
    assert DCR.get_fundamental_picks("IN").empty
    assert calls[0][0][1:] == ["full_indian_market_scan.py", "--workers", "10"]
    assert calls[0][0][0] == sys.executable and calls[0][1] == 14400
    assert DCR.get_fundamental_picks("US").empty
    assert calls[1][0][1:] == ["full_us_market_scan.py", "--workers", "10", "--min-price", "2"]
    write_xlsx(tmp_path / "us_full_scan" / "us_full_scan_1.xlsx", {"Coffee_Can": [{"Symbol": "Z"}]})
    DCR.get_fundamental_picks("US")
    assert len(calls) == 2                                           # files exist: no run
    DCR.get_fundamental_picks("US", run_fresh=True)
    assert len(calls) == 3


# ── get_street_talk ───────────────────────────────────────────────────────────
def fake_pipeline(monkeypatch, record):
    class S:
        def __init__(self, score, label, n, heads):
            self.score, self.label, self.n_articles, self.top_headlines = score, label, n, heads

    class SP:
        def get_market_mood(self, market):
            record["mood_market"] = market
            return {"mood": "POSITIVE", "score": 0.3, "n_articles": 9}

        def get_batch(self, symbols, market):
            record["batch"] = (list(symbols), market)
            return {s: S(0.5, "POSITIVE", 3, [{"title": "a"}, {"title": "b"}, {"title": "c"}])
                    for s in symbols}
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", types.SimpleNamespace(SentimentPipeline=SP))


def test_street_talk_shapes_results_and_trims_headlines(DCR, monkeypatch):
    rec = {}
    fake_pipeline(monkeypatch, rec)
    mood, talk = DCR.get_street_talk(["AAA", "BBB"], "US")
    assert mood["mood"] == "POSITIVE" and rec["mood_market"] == "US" and rec["batch"] == (["AAA", "BBB"], "US")
    assert talk["AAA"] == {"score": 0.5, "label": "POSITIVE", "n_articles": 3,
                           "headlines": [{"title": "a"}, {"title": "b"}]}


def test_street_talk_no_symbols_skips_batch(DCR, monkeypatch):
    rec = {}
    fake_pipeline(monkeypatch, rec)
    mood, talk = DCR.get_street_talk([], "IN")
    assert talk == {} and "batch" not in rec and mood["n_articles"] == 9


def test_street_talk_without_sentiment_module(DCR, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", None)
    assert DCR.get_street_talk(["A"], "IN") == ({}, {})


# ── find_convergence ──────────────────────────────────────────────────────────
def talk_entry(label, score, n=2, heads=None):
    return {"label": label, "score": score, "n_articles": n,
            "headlines": [{"title": "H" * 100}] if heads is None else heads}


def test_convergence_labels_sorting_and_filters(DCR):
    picks = pd.DataFrame({"Symbol": ["N1", "P1", "NG", "P2", "ND", "NONE", "EMPTY"],
                          "Tier": ["T"] * 7})
    talk = {"N1": talk_entry("NEUTRAL", 0.0), "P1": talk_entry("POSITIVE", 0.4),
            "NG": talk_entry("NEGATIVE", -0.6), "P2": talk_entry("POSITIVE", 0.9),
            "ND": talk_entry("NO_DATA", 0.0), "EMPTY": talk_entry("POSITIVE", 0.1, heads=[])}
    out = DCR.find_convergence(picks, talk)
    assert list(out["Symbol"]) == ["P2", "P1", "EMPTY", "N1", "NG"]       # both-bullish, neutral, negative
    assert list(out["Convergence"]) == ["✅ BOTH BULLISH"] * 3 + ["— news neutral", "⚠️ NEWS NEGATIVE"]
    assert "_rank" not in out.columns
    assert out.set_index("Symbol").loc["P1", "Top_Headline"] == "H" * 80
    assert out.set_index("Symbol").loc["EMPTY", "Top_Headline"] == ""
    assert (out["Fundamental"] == "STRONG").all()


def test_convergence_empty_inputs(DCR):
    assert DCR.find_convergence(pd.DataFrame({"Symbol": ["A"], "Tier": ["T"]}), {}).empty
    assert DCR.find_convergence(pd.DataFrame({"Symbol": [], "Tier": []}), {"A": talk_entry("POSITIVE", 1)}).empty
    assert DCR.find_convergence(pd.DataFrame({"Symbol": ["A"], "Tier": ["T"]}),
                                {"A": {}}).empty                      # falsy talk entry skipped


# ── build_report / rendering ──────────────────────────────────────────────────
def sample_report(DCR, market="IN", empty=False):
    picks = pd.DataFrame() if empty else pd.DataFrame({
        "Symbol": ["AAA", "BBB"], "Tier": ["Triple Hit", "Single-Screen"],
        "Screens": ["Coffee", "—"], "Piotroski": [8, None]})
    talk = {} if empty else {"AAA": talk_entry("POSITIVE", 0.5, 4, [{"title": "Great quarter"}]),
                             "BBB": talk_entry("NEGATIVE", -0.4, 2, [{"title": "Probe opened"}]),
                             "CCC": talk_entry("NO_DATA", 0.0, 0, [])}
    mood = {} if empty else {"mood": "POSITIVE", "score": 0.25, "n_articles": 12}
    conv = DCR.find_convergence(picks, talk) if not empty else pd.DataFrame()
    return {"market": market, "generated": "2026-01-01T09:00:00", "picks": picks, "mood": mood,
            "talk": talk, "convergence": conv}


def test_build_report_limits_symbols_to_top_25(DCR, monkeypatch):
    picks = pd.DataFrame({"Symbol": [f"S{i}" for i in range(40)], "Tier": ["T"] * 40})
    seen = {}
    monkeypatch.setattr(DCR, "get_fundamental_picks", lambda m, fresh: picks)
    monkeypatch.setattr(DCR, "get_street_talk",
                        lambda syms, m: seen.update(syms=list(syms), market=m) or ({"mood": "x"}, {}))
    rep = DCR.build_report("US", True)
    assert seen["syms"] == [f"S{i}" for i in range(25)] and seen["market"] == "US"
    assert rep["market"] == "US" and rep["picks"] is picks and rep["mood"] == {"mood": "x"}
    assert rep["convergence"].empty and "generated" in rep


def test_build_report_empty_picks_no_symbols(DCR, monkeypatch):
    seen = {}
    monkeypatch.setattr(DCR, "get_fundamental_picks", lambda m, fresh: pd.DataFrame())
    monkeypatch.setattr(DCR, "get_street_talk", lambda syms, m: seen.update(syms=syms) or ({}, {}))
    DCR.build_report("IN", False)
    assert seen["syms"] == []


def test_print_report_full(DCR, capsys):
    DCR.print_report(sample_report(DCR))
    out = capsys.readouterr().out
    assert "COMPONENT 1" in out and "AAA" in out and "Triple Hit" in out
    assert "Market mood: POSITIVE (+0.25) from 12 articles (Moneycontrol/ET/BusinessLine/LiveMint)" in out
    assert out.index("AAA", out.index("COMPONENT 2")) < out.index("BBB", out.index("COMPONENT 2"))   # by score desc
    assert "CCC" not in out                                              # NO_DATA hidden
    assert "🎯 1 HIGH-CONVICTION" in out and "news +0.50 (4 art) — Great quarter" in out
    assert "⚠️  1 CAUTION" in out and "news -0.40 — Probe opened" in out


def test_print_report_us_source_and_empty(DCR, capsys):
    rep = sample_report(DCR, "US")
    DCR.print_report(rep)
    assert "(CNBC/MarketWatch)" in capsys.readouterr().out
    DCR.print_report(sample_report(DCR, empty=True))
    out = capsys.readouterr().out
    assert "No fundamental picks today." in out
    assert "No picks had matching news coverage today." in out and "Market mood" not in out


def test_to_html_sections_colours_and_rows(DCR):
    html = DCR.to_html(sample_report(DCR))
    assert "<h2>📊 Stock Picks Based on Fundamentals</h2>" in html
    assert "<tr><td><b>AAA</b></td><td>Triple Hit</td><td>Coffee</td><td>8.0</td></tr>" in html
    assert 'color:#2e7d32">POSITIVE (+0.25)</b>' in html and "Moneycontrol · ET" in html
    assert "<td style='color:#c62828'>NEGATIVE</td><td>-0.40</td>" in html
    assert "CCC" not in html
    assert "<li><b>AAA</b> [Triple Hit] — news +0.50 (4 articles): Great quarter</li>" in html
    assert "<li><b>BBB</b> — news -0.40: Probe opened</li>" in html
    us = DCR.to_html(sample_report(DCR, "US"))
    assert "CNBC · MarketWatch" in us
    neutral = sample_report(DCR)
    neutral["mood"] = {"mood": "NEUTRAL", "score": 0.0, "n_articles": 1}
    assert 'color:#f57f17">NEUTRAL' in DCR.to_html(neutral)


def test_to_html_empty_report(DCR):
    html = DCR.to_html(sample_report(DCR, empty=True))
    assert "No fundamental picks had news coverage today." in html
    assert "Market mood" not in html and html.count("<table>") == 2


def test_save_writes_json_and_optional_html(DCR, tmp_path):
    DCR.save(sample_report(DCR), emit_html=False)
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert len(files) == 1 and files[0].startswith("combined_IN_") and files[0].endswith(".json")
    payload = json.loads((tmp_path / "out" / files[0]).read_text())
    assert payload["market"] == "IN" and payload["mood"]["n_articles"] == 12
    assert [p["Symbol"] for p in payload["picks"]] == ["AAA", "BBB"]
    assert payload["convergence"][0]["Convergence"] == "✅ BOTH BULLISH"
    assert payload["talk"]["AAA"]["label"] == "POSITIVE"
    DCR.save(sample_report(DCR, "US"), emit_html=True)
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any(n.startswith("combined_US_") and n.endswith(".html") for n in names)
    assert any(n.startswith("combined_US_") and n.endswith(".json") for n in names)


def test_save_empty_report_has_empty_lists(DCR, tmp_path):
    DCR.save(sample_report(DCR, empty=True), emit_html=False)
    payload = json.loads(next((tmp_path / "out").iterdir()).read_text())
    assert payload["picks"] == [] and payload["convergence"] == [] and payload["talk"] == {}


def test_main_wires_arguments(DCR, monkeypatch):
    calls = []
    monkeypatch.setattr(DCR, "build_report", lambda m, fresh: calls.append(("build", m, fresh)) or "REP")
    monkeypatch.setattr(DCR, "print_report", lambda r: calls.append(("print", r)))
    monkeypatch.setattr(DCR, "save", lambda r, html: calls.append(("save", r, html)))
    monkeypatch.setattr(sys, "argv", ["x", "--market", "US", "--run-fresh", "--html"])
    DCR.main()
    assert calls == [("build", "US", True), ("print", "REP"), ("save", "REP", True)]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x"])
    DCR.main()
    assert calls[0] == ("build", "IN", False) and calls[2] == ("save", "REP", False)
