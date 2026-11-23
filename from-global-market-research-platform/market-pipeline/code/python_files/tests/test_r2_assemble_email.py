"""Offline tests for assemble_email (report HTML sections, snapshot, main wiring)."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import assemble_email as ae


@pytest.fixture
def tree(monkeypatch, tmp_path):
    monkeypatch.setattr(ae, "PF", tmp_path)
    monkeypatch.setattr(ae, "OUT_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    return tmp_path


# ------------------------------------------------------------------ file lookup
def test_latest_json_picks_newest_by_name(tree):
    assert ae.latest_json("IN") is None
    d = tree / "combined_report_results"
    d.mkdir()
    (d / "combined_IN_20260101.json").write_text(json.dumps({"v": 1}))
    (d / "combined_IN_20260301.json").write_text(json.dumps({"v": 3}))
    (d / "combined_US_20260401.json").write_text(json.dumps({"v": 9}))
    assert ae.latest_json("IN") == {"v": 3} and ae.latest_json("US") == {"v": 9}


def test_latest_scan_xlsx_per_market(tree):
    assert ae.latest_scan_xlsx("IN") is None
    for d, names in (("indian_full_scan", ["indian_full_scan_1.xlsx", "indian_full_scan_2.xlsx"]), ("us_full_scan", ["us_full_scan_5.xlsx"])):
        (tree / d).mkdir()
        for n in names:
            (tree / d / n).write_text("x")
    assert ae.latest_scan_xlsx("IN").endswith("indian_full_scan_2.xlsx")
    assert ae.latest_scan_xlsx("US").endswith("us_full_scan_5.xlsx")


# ------------------------------------------------------------------ snapshot
class _Hist:
    def __init__(self, h):
        self.h = h

    def history(self, period):
        if isinstance(self.h, Exception):
            raise self.h
        return self.h


def _closes(vals):
    return pd.DataFrame({"Close": vals})


def test_market_snapshot_regimes_and_vix(monkeypatch):
    up = _closes(list(np.full(200, 100.0)) + [110.0])  # last 200 bars average 100.05 -> above
    down = _closes(list(np.full(200, 100.0)) + [90.0])
    table = {"^NSEI": up, "^GSPC": down, "^INDIAVIX": _closes([12.0, 14.5])}
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: _Hist(table[t])
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    s = ae.market_snapshot()
    assert s["nifty"]["regime"] == "above 200-DMA (bullish)" and s["nifty"]["label"] == "Nifty 50" and s["nifty"]["last"] == 110.0
    assert s["sp500"]["regime"] == "below 200-DMA (bearish)"
    assert s["nifty"]["dma200"] == pytest.approx(np.mean([100.0] * 199 + [110.0]))
    assert s["vix"] == 14.5


def test_market_snapshot_equal_to_dma_is_bearish(monkeypatch):
    flat = _closes(list(np.full(210, 100.0)))
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: _Hist(flat if t != "^INDIAVIX" else pd.DataFrame())
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    s = ae.market_snapshot()
    assert s["nifty"]["regime"] == "below 200-DMA (bearish)"  # strict '>' comparison
    assert s["vix"] is None


def test_market_snapshot_failures(monkeypatch):
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda t: _Hist(RuntimeError("down") if t == "^NSEI" else pd.DataFrame())
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    s = ae.market_snapshot()
    assert s["nifty"] is None and s["sp500"] is None and s["vix"] is None
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert "error" in ae.market_snapshot()


def test_snapshot_html_variants():
    html = ae.snapshot_html({"nifty": {"last": 22500.4, "dma200": 21000.0, "regime": "above 200-DMA (bullish)"}, "vix": 13.456,
                             "sp500": {"last": 5500.0, "regime": "below 200-DMA (bearish)"}})
    assert "Nifty 50 <b>22,500</b> vs 200-DMA 21,000" in html and "India VIX <b>13.46</b>" in html and "S&amp;P 500 <b>5,500</b>" in html
    assert html.count("&nbsp;|&nbsp;") == 2 and html.startswith("<p>")
    assert "India VIX <b>0.00</b>" in ae.snapshot_html({"vix": 0.0})  # a zero VIX is data, not missing
    assert "Market snapshot unavailable" in ae.snapshot_html({})
    assert "Market snapshot unavailable" in ae.snapshot_html({"nifty": None, "vix": None})


# ------------------------------------------------------------------ darvas section
def _scan(path, rows, sheet="Darvas_Signals"):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name=sheet, index=False)


def test_darvas_section_filters_and_formats(tree):
    assert "scan xlsx unavailable" in ae.darvas_section("IN")
    d = tree / "indian_full_scan"
    d.mkdir()
    rows = [
        {"Symbol": "AAA", "Darvas_Signal": "BREAKOUT_BUY", "Position_in_Box%": 100.0, "LTP": 1234.5, "Suffix": ".NS"},
        {"Symbol": "BBB", "Darvas_Signal": "breakout_buy", "Position_in_Box%": 120.0, "LTP": float("nan"), "Suffix": ""},
        {"Symbol": "CCC", "Darvas_Signal": "BREAKOUT_BUY", "Position_in_Box%": 99.9, "LTP": 5.0, "Suffix": ".NS"},
        {"Symbol": "DDD", "Darvas_Signal": "BREAKOUT_BUY", "Position_in_Box%": 120.1, "LTP": 5.0, "Suffix": ".NS"},
        {"Symbol": "EEE", "Darvas_Signal": "BREAKDOWN_SELL", "Position_in_Box%": 110.0, "LTP": 5.0, "Suffix": ".NS"},
    ]
    _scan(d / "indian_full_scan_1.xlsx", rows)
    html = ae.darvas_section("IN")
    assert "fresh box breakouts (top 2)" in html  # 100 and 120 are inclusive bounds; 99.9/120.1 excluded
    assert "<b>AAA</b> &middot; .NS</td><td>₹1,234.50</td>" in html
    bbb_row = html[html.index("<b>BBB</b>"):].split("</tr>")[0]
    assert bbb_row.endswith("<td>&mdash;</td>")  # NaN LTP shows a dash, not 'nan'
    assert "CCC" not in html and "DDD" not in html and "EEE" not in html


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="darvas_section: a blank Suffix cell is NaN (truthy), so the row renders ' &middot; nan' instead of no exchange tag")
def test_darvas_section_blank_suffix_has_no_exchange_tag(tree):
    d = tree / "indian_full_scan"
    d.mkdir()
    _scan(d / "indian_full_scan_1.xlsx", [{"Symbol": "BBB", "Darvas_Signal": "BREAKOUT_BUY", "LTP": 5.0, "Suffix": None}])
    assert "&middot;" not in ae.darvas_section("IN")


def test_darvas_section_top_limit_empty_and_missing_sheet(tree):
    d = tree / "us_full_scan"
    d.mkdir()
    rows = [{"Symbol": f"S{i}", "Darvas_Signal": "BREAKOUT_BUY", "LTP": 10.0 + i} for i in range(20)]
    _scan(d / "us_full_scan_1.xlsx", rows)
    html = ae.darvas_section("US", top=3)
    assert "top 3" in html and "$10.00" in html and html.count("<tr><td>") == 3  # no position column -> no position filter
    _scan(d / "us_full_scan_2.xlsx", [{"Symbol": "X", "Darvas_Signal": "BREAKDOWN_SELL"}])
    assert "no fresh box breakouts today" in ae.darvas_section("US")
    _scan(d / "us_full_scan_3.xlsx", [{"Symbol": "X"}], sheet="Other")
    assert "no Darvas_Signals sheet" in ae.darvas_section("US")
    (d / "us_full_scan_4.xlsx").write_text("not an excel file")
    assert "Darvas extraction failed" in ae.darvas_section("US")


# ------------------------------------------------------------------ news picks
def test_news_picks_block_rows_and_failures(monkeypatch):
    mod = types.ModuleType("news_picks")
    pick = {"symbol": "TCS", "exchange": "NSE", "name": "Tata Consultancy Services Limited Extra", "mentions": 4, "label": "POSITIVE", "score": 0.4567, "headline": "TCS wins deal"}
    neg = dict(pick, symbol="BAD", label="NEGATIVE", score=-0.5)
    neu = dict(pick, symbol="MEH", label="MIXED", score=0.0)

    def np_(mk, top):
        if mk == "IN":
            return [pick, neg, neu]
        raise RuntimeError("feed down")
    mod.news_picks = np_
    monkeypatch.setitem(sys.modules, "news_picks", mod)
    html = ae.news_picks_block()
    assert "top 3 in the news" in html and "color:#2e7d32" in html and "color:#c62828" in html and "color:#f57f17" in html
    assert "Tata Consultancy Services Limite" in html and "Tata Consultancy Services Limited" not in html  # name truncated to 32 chars
    assert "+0.46" in html and "-0.50" in html
    assert "US: news picks failed (feed down)" in html
    assert "No clearly-named stocks in today's headlines across markets" not in html


def test_news_picks_block_empty_and_import_failure(monkeypatch):
    mod = types.ModuleType("news_picks")
    mod.news_picks = lambda mk, top: []
    monkeypatch.setitem(sys.modules, "news_picks", mod)
    html = ae.news_picks_block(("IN",))
    assert "No clearly-named stocks in today's headlines.</p>" in html and "across markets" in html
    monkeypatch.setitem(sys.modules, "news_picks", None)
    assert "News picks unavailable" in ae.news_picks_block()


# ------------------------------------------------------------------ convergence
@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="convergence_block: `rep.get('convergence', [])` returns None for a JSON null (build() guards with `or []`), so iterating raises TypeError")
def test_convergence_block_tolerates_null_convergence():
    assert "No fundamental picks had matching news coverage today." in ae.convergence_block({"EU": {"convergence": None}})


def test_convergence_block_buckets():
    reports = {"IN": {"convergence": [{"Symbol": "AAA", "Convergence": "BOTH BULLISH", "Tier": "Triple Hit", "News_Score": 0.5, "N_Articles": 3, "Top_Headline": "good"},
                                      {"Symbol": "BBB", "Convergence": "FUND POS / NEWS NEGATIVE", "News_Score": -0.3, "Top_Headline": "bad"},
                                      {"Symbol": "CCC", "Convergence": "NEUTRAL", "News_Score": 0.0}]},
               "US": None}
    html = ae.convergence_block(reports)
    assert "AAA</b> [Triple Hit &middot; IN] &mdash; news +0.50 (3 art): good" in html
    assert "BBB</b> [IN] &mdash; news -0.30: bad" in html and "CCC" not in html
    assert "High-conviction" in html and "Caution" in html
    only_caution = ae.convergence_block({"IN": {"convergence": [{"Symbol": "B", "Convergence": "NEGATIVE", "News_Score": -1.0}]}})
    assert "High-conviction" not in only_caution and "Caution" in only_caution
    none = ae.convergence_block({"IN": {}, "US": None})
    assert "No fundamental picks had matching news coverage today." in none and "<ul>" not in none


# ------------------------------------------------------------------ picks table
def test_picks_table_tiers_caps_and_note():
    def p(sym, tier, ltp=10.0, pio=7):
        return {"Symbol": sym, "Tier": tier, "LTP": ltp, "Screens": 4, "Piotroski": pio}
    picks = ([p(f"T{i}", "Triple Hit") for i in range(17)] + [p("M1", "Multi-Screen", ltp="nan", pio="nan")]
             + [p(f"S{i}", "Single-Screen") for i in range(12)])
    html = ae.picks_table({"market": "US", "picks": picks})
    assert "US (NASDAQ/NYSE) &mdash; 17 triple, 1 multi-screen" in html
    assert html.count("<b>T") == 15 and "T15" not in html  # triple capped at 15
    assert html.count("<b>S") == 10 and "S10" not in html  # single capped at 10
    assert "<b>M1</b></td><td>&mdash;</td><td>Multi-Screen</td><td>4</td><td>&mdash;</td>" in html
    assert "$10.00" in html and "India's full scan" not in html
    only_single = ae.picks_table({"market": "IN", "picks": [p("S", "Single-Screen")]})
    assert "India's full scan carried no fundamentals this run" in only_single and "₹10.00" in only_single
    with_multi = ae.picks_table({"market": "IN", "picks": [p("M", "Multi-Screen")]})
    assert "India's full scan" not in with_multi
    assert ae.picks_table({"market": "US", "picks": []}) == "<p>No fundamental picks for US today.</p>"


# ------------------------------------------------------------------ talk block
def test_talk_block_sorted_by_score_and_filters():
    rep = {"market": "IN", "mood": {"mood": "NEGATIVE", "score": -0.25, "n_articles": 40},
           "talk": {"LOW": {"label": "NEGATIVE", "score": -0.4, "n_articles": 2, "headlines": [{"title": "x" * 100}]},
                    "HIGH": {"label": "POSITIVE", "score": 0.6, "n_articles": 5, "headlines": ["plain string"]},
                    "NONE": {"label": "NO_DATA", "score": 0.9}, "NOLABEL": {"label": None, "score": 0.8},
                    "MID": {"label": "MIXED", "score": 0.0}}}
    html = ae.talk_block(rep)
    assert html.index("HIGH") < html.index("MID") < html.index("LOW")  # descending score
    assert "NONE" not in html and "NOLABEL" not in html
    assert "Mood: <b style='color:#c62828'>NEGATIVE (-0.25)</b> from 40 live articles (Moneycontrol" in html
    assert "x" * 70 in html and "x" * 71 not in html  # headline truncated to 70
    assert "<td style='color:#f57f17'>MIXED</td><td>+0.00</td><td>0</td>" in html
    us = ae.talk_block({"market": "US", "mood": {}, "talk": {}})
    assert "No per-ticker coverage today." in us and "Mood:" not in us
    assert "CNBC · MarketWatch" in ae.talk_block({"market": "US", "mood": {"mood": "POSITIVE", "score": 0.1}, "talk": None})


# ------------------------------------------------------------------ build + main
def test_build_assembles_sections(tree, monkeypatch):
    mod = types.ModuleType("news_picks")
    mod.news_picks = lambda mk, top: []
    monkeypatch.setitem(sys.modules, "news_picks", mod)
    reports = {"IN": {"market": "IN", "picks": [{"Symbol": "A", "Tier": "Triple Hit", "LTP": 1, "Screens": 5, "Piotroski": 8}], "mood": {"mood": "POSITIVE", "score": 0.2, "n_articles": 3},
                      "convergence": [{"Symbol": "A", "Convergence": "BOTH BULLISH", "News_Score": 0.1, "N_Articles": 1}], "talk": {}},
               "US": None}
    html = ae.build(reports, {"vix": 12.0}, "DEGRADED NOTE")
    assert "Fundamentals: 1 picks" in html and "Street mood: IN POSITIVE" in html and "Convergence: 1" in html
    assert '<div class="warn">DEGRADED NOTE</div>' in html and ae.DISCLAIMER in html
    assert "India (NSE/BSE) &mdash; 1 triple" in html and "US (NASDAQ/NYSE) &mdash;" not in html
    assert html.index("Market snapshot") < html.index("CONVERGENCE") < html.index("COMPONENT 1") < html.index("COMPONENT 2")
    bare = ae.build({"IN": None, "US": None}, {}, "")
    assert "Street mood: n/a" in bare and "Fundamentals: 0 picks" in bare and 'class="warn"' not in bare.split("</style>")[1]


def test_main_writes_both_files_and_reports_counts(tree, monkeypatch, capsys):
    d = tree / "combined_report_results"
    d.mkdir()
    (d / "combined_IN_1.json").write_text(json.dumps({"market": "IN", "picks": [{"Symbol": "A", "Tier": "Single-Screen", "LTP": 1, "Screens": 1}], "convergence": [], "talk": {}}))
    monkeypatch.setattr(ae, "market_snapshot", lambda: {})
    monkeypatch.setattr(ae, "news_picks_block", lambda *a, **k: "<p>news</p>")
    monkeypatch.setattr(ae, "darvas_section", lambda m, top=15: f"<p>darvas {m}</p>")
    ae.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("EMAIL_HTML=") and out[0].endswith(f"daily_market_report_{ae.DATE}.html")
    assert out[1].endswith("daily_market_report_artifact.html")
    assert "PICKS_IN=1" in out and "PICKS_US=0" in out
    html = (tree / "reports" / f"daily_market_report_{ae.DATE}.html").read_text()
    assert "US combined report JSON missing this run." in html and "India combined report JSON missing" not in html
    assert (tree / "reports" / "daily_market_report_artifact.html").read_text() == html


def test_main_survives_unwritable_output(tree, monkeypatch, capsys):
    monkeypatch.setattr(ae, "OUT_DIR", tree / "no_such_dir")  # write_text raises OSError
    monkeypatch.setattr(ae, "market_snapshot", lambda: {})
    monkeypatch.setattr(ae, "news_picks_block", lambda *a, **k: "")
    monkeypatch.setattr(ae, "darvas_section", lambda m, top=15: "")
    ae.main()
    out = capsys.readouterr().out
    assert out.count("failed:") == 2 and "PICKS_IN=0" in out
