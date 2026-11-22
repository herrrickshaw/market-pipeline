"""build_mailer.py -- the HTML section builders behind the daily brief.

Everything reads from tmp_path (the module globs relative paths, so each test
chdirs into a sandbox). Network/engine modules are replaced via sys.modules.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

import build_mailer as BM
import liquidity as liq


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "bhav"))
    monkeypatch.setattr(liq, "INDEX_PATH", tmp_path / "liq.parquet")
    monkeypatch.setattr(liq, "_CACHE_MAP", {})
    return tmp_path


def write_liq(tmp_path, rows):
    pd.DataFrame(rows).to_parquet(tmp_path / "liq.parquet", index=False)


# ── tiny helpers ──────────────────────────────────────────────────────────────
def test_tv_formats_millions_or_dash():
    assert BM._tv(2_500_000) == "$2.5M"
    assert BM._tv(0) == "$0.0M"
    assert BM._tv(float("nan")) == "—"
    assert BM._tv(None) == "—"


@pytest.mark.parametrize("v,exp", [("7", 7), (7.9, 7), ("8.0", 8), (None, None), ("x", None),
                                   ("", None), (float("nan"), None)])
def test_pio(v, exp):
    assert BM._pio(v) == exp


def test_table_wraps_headers_and_rows():
    html = BM._table(["A", "B"], ["<tr><td>1</td></tr>", "<tr><td>2</td></tr>"])
    assert html.startswith("<div class='tw' style='overflow-x:auto")
    assert html.count("<th align='left' style='padding:5px 8px'>") == 2
    assert "<th align='left' style='padding:5px 8px'>A</th>" in html
    assert html.index("<td>1</td>") < html.index("<td>2</td>")
    assert html.endswith("</table></div>") and "min-width:520px" in html


# ── _load_combined ────────────────────────────────────────────────────────────
def test_load_combined_picks_latest_file_and_survives_bad_json(sandbox):
    d = sandbox / "combined_report_results"
    d.mkdir()
    assert BM._load_combined("IN") == {}
    (d / "combined_IN_20260101.json").write_text(json.dumps({"v": 1}))
    (d / "combined_IN_20260301.json").write_text(json.dumps({"v": 3}))
    (d / "combined_IN_20260201.json").write_text(json.dumps({"v": 2}))
    (d / "combined_US_20260401.json").write_text(json.dumps({"v": 9}))
    assert BM._load_combined("IN") == {"v": 3}
    assert BM._load_combined("US") == {"v": 9}
    (d / "combined_IN_20260501.json").write_text("{broken")
    assert BM._load_combined("IN") == {}                  # latest file is the broken one: no fallback


# ── convergence / talk / news rows ────────────────────────────────────────────
def test_convergence_empty_message():
    for data in ({}, {"convergence": []},
                 {"convergence": [{"Convergence": "NEUTRAL"}]}):
        out = BM._convergence_html("IN", data)
        assert out == ("<p style='color:#777;font-size:13px'>IN: no fundamental picks "
                       "had matching news coverage today.</p>")


def test_convergence_both_and_caution_blocks():
    data = {"convergence": [
        {"Symbol": "AAA", "Tier": "Triple Hit", "Convergence": "BOTH BULLISH",
         "News_Score": 0.456, "N_Articles": 7, "Top_Headline": "Up"},
        {"Symbol": "BBB", "Convergence": "NEGATIVE NEWS", "News_Score": -0.3,
         "Top_Headline": "Down"},
        {"Symbol": "CCC", "Convergence": "x"}]}
    out = BM._convergence_html("🇮🇳 India", data)
    assert "🎯 🇮🇳 India high-conviction" in out and "⚠️ 🇮🇳 India caution" in out
    assert "<li><b>AAA</b> [Triple Hit] — news +0.46 (7 articles): Up</li>" in out
    assert "<li><b>BBB</b> — news -0.30: Down</li>" in out
    assert "CCC" not in out
    only_both = BM._convergence_html("X", {"convergence": [data["convergence"][0]]})
    assert "high-conviction" in only_both and "caution" not in only_both
    only_caut = BM._convergence_html("X", {"convergence": [data["convergence"][1]]})
    assert "caution" in only_caut and "high-conviction" not in only_caut


def test_talk_rows_filter_sort_cap_and_colour():
    talk = {"A": {"label": "POSITIVE", "score": 0.2, "n_articles": 3},
            "B": {"label": "NEGATIVE", "score": -0.5, "n_articles": 1},
            "C": {"label": "NO_DATA", "score": 9},
            "D": {"score": 1.0},                       # no label -> still scored (label None excluded)
            "E": "junk",
            "F": {"label": "NEUTRAL", "score": 0.9, "n_articles": 2}}
    rows = BM._talk_rows({"talk": talk})
    syms = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    assert syms == ["F", "A", "B"]                       # by score desc; NO_DATA/None/junk dropped
    assert "color:#777" in rows[0] and "color:#1b7f37" in rows[1] and "color:#b00" in rows[2]
    assert "+0.90" in rows[0] and "<td>2</td>" in rows[0] and "-0.50" in rows[2]
    assert len(BM._talk_rows({"talk": talk}, cap=2)) == 2
    assert BM._talk_rows({}) == [] and BM._talk_rows({"talk": None}) == []


def test_news_rows_use_cap_default_and_swallow_errors(monkeypatch):
    seen = {}

    def fake(market, top):
        seen.update(market=market, top=top)
        return [{"symbol": "AAA", "name": "Alpha", "label": "POSITIVE", "score": 0.5,
                 "mentions": 4, "headline": "Big news"},
                {"symbol": "BBB", "name": "Beta", "label": "WEIRD", "score": -0.25,
                 "mentions": 1, "headline": "h"}]
    monkeypatch.setattr(BM.npk, "news_picks", fake)
    rows = BM._news_rows("IN")
    assert seen == {"market": "IN", "top": 12}
    assert "color:#1b7f37" in rows[0] and "POSITIVE (+0.50)" in rows[0] and "Big news" in rows[0]
    assert "color:#777" in rows[1] and "WEIRD (-0.25)" in rows[1]
    BM._news_rows("US", top=3)
    assert seen["top"] == 3
    BM._news_rows("US", top=0)
    assert seen["top"] == 0                               # explicit 0 is not "default"
    monkeypatch.setattr(BM.npk, "news_picks", lambda m, top: 1 / 0)
    assert BM._news_rows("IN") == []


# ── market picks ──────────────────────────────────────────────────────────────
def test_market_picks_rows_sort_by_tier_then_liquidity_and_cap(sandbox):
    write_liq(sandbox, [
        {"Symbol": "S1", "Market": "IN", "turnover_usd": 9e6, "ltp": 1},   # High
        {"Symbol": "S2", "Market": "IN", "turnover_usd": 6e5, "ltp": 1},   # Medium
        {"Symbol": "S3", "Market": "IN", "turnover_usd": 1e5, "ltp": 1},   # Low
        {"Symbol": "M1", "Market": "IN", "turnover_usd": 9e6, "ltp": 1}])
    picks = [{"Symbol": "S3", "Tier": "Single-Screen", "Screens": "P"},
             {"Symbol": "S1", "Tier": "Single-Screen", "Screens": "P"},
             {"Symbol": "M1", "Tier": "Multi-Screen", "Screens": "P+C"},
             {"Symbol": "S2", "Tier": "Single-Screen", "Screens": "P"},
             {"Symbol": "T1", "Tier": "Triple Hit", "Screens": "P+C+M"},
             {"Symbol": "U1", "Tier": "Odd", "Screens": "?"}]
    rows = BM._market_picks_rows({"picks": picks}, "IN", cap=100)
    order = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    # Triple Hit, Multi, then Single sorted High/Medium/Low, unknown tier last
    assert order == ["T1", "M1", "S1", "S2", "S3", "U1"]
    assert "$9.0M" in rows[1] and "color:#1b7f37" in rows[1] and ">High<" in rows[1]
    assert "—" in rows[0] and "color:#b00" in rows[0]            # T1 not in index: NaN turnover -> Low
    assert len(BM._market_picks_rows({"picks": picks}, "IN", cap=2)) == 2
    assert len(BM._market_picks_rows({"picks": picks}, "IN")) == 6     # default cap 24


@pytest.mark.parametrize("data", [{}, {"picks": []}, {"picks": None}])
def test_market_picks_rows_empty(data, sandbox):
    assert BM._market_picks_rows(data, "IN") == []


# ── fundamentals picks (EU/JP/KR) ─────────────────────────────────────────────
def fundamentals_xlsx(path, rows, sheet="Fundamentals"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, sheet_name=sheet, index=False)


def test_fundamentals_picks_filters_sorts_and_labels(sandbox):
    fundamentals_xlsx(sandbox / "scan" / "m_20260101.xlsx", [
        {"Code": "A1", "Name": "Alpha, Inc", "Piotroski_Score": 9, "Piotroski_Strong": "yes",
         "CoffeeCan": "PASS"},
        {"Code": "B2", "Name": "Beta", "Piotroski_Score": 7, "Piotroski_Strong": "NO",
         "CoffeeCan": "FAIL"},
        {"Code": "C3", "Name": "Gamma", "Piotroski_Score": 6, "Piotroski_Strong": "NO",
         "CoffeeCan": "pass"},
        {"Code": "D4", "Name": "Delta", "Piotroski_Score": 6, "Piotroski_Strong": "NO",
         "CoffeeCan": "FAIL"},
        {"Code": "E5", "Name": "Eps", "Piotroski_Score": np.nan, "Piotroski_Strong": "YES",
         "CoffeeCan": "FAIL"},
        {"Code": "F6", "Name": "Phi", "Piotroski_Score": 8, "Piotroski_Strong": "NO",
         "CoffeeCan": "FAIL"}])
    rows = BM._fundamentals_picks_rows(str(sandbox / "scan" / "m_*.xlsx"), ["Code"], True,
                                       "CoffeeCan")
    codes = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    assert codes == ["A1", "F6", "B2", "C3", "E5"]              # by score desc, NaN score last; D4 out
    assert "<td>Alpha</td>" in rows[0]                           # text before first comma
    assert "Triple Hit" in rows[0] and "Piotroski+CoffeeCan" in rows[0] and "<td>9</td>" in rows[0]
    assert "Multi-Screen" in rows[1] and "<td>Piotroski</td>" in rows[1]
    assert "<td>CoffeeCan</td>" in rows[3] and "<td>6</td>" in rows[3]      # coffee only
    assert "<td>n/a</td>" in rows[4] and "<td>Piotroski</td>" in rows[4]    # strong flag, no score


def test_fundamentals_picks_score_threshold_is_seven_inclusive(sandbox):
    fundamentals_xlsx(sandbox / "s" / "x_1.xlsx", [
        {"Symbol": "S7", "Piotroski_Score": 7, "CoffeeCan_Class": "FAIL"},
        {"Symbol": "S6", "Piotroski_Score": 6, "CoffeeCan_Class": "FAIL"}])
    rows = BM._fundamentals_picks_rows("s/x_*.xlsx", ["Symbol"], False, "CoffeeCan_Class")
    assert len(rows) == 1 and "S7" in rows[0]
    # without the strong-flag support, a "Piotroski_Strong" column is ignored
    fundamentals_xlsx(sandbox / "s" / "y_1.xlsx", [
        {"Symbol": "ST", "Piotroski_Score": 3, "Piotroski_Strong": "YES",
         "CoffeeCan_Class": "FAIL"}])
    assert BM._fundamentals_picks_rows("s/y_*.xlsx", ["Symbol"], False, "CoffeeCan_Class") == []


def test_fundamentals_picks_ticker_column_priority_name_default_and_top(sandbox):
    fundamentals_xlsx(sandbox / "s" / "z_1.xlsx", [
        {"Code": "7203", "YF_Ticker": "7203.T", "Piotroski_Score": 9, "CoffeeCan": "FAIL"},
        {"Code": "6758", "YF_Ticker": "6758.T", "Piotroski_Score": 8, "CoffeeCan": "FAIL"},
        {"Code": "1111", "YF_Ticker": "x", "Piotroski_Score": 7, "CoffeeCan": "FAIL"}])
    rows = BM._fundamentals_picks_rows("s/z_*.xlsx", ["Code", "YF_Ticker"], True, "CoffeeCan",
                                       top=2)
    assert len(rows) == 2
    assert "<b>7203</b></td><td>7203</td>" in rows[0]            # no Name column: name defaults to code
    assert "<b>6758</b>" in rows[1]
    only_yf = BM._fundamentals_picks_rows("s/z_*.xlsx", ["YF_Ticker", "Code"], True, "CoffeeCan",
                                          top=1)
    assert "<b>7203.T</b>" in only_yf[0]                           # earlier column wins


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="a blank Code cell reads back from Excel as NaN; str(NaN or '') is "
                          "'nan' (truthy), so the loop never falls through to YF_Ticker and the "
                          "row is labelled 'nan'")
def test_fundamentals_picks_blank_code_falls_back_to_next_column(sandbox):
    fundamentals_xlsx(sandbox / "s" / "q_1.xlsx", [
        {"Code": None, "YF_Ticker": "7203.T", "Piotroski_Score": 9, "CoffeeCan": "FAIL"}])
    rows = BM._fundamentals_picks_rows("s/q_*.xlsx", ["Code", "YF_Ticker"], True, "CoffeeCan")
    assert "<b>7203.T</b>" in rows[0]


def test_fundamentals_picks_missing_inputs(sandbox):
    assert BM._fundamentals_picks_rows("nothing/*.xlsx", ["Code"], True, "CoffeeCan") == []
    fundamentals_xlsx(sandbox / "s" / "w_1.xlsx", [{"a": 1}], sheet="Other")
    assert BM._fundamentals_picks_rows("s/w_*.xlsx", ["Code"], True, "CoffeeCan") == []
    (sandbox / "s" / "bad_1.xlsx").write_text("not excel")
    assert BM._fundamentals_picks_rows("s/bad_*.xlsx", ["Code"], True, "CoffeeCan") == []


def test_fundamentals_picks_uses_newest_file(sandbox):
    fundamentals_xlsx(sandbox / "s" / "n_1.xlsx", [{"Code": "OLD", "Piotroski_Score": 9,
                                                   "CoffeeCan": "PASS"}])
    fundamentals_xlsx(sandbox / "s" / "n_2.xlsx", [{"Code": "NEW", "Piotroski_Score": 9,
                                                   "CoffeeCan": "PASS"}])
    rows = BM._fundamentals_picks_rows("s/n_*.xlsx", ["Code"], True, "CoffeeCan")
    assert len(rows) == 1 and "NEW" in rows[0]


def test_market_wrappers_pass_their_scan_spec(monkeypatch):
    calls = []
    monkeypatch.setattr(BM, "_fundamentals_picks_rows", lambda *a, **k: calls.append((a, k)) or ["r"])
    assert BM._eu_picks_rows(top=3) == ["r"]
    assert BM._jp_picks_rows(top=4) == ["r"]
    assert BM._kr_picks_rows() == ["r"]
    (eu, eu_k), (jp, jp_k), (kr, kr_k) = calls
    assert eu == ("european_scan/european_market_scan*.xlsx",)
    assert eu_k == {"ticker_cols": ["Symbol"], "has_strong_flag": False,
                    "coffee_col": "CoffeeCan_Class", "top": 3}
    assert jp == ("japan_scan/japan_market_scan*.xlsx",) and jp_k["top"] == 4
    assert jp_k["has_strong_flag"] is True and jp_k["ticker_cols"] == ["Code", "YF_Ticker"]
    assert kr == ("korea_scan/korea_market_scan*.xlsx",) and kr_k["top"] == 15
    assert kr_k["coffee_col"] == "CoffeeCan"


# ── coverage notes ────────────────────────────────────────────────────────────
def test_scope_note_names_covered_and_parked(monkeypatch):
    wm = types.SimpleNamespace(COVERED=["IN", "US"],
                               PARKED={"JP": "no ratios", "XX": "ignored", "KR": "scores", "EU": "gap"})
    monkeypatch.setitem(sys.modules, "watchlist_markets", wm)
    out = BM._coverage_scope_note()
    assert "Per-name coverage: IN + US only." in out
    assert "<b>JP</b> no ratios · <b>KR</b> scores · <b>EU</b> gap" in out
    assert "ignored" not in out
    monkeypatch.setitem(sys.modules, "watchlist_markets", None)
    assert BM._coverage_scope_note() == ""


@pytest.fixture
def fund_store(tmp_path, monkeypatch):
    import data_registry as R
    monkeypatch.setattr(R, "FUND_DIR", tmp_path, raising=False)
    return tmp_path


def write_fund(path, n, ages_days, nan_rows=0):
    now = pd.Timestamp.now(tz="UTC")
    d = pd.DataFrame({
        "ticker": [f"{chr(65 + i % 26)}{i}" for i in range(n)],
        "cfo": 1.0, "total_assets": 2.0,
        "collected_at": [(now - pd.Timedelta(days=float(a), minutes=1)).isoformat() for a in ages_days]})
    if nan_rows:
        d.loc[:nan_rows - 1, "cfo"] = np.nan
    d.to_parquet(path / "IN_current.parquet", index=False)


def test_fund_note_filling_when_store_missing_or_small(fund_store):
    out = BM._fund_coverage_note()                           # no parquet at all
    assert "Fundamentals coverage is still filling" in out
    write_fund(fund_store, 49, [1] * 49)
    assert "still filling" in BM._fund_coverage_note()
    write_fund(fund_store, 50, [1] * 50)
    assert "still filling" not in BM._fund_coverage_note()
    write_fund(fund_store, 50, [1] * 50, nan_rows=1)          # one row lacks cfo -> 49 usable
    assert "still filling" in BM._fund_coverage_note()


def test_fund_note_age_and_staleness_boundary(fund_store):
    write_fund(fund_store, 60, [3] * 60)
    out = BM._fund_coverage_note()
    assert "60 names, 26 first-letters, median 3d old." in out and "STALE" not in out
    write_fund(fund_store, 60, [10] * 60)
    assert "median 10d old" in BM._fund_coverage_note() and "STALE" not in BM._fund_coverage_note()
    write_fund(fund_store, 60, [11] * 60)
    assert "median 11d old" in BM._fund_coverage_note() and "⚠ STALE" in BM._fund_coverage_note()


def test_fund_note_uses_median_not_newest(fund_store):
    write_fund(fund_store, 60, [20] * 59 + [0])
    out = BM._fund_coverage_note()
    assert "median 20d old" in out and "STALE" in out


# ── darvas section ────────────────────────────────────────────────────────────
def darvas_df(n1=2, n2=1, n3=0):
    rows = ([{"Tier": 1}] * n1) + ([{"Tier": 2}] * n2) + ([{"Tier": 3}] * n3)
    return pd.DataFrame(rows)


@pytest.fixture
def fake_darvas(monkeypatch):
    holder = {"df": darvas_df(), "seen": {}}

    def build(market, write_csv):
        holder["seen"].update(market=market, write_csv=write_csv)
        return "", holder["df"]
    monkeypatch.setattr(BM, "dbrk", types.SimpleNamespace(
        build=build, rows_html=lambda df, m: f"<tr><td>ROWS {len(df)} {m}</td></tr>"))
    return holder


def set_regime(monkeypatch, trend, vol="MID", boom=False):
    def get_regime(m):
        if boom:
            raise RuntimeError("no regime")
        return {"trend": trend, "vol": vol}
    monkeypatch.setitem(sys.modules, "market_regime", types.SimpleNamespace(get_regime=get_regime))


def test_darvas_unavailable_paths(monkeypatch):
    monkeypatch.setattr(BM, "dbrk", None)
    assert "not importable" in BM._darvas_section("IN")

    def boom(m, write_csv):
        raise RuntimeError("scan gone")
    monkeypatch.setattr(BM, "dbrk", types.SimpleNamespace(build=boom))
    assert "Darvas fragment for IN unavailable (scan gone)" in BM._darvas_section("IN")


def test_darvas_empty_frame(fake_darvas):
    fake_darvas["df"] = pd.DataFrame()
    out = BM._darvas_section("EU")
    assert "🇪🇺 Europe — Darvas Breakouts" in out and "No fresh breakouts" in out
    assert fake_darvas["seen"] == {"market": "EU", "write_csv": False}


def test_darvas_counts_cap_and_label(fake_darvas, monkeypatch):
    set_regime(monkeypatch, "bull", "LOW")
    fake_darvas["df"] = darvas_df(n1=3, n2=2, n3=5)
    out = BM._darvas_section("US", cap=4)
    assert "(3 Tier-1 · 2 Tier-2, top 4 of 10 shown)" in out and "ROWS 4 US" in out
    assert "🇺🇸 US" in out and "BULL regime · LOW vol" in out
    assert "color:#1a7f37" in out and "watch-only" not in out and "size down" not in out
    exact = BM._darvas_section("US", cap=10)                      # len == cap: no truncation note
    assert "shown" not in exact and "ROWS 10 US" in exact
    default = BM._darvas_section("US")                            # default cap 30
    assert "ROWS 10 US" in default
    assert "ZZ — Darvas" in BM._darvas_section("ZZ")              # unknown market keeps its code


@pytest.mark.parametrize("trend,vol,needle,colour", [
    ("bear", "LOW", "breakouts historically dead in bear", "#c62828"),
    ("chop", "HIGH", "chop + HIGH vol: weakest breakout", "#b26a00"),
    ("bull", "HIGH", "HIGH vol: breakout follow-through", "#1a7f37"),
    ("chop", "MID", None, "#b26a00"),
    ("unknown", "unknown", None, "#777"),
])
def test_darvas_regime_messages(fake_darvas, monkeypatch, trend, vol, needle, colour):
    set_regime(monkeypatch, trend, vol)
    out = BM._darvas_section("IN")
    assert f"font-weight:600;color:{colour}" in out
    assert f"{trend.upper()} regime · {vol} vol" in out
    messages = ["historically dead in bear", "weakest breakout cell", "follow-through"]
    present = [m for m in messages if m in out]
    if needle is None:
        assert present == []
    else:
        assert len(present) == 1 and needle in out


def test_darvas_regime_failure_degrades_to_unknown(fake_darvas, monkeypatch):
    set_regime(monkeypatch, "x", boom=True)
    out = BM._darvas_section("IN")
    assert "UNKNOWN regime · unknown vol" in out and "color:#777" in out


# ── correlation section ───────────────────────────────────────────────────────
def corr_file(root, key, text):
    d = root / "correlation_scan"
    d.mkdir(exist_ok=True)
    (d / f"{key}_correlation_clusters.txt").write_text(text)


def test_corr_missing_files_say_na(sandbox):
    out = BM._corr_section()
    assert out.count("n/a (no correlation scan yet)") == 2
    assert "🇮🇳 India (NSE)" in out and "🇺🇸 US" in out


def test_corr_parses_header_clusters_and_limits(sandbox):
    members = ", ".join(f"'S{i}'" for i in range(10))
    corr_file(sandbox, "nse", "Scan: 1,234 symbols, period=1y\n"
              f"  (10) [{members}]\n  (3) ['A', 'B', 'C']\n  (2) ['D', 'E']\n  (2) ['F', 'G']\n")
    out = BM._corr_section(top_n=3)
    assert "(1,234 symbols, 4 clusters found)" in out
    assert "<li>(10) S0, S1, S2, S3, S4, S5, S6, S7 …</li>" in out    # 8 shown + ellipsis
    assert "<li>(3) A, B, C</li>" in out and "<li>(2) D, E</li>" in out
    assert "F, G" not in out                                          # top_n=3
    one = BM._corr_section(top_n=1)
    assert "(3) A, B, C" not in one and "S0" in one


def test_corr_exactly_eight_members_has_no_ellipsis(sandbox):
    members = ", ".join(f"'S{i}'" for i in range(8))
    corr_file(sandbox, "us", f"x 5 symbols period=1y\n(8) [{members}]\n")
    out = BM._corr_section()
    assert "<li>(8) S0, S1, S2, S3, S4, S5, S6, S7</li>" in out


def test_corr_no_clusters_and_unknown_header(sandbox):
    corr_file(sandbox, "nse", "300 symbols period=1y\nno clusters here\n")
    corr_file(sandbox, "us", "garbage only\n")
    out = BM._corr_section()
    assert "300 symbols scanned, no clusters &gt;1 found" in out
    assert "? symbols scanned, no clusters" in out


def test_corr_bad_member_list_and_bad_line_skipped(sandbox):
    corr_file(sandbox, "nse", "9 symbols period=1y\n(2) [not python]\n(zzz) ['X']\n(2) ['K', 'L']\n")
    out = BM._corr_section()
    assert "<li>(2) </li>" in out and "(2) K, L" in out and "zzz" not in out


def test_corr_unreadable_file_degrades_per_market(sandbox):
    (sandbox / "correlation_scan" / "nse_correlation_clusters.txt").mkdir(parents=True)  # a dir
    corr_file(sandbox, "us", "2 symbols period=1y\n(2) ['A', 'B']\n")
    out = BM._corr_section()
    assert "n/a (error reading correlation scan)" in out and "(2) A, B" in out


# ── as-of banner ──────────────────────────────────────────────────────────────
def test_as_of_empty_when_no_scans(sandbox):
    assert BM._as_of_html() == ""


def test_as_of_lists_scans_with_mtime(sandbox):
    f = sandbox / "us_full_scan" / "us_full_scan_20260101.xlsx"
    f.parent.mkdir()
    f.write_text("x")
    t = dt.datetime(2026, 3, 4, 7, 5).timestamp()
    os.utime(f, (t, t))
    k = sandbox / "korea_scan" / "korea_market_scan_1.xlsx"
    k.parent.mkdir()
    k.write_text("x")
    os.utime(k, (t, t))
    before = dt.datetime.now()
    out = BM._as_of_html()
    after = dt.datetime.now()
    assert "🇺🇸 US" in out and "scanned 04 Mar 07:05" in out and "🇰🇷 Korea" in out
    assert "Europe" not in out and "Japan" not in out
    assert "Prices are last official closes" in out
    assert f"{before:%d %b %Y}" in out or f"{after:%d %b %Y}" in out
    assert out.count("<tr>") == 2


# ── CCC rows ──────────────────────────────────────────────────────────────────
def ccc_frame():
    return pd.DataFrame({
        "Symbol": ["HI", "MID", "LOWL", "NEG", "NAN", "UNK"],
        "Name": ["High Co", "Mid Co", "Low Co", "Neg Co", "Nan Co", "Unk Co"],
        "Cash_Cycle": ["30.0", "10", "5", "-4.5", "n/a", "1"],
        "ROCE": [20, 15, 10, 30, 5, 1]})


@pytest.fixture
def ccc_env(sandbox, monkeypatch):
    (sandbox / "cache_seed").mkdir()
    write_liq(sandbox, [
        {"Symbol": "HI", "Market": "IN", "turnover_usd": 9e6, "ltp": 1},
        {"Symbol": "MID", "Market": "IN", "turnover_usd": 6e5, "ltp": 1},
        {"Symbol": "LOWL", "Market": "IN", "turnover_usd": 1e5, "ltp": 1},
        {"Symbol": "NEG", "Market": "IN", "turnover_usd": 7e6, "ltp": 1}])
    holder = {"df": ccc_frame()}
    monkeypatch.setitem(sys.modules, "screener_in",
                        types.SimpleNamespace(ccc_screen=lambda: holder["df"]))
    return holder


def test_ccc_rows_filter_liquidity_sort_and_cache(ccc_env, sandbox):
    rows = BM._ccc_rows()
    syms = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    assert syms == ["NEG", "MID", "HI"]                      # ascending CCC; Low/Unknown/NaN dropped
    assert "-4.5" in rows[0] and "<td>Neg Co</td>" in rows[0] and "<td>30</td>" in rows[0]
    assert "color:#b8860b" in rows[1] and ">Medium<" in rows[1] and "30.0" in rows[2]
    assert (sandbox / "cache_seed" / "india_ccc_screen.parquet").exists()
    assert len(BM._ccc_rows(top=2)) == 2


def test_ccc_rows_fall_back_to_cache_when_fetch_fails(ccc_env, sandbox, monkeypatch):
    BM._ccc_rows()                                            # populates the cache
    monkeypatch.setitem(sys.modules, "screener_in", types.SimpleNamespace(
        ccc_screen=lambda: (_ for _ in ()).throw(RuntimeError("blocked"))))
    assert len(BM._ccc_rows()) == 3
    ccc_env["df"] = pd.DataFrame()
    monkeypatch.setitem(sys.modules, "screener_in", types.SimpleNamespace(ccc_screen=lambda: ccc_env["df"]))
    assert len(BM._ccc_rows()) == 3                           # empty live result -> cache too


def test_ccc_rows_no_data_anywhere(ccc_env, monkeypatch):
    ccc_env["df"] = pd.DataFrame()
    assert BM._ccc_rows() == []                               # no cache file either
    ccc_env["df"] = pd.DataFrame({"Symbol": ["HI"], "Other": [1]})
    assert BM._ccc_rows() == []                               # no Cash_Cycle column


# ── market snapshot ───────────────────────────────────────────────────────────
def fake_yf(monkeypatch, series_by_ticker):
    def download(tkr, **kw):
        s = series_by_ticker[tkr]
        if isinstance(s, Exception):
            raise s
        return pd.DataFrame({"Close": s})
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))


def test_snapshot_without_yfinance_shows_na(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    set_regime(monkeypatch, "bull", boom=True)               # regime strip fails -> ""
    out = BM._market_snapshot_html()
    assert out.count("n/a") >= 9 and "Regime gate" not in out
    for name in ("Nifty 50", "S&P 500", "Euro Stoxx 50"):
        assert name in out


def test_snapshot_dma_stance_and_cross(monkeypatch):
    up = np.linspace(100, 300, 260)
    down = np.linspace(300, 100, 260)
    short = np.linspace(100, 110, 60)
    fake_yf(monkeypatch, {"^NSEI": up, "^GSPC": down, "^STOXX50E": short})
    set_regime(monkeypatch, "bull")
    out = BM._market_snapshot_html()
    cells = out.split("<td style='padding:8px 12px")[1:]
    nifty, spx, stoxx = cells
    assert f"{up[-1]:,.0f}" in nifty and "above 200-DMA (bullish)" in nifty
    assert f"200-DMA {up[-200:].mean():,.0f}" in nifty and f"50-DMA {up[-50:].mean():,.0f}" in nifty
    assert "above 50-DMA" in nifty and "Golden Cross" in nifty and "#2e7d32" in nifty
    assert "below 200-DMA (cautious)" in spx and "below 50-DMA" in spx and "Death Cross" in spx
    assert "#c62828" in spx
    assert "above 50-DMA" in stoxx and "200-DMA n/a" in stoxx and "Cross" not in stoxx
    assert "n/a · 200-DMA" in stoxx and "above 200" not in stoxx


def test_snapshot_dma_window_boundaries(monkeypatch):
    for n, has50, has200 in ((49, False, False), (50, True, False), (199, True, False),
                             (200, True, True)):
        fake_yf(monkeypatch, {t: np.linspace(100, 150, n) for t in ("^NSEI", "^GSPC", "^STOXX50E")})
        set_regime(monkeypatch, "bull")
        out = BM._market_snapshot_html()
        assert ("above 50-DMA" in out) is has50, n
        assert ("above 200-DMA" in out) is has200, n


def test_snapshot_download_errors_isolated_per_index(monkeypatch):
    good = np.linspace(100, 300, 260)
    fake_yf(monkeypatch, {"^NSEI": RuntimeError("boom"), "^GSPC": good, "^STOXX50E": good})
    set_regime(monkeypatch, "chop", boom=True)
    out = BM._market_snapshot_html()
    cells = out.split("<td style='padding:8px 12px")[1:]
    assert ">n/a<" in cells[0] and "above 200-DMA" in cells[1] and "above 200-DMA" in cells[2]


def test_snapshot_regime_strip(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    set_regime(monkeypatch, "bear", "HIGH")
    out = BM._market_snapshot_html()
    assert "Regime gate:" in out and out.count("BEAR") == 6 and "/HIGH" in out
    for flag in ("🇮🇳", "🇺🇸", "🇪🇺", "🇯🇵", "🇰🇷", "🇨🇳"):
        assert flag in out
    assert "color:#c62828" in out


# ── build() ───────────────────────────────────────────────────────────────────
@pytest.fixture
def build_env(sandbox, monkeypatch):
    for stub in ("_as_of_html", "_market_snapshot_html", "_fund_coverage_note",
                 "_coverage_scope_note", "_corr_section"):
        monkeypatch.setattr(BM, stub, lambda *a, _s=stub, **k: f"[{_s}]")
    monkeypatch.setattr(BM, "_ccc_rows", lambda: [])
    monkeypatch.setattr(BM, "_news_rows", lambda m: [])
    for name in ("_eu_picks_rows", "_jp_picks_rows", "_kr_picks_rows"):
        monkeypatch.setattr(BM, name, lambda: [])
    monkeypatch.setattr(BM, "_darvas_section", lambda m: f"[darvas {m}]")
    monkeypatch.setitem(sys.modules, "carry_fx", types.SimpleNamespace(carry_html=lambda: "<p>carry!</p>"))
    allh = pd.DataFrame({
        "Market": ["IN", "TW", "KR", "TW", "ZZ", "JP"],
        "Symbol": ["a", "t1", "k1", "t2", "z1", "j1"],
        "ret_126": [50.0, 10.0, 30.0, 40.0, 99.0, 5.0], "rsi14": [55, 60, 65, 70, 75, 80]})
    monkeypatch.setattr(BM.rga, "load_highlights", lambda: allh)
    write_liq(sandbox, [{"Symbol": s, "Market": m, "turnover_usd": 9e6, "ltp": 1}
                        for s, m in zip(allh.Symbol, allh.Market)])
    monkeypatch.setattr(BM.mp, "load", lambda: pd.DataFrame({
        "Market": ["IN"], "Index": ["NIFTY"], "CAGR%": [12.3], "Return_1y%": [8.1], "Sharpe": [0.9]}))
    return sandbox


def test_build_assembles_subject_text_html(build_env):
    subject, text, html = BM.build()
    today = dt.date.today().strftime("%d %b %Y")
    assert subject == f"📈 Daily Market Brief — {today}"
    assert text.startswith(f"Daily Market Brief — {today}\n")
    assert "India mood: n/a (+0.00). US mood: n/a (+0.00)." in text
    assert "NOT investment advice" in text
    assert "mood n/a (+0.00) from 0 articles" in html
    for marker in ("[darvas IN]", "[darvas US]", "[darvas EU]", "<p>carry!</p>", "[_corr_section]",
                   "[_as_of_html]", "[_market_snapshot_html]", "no picks</p>"):
        assert marker in html
    assert "no picks (run daily_combined_report.py --market US)" in html
    assert "no European scan available today" in html and "no Japan scan" in html
    assert "no Korea scan available today" in html
    # momentum rows: top 15 (here all 6), signed rounded return
    assert "'>IN</td><td>a</td><td>+50</td><td>55</td>" in html
    # other markets: best 6mo mover per non-IN market, in the module's market order
    other = html[html.index("Other markets"):html.index("20-market 5-year")]
    assert other.index("Taiwan") < other.index("Korea") < other.index("Japan")
    assert "<b>t2</b></td><td>+40%" in other and "t1" not in other
    assert "z1" not in other and "S.Africa" not in other
    assert "<td>NIFTY</td><td>12.3</td><td>8.1</td><td>0.9</td>" in html


def test_build_uses_combined_json_and_mood(build_env):
    d = build_env / "combined_report_results"
    d.mkdir()
    (d / "combined_IN_1.json").write_text(json.dumps(
        {"mood": {"mood": "BULLISH", "score": 0.5, "n_articles": 12}, "picks": []}))
    _, _, html = BM.build()
    assert "mood BULLISH (+0.50) from 12 articles" in html
    assert "mood n/a (+0.00) from 0 articles" in html          # US side has no file


def test_build_carry_failure_message_truncated(build_env, monkeypatch):
    def boom():
        raise RuntimeError("x" * 100)
    monkeypatch.setitem(sys.modules, "carry_fx", types.SimpleNamespace(carry_html=boom))
    _, _, html = BM.build()
    assert "carry FX unavailable (" + "x" * 40 + ")" in html and "x" * 41 not in html


def test_build_tolerates_picker_and_corr_failures(build_env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    for n in ("_eu_picks_rows", "_jp_picks_rows", "_kr_picks_rows"):
        monkeypatch.setattr(BM, n, boom)
    monkeypatch.setattr(BM, "_corr_section", boom)
    _, _, html = BM.build()
    assert "no European scan available today" in html
    assert "Correlation scan unavailable." in html


def test_build_picks_tables_when_rows_present(build_env, monkeypatch):
    monkeypatch.setattr(BM, "_eu_picks_rows", lambda: ["<tr><td>EUROW</td></tr>"])
    monkeypatch.setattr(BM, "_ccc_rows", lambda: ["<tr><td>CCCROW</td></tr>"])
    _, _, html = BM.build()
    assert "EUROW" in html and "CCCROW" in html and "no European scan" not in html


# ── token logging ─────────────────────────────────────────────────────────────
def test_log_tokens_payload_and_minimums(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    real_exists = BM.Path.exists
    monkeypatch.setattr(BM.Path, "exists",
                        lambda self: True if self.name == "log-token-usage.js" else real_exists(self))
    BM._log_tokens_to_ruflo("subj", "text", "<b>x</b>", duration_seconds=1.5)
    cmd, kw = calls[0]
    payload = json.loads(kw["input"])
    assert cmd[0] == "node" and cmd[1].endswith("log-token-usage.js") and kw["timeout"] == 5
    assert payload["inputTokens"] == 500 and payload["outputTokens"] == 2000     # floors
    assert payload["durationSeconds"] == 1.5 and payload["taskId"] == "watchlist_mailer"
    big = "y" * 40_000
    BM._log_tokens_to_ruflo("s" * 4000, "t" * 4000, big)
    payload = json.loads(calls[1][1]["input"])
    assert payload["inputTokens"] == 2000 and payload["outputTokens"] == 10_000   # chars // 4


def test_log_tokens_silent_when_logger_missing_or_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(BM.Path, "exists", lambda self: False)
    BM._log_tokens_to_ruflo("s", "t", "h")
    assert calls == []
    monkeypatch.setattr(BM.Path, "exists", lambda self: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: 1 / 0)
    BM._log_tokens_to_ruflo("s", "t", "h")                    # must not raise
