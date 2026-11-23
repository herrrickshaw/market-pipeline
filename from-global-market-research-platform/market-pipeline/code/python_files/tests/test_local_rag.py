"""local_rag.py -- intent routing, grounded extractors over report CSV/markdown files,
psql-backed queries (subprocess faked), corpus chunking and the answer() pipeline.

DataFrame.to_markdown needs the optional `tabulate`, which CI does not install, so the table
renderer is replaced by a plain-text one; the logic under test is what is selected and shown."""

import subprocess
import sys

import pandas as pd
import pytest

import local_rag as LR


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(LR, "HERE", tmp_path)
    monkeypatch.setattr(LR, "REP", tmp_path / "reports")
    monkeypatch.setattr(
        LR, "_md_table", lambda df, n=None: (df.head(n) if n else df).to_csv(index=False).strip()
    )
    return tmp_path


def rep(env, name, text=None, df=None):
    p = env / "reports" / name
    if df is not None:
        df.to_csv(p, index=False)
    else:
        p.write_text(text)
    return p


# ── market parsing ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "q,mk",
    [
        ("buy signals in India", "IN"),
        ("Indian picks", "IN"),
        ("korean value", "KR"),
        ("Japan buy", "JP"),
        ("european cheap", "EU"),
        ("china tech", "CN"),
        ("united states stocks", "US"),
        ("america", "US"),
        ("signals for us", "US"),
        ("what about uk", "EU"),
        ("eu zones", "EU"),
        ("kr zones", "KR"),
        ("jp now", "JP"),
        ("cn now", "CN"),
        ("buy signals", None),
        ("business results", None),
        ("tell me", None),
    ],
)
def test_market_from_question(q, mk):
    assert LR._market_from_q(q) == mk


def test_market_codes_need_word_boundaries():
    assert LR._market_from_q("focus on usual cases") is None  # ' us' inside a word
    assert LR._market_from_q("us") == "US"
    assert LR._market_from_q("find us now") == "US"


def test_market_full_names_win_over_codes():
    assert LR._market_from_q("korea vs us") == "KR"


# ── routing ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "q,fn",
    [
        ("how long does the pipeline take", LR.pipeline_perf),
        ("is it slow", LR.pipeline_perf),
        ("re-entry candidates", LR.reentry),
        ("reentry queue", LR.reentry),
        ("tier 3 anomalies", LR.reentry),
        ("market playbook please", LR.playbook),
        ("what strategy for japan market", LR.playbook),
        ("where is the edge", LR.edge_map),
        ("fat pitch grid", LR.edge_map),
        ("speculation by sector", LR.speculation),
        ("pb roe regime", LR.speculation),
        ("etl status", LR.etl_status),
        ("load batch lineage", LR.etl_status),
        ("edinet roe leaders", LR.global_fund),
        ("global fundamentals coverage", LR.global_fund),
        ("income statement", LR.income_statement),
        ("profit and loss", LR.income_statement),
        ("balance sheet", LR.balance_sheet),
        ("top stocks", LR.picks),
        ("any picks?", LR.picks),
        ("sell signals", LR.signals),
        ("watchlist", LR.signals),
        ("underpriced in india", LR.valuation),
        ("mispriced", LR.valuation),
        ("is the data sufficient", LR.sufficiency),
        ("data sources", LR.sources),
    ],
)
def test_route_picks_the_extractor(q, fn):
    assert LR.route(q) is fn


def test_route_unmatched_and_priority():
    assert LR.route("what is the weather") is None
    assert LR.route("how long do the buy signals take") is LR.pipeline_perf  # earlier intent wins


# ── file-backed extractors ────────────────────────────────────────────────────
def test_income_statement_and_balance_sheet(env):
    assert "income_statement_by_geo.csv not found" in LR.income_statement()
    assert "balance_sheet_by_geo.csv not found" in LR.balance_sheet()
    rep(env, "income_statement_by_geo.csv", df=pd.DataFrame({"geo": ["IN"], "rev": [5]}))
    rep(env, "balance_sheet_by_geo.csv", df=pd.DataFrame({"geo": ["US"], "assets": [9]}))
    assert LR.income_statement().endswith("geo,rev\nIN,5")
    assert (
        LR.balance_sheet().startswith("**Balance sheet by geography:**")
        and "US,9" in LR.balance_sheet()
    )


def picks_df():
    return pd.DataFrame(
        {
            "market": ["IN", "IN", "US"],
            "symbol": ["A", "B", "C"],
            "name": ["a", "b", "c"],
            "filter": ["f"] * 3,
            "direction": ["long"] * 3,
            "pe": [10, 12, 15],
            "roe": [0.2, 0.3, 0.1],
            "extra": [1, 2, 3],
        }
    )


def test_picks_prefers_the_live_playbook_and_filters_by_market(env):
    rep(env, "playbook_picks.csv", df=picks_df())
    out = LR.picks("picks for india")
    assert (
        "Live playbook picks — IN" in out
        and "A,a,f,long,10,0.2" in out
        and "C" not in out.split("\n\n", 1)[1]
    )
    assert "extra" not in out
    allm = LR.picks("any picks")
    assert "(all markets)" in allm and "C,c" in allm
    assert len(LR.picks("x").split("\n")) >= 5


def test_picks_falls_back_to_the_learned_model(env):
    assert "No picks for IN" in LR.picks("")
    pd.DataFrame({"t": ["A", "B", "C"], "p": [0.01, 0.05, -0.02], "z": [0, 0, 0]}).to_csv(
        env / "reports" / "recs_KR.csv", index=False
    )
    out = LR.picks("korea picks")
    assert out.startswith("**Top KR picks**")
    rows = out.split("\n")[-4:]
    assert rows[1].startswith("B,5.0%") and rows[-1].startswith(
        "C,-2.0%"
    )  # sorted desc, formatted %


def test_picks_unknown_market_in_playbook_falls_through(env):
    rep(env, "playbook_picks.csv", df=picks_df())
    rep(env, "recs_JP.csv", df=pd.DataFrame({"t": ["Z"], "p": [0.1]}))
    assert LR.picks("japan picks").startswith("**Top JP picks**")


def test_signals_summary_and_actionable_names(env):
    pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "market": ["IN", "IN", "IN", "US"],
            "status": ["signal", "held", "sold", "signal"],
        }
    ).to_csv(env / "watchlist.csv", index=False)
    out = LR.signals("india buy signals")
    assert "Watchlist signals — IN" in out and "Actionable names (IN): A, B" in out
    assert "Not investment advice" in out
    allm = LR.signals("signals")
    assert "Watchlist signals:" in allm and "Actionable names" not in allm
    assert "IN,held,1" in allm and "US,signal,1" in allm


def valuation_df():
    return pd.DataFrame(
        {
            "market": ["india", "india", "us", "KR"],
            "name": ["a", "b", "c", "d"],
            "pe": [1, 2, 3, 4],
            "pb": [1, 1, 1, 1],
            "roe": [0.1] * 4,
            "valuation_z": [-2.0, 1.0, 3.0, -0.5],
            "verdict": ["x"] * 4,
            "junk": [0] * 4,
        }
    )


def test_valuation_sides_and_market_normalisation(env):
    assert "valuation_clusters.csv not found" in LR.valuation("")
    rep(env, "valuation_clusters.csv", df=valuation_df())
    under = LR.valuation("underpriced")
    assert "UNDERPRICED" in under and under.split("\n\n")[1].split("\n")[1].startswith("IN,a")
    over = LR.valuation("expensive")
    assert "OVERPRICED" in over and over.split("\n\n")[1].split("\n")[1].startswith("US,c")
    both = LR.valuation("valuation")
    assert "both tails" in both and both.split("\n\n")[1].split("\n")[1].startswith("US,c")
    ind = LR.valuation("cheap india")
    assert "· IN" in ind and "US" not in ind.split("\n\n", 1)[1]
    assert "junk" not in ind


def test_sufficiency_and_sources(env):
    assert "run `python data_sufficiency.py`" in LR.sufficiency()
    rep(
        env,
        "data_sufficiency.md",
        "# top\n\n| market | fund tickers | ok |\n|--|--|--|\n| IN | 5 | y |\n\n## next\nzzz",
    )
    out = LR.sufficiency()
    assert "| IN | 5 | y |" in out and "zzz" not in out
    rep(env, "data_sufficiency.md", "no table here " * 200)
    assert len(LR.sufficiency()) < 1600 and "no table here" in LR.sufficiency()
    assert "source_registry.py" in LR.sources()
    (env / "docs" / "DATA_SOURCES.md").write_text("S" * 5000)
    out = LR.sources()
    assert out.startswith("**Data sources registry:**") and out.count("S") == 2500


def test_playbook_market_section_or_overview(env):
    assert "market_playbook.py" in LR.playbook("")
    rep(
        env,
        "market_playbook.md",
        "# Playbook\n\n## IN — india edges\nbuy momentum\n\n## US — us edges\nrevert\n",
    )
    assert "buy momentum" in LR.playbook("india playbook") and "revert" not in LR.playbook(
        "india playbook"
    )
    assert "Market playbook — retail-accessible edges" in LR.playbook("playbook")
    assert "revert" in LR.playbook("playbook for korea")  # KR has no section: overview


def test_edge_map_columns(env):
    assert "edge_matrix.py" in LR.edge_map("")
    rep(env, "edge_matrix.csv", df=pd.DataFrame({"strategy": ["s"], "IN": [1], "US": [2]}))
    assert "strategy,IN,US" in LR.edge_map("anything")
    assert (
        "strategy,IN\n" in LR.edge_map("india")
        and "US" not in LR.edge_map("india").split("\n\n", 1)[1]
    )
    assert "strategy,IN,US" in LR.edge_map("korea")  # column absent: all shown


def test_speculation_sorted_by_r2(env):
    assert "fundamentals_vs_speculation.py" in LR.speculation("")
    rep(
        env,
        "fundamentals_vs_speculation.csv",
        df=pd.DataFrame(
            {
                "market": ["IN", "IN", "US"],
                "sector": ["a", "b", "c"],
                "R2_pb_roe": [0.2, 0.9, 0.5],
                "n": [1, 2, 3],
            }
        ),
    )
    body = LR.speculation("").split("\n\n")[1].split("\n")
    assert [r.split(",")[1] for r in body[1:]] == ["b", "c", "a"]
    ind = LR.speculation("india").split("\n\n")[1].split("\n")
    assert [r.split(",")[1] for r in ind[1:]] == ["b", "a"]
    rep(
        env,
        "fundamentals_vs_speculation.csv",
        df=pd.DataFrame({"sector": ["a", "b"], "x": [1, 2], "r": [0.1, 0.9], "n": [1, 1]}),
    )
    assert (
        LR.speculation("").split("\n\n")[1].split("\n")[1].startswith("b")
    )  # falls back to the second-last column


def test_reentry_sections(env):
    assert LR.reentry("") == "run `python reentry_engine.py` first."
    rep(env, "reentry_candidates.csv", df=pd.DataFrame({"s": []}))
    assert "_none today_" in LR.reentry("")
    rep(env, "reentry_candidates.csv", df=pd.DataFrame({"s": ["A"]}))
    rep(env, "reentry_recent.csv", df=pd.DataFrame({"market": ["IN", "US"], "s": ["x", "y"]}))
    out = LR.reentry("india")
    assert "s\nA" in out and "Recent re-entry names (last 45d, 1)" in out and "boomerang" in out
    assert "(last 45d, 2)" in LR.reentry("")


# ── psql-backed extractors ────────────────────────────────────────────────────
class Proc:
    def __init__(self, out="", rc=0):
        self.stdout, self.returncode = out, rc


def fake_psql(monkeypatch, out="", rc=0, boom=False):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if boom:
            raise FileNotFoundError("psql")
        return Proc(out, rc)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_psql_df_parsing_and_failures(monkeypatch):
    calls = fake_psql(monkeypatch, "a,b\n1,2\n")
    d = LR._psql_df("SELECT 1")
    assert list(d.columns) == ["a", "b"] and d.iloc[0].tolist() == [1, 2]
    assert calls[0] == ["psql", "-d", "market_data", "--csv", "-c", "SELECT 1"]
    fake_psql(monkeypatch, "a\n1\n", rc=1)
    assert LR._psql_df("x") is None
    fake_psql(monkeypatch, "   \n")
    assert LR._psql_df("x") is None
    fake_psql(monkeypatch, boom=True)
    assert LR._psql_df("x") is None


def test_global_fund_queries(env, monkeypatch):
    sqls = []
    monkeypatch.setattr(
        LR, "_psql_df", lambda sql: sqls.append(sql) or pd.DataFrame({"m": ["IN"], "n": [1]})
    )
    out = LR.global_fund("ROE leaders in india")
    assert "ROE leaders (FY2024+, official filings) · IN" in out and "Sources: EDGAR(US)" in out
    assert "WHERE market='IN' AND" in sqls[0] and "ORDER BY 4 DESC" in sqls[0]
    LR.global_fund("best roe")
    assert "WHERE market" not in sqls[1] and "global_fundamentals  WHERE\n" in sqls[1]
    cov = LR.global_fund("fundamentals coverage for korea")
    assert (
        "global_fundamentals coverage (14 markets, one schema) · KR" in cov
        and "GROUP BY market" in sqls[2]
    )
    monkeypatch.setattr(LR, "_psql_df", lambda sql: None)
    assert "not reachable" in LR.global_fund("coverage")
    monkeypatch.setattr(LR, "_psql_df", lambda sql: pd.DataFrame())
    assert "not reachable" in LR.global_fund("coverage")


def test_etl_status_and_pipeline_perf(env, monkeypatch):
    monkeypatch.setattr(LR, "_psql_df", lambda sql: None)
    assert "no ETL batches" in LR.etl_status()
    assert (
        "step_timings unreachable" in LR.pipeline_perf()
        and "Performance decisions" in LR.pipeline_perf()
    )
    monkeypatch.setattr(LR, "_psql_df", lambda sql: pd.DataFrame({"batch_id": [3]}))
    assert "ETL batch lineage" in LR.etl_status() and "batch_id\n3" in LR.etl_status()
    assert "batch_id\n3" in LR.pipeline_perf()


# ── corpus / retrieval / answer ───────────────────────────────────────────────
def test_build_corpus_chunks_on_headings_and_drops_short_chunks(env):
    long_a = "alpha " * 30
    rep(env, "a.md", f"# Title\n{long_a}\n## Sub\nshort\n## Third\n{'beta ' * 30}")
    (env / "docs" / "b.md").write_text(f"### Doc\n{'gamma ' * 30}")
    docs = LR.build_corpus()
    assert [d["source"] for d in docs] == ["a.md", "a.md", "b.md"]
    assert all(len(d["text"]) > 80 for d in docs) and docs[1]["text"].startswith("## Third")


def test_build_corpus_boundary_length(env):
    rep(env, "a.md", "x" * 80 + "\n## b\n" + "y" * 81)  # first chunk is exactly 80 chars
    docs = LR.build_corpus()
    assert len(docs) == 1 and docs[0]["text"].startswith("## b")
    rep(env, "a.md", "x" * 81)
    assert len(LR.build_corpus()) == 1


def test_local_llm_tries_both_endpoints(monkeypatch):
    import requests

    seen = []

    class R:
        def __init__(self, code, text=""):
            self.status_code, self._t = code, text

        def json(self):
            return {"choices": [{"message": {"content": f" {self._t} "}}]}

    def post(url, json, timeout):
        seen.append((url, json["model"], timeout))
        if "1234" in url:
            raise OSError("down")
        return R(200, "hello")

    monkeypatch.setattr(requests, "post", post)
    assert LR.local_llm("q") == "hello"
    assert [s[1] for s in seen] == ["local-model", "llama3"] and seen[0][2] == 60
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(500))
    assert LR.local_llm("q") is None
    monkeypatch.setattr(requests, "post", lambda *a, **k: 1 / 0)
    assert LR.local_llm("q") is None


def test_local_llm_without_requests(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    assert LR.local_llm("q") is None


def test_answer_structured_adds_context(env, monkeypatch):
    monkeypatch.setattr(
        LR, "retrieve", lambda q, k: [({"source": "x.md", "text": "line1\nline2"}, 0.5)]
    )
    rep(env, "income_statement_by_geo.csv", df=pd.DataFrame({"g": ["IN"]}))
    out = LR.answer("income statement")
    assert out.endswith("---\n*Context:* x.md — line1 line2")
    monkeypatch.setattr(LR, "retrieve", lambda q, k: [])
    assert "---" not in LR.answer("income statement")


def test_answer_open_question_paths(env, monkeypatch):
    monkeypatch.setattr(LR, "retrieve", lambda q, k: [])
    assert LR.answer("what is the weather").startswith("No relevant context found")
    ctx = [({"source": "a.md", "text": "T" * 900}, 0.4321), ({"source": "b.md", "text": "u"}, 0.1)]
    monkeypatch.setattr(LR, "retrieve", lambda q, k: ctx if k == 4 else [])
    monkeypatch.setattr(LR, "local_llm", lambda p: None)
    out = LR.answer("what is the weather")
    assert "extractive answer" in out and "**a.md** (score 0.43)" in out and out.count("T") == 700
    prompts = []
    monkeypatch.setattr(LR, "local_llm", lambda p: prompts.append(p) or "LLM says hi")
    out = LR.answer("what is the weather")
    assert out == "LLM says hi\n\n*(sources: a.md, b.md)*"
    assert (
        prompts[0].startswith("Context:\n[a.md] ") and "Question: what is the weather" in prompts[0]
    )


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_modes(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert LR.main() == 0
    assert capsys.readouterr().out.strip() == str(LR.__doc__).strip()
    monkeypatch.setattr(LR, "answer", lambda q: f"ANSWER<{q}>")
    monkeypatch.setattr(sys, "argv", ["x", "hello", "world"])
    assert LR.main() == 0 and "ANSWER<hello world>" in capsys.readouterr().out
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, cwd: ran.append((cmd, cwd)))
    monkeypatch.setattr(sys, "argv", ["x", "--refresh", "q"])
    LR.main()
    assert ran[0][0][1].endswith("trading_financials.py") and ran[0][1] == env
    assert "refreshing financial statements" in capsys.readouterr().out


# ── mutation follow-ups: row caps and context depth ───────────────────────────
def big_valuation_df(n=20):
    return pd.DataFrame(
        {
            "market": ["india"] * n,
            "name": [f"co{i:02d}" for i in range(n)],
            "pe": [10.0] * n,
            "pb": [1.0] * n,
            "roe": [0.1] * n,
            "valuation_z": [float(i - 10) for i in range(n)],
            "verdict": ["x"] * n,
        }
    )


@pytest.mark.parametrize("q", ["underpriced", "expensive", "valuation"])
def test_valuation_lists_at_most_fifteen_rows(env, q):
    rep(env, "valuation_clusters.csv", df=big_valuation_df())
    table = LR.valuation(q).split("\n\n")[1].split("\n")
    assert len(table) == 1 + 15  # header + 15 rows


def test_answer_structured_context_uses_two_chunks(env, monkeypatch):
    ks = []
    monkeypatch.setattr(LR, "retrieve", lambda q, k: ks.append(k) or [])
    rep(env, "income_statement_by_geo.csv", df=pd.DataFrame({"g": ["IN"]}))
    LR.answer("income statement")
    assert ks == [2]
