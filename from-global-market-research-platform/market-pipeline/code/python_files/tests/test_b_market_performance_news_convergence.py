"""Offline tests for market_performance (index metrics) and news_convergence (headline matching)."""

import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

import market_performance as mp
import news_convergence as nc


# ================================================================ market_performance
def _series(values, start="2021-01-01", freq="B"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float
    )


def test_metrics_requires_sixty_clean_bars():
    assert mp._metrics(_series([1.0] * 59)) == {}
    assert mp._metrics(_series([1.0, np.nan] * 40)) == {}  # 40 clean bars after dropna
    assert mp._metrics(_series(np.linspace(100, 110, 60))) != {}


def test_metrics_known_values_on_geometric_series():
    # daily +0.1% for 500 business days
    vals = 100 * 1.001 ** np.arange(500)
    s = _series(vals)
    m = mp._metrics(s)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    assert m["Years"] == round(yrs, 1)
    assert m["Total_5y%"] == round((vals[-1] / vals[0] - 1) * 100, 1)
    assert m["CAGR%"] == round(((vals[-1] / vals[0]) ** (1 / yrs) - 1) * 100, 1)
    assert m["MaxDD%"] == 0.0
    assert m["Vol_ann%"] == pytest.approx(0.0, abs=0.05)


def test_metrics_vol_sharpe_and_drawdown():
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0005, 0.01, 400)
    s = _series(100 * np.cumprod(1 + rets))
    m = mp._metrics(s)
    dr = s.pct_change().dropna()
    assert m["Vol_ann%"] == round(dr.std() * np.sqrt(252) * 100, 1)
    assert m["Sharpe"] == round((dr.mean() * 252) / (dr.std() * np.sqrt(252)), 2)
    dd = ((s - s.cummax()) / s.cummax()).min() * 100
    assert m["MaxDD%"] == round(dd, 1) and m["MaxDD%"] < 0


def test_metrics_one_year_and_ytd_windows():
    idx = pd.date_range("2023-01-02", "2025-06-30", freq="B")
    vals = np.linspace(100, 200, len(idx))
    s = pd.Series(vals, index=idx)
    m = mp._metrics(s)
    last = s.index[-1]
    s1 = s[s.index >= last - pd.Timedelta(days=365)]
    assert m["Return_1y%"] == round((s1.iloc[-1] / s1.iloc[0] - 1) * 100, 1)
    ytd = s[s.index >= pd.Timestamp(last.year, 1, 1)]
    assert m["YTD%"] == round((ytd.iloc[-1] / ytd.iloc[0] - 1) * 100, 1)
    assert 0 < m["YTD%"] < m["Return_1y%"] < m["Total_5y%"]


def test_metrics_ytd_none_when_only_one_bar_this_year():
    idx = list(pd.date_range("2024-01-02", periods=100, freq="B")) + [pd.Timestamp("2025-01-02")]
    s = pd.Series(np.linspace(100, 120, 101), index=pd.DatetimeIndex(idx))
    m = mp._metrics(s)
    assert m["YTD%"] is None and m["Return_1y%"] is not None


def test_metrics_constant_series_has_no_sharpe():
    m = mp._metrics(_series([50.0] * 80))
    assert m["Sharpe"] is None and m["Vol_ann%"] == 0.0 and m["MaxDD%"] == 0.0


def test_metrics_negative_return_and_no_negative_cagr_blowup():
    s = _series(np.linspace(100, 50, 300))
    m = mp._metrics(s)
    assert m["Total_5y%"] == -50.0 and m["CAGR%"] < 0 and m["MaxDD%"] < 0 and m["Sharpe"] < 0


def test_metrics_negative_first_price_has_no_cagr():
    s = _series(np.linspace(-5.0, 10.0, 80))
    assert mp._metrics(s).get("CAGR%") is None


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="source bug: `tot = (last / first - 1)` runs before the `first > 0` guard that "
    "protects only CAGR, so a zero first price raises instead of being handled",
)
def test_metrics_zero_first_price_is_handled():
    s = _series(np.linspace(0.0, 10.0, 80))
    assert isinstance(mp._metrics(s), dict)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: `round(x, 1) if x else None` treats a legitimate 0.0 (flat 5y CAGR, "
    "zero 1y return, zero Sharpe) as missing and reports None instead of 0.0",
)
def test_metrics_flat_index_reports_zero_cagr_not_none():
    vals = np.r_[np.linspace(100, 120, 100), np.linspace(120, 100, 100)]
    m = mp._metrics(_series(vals))  # ends exactly where it started
    assert m["Total_5y%"] == 0.0
    assert m["CAGR%"] == 0.0


def test_metrics_does_not_mutate_input():
    s = _series(np.linspace(100, 120, 100))
    s.iloc[3] = np.nan
    before = s.copy()
    mp._metrics(s)
    pd.testing.assert_series_equal(s, before)


# ---------------------------------------------------------------- analyse / load
def _fetch(monkeypatch, data, calls=None):
    def fetch(tickers, **kw):
        if calls is not None:
            calls.append((tickers, kw))
        return data

    monkeypatch.setattr(mp, "multi_fetch", fetch)


@pytest.fixture
def out(monkeypatch, tmp_path):
    p = tmp_path / "cache_seed" / "market_performance_5y.parquet"
    p.parent.mkdir()
    monkeypatch.setattr(mp, "OUT_PARQUET", p)
    return p


def _df(values):
    return pd.DataFrame({"Close": _series(values)})


def test_analyse_ranks_by_cagr_and_persists(monkeypatch, out, capsys):
    calls = []
    _fetch(
        monkeypatch,
        {
            "^GSPC": _df(np.linspace(100, 200, 400)),  # US: strong
            "^NSEI": _df(np.linspace(100, 120, 400)),  # IN: weaker
            "^N225": _df(np.linspace(100, 50, 400)),  # JP: negative
            "^HSI": pd.DataFrame(),  # empty -> "no data"
            "^FTSE": _df(np.linspace(100, 101, 30)),  # too short -> skipped silently
        },
        calls,
    )
    res = mp.analyse()
    assert res["Market"].tolist() == ["US", "IN", "JP"]
    assert res["Index"].iloc[0] == "S&P 500" and res["Ticker"].iloc[0] == "^GSPC"
    tickers, kw = calls[0]
    assert tickers == [v[0] for v in mp.INDEX.values()]
    assert kw["order"] == ("yahoo", "stooq") and kw["period"] == "5y" and kw["min_bars"] == 60
    assert out.exists() and out.with_suffix(".xlsx").exists()
    assert pd.read_parquet(out)["Market"].tolist() == ["US", "IN", "JP"]
    assert (res["as_of"] == pd.Timestamp.today().normalize()).all()
    text = capsys.readouterr().out
    assert "HK (^HSI): no data" in text and "NOT investment advice" in text


def test_analyse_quiet_and_empty(monkeypatch, out, capsys):
    _fetch(monkeypatch, {})
    res = mp.analyse(verbose=False)
    assert res.empty and not out.exists()
    assert capsys.readouterr().out == ""


def test_load_roundtrip_and_missing(monkeypatch, out):
    assert mp.load().empty
    _fetch(monkeypatch, {"^GSPC": _df(np.linspace(100, 200, 400))})
    mp.analyse(verbose=False)
    assert mp.load()["Market"].tolist() == ["US"]


def test_index_table_has_twenty_unique_markets():
    assert len(mp.INDEX) == 20 and len({v[0] for v in mp.INDEX.values()}) == 20


# ================================================================ news_convergence
@pytest.mark.parametrize(
    "title",
    [
        "Market wrap: Top gainers and losers on Nifty and Sensex today",
        "Stocks to watch: Reliance, TCS",
        "Closing Bell: Sensex ends higher",
        "F&O ban list",
        "NIFTY closes at record",
        "Trade setup for Monday",
        "MARKET LIVE updates",
        "Buzzing stocks today",
    ],
)
def test_generic_headlines_are_matched(title):
    assert nc.GENERIC.search(title)


@pytest.mark.parametrize(
    "title", ["Reliance posts record profit", "Infosys wins deal", "Sensible plan"]
)
def test_company_headlines_are_not_generic(title):
    assert not nc.GENERIC.search(title)


@pytest.mark.parametrize(
    "title",
    [
        "SEBI probe into Acme",
        "Auditor resigns at Acme",
        "CFO resigns",
        "Acme faces fraud allegations",
        "Rating downgraded",
        "Acme downgrades guidance",
        "Insolvency proceedings begin",
        "NCLT admits plea",
        "Promoter shares pledged",
        "Acme issues profit warning",
        "Acme writeoff of 200 crore",
        "Acme fined by SEBI",
        "Qualified opinion from auditor",
        "guidance cut announced",
        "Acme to delist",
        "Acme defaults on loan",
        "Lawsuit filed against Acme",
        "Raid at Acme offices",
    ],
)
def test_red_flag_regex_matches(title):
    assert nc.RED_FLAG.search(title)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug: the pattern `writ[e-]?off` accepts 'writeoff' and 'writ-off' but not the "
    "standard spelling 'write-off' (it needs both 'e' and '-'), so the commonest red-flag word "
    "is not detected",
)
def test_red_flag_regex_matches_write_off():
    assert nc.RED_FLAG.search("Acme announces write-off of Rs 200 crore")


@pytest.mark.parametrize("title", ["Acme profit rises", "Acme wins order", "Finest quality award"])
def test_red_flag_regex_ignores_benign(title):
    assert not nc.RED_FLAG.search(title)


def test_clean_name_uses_symbol_master_with_fallbacks(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "symbol_master",
        types.SimpleNamespace(
            clean_name_for=lambda s: " ADANI ENTERPRISES " if s == "ADANIENT" else ""
        ),
    )
    assert nc._clean_name("ADANIENT") == "ADANI ENTERPRISES"
    assert nc._clean_name("XYZ") == "XYZ"  # empty name -> symbol
    monkeypatch.setitem(sys.modules, "symbol_master", None)
    assert nc._clean_name("ABC") == "ABC"


def test_tokens_drop_stopwords_short_words_and_sort_longest_first():
    toks = nc._tokens("TCS", "Tata Consultancy Services Ltd India")
    # 'Services', 'Ltd', 'India' dropped; only words longer than 3 chars; symbol always kept
    assert toks == ["Consultancy", "Tata", "TCS"]
    assert nc._tokens("ITC", "ITC") == ["ITC"]  # name token too short, symbol kept once
    assert set(nc._tokens("RELIANCE", "Reliance Industries Limited")) == {"Reliance", "RELIANCE"}
    toks = nc._tokens("X", "Abcd Efgh")
    assert toks[-1] == "X" and set(toks) == {"Abcd", "Efgh", "X"}


@pytest.fixture
def names(monkeypatch):
    monkeypatch.setattr(
        nc,
        "_clean_name",
        lambda s: {"ADANIENT": "ADANI ENTERPRISES", "TCS": "TATA CONSULTANCY"}.get(s, s),
    )


def _e(title, link="l", pub="p"):
    return {"title": title, "link": link, "published": pub}


def test_articles_require_title_match_and_skip_generic_blank_and_dupes(names):
    entries = [
        _e("Adani Enterprises wins airport bid", "l1", "d1"),
        _e("adani enterprises wins airport bid", "l2"),  # duplicate (case-insens.)
        _e("Market wrap: Adani Enterprises and Sensex today"),  # generic -> excluded
        _e("Tata Consultancy hires 5,000"),  # other company
        _e(""),
        _e(None),
        {"link": "x"},
        _e("ADANIENT hits upper circuit"),  # ticker symbol match
    ]
    arts = nc._articles_for("ADANIENT", entries)
    assert [a["title"] for a in arts] == [
        "Adani Enterprises wins airport bid",
        "ADANIENT hits upper circuit",
    ]
    assert arts[0] == {
        "title": "Adani Enterprises wins airport bid",
        "link": "l1",
        "published": "d1",
    }


def test_articles_use_word_boundaries(names):
    # "Adaniport" must not match token "Adani"; token "ADANIENT" must not match inside a longer word
    entries = [_e("Adaniport expands"), _e("SuperADANIENT news"), _e("Adani Enterprises news")]
    assert [a["title"] for a in nc._articles_for("ADANIENT", entries)] == ["Adani Enterprises news"]


def test_articles_title_only_not_summary(names):
    entries = [{"title": "Quarterly results season", "summary": "Adani Enterprises beat estimates"}]
    assert nc._articles_for("ADANIENT", entries) == []


def test_articles_regex_metacharacters_in_symbol_are_escaped(names):
    assert nc._articles_for("M&M", [_e("M&M launches SUV"), _e("MXM launches")]) == [
        {"title": "M&M launches SUV", "link": "l", "published": "p"}
    ]
    assert nc._articles_for("A+B", [_e("A+B merges")]) != []


def _scorer(monkeypatch, mapping, default=0.0):
    monkeypatch.setitem(
        sys.modules,
        "sentiment_pipeline",
        types.SimpleNamespace(score_text=lambda t: mapping.get(t, default)),
    )


def _arts(*titles):
    return [{"title": t} for t in titles]


def test_classify_no_news_and_no_scorer(monkeypatch):
    v, score, n, why = nc.classify([])
    assert (v, n, why) == ("NO_NEWS", 0, "no company-specific coverage") and np.isnan(score)
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", None)
    v, score, n, why = nc.classify(_arts("a", "b"))
    assert (v, n, why) == ("NO_SCORER", 2, "sentiment_pipeline unavailable") and np.isnan(score)


def test_classify_thresholds_are_inclusive(monkeypatch):
    _scorer(monkeypatch, {"neg": -0.15, "pos": 0.15, "ok": 0.1499, "bad": -0.1499})
    assert nc.classify(_arts("neg"))[0] == "DIVERGENT"
    assert nc.classify(_arts("pos"))[0] == "CONVERGENT"
    assert nc.classify(_arts("ok"))[0] == "NEUTRAL"
    assert nc.classify(_arts("bad"))[0] == "NEUTRAL"
    v, s, n, why = nc.classify(_arts("pos", "neg"))
    assert (v, s, n, why) == ("NEUTRAL", 0.0, 2, "routine coverage")


def test_classify_red_flag_outranks_positive_average(monkeypatch):
    _scorer(
        monkeypatch,
        {
            "Acme record profit": 0.9,
            "Acme wins big": 0.8,
            "Auditor resigns at Acme, a very long headline that should be truncated "
            + "x" * 50: 0.0,
        },
    )
    long = "Auditor resigns at Acme, a very long headline that should be truncated " + "x" * 50
    v, avg, n, why = nc.classify(_arts("Acme record profit", "Acme wins big", long))
    assert v == "DIVERGENT" and n == 3 and avg == pytest.approx((0.9 + 0.8) / 3)
    assert why == "material: " + long[:70]


def test_classify_reasons(monkeypatch):
    _scorer(monkeypatch, {"bad news": -0.5, "good news": 0.5})
    assert nc.classify(_arts("bad news"))[3] == "negative coverage"
    assert nc.classify(_arts("good news"))[3] == "positive coverage"


# ---------------------------------------------------------------- main
@pytest.fixture
def ledger(monkeypatch, tmp_path):
    p = tmp_path / "signal_ledger.parquet"
    monkeypatch.setattr(nc, "LEDGER", p)
    return p


def _write_ledger(path, rows):
    pd.DataFrame(rows, columns=["symbol", "market", "signal_date", "filter", "score"]).to_parquet(
        path
    )


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["news_convergence.py", *args])
    return nc.main()


def _provider(monkeypatch, entries):
    class P:
        def _all_entries(self):
            if isinstance(entries, Exception):
                raise entries
            return entries

    mod = types.SimpleNamespace(
        IndianRSSProvider=P, score_text=lambda t: -0.5 if "weak" in t else 0.0
    )
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", mod)


TODAY = pd.Timestamp(date.today())


def test_main_missing_ledger(monkeypatch, ledger, capsys):
    assert _run(monkeypatch) == 1
    assert "no signal ledger" in capsys.readouterr().out


def test_main_no_passes_for_day_or_market(monkeypatch, ledger, capsys):
    _write_ledger(ledger, [("AAA", "IN", TODAY - pd.Timedelta(days=1), "darvas", 5.0)])
    assert _run(monkeypatch) == 0
    assert "no IN filter passes" in capsys.readouterr().out
    _write_ledger(ledger, [("AAA", "US", TODAY, "darvas", 5.0)])
    assert _run(monkeypatch) == 0 and "no IN filter passes" in capsys.readouterr().out
    assert (
        _run(monkeypatch, "--market", "us", "--date", str(TODAY.date())) == 1
    )  # RSS stub not installed


def test_main_rss_failure_returns_one(monkeypatch, ledger, capsys):
    _write_ledger(ledger, [("AAA", "IN", TODAY, "darvas", 5.0)])
    _provider(monkeypatch, RuntimeError("feed down"))
    assert _run(monkeypatch) == 1
    assert "RSS unavailable (RuntimeError: feed down)" in capsys.readouterr().out


def test_main_reports_verdicts_orders_divergent_first_and_writes_csv(
    monkeypatch, ledger, tmp_path, capsys
):
    _write_ledger(
        ledger,
        [
            ("QUIET", "IN", TODAY, "darvas", 9.0),
            ("BADCO", "IN", TODAY, "piotroski", 6.0),
            ("GOODCO", "IN", TODAY, "darvas", 7.0),
            ("BADCO", "IN", TODAY, "darvas", 1.0),
            ("OTHERMKT", "US", TODAY, "darvas", 9.5),
        ],
    )
    monkeypatch.setattr(nc, "_clean_name", lambda s: s)
    _provider(monkeypatch, [_e("BADCO weak quarter, SEBI probe"), _e("GOODCO inks deal")])
    out = tmp_path / "sub" / "conv.csv"
    assert _run(monkeypatch, "--out", str(out)) == 0
    d = pd.read_csv(out)
    assert d["symbol"].tolist() == ["BADCO", "GOODCO", "QUIET"]  # DIVERGENT, NEUTRAL, NO_NEWS
    assert d.set_index("symbol").loc["BADCO", "verdict"] == "DIVERGENT"
    assert d.set_index("symbol").loc["QUIET", "verdict"] == "NO_NEWS"
    assert d.set_index("symbol").loc["GOODCO", "verdict"] == "NEUTRAL"
    assert len(d) == 3  # BADCO deduped, other market excluded
    assert d.set_index("symbol").loc["BADCO", "filter"] == "piotroski"  # highest score row kept
    text = capsys.readouterr().out
    assert "3 IN names passed a filter" in text and "2 headlines" in text
    assert "1 DIVERGENT" in text and "NO_NEWS" in text and "CONVERGENT does NOT mean" in text
    assert "1 of 3 have NO company-specific coverage" in text


def test_main_top_cap_and_hash_tiebreak_is_deterministic(monkeypatch, ledger, tmp_path):
    syms = [f"S{i:02d}" for i in range(12)]
    _write_ledger(ledger, [(s, "IN", TODAY, "darvas", 7.0) for s in syms])  # all tied
    monkeypatch.setattr(nc, "_clean_name", lambda s: s)
    _provider(monkeypatch, [])
    o1, o2 = tmp_path / "a.csv", tmp_path / "b.csv"
    _run(monkeypatch, "--top", "5", "--out", str(o1))
    _run(monkeypatch, "--top", "5", "--out", str(o2))
    a, b = pd.read_csv(o1), pd.read_csv(o2)
    assert len(a) == 5 and a["symbol"].tolist() == b["symbol"].tolist()
    assert a["symbol"].tolist() != sorted(syms)[:5]  # not the alphabetical head


def test_main_top_keeps_highest_scores_first(monkeypatch, ledger, tmp_path):
    _write_ledger(
        ledger,
        [
            ("LOW", "IN", TODAY, "darvas", 1.0),
            ("HIGH", "IN", TODAY, "darvas", 9.0),
            ("MID", "IN", TODAY, "darvas", 5.0),
        ],
    )
    monkeypatch.setattr(nc, "_clean_name", lambda s: s)
    _provider(monkeypatch, [])
    o = tmp_path / "o.csv"
    _run(monkeypatch, "--top", "2", "--out", str(o))
    assert set(pd.read_csv(o)["symbol"]) == {"HIGH", "MID"}


def test_main_no_divergence_message(monkeypatch, ledger, capsys):
    _write_ledger(ledger, [("QUIET", "IN", TODAY, "darvas", 9.0)])
    monkeypatch.setattr(nc, "_clean_name", lambda s: s)
    _provider(monkeypatch, [])
    _run(monkeypatch)
    assert "No divergences." in capsys.readouterr().out
