"""Offline tests for sentiment_pipeline (providers, RSS matching, aggregation, cache)."""

import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta

import pytest

_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp(prefix="mp_sent_"))  # the module creates ./sentiment_results at import
try:
    import sentiment_pipeline as sp
finally:
    os.chdir(_cwd)


class _FakeVader:
    """Minimal stand-in: +1 if 'good' in text, -1 if 'bad', else 0 (compound score)."""

    def polarity_scores(self, text):
        t = text.lower()
        return {"compound": 0.8 if "good" in t else (-0.8 if "bad" in t else 0.0)}


@pytest.fixture(autouse=True)
def vader(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_VADER", _FakeVader())
    monkeypatch.setattr(sp, "_VADER_OK", True)
    monkeypatch.setattr(sp, "CACHE_FILE", tmp_path / "cache" / "sentiment_cache.json")
    monkeypatch.setattr(sp, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    (tmp_path / "cache").mkdir()


# ---------------------------------------------------------------- scoring
def test_score_text_requires_text_and_vader(monkeypatch):
    assert sp.score_text("good news") == 0.8
    assert sp.score_text("bad news") == -0.8
    assert sp.score_text("") == 0.0 and sp.score_text(None) == 0.0
    monkeypatch.setattr(sp, "_VADER_OK", False)
    assert sp.score_text("good news") == 0.0


@pytest.mark.parametrize(
    "score,label",
    [
        (0.15, "POSITIVE"),
        (0.1499, "NEUTRAL"),
        (0.0, "NEUTRAL"),
        (-0.1499, "NEUTRAL"),
        (-0.15, "NEGATIVE"),
        (1.0, "POSITIVE"),
        (-1.0, "NEGATIVE"),
    ],
)
def test_label_of_thresholds(score, label):
    assert sp.label_of(score) == label


def test_dataclass_defaults_are_independent():
    a, b = sp.TickerSentiment("A", 0.0, "NEUTRAL", 0), sp.TickerSentiment("B", 0.0, "NEUTRAL", 0)
    a.providers.append("x")
    assert b.providers == [] and a.fetched_at == ""
    art = sp.Article("t", "s", "p", 0.1)
    assert art.url == "" and art.summary == ""


# ---------------------------------------------------------------- base provider
class _Prov(sp.NewsProvider):
    name = "P"
    env_key = "MP_TEST_KEY"
    min_interval = 2.0

    def fetch_news(self, ticker, market="IN"):
        return []


def test_provider_availability_depends_on_key_and_requests(monkeypatch):
    monkeypatch.delenv("MP_TEST_KEY", raising=False)
    assert _Prov().available is False
    monkeypatch.setenv("MP_TEST_KEY", "k")
    assert _Prov().available is True and _Prov().api_key == "k"
    monkeypatch.setattr(sp, "_REQ_OK", False)
    assert _Prov().available is False


def test_throttle_sleeps_only_the_remaining_interval(monkeypatch):
    sleeps = []
    clock = iter([100.0, 100.5, 105.0, 105.0])
    monkeypatch.setattr(sp.time, "time", lambda: next(clock))
    monkeypatch.setattr(sp.time, "sleep", lambda s: sleeps.append(s))
    p = _Prov()
    p._last_call = 99.0
    p._throttle()  # elapsed 1.0 (<2.0): sleeps the remaining 1.0
    assert sleeps == [1.0]
    p._throttle()  # last_call=100.5 -> elapsed 4.5 >= 2.0: no sleep
    assert sleeps == [1.0] and p._last_call == 105.0


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        sp.NewsProvider()


# ---------------------------------------------------------------- API providers
class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _http(monkeypatch, payload=None, exc=None):
    calls = []

    def get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if exc:
            raise exc
        return _Resp(payload)

    monkeypatch.setattr(sp, "requests", types.SimpleNamespace(get=get), raising=False)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    return calls


def _keyed(monkeypatch, cls, key="K"):
    monkeypatch.setenv(cls.env_key, key)
    return cls()


def test_providers_without_key_do_nothing(monkeypatch):
    calls = _http(monkeypatch, {})
    for cls in (
        sp.MarketauxProvider,
        sp.AlphaVantageProvider,
        sp.FinnhubProvider,
        sp.NewsDataProvider,
    ):
        monkeypatch.delenv(cls.env_key, raising=False)
        assert cls().fetch_news("TCS") == []
    assert calls == []


def test_marketaux_native_sentiment_with_title_fallback_and_symbol_mapping(monkeypatch):
    payload = {
        "data": [
            {
                "title": "TCS good",
                "published_at": "d1",
                "url": "u1",
                "description": "s1",
                "entities": [
                    {"symbol": "INFY.NS", "sentiment_score": -0.9},
                    {"symbol": "TCS.NS", "sentiment_score": 0.42},
                ],
            },
            {
                "title": "TCS bad quarter",
                "published_at": "d2",
                "url": "u2",
                "description": "s2",
                "entities": [{"symbol": "TCS.NS", "sentiment_score": None}],
            },
            {"title": "no entities good"},
        ]
    }
    calls = _http(monkeypatch, payload)
    arts = _keyed(monkeypatch, sp.MarketauxProvider).fetch_news("TCS", "IN")
    assert [a.sentiment for a in arts] == [0.42, -0.8, 0.8]
    assert arts[0].source == "Marketaux" and arts[0].url == "u1" and arts[0].summary == "s1"
    assert calls[0][1]["symbols"] == "TCS.NS" and calls[0][1]["api_token"] == "K"
    _http(monkeypatch, payload)
    calls = _http(monkeypatch, {"data": []})
    _keyed(monkeypatch, sp.MarketauxProvider).fetch_news("AAPL", "US")
    assert calls[0][1]["symbols"] == "AAPL"  # no .NS suffix outside India


def test_marketaux_errors_return_empty(monkeypatch):
    _http(monkeypatch, exc=RuntimeError("x"))
    assert _keyed(monkeypatch, sp.MarketauxProvider).fetch_news("TCS") == []
    _http(
        monkeypatch,
        {"data": [{"title": "t", "entities": [{"symbol": "TCS.NS", "sentiment_score": "bad"}]}]},
    )
    assert (
        _keyed(monkeypatch, sp.MarketauxProvider).fetch_news("TCS") == []
    )  # float('bad') -> swallowed


def test_alphavantage_ticker_score_then_overall_fallback(monkeypatch):
    payload = {
        "feed": [
            {
                "title": "a",
                "time_published": "t1",
                "url": "u",
                "summary": "s",
                "ticker_sentiment": [
                    {"ticker": "OTHER", "ticker_sentiment_score": "-0.5"},
                    {"ticker": "TCS.BSE", "ticker_sentiment_score": "0.33"},
                ],
                "overall_sentiment_score": "0.9",
            },
            {"title": "b", "ticker_sentiment": [], "overall_sentiment_score": "-0.25"},
            {"title": "c", "overall_sentiment_score": None},
        ]
    }
    calls = _http(monkeypatch, payload)
    arts = _keyed(monkeypatch, sp.AlphaVantageProvider).fetch_news("TCS", "IN")
    assert [a.sentiment for a in arts] == [0.33, -0.25, 0.0]
    assert calls[0][1]["tickers"] == "TCS.BSE" and calls[0][1]["function"] == "NEWS_SENTIMENT"
    calls = _http(monkeypatch, {"feed": []})
    _keyed(monkeypatch, sp.AlphaVantageProvider).fetch_news("AAPL", "US")
    assert calls[0][1]["tickers"] == "AAPL"


def test_alphavantage_error_returns_empty(monkeypatch):
    _http(monkeypatch, exc=RuntimeError("x"))
    assert _keyed(monkeypatch, sp.AlphaVantageProvider).fetch_news("TCS") == []


def test_finnhub_scores_headline_plus_summary_and_caps_fifteen(monkeypatch):
    data = [
        {"headline": f"good {i}", "summary": "s", "datetime": 1700000000 + i, "url": f"u{i}"}
        for i in range(20)
    ]
    calls = _http(monkeypatch, data)
    arts = _keyed(monkeypatch, sp.FinnhubProvider).fetch_news("TCS", "IN")
    assert len(arts) == 15 and arts[0].sentiment == 0.8 and arts[0].published == "1700000000"
    p = calls[0][1]
    assert p["symbol"] == "TCS.NS" and p["token"] == "K"
    frm, to = datetime.strptime(p["from"], "%Y-%m-%d"), datetime.strptime(p["to"], "%Y-%m-%d")
    assert (to - frm).days == 7
    calls = _http(monkeypatch, [])
    _keyed(monkeypatch, sp.FinnhubProvider).fetch_news("AAPL", "US")
    assert calls[0][1]["symbol"] == "AAPL"


def test_finnhub_non_list_payload_and_errors(monkeypatch):
    _http(monkeypatch, {"error": "limit"})
    assert _keyed(monkeypatch, sp.FinnhubProvider).fetch_news("TCS") == []
    _http(monkeypatch, exc=RuntimeError("x"))
    assert _keyed(monkeypatch, sp.FinnhubProvider).fetch_news("TCS") == []


def test_newsdata_scores_and_country_mapping(monkeypatch):
    payload = {
        "results": [{"title": "good deal", "description": None, "pubDate": "p", "link": "l"}] * 20
    }
    calls = _http(monkeypatch, payload)
    arts = _keyed(monkeypatch, sp.NewsDataProvider).fetch_news("TCS", "IN")
    assert (
        len(arts) == 15
        and arts[0].sentiment == 0.8
        and arts[0].summary == ""
        and arts[0].url == "l"
    )
    assert calls[0][1]["country"] == "in" and calls[0][1]["q"] == "TCS"
    calls = _http(monkeypatch, {"results": []})
    _keyed(monkeypatch, sp.NewsDataProvider).fetch_news("AAPL", "US")
    assert calls[0][1]["country"] == "us"
    _http(monkeypatch, exc=RuntimeError("x"))
    assert _keyed(monkeypatch, sp.NewsDataProvider).fetch_news("TCS") == []


# ---------------------------------------------------------------- RSS providers
def _feedparser(monkeypatch, per_url=None, boom_urls=()):
    calls = []

    def parse(url):
        calls.append(url)
        if url in boom_urls:
            raise RuntimeError("bad feed")
        return types.SimpleNamespace(entries=(per_url or {}).get(url, []))

    monkeypatch.setitem(sys.modules, "feedparser", types.SimpleNamespace(parse=parse))
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    return calls


def _entry(title, summary="", link="l", published="p"):
    return {"title": title, "summary": summary, "link": link, "published": published}


def test_rss_availability_follows_feedparser(monkeypatch):
    _feedparser(monkeypatch)
    assert sp.IndianRSSProvider().available is True
    monkeypatch.setitem(sys.modules, "feedparser", None)
    assert sp.IndianRSSProvider().available is False


def test_rss_all_entries_fetches_once_tags_outlet_and_isolates_failures(monkeypatch):
    first = sp.IndianRSSProvider.FEEDS["Moneycontrol"][0]
    bad = sp.IndianRSSProvider.FEEDS["EconomicTimes"][0]
    calls = _feedparser(
        monkeypatch,
        {first: [{"title": "T", "description": "D", "link": "L", "published": "P"}]},
        boom_urls={bad},
    )
    p = sp.IndianRSSProvider()
    entries = p._all_entries()
    total_urls = sum(len(v) for v in p.FEEDS.values())
    assert len(calls) == total_urls
    assert entries == [
        {"title": "T", "summary": "D", "link": "L", "published": "P", "outlet": "Moneycontrol"}
    ]
    assert p._feed_cache[bad] == []
    p._all_entries()
    assert len(calls) == total_urls  # cached within a run


def test_rss_summary_prefers_summary_over_description(monkeypatch):
    first = sp.IndianRSSProvider.FEEDS["Moneycontrol"][0]
    _feedparser(monkeypatch, {first: [{"title": "T", "summary": "S", "description": "D"}]})
    assert sp.IndianRSSProvider()._all_entries()[0]["summary"] == "S"


def _rss_with(monkeypatch, entries, outlet_url=None):
    url = outlet_url or sp.IndianRSSProvider.FEEDS["Moneycontrol"][0]
    _feedparser(monkeypatch, {url: entries})
    return sp.IndianRSSProvider()


def test_rss_company_name_phrase_match_is_precise(monkeypatch):
    p = _rss_with(
        monkeypatch,
        [
            _entry("Adani Enterprises Q4 profit good"),
            _entry("Adani Ports bad quarter"),
            _entry("Adani   Enterprises wins bid", summary="good"),
            _entry("Unrelated"),
        ],
    )
    arts = p.fetch_news("ADANIENT", "IN", company_name="ADANI ENTERPRISES")
    assert [a.title for a in arts] == [
        "Adani Enterprises Q4 profit good",
        "Adani   Enterprises wins bid",
    ]
    assert arts[0].source == "IndianRSS:Moneycontrol" and arts[0].sentiment == 0.8


def test_rss_single_token_name_and_short_tokens(monkeypatch):
    p = _rss_with(monkeypatch, [_entry("Infosys good results"), _entry("Info sys other")])
    assert len(p.fetch_news("INFY", company_name="INFOSYS")) == 1
    # tokens shorter than 3 characters are ignored; nothing left -> falls back to ticker rules
    assert p.fetch_news("AB", company_name="AB CD") == []


def test_rss_name_lookup_from_symbol_master_and_ticker_fallback(monkeypatch):
    p = _rss_with(monkeypatch, [_entry("RELIANCE jumps"), _entry("Reliance Retail news")])
    monkeypatch.setitem(
        sys.modules,
        "symbol_master",
        types.SimpleNamespace(clean_name_for=lambda t: "RELIANCE RETAIL"),
    )
    assert [a.title for a in p.fetch_news("RELIANCE")] == ["Reliance Retail news"]
    monkeypatch.setitem(
        sys.modules, "symbol_master", types.SimpleNamespace(clean_name_for=lambda t: "")
    )
    both = ["RELIANCE jumps", "Reliance Retail news"]  # ticker word match is case-insensitive
    assert [a.title for a in p.fetch_news("RELIANCE")] == both
    monkeypatch.setitem(sys.modules, "symbol_master", None)
    assert [a.title for a in p.fetch_news("RELIANCE")] == both


def test_rss_three_letter_name_tokens_are_part_of_the_phrase(monkeypatch):
    p = _rss_with(monkeypatch, [_entry("ABB India posts profit"), _entry("India GDP grows")])
    assert [a.title for a in p.fetch_news("ABB", company_name="ABB INDIA")] == [
        "ABB India posts profit"
    ]


def test_rss_ticker_fallback_skips_short_and_common_word_tickers(monkeypatch):
    p = _rss_with(
        monkeypatch,
        [_entry("TCS wins"), _entry("The GOLD price"), _entry("ABCD up"), _entry("ABCDEF up")],
    )
    monkeypatch.setitem(
        sys.modules, "symbol_master", types.SimpleNamespace(clean_name_for=lambda t: "")
    )
    assert p.fetch_news("TCS") == []  # < 4 characters
    assert p.fetch_news("GOLD") == []  # common English word
    assert len(p.fetch_news("ABCD")) == 1  # exactly 4 chars allowed, word-boundary only
    assert len(p.fetch_news(" abcd ")) == 1  # stripped and upper-cased


def test_rss_fetch_news_only_for_india_and_with_feedparser(monkeypatch):
    p = _rss_with(monkeypatch, [_entry("Reliance good")])
    assert p.fetch_news("RELIANCE", "US", company_name="RELIANCE") == []
    monkeypatch.setitem(sys.modules, "feedparser", None)
    assert sp.IndianRSSProvider().fetch_news("RELIANCE", company_name="RELIANCE") == []


def test_rss_caps_fifteen_and_truncates_summary(monkeypatch):
    entries = [_entry(f"Reliance item {i}", summary="s" * 500) for i in range(20)]
    arts = _rss_with(monkeypatch, entries).fetch_news("RELIANCE", company_name="RELIANCE")
    assert len(arts) == 15 and len(arts[0].summary) == 200


def test_rss_market_mood(monkeypatch):
    assert sp.IndianRSSProvider.fetch_market_mood is not None
    p = _rss_with(
        monkeypatch, [_entry("good a"), _entry("good b"), _entry("bad c"), _entry("flat d")]
    )
    mood = p.fetch_market_mood()
    assert mood["mood"] == "POSITIVE" and mood["score"] == round((0.8 + 0.8 - 0.8) / 3, 3)
    assert mood["n_articles"] == 4
    assert mood["by_outlet"] == {o: len(u) for o, u in p.FEEDS.items()}


def test_rss_market_mood_edge_cases(monkeypatch):
    p = _rss_with(monkeypatch, [_entry("flat"), _entry("flat too")])
    assert p.fetch_market_mood() == {"mood": "NEUTRAL", "score": 0.0, "n": 2}
    assert _rss_with(monkeypatch, []).fetch_market_mood() == {}
    monkeypatch.setitem(sys.modules, "feedparser", None)
    assert sp.IndianRSSProvider().fetch_market_mood() == {}


def test_us_rss_provider_routes_us_only_and_tags_market(monkeypatch):
    url = sp.USRSSProvider.FEEDS["CNBC"][0]
    _feedparser(monkeypatch, {url: [_entry("Apple Inc good quarter")]})
    p = sp.USRSSProvider()
    assert p.name == "USRSS" and "CNBC" in p.FEEDS and "Moneycontrol" not in p.FEEDS
    arts = p.fetch_news("AAPL", "US", company_name="APPLE INC")
    assert [a.source for a in arts] == ["USRSS:CNBC"]
    assert p.fetch_news("AAPL", "IN", company_name="APPLE INC") == []
    assert p.fetch_market_mood()["market"] == "US"
    monkeypatch.setitem(sys.modules, "feedparser", None)
    assert sp.USRSSProvider().fetch_market_mood() == {}


# ---------------------------------------------------------------- pipeline
class _Fake(sp.NewsProvider):
    def __init__(self, name, arts, active=True):
        super().__init__()
        self.name, self._arts, self._active, self.calls = name, arts, active, []

    @property
    def available(self):
        return self._active

    def fetch_news(self, ticker, market="IN"):
        self.calls.append((ticker, market))
        return [sp.Article(a.title, a.source, a.published, a.sentiment) for a in self._arts]


def _art(title, s, src="x"):
    return sp.Article(title, src, "p", s)


def _pipeline(monkeypatch, providers):
    monkeypatch.setattr(
        sp, "MarketauxProvider", lambda: providers[0] if providers else _Fake("M", [], False)
    )
    for cls in (
        "AlphaVantageProvider",
        "FinnhubProvider",
        "NewsDataProvider",
        "IndianRSSProvider",
        "USRSSProvider",
    ):
        monkeypatch.setattr(sp, cls, lambda: _Fake("off", [], False))
    return sp.SentimentPipeline()


def test_pipeline_active_providers_and_status(monkeypatch):
    s = _pipeline(monkeypatch, [])
    assert s.active == [] and s.status().startswith("No providers active. Set API keys:")
    s = _pipeline(monkeypatch, [_Fake("Marketaux", [])])
    assert [p.name for p in s.active] == ["Marketaux"]
    assert s.status() == "Active providers: Marketaux"


def test_pipeline_weighted_mean_clamping_and_top_headlines(monkeypatch):
    p1 = _Fake("Marketaux", [_art("a", 0.5), _art("b", 5.0)])  # 5.0 is clamped to 1.0
    s = _pipeline(monkeypatch, [p1])
    ts = s.get_ticker_sentiment("TCS", "IN")
    assert ts.score == round((0.5 * 1.2 + 1.0 * 1.2) / (1.2 + 1.2), 4) == 0.75
    assert ts.label == "POSITIVE" and ts.n_articles == 2 and ts.providers == ["Marketaux"]
    assert [h["sentiment"] for h in ts.top_headlines] == [1.0, 0.5]  # by |sentiment|, desc
    assert p1.calls == [("TCS", "IN")]


def test_pipeline_top_headlines_capped_and_title_truncated(monkeypatch):
    arts = [_art("x" * 200, 0.1 * i) for i in range(1, 9)]
    ts = _pipeline(monkeypatch, [_Fake("Marketaux", arts)]).get_ticker_sentiment("T")
    assert len(ts.top_headlines) == 5 and len(ts.top_headlines[0]["title"]) == 90
    assert ts.top_headlines[0]["sentiment"] == 0.8


def test_pipeline_no_data_not_cached(monkeypatch):
    s = _pipeline(monkeypatch, [_Fake("Marketaux", [])])
    ts = s.get_ticker_sentiment("TCS")
    assert (ts.label, ts.score, ts.n_articles, ts.providers) == ("NO_DATA", 0.0, 0, [])
    assert s._cache == {} and not sp.CACHE_FILE.exists()


def test_pipeline_unknown_provider_name_weighs_one(monkeypatch):
    p = _Fake("Mystery", [_art("a", 0.4), _art("b", 0.2)])
    s = _pipeline(monkeypatch, [p])
    assert s.get_ticker_sentiment("T").score == 0.3


def test_pipeline_caches_for_six_hours_per_market_and_ticker(monkeypatch):
    p = _Fake("Marketaux", [_art("a", 0.5)])
    s = _pipeline(monkeypatch, [p])
    first = s.get_ticker_sentiment("TCS", "IN")
    again = s.get_ticker_sentiment("TCS", "IN")
    assert (
        len(p.calls) == 1
        and again.score == first.score
        and again.top_headlines == first.top_headlines
    )
    s.get_ticker_sentiment("TCS", "US")  # different market -> different key
    assert len(p.calls) == 2
    assert set(json.loads(sp.CACHE_FILE.read_text())) == {"IN:TCS", "US:TCS"}


@pytest.mark.parametrize(
    "name,weight",
    [
        ("Marketaux", 1.2),
        ("AlphaVantage", 1.2),
        ("Finnhub", 1.0),
        ("NewsData", 0.8),
        ("IndianRSS", 1.1),
        ("USRSS", 1.1),
        ("Mystery", 1.0),
    ],
)
def test_pipeline_provider_weights(monkeypatch, name, weight):
    assert sp.SentimentPipeline.PROVIDER_WEIGHT.get(name, 1.0) == weight
    s = _pipeline(monkeypatch, [_Fake(name, [_art("a", 1.0)])])
    s.active.append(_Fake("Mystery", [_art("b", 0.0)]))  # weight 1.0 baseline
    assert s.get_ticker_sentiment("T").score == round(weight / (weight + 1.0), 4)


def test_pipeline_cache_entry_exactly_at_ttl_is_stale(monkeypatch):
    fixed = datetime(2026, 7, 1, 12, 0, 0)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(sp, "datetime", FixedDT)
    s = _pipeline(monkeypatch, [])
    s._cache = {
        "k": {"fetched_at": (fixed - timedelta(hours=6)).isoformat()},
        "j": {"fetched_at": (fixed - timedelta(hours=6) + timedelta(seconds=1)).isoformat()},
    }
    assert s._is_fresh("k") is False and s._is_fresh("j") is True


def test_pipeline_cache_expiry_boundary_and_bad_timestamps(monkeypatch):
    s = _pipeline(monkeypatch, [_Fake("Marketaux", [])])
    now = datetime.now()
    s._cache = {
        "fresh": {"fetched_at": (now - timedelta(hours=5, minutes=59)).isoformat()},
        "stale": {"fetched_at": (now - timedelta(hours=6, minutes=1)).isoformat()},
        "bad": {"fetched_at": "not a date"},
        "missing": {},
    }
    assert s._is_fresh("fresh") is True
    assert s._is_fresh("stale") is False and s._is_fresh("bad") is False
    assert s._is_fresh("missing") is False and s._is_fresh("absent") is False
    assert sp.CACHE_TTL_HOURS == 6


def test_pipeline_loads_existing_cache_and_tolerates_corruption(monkeypatch):
    sp.CACHE_FILE.write_text(
        json.dumps(
            {
                "IN:X": {
                    "fetched_at": datetime.now().isoformat(),
                    "score": 0.5,
                    "label": "POSITIVE",
                    "n_articles": 3,
                    "providers": ["P"],
                    "top_headlines": [],
                }
            }
        )
    )
    s = _pipeline(monkeypatch, [])
    ts = s.get_ticker_sentiment("X", "IN")
    assert (ts.score, ts.label, ts.n_articles, ts.providers) == (0.5, "POSITIVE", 3, ["P"])
    sp.CACHE_FILE.write_text("{bad")
    assert _pipeline(monkeypatch, [])._cache == {}


def test_pipeline_get_batch_prints_progress_every_ten(monkeypatch, capsys):
    p = _Fake("Marketaux", [_art("a", 0.5)])
    s = _pipeline(monkeypatch, [p])
    out = s.get_batch([f"T{i}" for i in range(12)], "IN")
    assert list(out) == [f"T{i}" for i in range(12)]
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    assert lines == ["10/12 | POSITIVE:10 NEGATIVE:0", "12/12 | POSITIVE:12 NEGATIVE:0"]


def test_pipeline_market_mood_routes_by_market(monkeypatch):
    s = _pipeline(monkeypatch, [])
    assert s.get_market_mood("IN") == {} and s.get_market_mood("US") == {}
    s._rss_in = types.SimpleNamespace(fetch_market_mood=lambda: {"mood": "IN"})
    s._rss_us = types.SimpleNamespace(fetch_market_mood=lambda: {"mood": "US"})
    assert s.get_market_mood()["mood"] == "IN" and s.get_market_mood("US")["mood"] == "US"
    assert s.get_market_mood("JP")["mood"] == "IN"  # anything but US uses the India gauge


def test_pipeline_rss_handles_require_exact_provider_classes(monkeypatch):
    monkeypatch.setattr(sp, "MarketauxProvider", lambda: _Fake("M", [], False))
    for cls in ("AlphaVantageProvider", "FinnhubProvider", "NewsDataProvider"):
        monkeypatch.setattr(sp, cls, lambda: _Fake("off", [], False))
    monkeypatch.setitem(
        sys.modules,
        "feedparser",
        types.SimpleNamespace(parse=lambda u: types.SimpleNamespace(entries=[])),
    )
    s = sp.SentimentPipeline()
    assert type(s._rss_in).__name__ == "IndianRSSProvider"
    assert (
        type(s._rss_us).__name__ == "USRSSProvider"
    )  # subclass is not mistaken for the India provider


# ---------------------------------------------------------------- CLI
def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["sentiment_pipeline.py", *args])
    return sp.main()


def test_main_demo_mode_without_providers(monkeypatch, capsys):
    def _pipeline_patch():
        return types.SimpleNamespace(active=[], status=lambda: "No providers active.")  # noqa: E731

    monkeypatch.setattr(sp, "SentimentPipeline", _pipeline_patch)
    _main(monkeypatch, "--tickers", "TCS")
    out = capsys.readouterr().out
    assert "VADER scoring: available" in out and "Demo mode" in out
    assert "RELIANCE" in out and "YESBANK" in out and "TCS" in out and "NEUTRAL" in out
    monkeypatch.setattr(sp, "_VADER_OK", False)
    _main(monkeypatch, "--tickers", "TCS")
    assert "MISSING (pip install vaderSentiment)" in capsys.readouterr().out


def test_main_live_mode_prints_table_and_writes_json(monkeypatch, capsys):
    ts = sp.TickerSentiment(
        "TCS",
        0.42,
        "POSITIVE",
        3,
        ["Marketaux"],
        [{"title": "good deal", "sentiment": 0.8, "source": "x"}],
        "now",
    )
    pipe = types.SimpleNamespace(
        active=[1], status=lambda: "Active providers: Marketaux", get_batch=lambda t, m: {"TCS": ts}
    )
    monkeypatch.setattr(sp, "SentimentPipeline", lambda: pipe)
    _main(monkeypatch, "--tickers", "TCS", "--market", "US")
    out = capsys.readouterr().out
    assert "TCS" in out and "+0.420" in out and "[+0.80 x] good deal" in out
    files = list(sp.OUT_DIR.glob("sentiment_US_*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["TCS"]["score"] == 0.42


def test_main_rejects_unknown_market(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--tickers", "TCS", "--market", "JP")
