"""Offline tests for tool_registry (decorators, discovery, safe execution)."""

import sys
import types

import pytest

import tool_registry as tr


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(tr, "_SCREENERS", {})
    monkeypatch.setattr(tr, "_NEWS_SOURCES", {})
    monkeypatch.setattr(tr, "_ANALYSES", {})
    recorded = []
    fake = types.SimpleNamespace(record_analysis_run=lambda *a, **k: recorded.append((a, k)))
    monkeypatch.setitem(sys.modules, "system_state", fake)
    return recorded


# ---------------------------------------------------------------- screeners
def test_screener_registers_and_returns_function_unchanged():
    @tr.screener("hi", "High price", tier_weight=3)
    def hi(c):
        return c > 10

    assert hi(11) is True
    s = tr._SCREENERS["hi"]
    assert (s.key, s.description, s.tier_weight, s.fn) == ("hi", "High price", 3, hi)
    assert tr.list_screeners() == ["hi"]


def test_screener_description_falls_back_to_doc_then_key():
    @tr.screener("a")
    def a(c):
        """Doc text"""

    @tr.screener("b")
    def b(c):
        pass

    assert tr._SCREENERS["a"].description == "Doc text"
    assert tr._SCREENERS["b"].description == "b"
    assert tr._SCREENERS["b"].tier_weight == 1


def test_screener_reregistration_overwrites():
    tr.screener("x", "one")(lambda c: True)
    tr.screener("x", "two")(lambda c: False)
    assert tr.list_screeners() == ["x"] and tr._SCREENERS["x"].description == "two"


def test_run_screener_truthiness_and_unknown():
    tr.screener("t")(lambda c: c)
    assert tr.run_screener("t", 5) is True
    assert tr.run_screener("t", 0) is False
    assert tr.run_screener("t", []) is False
    assert tr.run_screener("missing", 1) is False


def test_run_screener_does_not_swallow_exceptions():
    tr.screener("boom")(lambda c: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        tr.run_screener("boom", 1)


def test_run_all_screeners_keeps_registration_order_and_isolates_failures():
    tr.screener("pos")(lambda c: c > 0)
    tr.screener("boom")(lambda c: 1 / 0)
    tr.screener("even")(lambda c: c % 2 == 0)
    tr.screener("neg")(lambda c: c < 0)
    assert tr.run_all_screeners(4) == ["pos", "even"]
    assert tr.run_all_screeners(-3) == ["neg"]
    assert tr.run_all_screeners(None) == []  # every fn raises/false


# ---------------------------------------------------------------- news sources
def test_news_source_feeds_decorator_has_no_custom_fn():
    @tr.news_source("reuters", feeds=["http://a", "http://b"], market="US")
    def _():
        pass

    n = tr.get_news_source("reuters")
    assert n.feeds == ["http://a", "http://b"] and n.market == "US" and n.fn is None
    assert tr.list_news_sources() == ["reuters"]


def test_news_source_custom_fetcher_kept_even_with_feeds():
    @tr.news_source("c", feeds=["http://a"])
    def fetch(ticker, market):
        return []

    assert tr.get_news_source("c").fn is fetch


def test_news_source_defaults():
    @tr.news_source("d")
    def _():
        pass

    n = tr.get_news_source("d")
    assert n.feeds == [] and n.market == "IN"
    assert n.fn is not None  # no feeds -> the `_` function is treated as custom
    assert tr.get_news_source("nope") is None


def test_news_source_feed_lists_not_shared_between_sources():
    tr.news_source("a")(lambda: None)
    tr.news_source("b")(lambda: None)
    tr.get_news_source("a").feeds.append("x")
    assert tr.get_news_source("b").feeds == []


# ---------------------------------------------------------------- analyses
def test_analysis_registration_and_run_records_ok(fresh):
    @tr.analysis("count", "Counts things")
    def count(ctx):
        return {"n": len(ctx)}

    assert tr.list_analyses() == ["count"]
    assert tr.run_analysis("count", [1, 2, 3]) == {"n": 3}
    ((args, kw),) = fresh
    assert args[:2] == ("count", "ok")
    assert kw["summary"] == {"n": 3} and kw["error"] is None and kw["duration_s"] >= 0


def test_analysis_description_fallbacks():
    tr.analysis("a")(lambda c: c)

    @tr.analysis("b")
    def b(c):
        """B doc"""

    assert tr._ANALYSES["a"].description == "a" and tr._ANALYSES["b"].description == "B doc"


def test_run_analysis_unknown_returns_none_and_records_nothing(fresh):
    assert tr.run_analysis("nope", 1) is None
    assert fresh == []


def test_run_analysis_error_recorded_and_reraised(fresh):
    @tr.analysis("bad")
    def bad(ctx):
        raise ValueError("kaput")

    with pytest.raises(ValueError, match="kaput"):
        tr.run_analysis("bad", None)
    ((args, kw),) = fresh
    assert args[:2] == ("bad", "error") and kw["summary"] is None and kw["error"] == "kaput"


def test_run_analysis_survives_persistence_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setitem(
        sys.modules, "system_state", types.SimpleNamespace(record_analysis_run=boom)
    )
    tr.analysis("ok")(lambda c: 42)
    assert tr.run_analysis("ok", None) == 42
    tr.analysis("bad")(lambda c: 1 / 0)
    with pytest.raises(ZeroDivisionError):  # the analysis' own error, not the db error
        tr.run_analysis("bad", None)


def test_run_analysis_survives_missing_system_state(monkeypatch):
    monkeypatch.setitem(sys.modules, "system_state", None)
    tr.analysis("ok")(lambda c: "fine")
    assert tr.run_analysis("ok", None) == "fine"


# ---------------------------------------------------------------- builtin bridge
def test_register_builtin_screeners_missing_domain_returns_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "domain", None)
    assert tr.register_builtin_screeners() == 0
    assert tr.list_screeners() == []


def test_register_builtin_screeners_wraps_specs(monkeypatch):
    class Spec:
        def __init__(self, name):
            self.name = name

        def explain(self):
            return f"explain {self.name}"

        def is_satisfied_by(self, c):
            return c == self.name

    mod = types.ModuleType("domain.screening.specifications")
    for cls in (
        "DarvasBoxSpec",
        "GoldenCrossSpec",
        "PiotroskiSpec",
        "CoffeeCanSpec",
        "MagicFormulaSpec",
        "BullCartelSpec",
    ):
        setattr(mod, cls, lambda cls=cls: Spec(cls))
    for name in ("domain", "domain.screening"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "domain.screening.specifications", mod)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert tr.register_builtin_screeners() == 6
    assert tr.list_screeners() == [
        "darvas",
        "golden_cross",
        "piotroski",
        "coffee_can",
        "magic_formula",
        "bull_cartel",
    ]
    assert tr._SCREENERS["darvas"].description == "explain DarvasBoxSpec"
    assert tr.run_screener("darvas", "DarvasBoxSpec") is True
    assert tr.run_screener("darvas", "other") is False
    assert tr.register_builtin_screeners() == 6 and len(tr.list_screeners()) == 6  # idempotent


def test_explicit_description_beats_docstring():
    @tr.screener("both", "explicit text")
    def both(c):
        """doc text"""

    @tr.analysis("both_a", "explicit analysis")
    def both_a(c):
        """doc text"""

    assert tr._SCREENERS["both"].description == "explicit text"
    assert tr._ANALYSES["both_a"].description == "explicit analysis"
