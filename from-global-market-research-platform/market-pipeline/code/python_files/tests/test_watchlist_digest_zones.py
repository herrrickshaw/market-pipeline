"""Zone rules in watchlist_digest.

* zone_series: per-bar EMA trend zone (BUY close>EMA20>EMA50, SELL close<EMA50)
  — used for EVICTION in every market.
* assign_recommendations: the per-market RECOMMENDATION picked by
  backtest_zone_rules.py — trend, mean-revert (5d tercile), 1-month momentum
  (21d tercile) or 6-month momentum (126d tercile), ranked within the market.
* guard_droppable: held / value-hold / sold rows are never auto-dropped.
"""
import pandas as pd
import pytest

import watchlist_digest as wd


def closes(values, start="2026-01-01"):
    return pd.DataFrame({"Date": pd.bdate_range(start, periods=len(values)),
                         "Close": values})


class TestZoneSeries:
    def test_steady_uptrend_is_buy(self):
        z = wd.zone_series(closes([100 + i for i in range(60)]))
        assert z.iloc[-1] == "BUY"

    def test_steady_downtrend_is_sell(self):
        z = wd.zone_series(closes([200 - i for i in range(60)]))
        assert z.iloc[-1] == "SELL"

    def test_pullback_between_emas_is_hold(self):
        up = [100 + i for i in range(60)]            # EMA20 ~150, EMA50 ~137 at the end
        c = pd.Series(up, dtype=float)
        e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        mid = (e20 + e50) / 2                         # below EMA20, above EMA50
        z = wd.zone_series(closes(up + [mid]))
        assert z.iloc[-1] == "HOLD"

    def test_bars_are_ordered_by_date_not_by_row(self):
        df = closes([100 + i for i in range(60)]).iloc[::-1]
        assert wd.zone_series(df).iloc[-1] == "BUY"

    def test_minimum_history(self):
        n = wd.MIN_BARS_FOR_ZONE
        assert wd.zone_series(closes(list(range(1, n)))) is None
        assert wd.zone_series(closes(list(range(1, n + 1)))) is not None
        assert wd.zone_series(None) is None
        assert wd.zone_series(pd.DataFrame({"Date": []})) is None


def test_sell_streak_counts_only_the_trailing_run():
    z = pd.Series(["BUY", "SELL", "SELL", "HOLD", "SELL", "SELL", "SELL"])
    assert wd.sell_streak(z) == 3
    assert wd.sell_streak(pd.Series(["SELL", "SELL", "HOLD"])) == 0
    assert wd.sell_streak(None) == 0


@pytest.fixture
def rules(monkeypatch):
    def set_rules(flat=None, regime=None):
        monkeypatch.setattr(wd, "_ZONE_RULES", flat or {})
        monkeypatch.setattr(wd, "_ZONE_REGIME", regime or {})
    return set_rules


def _rows(mkt, **cols):
    n = len(next(iter(cols.values())))
    return [{"symbol": f"S{i}", "market": mkt, "missing": False, "zone": "HOLD",
             **{k: v[i] for k, v in cols.items()}} for i in range(n)]


class TestAssignRecommendations:
    def test_mean_revert_buys_the_oversold_tercile(self, rules):
        rules(flat={"KR": {"rule": "revert"}})
        rows = _rows("KR", d5=[-6, -4, -2, 2, 4, 6])
        wd.assign_recommendations(rows)
        assert [r["rec"] for r in rows] == ["BUY", "BUY", "HOLD", "SELL", "SELL", "SELL"]
        assert {r["rec_rule"] for r in rows} == {"revert"}

    def test_six_month_momentum_ranks_on_126_day_return(self, rules):
        # d5 and d126 disagree on purpose: the rule is 6-MONTH momentum.
        rules(flat={"US": {"rule": "mom126"}})
        rows = _rows("US", d5=[1, 2, 3, 4, 5, 6], d21=[0] * 6,
                     d126=[60, 50, 40, 30, 20, 10])
        wd.assign_recommendations(rows)
        assert [r["rec"] for r in rows] == ["BUY", "BUY", "BUY", "HOLD", "SELL", "SELL"]

    def test_one_month_momentum_ranks_on_21_day_return(self, rules):
        rules(flat={"JP": {"rule": "mom_st"}})
        rows = _rows("JP", d5=[1, 2, 3, 4, 5, 6], d21=[9, 8, 7, 3, 2, 1],
                     d126=[0] * 6)
        wd.assign_recommendations(rows)
        assert [r["rec"] for r in rows] == ["BUY", "BUY", "BUY", "HOLD", "SELL", "SELL"]

    def test_trend_rule_keeps_the_ema_zone(self, rules):
        rules(flat={"IN": {"rule": "trend"}})
        rows = _rows("IN", d5=[1, 2, 3, 4, 5, 6])
        rows[0]["zone"] = "SELL"
        wd.assign_recommendations(rows)
        assert rows[0]["rec"] == "SELL" and rows[1]["rec"] == "HOLD"

    def test_too_few_priced_names_falls_back_to_zone(self, rules):
        rules(flat={"KR": {"rule": "revert"}})
        rows = _rows("KR", d5=[-6, -4, 2, 4, 6])
        wd.assign_recommendations(rows)
        assert all(r["rec"] == "HOLD" for r in rows)

    def test_missing_rows_are_not_ranked(self, rules):
        rules(flat={"KR": {"rule": "revert"}})
        rows = _rows("KR", d5=[-6, -4, -2, 2, 4, 6])
        miss = {"symbol": "GONE", "market": "KR", "missing": True, "zone": "?", "d5": -99}
        wd.assign_recommendations(rows + [miss])
        assert (miss["rec"], miss["rec_rule"]) == ("?", "trend")

    def test_regime_map_overrides_flat_rule(self, rules):
        rules(flat={"KR": {"rule": "trend"}},
              regime={"KR": {"active_rule": "revert", "current_regime": "BEAR"}})
        rows = _rows("KR", d5=[-6, -4, -2, 2, 4, 6])
        wd.assign_recommendations(rows)
        assert rows[0]["rec"] == "BUY"
        assert rows[0]["rec_rule"] == "revert" and rows[0]["rec_regime"] == "BEAR"

    def test_markets_are_ranked_separately(self, rules):
        rules(flat={"KR": {"rule": "revert"}, "US": {"rule": "revert"}})
        kr = _rows("KR", d5=[-60, -40, -20, -10, -5, -1])
        us = _rows("US", d5=[1, 5, 10, 20, 40, 60])
        wd.assign_recommendations(kr + us)
        assert kr[0]["rec"] == "BUY" and us[0]["rec"] == "BUY"
        assert kr[-1]["rec"] == "SELL" and us[-1]["rec"] == "SELL"


def test_guard_droppable_protects_portfolio_rows():
    wl = pd.DataFrame({"status": ["held", "value-hold", "sold", "signal", "", "watch"]})
    droppable, protected = wd.guard_droppable(wl, list(wl.index))
    assert protected == [0, 1, 2, 4]          # blank status defaults to held
    assert droppable == [3, 5]
