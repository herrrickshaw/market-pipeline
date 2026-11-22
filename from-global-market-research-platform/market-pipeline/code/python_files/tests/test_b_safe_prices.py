"""Offline tests for safe_prices (Money, Bar, PriceSeries, median_age_days)."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import safe_prices as sp


# ================================================================ Money
def test_money_normalises_currency_and_repr():
    m = sp.Money(1234.5, "inr")
    assert m.currency == "INR" and m.amount == 1234.5
    assert repr(m) == "INR 1,234.50"


@pytest.mark.parametrize("bad", ["", "US", "USDX", None])
def test_money_rejects_bad_currency(bad):
    with pytest.raises(sp.UnsafePrice, match="3-letter"):
        sp.Money(1.0, bad)


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_money_rejects_missing_amount(bad):
    with pytest.raises(sp.UnsafePrice, match="cannot be NaN"):
        sp.Money(bad, "USD")


def test_money_zero_and_negative_are_allowed_amounts():
    assert sp.Money(0.0, "USD").amount == 0.0
    assert sp.Money(-5, "USD").amount == -5


def test_money_is_frozen():
    m = sp.Money(1.0, "USD")
    with pytest.raises(Exception):  # FrozenInstanceError
        m.amount = 2.0


def test_money_ordering_same_currency():
    a, b = sp.Money(1.0, "USD"), sp.Money(2.0, "usd")
    assert a < b and a <= b and b > a and b >= a
    assert not (b < a) and not (b <= a) and not (a > b) and not (a >= b)
    assert a <= sp.Money(1.0, "USD") and a >= sp.Money(1.0, "USD")
    assert not (a < sp.Money(1.0, "USD")) and not (a > sp.Money(1.0, "USD"))


@pytest.mark.parametrize("op", ["__lt__", "__le__", "__gt__", "__ge__"])
def test_money_cross_currency_comparison_raises(op):
    with pytest.raises(TypeError, match="cannot compare INR to USD"):
        getattr(sp.Money(1.0, "INR"), op)(sp.Money(1.0, "USD"))


@pytest.mark.parametrize("op", ["__lt__", "__le__", "__gt__", "__ge__"])
def test_money_comparison_with_bare_number_raises(op):
    with pytest.raises(TypeError, match="bare number has no currency"):
        getattr(sp.Money(1.0, "INR"), op)(5)


def test_money_equality_and_hash():
    assert sp.Money(1.0, "usd") == sp.Money(1.0, "USD")
    assert sp.Money(1.0, "USD") != sp.Money(2.0, "USD")
    assert sp.Money(1.0, "USD") != sp.Money(1.0, "EUR")
    assert sp.Money(1.0, "USD") != 1.0  # never equal to a bare number
    assert hash(sp.Money(1.0, "usd")) == hash(sp.Money(1.0, "USD"))
    assert len({sp.Money(1.0, "USD"), sp.Money(1.0, "usd"), sp.Money(1.0, "EUR")}) == 2


def test_money_conversion():
    usd = sp.Money(100.0, "INR").to("usd", 0.012)
    assert usd.currency == "USD" and usd.amount == pytest.approx(1.2)
    assert sp.Money(2.0, "USD").to("INR", 83.0) == sp.Money(166.0, "INR")


@pytest.mark.parametrize("rate", [0, -1.0, None, float("nan")])
def test_money_conversion_rejects_bad_rates(rate):
    with pytest.raises(sp.UnsafePrice, match="bad FX rate"):
        sp.Money(1.0, "INR").to("USD", rate)


def test_unsafe_price_is_a_value_error():
    assert issubclass(sp.UnsafePrice, ValueError)


# ================================================================ Bar
def test_bar_normalises_timestamp_and_datetime_to_date():
    for when in (
        pd.Timestamp("2024-03-05 15:30"),
        datetime(2024, 3, 5, 9, 0),
        date(2024, 3, 5),
        "2024-03-05",
    ):
        b = sp.Bar(when, 10.0)
        assert b.when == date(2024, 3, 5) and type(b.when) is date


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_bar_rejects_nan_close(bad):
    with pytest.raises(sp.UnsafePrice, match="cannot be NaN"):
        sp.Bar(date(2024, 1, 1), bad)


@pytest.mark.parametrize("bad", [0, 0.0, -1.5])
def test_bar_rejects_non_positive_close(bad):
    with pytest.raises(sp.UnsafePrice, match="must be positive"):
        sp.Bar(date(2024, 1, 1), bad)


def test_bar_smallest_positive_close_is_valid():
    assert sp.Bar(date(2024, 1, 1), 1e-9).close == 1e-9


def test_bar_age_days_against_explicit_and_default_asof():
    b = sp.Bar(date(2024, 1, 1), 5.0)
    assert b.age_days(date(2024, 1, 11)) == 10
    assert b.age_days(date(2024, 1, 1)) == 0
    assert b.age_days(date(2023, 12, 31)) == -1
    assert b.age_days() == (date.today() - date(2024, 1, 1)).days


def test_bar_repr_and_frozen():
    b = sp.Bar(date(2024, 1, 1), 1234.567)
    assert repr(b) == "Bar(2024-01-01 1,234.57)"
    with pytest.raises(Exception):
        b.close = 1.0


def test_bar_equality_by_value():
    assert sp.Bar(date(2024, 1, 1), 5.0) == sp.Bar(pd.Timestamp("2024-01-01"), 5.0)
    assert sp.Bar(date(2024, 1, 1), 5.0) != sp.Bar(date(2024, 1, 2), 5.0)


# ================================================================ PriceSeries
def _ps(values, start="2024-01-01", currency="", symbol="X"):
    idx = pd.bdate_range(start, periods=len(values))
    return sp.PriceSeries(pd.Series(values, index=idx), symbol=symbol, currency=currency)


def test_priceseries_requires_datetime_index():
    with pytest.raises(sp.UnsafePrice, match="DatetimeIndex"):
        sp.PriceSeries(pd.Series([1.0, 2.0]))


def test_priceseries_drops_trailing_and_inner_nans_and_coerces():
    p = _ps([10.0, np.nan, "12", "bad", np.nan])
    assert len(p) == 2
    assert p.last().close == 12.0  # trailing empty bars never become the 'last' price


def test_priceseries_last_carries_its_date():
    p = _ps([10.0, 11.0, np.nan])
    b = p.last()
    assert b.close == 11.0 and b.when == date(2024, 1, 2)


def test_priceseries_empty_has_no_last():
    assert _ps([np.nan, np.nan]).last() is None
    assert len(_ps([np.nan])) == 0


def test_priceseries_asof_is_never_a_lookahead():
    p = _ps([10.0, 11.0, 12.0], start="2024-01-01")  # Mon, Tue, Wed
    assert p.asof("2024-01-02").close == 11.0  # on the date itself
    assert p.asof("2024-01-03 23:00").close == 12.0
    assert p.asof("2024-01-06").close == 12.0  # weekend -> last settled bar
    assert p.asof("2023-12-31") is None
    assert p.asof(date(2024, 1, 1)).when == date(2024, 1, 1)


def test_priceseries_pct_between_end_defaults_to_last():
    p = _ps([100.0, 110.0, 121.0])
    assert p.pct_between("2024-01-01") == pytest.approx(21.0)
    assert p.pct_between("2024-01-01", "2024-01-02") == pytest.approx(10.0)
    assert p.pct_between("2024-01-02", "2024-01-02") == pytest.approx(0.0)


def test_priceseries_pct_between_refuses_bad_orderings_and_missing_bars():
    p = _ps([100.0, 110.0, 121.0])
    # end bar predates start bar -> None, not a bogus negative/positive number
    assert p.pct_between("2024-01-03", "2024-01-01") is None
    assert p.pct_between("2023-01-01") is None  # no bar at or before the start
    assert _ps([np.nan]).pct_between("2024-01-01") is None
    assert p.pct_between("2024-01-01", "2023-01-01") is None


def test_priceseries_pct_between_is_relative_to_start_close():
    p = _ps([200.0, 100.0])
    assert p.pct_between("2024-01-01") == pytest.approx(-50.0)


def test_priceseries_money_last():
    p = _ps([10.0, 11.0], currency="inr", symbol="TCS")
    assert p.money_last() == sp.Money(11.0, "INR")
    assert _ps([np.nan], currency="inr").money_last() is None
    with pytest.raises(sp.UnsafePrice, match="TCS: no currency set"):
        _ps([10.0], symbol="TCS").money_last()


def test_priceseries_currency_uppercased():
    assert _ps([1.0], currency="usd").currency == "USD"


# ================================================================ median_age_days
def test_median_age_days_uses_median_not_newest():
    asof = date(2024, 1, 31)
    bars = [sp.Bar(date(2024, 1, 31), 1.0)] + [sp.Bar(date(2024, 1, 27), 1.0)] * 4  # 0,4,4,4,4
    assert sp.median_age_days(bars, asof) == 4.0


def test_median_age_days_even_count_averages_middle_pair():
    asof = date(2024, 1, 31)
    bars = [sp.Bar(date(2024, 1, 31), 1.0), sp.Bar(date(2024, 1, 29), 1.0)]
    assert sp.median_age_days(bars, asof) == 1.0


def test_median_age_days_skips_none_and_handles_empty():
    asof = date(2024, 1, 31)
    assert sp.median_age_days([None, sp.Bar(date(2024, 1, 30), 1.0), None], asof) == 1.0
    assert sp.median_age_days([None, None], asof) is None
    assert sp.median_age_days([], asof) is None
    assert sp.median_age_days(iter([sp.Bar(date(2024, 1, 30), 1.0)]), asof) == 1.0


def test_median_age_days_default_asof_is_today():
    b = sp.Bar(date.today(), 1.0)
    assert sp.median_age_days([b]) == 0.0
