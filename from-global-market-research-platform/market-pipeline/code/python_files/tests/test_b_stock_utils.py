"""Offline tests for stock_utils (cleaning, statement helpers, download wrapper)."""
import types

import numpy as np
import pandas as pd
import pytest

import stock_utils as su


def _ohlcv(n=5, close=None, vol=None):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = close if close is not None else [10.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": vol if vol is not None else [100] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------- statement helpers
class _T:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_first_df_skips_none_empty_and_non_frames():
    good = pd.DataFrame({"a": [1]})
    t = _T(a=None, b=pd.DataFrame(), c="notdf", d=good)
    assert su.first_df(t, "missing", "a", "b", "c", "d") is good
    assert su.first_df(t, "a", "b") is None
    assert su.first_df(t) is None


def test_row_default_col_and_named_col():
    df = pd.DataFrame({"2024": [1.0, 5.0], "2023": [2.0, 6.0]}, index=["Rev", "NI"])
    assert su.row(df, "Rev") == 1.0
    assert su.row(df, "Rev", col=1) == 2.0
    assert su.row(df, "nope", "NI") == 5.0  # falls through to second name
    assert su.row(df, "nope") is None
    assert su.row(None, "Rev") is None
    assert su.row(pd.DataFrame(), "Rev") is None


def test_row_nan_and_out_of_range_column():
    df = pd.DataFrame({"a": [np.nan], "b": [3.0]}, index=["X"])
    assert su.row(df, "X") is None
    assert su.row(df, "X", col=1) == 3.0
    assert su.row(df, "X", col=9) is None  # IndexError swallowed


def test_row_nan_first_name_does_not_try_next_name():
    # a name that exists but is NaN returns None, not a later name's value
    df = pd.DataFrame({"a": [np.nan, 7.0]}, index=["X", "Y"])
    assert su.row(df, "X", "Y") is None


def test_series_drops_nan_and_picks_first_matching_name():
    df = pd.DataFrame({"a": [1.0, 3.0], "b": [np.nan, 4.0], "c": [2.0, 5.0]}, index=["X", "Y"])
    assert su.series(df, "X") == [1.0, 2.0]
    assert su.series(df, "nope", "Y") == [3.0, 4.0, 5.0]
    assert su.series(df, "nope") == []
    assert su.series(None, "X") == []
    assert su.series(pd.DataFrame(), "X") == []


def test_fin_metric_default_only_when_none():
    df = pd.DataFrame({"a": [0.0]}, index=["X"])
    assert su.fin_metric(df, ("X",), default=9) == 0.0  # zero is a value, not missing
    assert su.fin_metric(df, ("nope",), default=9) == 9
    assert su.fin_metric(df, ("nope",)) is None
    assert su.fin_metric(None, ("X",), default=1.5) == 1.5


# ---------------------------------------------------------------- clean_ohlcv
def test_clean_ohlcv_none_and_empty():
    assert su.clean_ohlcv(None) is None
    assert su.clean_ohlcv(pd.DataFrame()) is None


def test_clean_ohlcv_requires_close_column():
    df = pd.DataFrame({"Open": [1.0, 2.0]}, index=pd.date_range("2024", periods=2))
    assert su.clean_ohlcv(df) is None


def test_clean_ohlcv_sorts_dedups_keeps_last():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-01"])
    df = pd.DataFrame(
        {"Open": [3, 1, 2], "High": [3, 1, 2], "Low": [3, 1, 2], "Close": [3.0, 1.0, 2.0],
         "Volume": [1, 1, 1]},
        index=idx,
    )
    out = su.clean_ohlcv(df)
    assert list(out.index) == list(pd.to_datetime(["2024-01-01", "2024-01-03"]))
    assert out["Close"].tolist() == [2.0, 3.0]


def test_clean_ohlcv_coerces_strings_and_drops_nan_close():
    df = _ohlcv(4)
    df["Close"] = ["10", "bad", "12", "13"]
    out = su.clean_ohlcv(df)
    assert len(out) == 3
    assert out["Close"].dtype.kind == "f"


def test_clean_ohlcv_fills_ohlc_gaps_from_close_and_volume_zero():
    df = _ohlcv(3, close=[10.0, 11.0, 12.0])
    df.loc[df.index[1], ["Open", "High", "Low", "Volume"]] = np.nan
    out = su.clean_ohlcv(df)
    row1 = out.iloc[1]
    assert (row1["Open"], row1["High"], row1["Low"]) == (11.0, 11.0, 11.0)
    assert row1["Volume"] == 0


def test_clean_ohlcv_negative_volume_clipped_to_zero():
    df = _ohlcv(3, vol=[5, -7, 5])
    out = su.clean_ohlcv(df)
    assert out["Volume"].tolist() == [5, 0, 5]


def test_clean_ohlcv_drops_non_positive_prices():
    df = _ohlcv(4)
    df.loc[df.index[1], "Close"] = 0.0
    df.loc[df.index[2], "Low"] = -1.0
    out = su.clean_ohlcv(df)
    # row1 close 0 -> dropped; row2 Low=-1 -> OHLC repair cannot happen, dropped by >0 gate
    assert len(out) == 2


def test_clean_ohlcv_all_bad_returns_none():
    df = _ohlcv(2, close=[0.0, -1.0])
    assert su.clean_ohlcv(df) is None


def test_clean_ohlcv_repairs_ohlc_integrity():
    df = pd.DataFrame(
        {"Open": [10.0], "High": [9.0], "Low": [12.0], "Close": [11.0], "Volume": [1]},
        index=pd.date_range("2024", periods=1),
    )
    out = su.clean_ohlcv(df)
    assert out.iloc[0]["High"] == 12.0
    assert out.iloc[0]["Low"] == 9.0


def test_clean_ohlcv_jump_on_zero_volume_neutralised():
    df = _ohlcv(4, close=[10.0, 10.0, 100.0, 10.0], vol=[5, 5, 0, 5])
    out = su.clean_ohlcv(df)
    # the 100 bar (zero volume, +900%) is replaced by forward-fill of 10
    assert out["Close"].tolist() == [10.0, 10.0, 10.0, 10.0]


def test_clean_ohlcv_jump_with_volume_kept():
    df = _ohlcv(3, close=[10.0, 10.0, 100.0], vol=[5, 5, 5])
    out = su.clean_ohlcv(df)
    assert out["Close"].iloc[-1] == 100.0


def test_clean_ohlcv_jump_boundary_is_strictly_greater():
    # exactly +60% on zero volume is NOT neutralised (ret > max, not >=)
    df = _ohlcv(2, close=[10.0, 15.0], vol=[5, 0])  # ret is exactly 0.5
    out = su.clean_ohlcv(df, max_daily_move=0.5)
    assert out["Close"].iloc[-1] == 15.0
    df2 = _ohlcv(2, close=[10.0, 15.1], vol=[5, 0])
    out2 = su.clean_ohlcv(df2, max_daily_move=0.5)
    assert out2["Close"].iloc[-1] == 10.0


def test_clean_ohlcv_custom_move_threshold():
    df = _ohlcv(2, close=[10.0, 12.0], vol=[5, 0])
    assert su.clean_ohlcv(df, max_daily_move=0.1)["Close"].iloc[-1] == 10.0
    assert su.clean_ohlcv(df)["Close"].iloc[-1] == 12.0


def test_clean_ohlcv_min_bars_boundary():
    df = _ohlcv(3)
    assert su.clean_ohlcv(df, min_bars=3) is not None
    assert su.clean_ohlcv(df, min_bars=4) is None


def test_clean_ohlcv_without_volume_column():
    df = _ohlcv(3).drop(columns="Volume")
    out = su.clean_ohlcv(df)
    assert "Volume" not in out.columns and len(out) == 3


def test_clean_ohlcv_idempotent():
    once = su.clean_ohlcv(_ohlcv(6))
    twice = su.clean_ohlcv(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_ohlcv_does_not_mutate_input():
    df = _ohlcv(3)
    snap = df.copy()
    su.clean_ohlcv(df)
    pd.testing.assert_frame_equal(df, snap)


def test_clean_ohlcv_verbose_prints_only_when_something_dropped(capsys):
    su.clean_ohlcv(_ohlcv(3), ticker="AAA", verbose=True)
    assert capsys.readouterr().out == ""
    df = _ohlcv(3)
    df.loc[df.index[1], "Close"] = np.nan
    su.clean_ohlcv(df, ticker="AAA", verbose=True)
    out = capsys.readouterr().out
    assert "clean_ohlcv[AAA]: dropped 1 rows, neutralised 0 bad-print bars -> 2 clean bars" in out
    su.clean_ohlcv(df, verbose=True)
    assert "clean_ohlcv[?]" in capsys.readouterr().out


# ---------------------------------------------------------------- clean_financials
def test_clean_financials_none_empty():
    assert su.clean_financials(None) is None
    assert su.clean_financials(pd.DataFrame()) is None


def test_clean_financials_all_nan_returns_none():
    df = pd.DataFrame({"a": [np.nan], "b": [np.nan]}, index=["X"])
    assert su.clean_financials(df) is None


def test_clean_financials_orders_newest_first_and_drops_empty_cols():
    df = pd.DataFrame(
        {"2022-12-31": [1.0], "2024-12-31": [3.0], "2023-12-31": [np.nan], "2021": ["x"]},
        index=["X"],
    )
    out = su.clean_financials(df)
    assert list(out.columns) == ["2024-12-31", "2022-12-31"]
    assert out.loc["X"].tolist() == [3.0, 1.0]


def test_clean_financials_mixed_type_columns_left_as_is():
    df = pd.DataFrame([[1.0, 2.0]], columns=["a", 5], index=["X"])  # unsortable columns
    out = su.clean_financials(df)
    assert list(out.columns) == ["a", 5]


# ---------------------------------------------------------------- extract_ticker_df
def _bulk(tickers):
    frames = {t: _ohlcv(4, close=[10.0 + i + j for j in range(4)]) for i, t in enumerate(tickers)}
    return pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)


def test_extract_ticker_df_multiindex_picks_requested_ticker():
    raw = _bulk(["AAA", "BBB"])
    out = su.extract_ticker_df(raw, "BBB")
    assert out["Close"].iloc[0] == 11.0
    assert list(out.columns) == su.OHLCV_COLS


def test_extract_ticker_df_unknown_ticker_none():
    assert su.extract_ticker_df(_bulk(["AAA"]), "ZZZ") is None


def test_extract_ticker_df_flat_frame():
    out = su.extract_ticker_df(_ohlcv(3), "ANY")
    assert len(out) == 3


def test_extract_ticker_df_empty_after_dropna():
    df = pd.DataFrame({"Close": [np.nan, np.nan]}, index=pd.date_range("2024", periods=2))
    assert su.extract_ticker_df(df, "X") is None


def test_extract_ticker_df_clean_false_skips_cleaning_but_enforces_min_bars():
    df = _ohlcv(3)
    df.loc[df.index[0], "Close"] = -5.0  # would be dropped by cleaning
    out = su.extract_ticker_df(df, "X", clean=False)
    assert len(out) == 3
    assert su.extract_ticker_df(df, "X", clean=False, min_bars=4) is None
    assert su.extract_ticker_df(df, "X", clean=False, min_bars=3) is not None


def test_extract_ticker_df_clean_false_restricts_to_ohlcv_columns():
    df = _ohlcv(3)
    df["Extra"] = 1
    assert list(su.extract_ticker_df(df, "X", clean=False).columns) == su.OHLCV_COLS
    only = pd.DataFrame({"Other": [1.0]}, index=pd.date_range("2024", periods=1))
    assert list(su.extract_ticker_df(only, "X", clean=False).columns) == ["Other"]


def test_extract_ticker_df_min_bars_applies_when_cleaning():
    assert su.extract_ticker_df(_ohlcv(3), "X", min_bars=4) is None
    assert su.extract_ticker_df(_ohlcv(3), "X", min_bars=3) is not None


# ---------------------------------------------------------------- misc small helpers
@pytest.mark.parametrize(
    "msg,expected",
    [("Too Many Requests", True), ("HTTP 429", True), ("Rate limited", True),
     ("Invalid Crumb", True), ("timeout", False), ("", False)],
)
def test_is_rate_limited(msg, expected):
    assert su.is_rate_limited(Exception(msg)) is expected


@pytest.mark.parametrize(
    "tkr,expected",
    [("RELIANCE.NS", "RELIANCE"), ("X.BO", "X"), ("7203.T", "7203"), ("005930.KS", "005930"),
     ("035720.KQ", "035720"), ("AAPL", "AAPL"), ("2330.TW", "2330.TW"), ("1513.TWO", "1513.TWO"),
     ("ADS.DE", "ADS.DE"), ("A.NS.T", "A.NS"), (".T", "")],
)
def test_strip_suffix(tkr, expected):
    assert su._strip_suffix(tkr) == expected


def test_parallel_map_collects_non_none_and_swallows_errors(capsys):
    def fn(x):
        if x == 3:
            raise ValueError("boom")
        return None if x % 2 == 0 else x * 10

    out = su.parallel_map(fn, [1, 2, 3, 4, 5], workers=2, verbose=False)
    assert sorted(out) == [10, 50]
    assert capsys.readouterr().out == ""


def test_parallel_map_progress_lines(capsys):
    su.parallel_map(lambda x: x, list(range(4)), workers=1, progress_every=2, label="foo")
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    assert lines == ["2/4 foo processed, 2 results", "4/4 foo processed, 4 results"]


def test_parallel_map_empty():
    assert su.parallel_map(lambda x: x, [], verbose=False) == []


def test_pct_change():
    assert su.pct_change(110, 100) == pytest.approx(10.0)
    assert su.pct_change(90, -100) == pytest.approx(190.0)  # abs() denominator
    assert su.pct_change(0, 100) == pytest.approx(-100.0)
    assert su.pct_change(5, 0) is None
    assert su.pct_change(None, 5) is None
    assert su.pct_change(5, None) is None


def test_cagr():
    assert su.cagr(200, 100, 1) == pytest.approx(100.0)
    assert su.cagr(121, 100, 2) == pytest.approx(10.0)
    assert su.cagr(100, 0, 3) is None
    assert su.cagr(100, -5, 3) is None
    assert su.cagr(0, 100, 3) is None
    assert su.cagr(None, 100, 3) is None
    assert su.cagr(100, 50, 0) is None


def test_normalise_debt_to_equity_threshold():
    assert su.normalise_debt_to_equity(None) is None
    assert su.normalise_debt_to_equity(10) == 10  # not strictly greater
    assert su.normalise_debt_to_equity(10.01) == pytest.approx(0.1001)
    assert su.normalise_debt_to_equity(45.2) == pytest.approx(0.452)
    assert su.normalise_debt_to_equity(0.5) == 0.5


def test_market_cap_crores():
    assert su.market_cap_crores(None) == 0
    assert su.market_cap_crores(0) == 0
    assert su.market_cap_crores(5e9) == 500.0


# ---------------------------------------------------------------- bulk_download
class _FakeYF:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def download(self, batch, **kw):
        self.calls.append((list(batch), kw))
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture
def nosleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(su.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_bulk_download_without_yfinance_returns_empty(monkeypatch):
    monkeypatch.setattr(su, "_YF_OK", False)
    assert su.bulk_download(["A"]) == {}


def test_bulk_download_batches_period_and_suffix_strip(monkeypatch, nosleep, capsys):
    raw = _bulk(["A.NS", "B.NS", "C.NS"])
    fake = _FakeYF([raw, raw])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    out = su.bulk_download(["A.NS", "B.NS", "C.NS"], period="6mo", batch_size=2,
                           sleep_between=0.25, min_bars=2)
    assert set(out) == {"A", "B", "C"}
    assert [c[0] for c in fake.calls] == [["A.NS", "B.NS"], ["C.NS"]]
    assert fake.calls[0][1] == {"auto_adjust": True, "threads": True, "progress": False,
                                "period": "6mo"}
    assert nosleep == [0.25]  # sleeps between batches only, not after the last
    text = capsys.readouterr().out
    assert "bulk_download: 3 tickers, 2 batches (size=2, period=6mo)" in text
    assert "bulk_download complete: 3/3 loaded" in text


def test_bulk_download_start_end_replaces_period(monkeypatch, nosleep):
    fake = _FakeYF([_bulk(["A"]), _bulk(["A"])])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    su.bulk_download(["A"], start="2024-01-01", verbose=False)
    su.bulk_download(["A"], start="2024-01-01", end="2024-02-01", verbose=False)
    k0, k1 = fake.calls[0][1], fake.calls[1][1]
    assert k0["start"] == "2024-01-01" and "end" not in k0 and "period" not in k0
    assert k1["end"] == "2024-02-01"


def test_bulk_download_end_ignored_without_start(monkeypatch, nosleep):
    fake = _FakeYF([_bulk(["A"])])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    su.bulk_download(["A"], end="2024-02-01", verbose=False)
    assert "end" not in fake.calls[0][1] and fake.calls[0][1]["period"] == "1y"


def test_bulk_download_empty_response_breaks_without_retry(monkeypatch, nosleep, capsys):
    fake = _FakeYF([pd.DataFrame()])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    assert su.bulk_download(["A"]) == {}
    assert len(fake.calls) == 1
    assert "empty" in capsys.readouterr().out


def test_bulk_download_rate_limit_retries_with_backoff(monkeypatch, nosleep):
    fake = _FakeYF([Exception("429 Too Many"), Exception("Rate limit"), _bulk(["A"])])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    out = su.bulk_download(["A"], max_retries=3, min_bars=1, verbose=False)
    assert set(out) == {"A"}
    assert nosleep == [20, 40]


def test_bulk_download_rate_limit_gives_up_on_last_attempt(monkeypatch, nosleep):
    fake = _FakeYF([Exception("429"), Exception("429")])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    assert su.bulk_download(["A"], max_retries=2, verbose=False) == {}
    assert len(fake.calls) == 2
    assert nosleep == [20]


def test_bulk_download_other_error_does_not_retry(monkeypatch, nosleep, capsys):
    fake = _FakeYF([RuntimeError("kaboom " * 20)])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    assert su.bulk_download(["A"]) == {}
    assert len(fake.calls) == 1
    out = capsys.readouterr().out
    assert "ERROR: " + ("kaboom " * 20)[:50] in out


def test_bulk_download_min_bars_filters_short_history(monkeypatch, nosleep):
    fake = _FakeYF([_bulk(["A"])])
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su, "yf", fake, raising=False)
    assert su.bulk_download(["A"], min_bars=5, verbose=False) == {}
