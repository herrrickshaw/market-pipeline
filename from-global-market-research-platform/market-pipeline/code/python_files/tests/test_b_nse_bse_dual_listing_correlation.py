"""Offline tests for nse_bse_dual_listing_correlation (pairing, correlation, name verification)."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import nse_bse_dual_listing_correlation as dl


def _px(n=100, seed=0, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=pd.bdate_range(start, periods=n))


# ---------------------------------------------------------------- fetch_dual_listing_prices
class _YF:
    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def download(self, tickers, **kw):
        self.calls.append((list(tickers), kw))
        r = self.frames.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _raw(tickers):
    return pd.concat({t: pd.DataFrame({"Close": _px(seed=i)}) for i, t in enumerate(tickers)}, axis=1)


def test_fetch_batches_symbols_and_requests_both_suffixes(monkeypatch):
    yf = _YF([_raw(["A.NS", "A.BO", "B.NS", "B.BO"]), _raw(["C.NS", "C.BO"])])
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    out = dl.fetch_dual_listing_prices(["A", "B", "C"], period="6mo", batch_size=2)
    assert [c[0] for c in yf.calls] == [["A.NS", "A.BO", "B.NS", "B.BO"], ["C.NS", "C.BO"]]
    assert yf.calls[0][1]["period"] == "6mo" and yf.calls[0][1]["group_by"] == "ticker"
    assert set(out) == {"A", "B", "C"} and len(out["A"]["nse"]) == 100 and len(out["C"]["bse"]) == 100


def test_fetch_missing_side_is_none_and_failed_batch_skipped(monkeypatch, capsys):
    partial = _raw(["A.NS"])  # no A.BO column
    yf = _YF([partial, RuntimeError("rate limited"), _raw(["C.NS", "C.BO"])])
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    out = dl.fetch_dual_listing_prices(["A", "B", "C"], batch_size=1)
    assert out["A"]["nse"] is not None and out["A"]["bse"] is None
    assert "B" not in out and "C" in out  # the failed batch contributes nothing
    assert "batch failed: rate limited" in capsys.readouterr().out


def test_fetch_drops_nan_closes(monkeypatch):
    s = _px()
    s.iloc[3] = np.nan
    raw = pd.concat({"A.NS": pd.DataFrame({"Close": s}), "A.BO": pd.DataFrame({"Close": _px(seed=1)})}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", _YF([raw]))
    assert len(dl.fetch_dual_listing_prices(["A"])["A"]["nse"]) == 99


# ---------------------------------------------------------------- compute_dual_listing_correlations
def test_correlation_rows_sorted_and_gap_computed():
    base = _px(120, 1)
    noisy = base * (1 + np.random.default_rng(2).normal(0, 0.002, 120))
    indep = _px(120, 9)
    out = dl.compute_dual_listing_correlations({
        "SAME": {"nse": base, "bse": base * 1.01},
        "NOISY": {"nse": base, "bse": noisy},
        "INDEP": {"nse": base, "bse": indep}})
    assert out["symbol"].iloc[0] == "INDEP" and out["correlation"].is_monotonic_increasing
    same = out[out.symbol == "SAME"].iloc[0]
    assert same["correlation"] == pytest.approx(1.0, abs=1e-4) and same["common_days"] == 119
    assert same["price_gap_pct"] == pytest.approx(round(100 * (1 / 1.01 - 1), 3), abs=1e-3)  # (nse-bse)/bse
    assert same["nse_last_close"] == round(float(base.iloc[-1]), 2) and same["bse_last_close"] == round(float(base.iloc[-1] * 1.01), 2)


def test_correlation_min_common_days_boundary_and_misaligned_dates():
    a = _px(61)
    out = dl.compute_dual_listing_correlations({"X": {"nse": a, "bse": a}})
    assert len(out) == 1  # 61 prices -> 60 returns: exactly at the minimum
    out = dl.compute_dual_listing_correlations({"X": {"nse": a.iloc[:60], "bse": a.iloc[:60]}})
    assert out.empty  # 60 common prices -> 59 returns < 60
    shifted = a.copy()
    shifted.index = shifted.index + pd.Timedelta(days=400)
    assert dl.compute_dual_listing_correlations({"X": {"nse": a, "bse": shifted}}).empty
    assert len(dl.compute_dual_listing_correlations({"X": {"nse": a.iloc[:30], "bse": a.iloc[:30]}}, min_common_days=20)) == 1


def test_correlation_skips_missing_empty_and_constant_series():
    a = _px(100)
    const = pd.Series(50.0, index=a.index)
    out = dl.compute_dual_listing_correlations({
        "NONE": {"nse": a, "bse": None}, "EMPTY": {"nse": a, "bse": a.iloc[0:0]},
        "NOKEYS": {}, "CONST": {"nse": a, "bse": const}})
    assert out.empty and list(out.columns) == ["symbol", "correlation", "common_days", "nse_last_close",
                                               "bse_last_close", "price_gap_pct"]


# ---------------------------------------------------------------- _name_tokens
@pytest.mark.parametrize(
    "name,tokens",
    [("Reliance Industries Limited", {"RELIANCE", "INDUSTRIES"}), ("TATA CONSULTANCY SERVICES LTD.", {"TATA", "CONSULTANCY", "SERVICES"}),
     ("Nippon India ETF Nifty 50", {"NIPPON", "NIFTY", "50"}), ("A & B Co", set()), ("3P Land Holdings Ltd", {"3P", "LAND", "HOLDINGS"}),
     ("the bank of india", {"BANK"}), (None, set()), (123, set()), ("", set())],
)
def test_name_tokens(name, tokens):
    assert dl._name_tokens(name) == tokens


# ---------------------------------------------------------------- verify_dual_listing_names
def _anoms(*syms):
    return pd.DataFrame({"symbol": list(syms), "correlation": [0.3] * len(syms), "price_gap_pct": [1.0] * len(syms)})


def _fake_yf_names(monkeypatch, names, fail=()):
    calls = []

    class T:
        def __init__(self, t):
            self.t = t
            calls.append(t)

        @property
        def info(self):
            if self.t in fail:
                raise RuntimeError("rate limited")
            return {"longName": names.get(self.t)}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T))
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def test_verify_marks_same_different_and_unverifiable(monkeypatch):
    names = {"A.NS": "Acme Industries Limited", "A.BO": "ACME INDUSTRIES LTD",
             "B.NS": "Bharat Steel Limited", "B.BO": "Zenith Textiles Limited",
             "C.NS": "Cipla Limited", "C.BO": None}
    _fake_yf_names(monkeypatch, names)
    out = dl.verify_dual_listing_names(_anoms("A", "B", "C"), sleep_between=0)
    by = out.set_index("symbol")
    assert by.at["A", "same_company"] == True and by.at["B", "same_company"] == False  # noqa: E712
    assert by.at["C", "same_company"] is None or pd.isna(by.at["C", "same_company"])
    assert by.at["A", "nse_name"] == "Acme Industries Limited" and by.at["C", "bse_name"] == ""
    assert list(out.columns)[-3:] == ["nse_name", "bse_name", "same_company"]


def test_verify_overlap_threshold_is_sixty_percent_of_the_smaller_set(monkeypatch):
    names = {"A.NS": "ALPHA BETA GAMMA DELTA EPSILON", "A.BO": "ALPHA BETA GAMMA ZETA ETA",  # 3/5 = 0.6 -> same
             "B.NS": "ALPHA BETA GAMMA DELTA EPSILON", "B.BO": "ALPHA BETA ZETA ETA THETA"}  # 2/5 = 0.4 -> different
    _fake_yf_names(monkeypatch, names)
    by = dl.verify_dual_listing_names(_anoms("A", "B"), sleep_between=0).set_index("symbol")
    assert by.at["A", "same_company"] == True and by.at["B", "same_company"] == False  # noqa: E712


def test_verify_retries_with_exponential_backoff_then_gives_up(monkeypatch):
    calls, sleeps = _fake_yf_names(monkeypatch, {"A.BO": "Acme"}, fail={"A.NS"})
    out = dl.verify_dual_listing_names(_anoms("A"), sleep_between=0.5, max_retries=3)
    assert calls.count("A.NS") == 3 and out["nse_name"].iloc[0] == ""
    assert sleeps[:2] == [0.5, 1.0]  # backoff between attempts 1->2 and 2->3, none after the last
    assert sleeps.count(0.5) >= 3  # polite pause after each symbol side too


def test_verify_without_sleep_skips_pauses(monkeypatch):
    _, sleeps = _fake_yf_names(monkeypatch, {"A.NS": "X Y", "A.BO": "X Y"})
    dl.verify_dual_listing_names(_anoms("A"), sleep_between=0)
    assert sleeps == []


# ---------------------------------------------------------------- run
def _patch_run(monkeypatch, result, verify=None):
    monkeypatch.setattr(dl, "fetch_dual_listing_prices", lambda syms, period="1y": {"seen": list(syms), "period": period})
    monkeypatch.setattr(dl, "compute_dual_listing_correlations", lambda d: result)
    if verify is not None:
        monkeypatch.setattr(dl, "verify_dual_listing_names", verify)


def _result():
    return pd.DataFrame({"symbol": ["LOW", "MID", "HIGH"], "correlation": [0.3, 0.85, 0.999], "common_days": [100, 100, 100],
                         "nse_last_close": [1.0, 2.0, 3.0], "bse_last_close": [1.0, 2.0, 3.0], "price_gap_pct": [0.0, 0.1, 0.2]})


def test_run_writes_csv_report_and_flags_anomalies(monkeypatch, tmp_path, capsys):
    seen = {}

    def verify(an):
        seen["an"] = list(an["symbol"])
        return an.assign(nse_name="A", bse_name="B", same_company=[False])

    _patch_run(monkeypatch, _result(), verify)
    out = dl.run(symbols=["LOW", "MID", "HIGH"], output_dir=str(tmp_path))
    assert out is not None and seen["an"] == ["LOW"]  # only correlation < 0.8
    assert pd.read_csv(tmp_path / "nse_bse_dual_listing_correlations.csv")["symbol"].tolist() == ["LOW", "MID", "HIGH"]
    assert pd.read_csv(tmp_path / "nse_bse_dual_listing_anomalies_verified.csv")["same_company"].tolist() == [False]
    report = (tmp_path / "nse_bse_dual_listing_report.txt").read_text()
    assert "3 dual-listed symbols found out of 3 checked, period=1y" in report
    assert "mean:" in report and "p10:" in report and "Anomalies (correlation < 0.8, 1 found)" in report
    assert "1 CONFIRMED as genuine ticker collisions" in report and "CONFIRMED COLLISIONS" in report
    assert "3 dual-listed symbols, 1 below the 0.8 anomaly threshold, 1 confirmed genuine ticker collisions" in capsys.readouterr().out


def test_run_threshold_parameter_and_no_anomalies(monkeypatch, tmp_path):
    _patch_run(monkeypatch, _result(), lambda an: pytest.fail("no anomalies expected"))
    dl.run(symbols=["A"], anomaly_threshold=0.2, output_dir=str(tmp_path))
    assert "Anomalies (correlation < 0.2, 0 found)" in (tmp_path / "nse_bse_dual_listing_report.txt").read_text()
    assert "  none" in (tmp_path / "nse_bse_dual_listing_report.txt").read_text()
    assert not (tmp_path / "nse_bse_dual_listing_anomalies_verified.csv").exists()


def test_run_with_no_dual_listings(monkeypatch, tmp_path):
    empty = pd.DataFrame(columns=_result().columns)
    _patch_run(monkeypatch, empty)
    dl.run(symbols=["A", "B"], output_dir=str(tmp_path))
    assert "No dual-listed symbols found in this sample." in (tmp_path / "nse_bse_dual_listing_report.txt").read_text()


def test_run_sample_is_deterministic_and_capped(monkeypatch, tmp_path):
    got = []
    monkeypatch.setattr(dl, "fetch_dual_listing_prices", lambda syms, period="1y": got.append(list(syms)) or {})
    monkeypatch.setattr(dl, "compute_dual_listing_correlations", lambda d: pd.DataFrame(columns=_result().columns))
    syms = [f"S{i}" for i in range(50)]
    dl.run(symbols=syms, sample=5, output_dir=str(tmp_path))
    dl.run(symbols=syms, sample=5, output_dir=str(tmp_path))
    dl.run(symbols=syms[:3], sample=10, output_dir=str(tmp_path))
    assert got[0] == got[1] and len(got[0]) == 5 and got[0] != syms[:5]
    assert len(got[2]) == 3  # sample never exceeds the universe


def test_run_loads_nse_symbols_from_symbol_master(monkeypatch, tmp_path):
    master = pd.DataFrame({"symbol": ["TCS", "AAPL", "INFY"], "exchange": ["NSE", "NASDAQ", "NSE"]})
    monkeypatch.setitem(sys.modules, "symbol_master", types.SimpleNamespace(load_master=lambda auto_refresh: master))
    got = []
    monkeypatch.setattr(dl, "fetch_dual_listing_prices", lambda syms, period="1y": got.append(list(syms)) or {})
    monkeypatch.setattr(dl, "compute_dual_listing_correlations", lambda d: pd.DataFrame(columns=_result().columns))
    dl.run(output_dir=str(tmp_path))
    assert got == [["TCS", "INFY"]]
