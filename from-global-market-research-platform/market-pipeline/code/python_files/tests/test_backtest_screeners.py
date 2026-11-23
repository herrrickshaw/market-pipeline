"""backtest_screeners.py -- regime classification, forward returns, signal detectors,
fundamental event-study screens, statistics, ranking, Excel export and symbol universes.

The module does sys.exit() without yfinance and creates ./backtest_results at import, so it
is imported in a sandbox cwd with a fake yfinance and removed from sys.modules afterwards.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


class FakeYF(types.SimpleNamespace):
    pass


@pytest.fixture
def BS(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yf = FakeYF(download=lambda *a, **k: pd.DataFrame(), Ticker=lambda s: None)
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    sys.modules.pop("backtest_screeners", None)
    mod = importlib.import_module("backtest_screeners")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "bt")
    (tmp_path / "bt").mkdir()
    yield mod
    sys.modules.pop("backtest_screeners", None)


def index_frame(n=10, close=None, dma=None, slope=None, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "Close": close if close is not None else np.full(n, 100.0),
            "dma200": dma if dma is not None else np.full(n, 90.0),
            "dma200_sl": slope if slope is not None else np.full(n, 1.0),
        },
        index=idx,
    )


# ── regime ────────────────────────────────────────────────────────────────────
def test_regime_bull_bear_sideways(BS):
    d = pd.Timestamp("2024-01-05")
    assert BS.classify_regime(d, index_frame(close=100.0, dma=90.0, slope=1.0)) == "BULL"
    assert BS.classify_regime(d, index_frame(close=80.0, dma=90.0, slope=-1.0)) == "BEAR"
    assert BS.classify_regime(d, index_frame(close=100.0, dma=90.0, slope=-1.0)) == "SIDEWAYS"
    assert BS.classify_regime(d, index_frame(close=80.0, dma=90.0, slope=1.0)) == "SIDEWAYS"
    assert BS.classify_regime(d, index_frame(close=90.0, dma=90.0, slope=1.0)) == "SIDEWAYS"
    assert BS.classify_regime(d, index_frame(close=100.0, dma=90.0, slope=0.0)) == "SIDEWAYS"


def test_regime_unknown_cases(BS):
    d = pd.Timestamp("2024-01-05")
    assert BS.classify_regime(d, None) == "UNKNOWN"
    assert BS.classify_regime(d, pd.DataFrame()) == "UNKNOWN"
    assert BS.classify_regime(pd.Timestamp("2023-01-01"), index_frame()) == "UNKNOWN"  # before data
    nan_dma = index_frame()
    nan_dma["dma200"] = np.nan
    assert BS.classify_regime(d, nan_dma) == "UNKNOWN"


def test_regime_uses_last_bar_on_or_before_the_date(BS):
    f = index_frame(close=[100.0] * 3 + [50.0] * 7, dma=90.0, slope=[1.0] * 3 + [-1.0] * 7)
    assert BS.classify_regime(pd.Timestamp("2024-01-03"), f) == "BULL"
    assert BS.classify_regime(pd.Timestamp("2024-01-04"), f) == "BEAR"  # first bad bar
    assert BS.classify_regime(pd.Timestamp("2024-06-01"), f) == "BEAR"  # after the data: last bar


# ── forward returns ───────────────────────────────────────────────────────────
def test_forward_returns_net_of_round_trip_cost(BS):
    closes = 100.0 + np.arange(30.0)
    df = pd.DataFrame({"Close": closes})
    out = BS.compute_forward_returns(df, 0, 100.0)
    assert out["T+1d"] == pytest.approx(1.0 - 0.2)
    assert out["T+3d"] == pytest.approx(3.0 - 0.2)
    assert out["T+1wk"] == pytest.approx(5.0 - 0.2)
    assert out["T+1mo"] == pytest.approx(21.0 - 0.2)
    assert np.isnan(out["T+3mo"])  # exit bar 63 beyond the data


def test_forward_returns_without_costs_and_boundary(BS):
    df = pd.DataFrame({"Close": 100.0 + np.arange(22.0)})
    out = BS.compute_forward_returns(df, 0, 100.0, apply_costs=False)
    assert out["T+1mo"] == pytest.approx(21.0)  # bar 21 is the last (index n-1)
    out2 = BS.compute_forward_returns(df, 1, 101.0, apply_costs=False)
    assert np.isnan(out2["T+1mo"])  # exit idx 22 == n -> NaN
    assert BS.compute_forward_returns(df, 0, 100.0)["T+1d"] == pytest.approx(0.8)


def test_index_returns_horizons_and_failure_modes(BS):
    idx = index_frame(n=30, close=100.0 + np.arange(30.0))
    out = BS.compute_index_returns(idx, pd.Timestamp("2024-01-01"))
    assert out["T+1d"] == pytest.approx(1.0) and out["T+1mo"] == pytest.approx(21.0)
    assert np.isnan(out["T+3mo"])
    later = BS.compute_index_returns(idx, pd.Timestamp("2024-01-03"))  # entry bar 2
    assert later["T+1d"] == pytest.approx((103.0 - 102.0) / 102.0 * 100)
    for bad in (None, pd.DataFrame()):
        assert all(
            np.isnan(v) for v in BS.compute_index_returns(bad, pd.Timestamp("2024-01-01")).values()
        )
    before = BS.compute_index_returns(idx, pd.Timestamp("2000-01-01"))
    assert all(np.isnan(v) for v in before.values())


def test_index_returns_handles_timezone_aware_index(BS):
    idx = index_frame(n=30, close=100.0 + np.arange(30.0))
    idx.index = idx.index.tz_localize("Asia/Kolkata")
    sig = pd.Timestamp("2024-01-01", tz="Asia/Kolkata")
    assert BS.compute_index_returns(idx, sig)["T+1d"] == pytest.approx(1.0)


# ── is_financial_stock ────────────────────────────────────────────────────────
def test_financial_stock_detection(BS):
    assert BS.is_financial_stock("HDFCBANK") and BS.is_financial_stock("BAJFINANCE")
    assert BS.is_financial_stock("sbilife")
    assert not BS.is_financial_stock("TCS") and not BS.is_financial_stock("INFY")
    assert BS.is_financial_stock("XYZ", {"sector": "Financial Services"})
    assert BS.is_financial_stock("XYZ", {"industry": "Insurance - Life"})
    assert BS.is_financial_stock("XYZ", {"industry": "Capital Markets"})
    assert not BS.is_financial_stock("XYZ", {"sector": "Technology", "industry": "Software"})
    assert not BS.is_financial_stock("XYZ", {"sector": None, "industry": None})
    assert not BS.is_financial_stock("XYZ", {})


# ── Darvas ────────────────────────────────────────────────────────────────────
def darvas_frame(vol_spike=2000.0, n=240):
    idx = pd.bdate_range("2023-01-02", periods=n)
    highs = np.full(n, 100.0)
    closes = np.full(n, 99.0)
    vols = np.full(n, 1000.0)
    highs[200] = 110.0
    highs[201:204] = 105.0  # three lower highs confirm a box
    closes[210] = 111.0  # first close above the box
    highs[210] = 111.0
    vols[210] = vol_spike
    return pd.DataFrame(
        {"High": highs, "Low": highs - 2, "Close": closes, "Volume": vols}, index=idx
    )


def test_darvas_breakout_signal_fields(BS):
    sigs = BS.detect_darvas_signals(darvas_frame())
    assert [s["signal_idx"] for s in sigs] == [210]
    s = sigs[0]
    assert s["entry_price"] == 111.0 and s["box_top"] == 105.0 and s["vol_confirm"] is True
    assert s["date"] == darvas_frame().index[210]


def test_darvas_volume_confirmation_threshold(BS):
    assert BS.detect_darvas_signals(darvas_frame(vol_spike=1199.0)) == []  # < 1.2x average
    assert len(BS.detect_darvas_signals(darvas_frame(vol_spike=1200.0))) == 1  # exactly 1.2x
    assert len(BS.detect_darvas_signals(darvas_frame(vol_spike=500.0), require_volume=False)) == 1


def test_darvas_cooldown_and_guards(BS):
    f = darvas_frame()
    f.loc[f.index[211], "Close"] = 99.0
    f.loc[f.index[212], ["Close", "High"]] = 111.0
    f.loc[f.index[212], "Volume"] = 2000.0
    assert [s["signal_idx"] for s in BS.detect_darvas_signals(f, cooldown=1)] == [210, 212]
    assert [s["signal_idx"] for s in BS.detect_darvas_signals(f, cooldown=2)] == [210, 212]
    assert [s["signal_idx"] for s in BS.detect_darvas_signals(f, cooldown=3)] == [210]
    assert BS.detect_darvas_signals(None) == [] and BS.detect_darvas_signals(pd.DataFrame()) == []
    assert BS.detect_darvas_signals(darvas_frame().iloc[:209]) == []  # fewer than 210 bars


def test_darvas_has_no_lookahead(BS):
    f = darvas_frame()
    cut = f.copy()
    cut.iloc[212:, cut.columns.get_loc("Close")] = 500.0
    assert (
        BS.detect_darvas_signals(f)[0]["signal_idx"]
        == BS.detect_darvas_signals(cut)[0]["signal_idx"]
    )


# ── Golden cross ──────────────────────────────────────────────────────────────
def test_golden_cross_fires_once_on_the_crossing_bar(BS):
    closes = np.concatenate([np.full(200, 100.0), np.full(60, 80.0), np.full(100, 130.0)])
    df = pd.DataFrame({"Close": closes}, index=pd.bdate_range("2022-01-03", periods=len(closes)))
    sigs = BS.detect_golden_cross_signals(df)
    s50, s200 = df.Close.rolling(50).mean(), df.Close.rolling(200).mean()
    crossings = [
        i
        for i in range(1, len(df))
        if s50.iloc[i - 1] < s200.iloc[i - 1] and s50.iloc[i] > s200.iloc[i]
    ]
    start = len(df) - BS.LOOKBACK_DAYS
    expected = [i for i in crossings if i >= start]
    assert expected and [s["signal_idx"] for s in sigs] == expected
    s = sigs[0]
    assert s["entry_price"] == pytest.approx(df.Close.iloc[s["signal_idx"]])
    assert s["dma50"] == round(float(s50.iloc[s["signal_idx"]]), 2)


def test_golden_cross_window_and_guards(BS):
    assert BS.detect_golden_cross_signals(None) == []
    assert BS.detect_golden_cross_signals(pd.DataFrame({"Close": np.arange(204.0)})) == []
    steady = pd.DataFrame({"Close": np.full(300, 100.0)})
    assert BS.detect_golden_cross_signals(steady) == []
    # a cross that happened before the last 252 bars is ignored
    closes = np.concatenate(
        [np.full(200, 100.0), np.full(60, 80.0), np.full(40, 130.0), np.full(300, 130.0)]
    )
    old = pd.DataFrame({"Close": closes})
    assert BS.detect_golden_cross_signals(old) == []


# ── backtest runner ───────────────────────────────────────────────────────────
def test_run_technical_backtest_builds_rows(BS, monkeypatch):
    closes = 100.0 + np.arange(300.0)
    ohlc = pd.DataFrame({"Close": closes}, index=pd.bdate_range("2023-01-02", periods=300))
    sig = {"date": ohlc.index[100], "entry_price": closes[100], "signal_idx": 100, "box_top": 5.0}
    monkeypatch.setattr(BS, "detect_darvas_signals", lambda df: [sig])
    idx = index_frame(n=300, close=100.0 + np.arange(300.0), start="2023-01-02")
    out = BS.run_screener_backtest(
        "darvas",
        [("AAA", ".NS", ohlc), ("BBB", ".NS", None), ("CCC", ".NS", pd.DataFrame())],
        idx,
        technical=True,
    )
    assert len(out) == 1
    r = out.iloc[0]
    assert r["screener"] == "darvas" and r["symbol"] == "AAA" and r["regime"] == "BULL"
    assert r["T+1d"] == pytest.approx((closes[101] - closes[100]) / closes[100] * 100 - 0.2)
    assert r["bh_T+1d"] == pytest.approx(
        (idx.Close.iloc[101] - idx.Close.iloc[100]) / idx.Close.iloc[100] * 100
    )
    assert r["extra_info"] == str({"box_top": 5.0})


def test_run_technical_backtest_unknown_screener_and_errors(BS, monkeypatch):
    ohlc = pd.DataFrame({"Close": np.arange(300.0)})
    assert BS.run_screener_backtest(
        "nonsense", [("A", "", ohlc)], pd.DataFrame(), technical=True
    ).empty
    monkeypatch.setattr(BS, "detect_golden_cross_signals", lambda df: 1 / 0)
    assert BS.run_screener_backtest(
        "golden_cross", [("A", "", ohlc)], pd.DataFrame(), technical=True
    ).empty


def test_run_fundamental_backtest_parallel_event_study(BS, monkeypatch, capsys):
    closes = 100.0 + np.arange(100.0)
    ohlc = pd.DataFrame({"Close": closes}, index=pd.bdate_range("2024-01-01", periods=100))

    def fake_signal(sym, suffix, screener, df):
        if sym == "NO":
            return None
        if sym == "BOOM":
            raise RuntimeError("x")
        return {"date": df.index[10], "entry_price": float(df.Close.iloc[10]), "signal_idx": 10}

    monkeypatch.setattr(BS, "fetch_fundamental_signal", fake_signal)
    syms = [("YES", ".NS", ohlc), ("NO", ".NS", ohlc), ("BOOM", ".NS", ohlc)]
    out = BS.run_screener_backtest("piotroski", syms, pd.DataFrame(), technical=False, workers=2)
    assert list(out["symbol"]) == ["YES"] and out["regime"].iloc[0] == "UNKNOWN"
    assert out["extra_info"].iloc[0] == ""
    assert "[piotroski] 3/3 processed, 1 signals found" in capsys.readouterr().out
    monkeypatch.setattr(BS, "fetch_fundamental_signal", lambda *a: None)
    assert BS.run_screener_backtest("piotroski", syms, pd.DataFrame()).empty


# ── analysis ──────────────────────────────────────────────────────────────────
def signals_frame():
    rows = []
    vals = {"BULL": [2.0, 4.0, -1.0, 3.0], "BEAR": [-2.0, -4.0, 1.0]}
    for regime, rets in vals.items():
        for r in rets:
            rows.append(
                {
                    "screener": "darvas",
                    "regime": regime,
                    "T+1d": r,
                    "T+1mo": r * 2,
                    "bh_T+1d": 0.5,
                    "bh_T+1mo": 1.0,
                }
            )
    for r in (1.0, 1.0):  # only two obs: below the 3-obs floor
        rows.append(
            {
                "screener": "tiny",
                "regime": "BULL",
                "T+1d": r,
                "T+1mo": r,
                "bh_T+1d": 0.0,
                "bh_T+1mo": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_analyze_signals_hand_values(BS):
    st = BS.analyze_signals(signals_frame())
    row = st[(st.Screener == "darvas") & (st.Regime == "BULL") & (st.Horizon == "T+1d")].iloc[0]
    vals = pd.Series([2.0, 4.0, -1.0, 3.0])
    assert row["N_Signals"] == 4 and row["Stat_Flag"] == "⚠️ LOW N"
    assert row["Hit_Rate%"] == 75.0 and row["Avg_Return%"] == 2.0 and row["Median_Return%"] == 2.5
    assert row["Std_Dev%"] == round(vals.std(), 2)
    assert row["Sharpe"] == round(2.0 / vals.std(), 3)
    assert row["Sortino"] == 0  # one loss: std NaN -> 0
    assert row["Profit_Factor"] == 9.0
    assert row["Max_Win%"] == 4.0 and row["Max_Loss%"] == -1.0
    assert row["Avg_Win%"] == 3.0 and row["Avg_Loss%"] == -1.0
    assert row["Expected_Value%"] == round(0.75 * 3.0 + 0.25 * -1.0, 2)
    assert row["Alpha_vs_BH%"] == 1.5  # 2.0 - 0.5
    assert row["25th_Pct%"] == round(vals.quantile(0.25), 2)
    assert row["75th_Pct%"] == round(vals.quantile(0.75), 2)
    assert row["Max_Drawdown%"] == round(
        (
            ((1 + vals.sort_values().reset_index(drop=True) / 100).cumprod())
            / (1 + vals.sort_values().reset_index(drop=True) / 100).cumprod().cummax()
            - 1
        ).min()
        * 100,
        2,
    )


def test_analyze_signals_regimes_and_minimums(BS):
    st = BS.analyze_signals(signals_frame())
    assert set(st[st.Screener == "darvas"].Regime) == {"BULL", "BEAR", "ALL"}  # no SIDEWAYS rows
    allr = st[(st.Screener == "darvas") & (st.Regime == "ALL") & (st.Horizon == "T+1d")].iloc[0]
    assert allr["N_Signals"] == 7
    assert "tiny" not in set(st.Screener)  # needs >= 3 non-null returns
    assert BS.analyze_signals(pd.DataFrame()).empty


def test_analyze_signals_stat_flag_threshold_and_infinite_profit_factor(BS):
    n = BS.MIN_SIGNAL_THRESHOLD
    rows = [{"screener": "s", "regime": "BULL", "T+1d": float(i + 1)} for i in range(n)]
    st = BS.analyze_signals(pd.DataFrame(rows))
    r = st[(st.Regime == "BULL")].iloc[0]
    assert r["N_Signals"] == n and r["Stat_Flag"] == "OK"
    assert r["Profit_Factor"] == 999 or r["Profit_Factor"] > 1e6  # no losses
    assert r["Alpha_vs_BH%"] != r["Alpha_vs_BH%"]  # NaN: no benchmark column
    short = BS.analyze_signals(pd.DataFrame(rows[:-1]))
    assert short[short.Regime == "BULL"].iloc[0]["Stat_Flag"] == "⚠️ LOW N"


def test_heatmap_and_ranking(BS):
    st = BS.analyze_signals(signals_frame())
    hm = BS.build_heatmap(st, "Hit_Rate%")
    assert list(hm.columns) == ["T+1d", "T+1mo"]  # canonical horizon order
    assert hm.loc[("darvas", "BULL"), "T+1d"] == 75.0
    assert BS.build_heatmap(pd.DataFrame()).empty
    rk = BS.rank_screeners(st)
    r = rk[(rk.Screener == "darvas") & (rk.Regime == "BULL")].iloc[0]
    sub = st[(st.Screener == "darvas") & (st.Regime == "BULL")]
    one_mo = sub[sub.Horizon == "T+1mo"].iloc[0]
    assert r["Avg_Hit_Rate%"] == round(sub["Hit_Rate%"].mean(), 1)
    assert r["Sharpe_1mo"] == round(one_mo["Sharpe"], 3)
    assert r["Composite_Score"] == round(sub["Hit_Rate%"].mean() * one_mo["Sharpe"], 2)
    assert BS.rank_screeners(pd.DataFrame()).empty
    # sorted by Regime then descending score
    scores = rk.groupby("Regime")["Composite_Score"].apply(list)
    assert all(s == sorted(s, reverse=True) for s in scores)


def test_rank_without_one_month_row_scores_zero(BS):
    st = BS.analyze_signals(signals_frame())
    st = st[st.Horizon != "T+1mo"]
    rk = BS.rank_screeners(st)
    assert (rk["Sharpe_1mo"] == 0).all() and (rk["Composite_Score"] == 0).all()


def test_save_excel_sheets(BS):
    sig = signals_frame().assign(signal_date=pd.date_range("2024-01-01", periods=9))
    st = BS.analyze_signals(sig)
    path = BS.save_backtest_excel(
        sig,
        st,
        BS.rank_screeners(st),
        BS.build_heatmap(st, "Hit_Rate%"),
        BS.build_heatmap(st, "Avg_Return%"),
        "US",
    )
    assert path.parent == BS.DOWNLOAD_DIR and path.name.startswith("backtest_US_")
    sheets = pd.read_excel(path, sheet_name=None)
    assert {
        "DISCLAIMER",
        "HitRate_Heatmap",
        "AvgReturn_Heatmap",
        "Screener_Ranking",
        "Full_Stats",
        "darvas",
        "All_Signals",
    } <= set(sheets)
    assert len(sheets["All_Signals"]) == 9


def test_save_excel_with_empty_frames_writes_only_disclaimer(BS):
    e = pd.DataFrame()
    path = BS.save_backtest_excel(e, e, e, e, e)
    assert list(pd.read_excel(path, sheet_name=None)) == ["DISCLAIMER"]
    assert "_IN_" in path.name


# ── index download ────────────────────────────────────────────────────────────
def test_download_index_adds_dma_columns_and_handles_failure(BS, monkeypatch):
    n = 260
    raw = pd.DataFrame(
        {"Close": 100.0 + np.arange(n)}, index=pd.bdate_range("2023-01-02", periods=n)
    )
    seen = {}

    def dl(sym, **kw):
        seen.update(sym=sym, **kw)
        return raw

    monkeypatch.setattr(BS.yf, "download", dl)
    out = BS.download_index("US")
    assert seen["sym"] == "^GSPC" and seen["period"] == "2y"
    assert out["dma200"].iloc[-1] == pytest.approx(raw.Close.tail(200).mean())
    assert out["dma200_sl"].iloc[-1] == pytest.approx(
        out["dma200"].iloc[-1] - out["dma200"].iloc[-6]
    )
    BS.download_index("IN")
    assert seen["sym"] == "^NSEI"
    BS.download_index("XX")
    assert seen["sym"] == "^NSEI"  # unknown market falls back to Nifty
    monkeypatch.setattr(BS.yf, "download", lambda *a, **k: 1 / 0)
    assert BS.download_index("IN").empty


def test_download_index_flattens_multiindex_columns(BS, monkeypatch):
    n = 30
    cols = pd.MultiIndex.from_product([["Close"], ["^NSEI"]])
    raw = pd.DataFrame(
        np.arange(float(n)).reshape(n, 1) + 100,
        columns=cols,
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    monkeypatch.setattr(BS.yf, "download", lambda *a, **k: raw)
    out = BS.download_index("IN")
    assert "Close" in out.columns and len(out) == n


# ── fundamental signals (fake yfinance tickers) ───────────────────────────────
def stmt(rows, ncols=2):
    cols = pd.to_datetime(
        ["2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31", "2020-03-31"][:ncols]
    )
    return pd.DataFrame(
        {c: [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows)
    )


class FakeTicker:
    def __init__(self, inc=None, bal=None, cf=None, info=None, inc_q=None, mcap=None):
        self.income_stmt, self.balance_sheet, self.cash_flow = inc, bal, cf
        self.quarterly_income_stmt = inc_q
        self._info = info if info is not None else {}
        self._mcap = mcap

    @property
    def info(self):
        return self._info

    @property
    def fast_info(self):
        if self._mcap is None:
            raise RuntimeError("no fast info")
        return types.SimpleNamespace(market_cap=self._mcap)


def ohlc100():
    return pd.DataFrame(
        {"Close": 100.0 + np.arange(400.0)}, index=pd.bdate_range("2023-01-02", periods=400)
    )


def use_ticker(BS, monkeypatch, **kw):
    monkeypatch.setattr(BS.yf, "Ticker", lambda s: FakeTicker(**kw))


def test_fundamental_signal_date_and_alignment(BS, monkeypatch):
    inc = stmt({"Net Income": [1, 1]})
    t = FakeTicker(inc=inc)
    assert BS.get_fundamental_signal_date(t, "annual") == pd.Timestamp("2024-03-31")
    assert BS.get_fundamental_signal_date(FakeTicker(inc_q=inc), "quarterly") == pd.Timestamp(
        "2024-03-31"
    )
    assert BS.get_fundamental_signal_date(FakeTicker(), "annual") is None


def piotroski_frames(good=True):
    if good:
        inc = stmt(
            {"Net Income": [120, 100], "Total Revenue": [1000, 900], "Gross Profit": [500, 400]}
        )
        bal = stmt(
            {
                "Total Assets": [1000, 1000],
                "Long Term Debt": [100, 200],
                "Current Assets": [500, 400],
                "Current Liabilities": [300, 300],
                "Share Issued": [10, 10],
            }
        )
        cf = stmt({"Operating Cash Flow": [200, 100]})
    else:
        inc = stmt(
            {"Net Income": [-5, 10], "Total Revenue": [1000, 900], "Gross Profit": [300, 400]}
        )
        bal = stmt(
            {
                "Total Assets": [1000, 1000],
                "Long Term Debt": [300, 200],
                "Current Assets": [300, 400],
                "Current Liabilities": [300, 300],
                "Share Issued": [10, 10],
            }
        )
        cf = stmt({"Operating Cash Flow": [-10, 5]})
    return inc, bal, cf


def test_piotroski_signal_qualifies_at_seven_points(BS, monkeypatch):
    inc, bal, cf = piotroski_frames()
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf, info={"sector": "Technology"})
    sig = BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100())
    assert sig is not None and sig["signal_idx"] <= 399
    ohlc = ohlc100()
    assert sig["entry_price"] == pytest.approx(ohlc.Close.iloc[sig["signal_idx"]])
    expect = ohlc.index.searchsorted(pd.Timestamp("2024-03-31"), side="right")
    assert sig["signal_idx"] == min(expect, 399)


def test_piotroski_weak_company_and_financials_are_rejected(BS, monkeypatch):
    inc, bal, cf = piotroski_frames(good=False)
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf)
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is None
    inc, bal, cf = piotroski_frames()
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf, info={"sector": "Financial Services"})
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is None
    assert BS.fetch_fundamental_signal("HDFCBANK", ".NS", "piotroski", ohlc100()) is None


def test_fundamental_signal_needs_ohlc_and_a_statement(BS, monkeypatch):
    inc, bal, cf = piotroski_frames()
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf)
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", None) is None
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", pd.DataFrame()) is None
    use_ticker(BS, monkeypatch)  # no statements at all
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is None
    monkeypatch.setattr(BS.yf, "Ticker", lambda s: 1 / 0)
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is None


def test_signal_date_beyond_the_ohlc_is_clamped_to_the_last_bar(BS, monkeypatch):
    inc, bal, cf = piotroski_frames()
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf)
    short = ohlc100().iloc[:50]
    short.index = pd.bdate_range("2020-01-01", periods=50)  # all before the report date
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", short)["signal_idx"] == 49


def test_coffee_can_thresholds(BS, monkeypatch):
    inc = stmt(
        {
            "Total Revenue": [2000, 1500, 1000],
            "EBIT": [400, 300, 200],
            "Net Income": [300, 200, 100],
        },
        ncols=3,
    )
    bal = stmt(
        {"Total Assets": [1500, 1200, 1000], "Current Liabilities": [300, 250, 200]}, ncols=3
    )
    info = {"debtToEquity": 50}
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info=info, mcap=1e10)  # 1000 Cr
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is not None
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info={"debtToEquity": 150}, mcap=1e10)
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is None  # D/E 1.5
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info={"debtToEquity": 0.5}, mcap=1e10)
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is not None
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info=info, mcap=4.9e9)  # 490 Cr
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is None
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info={}, mcap=1e10)
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is None  # no D/E
    loss = inc.copy()
    loss.loc["Net Income", loss.columns[-1]] = -1
    use_ticker(BS, monkeypatch, inc=loss, bal=bal, info=info, mcap=1e10)
    assert BS.fetch_fundamental_signal("X", ".NS", "coffee_can", ohlc100()) is None  # a loss year


def test_magic_formula_thresholds(BS, monkeypatch):
    inc = stmt({"EBIT": [200, 100], "Net Income": [100, 50]})
    bal = stmt({"Total Assets": [1000, 900], "Current Liabilities": [200, 200]})
    info = {"totalDebt": 100, "totalCash": 0, "bookValue": 10}
    # ROIC = 200/800 = 25%; EV = 1000e7+100 ~ mcap => EY = 200/1e? scale: use mcap in same units as EBIT
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info=info, mcap=1500.0)
    sig = BS.fetch_fundamental_signal("X", ".NS", "magic_formula", ohlc100())
    assert sig is None  # mcap < 100 Cr
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, info=info, mcap=2e9)  # 200 Cr -> EY ~ 1e-7%
    assert BS.fetch_fundamental_signal("X", ".NS", "magic_formula", ohlc100()) is None
    big_ebit = stmt({"EBIT": [4e8, 1e8], "Net Income": [1e8, 5e7]})
    big_bal = stmt({"Total Assets": [2e9, 2e9], "Current Liabilities": [4e8, 4e8]})
    use_ticker(BS, monkeypatch, inc=big_ebit, bal=big_bal, info=info, mcap=3e9)  # ROIC 25%, EY 13%
    assert BS.fetch_fundamental_signal("X", ".NS", "magic_formula", ohlc100()) is not None
    use_ticker(BS, monkeypatch, inc=big_ebit, bal=big_bal, info={**info, "bookValue": -1}, mcap=3e9)
    assert BS.fetch_fundamental_signal("X", ".NS", "magic_formula", ohlc100()) is None


def test_bull_cartel_needs_five_quarters_and_growth(BS, monkeypatch):
    q = stmt(
        {
            "Total Revenue": [1500e7, 1400e7, 1300e7, 1200e7, 1000e7],
            "Net Income": [30e7, 25e7, 22e7, 21e7, 20e7],
        },
        ncols=5,
    )
    use_ticker(BS, monkeypatch, inc=stmt({"Net Income": [1, 1]}), inc_q=q)
    assert BS.fetch_fundamental_signal("X", ".NS", "bull_cartel", ohlc100()) is not None
    flat = q.copy()
    flat.loc["Net Income", flat.columns[0]] = 22e7  # profit growth only 10%
    use_ticker(BS, monkeypatch, inc=stmt({"Net Income": [1, 1]}), inc_q=flat)
    assert BS.fetch_fundamental_signal("X", ".NS", "bull_cartel", ohlc100()) is None
    four = q.iloc[:, :4]
    use_ticker(BS, monkeypatch, inc=stmt({"Net Income": [1, 1]}), inc_q=four)
    assert BS.fetch_fundamental_signal("X", ".NS", "bull_cartel", ohlc100()) is None


# ── symbol universes ──────────────────────────────────────────────────────────
def test_indian_symbols_fallback_and_top(BS, monkeypatch):
    monkeypatch.setitem(sys.modules, "nse", None)  # ImportError -> Nifty-50 fallback
    syms = BS.get_indian_symbols()
    assert len(syms) == 50 and syms[0] == ("ADANIENT", ".NS") and ("RELIANCE", ".NS") in syms
    assert BS.get_indian_symbols(top=3) == syms[:3]


def test_indian_symbols_from_bhavcopy_dataframe(BS, monkeypatch):
    class NSE:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def equityBhavcopy(self, d):
            return pd.DataFrame({"SERIES": ["EQ", "BE", "EQ"], "SYMBOL": [" ZED ", "BAD", "ABC"]})

    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=NSE))
    assert BS.get_indian_symbols() == [("ZED", ".NS"), ("ABC", ".NS")]


def test_indian_symbols_from_bhavcopy_csv(BS, monkeypatch, tmp_path):
    csv = tmp_path / "b.csv"
    pd.DataFrame({"SctySrs": ["EQ", "N1"], "TckrSymb": ["TCS", "JUNK"]}).to_csv(csv, index=False)

    class NSE:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def equityBhavcopy(self, d):
            return csv

    monkeypatch.setitem(sys.modules, "nse", types.SimpleNamespace(NSE=NSE))
    assert BS.get_indian_symbols() == [("TCS", ".NS")]


def test_us_symbols_nasdaq_files_filtering_and_dedupe(BS, monkeypatch):
    nasdaq = (
        "Symbol|Security Name|Test Issue|ETF\nAAPL|Apple|N|N\nTESTX|t|Y|N\nQQQ|etf|N|Y\n"
        "BRK.B|dot|N|N\nTOOLONG|x|N|N\nMSFT|ms|N|N\nFile Creation Time: x|||\n"
    )
    other = "ACT Symbol|Name|Test Issue|ETF\nAAPL|dup|N|N\nIBM|ibm|N|N\n"
    pages = {"nasdaqlisted": nasdaq, "otherlisted": other}

    class R:
        def __init__(self, t):
            self.text = t

        def raise_for_status(self):
            pass

    import requests

    monkeypatch.setattr(
        requests, "get", lambda url, **k: R(next(v for k, v in pages.items() if k in url))
    )
    syms = BS.get_us_symbols()
    assert syms == [("AAPL", ""), ("MSFT", ""), ("IBM", "")]
    assert BS.get_us_symbols(top=2) == [("AAPL", ""), ("MSFT", "")]


def test_us_symbols_sec_fallback(BS, monkeypatch):
    import requests

    class R:
        def __init__(self, data=None, text=""):
            self._d, self.text = data, text

        def raise_for_status(self):
            pass

        def json(self):
            return self._d

    def get(url, **k):
        if "sec.gov" in url:
            return R(
                {
                    "data": [
                        [1, "x", "aapl", "Nasdaq"],
                        [2, "y", "BRK.B", "NYSE"],
                        [3, "z", "TOOLONG", "NYSE"],
                        [4, "w", "XYZ", "OTC"],
                        [5, "v", "ibm", "NYSE"],
                    ]
                }
            )
        raise OSError("ftp down")

    monkeypatch.setattr(requests, "get", get)
    assert BS.get_us_symbols() == [("AAPL", ""), ("IBM", "")]
    monkeypatch.setattr(requests, "get", lambda *a, **k: 1 / 0)
    assert BS.get_us_symbols() == []


# ── mutation follow-ups ───────────────────────────────────────────────────────
def test_index_returns_horizon_landing_exactly_past_the_end_is_nan_only_for_that_horizon(BS):
    idx = index_frame(n=30, close=100.0 + np.arange(30.0))
    out = BS.compute_index_returns(idx, idx.index[9])  # 9 + 21 == 30 == len(index)
    assert out["T+1d"] == pytest.approx(1.0 / 109.0 * 100)
    assert np.isnan(out["T+1mo"]) and np.isnan(out["T+3mo"])
    last_ok = BS.compute_index_returns(idx, idx.index[8])  # 8 + 21 == 29, the last bar
    assert last_ok["T+1mo"] == pytest.approx(21.0 / 108.0 * 100)


def test_us_symbols_sec_fallback_skips_short_rows(BS, monkeypatch):
    import requests

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [[1, "x"], [2, "y", "msft"], [3, "z", "amd"], [4, "w", "ko", "NYSE"]]}

    def get(url, **k):
        if "sec.gov" in url:
            return R()
        raise OSError("ftp down")

    monkeypatch.setattr(requests, "get", get)
    # rows without an exchange column never qualify, and a 2-item row must not abort the loop
    assert BS.get_us_symbols() == [("KO", "")]


def test_piotroski_current_ratio_point_decides_a_seven_point_company(BS, monkeypatch):
    inc, bal, cf = piotroski_frames()
    inc = stmt({"Net Income": [120, 100], "Total Revenue": [1000, 900], "Gross Profit": [400, 400]})
    bal = stmt(
        {
            "Total Assets": [1000, 1000],
            "Long Term Debt": [100, 200],
            "Current Assets": [500, 400],
            "Current Liabilities": [300, 300],
            "Share Issued": [11, 10],  # dilution costs a point; flat gross margin costs another
        }
    )
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf, info={"sector": "Technology"})
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is not None
    weaker = bal.copy()
    weaker.loc["Current Assets", weaker.columns[0]] = 300  # current ratio no longer improving
    use_ticker(BS, monkeypatch, inc=inc, bal=weaker, cf=cf, info={"sector": "Technology"})
    assert BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", ohlc100()) is None


def test_bull_cartel_compares_latest_quarter_with_the_year_ago_quarter(BS, monkeypatch):
    q = stmt(
        {
            "Total Revenue": [
                1200e7,
                1000e7,
                1000e7,
                1000e7,
                1000e7,
            ],  # only the latest quarter grew
            "Net Income": [30e7, 25e7, 22e7, 21e7, 20e7],
        },
        ncols=5,
    )
    use_ticker(BS, monkeypatch, inc=stmt({"Net Income": [1, 1]}), inc_q=q)
    assert BS.fetch_fundamental_signal("X", ".NS", "bull_cartel", ohlc100()) is not None


def test_signal_dated_before_the_first_bar_enters_on_the_first_bar(BS, monkeypatch):
    inc, bal, cf = piotroski_frames()
    use_ticker(BS, monkeypatch, inc=inc, bal=bal, cf=cf, info={"sector": "Technology"})
    late = ohlc100()
    late.index = late.index + pd.Timedelta(days=2000)  # first bar is years after the report date
    sig = BS.fetch_fundamental_signal("TCS", ".NS", "piotroski", late)
    assert sig is not None and sig["signal_idx"] == 0
    assert sig["entry_price"] == pytest.approx(late.Close.iloc[0])


def test_worker_pool_default_is_eight(BS):
    assert BS.MAX_WORKERS == 8
