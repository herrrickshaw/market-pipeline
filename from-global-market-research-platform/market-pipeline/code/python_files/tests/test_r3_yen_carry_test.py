"""Offline tests for yen_carry_test (loader, episodes, event/asymmetry/carry/growth sections, CLI, fetch)."""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def yc(monkeypatch, tmp_path):
    sys.modules.pop("yen_carry_test", None)
    mod = importlib.import_module("yen_carry_test")
    cache = tmp_path / "cache_seed"
    cache.mkdir()
    monkeypatch.setattr(mod, "CACHE", cache)
    monkeypatch.setattr(mod, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(mod, "FX", cache / "usdjpy.parquet")
    monkeypatch.setattr(mod, "NKY", cache / "nikkei.parquet")
    monkeypatch.setattr(mod, "JPR", cache / "jp_rate_3m.parquet")
    monkeypatch.setattr(mod, "SPX", cache / "spx.parquet")
    monkeypatch.setattr(mod, "YIELDS", cache / "treasury_yields.parquet")
    monkeypatch.setattr(mod, "GROWTH", cache / "growth_controls.parquet")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "reports" / "yen_carry_linkage.md")
    monkeypatch.setattr(mod, "CREDS", tmp_path / "no_such_credentials.env")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    yield mod
    sys.modules.pop("yen_carry_test", None)


def _synth(start="2019-01-01", end="2025-06-30", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    usdjpy = 110 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    nikkei = 20000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    spx = 3000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return idx, usdjpy, nikkei, spx


def _write_prices(yc, idx, usdjpy, nikkei, spx):
    pd.DataFrame({"Date": idx, "usdjpy": usdjpy}).to_parquet(yc.FX, index=False)
    pd.DataFrame({"Date": idx, "nikkei": nikkei}).to_parquet(yc.NKY, index=False)
    pd.DataFrame({"Date": idx, "spx": spx}).to_parquet(yc.SPX, index=False)


def _d(n=500, seed=1, spikes=()):
    """Frame with the columns the sections need: jpy, spx_ret, nky_ret on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    d = pd.DataFrame(
        {
            "jpy": rng.normal(0, 0.003, n),
            "spx_ret": rng.normal(0.0004, 0.01, n),
            "nky_ret": rng.normal(0.0003, 0.012, n),
        },
        index=idx,
    )
    for pos, size in spikes:
        d.iloc[pos : pos + 5, d.columns.get_loc("jpy")] = size
    return d


# ------------------------------------------------------------------ _fred_key
def test_fred_key_env_has_priority_and_file_parsing(yc, monkeypatch, tmp_path):
    assert yc._fred_key() is None  # no env, creds file does not exist
    creds = tmp_path / "creds.env"
    monkeypatch.setattr(yc, "CREDS", creds)
    creds.write_text("OTHER=1\nFRED_API_KEY=\"abc123\"\n")
    assert yc._fred_key() == "abc123"
    creds.write_text("FRED_API_KEY='quoted'\n")
    assert yc._fred_key() == "quoted"
    creds.write_text("FRED_API_KEY=   \n")
    assert yc._fred_key() is None  # blank value
    creds.write_text("# FRED_API_KEY=commented\nNOPE=1\n")
    assert yc._fred_key() is None
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    assert yc._fred_key() == "from-env"  # environment wins over the file


# ------------------------------------------------------------------ load
def test_load_sign_convention_and_alignment(yc):
    idx, usdjpy, nikkei, spx = _synth(end="2019-03-29")
    _write_prices(yc, idx, usdjpy, nikkei, spx)
    d = yc.load()
    assert list(d.columns) == ["usdjpy", "nikkei", "spx", "jpy", "spx_ret", "nky_ret"]
    # yen move is minus the log change of USD/JPY (appreciation positive)
    assert d["jpy"].iloc[0] == pytest.approx(-np.log(usdjpy[1] / usdjpy[0]))
    assert d["spx_ret"].iloc[0] == pytest.approx(np.log(spx[1] / spx[0]))
    assert d["nky_ret"].iloc[0] == pytest.approx(np.log(nikkei[1] / nikkei[0]))
    assert len(d) == len(idx) - 1 and d.index.is_monotonic_increasing


def test_load_ffills_at_most_three_days_and_drops_missing_fx(yc):
    idx, usdjpy, nikkei, spx = _synth(end="2019-03-29")
    nik = pd.Series(nikkei, index=idx)
    # Nikkei holiday gap of 3 days (filled) and 4 days (the 4th row stays NaN and is dropped)
    nik.iloc[10:13] = np.nan
    nik.iloc[30:34] = np.nan
    _write_prices(yc, idx, usdjpy, nik.values, spx)
    d = yc.load()
    assert idx[10] in d.index and idx[11] in d.index and idx[12] in d.index
    assert idx[33] not in d.index and idx[30] in d.index
    assert d.loc[idx[11], "nikkei"] == nik.iloc[9]  # carried forward
    # a missing USD/JPY row removes the date altogether
    fx = pd.Series(usdjpy, index=idx)
    fx.iloc[20] = np.nan
    _write_prices(yc, idx, fx.values, nikkei, spx)
    assert idx[20] not in yc.load().index


def test_fwd_is_forward_looking_sum(yc):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    f = yc._fwd(s, 2)
    assert list(f.iloc[:3]) == [5.0, 7.0, 9.0]  # sum of the NEXT two
    assert f.iloc[3:].isna().all()


# ------------------------------------------------------------------ _episodes
def test_episodes_cluster_onset_and_peak(yc):
    d = _d(spikes=[(100, 0.02), (300, 0.03)])
    d["jpy"] = np.where(np.arange(len(d)) % 1 == 0, d["jpy"], 0)
    d.loc[:, "jpy"] = d["jpy"] * 0  # silence noise: only the spikes remain
    d.iloc[100:105, d.columns.get_loc("jpy")] = 0.02
    d.iloc[300:305, d.columns.get_loc("jpy")] = 0.03
    thr, onset, peak, eps = yc._episodes(d)
    assert thr == 0  # 95th percentile of a mostly-zero 5d sum
    assert len(eps) == 2
    assert onset == [d.index[100], d.index[300]]  # first day the 5d sum exceeds the threshold
    assert peak == [d.index[104], d.index[304]]  # day the 5d sum is largest
    assert eps[0][-1] == d.index[108]  # 5d sum stays positive until the spike rolls out


def test_episodes_gap_days_boundary(yc):
    d = _d()
    d["jpy"] = 0.0
    d.iloc[100:105, d.columns.get_loc("jpy")] = 0.02
    d.iloc[130:135, d.columns.get_loc("jpy")] = 0.02
    # crossings: 100..108 and 130..138 -> calendar gap between the last of one and first of the next
    gap = (d.index[130] - d.index[108]).days
    _, _, _, eps = yc._episodes(d, gap_days=gap)
    assert len(eps) == 1  # a gap of exactly gap_days is NOT split
    _, _, _, eps = yc._episodes(d, gap_days=gap - 1)
    assert len(eps) == 2
    assert (d.index[130] - d.index[108]).days == 30  # so the default 30-day gap also merges them
    assert len(yc._episodes(d)[3]) == 1


def test_episodes_peak_differs_from_onset(yc):
    d = _d()
    d["jpy"] = 0.0
    d.iloc[100:103, d.columns.get_loc("jpy")] = 0.01
    d.iloc[103:106, d.columns.get_loc("jpy")] = 0.04
    thr, onset, peak, eps = yc._episodes(d)
    assert len(eps) == 1
    w5 = d["jpy"].rolling(5).sum()
    assert peak[0] == w5.loc[eps[0]].idxmax() and peak[0] > onset[0]


def test_episodes_none_when_flat(yc):
    d = _d()
    d["jpy"] = 0.001  # constant: w5 never exceeds its own 95th percentile
    thr, onset, peak, eps = yc._episodes(d)
    assert eps == [] and onset == [] and peak == []


# ------------------------------------------------------------------ run_validate
def _aug24_frame(spike=True):
    idx = pd.bdate_range("2024-01-02", "2024-12-31")
    rng = np.random.default_rng(3)
    d = pd.DataFrame(
        {"jpy": rng.normal(0, 0.002, len(idx)),
         "spx_ret": rng.normal(0, 0.008, len(idx)),
         "nky_ret": rng.normal(0, 0.01, len(idx))},
        index=idx,
    )
    if spike:
        d.loc["2024-07-29":"2024-08-06", "jpy"] = 0.012
        d.loc["2024-08-05", "nky_ret"] = -0.13
        d.loc["2024-08-05", "spx_ret"] = -0.03
    return d


def test_validate_flags_august_2024(yc):
    d = _aug24_frame()
    thr, onset, peak, eps = yc._episodes(d)
    out = []
    yc.run_validate(d, out, eps)
    text = "\n".join(out)
    w = d.loc["2024-07-29":"2024-08-06"]
    assert f"**{w['jpy'].sum()*100:+.2f}%**" in text
    assert "5 Aug 2024 alone: Nikkei **-13.00%**" in text and "S&P -3.00%" in text
    assert "**flagged: YES**" in text and "the 5 Aug crossing was" in text
    assert "flagged: NO" not in text


def test_validate_flags_no_and_missing_window(yc):
    d = _aug24_frame(spike=False)
    out = []
    yc.run_validate(d, out, [])  # no episodes at all
    text = "\n".join(out)
    assert "flagged: NO" in text and "unvalidated" in text
    # episodes elsewhere do not cover 5 Aug 2024
    away = [[pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")]]
    out = []
    yc.run_validate(d, out, away)
    assert "flagged: NO" in "\n".join(out)
    # data that does not reach 2024: no window lines, still a verdict
    old = _d()
    out = []
    yc.run_validate(old, out, [])
    text = "\n".join(out)
    assert "yen move 29 Jul" not in text and "flagged: NO" in text
    # window present but 5 Aug itself missing: no single-day line
    d2 = _aug24_frame().drop(index=pd.Timestamp("2024-08-05"))
    out = []
    yc.run_validate(d2, out, [])
    assert "5 Aug 2024 alone" not in "\n".join(out)


# ------------------------------------------------------------------ run_events
def test_run_events_tables_and_numbers(yc):
    d = _d(n=600, spikes=[(150, 0.02), (350, 0.03), (500, 0.025)])
    d.loc[d.index[150:505], "jpy"] = d.loc[d.index[150:505], "jpy"]  # keep noise
    out = []
    onset, peak, eps = yc.run_events(d, out)
    text = "\n".join(out)
    assert len(eps) >= 3 and len(onset) == len(peak) == len(eps)
    assert f"**{len(eps)} episodes**" in text
    assert "**onset-dated**" in text and "**peak-dated**" in text
    assert text.count("| horizon | mkt |") == 2
    for h in (5, 21, 63):
        assert text.count(f"| {h}d | S&P 500 |") == 2 and text.count(f"| {h}d | Nikkei |") == 2
    # first table row numbers match an independent computation
    f5 = d["spx_ret"].rolling(5).sum().shift(-5)
    e = f5.reindex(onset).dropna()
    un = f5.dropna()
    assert f"| 5d | S&P 500 | {e.mean()*100:+.2f}% | {e.median()*100:+.2f}% | {(e > 0).mean()*100:.0f}% |" in text
    assert f"| {un.mean()*100:+.2f}% | **{(e.mean()-un.mean())*100:+.2f}pp** |" in text
    # contemporaneous table: one worst-episode column per market
    sums = [d.loc[ep[0]:ep[-1], "spx_ret"].sum() for ep in eps]
    assert f"| S&P 500 | {np.mean(sums)*100:+.2f}% | {np.median(sums)*100:+.2f}% | {min(sums)*100:+.2f}% |" in text


def test_run_events_skips_empty_forward_windows(yc):
    d = _d(n=100)
    d["jpy"] = 0.0
    d.iloc[95:100, d.columns.get_loc("jpy")] = 0.05  # episode in the last bars: no forward data
    out = []
    onset, peak, eps = yc.run_events(d, out)
    text = "\n".join(out)
    assert len(eps) == 1
    assert "| 5d | S&P 500 |" not in text and "| 63d | Nikkei |" not in text


# ------------------------------------------------------------------ run_asymmetry
def test_run_asymmetry_deciles_and_spread(yc):
    d = _d(n=700)
    out = []
    yc.run_asymmetry(d, out)
    text = "\n".join(out)
    assert text.count("| D") == 10
    assert "D1 ← strongest yen DEPRECIATION" in text and "D10 ← strongest yen APPRECIATION" in text
    w5 = d["jpy"].rolling(5).sum()
    t = pd.DataFrame({"y": w5, "s": d["spx_ret"].rolling(5).sum().shift(-5)}).dropna()
    assert f"{t['y'].corr(t['s']):+.3f}" in text
    # spread line is hi - lo of the first/last deciles
    dec = pd.qcut(t["y"], 10, labels=False) + 1
    g = t.groupby(dec)["s"].mean()
    assert f"**spread {(g.loc[10]-g.loc[1])*100:+.2f}pp**" in text


# ------------------------------------------------------------------ run_carry
def _write_rates(yc, d, jp_level=0.1, us_level=3.0, slope=0.0):
    idx = d.index
    pd.DataFrame({"Date": idx, "jp3m": jp_level + np.arange(len(idx)) * 0.0}).to_parquet(yc.JPR, index=False)
    us = us_level + slope * np.arange(len(idx)) + np.sin(np.arange(len(idx)) / 30.0)
    pd.DataFrame({"Date": idx, "3 Mo": us}).to_parquet(yc.YIELDS, index=False)


def test_run_carry_skipped_without_jp_rate(yc):
    out = []
    yc.run_carry(_d(), out)
    assert "JP rate unavailable — skipped." in "\n".join(out)


def test_run_carry_quartiles_and_compression(yc):
    d = _d(n=800)
    _write_rates(yc, d)
    out = []
    yc.run_carry(d, out)
    text = "\n".join(out)
    jp = 0.1
    us = pd.Series(3.0 + np.sin(np.arange(len(d)) / 30.0), index=d.index)
    diff = us - jp
    assert f"median **{diff.median():.2f}pp**" in text
    assert f"range {diff.min():.2f} .. {diff.max():.2f}pp" in text
    for q in ("Q1 narrowest", "Q2", "Q3", "Q4 widest"):
        assert f"| {q} |" in text
    assert "corr(carry differential, forward 1y S&P)" in text
    comp = diff.diff(63)
    assert f"<{comp.quantile(0.10):.2f}pp/63d" in text


# ------------------------------------------------------------------ run_growth
def _write_growth(yc, d, future=False):
    idx = d.index
    n = len(idx)
    rng = np.random.default_rng(7)
    cut = int(n * 0.6)  # growth is stable for the first 60% of the sample, then deteriorates
    claims = 220000 + rng.normal(0, 3000, n)
    claims[cut:] *= 1.4
    unrate = 4.0 + rng.normal(0, 0.02, n)
    unrate[cut:] += np.linspace(0, 1.5, n - cut)
    dates = idx + pd.DateOffset(years=20) if future else idx
    pd.DataFrame(
        {"Date": dates, "ICSA": claims, "UNRATE": unrate, "NFCI": rng.normal(-0.3, 0.2, n)}
    ).to_parquet(yc.GROWTH, index=False)


def test_run_growth_skips_without_inputs(yc):
    out = []
    yc.run_growth(_d(), out)
    assert "Growth controls or JP rate unavailable — skipped." in "\n".join(out)
    d = _d(n=800)
    _write_rates(yc, d)  # JP rate but no growth file
    out = []
    yc.run_growth(d, out)
    assert "unavailable — skipped" in "\n".join(out)


def test_run_growth_no_overlap_is_skipped(yc):
    d = _d(n=800)
    _write_rates(yc, d)
    _write_growth(yc, d, future=True)  # controls only start long after the price sample ends
    out = []
    yc.run_growth(d, out)
    assert "No overlapping sample — skipped." in "\n".join(out)


@pytest.mark.parametrize("seed,verdict", [(1, "does NOT survive"), (2, "SURVIVES")])
def test_run_growth_double_sort_and_regression(yc, seed, verdict):
    d = _d(n=900, seed=seed)
    _write_rates(yc, d, slope=-0.002)
    _write_growth(yc, d)
    out = []
    yc.run_growth(d, out)
    text = "\n".join(out)
    assert "### Double sort" in text and "### Regression" in text
    for gs in ("stable", "deteriorating"):
        assert f"| {gs} | no compression |" in text and f"| {gs} | COMPRESSING |" in text
    for spec in ("compression only", "growth only", "compression + growth"):
        assert f"| {spec} |" in text
    assert "**NBER's recession flag is deliberately excluded**" in text
    # exactly one verdict branch is printed (which one depends on the sign/size of the stable-growth effect)
    other = "SURVIVES" if verdict != "SURVIVES" else "does NOT survive"
    assert verdict in text and other not in text
    assert "overlapping days" in text
    # coefficient column: present for compression specs, em dash for the growth-only spec
    growth_row = next(line for line in out if line.startswith("| growth only |"))
    assert growth_row.rstrip().endswith("| — |")


# ------------------------------------------------------------------ run_synthesis
def test_run_synthesis_reports_sample_size(yc):
    d = _d(n=1000)
    out = []
    yc.run_synthesis(d, out)
    text = "\n".join(out)
    yrs = (d.index.max() - d.index.min()).days / 365.25
    assert f"{len(d):,} sessions over {yrs:.0f} years" in text
    assert "roughly **24 independent years**" in text
    assert text.count("**") > 10 and "My convexity hypothesis was WRONG" in text


# ------------------------------------------------------------------ main
def test_main_prints_help_without_flags(yc, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["yen_carry_test.py"])
    assert yc.main() == 1
    assert "--fetch" in capsys.readouterr().out


def test_main_requires_fetched_cache(yc, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["yen_carry_test.py", "--events"])
    assert yc.main() == 1
    assert "run --fetch first" in capsys.readouterr().out


def test_main_fetch_only_returns_zero_without_analysis(yc, monkeypatch):
    calls = []
    monkeypatch.setattr(yc, "fetch", lambda: calls.append(1) or 0)
    monkeypatch.setattr(yc, "load", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    monkeypatch.setattr(sys, "argv", ["yen_carry_test.py", "--fetch"])
    assert yc.main() == 0 and calls == [1]


def _full_cache(yc):
    idx, usdjpy, nikkei, spx = _synth(start="2018-01-01", end="2025-12-31")
    _write_prices(yc, idx, usdjpy, nikkei, spx)
    d = yc.load()
    _write_rates(yc, d)
    _write_growth(yc, d)
    return d


def test_main_all_writes_report(yc, monkeypatch, capsys):
    d = _full_cache(yc)
    monkeypatch.setattr(sys, "argv", ["yen_carry_test.py", "--all"])
    assert yc.main() == 0
    text = yc.REPORT.read_text()
    assert text.startswith("# The yen carry channel")
    for h in ("## 1. Sharp yen appreciation", "## 0. Validation", "## 2. Convexity", "## 3. The real funding",
              "## 5. Growth control", "## 4. What this establishes"):
        assert h in text
    assert f"({len(d):,} aligned sessions)" in text
    assert text.rstrip().endswith("common causation.*")
    assert f"-> {yc.REPORT}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flag,present,absent",
    [
        ("--events", ["## 1. Sharp", "## 0. Validation"], ["## 2. Convexity", "## 3.", "## 5.", "## 4."]),
        ("--asymmetry", ["## 2. Convexity"], ["## 1. Sharp", "## 3.", "## 5.", "## 4."]),
        ("--carry", ["## 3. The real funding"], ["## 1. Sharp", "## 2.", "## 5.", "## 4."]),
        ("--growth", ["## 5. Growth control"], ["## 1. Sharp", "## 2.", "## 3.", "## 4."]),
    ],
)
def test_main_single_sections(yc, monkeypatch, flag, present, absent):
    _full_cache(yc)
    monkeypatch.setattr(sys, "argv", ["yen_carry_test.py", flag])
    assert yc.main() == 0
    text = yc.REPORT.read_text()
    assert all(p in text for p in present) and not any(a in text for a in absent)


# ------------------------------------------------------------------ fetch
class _FakeResp:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise RuntimeError("500")

    def json(self):
        return self.payload


def _fake_yf(monkeypatch, multiindex=False):
    calls = []

    def download(tkr, start=None, progress=None, auto_adjust=None):
        calls.append((tkr, start, progress, auto_adjust))
        idx = pd.date_range("2024-01-01", periods=3, name="Date")
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [0, 0, 0]}, index=idx)
        if multiindex:
            df.columns = pd.MultiIndex.from_product([df.columns, [tkr]])
        return df

    m = types.ModuleType("yfinance")
    m.download = download
    monkeypatch.setitem(sys.modules, "yfinance", m)
    return calls


def test_fetch_prices_without_fred_key(yc, monkeypatch, capsys):
    calls = _fake_yf(monkeypatch)
    assert yc.fetch() == 0
    assert [c[0] for c in calls] == ["USDJPY=X", "^N225"]
    assert all(c[1] == "1990-01-01" and c[2] is False and c[3] is False for c in calls)
    fx = pd.read_parquet(yc.FX)
    assert list(fx.columns) == ["Date", "usdjpy"] and list(fx["usdjpy"]) == [1.0, 2.0, 3.0]
    assert list(pd.read_parquet(yc.NKY).columns) == ["Date", "nikkei"]
    out = capsys.readouterr().out
    assert "FRED_API_KEY not found" in out and not yc.JPR.exists()


def test_fetch_flattens_multiindex_columns(yc, monkeypatch):
    _fake_yf(monkeypatch, multiindex=True)
    yc.fetch()
    assert list(pd.read_parquet(yc.FX).columns) == ["Date", "usdjpy"]


def test_fetch_fred_series_with_publication_lag(yc, monkeypatch, capsys):
    _fake_yf(monkeypatch)
    monkeypatch.setenv("FRED_API_KEY", "dummy-test-value")
    urls = []

    def get(url, timeout=None):
        urls.append(url)
        if "IR3TIB01JPM156N" in url:
            return _FakeResp({"observations": [{"date": "2024-01-01", "value": "0.1"},
                                               {"date": "2024-02-01", "value": "."}]})
        sid = url.split("series_id=")[1].split("&")[0]
        return _FakeResp({"observations": [{"date": "2024-01-01", "value": "5"}, {"date": "2024-01-08", "value": "6"}]})

    import requests

    monkeypatch.setattr(requests, "get", get)
    assert yc.fetch() == 0
    jp = pd.read_parquet(yc.JPR)
    assert list(jp["jp3m"]) == [0.1]  # "." coerced to NaN and dropped
    g = pd.read_parquet(yc.GROWTH)
    assert set(g.columns) == {"Date", "ICSA", "UNRATE", "NFCI"}
    # each series is shifted by its own publication lag (7, 35, 10 days)
    by_date = g.set_index("Date")
    assert by_date["ICSA"].dropna().index[0] == pd.Timestamp("2024-01-08")
    assert by_date["UNRATE"].dropna().index[0] == pd.Timestamp("2024-02-05")
    assert by_date["NFCI"].dropna().index[0] == pd.Timestamp("2024-01-11")
    assert len(urls) == 4 and all("api_key=dummy-test-value" in u for u in urls)
    out = capsys.readouterr().out
    assert "dummy-test-value" not in out  # the key is never printed
    assert "lagged 7d" in out and "lagged 35d" in out and "lagged 10d" in out


def test_fetch_fred_failures_are_isolated(yc, monkeypatch, capsys):
    _fake_yf(monkeypatch)
    monkeypatch.setenv("FRED_API_KEY", "dummy-test-value")

    def get(url, timeout=None):
        if "IR3TIB01JPM156N" in url:
            return _FakeResp({}, fail=True)
        if "series_id=UNRATE" in url:
            return _FakeResp({"observations": [{"date": "2024-01-01", "value": "4"}]})
        raise ConnectionError("down")

    import requests

    monkeypatch.setattr(requests, "get", get)
    assert yc.fetch() == 0
    out = capsys.readouterr().out
    assert "FRED fetch failed (RuntimeError)" in out and "FRED ICSA failed (ConnectionError)" in out
    assert not yc.JPR.exists()
    g = pd.read_parquet(yc.GROWTH)
    assert list(g.columns) == ["Date", "UNRATE"]
    # when every growth series fails there is no growth file
    yc.GROWTH.unlink()
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: (_ for _ in ()).throw(OSError("all down")))
    yc.fetch()
    assert not yc.GROWTH.exists()
