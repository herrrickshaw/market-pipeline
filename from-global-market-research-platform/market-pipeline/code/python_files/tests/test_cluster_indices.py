"""cluster_indices.py -- within-group correlation and the sector-list fetch/cache fallback."""

import numpy as np
import pandas as pd
import pytest

import cluster_indices as CI


def test_within_corr_averages_group_means():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    fwd = pd.DataFrame({"A": a, "B": a, "C": -a, "D": rng.normal(size=200)})
    assert CI.within_corr(fwd, {"g1": ["A", "B"]}) == pytest.approx(1.0)
    assert CI.within_corr(fwd, {"g2": ["A", "C"]}) == pytest.approx(-1.0)
    assert CI.within_corr(fwd, {"g1": ["A", "B"], "g2": ["A", "C"]}) == pytest.approx(0.0)


def test_within_corr_uses_mean_of_pairs_within_each_group():
    a = np.arange(100.0)
    fwd = pd.DataFrame({"A": a, "B": a, "C": -a})
    # group of three: pairs AB=+1, AC=-1, BC=-1 -> mean -1/3
    assert CI.within_corr(fwd, {"g": ["A", "B", "C"]}) == pytest.approx(-1 / 3)


def test_within_corr_is_the_mean_not_the_sum_of_group_averages():
    a = np.arange(100.0)
    fwd = pd.DataFrame({"A": a, "B": a, "C": -a})
    # group 1: +1 ; group 2 (A,B,C): -1/3  -> mean 1/3
    assert CI.within_corr(fwd, {"g1": ["A", "B"], "g2": ["A", "B", "C"]}) == pytest.approx(1 / 3)


def test_within_corr_skips_small_missing_and_degenerate_groups():
    a = np.arange(50.0)
    fwd = pd.DataFrame({"A": a, "B": a, "K": 1.0, "L": 2.0})
    assert CI.within_corr(fwd, {"solo": ["A"]}) != CI.within_corr(fwd, {"solo": ["A"]})  # NaN
    assert np.isnan(CI.within_corr(fwd, {"solo": ["A"]}))
    assert np.isnan(CI.within_corr(fwd, {}))
    assert np.isnan(CI.within_corr(fwd, {"missing": ["X", "Y"], "one": ["A", "X"]}))
    # constant columns give NaN correlations, which are dropped; the other group still counts
    assert CI.within_corr(fwd, {"flat": ["K", "L"], "ok": ["A", "B"]}) == pytest.approx(1.0)
    assert np.isnan(CI.within_corr(fwd, {"flat": ["K", "L"]}))


class FakeResp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


CSV = b"Company Name,Industry,Symbol,Series\nA Ltd,Banks,  AAA ,EQ\nB Ltd,IT,BBB,EQ\n"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(CI, "CACHE", tmp_path / "sub" / "sectors.csv")
    return CI.CACHE


def test_sectors_fetch_success_writes_cache_and_strips_symbols(cache, monkeypatch):
    monkeypatch.setattr(CI.urllib.request, "urlopen", lambda req, timeout: FakeResp(CSV))
    s = CI.sectors()
    assert s.to_dict() == {"AAA": "Banks", "BBB": "IT"} and s.index.name == "Symbol"
    assert cache.read_bytes() == CSV


def test_sectors_sends_browser_user_agent(cache, monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        return FakeResp(CSV)

    monkeypatch.setattr(CI.urllib.request, "urlopen", fake)
    CI.sectors()
    assert seen == {"ua": CI.UA, "timeout": 45, "url": CI.TOTAL_MKT}


def test_sectors_falls_back_to_cache_on_failure(cache, monkeypatch, capsys):
    cache.parent.mkdir(parents=True)
    cache.write_bytes(CSV)

    def boom(req, timeout):
        raise OSError("blocked")

    monkeypatch.setattr(CI.urllib.request, "urlopen", boom)
    assert CI.sectors().to_dict() == {"AAA": "Banks", "BBB": "IT"}
    assert "sector fetch failed (blocked) — using cache" in capsys.readouterr().out


def test_sectors_rejects_html_error_page_and_uses_cache(cache, monkeypatch):
    cache.parent.mkdir(parents=True)
    cache.write_bytes(CSV)
    monkeypatch.setattr(
        CI.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResp(b"<html>Access denied</html>" * 40),
    )
    assert set(CI.sectors().index) == {"AAA", "BBB"}
    assert cache.read_bytes() == CSV  # the error page never overwrote it


def test_sectors_symbol_marker_must_be_in_first_400_bytes(cache, monkeypatch):
    late = b"x" * 400 + b"Symbol,Industry\nA,B\n"
    monkeypatch.setattr(CI.urllib.request, "urlopen", lambda req, timeout: FakeResp(late))
    with pytest.raises(SystemExit):
        CI.sectors()


def test_sectors_no_network_no_cache_exits_with_reason(cache, monkeypatch):
    def boom(req, timeout):
        raise OSError("offline")

    monkeypatch.setattr(CI.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="sector list unavailable and no cache: offline"):
        CI.sectors()
    assert not cache.exists()


def test_parameters():
    assert (CI.UNIV, CI.LOOKBACK, CI.MIN_GROUP) == (300, 252, 4)


# ═══ main(): the birds-of-a-feather study on a synthetic panel ══════════════════
import sys  # noqa: E402
import types  # noqa: E402


def factor_panel(n_per=30, n_clusters=4, n_days=1100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", periods=n_days)
    factors = rng.normal(0, 0.01, (n_days, n_clusters))
    cols, labels = {}, {}
    for c in range(n_clusters):
        for i in range(n_per):
            name = f"C{c}S{i:02d}"
            r = 1.0 * factors[:, c] + rng.normal(0, 0.006, n_days)
            cols[name] = 100 * np.cumprod(1 + r)
            labels[name] = f"Ind{c}"
    px = pd.DataFrame(cols, index=idx)
    tv = pd.DataFrame(1e6, index=idx, columns=px.columns)
    return px, tv, pd.Series(labels, name="Industry").rename_axis("Symbol")


@pytest.fixture
def study(tmp_path, monkeypatch):
    px, tv, ind = factor_panel()
    fake = types.SimpleNamespace(load_prices=lambda: (px, tv), contaminated=lambda p: ["C0S00"])
    monkeypatch.setitem(sys.modules, "pead_liquidity_study", fake)
    monkeypatch.setattr(CI, "sectors", lambda: ind)
    monkeypatch.setattr(CI, "OUT_MD", tmp_path / "ci.md")
    monkeypatch.setattr(CI, "UNIV", 200)
    return px, ind


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["cluster_indices", *argv])
    return CI.main()


def test_main_report_structure_and_ordering(study, monkeypatch):
    assert run_main(monkeypatch, "--k", "4", "--hold", "60", "--seed", "1") == 0
    md = CI.OUT_MD.read_text()
    assert md.startswith("# Birds of a feather")
    for h in ("| grouping |", "## by sub-period", "## sensitivity to k"):
        assert h in md
    rows = {
        x.split("|")[1].strip(" *"): [c.strip(" *") for c in x.split("|")[2:-1]]
        for x in md.splitlines()
        if x.startswith("| ")
        and x.count("|") == 5
        and any(k in x for k in ("RANDOM", "OFFICIAL", "DATA"))
    }
    rnd = float(rows["RANDOM (the null)"][0])
    off = float(rows["OFFICIAL NSE Industry"][0])
    dat = float(rows["DATA clusters (trailing corr)"][0])
    assert off > rnd + 0.05 and dat > rnd + 0.05  # real structure beats shuffled groups
    assert float(rows["OFFICIAL NSE Industry"][1]) == pytest.approx(off - rnd, abs=1e-4)
    assert (
        "k=4** (4 usable clusters)" in md
        and "- groups: 4 data clusters vs 4 official industries" in md
    )
    assert "**shipped**" not in md  # k=4 is not one of the recorded sweep rows


def test_main_excludes_contaminated_symbols_and_reports_universe(study, monkeypatch, capsys):
    run_main(monkeypatch, "--k", "4", "--hold", "60")
    out = capsys.readouterr().out
    assert "119 symbols · 120 with an NSE Industry label" in out  # C0S00 dropped from the panel


def test_main_marks_the_shipped_k_in_the_sweep_table(study, monkeypatch):
    run_main(monkeypatch, "--k", "45", "--hold", "60")
    md = CI.OUT_MD.read_text()
    assert "| 45 ⬅ **shipped** | 8 |" in md and "| 30 | 6 |" in md


def test_main_subperiod_rows_need_four_formations(study, monkeypatch):
    run_main(monkeypatch, "--k", "4", "--hold", "60")
    md = CI.OUT_MD.read_text()
    assert "| 2017-2020 |" in md and "| 2021-2026 |" not in md  # the panel ends in 2021


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="when no formation has >= 80 labelled names the results frame is empty and "
    "main() reads d.date on it, crashing instead of reporting 'no formations'",
)
def test_main_reports_when_no_formation_qualifies(study, monkeypatch):
    px, ind = study
    monkeypatch.setattr(CI, "sectors", lambda: ind.iloc[:50])
    assert run_main(monkeypatch, "--k", "4", "--hold", "60") in (0, 1)
