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
