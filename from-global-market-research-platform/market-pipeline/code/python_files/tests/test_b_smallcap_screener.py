"""Offline tests for smallcap_screener (exclusion flags, validation report, constituent screen)."""

import io
import sys
import types

import numpy as np
import pandas as pd
import pytest

import smallcap_screener as sc


def _panel(n=300, k=12, seed=0, turnover=5e7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"S{i:02d}" for i in range(k)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0), index=idx, columns=cols)
    tv = pd.DataFrame(turnover, index=idx, columns=cols)
    return px, tv


# ---------------------------------------------------------------- flags
def test_flags_columns_and_excluded_is_any_rule():
    px, tv = _panel()
    f = sc.flags(px, tv, 250, list(px.columns))
    assert list(f.columns) == ["downtrend", "distress", "blowup", "untradeable", "excluded"]
    assert (f["excluded"] == f[["downtrend", "distress", "blowup", "untradeable"]].any(axis=1)).all()
    assert f["untradeable"].sum() == 0 and f.index.tolist() == list(px.columns)


def test_flags_downtrend_is_last_below_200dma():
    px, tv = _panel()
    px.iloc[:, 0] = np.linspace(100, 200, len(px))  # steadily rising: last above the 200DMA
    px.iloc[:, 1] = np.linspace(200, 100, len(px))  # steadily falling: last below it
    f = sc.flags(px, tv, 250, list(px.columns))
    assert not f.loc["S00", "downtrend"] and f.loc["S01", "downtrend"]


def test_flags_distress_threshold_fifty_percent_of_three_year_high():
    px, tv = _panel()
    base = np.full(len(px), 100.0)
    for j, last in enumerate([49.9, 50.0, 50.1]):
        s = base.copy()
        s[100] = 100.0
        s[:249] = 100.0
        s[249] = last  # bar i-1 for i=250
        px.iloc[:, j] = s
    f = sc.flags(px, tv, 250, list(px.columns))
    assert f.loc["S00", "distress"] and not f.loc["S01", "distress"] and not f.loc["S02", "distress"]


def test_flags_blowup_is_top_decile_of_volatility_strict():
    px, tv = _panel(k=11, seed=3)
    px.iloc[:, 5] = 100 * np.cumprod(1 + np.random.default_rng(9).normal(0, 0.08, len(px)))  # wild
    f = sc.flags(px, tv, 250, list(px.columns))
    assert f.loc["S05", "blowup"] and f["blowup"].sum() == 1  # only strictly above the 90th percentile


def test_flags_untradeable_median_turnover_floor():
    px, tv = _panel()
    tv.iloc[:, 0] = sc.MIN_TURNOVER - 1
    tv.iloc[:, 1] = sc.MIN_TURNOVER  # exactly at the floor is tradeable
    f = sc.flags(px, tv, 250, list(px.columns))
    assert f.loc["S00", "untradeable"] and not f.loc["S01", "untradeable"]


def test_flags_median_turnover_ignores_brief_spikes():
    px, tv = _panel(turnover=1e6)  # all below floor
    tv.iloc[240:250, 0] = 1e12  # 10 spiky days out of the 60-bar window: median still low
    assert sc.flags(px, tv, 250, list(px.columns)).loc["S00", "untradeable"]


def test_flags_use_only_bars_before_i():
    px, tv = _panel()
    f1 = sc.flags(px, tv, 250, list(px.columns))
    px2 = px.copy()
    px2.iloc[250:, :] *= 0.01  # a crash at and after bar i must not be visible
    tv2 = tv.copy()
    tv2.iloc[250:, :] = 0
    pd.testing.assert_frame_equal(f1, sc.flags(px2, tv2, 250, list(px.columns)))


def test_flags_with_short_history_still_work():
    px, tv = _panel(n=60)
    f = sc.flags(px, tv, 40, list(px.columns))
    assert len(f) == len(px.columns)


@pytest.fixture
def fund(monkeypatch):
    px, tv = _panel()

    def preset(eps_now, eps_year, z):
        n = len(px)
        eps = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
        eps.iloc[:, :] = eps_now
        eps.iloc[: n - 100, :] = eps_year  # a year-ago level for the 253-bar lookback
        monkeypatch.setitem(sc._FUND, "pe", pd.DataFrame(10.0, index=px.index, columns=px.columns))
        monkeypatch.setitem(sc._FUND, "z", pd.DataFrame(z, index=px.index, columns=px.columns))
        monkeypatch.setitem(sc._FUND, "eps", eps)

    return px, tv, preset


def test_flags_fundamental_rules(fund):
    px, tv, preset = fund
    cols = list(px.columns)
    preset(eps_now=pd.Series([-1.0, 0.0, 1.0] + [10.0] * 9, index=cols).values, eps_year=10.0, z=0.0)
    z = sc._FUND["z"].copy()
    z.iloc[:, 3] = 2.0   # exactly 2.0 is not overvalued
    z.iloc[:, 4] = 2.01
    sc._FUND["z"] = z
    f = sc.flags(px, tv, 280, cols, fundamentals=True)
    assert f.loc["S00", "lossmaking"] and f.loc["S01", "lossmaking"] and not f.loc["S02", "lossmaking"]
    assert not f.loc["S03", "overvalued"] and f.loc["S04", "overvalued"]
    assert {"lossmaking", "overvalued", "earnings_fall"} <= set(f.columns)
    assert f.loc["S00", "excluded"] and f.loc["S04", "excluded"]


def test_flags_earnings_fall_needs_halving_from_a_positive_base(fund):
    px, tv, preset = fund
    cols = list(px.columns)
    now = pd.Series([4.9, 5.0, 5.1] + [10.0] * 9, index=cols)
    preset(eps_now=now.values, eps_year=10.0, z=0.0)
    f = sc.flags(px, tv, 280, cols, fundamentals=True)
    assert f.loc["S00", "earnings_fall"] and not f.loc["S01", "earnings_fall"] and not f.loc["S02", "earnings_fall"]
    preset(eps_now=1.0, eps_year=-5.0, z=0.0)  # negative year-ago EPS: not an "earnings fall"
    assert not sc.flags(px, tv, 280, cols, fundamentals=True)["earnings_fall"].any()


def test_flags_fundamentals_skip_year_ago_before_bar_253(fund):
    px, tv, preset = fund
    cols = list(px.columns)
    preset(eps_now=1.0, eps_year=10.0, z=0.0)
    f = sc.flags(px, tv, 253, cols, fundamentals=True)  # i not > 253 -> no year-ago comparison
    assert not f["earnings_fall"].any()


# ---------------------------------------------------------------- validate
class _Args:
    def __init__(self, **kw):
        self.hold, self.warmup, self.fundamentals = 21, 220, False
        self.__dict__.update(kw)


def _fake_study(monkeypatch, px, tv, bad=()):
    mod = types.SimpleNamespace(load_prices=lambda: (px, tv), contaminated=lambda p: list(bad))
    monkeypatch.setitem(sys.modules, "pead_liquidity_study", mod)


def test_validate_no_periods(monkeypatch, capsys):
    px, tv = _panel(n=300, k=20)
    _fake_study(monkeypatch, px, tv)
    monkeypatch.setattr(sc, "BAND_LO", 0)
    monkeypatch.setattr(sc, "BAND_HI", 20)  # band of 20 < 100 -> every formation skipped
    assert sc.validate(_Args()) == 1
    assert "no periods" in capsys.readouterr().out
    assert sc.HOLD == 21  # validate() overrides the module hold


def test_validate_report_tables_and_contaminated_names_dropped(monkeypatch, capsys):
    px, tv = _panel(n=700, k=130, seed=1)
    _fake_study(monkeypatch, px, tv, bad=["S000", "S001", "NOT_A_COLUMN"])
    px.columns = [f"S{i:03d}" for i in range(130)]
    tv.columns = px.columns
    monkeypatch.setattr(sc, "BAND_LO", 0)
    monkeypatch.setattr(sc, "BAND_HI", 130)
    rc = sc.validate(_Args(hold=63, warmup=300))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Small-cap exclusion rules — survivorship-free validation" in out
    assert "63-bar hold, 100bp charged" in out
    for lbl in ("full band (the index proxy)", "SCREENED (exclusions removed)", "the excluded names"):
        assert lbl in out
    for rule in ("downtrend", "blowup"):  # rules that removed no names have no row
        assert f"| {rule} |" in out
    assert "lossmaking" not in out  # fundamentals off
    assert "screened-minus-band per period" in out and "(t = " in out
    assert "(~128 names)" in out  # two contaminated names removed from the 130-name band


def test_validate_verdict_text_and_cost_on_full_band(monkeypatch, capsys):
    px, tv = _panel(n=700, k=130, seed=2)
    _fake_study(monkeypatch, px, tv)
    monkeypatch.setattr(sc, "BAND_LO", 0)
    monkeypatch.setattr(sc, "BAND_HI", 130)
    sc.validate(_Args(hold=63, warmup=300))
    out = capsys.readouterr().out
    assert ("excluding helps" in out) or ("EXCLUDING HURTS" in out)
    first_row = next(ln for ln in out.splitlines() if ln.startswith("| full band"))
    assert first_row.rstrip().endswith("+0.00pp |")  # the band is its own baseline


# ---------------------------------------------------------------- screen
def _cons_csv(rows):
    return pd.DataFrame(rows, columns=["Company Name", "Industry", "Symbol"]).to_csv(index=False).encode()


def _screen_env(monkeypatch, tmp_path, px, tv):
    monkeypatch.setattr(sc, "HERE", tmp_path)
    (tmp_path / "reports").mkdir()
    _fake_study(monkeypatch, px, tv)


class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_screen_downloads_caches_and_reports(monkeypatch, tmp_path, capsys):
    px, tv = _panel(n=300, k=12, seed=4)
    tv.iloc[:, 0] = 1.0  # S00 untradeable
    _screen_env(monkeypatch, tmp_path, px, tv)
    body = _cons_csv([("Co A", "Banks", "S00"), ("Co B", "Banks", " S01 "), ("Co C", "IT", "S02"),
                      ("Co D", "IT", "MISSING"), ("Co E", "IT", None)])
    seen = {}

    def urlopen(req, timeout):
        seen["ua"], seen["timeout"] = req.get_header("User-agent"), timeout
        return _Resp(body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    rc = sc.screen(types.SimpleNamespace(fundamentals=False))
    out = capsys.readouterr().out
    assert rc == 0 and seen["timeout"] == 45 and seen["ua"].startswith("Mozilla/5.0")
    assert (tmp_path / "cache_seed" / "smallcap250_constituents.csv").read_bytes() == body
    assert "constituents 4 · priced in panel 3" in out  # blank symbol dropped; MISSING not in panel
    assert "| untradeable | 1 |" in out
    saved = pd.read_csv(tmp_path / "reports" / "smallcap_screen.csv")
    assert set(saved["symbol"]) == {"S00", "S01", "S02"} and "Company Name" in saved.columns
    assert bool(saved.set_index("symbol").at["S00", "excluded"]) is True


def test_screen_falls_back_to_cache_or_fails(monkeypatch, tmp_path, capsys):
    px, tv = _panel(n=300, k=12, seed=4)
    _screen_env(monkeypatch, tmp_path, px, tv)

    def boom(req, timeout):
        raise OSError("blocked")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert sc.screen(types.SimpleNamespace(fundamentals=False)) == 1
    assert "constituent fetch failed and no cache" in capsys.readouterr().out
    cache = tmp_path / "cache_seed"
    cache.mkdir()
    (cache / "smallcap250_constituents.csv").write_bytes(_cons_csv([("Co A", "Banks", "S00"), ("Co B", "IT", "S01"), ("Co C", "IT", "S02")]))
    assert sc.screen(types.SimpleNamespace(fundamentals=False)) == 0
    assert "using cache" in capsys.readouterr().out


def test_screen_rejects_non_csv_shell_and_uses_cache(monkeypatch, tmp_path, capsys):
    px, tv = _panel(n=300, k=12, seed=4)
    _screen_env(monkeypatch, tmp_path, px, tv)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp(b"<html>SPA shell</html>"))
    assert sc.screen(types.SimpleNamespace(fundamentals=False)) == 1  # shell is not cached either
    assert not (tmp_path / "cache_seed" / "smallcap250_constituents.csv").exists()


# ---------------------------------------------------------------- main
def test_main_validate_writes_report_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sc, "HERE", tmp_path)
    monkeypatch.setattr(sc, "validate", lambda a: print("BODY LINE") or 0)
    monkeypatch.setattr(sys, "argv", ["smallcap_screener.py", "--validate", "--hold", "252", "--warmup", "100", "--fundamentals"])
    assert sc.main() == 0
    rp = tmp_path / "reports" / "smallcap_validation_hold252.md"
    text = rp.read_text()
    assert "BODY LINE" in text and "--validate --fundamentals --hold 252 --warmup 100" in text
    assert "wrote" in capsys.readouterr().out


def test_main_screen_and_usage(monkeypatch, capsys):
    monkeypatch.setattr(sc, "screen", lambda a: 7)
    monkeypatch.setattr(sys, "argv", ["smallcap_screener.py", "--screen"])
    assert sc.main() == 7
    monkeypatch.setattr(sys, "argv", ["smallcap_screener.py"])
    assert sc.main() != 7
