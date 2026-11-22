"""justified_mailer.py -- RSI, oversold/momentum pick selection, evidence HTML, watchlist sync."""

import smtplib
import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

import justified_mailer as JM


# ── rsi14 ─────────────────────────────────────────────────────────────────────
def test_rsi_all_gains_is_nan_and_all_losses_zero():
    up = JM.rsi14(pd.Series(np.arange(1.0, 40.0)))
    assert up.iloc[1:].isna().all()
    down = JM.rsi14(pd.Series(np.arange(40.0, 1.0, -1.0)))
    assert (down.iloc[1:] == 0.0).all()


def test_rsi_matches_wilder_recursion():
    rng = np.random.default_rng(5)
    c = pd.Series(50 + rng.normal(0, 1, 100).cumsum())
    d = c.diff().to_numpy()
    up = dn = None
    ref = []
    for x in d[1:]:
        u, l = max(x, 0.0), max(-x, 0.0)
        up = u if up is None else up + (u - up) / 14
        dn = l if dn is None else dn + (l - dn) / 14
        ref.append(100 - 100 / (1 + up / dn))
    assert np.allclose(JM.rsi14(c).to_numpy()[1:], ref)


def test_rsi_symmetric_series_is_near_fifty():
    assert 45 < JM.rsi14(pd.Series([10.0, 11.0] * 60)).iloc[-1] < 55


# ── helpers to build panels ───────────────────────────────────────────────────
def write_workbook(tmp_path, pattern_dir, fname, sheet_rows):
    p = tmp_path / pattern_dir / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p) as xw:
        pd.DataFrame(sheet_rows).to_excel(xw, sheet_name="All_Stocks", index=False)
    return p


def beaten_down(n=120, bounce_every=7, start=50.0):
    """Mostly falling closes with periodic small bounces: RSI lands well inside (2, 30)."""
    c, out = start, []
    for i in range(n):
        c *= 1.002 if (i % bounce_every == 0) else 0.99
        out.append(c)
    return np.array(out)


def panel(series_by_symbol):
    rows = []
    for sym, closes in series_by_symbol.items():
        dates = pd.bdate_range("2025-01-01", periods=len(closes))
        rows.append(pd.DataFrame({"Date": dates, "Symbol": sym, "Close": closes}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(JM, "HERE", tmp_path)
    panels = {m: tmp_path / f"{m}.parquet" for m in ("US", "KR", "JP", "IN")}
    monkeypatch.setattr(JM, "PANELS", panels)
    monkeypatch.setattr(JM, "BHAV", tmp_path)
    return tmp_path


# ── _tiers ────────────────────────────────────────────────────────────────────
def test_tiers_reads_latest_workbook_and_dedupes(env):
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        [{"Symbol": "OLD", "Liquidity_Tier": "T1_MOST_LIQUID", "Name": "Old"}],
    )
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_2.xlsx",
        [
            {"Symbol": "AAA", "Liquidity_Tier": "T1_MOST_LIQUID", "Name": "Alpha"},
            {"Symbol": "AAA", "Liquidity_Tier": "T5_MOST_ILLIQUID", "Name": "Dup"},
            {"Symbol": "BBB", "Liquidity_Tier": "T2_LIQUID"},
        ],
    )
    t = JM._tiers("US")
    assert list(t.index) == ["AAA", "BBB"]
    assert (
        t.loc["AAA", "tier"] == "T1_MOST_LIQUID" and t.loc["AAA", "name"] == "Alpha"
    )  # first kept
    assert t.loc["BBB", "name"] == "" or pd.isna(t.loc["BBB", "name"])


def test_tiers_uses_market_specific_symbol_column_and_missing_workbook(env):
    write_workbook(
        env,
        "korea_scan",
        "korea_market_scan_1.xlsx",
        [{"YF_Ticker": "005930.KS", "Liquidity_Tier": "T2_LIQUID", "Name": "Samsung"}],
    )
    assert list(JM._tiers("KR").index) == ["005930.KS"]
    assert JM._tiers("JP").empty


# ── oversold_picks ────────────────────────────────────────────────────────────
def test_oversold_picks_selection_rules(env):
    series = {
        "DEEP": beaten_down(120, 7),
        "SHALLOW": beaten_down(120, 3),
        "SHORT": beaten_down(99, 7),
        "EXACT": beaten_down(100, 7),
        "PENNY": beaten_down(120, 7, start=1.5) * 0.5,
        "ILLIQ": beaten_down(120, 7),
        "NOTIER": beaten_down(120, 7),
        "RISER": np.linspace(10, 50, 120),
        "ZOMBIE": np.linspace(50, 10, 120),
    }
    panel(series).to_parquet(env / "US.parquet", index=False)
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        [
            {"Symbol": s, "Liquidity_Tier": t, "Name": f"{s} Corp"}
            for s, t in [
                ("DEEP", "T1_MOST_LIQUID"),
                ("SHALLOW", "T3_MID"),
                ("SHORT", "T1_MOST_LIQUID"),
                ("EXACT", "T2_LIQUID"),
                ("PENNY", "T1_MOST_LIQUID"),
                ("ILLIQ", "T4_ILLIQUID"),
                ("RISER", "T1_MOST_LIQUID"),
                ("ZOMBIE", "T1_MOST_LIQUID"),
            ]
        ],
    )
    out = JM.oversold_picks("US")
    syms = list(out["symbol"])
    assert "SHORT" not in syms  # < 100 bars
    assert "EXACT" in syms  # exactly 100 bars is enough
    assert "PENNY" not in syms  # US $2 floor
    assert "ILLIQ" not in syms and "NOTIER" not in syms
    assert "RISER" not in syms  # RSI not < 30
    assert "ZOMBIE" not in syms  # all-down RSI == 0 is excluded (rsi > 2)
    assert {"DEEP", "SHALLOW", "EXACT"} <= set(syms)
    # most-oversold first, labelled with one decimal
    px = panel(series)
    rsi = {s: JM.rsi14(g.sort_values("Date")["Close"]).iloc[-1] for s, g in px.groupby("Symbol")}
    assert syms == sorted(syms, key=lambda s: rsi[s])
    row = out.set_index("symbol").loc["DEEP"]
    assert row["metric"] == f"{round(rsi['DEEP'], 1)} RSI" and row["name"] == "DEEP Corp"
    assert row["close"] == pytest.approx(series["DEEP"][-1])


def test_oversold_floor_only_applies_to_us(env):
    closes = beaten_down(120, 7, start=1.5)  # ends well under $2
    panel({"PENNY.KS": closes}).to_parquet(env / "KR.parquet", index=False)
    write_workbook(
        env,
        "korea_scan",
        "korea_market_scan_1.xlsx",
        [{"YF_Ticker": "PENNY.KS", "Liquidity_Tier": "T1_MOST_LIQUID", "Name": "Penny"}],
    )
    assert list(JM.oversold_picks("KR")["symbol"]) == ["PENNY.KS"]
    panel({"PENNY": closes}).to_parquet(env / "US.parquet", index=False)
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        [{"Symbol": "PENNY", "Liquidity_Tier": "T1_MOST_LIQUID"}],
    )
    assert JM.oversold_picks("US").empty


def test_oversold_floor_is_inclusive_at_two_dollars(env):
    base = beaten_down(120, 7)
    scale = 2.0 / base[-1]
    panel({"EDGE": base * scale}).to_parquet(env / "US.parquet", index=False)
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        [{"Symbol": "EDGE", "Liquidity_Tier": "T1_MOST_LIQUID"}],
    )
    out = JM.oversold_picks("US")
    assert list(out["symbol"]) == ["EDGE"] and out["close"].iloc[0] == pytest.approx(2.0)


def test_oversold_caps_at_top_n(env, monkeypatch):
    monkeypatch.setattr(JM, "TOP_N", 2)
    series = {f"S{i}": beaten_down(120, 7 + i % 3, start=50.0 + i) for i in range(5)}
    panel(series).to_parquet(env / "US.parquet", index=False)
    write_workbook(
        env,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        [{"Symbol": s, "Liquidity_Tier": "T1_MOST_LIQUID"} for s in series],
    )
    assert len(JM.oversold_picks("US")) == 2


# ── India momentum ────────────────────────────────────────────────────────────
def write_nse(env, rows):
    pd.DataFrame(rows, columns=["TckrSymb", "ISIN", "SctySrs", "FinInstrmNm"]).to_parquet(
        env / "nse.parquet", index=False
    )


def test_india_equity_names_allowlist_by_isin_prefix(env):
    write_nse(
        env,
        [
            ("RELIANCE", "INE002A01018", "EQ", "Reliance Industries"),
            ("RELIANCE", "INE002A01018", "EQ", "dup"),
            ("GOLDETF", "INF204KB14I2", "EQ", "Gold ETF"),
            ("GSEC", "IN0020200013", "EQ", "G-Sec"),
            ("BE_STOCK", "INE111A01011", "BE", "Trade-to-trade"),
            ("TCS", "INE467B01029", "EQ", "TCS"),
        ],
    )
    names = JM._india_equity_names()
    assert dict(names) == {"RELIANCE": "Reliance Industries", "TCS": "TCS"}


def india_panel(env):
    n = 300
    up = 100 * 1.002 ** np.arange(n)
    rise_fall = np.concatenate(
        [np.full(150, 100.0), np.linspace(100, 160, 100), np.linspace(160, 140, 50)]
    )
    falling = np.linspace(100, 50, n)
    series = {
        "UPTREND": up,
        "FADING": rise_fall,
        "FALLING": falling,
        "ETFX": up,
        "UNGATED": 50 * 1.0015 ** np.arange(n),
    }
    panel(series).to_parquet(env / "IN.parquet", index=False)
    write_nse(
        env,
        [
            ("UPTREND", "INE000000001", "EQ", "Up Trend Ltd"),
            ("FADING", "INE000000002", "EQ", "Fading Ltd"),
            ("FALLING", "INE000000003", "EQ", "Falling Ltd"),
            ("ETFX", "INF000000004", "EQ", "An ETF"),
            ("UNGATED", "INE000000005", "EQ", "Ungated Ltd"),
        ],
    )
    return series


def test_india_momentum_splits_near_high_and_momentum_lists(env):
    india_panel(env)
    out = JM.india_momentum_picks()
    rows = out.set_index("symbol")
    assert list(out["symbol"]) == ["UPTREND", "UNGATED", "FADING"]  # near-high first, then momentum
    assert "ETFX" not in rows.index and "FALLING" not in rows.index
    assert rows.loc["UPTREND", "metric"] == "100.0% of 52w-hi"
    assert rows.loc["UPTREND", "name"] == "Up Trend Ltd"
    fading = rows.loc["FADING"]
    assert fading["metric"].endswith("% 6m mom") and fading["close"] == pytest.approx(140.0)
    c = np.concatenate([np.full(150, 100.0), np.linspace(100, 160, 100), np.linspace(160, 140, 50)])
    assert fading["metric"] == f"{round((c[-1] / c[-127] - 1) * 100, 1)}% 6m mom"


def test_india_momentum_workbook_acts_as_liquidity_gate(env):
    india_panel(env)
    write_workbook(
        env,
        "indian_full_scan",
        "indian_full_scan_1.xlsx",
        [{"Symbol": "UPTREND"}, {"Symbol": "FADING"}],
    )
    assert list(JM.india_momentum_picks()["symbol"]) == ["UPTREND", "FADING"]


def test_india_momentum_requires_a_close_within_five_percent_of_high(env):
    near = np.concatenate([np.linspace(100, 200, 280), np.linspace(200, 190.0, 20)])  # 95.0% of hi
    far = np.concatenate([np.linspace(100, 200, 280), np.linspace(200, 189.9, 20)])  # 94.95%
    panel({"NEAR": near, "FAR": far}).to_parquet(env / "IN.parquet", index=False)
    write_nse(env, [("NEAR", "INE000000001", "EQ", "n"), ("FAR", "INE000000002", "EQ", "f")])
    rows = JM.india_momentum_picks().set_index("symbol")
    assert rows.loc["NEAR", "metric"] == "95.0% of 52w-hi"
    assert rows.loc["FAR", "metric"].endswith("% 6m mom")


# ── fundamentals overlay ──────────────────────────────────────────────────────
def write_combo(env, fname, rows):
    p = env / "reports" / fname
    p.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)


def test_overlay_filters_combos_and_formats(env):
    write_combo(
        env,
        "factor_combo_india_252d.csv",
        [
            {
                "combo": "piotroski",
                "testable": True,
                "n": 12345,
                "n_years": 9.4,
                "edge": 2.34,
                "t": 0.876,
                "years_positive": 6,
            },
            {
                "combo": "roce_plus",
                "testable": False,
                "n": 1,
                "n_years": 1,
                "edge": 1,
                "t": 1,
                "years_positive": 1,
            },
            {
                "combo": "momentum",
                "testable": True,
                "n": 5,
                "n_years": 5,
                "edge": 5,
                "t": 5,
                "years_positive": 5,
            },
        ],
    )
    write_combo(
        env,
        "factor_combo_us_252d.csv",
        [
            {
                "combo": "piotroski + roce_plus",
                "testable": True,
                "n": 999,
                "n_years": 8,
                "edge": -1.25,
                "t": -0.5,
                "years_positive": 3,
            }
        ],
    )
    html = JM.fundamentals_overlay()
    assert (
        "<tr><td>India</td><td>piotroski</td><td>12,345</td><td>9y</td><td>+2.3pp</td>"
        "<td>0.88</td><td>6/9</td></tr>" in html
    )
    assert "<td>US</td><td>piotroski + roce_plus</td><td>999</td><td>8y</td><td>-1.2pp</td>" in html
    assert "roce_plus</td><td>1" not in html and "momentum" not in html
    assert "Read honestly" in html and "illiquidity premium" in html


def test_overlay_empty_without_files_or_rows(env):
    assert JM.fundamentals_overlay() == ""
    write_combo(
        env,
        "factor_combo_us_252d.csv",
        [
            {
                "combo": "piotroski",
                "testable": False,
                "n": 1,
                "n_years": 1,
                "edge": 1,
                "t": 1,
                "years_positive": 1,
            }
        ],
    )
    assert JM.fundamentals_overlay() == ""


# ── section / build ───────────────────────────────────────────────────────────
def test_section_with_picks_truncates_names_and_formats_close():
    picks = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["N" * 50, ""],
            "metric": ["25.0 RSI", "28.1 RSI"],
            "close": [1234.5678, 2.0],
        }
    )
    html = JM.section("US", "🇺🇸 US", picks)
    assert "🇺🇸 US — RSI-14 oversold (&lt;30)" in html
    assert "edge <b>+1.85pp</b> vs equal-weight market, win rate 0.463" in html
    assert f'<td style="color:#666">{"N" * 34}</td>' in html and "N" * 35 not in html
    assert "<td>25.0 RSI</td><td>1,234.57</td>" in html and "<td>2.00</td>" in html
    assert html.count("<tr><td") == 2


def test_section_empty_message_still_shows_evidence():
    html = JM.section("KR", "🇰🇷 Korea", pd.DataFrame())
    assert "No qualifying names today." in html and "+2.04pp" in html and "<table" not in html


def test_build_orders_sections_and_includes_caveats(env):
    picks = {
        m: pd.DataFrame({"symbol": [f"{m}X"], "name": ["n"], "metric": ["m"], "close": [1.0]})
        for m in ("IN", "US", "KR", "JP")
    }
    html = JM.build(picks)
    assert f"Justified Brief — {date.today():%d %b %Y}" in html
    order = [
        html.index(x) for x in ("🇮🇳 India", "🇺🇸 US", "🇰🇷 Korea", "🇯🇵 Japan", "Excluded by evidence")
    ]
    assert order == sorted(order)
    for k, _ in JM.EXCLUDED:
        assert f"<b>{k}</b>" in html
    assert JM.CAVEATS in html and html.rstrip().endswith("</div>")
    assert "Fundamental overlay" not in html  # no CSVs in the sandbox


# ── send ──────────────────────────────────────────────────────────────────────
def fake_env(monkeypatch, values):
    monkeypatch.setitem(
        sys.modules, "env_loader", types.SimpleNamespace(get=lambda k, d="": values.get(k, d))
    )


def test_send_without_credentials_does_nothing(monkeypatch, capsys):
    fake_env(monkeypatch, {})
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("must not connect"))
    assert JM.send("<p>x</p>") is False
    assert "no creds — not sent" in capsys.readouterr().out
    fake_env(monkeypatch, {"GMAIL_USER": "u@example.com"})  # password missing
    assert JM.send("<p>x</p>") is False


def test_send_builds_html_message_and_defaults_recipient(monkeypatch):
    sent = {}

    class SMTP:
        def __init__(self, host, port):
            sent["addr"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, u, p):
            sent["login"] = (u, p)

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr(smtplib, "SMTP_SSL", SMTP)
    fake_env(monkeypatch, {"GMAIL_USER": "me@example.com", "GMAIL_APP_PASSWORD": "pw"})
    assert JM.send("<p>hi</p>") is True
    assert sent["addr"] == ("smtp.gmail.com", 465) and sent["login"] == ("me@example.com", "pw")
    msg = sent["msg"]
    assert msg["To"] == "me@example.com" and msg["From"] == "me@example.com"
    assert msg.get_content_type() == "text/html" and "Justified Brief" in msg["Subject"]
    fake_env(
        monkeypatch,
        {"GMAIL_USER": "me@example.com", "GMAIL_APP_PASSWORD": "pw", "MAIL_TO": "you@example.com"},
    )
    JM.send("<p>hi</p>")
    assert sent["msg"]["To"] == "you@example.com"


# ── watchlist sync ────────────────────────────────────────────────────────────
def picks_for(*syms):
    return pd.DataFrame({"symbol": list(syms), "name": "", "metric": "30.0 RSI", "close": 1.0})


def test_sync_watchlist_appends_only_new_pairs(env):
    pd.DataFrame(
        {
            "symbol": ["aaa", "OLD"],
            "market": ["us", "IN"],
            "status": ["held", "watch"],
            "note": ["a", "b"],
        }
    ).to_csv(env / "watchlist.csv", index=False)
    n = JM.sync_watchlist({"US": picks_for("AAA", "BBB", "BBB"), "IN": picks_for("AAA")})
    assert n == 2  # BBB (once) + AAA@IN; AAA@US exists
    wl = pd.read_csv(env / "watchlist.csv")
    assert list(wl["symbol"]) == ["aaa", "OLD", "BBB", "AAA"]
    new = wl.iloc[2:]
    assert list(new["status"]) == ["justified", "justified"] and list(new["market"]) == ["US", "IN"]
    assert new["note"].iloc[0] == f"30.0 RSI @ {date.today():%Y-%m-%d}"
    assert list(wl.iloc[:2]["status"]) == ["held", "watch"]  # existing rows untouched


def test_sync_watchlist_noop_cases(env):
    assert JM.sync_watchlist({"US": picks_for("A")}) == 0  # no watchlist.csv
    (env / "watchlist.csv").write_text("symbol,market,status,note\nA,US,held,x\n")
    before = (env / "watchlist.csv").read_text()
    assert JM.sync_watchlist({"US": picks_for("A")}) == 0
    assert (env / "watchlist.csv").read_text() == before


# ── main ──────────────────────────────────────────────────────────────────────
def test_main_writes_draft_and_honours_flags(env, monkeypatch, capsys):
    monkeypatch.setattr(JM, "OUT_HTML", env / "out" / "brief.html")
    empty = pd.DataFrame(columns=["symbol", "name", "metric", "close"])
    monkeypatch.setattr(JM, "india_momentum_picks", lambda: empty)
    monkeypatch.setattr(JM, "oversold_picks", lambda m: empty)
    calls = []
    monkeypatch.setattr(JM, "sync_watchlist", lambda p: calls.append(("sync", sorted(p))) or 3)
    monkeypatch.setattr(JM, "send", lambda h: calls.append(("send", "Justified Brief" in h)))
    monkeypatch.setattr(sys, "argv", ["x"])
    assert JM.main() == 0
    assert (env / "out" / "brief.html").read_text().startswith("<div")
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["x", "--send", "--sync-watchlist"])
    JM.main()
    assert calls == [("sync", ["IN", "JP", "KR", "US"]), ("send", True)]
    assert "watchlist.csv: +3 `justified` rows" in capsys.readouterr().out
