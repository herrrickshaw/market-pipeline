"""Offline tests for send_mailer (MIME assembly, credential guards, failure-isolated sections)."""

import email
import json
import sys
import types
from datetime import date

import pandas as pd
import pytest

import send_mailer as sm


class _Env:
    def __init__(self, **vals):
        self.vals = vals

    def get(self, key, default=""):
        return self.vals.get(key, default)


class _SMTP:
    instances = []

    def __init__(self, host, port):
        self.host, self.port, self.sent, self.login_args = host, port, [], None
        _SMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, pw):
        self.login_args = (user, pw)

    def sendmail(self, frm, to, msg):
        self.sent.append((frm, to, msg))


@pytest.fixture
def mailer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sm, "__file__", str(tmp_path / "send_mailer.py"))
    _SMTP.instances = []
    monkeypatch.setattr(sm.smtplib, "SMTP_SSL", _SMTP)
    return tmp_path


def _creds(monkeypatch, **extra):
    vals = {"GMAIL_USER": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}
    vals.update(extra)
    monkeypatch.setattr(sm, "_env", _Env(**vals))


# ---------------------------------------------------------------- send: credential guards
@pytest.mark.parametrize(
    "vals",
    [{}, {"GMAIL_USER": "me@example.com"}, {"GMAIL_APP_PASSWORD": "x" * 16},
     {"GMAIL_USER": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 15},
     {"GMAIL_USER": "me@example.com", "GMAIL_APP_PASSWORD": "PUT-YOUR-APP-PASSWORD-HERE"}],
)
def test_send_without_valid_credentials_saves_html_and_returns_false(mailer, monkeypatch, vals, capsys):
    monkeypatch.setattr(sm, "_env", _Env(**vals))
    assert sm.send("S", "t", "<b>h</b>") is False
    assert (mailer / "brief_today.html").read_text() == "<b>h</b>"
    assert _SMTP.instances == []
    assert "saved brief_today.html instead of sending" in capsys.readouterr().out


def test_send_accepts_a_sixteen_character_password(mailer, monkeypatch):
    _creds(monkeypatch)
    assert sm.send("S", "t", "h") is True
    assert not (mailer / "brief_today.html").exists()


# ---------------------------------------------------------------- send: MIME + transport
def _parse(instance):
    frm, to, raw = instance.sent[0]
    return frm, to, email.message_from_string(raw)


def test_send_plain_alternative_message_and_recipients(mailer, monkeypatch, capsys):
    _creds(monkeypatch, MAIL_TO="a@x.com, b@y.com")
    assert sm.send("Brief", "plain body", "<p>html body</p>") is True
    s = _SMTP.instances[0]
    assert (s.host, s.port, s.login_args) == ("smtp.gmail.com", 465, ("me@example.com", "x" * 16))
    frm, to, msg = _parse(s)
    assert frm == "me@example.com" and to == ["a@x.com", "b@y.com"]
    assert msg["Subject"] == "Brief" and msg["From"] == "me@example.com" and msg["To"] == "a@x.com, b@y.com"
    assert msg.get_content_type() == "multipart/alternative"
    kinds = [p.get_content_type() for p in msg.get_payload()]
    assert kinds == ["text/plain", "text/html"]
    assert "sent 'Brief' → a@x.com, b@y.com" in capsys.readouterr().out


def test_send_defaults_recipient_to_sender(mailer, monkeypatch):
    _creds(monkeypatch)
    sm.send("S", "t", "h")
    assert _parse(_SMTP.instances[0])[1] == ["me@example.com"]


def test_send_inline_images_and_attachments_shape(mailer, monkeypatch, capsys):
    _creds(monkeypatch)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    sm.send("S", "t", "<img src='cid:chart1'>", attachments=[("full.html", "<h1>full</h1>")],
            inline_images={"chart1": png})
    msg = _parse(_SMTP.instances[0])[2]
    assert msg.get_content_type() == "multipart/mixed"
    core, att = msg.get_payload()
    assert core.get_content_type() == "multipart/related"
    alt, img = core.get_payload()
    assert alt.get_content_type() == "multipart/alternative"
    assert img.get_content_type() == "image/png" and img["Content-ID"] == "<chart1>"
    assert 'filename="chart1.png"' in img["Content-Disposition"]
    assert att.get_content_type() == "text/html" and 'filename="full.html"' in att["Content-Disposition"]
    assert att.get_payload(decode=True).decode() == "<h1>full</h1>"
    assert "(+1 attachment)" in capsys.readouterr().out


def test_send_images_without_attachments_is_related_root(mailer, monkeypatch):
    _creds(monkeypatch)
    sm.send("S", "t", "h", inline_images={"a": b"\x89PNG" + b"0" * 10})
    assert _parse(_SMTP.instances[0])[2].get_content_type() == "multipart/related"


def test_send_attachment_without_images_is_mixed_with_alternative_first(mailer, monkeypatch):
    _creds(monkeypatch)
    sm.send("S", "t", "h", attachments=[("a.html", "x")])
    msg = _parse(_SMTP.instances[0])[2]
    assert msg.get_content_type() == "multipart/mixed"
    assert msg.get_payload()[0].get_content_type() == "multipart/alternative"


def test_send_writes_marker_after_success(mailer, monkeypatch):
    _creds(monkeypatch)
    sm.send("Subj", "t", "h")
    m = json.loads((mailer / "state" / "last_brief_sent.json").read_text())
    assert m["subject"] == "Subj" and m["to"] == "me@example.com" and m["pid"] > 0
    assert m["sent_at"].count(":") == 2 and "." not in m["sent_at"]  # timespec=seconds


def test_send_marker_failure_does_not_fail_the_send(mailer, monkeypatch, capsys):
    _creds(monkeypatch)
    (mailer / "state").write_text("a file, not a directory")
    assert sm.send("S", "t", "h") is True
    assert "send marker not written" in capsys.readouterr().out


def test_send_smtp_failure_propagates_and_writes_no_marker(mailer, monkeypatch):
    _creds(monkeypatch)

    class Boom(_SMTP):
        def sendmail(self, *a):
            raise OSError("connection reset")

    monkeypatch.setattr(sm.smtplib, "SMTP_SSL", Boom)
    with pytest.raises(OSError):
        sm.send("S", "t", "h")
    assert not (mailer / "state").exists()


# ---------------------------------------------------------------- _technical_section
class _Cur:
    def __init__(self, rows):
        self.rows, self.sql = rows, None

    def execute(self, sql):
        self.sql = sql

    def fetchall(self):
        return self.rows


def _fake_pg(monkeypatch, rows, connect=True):
    cur = _Cur(rows)
    pg = types.SimpleNamespace(connect=lambda: connect,
                               get_connection=lambda: types.SimpleNamespace(cursor=lambda: cur))
    wm = types.SimpleNamespace(norm=lambda m: str(m).upper(),
                               technical_markets=lambda tier: {2: ["EU", "JP"], 3: ["HK", "SG"]}[tier])
    monkeypatch.setitem(sys.modules, "pg_client", pg)
    monkeypatch.setitem(sys.modules, "watchlist_markets", wm)
    return cur


def test_technical_section_formats_tiers_and_skips_missing_markets(monkeypatch):
    cur = _fake_pg(monkeypatch, [("eu", date(2026, 7, 30), 12, 300), ("JP", date(2026, 7, 31), 5, 100),
                                 ("HK", date(2026, 7, 29), 3, 50)])
    html = sm._technical_section()
    assert "Tier 2 — daily, technical only:</b> EU 12/300 (30 Jul) · JP 5/100 (31 Jul)" in html
    assert "Tier 3 — weekly, technical only:</b> HK 3/50 (29 Jul)" in html and "SG" not in html
    assert "darvas_signal" in cur.sql and "market_daily.snapshots" in cur.sql


def test_technical_section_empty_cases(monkeypatch):
    _fake_pg(monkeypatch, [])
    assert sm._technical_section() == ""
    _fake_pg(monkeypatch, [("EU", date(2026, 7, 30), 1, 2)], connect=False)
    assert sm._technical_section() == ""
    _fake_pg(monkeypatch, [("EU", date(2026, 7, 30), 1, 2)])
    html = sm._technical_section()
    assert "Tier 2" in html and "Tier 3" not in html  # a tier with no data is omitted


def test_technical_section_is_failure_isolated(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pg_client", None)
    assert sm._technical_section() == ""
    assert "technical section skipped" in capsys.readouterr().out


# ---------------------------------------------------------------- _digest_section
def test_digest_section_failure_renders_error_strip(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "watchlist_digest", None)
    html, subj, full, pngs = sm._digest_section()
    assert "watchlist digest failed this morning" in html and subj == "" and full is None and pngs == {}
    assert "digest section failed (brief still sent)" in capsys.readouterr().out


def _fake_digest_stack(monkeypatch, tmp_path, rows, changed=False, viz_boom=False, bundles=True,
                       learned_boom=False, news_boom=False):
    wl = pd.DataFrame({"symbol": ["A", "B", "C"], "market": ["IN", "US", "JP"], "status": ["watch"] * 3})
    wl.to_csv(tmp_path / "watchlist.csv", index=False)
    log = {}

    def render(rows_, as_of, purged=None, full=False, images=None, bundles=None):
        log.setdefault("render", []).append((full, images, bundles, as_of))
        return "FULL" if full else "SHORT"

    def maintain(w):
        log["maintain_in"] = list(w["symbol"])
        return w, ["EVICT1"], ["PURGE1"], changed

    def annotate_learned(r):
        if learned_boom:
            raise RuntimeError("model missing")

    W = types.SimpleNamespace(
        __file__=str(tmp_path / "watchlist_digest.py"), maintain=maintain,
        build_rows=lambda w, frames_out: (frames_out.update({"k": 1}), rows)[1],
        assign_sectors=lambda r: None, assign_recommendations=lambda r: None,
        annotate_learned=annotate_learned, render=render, zone_series=object())
    WM = types.SimpleNamespace(covered=lambda m: m != "JP", restrict=lambda df: df[df["market"] != "JP"],
                               COVERED=["IN", "US"])
    monkeypatch.setitem(sys.modules, "watchlist_digest", W)
    monkeypatch.setitem(sys.modules, "watchlist_markets", WM)

    def sell(rows_):
        if news_boom:
            raise RuntimeError("no feed")

    monkeypatch.setitem(sys.modules, "sell_news", types.SimpleNamespace(annotate_sell_news=sell))

    def build_all(rows_, frames, zs):
        if viz_boom:
            raise RuntimeError("matplotlib missing")
        return {"chart1": b"PNGDATA"}

    monkeypatch.setitem(sys.modules, "watchlist_viz", types.SimpleNamespace(build_all=build_all))
    PB = types.SimpleNamespace(maybe_build=lambda a, f: {"built": "2026-07-01"},
                               value=lambda s, a, f: [{"b": 1}, {"b": 2}] if bundles else [])
    monkeypatch.setitem(sys.modules, "portfolio_bundles", PB)
    return log


def _rows():
    return [
        {"status": "watch", "last": "2026-07-30", "zone": "BUY", "days_in": 1},
        {"status": "signal", "last": "2026-07-31", "rec": "BUY", "zone": "HOLD", "days_in": 5},
        {"status": "justified", "last": None, "zone": "HOLD", "days_in": None},
        {"status": "watch", "missing": True, "last": "2026-08-01", "zone": "BUY"},
        {"status": "watch", "below_floor": True, "last": "2026-08-02", "zone": "BUY"},
        {"status": "held", "last": "2026-08-03", "zone": "BUY"},
    ]


def test_digest_section_happy_path_counts_and_renders_twice(monkeypatch, tmp_path, capsys):
    log = _fake_digest_stack(monkeypatch, tmp_path, _rows())
    html, subj, full, pngs = sm._digest_section()
    assert html == "SHORT" and full == "FULL" and pngs == {"chart1": b"PNGDATA"}
    # 3 picks (watch, signal, justified); buy-zone uses rec over zone -> 2 BUY; 1 new (days_in <= 1)
    assert subj == " + 📊 Picks (3 · 2 buy-zone · 2 bundles)"
    short, fullc = log["render"]
    assert short[0] is False and short[1] == {"chart1": "cid:chart1"} and short[3] == "2026-08-03"
    assert fullc[0] is True and fullc[1]["chart1"].startswith("data:image/png;base64,")
    assert log["maintain_in"] == ["A", "B"]  # JP is held back by the coverage gate
    out = capsys.readouterr().out
    assert "coverage gate held back 1 row(s) outside IN/US" in out
    assert "evicted EVICT1" in out and "purged PURGE1" in out and "bundles: 2 model portfolios" in out


def test_digest_section_writes_watchlist_back_only_when_changed_and_keeps_parked(monkeypatch, tmp_path):
    _fake_digest_stack(monkeypatch, tmp_path, _rows(), changed=False)
    sm._digest_section()
    assert pd.read_csv(tmp_path / "watchlist.csv")["symbol"].tolist() == ["A", "B", "C"]
    (tmp_path / "watchlist.csv").unlink()
    _fake_digest_stack(monkeypatch, tmp_path, _rows(), changed=True)
    sm._digest_section()
    assert pd.read_csv(tmp_path / "watchlist.csv")["symbol"].tolist() == ["A", "B", "C"]  # JP row re-appended


def test_digest_optional_blocks_are_failure_isolated(monkeypatch, tmp_path, capsys):
    _fake_digest_stack(monkeypatch, tmp_path, _rows(), viz_boom=True, learned_boom=True, news_boom=True, bundles=False)
    html, subj, full, pngs = sm._digest_section()
    assert html == "SHORT" and pngs == {} and subj.endswith("0 bundles)")
    out = capsys.readouterr().out
    assert "learned recs skipped (model missing)" in out and "sell-news skipped (no feed)" in out
    assert "digest charts skipped: matplotlib missing" in out
    assert "bundles:" not in out
