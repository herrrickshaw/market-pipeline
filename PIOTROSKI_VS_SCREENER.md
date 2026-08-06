# Our Piotroski F-Score vs screener.in's Piotroski filter

**Measured 2026-08-06 against the live screen. Universe: India (NSE).**

Reproducible artifact: [`reports/piotroski_vs_screener_20260806.csv`](reports/piotroski_vs_screener_20260806.csv) — 168 rows, our score, their score, delta, liquidity tier, ROCE, P/E.

---

## Summary

On the 168 names where our production India scan and screener.in's Piotroski filter both have an opinion:

| metric | value |
|---|--:|
| agree at ≥7 | **86.9%** |
| agree at ≥8 | **58.3%** |
| exact score match | **42.3%** (71/168) |
| mean delta (ours − theirs) | **−0.60** |
| we score lower on | 47% of names |
| we score higher on | 11% of names (never by more than +1) |

We are systematically about half a point below screener.in on the same companies. The disagreement is concentrated, explainable, and — after the 2026-08-05 corrections — mostly methodology rather than defect.

---

## 1. What screener.in's filter actually is

Public screen [`/screens/2/piotroski-scan/`](https://www.screener.in/screens/2/piotroski-scan/), query:

```
Piotroski score > 7
```

That is **8 or 9**, not 9. As fetched 2026-08-06: 536 results, of which 398 score 8 and 137 score 9. The screen exposes screener.in's own value in a `Piotski Scr` column, so this is a score-vs-score comparison, not merely set membership.

### Correction to the existing record

`CHANGELOG.md` (entry *2026-08-05 — Piotroski tests 5 and 7 corrected in all five market scans*) describes the calibration set as **"screener.in's 9s (n=490)"**. That characterisation is wrong: roughly **74% of that set are 8s**.

This is not screen drift. The persisted snapshot `cache_seed/public_screens/piotroski_9.parquet` (captured 2026-07-14) shows the same composition — 398 eights, 143 nines. The screen has always been `> 7`.

The measurements in that entry remain valid; only the label is wrong. But the distinction matters when quoting them: *"agreement at ≥8 rose 44% → 53%"* against a set that is three-quarters 8s is a materially weaker claim than the same sentence would be against a set of pure 9s. The filename `piotroski_9.parquet` reinforces the error and is worth renaming.

---

## 2. Coverage — most of their list is out of scope by design

| | count |
|---|--:|
| screener.in `>7` names | 535 fetched (536 stated) |
| BSE-only numeric codes (no NSE symbol) | 204 — outside our NSE universe |
| NSE-style symbols | 331 |
| **present in our production scan** | **168** |
| NSE-style but absent from our scan | 163 |

About **69% of their list is unreachable for us on purpose.** Our India scan runs ~1,380 names after the ₹1cr policy liquidity floor; screener.in screens the full listed universe with no liquidity filter. The 163 NSE-style names we lack are below that floor.

This is a deliberate design difference, not a coverage failure. Any headline "we only match N of their 536" that does not net out the liquidity floor and the BSE-only codes is misleading.

---

## 3. Agreement on the 168 overlap

| their score | n | our mean | we ≥7 | we ≥8 | exact |
|---|--:|--:|--:|--:|--:|
| 8 | 135 | 7.47 | 85.9% | 54.1% | 40.7% |
| 9 | 33 | 8.15 | 90.9% | 75.8% | 48.5% |
| **all** | **168** | **7.60** | **86.9%** | **58.3%** | **42.3%** |

Delta distribution (ours − theirs):

```
−4:  3     −3:  6     −2: 18     −1: 52     0: 71     +1: 18
```

We never exceed them by more than one point, but fall as much as four below. Agreement is notably better on their 9s than their 8s, which is the expected shape: a 9 requires every test to pass, leaving less room for an input-driven divergence.

**Reverse direction:** of our 186 names scoring ≥8, only **98 (52.7%)** appear on their list.

---

## 4. This confirms the 2026-08-05 fix is live

Today's workbook (`indian_full_scan_20260806_0630.xlsx`) yields:

| threshold | count | CHANGELOG post-fix figure |
|---|--:|--:|
| ≥7 | 405 | 405 ✅ |
| ≥8 | 186 | 186 ✅ |
| ==9 | 41 | 41 ✅ |

and the measured 86.9% agreement at ≥7 lands on the 86% that entry predicted. The ΔLeverage and no-dilution corrections are in production and behaving as measured.

---

## 5. Where the residual gap comes from

### Financials dominate the tail

The worst disagreements are banks, insurers and NBFCs, where the F-Score's leverage and asset-turnover tests are close to meaningless:

| symbol | name | ours | theirs | ROCE % |
|---|---|--:|--:|--:|
| IOB | Indian Overseas Bank | 4 | 8 | 6.07 |
| IDBI | IDBI Bank | 5 | 8 | 5.92 |
| GICRE | General Insurance Corp | 5 | 8 | 17.35 |
| FUSION | Fusion Finance | 5 | 8 | 5.92 |

A sector guard for financials would close more of the gap than any change to the nine tests.

### Everything else is the input source, not the formula

We score India off **yfinance annual statements**; screener.in scores off **filed standalone/consolidated statements**. Known friction:

- **Gross Profit** is frequently absent or differently defined in yfinance's India coverage, silently costing the ΔGross-margin point.
- yfinance annual data lags the latest filed annual result, so we can be comparing an older fiscal pair.

Deliberately **not** changed (per CHANGELOG): tests 8 (ΔGross margin) and 9 (ΔAsset turnover). We are stricter than screener.in on capex-cycle names and that is correct behaviour, not a bug.

If full reconcilability is ever the goal, the lever is the input source — score India off the screener.in / XBRL fundamentals already collected — not the formula.

---

## 6. Method notes

- screener.in's `/screen/raw/` and `/ratios/` are **login-gated**; both `SCREENER_EMAIL` and `SCREENER_PASSWORD` are empty in the credential store. No login was attempted. The public saved screen is readable anonymously and was used instead.
- The fetch was paced at **4s/page** across 22 pages. `screener_in.fetch_screen()` has no inter-page sleep and would risk the ~2.6 req/min throttle documented in `compare_screener_vs_yfinance.py`.
- Join key is the bare NSE symbol. BSE numeric codes cannot join and are reported separately rather than silently dropped.

## 7. Open items

1. Rename `piotroski_9.parquet` and correct the "screener.in's 9s" phrasing in `CHANGELOG.md`.
2. Consider a financials guard (skip or reweight leverage/turnover tests for banks, insurers, NBFCs).
3. The 2026-08-05 fix did **not** reach every Piotroski implementation in the tree — `screener_analysis.py`, `us_market_screener.py`, `backtest_screeners.py`, `walk_forward_backtest.py`, `stock_ddd/domain/screening/specifications.py` and several daily-report scripts still carry the pre-fix tests 5/7, so they now disagree with the five patched scans.
