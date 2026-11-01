# Scorecard: version 2.0 to version 3.0

What changed between the rebuilt analyzer (tag `v2.0.0`, commit `ae5b1ca`) and this release, measured rather than
claimed, and the free resources it was built from. Timings come from a shared 4-core Linux container, so treat them
as ratios more than absolute figures.

## Before and after

| Measure | v2.0.0 | v3.0.0 |
| --- | --- | --- |
| Automated tests | 3,357 | 5,004 |
| Implied volatility, 10,000 quotes | 7.65 s (Brent, one quote at a time) | 15 ms (vectorised Let's Be Rational) |
| Implied volatility, one quote | 590–840 µs | 19 µs |
| Implied-volatility accuracy | Brent root finder, no accuracy study | Within max(1e-14, 4·cond·ε) in volatility on all 66,738 quotes with time value, out of 97,086 cases priced at 60 digits with mpmath (worst 0.27 of that tolerance) |
| American put, error on the 4,495-option grid of Healy (2021) against QuantLib's QdFp engine at high precision | Leisen–Reimer tree, 2,000 steps: RMSE 1.5e-4, max 1.1e-3, 13.9 ms per option | Andersen–Lake–Offengenden engine ("accurate"): RMSE 1.6e-7, max 3.7e-6, about 0.9 ms per option |
| American put, a 41-strike strip at one expiry | 41 trees | One exercise boundary, reused: 0.8 ms |
| App: page settled after loading | 6.3–6.5 s | 3.0–3.3 s |
| App: data sent on first load | 354 KB | 56 KB |
| App: recompute after a spot change | 1.45–1.54 s, 283 KB | 0.59–0.71 s, 56 KB |
| App: first number on screen at 1440 × 900 | 174 px from the top, tabs at 303 px | 95 px, tabs at 203 px |

How each number was measured:

- Tests: `pytest --collect-only` on each tag.
- Implied volatility: `python benchmarks/bench_iv.py`; accuracy from `tests/test_lets_be_rational.py` (the 100,000-case
  study is marked `slow`).
- American options: `python benchmarks/bench_american.py`, `tests/test_american.py` and the oracle tests in
  `tests/oracle/` (QuantLib 1.44). The tree figures are the same grid priced with `binomial.price(..., american=True,
  method="lr")`.
- App: Chromium driven by Playwright, three loads each at 1440 × 900, before and after the change that made only the
  open tab compute.

## What version 3.0 adds

- **Engines.** A vectorised port of Jäckel's *Let's Be Rational* (revision 1520) for implied volatility, with a batch
  API that returns a reason for every rejected quote; and the Andersen–Lake–Offengenden spectral-collocation engine
  for American options, with a cached exercise boundary, three accuracy schemes and an honest fallback (flagged) for
  the double-boundary regime. In the app it is the "American (ALO)" pricing model, with smooth Greeks, the exercise
  boundary and American implied volatility.
- **Risk analytics.** Position risk in dollar terms with second-order Greeks (vanna, volga, charm, veta), a Greek
  ladder by full revaluation, a spot × vol × time scenario cube, vega by expiry and delta bucket, one-month-weighted
  vega, and strikes chosen by delta.
- **Market time and data.** A New York market clock (NYSE holidays from their rules, AM and PM settlement,
  minute-based year fractions, a variance clock with event days); a realistic synthetic option chain with exchange
  strike intervals, tick sizes, spreads and OCC symbols; adapters for chains users download themselves; and a data
  and licensing policy that the tests enforce (no scraped data, no third-party market data in the repository).
- **Interface.** A compact desk header (contract ticker, model badge, share link), tabs bound to the URL that compute
  only when opened, one card and chart system across tabs, self-hosted IBM Plex type, and a Methodology reference with
  verified citations that opens beside any tab.
- **Engineering.** Oracle tests against QuantLib and Jäckel's reference code that fail CI if skipped; 60-digit
  mpmath accuracy studies; a test-time network guard; an oldest-dependencies CI job; parallel test runs.

## Since version 3.0

The numbers a desk would have caught first (docs/improvement-guide.md section 2), fixed and measured:

| Measure | v3.0.0 | Now |
| --- | --- | --- |
| Automated tests (`pytest --collect-only`) | 5,004 | 6,127 (6,125 pass and 2 strict xfails on the recording platform; the 76 browser and network tests run nightly in CI, and CI collects 108 fewer, the golden surface's bit-identity checks) |
| 1-day at-the-money put (S 5,800, σ 18 %), cost of the next day | θ/365 = −10.66 on a 21.56 premium | Decay to expiry −21.56 (full revaluation), shown first |
| 7-day iron condor, P&L of one day | θ/365 = +$137.50 | +$151.97 by revaluation, beside θ/365 |
| 45-day short straddle, 99 % 1-day VaR (100,000 paths, seed 7) | $2,833 (implied vol frozen by default) | $3,862 with implied-vol moves by default, the spot-only $2,833 beside it |
| Quotes a few ulps above intrinsic (20,000 random double-precision quotes) | 9.5 % inverted more than 1 % off, up to 32× | Rejected as `no_time_value` (2,259); the rest within 7.7e-6 |
| Fed 2026 severely adverse vol shock | ×4.0 | ×2.73 (72.0 / 26.4, the Fed's table 2.A) |
| Monte Carlo 95 % interval with about 10 / 30 / 100 paths on the scarce side | Shown; covered 87.8 % / 92.2 % / 94.0 % | Withheld with the reason below 300 such paths |
| Synthetic SPX chain (spot 5,000) read with the Treasury rate and no dividend yield | Sidebar carry for every expiry: forwards 142 bp off; 49 quotes dropped as outside the no-arbitrage bounds; 62 calendar and 49 butterfly arbitrage points in the fit | Parity forwards and the arbitrage-free SVI fit: forwards within 0.22 bp, at-the-money vols within 0.0064 vol point, no quote dropped, no arbitrage, every fitted vol inside its bid/ask |
| 200 random noisy arbitrage-free surfaces, arbitrage left after the fit | Not constrained | None; no expiry's RMSE more than 8.6 % above its unconstrained fit |
| Local vol at listed monthly expiries, noise-free SSVI quotes, largest error | 2.1 vol points at the front (secants between expiries) | 0.10 vol point (dw/dT from the smiles' scaling), with a per-maturity error estimate |
| Tree decay against the closed form (501 steps, σ 20 %, strikes 80–120, 2 days to 10 years) | Not shown | Leisen–Reimer within 4.0e-6; Cox–Ross–Rubinstein within 1.5 % (interpolated trees; the same steps were off by 8.5 %) |

The desk redesign of every tab, measured in Chromium on one machine with one script (v3.0.0 on its only market, the
textbook one; now on the default synthetic index at 23-Oct-26 16:00):

| Measure (1,440 × 900, sidebar open) | v3.0.0 | Now |
| --- | --- | --- |
| Risk tab: page height; top of its first headline number | 4,612 px; 4,092 px (three VaR cards under six charts) | 837 px; 299 px (six pinned tiles, one report at a time) |
| Risk tab on a 390 px phone: page height | 5,630 px | 1,781 px |
| Strategy tab: top of the summary tiles; top of the payoff chart | 672 px; 1,002 px | 207 px; 455 px (the spec's limits: 320 and 480) |
| Surface tab: page height; top of the smile | 2,611 px; 712 px | 1,388 px; 372 px |
| Time from opening a tab to an idle page, first visit | 2.5 to 6.4 s | 2.6 to 3.4 s |
| Numbers written by hand in the tabs (the DK-4 lint) | Not checked | None: every tab formats through the number standard, enforced by a test |

What the desk build sends and how long a page takes to settle, measured with `python scripts/app_payload.py` (3
repeats; websocket bytes as the page receives them, after the per-message compression is undone; time from the click
or navigation to the last DOM change before 600 ms of quiet, Streamlit idle; 1,440 × 900, dark, on the synthetic
index at 23-Oct-26 16:00), on 1-Nov-26 before the audit's fixes (commit `21863b7`) and after them, on the shared
4-core container (load averages 4.1 and 2.0 to 2.8, so the timings are indicative; the byte counts did not vary
between repeats). v3.0.0's 56 KB a load and a rerun (above) came from another script on the one textbook page.

| Measure | Data sent, before → after | Time to a settled page, median, before → after |
| --- | --- | --- |
| First load after the server starts (Pricing) | 67 → 73 KB | 6.3 → 5.7 s |
| First load of a tab in a fresh page: Pricing, Strategy, Risk | 67, 157, 52 → 73, 112, 54 KB | 3.9, 4.3, 2.9 → 3.8, 3.8, 3.6 s |
| First load of a tab in a fresh page: Surface, IV, Methodology | 88, 48, 96 → 89, 49, 109 KB | 4.2, 2.9, 1.9 → 3.9, 3.0, 2.0 s |
| Pricing: strike step of 5 points | 53 → 58 KB | 0.95 → 0.90 s |
| Tab switch from Pricing to Strategy | 142 → 97 KB | 1.44 → 1.27 s |
| Strategy: what-if spot step | 143 → 97 KB | 0.68 → 0.78 s |
| Strategy: template change | 142 → 95 KB | 1.09 → 1.13 s |

The P&L chart's grid of 301 points instead of 600 takes about a third off every Strategy rerun. The websocket's
per-message compression, on since the audit, about halves the bytes on the wire besides (a Pricing strike step 58
→ 25 KB, a Strategy what-if step 143 → 76 KB). The spec's latency budgets (0.6 s for a strike step or a tab switch,
0.5 s for a what-if step) are still missed on this machine; ROADMAP DK-7 tracks them.

## Free-resources ledger

Everything the project uses is free.

| Resource | Licence | Used for |
| --- | --- | --- |
| Python, NumPy, SciPy, pandas | PSF, BSD | The numerical core |
| Streamlit and Streamlit Community Cloud | Apache 2.0; free hosting | The app and its deployment |
| Plotly | MIT | Charts |
| QuantLib | BSD 3-Clause | Test oracle only (European, binomial, finite-difference and QdFp American engines) |
| Let's Be Rational reference code (P. Jäckel) and `py_lets_be_rational` | Permissive notice; MIT | The port in `lets_be_rational.py` (notice in `THIRD_PARTY_NOTICES.md`) and a test reference |
| mpmath | BSD | 60-digit reference values |
| pytest, pytest-xdist, ruff | MIT | Tests and linting |
| Hypothesis | MPL 2.0 | Property tests |
| fontTools and Brotli | MIT | Font-metric tests (reading the self-hosted WOFF2 fonts) |
| Pillow | MIT-CMU | Shrinking the README screenshots |
| Playwright and Chromium | Apache 2.0, BSD | Browser tests and screenshots |
| IBM Plex Sans and Mono | SIL Open Font License | Self-hosted type |
| GitHub and GitHub Actions | Free for public repositories | Code hosting and CI |
| Papers: Jäckel (2015) *Let's Be Rational*; Andersen, Lake and Offengenden (2016); Healy (2021); Leisen and Reimer (1996); Gatheral (2006); Gatheral and Jacquier (2014); Acerbi and Tasche (2002) | Publicly available | Methods; full references in the app's Methodology tab |
| NYSE holiday calendar, Cboe and OCC product specifications | Public web pages | The market clock, settlement rules and OCC symbols |

Development was AI-assisted: the code, tests and documents were written with an AI coding assistant (on a paid
plan) under the author's direction. Every result above is reproducible from the repository with the free
tools listed.
