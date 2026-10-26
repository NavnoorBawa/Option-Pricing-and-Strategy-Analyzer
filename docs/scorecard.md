# Scorecard: version 2.0 to version 3.0

What changed between the rebuilt analyzer (tag `v2.0.0`, commit `ae5b1ca`) and this release, measured rather than
claimed, and the free resources it was built from. Timings come from a shared 4-core Linux container, so treat them
as ratios more than absolute figures.

## Before and after

| Measure | v2.0.0 | v3.0.0 |
| --- | --- | --- |
| Automated tests | 3,357 | 4,953 |
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
  the double-boundary regime.
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
| Playwright and Chromium | Apache 2.0, BSD | Browser tests and screenshots |
| IBM Plex Sans and Mono | SIL Open Font License | Self-hosted type |
| GitHub and GitHub Actions | Free for public repositories | Code hosting and CI |
| Papers: Jäckel (2015) *Let's Be Rational*; Andersen, Lake and Offengenden (2016); Healy (2021); Leisen and Reimer (1996); Gatheral (2006); Gatheral and Jacquier (2014); Acerbi and Tasche (2002) | Publicly available | Methods; full references in the app's Methodology tab and in `docs/methods/` |
| NYSE holiday calendar, Cboe and OCC product specifications | Public web pages | The market clock, settlement rules and OCC symbols |

Development was AI-assisted: the code, tests and documents were written with Claude Code (an AI coding assistant,
on a paid plan) under the author's direction. Every result above is reproducible from the repository with the free
tools listed.
