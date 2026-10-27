# Roadmap

Where the project stands and what remains before it meets the standard of an internal options desk tool at a
top-tier trading firm. Item codes (for example `SF-3`) refer to the detailed plan with acceptance tests, kept in the
git history: `git show ee47346:docs/roadmap/plan-items.md`; the per-tab interface specification is at
`git show ee47346:docs/design/ui-spec.md` (its Appendix B records decisions already taken).
[docs/improvement-guide.md](docs/improvement-guide.md) explains what matters most, why, and how to work on it.

## The bar

The project is done when all of these hold, each shown by a test or a measurement in the repository:

1. **Numerics.** Every engine is checked in CI against an independent oracle (QuantLib, mpmath or a published table)
   with stated tolerances; property tests cover the no-arbitrage identities; mutation testing kills at least 90 % of
   non-cosmetic mutants in the pricing core.
2. **Market realism.** Forwards and discount factors come from put–call parity per expiry; the volatility surface is
   fitted to bid/ask with spread weights and no static arbitrage; probabilities and strategy legs use the fitted
   smile; discrete dividends, a trading-time clock and same-day (0DTE) expiries are supported in the app.
3. **Risk.** Full-revaluation scenario cubes, P&L explain, and VaR/ES whose backtests (Kupiec, Christoffersen, Basel
   traffic light, Acerbi–Szekely) are published in a generated validation report.
4. **Interface.** Every "must" item of the UI specification is built, with browser acceptance tests and performance
   budgets enforced in CI, on desktop and phone, in light and dark.
5. **Engineering.** Typed core (mypy strict on the numerical modules), a coverage gate of at least 95 %, pinned and
   audited CI actions, a documentation site, a changelog, citation metadata and benchmark history.
6. **Honesty.** README, scorecard and Methodology describe only what the code and tests show, and say what is not
   done.

## Done (version 3.0)

- Desk design system: theme tokens, self-hosted IBM Plex, chart house style, number formats (DK-1 to DK-4).
- App shell: shared cards, one chart entry point, model settings split out, Methodology split by area (P0-2); tabs
  bound to the URL that compute only when opened, and a UI-free position builder (P0-1); compact desk header with
  contract ticker, model badge and share link (DK-5); Methodology reference with verified citations and a drawer
  beside any tab (DK-6).
- Build and CI: pytest-xdist, oldest-dependencies job, oracle job that fails when skipped, network guard (P0-3).
- Let's Be Rational implied volatility, scalar and batch, with a 60-digit accuracy study (PR-1).
- Andersen–Lake–Offengenden American engine validated against QuantLib (PR-2), available in the app as the
  "American (ALO)" model with Greeks, the exercise boundary and American implied volatility (most of PR-3 and PR-4).
- Position risk with dollar and second-order Greeks, Greek ladder, scenario cube, vega buckets, one-month-weighted
  vega, strikes by delta, input validation (RK-1, RK-2, RK-9).
- Market clock, synthetic option chain, chain adapters, enforced data policy (DT-1, DT-4, DT-5, DT-6).
- Volatility-surface code split into a package with golden-output tests (SF-2).

## Remaining

### Volatility surface
- [ ] SF-3 Forwards and discount factors per expiry from put–call parity, Black-76 inversion (turns the strict xfails
  in `tests/test_market_data.py` into passes).
- [ ] SF-4 Spread-weighted SVI calibration constrained against butterfly and calendar arbitrage.
- [ ] SF-8 Wing extrapolation within Lee's moment bound, flagged in charts.
- [ ] SF-1 Surface tab message severity and empty cells.
- [ ] SF-5 Vectorised inversion at chain scale and de-Americanised quotes.
- [ ] SF-6 Desk conventions on the Surface tab: delta axes, 25-delta risk reversal and butterfly, term structure,
  fit diagnostics.
- [ ] SF-11 Data-source panel, carry diagnostics, and a published fitted surface that other tabs read.
- [ ] SF-9 Trading-time clock for short-dated expiries.
- [ ] SF-7 Andreasen–Huge arbitrage-free interpolation (optional precision mode).
- [ ] SF-10 Quote-level static arbitrage with minimal L1 repair (could).

### Pricing
- [ ] PR-4 (rest) Batch American implied volatility and a de-Americanisation API in the library.
- [ ] PR-5 Discrete cash dividends with a finite-difference engine on the spot model (removes the escrowed-dividend
  approximation and its strict xfail).
- [ ] PR-6 Monte Carlo honesty (reliability flag, coverage study) and randomised quasi-Monte Carlo.
- [ ] PR-7 Pricing and IV tabs in the desk layout of the UI specification.
- [ ] RK-12 American revaluation in Strategy and Risk through an ALO price grid.
- [ ] PR-8 Heston with the Albrecher characteristic function and calibration (could).

### Risk
- [ ] RK-6 Exact delta-gamma-theta VaR as the analytic comparator.
- [ ] RK-3 P&L explain: second-order Taylor waterfall and Shapley attribution.
- [ ] RK-4 VaR/ES backtesting library and a synthetic market-history generator.
- [ ] RK-5 Historical and filtered historical simulation, calibrated dynamics, realistic defaults.
- [ ] RK-7 Volatility-surface shocks and smile regimes, legs read from the fitted surface.
- [ ] RK-8 Margin-style risk arrays (FINRA 4210(g) portfolio margin; SPAN-like array as a library function).
- [ ] RK-11 Futures legs and per-leg exercise style.
- [ ] RK-10 Euler ES contributions by leg and expiry (could).

### Strategy and Risk interface
- [ ] BK-1 Risk console: headline strip and report selector.
- [ ] BK-2 Strategy desk layout: builder bar, package market, blotter, dated P&L curves, Greek strip.
- [ ] BK-3 Leg implied vols from the fitted surface.
- [ ] UI-4 Shared what-if strip; UI-9 date × spot P&L table; UI-10 apply a scenario cell; UI-13 risk-neutral ranges
  from the smile; UI-8 structure steppers and delta-hedge ticket; UI-24 undo.
- [ ] BK-4 Position CSV import, expiry ladder, pin risk (could).

### App shell and data
- [ ] UI-1 Price cards move into the Pricing tab; UI-2 market-level header; UI-3 and DK-4 one number standard
  everywhere; UI-5 stale-state marking; UI-6 phone pass; UI-22 colour roles; UI-28 provenance in every export.
- [ ] DT-2 Sidebar: expiry dates, intraday valuation time, 0DTE, presets (synthetic index default, textbook preset).
- [ ] DT-2a Cboe trading sessions; DT-5a quote times from uploaded files; DT-10 SPXW daily listings; RK-9a strikes
  by smile delta snapped to listed strikes.
- [ ] DT-3 Treasury curve (Federal Reserve H.15) and cash dividend schedules.
- [ ] UI-23 Trader marks on the smile; UI-26 model-free 30-day vol; UI-27 realised vol and breakeven move; DT-9 and
  UI-25 earnings events.
- [ ] DK-7 Browser regression and performance-budget suite in CI.
- [ ] DK-8 Keyboard shortcuts and export menus.
- [ ] DT-7 Local-only live adapters; DT-8 weekly network canary (could).

### Validation and engineering
- [ ] VL-1 QuantLib oracle tests for European prices and Greeks and for binomial trees (LBR and ALO oracles exist).
- [ ] VL-2 Hypothesis property suite on the documented domain.
- [ ] VL-3 CI hygiene: actions pinned to commit SHAs, zizmor clean, pre-commit, nox, macOS and Windows jobs.
- [ ] VL-4 Typing gate (mypy, `py.typed`).
- [ ] VL-5 Coverage gate and badge.
- [ ] VL-6 Benchmarks tracked over time.
- [ ] VL-7 Generated validation report in a model-risk structure, shown in Methodology › Validation.
- [ ] VL-8 Scorecard generated by a script from the tagged baseline.
- [ ] VL-9 Documentation site; VL-10 changelog, citation file and release workflow; VL-11 mutation testing and a
  nightly run.

### Known limitations to remove
- Strategy legs are priced as European options with Black–Scholes–Merton at flat leg volatilities.
- Probabilities in the Strategy tab are log-normal.
- App expiries are whole calendar days (the library supports fractional days).
- Browser acceptance tests run locally only (`pytest -m e2e`), not in CI.
- Bit-exact surface golden tests skip on CPUs other than the recording one; the tolerance versions run everywhere.
