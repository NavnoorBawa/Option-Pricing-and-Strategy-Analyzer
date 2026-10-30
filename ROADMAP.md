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

## Done since 3.0

The wrong numbers of [docs/improvement-guide.md](docs/improvement-guide.md) section 2, items 1 to 5, each reviewed by
an independent adversarial check before it merged:

- Theta per day beside the one-day decay by full revaluation, which leads when the two differ by more than 10 %
  (item 1). On a tree the decay is the difference of two trees: Leisen–Reimer keeps its steps (within 4.0e-6 of the
  closed form at 501 steps and 20 % vol); Cox–Ross–Rubinstein interpolates between the two trees whose time steps
  bracket this one's (within 1.5 %, where the same steps were off by up to 8.5 %).
- VaR with implied-vol moves on by default, beside the spot-only VaR (item 2): frozen implied vol understated a
  45-day short straddle's 99 % VaR by 27 % ($2,833 against $3,862; 24 % to 27 % over seeds 0 to 9).
- Implied vol rejects quotes whose time value is lost in rounding as `no_time_value`, in the European, tree and ALO
  solvers, and the batch result carries the vega (item 3): on 20,000 random quotes the worst error on the rest is
  7.7e-6.
- Stress scenarios pinned to their sourced closes by a test; the Fed 2026 vol shock is 72.0 / 26.4 = 2.73, not 4.0
  (item 4).
- Monte Carlo intervals withheld below 300 paths on the scarce side of the strike or 5,000 independent samples, with
  the coverage study as binomial tests (item 5; the reliability part of PR-6).

## Remaining

### Volatility surface
- [x] SF-3 Forwards and discount factors per expiry from put–call parity, Black-76 inversion. On the synthetic SPX
  chain read with the Treasury rate, forwards land within 0.22 bp and at-the-money vols within 0.0064 vol point, with
  no arbitrage; the sidebar forward was 85 to 142 bp off. American chains are read at the given rates (parity does not
  hold for them), with a message saying so.
- [x] SF-4 Spread-weighted SVI calibration constrained against butterfly and calendar arbitrage (SLSQP with analytic
  gradients and cutting planes). 200 random noisy surfaces end with no static arbitrage, no expiry's RMSE more than
  8.6 % above its unconstrained fit; Gatheral and Jacquier's arbitrageable slice is fitted 0.63 vol point away with a
  positive density. The SF-3 acceptance is restated in spread terms: on the synthetic chain every fitted vol is inside
  the bid/ask and the median error is 0.13 to 0.22 of the half-spread. An expiry where no arbitrage-free smile is
  found keeps its unconstrained fit with a note, never silently.
- [x] SF-8 Wing extrapolation within Lee's moment bound (wing slope at most 2), out to four at-the-money standard
  deviations and at most |ln K/F| = 1.5, flagged as extrapolated in the readout and drawn dotted on the smiles.
- [ ] SF-1 Surface tab message severity and empty cells.
- [ ] SF-5 Vectorised inversion at chain scale and de-Americanised quotes.
- [ ] SF-6 Desk conventions on the Surface tab: delta axes, 25-delta risk reversal and butterfly, term structure,
  fit diagnostics.
- [ ] SF-11 Data-source panel, carry diagnostics, and a published fitted surface that other tabs read. Done: the
  synthetic chain's fitted surface is published to the sidebar's contract (its vol, forward and carry) and shown by
  default on the Surface tab, with the chain's quotes for download. Left: the data panel, filters and the FIT flag.
- [ ] SF-9 Trading-time clock for short-dated expiries.
- [ ] SF-7 Andreasen–Huge arbitrage-free interpolation (optional precision mode).
- [ ] SF-10 Quote-level static arbitrage with minimal L1 repair (could).

### Pricing
- [ ] PR-4 (rest) Batch American implied volatility and a de-Americanisation API in the library.
- [ ] PR-5 Discrete cash dividends with a finite-difference engine on the spot model (removes the escrowed-dividend
  approximation and its strict xfail).
- [ ] PR-6 Monte Carlo honesty: the reliability flag and coverage study are done; randomised quasi-Monte Carlo, the
  small-σ ultima limit, NaN input validation and a flag for heavy-tailed samples (coverage 0.94 at σ√T of 2.5 to 3)
  remain.
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
  Done: the Synthetic index market (default) and the Textbook one, a valuation time on 5-minute steps defaulting to
  the last cash close, listed expiries with their settlement, strikes on the listed grid with an ATMF button, the
  surface's vol or a marked override, the carry implied by parity, and links that carry all of it. Left: an ETF
  (American) preset, 0DTE listings (DT-10), and a typed spot under a chain.
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
