# Improvement guide

How to take this project from version 3.0 to the standard of an internal options desk tool at a top-tier trading
firm. Written for whoever works on it next, human or AI agent, so that every session starts from the same facts
and the same priorities. `ROADMAP.md` is the checklist; this guide explains what matters most, why, and how to
work so that quality holds.

## 1. Start every session here

1. Install: `pip install -r requirements-oracle.txt brotli fonttools playwright`. Chromium for Playwright is
   usually preinstalled under `/opt/pw-browsers`; never run `playwright install`.
2. Gates, all must pass before and after your change:
   `ruff check . && ruff format --check . && python -m pytest -q -n auto --dist loadfile`
   Expected noise: about 103 skips in `tests/test_surface_golden.py` (bit-exact goldens skip on a CPU other than
   the recording one; the tolerance versions still run) and 4 strict xfails that mark SF-3, SF-4 and PR-5 work.
3. Read `ROADMAP.md` (the "Next session" note at its top, if any, says what is in progress), `git log -20`, and the
   detailed plan for the item you take: `git show ee47346:docs/roadmap/plan-items.md` and, for interface work,
   `git show ee47346:docs/design/ui-spec.md` (Appendix B records decisions already taken). Read them from history;
   do not add them back to the tree.
4. Work on a branch, commit and push after every verified step (cloud containers reset without warning), then
   fast-forward `main` and watch the GitHub Actions run. CI runs only on pushes to `main` and on pull requests.

## 2. What a top desk would notice first

Four independent reviewers (a market maker, a quant developer, a model-risk validator and a product designer) went
through the shipped project with the code and the running app; their reports with evidence are in
`docs/reviews/panel-v3.0.md`. What follows ranks their findings together with the roadmap. Fix in this order unless
`ROADMAP.md` says otherwise.

### Wrong or misleading numbers (fix first, each is small)

1. **"Theta / day" is instantaneous θ/365, wrong by 2× inside three days to expiry and meaningless for 0DTE.** A
   1-DTE at-the-money put worth 21.56 shows −10.66/day; the true one-day decay is the whole premium. Show one-day
   decay by full revaluation (`position_value(t=1/365)` exists) beside θ/365, and switch the headline to it when the
   two differ by more than 10 %. (Market maker, finding 3.)
2. **VaR ignores vega by default.** The "Simulate implied-vol moves" toggle is off, so a 45-day short straddle's
   99 % VaR reads $2,833 instead of $3,862 with the app's own suggested index dynamics: a 27 % understatement on the
   most common vol position. Default it on; show spot-only and spot-plus-vol VaR side by side. (MM 2, RK-5.)
3. **The implied-vol solver "solves" quotes with no time value.** A quote a few ulps above intrinsic inverts to an
   arbitrary 0–219 % with reason `""`; 9.5 % of random double-precision quotes come back more than 1 % off. Treat
   `otm_price <= 1e-12·max(S, K)` as no time value, add a `no_time_value` reason and a `vega` field to the batch
   result, and pin the behaviour on both sides of intrinsic. (Quant dev 1; sharpens SF-4/SF-5.)
4. **The Fed severely-adverse scenario overstates the vol shock by about 47 %:** `vol_multiplier=4.0` where the
   Fed's jump-off VIX of 26.4 (its table 2.A: the quarter's highest close) and peak of 72.0 give 2.73. Fix the
   multiplier, state the jump-off, and pin every historical scenario to its sourced number with a test. (Validator 3.)
5. **Monte Carlo intervals under-cover when few paths finish in the money** (coverage 0.85–0.91 below about 300
   in-the-money paths) and the app shows them with no flag. Flag "interval unreliable" below that threshold and make
   the coverage test a two-sided binomial check across in-the-money levels. (Validator 2; this is PR-6's threshold.)
6. **Local volatility at the front expiry is off by up to 2 vol points on a listed monthly grid with noise-free
   quotes**, because ∂w/∂T is a secant between expiries. Use a monotone interpolant in T (PCHIP) and report the
   truncation error; add the listed-expiry grid as a test with a 0.5-point bound. (Validator 1; not on the roadmap.)
7. **Expiry dates ignore the exchange calendar** (`days=4` gives a Saturday) and the market clock is not wired to
   the app at all. (MM 4, designer 10; this is DT-2, whose library half is done.)

### The market, not the textbook (largest value)

8. **Forwards and discounting come from the sidebar.** The project's own arbitrage-free synthetic chain, fitted with
   the generator's own rate and yield, is reported as 136 quotes outside the bounds, 21 calendar violations and
   1.36 vol points of RMSE, because the forward is `S·e^{(r−q)T}` rather than the chain's parity forward. SF-3 first,
   with this exact run as its acceptance test (zero drops, zero violations, RMSE within the noise); SF-4 and SF-8
   follow. (Quant dev 2.)
9. **Strategy legs are priced at one flat vol while the Surface tab already builds a skewed surface** it never
   reads; a default iron condor sells 33–36-delta strikes with a 45.6 % probability of profit, which no desk would
   do. Leg vols from the published surface (BK-3), strikes by delta with 20Δ/5Δ defaults (RK-9a; the library has
   `build_template(strikes_by="delta")`), then surface shocks (RK-7) and smile-implied probabilities (UI-13).
   (MM 6, 7.)
10. **The "done" desk-risk library is unreachable from the app:** `greek_ladder`, `vega_buckets`, `weighted_vega`,
    `strike_for_delta`, `sample_chain` and `read_chain` have zero call sites under `options_analyzer/ui/`. Uploading
    the repository's own Cboe-layout fixture to the Surface tab shows a raw pandas parser error because the uploader
    never calls `read_chain`. Wire each one (Risk: ladder and vega-bucket tables; Strategy: strikes-by-delta
    selector; Surface: `read_chain` first with its notes shown, and a "load synthetic chain" button). Until then,
    mark these roadmap items "library only". (MM 5, designer 1.)
11. **Risk-tab scenario grids are unusable for short-dated positions:** spot ±30 % in 5 % steps puts the whole ±2σ
    region of a 7-day straddle inside one row. Default the spot axis to multiples of σ√T (as the time-decay chart
    already does), make vol shocks absolute vol points, and show the one-day-rolled grid by default. (MM 1; UI-4/UI-9.)
12. **VaR, P&L explain and backtests:** add the quadrature oracle for option VaR/ES that the validator wrote (the
    engine passes it), then RK-6 analytic delta-gamma VaR, RK-3 P&L explain, RK-4 and RK-5 backtests, all published
    through the validation report. (Validator 4.)

### The interface (what a user sees in the first ten seconds)

13. **No tab shows its primary chart above the fold at 1440 × 900:** the Strategy P&L chart starts at 1,002 px, the
    Risk VaR section at 3,935 px, the Surface smile at 1,700 px below two half-empty 3-D plots. Two-column Strategy
    (legs and cards left, chart right), a Risk headline strip with a report selector (BK-1), Surface parameters in a
    popover with the smile first and the 3-D plots demoted (SF-6). Acceptance: first chart top under 600 px on
    every tab. (Designer 2, 4, 12.)
14. **Five number standards on one screen** (quantity `1.00` and `1`, dollars at 4, 2 and 0 decimals, Greeks at 6,
    4 and 2). Publish the standard as a table in `format.py`, make every column_config and card use it, and add a
    test that checks the formats against the table. (Designer 3; DK-4/UI-3.)
15. **The option chain is hidden in a collapsed expander** with a fixed 5-point step (a 1-day SPX chain shows
    `9.29e-134`), no delta or IV columns and no ATM highlight; the IV tab takes one price with no bid/ask, so there
    is no bid/ask vol or spread in points. Make the chain the Pricing tab's main table at exchange strike intervals
    over ±3σ√T; add bid/ask inputs to the IV tab. (MM 8, 9; designer 7.)
16. **Phones:** scenario heatmap cells at 9 px, the legs editor needing a hidden sideways scroll, tables cut mid-word.
    Rule: no chart text below 11 px; under 600 px render legs as stacked cards. (Designer 5, 9; UI-6.)
17. **Smaller polish worth doing in one pass:** explanatory paragraphs that belong in the Methodology drawer;
    checkbox widgets in a read-only stress table; unlabelled strike lines on the P&L chart; the ALO badge wrapping
    the ticker and shifting the page; Monte Carlo intervals hidden in a caption; one shared dollar-Greek strip for
    Strategy and Risk. (Designer 6, 8, 11, 13, 14, 15.)

### Validation and engineering (cheap, and reviewers look for them)

18. **Oracle gaps with the tests already written by the validator:** QuantLib European prices, Greeks and trees
    (compare puts, since calls are priced as symmetric puts, and use ALO "high" as the American tree reference, not
    QuantLib's erratic LR tree); American Greeks against QdFp with the time-scaling trick for theta (QuantLib's FD
    theta is wrong near expiry, by 5× at 2 days); the double-boundary fallback over 6–8 cases with a T-dependent
    tolerance (its error is 1e-4 beyond a year, not the documented 1e-5). (Validator 5, 6, 7; VL-1.)
19. **Tighten tolerances that are loose relative to the engines' measured error** (tree theta `abs=0.02` on a theta
    of −6) to measured error × 3, with the measurement in the test comment; mark the four 13–21 s tests `slow`; add
    a nightly workflow with `HYPOTHESIS_PROFILE=nightly`, the slow tests and the oracles, and move `slow` out of the
    push job. (Validator 8, 13; quant dev 8; VL-2, VL-11.)
20. **Typing gate is cheaper than it looks:** 141 of 147 mypy errors come from one unannotated helper
    (`black_scholes._prepare`); annotate it, add `[tool.mypy]` strict on the engines that are already clean, ship
    `py.typed`, add the CI job. Coverage is 94–99 % everywhere except `american.py` (89 %, mostly overflow branches):
    set the gate now, per package. (Quant dev 5, 7; VL-4, VL-5.)
21. **Reproducible deployment:** unpinned runtime ranges, no lock file, no Python pin for Streamlit Cloud, and
    `python-dateutil` imported but not declared. Compile a lock file, declare the dependency, pin the cloud Python.
    (Quant dev 6; not on the roadmap.)
22. **Method notes and the validation report:** only the American engine has a method note; write
    `docs/methods/{lets_be_rational,binomial,monte_carlo,var,vol_surface}.md` in the same structure, each with a test
    that reproduces its headline numbers; have VL-7's generator emit the tables the validator produced (oracle error
    by scheme, Greek errors, MC coverage by in-the-money paths, VaR/ES against quadrature, local-vol error by expiry
    spacing) and fail CI when one exceeds its tolerance. (Validator 9, 11.)
23. **One result shape across engines:** three Greek result types and two calling conventions today, with per-engine
    adapters in the UI; one frozen `Greeks` mapping and one `price(...)` signature, old names kept as wrappers for a
    release. Return snake_case columns from `run_stress_tests` and `leg_table` as the module's own convention says.
    Promote the `lets_be_rational` internals the solver relies on to public functions. (Quant dev 3, 10, 11.)
24. **Honesty fixes:** README says the implied-vol accuracy holds on "97,086 cases"; the bound is proven on the
    66,738 quotes with time value, as the scorecard says. Say so, and say what happens to the rest. (Validator 10.)

## 3. Rules that keep the quality up

- **Independent oracles over self-consistency.** A test that compares the code with itself proves little. Use
  QuantLib, mpmath at 40–60 digits, published tables (Healy 2021, Haug 2007) and closed forms. Mark them `oracle`;
  the CI oracle job fails if one is skipped.
- **State tolerances and where they come from.** "Within 4·cond·ε" with the conditioning defined beats "approx".
- **Timing tests are loose guards, not benchmarks.** GitHub's shared runners are 2–3× slower than a laptop; put
  measured timings in `benchmarks/` and in `docs/scorecard.md`, and assert only against an order-of-magnitude
  slowdown.
- **Every engine ships with its method note** in `docs/methods/` (model, numerical method, error, limits,
  references with DOI or stable URL) and a paragraph in the app's Methodology.
- **Honest documentation.** README, scorecard and Methodology describe only what the code and tests show, and say
  what is not done. Remove a claim before you ship a change that invalidates it.
- **Streamlit floor 1.65.** Every API used must exist there (the oldest-dependencies CI job runs it). Check
  `st.<name>` signatures in a 1.65 venv before using something new.
- **Plain language in the UI**, units on every number, no emoji, one minus sign (U+2212), losses negative.
- **Licences.** Permissive or OFL only; GPL code only as a source of reference numbers. Keep
  `THIRD_PARTY_NOTICES.md` current. The public app uses synthetic data and uploads only; never scrape market data
  (`docs/data_policy.md`, enforced by tests).
- **Never** force-push, rewrite `main`'s history, leave `main` red, or put an AI model name or identifier in the
  repository or in commit messages.
- **Review your own diff as a hostile reviewer would** before shipping; if you can run subagents, have one verify
  the item independently against the acceptance tests.

## 4. How to judge the interface

Open every tab in Chromium at 1440 × 900 and 390 × 844, dark and light, and ask what a senior options market
maker and a principal product designer would say. Concretely: is the first number a user needs on screen without
scrolling; are the units and conventions the desk's (vol points, bp, delta-strike axes, 25-delta risk reversal and
butterfly, dollar Greeks); is anything computed that the open tab does not need; does any text explain instead of
show; do charts share one house style; do phones get the same information in a narrower layout; is stale state
marked while the app recomputes. Record what you find as new `ROADMAP.md` items with screenshot evidence under a
scratch folder, never in the tree.

## 5. What "done" looks like

`ROADMAP.md` › "The bar" lists six criteria. Each is shown by a test or a measurement in the repository, not by a
sentence in a document. When all six hold, update the scorecard against the `v2.0.0` and `v3.0.0` baselines, tag a
release, and refresh the screenshots with `scripts/screenshots.py`.
