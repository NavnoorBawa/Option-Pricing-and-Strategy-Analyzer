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

Ranked by how much each gap costs a professional user. Fix in this order unless `ROADMAP.md` says otherwise.

1. **Forwards and discounting come from the sidebar, not the market.** Every surface fit uses one rate and one
   dividend yield. A desk reads the forward and discount factor per expiry from put–call parity, and the
   synthetic chain already exposes the error (the strict xfails in `tests/test_market_data.py`: forwards off by
   about 1.5 points at 11 days). Do SF-3 first; SF-4 (spread-weighted, arbitrage-constrained SVI) and SF-8 (wings
   within Lee's bound) follow directly.
2. **Strategy legs ignore the smile.** Legs are priced with Black–Scholes–Merton at a flat vol, probabilities are
   log-normal, and the Risk tab's scenarios move spot and vol independently of the surface. BK-3 (leg vols from
   the fitted surface), RK-7 (surface shocks, sticky-strike and sticky-delta) and UI-13 (risk-neutral ranges from
   the smile) turn the Strategy and Risk tabs from a textbook into a desk tool.
3. **VaR has no backtest and no analytic comparator.** A VaR number without Kupiec, Christoffersen and the Basel
   traffic light is unverifiable; without delta-gamma VaR beside it, a user cannot see how much of the number is
   non-linearity. RK-6, RK-4 and RK-5, published through the validation report (VL-7).
4. **No P&L explain.** A risk tab that cannot attribute a day's P&L to delta, gamma, vega, theta and the residual
   is not a risk tab. RK-3 (second-order Taylor waterfall plus Shapley attribution, which has no order dependence).
5. **The interface still shows the textbook, not the market.** Price cards describe one contract on every tab,
   defaults are S = K = 100 and 20 % vol, the Pricing tab has no chain, and numbers are formatted inconsistently.
   UI-1, UI-2, DK-4 with UI-3, DT-2 (synthetic index preset, expiry dates, intraday time, 0DTE), then PR-7, BK-1,
   BK-2, SF-6 and SF-11 from the UI specification.
6. **Discrete dividends are approximated.** The synthetic ETF chain uses the escrowed-dividend model (its put bias
   is documented and xfailed). PR-5 adds a finite-difference engine on the spot model with cash dividends.
7. **Validation has gaps a model-risk reviewer would flag.** European prices, Greeks and the trees have no
   QuantLib oracle in CI (only the American engine and Let's Be Rational do); there is no property-based suite;
   no typing gate; no coverage gate; no generated validation report. VL-1, VL-2, VL-4, VL-5, VL-7, then VL-11
   (mutation testing).
8. **Monte Carlo reports an interval even when it is unreliable** (few in-the-money paths), and has no
   quasi-random option. PR-6.
9. **Engineering hygiene.** CI actions by tag not SHA, no pre-commit, no benchmark history, no changelog, no
   documentation site. VL-3, VL-6, VL-9, VL-10. Cheap, and reviewers look for them.

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
