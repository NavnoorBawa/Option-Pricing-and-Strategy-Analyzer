# Options Pricing & Strategy Analyzer

[![CI](https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://option-pricing-and-strategy-analyzer.streamlit.app)

**Live app: <https://option-pricing-and-strategy-analyzer.streamlit.app>**

An options analytics workstation built with Streamlit. It prices European and American options, shows Greeks to
third order, builds and analyses any multi-leg strategy, stress-tests it and measures its Value-at-Risk, and turns
implied-volatility quotes into arbitrage-checked surfaces and local volatility.

The numerical core is a standalone, tested Python package (`options_analyzer`, needing only NumPy, SciPy and pandas).
Its engines are validated against independent libraries (QuantLib, Jäckel's Let's Be Rational, 60-digit mpmath), and
the Streamlit app is a thin interface on top of it.

![Strategy](docs/images/strategy.png)

## Highlights

- **Implied volatility in 19 µs per quote, 15 ms per 10,000**: a vectorised port of Jäckel's *Let's Be Rational*
  (revision 1520), accurate to a few ulps of the problem's own conditioning on 97,086 cases checked at 60 digits.
- **American options by spectral collocation**: the Andersen–Lake–Offengenden engine prices the 4,495-put grid of
  Healy (2021) with an RMSE of 1.6e-7 against QuantLib's high-precision engine, in under a millisecond per option. A
  2,000-step binomial tree gives an RMSE of 1.5e-4 in 14 ms. In the app it is the "American (ALO)" pricing model, with
  smooth Greeks, the early-exercise boundary and American implied volatility.
- **Desk-style risk**: dollar and second-order position Greeks, a Greek ladder by full revaluation, vega by expiry
  and delta bucket, one-month-weighted vega, and strikes chosen by delta.
- **Market time done properly**: a New York clock with the NYSE calendar, AM and PM settlement, minute-based year
  fractions and event variance, plus a realistic synthetic option chain and adapters for downloaded chains.
- **A fast, linkable desk interface**: a compact header with the contract ticker and model, tabs bound to the URL that
  compute only when opened (the page settles in half the time and sends a sixth of the data), and a Methodology reference that opens beside any
  tab.

The [scorecard](docs/scorecard.md) lists every before-and-after measurement and the free resources the project is
built from.

## Features

**Pricing**
- Black–Scholes–Merton with continuous dividend yield, exact at every edge case (expiry, zero volatility).
- American (ALO): the Andersen–Lake–Offengenden engine with fast, accurate and high schemes, Greeks by finite
  differences of the engine (closed form where the option is European, one-sided where a bump crosses an exercise
  regime, and the app says which), the early-exercise premium and the exercise boundary, and a plain statement
  whenever the engine falls back to a tree.
- Binomial trees: Leisen–Reimer (≈O(1/n²) convergence for European exercise) and Cox–Ross–Rubinstein, European or
  American exercise, with bump-and-reprice tree Greeks, convergence charts and the early-exercise boundary.
- Monte Carlo with exact terminal sampling, antithetic and control variates, reproducible seeds and 95 % confidence
  intervals.
- All 14 Greeks (delta … ultima, plus rho and dividend rho) for calls and puts, with explicit units, a sensitivity
  explorer and spot × volatility price heatmaps.

**Strategy**
- 26 templates (verticals, straddles and strangles, iron butterfly and condor, butterflies, ratio backspreads,
  collars, jade lizard, synthetics, calendars and diagonals) or any custom combination of calls, puts and stock.
- Every leg is editable (side, size, strike, expiry, implied volatility and fill price), and edits survive tab
  switches and input changes. Contract multiplier and commissions are included.
- P&L at expiry, today and at any date in between, with an implied-volatility shift, exact breakevens, unlimited-risk
  detection, probability of profit, expected P&L and position and per-leg Greeks.

**Risk** (for the position built on the Strategy tab, or the template named in the link)
- P&L heatmaps of spot × implied vol and spot × date, and time-decay curves, all from full revaluation.
- Stress tests calibrated to Black Monday, the dot-com bust, 2008, Volmageddon, the COVID crash, the 2022 rate shock,
  the April 2025 tariff shock and the Fed's 2026 severely adverse scenario, plus custom shocks.
- Monte Carlo VaR and Expected Shortfall with Normal, Student-t or GARCH(1,1) returns and optional implied-vol
  dynamics, compared with delta-normal VaR.

**Surface and IV**
- Arbitrage-free SSVI surfaces or your own quotes (implied vols, or option prices inverted with Black–Scholes–Merton),
  fitted with an SVI smile per expiry and turned into local volatility with Gatheral's formula; calendar and butterfly
  arbitrage are detected and reported, never silently patched.
- An implied-volatility solver for European and American quotes that checks the no-arbitrage bounds first; under the
  American (ALO) model it inverts the engine's price.

**Desk shell**
- A header with the contract ticker (spot, strike, expiry and days, vol, rates, forward), the model in use and a
  Share link that reproduces the exact view: the open tab, the sidebar inputs and the strategy template.
- Only the open tab computes, so switching tabs and changing inputs stays fast. Old links keep working.
- The Methodology tab documents every model, formula and assumption with verified references, and opens as a drawer
  beside whichever tab is open.

| Pricing | Risk |
| --- | --- |
| ![Pricing](docs/images/pricing.png) | ![Risk](docs/images/risk.png) |
| **Surface** | **IV** |
| ![Surface](docs/images/surface.png) | ![Implied volatility](docs/images/iv.png) |

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer.git
cd Option-Pricing-and-Strategy-Analyzer
pip install -r requirements.txt
streamlit run main.py
```

The app opens at <http://localhost:8501>. A dev container (VS Code / GitHub Codespaces) is included and starts the
app automatically. To install it as a package instead, `pip install ".[app]"` and run `options-analyzer`;
`pip install .` installs only the numerical core.

## Using the library

```python
import numpy as np
from options_analyzer import american, black_scholes as bs, implied_vol, scenarios, strategies as st

bs.price(100, 105, 0.5, 0.04, 0.22, q=0.01, option_type="put")  # 8.1319
american.price(100, 105, 0.5, 0.04, 0.22, 0.01, option_type="put")  # 8.3696 (Andersen-Lake-Offengenden)
american.greeks(100, 105, 0.5, 0.04, 0.22, 0.01, option_type="put")["delta"]  # -0.5768, smooth American Greeks
implied_vol.implied_volatility(8.50, 100, 105, 0.5, 0.04, 0.01, "put")  # 0.2332

strikes = np.array([90, 95, 100, 105, 110.0])
calls = bs.price(100, strikes, 0.5, 0.04, 0.22, q=0.01, option_type="call")
implied_vol.implied_volatility_batch(calls, 100, strikes, 0.5, 0.04, 0.01, "call")  # 0.22 for every strike

market = st.Market(spot=100, rate=0.04, volatility=0.22, expiry=0.5, dividend_yield=0.01)
condor = st.build_template("Iron Condor", strike=100, width=5)
st.analyze(condor, market).breakevens  # [91.909, 108.091]
st.position_risk(condor, market).dollar_delta  # -0.411, delta times spot
scenarios.greek_ladder(condor, market)  # P&L and Greeks at spot -20 % … +20 %, by full revaluation
st.strike_for_delta(-0.16, market, 0.5, 0.22, "put")  # 88.07, the 16-delta put strike
```

`options_analyzer.clock` holds the market clock, and `options_analyzer.market_data` holds the synthetic chain
(`sample_chain`) and the chain adapters (`read_chain`).

## Validation

- **Independent oracles** (`pytest -m oracle`): American prices against QuantLib's QdFp and finite-difference
  engines, and implied volatilities against Jäckel's own reference implementation. CI installs the oracle libraries
  and fails if any oracle test is skipped.
- **High-precision references**: Greeks against 40-digit mpmath derivatives, 100,000 implied-volatility cases priced at
  60 digits, and the 4,495-put American grid of Healy (2021) generated with QuantLib and stored in the repository, so
  the main suite needs no QuantLib.
- **Invariants**: put–call parity and symmetry, no-arbitrage bounds, monotonicity, Greeks against finite
  differences, the Greek-ladder identities across all 26 strategy templates, and the vega-bucket sums.
- **App tests**: Streamlit AppTests of every tab and deep link, plus Chromium browser tests of the layout, run locally
  with `pytest -m e2e`.
- The app's Methodology tab documents every model with its references, and
  [`docs/methods/american_alo.md`](docs/methods/american_alo.md) states the American engine's error and limits.

## Conventions

- Time to expiry is calendar days / 365 in the app; the library's clock also gives minute-based year fractions to
  the settlement time. Rates and dividend yield are continuously compounded.
- Greeks are displayed per calendar day (theta, charm, veta, color), per volatility point (vega, vanna, zomma …), per
  1 % rate move (rho) and per $1 spot move (delta, gamma, speed).
- VaR horizons are in trading days (252 per year).
- Strategy quantities are signed (+ long, − short); one unit is an option on one share. P&L is shown for the whole
  position (contracts × multiplier).

## Project structure

```
main.py                     Streamlit entry point (also Streamlit Community Cloud's)
options_analyzer/
  black_scholes.py          BSM prices and analytical Greeks
  binomial.py               CRR / Leisen-Reimer trees, American exercise
  american.py               Andersen-Lake-Offengenden American engine
  monte_carlo.py            Variance-reduced Monte Carlo
  lets_be_rational.py       Jäckel's Let's Be Rational, vectorised
  implied_vol.py            Bounds-checked implied volatility, scalar and batch
  strategies.py             Leg-based strategy engine, templates, position risk
  scenarios.py              Stress tests, scenario grids, Greek ladder, vega buckets
  risk.py                   Monte Carlo VaR / Expected Shortfall
  vol_surface/              SVI, SSVI, quote parsing, surface fitting, local volatility
  clock.py                  Market clock: NYSE calendar, settlement times, variance clock
  market_data/              Synthetic option chain, chain adapters, schema and OCC symbols
  ui/                       Streamlit interface (one module per tab, desk header, Methodology)
tests/                      pytest suite (engines, oracles, invariants, app); tests/oracle needs QuantLib
benchmarks/                 Timing scripts for the engines
docs/                       Scorecard, method notes, data policy, screenshots
```

## Development

```bash
pip install -r requirements-dev.txt     # app dependencies plus pinned test and lint tools
pytest -n auto --dist loadfile           # 5,000 tests in parallel
pip install -r requirements-oracle.txt && pytest -m oracle   # the QuantLib, Let's Be Rational and mpmath oracles
ruff check . && ruff format --check .
```

Continuous integration runs the linters, the suite on Python 3.10–3.14, the oracle tests, the suite against the
oldest supported dependency versions, and a wheel build. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Data

The public app uses synthetic data and files you upload; it never scrapes market data. The
[data policy](docs/data_policy.md) lists each source's terms, and the tests enforce it.

## Limitations

- Strategy valuation uses Black–Scholes–Merton at each leg's implied volatility (no smile dynamics, European
  exercise), and probabilities are log-normal.
- The app's expiries are whole calendar days; the library supports fractional days down to same-day expiries.
- Transaction costs beyond a flat commission, margin and discrete dividends are not modelled.

[ROADMAP.md](ROADMAP.md) lists what is done and what remains, and
[docs/improvement-guide.md](docs/improvement-guide.md) says how to take it further. This project is for education and analysis only and is
**not investment advice**. Version 2.0 was a ground-up rebuild
after a full audit of the original app ([AUDIT.md](AUDIT.md)); version 3.0 is summarised in the
[scorecard](docs/scorecard.md).

## Author

Created by [Navnoor Bawa](https://www.linkedin.com/in/navnoorbawa/). Released under the [MIT License](LICENSE).
