## Validation

Every engine is checked by the project's automated tests, which continuous integration runs on Python 3.10 to 3.14
and against the oldest supported dependency versions. The figures below are the tests' measured results; the file
that holds each check is named in brackets.

### Independent oracles
* **American options.** The Andersen–Lake–Offengenden engine prices the 4,495 puts of Healy (2021, Table 5) with an
  RMSE of 1.6e-7 and a largest error of 3.7e-6 against QuantLib's QdFp engine at high precision, in about 0.9 ms
  an option; a 2,000-step Leisen–Reimer tree gives an RMSE of 1.5e-4 in 14 ms. The grid is generated with QuantLib
  and stored in the repository, so the main suite needs no QuantLib (`tests/test_american.py`,
  `tests/oracle/test_ql_alo.py`).
* **Implied volatility.** The port of Let's Be Rational is checked against Jäckel's own reference implementation
  (`tests/oracle/test_lbr_reference.py`). Continuous integration installs QuantLib and the reference code and fails
  if any oracle test is skipped.

### High-precision references
* Implied volatility is within ${\max(10^{-14},\ 4\,\text{cond}\,\varepsilon)}$ of the true vol on all 66,738 quotes
  with time value among 97,086 cases priced at 60 digits with mpmath (the worst at 0.27 of that tolerance); the
  rest round to intrinsic. On 20,000 random double-precision quotes, every quote with time value above
  ${10^{-12}\max(S, K)}$ inverts to within 1e-5 of the vol that priced it (the worst 7.7e-6), and the other 2,259
  are refused as having no time value (`tests/test_lets_be_rational.py`, `tests/test_implied_vol.py`).
* The Greeks are checked against 40-digit mpmath derivatives (`tests/test_black_scholes.py`).
* The trees' one-day decay is checked against the closed form at 501 steps, volatility 20%, strikes 80 to 120 and
  2 days to 10 years: Leisen–Reimer within a relative 4.0e-6, Cox–Ross–Rubinstein within 1.5%
  (`tests/test_time_decay.py`).

### Coverage studies
The Monte Carlo 95% interval's coverage is checked with binomial tests at 10 to 1,000 paths on the scarce side of
the strike: with about 10, 30 and 100 such paths it covered 87.8%, 92.2% and 94.0% of the time, so the app withholds
the interval, with the reason, below 300 of them (`tests/test_monte_carlo.py`). The VaR backtests' Kupiec and
Christoffersen statistics are checked against their formulas term by term (`tests/test_backtest.py`).

### The volatility surface
* On the synthetic SPX chain read back as an upload, the parity forwards are within 0.22 bp of the generator's and
  the at-the-money vols within 0.0064 vol point, with no quote dropped, no arbitrage and every fitted vol inside
  its bid/ask (`tests/test_forwards.py`).
* On 200 random noisy arbitrage-free surfaces the arbitrage-free fit leaves no arbitrage, and no expiry's RMSE is
  more than 8.6% above its unconstrained fit (`tests/test_svi_constraints.py`).
* Local vol at listed monthly expiries, from noise-free SSVI quotes, is within 0.10 vol point
  (`tests/test_vol_surface.py`).

### Invariants
Put–call parity and symmetry, the no-arbitrage bounds, monotonicity, the Greeks against finite differences, the
Greek ladder's identities across all 26 strategy templates and the vega-bucket sums are tested as properties
(`tests/test_greek_ladder.py`, `tests/test_vega_buckets.py` and the engines' own test files).

### The app
Streamlit's headless tests run every tab and deep link (`tests/test_app*.py`), and browser tests in Chromium check
the layout at desktop and phone widths, in light and dark themes (`tests/test_theme.py` and the `tests/test_*_e2e.py`
files, run locally with `pytest -m e2e`). A test enforces the data policy: the public app fetches no market data.
