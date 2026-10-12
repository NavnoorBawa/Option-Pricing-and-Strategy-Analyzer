# Audit report

This report records the full audit of version 1 of the Options Pricing & Strategy Analyzer (a single 4,331-line
`main.py`) and how version 2 resolves each problem. Version 2 is a ground-up rebuild; nothing of the original
implementation is carried over except its purpose, its author credit and its licence.

## How the audit was done

- **Line-by-line read** of the whole code base, README and configuration.
- **Seven domain audits** (pricing models, Greeks, strategies, implied volatility, VaR/GARCH/stress tests,
  scenario analysis, volatility surfaces). Each finding is backed by evidence from running the original code:
  finite differences against 40-digit arithmetic, reference values (Hull, QuantLib), put-call parity, round-trips,
  memory and timing measurements, and the Streamlit app driven headlessly. The most consequential findings were
  then re-checked independently; a planned second, adversarial pass over every original-code finding did not run,
  which is moot for the fixes because none of the original implementation remains.
- **Three research tracks** on current best practice: numerical methods (implied-vol solvers, trees, Monte Carlo,
  Greek conventions), risk and volatility (Gatheral local volatility, SSVI, Acerbi-Tasche VaR/ES, GARCH variance
  targeting, historical crisis data) and Streamlit/product practice compared with leading options tools.
- **Independent adversarial review of the rebuild**, whose findings were reproduced and fixed before release (see
  the last section).

The domain audits alone found **186 issues: 26 critical, 35 high,
72 medium and 53 low**. Interface, reliability and project issues are listed
separately below. Every issue is resolved in version 2.

## The most serious problems

**Numbers that were invented or wrong**
- The time-decay chart did not compute anything: it drew hard-coded curves (a comment reads "adjust to match image
  values", with constants such as -10.45 and 1.9), and the "Daily Theta / Weekly / Monthly" panel was derived from
  those curves. *Removed; time decay is now re-priced from the actual legs.*
- Three higher-order Greeks had wrong formulas (speed had an extra factor of d1 and flipped sign for OTM calls,
  zomma was off by a factor of σ√T, color was wrong under either time convention) and put charm had the wrong
  sign. *All 14 Greeks are now verified against 40-digit finite differences.*
- The dividend-yield input (and the volatility and rate term-structure and skew inputs) were collected but never
  used: a 3% yield still showed the q = 0 price. *Dividends now flow through every model; the unused inputs are
  gone.*
- The implied-volatility solver returned a hard-coded 30% for any option within 3.65 days of expiry, clamped every
  answer to 5%-100%, returned numbers for impossible quotes and failed 58.9% of a well-posed round-trip test;
  with its own default inputs it showed 5.00% instead of 3.54%. *Now 0 failures on 2,520 cases, with explicit
  errors for impossible quotes.*
- The binomial tree did not check its risk-neutral probability: at σ = 1%, r = 10% and 10 steps it priced a \$9.52
  call at **\$2,109** and produced negative put prices. *Arbitrage-checked lattices with a Leisen-Reimer option.*
- Black-Scholes at zero volatility returned undiscounted intrinsic value (a call worth \$4.88 showed \$0.00), so the
  three models disagreed on the same input.
- Strategy maximum profit and loss were read off a ±30% price grid, so unlimited risk was shown as a small finite
  number and the probability of profit collapsed to 0% or 100% when a breakeven fell outside the grid; it also
  ignored the drift term. A "Kelly criterion" with the wrong dimensions and a "Sharpe ratio" over a uniform price
  grid were presented as metrics.
- The scenario engine valued positions at expiry with fixed entry premiums, so single-option volatility, time and
  rate scenarios showed exactly zero sensitivity and spread scenarios had the wrong sign.
- The "GARCH" model mixed annual and daily units, so simulated volatility decayed towards 0.3% a year, its variance
  shocks were independent of the return shocks, and it simulated at least four days for a one-day horizon.
- VaR valued the current portfolio and the simulated portfolio with different instruments (a bull call spread
  showed a *negative* VaR), and the stress test always priced a long call whatever position was selected.
- Local volatility double-counted the dividend yield (a flat 20% surface came out between 11% and 90% with a 3%
  yield), the smoothing slider collapsed the surface to a single polynomial at every setting above 0.05, invalid
  points were silently replaced by implied volatility, and the exported "LocalVolSurface" was the ATM term
  structure copied into every strike column.

**Reliability, performance and security**
- Monte Carlo at the maximum settings allocated about 600-800 MB and took 6.5 s on every rerun (close to Streamlit
  Community Cloud's memory limit); a 1,000-step binomial tree took about a second per option; VaR took 15-42 s at
  100,000 simulations.
- Matplotlib's global pyplot state was used from Streamlit's multi-threaded runner and figures were never closed,
  so concurrent sessions could corrupt each other's charts and memory grew with every interaction.
- The app crashed with a risk-free rate below 1%, with K = 0, and with the Monte Carlo VaR box ticked (duplicate
  widget IDs); a catch-all handler and debug expanders showed raw tracebacks to users.
- `.streamlit/config.toml` disabled Streamlit's XSRF and CORS protections for every deployment.

## Before and after

| Measure | Version 1 | Version 2 |
| --- | --- | --- |
| Implied-vol round-trip failures | 58.9% of well-posed cases | 0 of 2,520 cases |
| Greek formulas verified | none (4 wrong) | 14 of 14, to ~1e-15 |
| Binomial, 1,000 steps, call + put | ~1.9 s | 8 ms (European), 33 ms (American) |
| Monte Carlo at maximum settings | 6.5 s, ~600 MB peak | 1,000,000 paths in 50 ms, 44 MB peak |
| VaR, 100,000 simulations | 15-42 s | 42 ms (Normal), 59 ms (10-day GARCH) |
| Monte Carlo standard error at equal cost | baseline | about 6x smaller (antithetic + control variate) |
| Rerun time after a widget change | ~1-2 s | ~0.4 s |
| Automated tests | 0 | 1,250+ (math, invariants, QuantLib references, UI) |

## Findings by area

Each table lists every issue found in that part of version 1 (line numbers refer to the original `main.py`),
followed by how version 2 resolves the area as a whole.

### Pricing models (Black-Scholes, binomial, Monte Carlo) (24 issues: 1 critical, 6 high, 9 medium, 8 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | The Dividend Yield input has no effect on any pricing model, so dividend-paying underlyings are mispriced | 2008-2011, 2021-2052, 15-53, 56-120, ... |
| High | Vol term structure, rate term structure and skew inputs are collected but never used | 1984-2016, 2021-2052 |
| High | black_scholes_calc gives wrong prices at sigma=0 (and sigma<0): it returns undiscounted spot intrinsic instead of discounted forward intrinsic | 38-43 |
| High | Binomial tree gives absurd and negative prices when the risk-neutral probability p falls outside [0,1] | 83-86, 106-118, 1974 |
| High | Monte Carlo at the maximum slider settings uses ~0.6-0.8 GB and ~6.5 s, recomputed on every rerun, and half of it is thrown away | 150-163, 177, 1977-1978, 2042-2052, 2... |
| High | Binomial pricer is an O(n²) pure-Python double loop with an O(n²) matrix: 1.2-1.6 s per rerun at 1000 steps | 89-120, 1974 |
| High | black_scholes_calc is scalar-only and uses scipy.stats.norm.cdf (~56 µs per scalar), forcing slow Python loops across the app | 38, 46-53 (callers 893-980, 1067-1086... |
| Medium | Monte Carlo is unseeded, uses the legacy global RNG, and the displayed price changes on every rerun | 155, 2042-2047, 866 |
| Medium | Call and put Monte Carlo prices use independent random draws: put-call parity fails and the cost doubles | 2042-2047 |
| Medium | No variance reduction in Monte Carlo, so it needs up to ~7× more simulations for the same accuracy | 155-175 |
| Medium | The Monte Carlo 'Time Steps' control does nothing for pricing accuracy and only adds cost | 150-163, 1978 |
| Medium | CRR binomial oscillates between even and odd step counts, so the displayed price flips across BS as the slider moves | 83-86, 1974 |
| Medium | No input validation in the pricing core: K=0 crashes, S or K<0 displays \$nan, T<0 is priced silently, and binomial/MC return NaN at T=0 or sigma=0 | 38-47, 83-86, 150-152, 1965-1969, 202... |
| Medium | Model prices are mixed with Black-Scholes values downstream, so outputs are inconsistent with the selected model | 2143-2150, 354-373, 2300-2321 |
| Medium | option_type and style are not validated: any string other than 'call' is priced as a put, and 'European' (capitalised) is priced as American | 40-53, 100-118, 166-169 |
| Medium | No caching of price calculations: every widget interaction reprices from scratch | 2021-2052, 4299-4303 |
| Low | The Monte Carlo '±' value is an unlabelled 1-standard-error band, shown at a different precision from the price | 2048-2051 |
| Low | Monte Carlo standard error uses the population std (ddof=0) | 175 |
| Low | Price formatting shows '\$-27.22', '\$nan' and '\$0.00' for sub-cent values | 2028, 2037, 2049-2050 |
| Low | Binomial stock tree overflows for large σ·√T·√n, and u**k·d**k accumulates error | 93-94 |
| Low | black_scholes_calc edge branch is redundant and returns Python int 0 instead of a float | 40-43 |
| Low | Unknown model_type silently falls through to Monte Carlo | 2039 |
| Low | calculate_option_prices mixes presentation with computation and returns an ad-hoc tuple | 2021-2052 |
| Low | Duplicate definitions of display_option_prices and display_greeks | 1909-1926, 2095-2112, 1928-1952, 2114... |

**Resolution.** Rebuilt as `black_scholes.py`, `binomial.py` and `monte_carlo.py`: vectorised NumPy code with the dividend yield wired through every model, exact zero-volatility and expiry limits, validated inputs, a Leisen-Reimer tree next to CRR (with arbitrage checks and a fallback lattice), exact terminal sampling with antithetic and control variates, seeded common random numbers, `ddof=1` standard errors and 95% confidence intervals. The unused term-structure and skew inputs were removed; prices are cached with `st.cache_data`.

### Greeks (26 issues: 3 critical, 2 high, 13 medium, 8 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Speed is wrong: the formula has an extra factor of d1, so the magnitude is wrong and the sign flips for OTM calls | 313 |
| Critical | Zomma is wrong by a factor of sigma*sqrt(T) | 316 |
| Critical | Color (dGamma/dt) formula is wrong under either time convention | 319 |
| High | Put charm has the wrong sign | 304-307 |
| High | Greeks shown for Binomial (American) and Monte Carlo models are always European Black-Scholes Greeks | 2140-2150, 617-661, 2425-2428 |
| Medium | Veta sign convention (d vega/d tau) contradicts theta and charm (calendar time) in the same panel | 309-310, 2161 |
| Medium | Greek units are mixed and unlabelled: theta per day, vega per 1% vol, but vanna/volga/ultima/zomma per 1.00 vol and charm/veta/color per year | 233, 240-246, 298-322, 1936-1948, 215... |
| Medium | Day-count inconsistency: theta uses /365 while the rest of the app uses 252 trading days | 242, 246, 1226-1227, 1667, 1779-1792,... |
| Medium | weekly_decay/monthly_decay multiply calendar-day theta by trading-day counts (5, 21) and understate decay by ~30% | 1224-1228 |
| Medium | sigma=0 with T>0 returns wrong, discontinuous Greeks (spot moneyness instead of forward, and theta forced to 0) | 203-218 |
| Medium | No input validation: K=0 raises ZeroDivisionError, S=0 gives NaN gamma, negative S/K give all-NaN, negative sigma/T are silently treated as expired | 203-222, 279-289 |
| Medium | Inconsistent option_type API: calculate_greeks treats anything other than exact 'Call' as a put, e.g. 'call' silently returns put Greeks | 205, 235, 243, 304 |
| Medium | Dividend yield, vol/rate term structure and skew inputs are ignored by all Greek calculations | 1980-2016, 180-328, 617-661 |
| Medium | Basic tab shows only call Greeks although call and put prices are both displayed; puts (and put charm) are never shown | 2143-2150 |
| Medium | 'Position Greeks' label is misleading: the Basic tab shows a single option's per-share Greeks with no position size or contract multiplier | 1932, 2118, 2145, 2428 |
| Medium | Rounding to 3 decimals hides real values (gamma shows 0.0 for high-priced underlyings, spread thetas show -0.0) | 1936-1948, 2122-2134, 2158-2174 |
| Medium | Strategy Greeks use hidden hard-coded strikes (0.9K, 1.1K, 0.855K, 1.155K); iron condor wings are asymmetric | 641-642, 658-661 |
| Medium | calculate_strategy_performance reports vanna/volga of a single ATM call whatever the strategy | 1133-1134, 1218-1219 |
| Low | Zomma and Ultima descriptions are wrong ('Gamma-Volga', 'Volga-Volga') | 2172, 2174 |
| Low | Rho is missing from the first-order Greeks | 180-253, 1928-1952 |
| Low | Unknown strategy names silently return all-zero Greeks, while the P&L function falls back to Long Call | 645, 663-750, 438-440 |
| Low | Dead strategy branches: 'Protected Short Sale', 'Reverse Hedge', 'Naked Call Writing' are unreachable and have no matching P&L | 673-689, 2373-2375, 4239-4240 |
| Low | At T=0 and S==K both call and put delta are 0, which violates delta parity | 205-218 |
| Low | Duplicate definitions of display_greeks and display_option_prices (the second shadows the first) | 1909-1952, 2095-2138 |
| Low | Unused and duplicated work in the strategy Greek and performance path; slow scalar SciPy calls | 648-661, 1129-1134, 1185 |
| Low | No automated tests guard the Greek formulas | 180-328, 617-750 |

**Resolution.** One analytical function returns all 14 Greeks for calls and puts with dividends, every one verified against 40-digit finite differences in the tests. All time Greeks use calendar time, units are shown next to every value, rho and dividend rho were added, the American tree's own delta/gamma/theta are shown for binomial American pricing, and strategy Greeks are summed over the actual legs.

### Strategy analysis (28 issues: 3 critical, 10 high, 9 medium, 6 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Max profit / max loss come from the edges of a finite spot grid: unlimited risk shows as a small number and bounded extremes are wrong | 1137, 1178-1182, 1292-1304, 2415, 245... |
| Critical | Profit probability collapses to 0% or 100% when a break-even lies outside the 0.7S–1.3S grid | 1137, 1146-1147, 1150-1175 |
| Critical | Kelly criterion formula is dimensionally wrong (scales with 1/price) and uses the wrong inputs | 1192-1201, 2465 |
| High | Profit probability ignores the lognormal drift term (r − σ²/2)T | 1159, 1161, 1168-1172 |
| High | 'Sharpe Ratio', 'P&L Volatility' and 'expected_pnl' are the mean/std of P&L over a uniform price grid, not over a distribution | 1137, 1187-1190, 1208, 1220-1221, 246... |
| High | T=0 and σ=0 give contradictory metrics (100% profit probability with \$0 max profit) and divide by zero | 1159-1172, 1175, 38-43 |
| High | Negative interest rates are rejected and take down the whole app | 338-340, 4323-4327 |
| High | Input validation misses K=0 (crash), S=0 (silent degenerate output), NaN/inf, and rejects numpy ints | 334-340 |
| High | Leg premiums mix sources: ATM legs use the selected model (unseeded MC, coarse or American binomial), other legs use BS, and Greeks always use BS | 354-373, 2300-2303, 2316-2317, 2425-2427 |
| High | 'Max Loss' shows a positive dollar loss for positions that never lose, and negative max profits / negative risk-reward are displayed | 1178-1182, 2450-2452 |
| High | Evaluation and chart grids are centred on spot, ignore where the strikes are, and do not contain the kinks | 1137, 2415 |
| High | Strikes are hard-coded (0.9K/1.1K, condor wings 0.855K/1.155K, asymmetric), cannot be set by the user, and are never disclosed | 343-348, 641-659, 1265-1266, 2395-2412 |
| High | matplotlib pyplot state machine and global style are not thread-safe: figures get corrupted with concurrent sessions and all other charts turn dark | 1269-1317, 1344, 2416-2421 |
| Medium | Profit calculation ignores cost of carry: premiums and stock paid today are compared with payoffs at T without compounding | 377-436 |
| Medium | Break-even points are the grid point before the sign change, not interpolated, and three different values appear on one page | 1146-1147, 1288-1289, 1309 |
| Medium | Chart strike lines are wrong: a 'Strike Price' line where no leg exists, one spread leg missing, condor wings missing | 1276-1285 |
| Medium | Figures are never closed: memory grows outside Streamlit, and inside it cleanup depends on an implicit runtime hook | 1270, 1346, 2421 |
| Medium | Strategy chart is expiry-only: there is no mark-to-market (T+0) curve, and time/vol inputs only shift entry premiums | 331-443, 1239-1316, 2415-2428 |
| Medium | 'vol_move_up' and price sensitivities are computed incorrectly (sign flips, zeros) and are never displayed | 1229-1236 |
| Medium | Strategy 'advanced Greeks' (vanna, volga) are always those of a single long ATM call | 1134, 1218-1219 |
| Medium | Unknown strategy names silently fall back to a Long Call; the Greeks function supports a different strategy set | 438-440, 664-750, 2412 |
| Medium | No caching and loop-based P&L: all strategy work (plus a 200 ms PNG render) reruns on every widget change, even when the tab isn't open | 375-440, 1129-1143, 1232-1235, 2415-2428 |
| Low | Weekly/monthly decay use 5 and 21 multipliers on a per-calendar-day theta | 1224-1228 (theta /365 at 241-246) |
| Low | Chart annotations and description text contradict each other and are missing for two strategies | 1296-1304, 1319-1344 |
| Low | Chart layout: max-profit label collides with the title, gets clipped at the right edge, and the text box overlaps the axis area | 1296-1304, 1309-1310, 1344 |
| Low | Performance panel shows misleading labels and formats and omits what matters (break-evens, net debit/credit, assumptions) | 1182, 1201, 2441-2468 |
| Low | Strategy coverage and position model fall short of standard analysers | 2370-2393, 331-443 |
| Low | 'call_value > 0' sentinel conflates 'not provided' with a legitimately zero or invalid price | 331, 354-357 |

**Resolution.** Replaced by a leg-based engine (`strategies.py`): 26 editable templates plus fully custom legs with their own strike, expiry, IV and fill price. Breakevens are exact roots, unlimited profit or loss is detected from the payoff slope (including the S -> infinity limit), the probability of profit integrates the log-normal density with drift over every profitable interval, and P&L is shown at expiry, today and any date. The Sharpe and Kelly figures were removed because they were not meaningful. Charts moved to Plotly.

### Implied volatility (23 issues: 5 critical, 1 high, 8 medium, 9 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Implied-vol solver fails the round-trip on 58.9% of well-posed cases; replace it with a bracketed, arbitrage-bounded root finder (Brent, or Jäckel's 'Let's Be Rational') | 476-614 |
| Critical | Hard-coded 'return 0.3' for T <= 0.01 fabricates a 30% implied vol for every option with 3.65 days or less to expiry (and for T<=0) | 497-499 |
| Critical | Volatility is clamped to [5%, 100%], so every IV below 5% or above 100% is silently replaced by the clamp value | 502, 563-564, 595-596, 600 |
| Critical | No no-arbitrage bound check: prices below discounted intrinsic, zero, negative, NaN, inf, or above the upper bound return 5% or 100% with a success message | 504-511, 2501-2508 |
| Critical | The tool's default inputs produce a wrong IV: a \$5.00 call at the default parameters shows 5.00% (true 3.54%) with a '-75%' delta | 2495-2496, 2503-2513 |
| High | The 20-point grid fallback returns vols quantised to 0.05 steps, reports them as success, and discards a better Newton iterate | 598-612 |
| Medium | Post-loop convergence check uses a stale price, and max_iterations=0 crashes | 567-599, 612-614 |
| Medium | Absolute price tolerance (1e-8) is scale-dependent: tiny prices 'converge' at the initial guess and low-priced underlyings get biased IVs | 476, 574-576, 2505 |
| Medium | option_type is lower-cased for the branch logic but passed raw to the case-sensitive pricer; unknown strings silently price as puts | 505, 517, 529, 551, 569, 605 (and pri... |
| Medium | No validation of S, K, T or finiteness: invalid inputs return 5% or crash with raw Python errors | 497-514, 579 |
| Medium | IV-vs-model delta metric: an ambiguous relative %, misleading green/red colouring, and a crash or 'inf%' when model vol is 0 | 2510-2516 |
| Medium | The 'Initial Volatility Guess' slider is unused and misleading | 2498-2499, 2503-2506 |
| Medium | The IV tool ignores the sidebar dividend yield and the model style (American binomial), so it inverts the wrong model | 2480-2481, 2489, 2503-2506 |
| Medium | The pricer's sigma<=0 branch returns undiscounted intrinsic, which is discontinuous with the sigma->0+ limit the IV bounds rely on | 38-43 (used by 529, 551, 569, 605) |
| Low | Ad-hoc Newton safeguards (absolute vega floor, 0.1 step cap, clamps, dead sign branch) give no convergence guarantee | 566-596 |
| Low | Initial-guess heuristics are inaccurate and scale-dependent (Brenner-Subrahmanyam misapplied, spot-based moneyness, absolute \$0.01 floor) | 504-561 |
| Low | Remove the dead, broken commented-out L-BFGS-B version and the unused scipy.optimize.minimize import | 11, 445-475 |
| Low | Button-gated result with no diagnostics: disappears on any interaction, no repricing check, bounds, vega or convergence info | 2501-2518 |
| Low | Catch-all exception handler shows raw Python internals to users | 2517-2518 |
| Low | Market price input: no lower bound, 2-decimal display, and no help text on valid range | 2495 |
| Low | Docstring and README misdescribe the algorithm and hide its silent defaults | 477-496; README.md L33 |
| Low | No automated tests for implied volatility | 476-614 |
| Low | Scalar scipy.stats.norm makes each IV call take milliseconds, which rules out smiles and surfaces | 15-53, 567-612 |

**Resolution.** `implied_vol.py` checks the no-arbitrage bounds, converts in-the-money quotes to their out-of-the-money equivalent and solves with Brent's method on a bracket: 0 failures on a 2,520-case grid (vol 1%-300%, expiries down to 17 hours), versus 58.9% before. American quotes can be inverted on the binomial tree. Impossible quotes produce an explanation, not a number.

### Value-at-Risk, GARCH and stress testing (29 issues: 3 critical, 5 high, 12 medium, 9 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Stress test always prices a 'Long Call', ignoring the Position Type the user selected in the VaR tool | 2283-2297 (handle_stress_testing hard... |
| Critical | stress_test_portfolio values every position as one long call or one long put by substring 'Call' (drops stock legs, short legs and multi-leg structures) | 1063-1070, 1078-1089 |
| Critical | GARCH recursion mixes annualised variance with daily parameters, so simulated volatility decays toward 0.3%/yr | 752-787 (omega=1e-6, alpha=0.1, beta=... |
| High | 'GARCH' variance shocks are independent of the return shocks, so this is not a GARCH model | 776-784 (own z), 846-860 (separate np... |
| High | GARCH branch simulates the wrong horizon (minimum 4 days) while revaluing at T - horizon and labelling it as 1 day | 839 (n_steps = max(int(horizon*252)+1... |
| High | var_calculator values the current and simulated portfolio with different instruments, and Bear Call / Bull Put spreads have the wrong sign | 900-907 (current value: single call/p... |
| High | Duplicate widget IDs crash the Quant tab when the Monte Carlo VaR checkbox is on (StreamlitDuplicateElementId) | 2210-2226 vs 3567-3587 (identical st.... |
| High | O(n_sim x n_steps) pure-Python loops with scalar SciPy calls: 15-42 s at 100k simulations | 846-860 (per-path, per-step scalar np... |
| Medium | Monte Carlo view recomputes VaR (and stress tests) on every rerun, with no caching | 2205-2206, 2229-2246, 2249-2250 |
| Medium | np.random.seed(42) reseeds the process-global RNG in one branch only; the GARCH branch is unseeded | 866 (seed in constant-vol branch), 77... |
| Medium | VaR revaluation holds implied vol constant (no vega risk), even in GARCH mode | 923-980 (reprices with vols[i]), 840 ... |
| Medium | Skewness/kurtosis divide by zero P&L std, showing 'nan' on reachable inputs (deep OTM strike, vol=0, T=0 OTM) | 1000-1002, 2275-2276 |
| Medium | No input validation in var_calculator: S<=0, K<=0, sigma<=0, df<=2 and confidence outside (0,1) give NaN or meaningless VaR | 821-832, 835, 856/874 (sqrt((df-2)/df... |
| Medium | Stress-test % change divides by a tiny or negative portfolio value (1e20 %, sign flips) and reports 0% when the value is 0 | 1092-1093 |
| Medium | Stressed rate is floored at 0.1%, so rate cuts at low or negative base rates become rate hikes | 1080 |
| Medium | 'Historical' stress scenarios understate the real events; the Tech_Bubble rate move has the wrong sign | 1056-1061 |
| Medium | Stress chart leaks matplotlib figures and uses the non-thread-safe pyplot state machine | 2338-2366 |
| Medium | VaR tool results disappear on any interaction (st.button pattern), and stress testing must be ticked before clicking | 3587-3613 |
| Medium | Result labels are misleading: 'Volatility' is P&L std in \$, Worst/Best use opposite sign conventions, negative VaR shown as '\$-x', GARCH horizon mislabelled | 2256-2281, 1010-1013 |
| Medium | Student-t option is presented as more conservative, but at 90-95% it lowers VaR | 2214-2215, 3575-3576 (help text), 871... |
| Low | Stress design gaps: instantaneous shocks with no time decay, multiplicative vol only, no upside or vol-crush scenarios | 1074-1089 |
| Low | Stress chart styling depends on a global style set elsewhere, and labels use absolute dollar offsets | 2343, 2349, 2352-2364 (relies on plt.... |
| Low | VaR/ES quantile convention: off-by-one index, float truncation, and ES = NaN when var_index == 0 | 993-997 |
| Low | No Monte Carlo error bars on VaR/ES; the fixed seed hides the sampling noise | 993-1017, 2252-2281 |
| Low | Model choices undocumented: risk-neutral drift with a normal convexity term under t, single t draw scaled by sqrt(h) for multi-day horizons, RMS of option IVs as the underlying vol | 835, 850, 873-874, 878 |
| Low | Monte Carlo-tab VaR is always a Long Call, is unlabelled, and ignores the sidebar simulation count | 2205-2206, 2230-2244, 2250 |
| Low | stress_test_portfolio list-broadcasting breaks for NumPy array inputs (multiplies values instead of repeating them) | 1035-1052 |
| Low | visualize_stress_test_results silently fabricates 0.0 P&L for malformed results | 2318-2325 |
| Low | GARCH parameters are hidden, uncalibrated and unvalidated; the vol array has an unused extra column | 752 (alpha=0.1, beta=0.8, omega=1e-6 ... |

**Resolution.** `risk.py` revalues the actual position (all legs) on vectorised simulations under Normal, Student-t (daily shocks, unit variance) or GARCH(1,1) with variance targeting and shared shocks, with optional implied-vol dynamics. VaR/ES use the Acerbi-Tasche estimators with a confidence interval, seeds are local, legs expiring inside the horizon settle on their expiry day, and inputs are validated. Stress tests (`scenarios.py`) re-price every leg under scenarios calibrated to published data.

### Scenario analysis (24 issues: 6 critical, 4 high, 8 medium, 6 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Scenario engine prices the position at expiry with fixed entry premiums, so its vol, time and rate scenarios are not mark-to-market. Single-leg and ATM-leg strategies show exactly zero vol, time and rate sensitivity | 1681-1682, 1711-1736, 1738-1774, 1795... |
| Critical | Vol and rate scenarios for spreads and Iron Condor have the wrong sign, because the shock reprices the entry debit or credit instead of the held position | 1714-1719, 1798-1803, 360-373, 391, 4... |
| Critical | 'Current P&L' baseline is the at-expiry P&L at unchanged spot (for example -premium), not the position's current P&L of 0 | 1681-1682, 1874, 4037, 1692, 1719, 17... |
| Critical | Time-decay chart plots hard-coded fabricated curves ('-10.45 from image', 1.9, -5.0, 1.0, decay 0.3*tf^2) instead of computed values | 3750-3847 (3754, 3764-3793, 3796-3809... |
| Critical | Time Decay Metrics panel (Daily Theta / Weekly / Monthly) shows fabricated or wrong-sign numbers | 3887-3939, 1790-1792 |
| Critical | The expiry-time special case (time_left<=0) uses wrong formulas: Covered Call gives max(0,K-S), and the spreads use a single ATM premium as the spread debit | 1743-1767 |
| High | Scenario analysis crashes whenever the risk-free rate is below 1% (including r=0) or negative | 1668, 1817, 1841, 1847, 1850 (shifts)... |
| High | risk_metrics are wrong: 'Maximum Potential Loss' is \$0 for long-premium strategies, gain is overstated by the premium, 'Worst-Case Loss' can be negative, and the risk/reward ratio is inverted | 1878-1901, 4081-4091 |
| High | The time-scenario grid is defined wrongly: it is fixed at 0-126 days of remaining life regardless of T, the 'one day' effect runs backwards at expiry, and 'one week' uses 10 days | 1667, 1741-1793, 4191 |
| High | Every rerun leaks 5 matplotlib figures (about 19 MB per click), and pyplot global state is used in a multi-session server | 3633-3671, 3681-3716, 3737-3885, 3950... |
| Medium | The time-decay chart layout breaks at default inputs because the 'Current' marker is placed off-chart and the label is anchored at ylim()[0]*0.9 | 3859-3864, 3883, 3703-3704, 3982-3983... |
| Medium | The 'historical' stress scenarios are uncalibrated and one-sided, and apply multi-month drawdowns instantly with relative vol multipliers | 1838-1863, 1813-1836 |
| Medium | Sensitivity metrics are sign-less grid averages labelled as local Greeks, and the normalised bars compare incompatible units | 1694-1700, 1721-1727, 1866-1876, 3726... |
| Medium | The Strategy Guidance HTML leaks a literal '</div></div>' code block because of blank lines from get_strategy_risk_profile | 4097-4104, 4109-4179 |
| Medium | Categorical bar labels collide for low-priced underlyings or low vol, so bars stack on the same x position | 3690-3691, 3960-3961, 3979 |
| Medium | Results disappear on any widget change, the 'Number of scenarios' slider does nothing, and the model prices passed in are ignored | 4235-4272 (4252-4255 slider, 4257 but... |
| Medium | Edge-case inputs produce NaN output or crashes (sigma=0, S=0, K=0), and the premium>0 sentinel reprices zero-value legs at shocked inputs | 1665-1668, 1698-1700, 1725-1727, 1866... |
| Medium | 'Max Gain' and 'Max Loss' side panels show grid-limited absolute values with misleading labels and colours, and ignore the P&L-change toggle | 3721-3729, 4000-4008, 1706-1707, 1733... |
| Low | The Market Stress chart ignores the P&L-change toggle, its 'current' marker is never drawn, and its label offsets are hard-coded | 3615-3671 |
| Low | The comparative risk chart uses fragile float index lookups and wrong labels ('1 Week Passes' = 10 days backwards; Long Call market crash shown as \$0) | 4186-4193 |
| Low | The code mixes trading-day (252) and calendar-day (365) conventions | 1667, 1779, 1782, 1786, 3861, 3903-39... |
| Low | Unreachable code after return, and computed-but-unused outputs | 1903-1906, 1777-1789, 1795-1811, 1813... |
| Low | The strategy guidance is incomplete, depends on substring order, and contradicts the displayed numbers; the tool offers fewer strategies than the app supports | 4109-4179, 4239-4240 |
| Low | Inline CSS 'style="\$color: ..."' is invalid, so theta colours never apply | 3934-3936 |

**Resolution.** Every scenario number is now a mark-to-market re-pricing of each leg at the shocked spot, implied volatility, rate and remaining time. The fabricated time-decay curves and metrics were deleted; time decay, the spot x IV heatmap and the stress table are all computed from the same valuation as the payoff chart.

### Volatility surface and local volatility (32 issues: 5 critical, 7 high, 13 medium, 7 low)

| Severity | Issue | Lines |
| --- | --- | --- |
| Critical | Dupire numerator double-counts the dividend: prices come from BS(S, r−q) (= e^{qT}·true price) but the numerator still adds q·C | 1429-1435, 1495-1497 |
| Critical | The smoothing slider does nothing between 0.05 and 1.0: s=level·m·n collapses both splines to one global bicubic polynomial | 1400-1410, 1515-1526, 2570-2574 |
| Critical | Exported 'LocalVolSurface' is fake: the ATM term structure repeated across every strike | 3493, 3495, 3519-3523 |
| Critical | Invalid Dupire points are silently replaced by the implied vol and still shown as local vol | 1502-1513, 3478-3490 |
| Critical | One-sided finite differences at the grid edges, and a coarse forward T-difference, corrupt the 42% of displayed points on the boundary | 1444-1482, 1413-1418, 1525 |
| High | Clipping is undone by the output spline: negative and above-200% local vols are displayed and exported | 1510-1522 |
| High | Uploaded CSV is not validated for units, spot consistency or sanity, so garbage is silently turned into 'Good' surfaces | 2624-2634, 2686-2717 |
| High | 'Cubic Spline' gap filling overwrites every market quote and extrapolates to negative vols on realistic chains | 2761-2776 |
| High | 'Thin Plate Spline' gap filling is O(N³) in memory and time, and its result depends on the price unit | 2778-2792 |
| High | Simulated Market skew sign is the reverse of its help text and of the Custom Parameters mode | 2608-2609, 2662, 2682 |
| High | Custom Parameters crashes the page when the sidebar volatility is outside [0.01, 1.0] | 2618 |
| High | Results disappear on any interaction, including clicking a download button | 2590-2599, 3534-3557 |
| Medium | Degenerate uploads crash with raw SciPy errors and a full traceback | 1405-1410, 1421-1422, 2688, 2697-2717... |
| Medium | 'Linear' gap filling interpolates in index space, not in strike and maturity units | 2743-2759 |
| Medium | The surface generators produce arbitrageable surfaces without warning (the term-structure slider allows calendar arbitrage; the 0.05 floor adds kinks) | 2612-2613, 2657-2666, 2675-2684 |
| Medium | Equal moneyness bounds or spot ≤ 0 crash with 'y must be strictly increasing' | 2621, 2655, 2670-2673, 1405 |
| Medium | Slow pure-Python loops: 2,500+ scalar BS calls, nested FD and Dupire loops, and an O(m·n·N) grid-fill loop | 1429-1513, 2697-2713 |
| Medium | Matplotlib figures are never closed, so memory leaks with every rerun | 2805-2840, 2846-2871, 2880-2898, 2912... |
| Medium | The 'Surface Stability' metric depends on grid size and cannot detect failures | 1632-1639, 3452-3455 |
| Medium | 'ATM', '90%' and '110%' points use the nearest grid strike without a range check, so they are mislabelled when those levels are off-grid | 1561-1569, 1606-1617, 2878-2890, 3253... |
| Medium | Calculation diagnostics are misleading | 1393-1398, 1470, 1482, 1508, 1513, 34... |
| Medium | Plotly 3-D chart: z-axis ticks are hard-coded to 5%–30% and the configured colorbar is never applied | 3175-3185, 3216-3226, 2964-2995 |
| Medium | The only interactive 3-D chart shows implied vol only, never the local-vol surface the tool is named for | 2934-3244 |
| Medium | Tracebacks and internal debug output are shown to end users, bypassing showErrorDetails='type' | 2597-2599, 2943-2958, 3297-3302 |
| Medium | Absolute 1e-6 denominator threshold makes local vol depend on the price scale | 1500-1503 |
| Low | 'cividis' is offered but silently rendered as Viridis in Plotly | 2584-2587, 2966-2976 |
| Low | Cascading fallback visualisation code (about 110 lines) is dead weight, with a bare except and wrong view labels | 3297-3408 |
| Low | Wrong MIME type for .xlsx, and the export is incomplete and hard to reuse | 3506-3559 |
| Low | Dividend and rate inputs are inconsistent with the rest of the app, and rate arrays use the wrong rate in Dupire | 2565-2568, 1387-1391, 1420-1425, 1490... |
| Low | The About text has invalid escape sequences and overclaims robustness | 2526-2545 |
| Low | Deprecated and legacy APIs | 1380, 1421-1422, 2647, 2788, 3230, 32... |
| Low | Dead code and unused imports in the vol-surface path | 1379-1380, 1417, 1569, 2647, 2741, 27... |

**Resolution.** `vol_surface.py` works in total implied variance and computes local volatility with Gatheral's formula (matches a price-based Dupire calculation to 1e-9). Surfaces come from arbitrage-free SSVI or from uploaded quotes, which are validated and smoothed per maturity (penalty chosen by cross-validation). Calendar and butterfly arbitrage are counted and left undefined instead of being patched, and the real local-vol grid is exported.

### Interface, reliability and project hygiene

| Severity | Issue | Resolution in version 2 |
| --- | --- | --- |
| Critical | Streamlit XSRF and CORS protections disabled in the committed `config.toml` | Protections restored to their secure defaults; only the dev container disables them on the command line, as GitHub Codespaces requires |
| High | Results of every button-gated tool (implied vol, local vol, VaR, scenarios) disappeared on the next interaction, including clicking a download button | Tools compute reactively from cached functions; downloads use `on_click="ignore"` |
| High | Matplotlib pyplot used from Streamlit's multi-threaded runner; figures never closed; global dark style set by one chart | All charts are Plotly figures built by pure functions and themed by Streamlit |
| High | Crashes on reachable inputs (K = 0, rate below 1%, duplicate widget IDs with the Monte Carlo VaR option) | Validated inputs with bounds; 110+ UI interactions and extreme inputs are exercised headlessly in testing with zero exceptions |
| High | No caching: every interaction recomputed every model, including heavy simulations | `st.cache_data` with bounded caches and fixed seeds; Streamlit fragments isolate the heavy tabs |
| Medium | Inputs in decimals with two-decimal formatting (a 4.33% rate or 15.5% volatility could not be entered); time in fractional years | Percent inputs with sensible steps and bounds; expiry in calendar days; settings bound to the URL so a setup can be shared |
| Medium | Catch-all exception handler, debug expanders and printed tracebacks shown to users; `warnings.filterwarnings('ignore')` hid numerical problems | Errors are prevented by validation or shown as plain-language messages; `showErrorDetails = "type"`; warnings are not suppressed |
| Medium | Greeks shown only for calls under a misleading "Position Greeks" label, rounded to 3 decimals | Call and put Greeks with units and definitions, higher orders on demand, position Greeks in dollars on the strategy tab |
| Medium | Three different strategy lists in three tabs, with hidden hard-coded strikes | One position, built once in the Strategy Builder, is analysed by every risk tool |
| Medium | Hard-coded HTML cards and colours that break in the light theme; HTML injected with `unsafe_allow_html` | Native `st.metric`, `st.dataframe` and theme-aware Plotly; no raw HTML |
| Medium | Markdown rendered amounts such as "\$4.88 ... \$100" as LaTeX | Dollar signs escaped wherever amounts appear in Markdown |
| Low | Duplicate function definitions, dead code (commented-out solver, unreachable branches), unused imports (seaborn, matplotlib colormaps, `minimize`), 554 whitespace warnings | Code base rewritten; `ruff check` and `ruff format` pass and run in CI |
| Low | README told users to run a file that does not exist (`options_pricing_app.py`), claimed Python 3.7+ support and advertised the Kelly/Sharpe metrics | README rewritten and verified: the run command, Python 3.10+ (tested with the minimum dependency versions) and every documented feature |
| Low | `requirements.txt` listed unused packages, missed `xlsxwriter` (the Excel export silently disappeared) and used floors too old for the APIs used | Accurate runtime requirements with tested floors; `pyproject.toml` for tooling |
| Low | No tests, no CI, a monolithic 4,331-line file, no type hints | A package with separate math and UI layers, type hints, 1,250+ tests and a CI workflow on Python 3.10-3.13 |
| Low | Dev container on an older Debian image | Python 3.12 on Debian bookworm, with ruff and the test tools installed |

## Independent review of version 2

Before release, version 2 itself was reviewed by independent agents instructed to break it. Each finding below was
reproduced, fixed and covered by a regression test.

| Severity | Issue found in the rebuild | Fix |
| --- | --- | --- |
| High | A far out-of-the-money option whose premium was below a fixed zero tolerance had no breakeven and an inverted probability of profit (100% for a lottery ticket) | Signs are classified with a tolerance relative to the size of the terms being summed; probability of profit integrates every profitable run of the payoff |
| High | VaR settled options that expire inside the horizon on the horizon's price (a 7-day calendar showed \$73 instead of about \$633) | Expiring legs settle on the simulated price at the close of their expiry day; the cash then earns the risk-free rate |
| High | With a dividend yield, the total-return stock valuation made a covered call's profit unlimited | A share is valued as price plus reinvested cash dividends: delta 1, covered-call profit capped at K - S0 + C + D |
| Medium | Position theta ignored the stock leg's dividend accrual | Stock theta is the dividend accrual; position theta matches finite differences |
| Medium | A zero-P&L plateau (zero-cost risk reversal) was reported as one midpoint breakeven, overstating the probability of profit (56% vs 37%) | Both ends of a plateau are reported; probabilities match the closed form |
| Medium | Implied-vol shocks in VaR inherited fat tails from Student-t/GARCH returns, so the vol multiplier lost its mean of one | Gaussian copula on rank scores, plus tenor damping |
| Medium | Student-t VaR built an (horizon x paths) matrix: 252 MB peak at the maximum settings | Daily accumulation: 81 MB peak |
| Medium | The analysis grid ignored long-dated legs' volatility, understating extremes of multi-expiry positions | Grid spans every live leg's volatility and tenor, and the S -> infinity limit is a candidate extreme |
| Low | The expected-P&L quadrature window was too narrow at extreme volatility x time | Window extends to 12 + 2 sigma sqrt(T); matches the closed form at sigma = 500%, T = 10 years |
| Low | Dot-com Fed move and Fed 2026 scenario figures | Corrected to -425 bp and to the published -58% equities and -360 bp (federalreserve.gov) |
| Low | The heatmap and the stress table applied the same implied-vol shock differently | Only one-month (VIX-based) shocks are tenor-damped and the table says so |
| Low | Riskless positions (box spreads) showed a negative maximum loss and noise-driven skewness | Labelled as guaranteed profit; higher moments reported as not applicable |
| Low | VaR configuration accepted NaN/infinite parameters and fractional horizons | Validated, including a minimum number of tail observations |
| Low | Documentation promised a zero expected P&L also when fills, fees or leg IVs differ from the model | Wording corrected; time-spread templates require an explicit expiry |
