"""Methodology tab: models, conventions and limitations."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        r"""
### Conventions
* Time to expiry $T$ = calendar days / 365. Rates $r$ and dividend yield $q$ are continuously compounded.
* Greeks are shown per calendar day (theta, charm, veta, color), per volatility point (vega, vanna, zomma),
  per 1 % rate move (rho) and per \$1 move in the underlying (delta, gamma, speed). Every time Greek is the
  change as calendar time *passes*: theta is usually negative for a long option (deep in-the-money puts and
  calls on high-dividend stocks can gain value as time passes).
* Value-at-Risk horizons are in trading days (252 per year), because volatility accrues with trading time.
* Strategy quantities are signed (+ long, − short); one unit is an option on one share. The contract
  multiplier converts to dollar P&L.

### Black–Scholes–Merton

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$d_{1,2} = \dfrac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)\,T}{\sigma\sqrt{T}}$$

With zero volatility the value is the discounted forward intrinsic value
$\max(\pm(S e^{-qT} - K e^{-rT}), 0)$. All fourteen Greeks (to third order) are analytical and are
verified against high-precision finite differences in the test-suite.

### Binomial trees
* **Cox–Ross–Rubinstein:** $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$,
  $p = (e^{(r-q)\Delta t} - d)/(u - d)$. The tree refuses inputs with $p \notin (0,1)$ (arbitrage).
* **Leisen–Reimer** (the default): Peizer–Pratt inversion centres the tree on the strike; convergence is
  smooth, roughly second order for European exercise and first order with early exercise (an odd number of
  steps is used).
* Calls are priced as the symmetric put, $C(S, K, T; r, q) = P(K, S, T; q, r)$ (McDonald–Schroder). On the
  lattice this is an exact change of numeraire, and a put's payoff is bounded by its strike, so the value
  stays accurate even when extreme volatility × expiry forces the tree to keep its node prices between
  $e^{-700}$ and $e^{700}$. Such a capped lattice computes each level's prices from their logarithms, so a
  capped or zero-rounded node never feeds the exercise test at an earlier step.
* American exercise compares continuation and exercise value at every node; delta and gamma are read off
  the tree, theta from the step-2 middle node (CRR) or a central difference in time (Leisen–Reimer), and
  vega, rho and dividend rho from repricing on a tree with the same steps (±0.5 vol point, ±25 bp). The
  early-exercise boundary is the critical price where the exercise gain changes sign, interpolated between
  adjacent nodes, averaged over consecutive steps (whose nodes interleave) and fitted to be monotone in
  time, as the exact boundary is. The early-exercise premium is measured on the same tree with European
  exercise, so the discretisation error cancels; the American reference in the convergence chart
  extrapolates Leisen–Reimer trees with 2,001 and 4,003 steps and is kept within the American no-arbitrage
  bounds: at least the European and the exercise value, at most the perpetual American value.

### Monte Carlo
European payoffs depend only on $S_T$, which is sampled exactly:
$S_T = S\exp\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}Z\big)$. Antithetic pairs $(Z, -Z)$ and a
control variate (the discounted terminal price, with known mean $S e^{-qT}$) reduce the variance; the
standard error uses the unbiased sample deviation of the independent samples. It is left blank when the
sample cannot estimate it: every simulated price finished out of the money, or every one in the money with
the control variate on (the residuals are then constant). Calls and puts share
draws and a fixed seed makes results reproducible. The sample-path chart draws the first 60 of these
scenarios, each filled in from today to its sampled $S_T$ with a Brownian bridge (exact for GBM given the
end point).

### Implied volatility
Quotes outside the no-arbitrage bounds
$\max(S e^{-qT} - K e^{-rT}, 0) \le C < S e^{-qT}$ (and the put analogue) are rejected, because no
volatility can produce them. Otherwise the quote is converted to its out-of-the-money equivalent through
put–call parity and solved with Brent's method, which always converges on a bracketed root. American quotes
must be at least the exercise value and the European lower bound, and below the most exercise can be worth:
$K\max(1, e^{-rT})$ for a put (more than the strike when the rate is negative) and $S\max(1, e^{-qT})$ for a
call. They are inverted on the sidebar's binomial tree when that model is selected (so the
headline American price gives back the sidebar volatility), otherwise on a 501-step Leisen–Reimer tree; a
quote equal to the option's zero-volatility value (for example the value of exercising now) does not
identify a volatility and is reported as such.

### Strategies
Each position is a list of legs. Options are valued with Black–Scholes–Merton at each leg's implied
volatility. A share is worth its price plus the dividends it has paid, reinvested at the risk-free rate,
$S_t + S_0\,(e^{rt} - e^{(r-q)t})$, so its delta is exactly one and a covered call's profit stays capped.
The entry cost is the model value unless you enter your own fill price, so the "today" curve starts at zero
for legs entered at model value. Breakevens are exact roots of the P&L function, maximum profit or loss is
reported as unlimited when the P&L keeps rising or falling with the spot, and the probability of profit
integrates the log-normal density of the price at the front expiry over the profitable intervals (drift = the
risk-free rate by default). Premium financing is ignored unless you switch it on; early exercise is ignored,
whatever the sidebar pricing model. Multi-strike templates space their strikes $w$ apart, a percentage of the
strike rounded to the cent (to two significant figures below ten cents). By default $w$ is about half the
expected move, $w \approx \tfrac12 K\sigma\sqrt{T}$, rounded to a strike step of 1, 2, 2.5 or 5 times a power
of ten and capped at 25 % of the strike.

### Scenarios and stress tests
Every scenario re-prices all legs at the shocked spot, implied volatility, rate and remaining time. Historical
analogues use the S&P 500 move, the VIX ratio and the policy-rate change of each episode (the Fed scenario
uses the published 2026 severely adverse paths). Those implied-vol shocks are one-month shocks, damped for
longer tenors by $(T/\text{1M})^{-0.25}$ by default, since long-dated volatility moves much less than
one-month volatility in a crisis; hypothetical IV shocks and the heatmap apply literally to every leg.
The spot × date grid, the time-decay curves and the position-Greek profiles move the spot in standard
deviations $\sigma\sqrt{T}$ of the log price at the front expiry, so their range follows the volatility and
the time left.

### Value-at-Risk and Expected Shortfall
Full revaluation on simulated spot (and optionally implied-vol) paths:
* **Normal** — Gaussian daily returns.
* **Student-t** — daily shocks $t_\nu\sqrt{(\nu-2)/\nu}$ (unit variance) summed over the horizon.
* **GARCH(1,1)** — $h_{t+1} = \omega + \alpha\varepsilon_t^2 + \beta h_t$ in daily units with variance
  targeting $\omega = (1-\alpha-\beta)\,\sigma^2/252$; the same shocks drive returns and variance.
* **Implied-vol moves** — log-normal with a chosen vol-of-vol, correlated with the spot return through a
  Gaussian copula (so the multiplier keeps mean one under fat tails) and damped for longer tenors.

Options that expire before the horizon are settled on the simulated price at the close of their expiry day,
and the cash earns the risk-free rate until the horizon.

VaR and ES use the Acerbi–Tasche estimators: with $w = \lfloor n(1-\alpha)\rfloor$ worst outcomes, VaR is the
$w$-th worst loss and ES the average of the $w$ worst losses. The delta-normal VaR
$|\Delta| S \sigma \sqrt{h/252}\, z_\alpha$ is shown for comparison.

### Local volatility
Surfaces are handled as total implied variance $w(y,T) = \sigma_{\text{BS}}^2 T$ in forward log-moneyness
$y = \ln(K/F_T)$, and local variance follows Gatheral's form of Dupire's equation

$$\sigma_{\text{loc}}^2 = \frac{\partial_T w}{1 - \frac{y}{w}\partial_y w + \frac14\left(-\frac14 - \frac1w + \frac{y^2}{w^2}\right)(\partial_y w)^2 + \frac12 \partial_{yy} w}.$$

The numerator and denominator are exactly the calendar and butterfly no-arbitrage conditions; where either
fails local volatility is undefined and is reported rather than patched. The parametric surface is SSVI
(Gatheral & Jacquier, 2014), arbitrage-free when $\eta(1+|\rho|) \le 2$. Uploaded quotes (implied vols, or
option prices inverted with Black–Scholes–Merton, out-of-the-money side preferred) are fitted per maturity
with Gatheral's raw SVI smile

$$w(y) = a + b\left(\rho\,(y-m) + \sqrt{(y-m)^2 + s^2}\right)$$

by least squares on implied vols, with $b \ge 0$, $|\rho| < 1$, $s > 0$, a non-negative minimum variance and
Lee's wing bound $b(1+|\rho|) \le 2$. The search starts from Zeliade's quasi-explicit fits: for fixed $m$ and
$s$ the smile is linear in $a$, $bs$ and $\rho bs$, solved on a grid of $(m, s)$ with $|\rho| < 1$ imposed, so
steep, nearly linear equity skews are followed too. SSVI slices are exactly SVI, so noise-free SSVI quotes are
recovered. A maturity column whose header gives no unit is read as years and checked against the quotes. Read
as years, a skewed smile a few days long is steeper than any arbitrage-free smile that long can be (its vols
price a put below a put at a lower strike), and option prices fall below their intrinsic value or imply vols
about 19 times too low; quotes like that get a warning to name the unit.
$\partial_y w$ and $\partial_{yy} w$ are analytic; $\partial_T w$ at fixed $y$ comes from secants between
neighbouring maturities evaluated at the same $y$ — their Fritsch–Butland weighted harmonic mean inside, the
one-sided three-point formula at the first maturity (the one-sided secant where that is not positive), and the
one-sided secant at the last, where quote noise outweighs the secant's truncation error. A secant that falls
by more than the quotes' precision is calendar arbitrage; one that changes by less (two expiries a day apart,
say) cannot resolve $\partial_T w$ and is not used, and where no secant is left local vol is left blank.
Implied vol is shown only inside each maturity's quoted strikes, and local vol only where the neighbouring
maturities are quoted near the same forward moneyness.

### Limitations
Constant volatility per leg (no smile dynamics in strategy valuation), European exercise for strategies,
no bid-ask spreads or transaction costs beyond a flat per-contract commission, no margin or discrete
dividends, and log-normal probabilities. This tool is for education
and analysis only and is not investment advice.

### References
Hull, *Options, Futures and Other Derivatives* · Haug, *The Complete Guide to Option Pricing Formulas* ·
Leisen & Reimer (1996) · Glasserman, *Monte Carlo Methods in Financial Engineering* · Acerbi & Tasche (2002) ·
Gatheral, *The Volatility Surface* (2006) · Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces* (2014).
"""
    )
