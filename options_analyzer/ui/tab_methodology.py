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
  change as calendar time *passes*, so a long option has negative theta.
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
* **Leisen–Reimer:** Peizer–Pratt inversion centres the tree on the strike; convergence is smooth and
  roughly second order (an odd number of steps is used).
* American exercise compares continuation and exercise value at every node; delta, gamma and theta are read
  off the tree, and the early-exercise boundary is traced from the exercised nodes.

### Monte Carlo
European payoffs depend only on $S_T$, which is sampled exactly:
$S_T = S\exp\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}Z\big)$. Antithetic pairs $(Z, -Z)$ and a
control variate (the discounted terminal price, with known mean $S e^{-qT}$) reduce the variance; the
standard error uses the unbiased sample deviation of the independent samples. Calls and puts share draws and
a fixed seed makes results reproducible.

### Implied volatility
Quotes outside the no-arbitrage bounds
$\max(S e^{-qT} - K e^{-rT}, 0) \le C < S e^{-qT}$ (and the put analogue) are rejected, because no
volatility can produce them. Otherwise the quote is converted to its out-of-the-money equivalent through
put–call parity and solved with Brent's method, which always converges on a bracketed root. American quotes
are inverted on a 501-step Leisen–Reimer tree against American bounds; a quote equal to the option's
zero-volatility value (for example the value of exercising now) does not identify a volatility and is
reported as such.

### Strategies
Each position is a list of legs. Options are valued with Black–Scholes–Merton at each leg's implied
volatility. A share is worth its price plus the dividends it has paid, reinvested at the risk-free rate,
$S_t + S_0\,(e^{rt} - e^{(r-q)t})$, so its delta is exactly one and a covered call's profit stays capped.
The entry cost is the model value unless you enter your own fill price, so the "today" curve starts at zero
for legs entered at model value. Breakevens are exact roots of the P&L function, maximum profit or loss is
reported as unlimited when the P&L keeps rising or falling with the spot, and the probability of profit
integrates the log-normal density of the price at the front expiry over the profitable intervals (drift = the
risk-free rate by default). Premium financing is ignored unless you switch it on; early exercise is ignored.

### Scenarios and stress tests
Every scenario re-prices all legs at the shocked spot, implied volatility, rate and remaining time. Historical
analogues use the S&P 500 move, the VIX ratio and the policy-rate change of each episode (the Fed scenario
uses the published 2026 severely adverse paths). Those implied-vol shocks are one-month shocks, damped for
longer tenors by $(T/\text{1M})^{-0.25}$ by default, since long-dated volatility moves much less than
one-month volatility in a crisis; hypothetical IV shocks and the heatmap apply literally to every leg.

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
$|\Delta| S \sigma \sqrt{h}\, z_\alpha$ is shown for comparison.

### Local volatility
Surfaces are handled as total implied variance $w(y,T) = \sigma_{\text{BS}}^2 T$ in forward log-moneyness
$y = \ln(K/F_T)$, and local variance follows Gatheral's form of Dupire's equation

$$\sigma_{\text{loc}}^2 = \frac{\partial_T w}{1 - \frac{y}{w}\partial_y w + \frac14\left(-\frac14 - \frac1w + \frac{y^2}{w^2}\right)(\partial_y w)^2 + \frac12 \partial_{yy} w}.$$

The numerator and denominator are exactly the calendar and butterfly no-arbitrage conditions; where either
fails local volatility is undefined and is reported rather than patched. The parametric surface is SSVI
(Gatheral & Jacquier, 2014), arbitrage-free when $\eta(1+|\rho|) \le 2$; uploaded quotes are smoothed per
maturity with a cubic smoothing spline (penalty by generalised cross-validation) and interpolated in maturity
with a monotone (PCHIP) interpolant of total variance.

### Limitations
Constant volatility per leg (no smile dynamics in strategy valuation), European exercise for strategies,
no transaction costs, margin or discrete dividends, and log-normal probabilities. This tool is for education
and analysis only and is not investment advice.

### References
Hull, *Options, Futures and Other Derivatives* · Haug, *The Complete Guide to Option Pricing Formulas* ·
Leisen & Reimer (1996) · Glasserman, *Monte Carlo Methods in Financial Engineering* · Acerbi & Tasche (2002) ·
Gatheral, *The Volatility Surface* (2006) · Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces* (2014).
"""
    )
