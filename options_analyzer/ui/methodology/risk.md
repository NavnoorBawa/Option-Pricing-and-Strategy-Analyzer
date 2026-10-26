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
