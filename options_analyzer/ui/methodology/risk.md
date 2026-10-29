## Risk

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
Full revaluation on simulated spot and implied-vol paths:
* **Normal** — Gaussian daily returns.
* **Student-t** — daily shocks ${t_\nu\sqrt{(\nu-2)/\nu}}$ (unit variance) summed over the horizon.
* **GARCH(1,1)** — ${h_{t+1} = \omega + \alpha\varepsilon_t^2 + \beta h_t}$ in daily units, with variance
  targeting ${\omega = (1-\alpha-\beta)\,\sigma^2/252}$ and the same shocks driving returns and variance.
* **Implied-vol moves** — one-month implied vol is log-normal with a chosen vol of vol, correlated with the spot
  return through a Gaussian copula (so the multiplier keeps mean one under fat tails) and damped for longer
  tenors. They are on by default, with index (VIX-like) dynamics: vol of vol 100 % a year and spot–vol
  correlation −0.7.

With implied-vol moves on, the tab shows the spot-only VaR and ES beside the result: the same simulated spot
paths (one seed) re-priced with implied vol frozen, so the gap between the two is what the implied-vol moves
add, the vega risk. On a 45-day at-the-money short straddle on an index (spot 5,800, volatility 16 %, rate
4.3 %, dividend yield 1.3 %, one 100-share contract, 100,000 paths, seed 7, the app's default), the one-day
99 % VaR is \$3,862 with implied-vol moves and \$2,833 without them: frozen implied vol understates it by 27 %
(24 % to 27 % over seeds 0 to 9).

Options that expire before the horizon are settled on the simulated price at the close of their expiry day,
and the cash earns the risk-free rate until the horizon.

VaR and ES use the Acerbi–Tasche estimators: with ${w = \lfloor n(1-\alpha)\rfloor}$ worst outcomes, VaR is the
$w$-th worst loss and ES the average of the $w$ worst losses. The delta-normal VaR
$|\Delta| S \sigma \sqrt{h/252}\, z_\alpha$, with Normal returns, is shown for comparison in the table of P&L
statistics. It leaves out the position's gamma, its time decay over the horizon, its vega (with implied-vol
moves on) and the fatter tails of Student-t returns, or of GARCH returns beyond one day.
