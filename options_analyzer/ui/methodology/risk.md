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
Full revaluation on simulated spot (and optionally implied-vol) paths:
* **Normal** — Gaussian daily returns.
* **Student-t** — daily shocks ${t_\nu\sqrt{(\nu-2)/\nu}}$ (unit variance) summed over the horizon.
* **GARCH(1,1)** — ${h_{t+1} = \omega + \alpha\varepsilon_t^2 + \beta h_t}$ in daily units, with variance
  targeting ${\omega = (1-\alpha-\beta)\,\sigma^2/252}$ and the same shocks driving returns and variance.
* **Implied-vol moves** — log-normal with a chosen vol-of-vol, correlated with the spot return through a
  Gaussian copula (so the multiplier keeps mean one under fat tails) and damped for longer tenors.

Options that expire before the horizon are settled on the simulated price at the close of their expiry day,
and the cash earns the risk-free rate until the horizon.

VaR and ES use the Acerbi–Tasche estimators: with ${w = \lfloor n(1-\alpha)\rfloor}$ worst outcomes, VaR is the
$w$-th worst loss and ES the average of the $w$ worst losses. The delta-normal VaR
$|\Delta| S \sigma \sqrt{h/252}\, z_\alpha$ is shown for comparison.
