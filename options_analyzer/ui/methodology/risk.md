## Risk

### Scenarios and stress tests
Every scenario re-prices all legs at the shocked spot, implied volatility, rate and remaining time. Historical
analogues take each episode between two closes: the S&P 500 move (to 0.1%), the ratio of the VIX closes as the
implied-vol multiplier (the VXO for 1987) and the change in the Fed funds target (the midpoint of the target range
from December 2008), with the VIX, VXO and Fed funds data from FRED (series VIXCLS, VXOCLS, DFEDTAR, DFEDTARL and
DFEDTARU). The Fed scenario uses the published 2026 severely adverse paths from their Q4 2025 jump-off: equities
−57.8% at the Q3 2026 trough, the VIX from 26.4 to its Q2 2026 peak of 72.0 (a multiplier of 2.73) and the 3-month
Treasury rate from 3.7% to 0.1%. It applies the VIX peak and the equity trough at once, although the Fed's paths
reach them a quarter apart. Every description quotes its dates and levels, and a test checks them, and the Fed
figures above, against the sources. The rate shock shifts the model's risk-free rate: by the change in the Fed
funds target for a historical analogue, by the change in the 3-month Treasury rate for the Fed scenario. The vol
ratio multiplies each leg's current implied vol; it does not set implied vol to the index's level. The Fed's VIX
is the highest close of each quarter, so its shock is the ratio of two spikes, and a one-month leg marked at a
calm 15% goes to 41%, not to the Fed's 72%. Those implied-vol shocks are one-month shocks, damped for longer
tenors by $(T/\text{1M})^{-0.25}$ by default, since long-dated volatility moves much less than one-month
volatility in a crisis; hypothetical IV shocks and the heatmap apply literally to every leg.
The spot × date grid, the time-decay curves and the position-Greek profiles move the spot in standard
deviations $\sigma\sqrt{T}$ of the log price at the front expiry, so their range follows the volatility and
the time left. Under the time-decay curves the one-day decay by repricing stands beside theta / 365 (see
Conventions), and the theta profile draws it dashed beside theta / 365 at each date: the position's value a calendar
day after the date minus its value on the date, at every spot of the axis. Within a few days of expiry the two can
part a long way in either direction.

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

### P&L explain
The library explains the P&L of a market move in two ways; the Risk tab does not show them yet. The P&L is the
full revaluation: each leg's value at the new spot, implied vols and rate after the time elapsed, minus its value
today. The Greek waterfall splits it into the terms of a Taylor expansion around today's market, with today's
Greeks, the spot move $\delta S$, each leg's vol shift $\delta\sigma$, the rate shift $\delta r$ and the time
elapsed $\delta t$:
* **First order** — delta ${\Delta\,\delta S}$, vega ${\nu\,\delta\sigma}$, rho ${\rho\,\delta r}$ and theta
  ${\Theta_{\delta t} = V(t + \delta t) - V(t)}$, the time decay by repricing with nothing else moving (for one
  day, the one-day decay above).
* **Second order** — gamma ${\tfrac12\,\Gamma\,\delta S^2}$, vanna ${\text{vanna}\,\delta S\,\delta\sigma}$, volga
  ${\tfrac12\,\text{volga}\,\delta\sigma^2}$, charm ${\text{charm}\,\delta S\,\delta t}$ and veta
  ${\text{veta}\,\delta\sigma\,\delta t}$.
* **Unexplained** — $U$, the P&L less the nine terms.

Theta as $\theta\,\delta t$ would leave ${\tfrac12\,\partial_{tt}V\,\delta t^2}$ and the higher orders in time
unexplained: 0.8 % of a 30-day at-the-money straddle's one-day decay, and half of a 1-day at-the-money option's,
whose value falls with the square root of the time left (the $\delta t^2$ term is then only a quarter of the gap).
The unexplained P&L $U$ is third order in the spot and vol moves, so halving them divides it by about 8 (7.84,
8.10 and 8.11 on an iron condor); the second-order terms of a rate move are left in it too. The quality
${|U| / \max(|\text{P\&L}|, f)}$ is the share of the P&L that the Greeks miss. Its floor $f$, a tenth of the sum of
the nine terms' sizes, comes into play when the P&L is smaller than that, as when a day's time decay offsets the
gamma of a spot move: such a P&L is a small difference of larger amounts, and the share of it that the Greeks miss
means little. With this floor the quality never exceeds 11. A floor in the notional would be the wrong scale for
options: one basis point of a 7-day index iron condor's notional is \$232, more than its whole day of time decay,
\$151.97. On a 30-day at-the-money short straddle (spot and strike 100, vol 20 %, rate 4 %, no dividends, one
100-share contract), a day on which spot falls 4 % and implied vol rises 6 points loses \$188.36, and the nine terms
leave \$25.17 unexplained (13.4 %): mostly zomma, since the higher vol lowers the straddle's gamma (its third-order
term ${\tfrac12\,\partial_{SS\sigma}V\,\delta S^2\,\delta\sigma}$ is +\$33.19).

The Shapley split shares the same P&L exactly between the four factors spot, vol, time and rate. With $v(A)$ the
P&L when only the factors in $A$ move (16 revaluations), factor $i$ gets
${\phi_i = \sum_A w_{|A|}\,[v(A \cup \{i\}) - v(A)]}$ over the sets $A$ of the other three factors, with the
weights ${w_k = k!\,(3 - k)!/4!}$ (1/4, 1/12, 1/12 and 1/4 for 0 to 3 factors): its average contribution over
the 24 orders in which the factors can be moved one after the other. The parts add up to the P&L, with no
residual, and depend on no order: on the straddle's day they are spot −\$72.17, vol −\$124.04 and time +\$7.85,
while moving the factors in turn credits the spot move with anything from −\$84.87 to −\$59.72, depending on the
order. Each leg is explained on its own, and a position's explain is the sum of its legs'.

For the P&L attribution test of a backtest, the Spearman correlation of the ranks of the hypothetical and
risk-theoretical P&L and the Kolmogorov–Smirnov distance between their empirical distributions are computed as the
Basel framework defines them (MAR32.36 to 32.41).
