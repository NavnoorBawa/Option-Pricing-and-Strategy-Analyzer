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

The table also gives the delta-gamma-theta VaR and ES, the analytic comparator: exact for the P&L of today's
position Greeks to second order in the spot move $x$ and first order in time,
${\theta h/252 + \Delta x + \tfrac12\Gamma x^2}$, with $\theta$ per year and $x$ Normal with mean zero and standard
deviation ${s = S\sigma\sqrt{h/252}}$. Completing the square, this is
${\theta h/252 - \Delta^2/(2\Gamma) + \tfrac12\Gamma s^2\,Y}$ with $Y$ non-central chi-square with one degree of
freedom and non-centrality ${\lambda = (\Delta/(\Gamma s))^2}$. VaR is read off its quantile, the lower tail of $Y$
for long gamma and the upper tail for short gamma; ES integrates the same distribution numerically, within
${3 \times 10^{-13}}$ of the P&L's size in tests against 50-digit mpmath. At ${\Gamma = 0}$ it is the delta-normal
VaR minus the time decay ${\theta h/252}$, and it moves there continuously as ${\Gamma \to 0}$. Its gap to the full
revaluation shows the spot risk beyond gamma, how the time decay and gamma change as expiry nears, the legs that
expire within the horizon, the vega and the fatter tails. On the 45-day straddle above with implied vol frozen, the
one-day 99 % VaR is \$2,882 delta-gamma-theta and \$2,833 by full revaluation (\$1,016 delta-normal). On ten
strategy templates, at least one of each kind (30 days, spot 100, volatility 20 %, 400,000 paths), the one-day gap
is at most 3.2 %; on a put ratio backspread it is 10 %, since its gamma grows as the spot falls towards its two long
puts. It is a check for short horizons: an option's time decay speeds up as expiry nears, which today's theta
leaves out. Over 20 trading days of the 30-day long straddle, full revaluation loses \$373 at an unchanged spot, about
its worst outcome, where ${\theta h/252}$ says \$221, and the 99 % VaR is \$372 by full revaluation against \$222
delta-gamma-theta.

### Backtesting
The backtests are in the library (`options_analyzer/backtest.py`); the Risk tab does not run them yet. A VaR model
is judged by its exceptions, the days whose loss exceeds the VaR. Kupiec's (1995) likelihood ratio tests their rate,
with the exact binomial p-value beside the chi-square one; Christoffersen's (1998) Markov test asks whether they come
in clusters; and the Basel traffic light of MAR32 counts them: in 250 days at 99 %, 0 to 4 exceptions are green
(capital multiplier 1.50), 5 to 9 amber (1.70 to 1.92) and 10 or more red (2.00). Other sample sizes follow the
Basel Committee's 1996 rule, amber from a cumulative binomial probability of 95 % and red from 99.99 %. Basel's
traffic light is defined at 99 % only (at 97.5 % MAR32 sets just a desk's limit of 30 exceptions in 12 months); the
zones the library also gives at 97.5 % apply the same rule as its own extension. Expected shortfall is tested by
Acerbi and Szekely's (2014) $Z_2$, with its critical value simulated under the model, and two models are ranked by
the FZ0 loss of Patton, Ziegel and Chen (2019) with Diebold and Mariano's (1995) test and a Newey–West variance.
MAR32's P&L attribution test (the Spearman correlation and the Kolmogorov–Smirnov distance between the desk's and
the risk model's P&L) is there too. A rolling backtest builds the position at every close, lets each model forecast
its next-day P&L, and scores the forecasts against the P&L of that position held to the next close, re-priced at the
next close's spot and implied vol.

A year of data tells little. At 250 days and 99 %, Kupiec's test at a nominal 5 % rejects a correct model 9.5 % of
the time (the binomial is discrete) and a model whose true exception rate is 2 %, twice the forecast, only 24 % of
the time; ten years (2,520 days) raise that to 98.7 %. The traffic light leaves that model green 44 % of the time.
Christoffersen's independence test rejects exceptions that follow one another a quarter of the time in only 34 % of
such years. $Z_2$ catches a volatility 20 % above the model's in 77 % of years, but tails as fat as a Student-t's
with 5 degrees of freedom, at the model's variance, in only 16 %.

The tests check themselves on a synthetic history, labelled "Synthetic — not market data": ten years of GARCH(1,1)
returns with Student-t shocks, their volatility set by a log-normal implied-vol index with the Risk tab's index
dynamics, on which the 45-day at-the-money short straddle above is struck again at every close. At 99 % over 2,020
days (20.2 exceptions expected, red from 39), the Normal model with implied vol frozen has 50 exceptions (red), the
Risk tab's default with implied-vol moves 31 (amber), both with 5,000 paths a day, and historical simulation on the
last 500 days with Hull and White's (1998) volatility scaling 17 (green); over ten seeds, 42 to 76, 21 to 39 and 15
to 25.
