# VaR and ES backtesting, and a synthetic market history

`options_analyzer/backtest.py` tests VaR and ES models against realised P&L, ranks two models, computes how much each
test can detect at a given sample size, and generates a synthetic daily history to run them on. Everything is
written from the papers with NumPy and SciPy; packages under the GPL (rugarch, ExactVaRTest, esback) were not used,
not even for reference values: every reference number below comes from a paper, a Basel document or a closed form.
The Risk tab does not run these backtests yet; the library and its tests do.

## Model

A model forecasts, at each close, the distribution of the next day's P&L $X_t$; its VaR and ES at confidence $c$
are positive losses, read with the Acerbi-Tasche estimators of `options_analyzer/risk.py` when the model is given as
simulated P&L. A day is an exception when $X_t < -\mathrm{VaR}_t$; with $p = 1 - c$ the exceptions of a correct
model are independent Bernoulli($p$) draws.

**Kupiec (1995)**, proportion of failures. With $x$ exceptions in $n$ days and $\hat p = x / n$,

$$\mathrm{LR}_{\mathrm{POF}} = -2 \ln\big[(1 - p)^{n - x} p^x\big] + 2 \ln\big[(1 - \hat p)^{n - x} \hat p^{\,x}\big],$$

asymptotically $\chi^2(1)$. The library also returns the exact p-value of the same statistic (the binomial
probability of an LR at least as large) and the binomial tail $P(X \ge x)$.

**Christoffersen (1998)**, independence. With $n_{ij}$ the days in state $j$ after a day in state $i$ (1 an
exception), $\pi_{01} = n_{01} / (n_{00} + n_{01})$, $\pi_{11} = n_{11} / (n_{10} + n_{11})$ and
$\pi = (n_{01} + n_{11}) / (n - 1)$:

$$\mathrm{LR}_{\mathrm{ind}} = -2 \ln \frac{(1 - \pi)^{n_{00} + n_{10}} \pi^{n_{01} + n_{11}}}
  {(1 - \pi_{01})^{n_{00}} \pi_{01}^{n_{01}} (1 - \pi_{11})^{n_{10}} \pi_{11}^{n_{11}}}
  \sim \chi^2(1),$$

and $\mathrm{LR}_{\mathrm{cc}}$, the same ratio with $p$ in place of $\pi$ in the numerator, $\chi^2(2)$. Both
condition on the first day, as in the paper, so $\mathrm{LR}_{\mathrm{cc}} = \mathrm{LR}_{\mathrm{uc}} +
\mathrm{LR}_{\mathrm{ind}}$ exactly, with $\mathrm{LR}_{\mathrm{uc}}$ Kupiec's statistic on the last $n - 1$ days. (Some
software adds Kupiec's statistic on all $n$ days instead.)

**Traffic light.** MAR32.9 Table 1 (BCBS 2020): in 250 days at 99 %, 0 to 4 exceptions are green (multiplier
1.50), 5 to 9 amber (1.70, 1.76, 1.83, 1.88, 1.92) and 10 or more red (2.00). Those are the numbers of exceptions
at which the cumulative binomial probability reaches 95 % and 99.99 %, the rule BCBS (1996) gives for other sample
sizes (it calls the amber zone "yellow"); the library applies that rule to any sample size and coverage, and gives
the multiplier only at 250 days and 99 %.

**Acerbi and Szekely (2014)**, test 2 of ES at tail probability $\alpha$:

$$Z_2 = \sum_{t=1}^{T} \frac{X_t I_t}{T \alpha\, \mathrm{ES}_{\alpha,t}} + 1, \qquad I_t = 1\{X_t \le -\mathrm{VaR}_{\alpha,t}\},$$

zero on average under the model and negative when the tail losses exceed the forecast. The critical value is the
5 % quantile of $Z_2$ simulated under the model. A P&L exactly at $-\mathrm{VaR}$ counts as a tail day: that is the
paper's indicator $I'_t$ for a scenario set whose tail holds exactly $\alpha$ of the scenarios, and it keeps the mean
of the simulated $Z_2$ at zero when the null is drawn from a model's finite scenario set.

**FZ0 (Patton, Ziegel and Chen 2019, eq. 6)**, in the VaR/ES sign convention ($v = -\mathrm{VaR}$, $e = -\mathrm{ES}$
in the paper):

$$L = \frac{\max(-X - \mathrm{VaR}, 0)}{\alpha\, \mathrm{ES}} + \frac{\mathrm{VaR}}{\mathrm{ES}} + \ln \mathrm{ES} - 1,$$

strictly consistent for VaR and ES jointly (smallest in expectation at the true pair) and homogeneous of degree
zero. **Diebold and Mariano (1995)** compare two models' mean losses: $\bar d / \sqrt{\hat\sigma^2 / T}$ with
$\hat\sigma^2$ the Newey and West (1987) long-run variance of the loss differential (Bartlett weights,
$\lfloor 4 (T/100)^{2/9} \rfloor$ lags by default), asymptotically standard Normal.

**P&L attribution (MAR32.34 to MAR32.42).** The Spearman correlation of hypothetical and risk-theoretical P&L and
the Kolmogorov-Smirnov distance between their empirical distributions: green with a correlation above 0.80 and a
distance below 0.09, red with a correlation below 0.70 or a distance above 0.12, amber otherwise.

**Rolling backtest.** At each close $t$ from the end of the window, `build_position(close, implied vol)` gives the
position; each model forecasts its one-day P&L from the last `window` days; the hypothetical P&L re-prices the same
legs at the next close's spot, every leg's implied vol multiplied by the index's change (damped for tenors beyond a
month by $(\tau / 1\mathrm{M})^{-0.25}$, as the VaR engine and the stress tests do) and one trading day ($1/252$ year)
less to expiry. Two models come with the library: the app's VaR engine (`monte_carlo_model`, any `VaRConfig` with a
one-day horizon) and historical simulation by full revaluation on the window's joint spot and implied-vol moves
(`historical_model`), optionally with Hull and White's (1998) EWMA volatility scaling (decay 0.94).

**Synthetic history.** Daily, labelled "Synthetic — not market data" (`SYNTHETIC_LABEL`): the log of a one-month
implied-vol index is an Ornstein-Uhlenbeck process (median 18 %, vol of vol $\eta = 100$ % a year, mean reversion 4 a
year, a half-life of 0.17 year); the index's shocks are joined to the spot's by a Gaussian copula with correlation
$\rho = -0.7$; the spot's daily variance is $(\mathrm{IV}_{t-1} / 1.2)^2 / 252$ times a unit-mean GARCH(1,1)
factor ($\alpha = 0.06$, $\beta = 0.90$) driven by unit-variance Student-t shocks with 5 degrees of freedom.
Implied vol is thus 1.2 times the next day's volatility when the GARCH factor is at its mean, and the realised
volatility of the spot is about 15 % to 16 % a year (15.2 % to 16.3 % over 10,000 days at seeds 0 to 2). With no vol
of vol the returns are a plain GARCH(1,1)-t.

## Numerical method

* Kupiec's exact p-value and every traffic-light probability are exact binomial sums over all $n + 1$ outcomes.
* The power of Kupiec's test is exact: the binomial probability, under the true rate, of the outcomes whose LR
  exceeds the $\chi^2(1)$ critical value. Christoffersen's power is simulated (10,000 paths of a stationary
  two-state Markov chain, seed 1998; standard error at most 0.005).
* $Z_2$ under a zero-mean Normal or Student-t model does not depend on the scale, so its null is simulated from the
  standard distribution, once per sample size. Only tail days contribute: each simulated history draws its number
  of tail days from the binomial and then that many tail values by inverting the distribution below the VaR. This
  is exact in distribution and needs about $T \alpha$ draws per history instead of $T$ (100,000 histories, seed
  2014, in 0.06 s for the Normal; the 5 % quantile moves by about 0.003 with the seed). For any other model, pass
  P&L simulated under it; the rolling backtest draws every day's P&L from that day's scenarios (2,000 histories).
* The Newey-West variance is a direct sum of the autocovariances; Hull and White's EWMA is a first-order filter
  (`scipy.signal.lfilter`) started at the window's mean square.

## Validation

`tests/test_backtest.py` checks every number against a reference computed in the test or published:

| Check | Reference | Result |
| --- | --- | --- |
| Kupiec LR at $n = 250$, $p = 1$ % for 0, 4, 5, 10 exceptions | the formula with `math.log` | 5.0252, 0.7691, 1.9568, 12.9555, to 1e-12 relative |
| 10 exceptions: $\chi^2$ p-value and $P(X \ge 10)$ | `scipy.stats` | 0.0003 and 0.0003 |
| Cumulative probability of 0 to 10 exceptions | BCBS (1996) Table 2 | within its printed 0.01 % |
| Zones and multipliers | MAR32.9 Table 1 | identical |
| Traffic light type 1 and type 2 errors at 98 % to 95 % coverage | BCBS (1996) Table 1 | within its printed 0.1 % |
| Exceptions on days 100 to 104 of 250 | the formula | counts (243, 1, 1, 4), $\mathrm{LR}_{\mathrm{ind}}$ 30.985 |
| Christoffersen's simulated power, 8 days | exact enumeration of 256 paths | within 4 standard errors |
| $Z_2$ 5 % critical value, 250 days, $\alpha = 2.5$ %: Normal, $t_{100}$, $t_{10}$, $t_5$, $t_3$ | Acerbi and Szekely (2014) Table 4: −0.70, −0.70, −0.71, −0.74, −0.82 | within 0.02 |
| $Z_2$ null and power at 20 % more volatility | brute-force simulation of full histories | within 0.02 |
| FZ0 at $\alpha = 5$ % for N(0, 1) | closed form; true VaR 1.645, ES 2.063 | minimised at (1.645, 2.063) within 0.002 |
| Kupiec rejections of the true VaR, 200 replications of 2,500 days | binomial 95 % band around 5 % | inside (4 to 16) |
| PLA metrics | `scipy.stats.spearmanr`, `ks_2samp` | to 1e-12 relative |
| KS p-values at 0.09 and 0.12 | MAR32.42 quotes 0.264 and 0.055 | 0.2634 and 0.0546 |
| Newey-West variance | explicit sums | to 1e-11 relative |
| Hypothetical P&L of the 45-day straddle | Black-Scholes in the test | to 1e-10 relative |

## Power

How often each test, at a nominal 5 %, rejects a model at 99 % VaR (97.5 % ES for $Z_2$), by sample size. The
first row of each test is its actual size, the others its power; the simulated rows carry a standard error of at
most 0.5 point (Christoffersen) and 0.2 point ($Z_2$). `tests/test_backtest.py` recomputes this table.

| Test, true model | 250 days | 500 days | 1,000 days | 2,520 days |
| --- | --- | --- | --- | --- |
| Kupiec, correct model | 9.5 % | 7.1 % | 5.5 % | 5.8 % |
| Kupiec, true exception rate 2 % | 24.3 % | 54.4 % | 78.2 % | 98.7 % |
| Traffic light out of green, correct model | 10.8 % | 6.7 % | 8.2 % | 5.3 % |
| Traffic light out of green, true exception rate 2 % | 56.1 % | 66.9 % | 89.7 % | 99.4 % |
| Christoffersen independence, independent exceptions | 1.3 % | 1.4 % | 1.7 % | 1.9 % |
| Christoffersen independence, 25 % chance of an exception after one | 34.1 % | 52.9 % | 78.6 % | 96.7 % |
| $Z_2$, correct model | 5.0 % | 5.2 % | 5.2 % | 5.0 % |
| $Z_2$, volatility 20 % above the model's | 77.3 % | 95.1 % | 99.8 % | 100.0 % |
| $Z_2$, Student-t(5) tails at the model's variance | 15.8 % | 21.3 % | 28.9 % | 48.3 % |

A year of exceptions cannot tell a 99 % VaR from one that is breached twice as often: Kupiec's test rejects the
second only 24 % of the time while rejecting a correct model 9.5 % of the time. Christoffersen's test is badly
undersized with so few exceptions (1.3 % at a nominal 5 %). $Z_2$ sees a level error quickly but a pure tail-shape
error slowly.

## Synthetic history results

The 45-day at-the-money short straddle of the review panel's case (index at 5,800 at the start, rate 4.3 %, dividend
yield 1.3 %, one 100-share contract) re-struck at every close of the synthetic history, window 500 days, 2,020 days
of backtest at 99 % (20.2 exceptions expected; amber from 28, red from 39):

| Model | Exceptions, seed 0 | Zone, seed 0 | Exceptions over seeds 0 to 9 |
| --- | --- | --- | --- |
| Normal, implied vol frozen (5,000 paths a day) | 50 | red | 42 to 76, red every time |
| Normal with index implied-vol dynamics, the Risk tab's default (5,000 paths a day) | 31 | amber | 21 to 39: 4 green, 5 amber, 1 red |
| Historical simulation with Hull and White scaling | 17 | green | 15 to 25, green every time |

$Z_2$ rejects the frozen-vol model at every seed, at 99 % and 97.5 %. The slow test reruns seed 0; the ten-seed
ranges are from scratch runs of about 20 s each.

## Limits

* **The synthetic history is not the market.** Its log implied vol is symmetric, with no spikes: the VIX jumps up
  and decays, while this index also dips lower than the VIX ever has (to 5.4 % to 5.8 % over 40 synthetic years at
  seeds 0 to 2, against a VIX low near 9 %). The spot's variance is set by the index with a day's lag times the
  GARCH factor, so implied vol is a better forecast of realised vol than it is in markets. Results on it show the
  tests work, not how a model would fare on market data.
* **The Monte Carlo model uses the app's seed every day**, as the app does, so its simulation error is the same each
  day rather than averaging out (with 10,000 paths the 99 % VaR of a share is its 1.06 % quantile).
* **The traffic light beyond 250 days at 99 % is an extension.** BCBS (1996) gives the cumulative-probability rule
  for other sample sizes at 99 %; applying it at 97.5 % is this library's choice, and no multiplier is defined there.
* **$Z_2$'s Normal and Student-t nulls assume a zero-mean scale family.** For an option position's P&L, pass the
  model's simulated P&L (the rolling backtest does).
* **Diebold-Mariano is asymptotic.** With AR(1) loss differentials (coefficient 0.3, 1,000 days) it rejects 5.2 %
  of the time at a nominal 5 % with the default lags (13.8 % without lags, which is why the lags are there); no
  small-sample correction is applied.

## References

* Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*
  3(2), 73-84. [doi:10.3905/jod.1995.407942](https://doi.org/10.3905/jod.1995.407942)
* Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review* 39(4), 841-862.
  [doi:10.2307/2527341](https://doi.org/10.2307/2527341)
* Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction
  with the internal models approach to market risk capital requirements.
  [bis.org/publ/bcbs22.htm](https://www.bis.org/publ/bcbs22.htm)
* Basel Committee on Banking Supervision (2020). MAR32, Backtesting and P&L attribution test requirements (version
  of 27 March 2020, in force from 1 January 2023).
  [bis.org](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327)
* Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. MSCI research paper, December 2014.
  [msci.com](https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b)
* Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and
  Value-at-Risk). *Journal of Econometrics* 211(2), 388-413.
  [doi:10.1016/j.jeconom.2018.10.008](https://doi.org/10.1016/j.jeconom.2018.10.008)
* Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic
  Statistics* 13(3), 253-263. [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
* Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix. *Econometrica* 55(3), 703-708. [doi:10.2307/1913610](https://doi.org/10.2307/1913610)
* Hull, J. and White, A. (1998). Incorporating volatility updating into the historical simulation method for
  value-at-risk. *Journal of Risk* 1(1), 5-19. [doi:10.21314/JOR.1998.001](https://doi.org/10.21314/JOR.1998.001)
