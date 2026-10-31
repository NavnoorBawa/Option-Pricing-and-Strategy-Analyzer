# P&L explain: a second-order Greek waterfall and a Shapley split

`options_analyzer/attribution.py` explains the P&L of a market move (spot, implied vols, rate and time) in two ways,
leg by leg and for the position, and computes the two statistics of the Basel P&L attribution test. The numbers below
are reproduced by `tests/test_attribution.py`.

## Model

Every leg is valued as `strategies.position_value` values it: Black-Scholes-Merton at the leg's own implied vol and
remaining time (ACT/365), a share at its price plus its accrued dividends. The move takes today's market, spot $S_0$,
each leg's vol $\sigma_i$, rate $r$, to spot $S_1 = S_0 + \delta S$, vols $\sigma_i + \delta\sigma_i$ and rate
$r + \delta r$, $\delta t$ = `days` / 365 years later. The P&L is the full revaluation

$$\text{P\&L} = \sum_i q_i \big[V_i(S_1, \sigma_i + \delta\sigma_i, r + \delta r, \delta t) - V_i(S_0, \sigma_i, r, 0)\big],$$

each leg's two values differenced on their own, so that a small P&L is not lost in the rounding of a large value. The
vol shifts are given as shifts (`vol_shift`) or as the legs' new vols (`vols`, as when re-marking legs to a surface;
then $\delta\sigma_i$ is the new vol less today's). The rate shift moves the options' rate only, as in
`position_value`; the dividends a share accrues keep today's rate.

## The Greek waterfall (`taylor_explain`)

With the Greeks of today (`strategies.position_risk`, leg by leg),

$$
\begin{aligned}
\text{P\&L} &= \Delta\,\delta S + \tfrac12\Gamma\,\delta S^2 + \nu\,\delta\sigma + \Theta_{\delta t} + \rho\,\delta r
  + \text{vanna}\,\delta S\,\delta\sigma + \tfrac12\,\text{volga}\,\delta\sigma^2 \\
&\quad + \text{charm}\,\delta S\,\delta t + \text{veta}\,\delta\sigma\,\delta t + U,
\end{aligned}
$$

where a term with $\delta\sigma$ sums over the legs, each with its own Greek and shift, and $U$ is the unexplained
P&L. Each term is reported in dollars, the Greeks in desk units: vega per vol point times the shift in points, rho per
1 % times the shift in percent, charm and veta per calendar day times the days.

**Theta by repricing.** $\Theta_{\delta t} = V(S_0, \sigma, r, \delta t) - V(S_0, \sigma, r, 0)$, the change in
value with nothing but time moving (for one day, `strategies.time_decay`), not $\theta\,\delta t$. It is exact for
the passage of time alone, so a quiet day is explained in full; $\theta\,\delta t$ leaves
$\tfrac12 \partial_{tt} V\,\delta t^2$ and the higher orders unexplained: $6.5 \times 10^{-4}$ per unit (0.8 % of the
one-day decay) on a 30-day at-the-money straddle, 98 % of it the $\delta t^2$ term, and half the decay of a 1-day
at-the-money option (an index put worth 21.56 decays by 21.56, while $\theta / 365$ reads $-10.66$). There the
$\delta t^2$ term is only $-2.73$ of the $-10.90$ gap: an at-the-money option's value near expiry is about
$c\sqrt{T - t}$, which no series in powers of $\delta t$ follows over its last day ($\theta\,\delta t = -V/2$,
$\tfrac12 \partial_{tt} V\,\delta t^2 = -V/8$, decay $-V$). Charm and veta correct today's delta and vega for the
time elapsed: the cross effect of a move and the time by revaluation,
$V(S_1, \delta t) - V(S_1, 0) - V(S_0, \delta t) + V(S_0, 0)$, differs from $\text{charm}\,\delta S\,\delta t$ by
1.3 % on a 30-day risk reversal over a day with a $-1$ % move, and the gap halves as the move and the time halve
(the same for veta: 0.9 % on the straddle with $+2$ vol points).

**What is left.** $U$ is third order in $\delta S$ and $\delta\sigma$: on an iron condor (strikes 90/95/105/110,
30 days, vol 20 %, rate 4 %), halving a $-4$ % spot and $+6$ vol point move three times divides $U$ by 7.84, 8.10
and 8.11 (7.87, 8.11 and 8.11 with a day of time halved along). The second-order terms of a rate move are not in the
expansion: with a 25 bp rise as well the ratios are 7.64, 7.65 and 7.29, heading for 4. The largest is
$\partial_{Sr}V\,\delta S\,\delta r$, which a straddle's near-zero rho hides: \$1.14 per contract on the stress day
below with a 25 bp rise, against a rho term of −\$0.05.

**Quality.** $|U| / \max(|\text{P\&L}|, f)$, the share of the P&L the Greeks miss, with the floor $f$ a tenth of
the gross of the nine terms, $0.1 \sum_k |T_k|$ (a leg's own terms for a leg, the position's net terms for the
position). Small moves need no floor, since $U$ is third order in them; the floor is for a P&L that is a small
difference of larger amounts. A long straddle on a day when a 1 % rise is offset by a 0.614-point vol fall makes
\$0.0073 a contract, its delta (+\$6.85), gamma (+\$6.93) and vega (−\$13.99) terms cancelling, and the Greeks miss
\$0.144, 20 times the P&L; against the floor, \$2.785, the quality is 0.052. Whatever the move, the quality is at most
$1 + 1/0.1 = 11$: when $|U| > 1.1 \sum_k |T_k|$ the P&L is at least $|U| - \sum_k |T_k|$, and otherwise the floor is
at least $|U| / 11$. A floor in the notional, $10^{-4} \sum_i |q_i| S_0$, would be the wrong scale for options: for a
7-day index iron condor (4 legs, spot 5,800, 100 shares a contract) it is \$232, more than the condor's whole day of
time decay (\$151.97). On a day when spot falls 0.5 % and implied vol rises half a point the condor makes \$43.15 and
the Greeks miss \$6.21, which the notional floor would report as a quality of 0.027 instead of 0.144 (an iron
butterfly: 0.012 instead of 0.176). Against a tenth of the terms neither day is floored.

**The stress day.** A 30-day at-the-money short straddle (spot and strike 100, vol 20 %, rate 4 %, no dividends,
one 100-share contract), spot $-4$ % and implied vol $+6$ points over one day:

| Term | \$ | Term | \$ |
|---|---:|---|---:|
| delta | 27.42 | vanna | −2.73 |
| gamma | −110.91 | volga | −0.05 |
| vega | −136.74 | charm | −0.46 |
| theta | 7.69 | veta | 2.26 |
| rho | 0.00 | unexplained | 25.17 |
| | | **full P&L** | **−188.36** |

The quality is 0.134. Most of $U$ is zomma, $\tfrac12\partial_{SS\sigma}V\,\delta S^2\,\delta\sigma = +\$33.19$: the
higher vol lowers the straddle's gamma. The project's plan quoted a "second-order residual" of 27.04 for this day; that
is $U$ without the charm and veta terms and with $\theta\,\delta t$ for theta (the test rebuilds it to 0.01).

## The Shapley split (`shapley_explain`)

The P&L is shared between the four factors $F$ = {spot, vol, time, rate}. With $v(A)$ the P&L when only the factors
in $A$ move (16 revaluations of each leg, in one vectorised call),

$$\phi_i = \sum_{A \subseteq F \setminus \{i\}} \frac{|A|!\,(3 - |A|)!}{4!}\,\big(v(A \cup \{i\}) - v(A)\big),$$

the weights being $\tfrac14, \tfrac1{12}, \tfrac1{12}, \tfrac14$ for $|A| = 0, 1, 2, 3$: the average of factor $i$'s
contribution over the 24 orders in which the factors can be moved one after the other (Shapley 1953). The parts add up
to the P&L (efficiency), a factor that does not move gets exactly zero, and they are additive over legs. A sequential
attribution credits each interaction to whichever factor moves later: on the stress day above its spot part ranges
from −\$84.87 to −\$59.72 and its vol part from −\$136.77 to −\$111.18 over the 24 orders, where the Shapley
parts are spot −\$72.17, vol −\$124.04, time +\$7.85 and rate 0.

## P&L attribution statistics (`pla_metrics`)

For a desk's hypothetical P&L (HPL) and risk-theoretical P&L (RTPL), day by day, as MAR32.36 to 32.41 define them:
the Spearman correlation is the Pearson correlation of the two series' ranks (ties get their average rank, which
MAR32 leaves open; a constant series has no ranking and gives NaN), and the Kolmogorov-Smirnov statistic is the
largest absolute difference between the two empirical distribution functions, $F(x)$ = (observations $\le x$) / $n$.
MAR32.42 puts a desk in the green zone when the correlation is above 0.80 and the KS statistic below 0.09, and in the
red zone when the correlation is below 0.70 or the KS statistic above 0.12; zoning belongs to the backtest.

## Error

| Check | Reference | Measured worst | Tolerance |
|---|---|---:|---:|
| Taylor terms and P&L, 3 positions with dividends, skew shifts and a rate move | mpmath derivatives of BSM written out in the test, 40 digits | $1.5 \times 10^{-14}$ per unit | $10^{-11}$ |
| Shapley parts | average of the 24 sequential attributions of the whole position | $6.2 \times 10^{-15}$ | $10^{-12}$ |
| Shapley parts add up to the P&L, 26 templates × 4 moves × 3 markets | | $1.8 \times 10^{-15}$ | $10^{-9}$ |
| Legs add up to the position | Greeks and revaluations of the whole position | $1.4 \times 10^{-14}$ | $10^{-11}$ |
| One day of time alone, theta + charm (by construction: a regression check) | `position_value` a day later, including 1-day index options | $4.5 \times 10^{-13}$ | $10^{-6}$ |
| Charm and veta, a $-1$ % move or $+2$ vol points over a day, then both halved 3 times | the cross effect by 4 revaluations | gap 1.3 % (charm, risk reversal), 0.9 % (veta, straddle), halving | each halving divides it by 1.3 or more |
| Quality with the default floor, every template, random moves | the bound $1 + 1/0.1$ | at most 11 | 11 |
| Spearman and KS on 250-day series with ties | SciPy `spearmanr` and `ks_2samp` | $6.9 \times 10^{-18}$ | $10^{-12}$ |

## Limits

- Black-Scholes-Merton legs with sticky-strike vols: a vol shift is what the caller passes for each leg; the spot
  move does not move the vols (no smile dynamics), and the Shapley "vol" factor moves all legs' vols together.
- The expansion has no second-order rate terms (see above) and no $\partial_{tt}V$ term (theta is repriced instead).
- A leg that expires within the period is settled at its intrinsic value, which no expansion of today's value
  follows: the Greeks then miss much of that leg's P&L (31 % of a short 30-day call's over 40 days in the tests), and the
  quality shows it.
- The quality floor is a convention, not a statistic; pass `floor` to use a desk's own threshold in dollars.

## References

- Basel Committee on Banking Supervision (2020), *Basel Framework*, MAR32 "Internal models approach: backtesting and
  P&L attribution test requirements", version in force from 1 January 2023, paragraphs 32.34 to 32.42,
  <https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327>.
- L. S. Shapley (1953), "A value for n-person games", in H. W. Kuhn and A. W. Tucker (eds.), *Contributions to the
  Theory of Games II*, Annals of Mathematics Studies 28, Princeton University Press, 307-317,
  doi:[10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018).
