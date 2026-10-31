## Surface

### Local volatility
Surfaces are handled as total implied variance ${w(y,T) = \sigma_{\text{BS}}^2 T}$ in forward log-moneyness
${y = \ln(K/F_T)}$, and local variance follows Gatheral's form of Dupire's equation

$$
\begin{aligned}
\sigma_{\text{loc}}^2 &= \frac{\partial_T w}{g}, \\
g &= 1 - \frac{y}{w}\,\partial_y w \\
  &+ \frac14\left(-\frac14 - \frac1w + \frac{y^2}{w^2}\right)(\partial_y w)^2 + \frac12\,\partial_{yy} w.
\end{aligned}
$$

The numerator and $g$ are exactly the calendar and butterfly no-arbitrage conditions; where either fails
local volatility is undefined and is reported rather than patched.

$\partial_y w$ and $\partial_{yy} w$ are analytic. $\partial_T w$ at fixed $y$ has two parts. A smile that keeps
its shape in $y/\sqrt{\theta}$, with $\theta(T)$ the at-the-money total variance (as an SSVI smile nearly does, and
an equity index skew roughly does), changes with $T$ at a rate its own slice gives:

$$
\partial_T w = \frac{\theta'(T)}{\theta}\left(w - \tfrac12\,y\,\partial_y w\right),
$$

with $\theta'$ from the parabola through $\ln\theta$ against $\ln T$ at the maturity and its two neighbours,
through the last three at the last maturity. There that parabola is kept between the slopes a straight line and a
power law through the last two slices give, and $\theta'$ is at most three times the secant of $\theta$ next to it.
What the smile does beyond that scaling comes from the parabola in $T$ through the neighbouring slices at the same
$y$: through the first three maturities at the first, the maturity and its two neighbours inside, and from the line
through the last two at the last, where noise in total variance, which grows with $T$, outweighs the truncation
error. Secants in $T$ alone miss the scaling, whose share of $w$ near the money grows like $\sqrt{T}$: with
noise-free SSVI quotes at the listed expiries from 30 days to two years, local vol was up to 2.1 vol points off at
30 days with them, and is within 0.1 vol point now (0.45 on weekly expiries to 28 days followed by 60, 91, 182 and
365 days, at the year after a half-year gap; 0.07 every two weeks to a year; 0.03 on quarterly expiries to two
years; 0.54 and 0.63 on a steep equity skew on the weekly and the listed expiries, against 1.37 and 0.96 with
secants); a flat surface gives its flat local vol exactly. With quotes noisy to 0.2 vol point the noise dominates:
the RMS error is 0.93 vol point at the first expiry, 0.58 inside and 0.87 at the last (1.28, 0.56 and 0.89 with
secants).

The last expiry is a trade-off. With $\theta'$ from the last two slices alone it was worse than with secants on a
third of clean surfaces, by up to 3.3 times (1.96 against 1.19 vol points on that equity skew, weekly expiries);
from the last three without the bracket its clean error halves again, but an at-the-money vol falling from 80% to
20% within months, on expiries at 30, 91, 365 and 730 days, puts it 4.2 vol points off instead of 0.74, and the
noise grows. With the bracket it is no worse than with secants on any of 245 clean surfaces on the listed, weekly
and quarterly expiries, but it still is on 8 of 52 where the last two expiries are a year apart (30, 91, 365 and
730 days), by up to 2.4 times, and on the synthetic index chain, whose prices sit on ticks, its last expiry is 0.60
vol point off on average over 16 draws of the chain's noise, against 0.55 with secants.

Secants between neighbouring maturities, evaluated at the same $y$, still decide where $\partial_T w$ stands: one
that falls by more than the quotes' precision is calendar arbitrage, and one that changes by less (two expiries a
day apart, say) cannot resolve $\partial_T w$, so where no secant is left local vol is left blank. Where the formula
above cannot use all three maturities (one quoted far from the forward or from $y$, a change within the quotes'
precision, or a slope above three times the smaller secant: Hyman's limit, beyond which a cubic through the slices
is no longer guaranteed to be monotone) it uses the maturity and one neighbour, and failing that the secants give
$\partial_T w$: their Fritsch–Butland weighted harmonic mean inside, the one-sided three-point formula at the first
maturity and the one-sided secant at the last.
Implied vol is shown only inside each maturity's quoted strikes, and local vol only where the neighbouring
maturities are quoted near the same forward moneyness.

The truncation error of local vol is estimated at each maturity as the largest change over its strikes when
$\partial_T w$ comes from one neighbouring slice more. On noise-free quotes it is close to the actual error: over
1,196 maturities of 60 random SSVI surfaces its median is 1.11 times the error, and twice it covers the error at 91%
of them. It is an estimate, not a bound: it falls several-fold short where the expiries are too far apart for the
term structure (0.6 against 2.5 vol points at a year on expiries at 30, 91, 365 and 730 days with an at-the-money
vol falling from 80% to 20%), and with noisy quotes it moves with the noise rather than measuring it.

### Forwards and discounting
Option prices are inverted at each expiry's own forward $F$ and discount factor $D$, read from the quotes by
put–call parity: for European options ${C(K) - P(K) = D\,(F - K)}$ at every strike, so the call-minus-put prices of
an expiry lie on a line in $K$ with slope ${-D}$ and root $F$. The line is fitted by weighted least squares to the
call/put pairs within the wider of 8% and 1.5 standard deviations ($\sigma\sqrt{T}$, read from the at-the-money
straddle) of a first-guess forward, each pair weighted by ${1/(e_C + e_P)^2}$ with $e$ half its bid/ask spread, and
pairs more than 3.5 scaled median absolute deviations from the line (and outside their own spreads) are removed.
Where the slope is too uncertain (standard error above 0.0005, typical of short expiries with few strikes) or
implies a rate outside −10% to 50%, $D$ comes from the sidebar rate plus the spread over it that the other
expiries' slopes measured (the sidebar rate alone where none did) — options discount at a financing (box) rate
tens of basis points above Treasuries — and $F$ is the weighted mean of ${K + (C-P)/D}$. An expiry with fewer than
six strikes quoted on both sides keeps the sidebar rate and dividend yield. Black-76 with $F$ and $D$ is
Black–Scholes–Merton at the rate ${-\ln D / T}$ and the yield ${-\ln D / T - \ln(F/S)/T}$, so the existing
solver inverts each quote at its own expiry's carry; the Forwards and discounting table on the Surface tab lists
them with the gap to the sidebar forward in basis points.

On the synthetic SPX chain (spot 5,000, 26 October 2026 at 10:00, box rate 45 bp above the Treasury rate) read
with the Treasury rate and no dividend yield, the parity forwards come out within 0.22 bp of the generator's and
the at-the-money vols within 0.0064 vol point, with no arbitrage point and fit RMSEs of 0.073 vol point at most,
whichever the calibration. With the sidebar forward instead (the Sidebar r, q choice) the forwards are up to 142 bp
off and the at-the-money vols up to 0.17 vol point; fitted unconstrained, the surface has 69 calendar and 40
butterfly grid points and fits up to 1.60 vol points from the quotes, and the arbitrage-free fit holds those out at
an RMSE of up to 1.98. With the generator's own 1.2% yield in the sidebar the forward is still up to 85 bp off and
the RMSE up to 2.38, whichever the calibration.

Parity is an equality only for European exercise: an American put is worth more than parity says by its
early-exercise premium, which parity reads as a lower forward (on a chain priced as American with no dividend, a 5%
rate and a 30% vol, expiries from 30 days to two years, as an implied dividend yield of 0.9–1.6% and forwards 9 to
179 bp low, which would have raised the carry-gap warning at every expiry). When the sidebar's pricing model is
American, parity is not used: every expiry takes the sidebar rate and dividend yield, and the Forwards control says
so. The prices are still inverted as European options, which overstates their implied vols by the early-exercise
premium, until de-Americanisation (roadmap items PR-4 and SF-5) is in.

In-the-money quotes whose time value at the expiry's forward is within their own bid/ask spread are dropped: they
are the in-the-money side at strikes whose out-of-the-money option has no bid, and say nothing about the vol.

### SSVI and SVI fits
The parametric surface is SSVI (Gatheral & Jacquier, 2014), arbitrage-free when ${\eta(1+|\rho|) \le 2}$. Uploaded
quotes (implied vols, or option prices inverted with Black-76 at each expiry's forward; where a Type column gives
both a call and a put at one strike, the out-of-the-money one against that forward is used) are fitted per
maturity with Gatheral's raw SVI smile

$$
w(y) = a + b\left(\rho\,(y-m) + \sqrt{(y-m)^2 + s^2}\right)
$$

by least squares on implied vols, with ${b \ge 0}$, ${|\rho| < 1}$, ${s > 0}$, a non-negative minimum variance and
Lee's wing bound ${b(1+|\rho|) \le 2}$ (a penalty in the search, then $b$ cut to the bound, so no fitted wing is
steeper than 2 with or without the arbitrage constraints). The search starts from Zeliade's quasi-explicit fits:
for fixed $m$ and $s$ the smile is linear in $a$, $bs$ and $\rho bs$, solved on a grid of $(m, s)$ with
${|\rho| < 1}$ imposed, so steep, nearly linear equity skews are followed too. SSVI slices are exactly SVI, so
noise-free SSVI quotes are recovered.

Each quote's vol error is weighted by one over its half-spread in vol (its price error, half the bid/ask spread,
over its vega), divided by the median weight and capped at five times it, so tight at-the-money quotes count more
than wing quotes a few ticks wide; quoted implied vols, whose spreads are unknown, weigh equally, and prices with
no bid/ask and on no price grid are weighted by their vega. Where wing spreads in vol are about 15 times the
at-the-money ones, the weighted fit's largest error within one standard deviation of the money is 0.58 of the
unweighted fit's. The summary table says how each expiry's fit sits within the spreads: the share of quotes whose
fitted vol is inside their bid/ask in vol, and the median error in half-spreads (blank where the quotes have no
spread). Its Fit RMSE is unweighted, so it can exceed an unweighted fit's: on a $100 chain quoted $0.05 either side
of tick-rounded mids the 30-day fit is 0.73 vol point RMSE from the quotes (an unweighted fit, 0.68), with every
fitted vol inside its bid/ask and a median error of 0.18 half-spread.

By default the calibration is arbitrage-free. Each smile must keep Durrleman's ${g > 0}$ (a positive density) over
its quoted range widened by 10% and out to its wing reach, four at-the-money standard deviations ${4\sqrt{w(0)}}$
(at most ${|\ln K/F| = 1.5}$), and its total variance must not fall below the previous expiry's where both are
quoted. Both are checked on 2,001 points (more around the smile's vertex); where either fails, the smile is replaced
by the nearest one that meets ${g \ge 0.001}$ and ${w_T \ge w_{T^-}}$, found by sequential quadratic programming
with analytic gradients from the unconstrained fit (and, when that ends more than 10% worse, from the quasi-explicit
starts until one ends within 10%, and if none does, from the previous expiry's smile raised to these quotes and from
a nearly flat smile above it, which meets both constraints), with the points that still fail added to the
constraints (a few of each stretch of them where there are more than 200). On arbitrage-free quotes nothing binds
and the fit is unchanged; Gatheral and Jacquier's arbitrageable example slice (Vogt's) is fitted 0.63 vol point
(RMSE) away with a positive density, and over 200 random arbitrage-free surfaces with 0.2 vol point of noise no
expiry's RMSE grew by more than 8.6% and the slowest surface took 0.6 s. The search has a budget of 600 solver
iterations per expiry and 4,000 per surface, which none of those surfaces, the sample quotes or the synthetic SPX
chains comes near (at most 419 on an expiry and 940 on a surface). 20 expiries of 41 strikes with 8 vol points of
noise, which took 2.8 to 14 s before these budgets and a faster search (3.4 to 5.0 s on one processor thread), now
take 1.8 to 2.9 s on a lightly loaded four-core machine, up to 10 s when it is busy, and 2.4 to 4.5 s on one thread,
using 2,761 to 3,872 iterations; the surface budget caps any chain at about 3 to 4.5 s of search on one thread,
however noisy. An expiry where no start finds an arbitrage-free smile within the budgets keeps its unconstrained
fit, and a note names it, the arbitrage left and whether the surface's budget had run out. The summary table's
Arbitrage cost is what the constraints add to each expiry's spread-weighted RMSE (the error the fit minimises; blank
where they could not be met), a note names the expiries where the cost is material, and the Unconstrained
calibration shows the quotes' own arbitrage instead.

### Wing extrapolation
Beyond its quotes a smile is read off its SVI wings out to the same reach of four standard deviations from the
forward, at most ${|\ln K/F| = 1.5}$, where the density is held positive; an SVI wing grows linearly with a
total-variance slope ${b(1+|\rho|) \le 2}$, Lee's moment bound (an inequality of the arbitrage-free search; the
unconstrained fit penalises a steeper wing and then cuts ${b}$ to the bound, as the penalty alone left slopes up to
2.0024 on quotes far steeper than Lee allows). Extrapolated vols are labelled as such: the contract readout says
"extrapolated", and the Smiles chart draws them dotted beyond each expiry's quotes. Local vol is not extrapolated.

The reach is fixed rather than a setting, because it is part of the fit: the arbitrage-free calibration holds the
density positive out to it, so a vol read further out would come from a wing nobody checked for arbitrage, and a
reach changed by the user would change every fit. Four standard deviations lie well beyond a 1-delta strike (about
2.3); further out the straight SVI wing is an assumption, not a measurement. The cap of 1.5 (K/F from 0.22 to 4.5)
binds only for long, volatile expiries; without the standard-deviation scaling it would hold a six-hour expiry's
density positive a hundred standard deviations out, which bent its smile 0.3 vol point off at the money.

### Static arbitrage in the quotes
The fit can smooth over quotes that are not free of arbitrage, so the quotes themselves are also
checked without a model: a butterfly of three neighbouring strikes that costs less than nothing even if each
quote's vol is half a point off, or a put priced below one at a lower strike (a call above one) even if each
is a vol point off, is reported as static arbitrage in the quotes. Both checks also let each price be off by
one step of the file's price grid (a midpoint of bids and asks in 0.05 ticks lies on a 0.025 grid) and, where
the file gives a bid and an ask, by half the spread, or as far as the farther of the two for a price away from
their midpoint, so a reported butterfly costs less than nothing bought at the ask and sold at the bid whether
the price is their midpoint or a Price, Mid or Mark column. The price behind a quoted implied vol is unknown,
so it may be off by 0.01% of the forward or 0.005 (at most 1% of the strike spacing, for sub-cent strike
grids), whichever is more, and arbitrage that half a 0.05 tick (at most 5% of the strike spacing) in each price
would remove is reported as possible rather than certain: vols inverted from rounded prices can show it.
