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

$\partial_y w$ and $\partial_{yy} w$ are analytic; $\partial_T w$ at fixed $y$ comes from secants between
neighbouring maturities evaluated at the same $y$ — their Fritsch–Butland weighted harmonic mean inside, the
one-sided three-point formula at the first maturity (the one-sided secant where that is not positive), and the
one-sided secant at the last, where quote noise outweighs the secant's truncation error. A secant that falls
by more than the quotes' precision is calendar arbitrage; one that changes by less (two expiries a day apart,
say) cannot resolve $\partial_T w$ and is not used, and where no secant is left local vol is left blank.
Implied vol is shown only inside each maturity's quoted strikes, and local vol only where the neighbouring
maturities are quoted near the same forward moneyness.

### Forwards and discounting
Option prices are inverted at each expiry's own forward $F$ and discount factor $D$, read from the quotes by
put–call parity: for European options ${C(K) - P(K) = D\,(F - K)}$ at every strike, so the call-minus-put prices of
an expiry lie on a line in $K$ with slope ${-D}$ and root $F$. The line is fitted by weighted least squares to the
call/put pairs within the wider of 8% and 1.5 standard deviations ($\sigma\sqrt{T}$, read from the at-the-money
straddle) of a first-guess forward, each pair weighted by ${1/(e_C + e_P)^2}$ with $e$ half its bid/ask spread, and
pairs more than 3.5 scaled median absolute deviations from the line (and outside their own spreads) are removed.
Where the slope is too uncertain (standard error above 0.0005, typical of short expiries with few strikes), $D$
comes from the sidebar rate plus the spread over it that the other expiries measured — options discount at a
financing (box) rate tens of basis points above Treasuries — and $F$ is the weighted mean of ${K + (C-P)/D}$. An
expiry with fewer than six strikes quoted on both sides keeps the sidebar rate and dividend yield. Black-76 with
$F$ and $D$ is Black–Scholes–Merton at the rate ${-\ln D / T}$ and the yield ${-\ln D / T - \ln(F/S)/T}$, so the
existing solver inverts each quote at its own expiry's carry; the Forwards and discounting table on the Surface
tab lists them with the gap to the sidebar forward in basis points. On the synthetic SPX chain read with the
Treasury rate and no dividend yield, the forwards come out within 0.3 bp and the at-the-money vols within 0.01 vol
point of the generator's; with the sidebar forward 135 quotes fell outside the no-arbitrage bounds and the fit
reported 26 calendar violations. Parity is an equality only for European exercise.

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
Lee's wing bound ${b(1+|\rho|) \le 2}$. The search starts from Zeliade's quasi-explicit fits: for fixed $m$ and
$s$ the smile is linear in $a$, $bs$ and $\rho bs$, solved on a grid of $(m, s)$ with ${|\rho| < 1}$ imposed, so
steep, nearly linear equity skews are followed too. SSVI slices are exactly SVI, so noise-free SSVI quotes are
recovered.

Each quote's vol error is weighted by one over its half-spread in vol (its price error, half the bid/ask spread,
over its vega), divided by the median weight and capped at five times it, so tight at-the-money quotes count more
than wing quotes a few ticks wide; quoted implied vols, whose spreads are unknown, weigh equally. Where wing spreads
in vol are about 15 times the at-the-money ones, the weighted fit's largest error within one standard deviation of
the money is 0.58 of the unweighted fit's.

By default the calibration is arbitrage-free. Each smile must keep Durrleman's ${g > 0}$ (a positive density) over
its quoted range widened by 10% and out to its wing reach, four at-the-money standard deviations ${4\sqrt{w(0)}}$ (at
most ${|\ln K/F| = 1.5}$), and its total variance must not fall below the previous expiry's where both are quoted.
Both are checked on 2,001 points (more around the smile's vertex); where either fails, the smile is replaced by the
nearest one that meets ${g \ge 0.001}$ and ${w_T \ge w_{T^-}}$, found by sequential quadratic programming with
analytic gradients from the unconstrained fit (and from the quasi-explicit starts when that ends worse), with any
point that still fails added to the constraints. On arbitrage-free quotes nothing binds and the fit is unchanged;
Gatheral and Jacquier's arbitrageable example slice (Vogt's) is fitted 0.63 vol point (RMSE) away with a positive
density, and over 200 random arbitrage-free surfaces with 0.2 vol point of noise no expiry's RMSE grew by more
than 9% and the slowest surface took 0.65 s. The summary table shows what the constraints cost each expiry, a
note names the expiries where the cost is material, and the Unconstrained calibration shows the quotes' own
arbitrage instead.

### Wing extrapolation
Beyond its quotes a smile is read off its SVI wings out to the same reach of four standard deviations, where the
density is held positive; an SVI wing grows linearly with a total-variance slope ${b(1+|\rho|) \le 2}$, Lee's
moment bound. Extrapolated vols are labelled as such (the contract readout says "extrapolated"); local vol is not
extrapolated.

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
