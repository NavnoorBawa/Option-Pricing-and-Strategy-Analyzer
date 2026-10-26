### Local volatility
Surfaces are handled as total implied variance $w(y,T) = \sigma_{\text{BS}}^2 T$ in forward log-moneyness
$y = \ln(K/F_T)$, and local variance follows Gatheral's form of Dupire's equation

$$\sigma_{\text{loc}}^2 = \frac{\partial_T w}{1 - \frac{y}{w}\partial_y w + \frac14\left(-\frac14 - \frac1w + \frac{y^2}{w^2}\right)(\partial_y w)^2 + \frac12 \partial_{yy} w}.$$

The numerator and denominator are exactly the calendar and butterfly no-arbitrage conditions; where either
fails local volatility is undefined and is reported rather than patched. The parametric surface is SSVI
(Gatheral & Jacquier, 2014), arbitrage-free when $\eta(1+|\rho|) \le 2$. Uploaded quotes (implied vols, or
option prices inverted with Black–Scholes–Merton; where a Type column gives both a call and a put at one
strike, the out-of-the-money one is used) are fitted per maturity with Gatheral's raw SVI smile

$$w(y) = a + b\left(\rho\,(y-m) + \sqrt{(y-m)^2 + s^2}\right)$$

by least squares on implied vols, with $b \ge 0$, $|\rho| < 1$, $s > 0$, a non-negative minimum variance and
Lee's wing bound $b(1+|\rho|) \le 2$. The search starts from Zeliade's quasi-explicit fits: for fixed $m$ and
$s$ the smile is linear in $a$, $bs$ and $\rho bs$, solved on a grid of $(m, s)$ with $|\rho| < 1$ imposed, so
steep, nearly linear equity skews are followed too. SSVI slices are exactly SVI, so noise-free SSVI quotes are
recovered. The fit can smooth over quotes that are not free of arbitrage, so the quotes themselves are also
checked without a model: a butterfly of three neighbouring strikes that costs less than nothing even if each
quote's vol is half a point off, or a put priced below one at a lower strike (a call above one) even if each
is a vol point off, is reported as static arbitrage in the quotes. Both checks also let each price be off by
one step of the file's price grid (a midpoint of bids and asks in 0.05 ticks lies on a 0.025 grid) and, where
the file gives a bid and an ask, by half the spread, or as far as the farther of the two for a price away from
their midpoint, so a reported butterfly costs less than nothing bought at the ask and sold at the bid whether
the price is their midpoint or a Price, Mid or Mark column. The price behind a quoted implied vol is unknown,
so it may be off by 0.01% of the forward or 0.005 (at most 1% of the strike spacing, for sub-cent strike
grids), whichever is more, and arbitrage that half a 0.05 tick (at most 5% of the strike spacing) in each price
would remove is reported as possible rather than certain: vols inverted from rounded prices can show it. A maturity header may state days,
years, months, weeks, hours or minutes ('Maturity (months)'); a bare 'm' could be months or minutes and is
refused, and times to expiry within half a day of each other are taken as one expiry (keeping each quote's
total variance). A maturity column whose header gives no unit is converted from days when its values go above
30, refused when they are whole numbers from 5 to 30 (they could be years or days), and otherwise read as years,
with a warning to name the unit when they are small whole numbers or go above 3 (listed options rarely run
longer), as they could then be days.
$\partial_y w$ and $\partial_{yy} w$ are analytic; $\partial_T w$ at fixed $y$ comes from secants between
neighbouring maturities evaluated at the same $y$ — their Fritsch–Butland weighted harmonic mean inside, the
one-sided three-point formula at the first maturity (the one-sided secant where that is not positive), and the
one-sided secant at the last, where quote noise outweighs the secant's truncation error. A secant that falls
by more than the quotes' precision is calendar arbitrage; one that changes by less (two expiries a day apart,
say) cannot resolve $\partial_T w$ and is not used, and where no secant is left local vol is left blank.
Implied vol is shown only inside each maturity's quoted strikes, and local vol only where the neighbouring
maturities are quoted near the same forward moneyness.
