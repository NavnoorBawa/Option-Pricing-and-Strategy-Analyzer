## Data and time

### Sources
The app fetches no market data. Spot, strike, days to expiry, volatility, rate and dividend yield are typed in the
sidebar. The Surface tab draws either a parametric SSVI surface or option quotes uploaded as a CSV file; its two
sample files are generated from the sidebar's spot, rate and dividend yield.

### Time
* Time to expiry $T$ = calendar days / 365. The sidebar takes whole days from 1 to 3,650.
* Value-at-Risk horizons are in trading days (252 per year), because volatility accrues with trading time.

### Maturities in uploaded files
A maturity header may state days, years, months, weeks, hours or minutes ('Maturity (months)'); a bare 'm' could
be months or minutes and is refused, and times to expiry within half a day of each other are taken as one expiry
(keeping each quote's total variance). A maturity column whose header gives no unit is converted from days when
its values go above 30, refused when they are whole numbers from 5 to 30 (they could be years or days), and
otherwise read as years, with a warning to name the unit when they are small whole numbers or go above 3 (listed
options rarely run longer), as they could then be days.
