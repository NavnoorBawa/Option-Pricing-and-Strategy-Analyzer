## Data and time

### Sources
The app fetches no market data. The sidebar's Market menu picks one of two markets:

* **Synthetic index** (the default): an SPX-like option chain that the app's own generator prices at the
  valuation time, labelled "Synthetic — not market data" wherever it appears. The level is 5,800, the Treasury
  zero rate a flat 4.00%, the options discount at the box rate 45 bp above it, and the dividend yield is 1.30%.
  SPX monthlies (AM-settled, third Friday) and SPXW weeklies (PM-settled) are listed, with bid/ask spreads and
  ticks of $0.05 below $3 and $0.10 from $3. The app then reads the chain as it would an upload: each expiry's
  forward and discount factor from put-call parity on its quotes, the prices inverted with Black-76, and an
  arbitrage-free, spread-weighted SVI smile fitted to each expiry (Volatility surface, below). That fitted surface
  is the one every tab uses: the contract is a listed expiry and a strike on the listed 5-point grid, its vol is
  the surface's at that strike and expiry (or a typed override, marked in amber), and its rate and dividend yield
  are the ones implied by the expiry's forward $F$ and discount factor $D$: ${r_T = -\ln D / T}$ and
  ${q_T = r_T - \ln(F/S) / T}$, so Black-Scholes-Merton with them is Black-76 on $F$ and $D$. Read back at
  23-Oct-26 16:00, the forwards are within 0.3 bp of the generator's, the discount factors within 2.0e-4, and the
  at-the-money vols within 0.006 vol point.
* **Textbook**: spot, strike, days to expiry, volatility, rate and dividend yield are typed (S = K = 100, one year,
  σ 20%, r 5%, q 0 to start, the inputs of version 2). A link with any of the typed inputs and no market opens
  this one, so links from version 2 reproduce their numbers.

The Surface tab draws the synthetic chain's surface (under the index), a parametric SSVI surface, or option quotes
uploaded as a CSV file; its two sample files are generated from the sidebar's spot, rate and dividend yield.

### Time
* Under the index, the valuation time is a New York date and time on 5-minute steps, by default the last completed
  cash close (16:00, or 13:00 on an early close), so that every visitor sees the same numbers all day; a shared
  link holds it. Time to expiry runs from it to each expiry's settlement, in minutes / 525,600.
* Under the textbook market, time to expiry $T$ = calendar days / 365. The sidebar takes whole days from 1 to 3,650.
* Value-at-Risk horizons are in trading days (252 per year), because volatility accrues with trading time.

### Maturities in uploaded files
A maturity header may state days, years, months, weeks, hours or minutes ('Maturity (months)'); a bare 'm' could
be months or minutes and is refused, and times to expiry within half a day of each other are taken as one expiry
(keeping each quote's total variance). A maturity column whose header gives no unit is converted from days when
its values go above 30, refused when they are whole numbers from 5 to 30 (they could be years or days), and
otherwise read as years, with a warning to name the unit when they are small whole numbers or go above 3 (listed
options rarely run longer), as they could then be days.
