## Strategy

Each position is a list of legs. Options are valued with Black–Scholes–Merton at each leg's implied volatility. A
share is worth its price plus the dividends it has paid, reinvested at the risk-free rate,
${S_t + S_0\,(e^{rt} - e^{(r-q)t})}$, so its delta is exactly one and a covered call's profit stays capped.

The entry cost is the model value unless you enter your own fill price, so the "today" curve starts at zero for
legs entered at model value.

The time-decay card shows theta per day (theta / 365) or the *1-day decay*, the change in the position's value over
the next calendar day by repricing every leg with spot and implied volatility unchanged (a leg that expires within
the day at its intrinsic value): the decay when they differ by more than 10 % of the larger, and as *decay to expiry*
when a leg expires within the day (see Conventions). Its help gives both. A 7-day short iron condor on an index at
5,800 (long the 5,700 put and the 5,900 call, short the 5,750 put and the 5,850 call, one contract of 100 shares,
volatility 16 %, rate 4.3 %, dividend yield 1.3 %) keeps theta, +\$137.50 per day against a decay of +\$151.97,
9.5 % apart; a 1-day at-the-money put shows the decay, its whole premium. The legs table shows both for every leg.

Breakevens are exact roots of the P&L function, maximum profit or loss is reported as unlimited when the P&L keeps
rising or falling with the spot, and the probability of profit integrates the log-normal density of the price at
the front expiry over the profitable intervals (drift = the risk-free rate by default). Premium financing is
ignored unless you switch it on; early exercise is ignored, whatever the sidebar pricing model.

Multi-strike templates space their strikes $w$ apart, a percentage of the strike rounded to the cent (to two
significant figures below ten cents). By default $w$ is about half the expected move,
${w \approx \tfrac12 K\sigma\sqrt{T}}$, rounded to a strike step of 1, 2, 2.5 or 5 times a power of ten and capped at
25 % of the strike.
