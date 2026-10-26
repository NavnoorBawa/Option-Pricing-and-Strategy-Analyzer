## Strategy

Each position is a list of legs. Options are valued with Black–Scholes–Merton at each leg's implied volatility. A
share is worth its price plus the dividends it has paid, reinvested at the risk-free rate,
$S_t + S_0\,(e^{rt} - e^{(r-q)t})$, so its delta is exactly one and a covered call's profit stays capped.

The entry cost is the model value unless you enter your own fill price, so the "today" curve starts at zero for
legs entered at model value.

Breakevens are exact roots of the P&L function, maximum profit or loss is reported as unlimited when the P&L keeps
rising or falling with the spot, and the probability of profit integrates the log-normal density of the price at
the front expiry over the profitable intervals (drift = the risk-free rate by default). Premium financing is
ignored unless you switch it on; early exercise is ignored, whatever the sidebar pricing model.

Multi-strike templates space their strikes $w$ apart, a percentage of the strike rounded to the cent (to two
significant figures below ten cents). By default $w$ is about half the expected move,
$w \approx \tfrac12 K\sigma\sqrt{T}$, rounded to a strike step of 1, 2, 2.5 or 5 times a power of ten and capped at
25 % of the strike.
