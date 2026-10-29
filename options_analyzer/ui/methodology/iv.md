## Implied vol

Quotes outside the no-arbitrage bounds ${\max(S e^{-qT} - K e^{-rT}, 0) \le C < S e^{-qT}}$ (and the put analogue)
are rejected, because no volatility can produce them. Otherwise the quote is converted to its out-of-the-money
equivalent through put–call parity, which keeps its time value, and inverted with Jäckel's *Let's Be Rational*: two
Householder steps reach machine precision, and Brent's method on a bracket takes over wherever they do not report
convergence.

**Quotes with no time value.** Converting an in-the-money quote subtracts the discounted intrinsic value
(${S e^{-qT} - K e^{-rT}}$ for a call: the lower bound, and the option's value at zero volatility), so the time
value that is left carries the rounding of the whole price, and close to that value the last digits of the price
decide the volatility: a 2-day call struck at \$60 with the spot at \$100 and a rate of 4 % is worth
\$40.0131 at 0 % volatility (not \$40: the strike is discounted), ${10^{-13}}$ dollars more at 94.5 % and half a
cent more at 230 %. An in-the-money quote within ${10^{-12}\max(S, K)}$ of the lower bound (the margin the bound
check already allows below it) is therefore reported as having no time value instead of being given a volatility,
and so is an out-of-the-money quote of zero. An out-of-the-money quote is all time value and keeps its full relative
precision, so it is inverted however small it is. On 20,000 random quotes priced in double precision, every quote
with more time value than that margin comes back within $10^{-5}$ (relative) of the volatility that priced it.

American quotes must be at least the exercise value and the European lower bound, and below the most exercise can
be worth: $K\max(1, e^{-rT})$ for a put (more than the strike when the rate is negative) and $S\max(1, e^{-qT})$
for a call. They are inverted on the sidebar's binomial tree when that model is selected, with the sidebar's
American engine and its scheme when that is (so the headline American price gives back the sidebar volatility), and
otherwise on a 501-step Leisen–Reimer tree. The engine values zero volatility exactly, so its search starts there.
A quote no more than ${10^{-12}\max(S, K)}$ above the option's zero-volatility value has no time value either:
every volatility up to some level gives a value that close to it (deep in the money exactly that value: every
volatility at which exercising at once stays optimal, up to 38.5 % for a one-year put struck at \$100 with the spot
at \$60 and a rate of 5 %). The margin applies out of the money too, because near zero volatility the American
engines are accurate only in absolute terms, far more coarsely than a quote's rounding; just above it a volatility
is only as good as the engine there.
