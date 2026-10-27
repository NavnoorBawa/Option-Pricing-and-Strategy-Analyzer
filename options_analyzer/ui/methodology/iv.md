## Implied vol

Quotes outside the no-arbitrage bounds ${\max(S e^{-qT} - K e^{-rT}, 0) \le C < S e^{-qT}}$ (and the put analogue)
are rejected, because no volatility can produce them. Otherwise the quote is converted to its out-of-the-money
equivalent through put–call parity and solved with Brent's method, which always converges on a bracketed root.

American quotes must be at least the exercise value and the European lower bound, and below the most exercise can
be worth: $K\max(1, e^{-rT})$ for a put (more than the strike when the rate is negative) and $S\max(1, e^{-qT})$
for a call. They are inverted on the sidebar's binomial tree when that model is selected, with the sidebar's
American engine and its scheme when that is (so the headline American price gives back the sidebar volatility), and
otherwise on a 501-step Leisen–Reimer tree. The engine values zero volatility exactly, so its search starts there.
A quote equal to the option's zero-volatility value (for example the value of exercising now) does not identify a
volatility and is reported as such.
