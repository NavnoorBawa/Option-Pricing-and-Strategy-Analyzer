## Strategy

### Legs and valuation
Each position is a list of legs, and every leg is valued with Black–Scholes–Merton at its own implied volatility and
expiry, with European exercise. Under the synthetic index a leg holds a listed expiry (a date and its settlement:
SPX monthlies AM at the opening prints, SPXW weeklies PM at the close) and a listed strike; its vol is the published
surface's at that strike and expiry (*surface*; *extrap.* beyond the expiry's quotes, on its Lee-bounded wing;
*unlisted* for a date the chain no longer lists, read from the surface in total variance) unless you type an
override, and its bid and ask come from the chain. The index itself cannot be traded, so it takes no share leg. In
the textbook market a leg keeps days to expiry and the flat sidebar vol, as in version 2, unless you type its own.
Every leg is valued sticky strike: when spot moves in a scenario, each leg keeps the vol of its strike. Legs on a
later expiry than the front one take the front expiry's rate and dividend yield, not their own expiry's (on the
default call calendar, about 2 bp off the later expiry's own discount rate, \$2.69 a contract).

A share (textbook market only) is worth its price plus the dividends it has paid, reinvested at the risk-free rate,
${S_t + S_0\,(e^{rt} - e^{(r-q)t})}$, so its delta is exactly one and a covered call's profit stays capped.

### Entry and costs
A leg's entry price is the fill you type; left blank, it is the chain's mid under the index (the model value where
the leg has no two-sided quote) and the model value in the textbook market. The Net premium tile gives the
position's premium at that entry, and the T+0 curve starts at the entry's distance from the model value: zero for
legs entered at model value. Under the index the Package tile shows the structure's natural market per share: for a
credit, the shorts' bids less the longs' asks to the shorts' asks less the longs' bids (the reverse for a debit), the
package mid against its model value, and the cost of crossing half the spread, which closing at the market rather
than the mid costs again. A commission per contract is charged when the position is opened; no other transaction
cost is modelled. Premium financing is ignored unless you switch it on.

### Templates
Under the index a template opens on the AM monthly nearest 45 days. With strikes by delta, the short strikes are the
listed strikes whose own delta (spot convention, at their own surface vol) is nearest the target (10 to 30; 16 by
default), and each wing sits on the first listed strike at least the wing width beyond its short (50 to 200 index
points; 100 by default), equal on both sides; a structure with legs at the money centres on the listed strike
nearest the forward and sets its other strikes by the wing width.

With strikes by width (the only choice in the textbook market), the strikes are spaced $w$ apart (${K \pm w}$,
${K \pm 2w}$) around the listed strike nearest the forward under the index and around the sidebar strike in the
textbook market, $w$ a percentage of the strike rounded to the cent (to two significant figures below ten cents),
or to listed strikes under the index. By default $w$ is about half the expected move,
${w \approx \tfrac12 K\sigma\sqrt{T}}$, rounded to a strike step of 1, 2, 2.5 or 5 times a power of ten and capped at
25% of the strike.

### Tiles and probabilities
Maximum profit, maximum loss and the breakevens are read from the P&L at the front expiry. Breakevens are exact
roots of that P&L, and a profit or loss that keeps growing with spot is reported as unlimited, except a long
calendar's or diagonal's loss, which grows without bound only because a dividend yield makes its long back call gain
less than \$1 for each \$1 rise: its tile gives the worst P&L over the central 99.9% range of the price at the front
expiry, and says so.

The probability of profit is the probability that the P&L at the front expiry is positive, integrated over the spot
intervals where it is. Under the index it is risk-neutral from the published smile at the front expiry: the
risk-neutral distribution of the price is read from the smile's call prices (Breeden–Litzenberger),
${P(S_T \le K) = 1 + \partial C/\partial K \,/\, D}$, so it carries the skew. In the textbook market it is
risk-neutral at the flat sidebar vol (a log-normal price, N(d₂)). Switched off in the position settings, it uses
your own expected return at the flat at-the-money-forward vol (log-normal). The tile's help gives the expected P&L
and its standard deviation under the same distribution, and the P&L chart shades the 68% and 95% ranges of the price
at the front expiry from it.

### Time decay
The blotter shows, for every leg and in total, theta per day (theta / 365, the instantaneous rate) beside the
*1-day decay*: the change in the position's value over the next calendar day by repricing every leg with spot and
implied volatility unchanged, a leg that expires within the day at its intrinsic value, when the column reads *to
expiry* (see Conventions). A 7-day short iron condor on an index at 5,800 (long the 5,700 put and the 5,900 call,
short the 5,750 put and the 5,850 call, one contract of 100 shares, volatility 16%, rate 4.3%, dividend yield 1.3%)
has a theta of +\$137.50 per day against a decay of +\$151.97, 9.5% apart; a 1-day at-the-money put decays by its
whole premium.

Early exercise is ignored whatever the sidebar's pricing model: the textbook market's American engine prices the
Pricing and IV tabs, and under it the Strategy tab says what early exercise adds to the sidebar contract.
