## Pricing engines

### Black–Scholes–Merton

$$
\begin{aligned}
C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \\
P &= K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),
\end{aligned}
$$

$$
d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)\,T}{\sigma\sqrt{T}}.
$$

With zero volatility the value is the discounted forward intrinsic value
${\max(\pm(S e^{-qT} - K e^{-rT}), 0)}$. All fourteen Greeks (to third order) are analytical and are
verified against high-precision finite differences in the test-suite.

### Binomial trees
* **Cox–Ross–Rubinstein:** ${u = e^{\sigma\sqrt{\Delta t}}}$, ${d = 1/u}$,
  ${p = (e^{(r-q)\Delta t} - d)/(u - d)}$. The tree refuses inputs with ${p \notin (0,1)}$ (arbitrage).
* **Leisen–Reimer** (the default): Peizer–Pratt inversion centres the tree on the strike; convergence is
  smooth, roughly second order for European exercise and first order with early exercise (an odd number of
  steps is used).
* Calls are priced as the symmetric put, ${C(S, K, T; r, q) = P(K, S, T; q, r)}$ (McDonald–Schroder). On the
  lattice this is an exact change of numeraire, and a put's payoff is bounded by its strike, so the value
  stays accurate even when extreme volatility × expiry forces the tree to keep its node prices between
  $e^{-700}$ and $e^{700}$. Such a capped lattice computes each level's prices from their logarithms, so a
  capped or zero-rounded node never feeds the exercise test at an earlier step.
* American exercise compares continuation and exercise value at every node; delta and gamma are read off
  the tree, theta from the step-2 middle node (CRR) or a central difference in time (Leisen–Reimer), and
  vega, rho and dividend rho from repricing on a tree with the same steps (±0.5 vol point, ±25 bp). The
  early-exercise boundary is the critical price where the exercise gain changes sign, interpolated between
  adjacent nodes, averaged over consecutive steps (whose nodes interleave) and fitted to be monotone in
  time, as the exact boundary is. The early-exercise premium is measured on the same tree with European
  exercise, so the discretisation error cancels; the American reference in the convergence chart
  extrapolates Leisen–Reimer trees with 2,001 and 4,003 steps and is kept within the American no-arbitrage
  bounds: at least the European and the exercise value, at most the perpetual American value.

### American engine (Andersen–Lake–Offengenden)
The sidebar's *American (ALO)* model values American calls and puts by spectral collocation of the
early-exercise boundary (Andersen, Lake and Offengenden 2016), after the design of QuantLib's QdFp engine. A
put is worth its European value plus the early-exercise premium, an integral over the boundary $B(\tau)$. The
boundary solves a fixed-point equation and is interpolated as ${H = \ln(B/X)^2}$ on Chebyshev nodes in
$\sqrt{\tau}$; every integral is a Gauss–Legendre rule after a cosine change of variable, and a call is the
symmetric put (McDonald–Schroder). The *fast*, *accurate* (the default) and *high* schemes use 8, 15 and 33
nodes, and a stiff option (a long expiry, or high rates or dividends at low volatility) is handed to the next
scheme. On the 4,495 puts of Healy (2021, Table 5), strike \$100, *accurate* is within \$0.000005 of QuantLib's
high-precision values. Early exercise never pays for a call when ${q \le 0}$ and ${r \ge q}$, nor for a put when
${r \le 0}$ and ${q \ge r}$: both then equal Black–Scholes–Merton, and the app says why. Where the exercise region
lies between two boundaries (${q < r < 0}$ for a put; Healy 2021) or the boundary iteration does not converge,
the option falls back to the extrapolated Leisen–Reimer trees above, and the app says so. Greeks are central
finite differences of the engine with its scheme and boundary equation held fixed (spot and expiry ±0.01%,
volatility ±0.01 vol point, rate and dividend yield ±1 bp); where a bump one way would change how the option
is valued, the difference is taken on the other side. An option valued as European has the closed-form Greeks,
and one exercised at once those of its exercise value. At a rate and dividend yield of zero both options are
European, but a 1 bp move one way makes early exercise pay, so their rho and dividend rho are one-sided: the
slopes on the side where it does not. American implied volatility inverts the engine with Brent's method. The
method, its error and its limits are written up in `docs/methods/american_alo.md`.

References for the engine: Andersen, L., Lake, M. and Offengenden, D. (2016). High-performance American option
pricing. *Journal of Computational Finance* 20(1), 39–87.
[doi:10.21314/JCF.2016.312](https://doi.org/10.21314/JCF.2016.312) · Healy, J. (2021). Pricing American
options under negative rates. *Journal of Computational Finance* 25(1).
[doi:10.21314/JCF.2021.004](https://doi.org/10.21314/JCF.2021.004),
[arXiv:2109.15157](https://arxiv.org/abs/2109.15157) · Spanderen, K. (2022). QuantLib `QdFpAmericanEngine`
(BSD-3-Clause).
[github.com/lballabio/QuantLib](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/qdfpamericanengine.cpp)

### Monte Carlo
European payoffs depend only on $S_T$, which is sampled exactly:
${S_T = S\exp\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}Z\big)}$. Antithetic pairs ${(Z, -Z)}$ and a
control variate (the discounted terminal price, with known mean $S e^{-qT}$) reduce the variance; the
standard error uses the unbiased sample deviation of the independent samples. It is left blank when the
sample cannot estimate it: every simulated price finished out of the money, or every one in the money with
the control variate on (the residuals are then constant). Calls and puts share
draws and a fixed seed makes results reproducible. The sample-path chart draws the first 60 of these
scenarios, each filled in from today to its sampled $S_T$ with a Brownian bridge (exact for GBM given the
end point).

### One-day decay
The Greeks table, the option chain and the sensitivity explorer (dashed, in its theta view) show the *1-day decay*
beside theta per day (see Conventions): the value with one calendar day less to expiry minus the value now, all else
equal, or the intrinsic value minus the value now when the option expires within the day. It uses the model of the
table it is in: the closed form for Black–Scholes–Merton, and the engine for the American engine. Under a binomial
tree the Greeks table stays Black–Scholes–Merton's for European exercise, decay included, so with American exercise
it is not the decay of the tree's American value in the price cards; the tree's Greeks give that one, and put it
first by the same 10 % rule against the tree's own theta. Under Monte Carlo the decay is the closed-form one: the
simulation estimates that same value, and the difference of two simulations would only add sampling error.

On a tree the decay is the difference of two trees, so how the shorter tree is built decides its accuracy. Measured
against the closed-form decay of European options (spot 100, strikes 80 to 120 in steps of 0.5, rate 5 %, no
dividends, every expiry from 2 to 60 days and 13 more out to 3,650 days, 501 steps, decays of at least one cent) at
10 %, 20 % and 50 % volatility: a Leisen–Reimer tree one day shorter keeps its steps, and as its error is smooth in
$T$ the two errors cancel, to within ${1.6 \times 10^{-6}}$, ${4.0 \times 10^{-6}}$ and ${1.2 \times 10^{-5}}$ of
the decay. The Cox–Ross–Rubinstein error swings with the strike's place among the final nodes, so the same steps
were off by up to 4.4 %, 8.5 % and 6.0 %. Its shorter tree keeps the time step instead, which keeps the final nodes
in place: the value is interpolated, linearly in the step count, between the two trees with the step count's parity
whose time steps bracket the current one. That is within 0.55 %, 1.5 % and 0.66 % of the decay (99 % of cases within
0.08 %, 0.12 % and 0.19 %); the nearest of those two trees alone was off by up to 1.1 %, 2.3 % and 2.2 %. Both
trees' errors grow as the steps fall: at 20 % volatility and 101 steps, up to 11 % for Cox–Ross–Rubinstein (99 % of
cases within 1.2 %) and ${9.7 \times 10^{-5}}$ for Leisen–Reimer; at 2,000 steps, 0.06 % and ${1.7 \times 10^{-7}}$.
