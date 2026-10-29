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

The 95% interval, price ± 1.96 standard errors, is shown under each price card and in the diagnostics table.
It assumes the estimate is close to normal, which fails when few simulated prices finish on the side of the
strike that its error rests on: the in-the-money side when the payoff is averaged as it is, and whichever
side is scarcer with the control variate, which takes out the payoff's linear part (a deep in-the-money call
is then as uncertain as the out-of-the-money put on the same draws), and in a heavy-tailed sample (below).
Over 20,000 seeds per level (a 30-day call on ${S = 100}$ at ${\sigma = 20\%}$ and ${r = 5\%}$, 20,000
antithetic paths with the control variate, a strike for each level) the interval covered the
Black–Scholes–Merton value in 87.8% of runs with about 10 such paths, 92.2% with 30, 94.0% with 100, 94.6%
with 300 and 94.7% with 1,000, against 94.8% at the money; a put, a call without either variance reduction
and 10,000 or 100,000 paths behave alike. Below 300 such paths, or when the standard error cannot be
estimated, the result is flagged unreliable and the app says how many paths finished in (or out of) the
money instead of showing the interval. The flag also sorts the runs of one level: at about 300 expected
paths, those with at least 300 covered 95.3% and the flagged ones 93.8%. The test-suite checks this as a
two-sided binomial test at about 10, 30, 100, 300 and 1,000 paths on the scarce side, 4,000 seeds each: the
intervals flagged reliable cover the value at a rate inside the 99% binomial band around 95% and of at least
93%, and the flag is raised wherever the coverage falls short. What the flag does not catch is the shortfall
of heavy-tailed samples (very large ${\sigma\sqrt{T}}$, priced from the put and parity as the diagnostics
describe) with many paths on both sides: at ${\sigma\sqrt{T}}$ of 2 to 3 without the control variate their
intervals covered the value in 91 to 94% of 400 runs, with thousands of paths on each side of the strike.
