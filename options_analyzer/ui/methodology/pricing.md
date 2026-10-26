## Pricing engines

### Black–Scholes–Merton

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$d_{1,2} = \dfrac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)\,T}{\sigma\sqrt{T}}$$

With zero volatility the value is the discounted forward intrinsic value
$\max(\pm(S e^{-qT} - K e^{-rT}), 0)$. All fourteen Greeks (to third order) are analytical and are
verified against high-precision finite differences in the test-suite.

### Binomial trees
* **Cox–Ross–Rubinstein:** $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$,
  $p = (e^{(r-q)\Delta t} - d)/(u - d)$. The tree refuses inputs with $p \notin (0,1)$ (arbitrage).
* **Leisen–Reimer** (the default): Peizer–Pratt inversion centres the tree on the strike; convergence is
  smooth, roughly second order for European exercise and first order with early exercise (an odd number of
  steps is used).
* Calls are priced as the symmetric put, $C(S, K, T; r, q) = P(K, S, T; q, r)$ (McDonald–Schroder). On the
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

### Monte Carlo
European payoffs depend only on $S_T$, which is sampled exactly:
$S_T = S\exp\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}Z\big)$. Antithetic pairs $(Z, -Z)$ and a
control variate (the discounted terminal price, with known mean $S e^{-qT}$) reduce the variance; the
standard error uses the unbiased sample deviation of the independent samples. It is left blank when the
sample cannot estimate it: every simulated price finished out of the money, or every one in the money with
the control variate on (the residuals are then constant). Calls and puts share
draws and a fixed seed makes results reproducible. The sample-path chart draws the first 60 of these
scenarios, each filled in from today to its sampled $S_T$ with a Brownian bridge (exact for GBM given the
end point).
