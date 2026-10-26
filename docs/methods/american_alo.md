# American options: Andersen-Lake-Offengenden spectral collocation

`options_analyzer/american.py` values American calls and puts in the Black-Scholes-Merton model by solving for the
early-exercise boundary on a Chebyshev grid (Andersen, Lake and Offengenden 2016). It follows the design of
QuantLib's BSD-licensed `QdFpAmericanEngine` (Spanderen 2022), written from the papers; QuantLib supplies the
reference prices it is validated against.

## Model

The spot follows $dS = (r - q) S\,dt + \sigma S\,dW$ under the risk-neutral measure, with constant rate $r$,
dividend yield $q$ and volatility $\sigma$ (continuously compounded, per year). Write $\tau$ for the time to expiry,
$\Phi$ and $\phi$ for the standard normal distribution and density, and

$$d_\pm(\tau, z) = \frac{\ln z + (r - q)\tau}{\sigma\sqrt\tau} \pm \frac{\sigma\sqrt\tau}{2}.$$

A put is exercised as soon as the spot falls to the boundary $B(\tau)$. Its value is the European value $v$ plus
the early-exercise premium (Kim 1990):

$$V(\tau, S) = v(\tau, S) + \int_0^\tau \Big[ rK e^{-r(\tau - u)} \Phi\big(-d_-(\tau - u, S / B(u))\big)
  - qS e^{-q(\tau - u)} \Phi\big(-d_+(\tau - u, S / B(u))\big) \Big]\,du .$$

At expiry the boundary is $X = K \min(1, r / q)$ (and $X = K$ when $q \le 0$). Calls are valued as puts through
McDonald and Schröder's (1998) symmetry $C(S, K; r, q) = P(K, S; q, r)$, so everything below is about the put.

## The boundary equation

Value matching at the boundary gives the fixed-point equation of Andersen et al. (2016)

$$B(\tau) = K e^{-(r - q)\tau} \frac{N(\tau, B)}{D(\tau, B)},$$

whose numerator and denominator are integrals over the boundary at earlier times to expiry. Two forms are used,
as in QuantLib:

- the **cdf form**, when $|r - q| \ge 10^{-3}$:

  $$N = \Phi\big(d_-(\tau, \tfrac{B(\tau)}{K})\big) + r \int_0^\tau e^{ru} \Phi\big(d_-(\tau - u, \tfrac{B(\tau)}{B(u)})\big)\,du, \qquad
    D = \Phi\big(d_+(\tau, \tfrac{B(\tau)}{K})\big) + q \int_0^\tau e^{qu} \Phi\big(d_+(\tau - u, \tfrac{B(\tau)}{B(u)})\big)\,du;$$

- the **density form**, when $|r - q| < 10^{-3}$ (it converges faster there: in the example below, at $r = q$, the
  value settles to $10^{-12}$ in 8 iterations, with the cdf form in 14):

  $$N = \frac{\phi(d_-(\tau, \cdot))}{\sigma\sqrt\tau} + r \int_0^\tau \frac{e^{ru}\,\phi(d_-(\tau - u, \cdot))}{\sigma\sqrt{\tau - u}}\,du, \qquad
    D = \frac{\phi(d_+(\tau, \cdot))}{\sigma\sqrt\tau} + \Phi(d_+(\tau, \cdot))
      + q \int_0^\tau e^{qu} \Big[\frac{\phi(d_+(\tau - u, \cdot))}{\sigma\sqrt{\tau - u}} + \Phi(d_+(\tau - u, \cdot))\Big]\,du .$$

## Numerical method

1. **Representation.** The boundary is smooth in $\sqrt\tau$ after a change of variable: the engine interpolates
   $H = \ln(B / X)^2$ by a degree-$n$ polynomial in $z = 2\sqrt{\tau / T} - 1$ through the Chebyshev-Lobatto nodes
   $z_i = -\cos(i\pi / n)$, $i = 0, \dots, n$. $H = 0$ at $z = -1$ (expiry), and $B = X e^{-\sqrt H}$ is positive
   and at most $X$ by construction.
2. **Initial guess.** Li's (2010) QD+ approximation of the boundary at every node, solved by Halley's method.
3. **Iteration.** One Jacobi-Newton step, $B \leftarrow B - (f(B) - B) / (f'(B) - 1)$ with $f$ the right-hand side
   of the fixed-point equation differentiated through its terms outside the integrals, then $m - 1$ plain steps
   $B \leftarrow f(B)$ at all nodes at once.
4. **Quadrature.** Every integral is a Gauss-Legendre rule in $\theta \in [0, \pi]$ after the cosine map
   $u = \tau (1 - \cos\theta) / 2$: $\sqrt u = \sqrt\tau \sin(\theta/2)$ and $\sqrt{\tau - u} = \sqrt\tau \cos(\theta/2)$,
   so the square-root behaviour at both ends (and the $1/\sqrt{\tau - u}$ of the density form, since
   $du / \sqrt{\tau - u} = \sqrt\tau \sin(\theta/2)\,d\theta$) becomes smooth and the rule converges spectrally. The
   boundary integrals use $l$ points per node, the premium $p$ points, with $B(u)$ taken from the interpolant.
5. **Value.** The premium integral above, with the European value from the closed form; the result is floored at
   $\max(v, K - S)$, and spots at or below $B(T)$ (today's boundary) are worth $\max(v, K - S)$, which is $K - S$
   wherever exercise is optimal.

A **scheme** fixes $(l, m, n, p)$:

| Scheme | $(l, m, n, p)$ | Use |
|---|---|---|
| `fast` | (8, 3, 7, 24) | quick looks |
| `accurate` | (24, 8, 14, 48) | the default |
| `high` | (64, 16, 32, 128) | reference values |

The boundary is proportional to the strike, so the normalised boundary $B/K$ is cached per
$(T, r, q, \sigma, \text{scheme})$ (4,096 entries): a strike strip, or the spots of a profile, solves one boundary.
A volatility smile changes $\sigma$ with the strike and needs one solve per strike.

## Regimes

Healy (2021) and Andersen and Lake (2021) classify the exercise region of a put:

- $r \le 0$ and $q \ge r$: early exercise is never optimal; the value is the European one (`method = "european"`).
- $q < r < 0$: the exercise region lies between **two** boundaries. This engine, like QuantLib's (which refuses the
  case), solves one. The value falls back to `binomial.extrapolated_price`, a Richardson extrapolation of
  Leisen-Reimer (1996) trees with 2,001 and 4,002 steps (about 55 ms per option), and `price_details` reports
  `fallback = True`, `fallback_reason = "double boundary"`.
- otherwise one boundary, solved as above. $\sigma = 0$ is valued exactly, as the best deterministic exercise date
  of $K e^{-rt} - S e^{-qt}$.

Calls follow by symmetry (a call is European when $q \le 0$ and $r \ge q$, double-boundary when $r < q < 0$).

**Convergence check.** If the last fixed-point step still moves $\ln B$ by more than $10^{-2}$ at some node, the
boundary is rejected and the option goes to the same tree (`fallback_reason = "not converged"`;
`exercise_boundary` raises). On the 4,495-put grid the largest last step is $5.5 \times 10^{-4}$ (`fast`),
$1.2 \times 10^{-5}$ (`accurate`) and $2.8 \times 10^{-7}$ (`high`). The only rejections found in the tests are
`fast` on 5- and 20-year puts with $r = 0$, $q = -5\%$, $\sigma = 150\%$.

## Error

Measured with QuantLib 1.44 (`tests/test_american.py`, `tests/oracle/test_ql_alo.py`,
`benchmarks/bench_american.py`). The reference grid is the 4,495 puts of Healy (2021, Table 5): $K = 100$,
$S \in \{25, \dots, 200\}$, $T \in \{1/12, \dots, 1\}$, $r \in \{2, \dots, 10\}\%$, $q \in \{0, \dots, 12\}\%$,
$\sigma \in \{10, \dots, 60\}\%$, the options worth at least 0.5. It is priced by `scripts/gen_alo_reference.py`
with QdFp's high-precision scheme (`qdfp_high`) and with tanh-sinh quadrature to $10^{-13}$, 24 iterations and
65 nodes (`qdfp_strict`), and committed as `tests/fixtures/alo_grid_qdfp.npz`.

| Scheme | RMSE vs `qdfp_high` | max vs `qdfp_high` | RMSE vs `qdfp_strict` | max vs `qdfp_strict` | Target (RMSE vs `qdfp_high`) |
|---|---|---|---|---|---|
| `fast` | 3.7e-5 | 7.1e-4 | 3.7e-5 | 7.1e-4 | ≤ 1e-4 |
| `accurate` | 1.6e-7 | 3.7e-6 | 1.7e-7 | 4.8e-6 | ≤ 5e-6, max ≤ 5e-5 |
| `high` | 5.2e-8 | 1.0e-6 | 1.8e-9 | 5.4e-8 | ≤ 3e-7 |

`high`'s error against `qdfp_high` is that column's own error: the two QuantLib columns differ by up to
$1.04 \times 10^{-6}$ (RMSE $5.2 \times 10^{-8}$), always near the exercise boundary. For comparison, QuantLib's own
fast, accurate and high schemes have RMSE $6.9 \times 10^{-5}$, $3.2 \times 10^{-6}$ and $7.4 \times 10^{-8}$
against `qdfp_strict` on 300 of these options.

Outside the grid (80 random calls and puts, expiries from a day to 30 years, volatilities 2 % to 200 %, rates from
-5 % to 25 %, negative and zero dividend yields), the largest $|V - V_\text{QdFp strict}| / \max(1, V)$ is
$2.0 \times 10^{-4}$ (`fast`), $2.7 \times 10^{-6}$ (`accurate`) and $7.1 \times 10^{-9}$ (`high`).

The example of Andersen et al. (2016), $S = K = 100$, $q = 5\%$, $\sigma = 25\%$, one year: the early-exercise
premium is 0.1069527027479 at $r = 5\%$ (density form) and 0.3671111406253 at $r = 7.5\%$ (cdf form), from QdFp at
its tightest settings; `high` agrees to $1.0 \times 10^{-12}$ and $4.6 \times 10^{-11}$.

The double-boundary put ($S = K = 100$, $r = -1\%$, $q = -2\%$, $\sigma = 20\%$, one year) falls back to 7.6252878;
QuantLib's finite-difference engine on a 4,000 × 4,000 grid gives 7.6252621.

Properties checked (Hypothesis and parameter sweeps): put-call symmetry to $10^{-14}$ relative; a call with
$q = 0 \le r$ equals Black-Scholes to $10^{-12}$; puts with $r \le 0 \le q - r$ equal Black-Scholes; the value lies
between $\max(v, \text{intrinsic})$ and `binomial.american_upper_bound`; it is monotone in $\sigma$, $T$ and $K$
(no decrease beyond $1.4 \times 10^{-14}$); the boundary starts at $X$.

## Speed

Median wall-clock times on the development machine (one CPU core, NumPy), with the boundary cache cleared:

| Scheme | One option (its own boundary) | Whole grid, one call (600 boundaries) | 41-strike strip (one boundary) |
|---|---|---|---|
| `fast` | 0.38 ms | 0.24 s | 0.46 ms |
| `accurate` | 0.64 ms | 0.38 s | 0.82 ms |
| `high` | 1.52 ms | 1.21 s | 2.06 ms |

The targets were 1 ms (`fast`) and 5 ms (`accurate`) per option and 15 ms for the strip.

## Limits

- Constant $r$, $q$ and $\sigma$: no term structures, no discrete dividends, no smile within one boundary.
- The double-boundary regime is valued on the tree, not by collocation: about $10^{-5}$ accurate and 55 ms per
  option. Andersen and Lake (2021) and Healy (2021) extend the collocation to two boundaries.
- `fast` interpolates the boundary with 8 nodes. Just before expiry at very high volatility the interpolant ripples:
  the put boundary can rise with time to expiry by up to 2.1 % of its value at $\sigma = 150\%$ (8e-6 up to
  $\sigma = 20\%$; `accurate` 3.0e-5, `high` 4.1e-6). Use `accurate` or `high` when the boundary itself is
  shown.
- The convergence threshold ($10^{-2}$ in $\ln B$) guards against divergence, not against slow convergence: a
  boundary that passes it can still be less accurate than the scheme's grid error far outside the validated domain
  (expiries of decades, volatilities above 150 %).

## References

- L. Andersen and M. Lake (2021), "Fast American option pricing: the double-boundary case", *Wilmott* 2021(116),
  30-41, doi:[10.1002/wilm.10969](https://doi.org/10.1002/wilm.10969).
- L. Andersen, M. Lake and D. Offengenden (2016), "High-performance American option pricing", *Journal of
  Computational Finance* 20(1), 39-87, doi:[10.21314/JCF.2016.312](https://doi.org/10.21314/JCF.2016.312).
- J. Healy (2021), "Pricing American options under negative rates", *Journal of Computational Finance* 25(1),
  doi:[10.21314/JCF.2021.004](https://doi.org/10.21314/JCF.2021.004), arXiv:[2109.15157](https://arxiv.org/abs/2109.15157).
- I. J. Kim (1990), "The analytic valuation of American options", *Review of Financial Studies* 3(4), 547-572,
  doi:[10.1093/rfs/3.4.547](https://doi.org/10.1093/rfs/3.4.547).
- D. Leisen and M. Reimer (1996), "Binomial models for option valuation - examining and improving convergence",
  *Applied Mathematical Finance* 3(4), 319-346, doi:[10.1080/13504869600000015](https://doi.org/10.1080/13504869600000015).
- M. Li (2010), "Analytical approximations for the critical stock prices of American options: a performance
  comparison", *Review of Derivatives Research* 13(1), 75-99,
  doi:[10.1007/s11147-009-9044-3](https://doi.org/10.1007/s11147-009-9044-3).
- R. McDonald and M. Schroder (1998), "A parity result for American options", *Journal of Computational Finance*
  1(3), 5-13, doi:[10.21314/JCF.1998.010](https://doi.org/10.21314/JCF.1998.010).
- K. Spanderen (2022), QuantLib `QdFpAmericanEngine` (BSD-3-Clause),
  <https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/qdfpamericanengine.cpp>.
