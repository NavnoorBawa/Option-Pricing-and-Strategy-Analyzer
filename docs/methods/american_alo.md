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
  value settles to $10^{-12}$ in 8 iterations, with the cdf form in 14; option `fixed_point` picks one form for
  every input):

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
5. **Scaling.** No exponential factor above 1 is formed, so that no rate, dividend yield or expiry overflows. With
   $a = \max(q, 0)$ the fixed point is solved as
   $B = K e^{(q - a)\tau}\,\big(e^{-r\tau} N\big) / \big(e^{-a\tau} D\big)$, whose integrands carry
   $e^{-r(\tau - u)}$ and $e^{qu - a\tau}$ ($r \ge 0$ wherever a boundary is solved); the European value and,
   when $q < 0$, the dividend leg of the premium are taken as $K e^{-rT + \ln\Phi(-d_-)}$ and
   $S e^{-qT + \ln\Phi(-d_+)}$, which are bounded by $K e^{-rT}$.
6. **Value.** The premium integral above, with the European value from the closed form. Spots at or below $B(T)$
   (today's boundary) are worth $\max(v, K - S)$, which is $K - S$ wherever exercise is optimal. Every other value
   is kept within the no-arbitrage bounds: at least $v$, $K - S$ and the best deterministic exercise date
   $\max_{0 \le t \le T} (K e^{-rt} - S e^{-qt})^+$ (exercising at a fixed date $t$ is worth
   $e^{-rt} E[(K - S_t)^+] \ge (K e^{-rt} - S e^{-qt})^+$ by Jensen's inequality), and at most $K$ and, for
   $r > 0$, the perpetual put (Merton 1973), which no finite expiry exceeds. The true value lies between the
   bounds, so moving a value onto a bound it crosses never takes it farther from the truth.

A **scheme** fixes $(l, m, n, p)$:

| Scheme | $(l, m, n, p)$ | Use |
|---|---|---|
| `fast` | (8, 3, 7, 24) | quick looks |
| `accurate` | (24, 8, 14, 48) | the default |
| `high` | (64, 16, 32, 128) | reference values |

**Hand-over.** A scheme's error grows with the stiffness
$\Lambda = \max\big(T c^2 / \sigma^2,\ T \max(|r|, |q|)\big)$, $c = \max(|r|, |q|, |r - q|)$: drift against
diffusion, and discounting, over the life of the option. The boundary moves on the time scales $\sigma^2 / c^2$ and
$1 / \max(|r|, |q|)$, and the stiffer the option the closer to expiry ($z = -1$) the interpolant has to bend. Over
2,500 random puts (expiries 0.01 to 100 years, volatilities 1 % to 150 %, rates 0 to 30 %, dividend yields
$-15$ % to 35 %) against a 65-node scheme, the largest error relative to $\max(1, V)$ is:

| $\Lambda$ up to | 1 | 5 | 8 | 25 | 100 | $10^4$ | beyond |
|---|---|---|---|---|---|---|---|
| `fast` | 9.7e-5 | 3.9e-4 | 4.2e-4 | 3.2e-2 | 2.8e-1 | 1.0 | 2.1e-2 |
| `accurate` | 2.4e-6 | 7.6e-6 | 1.0e-5 | 2.7e-5 | 8.0e-5 | 1.3e-3 | 2.8e-3 |
| `high` | 2.6e-8 | 8.0e-8 | 1.2e-7 | 2.7e-7 | 8.5e-7 | 1.0e-5 | 2.4e-4 |

So beyond $\Lambda = 5$ `fast` hands the option to `accurate`, and beyond 25 `accurate` hands it to `high`;
`price_details(...).scheme` names the scheme used (a `Scheme` object of one's own is used as given). On the grid
below, $\Lambda \le 1.44$, so nothing is handed over there. The value jumps where the scheme changes, by the error
of the scheme handed over (at most about $4 \times 10^{-4} \max(1, V)$ for `fast`).

The boundary is proportional to the strike, so the normalised boundary $B/K$ is cached per
$(T, r, q, \sigma, \text{scheme}, \text{form})$ (4,096 entries): a strike strip, or the spots of a profile, solves
one boundary. A volatility smile changes $\sigma$ with the strike and needs one solve per strike. A batch is grouped
by $(T, r, \sigma, q)$ with one sort, so grouping costs $O(N \log N)$ (11 ms for 400,000 options in 4,000 groups).

## Regimes

Healy (2021) and Andersen and Lake (2021) classify the exercise region of a put:

- $r \le 0$ and $q \ge r$: early exercise is never optimal; the value is the European one (`method = "european"`).
- $q < r < 0$: the exercise region lies between **two** boundaries. This engine, like QuantLib's (which refuses the
  case), solves one. The value falls back to `binomial.extrapolated_price`, a Richardson extrapolation of
  Leisen-Reimer (1996) trees with 2,001 and 4,003 steps (Leisen-Reimer trees have an odd number of steps), and
  `price_details` reports `fallback = True`, `fallback_reason = "double boundary"`. The tree values each option on
  its own, at about 60 ms per option: a 41-strike strip takes 2.2 s, far above the 15 ms of a collocation strip,
  so a chain or a profile in this regime needs a loading state. Where the bounds meet (at extreme total
  volatility the European value reaches $K e^{-rT}$, the upper bound) no tree is built, and the value is that
  bound (`method = "exact"`).
- otherwise one boundary, solved as above.

**Closed forms** (`method = "exact"`, `critical_spot` not computed):

- $\sigma = 0$: the best deterministic exercise date of $K e^{-rt} - S e^{-qt}$ (now, at expiry, or at
  $t^* = \ln(rK / (qS)) / (r - q)$ when $q > r > 0$).
- $\sigma\sqrt T < 10^{-8}$: the same, floored at the European value. The error is of the order of vega times
  $\sigma$, which for a European option is at most $S e^{-qT} \sigma \sqrt T / \sqrt{2\pi}$, here
  $4 \times 10^{-9} S e^{-qT}$ (the tests find the engine and this closed form within $4 \times 10^{-9} S$ of each
  other across the threshold); the engine is no more accurate there, its boundary layers
  being thinner than its quadrature rules resolve (before this branch, a 20-year put with $q > r$ at
  $\sigma = 10^{-10}$ came out 1.8 below its zero-volatility value, and $\sigma \le 10^{-77}$ overflowed).
- $\max(|r|, |q|)\,T < 10^{-16}$: $\max(v, K - S)$. The premium is at most
  $(|r| K + |q| S)\,T\,e^{\max(|r|, |q|) T}$, below rounding.

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
$5.0 \times 10^{-5}$ (`fast`), $9.4 \times 10^{-7}$ (`accurate`) and $3.1 \times 10^{-9}$ (`high`); before the
hand-over it was $2.0 \times 10^{-4}$ and $2.7 \times 10^{-6}$. At low volatility (40 options at 1 % to 6 %, rates
up to 30 %, expiries up to 30 years) it is $2.1 \times 10^{-5}$, $3.7 \times 10^{-8}$ and $3.7 \times 10^{-9}$;
before the hand-over `fast` was 39 % low on a 5-year put with $r = 25$ % and $\sigma = 2$ % (0.01800 against
0.029419), and is now within $2 \times 10^{-11}$.

Long expiries: the put with $S = K = 100$, $r = 5$ %, $q = 0$, $\sigma = 20$ % is worth 12.2021227 at 30 years
(QdFp strict; `fast` +2.8e-3, `accurate` -3.0e-5, `high` -3.9e-7, `fast` not handed over at $\Lambda = 1.9$) and
approaches the perpetual put, 12.3200329, from below: every scheme gives it to $10^{-6}$ from 1,000 to 100,000
years (before the hand-over `fast` gave 13.5713, 6.5480 and 25.6631 at 1,000, 3,000 and 10,000 years). Values
never leave the no-arbitrage bounds (before the bounds were applied, a sweep of expiries from 1 to 100 years
found `fast` above the perpetual bound in 94 cases, by up to $7.3 \times 10^{-2}$, `accurate` in 89 and `high` in
44), and along $\pm 10$ % lines in $T$ up to 30 years the largest step down is $7.5 \times 10^{-9}$, at the error
of `high` on a put near its perpetual value.

The fixed-point form: the automatic choice changes at $|r - q| = 10^{-3}$, and the value jumps there by the
change of discretisation error, up to $3.9 \times 10^{-4}$ (`fast`), $1.1 \times 10^{-6}$ (`accurate`) and
$2 \times 10^{-9}$ (`high`) on a 10-year put; the slope predicts $3 \times 10^{-10}$ over the $2 \times 10^{-12}$
step. Bumped Greeks across that line should fix the form with `fixed_point`. Either form converges: forced on the
whole grid, the density form has RMSE $5.7 \times 10^{-6}$, $2.8 \times 10^{-7}$ and $1.9 \times 10^{-7}$ against
`qdfp_strict` (max $2.6 \times 10^{-4}$, $1.7 \times 10^{-5}$, $1.3 \times 10^{-5}$); far from $r = q$ it needs
more iterations than `high` gives it, which is why the cdf form is the default there.

The example of Andersen et al. (2016), $S = K = 100$, $q = 5\%$, $\sigma = 25\%$, one year: the early-exercise
premium is 0.1069527027479 at $r = 5\%$ (density form) and 0.3671111406253 at $r = 7.5\%$ (cdf form), from QdFp at
its tightest settings; `high` agrees to $1.0 \times 10^{-12}$ and $4.6 \times 10^{-11}$.

The double-boundary put ($S = K = 100$, $r = -1\%$, $q = -2\%$, $\sigma = 20\%$, one year) falls back to 7.6252878;
QuantLib's finite-difference engine on a 4,000 × 4,000 grid gives 7.6252621.

Properties checked (Hypothesis and parameter sweeps): put-call symmetry to $10^{-14}$ relative; a call with
$q = 0 \le r$ equals Black-Scholes to $10^{-12}$; puts with $r \le 0 \le q - r$ equal Black-Scholes; the value lies
between $\max(v, \text{intrinsic})$ and `binomial.american_upper_bound`, and, up to 30 years, within the bounds of
`binomial._american_bounds` (with the perpetual put); it is monotone in $\sigma$, $T$ and $K$ (no decrease beyond
$1.4 \times 10^{-14}$ up to 5 years, $7.5 \times 10^{-9}$ up to 30); the boundary starts at $X$. The boundary is
not exactly monotone in $\tau$ (see Limits). Inputs at the edges of the double range (subnormal volatilities,
expiries down to $5 \times 10^{-324}$ years, $rT$ and $|q|T$ beyond 709, volatilities of 5,000 %) stay finite,
quiet and within the bounds.

## Greeks and implied volatility

`greeks` returns delta, gamma, theta, vega, rho and dividend rho by central differences of `price`: the spot and
the time to expiry are moved by $\pm 10^{-4}$ of their values, the volatility, the rate and the dividend yield by
$\pm 10^{-4}$ (`GREEK_BUMPS`). Every bump is priced with the scheme that solved the option (after any hand-over) and
the form of its fixed-point equation, held fixed, so that no difference straddles one of the jumps under Limits; a
plain difference with `fast` at $\Lambda$ just below 5 (a one-year put, $r = 10\%$, $\sigma = 4.475\%$) misreads
vega by 0.9 %. Held fixed, the engine is smooth in every input within a regime: at $S = K = 100$, one year,
$r = 5\%$, $\sigma = 20\%$ the put's Greeks agree with bumps ten times smaller to $4 \times 10^{-7}$ of each Greek.
Where the option falls back to the tree, the differences are those of tree values, which are far less smooth.

The value is not smooth where the way the option is valued changes, so no difference is taken across such a line:

* An option valued as European gets the closed-form Black-Scholes-Merton Greeks. At $r = q = 0$ both the call
  and the put are European, but a 1 bp move one way makes early exercise pay (a lower rate or a higher yield for
  the call, a higher rate or a lower yield for the put), and the value has a kink: the premium grows about in
  proportion to the move, since the boundary at expiry is the strike. A central difference there averages two
  slopes and missed rho and dividend rho by about 3 % at the money (6 to 7 % for a call at $S = 120$); the closed
  form is the slope on the European side. It is also free of the rounding of differences of a value as large as
  the strike: at $S = 0.01$, $K = 10^7$, 30 days, $r = -5\%$, $q = 5\%$, $\sigma = 0.1\%$ a central difference
  gave the put a gamma of $-1863$.
* An option exercised at once, at or beyond its critical spot, is worth its intrinsic value nearby: delta is
  $\pm 1$ and the other Greeks are zero. Differences would be rounding there too (a put delta of $-1.00024$ at
  $S = 0.01$, $K = 10^7$, 30 days, $r = 5\%$, $\sigma = 10\%$).
* Where one bump changes the method (the regime, a fallback, or exercise at once), the derivative is taken on the
  other side alone: one-sided differences over half the bump and over a quarter of it, extrapolated (Richardson;
  third order for the slope, second for the curvature). A call at a negative rate without dividends is one: at
  $q - 10^{-4}$ it lies in the double-boundary regime and is valued on the tree. Its dividend rho at $r = -3\%$ is
  $-36.81271$ where the central difference gave $-36.8199$; at $r = -0.05\%$, where the value bends sharply near
  the edge, it is $-49.15829$ against $-49.15836$ from one-sided differences fifty times finer, where the central
  difference gave $-49.1644$.

`AmericanGreeks.basis` and `AmericanGreeks.one_sided` say which applied; the app states them under its Greeks
table. Over 4,320 options (spots and strikes from 0.01 to $10^7$, expiries from a day to ten years, rates from
$-5\%$ to 50 %, volatilities from 0.1 % to 500 %, dividend yields from 0 to 50 %, calls and puts), no delta
leaves $[0, 1]$ (call) or $[-1, 0]$ (put) by more than $5 \times 10^{-10}$, and no gamma is below $-10^{-6}$ but
one of $-3.5 \times 10^{-6}$, from the tree values of a fallback. Central differences throughout broke one of these
limits 343 times (deltas down to $-1.00024$, gammas down to $-3725$).

`implied_vol.implied_volatility_alo` inverts `price` with Brent's method. The quote is first checked against the
American bounds, with the messages of the tree solver; since the engine values $\sigma = 0$ exactly, the bracket
starts there. A quote no more than $10^{-12}\max(S, K)$ above the zero-volatility value (or up to
$10^{-10}\max(S, K)$ below it) is rejected with the reason `no_time_value`: every volatility up to some level gives
that value (for a deep in-the-money put, every volatility at which exercising at once stays optimal), so the quote
does not identify one. Unlike the European solver, which inverts a tiny out-of-the-money quote to full relative
precision, this margin applies out of the money too, because near zero volatility the engine is accurate only in
absolute terms (below a total volatility of $10^{-8}$ it is the zero-volatility value, within about
$4 \times 10^{-9} S$).

## Speed

Median wall-clock times on the development machine (one CPU core, NumPy), with the boundary cache cleared:

| Scheme | One option (its own boundary) | Whole grid, one call (600 boundaries) | 41-strike strip (one boundary) |
|---|---|---|---|
| `fast` | 0.40 ms | 0.27 s | 0.52 ms |
| `accurate` | 0.70 ms | 0.53 s | 0.80 ms |
| `high` | 1.63 ms | 1.25 s | 2.22 ms |

The targets were 1 ms (`fast`) and 5 ms (`accurate`) per option and 15 ms for the strip.

## Limits

- Constant $r$, $q$ and $\sigma$: no term structures, no discrete dividends, no smile within one boundary.
- The double-boundary regime is valued on the tree, not by collocation: about $10^{-5}$ accurate and 60 ms per
  option (2.2 s for a 41-strike strip). Andersen and Lake (2021) and Healy (2021) extend the collocation to two
  boundaries.
- The boundary should never rise with time to expiry (fall, for a call); the Chebyshev interpolant ripples between
  its nodes, most just before expiry at high volatility. `fast` interpolates with 8 nodes: its put boundary can
  rise by up to $8.1 \times 10^{-3}$ of its value at volatilities up to 60 % and $2.1 \times 10^{-2}$ at 150 %
  (`accurate` $9.1 \times 10^{-8}$ and $3.0 \times 10^{-5}$, `high` $6.8 \times 10^{-9}$ and
  $3.9 \times 10^{-6}$). Use `accurate` or `high` when the boundary itself is shown.
- The value jumps where a scheme hands over ($\Lambda = 5$ and 25) and where the automatic fixed-point form
  changes ($|r - q| = 10^{-3}$), by the error of the scheme. Finite differences across those lines need a fixed
  `Scheme` object and `fixed_point`.
- The convergence threshold ($10^{-2}$ in $\ln B$) guards against divergence, not against slow convergence. Beyond
  $\Lambda = 10^4$ even `high` errs by up to $2.4 \times 10^{-4}$ of $\max(1, V)$, and at absurd horizons it fails:
  the put above at a million years ($rT = 5 \times 10^4$) comes out 12.149, 1.4 % below the perpetual value, the
  bounds not being tight there.

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
- R. C. Merton (1973), "Theory of rational option pricing", *Bell Journal of Economics and Management Science*
  4(1), 141-183, doi:[10.2307/3003143](https://doi.org/10.2307/3003143).
- K. Spanderen (2022), QuantLib `QdFpAmericanEngine` (BSD-3-Clause),
  <https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/qdfpamericanengine.cpp>.
