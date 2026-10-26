# This module is a port of the reference implementation of "Let's Be Rational", revision 1520, which resides at
# www.jaeckel.org/LetsBeRational.7z . Its notice, preserved as it requires:
#
# ======================================================================================
# Copyright © 2013-2024 Peter Jäckel.
#
# Permission to use, copy, modify, and distribute this software is freely granted,
# provided that this notice is preserved.
#
# WARRANTY DISCLAIMER
# The Software is provided "as is" without warranty of any kind, either express or implied,
# including without limitation any implied warranties of condition, uninterrupted use,
# merchantability, fitness for a particular purpose, or non-infringement.
# ======================================================================================
"""Black implied volatility to machine precision: Peter Jäckel's "Let's Be Rational".

A port of the reference implementation of P. Jäckel, "Let's Be Rational", Wilmott, January 2015, pp. 40-53, at
revision 1520 (2024-02-16, www.jaeckel.org). The notation follows the paper: ``x = ln(F/K)``, ``s = σ√T`` and
``θ = +1`` for a call, ``-1`` for a put. The normalised Black function

    b(x, s, θ) = θ·[e^(x/2)·Φ(θ·(x/s + s/2)) - e^(-x/2)·Φ(θ·(x/s - s/2))]

is the undiscounted Black price divided by ``√(FK)``. Inversion maps the quote to the out-of-the-money side
(``θx ≤ 0``), picks one of four branches by the price, starts from a rational-cubic interpolation of a transformed
price in that branch (within 13 % of the answer in every test, usually far closer) and takes two Householder(3)
steps on an objective chosen per branch (Householder(4) for ``|x|`` above 190 or 580). That reaches the attainable
accuracy ``(1 + |b/(s·∂b/∂s)|)·ε``, the error that rounding the price alone causes.

Every public function broadcasts over NumPy arrays (array code that evaluates each branch once on its own rows)
and has a scalar path written with the ``math`` module, used when all inputs are scalars, which returns a Python
``float``. The two paths agree to rounding, and on the same rules outside the domain: a NaN or negative volatility,
a forward or strike that is not positive and finite, or a negative expiry gives NaN. Neither raises: where ``math``
would raise on overflow or a domain error, the scalar path recomputes on the array path, which follows IEEE
arithmetic. NumPy's ``exp`` and ``sinh`` can differ from the C library's in the last bit, so a price within an ulp
of the intrinsic value or the maximum may fall on different sides of it on the two paths.
:mod:`options_analyzer.implied_vol` builds the spot-model solvers on this module and makes those decisions with
the C library on both of its paths.

Differences from the reference implementation:

* ``scipy.special`` (``erfc``, ``erfcx``, ``erf``, ``erfinv``, ``ndtri``) replaces W. J. Cody's error functions and
  Jäckel's inverse normal; ``math.erfc`` and ``math.erf`` are used on the scalar path. Arguments of ``erfcx`` are
  always positive, where SciPy's version is accurate to a few units in the last place.
* The coefficients of the asymptotic expansion are generated from their closed form (they reproduce the
  reference's literals exactly), and the branch of Cody's split that cannot occur (``q₁ ≥ 0.46875 > q₂``) is
  omitted.
* The reference has no failure signal. Here the iteration is reported as *converged* when its result is finite
  and positive and its last Householder step was below ``1e-4`` of the volatility. The steps converge with
  fourth order, the next error being about ``C·δ⁴`` after a step ``δ`` with ``C ≤ 2`` measured, so after such a
  step the result is accurate to ``ε``. On ``|x| ≤ 80·s`` and ``s ≤ 20`` the second step is at most ``4.4e-5``;
  in a sample up to ``|x| = 700`` and ``s = 100`` only ``|x| > 182`` with ``24 < s < 41`` was flagged. Callers fall
  back to a bracketing solver on flagged rows.
* Prices below the intrinsic value give NaN and prices at or above the maximum (``F`` for a call, ``K`` for a put)
  give ``inf``, where the reference returns ``-DBL_MAX`` and ``DBL_MAX``.
* :func:`black` and :func:`implied_black_volatility` take ``ln(F/K)`` from ``log1p`` near the money, where
  ``log(F/K)`` has an absolute error of ``ε`` that near the forward costs the volatility up to ``1e-14``, and from
  ``ln F - ln K`` where ``F/K`` under- or overflows.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from functools import partial
from math import comb, prod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf, erfc, erfcx, erfinv, ndtri

__all__ = [
    "black",
    "implied_black_volatility",
    "normalised_black",
    "normalised_implied_volatility",
    "normalised_vega",
]

# ------------------------------------------------------------------------------------------------ constants

_EPS = 2.220446049250313e-16  # DBL_EPSILON
_DBL_MIN = 2.2250738585072014e-308
_DBL_MAX = 1.7976931348623157e308
_SQRT_DBL_MAX = math.sqrt(_DBL_MAX)

_TWO_PI = 6.283185307179586476925286766559005768394338798750
_SQRT_TWO = 1.4142135623730950488016887242096980785696718753769
_SQRT_TWO_PI = 2.5066282746310005024157652848110452530069867406099
_LN_TWO_PI = 1.8378770664093454835606594728112352797227949472756
_SQRT_PI_OVER_TWO = 1.253314137315500251207882642405522626503493370305
_SQRT_THREE = 1.732050807568877293527446341505872366942805253810
_SQRT_ONE_OVER_THREE = 0.577350269189625764509148780501957455647601751270
_TWO_PI_OVER_SQRT_TWENTY_SEVEN = 1.209199576156145233729385505094770488189377498728
_SQRT_THREE_OVER_THIRD_ROOT_TWO_PI = 0.938643487427383566075051356115075878414688769574
_PI_OVER_SIX = 0.523598775598298873077107230546583814032861566563
_ONE_OVER_SQRT_TWO = 1 / _SQRT_TWO
_ONE_OVER_SQRT_TWO_PI = 1 / _SQRT_TWO_PI
_TWO_OVER_SQRT_TWO_PI = 2 / _SQRT_TWO_PI
_HALF_OVER_SQRT_TWO = 0.5 / _SQRT_TWO

# Regions of the normalised Black function for θx < 0, with h = θx/s and t = s/2 (Figure-6 of revision 1517):
#   I.   h < η and t < (τ + ½) + (|h| - |η|): asymptotic expansion of b/b' in powers of (h/((h+t)(h-t)))²;
#   II.  η ≤ h ≤ 0 and t < τ + (½/|η|)·|h|:   Taylor expansion of b/b' in t;
#   III. otherwise:                           erfc/erfcx, choosing per term the form with fewest exponentials.
_ETA = -13.0
_TAU = 2 * math.sqrt(math.sqrt(math.sqrt(math.sqrt(_EPS))))  # 2·ε^(1/16) ≈ 0.21
_CODY_THRESHOLD = 0.46875

#: The iteration has converged when its last step is below this fraction of the volatility (see the module notes).
_CONVERGED_STEP = 1e-4
_ITERATIONS = 2

# Rational cubic interpolation (Delbourgo and Gregory 1985): limits of the control parameter r.
_MINIMUM_RATIONAL_CUBIC_CONTROL = -(1 - math.sqrt(_EPS))
_MAXIMUM_RATIONAL_CUBIC_CONTROL = 2 / (_EPS * _EPS)


def _asymptotic_coefficients(n_max: int) -> tuple[tuple[float, ...], ...]:
    """Coefficients of A_n(e) = (-1)^n·2·(2n-1)!!·Σ_k C(2n+1, 2k+1)·e^k, in ascending powers of e.

    These are the terms of the expansion of (r/t)·[Y(h+t) - Y(h-t)] in q = (h/r)², r = (h+t)(h-t), e = (t/h)²,
    that follows from Abramowitz and Stegun 26.2.12, Y(z) = Φ(z)/φ(z) ~ -1/z·Σ (-1)^n (2n-1)!!/z^(2n).
    """
    return tuple(
        tuple(float((-1) ** n * 2 * prod(range(2 * n - 1, 0, -2)) * comb(2 * n + 1, 2 * k + 1)) for k in range(n + 1))
        for n in range(n_max + 1)
    )


_A = _asymptotic_coefficients(16)
# Upper bounds of -h - t + τ + ½ beyond which one more term is dropped: 17 terms (A0..A16) below the first.
_ASYMPTOTIC_THRESHOLDS = (12.347, 12.958, 13.729, 14.718, 16.016, 17.769, 20.221, 23.816, 29.419, 38.93, 57.171, 99.347)


def _horner(coefficients: Sequence[float], x: Any) -> Any:
    """Σ c_k x^k for ascending coefficients, for a float or an array."""
    result = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        result = c + x * result
    return result


# ------------------------------------------------------------------- rational approximations (float or array)
# The Remez-optimised minimax rational functions of revision 1520, as (numerator, denominator) coefficients in
# ascending powers. Horner's rule evaluates them with the reference's own operation order, so the results agree
# with its nested expressions to the last bit.


def _rational(coefficients: tuple[Sequence[float], Sequence[float]], x: Any) -> Any:
    numerator, denominator = coefficients
    return _horner(numerator, x) / _horner(denominator, x)


# Y'(h) = 1 + h·Y(h) for h < -4 is w·(1 + w·P(w)/Q(w)) with w = 1/h²: order (5,6), accuracy better than 9.8e-17.
_YPRIME_TAIL = (
    (-2.9999999999994663866, -1.7556263323542206288e2, -3.4735035445495633334e3, -2.7805745693864308643e4,
     -8.3836021460741980839e4, -6.6818249032616849037e4),
    (1.0, 6.3520877744831739102e1, 1.4404389037604337538e3, 1.4562545638507033944e4, 6.6886794165651675684e4,
     1.2569970380923908488e5, 6.9286518679803751694e4),
)  # fmt: skip
# Y'(h) for -4 ≤ h ≤ -0.46875, in powers of h: order (7,7), accuracy better than 1.6e-16.
_YPRIME_MIDDLE = (
    (1.0000000000594317229, -6.1911449879694112749e-1, 2.2180844736576013957e-1, -4.5650900351352987865e-2,
     5.545521007735379052e-3, -3.0717392274913902347e-4, 4.2766597835908713583e-8, 8.4592436406580605619e-10),
    (1.0, -1.8724286369589162071, 1.5685497236077651429, -7.6576489836589035112e-1, 2.3677701403094640361e-1,
     -4.6762548903194957675e-2, 5.5290453576936595892e-3, -3.0822020417927147113e-4),
)  # fmt: skip
# 1 - erfcx(x) = x·(2/√π - x·P(x)/Q(x)) for -1/5 ≤ x ≤ 1/3, without cancellation: order (4,5), accuracy 2.5e-17.
_ONE_MINUS_ERFCX = (
    (1.0000000000000002, 1.1514967181784756, 5.7689001208873741e-1, 1.4069188744609651e-1, 1.4069285713634565e-2),
    (1.0, 1.9037494962421563, 1.5089908593742723, 6.2486081658640257e-1, 1.358008134514386e-1, 1.2463320728346347e-2),
)  # fmt: skip
_TWO_OVER_SQRT_PI = 1.128379167095512573896

# b_l(x)/b_max(x) and b_u(x)/b_max(x), the branch edges of the initial guess, as functions of y = s_c = √(2|x|),
# in four branches each. Branch I of each is y²·(c₂ + y·(c₃ + y·g)) (b_l) or c₀ + y²·(c₂ + y·g) (b_u) with g a
# rational function; the others are rational functions of y.
_BL_EDGES = (0.7099295739719539, 2.6267851073127395, 7.348469228349534)  # |x| = 0.252, 3.45, 27
_BL_I = (  # order (5,4), accuracy better than 7.43e-17
    (8.0741072372882856924e-2, 9.8078911786358897272e-2, 3.9760631445677058375e-2, 5.9716928459589189876e-3,
     -6.4036399341479799981e-6, 4.5425102093616062245e-7),
    (1.0, 1.8594977672287664353, 1.3658801475711790419, 4.6132707108655653215e-1, 6.1254597049831720643e-2),
)  # fmt: skip
_BL_II = (  # order (6,6), accuracy better than 8.77e-17
    (1.9795737927598581235e-9, -2.7081288564685588037e-8, 7.5610142272549044609e-2, 6.917130174466834016e-2,
     2.9537058950963019803e-2, 6.5849252702302307774e-3, 6.9711400639834715731e-4),
    (1.0, 2.1941448525586579756, 2.1297103549995181357, 1.1571483187179784072, 3.7831622253060456794e-1,
     7.1714862448829349869e-2, 6.6361975827861200167e-3),
)  # fmt: skip
_BL_III = (  # order (6,6), accuracy better than 7.49e-17
    (-9.3325115354837883291e-5, 5.3118033972794648837e-4, 7.4114855448345002595e-2, 7.4039658186822817454e-2,
     3.9225177407687604785e-2, 1.0022913378254090083e-2, 1.7012579407246055469e-3),
    (1.0, 2.2217238132228132256, 2.3441816707087403282, 1.3912323646271141826, 5.3231258443501838354e-1,
     1.1744005919716101572e-1, 1.6195405895930935811e-2),
)  # fmt: skip
_BL_IV = (  # order (6,6), accuracy better than 8.4e-17
    (1.4500072297240603183e-3, -1.5116692485011195757e-3, 7.1682178310936334831e-2, 3.921610857820463493e-2,
     2.9342405658628443931e-2, 5.1832526171631521426e-3, 1.6930208078421474854e-3),
    (1.0, 1.6176313502305414664, 1.6823159175281531664, 8.4878307567372222113e-1, 3.7543742137375791321e-1,
     7.126137099644302999e-2, 1.6116992546788676159e-2),
)  # fmt: skip
_BU_EDGES = (0.7745966692414833, 1.7888543819998317, 6.164414002968976)  # |x| = 0.3, 1.6, 19
_BU_I = (  # order (5,4), accuracy better than 9.2e-17
    (-6.063099881233561706e-2, -8.1011946637120604985e-2, -4.2505564862438753828e-2, -8.9880000946868691788e-3,
     -7.5603072110443268356e-6, 4.3879556621540147458e-7),
    (1.0, 1.8400371530721828756, 1.5709283443886143691, 6.8913245453611400484e-1, 1.4703173061720980923e-1),
)  # fmt: skip
_BU_II = (  # order (6,5), accuracy better than 8.4e-17
    (7.8990944435755287611e-1, -1.2655410534988972886, -2.8803040699221003256, -2.6936198689113258727,
     -1.1213067281643205754, -2.1277793801691629892e-1, 5.1486445905299802703e-6),
    (1.0, -1.6021222722060444448, -3.7242680976480704555, -3.2083117718907365085, -1.2922333835930958583,
     -2.3762328334050001161e-1),
)  # fmt: skip
_BU_III = (  # order (6,6), accuracy better than 7.7e-17
    (7.8990640048967596475e-1, 1.5993699253596663678, 1.6481729039140370242, 9.8227188109869200166e-1,
     3.6313557966186936883e-1, 7.8277036261179606301e-2, 9.3404307364538726214e-3),
    (1.0, 2.0247407005640401446, 2.0087454279103740489, 1.1627561803056961973, 4.2004672123723823581e-1,
     8.9130862793887234546e-2, 1.0436767768858021717e-2),
)  # fmt: skip
_BU_IV = (  # order (6,6), accuracy better than 3.9e-17
    (7.91133825948419359e-1, 1.24653733210880042, 1.32747426980537386, 6.95009705717846778e-1,
     3.05965944268228457e-1, 6.02200363391352887e-2, 1.29050244454344842e-2),
    (1.0, 1.58117486714634672, 1.60144713247629644, 8.30040185836882436e-1, 3.53071863813401531e-1,
     6.95901684131758475e-2, 1.44197580643890011e-2),
)  # fmt: skip


def _yprime_tail(h: Any) -> Any:
    w = 1 / (h * h)
    return w * (1 + w * _horner(_YPRIME_TAIL[0], w) / _horner(_YPRIME_TAIL[1], w))


def _one_minus_erfcx_rational(x: Any) -> Any:
    return x * (_TWO_OVER_SQRT_PI - x * _horner(_ONE_MINUS_ERFCX[0], x) / _horner(_ONE_MINUS_ERFCX[1], x))


def _small_t_expansion(h: Any, t: Any, a: Any) -> Any:
    """b/b' = Y(h+t) - Y(h-t) to order t^13 for h ≤ 0 and small t, given a = Y'(h)."""
    h2 = h * h
    t2 = t * t
    b0 = 2 * a
    b1 = (-1 + a * (3 + h2)) / 3
    b2 = (-7 - h2 + a * (15 + h2 * (10 + h2))) / 60
    b3 = (-57 + (-18 - h2) * h2 + a * (105 + h2 * (105 + h2 * (21 + h2)))) / 2520
    b4 = (-561 + h2 * (-285 + (-33 - h2) * h2) + a * (945 + h2 * (1260 + h2 * (378 + h2 * (36 + h2))))) / 181440
    b5 = (
        -6555
        + h2 * (-4680 + h2 * (-840 + (-52 - h2) * h2))
        + a * (10395 + h2 * (17325 + h2 * (6930 + h2 * (990 + h2 * (55 + h2)))))
    ) / 19958400
    b6 = (
        -89055
        + h2 * (-82845 + h2 * (-20370 + h2 * (-1926 + (-75 - h2) * h2)))
        + a * (135135 + h2 * (270270 + h2 * (135135 + h2 * (25740 + h2 * (2145 + h2 * (78 + h2))))))
    ) / 3113510400
    return t * (b0 + t2 * (b1 + t2 * (b2 + t2 * (b3 + t2 * (b4 + t2 * (b5 + b6 * t2))))))


def _bl_over_bmax(y: Any) -> Any:
    def branch_i(y: Any) -> Any:
        return (y * y) * (0.07560996640296361767172 + y * (y * _rational(_BL_I, y) - 0.09672719281339436290858))

    pieces = (branch_i, partial(_rational, _BL_II), partial(_rational, _BL_III), partial(_rational, _BL_IV))
    return _piecewise(y, _BL_EDGES, pieces)


def _bu_over_bmax(y: Any) -> Any:
    def branch_i(y: Any) -> Any:
        return 0.7899085945560627246288 + (y * y) * (0.0614616805805147403487 + y * _rational(_BU_I, y))

    pieces = (branch_i, partial(_rational, _BU_II), partial(_rational, _BU_III), partial(_rational, _BU_IV))
    return _piecewise(y, _BU_EDGES, pieces)


def _piecewise(x: Any, edges: Sequence[float], pieces: Sequence[Callable[[Any], Any]]) -> Any:
    """``pieces[i](x)`` below ``edges[i]`` (first match), the last piece above the last edge; float or array."""
    if isinstance(x, float):
        for edge, piece in zip(edges, pieces[:-1], strict=True):
            if x < edge:
                return piece(x)
        return pieces[-1](x)
    return np.select([x < edge for edge in edges], [piece(x) for piece in pieces[:-1]], pieces[-1](x))


def _householder3_factor(nu: Any, h2: Any, h3: Any) -> Any:
    """The step is ν times this, with ν = -g/g′ and h₂, h₃ the second and third derivatives of g over g′."""
    return (1 + 0.5 * h2 * nu) / (1 + nu * (h2 + h3 * nu * (1.0 / 6.0)))


def _householder4_factor(nu: Any, h2: Any, h3: Any, h4: Any) -> Any:
    """As :func:`_householder3_factor`, with h₄ the fourth derivative of g over g′ as well."""
    return (1 + nu * (h2 + nu * h3 * (1.0 / 6.0))) / (
        1 + nu * (1.5 * h2 + nu * (h2 * h2 * 0.25 + h3 * (1.0 / 3.0) + nu * h4 * (1.0 / 24.0)))
    )


# ------------------------------------------------------------------------------------- scalar path (math)


def _erfcx(x: float) -> float:
    return float(erfcx(x))


def _vega_s(x: float, s: float) -> float:
    h = x / s
    t = 0.5 * s
    return _ONE_OVER_SQRT_TWO_PI * math.exp(-0.5 * (h * h + t * t))


def _inv_vega_s(x: float, s: float) -> float:
    h = x / s
    t = 0.5 * s
    return _SQRT_TWO_PI * math.exp(0.5 * (h * h + t * t))


def _ln_vega_s(x: float, s: float) -> float:
    h = x / s
    t = 0.5 * s
    return -(_LN_TWO_PI * 0.5) - 0.5 * (h * h + t * t)


def _yprime_s(h: float) -> float:
    if h < -4:
        return _yprime_tail(h)
    if h <= -_CODY_THRESHOLD:
        return _rational(_YPRIME_MIDDLE, h)
    return 1 + h * _SQRT_PI_OVER_TWO * _erfcx(-_ONE_OVER_SQRT_TWO * h)


def _asymptotic_expansion_s(h: float, t: float) -> float:
    """b/b' in region I (h < η), with as many terms as the distance from the region's edge requires."""
    e = (t / h) * (t / h)
    r = (h + t) * (h - t)
    q = (h / r) * (h / r)
    first_dropped = bisect.bisect_right(_ASYMPTOTIC_THRESHOLDS, -h - t + _TAU + 0.5)
    omega = 0.0
    for n in range(16 - first_dropped, 4, -1):
        omega = q * (_horner(_A[n], e) + omega)
    omega = _A[0][0] + q * (
        _horner(_A[1], e) + q * (_horner(_A[2], e) + q * (_horner(_A[3], e) + q * (_horner(_A[4], e) + omega)))
    )
    return (t / r) * omega


def _black_region_iii_s(x: float, s: float) -> float:
    h = x / s
    t = 0.5 * s
    q1 = -_ONE_OVER_SQRT_TWO * (h + t)
    q2 = -_ONE_OVER_SQRT_TWO * (h - t)  # q2 > q1 since t > 0
    if q2 < _CODY_THRESHOLD:
        two_b = math.exp(0.5 * x) * math.erfc(q1) - math.exp(-0.5 * x) * math.erfc(q2)
    elif q1 < _CODY_THRESHOLD:
        two_b = math.exp(0.5 * x) * math.erfc(q1) - math.exp(-0.5 * (h * h + t * t)) * _erfcx(q2)
    else:
        two_b = math.exp(-0.5 * (h * h + t * t)) * (_erfcx(q1) - _erfcx(q2))
    return max(0.5 * two_b, 0.0)


def _is_region_i(x: Any, s: Any) -> Any:
    return (x < s * _ETA) & (s * (0.5 * s - (_TAU + 0.5 + _ETA)) + x < 0)


def _is_region_ii(x: Any, s: Any) -> Any:
    return s * (s - (2 * _TAU)) - x / _ETA < 0


def _normalised_black_otm_s(x: float, s: float) -> float:
    """b(x, s) for x < 0 and s > 0."""
    if _is_region_i(x, s):
        return _asymptotic_expansion_s(x / s, 0.5 * s) * _vega_s(x, s)
    if _is_region_ii(x, s):
        h = x / s
        return _small_t_expansion(h, 0.5 * s, _yprime_s(h)) * _vega_s(x, s)
    return _black_region_iii_s(x, s)


def _scaled_black_and_ln_vega_s(x: float, s: float) -> tuple[float, float]:
    """(b/b', ln b') for x < 0 and s > 0, without forming the possibly underflowing b."""
    ln_vega = _ln_vega_s(x, s)
    if _is_region_i(x, s):
        return _asymptotic_expansion_s(x / s, 0.5 * s), ln_vega
    if _is_region_ii(x, s):
        h = x / s
        return _small_t_expansion(h, 0.5 * s, _yprime_s(h)), ln_vega
    return _black_region_iii_s(x, s) * math.exp(-ln_vega), ln_vega


def _one_minus_erfcx_s(x: float) -> float:
    if x < -1.0 / 5.0 or x > 1.0 / 3.0:
        return 1 - _erfcx(x)
    return _one_minus_erfcx_rational(x)


def _rational_cubic_interpolation(
    x: float, x_l: float, x_r: float, y_l: float, y_r: float, d_l: float, d_r: float, r: float
) -> float:
    """Delbourgo-Gregory rational cubic through (x_l, y_l), (x_r, y_r) with end slopes d_l, d_r and control r."""
    h = x_r - x_l
    if abs(h) <= 0:
        return 0.5 * (y_l + y_r)
    t = (x - x_l) / h
    if not r >= _MAXIMUM_RATIONAL_CUBIC_CONTROL:
        omt = 1 - t
        t2 = t * t
        omt2 = omt * omt
        return (y_r * t2 * t + (r * y_r - h * d_r) * t2 * omt + (r * y_l + h * d_l) * t * omt2 + y_l * omt2 * omt) / (
            1 + (r - 3) * t * omt
        )
    return y_r * t + y_l * (1 - t)  # linear interpolation without over- or underflow


def _rational_cubic_control_s(x_l, x_r, y_l, y_r, d_l, d_r, second_derivative, prefer_shape, *, left):
    """Convex control parameter that fits the second derivative at the left (or right) end, as in the reference."""
    h = x_r - x_l
    numerator = 0.5 * h * second_derivative + (d_r - d_l)
    denominator = ((y_r - y_l) / h - d_l) if left else (d_r - (y_r - y_l) / h)
    if abs(denominator) < _DBL_MIN:
        r = _MAXIMUM_RATIONAL_CUBIC_CONTROL if numerator > 0 else _MINIMUM_RATIONAL_CUBIC_CONTROL
    else:
        r = numerator / denominator
    # The minimum control parameter for a shape-preserving interpolation (formulas 3.8 and 3.18 of Delbourgo-Gregory).
    slope = (y_r - y_l) / (x_r - x_l)
    monotonic = d_l * slope >= 0 and d_r * slope >= 0
    convex = d_l <= slope <= d_r
    concave = d_l >= slope >= d_r
    if not monotonic and not convex and not concave:
        return max(r, _MINIMUM_RATIONAL_CUBIC_CONTROL)
    r1 = r2 = -_DBL_MAX
    if monotonic:
        if abs(slope) >= _DBL_MIN:
            r1 = (d_r + d_l) / slope
        elif prefer_shape:
            r1 = _MAXIMUM_RATIONAL_CUBIC_CONTROL
    if convex or concave:
        s_m_d_l = slope - d_l
        d_r_m_s = d_r - slope
        if not (abs(s_m_d_l) < _DBL_MIN or abs(d_r_m_s) < _DBL_MIN):
            r2 = max(abs((d_r - d_l) / d_r_m_s), abs((d_r - d_l) / s_m_d_l))
        elif prefer_shape:
            r2 = _MAXIMUM_RATIONAL_CUBIC_CONTROL
    elif monotonic and prefer_shape:
        r2 = _MAXIMUM_RATIONAL_CUBIC_CONTROL
    return max(r, _MINIMUM_RATIONAL_CUBIC_CONTROL, r1, r2)


def _f_lower_map_s(x: float, s: float) -> tuple[float, float, float]:
    """The lowest branch's transformation f(b) (formula 4.36) and its first two derivatives at s."""
    ax = abs(x)
    z = _SQRT_ONE_OVER_THREE * ax / s
    y = z * z
    s2 = s * s
    phi_cdf = 0.5 * math.erfc(_ONE_OVER_SQRT_TWO * z)
    phi_pdf = _ONE_OVER_SQRT_TWO_PI * math.exp(-0.5 * z * z)
    fpp = (
        _PI_OVER_SIX
        * y
        / (s2 * s)
        * phi_cdf
        * (8 * _SQRT_THREE * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * phi_cdf / phi_pdf)
        * math.exp(2 * y + 0.25 * s2)
    )
    phi2 = phi_cdf * phi_cdf
    fp = _TWO_PI * y * phi2 * math.exp(y + 0.125 * s * s)
    f = _TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * (phi2 * phi_cdf)
    return f, fp, fpp


def _inverse_f_lower_map_s(x: float, f: float) -> float:  # formula 4.38
    return abs(
        x
        / (_SQRT_THREE * float(ndtri(_SQRT_THREE_OVER_THIRD_ROOT_TWO_PI * float(np.cbrt(f)) / float(np.cbrt(abs(x))))))
    )


def _f_upper_map_s(x: float, s: float) -> tuple[float, float, float]:
    f = 0.5 * math.erfc(_HALF_OVER_SQRT_TWO * s)
    w = (x / s) * (x / s)
    fp = -0.5 * math.exp(0.5 * w)
    fpp = _SQRT_PI_OVER_TWO * math.exp(w + 0.125 * s * s) * w / s
    return f, fp, fpp


def _step_middle_s(beta: float, x: float, s: float, bmax: float) -> float:
    """Householder(3) step on g(s) = b(s) - β."""
    b = _normalised_black_otm_s(x, s)
    nu = (beta - b) * _inv_vega_s(x, s)
    h = x / s
    x2_over_s3 = (h * h) / s
    h2 = x2_over_s3 - s * 0.25
    h3 = h2 * h2 - 3 * (x2_over_s3 / s) - 0.25
    return nu * _householder3_factor(nu, h2, h3)


def _step_lowest_s(beta: float, x: float, s: float, bmax: float) -> float:
    """Householder step on g(s) = 1/ln b(s) - 1/ln β (fourth order for x < -190)."""
    bx, ln_vega = _scaled_black_and_ln_vega_s(x, s)
    ln_b = math.log(bx) + ln_vega
    ln_beta = math.log(beta)
    bpob = 1 / bx
    h = x / s
    x2_over_s3 = h * h / s
    b_h2 = x2_over_s3 - s / 4
    nu = (ln_beta - ln_b) * ln_b / ln_beta / bpob
    lam = 1 / ln_b
    otl = 1 + 2 * lam
    h2 = b_h2 - bpob * otl
    c = 3 * (x2_over_s3 / s)
    b_h3 = b_h2 * b_h2 - c - 0.25
    sq_bpob = bpob * bpob
    bppob = b_h2 * bpob
    mu = 6 * lam * (1 + lam)
    h3 = b_h3 + sq_bpob * (2 + mu) - bppob * 3 * otl
    if x < -190:
        h4 = (
            (b_h2 * (b_h3 - 0.5) - (b_h2 - 2 / s) * 2 * c)
            - bpob * (sq_bpob * (6 + lam * (22 + lam * (36 + lam * 24))) - bppob * (12 + 6 * mu))
            - bppob * b_h2 * 3 * otl
            - b_h3 * bpob * 4 * otl
        )
        return nu * _householder4_factor(nu, h2, h3, h4)
    return nu * _householder3_factor(nu, h2, h3)


def _step_highest_s(beta: float, x: float, s: float, bmax: float) -> float:
    """Householder step on g(s) = ln((b_max - β)/(b_max - b(s))) (fourth order for x < -580)."""
    h = x / s
    t = s / 2
    # b_max - b(s) = √(π/2)·[erfcx((t+h)/√2) + erfcx((t-h)/√2)]·b'(s), without cancellation; here t + h > 0.
    gp = _TWO_OVER_SQRT_TWO_PI / (_erfcx((t + h) * _ONE_OVER_SQRT_TWO) + _erfcx((t - h) * _ONE_OVER_SQRT_TWO))
    b_bar = _vega_s(x, s) / gp
    g = math.log((bmax - beta) / b_bar)
    x2_over_s3 = (h * h) / s
    b_h2 = x2_over_s3 - s / 4
    c = 3 * (x2_over_s3 / s)
    b_h3 = b_h2 * b_h2 - c - 0.25
    nu = -g / gp
    h2 = b_h2 + gp
    h3 = b_h3 + gp * (2 * gp + 3 * b_h2)
    if x < -580:
        h4 = (b_h2 * (b_h3 - 0.5) - (b_h2 - 2 / s) * 2 * c) + gp * (
            6 * gp * (gp + 2 * b_h2) + 3 * b_h2 * b_h2 + 4 * b_h3
        )
        return nu * _householder4_factor(nu, h2, h3, h4)
    return nu * _householder3_factor(nu, h2, h3)


def _lets_be_rational_s(beta: float, x: float, iterations: int = _ITERATIONS) -> tuple[float, bool]:
    """Normalised implied volatility of an out-of-the-money normalised price ``beta`` at ``x = θ·ln(F/K) ≤ 0``.

    Returns ``(s, converged)``: ``s`` is 0 for ``beta == 0``, NaN below it, ``inf`` at or above ``b_max``. Raises
    ``ArithmeticError`` or ``ValueError`` where the ``math`` module over- or underflows on extreme inputs (the public
    functions then take the array path, :func:`implied_vol.implied_volatility` its Brent fallback).
    """
    if not beta > 0:
        return (0.0, True) if beta == 0 else (math.nan, True)
    bmax = math.exp(0.5 * x)
    if beta >= bmax:
        return math.inf, True
    if x == 0:  # exactly at the forward: b(0, s) = erf(s/√8)
        return 2 * _SQRT_TWO * float(erfinv(beta)), True
    sqrt_ax = math.sqrt(-x)
    s_c = _SQRT_TWO * sqrt_ax
    ome = _one_minus_erfcx_s(sqrt_ax)
    b_c = 0.5 * bmax * ome  # b(x, s_c) = e^(x/2)/2·(1 - erfcx(√|x|))
    step = _step_middle_s
    if beta < b_c:
        s_l = s_c - _SQRT_PI_OVER_TWO * ome
        b_l = _bl_over_bmax(s_c) * bmax
        if beta < b_l:  # lowest branch
            f_l, fp_l, fpp_l = _f_lower_map_s(x, s_l)
            r_ll = _rational_cubic_control_s(0.0, b_l, 0.0, f_l, 1.0, fp_l, fpp_l, True, left=False)
            f = _rational_cubic_interpolation(beta, 0.0, b_l, 0.0, f_l, 1.0, fp_l, r_ll)
            if not f > 0:  # round-off for extreme |x|: quadratic through f(0) = 0, f(b_l) and f'(0) = 1
                t = beta / b_l
                f = (f_l * t + b_l * (1 - t)) * t
            s = _inverse_f_lower_map_s(x, f)
            step = _step_lowest_s
        else:  # lower middle
            inv_v_l = _inv_vega_s(x, s_l)
            inv_v_c = _SQRT_TWO_PI / bmax
            r_lm = _rational_cubic_control_s(b_l, b_c, s_l, s_c, inv_v_l, inv_v_c, 0.0, False, left=False)
            s = _rational_cubic_interpolation(beta, b_l, b_c, s_l, s_c, inv_v_l, inv_v_c, r_lm)
    else:
        s_u = s_c + _SQRT_PI_OVER_TWO * (2 - ome)
        b_u = _bu_over_bmax(s_c) * bmax
        if beta <= b_u:  # upper middle
            inv_v_c = _SQRT_TWO_PI / bmax
            inv_v_u = _inv_vega_s(x, s_u)
            r_um = _rational_cubic_control_s(b_c, b_u, s_c, s_u, inv_v_c, inv_v_u, 0.0, False, left=True)
            s = _rational_cubic_interpolation(beta, b_c, b_u, s_c, s_u, inv_v_c, inv_v_u, r_um)
        else:  # highest branch
            f_u, fp_u, fpp_u = _f_upper_map_s(x, s_u)
            f = -_DBL_MAX
            if -_SQRT_DBL_MAX < fpp_u < _SQRT_DBL_MAX:
                r_uu = _rational_cubic_control_s(b_u, bmax, f_u, 0.0, fp_u, -0.5, fpp_u, True, left=True)
                f = _rational_cubic_interpolation(beta, b_u, bmax, f_u, 0.0, fp_u, -0.5, r_uu)
            if f <= 0:  # quadratic through f(b_u), f(b_max) = 0 and f'(b_max) = -1/2
                h = bmax - b_u
                t = (beta - b_u) / h
                f = (f_u * (1 - t) + 0.5 * h * t) * (1 - t)
            s = -2 * float(ndtri(f))
            if beta > 0.5 * bmax:
                step = _step_highest_s
    ds = -_DBL_MAX
    for _ in range(iterations):
        if not abs(ds) > _EPS * s:
            break
        ds = step(beta, x, s, bmax)
        s += ds
    return s, bool(math.isfinite(s) and s > 0 and abs(ds) <= _CONVERGED_STEP * s)


def _normalised_intrinsic_s(x: float) -> float:  # x = θ·ln(F/K)
    return 2 * math.sinh(0.5 * x) if x > 0 else 0.0


def _normalised_black_s(x: float, s: float, theta: float) -> float:
    if math.isnan(x) or not s >= 0:  # as the array path: no price for a NaN or negative volatility
        return math.nan
    if x == 0:
        return math.erf(_HALF_OVER_SQRT_TWO * s)
    tx = -x if theta < 0 else x
    return _normalised_intrinsic_s(tx) + (_normalised_black_otm_s(-abs(x), s) if s > 0 else 0.0)


# -------------------------------------------------------------------------------------- vector path (NumPy)


def _vega_v(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    h = x / s
    t = 0.5 * s
    return _ONE_OVER_SQRT_TWO_PI * np.exp(-0.5 * (h * h + t * t))


def _inv_vega_v(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    h = x / s
    t = 0.5 * s
    return _SQRT_TWO_PI * np.exp(0.5 * (h * h + t * t))


def _ln_vega_v(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    h = x / s
    t = 0.5 * s
    return -(_LN_TWO_PI * 0.5) - 0.5 * (h * h + t * t)


def _yprime_v(h: np.ndarray) -> np.ndarray:
    out = np.empty_like(h)
    tail = h < -4
    middle = ~tail & (h <= -_CODY_THRESHOLD)
    near = ~tail & ~middle
    out[tail] = _yprime_tail(h[tail])
    out[middle] = _rational(_YPRIME_MIDDLE, h[middle])
    out[near] = 1 + h[near] * _SQRT_PI_OVER_TWO * erfcx(-_ONE_OVER_SQRT_TWO * h[near])
    return out


def _asymptotic_expansion_v(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    e = (t / h) * (t / h)
    r = (h + t) * (h - t)
    q = (h / r) * (h / r)
    first_dropped = np.searchsorted(_ASYMPTOTIC_THRESHOLDS, -h - t + _TAU + 0.5, side="right")
    omega = np.zeros_like(h)
    for n in range(16, 4, -1):
        omega = np.where(first_dropped <= 16 - n, q * (_horner(_A[n], e) + omega), omega)
    omega = _A[0][0] + q * (
        _horner(_A[1], e) + q * (_horner(_A[2], e) + q * (_horner(_A[3], e) + q * (_horner(_A[4], e) + omega)))
    )
    return (t / r) * omega


def _black_region_iii_v(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    h = x / s
    t = 0.5 * s
    q1 = -_ONE_OVER_SQRT_TWO * (h + t)
    q2 = -_ONE_OVER_SQRT_TWO * (h - t)
    two_b = np.full_like(x, np.nan)  # NaN inputs stay NaN
    m = q2 < _CODY_THRESHOLD
    two_b[m] = np.exp(0.5 * x[m]) * erfc(q1[m]) - np.exp(-0.5 * x[m]) * erfc(q2[m])
    m = ~m & (q1 < _CODY_THRESHOLD)
    two_b[m] = np.exp(0.5 * x[m]) * erfc(q1[m]) - np.exp(-0.5 * (h[m] * h[m] + t[m] * t[m])) * erfcx(q2[m])
    m = q1 >= _CODY_THRESHOLD
    two_b[m] = np.exp(-0.5 * (h[m] * h[m] + t[m] * t[m])) * (erfcx(q1[m]) - erfcx(q2[m]))
    return np.maximum(0.5 * two_b, 0.0)


def _scaled_black_v(x: np.ndarray, s: np.ndarray, *, scaled: bool) -> np.ndarray:
    """b(x, s) for x < 0 and s > 0, or b/b' with ``scaled``, region by region."""
    out = np.empty_like(x)
    r1 = _is_region_i(x, s)
    r2 = ~r1 & _is_region_ii(x, s)
    r3 = ~r1 & ~r2
    for region, expansion in ((r1, False), (r2, True)):
        if region.any():
            xr, sr = x[region], s[region]
            h = xr / sr
            bx = _small_t_expansion(h, 0.5 * sr, _yprime_v(h)) if expansion else _asymptotic_expansion_v(h, 0.5 * sr)
            out[region] = bx if scaled else bx * _vega_v(xr, sr)
    if r3.any():
        xr, sr = x[r3], s[r3]
        b = _black_region_iii_v(xr, sr)
        out[r3] = b * np.exp(-_ln_vega_v(xr, sr)) if scaled else b
    return out


def _rational_cubic_interpolation_v(x, x_l, x_r, y_l, y_r, d_l, d_r, r):
    h = x_r - x_l
    t = (x - x_l) / h
    omt = 1 - t
    t2 = t * t
    omt2 = omt * omt
    cubic = (y_r * t2 * t + (r * y_r - h * d_r) * t2 * omt + (r * y_l + h * d_l) * t * omt2 + y_l * omt2 * omt) / (
        1 + (r - 3) * t * omt
    )
    linear = y_r * t + y_l * (1 - t)
    return np.where(np.abs(h) <= 0, 0.5 * (y_l + y_r), np.where(r >= _MAXIMUM_RATIONAL_CUBIC_CONTROL, linear, cubic))


def _rational_cubic_control_v(x_l, x_r, y_l, y_r, d_l, d_r, second_derivative, prefer_shape, *, left):
    h = x_r - x_l
    numerator = 0.5 * h * second_derivative + (d_r - d_l)
    denominator = ((y_r - y_l) / h - d_l) if left else (d_r - (y_r - y_l) / h)
    fallback = np.where(numerator > 0, _MAXIMUM_RATIONAL_CUBIC_CONTROL, _MINIMUM_RATIONAL_CUBIC_CONTROL)
    r = np.where(np.abs(denominator) < _DBL_MIN, fallback, numerator / denominator)
    slope = (y_r - y_l) / (x_r - x_l)
    monotonic = (d_l * slope >= 0) & (d_r * slope >= 0)
    convex = (d_l <= slope) & (slope <= d_r)
    concave = (d_l >= slope) & (slope >= d_r)
    shaped = convex | concave
    maximum = _MAXIMUM_RATIONAL_CUBIC_CONTROL if prefer_shape else -_DBL_MAX
    r1 = np.where(monotonic, np.where(np.abs(slope) >= _DBL_MIN, (d_r + d_l) / slope, maximum), -_DBL_MAX)
    s_m_d_l = slope - d_l
    d_r_m_s = d_r - slope
    degenerate = (np.abs(s_m_d_l) < _DBL_MIN) | (np.abs(d_r_m_s) < _DBL_MIN)
    r2_shaped = np.where(degenerate, maximum, np.maximum(np.abs((d_r - d_l) / d_r_m_s), np.abs((d_r - d_l) / s_m_d_l)))
    r2 = np.where(shaped, r2_shaped, np.where(monotonic, maximum, -_DBL_MAX))
    minimum = np.where(monotonic | shaped, np.maximum(r1, r2), -_DBL_MAX)
    return np.maximum(r, np.maximum(_MINIMUM_RATIONAL_CUBIC_CONTROL, minimum))


def _guess_lowest_v(beta, x, s_l, b_l):
    ax = np.abs(x)
    z = _SQRT_ONE_OVER_THREE * ax / s_l
    y = z * z
    s2 = s_l * s_l
    phi_cdf = 0.5 * erfc(_ONE_OVER_SQRT_TWO * z)
    phi_pdf = _ONE_OVER_SQRT_TWO_PI * np.exp(-0.5 * z * z)
    fpp = (
        _PI_OVER_SIX
        * y
        / (s2 * s_l)
        * phi_cdf
        * (8 * _SQRT_THREE * s_l * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * phi_cdf / phi_pdf)
        * np.exp(2 * y + 0.25 * s2)
    )
    phi2 = phi_cdf * phi_cdf
    fp = _TWO_PI * y * phi2 * np.exp(y + 0.125 * s_l * s_l)
    f_l = _TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * (phi2 * phi_cdf)
    zero, one = np.zeros_like(beta), np.ones_like(beta)
    r_ll = _rational_cubic_control_v(zero, b_l, zero, f_l, one, fp, fpp, True, left=False)
    f = _rational_cubic_interpolation_v(beta, zero, b_l, zero, f_l, one, fp, r_ll)
    t = beta / b_l
    f = np.where(f > 0, f, (f_l * t + b_l * (1 - t)) * t)
    return np.abs(x / (_SQRT_THREE * ndtri(_SQRT_THREE_OVER_THIRD_ROOT_TWO_PI * np.cbrt(f) / np.cbrt(ax))))


def _guess_highest_v(beta, x, s_u, b_u, bmax):
    f_u = 0.5 * erfc(_HALF_OVER_SQRT_TWO * s_u)
    w = (x / s_u) * (x / s_u)
    fp = -0.5 * np.exp(0.5 * w)
    fpp = _SQRT_PI_OVER_TWO * np.exp(w + 0.125 * s_u * s_u) * w / s_u
    zero, half = np.zeros_like(beta), np.full_like(beta, -0.5)
    r_uu = _rational_cubic_control_v(b_u, bmax, f_u, zero, fp, half, fpp, True, left=True)
    f = np.where(
        (fpp > -_SQRT_DBL_MAX) & (fpp < _SQRT_DBL_MAX),
        _rational_cubic_interpolation_v(beta, b_u, bmax, f_u, zero, fp, half, r_uu),
        -_DBL_MAX,
    )
    h = bmax - b_u
    t = (beta - b_u) / h
    f = np.where(f <= 0, (f_u * (1 - t) + 0.5 * h * t) * (1 - t), f)
    return -2 * ndtri(f)


def _step_middle_v(beta, x, s, bmax):
    b = _scaled_black_v(x, s, scaled=False)
    nu = (beta - b) * _inv_vega_v(x, s)
    h = x / s
    x2_over_s3 = (h * h) / s
    h2 = x2_over_s3 - s * 0.25
    h3 = h2 * h2 - 3 * (x2_over_s3 / s) - 0.25
    return nu * _householder3_factor(nu, h2, h3)


def _step_lowest_v(beta, x, s, bmax):
    bx = _scaled_black_v(x, s, scaled=True)
    ln_b = np.log(bx) + _ln_vega_v(x, s)
    h = x / s
    ln_beta = np.log(beta)
    bpob = 1 / bx
    x2_over_s3 = h * h / s
    b_h2 = x2_over_s3 - s / 4
    nu = (ln_beta - ln_b) * ln_b / ln_beta / bpob
    lam = 1 / ln_b
    otl = 1 + 2 * lam
    h2 = b_h2 - bpob * otl
    c = 3 * (x2_over_s3 / s)
    b_h3 = b_h2 * b_h2 - c - 0.25
    sq_bpob = bpob * bpob
    bppob = b_h2 * bpob
    mu = 6 * lam * (1 + lam)
    h3 = b_h3 + sq_bpob * (2 + mu) - bppob * 3 * otl
    h4 = (
        (b_h2 * (b_h3 - 0.5) - (b_h2 - 2 / s) * 2 * c)
        - bpob * (sq_bpob * (6 + lam * (22 + lam * (36 + lam * 24))) - bppob * (12 + 6 * mu))
        - bppob * b_h2 * 3 * otl
        - b_h3 * bpob * 4 * otl
    )
    return nu * np.where(x < -190, _householder4_factor(nu, h2, h3, h4), _householder3_factor(nu, h2, h3))


def _step_highest_v(beta, x, s, bmax):
    h = x / s
    t = s / 2
    gp = _TWO_OVER_SQRT_TWO_PI / (erfcx((t + h) * _ONE_OVER_SQRT_TWO) + erfcx((t - h) * _ONE_OVER_SQRT_TWO))
    b_bar = _vega_v(x, s) / gp
    g = np.log((bmax - beta) / b_bar)
    x2_over_s3 = (h * h) / s
    b_h2 = x2_over_s3 - s / 4
    c = 3 * (x2_over_s3 / s)
    b_h3 = b_h2 * b_h2 - c - 0.25
    nu = -g / gp
    h2 = b_h2 + gp
    h3 = b_h3 + gp * (2 * gp + 3 * b_h2)
    h4 = (b_h2 * (b_h3 - 0.5) - (b_h2 - 2 / s) * 2 * c) + gp * (6 * gp * (gp + 2 * b_h2) + 3 * b_h2 * b_h2 + 4 * b_h3)
    return nu * np.where(x < -580, _householder4_factor(nu, h2, h3, h4), _householder3_factor(nu, h2, h3))


def _lets_be_rational_v(
    beta: np.ndarray, x: np.ndarray, iterations: int = _ITERATIONS
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_lets_be_rational_s` on 1-D arrays: ``(s, converged)``."""
    beta = np.asarray(beta, dtype=float)
    x = np.asarray(x, dtype=float)
    s_out = np.full(beta.shape, np.nan)
    converged = np.ones(beta.shape, dtype=bool)
    with np.errstate(all="ignore"):  # IEEE inf/NaN on extreme inputs end up as non-converged rows
        bmax_all = np.exp(0.5 * x)
        s_out[beta == 0] = 0.0  # also where b_max underflows to 0, as on the scalar path
        above = (beta > 0) & (beta >= bmax_all)
        s_out[above] = np.inf
        live = (beta > 0) & ~above
        atm = live & (x == 0)
        s_out[atm] = 2 * _SQRT_TWO * erfinv(beta[atm])
        rows = np.flatnonzero(live & (x < 0))
        converged[live & ~atm & ~(x < 0)] = False  # x NaN or positive
        if rows.size == 0:
            return s_out, converged
        b, xr, bmax = beta[rows], x[rows], bmax_all[rows]
        sqrt_ax = np.sqrt(-xr)
        s_c = _SQRT_TWO * sqrt_ax
        ome = np.where(
            (sqrt_ax < -1.0 / 5.0) | (sqrt_ax > 1.0 / 3.0), 1 - erfcx(sqrt_ax), _one_minus_erfcx_rational(sqrt_ax)
        )
        b_c = 0.5 * bmax * ome
        s_l = s_c - _SQRT_PI_OVER_TWO * ome
        s_u = s_c + _SQRT_PI_OVER_TWO * (2 - ome)
        b_l = _bl_over_bmax(s_c) * bmax
        b_u = _bu_over_bmax(s_c) * bmax
        inv_v_c = _SQRT_TWO_PI / bmax

        s = np.empty_like(b)
        objective = np.zeros(b.shape, dtype=np.int8)  # 0: b - β, 1: lowest branch, 2: highest branch
        lower = b < b_c
        m = lower & (b < b_l)
        s[m] = _guess_lowest_v(b[m], xr[m], s_l[m], b_l[m])
        objective[m] = 1
        m = lower & (b >= b_l)
        inv_v_l = _inv_vega_v(xr[m], s_l[m])
        r = _rational_cubic_control_v(b_l[m], b_c[m], s_l[m], s_c[m], inv_v_l, inv_v_c[m], 0.0, False, left=False)
        s[m] = _rational_cubic_interpolation_v(b[m], b_l[m], b_c[m], s_l[m], s_c[m], inv_v_l, inv_v_c[m], r)
        m = ~lower & (b <= b_u)
        inv_v_u = _inv_vega_v(xr[m], s_u[m])
        r = _rational_cubic_control_v(b_c[m], b_u[m], s_c[m], s_u[m], inv_v_c[m], inv_v_u, 0.0, False, left=True)
        s[m] = _rational_cubic_interpolation_v(b[m], b_c[m], b_u[m], s_c[m], s_u[m], inv_v_c[m], inv_v_u, r)
        m = ~lower & (b > b_u)
        s[m] = _guess_highest_v(b[m], xr[m], s_u[m], b_u[m], bmax[m])
        objective[m & (b > 0.5 * bmax)] = 2

        ds = np.full_like(b, -_DBL_MAX)
        for kind, step in enumerate((_step_middle_v, _step_lowest_v, _step_highest_v)):
            mine = objective == kind
            for _ in range(iterations):
                go = mine & (np.abs(ds) > _EPS * s)
                if not go.any():
                    break
                ds[go] = step(b[go], xr[go], s[go], bmax[go])
                s[go] += ds[go]
        s_out[rows] = s
        converged[rows] = np.isfinite(s) & (s > 0) & (np.abs(ds) <= _CONVERGED_STEP * s)
    return s_out, converged


def _normalised_black_v(x: np.ndarray, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)  # NaN where x is NaN or s is NaN or negative
    with np.errstate(all="ignore"):
        valid = ~np.isnan(x) & (s >= 0)
        atm = valid & (x == 0)
        out[atm] = erf(_HALF_OVER_SQRT_TWO * s[atm])
        rest = valid & (x != 0)
        tx = np.where(theta < 0, -x, x)[rest]
        intrinsic = np.where(tx > 0, 2 * np.sinh(0.5 * tx), 0.0)
        xr, sr = -np.abs(x[rest]), s[rest]
        time_value = np.zeros_like(xr)
        live = sr > 0
        time_value[live] = _scaled_black_v(xr[live], sr[live], scaled=False)
        out[rest] = intrinsic + time_value
    return out


# ------------------------------------------------------------------------------------------------ public API
# Each function takes the math path when all of its inputs are scalars. The math module raises on overflow and on
# domain errors where NumPy's IEEE arithmetic returns inf or NaN; on such inputs the function recomputes on the
# array path instead, so that the two paths agree on every input (and neither raises).

_MATH_ERRORS = (ArithmeticError, ValueError)


def _all_scalar(*values: Any) -> bool:
    return all(np.ndim(v) == 0 for v in values)


def _floats(*values: Any) -> list[float]:
    return [float(v) for v in values]


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    return list(np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values)))


def _scalar_if_0d(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _log_ratio_s(F: float, K: float) -> float:
    """``ln(F/K)`` to relative accuracy also near the money (within a factor of two ``F - K`` is exact).

    Where ``F/K`` under- or overflows, ``ln F - ln K`` instead.
    """
    ratio = F / K
    if 0.5 <= ratio <= 2:
        return math.log1p((F - K) / K)
    return math.log(ratio) if 0 < ratio < math.inf else math.log(F) - math.log(K)


def _log_ratio_v(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    ratio = F / K
    far = np.where((ratio > 0) & (ratio < np.inf), np.log(ratio), np.log(F) - np.log(K))
    return np.where((ratio >= 0.5) & (ratio <= 2), np.log1p((F - K) / K), far)


def _valid_forward_and_strike(forward: Any, strike: Any) -> Any:
    """Both positive and finite; a bool or a boolean array."""
    return (forward > 0) & (forward < np.inf) & (strike > 0) & (strike < np.inf)


def normalised_black(x: ArrayLike, s: ArrayLike, theta: ArrayLike = 1.0) -> float | np.ndarray:
    """Normalised Black price ``b(x, s, θ) = θ·[e^(x/2)·Φ(θ(x/s + s/2)) - e^(-x/2)·Φ(θ(x/s - s/2))]``.

    Accurate to a few times ``(1 + |s·∂b/∂s / b|)·ε`` relative (the error that rounding ``s`` alone causes), also
    where the naive formula cancels or underflows: far out of the money, very low or very high volatility.

    Parameters
    ----------
    x : array_like
        Log-moneyness ``ln(F/K)``.
    s : array_like
        Total volatility ``σ√T`` (non-negative); ``s = 0`` gives the intrinsic value, ``s = inf`` the maximum
        ``e^(θx/2)``, a negative or NaN ``s`` gives NaN.
    theta : array_like, default 1.0
        ``+1`` for a call, ``-1`` for a put (any negative value counts as a put).

    Returns
    -------
    float or numpy.ndarray
        The undiscounted Black price divided by ``√(FK)``; a float when all inputs are scalars.

    Examples
    --------
    At the forward, ``b(0, s) = erf(s/√8)``:

    >>> normalised_black(0.0, 0.5)
    0.19741265136584743
    >>> round(normalised_black(-0.5, 0.3), 15)
    0.005898232610535
    """
    if _all_scalar(x, s, theta):
        try:
            return _normalised_black_s(*_floats(x, s, theta))
        except _MATH_ERRORS:  # e.g. the intrinsic value 2·sinh(x/2) overflows: inf on the array path
            pass
    xa, sa, ta = _arrays(x, s, theta)
    return _scalar_if_0d(_normalised_black_v(xa.ravel(), sa.ravel(), ta.ravel()).reshape(xa.shape))


def normalised_vega(x: ArrayLike, s: ArrayLike) -> float | np.ndarray:
    """``∂b(x, s, θ)/∂s = exp(-((x/s)² + (s/2)²)/2)/√(2π)``, the same for calls and puts.

    Parameters
    ----------
    x : array_like
        Log-moneyness ``ln(F/K)``.
    s : array_like
        Total volatility ``σ√T``; ``s = 0`` gives 0 (or ``1/√(2π)`` at ``x = 0``), a negative or NaN ``s`` NaN.

    Returns
    -------
    float or numpy.ndarray
        The vega of :func:`normalised_black` per unit of ``s``; a float when all inputs are scalars.

    Examples
    --------
    >>> round(normalised_vega(0.0, 0.0), 15)
    0.398942280401433
    """
    if _all_scalar(x, s):
        x0, s0 = _floats(x, s)
        if math.isnan(x0) or not s0 >= 0:
            return math.nan
        if s0 == 0:
            return _ONE_OVER_SQRT_TWO_PI if x0 == 0 else 0.0
        return _vega_s(x0, s0)
    xa, sa = _arrays(x, s)
    with np.errstate(all="ignore"):
        vega = _vega_v(xa, np.where(sa > 0, sa, 1.0))
        vega = np.where(sa > 0, vega, np.where(xa == 0, _ONE_OVER_SQRT_TWO_PI, 0.0))
        return np.where(np.isnan(xa) | ~(sa >= 0), np.nan, vega)


def _black_s(F: float, K: float, sigma: float, T: float, theta: float) -> float:
    s = sigma * math.sqrt(T) if T >= 0 else math.nan
    if not (_valid_forward_and_strike(F, K) and s >= 0):
        return math.nan
    if K == F:
        return F * math.erf(_HALF_OVER_SQRT_TWO * s)
    intrinsic = max(K - F if theta < 0 else F - K, 0.0)
    if s == 0:
        return intrinsic
    return intrinsic + math.sqrt(F) * math.sqrt(K) * _normalised_black_otm_s(-abs(_log_ratio_s(F, K)), s)


def black(F: ArrayLike, K: ArrayLike, sigma: ArrayLike, T: ArrayLike, theta: ArrayLike = 1.0) -> float | np.ndarray:
    """Undiscounted Black (1976) price of a call (``theta = 1``) or put (``theta = -1``) on a forward.

    Parameters
    ----------
    F, K : array_like
        Forward and strike (positive and finite).
    sigma : array_like
        Volatility per square-root year (decimal, 0.2 = 20 %; non-negative).
    T : array_like
        Time to expiry in years (non-negative).
    theta : array_like, default 1.0
        ``+1`` for a call, ``-1`` for a put.

    Returns
    -------
    float or numpy.ndarray
        ``max(θ(F - K), 0) + √(FK)·b(-|ln(F/K)|, σ√T)``, accurate in relative terms also far out of the money, and
        NaN outside the domain above (or where ``σ√T`` is NaN); a float when all inputs are scalars.

    Examples
    --------
    >>> round(black(100.0, 110.0, 0.2, 1.0), 10)
    4.2920109414
    """
    if _all_scalar(F, K, sigma, T, theta):
        try:
            return _black_s(*_floats(F, K, sigma, T, theta))
        except _MATH_ERRORS:
            pass
    fa, ka, va, tma, tha = _arrays(F, K, sigma, T, theta)
    with np.errstate(all="ignore"):
        sa = va * np.sqrt(tma)
        intrinsic_a = np.maximum(np.where(tha < 0, ka - fa, fa - ka), 0.0)
        xa = -np.abs(_log_ratio_v(fa, ka))
        b = _normalised_black_v(xa.ravel(), sa.ravel(), np.ones(xa.size)).reshape(xa.shape)
        price = np.where(ka == fa, fa * erf(_HALF_OVER_SQRT_TWO * sa), intrinsic_a + np.sqrt(fa) * np.sqrt(ka) * b)
        return _scalar_if_0d(np.where(_valid_forward_and_strike(fa, ka) & (sa >= 0), price, np.nan))


def _normalised_implied_volatility_s(beta: float, x: float, theta: float) -> tuple[float, bool]:
    return _lets_be_rational_s(beta - _normalised_intrinsic_s(-x if theta < 0 else x), -abs(x))


def normalised_implied_volatility(
    beta: ArrayLike, x: ArrayLike, theta: ArrayLike = 1.0, *, full_output: bool = False
) -> Any:
    """Total implied volatility ``s = σ√T`` such that :func:`normalised_black` ``(x, s, θ) = β``.

    Parameters
    ----------
    beta : array_like
        Normalised price ``price/√(FK)`` (undiscounted).
    x : array_like
        Log-moneyness ``ln(F/K)``.
    theta : array_like, default 1.0
        ``+1`` for a call, ``-1`` for a put.
    full_output : bool, default False
        Also return a boolean (array) that is False where the two Householder steps did not converge.

    Returns
    -------
    float or numpy.ndarray, or a tuple ``(s, converged)``
        ``s`` (0 at the intrinsic value, NaN below it, ``inf`` at or above the maximum ``e^(θx/2)``); a float when
        all inputs are scalars.

    Examples
    --------
    >>> beta = normalised_black(-0.5, 0.3)
    >>> round(normalised_implied_volatility(beta, -0.5), 15)
    0.3
    >>> normalised_implied_volatility(0.5, 0.5, 1.0)  # below 2·sinh(0.25) = 0.505, this call's intrinsic value
    nan
    """
    if _all_scalar(beta, x, theta):
        try:
            s, converged = _normalised_implied_volatility_s(*_floats(beta, x, theta))
        except _MATH_ERRORS:
            pass
        else:
            return (s, converged) if full_output else s
    ba, xa, tha = _arrays(beta, x, theta)
    with np.errstate(all="ignore"):
        tx = np.where(tha < 0, -xa, xa)
        otm = ba - np.where(tx > 0, 2 * np.sinh(0.5 * tx), 0.0)
    sa, ca = _lets_be_rational_v(otm.ravel(), -np.abs(xa.ravel()))
    s_out, flags = _scalar_if_0d(sa.reshape(ba.shape)), ca.reshape(ba.shape)
    return (s_out, bool(flags) if flags.ndim == 0 else flags) if full_output else s_out


def _implied_black_volatility_s(price: float, F: float, K: float, T: float, theta: float) -> tuple[float, bool]:
    if not (_valid_forward_and_strike(F, K) and 0 < T < math.inf):
        return math.nan, True
    if price >= (K if theta < 0 else F):
        return math.inf, True
    intrinsic = K - F if theta < 0 else F - K
    beta = (price - intrinsic if intrinsic > 0 else price) / (math.sqrt(F) * math.sqrt(K))
    s, converged = _lets_be_rational_s(beta, -abs(_log_ratio_s(F, K)))
    return s / math.sqrt(T), converged


def implied_black_volatility(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike, theta: ArrayLike = 1.0, *, full_output: bool = False
) -> Any:
    """Black (1976) implied volatility of an undiscounted option price on a forward.

    Parameters
    ----------
    price : array_like
        Undiscounted option price (the premium divided by the discount factor).
    F, K : array_like
        Forward and strike (positive and finite).
    T : array_like
        Time to expiry in years (positive and finite).
    theta : array_like, default 1.0
        ``+1`` for a call, ``-1`` for a put.
    full_output : bool, default False
        Also return a boolean (array) that is False where the two Householder steps did not converge.

    Returns
    -------
    float or numpy.ndarray, or a tuple ``(sigma, converged)``
        Volatility per square-root year: 0 at the intrinsic value, NaN below it or outside the domain above,
        ``inf`` at or above the maximum price (``F`` for a call, ``K`` for a put); a float when all inputs are
        scalars.

    Examples
    --------
    >>> price = black(100.0, 110.0, 0.2, 1.0)
    >>> round(implied_black_volatility(price, 100.0, 110.0, 1.0), 15)
    0.2
    """
    if _all_scalar(price, F, K, T, theta):
        try:
            vol, converged = _implied_black_volatility_s(*_floats(price, F, K, T, theta))
        except _MATH_ERRORS:
            pass
        else:
            return (vol, converged) if full_output else vol
    pa, fa, ka, ta, tha = _arrays(price, F, K, T, theta)
    with np.errstate(all="ignore"):
        valid = _valid_forward_and_strike(fa, ka) & (ta > 0) & (ta < np.inf)
        intrinsic_a = np.where(tha < 0, ka - fa, fa - ka)
        beta_a = np.where(intrinsic_a > 0, pa - intrinsic_a, pa) / (np.sqrt(fa) * np.sqrt(ka))
        sa, ca = _lets_be_rational_v(beta_a.ravel(), -np.abs(_log_ratio_v(fa, ka)).ravel())
        at_maximum = pa >= np.where(tha < 0, ka, fa)
        sa = np.where(at_maximum, np.inf, sa.reshape(pa.shape))
        vol_a = _scalar_if_0d(np.where(valid, sa / np.sqrt(ta), np.nan))
    ca = ca.reshape(pa.shape) | at_maximum | ~valid  # as the scalar path: nothing to converge
    return (vol_a, bool(ca) if ca.ndim == 0 else ca) if full_output else vol_a
