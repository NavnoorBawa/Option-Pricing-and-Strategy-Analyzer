"""options_analyzer.lets_be_rational against py_lets_be_rational, an independent port of Jäckel's reference code.

py_lets_be_rational (MIT, pinned in requirements-oracle.txt) is a pure-Python port of the 2013 revision of "Let's Be
Rational". Revision 1520, which options_analyzer ports, moved the boundary of the asymptotic expansion from h = -10
to -13, evaluates 1 + h·Y(h) in the Taylor expansion without cancellation, and replaces the Black evaluations at
the branch edges b_l and b_u with rational approximations. So the two must agree on implied volatilities to the
attainable accuracy and on the four-branch initial guess to rounding, and the newer Black function must be at
least as accurate (judged against 30-digit mpmath).
"""

from __future__ import annotations

import numpy as np
import pytest

from options_analyzer import lets_be_rational as lbr

plbr = pytest.importorskip("py_lets_be_rational")
mp = pytest.importorskip("mpmath")

pytestmark = pytest.mark.oracle

EPS = np.finfo(float).eps


def _exact(x, s, theta: float):
    """b(x, s, θ), its normalised vega and the time value in units of b's last place, at 30 digits."""
    with mp.workdps(30):
        x, s = mp.mpf(x), mp.mpf(s)
        h, t = x / s, s / 2

        def ncdf(z):
            return mp.erfc(-z / mp.sqrt(2)) / 2

        b = theta * (mp.exp(x / 2) * ncdf(theta * (h + t)) - mp.exp(-x / 2) * ncdf(theta * (h - t)))
        vega = mp.exp(-(h * h + t * t) / 2) / mp.sqrt(2 * mp.pi)
        intrinsic = max(theta * (mp.exp(x / 2) - mp.exp(-x / 2)), 0)
        return b, vega, (b - intrinsic) / (EPS * b) if b > 0 else 0


def _cases(n: int, seed: int, *, ratio_max: float = 40.0):
    """Out-of-the-money and in-the-money calls and puts: s in [1e-4, 5], |x| up to ratio_max·s, prices ≥ 1e-300."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        s = 10 ** rng.uniform(-4, np.log10(5))
        x = rng.choice([-1.0, 1.0]) * rng.uniform(0, ratio_max) * s
        theta = rng.choice([-1.0, 1.0])
        b, vega, time_value_ulps = _exact(x, s, theta)
        if b < 1e-300 or time_value_ulps < 1000:  # in the money, keep only quotes whose time value survives rounding
            continue
        rows.append((x, s, theta, float(b), float(1 + abs(b / (s * vega)))))
    return [np.array(column) for column in zip(*rows, strict=True)]


def _py_implied(beta: float, x: float, theta: float) -> float:
    return plbr.normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, theta)


def _assert_agreement(ours, theirs, truth, cond):
    """Ours within max(1e-14, 4·cond·ε) of the true volatility, and the two within max(2e-15, 12·cond·ε) of each other.

    Each implementation's error reaches a few ε times cond, so they may differ by about the sum: a flat 1e-15 is not
    attainable for two double-precision implementations once cond exceeds about 1.1. In the money
    py_lets_be_rational's own error reaches about 7.5·cond·ε (it computes e^(x/2) - e^(-x/2) where revision 1520 uses
    2·sinh(x/2)), and the platform's libm moves it by a little more, so the agreement bound is 12·cond·ε; our accuracy
    is held to 4·cond·ε against the 60-digit truth. Returns the relative differences.
    """
    assert np.all(np.abs(ours - truth) <= np.maximum(1e-14, 4 * cond * EPS) * truth)
    difference = np.abs(ours - theirs) / truth
    assert np.all(difference <= np.maximum(2e-15, 12 * cond * EPS)), np.max(difference / (cond * EPS))
    return difference


def test_implied_volatilities_agree():
    x, s, theta, beta, cond = _cases(3_000, seed=1520)
    ours = lbr.normalised_implied_volatility(beta, x, theta)
    theirs = np.array([_py_implied(*args) for args in zip(beta, x, theta, strict=True)])
    difference = _assert_agreement(ours, theirs, s, cond)
    out_of_the_money = theta * x <= 0
    # Out of the money (2,537 quotes) every one agrees within 1e-15. Measured: at most 7.8e-16 apart (8.9e-16 with
    # NumPy 1.26/SciPy 1.11).
    assert np.all(difference[out_of_the_money] <= 1e-15), np.max(difference[out_of_the_money])
    # In the money they part where 0.11 < |x| < 0.2: py_lets_be_rational subtracts the intrinsic value as
    # e^(x/2) - 1/e^(x/2), which revision 1520 replaced by 2·sinh(x/2) ("incurs a larger roundoff error"). Measured:
    # 12 of the 463 in-the-money quotes (13 with NumPy 1.26/SciPy 1.11) beyond max(1e-15, 4·cond·ε) apart; on each
    # py_lets_be_rational's error is 3.9 to 7.5·cond·ε and ours at most 0.7·cond·ε.
    in_the_money = ~out_of_the_money
    apart = in_the_money & (difference > np.maximum(1e-15, 4 * cond * EPS))
    assert apart.sum() <= 0.05 * in_the_money.sum()
    assert np.all(np.abs(ours - s)[apart] < np.abs(theirs - s)[apart])  # where they part, theirs is the one off


def test_price_level_implied_volatilities_agree():
    rng = np.random.default_rng(3)
    F, n = 100.0, 400
    K = F * np.exp(rng.uniform(-1.5, 1.5, n))
    T = 10 ** rng.uniform(-2, 1, n)
    sigma = 10 ** rng.uniform(-1.5, 0.3, n)
    for theta in (1.0, -1.0):
        rows = []
        for i in range(n):
            with mp.workdps(30):  # the price of exactly these F, K, σ and T (not of a rounded ln(F/K) or σ√T)
                x, s = mp.log(mp.mpf(F) / mp.mpf(K[i])), mp.mpf(sigma[i]) * mp.sqrt(mp.mpf(T[i]))
                b, vega, time_value_ulps = _exact(x, s, theta)
                price = float(b * mp.sqrt(mp.mpf(F) * mp.mpf(K[i])))
                cond = float(1 + abs(b / (s * vega)))
            if price < 1e-300 or time_value_ulps < 1000:
                continue
            ours = lbr.implied_black_volatility(price, F, K[i], T[i], theta)
            theirs = plbr.implied_volatility_from_a_transformed_rational_guess(price, F, K[i], T[i], theta)
            rows.append((ours, theirs, sigma[i], cond))
        ours, theirs, truth, cond = (np.array(column) for column in zip(*rows, strict=True))
        assert len(rows) > 250
        difference = _assert_agreement(ours, theirs, truth, cond)
        # Measured: 97.6 to 99.3 % within 1e-15 and at most 2.4e-15 apart (NumPy 1.26/SciPy 1.11 to 2.4/1.17).
        assert np.mean(difference <= 1e-15) >= 0.95


def test_four_branch_initial_guesses_agree():
    # Zero iterations returns the initial guess of the branch the quote falls in: the same up to the newer rational
    # approximations of the branch edges b_l and b_u (measured 3e-14 relative).
    x, _, theta, beta, _ = _cases(3_000, seed=7)
    x_otm = -np.abs(x)
    otm = theta * x <= 0  # the guess is for the out-of-the-money price
    worst = 0.0
    for i in np.flatnonzero(otm):
        ours, converged = lbr._lets_be_rational_s(beta[i], x_otm[i], iterations=0)
        theirs = plbr.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta[i], x_otm[i], 1.0, 0
        )
        assert not converged  # no step was taken
        worst = max(worst, abs(ours - theirs) / theirs)
    vector, _ = lbr._lets_be_rational_v(beta[otm], x_otm[otm], iterations=0)
    scalar = np.array(
        [lbr._lets_be_rational_s(b, v, iterations=0)[0] for b, v in zip(beta[otm], x_otm[otm], strict=True)]
    )
    assert worst <= 1e-12
    np.testing.assert_allclose(vector, scalar, rtol=1e-14)


def test_normalised_black_is_at_least_as_accurate():
    x, s, theta, _, _ = _cases(2_000, seed=11)
    ours = lbr.normalised_black(x, s, theta)
    theirs = np.array([plbr.normalised_black(*args) for args in zip(x, s, theta, strict=True)])
    worst_ours = worst_theirs = 0.0
    for i in range(len(x)):
        b, vega, _ = _exact(x[i], s[i], theta[i])
        attainable = float(1 + abs(s[i] * vega / b)) * EPS
        assert abs(ours[i] - theirs[i]) <= 16 * attainable * float(b)  # their own error reaches 9 times attainable
        worst_ours = max(worst_ours, float(abs(ours[i] / b - 1)) / attainable)
        worst_theirs = max(worst_theirs, float(abs(theirs[i] / b - 1)) / attainable)
    assert worst_ours <= worst_theirs
    assert worst_ours <= 5
