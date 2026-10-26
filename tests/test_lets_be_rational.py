"""Let's Be Rational (options_analyzer.lets_be_rational) against 60-digit mpmath.

The accuracy study draws total volatilities s = σ√T in [1e-4, 5], log-moneyness |x| = |ln F/K| up to 40·s, calls and
puts, prices the options with mpmath at 60 significant digits, rounds the prices to doubles (keeping those of at
least 1e-300) and inverts them. Rounding the price already moves the volatility by up to ``cond·ε/2``, where
``cond = 1 + |b/(s·∂b/∂s)|`` is the condition number of the inversion (Jäckel's attainable accuracy is
``cond·ε``), so the requirement is a relative error of at most ``max(1e-14, 4·cond·ε)``.

Where an in-the-money option's time value is below two units in the last place of its price, rounding the price
loses the time value altogether and no volatility is recoverable (the linearised ``cond`` understates this). There
the inversion must be backward stable instead: the returned volatility reprices the quote to within 4 units in the
last place, or is NaN when the rounded quote is already below the rounded intrinsic value.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from options_analyzer import lets_be_rational as lbr

mp = pytest.importorskip("mpmath")

EPS = np.finfo(float).eps


def _exact_black(x: float, s: float, theta: float, dps: int = 60):
    """b(x, s, θ) and ∂b/∂s with mpmath (s = 0 gives the intrinsic value, s = inf the maximum e^(θx/2))."""
    with mp.workdps(dps):
        x, theta = mp.mpf(x), int(theta)
        if s == 0:
            return max(theta * (mp.exp(x / 2) - mp.exp(-x / 2)), mp.mpf(0)), mp.mpf(0)
        if s == math.inf:
            return mp.exp(theta * x / 2), mp.mpf(0)
        s = mp.mpf(s)
        h, t = x / s, s / 2

        def ncdf(z):
            return mp.erfc(-z / mp.sqrt(2)) / 2

        b = theta * (mp.exp(x / 2) * ncdf(theta * (h + t)) - mp.exp(-x / 2) * ncdf(theta * (h - t)))
        return b, mp.exp(-(h * h + t * t) / 2) / mp.sqrt(2 * mp.pi)


def _study_cases(n: int, seed: int):
    """x, s, θ for the study: |x|/s uniform on [0, 40] (a quarter log-uniform down to 1e-8), 3 % at x = 0."""
    rng = np.random.default_rng(seed)
    s = 10 ** rng.uniform(-4, np.log10(5), n)
    ratio = np.where(rng.random(n) < 0.75, rng.uniform(0, 40, n), 10 ** rng.uniform(-8, np.log10(40), n))
    x = rng.choice([-1.0, 1.0], n) * ratio * s
    x[rng.random(n) < 0.03] = 0.0
    return x, s, rng.choice([-1.0, 1.0], n)


def _study(n: int, seed: int) -> dict:
    """Invert n exactly priced quotes with the array path; return the measurements and the failures."""
    x, s, theta = _study_cases(n, seed)
    beta, cond, time_value_ulps = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        b, vega = _exact_black(x[i], s[i], theta[i])
        beta[i] = float(b)
        cond[i] = min(float(1 + abs(b / (s[i] * vega))), 1e300) if b > 0 else np.inf
        intrinsic, _ = _exact_black(x[i], 0, theta[i])
        time_value_ulps[i] = float((b - intrinsic) / (EPS * b)) if b > 0 else 0.0
    keep = beta >= 1e-300
    x, s, theta, beta, cond, time_value_ulps = (a[keep] for a in (x, s, theta, beta, cond, time_value_ulps))
    vol, converged = lbr.normalised_implied_volatility(beta, x, theta, full_output=True)
    error = np.abs(vol - s) / s
    identifiable = time_value_ulps >= 2
    tolerance = np.maximum(1e-14, 4 * cond * EPS)
    failures = [
        f"x={x[i]!r} s={s[i]!r} theta={theta[i]:+.0f}: error {error[i]:.3g} > {tolerance[i]:.3g}"
        for i in np.flatnonzero(identifiable & ~(error <= tolerance))
    ]
    for i in np.flatnonzero(~identifiable):  # backward stability where the time value was lost in rounding
        if np.isnan(vol[i]):
            continue
        repriced, _ = _exact_black(x[i], float(vol[i]), theta[i])
        if not abs(float(repriced) - beta[i]) <= 4 * EPS * beta[i]:
            failures.append(f"x={x[i]!r} s={s[i]!r} theta={theta[i]:+.0f}: {vol[i]!r} reprices to {float(repriced)!r}")
    return {
        "x": x,
        "s": s,
        "theta": theta,
        "beta": beta,
        "cond": cond,
        "vol": vol,
        "converged": converged,
        "identifiable": identifiable,
        "error": error,
        "worst_over_tolerance": float(np.max((error / tolerance)[identifiable])),
        "failures": failures,
    }


def _assert_study(result: dict) -> None:
    n = len(result["x"])
    assert not result["failures"], f"{len(result['failures'])} of {n} cases fail, e.g. " + "; ".join(
        result["failures"][:5]
    )
    assert result["converged"].all(), f"{(~result['converged']).sum()} cases report no convergence"
    assert result["worst_over_tolerance"] < 1  # recorded in the message of the slow test


@pytest.mark.oracle
def test_accuracy_study_fast_subset():
    result = _study(2_000, seed=20240216)
    _assert_study(result)
    well_conditioned = result["identifiable"] & (result["cond"] < 10)
    assert well_conditioned.sum() > 1_000
    assert result["error"][well_conditioned].max() <= 4e-15


@pytest.mark.oracle
@pytest.mark.slow
def test_accuracy_study_100k():
    # 100,000 cases (about 35 s of mpmath), of which 97,086 have prices of at least 1e-300 and 66,738 keep their
    # time value. Measured: worst error 0.27 of the tolerance; at most 1.54e-15 where cond < 10 (60,088 quotes);
    # every quote converged in the two Householder steps.
    result = _study(100_000, seed=1)
    _assert_study(result)
    assert result["identifiable"].sum() > 60_000


@pytest.mark.oracle
def test_scalar_path_agrees_with_the_array_path():
    result = _study(1_500, seed=7)
    rows = np.flatnonzero(result["identifiable"])
    scalar = np.array(
        [lbr.normalised_implied_volatility(result["beta"][i], result["x"][i], result["theta"][i]) for i in rows]
    )
    difference = np.abs(scalar - result["vol"][rows]) / result["s"][rows]
    assert np.all(difference <= 4 * result["cond"][rows] * EPS)
    assert np.mean(scalar == result["vol"][rows]) > 0.5  # most agree to the last bit


@pytest.mark.oracle
def test_normalised_black_is_accurate_to_its_condition_number():
    # The attainable relative accuracy of b is (1 + |s·b'/b|)·ε; SciPy's erfcx (up to ~3.7 ulp) costs a little.
    x, s, theta = _study_cases(3_000, seed=11)
    worst = 0.0
    for scalar_path in (False, True):
        values = (
            np.array([lbr.normalised_black(*args) for args in zip(x, s, theta, strict=True)])
            if scalar_path
            else lbr.normalised_black(x, s, theta)
        )
        for i in range(len(x)):
            b, vega = _exact_black(x[i], s[i], theta[i])
            if b < 1e-300:
                continue
            attainable = float(1 + abs(s[i] * vega / b)) * EPS
            worst = max(worst, abs(values[i] / float(b) - 1) / attainable)
    assert worst <= 5  # measured 3.4 over the 100,000 cases of the study


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("x", "s", "where"),
    [
        (-20.0, 1.0, "region I (asymptotic expansion), lowest branch"),
        (-30.0, 1.0, "region I far from its edge (8 terms; fewer need prices below DBL_MIN)"),
        (-0.5, 0.1, "region II (small-t expansion), lowest branch"),
        (-6.0, 0.6, "region II near h = -13 where Y'(h) needs its Remez form"),
        (-0.5, 1.0, "region III (erfc/erfcx), upper middle branch"),
        (-2.0, 0.3, "lowest branch, |x| in b_l's branch II"),
        (-2.0, 1.8, "lower middle branch"),
        (-2.0, 2.5, "upper middle branch"),
        (-2.0, 6.0, "highest branch, objective ln((b_max - β)/(b_max - b))"),
        (-0.01, 3.0, "highest branch near the money"),
        (-30.0, 5.0, "lowest branch, |x| in b_l's branch IV"),
        (-250.0, 30.0, "lowest branch with the fourth-order step (x < -190)"),
        (-650.0, 30.0, "lowest branch at the edge of double precision"),
        (-0.3, 10.0, "highest branch, price within 1e-3 of the maximum"),
        (0.0, 1e-8, "at the forward: s = √8·erfinv(β)"),
        (0.0, 4.0, "at the forward, high volatility"),
    ],
)
@pytest.mark.parametrize("theta", [1.0, -1.0])
def test_each_branch_and_region_round_trips(x, s, where, theta):
    b, vega = _exact_black(-x if theta < 0 else x, s, theta)  # out of the money for both calls and puts
    beta = float(b)
    cond = float(1 + abs(b / (s * vega)))
    x_signed = -x if theta < 0 else x
    tolerance = max(1e-14, 4 * cond * EPS)
    scalar = lbr.normalised_implied_volatility(beta, x_signed, theta)
    vector = lbr.normalised_implied_volatility(np.array([beta]), np.array([x_signed]), theta)[0]
    assert abs(scalar - s) <= tolerance * s, where
    assert abs(vector - s) <= tolerance * s, where
    assert lbr.normalised_black(x_signed, s, theta) == pytest.approx(beta, rel=8 * EPS * (1 + float(s * vega / b)))


def test_asymptotic_coefficients_are_the_references_literals():
    # A spot check of the closed form against literals copied from revision 1520 (A1, A2, A9 and A16 in full).
    assert lbr._A[0] == (2.0,)
    assert lbr._A[1] == (-6.0, -2.0)
    assert lbr._A[2] == (30.0, 60.0, 6.0)
    assert lbr._A[9] == (
        -1.30945815e9,
        -6.678236565e10,
        -8.013883878e11,
        -3.4726830138e12,
        -6.3665855253e12,
        -5.2090245207e12,
        -1.8699062382e12,
        -2.671294626e11,
        -1.178512335e10,
        -6.891885e7,
    )
    assert lbr._A[16][0] == 1.26653197415257012e19
    assert lbr._A[16][8] == 4.47816195865351041e26
    assert lbr._A[16][-1] == 3.8379756792502125e17
    assert [len(a) for a in lbr._A] == list(range(1, 18))


@pytest.mark.oracle
def test_rational_approximations_match_their_functions():
    # The Remez tables of revision 1520 against the functions they approximate (60-digit mpmath); this checks the
    # transcription of every coefficient and branch edge.
    with mp.workdps(60):

        def yprime(h):  # 1 + h·Φ(h)/φ(h)
            h = mp.mpf(h)
            return 1 + h * mp.erfc(-h / mp.sqrt(2)) / 2 / (mp.exp(-h * h / 2) / mp.sqrt(2 * mp.pi))

        for h in np.concatenate([-np.geomspace(4, 1e4, 300), -np.linspace(0.46875, 4, 300)]):
            approx = lbr._yprime_tail(h) if h < -4 else lbr._rational(lbr._YPRIME_MIDDLE, h)
            assert approx == pytest.approx(float(yprime(h)), rel=4 * EPS), h
        for v in np.linspace(-0.2, 1 / 3, 301):
            exact = 1 - mp.exp(mp.mpf(v) ** 2) * mp.erfc(mp.mpf(v))
            assert lbr._one_minus_erfcx_rational(v) == pytest.approx(float(exact), rel=4 * EPS, abs=1e-300)
        # b_l = b(x, s_l) and b_u = b(x, s_u) over b_max = e^(x/2), with s_l, s_u = √(2|x|) ∓ √(π/2)·(1 ∓ erfcx(√|x|)).
        for x in -np.concatenate([np.geomspace(1e-6, 700, 400), [0.252, 0.3, 1.6, 3.45, 19, 27]]):
            ax = mp.mpf(-x)
            s_c = mp.sqrt(2 * ax)
            erfcx = mp.exp(ax) * mp.erfc(mp.sqrt(ax))
            for s_edge, approx in (
                (s_c - mp.sqrt(mp.pi / 2) * (1 - erfcx), lbr._bl_over_bmax(float(s_c))),
                (s_c + mp.sqrt(mp.pi / 2) * (1 + erfcx), lbr._bu_over_bmax(float(s_c))),
            ):
                exact, _ = _exact_black(x, float(s_edge), 1.0)
                # The reference documents better than 1e-16 for the rational functions in exact arithmetic;
                # evaluating them in doubles and rounding s_edge leave a few ulps.
                assert approx == pytest.approx(float(exact / mp.exp(-ax / 2)), rel=1e-14), x


def test_special_values_on_both_paths():
    cases = [
        # (beta, x, theta, expected s)
        (0.0, -0.3, 1.0, 0.0),  # out of the money at zero price
        (-1e-12, -0.3, 1.0, math.nan),  # below the (zero) intrinsic value
        (math.exp(-0.15), -0.3, 1.0, math.inf),  # at the maximum b_max = e^(x/2)
        (2.0, -0.3, 1.0, math.inf),
        (2 * math.sinh(0.15), 0.3, 1.0, 0.0),  # an in-the-money call at its intrinsic value
        (0.1, 0.3, 1.0, math.nan),  # below the intrinsic value 0.3045
        (math.nan, -0.3, 1.0, math.nan),
    ]
    beta, x, theta, expected = (np.array(column) for column in zip(*cases, strict=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vector = lbr.normalised_implied_volatility(beta, x, theta)
        scalar = [lbr.normalised_implied_volatility(*case[:3]) for case in cases]
    np.testing.assert_array_equal(vector, expected)
    np.testing.assert_array_equal(np.array(scalar), expected)


@pytest.mark.oracle
@pytest.mark.parametrize("beta", [1e-300, 1e-100, 1e-16, 1e-3, 0.5, 0.99, 1 - 1e-12])
def test_at_the_forward_the_inverse_is_exact(beta):
    with mp.workdps(60):
        exact = float(2 * mp.sqrt(2) * mp.erfinv(mp.mpf(beta)))
    assert lbr.normalised_implied_volatility(beta, 0.0) == pytest.approx(exact, rel=4 * EPS)
    assert lbr.normalised_implied_volatility(np.array([beta]), 0.0)[0] == pytest.approx(exact, rel=4 * EPS)


def test_black_limits_and_parity():
    assert lbr.normalised_black(0.4, 0.0, 1.0) == pytest.approx(2 * math.sinh(0.2), rel=1e-15)  # intrinsic
    assert lbr.normalised_black(0.4, 0.0, -1.0) == 0.0
    assert lbr.normalised_black(-0.4, 0.0, 1.0) == 0.0
    assert np.isnan(lbr.normalised_black(math.nan, 0.2))
    assert lbr.normalised_vega(0.0, 0.0) == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-15)
    assert lbr.normalised_vega(-0.4, 0.0) == 0.0
    # Put-call parity b(x, s, +1) - b(x, s, -1) = 2·sinh(x/2); reciprocal-strike symmetry b(x, s, θ) = b(-x, s, -θ).
    x = np.linspace(-3, 3, 61)
    s = np.full_like(x, 0.7)
    calls, puts = lbr.normalised_black(x, s, 1.0), lbr.normalised_black(x, s, -1.0)
    assert np.all(np.abs(calls - puts - 2 * np.sinh(x / 2)) <= 4 * EPS * np.maximum(calls, 1))
    np.testing.assert_array_equal(calls, lbr.normalised_black(-x, s, -1.0))


@pytest.mark.oracle
def test_black_price_and_its_inverse():
    F, K, T = 100.0, np.array([20.0, 80.0, 100.0, 125.0, 400.0]), 0.75
    for theta in (1.0, -1.0):
        price = lbr.black(F, K, 0.3, T, theta)
        assert isinstance(lbr.black(F, 125.0, 0.3, T, theta), float)
        for k, value in zip(K, price, strict=True):
            exact, _ = _exact_black(math.log(F / k), 0.3 * math.sqrt(T), theta)
            assert value == pytest.approx(float(exact * mp.sqrt(F * k)), rel=1e-13)
            assert lbr.black(F, k, 0.3, T, theta) == pytest.approx(value, rel=4 * EPS)
        vol = lbr.implied_black_volatility(price, F, K, T, theta)
        np.testing.assert_allclose(lbr.black(F, K, vol, T, theta), price, rtol=4 * EPS)  # backward stable
        out_of_the_money = theta * (F - K) <= 0  # deep in the money the time value is below the price's rounding
        np.testing.assert_allclose(vol[out_of_the_money], 0.3, rtol=1e-14)
        assert lbr.implied_black_volatility(float(price[1]), F, 80.0, T, theta) == pytest.approx(0.3, rel=1e-13)
    assert lbr.implied_black_volatility(100.0, 100.0, 90.0, 1.0, 1.0) == math.inf  # at the maximum F
    assert np.isnan(lbr.implied_black_volatility(5.0, 100.0, 90.0, 1.0, 1.0))  # below the intrinsic 10
    assert np.isnan(lbr.implied_black_volatility(5.0, 100.0, 100.0, 0.0, 1.0))  # no time to expiry
    vol, converged = lbr.implied_black_volatility(np.array([4.0, 100.0]), 100.0, 100.0, 1.0, full_output=True)
    assert converged.all() and vol[1] == math.inf


def test_shapes_and_types():
    beta = lbr.normalised_black(np.array([[-0.5], [-0.1]]), np.array([0.1, 0.2, 0.3]))
    assert beta.shape == (2, 3)
    vol = lbr.normalised_implied_volatility(beta, np.array([[-0.5], [-0.1]]))
    np.testing.assert_allclose(vol, np.broadcast_to([0.1, 0.2, 0.3], (2, 3)), rtol=1e-14)
    assert isinstance(lbr.normalised_black(np.float64(-0.5), 1), float)
    assert isinstance(lbr.normalised_implied_volatility(0.1, -0.5), float)
    s, converged = lbr.normalised_implied_volatility(0.1, -0.5, full_output=True)
    assert isinstance(s, float) and converged is True


def test_no_row_is_flagged_on_the_documented_domain():
    # The module promises that the convergence flag only trips beyond |x| ≈ 200: here |x| ≤ 80·s with s ≤ 20.
    rng = np.random.default_rng(5)
    s = 10 ** rng.uniform(-4, np.log10(20), 50_000)
    x = -rng.uniform(0, 80, s.size) * s
    beta = lbr.normalised_black(x, s)
    valid = (beta > 1e-300) & (beta < np.exp(x / 2))
    beta, x, s = beta[valid], x[valid], s[valid]
    vol, converged = lbr.normalised_implied_volatility(beta, x, full_output=True)
    assert converged.all()
    ln_vega = -0.5 * np.log(2 * np.pi) - 0.5 * ((x / s) ** 2 + (s / 2) ** 2)  # without underflow
    with np.errstate(over="ignore"):
        cond = 1 + np.exp(np.log(beta) - np.log(s) - ln_vega)
    assert np.all(np.abs(vol - s) <= np.maximum(1e-14, 8 * cond * EPS) * s)


def test_the_flag_trips_when_the_steps_are_cut_short():
    beta = lbr.normalised_black(-0.5, 0.3)
    assert lbr._lets_be_rational_s(beta, -0.5, iterations=0)[1] is False
    assert not lbr._lets_be_rational_v(np.array([beta]), np.array([-0.5]), iterations=0)[1][0]
    s, converged = lbr._lets_be_rational_s(beta, -0.5, iterations=2)
    assert converged and s == pytest.approx(0.3, rel=4 * EPS)


def test_extreme_inputs_raise_no_warnings():
    x = np.array([-700.0, -650.0, -1e-300, -1e-12, 700.0, -0.5, -0.5, -0.5])
    s = np.array([30.0, 1.0, 1e-4, 50.0, 30.0, 1e-300, 1e300, math.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        beta = lbr.normalised_black(x, s)
        vol = lbr.normalised_implied_volatility(beta, x)
        for args in zip(beta, x, strict=True):
            lbr.normalised_implied_volatility(*args)
        lbr.normalised_implied_volatility(np.full(3, 1e-310), np.array([-0.1, -1.0, -10.0]))
    assert np.all(np.isfinite(vol) | np.isnan(vol) | np.isinf(vol))
