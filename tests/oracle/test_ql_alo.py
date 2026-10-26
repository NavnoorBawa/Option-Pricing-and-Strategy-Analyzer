"""options_analyzer.american against QuantLib (BSD-3-Clause, pinned in requirements-oracle.txt).

* The committed reference grid is what QuantLib computes: rows of tests/fixtures/alo_grid_qdfp.npz repriced with
  the generator's own code (scripts/gen_alo_reference.py) agree to 1e-12.
* Calls and puts well outside the grid (expiries from a day to 30 years, volatilities from 2 % to 200 %, negative
  dividend yields, zero rates) against QdFp at strict settings, which converge to the same equation.
* The QD+ initial guess reproduces QuantLib's QdPlus engine (Li 2010), which checks its expanded equation.
* The regimes are QuantLib's: QdFp returns the European value where early exercise is never optimal and refuses the
  double-boundary put, whose fallback value matches QuantLib's finite-difference engine on a 4000 x 4000 grid.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from options_analyzer import american as am
from options_analyzer import black_scholes as bs

ql = pytest.importorskip("QuantLib")

pytestmark = pytest.mark.oracle

ROOT = Path(__file__).resolve().parents[2]
TODAY = ql.Date(2, 1, 2023)


def _generator():
    spec = importlib.util.spec_from_file_location("gen_alo_reference", ROOT / "scripts" / "gen_alo_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option(kind: str, S: float, K: float, days: int, r: float, q: float, sigma: float, engine):
    """A QuantLib American option expiring in ``days`` days on an Actual/365 count (T = days / 365 exactly)."""
    ql.Settings.instance().evaluationDate = TODAY
    day_count = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, day_count)),
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.AmericanExercise(TODAY, TODAY + days))
    option.setPricingEngine(engine(process))
    return option


def _strict(process):
    return ql.QdFpAmericanEngine(process, ql.QdFpTanhSinhIterationScheme(24, 64, 1e-12))


def test_the_fixture_matches_quantlib():
    generator = _generator()
    data = generator.read()
    assert str(data["quantlib_version"]) == ql.__version__
    engines = generator.schemes()
    rows = np.random.default_rng(2016).choice(data["spot"].size, size=12, replace=False)
    for i in rows:
        months = round(float(data["maturity"][i]) * 12)
        args = (float(data["spot"][i]), float(data["strike"]), months, float(data["rate"][i]))
        args += (float(data["dividend_yield"][i]), float(data["volatility"][i]))
        for column in ("qdfp_high", "qdfp_strict"):
            assert generator.qdfp_put(*args, engines[column]) == pytest.approx(float(data[column][i]), abs=1e-12)


def _random_options(count: int, seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < count:
        kind = str(rng.choice(["call", "put"]))
        K = float(100 * math.exp(rng.uniform(-1.0, 1.0)))
        days = round(10 ** rng.uniform(0.0, math.log10(30 * 365)))
        sigma = float(10 ** rng.uniform(math.log10(0.02), math.log10(2.0)))
        r = float(rng.choice([rng.uniform(-0.05, 0.25), 0.0, rng.uniform(0.0, 0.02)]))
        q = float(rng.choice([rng.uniform(-0.1, 0.2), 0.0, r, r + 1e-4]))
        if am.exercise_regime(r, q, option_type=kind) != "double":
            cases.append((kind, 100.0, K, days, r, q, sigma))
    return cases


# Largest |ours - QdFp strict| / max(1, value) over these 80 options, measured: fast 2.0e-4, accurate 2.7e-6 and
# high 7.1e-9 (the worst case is a 6.4-year put with r = 17 %, q = 0); the long expiries need the most iterations.
WIDE_TOLERANCE = {"fast": 5e-4, "accurate": 5e-6, "high": 2e-8}


def test_calls_and_puts_far_outside_the_grid():
    worst = dict.fromkeys(am.SCHEMES, 0.0)
    for kind, S, K, days, r, q, sigma in _random_options(80, seed=7):
        reference = _option(kind, S, K, days, r, q, sigma, _strict).NPV()
        for scheme in am.SCHEMES:
            ours = am.price(S, K, days / 365, r, sigma, q, option_type=kind, scheme=scheme)
            worst[scheme] = max(worst[scheme], abs(ours - reference) / max(1.0, reference))
    for scheme, tolerance in WIDE_TOLERANCE.items():
        assert worst[scheme] <= tolerance, (scheme, worst[scheme])


@pytest.mark.parametrize(
    ("S", "K", "days", "r", "q", "sigma"),
    [
        (100, 100, 365, 0.05, 0.05, 0.25),
        (100, 95, 365, 0.075, 0.05, 0.25),
        (36, 40, 730, 0.075, 0.01, 0.4),
        (100, 90, 548, 0.025, -0.05, 0.4),
        (100, 120, 91, 0.10, 0.0, 0.15),
        (100, 100, 3650, 0.04, 0.02, 0.3),
    ],
)
def test_the_qd_plus_initial_guess_is_lis_approximation(S, K, days, r, q, sigma):
    """Our QD+ boundary on 8 Chebyshev nodes, with QuantLib's premium integral, is QuantLib's QdPlus engine."""
    T = days / 365
    scheme = am.Scheme("qd-plus", l=8, m=1, n=7, p=512)
    x_max = am._put_x(r, q)
    h = np.zeros(scheme.n + 1)
    h[1:] = np.log(am._qd_plus(T * am._grid(scheme).tau_share, r, q, sigma, x_max) / x_max) ** 2
    premium = K * am._premium(np.array([S / K]), T, r, q, sigma, h, scheme)[0]
    ours = max(bs.price(S, K, T, r, sigma, q, "put") + max(premium, 0.0), K - S)

    def qd_plus(process):
        return ql.QdPlusAmericanEngine(process, 8, ql.QdPlusAmericanEngine.Halley, 1e-12)

    assert ours == pytest.approx(_option("put", S, K, days, r, q, sigma, qd_plus).NPV(), abs=1e-8)


@pytest.mark.parametrize(
    ("kind", "r", "q"), [("put", -0.01, 0.02), ("put", 0.0, 0.03), ("call", 0.03, 0.0), ("call", 0.02, -0.01)]
)
def test_european_regimes_agree_with_quantlib(kind, r, q):
    reference = _option(kind, 100.0, 105.0, 365, r, q, 0.3, _strict).NPV()
    details = am.price_details(100.0, 105.0, 1.0, r, 0.3, q, option_type=kind)
    assert details.method == "european"
    assert details.value == pytest.approx(reference, rel=1e-12)


def test_the_double_boundary_put_against_quantlib_finite_differences():
    """QdFp refuses q < r < 0 for a put; the tree fallback is within 1e-4 of QuantLib's FD engine (4000 x 4000)."""
    with pytest.raises(RuntimeError, match="double-boundary"):
        _option("put", 100.0, 100.0, 365, -0.01, -0.02, 0.2, _strict).NPV()

    def finite_differences(process):
        return ql.FdBlackScholesVanillaEngine(process, 4000, 4000)

    reference = _option("put", 100.0, 100.0, 365, -0.01, -0.02, 0.2, finite_differences).NPV()
    assert reference == pytest.approx(7.62526, abs=5e-6)  # the value the plan quotes
    details = am.price_details(100.0, 100.0, 1.0, -0.01, 0.2, -0.02, option_type="put")
    assert details.fallback
    assert abs(details.value - reference) <= 1e-4  # measured 2.6e-5
