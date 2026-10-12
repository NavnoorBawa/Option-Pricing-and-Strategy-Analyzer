import itertools

import mpmath as mp
import numpy as np
import pytest

from options_analyzer import black_scholes as bs

mp.mp.dps = 40


def _mp_price(S, K, T, r, sigma, q, phi):
    S, K, T, r, sigma, q = map(mp.mpf, (S, K, T, r, sigma, q))
    d1 = (mp.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * mp.sqrt(T))
    d2 = d1 - sigma * mp.sqrt(T)
    return phi * (S * mp.e ** (-q * T) * mp.ncdf(phi * d1) - K * mp.e ** (-r * T) * mp.ncdf(phi * d2))


CASES = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),
    (42, 40, 0.5, 0.10, 0.20, 0.03),
    (100, 130, 0.25, -0.01, 0.45, 0.02),
    (100, 70, 2.0, 0.03, 0.10, 0.05),
    (50, 55, 0.05, 0.02, 0.60, 0.00),
]


def test_hull_reference_values():
    # Hull, Options Futures & Other Derivatives, Example 15.6.
    assert bs.price(42, 40, 0.5, 0.10, 0.20, option_type="call") == pytest.approx(4.7594, abs=1e-4)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, option_type="put") == pytest.approx(0.8086, abs=1e-4)
    assert bs.price(100, 100, 1.0, 0.05, 0.20, option_type="call") == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    c = bs.price(S, K, T, r, sigma, q, "call")
    p = bs.price(S, K, T, r, sigma, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize(("option_type", "phi"), [("call", 1), ("put", -1)])
def test_greeks_match_high_precision_derivatives(S, K, T, r, sigma, q, option_type, phi):
    g = bs.greeks(S, K, T, r, sigma, q, option_type)

    def V(**kw):
        args = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q} | kw
        return _mp_price(args["S"], args["K"], args["T"], args["r"], args["sigma"], args["q"], phi)

    reference = {
        "delta": mp.diff(lambda x: V(S=x), S),
        "gamma": mp.diff(lambda x: V(S=x), S, 2),
        "speed": mp.diff(lambda x: V(S=x), S, 3),
        "vega": mp.diff(lambda x: V(sigma=x), sigma),
        "vomma": mp.diff(lambda x: V(sigma=x), sigma, 2),
        "ultima": mp.diff(lambda x: V(sigma=x), sigma, 3),
        "theta": -mp.diff(lambda x: V(T=x), T),
        "rho": mp.diff(lambda x: V(r=x), r),
        "epsilon": mp.diff(lambda x: V(q=x), q),
        "vanna": mp.diff(lambda a, b: V(S=a, sigma=b), (S, sigma), (1, 1)),
        "charm": -mp.diff(lambda a, b: V(S=a, T=b), (S, T), (1, 1)),
        "veta": -mp.diff(lambda a, b: V(sigma=a, T=b), (sigma, T), (1, 1)),
        "zomma": mp.diff(lambda a, b: V(S=a, sigma=b), (S, sigma), (2, 1)),
        "color": -mp.diff(lambda a, b: V(S=a, T=b), (S, T), (2, 1)),
    }
    assert set(reference) == set(bs.GREEK_ORDER)
    for name, value in reference.items():
        assert g[name] == pytest.approx(float(value), rel=1e-9, abs=1e-12), name


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
def test_greek_identities(S, K, T, r, sigma, q):
    c = bs.greeks(S, K, T, r, sigma, q, "call")
    p = bs.greeks(S, K, T, r, sigma, q, "put")
    assert c["delta"] - p["delta"] == pytest.approx(np.exp(-q * T))
    assert c["charm"] - p["charm"] == pytest.approx(q * np.exp(-q * T))
    for name in ("gamma", "vega", "vanna", "vomma", "veta", "speed", "zomma", "color", "ultima"):
        assert c[name] == pytest.approx(p[name], rel=1e-10, abs=1e-14), name


def test_zero_volatility_is_discounted_forward_intrinsic():
    # Old code returned spot intrinsic (0 here), which is discontinuous in sigma.
    assert bs.price(100, 100, 1, 0.05, 0.0, 0.0, "call") == pytest.approx(100 - 100 * np.exp(-0.05))
    assert bs.price(100, 100, 1, 0.05, 1e-9, 0.0, "call") == pytest.approx(100 - 100 * np.exp(-0.05))
    assert bs.price(100, 110, 1, 0.05, 0.0, 0.0, "put") == pytest.approx(110 * np.exp(-0.05) - 100)
    g = bs.greeks(100, 90, 0.5, 0.05, 0.0, 0.02, "call")
    assert g["delta"] == pytest.approx(np.exp(-0.01))
    assert g["gamma"] == 0.0


def test_expired_and_boundary_inputs():
    assert bs.price(105, 100, 0.0, 0.05, 0.2, option_type="call") == pytest.approx(5.0)
    assert bs.price(95, 100, -1.0, 0.05, 0.2, option_type="put") == pytest.approx(5.0)
    assert bs.price(0.0, 100, 1, 0.05, 0.2, option_type="call") == 0.0
    assert bs.price(0.0, 100, 1, 0.05, 0.2, option_type="put") == pytest.approx(100 * np.exp(-0.05))
    g = bs.greeks(105, 100, 0.0, 0.05, 0.2, option_type="call")
    assert g["delta"] == 1.0 and g["theta"] == 0.0


def test_vectorised_and_scalar_outputs():
    spots = np.linspace(50, 150, 11)
    values = bs.price(spots, 100, 1.0, 0.05, 0.2)
    assert isinstance(values, np.ndarray) and values.shape == (11,)
    assert np.all(np.diff(values) > 0)
    assert isinstance(bs.price(100, 100, 1, 0.05, 0.2), float)
    grid = bs.price(spots[:, None], 100, 1.0, 0.05, np.array([0.1, 0.2, 0.3])[None, :])
    assert grid.shape == (11, 3)


def test_prices_respect_no_arbitrage_bounds():
    for S, K, T, sigma in itertools.product([1, 50, 100, 1e4], [1, 100, 1e4], [1e-6, 0.01, 1, 30], [1e-6, 0.2, 5]):
        c = bs.price(S, K, T, 0.03, sigma, 0.01, "call")
        p = bs.price(S, K, T, 0.03, sigma, 0.01, "put")
        assert max(S * np.exp(-0.01 * T) - K * np.exp(-0.03 * T), 0) - 1e-9 <= c <= S * np.exp(-0.01 * T) + 1e-9
        assert max(K * np.exp(-0.03 * T) - S * np.exp(-0.01 * T), 0) - 1e-9 <= p <= K * np.exp(-0.03 * T) + 1e-9


@pytest.mark.parametrize("bad", [{"S": -1}, {"K": -1}, {"sigma": -0.1}, {"S": np.nan}])
def test_invalid_inputs_raise(bad):
    args = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2} | bad
    with pytest.raises(ValueError):
        bs.price(**args)


def test_option_type_validation():
    assert bs.option_sign("CALL") == 1 and bs.option_sign("p") == -1
    with pytest.raises(ValueError):
        bs.option_sign("straddle")


def test_display_metadata_covers_every_greek():
    assert list(bs.GREEKS) == list(bs.GREEK_ORDER)


def test_delta_at_the_money_at_expiry_keeps_parity():
    call = bs.greeks(100, 100, 0.0, 0.05, 0.2, option_type="call")["delta"]
    put = bs.greeks(100, 100, 0.0, 0.05, 0.2, option_type="put")["delta"]
    assert (call, put) == (0.5, -0.5)


@pytest.mark.parametrize("sigma", [0.0, 1e-9])
def test_charm_parity_holds_in_the_zero_volatility_limit(sigma):
    c = bs.greeks(100, 90, 0.5, 0.05, sigma, 0.04, "call")["charm"]
    p = bs.greeks(100, 90, 0.5, 0.05, sigma, 0.04, "put")["charm"]
    assert c == pytest.approx(0.04 * np.exp(-0.02)) and p == pytest.approx(0.0)
