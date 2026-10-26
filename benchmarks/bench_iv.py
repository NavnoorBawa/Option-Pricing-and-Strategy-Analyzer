"""Implied-volatility timings: Let's Be Rational (scalar and batch) against the Brent solver it replaced.

Run from the repository root::

    python benchmarks/bench_iv.py               # 10,000 quotes, 7 repeats of the batch
    python benchmarks/bench_iv.py --legacy-all  # also time the Brent solver on all quotes (about 8 s)

The quotes are a spot-100 chain: expiries from a day to five years, volatilities from 5 % to 150 %, strikes within
three standard deviations of the forward, calls and puts (in and out of the money), rates -2 % to 10 %, dividend
yields 0 to 5 %. Each line prints a median of wall-clock times on this machine.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options_analyzer import black_scholes as bs
from options_analyzer import implied_vol as iv
from options_analyzer import lets_be_rational as lbr


def quotes(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = 10 ** rng.uniform(np.log10(1 / 365), np.log10(5), n)
    sigma = 10 ** rng.uniform(np.log10(0.05), np.log10(1.5), n)
    r = rng.uniform(-0.02, 0.1, n)
    q = rng.uniform(0, 0.05, n)
    K = 100 * np.exp((r - q) * T + rng.uniform(-3, 3, n) * sigma * np.sqrt(T))
    kind = np.where(rng.random(n) < 0.5, "call", "put")
    price = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    return price, K, T, r, q, kind, sigma


def per_call(function, rows, repeats: int = 1) -> float:
    """Median seconds of single calls."""
    times = []
    for _ in range(repeats):
        for args in rows:
            start = time.perf_counter()
            function(*args)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def whole(function, repeats: int) -> float:
    """Median seconds of whole calls."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def legacy_brent(price, S, K, T, r, q, kind):
    """The solver implied_volatility used before Let's Be Rational: Brent's method on the out-of-the-money price."""
    phi = bs.option_sign(kind)
    fwd_s, pv_k = iv._discounted(S, K, T, r, q)
    otm_type = "put" if fwd_s >= pv_k else "call"
    otm_price = price if bs.option_sign(otm_type) == phi else price - phi * (fwd_s - pv_k)
    return iv._brent_implied_volatility(otm_price, S, K, T, r, q, otm_type, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quotes", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--legacy-all", action="store_true", help="time the Brent solver on every quote")
    args = parser.parse_args()

    price, K, T, r, q, kind, sigma = quotes(args.quotes)
    rows = [
        (float(p), 100.0, float(k), float(t), float(rr), float(qq), str(c))
        for p, k, t, rr, qq, c in zip(price, K, T, r, q, kind, strict=True)
    ]
    vol = iv.implied_volatility_batch(price, 100.0, K, T, r, q, kind)
    # The prices carry black_scholes.price's rounding, which deep in the money (large cond) moves the volatility most.
    error = np.nanmax(np.abs(vol - sigma) / sigma)
    print(f"{args.quotes:,} quotes; batch volatility vs the input volatility: max relative error {error:.2e}")
    print()

    batch = whole(lambda: iv.implied_volatility_batch(price, 100.0, K, T, r, q, kind), args.repeats)
    print(f"implied_volatility_batch, {args.quotes:,} quotes:     {batch * 1e3:8.2f} ms")
    scalar = per_call(iv.implied_volatility, rows[:2_000], repeats=3)
    print(f"implied_volatility, one quote (median):    {scalar * 1e6:8.2f} µs")
    loop = whole(lambda: [iv.implied_volatility(*row) for row in rows], 1)
    print(f"implied_volatility in a loop, {args.quotes:,} quotes: {loop * 1e3:8.2f} ms")

    fwd = 100 * np.exp((r - q) * T)
    undiscounted = price * np.exp(r * T)
    theta = np.where(kind == "call", 1.0, -1.0)
    black_rows = list(zip(undiscounted.tolist(), fwd.tolist(), K.tolist(), T.tolist(), theta.tolist(), strict=True))
    core = per_call(lbr.implied_black_volatility, black_rows[:2_000], repeats=3)
    print(f"lets_be_rational.implied_black_volatility, scalar (median): {core * 1e6:8.2f} µs")
    core_batch = whole(lambda: lbr.implied_black_volatility(undiscounted, fwd, K, T, theta), args.repeats)
    print(f"lets_be_rational.implied_black_volatility, {args.quotes:,} at once:  {core_batch * 1e3:8.2f} ms")

    legacy_rows = rows if args.legacy_all else rows[:1_000]
    legacy = per_call(legacy_brent, legacy_rows)
    print(f"legacy Brent solver, one quote (median):   {legacy * 1e6:8.2f} µs")
    if args.legacy_all:
        total = whole(lambda: [legacy_brent(*row) for row in rows], 1)
        print(f"legacy Brent solver in a loop, {args.quotes:,} quotes: {total:8.2f} s")
    try:
        import py_lets_be_rational as plbr
    except ImportError:
        return
    reference = per_call(plbr.implied_volatility_from_a_transformed_rational_guess, black_rows[:2_000], repeats=3)
    print(f"py_lets_be_rational (pure Python), scalar (median): {reference * 1e6:8.2f} µs")


if __name__ == "__main__":
    main()
