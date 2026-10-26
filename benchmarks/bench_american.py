"""American engine timings and accuracy: options_analyzer.american on the 4,495-put grid of Healy (2021, Table 5).

Run from the repository root::

    python benchmarks/bench_american.py             # our three schemes
    python benchmarks/bench_american.py --quantlib  # also time QuantLib's QdFp schemes on the same options

For each scheme it prints the error against the two QuantLib columns of tests/fixtures/alo_grid_qdfp.npz (QdFp
high precision, and QdFp at strict settings), the median time of one option priced on its own (its boundary
solved from scratch), the time to price the whole grid in one call (600 boundaries, one per expiry, rate, dividend
yield and volatility) and the median time of a 41-strike strip at one expiry (one boundary). Times are wall-clock
medians on this machine.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gen_alo_reference as reference  # noqa: E402

from options_analyzer import american as am  # noqa: E402


def median_seconds(function, cases, repeats: int = 3) -> float:
    """Best of ``repeats`` medians of single calls, with the boundary cache cleared before each call."""
    medians = []
    for _ in range(repeats):
        times = []
        for case in cases:
            am._put_boundary.cache_clear()
            start = time.perf_counter()
            function(*case)
            times.append(time.perf_counter() - start)
        medians.append(statistics.median(times))
    return min(medians)


def ours(grid: dict[str, np.ndarray], sample: np.ndarray) -> None:
    S, T, r, q, sigma = (grid[name] for name in ("spot", "maturity", "rate", "dividend_yield", "volatility"))
    cases = [(float(S[i]), 100.0, float(T[i]), float(r[i]), float(sigma[i]), float(q[i])) for i in sample]
    strikes = np.linspace(80.0, 120.0, 41)
    print(
        f"{'scheme':<9} {'(l, m, n, p)':<16} {'RMSE high':>10} {'max high':>9} {'RMSE strict':>11} {'max strict':>10}"
        f" {'1 option':>9} {'grid':>7} {'41 strikes':>10}"
    )
    for name, scheme in am.SCHEMES.items():
        am._put_boundary.cache_clear()
        start = time.perf_counter()
        values = am.price(S, 100.0, T, r, sigma, q, option_type="put", scheme=name)
        batch = time.perf_counter() - start
        errors = [values - grid[column] for column in ("qdfp_high", "qdfp_strict")]
        rmse = [float(np.sqrt(np.mean(e**2))) for e in errors]
        largest = [float(np.abs(e).max()) for e in errors]

        def one(S_, K_, T_, r_, sigma_, q_, scheme=name):
            am.price(S_, K_, T_, r_, sigma_, q_, option_type="put", scheme=scheme)

        single = median_seconds(one, cases)
        strip = median_seconds(
            lambda scheme=name: am.price(100.0, strikes, 0.5, 0.05, 0.25, 0.02, option_type="put", scheme=scheme),
            [()] * 20,
        )
        spec = f"({scheme.l}, {scheme.m}, {scheme.n}, {scheme.p})"
        print(
            f"{name:<9} {spec:<16} {rmse[0]:>10.1e} {largest[0]:>9.1e} {rmse[1]:>11.1e} {largest[1]:>10.1e}"
            f" {single * 1e3:>7.2f}ms {batch:>6.2f}s {strip * 1e3:>8.2f}ms"
        )


def quantlib(grid: dict[str, np.ndarray], sample: np.ndarray) -> None:
    import QuantLib as ql

    schemes = {
        "QdFp fast": ql.QdFpAmericanEngine.fastScheme(),
        "QdFp accurate": ql.QdFpAmericanEngine.accurateScheme(),
        "QdFp high": ql.QdFpAmericanEngine.highPrecisionScheme(),
    }
    print(f"\nQuantLib {ql.__version__} on {sample.size} of the options (C++ through its Python wrapper):")
    for name, scheme in schemes.items():
        errors, times = [], []
        for i in sample:
            args = (float(grid["spot"][i]), 100.0, round(float(grid["maturity"][i]) * 12), float(grid["rate"][i]))
            args += (float(grid["dividend_yield"][i]), float(grid["volatility"][i]))
            start = time.perf_counter()
            value = reference.qdfp_put(*args, scheme)
            times.append(time.perf_counter() - start)
            errors.append(value - float(grid["qdfp_strict"][i]))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        print(
            f"{name:<14} RMSE against strict {rmse:.1e}, median {statistics.median(times) * 1e3:.2f} ms per option"
            " (including QuantLib's object set-up)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=300, help="options timed one at a time")
    parser.add_argument("--quantlib", action="store_true", help="also time QuantLib's QdFp engine")
    args = parser.parse_args()
    grid = reference.read()
    sample = np.random.default_rng(0).choice(grid["spot"].size, size=args.sample, replace=False)
    ours(grid, sample)
    if args.quantlib:
        quantlib(grid, sample)


if __name__ == "__main__":
    main()
