"""Reference prices for the American engine: the 4,495-put grid of Healy (2021, Table 5), priced with QuantLib.

The grid is the test set of Andersen, Lake and Offengenden (2016) as Healy (2021, Table 5) lists it: strike 100,
spots {25, 50, 80, 90, 100, 110, 120, 150, 175, 200}, expiries {1/12, 0.25, 0.5, 0.75, 1} years, rates
{2, 4, 6, 8, 10} %, dividend yields {0, 4, 8, 12} % and volatilities {10, ..., 60} %: 6,000 American puts, of which
the 4,495 worth at least 0.5 are kept. Each is priced twice with QuantLib's ``QdFpAmericanEngine`` (BSD-3-Clause,
pinned in requirements-oracle.txt):

* ``qdfp_high``: its high-precision scheme (tanh-sinh quadrature to 1e-10, 10 iterations, 31 Chebyshev nodes), the
  reference the acceptance criteria of options_analyzer/american.py are stated against;
* ``qdfp_strict``: tanh-sinh to 1e-13, 24 iterations and 65 nodes, which shows how accurate ``qdfp_high`` itself is
  (it differs by up to about 1e-6 near the exercise boundary).

Both use QuantLib's own choice of fixed-point equation. The maturities are 1, 3, 6, 9 and 12 months on a 30/360 day
count, which gives the year fractions exactly (30/360 is 1/12 to the last bit).

The result is written as tests/fixtures/alo_grid_qdfp.npz, so that the main test suite checks the engine's accuracy
without QuantLib. It is a NumPy ``.npz`` archive, base64-encoded after a text header, because every fixture starts
with a provenance header (tests/test_data_policy.py reads each one as UTF-8 text); :func:`read` decodes it. The zip
members carry a fixed date, so regenerating with the same QuantLib gives the same file.

Run from the repository root (QuantLib from requirements-oracle.txt)::

    python scripts/gen_alo_reference.py              # regenerate the fixture (about 6 minutes)
    python scripts/gen_alo_reference.py --check 200  # reprice 200 random rows of the committed fixture
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "alo_grid_qdfp.npz"

STRIKE = 100.0
RATES = (0.02, 0.04, 0.06, 0.08, 0.10)
DIVIDEND_YIELDS = (0.0, 0.04, 0.08, 0.12)
SPOTS = (25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0)
MONTHS = (1, 3, 6, 9, 12)
VOLATILITIES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
#: Healy (2021) and Andersen et al. (2016) leave out the options worth less than this.
MIN_PRICE = 0.5
#: QdFpTanhSinhIterationScheme(iterations, nodes - 1, tolerance) of the ``qdfp_strict`` column.
STRICT_SCHEME = (24, 64, 1e-13)

#: The fixture's provenance header (tests/test_data_policy.py). Keep it in step with the constants above.
HEADER = """\
# Provenance: synthetic
# Description: model prices, not market data. The 4,495 American puts of Healy (2021, Table 5), the test set of
# Andersen, Lake and Offengenden (2016): K = 100, S in {25, 50, 80, 90, 100, 110, 120, 150, 175, 200}, T in
# {1/12, 0.25, 0.5, 0.75, 1} years, r in {2, 4, 6, 8, 10} %, q in {0, 4, 8, 12} % and sigma in {10, ..., 60} %,
# keeping the 4,495 of the 6,000 worth at least 0.5.
# Prices: QuantLib's QdFpAmericanEngine (BSD-3-Clause), qdfp_high with its high-precision scheme and qdfp_strict
# with tanh-sinh quadrature to 1e-13, 24 iterations and 65 Chebyshev nodes.
# Arrays: spot, strike, maturity (years), rate, dividend_yield, volatility, qdfp_high, qdfp_strict, quantlib_version.
# Generator: scripts/gen_alo_reference.py
# Format: a NumPy .npz archive, base64-encoded after this header so that the file is text with a provenance header
# like every fixture; read() in the generator decodes it.
"""


def qdfp_put(spot: float, strike: float, months: int, r: float, q: float, sigma: float, scheme) -> float:
    """QuantLib QdFp value of an American put expiring in ``months`` 30/360 months (``months / 12`` years)."""
    import QuantLib as ql

    today = ql.Date(2, 1, 2023)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Thirty360(ql.Thirty360.BondBasis)
    maturity = today + ql.Period(months, ql.Months)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)),
    )
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, strike), ql.AmericanExercise(today, maturity))
    option.setPricingEngine(ql.QdFpAmericanEngine(process, scheme))
    return option.NPV()


def schemes() -> dict:
    """The two QuantLib schemes of the fixture's columns."""
    import QuantLib as ql

    return {
        "qdfp_high": ql.QdFpAmericanEngine.highPrecisionScheme(),
        "qdfp_strict": ql.QdFpTanhSinhIterationScheme(*STRICT_SCHEME),
    }


def build(progress: bool = True) -> dict[str, np.ndarray]:
    """Price the whole grid: rows in the order (r, q, T, sigma, S), the options worth at least MIN_PRICE."""
    import QuantLib as ql

    engines = schemes()
    rows = []
    start = time.perf_counter()
    for r, q, months, sigma, spot in itertools.product(RATES, DIVIDEND_YIELDS, MONTHS, VOLATILITIES, SPOTS):
        high = qdfp_put(spot, STRIKE, months, r, q, sigma, engines["qdfp_high"])
        if high >= MIN_PRICE:
            strict = qdfp_put(spot, STRIKE, months, r, q, sigma, engines["qdfp_strict"])
            rows.append((spot, months / 12, r, q, sigma, high, strict))
            if progress and len(rows) % 500 == 0:
                print(f"{len(rows)} options priced in {time.perf_counter() - start:.0f} s", file=sys.stderr)
    columns = np.array(rows).T
    names = ("spot", "maturity", "rate", "dividend_yield", "volatility", "qdfp_high", "qdfp_strict")
    arrays = dict(zip(names, columns, strict=True))
    arrays["strike"] = np.array(STRIKE)
    arrays["quantlib_version"] = np.array(ql.__version__)
    return arrays


def write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write ``arrays`` as a compressed .npz archive, base64-encoded after the provenance header."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    encoded = base64.encodebytes(buffer.getvalue()).decode("ascii")
    path.write_text(HEADER + encoded, encoding="ascii")


def read(path: Path = FIXTURE) -> dict[str, np.ndarray]:
    """The arrays of a fixture written by :func:`write`."""
    lines = path.read_text(encoding="ascii").splitlines()
    payload = base64.b64decode("".join(line for line in lines if not line.startswith("#")))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def check(count: int, seed: int = 0) -> float:
    """Reprice ``count`` random rows of the committed fixture; returns the largest difference of either column."""
    data = read()
    engines = schemes()
    rows = np.random.default_rng(seed).choice(data["spot"].size, size=min(count, data["spot"].size), replace=False)
    worst = 0.0
    for i in rows:
        months = round(float(data["maturity"][i]) * 12)
        spot, strike, r = float(data["spot"][i]), float(data["strike"]), float(data["rate"][i])
        q, sigma = float(data["dividend_yield"][i]), float(data["volatility"][i])
        for column, scheme in engines.items():
            worst = max(worst, abs(qdfp_put(spot, strike, months, r, q, sigma, scheme) - float(data[column][i])))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=FIXTURE, help="where to write the fixture")
    parser.add_argument("--check", type=int, metavar="N", help="reprice N random rows of the committed fixture")
    args = parser.parse_args()
    if args.check:
        print(f"largest difference over {args.check} rows: {check(args.check):.3e}")
        return
    arrays = build()
    write(args.output, arrays)
    print(f"wrote {arrays['spot'].size} options to {args.output}")


if __name__ == "__main__":
    main()
