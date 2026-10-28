"""Forwards and discount factors per expiry from put-call parity (roadmap item SF-3).

For European options on one underlying and one expiry, put-call parity holds at every strike::

    C(K) - P(K) = DF (F - K)

with ``F`` the forward of the expiry and ``DF`` the discount factor the options are priced with (the box rate, which
for index options sits tens of basis points above Treasury rates). So the call-minus-put prices of an expiry lie on
a straight line in the strike whose slope is ``-DF`` and whose root is ``F``. Reading both from the quotes, instead
of building the forward from a sidebar rate and dividend yield, removes the largest error in a fitted surface: on
the project's own arbitrage-free synthetic SPX chain the sidebar forward put 135 out-of-the-money quotes outside the
no-arbitrage bounds, flagged 26 calendar violations and left SVI fits 1.33 vol points from the quotes (see
``tests/test_forwards.py``).

:func:`implied_forwards` estimates them expiry by expiry:

1. pair the call and the put at each strike (several quotes of one option, as from two option roots settling on the
   same day, are averaged with weights ``1 / error^2``);
2. keep the pairs near the money: ``|ln(K / F0)|`` within :data:`NEAR_THE_MONEY` or :data:`NEAR_SDS` standard
   deviations ``sigma sqrt(T)``, whichever is wider, of a first guess ``F0`` (the strike where ``|C - P|`` is
   smallest, moved by ``(C - P) / DF``), or the :data:`MIN_NEAR_PAIRS` nearest ``F0`` where fewer lie that close.
   ``sigma sqrt(T)`` is read from the straddle at that strike, ``C + P = 2 DF F (2 N(sigma sqrt(T) / 2) - 1)``, so
   the window needs no model. Deep in-the-money options quote wide and their weights are small, but a window that
   stays at 8 % for a two-year expiry (a third of a standard deviation) left the slope 0.6-0.7 % uncertain there;
   1.5 standard deviations measure it to 1.5e-4;
3. fit ``C - P = a - DF K`` by weighted least squares, each pair weighted by ``1 / (e_C + e_P)^2`` where ``e`` is
   how far each price may be off (half its bid/ask spread, see ``parse_quotes``), then remove pairs whose
   standardised residual is more than :data:`TRIM_MADS` median absolute deviations from the median, refitting until
   none is removed (at most :data:`TRIM_ROUNDS` rounds);
4. ``F = a / DF``. Where fewer than :data:`MIN_PARITY_PAIRS` pairs remain the expiry keeps the forward and discount
   factor of the rate and dividend yield it was given. Where the slope is too uncertain (standard error above
   :data:`MAX_DISCOUNT_SE`, typical of short expiries whose few strikes span a narrow range, where it hardly matters)
   or implausible (an implied rate outside :data:`RATE_LIMITS`), the discount factor comes from the given rate and
   the forward is the weighted mean of ``K + (C - P) / DF`` over the same pairs. Near the money that mean barely
   depends on ``DF``: a discount factor off by ``d`` (relative) moves each term by ``d (F - K)``.

Black-76 with ``F`` and ``DF`` is Black-Scholes-Merton with the rate ``r_T = -ln(DF) / T`` and the dividend yield
``q_T = r_T - ln(F / S) / T`` (``S e^{-q_T T} = DF F`` and ``K e^{-r_T T} = DF K``), so the quotes are inverted with
the existing solver at those per-expiry carries.

Parity is an equality only for European exercise. American chains (listed single-stock and ETF options) need their
quotes de-Americanised first (roadmap item SF-5); this module does not do that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtri

__all__ = [
    "FORWARD_COLUMNS",
    "FORWARD_SOURCES",
    "MAX_DISCOUNT_SE",
    "MIN_CARRY_GAP_BP",
    "MIN_PARITY_PAIRS",
    "carry_gaps",
    "implied_forwards",
]

#: Fewer call/put pairs than this left after trimming, and the expiry keeps the forward of the given rates.
MIN_PARITY_PAIRS = 6
#: Pairs within this distance of the first-guess forward in ``|ln K/F0|`` (8 %), or within NEAR_SDS standard
#: deviations of the log price at expiry where that is wider, are used...
NEAR_THE_MONEY = 0.08
NEAR_SDS = 1.5
#: ...or this many nearest it, where fewer lie within that distance.
MIN_NEAR_PAIRS = 20
#: A slope (discount factor) with a larger standard error is replaced by the given rate's discount factor: 5 bp of
#: the discount factor moves the undiscounted out-of-the-money prices, hence their implied vols, by 0.05 % (relative)
#: at most, while a slope that uncertain would give the implied rate an error of 5 bp / T.
MAX_DISCOUNT_SE = 5e-4
#: Implied rates outside this range (continuously compounded) are not believed; the given rate's discount factor is
#: used instead.
RATE_LIMITS = (-0.10, 0.50)
#: Outlier trimming: standardised residuals beyond this many scaled median absolute deviations from their median.
TRIM_MADS = 3.5
TRIM_ROUNDS = 5
_MAD_SCALE = 1.4826  # a median absolute deviation times this estimates the standard deviation of normal noise

#: A parity forward this far from the given rates' forward (basis points of the forward), or half the at-the-money
#: spread where that is wider, says the rate or dividend yield given does not match the quotes.
MIN_CARRY_GAP_BP = 5.0

#: How each expiry's forward and discount factor were obtained.
FORWARD_SOURCES = {
    "parity": "put-call parity: forward and discount factor fitted to the call/put pairs",
    "parity_forward": "put-call parity for the forward; discount factor from the rate input (slope too uncertain)",
    "rates": "the rate and dividend yield inputs (too few call/put pairs)",
}
#: The columns of :func:`implied_forwards`' table.
FORWARD_COLUMNS = (
    "maturity",
    "forward",
    "discount_factor",
    "rate",
    "dividend_yield",
    "pairs",
    "residual",
    "forward_se",
    "discount_se",
    "gap_bp",
    "atm_half_spread_bp",
    "source",
)


def _pair_prices(group: pd.DataFrame, floor: float) -> pd.DataFrame:
    """One call and one put price per strike of an expiry, with how far each may be off: several quotes of one
    option are averaged with weights ``1 / error^2``. Strikes quoted for only one side are dropped."""
    error = np.maximum(group["price_error"].to_numpy(float), floor)
    error = np.where(np.isfinite(error), error, floor)
    frame = pd.DataFrame(
        {
            "strike": group["strike"].to_numpy(float),
            "call": (group["type"] == "call").to_numpy(),
            "weight": 1.0 / error**2,
            "price": group["price"].to_numpy(float),
        }
    )
    frame["weighted"] = frame["weight"] * frame["price"]
    sides = frame.groupby(["strike", "call"])[["weight", "weighted"]].sum()
    sides["price"] = sides["weighted"] / sides["weight"]
    sides["error"] = 1.0 / np.sqrt(sides["weight"])
    wide = sides[["price", "error"]].unstack("call").dropna()
    if wide.empty or ("price", True) not in wide or ("price", False) not in wide:
        return pd.DataFrame(columns=["strike", "call", "put", "call_error", "put_error"])
    return pd.DataFrame(
        {
            "strike": wide.index.to_numpy(float),
            "call": wide[("price", True)].to_numpy(float),
            "put": wide[("price", False)].to_numpy(float),
            "call_error": wide[("error", True)].to_numpy(float),
            "put_error": wide[("error", False)].to_numpy(float),
        }
    )


def _near_the_money(pairs: pd.DataFrame, discount: float) -> pd.DataFrame:
    """The pairs near a first-guess forward (see the module docstring, step 2)."""
    difference = (pairs["call"] - pairs["put"]).to_numpy()
    strikes = pairs["strike"].to_numpy()
    nearest = int(np.argmin(np.abs(difference)))
    guess = strikes[nearest] + difference[nearest] / discount
    if not guess > 0:
        guess = strikes[nearest]
    # sigma sqrt(T) from the straddle: C + P = 2 DF F (2 N(s/2) - 1) at the money.
    straddle = float(pairs["call"].iloc[nearest] + pairs["put"].iloc[nearest]) / (4.0 * discount * guess)
    sd = 2.0 * float(ndtri(min(max(straddle, 0.0), 0.49) + 0.5))
    distance = np.abs(np.log(strikes / guess))
    near = distance < max(NEAR_THE_MONEY, NEAR_SDS * sd)
    if near.sum() < MIN_NEAR_PAIRS:
        near = distance <= np.sort(distance)[min(MIN_NEAR_PAIRS, len(distance)) - 1]
    return pairs[near]


def _weighted_line(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least squares ``y = a + b x``: ``(coefficients, covariance, residuals)``.

    The covariance scales the inverse normal matrix by the weighted residual variance (``n - 2`` degrees of
    freedom), so it reflects the scatter actually seen rather than the stated price errors, which are bid/ask
    half-spreads rather than standard deviations.
    """
    # Centre the strikes: around a forward of 5,000 the uncentred normal matrix is ill-conditioned.
    centre = float(np.average(x, weights=weight))
    design = np.column_stack([np.ones_like(x), x - centre])
    normal = design.T @ (design * weight[:, None])
    coef = np.linalg.solve(normal, design.T @ (weight * y))
    residuals = y - design @ coef
    dof = max(len(x) - 2, 1)
    scale = float(np.sum(weight * residuals**2) / dof)
    covariance = scale * np.linalg.inv(normal)
    # Back to y = a + b x: a = c0 - b centre.
    to_raw = np.array([[1.0, -centre], [0.0, 1.0]])
    return to_raw @ coef, to_raw @ covariance @ to_raw.T, residuals


def _trimmed_fit(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The parity line of the pairs with outliers removed (module docstring, step 3).

    Returns ``(coefficients, covariance, kept, residuals)``; ``kept`` is a boolean mask over ``pairs``.
    """
    strike = pairs["strike"].to_numpy(float)
    difference = (pairs["call"] - pairs["put"]).to_numpy(float)
    error = (pairs["call_error"] + pairs["put_error"]).to_numpy(float)
    weight = 1.0 / error**2
    kept = np.ones(len(pairs), dtype=bool)
    for _ in range(TRIM_ROUNDS):
        coef, covariance, residuals = _weighted_line(strike[kept], difference[kept], weight[kept])
        standardised = (difference - (coef[0] + coef[1] * strike)) / error
        centre = np.median(standardised[kept])
        spread = _MAD_SCALE * np.median(np.abs(standardised[kept] - centre))
        if not spread > 0:
            break
        # A pair within its own price error of the line is never an outlier, however small the scatter of the
        # others (exact model prices scatter by rounding only).
        keep = kept & (np.abs(standardised - centre) <= max(TRIM_MADS * spread, 1.0))
        if keep.sum() == kept.sum() or keep.sum() < MIN_PARITY_PAIRS:
            break
        kept = keep
    coef, covariance, residuals = _weighted_line(strike[kept], difference[kept], weight[kept])
    return coef, covariance, kept, residuals


@dataclass(frozen=True, eq=False)  # holds a table: compared by identity
class _ParityFit:
    """The parity line of one expiry's pairs near the money, before its discount factor is settled."""

    T: float
    used: pd.DataFrame  # the pairs kept after trimming
    intercept: float  # a = DF F
    discount: float  # the fitted slope, -b
    covariance: np.ndarray  # of (a, b)
    residuals: np.ndarray

    @property
    def weight(self) -> np.ndarray:
        return 1.0 / (self.used["call_error"] + self.used["put_error"]).to_numpy(float) ** 2

    @property
    def discount_se(self) -> float:
        return math.sqrt(max(float(self.covariance[1, 1]), 0.0))

    @property
    def measured(self) -> bool:
        """Whether the slope measures the discount factor (module docstring, step 4)."""
        if not (self.discount > 0 and self.discount_se <= MAX_DISCOUNT_SE):
            return False
        return RATE_LIMITS[0] <= -math.log(self.discount) / self.T <= RATE_LIMITS[1]


def _parity_fit(group: pd.DataFrame, T: float, given_discount: float, floor: float) -> _ParityFit | None:
    """The parity line of one expiry (module docstring, steps 1-3), or None with too few pairs."""
    pairs = _pair_prices(group, floor)
    if len(pairs) < MIN_PARITY_PAIRS:
        return None
    near = _near_the_money(pairs, given_discount)
    if len(near) < MIN_PARITY_PAIRS:
        return None
    coef, covariance, kept, residuals = _trimmed_fit(near)
    if kept.sum() < MIN_PARITY_PAIRS:
        return None
    return _ParityFit(T, near[kept], float(coef[0]), float(-coef[1]), covariance, residuals)


def _settled(fit: _ParityFit, discount: float | None) -> dict[str, float | int | str] | None:
    """The forward of a parity fit (module docstring, step 4): ``a / DF`` with the fitted slope, or, given another
    ``discount`` factor, the weighted mean of ``K + (C - P) / DF``; None if it is not a positive number."""
    weight = fit.weight
    if discount is None:
        a, b_var, cov = fit.intercept, fit.covariance[1, 1], fit.covariance[0, 1]
        discount = fit.discount
        forward = a / discount
        # Delta method for a / DF with DF = -b: var = var(a)/DF^2 + a^2 var(b)/DF^4 + 2 a cov(a, b)/DF^3.
        variance = fit.covariance[0, 0] / discount**2 + a**2 * b_var / discount**4 + 2.0 * a * cov / discount**3
        discount_se, source = fit.discount_se, "parity"
    else:
        strike = fit.used["strike"].to_numpy(float)
        each = strike + (fit.used["call"] - fit.used["put"]).to_numpy(float) / discount
        forward = float(np.average(each, weights=weight))
        variance = float(np.sum(weight * (each - forward) ** 2) / max(len(each) - 1, 1) / np.sum(weight))
        discount_se, source = math.nan, "parity_forward"
    if not (forward > 0 and math.isfinite(forward)):
        return None
    atm = int(np.argmin(np.abs(fit.used["strike"].to_numpy(float) - forward)))
    half_spread = 0.5 * float(fit.used["call_error"].iloc[atm] + fit.used["put_error"].iloc[atm])
    return {
        "forward": float(forward),
        "discount_factor": float(discount),
        "pairs": len(fit.used),
        "residual": float(np.sqrt(np.average(fit.residuals**2, weights=weight))),
        "forward_se": math.sqrt(max(float(variance), 0.0)),
        "discount_se": discount_se,
        "atm_half_spread_bp": 1e4 * half_spread / (discount * forward),
        "source": source,
    }


def _rate_spread(fits: list[_ParityFit], rate: float) -> float | None:
    """The implied rate's spread over ``rate``, pooled over the expiries whose slope measured it (weights ``1 /
    SE(r_T)^2`` with ``SE(r_T) = SE(DF) / (DF T)``), or None if none did.

    Option prices discount at a financing rate (the box rate) tens of basis points above Treasuries; an expiry whose
    own slope is too uncertain gets its discount factor from ``rate`` plus this spread rather than from ``rate``
    alone. On the synthetic chain, whose box rate is 45 bp above its Treasury rate, the last three expiries (1.4 to
    1.9 years) measure no slope of their own, and ``rate`` alone put their discount factors 0.6-0.8 % too high and
    their at-the-money vols 0.09-0.13 vol point too low.
    """
    measured = [fit for fit in fits if fit.measured]
    if not measured:
        return None
    spreads = np.array([-math.log(fit.discount) / fit.T - rate for fit in measured])
    errors = np.array([max(fit.discount_se, 1e-12) / (fit.discount * fit.T) for fit in measured])
    return float(np.average(spreads, weights=1.0 / errors**2))


def implied_forwards(
    prices: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float,
    *,
    expiry: pd.Series | np.ndarray | None = None,
    parity: bool = True,
) -> pd.DataFrame:
    """Each expiry's forward and discount factor from put-call parity on its option prices.

    Parameters
    ----------
    prices : pandas.DataFrame
        Option prices with columns ``strike``, ``maturity`` (years), ``type`` (``"call"``/``"put"``), ``price``
        (discounted premium) and ``price_error`` (how far the price may be off, e.g. half its bid/ask spread; NaN
        where unknown).
    spot : float
        Underlying price, for the implied dividend yield and the gap to the given carry.
    rate, dividend_yield : float
        The rate and dividend yield the forward otherwise comes from (continuously compounded, decimal): used where
        parity cannot be read, and as the reference of ``gap_bp``.
    expiry : array_like of int, optional
        An expiry label per row (rows with equal labels are one expiry, at their median maturity); by default each
        distinct maturity is one expiry.
    parity : bool, default True
        ``False`` gives every expiry the forward and discount factor of ``rate`` and ``dividend_yield`` (source
        ``"rates"``): the same table, for display, when parity is not wanted.

    Returns
    -------
    pandas.DataFrame
        One row per expiry, in order of maturity, with the columns of :data:`FORWARD_COLUMNS`: ``forward`` and
        ``discount_factor``; the implied ``rate`` (``-ln(DF) / T``) and ``dividend_yield`` (``rate - ln(F/S) / T``);
        the number of call/put ``pairs`` used and the weighted RMS ``residual`` of their call-minus-put prices
        around the fitted line (price units); the standard errors ``forward_se`` and ``discount_se`` (NaN where the
        discount factor was not fitted); ``gap_bp``, the forward's distance from ``spot e^{(rate - dividend_yield)
        T}`` in basis points of the forward; ``atm_half_spread_bp``, half the at-the-money call and put spreads in
        basis points of the discounted forward (the precision the quotes give the forward); and the ``source``, a
        key of :data:`FORWARD_SOURCES`.

    Examples
    --------
    Noise-free European prices at a 5 % rate and a 2 % dividend yield, read with a 4 % rate and no yield:

    >>> from options_analyzer import black_scholes as bs
    >>> strikes = np.arange(80.0, 121.0, 5.0)
    >>> rows = [(K, 0.5, kind, bs.price(100.0, K, 0.5, 0.05, 0.2, 0.02, kind), 0.01)
    ...         for K in strikes for kind in ("call", "put")]
    >>> quotes = pd.DataFrame(rows, columns=["strike", "maturity", "type", "price", "price_error"])
    >>> table = implied_forwards(quotes, 100.0, 0.04, 0.0)
    >>> round(float(table["forward"].iloc[0]), 10) == round(100.0 * math.exp(0.03 * 0.5), 10)
    True
    >>> round(float(table["rate"].iloc[0]), 10), round(float(table["dividend_yield"].iloc[0]), 10)
    (0.05, 0.02)
    >>> table["source"].iloc[0], int(table["pairs"].iloc[0])
    ('parity', 9)
    """
    spot = float(spot)
    data = prices.reset_index(drop=True)
    labels = np.asarray(data["maturity"].to_numpy(float) if expiry is None else expiry)
    # A floor on each price's error: zero errors (model prices on no grid) would make the weights infinite.
    floor = 1e-10 * max(spot, float(data["strike"].max()) if len(data) else spot)
    expiries = []
    for label in np.unique(labels):
        group = data[labels == label]
        T = float(np.median(group["maturity"].to_numpy(float)))
        fit = _parity_fit(group, T, math.exp(-rate * T), floor) if parity else None
        expiries.append((label, T, fit))
    spread = _rate_spread([fit for _, _, fit in expiries if fit is not None], rate)
    rows = []
    for label, T, fit in expiries:
        row: dict[str, float | int | str] = {
            "expiry": label,
            "maturity": T,
            "forward": spot * math.exp((rate - dividend_yield) * T),
            "discount_factor": math.exp(-rate * T),
            "pairs": 0,
            "source": "rates",
        }
        if fit is not None:
            discount = None if fit.measured else math.exp(-(rate + (spread or 0.0)) * T)
            row.update(_settled(fit, discount) or {})
        rows.append(row)
    table = pd.DataFrame(rows, columns=["expiry", *FORWARD_COLUMNS])
    if table.empty:
        return table
    T = table["maturity"].to_numpy(float)
    discount = table["discount_factor"].to_numpy(float)
    forward = table["forward"].to_numpy(float)
    table["rate"] = -np.log(discount) / T
    table["dividend_yield"] = table["rate"] - np.log(forward / spot) / T
    table["gap_bp"] = 1e4 * (np.log(forward / spot) - (rate - dividend_yield) * T)
    return table.sort_values("maturity", kind="stable").reset_index(drop=True)


def carry_gaps(table: pd.DataFrame) -> np.ndarray:
    """Which expiries of :func:`implied_forwards`' table have a parity forward more than :data:`MIN_CARRY_GAP_BP`,
    or half the at-the-money call and put spreads where that is wider, from the forward of the given rate and
    dividend yield: the quotes resolve the difference, so the given carry does not match them.

    Examples
    --------
    >>> table = pd.DataFrame({"gap_bp": [3.0, -12.0, 12.0, 40.0], "atm_half_spread_bp": [1.0, 2.0, 15.0, 1.0],
    ...                       "source": ["parity", "parity", "parity", "rates"]})
    >>> carry_gaps(table).tolist()
    [False, True, False, False]
    """
    gap = np.abs(table["gap_bp"].to_numpy(float))
    allowed = np.fmax(MIN_CARRY_GAP_BP, table["atm_half_spread_bp"].to_numpy(float))
    return (table["source"] != "rates").to_numpy() & (gap > allowed)
