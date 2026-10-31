"""An illustrative portfolio-margin requirement in the style of FINRA Rule 4210(g).

FINRA 4210(g) lets a broker margin a customer's listed options by the largest theoretical loss of each product group
over a range of moves in the underlying (https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210):

* (g)(2)(F), *theoretical gains and losses*: the gain or loss "at ten equidistant intervals (valuation points)"
  over a range set by the product class: +6% / -8% for a high-capitalisation, broad-based market index (SPX),
  +/-10% for another broad-based index and +/-15% for an equity or a narrow-based index;
* (g)(7)(B), the minimum: "$.375 for each listed option ... multiplied by the contract's or instrument's
  multiplier", not more than the market value of a long contract.

The requirement is the greater of the two. This module follows that arithmetic on one position with the app's own
valuation (Black-Scholes-Merton at each leg's implied vol, held fixed), and nothing more: no offsets between product
groups, no OCC theoretical prices, no house add-ons and no concentration charges. It is illustrative, not a
broker's requirement, and the app labels it so.

The rule's "ten equidistant intervals (valuation points)" is read as ten points, both ends of the range included
(-8%, -6.44%, ..., +6% for SPX), as the app's spec draws them. Ten intervals of a range would be eleven points; the
largest loss of a position whose P&L is monotone in spot (a vertical spread, a single option) is at an end of the
range either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import scenario_cube
from .strategies import Market, Strategy, _number, position_value

__all__ = [
    "MINIMUM_PER_CONTRACT",
    "PRODUCT_CLASSES",
    "VALUATION_POINTS",
    "PortfolioMargin",
    "minimum_requirement",
    "portfolio_margin",
    "valuation_points",
]

#: The valuation-point range of each product class, as (largest fall, largest rise) of the underlying: FINRA
#: 4210(g)(2)(F)'s table.
PRODUCT_CLASSES: dict[str, tuple[float, float]] = {
    "high_cap_index": (-0.08, 0.06),  # a high-capitalisation, broad-based market index, such as the S&P 500
    "broad_index": (-0.10, 0.10),  # another broad-based market index
    "equity": (-0.15, 0.15),  # an equity security or a narrow-based index
}
#: Names of the product classes as the app writes them.
CLASS_NAMES = {
    "high_cap_index": "high-cap broad-based index",
    "broad_index": "broad-based index",
    "equity": "equity or narrow-based index",
}
#: Valuation points of the range (FINRA 4210(g)(2)(F)).
VALUATION_POINTS = 10
#: The minimum per listed option contract, times its multiplier (FINRA 4210(g)(7)(B)): $0.375.
MINIMUM_PER_CONTRACT = 0.375


@dataclass(frozen=True)
class PortfolioMargin:
    """The illustrative requirement of one position, from :func:`portfolio_margin`.

    Attributes
    ----------
    product_class : str
        A key of :data:`PRODUCT_CLASSES`.
    points : numpy.ndarray
        The ten spot moves (relative: -0.08 is -8%).
    pnl : numpy.ndarray
        The position's P&L at each point ($), against its value today, implied vols unchanged.
    worst : float
        The lowest of ``pnl`` ($): negative when the position loses at some point.
    minimum : float
        The (g)(7)(B) minimum ($): $0.375 x multiplier per option contract, a long one's at most its value.
    requirement : float
        The greater of the largest theoretical loss (``-worst``, when negative) and ``minimum`` ($, never negative).
    """

    product_class: str
    points: np.ndarray
    pnl: np.ndarray
    worst: float
    minimum: float
    requirement: float

    @property
    def worst_point(self) -> float:
        """The spot move of the worst point (relative)."""
        return float(self.points[int(np.argmin(self.pnl))])


def valuation_points(product_class: str = "high_cap_index") -> np.ndarray:
    """The ten equidistant spot moves of a product class, from its largest fall to its largest rise.

    Parameters
    ----------
    product_class : str, default "high_cap_index"
        A key of :data:`PRODUCT_CLASSES`.

    Returns
    -------
    numpy.ndarray
        Ten relative moves, ends included.

    Raises
    ------
    ValueError
        If the product class is unknown.

    Examples
    --------
    >>> (100 * valuation_points()).round(2).tolist()
    [-8.0, -6.44, -4.89, -3.33, -1.78, -0.22, 1.33, 2.89, 4.44, 6.0]
    >>> (100 * valuation_points("equity")[[0, -1]]).tolist()
    [-15.0, 15.0]
    """
    if product_class not in PRODUCT_CLASSES:
        raise ValueError(f"product_class must be one of {tuple(PRODUCT_CLASSES)}, got {product_class!r}")
    low, high = PRODUCT_CLASSES[product_class]
    return np.linspace(low, high, VALUATION_POINTS)


def minimum_requirement(strategy: Strategy, market: Market, *, contracts: float = 1, multiplier: float = 100) -> float:
    """FINRA 4210(g)(7)(B)'s minimum: $0.375 times the multiplier for each option contract, a long one's at most its
    market value.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position (leg quantities per contract) and the market that values its long legs.
    contracts : float, default 1
        Contracts of the position: each option leg is ``|quantity| x contracts`` listed contracts.
    multiplier : float, default 100
        Shares (index units) per contract.

    Returns
    -------
    float
        The minimum ($). Shares carry none.

    Examples
    --------
    An SPX iron condor, one of each leg, is four contracts: 4 x $37.50.

    >>> from options_analyzer.strategies import build_template
    >>> condor = build_template("Iron Condor", 5800, 100, 55 / 365)
    >>> minimum_requirement(condor, Market(5800, 0.045, 0.16, 55 / 365, 0.013))
    150.0

    A long option worth less than $37.50 a contract is charged its value:

    >>> from options_analyzer.strategies import Leg
    >>> far = Strategy("far call", (Leg("call", 1, 7000),))
    >>> round(minimum_requirement(far, Market(5800, 0.045, 0.16, 7 / 365, 0.013)), 6)
    0.0
    """
    contracts, multiplier = _number("contracts", contracts), _number("multiplier", multiplier)
    floor = MINIMUM_PER_CONTRACT * multiplier
    total = 0.0
    for leg in strategy.legs:
        if not leg.is_option:
            continue
        per_contract = floor
        if leg.quantity > 0:
            value = float(position_value(Strategy("leg", (leg,)), market)) / leg.quantity * multiplier
            per_contract = min(floor, max(value, 0.0))
        total += abs(leg.quantity) * abs(contracts) * per_contract
    return total


def portfolio_margin(
    strategy: Strategy,
    market: Market,
    *,
    product_class: str = "high_cap_index",
    contracts: float = 1,
    multiplier: float = 100,
) -> PortfolioMargin:
    """An illustrative FINRA 4210(g)-style portfolio-margin requirement of one position.

    The position is revalued at the ten equidistant spot moves of its product class (:func:`valuation_points`),
    implied vols and time unchanged, by :func:`~options_analyzer.scenarios.scenario_cube`; the requirement is the
    greater of the largest loss and the (g)(7)(B) minimum (:func:`minimum_requirement`).

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position (leg quantities per contract) and today's market.
    product_class : str, default "high_cap_index"
        A key of :data:`PRODUCT_CLASSES`: ``"high_cap_index"`` for SPX.
    contracts : float, default 1
        Contracts of the position.
    multiplier : float, default 100
        Shares (index units) per contract.

    Returns
    -------
    PortfolioMargin

    Raises
    ------
    ValueError
        If the product class is unknown or ``contracts`` or ``multiplier`` is not a finite number.

    Examples
    --------
    A short at-the-money SPX put, 30 days: the requirement is its loss at -8%.

    >>> from options_analyzer.strategies import Leg
    >>> put = Strategy("short put", (Leg("put", -1, 5800),))
    >>> margin = portfolio_margin(put, Market(5800, 0.045, 0.16, 30 / 365, 0.013))
    >>> round(margin.worst), round(margin.requirement), margin.worst_point
    (-35386, 35386, -0.08)
    """
    contracts, multiplier = _number("contracts", contracts), _number("multiplier", multiplier)
    points = valuation_points(product_class)
    pnl = scenario_cube(strategy, market, points, [0.0], multiplier=contracts * multiplier)[0, 0]
    worst = float(np.min(pnl))
    minimum = minimum_requirement(strategy, market, contracts=contracts, multiplier=multiplier)
    return PortfolioMargin(product_class, points, pnl, worst, minimum, max(-worst, minimum, 0.0))
