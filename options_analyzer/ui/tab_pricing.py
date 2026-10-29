"""Pricing tab and the headline price cards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr

from .. import black_scholes as bs
from ..american import AmericanPrice
from ..strategies import ONE_DAY
from . import cache, charts
from .components import card_row
from .format import DECAY_SWITCH, decay_label, decay_leads, md, money, pct
from .inputs import Inputs

#: Monte Carlo scenarios drawn on the sample-paths chart.
SAMPLE_PATHS = 60
#: The confidence interval of a simulated price, in no-break spaces: it never splits over two lines (on a phone "SE"
#: stood alone on the second line of the caption under the price cards).
_CI_FORMULA = "95% CI = price ± 1.96 SE".replace(" ", "\xa0")
#: What a heavy-tailed Monte Carlo sample does instead of the usual estimator (see ``monte_carlo``).
HEAVY_TAIL_NOTE = (
    "Heavy-tailed sample (large σ√T): the put is averaged directly and the call follows from put-call parity"
)
#: Why the American engine values an option at its European value. ``american.exercise_regime`` finds early exercise
#: never optimal for a call with ``q <= 0`` and ``q <= r`` and for a put with ``r <= 0`` and ``r <= q``; the sidebar's
#: dividend yield is never negative, so that is a call without dividends at a rate of at least zero, and a put at a
#: rate of at most zero.
NEVER_EARLY = {
    "call": (
        "with no dividend yield, exercising a call early never pays (it would pay the strike early and gain no "
        "dividends)"
    ),
    "put": "at a rate of zero or below, exercising a put early never pays (the strike would earn no interest)",
}
#: Why the American engine hands an option to the tree (``AmericanPrice.fallback_reason``).
FALLBACK_REASONS = {
    "double boundary": "its exercise region lies between two boundaries, which the engine does not solve",
    "not converged": "the engine's iteration for the exercise boundary did not converge at these inputs",
}
#: How the American engine valued an option (``AmericanPrice.method``), for the engine table.
ALO_METHODS = {
    "alo": "Spectral collocation",
    "european": "European value (never exercised early)",
    "tree": "Leisen-Reimer trees (fallback)",
    "exact": "Closed form",
    "intrinsic": "Intrinsic value",
}
#: The engine's Greeks, as the Greeks table shows them (names of ``black_scholes.GREEKS``).
ALO_GREEKS = ("delta", "gamma", "theta", "vega", "rho", "epsilon")


@dataclass(frozen=True)
class Prices:
    call: float
    put: float
    call_note: str = ""
    put_note: str = ""
    tree: dict | None = None
    mc: dict | None = None
    #: Values on the same tree with European exercise (American tree only).
    european_tree: dict | None = None
    #: The American engine's results (``american.price_details``) and Greeks (``american.greeks``), by option type.
    alo: dict | None = None
    alo_greeks: dict | None = None


def compute_prices(inputs: Inputs) -> Prices:
    """Model prices for the headline cards. Raises ``ValueError`` on invalid tree inputs."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    bs_call = bs.price(S, K, T, r, sigma, q, "call")
    bs_put = bs.price(S, K, T, r, sigma, q, "put")
    if model.kind == "binomial":
        tree = cache.tree_results(S, K, T, r, sigma, q, model.steps, model.american, model.tree_method)
        # The early-exercise premium is measured on the same lattice, so the tree's discretisation error cancels.
        european = (
            cache.tree_prices(S, K, T, r, sigma, q, model.steps, False, model.tree_method) if model.american else None
        )
        notes = {}
        for kind, ref in (("call", bs_call), ("put", bs_put)):
            if model.american:
                premium = tree[kind].price - european[kind]
                notes[kind] = f"Early-exercise premium {money(premium, 4, signed=True)} (same tree, European exercise)"
            else:
                notes[kind] = f"Tree error vs BSM {money(tree[kind].price - ref, 6, signed=True)}"
        return Prices(
            tree["call"].price, tree["put"].price, notes["call"], notes["put"], tree=tree, european_tree=european
        )
    if model.kind == "monte_carlo":
        mc = cache.mc_results(S, K, T, r, sigma, q, model.n_paths, model.seed, model.antithetic, model.control_variate)
        notes = {
            kind: (f"{HEAVY_TAIL_NOTE}. " if res.heavy_tailed else "")
            + (
                f"95% CI {money(res.ci95[0], 3)} – {money(res.ci95[1], 3)} · SE {money(res.std_error, 4)}"
                if np.isfinite(res.std_error)
                else "Every simulated price finished on one side of the strike: the error cannot be estimated"
            )
            for kind, res in mc.items()
        }
        return Prices(mc["call"].price, mc["put"].price, notes["call"], notes["put"], mc=mc)
    if model.kind == "alo":
        details, greeks = cache.alo_results(S, K, T, r, sigma, q, model.scheme)
        notes = {kind: _alo_note(kind, res) for kind, res in details.items()}
        return Prices(
            details["call"].value,
            details["put"].value,
            notes["call"],
            notes["put"],
            alo=details,
            alo_greeks=greeks,
        )
    return Prices(bs_call, bs_put, "Closed-form European value", "Closed-form European value")


def _alo_note(kind: str, res: AmericanPrice) -> str:
    """The tooltip of a price card under the American engine: its early-exercise premium, or why it has none, or
    why the engine fell back to the tree."""
    if res.fallback:
        return f"Valued on Leisen-Reimer trees, not by the engine: {FALLBACK_REASONS[res.fallback_reason]}"
    if res.method == "european":
        return f"Equals the Black-Scholes-Merton value: {NEVER_EARLY[kind]}"
    return (
        f"Early-exercise premium {money(res.premium, 4, signed=True)} over the European value {money(res.european, 4)}"
    )


def alo_fallbacks(prices: Prices) -> list[str]:
    """Plain statements of the options the American engine handed to the tree, and why (empty when none)."""
    notes = []
    for kind, res in (prices.alo or {}).items():
        if res.fallback:
            detail = ""
            if res.fallback_reason == "not converged":
                detail = (
                    f" (its last step still moved the logarithm of the boundary by {res.last_step:.2g}, where at most "
                    "0.01 passes)"
                )
            notes.append(
                f"The {kind} is valued on Leisen-Reimer trees (2,001 and 4,003 steps, extrapolated) instead of the "
                f"American engine: {FALLBACK_REASONS[res.fallback_reason]}{detail}."
            )
    return notes


def _model_card(inputs: Inputs, prices: Prices) -> tuple[str, str, str, str]:
    """``(label, value, caption, help)`` of the fourth headline card: what the sidebar model adds or how far it
    is from the exact answer. (Put-call parity would read zero for every European model by construction.) The
    caption under the cards gives the call's and the put's figures where the card shows the larger of the two; it
    is empty for the closed form, whose card holds the call's figure and its tooltip the put's."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    if model.kind == "binomial" and model.american:
        premium = {kind: prices.tree[kind].price - prices.european_tree[kind] for kind in ("call", "put")}
        larger = max(premium, key=lambda kind: abs(premium[kind]))
        return (
            "Exercise premium",
            money(premium[larger], 4),
            f"Early-exercise premium: call {money(premium['call'], 4, signed=True)}, "
            f"put {money(premium['put'], 4, signed=True)} (against the same tree with European exercise)",
            "American value minus the value on the same tree with European exercise (the larger of the call's and "
            "the put's is shown). Measured on one lattice, so the tree's discretisation error cancels.",
        )
    if model.kind == "alo":
        premium = {kind: prices.alo[kind].premium for kind in ("call", "put")}
        larger = max(premium, key=lambda kind: abs(premium[kind]))
        never = "".join(
            f" The {kind} equals its Black-Scholes-Merton value: {NEVER_EARLY[kind]}."
            for kind, res in prices.alo.items()
            if res.method == "european"
        )
        return (
            "Exercise premium",
            money(premium[larger], 4),
            f"Early-exercise premium: call {money(premium['call'], 4, signed=True)}, "
            f"put {money(premium['put'], 4, signed=True)} (over the closed-form European value).{never}",
            "American value minus the closed-form Black-Scholes-Merton value of the same option with European "
            "exercise (the larger of the call's and the put's is shown): what the right to exercise early is worth.",
        )
    if model.kind == "binomial":
        error = {kind: prices.tree[kind].price - bs.price(S, K, T, r, sigma, q, kind) for kind in ("call", "put")}
        return (
            "Tree error vs BSM",
            money(max(abs(e) for e in error.values()), 6),
            f"Error against the closed-form value: call {money(error['call'], 6, signed=True)}, "
            f"put {money(error['put'], 6, signed=True)}",
            "Tree value minus the closed-form Black-Scholes-Merton value, the exact answer for European exercise "
            "(the larger of the call's and the put's is shown). It shrinks as the number of steps grows.",
        )
    if model.kind == "monte_carlo":
        se = {kind: prices.mc[kind].std_error for kind in ("call", "put")}
        finite = [v for v in se.values() if np.isfinite(v)]
        heavy = any(res.heavy_tailed for res in prices.mc.values())
        return (
            "Std. error",
            money(max(finite), 4) if finite else "–",
            f"Standard error: call {money(se['call'], 4)}, put {money(se['put'], 4)}; {_CI_FORMULA}"
            + (f". {HEAVY_TAIL_NOTE}." if heavy else ""),
            "Standard error of the simulated prices, after any variance reduction in use (the header's badge names "
            "it: AV for antithetic variates, CV for a control variate). The larger of the call's and the put's is "
            "shown; it falls as one over the square root of the number of paths.",
        )
    vst = sigma * np.sqrt(T)
    if vst > 0:
        itm = float(ndtr((np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / vst))
    else:
        itm = float(np.sign(S * np.exp((r - q) * T) - K) + 1.0) / 2.0  # certain, or 1/2 exactly at the strike
    return (
        "Call ITM probability",
        pct(itm, 2),
        "",
        "Probability that the call finishes in the money under the pricing (risk-neutral) measure, N(d₂); the "
        f"put's is 1 − N(d₂) = {pct(1.0 - itm, 2)}. It uses the risk-free drift, so it is not a real-world forecast.",
    )


#: Keys of the headline price-card row and of its two pairs.
PRICE_CARDS = ("price_cards", ("price_cards_prices", "price_cards_model"))


def render_price_cards(inputs: Inputs, prices: Prices) -> None:
    S, T, r, q = inputs.spot, inputs.expiry, inputs.rate, inputs.dividend_yield
    forward = S * np.exp((r - q) * T)
    label, value, caption, help_text = _model_card(inputs, prices)
    # The per-option notes are in the tooltips so the cards stay short on narrow screens.
    cards = [
        ("Call value", money(prices.call, 4), md(prices.call_note)),
        ("Put value", money(prices.put, 4), md(prices.put_note)),
        ("Forward price", money(forward, 4), f"F = S·e^((r−q)T). Discount factor e^(−rT) = {np.exp(-r * T):.6f}."),
        (label, value, md(help_text)),
    ]
    card_row(*PRICE_CARDS, [(name, text, {"help": help_text}) for name, text, help_text in cards])
    # The header's badge names the model, so the caption only adds what the cards do not show.
    if caption:
        st.caption(md(caption))
    for note in alo_fallbacks(prices):
        st.warning(note, icon=":material/warning:")


def _european_note(inputs: Inputs) -> str:
    """Caption for views that always use Black-Scholes-Merton, whatever the sidebar model."""
    model = inputs.model
    if model.kind == "bsm":
        return ""
    early = " Early exercise is ignored here." if model.is_american else ""
    used = "the price cards and the model diagnostics"
    if model.kind == "alo":  # the engine's Greeks replace the analytical ones
        used = "the price cards, the Greeks and the model diagnostics"
    return (
        f"Closed-form Black-Scholes-Merton values for European exercise, not the sidebar's {model.label} "
        f"(used for {used}).{early}"
    )


# --- One-day decay --------------------------------------------------------------------------------------------
# Theta / 365 is the instantaneous rate of decay. Inside a few days of expiry it is far from what a day costs: a
# 1-day at-the-money SPX put (5,800, 18 % volatility, rate 4.3 %, dividend yield 1.3 %) worth $21.56 shows theta
# -$10.66 a day, and loses all $21.56 by expiry. The decay below reprices the option instead.


def to_expiry(expiry: float) -> bool:
    """Whether an option with ``expiry`` years left expires within the next calendar day, so that its one-day decay
    is its decay to expiry."""
    return expiry <= ONE_DAY


def closed_form_decay(S, K, T, r, sigma, q, option_type: str):
    """Black-Scholes-Merton one-day decay ($ per share): the value with one calendar day less to expiry, all else
    equal, minus the value now. With a day or less to expiry, the value at expiry (the intrinsic value
    ``max(±(S − K), 0)``) minus the value now. Broadcasts like :func:`black_scholes.price`."""
    later = bs.price(S, K, np.maximum(np.asarray(T, dtype=float) - ONE_DAY, 0.0), r, sigma, q, option_type)
    return later - bs.price(S, K, T, r, sigma, q, option_type)


def later_trees(steps: int, expiry: float, method: str) -> tuple[tuple[int, float], ...]:
    """The trees that value an option one calendar day later, as ``(steps, weight)`` pairs whose weights add up to
    one (``steps`` and ``expiry`` are the current tree's, ``expiry`` more than a day).

    Measured on European options against the closed-form decay (spot 100, strikes 80 to 120 in steps of 0.5, rate
    5 %, no dividends, 2 to 60 days and 13 expiries from 75 to 3,650 days, 501 steps), on every one-day decay of at
    least 1 cent per share, at 10 %, 20 % and 50 % volatility:

    * Leisen-Reimer keeps its steps: its error is smooth in the expiry, so the two trees' errors cancel, to within
      1.6e-6, 4.0e-6 and 1.2e-5 of the decay (medians 3e-7 or less).
    * The Cox-Ross-Rubinstein error swings with the strike's place among the final nodes. With the same steps the
      two errors did not cancel: up to 4.4 %, 8.5 % and 6.0 % of the decay. The shorter tree therefore keeps the
      time step, which keeps the final nodes where they were: its value is interpolated linearly in the step count
      between the two counts with the current tree's parity (the final nodes of an odd and an even tree interleave)
      that bracket ``steps · (expiry − 1 day) / expiry``, whose time steps bracket the current tree's. Within 0.55 %,
      1.5 % and 0.66 % of the decay (99 % of cases within 0.08 %, 0.12 % and 0.19 %), where the next count up with
      the parity alone was off by up to 2.1 %, 3.0 % and 2.9 %, and the nearest count by up to 1.1 %, 2.3 % and
      2.2 %.
    * Both scale with the steps. At 20 % volatility: with 101 steps, Cox-Ross-Rubinstein up to 11 % (99 % of cases
      within 1.2 %) and Leisen-Reimer 9.7e-5; with 2,000 steps (strikes in steps of 2), 0.062 % and 1.7e-7. With a
      3 % dividend yield and 501 steps, 0.64 % and 4.1e-6.

    The lower count's time step is slightly longer than the current tree's, so its lattice can fail to be
    arbitrage-free where the current one only just is (p lies in (0, 1) exactly while |r − q| Δt < σ√Δt); the
    caller then uses the upper count alone (:func:`tree_decay`), whose time step is never longer. The lower count is
    also left out when it is below one step or carries no weight (a whole number of steps keeps the time step
    exactly).
    """
    if method != "crr":
        return ((steps, 1.0),)
    exact = steps * (expiry - ONE_DAY) / expiry
    lower = math.floor(exact + 1e-9)  # a whole number is not rounded down
    lower -= (lower - steps) % 2
    weight = max(exact - lower, 0.0) / 2.0  # of the upper count, two steps above
    if lower < 1:
        return ((lower + 2, 1.0),)
    if weight < 1e-9:
        return ((lower, 1.0),)
    return ((lower, 1.0 - weight), (lower + 2, weight))


def greeks_decay(inputs: Inputs, prices: Prices) -> dict[str, float]:
    """One-day decay of the call and the put ($ per share) by the model of the Greeks table: the American engine
    under that model, Black-Scholes-Merton otherwise.

    Under a binomial tree or Monte Carlo the Greeks table is Black-Scholes-Merton's, so its decay is too, for
    European exercise: the tree's own, with the tree's exercise style, is in the tree's Greeks (:func:`tree_decay`),
    and the Greeks caption says that an American tree's is there (:func:`_decay_note`). Monte Carlo estimates the
    Black-Scholes-Merton value, whose decay the closed form gives exactly, where the difference of two simulations
    would add sampling error.
    """
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    if inputs.model.kind != "alo":
        return {kind: float(closed_form_decay(S, K, T, r, sigma, q, kind)) for kind in cache.KINDS}
    decay = {}
    for kind, res in prices.alo.items():
        if to_expiry(T):
            later = max(bs.option_sign(kind) * (S - K), 0.0)
        else:
            later = cache.alo_value(S, K, T - ONE_DAY, r, sigma, q, kind, inputs.model.scheme)
        decay[kind] = later - res.value
    return decay


def tree_decay(inputs: Inputs, prices: Prices) -> tuple[dict[str, float], tuple[tuple[int, float], ...]]:
    """One-day decay of the call and the put ($ per share) on the sidebar's tree, and the shorter trees it used.

    The value on trees one calendar day shorter (:func:`later_trees`, weighted) minus the value on the tree now,
    with the tree's exercise style; with a day or less to expiry, the intrinsic value minus the value now (and no
    trees). Where the lower of two Cox-Ross-Rubinstein counts has no arbitrage-free lattice, the upper one alone.
    """
    model = inputs.model
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    if to_expiry(T):
        later = {kind: max(bs.option_sign(kind) * (S - K), 0.0) for kind in cache.KINDS}
        return {kind: later[kind] - prices.tree[kind].price for kind in cache.KINDS}, ()

    def value(steps: int) -> dict[str, float]:
        return cache.tree_prices(S, K, T - ONE_DAY, r, sigma, q, steps, model.american, model.tree_method)

    trees = later_trees(prices.tree["call"].steps, T, model.tree_method)
    try:
        values = [(value(steps), weight) for steps, weight in trees]
    except ValueError:  # the lower count's longer time step has no arbitrage-free lattice
        trees = ((trees[-1][0], 1.0),)
        values = [(value(trees[0][0]), 1.0)]
    decay = {
        kind: math.fsum(weight * later[kind] for later, weight in values) - prices.tree[kind].price
        for kind in cache.KINDS
    }
    return decay, trees


def _decay_row(decay: dict[str, float], expiring: bool) -> dict:
    """The one-day decay as a row of the Greeks table."""
    return {
        "Greek": decay_label(expiring),
        "Call": decay["call"],
        "Put": decay["put"],
        "Unit": "until expiry, by repricing" if expiring else "over the next calendar day, by repricing",
        "Definition": "V(expiry) − V" if expiring else "V(T − 1 day) − V(T)",
        "Order": 1,
    }


def _with_decay(rows: list[dict], theta: dict[str, float], decay: dict[str, float], expiring: bool) -> list[dict]:
    """``rows`` with the one-day decay next to the theta row: before it when it leads (:func:`decay_leads`) for the
    call or the put, after it otherwise. ``theta`` is theta / 365 of each option."""
    leads = any(decay_leads(theta[kind], decay[kind], expiring) for kind in cache.KINDS)
    at = next(i for i, row in enumerate(rows) if row["Greek"].startswith("Theta")) + (0 if leads else 1)
    return [*rows[:at], _decay_row(decay, expiring), *rows[at:]]


def _greeks_table(inputs: Inputs, higher_order: bool, decay: dict[str, float]) -> pd.DataFrame:
    args = (inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield)
    call = bs.greeks(*args, "call")
    put = bs.greeks(*args, "put")
    rows = []
    for name, info in bs.GREEKS.items():
        if info.order > 1 and not higher_order:
            continue
        rows.append(
            {
                "Greek": f"{info.label} {info.symbol}".strip(),
                "Call": call[name] * info.scale,
                "Put": put[name] * info.scale,
                "Unit": info.unit,
                "Definition": info.description,
                "Order": info.order,
            }
        )
    theta = {"call": call["theta"] / 365.0, "put": put["theta"] / 365.0}
    return pd.DataFrame(_with_decay(rows, theta, decay, to_expiry(inputs.expiry)))


def _alo_greeks_table(inputs: Inputs, prices: Prices, decay: dict[str, float]) -> pd.DataFrame:
    """The American engine's Greeks (``american.greeks``) in the layout and units of
    :func:`_greeks_table`."""
    rows = []
    for name in ALO_GREEKS:
        info = bs.GREEKS[name]
        rows.append(
            {
                "Greek": f"{info.label} {info.symbol}".strip(),
                "Call": prices.alo_greeks["call"][name] * info.scale,
                "Put": prices.alo_greeks["put"][name] * info.scale,
                "Unit": info.unit,
                "Definition": info.description,
                "Order": info.order,
            }
        )
    theta = {kind: prices.alo_greeks[kind]["theta"] / 365.0 for kind in cache.KINDS}
    return pd.DataFrame(_with_decay(rows, theta, decay, to_expiry(inputs.expiry)))


#: How the Greeks table under the American engine is computed (american.GREEK_BUMPS).
ALO_GREEKS_NOTE = (
    "American Greeks of the sidebar engine, in the units shown (theta per calendar day of elapsed time): central "
    "finite differences of its value with the spot and the time to expiry moved by ±0.01%, the volatility by "
    "±0.01\xa0vol pt and the rate and dividend yield by ±1\xa0bp, the scheme and the boundary equation held fixed. "
    "Second- and third-order Greeks are shown for Black-Scholes-Merton only."
)
#: At a rate and dividend yield of zero both options are European, at the edge of early exercise (american.greeks).
ALO_ZERO_CARRY_NOTE = (
    "At a rate and dividend yield of zero, a 1\xa0bp move one way makes early exercise pay (a lower rate or a higher "
    "dividend yield for the call, a higher rate or a lower dividend yield for the put), so rho and dividend rho are "
    "one-sided here: the slopes on the side where it does not."
)
#: The Greeks as the caption names them.
_GREEK_WORDS = {"epsilon": "dividend rho"}


def _join(words: list[str]) -> str:
    return " and ".join(words) if len(words) < 3 else ", ".join(words[:-1]) + " and " + words[-1]


def _alo_greeks_note(inputs: Inputs, prices: Prices) -> str:
    """The caption of the Greeks table under the American engine: how they are computed, and for which option
    each exception holds (``AmericanGreeks.basis`` and ``one_sided``, and the fallback to the tree)."""
    greeks, details = prices.alo_greeks, prices.alo
    notes = [ALO_GREEKS_NOTE, _decay_note(inputs)]
    closed = [kind for kind in cache.KINDS if greeks[kind].basis == "closed form"]
    if len(closed) == 2:
        notes.append(
            "The call and the put equal their Black-Scholes-Merton values, so their Greeks are the closed-form ones."
        )
    elif closed:
        notes.append(f"The {closed[0]} equals its Black-Scholes-Merton value, so its Greeks are the closed-form ones.")
    if len(closed) == 2 and inputs.rate == 0.0 and inputs.dividend_yield == 0.0:
        notes.append(ALO_ZERO_CARRY_NOTE)
    for kind in cache.KINDS:
        if greeks[kind].basis == "exercise value":
            side, payoff, delta = ("above", "S − K", "1") if kind == "call" else ("below", "K − S", "−1")
            notes.append(
                f"The {kind} is exercised at once at this spot (at or {side} its critical spot), so its Greeks are "
                f"those of its exercise value {payoff}: delta {delta}, the others zero."
            )
        if sided := [_GREEK_WORDS.get(name, name) for name in greeks[kind].one_sided]:
            notes.append(
                f"The {kind}'s {_join(sided)} {'is' if len(sided) == 1 else 'are'} one-sided: a bump one way would "
                "change how the option is valued (a different exercise regime, the tree instead of the engine or "
                "the reverse, or exercise at once), so the difference is taken on the other side alone."
            )
        if details[kind].fallback:
            notes.append(
                f"The {kind}'s are differences of tree values (see the warning above), far less smooth than the "
                "engine's."
            )
    return " ".join(notes)


def _sensitivity(inputs: Inputs) -> None:
    quantities = {
        "Value": ("value", 1.0, "Option value ($)"),
        "Delta": ("delta", 1.0, "Delta"),
        "Gamma": ("gamma", 1.0, "Gamma"),
        # Theta / 365 with the 1-day decay beside it (dashed); the decay alone, repriced one calendar day later
        # (closed_form_decay), against the days to expiry shows where theta / 365 stops being the cost of a day.
        "Theta / day": ("theta", 1 / 365, "Theta / 365 and 1-day decay ($)"),
        "1-day decay": ("decay", 1.0, "1-day decay ($)"),
        "Vega / vol pt": ("vega", 0.01, "Vega ($ per vol point)"),
        "Rho / 1%": ("rho", 0.01, "Rho ($ per 1% rate)"),
    }
    axes = {"Spot price": "spot", "Volatility": "vol", "Days to expiry": "days"}
    c1, c2 = st.columns(2)
    quantity = c1.selectbox("Quantity", list(quantities), key="sens_quantity", persist_state="session")
    axis = c2.selectbox("Against", list(axes), key="sens_axis", persist_state="session")
    name, scale, y_title = quantities[quantity]

    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    kind = axes[axis]
    if kind == "spot":
        lo, hi = 0.5 * min(S, K), 1.5 * max(S, K)
        x = np.linspace(lo, hi, 301)
        args, current, strike_x, x_title = (x, K, T, r, sigma, q), S, K, "Spot price"
    elif kind == "vol":
        x = np.linspace(0.01, max(1.0, 2.5 * sigma), 301)
        args, current, strike_x, x_title = (S, K, T, r, x, q), sigma * 100, None, "Volatility (%)"
    else:
        x = np.linspace(1, max(inputs.days * 1.5, 30), 301)
        args, current, strike_x, x_title = (S, K, x / 365.0, r, sigma, q), inputs.days, None, "Days to expiry"

    def curve(option_type):
        if name == "value":
            return bs.price(*args, option_type)
        if name == "decay":
            return closed_form_decay(*args, option_type)
        return bs.greeks(*args, option_type)[name] * scale

    # Against spot or volatility every point has the sidebar's expiry, so within a day of it the decay is the decay
    # to expiry; against the days left only the first point (1 day) is, and "1-day decay" covers it.
    decay = decay_label(kind != "days" and to_expiry(T))
    dashed = None
    if name == "decay":
        y_title = f"{decay} ($)"
    elif name == "theta":
        # The repriced day beside the rate, dashed: they part within a few days of expiry.
        y_title = f"Theta / 365 and {decay.lower()} ($)"
        dashed = ("Theta / 365", decay, closed_form_decay(*args, "call"), closed_form_decay(*args, "put"))
    x_plot = x * 100 if kind == "vol" else x
    fig = charts.sensitivity_chart(
        x_plot, curve("call"), curve("put"), x_title, y_title, current, strike_x, dashed=dashed
    )
    charts.show(fig, key="sensitivity_chart", alt=f"Call and put {y_title.lower()} against {x_title.lower()}")
    if note := _european_note(inputs):
        st.caption(note)


def _heatmaps(inputs: Inputs) -> None:
    c1, c2 = st.columns(2)
    spot_span = c1.slider("Spot range (±%)", 5, 50, 20, 5, key="hm_spot", persist_state="session") / 100
    vol_span = c2.slider("Volatility range (±%)", 10, 90, 50, 10, key="hm_vol", persist_state="session") / 100
    spots = inputs.spot * np.linspace(1 - spot_span, 1 + spot_span, 9)
    vols = np.maximum(inputs.volatility * np.linspace(1 - vol_span, 1 + vol_span, 9), 1e-4)
    grid_s, grid_v = np.meshgrid(spots, vols)
    args = (inputs.strike, inputs.expiry, inputs.rate)
    # One above the other at full width: side by side, each cell was too narrow for its value (5 to 9 px text).
    for kind in ("call", "put"):
        values = bs.price(grid_s, args[0], args[1], args[2], grid_v, inputs.dividend_yield, kind)
        charts.show(
            charts.price_heatmap(spots, vols, values, f"{kind.capitalize()} value ($)"),
            key=f"heatmap_{kind}",
            alt=f"Heatmap of the {kind} value ($) over spot price and volatility",
        )
    if note := _european_note(inputs):
        st.caption(note)


def _chain(inputs: Inputs) -> None:
    """Black-Scholes-Merton values and key Greeks across strikes at the sidebar expiry and volatility."""
    S, T, r, sigma, q = inputs.spot, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield
    strikes = np.unique(np.round(inputs.strike * np.linspace(0.8, 1.2, 9), 2))
    call = bs.greeks(S, strikes, T, r, sigma, q, "call")
    put = bs.greeks(S, strikes, T, r, sigma, q, "put")
    decay = decay_label(to_expiry(T))
    chain = pd.DataFrame(
        {
            "Call value": bs.price(S, strikes, T, r, sigma, q, "call"),
            "Call delta": call["delta"],
            "Call theta / day": call["theta"] / 365,
            f"Call {decay.lower()}": closed_form_decay(S, strikes, T, r, sigma, q, "call"),
            "Strike": strikes,
            "Put value": bs.price(S, strikes, T, r, sigma, q, "put"),
            "Put delta": put["delta"],
            "Put theta / day": put["theta"] / 365,
            f"Put {decay.lower()}": closed_form_decay(S, strikes, T, r, sigma, q, "put"),
            "Gamma": call["gamma"],
            "Vega / vol pt": call["vega"] / 100,
        }
    )
    st.dataframe(
        chain,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="%.2f"),
            "Call value": st.column_config.NumberColumn(format="%.4f"),
            "Put value": st.column_config.NumberColumn(format="%.4f"),
            "Call delta": st.column_config.NumberColumn(format="%.3f"),
            "Put delta": st.column_config.NumberColumn(format="%.3f"),
            "Call theta / day": st.column_config.NumberColumn(format="%.4f"),
            "Put theta / day": st.column_config.NumberColumn(format="%.4f"),
            f"Call {decay.lower()}": st.column_config.NumberColumn(format="%.4f"),
            f"Put {decay.lower()}": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Vega / vol pt": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Strikes from 80% to 120% of the sidebar strike, flat volatility (no smile), same expiry. Theta / day is "
        f"theta / 365; the {decay.lower()} reprices each option "
        + ("at expiry (its intrinsic value). " if to_expiry(T) else "with one calendar day less to expiry. ")
        + _european_note(inputs)
    )


def _tree_repricing(steps: int, expiry: float, method: str, trees: tuple[tuple[int, float], ...]) -> str:
    """How the tree table's decay reprices, from the shorter trees it used (``trees``, from :func:`tree_decay`).

    It says "this tree's time step" only where the shorter trees keep it: two trees interpolated to it, or one
    whose time step equals it to 1e-9; otherwise it says how much shorter the one tree's time step is, and why.
    """
    if not trees:
        return "the intrinsic value at expiry minus the tree value"
    if method != "crr":
        return (
            "the value on a tree one day shorter with the same steps minus the value now (Leisen-Reimer's error is "
            "smooth in the expiry, so the two trees' errors cancel)"
        )
    exact = steps * (expiry - ONE_DAY) / expiry  # the steps of a tree one day shorter with this tree's time step
    kept = "(at the same time step the final nodes stay in place, so the trees' oscillating errors largely cancel)"
    if len(trees) == 2:
        (lower, _), (upper, _) = trees
        return (
            f"the value one day later, interpolated between trees with {lower:,} and {upper:,} steps to this tree's "
            f"time step ({exact:,.2f} steps), minus the value now {kept}"
        )
    ((later, _),) = trees
    shorter = 1.0 - exact / later  # how much shorter its time step is than this tree's, as a fraction
    if abs(shorter) < 1e-9:
        return (
            f"the value on a tree one day shorter with {later:,} steps, this tree's time step, minus the value now "
            + kept
        )
    why = (
        "the next count down has no arbitrage-free lattice here"
        if len(later_trees(steps, expiry, method)) == 2
        else "no smaller count keeps it"
    )
    return (
        f"the value on a tree one day shorter with {later:,} steps, a time step {shorter:.2%} shorter than this "
        f"tree's ({why}), minus the value now"
    )


def _tree_diagnostics(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    st.markdown("**Tree Greeks** (on the lattice, so they include the early-exercise feature)")
    decay, trees = tree_decay(inputs, prices)
    expiring = to_expiry(T)
    label = decay_label(expiring)
    # The headline rule of the Greeks table, on the tree's own theta: the decay column comes first when it leads.
    leads = any(decay_leads(res.theta / 365, decay[kind], expiring) for kind, res in prices.tree.items())
    rows = []
    for kind, res in prices.tree.items():
        time = {"Theta / day": res.theta / 365, label: decay[kind]}
        rows.append(
            {"Option": kind.capitalize(), "Value": res.price, "Delta": res.delta, "Gamma": res.gamma}
            | (dict(reversed(time.items())) if leads else time)
            | {"Vega / vol pt": res.vega / 100, "Rho / 1%": res.rho / 100, "Div. rho / 1%": res.epsilon / 100}
        )
    steps_used = prices.tree["call"].steps
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "Value": st.column_config.NumberColumn(format="%.4f"),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Theta / day": st.column_config.NumberColumn(format="%.4f"),
            label: st.column_config.NumberColumn(format="%.4f"),
            "Vega / vol pt": st.column_config.NumberColumn(format="%.4f"),
            "Rho / 1%": st.column_config.NumberColumn(format="%.4f"),
            "Div. rho / 1%": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    first = (
        f" The {label.lower()} comes first: "
        + (
            "the option expires within the day."
            if expiring
            else f"it differs from theta / 365 by more than {DECAY_SWITCH:.0%} of the larger, for the call or the put."
        )
        if leads
        else ""
    )
    st.caption(
        f"Dollars per share on a {steps_used:,}-step tree. Delta and gamma come from its first two time steps, "
        "theta from the second step (CRR) or from repricing with the expiry moved (Leisen-Reimer). Theta / day is "
        f"theta / 365; the {label.lower()} is {_tree_repricing(steps_used, T, model.tree_method, trees)}.{first} "
        "Vega, rho and dividend rho (per 1% move in the dividend yield) reprice on a tree with the same steps, with "
        "the volatility moved ±0.5 vol pt and the rates ±25 bp."
    )

    st.markdown("**Convergence with the number of steps**")
    series, references = cache.tree_convergence(S, K, T, r, sigma, q, model.steps, model.american)
    reference_label = "Converged" if model.american else "Black-Scholes-Merton"
    reference = "converged American value" if model.american else "Black-Scholes-Merton value"
    for col, kind in zip(st.columns(2), ("call", "put"), strict=True):
        charts.show(
            charts.convergence_chart(series[kind], references[kind], reference_label, kind.capitalize()),
            key=f"tree_convergence_{kind}",
            alt=f"{kind.capitalize()} value on each lattice against the number of tree steps, with the {reference}",
            container=col,
        )
    st.caption(
        f"Step counts from 10 to the sidebar's {model.steps:,}. CRR oscillates between odd and even step counts; "
        "Leisen-Reimer converges smoothly, "
        + (
            "at first order with early exercise. The converged value extrapolates Leisen-Reimer trees with 2,001 "
            "and 4,003 steps (Richardson), kept within the no-arbitrage bounds of an American option."
            if model.american
            else "at roughly O(1/n²), to the closed-form value."
        )
    )
    if model.american:
        boundaries = {}
        for kind, res in prices.tree.items():
            if res.exercise_boundary is not None and len(res.exercise_boundary[0]):
                boundaries[f"{kind.capitalize()} boundary"] = (
                    res.exercise_boundary[0] * 365,
                    res.exercise_boundary[1],
                )
        st.markdown("**Early-exercise boundary**")
        if boundaries:
            charts.show(
                charts.exercise_boundary_chart(boundaries, K),
                key="exercise_boundary",
                alt="Early-exercise boundary: the critical spot price against days from today, with the strike",
            )
            st.caption(
                "Exercising immediately is optimal when the spot is beyond the boundary: below it for a put, "
                "above it for a call. With a non-negative rate and no dividends an American call is never "
                "exercised early. The critical price at each step is interpolated between the last exercised "
                "node and the next one, and consecutive steps are averaged (their nodes interleave)."
            )
        else:
            st.info(
                "Early exercise is never optimal for these inputs, or the spot is already in the exercise region "
                "at every step the tree resolves (e.g. a deep in-the-money put worth its intrinsic value)."
            )


def _monte_carlo_diagnostics(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    times, paths = cache.mc_paths(S, T, r, sigma, q, model.n_paths, model.seed, model.antithetic, SAMPLE_PATHS)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Sample paths** ({len(paths)} of the {model.n_paths:,} scenarios behind the price)")
        charts.show(
            charts.paths_chart(times * 365, paths, K, S * np.exp((r - q) * times), "Risk-neutral mean (forward)"),
            key="mc_paths",
            alt=f"{len(paths)} simulated price paths and their risk-neutral mean (the forward) against days from "
            "today, with the strike",
        )
        st.caption(
            "The price samples each scenario's price at expiry exactly; its path from today is filled in with a "
            "Brownian bridge. The purple line is the forward S·e^((r−q)t), the expected price of a scenario at each "
            "date under the pricing measure; the average of the simulated scenarios differs from it by sampling noise."
        )
    with c2:
        st.markdown("**Terminal price distribution**")
        s_t = cache.mc_terminal(S, T, r, sigma, q, model.n_paths, model.seed, model.antithetic)
        grid = np.linspace(np.percentile(s_t, 0.05), np.percentile(s_t, 99.95), 400)
        mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
        sd = sigma * np.sqrt(T)
        pdf = np.exp(-0.5 * ((np.log(grid) - mu) / sd) ** 2) / (grid * sd * np.sqrt(2 * np.pi))
        charts.show(
            charts.histogram_chart(s_t, "Underlying price at expiry", [(K, "strike", charts.NEUTRAL)], pdf=(grid, pdf)),
            key="mc_terminal",
            alt="Histogram of the simulated underlying prices at expiry, with the log-normal density and the strike",
        )
    rows = [
        {
            "Option": kind.capitalize(),
            "Monte Carlo": res.price,
            "Std. error": res.std_error,
            "95% CI low": res.ci95[0],
            "95% CI high": res.ci95[1],
            "BSM": bs.price(S, K, T, r, sigma, q, kind),
            "Error / SE": (res.price - bs.price(S, K, T, r, sigma, q, kind)) / res.std_error
            if res.std_error > 0
            else np.nan,
            "In the money": res.itm_fraction,
        }
        for kind, res in prices.mc.items()
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        placeholder="",  # undefined values are left blank, as the caption says (Streamlit would write "None")
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in ("Monte Carlo", "Std. error", "95% CI low", "95% CI high", "BSM")
        }
        | {
            "Error / SE": st.column_config.NumberColumn(format="%.2f"),
            "In the money": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Terminal prices are sampled exactly from the log-normal distribution (no time-stepping bias). "
        "|Error / SE| below about 2 is consistent with an unbiased estimator. The standard error is left blank "
        "when the sample cannot estimate it: every price finished out of the money, or every one in the money "
        "with the control variate on. At very large σ√T (hundreds of percent of volatility over years) the mean of "
        "the price at expiry rests on scenarios too rare to draw: when the simulated discounted prices miss their "
        "known mean S·e^(−qT) by more than 5 standard errors, the put, whose payoff is at most its strike, is "
        "averaged directly (no control variate) and the call follows from put-call parity. Without the control "
        "variate the call's own average fails much sooner: from σ√T = 2 its 95% interval misses the value more "
        "often than 1 run in 20 (about 1 in 5 at σ√T = 3), so from σ√T = 2 on the call follows from the put by "
        "parity as well. Every estimate is kept within the no-arbitrage bounds of a European option."
        + (
            " This sample is heavy-tailed: the prices above come from the put's direct average and parity."
            if any(res.heavy_tailed for res in prices.mc.values())
            else ""
        )
    )


def _alo_diagnostics(inputs: Inputs, prices: Prices) -> None:
    """The American engine: how it valued each option, and the early-exercise boundary it solved."""
    K, T, r, sigma, q = inputs.strike, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield
    model = inputs.model
    st.markdown("**Engine results**")

    def solved(value: float, res: AmericanPrice) -> float | None:
        return value if res.method == "alo" and np.isfinite(value) else None

    rows = [
        {
            "Option": kind.capitalize(),
            "American": res.value,
            "European": res.european,
            "Premium": res.premium,
            "Critical spot": solved(res.critical_spot, res),
            "Method": ALO_METHODS.get(res.method, res.method),
            "Scheme used": res.scheme if res.method == "alo" else None,
            "Last step": solved(res.last_step, res),
        }
        for kind, res in prices.alo.items()
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        placeholder="–",
        column_config={
            "American": st.column_config.NumberColumn(format="%.4f"),
            "European": st.column_config.NumberColumn(format="%.4f"),
            "Premium": st.column_config.NumberColumn(format="%.4f"),
            "Critical spot": st.column_config.NumberColumn(format="%.4f"),
            "Last step": st.column_config.NumberColumn(format="%.1e"),
        },
    )
    st.caption(
        "Dollars per share. The premium is the American value minus the closed-form European value. Today the put "
        "is exercised at or below its critical spot, the call at or above it. A stiff option (a long expiry, or "
        "high rates or dividends at low volatility) is solved with a finer scheme than the one chosen. The last "
        "step, the engine's convergence check, is the largest change of the logarithm of the boundary in its final "
        "iteration; above 0.01 the option goes to the tree. A dash: not applicable to how the option was valued."
    )

    st.markdown("**Early-exercise boundary**")
    boundaries, missing = {}, []
    for kind, res in prices.alo.items():
        if res.method == "alo":
            tau, spots = cache.alo_boundary(K, T, r, sigma, q, kind, model.scheme)
            boundaries[f"{kind.capitalize()} boundary"] = ((T - tau[::-1]) * 365, spots[::-1])
        elif res.method == "european":
            missing.append(f"No {kind} boundary: {NEVER_EARLY[kind]}.")
        elif res.fallback:
            missing.append(f"No {kind} boundary: {FALLBACK_REASONS[res.fallback_reason]}.")
        else:
            missing.append(f"No {kind} boundary: the {kind} is valued in closed form here.")
    if boundaries:
        charts.show(
            charts.exercise_boundary_chart(boundaries, K),
            key="exercise_boundary",
            alt="Early-exercise boundary of the American engine: the critical spot price against days from today, "
            "with the strike",
        )
        ripple = (
            " The fast scheme's 8 nodes can make the curve ripple between them, by up to about 1% of its value (2% at "
            "volatilities near 150%); the accurate and high schemes draw it smoothly."
            if model.scheme == "fast"
            else ""
        )
        st.caption(
            "Exercising at once is optimal when the spot is beyond the boundary: at or below it for a put, at or "
            "above it for a call. At expiry it ends at K·min(1, r/q) for a put and K·max(1, r/q) for a call. The "
            f"engine's boundary at 101 dates, closest together near expiry, where it moves fastest.{ripple} "
            + " ".join(missing)
        )
    else:
        st.info(" ".join(missing))


def _model_diagnostics(inputs: Inputs, prices: Prices) -> None:
    if inputs.model.kind == "binomial":
        _tree_diagnostics(inputs, prices)
    elif inputs.model.kind == "monte_carlo":
        _monte_carlo_diagnostics(inputs, prices)
    elif inputs.model.kind == "alo":
        _alo_diagnostics(inputs, prices)
    else:
        st.info(
            "Black-Scholes-Merton is exact for European options under its assumptions. Switch the sidebar model to a "
            "binomial tree, Monte Carlo or the American engine to see convergence diagnostics, early exercise and "
            "simulated paths."
        )


def _decay_note(inputs: Inputs) -> str:
    """The sentences of the Greeks table's caption on its one-day decay row.

    Under an American binomial tree the table, decay included, is the European option's (Black-Scholes-Merton), so
    the note says plainly that it is not the decay of the American value in the price cards, which the tree's own
    Greeks give (:func:`tree_decay`): a 7-day put struck at 110 on 100 is exercised at once on the tree, so its
    value does not change in a day, while the European put gains 1.5 cents.
    """
    model = inputs.model
    expiring = to_expiry(inputs.expiry)
    value = "the engine's value" if model.kind == "alo" else "the value"
    later = "the intrinsic value at expiry" if expiring else f"{value} with one calendar day less to expiry"
    by = {
        "binomial": " by Black-Scholes-Merton for European exercise"
        + ("" if model.american else ", the value the tree converges to (the tree's own is in its Greeks below)"),
        "monte_carlo": " by Black-Scholes-Merton, whose value the simulation estimates",
    }.get(model.kind, "")
    first = (
        "it is listed first, as the option expires within the day"
        if expiring
        else f"it is listed first when it differs from the theta row (theta / 365) by more than {DECAY_SWITCH:.0%} "
        "of the larger, for the call or the put"
    )
    american = (
        " It is not the decay of the American value in the price cards, which the tree's Greeks give."
        if model.kind == "binomial" and model.american
        else ""
    )
    return f"{decay_label(expiring)}: {later} minus {value} now{by}, all else equal; {first}.{american}"


def render(inputs: Inputs, prices: Prices) -> None:
    st.subheader("Greeks")
    decay = greeks_decay(inputs, prices)
    if inputs.model.kind == "alo":
        table = _alo_greeks_table(inputs, prices, decay)
        note = _alo_greeks_note(inputs, prices)
    else:
        higher = st.toggle(
            "Show second- and third-order Greeks", value=False, key="higher_greeks", persist_state="session"
        )
        table = _greeks_table(inputs, higher, decay)
        note = (
            "Analytical Black-Scholes-Merton Greeks for European exercise, in the units shown "
            "(theta and the other time Greeks per calendar day of elapsed time). "
            + _decay_note(inputs)
            + (
                " Tree Greeks for the American option, including vega and rho, are under *Model diagnostics* below."
                if inputs.model.american and inputs.model.kind == "binomial"
                else ""
            )
        )
    st.dataframe(
        table.drop(columns="Order"),
        hide_index=True,
        column_config={
            # A fixed width: the grid sizes columns from the first rows it draws (and drops the longest name as an
            # outlier), which cut "Epsilon (dividend rho) ε" and, after the toggle, "Vomma (volga)" on phones.
            "Greek": st.column_config.TextColumn(width="medium"),
            "Call": st.column_config.NumberColumn(format="%.6f"),
            "Put": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.caption(note)

    st.subheader("Sensitivity explorer")
    _sensitivity(inputs)

    with st.expander("Price heatmaps (spot × volatility)"):
        _heatmaps(inputs)

    with st.expander("Theoretical option chain"):
        _chain(inputs)

    st.subheader("Model diagnostics")
    st.caption(inputs.model.label)
    _model_diagnostics(inputs, prices)
