"""The VaR and ES backtests (options_analyzer/backtest.py, roadmap item RK-4).

Every acceptance number of the item is checked against a reference computed here, not by the module: likelihood ratios
from their formulas with math.log, binomial tails from scipy.stats.binom, the Basel tables as printed (BCBS 1996 Tables
1 and 2, MAR32.9 Table 1, MAR32.42), Acerbi and Szekely's (2014) Table 4, and the minimum of the FZ0 loss from the
Normal's closed-form quantile and ES. Property tests check the identities (statistics never negative, p-values in
[0, 1], LR_cc = LR_uc + LR_ind). Simulations use fixed seeds, and each tolerance states its Monte Carlo error. The
ten-year backtest of the synthetic history is marked slow; a three-year version runs by default.
"""

from __future__ import annotations

import doctest
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import optimize, stats
from scipy.special import ndtri

from options_analyzer import backtest as bt
from options_analyzer import black_scholes as bs
from options_analyzer import risk
from options_analyzer import strategies as S
from options_analyzer.market_data import SYNTHETIC_LABEL

ROOT = Path(__file__).resolve().parents[1]
RISK_METHODOLOGY = ROOT / "options_analyzer" / "ui" / "methodology" / "risk.md"
METHOD_NOTE = ROOT / "docs" / "methods" / "backtest.md"


def _text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _loglik(count: float, probability: float) -> float:
    """``count * ln(probability)`` with 0 ln 0 = 0, as the likelihoods of the papers need."""
    return count * math.log(probability) if count else 0.0


def kupiec_reference(x: int, n: int, p: float) -> float:
    """Kupiec's LR_POF term by term (his eq. for the proportion of failures)."""
    rate = x / n
    return -2.0 * (_loglik(n - x, 1 - p) + _loglik(x, p) - _loglik(n - x, 1 - rate) - _loglik(x, rate))


def christoffersen_reference(n00: int, n01: int, n10: int, n11: int) -> float:
    """Christoffersen's LR_ind from the transition counts."""
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0  # a probability with no observation: its terms are all 0 ln 0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    iid = _loglik(n00 + n10, 1 - pi) + _loglik(n01 + n11, pi)
    markov = _loglik(n00, 1 - pi01) + _loglik(n01, pi01) + _loglik(n10, 1 - pi11) + _loglik(n11, pi11)
    return -2.0 * (iid - markov)


# --- Kupiec (acceptance 1 and 6) ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("exceptions", "lr"), [(0, 5.0252), (4, 0.7691), (5, 1.9568), (10, 12.9555)])
def test_kupiec_statistic_at_250_days_and_99_percent(exceptions, lr):
    res = bt.kupiec_pof(exceptions, 250, 0.01)
    reference = kupiec_reference(exceptions, 250, 0.01)
    # scipy's xlogy against math.log term by term: the same numbers up to rounding (identical at these four; at most
    # 3e-14 relative over all 251 outcomes, the largest where the statistic nears 0 and its terms cancel).
    assert res.lr == pytest.approx(reference, rel=1e-12)
    assert round(res.lr, 4) == lr
    assert res.p_value == pytest.approx(stats.chi2.sf(reference, 1), rel=1e-10)
    assert res.binomial_tail == pytest.approx(stats.binom.sf(exceptions - 1, 250, 0.01), rel=1e-12)
    assert (res.exceptions, res.observations, res.expected) == (exceptions, 250, 2.5)


def test_ten_exceptions_reject_asymptotically_and_exactly():
    res = bt.kupiec_pof(10, 250, 0.01)
    assert round(res.p_value, 4) == 0.0003  # chi-square(1): 3.19e-4
    assert round(res.binomial_tail, 4) == 0.0003  # P(X >= 10) = 2.50e-4
    # The exact p-value of the LR statistic: every number of exceptions whose LR is at least 12.9555. Only 10 and
    # more qualify (no exception at all scores 5.03), so it is the binomial tail.
    every = range(251)
    larger = [k for k in every if kupiec_reference(k, 250, 0.01) >= kupiec_reference(10, 250, 0.01)]
    assert larger == list(range(10, 251))
    assert res.exact_p_value == pytest.approx(stats.binom.sf(9, 250, 0.01), rel=1e-12)


@pytest.mark.parametrize("p", [0.01, 0.025])
def test_kupiec_rejects_a_true_var_at_its_nominal_rate(p):
    # Acceptance 6: 200 replications of 2,500 iid N(0, 1) days scored against the true VaR. The rejection rate at 5 %
    # must lie in the binomial 95 % band of 200 tests around 5 %, 4 to 16 rejections. (The test's exact size at 2,500
    # days, from the binomial, is 4.37 % at 99 % and 4.68 % at 97.5 %; measured 11 and 9 rejections at this seed.)
    rng = np.random.default_rng(6)
    exceptions = (rng.standard_normal((200, 2_500)) < ndtri(p)).sum(axis=1)
    rejections = sum(bt.kupiec_pof(int(x), 2_500, p).p_value < 0.05 for x in exceptions)
    low, high = stats.binom.interval(0.95, 200, 0.05)
    assert low <= rejections <= high
    assert bt.power_kupiec(2_500, p, p) == pytest.approx({0.01: 0.0437, 0.025: 0.0468}[p], abs=1e-4)


def test_kupiec_actual_size_and_power_in_a_year():
    # The rejection region of the 5 % test at 250 days and 99 %, from the reference statistic: no exception at all, or
    # seven and more. Its probability is the actual size (9.5 %, not 5 %) and, at a true rate of 2 %, the power (24 %).
    critical = stats.chi2.isf(0.05, 1)
    region = [k for k in range(251) if kupiec_reference(k, 250, 0.01) > critical]
    assert region == [0, *range(7, 251)]

    def rejected(p_true):
        return stats.binom.pmf(0, 250, p_true) + stats.binom.sf(6, 250, p_true)

    assert bt.power_kupiec(250, 0.01, 0.01) == pytest.approx(rejected(0.01), rel=1e-12)
    assert bt.power_kupiec(250, 0.02, 0.01) == pytest.approx(rejected(0.02), rel=1e-12)
    assert round(rejected(0.01), 4) == 0.0948 and round(rejected(0.02), 4) == 0.2427
    assert bt.power_kupiec(2_520, 0.02, 0.01) == pytest.approx(0.9866, abs=1e-4)  # ten years


@given(
    n=st.integers(1, 3_000),
    fraction=st.floats(0.0, 1.0),
    p=st.floats(1e-4, 0.5),
)
def test_kupiec_statistic_and_p_values_are_in_range(n, fraction, p):
    x = round(fraction * n)
    res = bt.kupiec_pof(x, n, p)
    assert res.lr >= 0.0
    assert 0.0 <= res.p_value <= 1.0 and 0.0 <= res.exact_p_value <= 1.0 and 0.0 <= res.binomial_tail <= 1.0
    assert res.lr == pytest.approx(kupiec_reference(x, n, p), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    ("args", "message"),
    [((5, 0), "observations"), ((-1, 250), "exceptions"), ((251, 250), "cannot exceed"), ((2.5, 250), "exceptions")],
)
def test_kupiec_refuses_impossible_counts(args, message):
    with pytest.raises(ValueError, match=message):
        bt.kupiec_pof(*args)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        bt.kupiec_pof(1, 250, 1.0)


# --- The traffic light (acceptance 2) ------------------------------------------------------------------------------

#: BCBS (1996), Table 2: the cumulative probability (%) of 0 to 10 exceptions in 250 days at 99 % coverage, as printed.
BCBS_TABLE_2 = [8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99]
#: MAR32.9 Table 1: the multiplier for 0 to 11 exceptions in 250 days.
MAR32_TABLE_1 = [1.50, 1.50, 1.50, 1.50, 1.50, 1.70, 1.76, 1.83, 1.88, 1.92, 2.00, 2.00]


def test_the_traffic_light_reproduces_the_basel_tables():
    # Acceptance 2: the binomial cdf at 4 and 9 exceptions, the edges of the green and amber zones.
    assert round(stats.binom.cdf(4, 250, 0.01), 4) == 0.8922 and round(stats.binom.cdf(9, 250, 0.01), 4) == 0.9997
    lights = [bt.traffic_light(k) for k in range(12)]
    for light, printed in zip(lights, BCBS_TABLE_2, strict=False):
        assert light.cumulative_probability == pytest.approx(stats.binom.cdf(light.exceptions, 250, 0.01), rel=1e-12)
        assert abs(100 * light.cumulative_probability - printed) <= 0.005 + 1e-9  # printed to two decimals
    assert [light.zone for light in lights] == ["green"] * 5 + ["amber"] * 5 + ["red"] * 2
    assert [light.multiplier for light in lights] == MAR32_TABLE_1
    assert {(light.amber_from, light.red_from) for light in lights} == {(5, 10)}
    assert bt.traffic_light(250).zone == "red" and bt.traffic_light(250).multiplier == 2.00


@pytest.mark.parametrize(("days", "coverage"), [(500, 0.99), (1_000, 0.99), (2_020, 0.99), (250, 0.975), (750, 0.975)])
def test_other_sample_sizes_follow_the_bcbs_cumulative_probability_rule(days, coverage):
    # BCBS (1996): for other sample sizes, amber begins where the cumulative probability reaches 95 % and red where it
    # reaches 99.99 %. Found here by walking the binomial cdf; MAR32's multipliers apply at 250 days and 99 % only.
    cdf = [stats.binom.cdf(k, days, 1 - coverage) for k in range(days + 1)]
    amber = next(k for k, c in enumerate(cdf) if c >= 0.95)
    red = next(k for k, c in enumerate(cdf) if c >= 0.9999)
    light = bt.traffic_light(0, days, coverage=coverage)
    assert (light.amber_from, light.red_from) == (amber, red)
    assert [bt.traffic_light(k, days, coverage=coverage).zone for k in (amber - 1, amber, red - 1, red)] == [
        "green", "amber", "amber", "red",
    ]  # fmt: skip
    assert light.multiplier is None


def test_traffic_light_error_rates_match_bcbs_1996_table_1():
    # BCBS (1996) Table 1, printed to 0.1 %: with 250 days, a correct 99 % model leaves green 10.8 % of the time; a
    # model whose true coverage is 98 %, 97 %, 96 % or 95 % stays green ("type 2" at a cut-off of 5) 43.9 %, 12.8 %,
    # 2.7 % or 0.5 % of the time, and out of the red zone (cut-off 10) 97.0 %, 77.9 %, 45.5 % or 19.5 %.
    printed = {0.02: (43.9, 97.0), 0.03: (12.8, 77.9), 0.04: (2.7, 45.5), 0.05: (0.5, 19.5)}
    assert abs(100 * bt.power_traffic_light(250, 0.01).amber_or_red - 10.8) <= 0.05
    for p_true, (green, not_red) in printed.items():
        power = bt.power_traffic_light(250, p_true)
        assert abs(100 * (1 - power.amber_or_red) - green) <= 0.05 + 1e-9
        assert abs(100 * (1 - power.red) - not_red) <= 0.05 + 1e-9


@given(days=st.integers(1, 2_000), coverage=st.floats(0.9, 0.999), fraction=st.floats(0.0, 1.0))
def test_the_zone_bounds_are_where_the_cumulative_probability_crosses(days, coverage, fraction):
    # Amber begins at the first count whose cumulative probability reaches 95 %, red at 99.99 % (the count before
    # stays below), so the zones never improve with more exceptions.
    p = 1 - coverage
    light = bt.traffic_light(round(fraction * days), days, coverage=coverage)
    for start, level in [(light.amber_from, 0.95), (light.red_from, 0.9999)]:
        assert stats.binom.cdf(start, days, p) >= level - 1e-12
        assert start == 0 or stats.binom.cdf(start - 1, days, p) < level
    assert light.amber_from <= light.red_from
    x = light.exceptions
    assert light.zone == ("red" if x >= light.red_from else "amber" if x >= light.amber_from else "green")


# --- Christoffersen (acceptance 3) ---------------------------------------------------------------------------------


def test_christoffersen_flags_five_exceptions_in_a_row():
    # Acceptance 3: exceptions on days 100 to 104 of 250. Five is the right number for 99 % (2.5 expected; Kupiec
    # does not reject), but they come together.
    hits = np.zeros(250, dtype=bool)
    hits[99:104] = True
    res = bt.christoffersen(hits, 0.01)
    assert (res.n00, res.n01, res.n10, res.n11) == (243, 1, 1, 4)
    assert res.lr_ind == pytest.approx(christoffersen_reference(243, 1, 1, 4), rel=1e-12)
    assert round(res.lr_ind, 3) == 30.985
    assert res.p_ind == pytest.approx(stats.chi2.sf(res.lr_ind, 1), rel=1e-10) and res.p_ind < 1e-7
    # Conditioned on the first day, as in the paper: LR_cc = LR_uc on the last 249 days + LR_ind.
    assert res.lr_cc == pytest.approx(kupiec_reference(5, 249, 0.01) + res.lr_ind, rel=1e-12)
    assert res.p_cc == pytest.approx(stats.chi2.sf(res.lr_cc, 2), rel=1e-10)
    assert bt.kupiec_pof(5, 250, 0.01).p_value > 0.1


def test_christoffersen_without_exceptions_or_with_one_at_the_end():
    quiet = bt.christoffersen(np.zeros(100, dtype=bool), 0.01)
    assert (quiet.n00, quiet.lr_ind, quiet.p_ind) == (99, 0.0, 1.0)
    assert quiet.lr_cc == pytest.approx(kupiec_reference(0, 99, 0.01), rel=1e-12)
    last = bt.christoffersen([0] * 99 + [1], 0.01)  # nothing follows the exception: pi11 has no observation
    assert (last.n00, last.n01, last.n10, last.n11, last.lr_ind) == (98, 1, 0, 0, 0.0)


@given(hits=st.lists(st.booleans(), min_size=2, max_size=400), p=st.floats(1e-3, 0.5))
def test_christoffersen_statistics_and_identity(hits, p):
    res = bt.christoffersen(hits, p)
    assert res.n00 + res.n01 + res.n10 + res.n11 == len(hits) - 1
    assert res.lr_ind >= 0 and res.lr_cc >= 0 and 0 <= res.p_ind <= 1 and 0 <= res.p_cc <= 1
    assert res.lr_ind == pytest.approx(christoffersen_reference(res.n00, res.n01, res.n10, res.n11), abs=1e-8)
    uc = bt.kupiec_pof(res.n01 + res.n11, len(hits) - 1, p).lr
    assert res.lr_cc == pytest.approx(uc + res.lr_ind, rel=1e-9, abs=1e-9)


def test_christoffersen_power_simulation_matches_exact_enumeration():
    # Eight days of a Markov chain (rate 20 %, probability 50 % of an exception after one): enumerate all 256 paths,
    # weight each by its probability under the chain, and add up those the tests reject. The simulation (40,000
    # paths) must agree within 4 standard errors (at most 0.0025 each).
    p, p11, n = 0.2, 0.5, 8
    p01 = p * (1 - p11) / (1 - p)
    exact_ind = exact_cc = 0.0
    for path in itertools.product((False, True), repeat=n):
        weight = p if path[0] else 1 - p
        for before, after in itertools.pairwise(path):
            stay = p11 if before else p01
            weight *= stay if after else 1 - stay
        res = bt.christoffersen(path, p)
        exact_ind += weight * (res.lr_ind > stats.chi2.isf(0.05, 1))
        exact_cc += weight * (res.lr_cc > stats.chi2.isf(0.05, 2))
    power = bt.power_christoffersen(n, p, p11, p, n_sims=40_000, seed=3)
    tolerance = 4 * 0.5 / math.sqrt(40_000)
    assert power.independence == pytest.approx(exact_ind, abs=tolerance)
    assert power.conditional_coverage == pytest.approx(exact_cc, abs=tolerance)


def test_christoffersen_in_a_year_is_weak():
    # 250 days at 99 %: independent exceptions are rejected 1.3 % of the time by LR_ind at a nominal 5 % (too few
    # exceptions for the asymptotics), and exceptions that follow one another a quarter of the time only 34 % of the
    # time. Measured with 10,000 paths (standard error at most 0.005).
    size = bt.power_christoffersen(250, 0.01, 0.01)
    clustered = bt.power_christoffersen(250, 0.01, 0.25)
    assert size.independence == pytest.approx(0.013, abs=0.01)
    assert clustered.independence == pytest.approx(0.34, abs=0.02)
    assert clustered.independence > size.independence and clustered.conditional_coverage > size.conditional_coverage


def test_christoffersen_refuses_bad_input():
    with pytest.raises(ValueError, match="at least two days"):
        bt.christoffersen([True], 0.01)
    with pytest.raises(ValueError, match="booleans"):
        bt.christoffersen([0, 2, 1], 0.01)
    with pytest.raises(ValueError, match="no Markov chain"):
        bt.power_christoffersen(250, 0.6, 0.1)


# --- Acerbi and Szekely's Z2 (acceptance 4) --------------------------------------------------------------------------

#: Acerbi and Szekely (2014), Table 4: the 5 % critical value of Z2 for 250 days at alpha = 2.5 %, location 0.
AS_TABLE_4 = {None: -0.70, 100.0: -0.70, 10.0: -0.71, 5.0: -0.74, 3.0: -0.82}


@pytest.mark.parametrize("dof", list(AS_TABLE_4), ids=lambda d: "normal" if d is None else f"t{d:g}")
def test_z2_critical_values_match_acerbi_szekely_table_4(dof):
    # Acceptance 4 (the Normal row): -0.70 +- 0.02. The table prints two decimals (0.005), and the 5 % quantile of
    # 100,000 simulated histories (50,000 for the Student-t rows, whose inverse cdf is slower) moves a little with
    # the seed: measured over seeds 2014 (the default), 0, 1 and 2, Normal -0.699 to -0.701, t100 -0.698 to -0.703,
    # t10 -0.708 to -0.713, t5 -0.730 to -0.743, t3 -0.811 to -0.824.
    alpha = 0.025
    if dof is None:
        q = stats.norm.isf(alpha)
        es = stats.norm.pdf(q) / alpha
    else:
        q = stats.t.isf(alpha, dof)
        es = stats.t.pdf(q, dof) * (dof + q * q) / (dof - 1) / alpha
    pnl = np.zeros(250)  # the statistic itself does not matter here
    null = "normal" if dof is None else "student_t"
    res = bt.acerbi_szekely_z2(pnl, q, es, alpha, null=null, dof=dof or 5.0, n_sims=100_000 if dof is None else 50_000)
    assert res.critical_value == pytest.approx(AS_TABLE_4[dof], abs=0.02)
    # No tail day: Z2 = 1, the largest value it takes, which the simulated histories without a tail day share; the
    # p-value counts those strictly below, so it is 1 - 0.975^250 = 0.9982 (Monte Carlo standard error 1.3e-4).
    assert res.statistic == 1.0 and res.p_value == pytest.approx(1 - 0.975**250, abs=6e-4)


def test_z2_null_simulation_agrees_with_brute_force():
    # The library draws only the tail days of each history; here 20,000 histories of 250 Normal days are drawn in
    # full. 5 % quantiles agree within 0.02 (the brute force's moves by about 0.006 at 20,000 histories) and the
    # brute force has mean 0 (within 0.01, four standard errors of a mean of 20,000 values of standard deviation 0.4).
    alpha, q = 0.025, stats.norm.isf(0.025)
    es = stats.norm.pdf(q) / alpha
    x = np.random.default_rng(11).standard_normal((20_000, 250))
    brute = np.where(x <= -q, x, 0.0).sum(axis=1) / (250 * alpha * es) + 1
    assert np.quantile(brute, 0.05) == pytest.approx(-0.70, abs=0.02)
    library = bt.acerbi_szekely_z2(x[0], q, es, alpha)  # 100,000 histories: about 0.003
    assert library.critical_value == pytest.approx(np.quantile(brute, 0.05), abs=0.02)
    assert abs(brute.mean()) < 0.01
    # Given the same simulated histories, the library computes the same statistics.
    given_null = bt.acerbi_szekely_z2(x[0], q, es, alpha, null=x[:5_000])
    assert given_null.critical_value == pytest.approx(np.quantile(brute[:5_000], 0.05), rel=1e-12)
    assert given_null.statistic == pytest.approx(brute[0], rel=1e-12)


@given(scale=st.floats(1e-3, 1e3), seed=st.integers(0, 1_000))
def test_z2_does_not_depend_on_the_scale_of_a_scale_family(scale, seed):
    x = np.random.default_rng(seed).standard_normal(250)
    assert bt.z2_statistic(scale * x, scale * 1.96, scale * 2.34, 0.025) == pytest.approx(
        bt.z2_statistic(x, 1.96, 2.34, 0.025), rel=1e-9, abs=1e-12
    )


def test_z2_power_and_size():
    # A Normal model scored on Normal P&L with 20 % more volatility (a year at alpha = 2.5 %): Z2 rejects 77 % of the
    # time; brute force with full draws agrees within 0.02 (each has a standard error of about 0.004 at 20,000
    # histories). With the model's own distribution the rejection rate is the size, 5 % within 0.005.
    alpha, q = 0.025, stats.norm.isf(0.025)
    es = stats.norm.pdf(q) / alpha
    rng = np.random.default_rng(12)
    null = rng.standard_normal((20_000, 250))
    critical = np.quantile(np.where(null <= -q, null, 0.0).sum(axis=1) / (250 * alpha * es) + 1, 0.05)
    wide = 1.2 * rng.standard_normal((20_000, 250))
    brute = np.mean(np.where(wide <= -q, wide, 0.0).sum(axis=1) / (250 * alpha * es) + 1 < critical)
    assert bt.power_z2(250, alpha, scale=1.2) == pytest.approx(brute, abs=0.02)
    assert round(bt.power_z2(250, alpha, scale=1.2), 2) == 0.77
    assert bt.power_z2(250, alpha) == pytest.approx(0.05, abs=0.005)
    assert bt.power_z2(250, alpha) < bt.power_z2(250, alpha, scale=1.1) < bt.power_z2(250, alpha, scale=1.2)


def test_z2_refuses_a_non_positive_es():
    with pytest.raises(ValueError, match="ES must be positive"):
        bt.acerbi_szekely_z2(np.zeros(10), 1.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        bt.acerbi_szekely_z2(np.zeros(10), 1.0, 2.0, null=np.zeros((200, 9)))


# --- FZ0 and Diebold-Mariano (acceptance 5) ------------------------------------------------------------------------


def fz0_paper(y, v, e, alpha):
    """Patton, Ziegel and Chen (2019), eq. 6, in their sign convention (v and e the quantile and ES of Y, negative)."""
    return -1 / (alpha * e) * (y <= v) * (v - y) + v / e + np.log(-e) - 1


def test_fz0_is_the_papers_loss_in_the_var_es_sign_convention():
    rng = np.random.default_rng(13)
    y, v = rng.standard_normal(1_000), rng.uniform(0.5, 3, 1_000)
    e = v + rng.uniform(0, 1, 1_000)
    np.testing.assert_allclose(bt.fz0_loss(y, v, e, 0.025), fz0_paper(y, -v, -e, 0.025), rtol=1e-13, atol=1e-13)


def test_fz0_is_smallest_at_the_true_var_and_es():
    # Acceptance 5: for N(0, 1) at alpha = 5 % the expected loss is smallest at VaR 1.645 and ES 2.063, (-1.64, -2.06)
    # in the paper's sign convention. The expectation is taken on 200,000 Normal quantiles at mid-probabilities (an
    # exact-enough sample: the closed form below agrees to 1e-5).
    alpha = 0.05
    y = ndtri((np.arange(200_000) + 0.5) / 200_000)

    def expected(params):
        var, es = params
        return math.inf if es <= 0 else float(bt.fz0_loss(y, var, es, alpha).mean())

    def closed_form(var, es):  # E[max(-Y - VaR, 0)] = phi(VaR) - VaR Phi(-VaR)
        tail = stats.norm.pdf(var) - var * stats.norm.sf(var)
        return tail / (alpha * es) + var / es + math.log(es) - 1

    for var, es in [(1.645, 2.063), (1.0, 1.5), (2.0, 3.0)]:
        assert expected((var, es)) == pytest.approx(closed_form(var, es), abs=1e-5)
    best = optimize.minimize(expected, x0=[1.0, 1.5], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12})
    truth = [stats.norm.isf(alpha), stats.norm.pdf(stats.norm.isf(alpha)) / alpha]
    assert best.x == pytest.approx(truth, abs=2e-3)
    assert [round(-v, 2) for v in best.x] == [-1.64, -2.06]


def newey_west_reference(d, lags):
    """Long-run variance with Bartlett weights, by explicit sums."""
    n, mean = len(d), sum(d) / len(d)
    gamma = [sum((d[t] - mean) * (d[t - k] - mean) for t in range(k, n)) / n for k in range(lags + 1)]
    return gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))


@pytest.mark.parametrize("lags", [0, 1, 5, 12])
def test_newey_west_variance_matches_explicit_sums(lags):
    rng = np.random.default_rng(14)
    d = np.zeros(300)
    for t in range(1, 300):  # AR(1), so the autocovariances matter
        d[t] = 0.6 * d[t - 1] + rng.standard_normal()
    assert bt.newey_west_variance(d, lags) == pytest.approx(newey_west_reference(d.tolist(), lags), rel=1e-11)


@given(values=st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=200), lags=st.integers(0, 30))
def test_newey_west_variance_is_never_negative(values, lags):
    assert bt.newey_west_variance(values, lags) >= 0.0


def test_diebold_mariano_statistic():
    rng = np.random.default_rng(15)
    a, b = rng.standard_normal(500) ** 2, rng.standard_normal(500) ** 2
    d = a - b
    # Without lags it is the t-statistic of the mean difference (variance with divisor T).
    plain = bt.diebold_mariano(a, b, lags=0)
    assert plain.statistic == pytest.approx(d.mean() / (d.std() / math.sqrt(500)), rel=1e-12)
    assert plain.p_value == pytest.approx(2 * stats.norm.sf(abs(plain.statistic)), rel=1e-12)
    default = bt.diebold_mariano(a, b)
    assert default.lags == math.floor(4 * (500 / 100) ** (2 / 9)) == 5
    assert default.statistic == pytest.approx(d.mean() / math.sqrt(newey_west_reference(d.tolist(), 5) / 500))
    assert bt.diebold_mariano(b, a).statistic == pytest.approx(-default.statistic, rel=1e-12)
    assert bt.diebold_mariano(a, a).statistic == 0.0 and bt.diebold_mariano(a, a).p_value == 1.0
    whole = np.arange(500.0)  # one model's loss exactly the other's plus one: no variance, infinite evidence
    shifted = bt.diebold_mariano(whole + 1, whole)
    assert shifted.statistic == math.inf and shifted.p_value == 0.0


def test_diebold_mariano_size_with_autocorrelated_losses():
    # Equal expected losses with AR(1) differentials (coefficient 0.3, 1,000 days): with Newey-West's default lags the
    # test rejects at about its nominal 5 % (measured 5.2 % over 1,000 replications; the band allows 3 standard
    # errors of 0.7 point and some finite-sample excess), without them it over-rejects (measured 13.8 %): the long-run
    # variance is what makes autocorrelated losses comparable.
    rng = np.random.default_rng(16)
    shocks = rng.standard_normal((1_000, 1_000))
    d = np.zeros_like(shocks)
    for t in range(1, 1_000):
        d[:, t] = 0.3 * d[:, t - 1] + shocks[:, t]
    zero = np.zeros(1_000)
    with_lags = np.mean([bt.diebold_mariano(row, zero).p_value < 0.05 for row in d])
    without = np.mean([bt.diebold_mariano(row, zero, lags=0).p_value < 0.05 for row in d])
    assert 0.03 <= with_lags <= 0.085 and without > 0.1


# --- P&L attribution (MAR32.34 to MAR32.42) -------------------------------------------------------------------------


def test_pla_metrics_are_spearman_and_kolmogorov_smirnov():
    rng = np.random.default_rng(17)
    hpl = rng.standard_t(4, 250)
    rtpl = 0.9 * hpl + rng.normal(0, 0.4, 250)
    rtpl[:5] = hpl[:5]  # ties across the two series do not matter; within one they share their average rank
    res = bt.pla_test(hpl, rtpl)
    assert res.spearman == pytest.approx(stats.spearmanr(hpl, rtpl).statistic, rel=1e-12)
    assert res.ks == pytest.approx(stats.ks_2samp(hpl, rtpl).statistic, rel=1e-12)
    assert res.ks_p_value == pytest.approx(stats.kstwobign.sf(math.sqrt(125) * res.ks), rel=1e-10)


def test_the_kolmogorov_smirnov_thresholds_are_mar32s_p_values():
    # MAR32.42 quotes p-values 0.264 at a distance of 0.09 and 0.055 at 0.12. Kolmogorov's asymptotic distribution
    # for two samples of 250 days, which the library uses, gives 0.2634 and 0.0546: the quotes to within a unit of
    # their last digit. (The exact two-sample distribution gives 0.0546 at 0.12 too; 0.09 is not a distance two
    # samples of 250 can have.)
    hpl = np.arange(250.0)
    for distance, quoted in [(0.09, 0.264), (0.12, 0.055)]:
        assert float(stats.kstwobign.sf(math.sqrt(250 * 250 / 500) * distance)) == pytest.approx(quoted, abs=1e-3)
    shifted = bt.pla_test(hpl, hpl + 22)  # 22 of the 250 steps of 0.004: a distance of 0.088
    assert shifted.ks == pytest.approx(0.088, abs=1e-12)
    assert shifted.ks_p_value == pytest.approx(stats.kstwobign.sf(math.sqrt(125) * 0.088), rel=1e-10)


@pytest.mark.parametrize(
    ("spearman", "ks", "zone"),
    [
        (0.81, 0.08, "green"),
        (0.80, 0.08, "amber"),  # green needs a correlation above 0.80
        (0.81, 0.09, "amber"),  # and a distance below 0.09
        (0.70, 0.12, "amber"),  # red needs a correlation below 0.70 or a distance above 0.12
        (0.69, 0.05, "red"),
        (0.95, 0.121, "red"),
    ],
)
def test_pla_zones_follow_mar32_42(spearman, ks, zone):
    assert bt.pla_zone(spearman, ks) == zone


def test_pla_refuses_unequal_or_constant_series():
    with pytest.raises(ValueError, match="same length"):
        bt.pla_test(np.arange(5.0), np.arange(6.0))
    with pytest.raises(ValueError, match="constant"):
        bt.pla_test(np.ones(10), np.arange(10.0))


# --- The synthetic history ------------------------------------------------------------------------------------------


def test_the_synthetic_history_is_labelled_and_reproducible():
    history = bt.synthetic_history(3, 300)
    assert history.attrs["source"] == SYNTHETIC_LABEL == "Synthetic — not market data"
    assert history.attrs["parameters"]["seed"] == 3
    assert list(history.columns) == ["close", "log_return", "iv_index", "true_vol"]
    assert history.index.name == "day" and len(history) == 301
    assert history["close"].iloc[0] == 5800.0 and history["iv_index"].iloc[0] == 0.18  # exactly
    assert np.isnan(history["log_return"].iloc[0])
    np.testing.assert_allclose(np.diff(np.log(history["close"])), history["log_return"].iloc[1:], rtol=0, atol=1e-12)
    assert history.equals(bt.synthetic_history(3, 300))
    assert not history.equals(bt.synthetic_history(4, 300))


def test_the_implied_vol_index_has_the_requested_vol_of_vol_and_correlation():
    # 10,000 days: the daily log change of the index has standard deviation eta / sqrt(252) up to the mean reversion
    # (measured eta 0.996, 1.015 and 0.996 at seeds 0 to 2; a sample standard deviation of 10,000 values moves by
    # 0.7 %) and correlation rho with the Normal score of the return's shock (measured -0.681 to -0.704 at seeds 0
    # to 3; a correlation's standard error here is 0.005, and mean reversion dilutes it slightly).
    history = bt.synthetic_history(0, 10_000)
    changes = np.diff(np.log(history["iv_index"].to_numpy()))
    assert changes.std() * math.sqrt(252) == pytest.approx(1.0, rel=0.03)
    variance = history["true_vol"].to_numpy()[:-1] ** 2 / 252
    shock = (history["log_return"].to_numpy()[1:] + variance / 2) / np.sqrt(variance)
    score = ndtri(stats.t.cdf(shock / math.sqrt(3 / 5), 5))  # a unit-variance t with 5 degrees of freedom
    assert np.corrcoef(changes, score)[0, 1] == pytest.approx(-0.7, abs=0.03)
    # The shocks are unit-variance Student-t(5) (Kolmogorov-Smirnov p-value 0.50 measured).
    assert stats.kstest(shock / math.sqrt(3 / 5), "t", args=(5,)).pvalue > 0.01


def test_without_vol_of_vol_the_returns_are_a_garch_t():
    history = bt.synthetic_history(1, 5_000, vol_of_vol=0.0, garch_alpha=0.08, garch_beta=0.9)
    assert np.all(history["iv_index"] == 0.18)
    # The variance is (0.18 / 1.2)^2 / 252 times g, and g follows the GARCH(1,1) recursion on the shocks.
    g = history["true_vol"].to_numpy() ** 2 / 0.15**2
    variance = g[:-1] * 0.15**2 / 252
    shock = (history["log_return"].to_numpy()[1:] + variance / 2) / np.sqrt(variance)
    np.testing.assert_allclose(g[1:], 1 - 0.08 - 0.9 + (0.08 * shock**2 + 0.9) * g[:-1], rtol=1e-10)
    assert g[0] == 1.0
    # Long-run volatility iv_level / vol_premium: 15 % a year (the sample's moves by a few per cent).
    assert history["log_return"].std() * math.sqrt(252) == pytest.approx(0.15, rel=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_days": 0},
        {"spot": -1.0},
        {"vol_of_vol": -0.1},
        {"spot_vol_correlation": 1.5},
        {"garch_alpha": 0.5, "garch_beta": 0.5},
        {"dof": 2.0},
        {"iv_level": math.nan},
    ],
)
def test_the_synthetic_history_refuses_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        bt.synthetic_history(0, **({"n_days": 10} | kwargs))


# --- The rolling backtest (acceptance 7) ---------------------------------------------------------------------------


def short_straddle(spot, iv):
    """The panel's case (docs/reviews/panel-v3.0.md, finding 2): a 45-day at-the-money short straddle on an index,
    re-struck at every close."""
    return S.build_template("Short Straddle", spot, 10), S.Market(spot, 0.043, iv, 45 / 365, 0.013)


def test_hypothetical_pnl_is_the_straddle_repriced_at_the_next_close():
    history = bt.synthetic_history(2, 120)
    closes, iv = history["close"].to_numpy(), history["iv_index"].to_numpy()
    models = {"hs": bt.historical_model()}
    res = bt.rolling_backtest(short_straddle, closes, iv, models, 100, (0.99,), multiplier=100)
    # Independently: the 45-day straddle re-priced a trading day later at the next close, its implied vol moved by
    # the index's change damped for its tenor, (tau / 1M)^-0.25 = 0.912 for tau = 45 / 365 - 1 / 252 years (the
    # index is a one-month vol; risk.py and the scenarios damp longer tenors the same way).
    t = np.arange(100, 120)
    tau = 45 / 365 - 1 / 252
    vol = iv[t] * (1 + (iv[t + 1] / iv[t] - 1) * (tau / (30 / 365)) ** -0.25)

    def straddle(spot, strike, years, vol):
        call = bs.price(spot, strike, years, 0.043, vol, 0.013, "call")
        return call + bs.price(spot, strike, years, 0.043, vol, 0.013, "put")

    expected = -100 * (straddle(closes[t + 1], closes[t], tau, vol) - straddle(closes[t], closes[t], 45 / 365, iv[t]))
    np.testing.assert_allclose(res.daily["pnl"], expected, rtol=1e-10, atol=1e-8)
    assert res.daily["day"].tolist() == list(range(101, 121))


def test_the_summary_is_the_tests_applied_to_the_daily_series():
    history = bt.synthetic_history(5, 400)
    models = {"normal": bt.monte_carlo_model(risk.VaRConfig(0.99, 1, 2_000)), "hs": bt.historical_model()}
    res = bt.rolling_backtest(short_straddle, history["close"], history["iv_index"], models, 100, (0.99, 0.975))
    assert res.source == SYNTHETIC_LABEL and res.summary.attrs["source"] == SYNTHETIC_LABEL
    assert (res.summary["history"] == SYNTHETIC_LABEL).all()
    assert len(res.summary) == 4 and len(res.daily) == 4 * 300  # 401 closes: 300 days after the 100-day window
    for row in res.summary.itertuples():
        daily = res.daily[(res.daily["model"] == row.model) & (res.daily["confidence"] == row.confidence)]
        hits = (daily["pnl"] < -daily["var"]).to_numpy()
        assert np.array_equal(hits, daily["exception"].to_numpy())
        p = 1 - row.confidence
        assert (row.days, row.exceptions, row.expected) == (300, hits.sum(), pytest.approx(300 * p))
        assert row.kupiec_p == bt.kupiec_pof(int(hits.sum()), 300, p).p_value
        assert row.christoffersen_p == bt.christoffersen(hits, p).p_cc
        assert row.zone == bt.traffic_light(int(hits.sum()), 300, coverage=row.confidence).zone
        assert row.last_250_exceptions == hits[-250:].sum()
        assert row.z2 == pytest.approx(bt.z2_statistic(daily["pnl"], daily["var"], daily["es"], p), rel=1e-12)
        assert row.fz0 == pytest.approx(bt.fz0_loss(daily["pnl"], daily["var"], daily["es"], p).mean(), rel=1e-12)


def test_a_correct_model_passes():
    # A history whose returns are exactly the Normal model's (no GARCH, no fat tails, implied vol constant and equal
    # to the volatility) and a stock position: the app's Normal model is then the truth, up to its simulation error
    # (with 10,000 paths and seed 7 its 99 % VaR is the 1.06 % quantile, its 97.5 % VaR the 2.37 % one). At seed 0:
    # 11 exceptions against 7.5 at 99 % (Kupiec p 0.23) and 16 against 18.75 at 97.5 % (0.51), green, Z2 p 0.06 and
    # 0.55; over seeds 0 to 5 the smallest p-value is Kupiec's 0.033 (seed 4). The Z2 critical values simulated from
    # the model's scenarios, -0.616 and -0.376, match those of a Normal P&L, -0.635 and -0.394, within 0.05 (a share's
    # one-day P&L is Normal up to a tiny log-normal skew; each simulated value moves by about 0.01 at 4,000 histories).
    history = bt.synthetic_history(
        0, 800, vol_of_vol=0.0, vol_premium=1.0, garch_alpha=0.0, garch_beta=0.0, dof=None, iv_level=0.2
    )

    def stock(spot, iv):
        return S.Strategy("Stock", (S.Leg("stock", 1),)), S.Market(spot, 0.04, iv, 0.5)

    models = {"normal": bt.monte_carlo_model(risk.VaRConfig(0.99, 1, 10_000))}
    res = bt.rolling_backtest(stock, history["close"], history["iv_index"], models, 50, (0.99, 0.975), z2_sims=4_000)
    for row in res.summary.itertuples():
        assert row.kupiec_p > 0.01 and row.christoffersen_p > 0.01 and row.zone == "green" and row.z2_p > 0.01
        p = 1 - row.confidence
        q = stats.norm.isf(p)
        normal = bt.acerbi_szekely_z2(np.zeros(row.days), q, stats.norm.pdf(q) / p, p)
        assert row.z2_critical == pytest.approx(normal.critical_value, abs=0.05)


def test_hull_white_scaling_matches_an_explicit_loop():
    rng = np.random.default_rng(18)
    returns = np.concatenate([0.01 * rng.standard_normal(250), 0.03 * rng.standard_normal(250)])
    variance = [np.mean(returns**2)]
    for r in returns:
        variance.append(0.94 * variance[-1] + 0.06 * r * r)
    expected = returns * np.sqrt(variance[-1] / np.array(variance[:-1]))
    np.testing.assert_allclose(bt._ewma_scaled(returns, 0.94), expected, rtol=1e-12)
    # After the volatility triples, the old quiet days are scaled up to about today's volatility.
    assert np.std(bt._ewma_scaled(returns, 0.94)[:250]) == pytest.approx(0.03, rel=0.3)


def test_rolling_backtest_refuses_inconsistent_input():
    history = bt.synthetic_history(0, 50)
    closes, iv = history["close"], history["iv_index"]
    models = {"hs": bt.historical_model()}

    def elsewhere(spot, vol):
        return short_straddle(spot + 1, vol)

    with pytest.raises(ValueError, match="market at the close"):
        bt.rolling_backtest(elsewhere, closes, iv, models, 20)
    with pytest.raises(ValueError, match="two closes more than the window"):
        bt.rolling_backtest(short_straddle, closes, iv, models, 50)  # 51 closes: no day after a 50-day window
    with pytest.raises(ValueError, match="same days"):
        bt.rolling_backtest(short_straddle, closes, iv[:-1], models, 20)
    with pytest.raises(ValueError, match="confidences"):
        bt.rolling_backtest(short_straddle, closes, iv, models, 20, (0.4,))
    with pytest.raises(ValueError, match="no model"):
        bt.rolling_backtest(short_straddle, closes, iv, {}, 20)
    with pytest.raises(ValueError, match=r"at least 100 for a 0\.99 VaR \(it returned 20\)"):
        bt.rolling_backtest(short_straddle, closes, iv, models, 20)  # 20 days: no scenario in the 1 % tail
    with pytest.raises(ValueError, match="one-day horizon"):
        bt.monte_carlo_model(risk.VaRConfig(0.99, 5, 2_000))
    with pytest.raises(ValueError, match="scaling"):
        bt.historical_model(scaling="garch")


#: The models of acceptance 7: the app's VaR engine with a Normal spot and implied vol frozen (the Risk tab's spot-only
#: run), and historical simulation with Hull and White's volatility scaling.
def _acceptance_models(paths: int) -> dict:
    return {
        "Normal, implied vol frozen": bt.monte_carlo_model(risk.VaRConfig(0.99, 1, paths)),
        "Historical, vol-scaled": bt.historical_model(scaling="ewma"),
    }


def test_frozen_implied_vol_fails_and_scaled_history_passes_on_three_synthetic_years():
    # The default-run version of acceptance 7 (about 2 s): 1,000 days, window 250, 750 days of backtest at 99 % (7.5
    # exceptions expected; amber from 12, red from 20). Measured at seed 0: the frozen-vol Normal model 16 exceptions
    # (amber) and Z2 -1.43 against a critical -0.66; vol-scaled historical simulation 3 (green) and Z2 +0.45. Over
    # seeds 0 to 9 the frozen-vol model is never green (12 to 38 exceptions) and Z2 rejects it at 9 of them; scaled
    # history is always green (2 to 7) and Z2 never rejects it.
    history = bt.synthetic_history(0, 1_000)
    res = bt.rolling_backtest(
        short_straddle, history["close"], history["iv_index"], _acceptance_models(2_000), 250, (0.99,), multiplier=100
    )
    frozen, scaled = (row for row in res.summary.itertuples())
    assert frozen.zone != "green" and frozen.z2 < frozen.z2_critical
    assert scaled.zone == "green" and scaled.z2 > scaled.z2_critical


@pytest.mark.slow  # about 20 s
def test_frozen_implied_vol_ends_red_and_scaled_history_does_not_on_ten_synthetic_years():
    # Acceptance 7: 2,520 synthetic days (seed 0), window 500, 2,020 days of backtest (20.2 exceptions expected at 99 %;
    # amber from 28, red from 39). Measured at seed 0: the frozen-vol Normal model 50 exceptions (red; 106 against
    # 50.5 at 97.5 %), vol-scaled historical simulation 17 (green; 40 at 97.5 %), and the Risk tab's default, the
    # Normal model with index implied-vol dynamics, 31 (amber; 50 at 97.5 %). Over seeds 0 to 9 (scratch runs, about
    # 20 s each): the frozen-vol model 42 to 76 (red every time), scaled history 15 to 25 (green every time), the
    # Risk tab's default 21 to 39 (4 green, 5 amber, 1 red). The Methodology quotes seed 0 and these ranges.
    history = bt.synthetic_history(0)
    models = _acceptance_models(5_000) | {
        "Normal, implied-vol moves": bt.monte_carlo_model(
            risk.VaRConfig(0.99, 1, 5_000, vol_of_vol=1.0, spot_vol_correlation=-0.7)
        )
    }
    res = bt.rolling_backtest(
        short_straddle, history["close"], history["iv_index"], models, 500, (0.99, 0.975), multiplier=100
    )
    rows = {(row.model, row.confidence): row for row in res.summary.itertuples()}
    frozen, scaled = rows["Normal, implied vol frozen", 0.99], rows["Historical, vol-scaled", 0.99]
    default = rows["Normal, implied-vol moves", 0.99]
    assert (frozen.days, frozen.expected, frozen.zone) == (2_020, pytest.approx(20.2), "red")
    assert frozen.z2 < frozen.z2_critical
    assert scaled.zone != "red" and scaled.z2 > scaled.z2_critical
    assert rows["Normal, implied vol frozen", 0.975].zone == "red"
    assert rows["Historical, vol-scaled", 0.975].zone != "red"
    light = bt.traffic_light(0, 2_020)
    assert (light.amber_from, light.red_from) == (28, 39)
    assert default.zone == "amber" and scaled.zone == "green"
    sentence = _risk_methodology_backtest_sentence(frozen.exceptions, default.exceptions, scaled.exceptions)
    assert sentence in _text(RISK_METHODOLOGY)


# --- Documentation ------------------------------------------------------------------------------------------------


def test_the_docstring_examples_run():
    failures, tried = doctest.testmod(bt, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tried > 20 and failures == 0


def test_the_methodology_quotes_the_measured_power():
    # The Methodology's power caveat, number by number, from the functions (the simulated ones at their default seeds
    # and sizes: 10,000 paths for Christoffersen, 100,000 histories for Z2, standard errors 0.005 and 0.0015).
    text = _text(RISK_METHODOLOGY)
    size, power = bt.power_kupiec(250, 0.01, 0.01), bt.power_kupiec(250, 0.02, 0.01)
    assert f"rejects a correct model {100 * size:.1f} % of the time" in text
    assert f"only {100 * power:.0f} % of the time; ten years (2,520 days) raise that to 98.7 %" in text
    assert f"{100 * bt.power_kupiec(2_520, 0.02, 0.01):.1f}" == "98.7"
    green = 1 - bt.power_traffic_light(250, 0.02).amber_or_red
    assert f"leaves that model green {100 * green:.0f} % of the time" in text
    clustered = bt.power_christoffersen(250, 0.01, 0.25).independence
    assert f"a quarter of the time in only {100 * clustered:.0f} % of such years" in text
    wide, fat = bt.power_z2(250, 0.025, scale=1.2), bt.power_z2(250, 0.025, dof=5)
    assert f"20 % above the model's in {100 * wide:.0f} % of years" in text
    assert f"at the model's variance, in only {100 * fat:.0f} %" in text


def _risk_methodology_backtest_sentence(frozen: int, default: int, scaled: int) -> str:
    return (
        f"the Normal model with implied vol frozen has {frozen} exceptions (red), the Risk tab's default with "
        f"implied-vol moves {default} (amber), both with 5,000 paths a day, and historical simulation on the last 500 "
        f"days with Hull and White's (1998) volatility scaling {scaled} (green)"
    )


#: The method note's power table: each row's function of the number of days (docs/methods/backtest.md, "Power").
POWER_ROWS = {
    "Kupiec, correct model": lambda n: bt.power_kupiec(n, 0.01, 0.01),
    "Kupiec, true exception rate 2 %": lambda n: bt.power_kupiec(n, 0.02, 0.01),
    "Traffic light out of green, correct model": lambda n: bt.power_traffic_light(n, 0.01).amber_or_red,
    "Traffic light out of green, true exception rate 2 %": lambda n: bt.power_traffic_light(n, 0.02).amber_or_red,
    "Christoffersen independence, independent exceptions": lambda n: bt.power_christoffersen(n, 0.01, 0.01)[0],
    "Christoffersen independence, 25 % chance of an exception after one": lambda n: bt.power_christoffersen(
        n, 0.01, 0.25
    )[0],
    "$Z_2$, correct model": lambda n: bt.power_z2(n, 0.025),
    "$Z_2$, volatility 20 % above the model's": lambda n: bt.power_z2(n, 0.025, scale=1.2),
    "$Z_2$, Student-t(5) tails at the model's variance": lambda n: bt.power_z2(n, 0.025, dof=5),
}


@pytest.mark.slow  # about 12 s: the Student-t rows invert the t distribution millions of times at 2,520 days
def test_the_method_note_power_table_is_what_the_functions_give():
    text = METHOD_NOTE.read_text(encoding="utf-8")
    for name, power in POWER_ROWS.items():
        cells = " | ".join(f"{100 * power(n):.1f} %" for n in (250, 500, 1_000, 2_520))
        assert f"| {name} | {cells} |" in text, name
