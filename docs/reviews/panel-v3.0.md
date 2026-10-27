# Expert review panel, version 3.0 (commit a7b0b0b, 2026-10-27)

Four independent reviews of the shipped project, each from one professional viewpoint: an options market maker, a
principal quantitative developer, a model-risk validator and a principal product designer. Each finding carries its
evidence (file and line, a command and its output, or a screenshot that lived in a scratch folder, written here as
`<scratch>/`). `docs/improvement-guide.md` distils them into priorities; `ROADMAP.md` tracks the work.

# market-maker

## Market-maker review: Option-Pricing-and-Strategy-Analyzer at a7b0b0b

Scripts and screenshots: `<scratch>/panel/market-maker/` (`shots.py`, `pricing_1dte.png`, `strategy_condor_7d.png`, `risk_var_7d.png`). All numbers below were reproduced with the library or `tests._apptest.make_app` on SPX-like inputs (S 5800, r 4.3 %, q 1.3 %).

**What is sound (checked, so the findings below are not about the numerics):** BSM, LBR implied vol and ALO agree with the LR tree under negative rates (r = −3 %, 1y ATM American call: ALO 6.87183, 2000-step LR 6.87187; the put correctly carries no premium). ALO prices a 2-hour ATM put in 6 ms; `implied_volatility` at T = 2 h returns 0.18000 exactly; `analyze` on a 0DTE straddle gives sensible breakevens/PoP. The library handles 0DTE, skew-by-leg, dividends and negative rates; the app is where the desk gaps are.

---

### 1. Risk-tab scenario grid is unusable for anything short-dated (spot ±30 % in 5 % steps, vol in relative −50 %…+100 %)
- **Evidence:** `risk_var_7d.png`: 7-DTE short straddle (1σ√T = 2.2 %) on a 13 × 6 grid whose every cell but the 0 % row reads −18k…−164k; the entire ±2σ region lives inside one row. `tab_risk.py:73-75` (`span` slider 10–60 %, default 30; `vol_changes = [-0.5,-0.25,0,0.25,0.5,1.0]` hard-coded, relative).
- **Change:** default the spot axis to multiples of σ√T (±0.25σ…±3σ, as `_time_decay` already does with `_sigma_t`, `tab_risk.py:143-147`) with an "absolute %" option; make the vol axis absolute vol points (±1, ±2, ±5) and keep the relative shocks as a stress choice. Show the 1-day-rolled grid by default (`days=1`), since that is the overnight risk a trader checks.
- **Effort:** S. Sharpens UI-4 / UI-9.

### 2. VaR ignores vega by default
- **Evidence:** `tab_risk.py:361-363` `Simulate implied-vol moves` toggle `value=False` → `vol_of_vol=0`. 45-DTE short straddle, 1 contract, 99 % 1-day: VaR $2,833 / ES $3,683 with vol frozen vs $3,862 / $4,737 with the app's own suggested 100 % vol-of-vol, ρ = −0.7 (`risk.value_at_risk`, 100k paths). A 27 % understatement on the most common vol position, while the help text itself says 100 % "matches index (VIX) dynamics".
- **Change:** default the toggle on with the stated index parameters; show "spot-only VaR" and "spot + vol VaR" side by side, and add vega-scaled analytic ES as the comparator (RK-6).
- **Effort:** S. Sharpens RK-5 "realistic defaults".

### 3. "Theta / day" is instantaneous θ/365 — wrong by 2× inside three days to expiry, and nonsense for 0DTE
- **Evidence:** 1-DTE ATM put: value 21.56, shown theta −10.66/day, true one-day decay −21.56 (whole premium). 1-DTE 5750 put: theta −6.97 vs true −5.19 (`pricing_1dte.png`, Greeks table). 7-DTE condor: strip shows +$137.50, full revaluation after one day +$151.97. 0DTE (2 h) short straddle: `position_risk` theta +$7,552/day on a position worth $1,259. Display sites: `tab_pricing.py:373,445,492`, `tab_strategy.py:844`, `tab_risk.py:159-160`.
- **Change:** show "1-day decay" by full revaluation (`position_value(t=1/365)` or `pnl_grid(days=1)`, both exist) next to θ/365, and switch the headline to the revaluation figure when they differ by >10 %. When DT-2 adds 0DTE, replace theta with "decay to close".
- **Effort:** S. Sharpens DT-2; without it the 0DTE work ships misleading numbers.

### 4. Expiry date is `today + days` with no exchange calendar; the market clock is not wired to the app
- **Evidence:** `make_app(days=4)` header → `31-Oct-26 (4d)` (Saturday); `days=5` → `01-Nov-26` (Sunday); `days=60` → `26-Dec-26` (Saturday). `ui/format.py:267-271`. `grep -rn "clock" options_analyzer/ui/` → no hits: `clock.py` (NYSE holidays, AM/PM settlement, `time_to_expiry`) is unused by the interface. A shared link with `days=7` shows a different expiry date tomorrow.
- **Change:** expiry *date* picker snapped to listed expiries (Mon/Wed/Fri for SPX, 3rd Friday monthlies) via `clock.is_trading_day` / `settlement_time`; store the date in the URL; derive T from `clock.time_to_expiry` (minutes, AM/PM). Days-to-expiry becomes derived, not input.
- **Effort:** M. This is DT-2 — the library half is done; say so in ROADMAP and scope the item to UI wiring.

### 5. The "done" desk-risk library is unreachable from the app
- **Evidence:** `grep -rn "vega_buckets\|weighted_vega\|greek_ladder\|scenario_cube\|strikes_by\|strike_for_delta\|sample_chain\|read_chain" options_analyzer/ui/` → zero hits. ROADMAP "Done" lists RK-1, RK-2, RK-9, DT-4, DT-5; a reviewer opening the app sees no Greek ladder, no vega by expiry/delta bucket, no 1M-weighted vega, no delta-placed strikes, no synthetic chain.
- **Change:** Risk tab: a Greek ladder table (`scenarios.greek_ladder(unit="sigma", roll_days=1)`) and vega buckets table; Strategy tab: a "Strikes by delta" selector (16Δ / 25Δ / 30Δ shorts, 5Δ wings) calling `build_template(strikes_by="delta")`; Surface tab: "Load synthetic chain" button using `market_data.sample_chain`. Mark these items "library only" in ROADMAP until then (The bar, item 6).
- **Effort:** M. Sharpens BK-1, BK-2, RK-9a.

### 6. Default iron condor sells 33–36-delta strikes and has a 45.6 % probability of profit
- **Evidence:** 7-DTE, 16 % vol: auto width $50 (0.86 % of strike, `tab_strategy.py:146 auto_width` ≈ half σ√T); legs table deltas −0.334 (5750 put) / +0.363 (5850 call); PoP metric 45.6 %, reward/risk 1.28 (`strategy_condor_7d.png`). No desk sells a condor with shorts inside 30Δ by default.
- **Change:** default `strikes_by="delta"` with (0.20, 0.05) for condors/strangles/jade lizards; keep width as the alternative. Show each leg's delta in the editor (currently only in the inception table).
- **Effort:** S (same code path as #5).

### 7. Every leg is priced at the flat sidebar vol — no skew, so credits and breakevens are wrong in the direction that matters
- **Evidence:** condor legs table IV column all 0.16; on SPX the 33Δ put carries ~2–3 vol points over the 36Δ call. The Surface tab already builds an SSVI surface with ρ = −0.6 (`tab_surface.py:76-85`) and exposes `VolSurface.implied_vol_at(K/S, T)` (`surface.py:83`), but `tab_strategy.py` never reads it.
- **Change:** toggle "Leg IVs from the surface" that fills `Leg.volatility` from the published surface (SSVI by default, fitted quotes when uploaded); label legs "IV 18.4 % (surface)". PoP from the smile-implied density is the follow-on (UI-13).
- **Effort:** M. Sharpens BK-3 / SF-11; do this before RK-7.

### 8. Pricing-tab "chain" is 80–120 % of strike in 9 rows, showing 1e-134 in scientific notation
- **Evidence:** 1-DTE SPX: strikes 4600, 4887.5 … 6900 (287.5 apart); put values `9.29e-134`, `1.07e-73`; caption "Strikes from 80 % to 120 % of the sidebar strike" (`tab_pricing.py:435-470`; AppTest dataframe 1).
- **Change:** strikes at the exchange interval (the intervals already live in `market_data/synthetic.py`) across ±3σ√T, prices at the tick, delta column, ATM row highlighted.
- **Effort:** S.

### 9. IV tab takes one price — no bid/ask, so no bid/ask vol or vol spread
- **Evidence:** `tab_iv.py:105-113`, single "Market price" input. A market maker's first read on any quote is the vol bid/ask and the spread in points; the batch solver and the adapters already parse bid/ask (`vol_surface/quotes.py:127-128, 482-485`).
- **Change:** optional bid and ask inputs → IV bid / mid / ask, spread in vol points, and "one tick = x vol points" (tick / vega). Flag when the bid is below intrinsic.
- **Effort:** S. Not in the roadmap (UI-23 and SF-5 are different items).

### 10. Trading-day VaR horizon vs calendar-day theta: two clocks in one tab
- **Evidence:** `risk.py:27-29` "a horizon of h days removes h/252 years from each option's life"; `_settlement_days` (`risk.py:228`) uses `expiry * TRADING_DAYS`. 7-DTE straddle: one calendar day of decay $760, one "trading day" (1/252 y) $1,122 — the VaR distribution carries 48 % more theta than the theta the same tab prints per calendar day, every day of the week, and nothing special over a weekend.
- **Change:** diffuse over trading time, decay over calendar time to the next session (exactly what `clock.VarianceClock.variance_time` already separates); horizon expressed as "to next close / N sessions".
- **Effort:** M. Sharpens SF-9 (the trading-time clock should cover Risk, not only Surface).

### 11. Historical stress episodes hit a 7-day position as instantaneous shocks by default
- **Evidence:** "Include episode duration" default off (`tab_risk.py:204-206`); GFC (517 days, −56.8 %) applied instantly to a 7-DTE straddle → −$319k; "Fed severely adverse" (274 days) → −$326k. For a position that expires in 7 days these are not scenarios.
- **Change:** for episodes longer than the front expiry, show the worst window of the position's own length (worst 1-day/5-day/20-day return of each episode: 1987 −20.5 %/1d, 2008 worst 5d −18 %, 2020 worst 5d −17 %) or default the duration toggle on and say the leg expired.
- **Effort:** M. Sharpens RK-5.

### 12. Share link drops the position
- **Evidence:** `header.py` `SHARE_NOTE`: "Edited legs, uploaded quotes and the other settings of a tab stay in this session." A ticket a colleague cannot reopen is not a ticket.
- **Change:** encode legs in the URL (`legs=p-1x5750@16,c-1x5850@14`), parsed by the UI-free position builder (P0-1).
- **Effort:** S/M. Sharpens BK-4 / UI-28.

### 13. Precision and headline cards are textbook, not desk
- **Evidence:** `pricing_1dte.png`: Call $55.6631, Put $5.1923, Forward $5,800.4767 on an SPX contract with a 0.05/0.10 tick; fourth card "Call ITM probability 82.20 %" (N(d2)).
- **Change:** prices to the tick (2 dp), forward to 2 dp; replace the fourth card with "Expected move ±$X (straddle) / ±Y %" and show delta in the card caption.
- **Effort:** S. Sharpens DK-4 / UI-3 / UI-27.

### 14. Surface uses one carry for every expiry and reports at K/S = 0.9/1/1.1
- **Evidence:** `surface.py:98,106,322` `y = log(K/S) − carry·T` with scalar `carry`; `surface_summary` at fixed K/S (`surface.py:539-543`); `_contract_readout` at K/S, not delta (`tab_surface.py:233-236`). With a single r−q, the fitted forward is off by the dividend-timing/term-structure error at every expiry, which shows up as a spurious calendar-arbitrage flag between a pre- and post-ex-div expiry.
- **Change:** this is SF-3 + SF-6 exactly; the one sharpening: report ATM/25Δ RR/25Δ BF per expiry in the summary table and drive the contract readout by delta, because that is the only form a desk compares against OVDV.
- **Effort:** M (already planned).

---

# quant-dev

## Principal quant-dev review — findings ranked by value to a professional user

Scratch artefacts (scripts, logs): `<scratch>/panel/quant-dev/` — `profile_core.py`, `probe_numerics.py`, `probe_iv_conditioning.py`, `profile_surface.py`, `mypy-default.txt`, `mypy-strict.txt`, `core-tests.log`. Nothing in the repository was modified (`git status --porcelain` empty before and after).

Overall: the numerical core is unusually well engineered (LBR port, ALO engine with bounds and honest fallbacks, 40–60-digit and QuantLib oracles, strict pytest config, network guard). The findings below are what still separates it from a desk tool; most are sharper than the roadmap's wording, and a few are not on it.

---

### 1. The implied-vol solver returns arbitrary volatilities (0 % … 219 %) for quotes whose time value is at rounding level, and reports them as solved — not on the roadmap
**What is wrong.** The lower-bound check tolerates a quote `1e-12·max(S,K)` *below* intrinsic (`implied_vol.py:139`, `:384`) and returns 0 for `otm_price <= 0` (`:188-189`, `:399`), but a quote a few ulps *above* intrinsic is inverted literally: a normalised price of 1e-16 at |x| = 0.65 is "solved" to 100 %+ vol. The result is discontinuous in the input's last bit and `REJECTION_REASONS` (`:57-66`) has no "no time value / ill-conditioned" code, so the IV tab and the Surface tab's price inversion (`quotes.py:423-438` drops only exact zeros) pass such numbers into the SVI fit.
**Evidence** (`probe_iv_conditioning.py`): 20,000 random BSM quotes priced in double precision and re-inverted → 1,899 (9.5 %) come back with IV more than 1 % off the vol that priced them (up to 32×), every one with time value ≤ 2.8e-16·max(S,K), all with `reason == ""`. Every quote with time value > 1e-12·max(S,K) inverts to < 6e-6 relative. A 2-day 60-strike call, S = 100: intrinsic + 0 → 0 %, intrinsic + 1e-13 → 94.5 %, intrinsic + ½ cent → 230 %. The accuracy study's tolerance `max(1e-14, 4·cond·ε)` (`tests/test_lets_be_rational.py`) is relative to the conditioning, so by construction it cannot see this.
**Change.** Treat `otm_price <= 1e-12·max(S,K)` (the same band already used below the bound) as "no time value": return 0/NaN with a new reason `no_time_value`, and add a `vega` (or `cond = 1/vega`) field to `ImpliedVolBatch` so SF-4's weighting and the Surface tab's notes can use it; add a test that pins the behaviour on both sides of intrinsic. Sharpens SF-4/SF-5. **Effort: S.**

### 2. The surface fit mislabels the project's own arbitrage-free synthetic chain as arbitrage-laden — concrete evidence for SF-3 being first
**Evidence** (`profile_surface.py`): `sample_chain(5000, 0.04, 0.012)` → `to_quotes` → `parse_quotes` → `fit_market_surface` with the generator's own rate and yield: 136 quotes dropped as "outside the no-arbitrage bounds", 21 calendar violations, 5 maturities flagged "no arbitrage-free smile can follow them", a butterfly note at 0.105y, max RMSE 1.36 vol points. The generator is SSVI (arbitrage-free) plus a box spread; the forward used by the fit is `S e^{(r-q)T}` from the sidebar (`surface.py:246`, `:322`), not the chain's box forward.
**Change.** Already SF-3; add this exact run as its acceptance test (expect zero dropped quotes, zero calendar violations, RMSE within the noise), since the current xfails in `tests/test_market_data.py` test only ATM vol and forwards. **Effort: part of SF-3 (M).**

### 3. Three incompatible Greek/price result shapes and two calling conventions across the engines — the UI already carries per-engine adapters
**What is wrong.** `black_scholes.greeks` → untyped `dict` of float-or-ndarray (`black_scholes.py:134-142`); `binomial.price_with_greeks` → `TreeResult` dataclass with `price`, `steps`, `exercise_boundary` (`binomial.py:51-76`); `american.greeks` → `AmericanGreeks(Mapping)` with `basis`/`one_sided` (`american.py:989-1023`); `monte_carlo.price_european(option_types=tuple) -> dict[str, MonteCarloResult]` (`monte_carlo.py:119-131`). `option_type` is positional in `bs`, `binomial`, `implied_vol` but keyword-only in `american.price` (`american.py:730-741`). `tab_pricing.py:85`, `:113`, `:338-354`, `:493` branch on which engine produced the object. Any future engine (PR-5 FD, PR-8 Heston, RK-12 ALO grids) will add a fourth shape.
**Change.** One frozen `Greeks` dataclass (Mapping over `GREEK_ORDER`, with `basis`, `one_sided`, `method`, `engine` fields; NaN where an engine has no value) returned by every engine, and one `price(S, K, T, r, sigma, q=0.0, *, option_type="call", **engine)` convention; keep the old names as thin wrappers for one release. **Effort: M.**

### 4. Batch American IV / de-Americanisation needs a vectorised solver, not a loop over `brentq` — sharpens PR-4 and SF-5 with numbers
**Evidence** (`profile_core.py`): `implied_volatility_alo` 2.7 ms per quote, `implied_volatility_american` (LR-501 tree) 55 ms per quote; a 41-strike expiry costs 110 ms (ALO) or 2.3 s (tree), a 17-expiry chain 1.9 s / 38 s. Yet `american.price` on a 41-strike strip costs 0.55 ms because the normalised boundary is shared and cached (`american.py:434`, `:821-823`); the scalar solver (`implied_vol.py:532-541`) throws that away by re-solving one boundary per Brent iterate per strike.
**Change.** `implied_volatility_alo_batch(prices, S, K, T, r, q, kind)` grouped by `(T, r, q)`: a vectorised safeguarded Newton (vega from the engine's own finite difference or from `_premium`) over all strikes of an expiry at one `sigma` iterate ≈ 20 × 0.55 ms ≈ 11 ms per expiry; expose the same loop as `de_americanise()` returning the European-equivalent price and the BSM IV. **Effort: M.**

### 5. Typing gate: 147 mypy errors without `--strict`, 141 of them from one untyped helper — VL-4 is cheaper than it looks
**Evidence** (`mypy-default.txt`, `mypy-strict.txt`, mypy 1.19): non-strict 147 errors (black_scholes.py 141, american.py 3, dateutil stubs 3); strict 392 (black_scholes 148, strategies 92, lets_be_rational 38, svi 20, quotes 18, binomial 17 …). The 141 in `black_scholes.py` all stem from `_prepare` (`:57`) having no annotations, so `ArrayLike` (which includes `str | bytes`) flows into arithmetic at `:96-113`, `:159-212`. No `py.typed`, no `[tool.mypy]` in `pyproject.toml`, no `types-python-dateutil` pin.
**Change.** Annotate `_prepare -> tuple[np.ndarray, ...]` and `_scalar_or_array -> float | np.ndarray` (removes ~140 errors in an hour); add `[tool.mypy]` with `strict = true` on `american`, `binomial`, `monte_carlo`, `risk`, `scenarios`, `clock` (3–5 errors each today) and `check_untyped_defs` elsewhere; ship `py.typed`; add the job to CI. **Effort: S for the gate, M for strict on `strategies`/`lets_be_rational`.**

### 6. The live deployment is not reproducible: unpinned runtime requirements, no lock file, no Python pin, one undeclared import — not on the roadmap
**Evidence.** Streamlit Cloud installs from `requirements.txt`, which holds ranges (`numpy>=1.26`, `scipy>=1.11`, `streamlit>=1.65,<2`); there is no lock file and no `.python-version`/`runtime.txt` (the devcontainer pins 3.12, CI's oldest job 3.11). `dateutil` is imported directly in `vol_surface/__init__.py:58`, `vol_surface/quotes.py:21`, `market_data/adapters.py:57` but is not in `dependencies` (it arrives via pandas). `ci.yml`'s test job comments "Network canaries and browser tests run in their own workflows", but `.github/workflows/` holds only `ci.yml` (DK-7/DT-8 are undone).
**Change.** `uv pip compile`/`pip-compile` → `requirements.lock` consumed by Streamlit Cloud and by a CI job that installs from it (keep `requirements.txt` ranges for the library); add `python-dateutil>=2.8` to `dependencies`; pin the cloud Python; fix or delete the stale comment. Complements VL-3 (action SHAs). **Effort: S.**

### 7. VL-5 coverage gate: the core is already at 94–99 % except `american.py` (89 %) — set the gate now, per package
**Evidence** (`core-tests.log`, core tests only, 4,070 passed in 142 s on 4 workers): `american.py` 89 % (53 statements missed), `ssvi.py` 93 %, `implied_vol.py` 94 %, `schema.py` 94 %, `binomial.py` 95 %, `surface.py` 97 %, everything else 98–100 %. UI modules are covered only by the AppTests.
**Change.** `--cov-fail-under` of 94 on `options_analyzer` minus `ui` today (raise with each engine), a separate 90 for `options_analyzer.ui`; the 53 missed lines in `american.py` are mostly the overflow branches (`_forward_put` log path, `_premium` with `q < 0`, `_check` errors) — cover them with the extreme inputs the docstrings already describe. **Effort: S.**

### 8. Test design: strong independence, but two structural blind spots
- Four single tests take 13–21 s and are not marked `slow` (`test_market_data.py::test_strike_grid_and_row_budget_over_spot_levels[index-3000-12000]` 21.4 s, `test_default_expiry_set_stays_under_10000_rows` 14.0 s, `test_binomial.py::test_leisen_reimer_accepts_the_maximum_step_count` 13.2 s, the `[high]`-scheme American tests 7–9 s each); they run in all six CI matrix jobs and in the documented `-m "not slow"` local loop. **Effort: S.**
- The LBR accuracy study's tolerance scales with the conditioning (finding 1), and the batch solver is tested for bit-identity with the scalar one (self-consistency). Add a conditioning-independent assertion: for quotes with time value ≤ 1e-12·max(S,K) the solver returns the documented no-time-value outcome, and for every other quote the IV error is below `1e-5` relative. **Effort: S.**
- Positives worth keeping: Greeks vs 40-digit mpmath, position risk vs QuantLib and finite differences, strategy analytics vs closed forms and Monte Carlo, `test_zero_row_is_position_risk_with_zero_pnl` is plumbing-only (greek_ladder calls position_risk) but is backed by the finite-difference tests.

### 9. The Surface tab's calibration costs 75 ms per maturity, 65 % of it in 136 `least_squares("lm")` calls — a budget for SF-4
**Evidence** (`profile_surface.py`): `fit_market_surface` on 17 maturities / 2,329 quotes = 1.27 s; `fit_svi` 1.65 s cumulative under cProfile, of which `least_squares` 1.16 s (136 calls: 3 starts × 20 nfev + up to 8 chunks of 50, `svi.py:161-194`), the Python `jacobian` 0.44 s (3,868 calls), `_svi_start_points` 0.48 s. A full index chain (17 expiries) therefore reruns in >1 s per sidebar change.
**Change.** When SF-4 replaces this with a spread-weighted, constrained fit, set an acceptance budget (≤ 30 ms per slice, ≤ 0.5 s per chain) and vectorise the start-point grid once per chain; until then, cache `fit_svi` per `(maturity, quote hash)` so a rate or spot change only refits what moved. **Effort: part of SF-4 (M).**

### 10. `run_stress_tests` and `leg_table` return display labels, against the module's own stated convention
**Evidence.** `scenarios.py:10-11`: "The risk tables use stable snake_case column and bucket names; display labels belong to the UI." `run_stress_tests` (`scenarios.py:220-231`) returns dicts keyed `"Spot change"`, `"Rate change (bp)"`, `"P&L"`, `"Legs expired"`; `strategies.leg_table` (`strategies.py:543-558`) returns `"Theta / day"`, `"Vega / vol pt"`, `"Days to expiry"`. `greek_ladder` and `vega_buckets` follow the convention. A library user (or RK-3/RK-7) cannot join these tables without string matching.
**Change.** Return `pd.DataFrame` with snake_case columns (`spot_change`, `rate_change_bp`, `pnl`, `legs_expired`, …) and move the labels to `ui/format.py`; keep a deprecation shim for one release. **Effort: S.**

### 11. Cross-module reliance on private names, chiefly `implied_vol` → `lets_be_rational._*`
**Evidence.** `implied_vol.py:193` (`lbr._lets_be_rational_s`), `:209` (`lbr._log_ratio_s`), `:403` (`lbr._lets_be_rational_v`); `scenarios.py:32` imports `strategies._number`; `surface.py:23-35` imports six underscored names from `local_vol`/`quotes`. `lets_be_rational.__all__` (`:73-79`) exposes five functions, none of which is the entry point the package actually uses, so the module's public surface is misleading and refactors of the LBR internals silently break the solver's contract.
**Change.** Promote `lets_be_rational_s/_v(beta, x) -> (s, converged)` and `log_ratio` to public, typed functions with docstrings; move `_number` to a small `options_analyzer._validation` module. **Effort: S.**

### 12. Pricing-tab tree work is 9 inductions per option and the convergence chart up to 0.9 s, recomputed on every step-slider move
**Evidence** (`profile_core.py`, `profile_surface.py`): `binomial.price_with_greeks` (LR-501, American) 64 ms = 1 tree + 2 for theta (`binomial.py:374-380`) + 6 for vega/rho/epsilon (`:304-313`), × 2 option types per rerun; the convergence sweep (`ui/cache.py:80-106`) 384 ms at the default 501 steps and 911 ms at 2,000 (American), and its cache key includes `max_steps` although the two `extrapolated_price` references (2001 + 4003 steps, 90 ms each) do not depend on it.
**Change.** Split `tree_convergence` into a cached reference (keyed on the contract only) and a cached sweep; read vega/rho/epsilon for the CRR/LR tree from one extra tree per bump using the already-built lattice (reuse `_lattice` and `ups_log`) or document that the tree Greeks beyond delta/gamma/theta are one-click-deeper. Minor next to the ALO model, which the app now defaults to for American exercise. **Effort: S–M.**

### 13. Engineering items the roadmap lists that are now cheap to close, with measured starting points
- VL-6 benchmark history: `profile_core.py` already reproduces the scorecard figures on this box (batch IV 18 ms/10k, ALO 1.5 ms uncached / 0.25 ms cached, LR-2001 23 ms, VaR 50k paths 33 ms); commit it as `benchmarks/bench_core.py` emitting JSON and chart it per tag. **S.**
- VL-1: a European-price/Greeks/tree oracle against QuantLib is a 60-line addition to `tests/oracle/test_ql_alo.py`'s `_option` helper (`:38-52`). **S.**
- VL-2: Hypothesis is used only in `tests/test_american.py` and the harness; the invariants in `test_black_scholes.py` (parity, bounds, monotonicity) are a ready-made property suite. **S.**

---

**What is already at the bar (no change needed):** zero-volatility American values agree exactly between the ALO closed form and the tree's 4,001-point search (`probe_numerics.py`, diffs ≤ 4e-15); LR tree theta matches BSM theta to 1e-6 per year; `implied_volatility_batch` and the scalar path agree bit for bit; the scalar and strip APIs of `american.price` behave as documented under hand-over and double-boundary fallback (16.5 ms, 7 ms and 101 ms respectively); the test harness (strict markers/xfail, RuntimeWarning as error, oracle-skip-fails-CI, network guard) is stronger than most internal desk repositories.

---

# model-validator

## Model-risk validation findings (SR 11-7 lens)

Scripts and raw outputs: `<scratch>/panel/model-risk/` (`probe_a_ql_european_tree.py` → `out_a.txt`, `probe_b_alo.py` → `out_b.txt`, `probe_c_mc_var_strat.py` → `out_c.txt`, `probe_d_followups.py` → `out_d.txt`, Fed scenario text in `fed/scen.txt`). Nothing in the repo was modified.

**What passed independent probing (worth recording in the validation report):** BS prices and all six first-order Greeks vs QuantLib `AnalyticEuropeanEngine`, 1,500 random options: worst 3.4e-12 relative. ALO vs QdFp strict on 240 off-grid options (1–730 days, S/K 0.37–2.7, q ≥ r, r = q): fast 7.6e-7, accurate 8.2e-9, high 4.0e-10. ALO theta vs QdFp strict (time-scaling trick, see #6): agree to 1e-5 at 1, 2, 7, 30, 365 days; vega/rho/dividend-rho to 1e-5. `implied_volatility` with dividends vs 40-digit mpmath, 400 quotes: worst 1.64·cond·ε (claim is 4). `strategies.analyze` PoP and E[P&L] vs mpmath quadrature: 4e-16 / 2e-14. Test count 5,004 confirmed. Fed 2026 figures (−58 %, VIX peak 72, 3-month 3.7 % → 0.1 %) confirmed against the Fed PDF, except the multiplier (#3).

---

### 1. Local volatility at the front expiry is off by up to 2 vol points on a listed (monthly) expiry grid, with noise-free quotes — and no test covers that spacing. (M)
- **Evidence:** `out_c.txt` §5: noise-free SSVI quotes at 30/60/91/182/273/365/547/730 days, implied-vol fit error 3e-11, yet local-vol error at T=0.082 is max 0.0213 (2.1 vol pts), median 0.44 pts; last expiry 0.66 pts. The only noise-free test, `tests/test_vol_surface.py:217` `test_noise_free_quotes_recover_the_surface…`, uses `range(182, 731, 14)` (two-week spacing from 6 months), where the secant error is <0.1 %; the noisy test (`:355`) tolerates 5 pts at the front. Methodology `surface.md:18-23` describes the secant scheme but states no error.
- **Change:** take ∂w/∂T from a monotone (PCHIP/Hyman) interpolant of total variance in T at fixed y, or from a parametric T-fit (SSVI-in-T), with the end formula's truncation error reported on the Surface tab; add the listed-expiry grid above as a test with a stated bound (≤0.5 pt). Sharpens SF-7/SF-4 (the roadmap has no item for the T-derivative itself).

### 2. Monte Carlo 95 % intervals under-cover once fewer than ~300 paths finish in the money; the app shows the interval with no flag. (S)
- **Evidence:** `out_d.txt` §4, 400 seeds, 20,000 antithetic paths, 30-day call: coverage 0.848 at ~10 ITM paths, 0.887 at ~20, 0.907 at ~50, 0.935 at ~100, 0.938 at ~200, 0.960 at ~400. The repo's only coverage test (`tests/test_monte_carlo.py:34`) is one ATM case with a 0.90–0.99 acceptance band, which cannot catch 0.887. `options_analyzer/ui/tab_pricing.py:103-106` prints the CI whenever the SE is finite; `monte_carlo.py:69-77` exposes `itm_fraction` but no reliability flag.
- **Change:** flag "interval unreliable" when `itm_fraction·n_paths < 300` (or switch the SE to a Wilson/bootstrap interval for rare payoffs); make the coverage test a 2-sided binomial check (e.g. 0.925–0.975 over 400 runs) across ITM levels. Sharpens PR-6 with the threshold.

### 3. The "Fed severely adverse (2026)" stress scenario overstates the Fed's vol shock by ~45 %. (S)
- **Evidence:** `options_analyzer/scenarios.py:163-170` uses `vol_multiplier=4.0` and says "VIX peaks at 72 (about 4x)". The Fed's final 2026 scenario (`fed/scen.txt` lines 246-247, 440, 616-619, 671-673) gives the jump-off VIX as 26 % in Q4 2025 and the peak as 72 %: ratio 2.77. The equity (−58 %) and rate (3.7 → 0.1 %) paths are correct. Also the shock applies the equity trough (Q3 2026) and the VIX peak (Q2 2026) simultaneously; say so in the description.
- **Change:** `vol_multiplier=72/26`, record the jump-off in the description, add a test that pins each historical/Fed scenario to its sourced numbers (the historical VIX ratios are consistent with FRED, spot-checked).

### 4. VaR/ES for option positions has no exact comparator in the tests; one is cheap and the engine passes it. (S)
- **Evidence:** `tests/test_risk.py` checks a stock position against the log-normal quantile (`:13`) and only "full revaluation > delta-normal" for options (`:52`). `out_c.txt` §2: against a 200,001-point Gauss quadrature of the single-factor P&L, 1-day short straddle VaR 0.8962 (CI 0.8816–0.9182) vs exact 0.8954, ES 1.1697 vs 1.1714; 10-day 7.335 (CI 7.215–7.457) vs exact 7.258; iron condor and covered call likewise inside the CI. Student-t 99.9 % stock VaR: 6.760 (CI 6.63–7.00) vs exact 6.948 — inside, but note how wide the order-statistic CI is at 200k paths.
- **Change:** add that quadrature as an oracle test (normal and Student-t, 1 and 10 days, three positions) with the CI as the tolerance; it also gives the first independent check of `_var_es` and `_settlement_days`. Sharpens RK-4/RK-6 (those add backtests and an analytic VaR, not an exact MC check).

### 5. VL-1 (QuantLib oracle for European prices, Greeks and trees) is a 30-minute job — `probe_a_ql_european_tree.py` is the test — with two traps the roadmap does not mention. (S)
- **Evidence:** `out_a.txt`: LR European matches QuantLib's `leisenreimer` to 2.7e-10 at identical steps; CRR American tree delta/gamma match QuantLib's to 1e-6. But (a) the project prices calls as the symmetric put (Methodology `pricing.md`), so "CRR" for a call differs from QuantLib's direct call lattice by 1.4e-2 at 50 steps (3-year call, q=10 % > r=8 %) — the oracle must compare puts, or calls against the put-symmetric lattice; (b) QuantLib's own LR American tree is erratic at some step counts (`out_d.txt` §2: n=51 call 4.3428 vs ours 4.4763, n=201 4.4392; n=501 put 6.0809 vs ours 6.0898; identical at 25, 101, 1001), so an American LR oracle must use converged references (ALO `high`), not step-matched QuantLib.
- **Change:** commit the probe as `tests/oracle/test_ql_european.py` with tolerances 1e-11 (BS), 1e-9 (LR European), and ALO `high` as the American tree reference; state trap (a) in the Pricing tab's convergence chart caption.

### 6. American Greeks have no external oracle in the repo (only a 4001-step tree at one point); an exact one exists and QuantLib's FD theta is *not* it. (S)
- **Evidence:** `tests/test_app_alo.py:400-408` is the only Greek cross-check (one ATM put, rel 1e-3). `out_b.txt` §2: QuantLib `FdBlackScholesVanillaEngine(2000,2000)` theta disagrees with ours by 4 % (7-day ATM put, −42.64 vs −40.99) and 5× (2-day K=105 put, −0.69 vs −3.20). `out_d.txt` §1 settles it in our favour: pricing the scaled contract V(fT, r/f, σ/√f, q/f) with QdFp strict and bumping ±1 day gives −40.99313 / −3.20072, matching the app to 1e-5; the European theta is −40.68 / −2.44, so these are not closed-form fallbacks.
- **Change:** add the time-scaling QdFp oracle for theta and central-difference QdFp for vega/rho/ε across ~10 cases incl. 1–7 DTE; document in `docs/methods/american_alo.md` that FD-engine Greeks are unsuitable references near expiry.

### 7. The double-boundary fallback is documented as "about 1e-5 accurate"; it is ~1e-4 beyond a year. (S)
- **Evidence:** `docs/methods/american_alo.md` (Limits, Regimes) and `tests/oracle/test_ql_alo.py:test_the_double_boundary_put_against_quantlib_finite_differences` rest on one 1-year case. `out_d.txt` §3, reference = Richardson of QuantLib FD 4000/8000: error +8.3e-6 at 1 year, +1.08e-4 at 5 years, +1.17e-4 at 10 years (always high, consistent with first-order LR bias not fully removed by `binomial.extrapolated_price`'s c/n assumption, `binomial.py:421-429`).
- **Change:** oracle over 6–8 double-boundary cases with a T-dependent tolerance (e.g. 2e-5·max(1,T)), and amend the method note.

### 8. Property testing is thin and never explores: 5 `@given` properties, all in `test_american.py`; CI runs `derandomize=True`; the `nightly` profile exists but no workflow uses it. (S)
- **Evidence:** `grep -c "@given" tests/*.py` → only `tests/test_american.py:5`; `tests/conftest.py:92-104`; `.github/workflows/ci.yml` has no `schedule:`; `CONTRIBUTING.md:29` mentions nightly. Meanwhile the 35-s 100k mpmath study (`test_lets_be_rational.py:127`) runs on every push.
- **Change:** a scheduled workflow (`HYPOTHESIS_PROFILE=nightly`, oracle + slow) and move `slow` out of the push job; VL-2 should name the identities to cover (parity, symmetry, monotonicity in σ/T/K, bounds, batch-vs-scalar IV) per module. Sharpens VL-2/VL-11.

### 9. Only one engine has a method note; the improvement guide's own rule ("every engine ships with its method note") is unmet for six. (M)
- **Evidence:** `ls docs/methods` → `american_alo.md` only. Undocumented limits found while probing: `extrapolated_price`'s Richardson assumption (#7); `implied_volatility_american` on the default 501-step LR tree returns vols 1.6e-5–4.9e-5 off (`out_b.txt` §4) while `iv.md` says only "inverted on a 501-step Leisen–Reimer tree"; MC CI reliability (#2); the local-vol T-derivative error (#1); VaR mixing trading-day diffusion with 1/252-year life reduction.
- **Change:** `docs/methods/{lets_be_rational,binomial,monte_carlo,var,vol_surface}.md` in the ALO note's structure (model, method, error, limits, references), each with a test that reproduces its headline numbers.

### 10. README overstates the IV accuracy population. (S)
- **Evidence:** README "accurate to a few ulps … on 97,086 cases checked at 60 digits"; the test (`tests/test_lets_be_rational.py:127-134`) applies the `max(1e-14, 4·cond·ε)` bound to the 66,738 quotes that keep time value, and the scorecard says so. 474 quotes rounding below intrinsic give NaN (same comment).
- **Change:** README → "66,738 quotes with time value out of 97,086 cases", and say what happens to the rest (Honesty criterion 6).

### 11. Methodology › Validation is a placeholder, and VL-7 has no content spec; the tables above are the content. (M)
- **Evidence:** `options_analyzer/ui/methodology/validation.md` (3 lines) and `tab_methodology.py:54` `NOT_GENERATED`; `scripts/validation_report.py` referenced at `:46` does not exist.
- **Change:** have VL-7's generator emit exactly: oracle error by scheme (grid + off-grid), Greek errors vs QdFp, MC coverage vs ITM paths, VaR/ES vs quadrature, local-vol error by expiry spacing, with commit, QuantLib version and tolerances; fail CI when a table exceeds its tolerance.

### 12. BS Greeks' mpmath oracle covers 5 parameter points, none near the documented edge cases. (S)
- **Evidence:** `tests/test_black_scholes.py:19-25` (`CASES`), `:46-78`; the degenerate branches in `black_scholes.py:160-175` (T→0, σ→0, S→0, kink averaging) are checked only against hand-written expectations (`:95-120`).
- **Change:** 200 random cases incl. 1-day, σ=0.5 %, S/K 0.3–3, negative r and q; it is the same harness, minutes of work; QuantLib agreement in #5 already shows the formulas are right, so this is a coverage statement for the report.

### 13. The tree Greeks' and American-IV tolerances in tests are loose relative to what the engines achieve. (S)
- **Evidence:** `tests/test_binomial.py:62-69` theta `abs=0.02` on a theta of about −6; `:200-206` vega/rho/ε `abs=1e-3` of ~0.12. My probe shows CRR tree delta/gamma agree with QuantLib to 1e-6 and theta to 6e-4 (`out_a.txt`), so a mutation that mis-scales theta by 0.3 % would pass today.
- **Change:** tighten to measured error × 3 and record the measurement in the test comment, as `test_american.py` does; it is the cheapest step toward the 90 % mutation-kill bar (VL-11).

---

# product-designer

# Product-design review of the running app (principal designer, data-dense trading software)

Method: `git archive` copy run on port 8795; Chromium via Playwright at 1440×900 and 390×844, dark and light, every tab plus model variants (ALO, Monte Carlo, binomial), uploads, popovers, drawer and edge states. All screenshots are under `<scratch>/panel/product-designer/shots/` (`<tab>_<desk|phone>_<dark|light>_<fold|full>.png`) and `.../shots/states/`. Scripts: `shoot.py`, `states.py`, `probe2.py` in the parent folder. Settle times were 2.5–3.3 s per cold tab load and 0.79 s for a spot change on Strategy; no exceptions on any tab in either theme or viewport, and no horizontal page scroll at 390 px. The shell (header, ticker, cards, tabs, Methodology drawer, share popover, light theme) is genuinely good; the findings below are what is left between it and a desk tool.

Ranked by value to a professional user.

---

### 1. The Surface tab cannot read the chain files the library was built to read; the user sees a pandas C-parser error
**Wrong:** The project ships `market_data.read_chain` (long, yfinance and Cboe layouts, with cleaning notes) but the Surface uploader calls only `vol_surface.parse_quotes`. Uploading the repository's own fixtures `tests/fixtures/chain_long.csv` and `chain_cboe_wide.csv` yields a red box: *"Could not build a surface from this file: Error tokenizing data. C error: Expected 3 fields in line 4, saw 4"*. A quant who downloads a Cboe chain, which is exactly what the data policy tells them to do, hits an internal library message, not a product message.
**Evidence:** `shots/states/surface_csv_long.png`, `surface_csv_cboe.png`; `options_analyzer/ui/tab_surface.py:198-199` (`except (QuoteParseError, ValueError, ParserError) as exc: st.error(f"Could not build a surface from this file: {exc}")`); `grep -n "market_data\|read_chain" options_analyzer/ui/*.py` → 0 matches; `read_chain("tests/fixtures/chain_long.csv", as_of=...)` → `(36, 9)` rows, 4 notes.
**Change:** In the uploader, try `read_chain` first (it detects the layout) and fall through to `parse_quotes`; render its notes as a "Dropped N quotes" expander; replace the raw exception with a product sentence that names the expected columns and links the two sample downloads. Sharpens SF-11 (data-source panel) and DT-5a, but the wiring itself is not on the roadmap.
**Effort:** M.

### 2. No tab shows its primary chart above the fold at 1440×900
**Wrong:** The first screen of every tab is controls and cards. Measured tops (page y, fold = 900): Strategy P&L chart 1,002 px; Surface first chart 712 px (only its title is visible); Risk VaR heading 3,935 px, Stress heading 2,543 px. On Pricing the first visual is the sensitivity line chart at ~690 px, below a 6-row Greeks table that repeats numbers also in the chain. A desk screen puts the payoff beside the builder, not under it.
**Evidence:** `shots/strategy_desk_dark_fold.png`, `surface_desk_dark_fold.png`, `risk_desk_dark_full.png` (4,945 px tall, 5.5 screens); `probe2.py` output above.
**Change:** Strategy: two columns at ≥1200 px (legs editor + cards left, P&L chart right, chart min-height 420). Surface: move the three SSVI number inputs and five sliders into a collapsed "Surface parameters" popover like the Strategy "Assumptions" one, and put the smile chart first. Risk: a headline strip (max loss, VaR, ES, delta-normal, position Greeks) at the top with a report selector. Sharpens BK-1, BK-2 and PR-7 with these numbers as the acceptance test ("first chart top < 600 px at 1440×900").
**Effort:** L (BK-2), M (Risk strip), S (Surface popover).

### 3. One screen shows five different number standards for the same quantity
**Wrong:** On Strategy alone, contract quantity is `1.00` in the legs editor but `1` in "Legs at inception"; strikes are `100` in the editor and `100.00` in the sidebar; dollar values appear at 4 dp (`$10.4506` model value), 2 dp (`$441.05`), 0 dp (`+504` heatmap cells, no `$`); Greeks at 6 dp (Pricing table), 4 dp (`-0.1026` Gamma card), 2 dp (`18.72` Delta). The Gamma card is the only Greek card without a unit in its label. The stress table's "Spot change" column mixes `-20.5%` and `15%` (Streamlit's `percent` format).
**Evidence:** `shots/strategy_desk_dark_full.png`, `risk_desk_dark_full.png` (stress table rows); `tab_strategy.py:48-50` (`Quantity` NumberColumn `format="%.2f"`), `:53-54` (`STRIKE_FORMAT`), `:993-998` (`$%.4f`); `tab_risk.py:262` (`format="percent"`).
**Change:** Publish the number standard as a table in `format.py` (price 2 dp for ≥ $1 else 4; quantities integer; Greeks 2 dp in shares/$; percentages 1 dp fixed) and make every column_config and card use it; add a test that greps `column_config` formats against the table. This is DK-4/UI-3; the sharpening is the concrete list above and the test.
**Effort:** M.

### 4. The Surface tab speaks textbook units, not desk units
**Wrong:** Smile x-axis is "Moneyness K/S"; the term-structure table gives "Maturity (years)" to 3 dp (`0.080`, `0.146`) rather than expiry dates or days; skew is "90–110 moneyness" rather than 25-delta risk reversal and butterfly; two 3D surfaces take 500 px of height each with the mesh occupying about a third of the box, while the desk-relevant smile-by-expiry chart sits 1,700 px down.
**Evidence:** `shots/surface_desk_dark_full.png` (3D boxes ~y 480–840, smiles at ~y 930, table at ~y 1550).
**Change:** Default the smile x-axis to delta (toggle to log-moneyness), label maturities as dates with days, add RR25/BF25 columns, demote the 3D surfaces to an expander. This is SF-6; the sharpening is to make the 3D demotion and the date labels part of it.
**Effort:** M.

### 5. The scenario heatmaps are unlabelled numbers on phones and lack units on desktop
**Wrong:** Cells read `+504`, `−157` with no `$`, and the unit rule ("in thousands (k) or millions (M) when large") lives in a caption below the chart. At 390 px the spot × IV grid is 6 columns of 3-digit labels at about 9 px, and the spot × date grid is 7 columns; neither is readable, and the x tick labels rotate to 45°.
**Evidence:** `shots/risk_phone_dark_full.png`, `risk_desk_dark_fold.png`; `charts.py:300, 590` (`texttemplate="%{text}"`).
**Change:** Put the unit in the colour-bar title or the axis title ("P&L, $"), and on viewports under 600 px drop cell text and show the value on tap, or reduce to a 5×7 grid. Sharpens UI-6 (phone pass) with a concrete rule: no chart text below 11 px.
**Effort:** S.

### 6. Explanatory prose where a number or a control should be
**Wrong:** Several blocks explain instead of show: the Pricing "Model diagnostics" box under BSM is an `st.info` telling the user to switch models; the IV tab opens with a two-line caption before any control; the Strategy template line ("Outlook: Moderately bullish · Structure: +1 call @ K, -1 call @ K+w — Buy a call and sell a higher-strike call: ...") runs to two lines; the MC and ALO diagnostics end in 6–8-line paragraphs; the share popover carries a 3-line paragraph under the URL.
**Evidence:** `shots/pricing_desk_dark_full.png` (info box at y≈1,220), `iv_desk_dark_fold.png`, `states/pricing_mc_full.png`, `states/risk_share_popover.png`.
**Change:** Move each paragraph into the Methodology drawer (already one click away) and leave a one-line caption with a "Methodology ›" link; under BSM show the diagnostics section as a compact put–call parity check (C − P vs S·e^(−qT) − K·e^(−rT)) instead of a prompt. Not on the roadmap.
**Effort:** S.

### 7. The theoretical option chain is hidden in an expander and has no desk columns
**Wrong:** The chain, the single most-used desk view, is collapsed by default under "Theoretical option chain" and lists Call value / Call delta / Call theta / Strike / Put ... with no bid–ask, no IV column, no ATM highlight, and strikes stepped by 5 from 80 to 120 regardless of spot. The improvement guide says "the Pricing tab has no chain"; it does, but it is invisible.
**Evidence:** `shots/states/pricing_chain_expander.png`.
**Change:** Make the chain the Pricing tab's main table (open by default, above the Greeks table), centre it on the ATM row with a highlighted band, add IV and Δ columns per side, and use the synthetic chain's exchange strike intervals (`market_data.sample_chain`) rather than a fixed 5-point step. Sharpens UI-1/PR-7.
**Effort:** M.

### 8. The stress table uses live-looking checkboxes in a read-only report
**Wrong:** "IV damped" is a `CheckboxColumn` in a `st.dataframe`; it renders as toggles that look editable but are not. Readers expect a mark, not a widget.
**Evidence:** `shots/risk_desk_dark_full.png` (table at y≈3,340); `tab_risk.py:266-269`.
**Change:** Render as text ("yes"/"—") or fold it into the IV multiplier column ("4.10× damped").
**Effort:** S.

### 9. The legs editor and position tables are unusable at 390 px
**Wrong:** On phones the data editor shows only Type/Quantity/Strike/Days; IV and Premium need an inner horizontal scroll with no affordance; "Legs at inception" is cut at "Cost $1,04"; the Surface term-structure table is cut at "Skew 90–11".
**Evidence:** `shots/states/phone_legs_editor.png`, `shots/strategy_phone_dark_full.png` (bottom), `surface_phone_dark_full.png`.
**Change:** Under 600 px, render legs as stacked cards (one per leg, two rows each) with an "Edit legs" dialog, and give wide tables a per-row "more" column or a transposed layout. Sharpens UI-6.
**Effort:** M.

### 10. The sidebar has no expiry date, no valuation time and a 4-dp year fraction
**Wrong:** Expiry is "Days to expiry 365" with the caption "T = 1.0000 years"; the header shows "27-Oct-27 (365d)". A desk user types an expiry date (or picks a listed one) and sees the day count; 4 dp on a year fraction is noise. The project already has the market clock and the NYSE calendar.
**Evidence:** `shots/pricing_desk_dark_fold.png` (sidebar); `inputs.py:57-65, 100`.
**Change:** Date input with listed-expiry presets (monthly/weekly from `clock`), computed days shown beside it, and "T = 1.00 y" at 2 dp. This is DT-2; the sharpening is to make the days-caption format part of it.
**Effort:** M.

### 11. The ALO badge wraps the ticker and shifts the whole page
**Wrong:** With the American engine selected the badge reads "ALO · accurate · American"; the ticker wraps to two lines and every element below moves down ~16 px, so the tabs sit at a different y than under BSM. Switching models should not reflow the shell.
**Evidence:** `shots/states/pricing_alo_full.png` vs `shots/pricing_desk_dark_fold.png`; `header.py` two-line width rule.
**Change:** Shorten the badge ("ALO · acc · Am" with the full text in its tooltip) or give the ticker a fixed two-line height on all models.
**Effort:** S.

### 12. The Surface tab's colour-scale selectbox is a full-width cosmetic control above the data
**Wrong:** "Colour scale: Viridis" occupies a full-width row between the arbitrage cards and the charts; a cosmetic choice sits above analytical content and pushes the first chart below the fold.
**Evidence:** `shots/surface_desk_dark_fold.png` (y≈650–700).
**Change:** Move it into the parameters popover or beside the download buttons; default to the theme's sequential ramp so it is rarely needed.
**Effort:** S.

### 13. The P&L chart has an unlabelled dashed vertical line
**Wrong:** On the Strategy chart the "spot" line is labelled but the second dashed line at the short strike (110) is not; on the condor there are four. Reference lines without labels force the reader to guess.
**Evidence:** `shots/strategy_desk_dark_full.png` (x≈830 in the chart); `tab_strategy.py:930-935` (`strategy.strikes` passed as vlines).
**Change:** Label strike lines at the top ("K 110") in the muted colour, and on hover show the leg.
**Effort:** S.

### 14. Card rows carry one-off labels instead of a Greek strip with one unit convention
**Wrong:** Strategy Greeks: "DELTA (SHARES) 18.72 · GAMMA −0.1026 · THETA / DAY −$0.14 · VEGA / VOL PT −$2.05": delta in shares, gamma unlabelled, theta and vega in dollars. Risk shows none of these at the top. The desk convention is dollar Greeks for all four, with shares-delta as a secondary line.
**Evidence:** `shots/strategy_desk_dark_full.png` (y≈840); `strategies.position_risk` already returns `dollar_delta` etc.
**Change:** One shared Greek strip component (Δ$, Γ$ per 1 %, Θ/day, Vega/pt) used by Strategy and Risk, with the share-count delta as the card caption. Sharpens UI-4 (shared what-if strip) and BK-1.
**Effort:** S.

### 15. The Monte Carlo price cards hide the confidence interval behind a caption
**Wrong:** Under MC the fourth card is "STD. ERROR $0.0088" and the caption says "95 % CI = price ± 1.96 SE"; the user has to compute the interval. The diagnostics table far below shows it.
**Evidence:** `shots/states/pricing_mc_full.png`.
**Change:** Card captions on the call and put ("10.4237 – 10.4582, 95 %") and the reliability flag from PR-6 in the same caption when the estimate is unreliable.
**Effort:** S.

### 16. The Methodology column is narrower than every other tab
**Wrong:** The Methodology sections end at x≈1,093 while the card row and tab rule end at 1,370; a reasonable reading measure, but the ragged right edge against the header looks unplanned.
**Evidence:** `shots/methodology_desk_dark_full.png`.
**Change:** Keep the measure but centre the column or give the right 20 % to a sticky contents list (the current contents line is inline text).
**Effort:** S.

### 17. Light-mode sliders use the dark amber as the track, which reads brown
**Wrong:** `[theme.light] primaryColor = "#9E5F00"` was darkened for text contrast, but on slider tracks and the segmented-control outline it reads as muddy brown; the dark theme's `#E8A33D` reads as the brand amber.
**Evidence:** `shots/surface_desk_light_fold.png` (sliders at y≈387, 461); `.streamlit/config.toml` light block.
**Change:** Keep `#9E5F00` for text and borders but override the slider track and thumb to a lighter amber (`#C98500`, the light categorical slot 3, 3.1:1 on white) in `desk.css` (`[data-testid="stSlider"]`).
**Effort:** S.

### 18. The Surface "found implied vol" info box is the only semantic alert used for a normal value
**Wrong:** A blue `st.info` with a crosshair icon announces the surface vol at the sidebar strike (23.65 % vs sidebar 20.00 %). Alert styling is for conditions, not readouts, and it is the largest element above the fold on Surface.
**Evidence:** `shots/surface_desk_dark_fold.png` (y≈575–625).
**Change:** Make it a fifth card ("Surface IV at K · 23.65 %" with the sidebar vol as the caption) in the existing card row.
**Effort:** S.

---

**What already meets the bar and should not be touched:** the header ticker and badge, the four-card grid tiers, the Methodology drawer with KaTeX, the share link semantics, the diverging and sequential palettes in both themes, the time-decay and position-Greek charts, and the absence of any exception or horizontal page scroll in 24 tab/viewport/theme combinations.
