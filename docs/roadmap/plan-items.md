# Upgrade plan: items and acceptance criteria

Working document for the v3 upgrade. Each item names the files it touches, what it does and the acceptance tests that
decide whether it is done. "Today" figures were measured on the v2.0.0 baseline (commit ae5b1ca). The item index with
file ownership comes first, then the sequencing notes, then the full text of each item.

SEQ: "PHASE 0 (lead, serial on main, about 2-3 days): P0-1 to P0-4. These items create the seams (lazy tabs and build_position; components.py; charts.show; model_config.py split from inputs.py; methodology split by area), pre-provision every dependency and the pytest and CI conventions, and tag v2.0.0. After P0, only the owner of a file edits it.\n\nFILE OWNERSHIP (disjoint, so each workstream gets its own git worktree):\n- WS-PRICING: black_scholes.py, implied_vol.py, binomial.py, monte_carlo.py, american.py, lets_be_rational.py, dividends.py, heston.py, THIRD_PARTY_NOTICES.md; ui/model_config.py, ui/cache.py, ui/tab_pricing.py, ui/tab_iv.py, ui/charts_pricing.py; their tests, oracle tests and bench files.\n- WS-SURFACE: options_analyzer/vol_surface/* (package); ui/tab_surface.py, ui/charts_surface.py, ui/surface_state.py.\n- WS-RISK: strategies.py, scenarios.py, risk.py, backtest.py, attribution.py, surface_risk.py, margin.py.\n- WS-BOOK: ui/tab_strategy.py, ui/tab_risk.py, ui/charts_risk.py, ui/risk_cache.py.\n- WS-DATA: clock.py, termstructure.py, market_data/*, ui/inputs.py, ui/data_cache.py, .github/workflows/canary.yml.\n- WS-DESK: both config.toml copies, ui/theme.py, ui/static/*, ui/charts.py, ui/format.py, ui/components.py, ui/app.py, the entry points, ui/tab_methodology.py, scripts/screenshots.py, tests/test_app.py, tests/e2e/*, .github/workflows/e2e.yml.\n- WS-VALIDATION: pyproject.toml, requirements*.txt, ci.yml and the other workflows, tests/conftest.py, tests/oracle/conftest.py, tests/test_properties.py, tests/test_packaging.py, benchmarks/conftest.py and test_bench.py, scripts/validation_report.py, scripts/scorecard.py, scripts/bench_table.py, docs/*, mkdocs.yml, CHANGELOG.md, CITATION.cff.\n- Shared methodology content: each workstream owns its own ui/methodology/<area>.md.\n- New tests, fixtures and bench files use names that are unique to their owner.\n- A change needed in someone else's file goes into that owner's queue or into INT-1, never into your branch.\n\nPHASE 1 (about 2 weeks, all seven workstreams in parallel):\n- Merge first, in week 1, because others depend on them: DK-1, DK-3, DK-4 (design system); DT-1 (clock) and the DividendSchedule/RateCurve part of DT-3; PR-1 (LBR batch API); SF-1 and SF-2 (package split); RK-1 (position_risk and ladder); VL-1, VL-2, VL-3.\n- Then: PR-2 -> PR-3 -> PR-4 -> PR-6; SF-3 -> SF-4 -> SF-8; RK-6, RK-2, RK-3, RK-5, RK-4, RK-9; DT-2, DT-4, DT-5, DT-6; DK-2, DK-5; VL-4 (non-blocking), VL-5, VL-6.\n- WS-BOOK starts in week 2 on the parts of BK-2 that only need DK-3/DK-4 and the existing core (tiles, Range control, P&L table).\n\nPHASE 2 (about 2-3 weeks, parallel):\n- PR-5 (needs DT-3), PR-7.\n- SF-5 (needs PR-1 and PR-4), SF-6, SF-7, SF-9 (needs DT-1), SF-11 (needs DT-4 and DT-5).\n- RK-7 (needs SF-8), RK-8.\n- BK-1 (needs RK-1 to RK-6), BK-3 (needs SF-8 and SF-11), the rest of BK-2.\n- DK-6, DK-7.\n- VL-7 (needs RK-4 and the oracle tests), VL-8, VL-9, VL-10.\n\nPHASE 3 (lead, serial, about 1 week): INT-1 merge train. Order: WS-DESK, WS-DATA, WS-PRICING, WS-RISK, WS-SURFACE, WS-BOOK, WS-VALIDATION, each followed by the full suite and e2e. Then INT-2 README and the v3.0.0 release. 'Could' items (PR-8, SF-10, RK-10, BK-4, DT-7, DT-8, DK-8, VL-11) start only after every 'must' has merged.\n\nCUT LINE if time runs short: ship every 'must' plus INT-1 and INT-2. Those alone fix the largest measured errors (forwards up to -318 bp, 101 vs 20 VaR exceptions, LR-501 RMSE 5.8e-4, 9.5 s page load) and produce the validation report and scorecard.\n\nDEFINITION OF DONE for every item:\n- edits only owned files;\n- ruff clean, and mypy clean on owned files;\n- no RuntimeWarning;\n- passes with pytest -n auto;\n- each new public function is typed, with keyword-only arguments after the core ones and a NumPy docstring with a runnable Examples doctest;\n- property, oracle or golden tests with the stated tolerances;\n- a benchmark if the code is on a hot path;\n- the owner's methodology file 
== WS0 Foundations (lead, serial on main before any worktree branches) [None, None, None, None]
== WS-PRICING Pricing engines and implied vol (core, plus the Pricing and IV tabs) [None, None, None, None, None, None, None, None]
== WS-SURFACE Volatility surface (calibration core and Surface tab) [None, None, None, None, None, None, None, None, None, None, None]
== WS-RISK Position and portfolio risk analytics (core library, no UI) [None, None, None, None, None, None, None, None, None, None]
== WS-BOOK Strategy Builder and Risk console (UI) [None, None, None, None]
== WS-DATA Market data, rates and time [None, None, None, None, None, None, None, None]
== WS-DESK Design system and app shell [None, None, None, None, None, None, None, None]
== WS-VALIDATION Independent validation and engineering [None, None, None, None, None, None, None, None, None, None, None]
== WS-INT Integration and release (lead, serial, last) [None, None]

WS0 Founda | P0-1 [P0] Lazy tabs and a UI-free position builder | must
WS0 Founda | P0-2 [P0] UI seams: shared components, one chart entry point, model-config split, methodol | must
WS0 Founda | P0-3 [P0] Build and CI skeleton every workstream builds on | must
WS0 Founda | P0-4 [P0] Tag the v2.0.0 baseline and record the starting numbers | should
WS-PRICING | PR-1 [P1, merge first] Vectorised Let's Be Rational implied vol (port of Jäckel revision 1 | must
WS-PRICING | PR-2 [P1] Andersen-Lake-Offengenden spectral-collocation American engine | must
WS-PRICING | PR-3 [P1] ALO in the app: engine selection, smooth American Greeks, correct convergence re | must
WS-PRICING | PR-4 [P1] Fast, unbiased American IV and a batch de-Americanisation API | must
WS-PRICING | PR-5 [P2] Discrete cash dividends: finite-difference engine on the spot model | should
WS-PRICING | PR-6 [P1] Monte Carlo honesty, randomised QMC and core edge cases | should
WS-PRICING | PR-7 [P2] Pricing and IV tabs in the desk layout | should
WS-PRICING | PR-8 [P3, gated] Heston with the Albrecher characteristic function, Lewis pricing and cali | could
WS-SURFACE | SF-1 [P1, day 1] Remove the literal 'None' cells and fix message severity on the Surface t | must
WS-SURFACE | SF-2 [P1] Split vol_surface.py into a package and break up its two largest functions | must
WS-SURFACE | SF-3 [P1] Forwards and discount factors per expiry from put-call parity, with Black-76 inv | must
WS-SURFACE | SF-4 [P1] Spread-weighted SVI calibration constrained against arbitrage | must
WS-SURFACE | SF-5 [P2] Vectorised inversion at chain scale and de-Americanised quotes | must
WS-SURFACE | SF-6 [P2] Desk conventions and diagnostics on the Surface tab | must
WS-SURFACE | SF-7 [P2] Andreasen-Huge arbitrage-free interpolation ('bid/ask precision' mode) | should
WS-SURFACE | SF-8 [P1] Wing extrapolation consistent with Lee's bound, flagged | should
WS-SURFACE | SF-9 [P2] Trading-time clock for short-dated expiries | should
WS-SURFACE | SF-10 [P3] Quote-level static arbitrage including calendar, with minimal L1 repair (HiGHS) | could
WS-SURFACE | SF-11 [P2] Data-source panel and carry diagnostics; publish the fitted surface to other ta | must
WS-RISK Po | RK-1 [P1, merge early] Second-order and dollar position Greeks, Greek ladder, scenario cub | must
WS-RISK Po | RK-2 [P1] Vega by expiry x strike bucket and 1M-equivalent vega | must
WS-RISK Po | RK-3 [P1] P&L explain: second-order Taylor waterfall and Shapley attribution | must
WS-RISK Po | RK-4 [P1] VaR/ES backtesting library and a synthetic market-history generator | must
WS-RISK Po | RK-5 [P1] Historical and filtered historical simulation, calibrated dynamics, realistic de | must
WS-RISK Po | RK-6 [P1] Exact delta-gamma-theta VaR as the analytic comparator | must
WS-RISK Po | RK-7 [P2] Vol-surface shocks and smile regimes, with legs read from the fitted surface | should
WS-RISK Po | RK-8 [P2] Margin-style risk arrays (SPAN-like and FINRA 4210(g) portfolio margin) | should
WS-RISK Po | RK-9 [P1] Strike selection by delta and consistent input validation | should
WS-RISK Po | RK-10 [P3] Euler ES contributions by leg and by expiry | could
WS-BOOK St | BK-1 [P2] Risk console: headline strip and report selector | must
WS-BOOK St | BK-2 [P2] Strategy Builder desk layout | should
WS-BOOK St | BK-3 [P2] Leg IVs from the fitted surface | must
WS-BOOK St | BK-4 [P3] Position CSV import, expiry ladder and pin risk | could
WS-DATA Ma | DT-1 [P1, merge early] Clock: valuation timestamps, settlement times, trading-time and eve | must
WS-DATA Ma | DT-2 [P1] Sidebar: expiry dates, intraday time, 0DTE and presets | must
WS-DATA Ma | DT-3 [P1] Term structures: Treasury curve (H.15 primary) and cash dividend schedules | should
WS-DATA Ma | DT-4 [P1] Realistic synthetic chain: default demo data and test oracle | must
WS-DATA Ma | DT-5 [P1] Upload adapters for chains users download themselves | must
WS-DATA Ma | DT-6 [P1] Data and licensing policy, enforced | must
WS-DATA Ma | DT-7 [P3] Optional local-only live adapters (Cboe JSON, yfinance) | could
WS-DATA Ma | DT-8 [P3] Weekly network canary for the public-domain endpoints | could
WS-DESK De | DK-1 [P1, merge early] Desk theme in config.toml: per-mode palettes with exactly ten chart | must
WS-DESK De | DK-2 [P1] Self-host the complete IBM Plex Sans and Mono | must
WS-DESK De | DK-3 [P1, merge early] Semantic colour tokens and the Plotly house style | must
WS-DESK De | DK-4 [P1] Desk number-format standard | should
WS-DESK De | DK-5 [P1] Desk shell: compact header strip, KPI styling, one stylesheet | must
WS-DESK De | DK-6 [P2] Methodology as a reference: contents, sections, drawer, validation summary | should
WS-DESK De | DK-7 [P2] Browser regression and performance-budget suite | should
WS-DESK De | DK-8 [P3] Shortcuts and export menus at shell level | could
WS-VALIDAT | VL-1 [P1, merge early] QuantLib and mpmath independent-oracle suite | must
WS-VALIDAT | VL-2 [P1] Hypothesis property suite | must
WS-VALIDAT | VL-3 [P1] CI hygiene and supply chain | should
WS-VALIDAT | VL-4 [P1, blocking from INT-1] Typing gate | should
WS-VALIDAT | VL-5 [P1] Coverage gate and badge without a third-party service | should
WS-VALIDAT | VL-6 [P1] Published benchmarks tracked over time | should
WS-VALIDAT | VL-7 [P2] Generated validation report in SR 26-2 structure | must
WS-VALIDAT | VL-8 [P2] Before/after scorecard and free-resources ledger (what one person achieved for f | must
WS-VALIDAT | VL-9 [P2] Documentation site | should
WS-VALIDAT | VL-10 [P2] Release engineering | should
WS-VALIDAT | VL-11 [P3] Mutation testing and nightly ongoing monitoring | could
WS-INT Int | INT-1 [P3] Merge train, cross-cutting cleanup and gates | must
WS-INT Int | INT-2 [P3] Reviewer-facing README and the v3.0.0 release | must

### P0-1 [P0] Lazy tabs and a UI-free position builder
DETAIL: 1. In ui/app.py, switch to st.tabs(key='tab', on_change='rerun', bind='query-params') (Streamlit 1.65) and render only the open tab (tab.open). Today all six tabs, 15 Plotly charts and 100k-path VaR run on every rerun. 2. Extract tab_strategy.build_position(inputs) -> Position | None from tab_strategy.render. It reads st.session_state['strategy_legs'] and the URL-bound selector keys and creates no widgets. 3. tab_risk.render(position) gets its position from build_position() when Strategy is closed. On a cold deep link it builds the bound template. 4. Add persist_state='session' to Strategy widgets that are not URL-bound (contracts, multiplier, commission, width), so their values survive while the tab is unrendered. 5. Keep the price cards outside the tabs. Start from the about-25-line scratch patch in scratchpad/phase1/u-ui/lazy_copy.
FILES: ['options_analyzer/ui/app.py', 'options_analyzer/ui/tab_strategy.py', 'options_analyzer/ui/tab_risk.py', 'tests/test_app.py', 'tests/test_app_strategy.py']
ACC: (1) Playwright, using the method of u-ui/compare_lazy.py (3 loads at 1440 px): DOM quiet within 3.5 s of navigation (today 9.52-9.97 s); first-load websocket traffic at most 60 KB (today 343,653 B); a Spot 100->105 rerun on Pricing within 0.9 s and at most 60 KB (today 1.73 s and 274 KB). (2) AppTest: with Pricing open, risk.value_at_risk is called 0 times (patched counter). With ?tab=Risk..., the VaR histogram is present and the Strategy data_editor is absent. (3) A cold deep link ?tab=Risk...&strategy=Iron+Condor shows that template's risk. (4) Leg edits survive Strategy -> Risk -> Strategy, and all 129 existing AppTests pass.
EFFORT M

### P0-2 [P0] UI seams: shared components, one chart entry point, model-config split, methodology split by area
DETAIL: (a) Move card_row and _cards_style from tab_pricing into a new ui/components.py; tab_iv already imports card_row. Every metric row in every tab should then use it, which fixes the phone stacking. (b) Add charts.show(fig, *, key, alt) and route all 19 st.plotly_chart calls through it. The Plotly config and accessibility text then live in one place owned by WS-DESK. (c) Move ModelConfig and the pricing-model widgets out of ui/inputs.py into a new ui/model_config.py; inputs.sidebar() calls model_config.model_sidebar(). WS-PRICING then owns model choice and WS-DATA owns market inputs. (d) Split the Methodology Markdown and LaTeX into options_analyzer/ui/methodology/{pricing,surface,risk,data,validation}.md as package data. tab_methodology renders them in order, so each workstream documents its own models in its own file. No behaviour change.
FILES: ['options_analyzer/ui/components.py', 'options_analyzer/ui/model_config.py', 'options_analyzer/ui/inputs.py', 'options_analyzer/ui/charts.py', 'options_analyzer/ui/tab_pricing.py', 'options_analyzer/ui/tab_iv.py', 'options_analyzer/ui/tab_strategy.py', 'options_analyzer/ui/tab_risk.py', 'options_analyzer/ui/tab_surface.py', 'options_analyzer/ui/tab_methodology.py', 'options_analyzer/ui/methodology/pricing.md', 'options_analyzer/ui/methodology/surface.md', 'options_analyzer/ui/methodology/risk.md', 'options_analyzer/ui/methodology/data.md', 'options_analyzer/ui/methodology/validation.md']
ACC: All 3,357 tests pass with only import-path edits. grep finds st.plotly_chart only in ui/charts.py. The AppTest element tree (labels, metric values, dataframe contents) is identical before and after, both for the default state and for ?model=Binomial+tree&style=American. The rendered Methodology markdown strings are unchanged (empty diff).
EFFORT M

### P0-4 [P0] Tag the v2.0.0 baseline and record the starting numbers
DETAIL: 1. Tag ae5b1ca as v2.0.0 (the owner pushes the tag). Every 'before' number in the scorecard (VL-8) is then measured against an immutable ref. 2. Write docs/scorecard/baseline.md with the phase-1 measurements and the path of the scratch script that produced each one:
- IV 589-765 µs per quote and 7.65 s for 10,000 quotes.
- LR-501 RMSE 5.8e-4 on the 4,495-put ALO grid.
- American IV 24-152 ms per quote with a +0.16 to +1.13 vol-bp bias.
- Page settle 9.5-10 s, 344 KB.
- SPX surface RMSE up to 1.59 vol pts with sidebar carry.
- Default VaR: 101 exceptions against 20.1 expected.
- mypy 184 errors, zizmor 12 findings.
FILES: ['docs/scorecard/baseline.md']
ACC: git rev-parse v2.0.0^{commit} equals ae5b1ca. Every baseline number names the script that regenerates it, and VL-8 reproduces each one within its stated noise on the v2.0.0 worktree.
EFFORT S

### DK-5 [P1] Desk shell: compact header strip, KPI styling, one stylesheet
DETAIL: In app.py, replace the 44 px H1 and the feature caption with a header row: st.container(horizontal=True, key='desk_header'). It holds:
- an 'OPTIONS ANALYTICS' small-caps brand;
- a Plex Mono contract ticker: S · K · expiry DD-Mon-YY (DTE) · σ · r · q · F;
- a model badge;
- a 'Share' popover with the exact-state URL (the inputs are already query-bound);
- a Methodology drawer button (DK-6).

Other changes:
- page_icon becomes ':material/candlestick_chart:' and st.logo shows an SVG monogram.
- ui/static/desk.css, injected once with st.html(Path), uses only allowlisted selectors (.st-key-*, stMetricLabel, stMetricValue, stMainBlockContainer). It styles uppercase 0.72rem metric labels with letter-spacing, values with tabular-nums and slashed-zero, and trims the top padding.
- The card container-query rules from P0-2 move into it.
- Tab labels are shortened for phones.
FILES: ['options_analyzer/ui/app.py', 'options_analyzer/ui/components.py', 'options_analyzer/ui/static/desk.css', 'tests/test_app.py']
ACC: Playwright at 1440x900:
- The header row is under 40 px high.
- The first KPI's top is under 200 px (today about 225).
- The tab bar's top is at most 260 px (today 386).

At 390x844, tab content starts at or above 330 px (today about 620), and tab labels are not truncated.

AppTest: the ticker updates after a spot change.

Static tests: desk.css contains no 'section[data-testid' and no selector outside the allowlist; no emoji appears in UI strings.
EFFORT M

### DK-6 [P2] Methodology as a reference: contents, sections, drawer, validation summary
DETAIL: tab_methodology renders ui/methodology/*.md (each workstream owns its own file):
- a table of contents with anchors;
- collapsed sections;
- references with years and DOI or arXiv IDs.

A 'Validation' section renders tables from ui/methodology/validation_results.json (generated by VL-7), with commit SHA and date.

The same content opens in a right-side drawer, st.dialog(position='right') from Streamlit 1.65, from the header button, so users never leave their tab.
FILES: ['options_analyzer/ui/tab_methodology.py', 'options_analyzer/ui/app.py', 'tests/test_app.py']
ACC: (1) AppTest renders every methodology file without an exception. (2) Playwright finds no raw '$$' in the DOM, so all maths is rendered. (3) With sections collapsed, the tab is at most 1,500 px tall at 1440 px and at most 2,500 px at 390 px (today 4,201 and 8,337). (4) The drawer is anchored right at full viewport height. (5) Every reference carries a URL, link-checked nightly.
EFFORT M

4
               total        used        free      shared  buff/cache   available
Mem:              15           0          15           0           0          15
Swap:              0           0           0
/dev/vda        252G   30G  7.6G  80% /
/dev/vda        252G   30G  7.6G  80% /
__init__.py
__pycache__
binomial.py
black_scholes.py
implied_vol.py
monte_carlo.py
risk.py
scenarios.py
strategies.py
ui
vol_surface.py
# Runtime dependencies (Streamlit Community Cloud installs from this file); keep them in step with pyproject.toml.
# Development tools are pinned in requirements-dev.txt, the independent test oracles in requirements-oracle.txt.
streamlit>=1.65,<2
numpy>=1.26
pandas>=2.1
plotly>=5.24
scipy>=1.11
tzdata>=2022.7; sys_platform == "win32"
xlsxwriter>=3.1
P0-1 M ['options_analyzer/ui/app.py', 'options_analyzer/ui/tab_strategy.py', 'options_analyzer/ui/tab_risk.py', 'tests/test_app.py', 'tests/test_app_strategy.py']
P0-2 M ['options_analyzer/ui/components.py', 'options_analyzer/ui/model_config.py', 'options_analyzer/ui/inputs.py', 'options_analyzer/ui/charts.py', 'options_analyzer/ui/tab_pricing.py', 'options_analyzer/ui/tab_iv.py', 'options_analyzer/ui/tab_strategy.py', 'options_analyzer/ui/tab_risk.py', 'options_analyzer/ui/tab_surface.py', 'options_analyzer/ui/tab_methodology.py', 'options_analyzer/ui/methodology/pricing.md', 'options_analyzer/ui/methodology/surface.md', 'options_analyzer/ui/methodology/risk.md', 'options_analyzer/ui/methodology/data.md', 'options_analyzer/ui/methodology/validation.md']
P0-3 S ['pyproject.toml', 'requirements.txt', 'requirements-dev.txt', 'requirements-oracle.txt', '.github/workflows/ci.yml', 'tests/conftest.py', 'tests/test_app.py', 'tests/test_app_pricing.py', 'tests/test_app_strategy.py', 'tests/test_app_surface.py']
P0-4 S ['docs/scorecard/baseline.md']
PR-1 M ['options_analyzer/lets_be_rational.py', 'options_analyzer/implied_vol.py', 'THIRD_PARTY_NOTICES.md', 'tests/test_lets_be_rational.py', 'tests/test_implied_vol.py', 'benchmarks/test_bench_iv.py']
PR-2 L ['options_analyzer/american.py', 'scripts/gen_alo_reference.py', 'tests/fixtures/alo_grid_qdfp.npz', 'tests/test_american.py', 'tests/oracle/test_ql_alo.py', 'benchmarks/test_bench_american.py']
PR-3 M ['options_analyzer/ui/model_config.py', 'options_analyzer/ui/cache.py', 'options_analyzer/ui/tab_pricing.py', 'options_analyzer/ui/charts_pricing.py', 'options_analyzer/binomial.py', 'tests/test_app_pricing.py', 'tests/test_american.py']
PR-4 M ['options_analyzer/implied_vol.py', 'options_analyzer/american.py', 'options_analyzer/ui/tab_iv.py', 'options_analyzer/ui/cache.py', 'tests/test_implied_vol.py', 'tests/test_american.py']
PR-5 L ['options_analyzer/dividends.py', 'options_analyzer/ui/tab_pricing.py', 'options_analyzer/ui/tab_iv.py', 'options_analyzer/ui/charts_pricing.py', 'tests/test_dividends.py', 'tests/oracle/test_ql_dividends.py']
PR-6 M ['options_analyzer/monte_carlo.py', 'options_analyzer/black_scholes.py', 'options_analyzer/binomial.py', 'tests/test_monte_carlo.py', 'tests/test_black_scholes.py', 'tests/test_binomial.py']
PR-7 L ['options_analyzer/ui/tab_pricing.py', 'options_analyzer/ui/tab_iv.py', 'options_analyzer/ui/charts_pricing.py', 'tests/test_app_pricing.py']
PR-8 L ['options_analyzer/heston.py', 'tests/test_heston.py', 'tests/oracle/test_ql_heston.py']
SF-1 S ['options_analyzer/ui/tab_surface.py', 'tests/test_app_surface.py']
SF-2 M ['options_analyzer/vol_surface/__init__.py', 'options_analyzer/vol_surface/svi.py', 'options_analyzer/vol_surface/ssvi.py', 'options_analyzer/vol_surface/local_vol.py', 'options_analyzer/vol_surface/quotes.py', 'options_analyzer/vol_surface/surface.py', 'tests/test_vol_surface.py']
SF-3 M ['options_analyzer/vol_surface/forwards.py', 'options_analyzer/vol_surface/quotes.py', 'options_analyzer/vol_surface/surface.py', 'options_analyzer/ui/tab_surface.py', 'tests/test_forwards.py', 'tests/test_vol_surface.py']
SF-4 M ['options_analyzer/vol_surface/svi.py', 'options_analyzer/vol_surface/surface.py', 'options_analyzer/ui/tab_surface.py', 'tests/test_svi_constraints.py']
SF-5 M ['options_analyzer/vol_surface/quotes.py', 'options_analyzer/ui/tab_surface.py', 'options_analyzer/ui/charts_surface.py', 'tests/test_vol_surface.py', 'tests/test_app_surface.py']
SF-6 L ['options_analyzer/vol_surface/diagnostics.py', 'options_analyzer/ui/tab_surface.py', 'options_analyzer/ui/charts_surface.py', 'tests/test_surface_diagnostics.py', 'tests/test_app_surface.py']
SF-7 L ['options_analyzer/vol_surface/andreasen_huge.py', 'options_analyzer/vol_surface/surface.py', 'options_analyzer/ui/tab_surface.py', 'tests/test_andreasen_huge.py', 'tests/oracle/test_ql_andreasen_huge.py']
SF-8 S ['options_analyzer/vol_surface/surface.py', 'options_analyzer/vol_surface/svi.py', 'tests/test_vol_surface.py']
SF-9 S ['options_analyzer/vol_surface/surface.py', 'options_analyzer/vol_surface/local_vol.py', 'options_analyzer/ui/tab_surface.py', 'tests/test_vol_surface.py']
SF-10 M ['options_analyzer/vol_surface/arbitrage.py', 'options_analyzer/ui/tab_surface.py', 'tests/test_arbitrage.py']
SF-11 M ['options_analyzer/ui/tab_surface.py', 'options_analyzer/ui/surface_state.py', 'options_analyzer/ui/charts_surface.py', 'tests/test_app_surface.py']
RK-1 M ['options_analyzer/strategies.py', 'options_analyzer/scenarios.py', 'tests/test_strategies.py', 'tests/test_scenarios.py']
RK-2 M ['options_analyzer/scenarios.py', 'tests/test_scenarios.py']
RK-3 M ['options_analyzer/attribution.py', 'tests/test_attribution.py']
RK-4 L ['options_analyzer/backtest.py', 'tests/test_backtest.py']
RK-5 M ['options_analyzer/risk.py', 'tests/test_risk.py']
RK-6 S ['options_analyzer/risk.py', 'tests/test_risk.py']
RK-7 L ['options_analyzer/surface_risk.py', 'tests/test_surface_risk.py']
RK-8 M ['options_analyzer/margin.py', 'tests/test_margin.py']
RK-9 S ['options_analyzer/strategies.py', 'tests/test_strategies.py']
RK-10 S ['options_analyzer/risk.py', 'tests/test_risk.py']
BK-1 L ['options_analyzer/ui/tab_risk.py', 'options_analyzer/ui/charts_risk.py', 'options_analyzer/ui/risk_cache.py', 'tests/test_app_risk.py']
BK-2 L ['options_analyzer/ui/tab_strategy.py', 'options_analyzer/ui/charts_risk.py', 'tests/test_app_strategy.py']
BK-3 M ['options_analyzer/ui/tab_strategy.py', 'tests/test_app_strategy.py']
BK-4 M ['options_analyzer/ui/tab_strategy.py', 'tests/test_app_strategy.py']
DT-1 M ['options_analyzer/clock.py', 'tests/test_clock.py', 'tests/fixtures/nyse_holidays_2026_2028.csv']
DT-2 M ['options_analyzer/ui/inputs.py', 'tests/test_app_inputs.py']
DT-3 M ['options_analyzer/termstructure.py', 'options_analyzer/market_data/__init__.py', 'options_analyzer/market_data/rates.py', 'options_analyzer/ui/data_cache.py', 'options_analyzer/ui/inputs.py', 'tests/test_termstructure.py', 'tests/test_market_data.py', 'tests/fixtures/h15_sample.csv', 'tests/fixtures/treasury_sample.csv']
DT-4 M ['options_analyzer/market_data/synthetic.py', 'tests/test_market_data.py']
DT-5 M ['options_analyzer/market_data/adapters.py', 'tests/test_market_data.py', 'tests/fixtures/chain_long.csv', 'tests/fixtures/chain_yfinance.csv', 'tests/fixtures/chain_cboe_wide.csv']
DT-6 S ['options_analyzer/ui/methodology/data.md', 'tests/test_market_data.py']
DT-7 M ['options_analyzer/market_data/live.py', 'tests/test_market_data.py']
DT-8 S ['.github/workflows/canary.yml', 'tests/test_market_data.py']
DK-1 S ['.streamlit/config.toml', 'options_analyzer/ui/.streamlit/config.toml', 'tests/test_theme.py']
DK-2 S ['options_analyzer/ui/static/fonts/IBMPlexSans-Regular.woff2', 'options_analyzer/ui/static/fonts/IBMPlexSans-Medium.woff2', 'options_analyzer/ui/static/fonts/IBMPlexSans-SemiBold.woff2', 'options_analyzer/ui/static/fonts/IBMPlexSans-Bold.woff2', 'options_analyzer/ui/static/fonts/IBMPlexMono-Regular.woff2', 'options_analyzer/ui/static/fonts/IBMPlexMono-Medium.woff2', 'options_analyzer/ui/static/fonts/LICENSE.txt', '.streamlit/config.toml', 'options_analyzer/ui/.streamlit/config.toml', 'options_analyzer/ui/streamlit_app.py', 'options_analyzer/ui/launch.py', 'main.py', 'tests/test_theme.py']
DK-3 M ['options_analyzer/ui/theme.py', 'options_analyzer/ui/charts.py', 'tests/test_charts.py', 'tests/test_theme.py']
DK-4 M ['options_analyzer/ui/format.py', 'tests/test_format.py']
DK-5 M ['options_analyzer/ui/app.py', 'options_analyzer/ui/components.py', 'options_analyzer/ui/static/desk.css', 'tests/test_app.py']
DK-6 M ['options_analyzer/ui/tab_methodology.py', 'options_analyzer/ui/app.py', 'tests/test_app.py']
DK-7 M ['tests/e2e/conftest.py', 'tests/e2e/test_style.py', 'tests/e2e/test_performance.py', '.github/workflows/e2e.yml', 'scripts/screenshots.py']
DK-8 S ['options_analyzer/ui/components.py', 'options_analyzer/ui/app.py', 'tests/test_app.py']
VL-1 M ['tests/oracle/conftest.py', 'tests/oracle/test_ql_european.py', 'tests/oracle/test_ql_binomial.py', 'tests/oracle/test_mpmath_iv.py', '.github/workflows/ci.yml']
VL-2 S ['tests/test_properties.py']
VL-3 S ['.github/workflows/ci.yml', '.github/dependabot.yml', '.pre-commit-config.yaml', 'noxfile.py', 'pyproject.toml', 'tests/test_packaging.py']
VL-4 M ['pyproject.toml', 'options_analyzer/py.typed', '.github/workflows/ci.yml']
VL-5 S ['pyproject.toml', '.github/workflows/ci.yml', 'tests/test_implied_vol_branches.py']
VL-6 S ['benchmarks/conftest.py', 'benchmarks/test_bench.py', 'scripts/bench_table.py', '.github/workflows/bench.yml', 'docs/benchmarks.md']
VL-7 M ['scripts/validation_report.py', 'docs/validation/index.md', 'docs/validation/results.json', 'options_analyzer/ui/methodology/validation_results.json', 'options_analyzer/ui/methodology/validation.md']
VL-8 M ['scripts/scorecard.py', 'docs/scorecard/index.md', 'docs/scorecard/ledger.md', 'tests/test_ledger.py']
VL-9 M ['mkdocs.yml', 'docs/index.md', 'docs/quickstart.md', 'docs/api/index.md', '.github/workflows/docs.yml']
VL-10 S ['CHANGELOG.md', 'CITATION.cff', '.github/workflows/release.yml']
VL-11 S ['.github/workflows/mutation.yml', '.github/workflows/nightly.yml']
INT-1 M ['options_analyzer/ui/cache.py', 'pyproject.toml', '.github/workflows/ci.yml', 'tests/e2e/test_style.py', 'tests/test_guards.py']
INT-2 S ['README.md', 'AUDIT.md', 'docs/history/audit-v1.md', 'docs/images/', 'CHANGELOG.md', 'options_analyzer/__init__.py']

### PR-1 [P1, merge first] Vectorised Let's Be Rational implied vol (port of Jäckel revision 1520)
DETAIL: New options_analyzer/lets_be_rational.py, written as NumPy array code with a math-module scalar path:
- normalised Black b(x,s), using the asymptotic branches with scipy.special.erfcx and ndtri;
- the four-branch rational initial guess;
- two third-order Householder steps.

Licensing: keep Jäckel's copyright and permission notice verbatim in the file header and in THIRD_PARTY_NOTICES.md. py_lets_be_rational (MIT) is the reading reference.

implied_vol.implied_volatility:
- keeps its scalar signature, model-free bound checks and ImpliedVolError messages;
- keeps Brent as a fallback where LBR flags non-convergence.

Add implied_volatility_batch(prices, S, K, T, r, q, option_type) -> ndarray. Rejected rows get NaN plus a per-row reason code.

Do not use py_vollib or py_vollib_vectorized at runtime: py_vollib is scalar, and py_vollib_vectorized fails under numba 0.68 with a TypingError.
ACC: (1) 100,000 random cases priced with 60-digit mpmath, covering |ln F/K| up to 40·σ√T, σ√T in [1e-4, 5], prices down to 1e-300, calls and puts: relative vol error at most 1e-14, or at most 4·cond·eps where vega is tiny. (2) Agrees with py_lets_be_rational (oracle group) to 1e-15. (3) The existing 2,659 test_implied_vol cases pass unchanged. (4) Speed on CI: a 10,000-quote batch within 50 ms (today 7.65 s); scalar median within 40 µs (today 589 µs). (5) Every out-of-bounds row returns the same ImpliedVolError reason as the scalar path.

### RK-1 [P1, merge early] Second-order and dollar position Greeks, Greek ladder, scenario cube
DETAIL: strategies.position_risk(strategy, market, spot=None, t=0) returns a PositionRisk dataclass:
- Δ (shares), $Δ = Δ·S;
- 1% Γ = Γ·S·0.01 (shares) and $Γ per 1% = ½Γ(0.01S)²;
- vega in $ per vol pt, θ in $ per day, ρ in $ per 1%;
- vanna, volga, charm and veta, which bs.greeks already computes (0 for stock legs).
position_greeks keeps returning its dict for compatibility.

scenarios.greek_ladder(strategy, market, shocks=(-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20)% or ±kσ, roll_days=0) returns a DataFrame of full-revaluation P&L plus the risk fields.

scenarios.scenario_cube(strategy, market, spot_changes, vol_shocks, days, vol_mode='relative'|'absolute'|'1m') returns an array of shape (days, vols, spots). pnl_grid gains vol_mode.

The library returns stable snake_case columns; display labels belong to the UI.
ACC: (1) The 0% ladder row equals position_risk to 1e-12, with P&L exactly 0. (2) For all 26 templates, the central difference of ladder P&L over ±1% matches $Δ·0.01 to relative 1e-3, and the difference of Δ matches 1% Γ the same way. (3) Vanna, volga and charm match finite differences of position_value to 1e-4 relative. (4) The ladder of a sum of legs equals the sum of their ladders. (5) For stock only, Δ = quantity and Γ = vega = 0 in every row. (6) scenario_cube[0, zero-vol index, :] equals today's pnl_grid, and vol_mode='1m' on a single 30-day expiry equals 'relative'.

### RK-2 [P1] Vega by expiry x strike bucket and 1M-equivalent vega
DETAIL: scenarios.vega_buckets(strategy, market, expiry_edges=(7,14,30,60,91,182,365,730) days, strike_axis='delta'|'moneyness') returns a pivot with row and column totals. The delta axis uses buckets 10Δp/25Δp/ATM/25Δc/10Δc, computed with each leg's own IV.

scenarios.weighted_vega(strategy, market, decay=DEFAULT_VOL_TENOR_DECAY) = Σ q·vega·σ·min(1, (T/1M)^-decay)/100, the P&L for a 1% relative shock to 1M vol, plus a √T variant. Net vega misled in the sample book: +162.52 raw against +31.94 1M-weighted.
ACC: (1) The buckets sum to position vega within 1e-10 (162.52 on the r-risk 7-leg book). (2) weighted_vega equals the run_stress_tests P&L for StressScenario(vol_multiplier=1.01, one_month_vol_shock=True) within 1%; measured 31.94 against 31.94. (3) A single-leg position has exactly one non-zero cell. (4) A leg exactly on a bucket edge is counted once.

### RK-9 [P1] Strike selection by delta and consistent input validation
DETAIL: 1. Add strategies.strike_for_delta(target, market, T, vol, option_type), and build_template(..., strikes_by='width'|'delta', deltas=...). Desks pick shorts around 16-30Δ, not at ±2.5% of spot. 2. Market and Leg reject NaN and inf in every numeric field with a message naming the field; today Market accepts rate=NaN and Leg accepts quantity=inf. 3. Leg expiry accepts fractional days, for 0DTE.
ACC: (1) An iron condor at 45 d and 18% vol with 16Δ shorts gives a short put Δ of -0.16 ± 1e-6 and a short call Δ of +0.16 ± 1e-6 at the leg vols. (2) NaN or inf in each Market and Leg field raises ValueError naming the field. (3) A leg 2 hours from expiry (T = 120/525,600) analyses without warnings.

### DT-1 [P1, merge early] Clock: valuation timestamps, settlement times, trading-time and event variance
DETAIL: New options_analyzer/clock.py, pure functions with no Streamlit:
- tz-aware America/New_York via zoneinfo, with tzdata on Windows;
- settlement at 16:00 ET (PM), or 09:30 ET (AM) for SPX/NDX/RUT/VIX monthlies;
- year_fraction = minutes/525,600, the Cboe VIX convention;
- NYSE holidays from the rules, including Good Friday via Easter and the observance rules, for 2000-2100.

VarianceClock(weekend_weight, holiday_weight, events):
- events is a list of (date, expected absolute move m), with event variance = (π/2)·m², a documented normal approximation;
- methods variance_time(t0, t1) and effective_vol(σ, t0, t1).
ACC: (1) Valuation at 10:00 ET on an expiry day gives T = 360/525,600 exactly. (2) AM and PM settlement differ by 390 minutes. (3) Holidays 2026-2028 equal NYSE's published list (fixture). (4) With weights 1 the VarianceClock equals calendar time. (5) variance_time(t0, t2) = variance_time(t0, t1) + variance_time(t1, t2) to 1e-15. (6) Total variance = σ²·τ_trading + Σ (π/2)m² to 1e-15. (7) BSM at the clock's T equals the hand formula.

### DT-4 [P1] Realistic synthetic chain: default demo data and test oracle
DETAIL: New market_data/synthetic.py with sample_chain(spot, rate_curve, dividends, style='index'|'etf', seed, as_of). It produces:
- OCC symbols, and weekly plus monthly expiries with AM/PM roots;
- 5-point strikes near ATM and coarser wings;
- ticks of 0.05 below $3 and 0.10 above;
- spreads widening with |k| and T, and both sides quoted at every strike;
- a box rate of UST + 45 bp, and some one-sided or zero-bid wings;
- SSVI vols in trading-time variance;
- for ETF style, American prices with a discrete-dividend step (from american/dividends once merged, the binomial tree before).
Everything is labelled 'Synthetic — not market data'.
ACC: (1) Output is deterministic per seed. (2) The noise-free version raises 0 quote-level arbitrage flags, and an injected butterfly is flagged. (3) Integration test owned here, enabled once SF-3 lands: sample_chain -> parse_quotes -> fit_market_surface recovers the generator's ATM vols within 0.1 vol pt and F within 2e-4. (4) The default expiry set stays under 10,000 rows. (5) Every column passes schema validation.

### DT-5 [P1] Upload adapters for chains users download themselves
DETAIL: New market_data/adapters.py normalises three layouts into the canonical schema parse_quotes accepts today, so no more header guessing is added to the parser:
- (a) the app's long schema;
- (b) a yfinance-style export (contractSymbol, strike, bid, ask, expiry);
- (c) Cboe's manual-download wide layout (calls | strike | puts).

Rules:
- OCC symbols parse as ROOT + YYMMDD + C/P + an 8-digit strike x 1000.
- T comes from clock.py, with the settlement style.
- Expired, crossed and zero-bid rows are dropped, each with a message.
- Vendor IV is kept only as a vendor_iv diagnostic: Yahoo's IV differs from Cboe's by more than 1 vol pt on 45% of OTM SPY quotes, and 838 of 10,040 contracts carry placeholders below 0.01.

Fixtures are fabricated values in the real layouts. The owner captures one manual Cboe download once to confirm its header, then replaces the values.
ACC: (1) The three fixture layouts give identical cleaned quotes. (2) 'SPXW261102C07930000' parses to (SPXW, 2026-11-02, call, 7930). (3) Expired and crossed rows are dropped with one message each. (4) The fit result is unchanged when vendor_iv is shuffled.

### DT-6 [P1] Data and licensing policy, enforced
DETAIL: ui/methodology/data.md lists every source with its terms and the public-demo rule (synthetic data plus uploads, no server-side scraping):
- Cboe delayed quotes: automated extraction prohibited, IPs blocked;
- cboe.com/terms: personal non-commercial use only;
- Yahoo terms 2(d)(ix);
- FRED: API key required, no caching; S&P 500 and VIX are copyrighted;
- Fed Board H.15: public domain, with citation;
- Treasury.

Also:
- .gitignore entries for data and cache directories;
- a test asserting that no cdn.cboe.com or finance.yahoo.com URL is reachable from the default code path.
ACC: The grep test passes. With HTTPS_PROXY pointed at a dead port, AppTest renders every tab on synthetic data and the bundled curve, with no wait beyond the timeouts. No real third-party chain or index history is in the repository (a check on the tests/fixtures provenance header).

### PR-2 [P1] Andersen-Lake-Offengenden spectral-collocation American engine
DETAIL: New options_analyzer/american.py, following the design of QuantLib's BSD-licensed QdFp engine (credit it in the docstring) and the scratch prototype r-pricing/scripts/alo_numpy.py.

Put core:
- X = K·min(1, r/q), and B = X·exp(-√H) with H interpolated on Chebyshev-Lobatto nodes in z = 2√(τ/T) - 1.
- The cdf-form fixed point when |r-q| >= 1e-3, otherwise the density form.
- A Jacobi-Newton step, then naive steps.
- Every integral on the cosine map u = τ(1 - cos θ)/2, which cut the error 10-100x in the prototype.

Calls by McDonald-Schröder symmetry. Schemes fast/accurate/high set (l, m, n, p).

Regimes:
- r <= 0 and q >= r: European value (Healy 2021, Prop. 2).
- Double boundary (q < r < 0 for a put), which QuantLib refuses: fall back to binomial.extrapolated_price and flag it.

API:
- price(S, K, T, r, sigma, q=0, *, option_type, scheme);
- exercise_boundary(...) -> (tau, B);
- a cache of the normalised boundary B/K per (T, r, q, σ), so a strike strip reuses one boundary.

Reference data: scripts/gen_alo_reference.py builds the Healy (2021) Table 5 grid of 4,495 puts with QdFp high precision. Commit the result as tests/fixtures/alo_grid_qdfp.npz, so the main suite checks accuracy without QuantLib installed.
ACC: On the 4,495-put grid against QdFp high precision:
- 'accurate': RMSE <= 5e-6, max error <= 5e-5, at most 5 ms per option on CI.
- 'high': RMSE <= 3e-7.
- 'fast': RMSE <= 1e-4 at <= 1 ms. If that cannot be reached, ship only 'accurate' and 'high'.

Properties:
- Put-call symmetry holds to 1e-14.
- A no-dividend call equals BSM to 1e-12.
- Puts with r <= 0 <= q-r equal BSM.
- The price is monotone in σ, T and K.
- It lies between max(European, intrinsic) and binomial.american_upper_bound.
- The boundary is non-decreasing in t and tends to X as τ -> 0.

The double-boundary case (r=-1%, q=-2%) returns the tree value within 1e-4 of QuantLib's FD 4000x4000 value of 7.62526. A 41-strike strip at one (T, r, q, σ) takes at most 15 ms.

### PR-6 [P1] Monte Carlo honesty, randomised QMC and core edge cases
DETAIL: Monte Carlo:
- When fewer than about 300 paths carry payoff, MonteCarloResult.reliable is False and the CI is NaN. At under 30 in-the-money paths, the 95% interval measured only 77-92% coverage.
- Add optional scrambled Sobol randomised QMC (scipy.stats.qmc), with its error taken from R independent scramblings.

Greeks: in bs.greeks, gate ultima so that σ <= 1e-77 returns the exact limit with no overflow warning.

Validation: NaN or inf in any of S, K, T, r, σ, q raises ValueError naming the field, in bs, binomial and monte_carlo. Binomial no longer says 'increase the number of steps' for NaN inputs.

New tests: zero-vol branch Greeks with r and q non-zero, and vectors mixing live and degenerate rows. These target the surviving mutants.
ACC: (1) mc_coverage over 1,000 seeds and m in {1,3,10,30,100,300}: every interval flagged reliable has coverage of at least 93%. (2) RQMC with 2^17 points and 16 scramblings: the ATM call's reported SE is at most 1e-4, and the true price lies within 3 SE in at least 99% of 200 seeds. (3) Hypothesis under -W error::RuntimeWarning: ultima is finite for σ in [1e-300, 1e-3]. (4) mutmut on black_scholes.py with its own tests kills at least 95% of non-cosmetic mutants (today 86.5%), including the d1 r+q sign mutant and the 18 survivors in _prepare.

### SF-2 [P1] Split vol_surface.py into a package and break up its two largest functions
DETAIL: Create an options_analyzer/vol_surface/ package:
- svi.py;
- ssvi.py;
- local_vol.py;
- quotes.py (header and unit guessing, inversion);
- surface.py (VolSurface, fit_market_surface, summary);
- __init__.py, re-exporting every name importable today, both public and the private ones the tests use.

Break up the two largest functions, keeping behaviour identical:
- parse_quotes (cyclomatic complexity 44): map_columns -> normalise_units -> maturities -> invert -> filter.
- fit_market_surface (complexity 39): prepare -> fit_slices -> evaluate_grid -> diagnostics.
ACC: The 174 vol-surface tests pass unchanged. The set of names in dir(vol_surface) before is a subset of the set after. Golden outputs (SVI parameters, grids, messages) for sample_quotes seeds 0-3 and the bundled sample CSVs are bit-identical. radon finds no function in the package above cyclomatic complexity 15. mypy (allow_redefinition) is clean on the package; today there are 8 Optional-narrowing errors.

### SF-3 [P1] Forwards and discount factors per expiry from put-call parity, with Black-76 inversion
DETAIL: New vol_surface/forwards.py with implied_forwards(quotes, spot, rate_fallback, exercise).

Estimation per expiry:
- Pair calls and puts at equal strikes with two-sided quotes near the money: |ln K/S| < 0.08, or the ~20 strikes around argmin |C-P|.
- Fit C - P = DF·F - DF·K by weighted least squares with weights 1/(spread_C + spread_P)^2, then 3-5 rounds of MAD trimming.
- Fallback when fewer than 6 pairs remain or SE(DF) > 5 bp: DF from the rate input or curve, and F as the weighted median of K + (C-P)/DF.
- American chains always take DF from the curve and use only near-ATM, short-dated pairs, because parity is only an inequality for American options.

Inversion and surface:
- Invert with per-row r_T = -ln(DF)/T and q_T = r_T - ln(F/S)/T, which is exactly Black-76.
- Choose the OTM side against F_T, not against S.
- VolSurface stores log_forward per maturity, interpolated linearly in T; the scalar carry stays for SSVI.

UI:
- A 'Forward source' control: put-call parity per expiry (the default when both sides are quoted) or sidebar r, q.
- A 'Forwards & discounting' table: F, DF, implied r, implied q, pairs used, residual, and the gap to the sidebar forward in bp.
- A warning when that gap exceeds max(5 bp, half the ATM spread).
ACC: (1) Synthetic chain (S=100, r=4.5%, true q=2%, SSVIParams(), 6 expiries 30-730 d, strikes 70-130 in steps of 2.5, bid/ask ±0.05 around tick-rounded mids), parsed with sidebar q=0: every F within 2 bp and DF within 2e-5; max fit RMSE < 0.4 vol pt; 0 calendar and 0 butterfly points; no static-arbitrage note. Today the same chain gives 1.52 vol pts and 18 + 18 points. (2) Call and put IVs at common strikes agree within their half-spread in vol. (3) A one-sided chain falls back with an explicit message. (4) Noise-free SSVI quotes with consistent carry are still recovered to 1e-9. (5) With constant r and q the result equals the old path to 1e-10.

### SF-4 [P1] Spread-weighted SVI calibration constrained against arbitrage
DETAIL: fit_svi(y, iv, T, weights=None, previous=None):

Weights:
- 1/max(half-spread in vol, floor), normalised by the median and capped at 5x the median.
- Vega weights when there is no spread; ones for files that hold only IVs.

Penalty rows:
- λ_g·min(g(k), 0) on ~200 points over the quoted range ±10%, where g is Durrleman's g from the analytic w, w', w''.
- λ_c·min(w(k) - w_prev(k), 0) only on the overlap of this slice's and the previous slice's quoted ranges. Enforcing it beyond the quotes made the prototype infeasible.

Fit short to long, starting from the unconstrained optimum so the penalties stay inactive on arbitrage-free data.

The summary gains 'Inside bid/ask (%)' and 'Median |err| / half-spread'. A 'Calibration: arbitrage-free (default) | unconstrained' toggle shows the RMSE cost per expiry.
ACC: (1) Vogt slice (Gatheral-Jacquier Example 3.1, parameters -0.0410, 0.1331, 0.3060, 0.3586, 0.4153) plus a second slice w+0.02 at T=1.5: 0 butterfly points, local vol defined at every quoted grid point, RMSE < 1 vol pt (prototype 0.63). (2) Two crossing slices: 0 calendar points, with the RMSE increase reported. (3) 200 random arbitrage-free SSVI surfaces with 0.2 vol pt of noise: 0 violations, and RMSE at most 1.1x the unconstrained fit. (4) The e3_weights study over 50 seeds: the weighted fit's median max error within ±1 sd is at most 0.7x the unweighted one. (5) With equal spreads the result is bit-identical to the unweighted fit. (6) The 174 existing tests pass.

### SF-8 [P1] Wing extrapolation consistent with Lee's bound, flagged
DETAIL: Add implied_vol_at(..., extrapolate=False) and the same flag on at_moneyness. With extrapolate=True, SVI is evaluated beyond the quoted y up to a user-set |k| cap, with a separate mask for extrapolated points. Charts show those points dashed or semi-transparent, and the contract readout says 'extrapolated'. Strategy legs (BK-3) and surface shocks (RK-7) need this to read wing vols.
ACC: The extrapolated total-variance slope stays at most 2 up to |k| = 3. For arbitrage-free fits g >= 0 in the extrapolated region. With the default False, behaviour is unchanged (existing tests pass).

### RK-3 [P1] P&L explain: second-order Taylor waterfall and Shapley attribution
DETAIL: New options_analyzer/attribution.py:
- taylor_explain(strategy, market0, spot1, vol_shift or per-leg vols, rate_shift, days) returns terms for delta, gamma, vega, theta, rho, vanna, volga, charm and veta, plus the unexplained residual.
- shapley_explain over the factors spot, vol, time and rate, using 16 position_value calls. It does not depend on factor order and leaves no residual; sequential attribution shifted by up to $138 depending on order.
- quality = |unexplained| / max(|full|, floor).
- pla_metrics(hpl, rtpl) returns Spearman and KS, for use by the backtest.
ACC: (1) The Shapley terms sum to the full-revaluation P&L within 1e-9. (2) The second-order residual falls about 8x each time the shocks halve: ratios in [7, 9] for the iron condor (measured 7.84, 8.10, 8.11). (3) Zero shocks give all terms 0. (4) A pure spot shock puts 100% of the Shapley P&L on spot. (5) For a one-day pure time shock, the θ plus charm terms are within 1e-6 of position_value(t=1/365) minus the base value. (6) Per-leg explains sum to the position explain. (7) Golden Short Straddle stress day (-4% spot, +6 vol pts): full P&L -188.36 per lot and second-order residual 27.04, both ±0.01.

### RK-4 [P1] VaR/ES backtesting library and a synthetic market-history generator
DETAIL: New options_analyzer/backtest.py, pure NumPy and SciPy, implemented from the papers. GPL packages (rugarch, ExactVaRTest, esback) are used only for reference values, never for code.

Tests:
- kupiec_pof: LR statistic, χ²(1) p-value and exact binomial tail.
- christoffersen: n00..n11, LR_ind and LR_cc.
- traffic_light: MAR32 Table 1 zones and multipliers 1.50/1.70/1.76/1.83/1.88/1.92/2.00.
- acerbi_szekely_z2, with simulated critical values.
- fz0_loss (Patton-Ziegel-Chen eq. 6) and diebold_mariano with a Newey-West variance.
- pla_test, with MAR32.42 zones.

rolling_backtest(build_position, closes, iv_index, models, window, confidences) produces static-position hypothetical P&L.

synthetic_history(seed) generates GARCH-t returns with a log-normal IV index (vol-of-vol, spot-vol correlation) for demos and CI.

Also report each test's power at the sample size. Kupiec at 250 days has an actual size of 9.5% and only 24% power when the true exception rate is 2%.
ACC: (1) Kupiec, n=250, p=1%: x=0 gives LR 5.0252, x=4 gives 0.7691, x=5 gives 1.9568, x=10 gives 12.9555 (asymptotic p 0.0003, binomial P(X>=10) 0.0003). (2) Binomial(250, 1%) CDF(4) = 0.8922 and CDF(9) = 0.9997, matching the MAR32 zones. (3) Christoffersen with exceptions on days 100-104 of 250: counts (243,1,1,4) and LR_ind 30.985. (4) Z2 under N(0,1) with T=250 and α=2.5%: 5% quantile -0.70 ± 0.02. (5) FZ0 at α=5% is minimised near (-1.64, -2.06). (6) Over 200 replications of 2,500 iid N(0,1) days with the true VaR, the Kupiec rejection rate lies within the binomial 95% band around nominal. (7) Fixed-seed synthetic history: the constant-IV normal model ends red, and vol-scaled historical simulation ends non-red.

### RK-5 [P1] Historical and filtered historical simulation, calibrated dynamics, realistic defaults
DETAIL: risk.HistoricalConfig(window=500, scaling='none'|'ewma'|'iv_index'|'garch', lambda=0.94) and risk.historical_var(strategy, market, returns, iv_changes=None, config, multiplier) -> VaRResult, reusing the Acerbi-Tasche _var_es.
- Full revaluation on joint spot and log-IV changes, with tenor damping.
- h-day horizons use overlapping sums, with a warning.

Filtered historical simulation: a GARCH(1,1) fit by Gaussian quasi-maximum likelihood (scipy.optimize, variance targeting optional), then a bootstrap of the standardised residuals.

calibrate_dynamics(closes, iv_index) returns the vol-of-vol, the Pearson and Spearman spot-vol correlation, and Student-t ν by maximum likelihood.

New VaRConfig defaults, documented: Student-t ν=5, with IV dynamics on (vol-of-vol 1.0, ρ -0.7). The current default, with IV moves off, had 101 exceptions against 20.1 expected on 10 years of S&P 500/VIX data.
ACC: (1) With 200,000 iid N(0,σ) returns, the historical-simulation VaR of a long call falls inside the order-statistic 95% interval of the normal Monte Carlo VaR. (2) iv_index scaling with a constant index gives a result identical to unscaled. (3) The GARCH fit recovers α and β within 0.02 on 20,000 simulated days. (4) calibrate_dynamics recovers η within 3% and ρ within 0.02 on 10,000 days. (5) On 5,000 GARCH days, GARCH-scaled historical simulation passes Kupiec at 99%, while unscaled historical simulation fails Christoffersen independence. (6) rolling_backtest on the fixed synthetic history: vol-scaled historical simulation has at most 1.5x the expected exceptions. (7) The 33 existing risk tests pass, updated only where they assert old defaults.

### RK-6 [P1] Exact delta-gamma-theta VaR as the analytic comparator
DETAIL: risk.delta_gamma_var(delta, gamma, theta, spot, vol, horizon_days, confidence):
- P&L = θh + Δx + ½Γx² with x ~ N(0, s²), s = Sσ√(h/252).
- For Γ ≠ 0, take the quantile of a shifted, scaled non-central χ²(1, λ = (Δ/(Γs))²) with scipy.stats.ncx2.ppf, choosing the tail by the sign of Γ.
- ES by numerical integration of the same distribution.

Store it in VaRResult. It replaces delta-normal VaR as the headline comparator: for a short straddle delta-normal gave 20.09 against 69.80 from full Monte Carlo. The gap to full revaluation is reported as a non-linearity and vega diagnostic.
ACC: (1) With Γ = 0 it equals delta-normal VaR minus θh. (2) It matches Monte Carlo of the quadratic P&L itself within the 95% interval (400,000 paths) for 10 templates. (3) At a 1-day horizon it is within 5% of full-revaluation Monte Carlo VaR for those templates (measured worst 3.5%; straddle 70.55 against 69.80, butterfly 20.10 against 19.67). (4) It is monotone in confidence.

### VL-1 [P1, merge early] QuantLib and mpmath independent-oracle suite
DETAIL: tests/oracle/conftest.py starts with pytest.importorskip('QuantLib') and provides builders:
- European: BlackCalculator with no calendars, so T is exact.
- American: Actual365Fixed, NullCalendar, T = days/365, FlatForward continuous; LR binomial and FD Douglas 800x800 engines.
- QuantLib's own relativeError definition (|diff|/spot). Using value-relative errors falsely reported 139 failures.

Tests:
- European grid, 2,000 random contracts in CI and 20,000 nightly: price |diff|/S <= 1e-13 and normalised Greeks <= 1e-11. Measured 2.2e-15 and 1.1e-12.
- LR-n against QuantLib LR-n: <= 1e-10·max(S,K) (measured 2.3e-12).
- Against QuantLib FD: <= 5e-5·max(S,K) (measured 2.8e-5).
- A verbatim replica of QuantLib's testLRBinomialEngines (72 contracts) with its tolerances.
- IV round trip against mpmath: |BS(iv) - p|/S <= 1e-12. QuantLib is not an IV oracle: it returned 0 for a 1.8e-86 price.
- A recorded case where QuantLib throws (S=39.18, K=19.71, T=1/365, r=3.1%, q=5.3%, σ=3.6%, put); ours stays finite and within american_bounds.

CI gets an 'oracle' job on Python 3.13.
ACC: pytest -m oracle is green in CI, and every measured maximum is at or below its tolerance. Patching d1 to use r+q makes the European test fail (measured price error 8.06e-2 within 6 s). Without QuantLib installed, the modules report skipped, not failed.

### VL-2 [P1] Hypothesis property suite
DETAIL: tests/test_properties.py, on the documented domain:
- S in {0} ∪ [1e-3, 1e4], ln K/S in ±2.5, T in {0} ∪ [1/365, 10], σ in {0} ∪ [1e-3, 3], r from -5% to 15%, q from 0 to 10%.
- Unrestricted floats produced false alarms that were conditioning artefacts.

Properties:
- put-call parity, model-free bounds, strike monotonicity and convexity;
- Γ and vega >= 0, and Δ and vega against finite differences;
- homogeneity in (S, K), and vector results equal to scalar ones on mixed degenerate rows;
- IV round trip in price space;
- American >= European >= intrinsic and <= the upper bound;
- lattice put-call symmetry, LR -> BSM convergence, American monotone in σ;
- Monte Carlo within 6 SE;
- vertical-spread width and breakevens;
- SSVI no-arbitrage conditions.

The two known defects (unreliable MC SE, tiny-σ ultima) go in as xfail(strict=True) referencing PR-6, and flip when it lands.
ACC: HYPOTHESIS_PROFILE=ci runs in under 60 s (measured 40 s at 3,000/300 examples) and prints a reproduction blob on failure. A nightly run at 5,000 examples is green. mutmut shows the added homogeneity and vector-scalar properties killing at least 19 of the black_scholes survivors.

### VL-3 [P1] CI hygiene and supply chain
DETAIL: 1. Pin every action to its 40-character commit SHA with a '# vN' comment, and set persist-credentials: false on every checkout. 2. Add .pre-commit-config.yaml with ruff-pre-commit, zizmor-pre-commit and pre-commit-hooks. 3. Add Dependabot groups for actions and dev tools. 4. Add noxfile.py with sessions tests/lint/typing/docs/bench. 5. Drop ruff target-version, which duplicates requires-python. 6. Add Python-version and Development Status classifiers. 7. Add macOS and Windows jobs that run the core, non-UI tests on Python 3.13. 8. Add a consistency check between requirements.txt and the pyproject runtime dependencies.
ACC: zizmor reports 0 findings (today 12). repo-review reports 0 failures among applicable checks (today 12). The macOS and Windows core jobs are green. A requirements/pyproject mismatch fails test_packaging.

### VL-4 [P1, blocking from INT-1] Typing gate
DETAIL: 1. Add [tool.mypy] with allow_redefinition, check_untyped_defs, strict_equality, warn_unused_ignores and warn_redundant_casts, plus pandas-stubs. allow_redefinition removes the 140 black_scholes errors caused by reassigning ArrayLike parameters. 2. Use strict overrides for black_scholes, binomial, implied_vol, lets_be_rational, american, monte_carlo, risk, scenarios, backtest and attribution. 3. Ship options_analyzer/py.typed. 4. The mypy CI step is non-blocking in P1-P2 and becomes blocking in INT-1. 5. Every workstream's definition of done includes being mypy-clean on the files it owns.
ACC: mypy exits 0 on the merged main. A deliberately wrong annotation in a strict module fails the lint job. repo-review MY100 passes. The wheel contains py.typed.

### VL-5 [P1] Coverage gate and badge without a third-party service
DETAIL: 1. Add [tool.coverage] settings: branch, source, relative_files and exclude_also. 2. In the Python 3.13 leg, run pytest -n auto --cov and pass the result to py-cov-action, which posts a PR comment and keeps the badge on a data branch. 3. Set fail_under to 94 during P1-P2 and raise it to 95 in INT-1. 4. Add tests for the uncovered implied_vol branches (lines 77, 79, 109, 118, 169-173, 199) and the uncovered binomial branches.
ACC: The CI coverage figure matches the local run (95% branch). A PR comment is posted. The badge renders from the data branch. implied_vol branch coverage is at least 95% (today 87%).

### VL-6 [P1] Published benchmarks tracked over time
DETAIL: 1. Create benchmarks/ with conftest.py and test_bench.py: the 13 measured cases plus engine benchmarks added by their owners, and a QuantLib comparison module using importorskip. Benchmarks stay out of testpaths. 2. scripts/bench_table.py renders docs/benchmarks.md from bench.json, including machine_info and commit_info. 3. bench.yml runs on pushes to main and passes results to github-action-benchmark (gh-pages benchmark-data, alert threshold 150%, comment on alert, no failure because shared runners are noisy). 4. README performance sentences are rewritten in INT-2 to link this page; the current 'about 25 ms' claim measured 15-21 ms.
ACC: bench.json is uploaded on every push to main. The dashboard is live on GitHub Pages. A planted 2x slowdown in binomial._induct triggers an alert comment.
