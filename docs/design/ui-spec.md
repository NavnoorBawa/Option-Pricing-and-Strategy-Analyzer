# UI specification: the Options Analytics desk (v0.2)

Status: revised draft for the owner and the workstream leads, written 2026-10-26 by the design lead after two reviews: the market-maker review (`design/review-mm/`) and the principal product designer review (`design/review-principal/`). It replaces v0.1 (`design/ui-spec-draft.md`). Every review point I agreed with is applied in the body. Each point I rejected, in whole or in part, has one line in Appendix A.

Applies to: the integration branch `claude/laughing-mendel-5j6iya` at `0e762c3`, the DK-5 branch (`worktree-wf_a9b96d93-f89-1` at `db50660`, desk header) and the RK-1 branch (`worktree-wf_a9b96d93-f89-12` at `96f0960`). `position_risk`, `greek_ladder`, `scenario_cube`, `vega_buckets` and `weighted_vega` are also present at `0e762c3` [V-me].

Nothing under the repository was modified. Supporting files: `design/spec/` (v0.1 checks) and `design/spec/v02/` (v0.2 numbers and checks, indexed in `spec/v02/notes.md`).

## How to read this

| Tag | Meaning |
|---|---|
| **[S]** | Sourced fact; the URL is next to it. "excerpt" means the page refused automated reads and the claim rests on a search excerpt. |
| **[P]** | The plan of record, `docs/roadmap/plan-items.md`. |
| **[V]** | Verified on the installed software. "V-me": checked by me (v0.1: `spec/sigcheck.py` on Streamlit 1.65.0 and 1.66.1; v0.2: `spec/v02/`). "V-probe": the principles-streamlit experiments E1 to E9 in Chromium on both versions. "V-audit": measured by the visual audit at `0e762c3`. "V-mm" and "V-pr": verified by the market-maker and principal reviews (evidence in `review-mm/notes.md` and `review-principal/notes.md`). |
| **[J]** | My judgement. It is not a sourced fact. |

Words used throughout:

- **Pricing contract**: the single option that the sidebar's Contract group describes (focus expiry, strike, vol). The Pricing and IV tabs price it. It no longer appears in the global header.
- **Focus expiry**: the one expiry that Pricing, IV and Surface show, held in the sidebar key `expiry`.
- **Position**: the multi-leg structure built on the Strategy tab; the Risk tab analyses it.
- **Source**: where market data comes from: the synthetic chain (DT-4), an uploaded chain (DT-5) or flat inputs.
- **Theo**: the model value at the published surface: the fit, or the trader's marks once UI-23 lands.
- **Mid vs fit**, **buy edge**, **sell edge**: defined once in section 2.5.
- **vp**: one vol point, 0.01 of implied volatility. **ATMF**: at the money forward, K = F. **SOQ**: the special opening quotation that settles AM SPX options.

**Example numbers.** Every wireframe number is a real output of the code at `0e762c3` on the default chain `sample_chain(5800, 0.043, 0.013, as_of=2026-10-23 16:00 ET, seed=0)` (box rate 4.75%, q 1.30%), produced by `spec/v02/spec_numbers.py`, `fit_numbers.py` and `vix_style.py` [V-me]. The exceptions are marked "illustrative": the ARB counts, the timing captions, and the VaR and Explain figures. FIT uses today's `fit_svi` before SF-4, and the 30D RR and BF use the generator's noise-free IVs, so the built app will differ in the second decimal.

### What changed from v0.1

1. A **context model** (2.0) defines once: spot, the forward per expiry, the what-if scope, the engine on each tab, the focus expiry, load precedence and staleness.
2. The **header** is market-level: valuation time, underlying, S, source, state chips and the engine of the open tab. Contract items move to Pricing and IV; the SCENARIO chip moves into the what-if strip.
3. **Edge** is no longer fit noise. Against the fit the column is "Mid vs fit"; per-side buy and sell edge appear only against marks or an override. One sign convention, in vol space. Nothing is forced to 0.00.
4. **Probabilities and bands** come from the smile (Breeden-Litzenberger), not N(d₂).
5. **Engines**: the global control is BSM or ALO. Tree and Monte Carlo live in Pricing's Engine diagnostics. Strategy and Risk say which engine they used.
6. **Index conventions**: Cboe hours, delta in ES equivalents, "Γ P&L 1%", a futures hedge, and FINRA's +6%/−8% for SPX.
7. **Signs**: one minus sign, losses negative, no accounting parentheses. The st.metric colour bug is fixed.
8. **Colour**: amber marks user state only; the widget primary becomes the link blue.
9. **Risk**: six reports. Margin becomes one line in Scenarios, Backtest moves to Methodology, and VaR leaves the pinned tiles.
10. **Strategy**: a package market replaces RETURN/RISK, the Greek tile row goes, wings are set in points, and the defaults are real and pass their own tests.
11. **Surface**: no Publish toggle and no teaching SSVI. The IV tab's chain mode becomes Surface's Quotes view, and the expiry table drops to 12 columns.
12. **Tests**: every test that relied on an AppTest selection or a heatmap click is re-based on callback unit tests plus Playwright.
13. **Shortcuts** move to Alt+Shift.
14. **New should items**: trader marks, undo, model-free 30D vol, realized vol and breakeven move, event vol, SPXW daily listings.
15. **Cut from v3**: command palette, punctuation keys, palette choice, live clock, the Margin report, 3-D local vol, the right-edge table fade.

### Findings about the plan that this spec has to work around

1. **[V-me] `plan-items.md` holds the full text (DETAIL, ACC) of 29 of its 65 items.** The other 36 have only a title and a file list, among them every UI item this spec is about: PR-3, PR-4, PR-7, SF-5, SF-6, SF-9, SF-11, RK-7, RK-8, BK-1, BK-2, BK-3, BK-4, DT-2, DT-3, DK-1, DK-2, DK-3, DK-4, DK-7, DK-8. Lines 149 to 174 hold stray terminal output. So for those items this spec supplies the UI detail and proposes acceptance tests, and each owner should adopt or amend them.
2. **Conflicts with the plan, resolved here. Each needs owner sign-off.**
   - P0-1 step 5, "Keep the price cards outside the tabs": the cards move into the Pricing tab (UI-1). The audit measured them pushing tab content to about 530 px on a phone and describing the wrong thing on Strategy, Risk and Surface [V-audit].
   - DK-5's ticker loses the contract items (K, expiry, σ) and r, q, for the same reason (UI-2). DK-5's "first KPI's top under 200 px" becomes "the first headline tile of the open tab".
   - The defaults change from S = K = 100, 365 days, 20%, 5% to the synthetic index preset (DT-2, DT-4). The old values survive as `?preset=textbook`, so the existing AppTests keep a fixture (2.3).
   - DT-2: under a chain source spot is a read-only observable, and the valuation time is one `st.datetime_input` (2.0, 2.3).
   - PR-3 and PR-6: the global model control offers only European (BSM) and American (ALO). The tree and Monte Carlo engines move to Pricing's Engine diagnostics.
   - RK-4: the backtest report leaves the Risk tab for Methodology › Validation (VL-7a).
   - RK-8: the SPAN-like array leaves the UI, the FINRA range follows the product class, and the Margin report becomes one line in Scenarios.
   - SF-7 and SF-9: no toggle appears until the item merges; SF-11's Publish toggle and the teaching SSVI source are dropped.
   - Promotions:
     - DK-4 (number formats), should → must: inconsistent formatting was the audit's most visible amateur signal [V-audit G2].
     - DK-7 (browser suite), should → must: every Playwright test here depends on it.
     - PR-7 (Pricing and IV desk layout), should → must [P: "should"].
     - BK-2 (Strategy desk layout), should → must [P: "should"]. Both are the primary screens of this tool [J].
     - DK-8, could → should, for the shortcut layer and export menus only: under a day of work [V-probe E1].
   - The plan's cut line ("ship every must plus INT-1 and INT-2") therefore grows. Section 4.5 restates it with the new musts and an order for trimming them.
3. **Tab labels stay** `Pricing · Strategy · Risk · Surface · IV · Methodology`, because they are the URL contract (`?tab=Surface`) [V: `app.TABS`].

---

## 1. Design principles

Ten principles, each held to a named test.

1. **One context, one library, everywhere.** Every number on screen derives from the context model (2.0): one valuation time, one source, one forward per expiry, one published surface. Each tab's badge names the engine that tab actually used. A view that uses another engine says so in its column header ("BSM ref").
   - Why: Bloomberg's MARS uses the same library as its single-security pricers, so a single-trade price and book risk always agree [S: [MARS brochure](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf)]. The audit's worst trust failure was two put deltas on one screen [V-audit P1].
   - Tests:
     - AppTest with the ETF preset and ALO on Pricing: no table or tile shows a European value unless its label contains "BSM".
     - AppTest with the ETF preset: the Strategy and Risk badges read `BSM · early exercise ignored` until RK-12 merges.
2. **Answer first.** Each tab opens with its headline tiles and its primary chart or table above the fold at 1440 × 900. Inputs that rarely change go to a popover, never above the answer.
   - Why: Few's pitfall #1, "exceeding the boundaries of a single screen" [S: [Few 2006](https://www.perceptualedge.com/articles/Whitepapers/Common_Pitfalls.pdf)]. The Strategy payoff sits at about 1,100 px today [V-audit S1].
   - Test: Playwright per tab (section 3).
3. **Desk units, one precision per quantity.**
   - Vega is $ per vol point; theta $ per calendar day; gamma per 1% move, with the position figure named "Γ P&L 1%".
   - Index delta is in $ and ES contracts.
   - Vol in %, vol changes in vp, rate changes in bp.
   - Only `format.py` and the column registry (UI-3) format numbers.
   - Why: [S: [IBKR Risk Navigator metrics, excerpt](https://www.interactivebrokers.com.hk/php/webhelp/navigator/metrics.htm); [ORATS definitions](https://orats.com/docs/definitions)].
   - Test: the static lint in UI-3 fails on any `:.Nf` or `%.Nf` format in a `tab_*.py` file.
4. **The market next to the model, with honest names.** Wherever a theo is shown and a quote exists, bid and ask sit beside it.
   - While theo is the fit to those same mids, the comparison is named "Mid vs fit". Per-side buy and sell edge appear only when theo comes from trader marks (UI-23) or an override.
   - A value inside the band is printed with its sign and an `in band` flag, never set to 0.00.
   - Why: Vola Dynamics draws bid/ask bars under the fit and labels a mid inside the band "no edge"; ORATS sets its smooth vol against bid/ask [S: [Vola Fitter](https://www.voladynamics.com/products/vola-fitter); [ORATS SMV](https://orats.com/blog/smoothing-options-implied-volatilities-using-orats-smv-system)]. Theo against its own fit is a residual, not an edge [V-mm, V-pr; J].
   - Test: the Pricing contract table, the Strategy blotter, the Surface Quotes view and the IV tiles all use the 2.5 convention, and no column is named "Edge" unless marks or an override are active.
5. **Provenance and hypothetical state are always visible.**
   - The header shows the source (SYNTHETIC, UPLOAD, FLAT), the valuation time, and an amber chip for each kind of user state: OVERRIDE, FIT (a non-default fit setting) and MARKS.
   - The what-if state is shown where it applies (the strip on Strategy and Risk) and as a muted note on every other tab.
   - Amber marks user state and nothing else; Reset is one click.
   - Why: Bloomberg's amber input fields [S: [HBS Bloomberg options guide](https://www.library.hbs.edu/services/help-center/bloomberg-options)]; IBKR's red what-if frame and Reset Scenario; Deribit's "Remove all Overrides" [S: [IBKR what-if portfolios](https://www.ibkrguides.com/traderworkstation/what-if-portfolios.htm); [Deribit Position Builder](https://support.deribit.com/hc/en-us/articles/31238900906781-Position-Builder)].
   - Test: an AppTest on every tab flips each override and each what-if shift and checks the chip or note.
6. **Uncertainty is printed.**
   - Printed: Monte Carlo standard error and interval; fit RMSE against half-spread; "extrapolated" flags; the VaR simulation interval; the engine's validated error in its badge help; `SYN HIST` inline on any statistic computed from synthetic history.
   - An unreliable value shows "–" with its reason, never a number.
   - Why: [S: [Vola error bars](https://www.voladynamics.com/products/vola-fitter); [ORATS `confidence`](https://orats.com/docs/definitions)]; PR-6 `reliable=False` [P].
7. **Colour means something, never alone.**
   - Profit and loss colours always come with a sign. Greeks are never coloured by sign.
   - `st.metric` delta colours are set explicitly from the sign, because a U+2212 minus is read as positive [V-pr].
   - Diverging maps are centred at zero and print their values.
   - Why: [S: [WCAG 2.2 SC 1.4.1](https://www.w3.org/TR/WCAG22/#use-of-color); [Okabe and Ito](https://jfly.uni-koeln.de/color/); [Bloomberg CVD, excerpt](https://www.bloomberg.com/ux/2021/10/14/designing-the-terminal-for-color-accessibility/)].
8. **Calm and fast.**
   - A sidebar change that keeps the source and the valuation time settles within 1.0 s. A new source or valuation time refits behind a skeleton (2.12).
   - Stale content in the tab body dims after 300 ms; the header and sidebar never dim.
   - Nothing moves or refreshes on its own.
   - Why: [S: [Nielsen response times](https://www.nngroup.com/articles/response-times-3-important-limits/); [Ström, UI density](https://mattstromawn.com/writing/ui-density); [WCAG 2.2.2](https://www.w3.org/TR/WCAG22/#pause-stop-hide)].
9. **Everything that matters is printed and reachable by keyboard.**
   - No value exists only in a hover.
   - Every chart or table selection has a widget route.
   - Tab switching, Reset and Undo have shortcuts.
   - Why: Jane Street: "Everything is accessible from the keyboard … You can't hover to see more" [S: [Signals and Threads](https://signalsandthreads.com/building-tools-for-traders/)]. Plotly charts cannot be operated by keyboard [J, principles-streamlit 2.11].
10. **Two levels of disclosure, one-line captions.**
    - Level 1 is the tiles and the primary view. Level 2 is a report selector, an expander, a popover or the Methodology drawer.
    - Captions are at most 140 characters; definitions go in `help=` or Methodology.
    - Why: [S: [Nielsen, progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/)]; the audit counted an 8-line Monte Carlo caption and 7 paragraphs on Risk [V-audit G5].
    - Test: an AppTest asserts every `st.caption` in a tab body is at most 140 characters.

---

## 2. Global shell

### 2.0 Context model (new)

Each piece of state has exactly one owner widget or module. Everything else reads it.

| State | Owner (widget or module) | URL | Read by | Rule |
|---|---|---|---|---|
| Valuation time | `st.datetime_input` key `asof`, `step=300` (sidebar) | `asof` (bound) | every tab | Default: the last completed cash close (2.3). An upload whose file states a quote time sets it (precedence below). `st.datetime_input` with `bind` and `step` exists on 1.65.0 and 1.66.1 [V-me `spec/v02/sigcheck_v02.txt`]; DK-7 checks how it serialises, and `_read_link` also accepts `2026-10-23T16:00`. |
| Source | `source` segmented control (sidebar) | `source` | every tab | SYNTHETIC, UPLOAD or FLAT. The Surface data panel shows it read-only. |
| Spot S | Computed under Synthetic and Upload; `spot` under Flat | `spot` (Flat only) | every tab | Under a chain source S is a market observable, shown muted. A move goes through the what-if spot shift `ws` under the declared dynamics `dyn`. Regenerating and refitting the chain per spot value takes 0.77 to 0.94 s with today's fitter [V-pr], and would act like sticky moneyness while the strip defaults to sticky strike, so the same "−2%" would give two P&Ls [V-pr; J]. |
| Forward and DF per expiry | `surface_state` (SF-3, SF-11) | `sf_fwd` | every tab | With `sf_fwd=parity` (default), Pricing, IV, Strategy, Risk and Surface all use each expiry's parity F and DF. The sidebar r and q are the fallback for expiries where parity fails (flagged `carry`), for custom dates and for `sf_fwd=sidebar`. The carry gap is shown in the Surface expiry table. |
| Rate shift `wr` | what-if strip | `wr` | Strategy, Risk | Holds S and the implied dividend yield fixed and raises every expiry's r, so each F moves by e^{Δr·T}. This is RK-1's rho convention (per 1% rate rise at fixed S) [P]. |
| Published surface | `surface_state.current()` | the filter params | every tab | Always published; no toggle. A non-default fit setting shows the amber FIT chip. |
| Marks (UI-23) | `surface_state` | `marks` | every tab | While any mark is non-zero, theo uses the marked surface, the MARKS chip shows, and Edge columns appear. |
| Focus expiry | `expiry` selectbox (sidebar) | `expiry` | Pricing, IV, Surface | Surface's expiry table and its ◂ ▸ buttons write it in callbacks. If the strike is not listed at a new focus expiry, it snaps to the nearest listed strike, with a one-line note. |
| Pricing strike | `strike` (sidebar) | `strike` | Pricing, IV | The Pricing chain's row click writes it in a callback. |
| Position | `tab_strategy` legs state; `strategy_expiry` | `strategy`, `legs` | Strategy, Risk | Legs store an expiry date and settlement. Each run converts them to a year fraction with `clock.time_to_expiry(asof, date, style=…)`. |
| What-if shifts | `ws`, `wv`, `wvmode`, `wd`, `wr` (UI-4) | same names | Strategy, Risk only | Pricing, IV and Surface show the market. Their context line ends with a muted note while any shift is non-zero: `Scenario on Strategy, Risk: spot −2.0% · vol +3.00 vp · T+5 ✕`. |
| Smile dynamics | `dyn` (Strategy's Assumptions popover) | `dyn` | Strategy, Risk | `sticky strike` (default) or `sticky moneyness` (RK-7). The shown delta follows it (RK-7a). |
| Engine | `model` (sidebar): `European (BSM)` · `American (ALO)` | `model` | per the engine policy below | Defaulted by the preset; disabled at European for the index preset, because SPX options are European-exercise [S: [Cboe SPX specifications](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/)]. |

**Engine policy.**

| Where | Index (European) | ETF (American) | Badge on that tab |
|---|---|---|---|
| Pricing, IV | BSM | ALO (scheme `fast` · `accurate` · `high` in Engine settings), or BSM if chosen | `BSM · European`, `ALO · accurate · American` |
| Pricing › Engine diagnostics | BSM, ALO × 3 schemes, LR 501 and 2,001, CRR 2,001, MC RQMC 2¹⁷ × 16 | same | named per row |
| Strategy, Risk | BSM on every leg | BSM until RK-12 merges, then a per-expiry ALO price grid | `BSM · European legs`; `BSM · early exercise ignored`; `ALO grid · American` |
| Surface | LBR inversion (PR-1); de-Americanised quotes for American roots (PR-4) | same | `SVI · arb‑free · parity F` |

- Badge help gives the engine's validated error from `validation_results.json`, for example "ALO accurate: RMSE ≤ 5e−6 against QdFp on the 4,495-put grid" [P: PR-2 ACC].
- Why Tree and Monte Carlo are not global: under the tree, the ±10 chain takes 0.7 to 2.0 s and a 121-point profile 3.9 to 11.7 s at 2,001 to 5,001 steps [V-pr]. Monte Carlo yields only Δ, and 100,000 paths per scenario cell meets no budget [V-mm].

**Precedence on load.**

1. Code defaults.
2. The preset.
3. Explicit URL parameters, which override the preset.
4. Legacy parameters (`days`, `model=binomial`, `strategy_category`), mapped by `_read_link`, each with a one-line note.
5. On an upload, the file's quote time sets `asof` when the layout carries one: Cboe's download layout has a `Date: October 26, 2026 at 10:00 AM EDT` line [V-me `tests/fixtures/chain_cboe_wide.csv`]. Otherwise `asof` stays, and the data panel reads "quote time unknown".
6. A later change of `asof` is allowed and flagged in the data panel as "valuation 2h 10m after quotes".

**Staleness.**

- A leg whose settlement falls at or before `asof` is shown greyed as `expired` and valued at intrinsic at the scenario spot. Its realised P&L is kept until the user removes it.
- A leg whose expiry is not listed in the current source keeps its date and takes its vol from the surface, interpolated in total variance, tagged `unlisted`.
- The share link carries the commit SHA and the seed. A link opened on another commit shows: "Made on `abc1234`; numbers may differ."

**Caching.** The chain and fit are cached with `st.cache_resource` per (source, `asof`, seed or file hash, fit settings). The end-of-day default is therefore shared by every visitor all day [J].

### 2.1 Page frame

```
┌ sidebar (resizable) ┐┌ main ──────────────────────────────────────────────────────────────────────────┐
│ Preset              ││ HEADER  OPTIONS ANALYTICS  23‑Oct‑26 16:00 ET · SPX 5,800.00                    │
│ Source              ││         [SYNTHETIC] [OVERRIDE] [FIT] [MARKS] [BSM · European] [Share] [Methodology]│
│ Valuation           ││ TABS  Pricing  Strategy  Risk  Surface  IV  Methodology                         │
│ Underlying          ││ ─ tab body: context line → tiles → primary view → secondary views → footer ─    │
│ Contract            ││   (amber chips appear only while that state is active)                          │
│ Model               ││                                                                                 │
│ v3.0 · Source       ││                                                                                 │
└─────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘
```

- `layout="wide"`, `initial_sidebar_state="auto"` (unchanged).
- Breakpoints [J]:
  - desktop ≥ 1280 px;
  - laptop 1024 to 1279 px;
  - tablet 641 to 1023 px;
  - phone ≤ 640 px. Streamlit stacks columns at ≤ 640 px [V-probe].
- Vertical rhythm: 0.75 rem between rows (DK-5 `desk.css`); Streamlit's default gap inside a tab.
- Tab labels: 0.875 rem, weight 600, through `.st-key-tab [role='tab'] p`. Today they are 12.25 px, the size of captions [V-audit G6].
- Section headings: `st.subheader` at the theme's h3 (1 rem, weight 600 [V: config]). At most one per panel. Chart titles never repeat a heading.
- **Context line.** Every tab body starts with one line that names what the tab is about:
  - Pricing and IV: the contract;
  - Strategy: the builder bar;
  - Risk: the position;
  - Surface: the data panel.

  On Pricing, IV and Surface the line ends with the muted scenario note while any what-if shift is non-zero.

### 2.2 Header strip (DK-5, extended by UI-2)

DK-5 builds `st.container(horizontal=True, key="desk_header")` with the brand, the ticker, the model badge, a Share popover and a Methodology button [V: DK-5 branch `header.py`]. This spec makes the ticker market-level and adds the state chips.

| Element | Component | Content | Rule |
|---|---|---|---|
| Valuation time | First ticker item | `23‑Oct‑26 16:00 ET` | From `asof`. Help names the session ("Cboe SPX closed · cash close"). |
| Underlying and spot | Ticker item | `SPX 5,800.00` | Root from the chain (OCC roots: SPX and SPXW share the `SPX` underlying [V: `synthetic.py`]) or the uploaded file. Under Flat: `FLAT 100.00`. |
| Source chip | `st.badge` | `SYNTHETIC` (gray); `UPLOAD · spy_chain.csv` (blue; name truncated to 20 characters, full name in help); `FLAT` (gray) | Always present. Help on SYNTHETIC: "Synthetic — not market data" (`synthetic.SYNTHETIC_LABEL`). Help on UPLOAD: quote time and age. |
| OVERRIDE chip | `st.badge(color="yellow")` | `OVERRIDE` | While any user override exists: the Pricing σ, a leg IV, a fallback carry. Help lists each one, e.g. "σ 16.00% on 20‑Nov‑26 5,815 (+0.54 vp vs surface)". |
| FIT chip | `st.badge(color="yellow")` | `FIT` | While any fit setting is not the default (unconstrained calibration, trading time, sidebar carry, changed filters). Help lists them. SF-9's time basis changes every IV in the app [V-mm]. |
| MARKS chip | `st.badge(color="yellow")` | `MARKS` | While any trader mark is non-zero (UI-23). Help lists the marked expiries. |
| Engine badge | `st.badge` | The open tab's engine (2.0 engine policy) | Help: the validated error. |
| Share | `st.popover` | The link and a note of what it holds | Lists `asof`, the what-if shifts, the open report, the legs, the marks, the commit SHA and the seed (UI-11). |
| Methodology | Button | Opens the DK-6 drawer with the open tab's section first | |

Header rules:

- Height at most 40 px at 1440 px with the sidebar open, in the worst state: UPLOAD with a 20-character name, OVERRIDE, FIT and MARKS, and the ALO badge.
  - Estimate [J, from DK-5's font metrics as measured by V-pr]: brand about 140 px; a 34-character ticker about 231 px; the action group about 410 px with the source chip; three state chips about 270 px. Total about 1,050 px against 1,038 px available.
  - So: if the DK-7 test fails, the three amber chips merge into one `STATE 3` chip whose help lists them.
- At most 2 lines at 390 px. On phones the chips shorten (`SYN`, `UPL`, `FLAT`, `OVR`, `FIT`, `MRK`) and the badge to `BSM`, `ALO` or `BSM eu`, through DK-5's `@container desk-header (width < 30rem)` [V: `desk.css`].
- Not in the header: K, expiry, σ, F, r and q. K, expiry and σ belong to Pricing and IV; the forward and carry are per expiry (2.0).

### 2.3 Sidebar (DT-2, DT-2a, DT-3, PR-3; owners WS-DATA for `inputs.py`, WS-PRICING for `model_config.py`)

Six groups, top to bottom. Every widget is bound to the URL unless noted.

| # | Group | Widgets (key → URL) | Default | Notes |
|---|---|---|---|---|
| 1 | Preset | `st.selectbox` `preset` | `Index (synthetic)` | Options: `Index (synthetic)`, `ETF, American (synthetic)`, `Textbook (S = K = 100, 1y)`. An `on_change` callback sets groups 2 to 6. Textbook reproduces v2 exactly: S = K = 100, 365 d, σ 20%, r 5%, q 0, BSM, Flat source. `Single stock with earnings` is added only when DT-9 merges, because `sample_chain` supports only the index and ETF styles today [V-pr]. |
| 2 | Source | `st.segmented_control` `source`: `Synthetic` · `Upload` · `Flat` | `Synthetic` | `Upload` with no file shows one caption and a button "Open Surface data panel", which sets `st.session_state["tab"] = "Surface"` in `on_click` [V-probe E1]. The single uploader lives in that panel (SF-11). `Flat` means no chain: S, r, q and σ are typed, and surfaces are off. |
| 3 | Valuation | `st.datetime_input` `asof` (step 300 s); a `Now` button (`on_click` sets `asof`); a status caption | The last completed cash close: 16:00 ET (13:00 on early closes) of the latest trading day at or before now, from `clock.market_close` and `previous_trading_day` [V: `clock.py`]. For an upload, the file's quote time (2.0). | Label `cash close 16:00 ET`. SPX options trade 9:30 to 16:15 ET, with a curb session 16:15 to 17:00 and GTH 20:15 to 09:25 [S: [Cboe SPX specifications](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/)]. The default stays at the cash close because that is when the index level is set [J]. The status caption names the product's session through the new `clock.session(product, ts)` (DT-2a): "Cboe SPX closed · GTH opens Sun 20:15 ET", "Cboe SPX regular · closes 16:15 ET (3h 12m)", "Holiday: Thanksgiving". NYSE holidays are used; GTH holiday exceptions are not modelled, and Methodology says so. The clock never ticks by itself. |
| 4 | Underlying | Synthetic or Upload: read-only muted lines, plus a `Fallback carry` expander with `rate_src` (`Curve` · `Flat`), `rate`, `div_src` (`Yield` · `Cash`, DT-3) and `div`. Flat: `spot` (2 dp, step 0.05% of spot), `rate`, `div`. | Index: S 5,800.00, the DT-3 curve plus the 45 bp box spread (`synthetic.BOX_SPREAD`), q 1.30% | Muted lines under a chain source: `S 5,800.00 · synthetic close`, then the focus expiry's `F 5,815.22 (parity) · DF 0.99640 · T 0.075970`. For an upload, also the file's S and the parity-implied S gap in bp. |
| 5 | Contract | `expiry` (the focus expiry): `st.selectbox` of listed expiries formatted `20‑Nov‑26 AM · 27d`, plus `Custom date…`; `strike`: `st.number_input`, step = the chain's strike interval (`synthetic.strike_interval`), plus an `ATMF` button that snaps to the listed strike nearest F; `vol_src`: `Surface` · `Override`, with `vol` (%) | The listed expiry nearest 30 DTE: 20‑Nov‑26 AM at 27.73 d beats 27‑Nov‑26 PM at 34.88 d [V-me]. K is the listed strike nearest F (5,815). Vol from the surface. | Help on an AM expiry: "last trade Thu 19‑Nov; settles at the SOQ" [S: Cboe]. An override draws an amber left rule (`.st-key-vol_override`), the caption "+0.54 vp vs surface 15.46%" and the OVERRIDE chip. A custom date has no market: market columns read `–`. The v2 `days` param maps to the nearest listed expiry, with a note. |
| 6 | Model | `model`: `European (BSM)` · `American (ALO)`; an `Engine settings` popover with the ALO scheme | Index: European, with American disabled. ETF: American (ALO, accurate). | Replaces the wrapping horizontal radio [V-audit P8]. Tree and Monte Carlo settings live in Engine diagnostics (3.1). |

The sidebar ends with one muted line: `v3.0.0 · abc1234 · Source · Methodology · Not investment advice`. The "Built by …" block moves to the About menu [V: `app.py` `menu_items`].

Typed inputs live only in the sidebar, the what-if strip, the Strategy builder and blotter, and the IV quote row. Computed values are muted Plex Mono. Overrides are amber, with a rule plus an asterisk [S: HBS above; J].

### 2.4 Navigation, URL state and keyboard

- **Tabs** (unchanged): `st.tabs(TABS, key="tab", on_change="rerun", bind="query-params")`. Only the open tab runs (P0-1). `TAB_ALIASES` stay.
- **What the link holds** (Share): every widget above, plus:
  - each tab's view settings (named in section 3);
  - `ws`, `wv`, `wvmode`, `wd`, `wr` and `dyn`;
  - the open Risk report `report`;
  - the legs `legs` (UI-11) and the marks `marks` (UI-23);
  - `v` = the commit SHA, and the seed.

  Uploaded files cannot travel in a link, and the Share note says so.
- **Hand-typed links**: `_read_link` maps `days` → `expiry`, `model=binomial` or `model=mc` → `model=European (BSM)` plus `pricing_diag=open`, and `preset`. A value it cannot use gets a one-line note, as `tab=` does today [V-audit G10].
- **Keyboard** (DK-8 core, should). Hidden shortcut buttons live in `st.container(key="hidden_keys")`, which CSS hides; this pattern works on 1.65 and 1.66 [V-probe E1].

| Keys | Action |
|---|---|
| Alt+Shift+1 … Alt+Shift+5 | Open Pricing, Strategy, Risk, Surface, IV (`on_click` sets `st.session_state["tab"]`) |
| Alt+Shift+M | Open the Methodology drawer, with the open tab's section first. The Methodology tab stays as the deep link and print surface; there is one renderer. |
| Alt+Shift+0 | Reset all what-if shifts |
| Alt+Shift+Z | Undo the last structural change (UI-24) |
| Alt+Shift+X | Export menu of the open tab's primary table |

- Why Alt+Shift:
  - Firefox on Linux binds Alt+1 to Alt+8 to its own tabs [S: [Mozilla support thread](https://support.mozilla.org/questions/1263764)].
  - Chrome documents Ctrl+1 to Ctrl+8 [S: [Chrome shortcuts](https://support.google.com/chrome/answer/157179)]; Alt+digit on Linux Chrome is undocumented.
  - C and R are reserved by Streamlit even with modifiers, and punctuation is refused [V-probe E4].
- Unverified [J]:
  - Whether Streamlit matches Alt+Shift+digit when the key value becomes `!` on a US layout. DK-7 tests headed Chrome and Firefox on Linux. If either fails, the fallback is Alt+Shift+P, G, K, U, I for the tabs and Alt+Shift+Q for Reset.
  - On macOS, Option+Shift+digit types a character into a focused input unless the handler prevents it. DK-7's Linux runners cannot test this, so it is a manual release check.
- Modifier shortcuts fire inside inputs [V-probe E1], so none of them may type a character a user needs.

### 2.5 Number standard (DK-4, promoted to must; owner WS-DESK, `format.py`)

One table governs every number. `format.py` gains the functions marked "new"; the others exist [V: `format.py`]. In `st.dataframe` the same kinds come from the column registry (UI-3).

| Quantity | Display | Example | Function |
|---|---|---|---|
| Spot, forward, index level | Thousands separator, 2 dp | `5,815.22` | `level` (DK-5 branch) |
| Strike | The chain's increment; no decimals when every listed strike is whole | `5,815`, `582.5` | `strike(k, increment)` (new) |
| Option value per share (theo, bid, ask, mid, entry, package) | 2 dp | `98.57` | `premium` (new) |
| Option value in diagnostics only | 6 dp; differences in scientific notation with U+2212 | `98.570661`, `−3.1e−7` | `premium(x, 6)`, `signed_sci` (new) |
| Money, unsigned (premium amounts, notionals) | $, 2 dp below $1,000, 0 dp from $1,000; in a column one precision chosen from its largest value. Credit or debit is a word, not a sign. | `$2,240 cr` | `money` (exists) |
| P&L and every loss figure | Signed; losses negative, including MAX LOSS, WORST STRESS, VaR and ES. No accounting parentheses. | `+$1,234`, `−$7,760` | `signed(prefix="$")` (exists) |
| Implied vol | %, 2 dp | `15.46%` | `vol` (new) |
| Vol change, RR, BF, skew | vp, 2 dp, signed | `+0.35 vp` | `vol_change` (new) |
| Mid vs fit, buy edge, sell edge | vp, 2 dp, signed; definitions below | `−0.07 vp` | `vol_change` |
| Rate, yield | %, 2 dp | `4.75%` | `pct` (exists) |
| Rate change, carry gap, spread | bp, 0 dp (2 dp below 1 bp, 1 dp below 10 bp), signed | `+45 bp`, `+0.05 bp` | `bp` (new) |
| Delta, per option | Signed, 3 dp, spot convention e^{−qT}N(d₁), the convention of `strike_for_delta` [V]. The Greeks table adds a `Δ fwd` row, ∂V/∂F. | `+0.508` | `greek(kind="delta", digits=3)` |
| Delta in pickers and buckets | Delta points | `16Δ`, `25Δp`, `ATM`, `25Δc` | `delta_label` (new) |
| Position delta | $Δ first. Index presets: ES-equivalent contracts, Δ ÷ (50 · e^{(r−q)T_ES}), with MES at 1/10 [S: [CME micro FAQ](https://www.cmegroup.com/education/frequently-asked-questions-micro-e-mini-equity-index-futures.html), $50 and $5 per point]. ETF and stock: shares. The label names the dynamics: `Δ (sticky strike)` or `Δ (sticky moneyness)` (RK-7a). | `−$19,525 · −0.07 ES`; `−435 sh · −$2.52M` | `money_short`, `futures_equiv` (new), `quantity` |
| Gamma, per option | `Γ 1%`: change in Δ for a 1% move, ΓS/100, 3 significant digits | `+0.0936` | `greek` |
| Gamma, position | `Γ P&L 1% ($)`: ½Γ(0.01S)² × units, the convexity P&L of a 1% move (RK-1 [P]). Help also gives ΓS²/100, the change in $Δ per 1% (200 times larger), so neither reading is mistaken for the other [V-mm, V-pr]. | `−$68` (help: `ΓS²/100 −$13,694`) | `dollar_gamma_1pct` (exists) |
| Vega | $ per vp: per share in contract tables, per position elsewhere | `6.37`, `−$306` | `dollar_vega` (exists) |
| Theta | $ per calendar day: Θ = ∂V/∂t ÷ 365 at fixed IVs (sticky strike), which is RK-1's definition [V: `PositionRisk`] | `+$41` | `dollar_theta` (exists) |
| Theta to next session | V(next session's close) − V(now) at fixed IVs; Friday 16:00 to Monday 16:00 counts 3 calendar days | `+$126` | `money` |
| Rho | $ per 1% rate rise at fixed S and q | `+$31` | `dollar_rho` (exists) |
| Vanna, volga, charm, veta and third order | 3 significant digits, in `PositionRisk` units | `+1.23e−4` | `greek` |
| Probability | %, 1 dp; `>99.9%` and `<0.1%` at the ends; always tagged with its basis: `RN smile` (from the published surface, 2.7 item 6), `flat vol` (N(d₂), Flat source only) or `view` (the user's drift at flat ATMF vol) | `73.5% RN smile` | `probability_text` (moved from `tab_strategy`) |
| Moneyness | K/F in %, 1 dp; k = ln(K/F), 3 dp | `97.5%`, `−0.025` | `moneyness` (new) |
| Quantity of a leg | Signed whole number | `+1`, `−2` | `quantity` (exists) |
| Date | `DD‑Mon‑YY`, with `AM` or `PM` where settlement matters | `20‑Nov‑26 AM` | `date` (exists) |
| Time left | Whole days from 30 days (`55d`); days and hours from 1 to 30 days (`27d 17h`, hours floored); hours and minutes below 1 day (`6h 30m`); `0DTE 2h 15m` on expiry day | `27d 17h` | `time_left(minutes)` (new) |
| Scenario date | `T+N` trading sessions plus the date | `T+10 · Fri 06‑Nov‑26` | `session_label` (new) |
| Timestamp | `DD‑Mon‑YY HH:MM ET` | `23‑Oct‑26 16:00 ET` | `timestamp` (new) |
| Year fraction | Diagnostics only, 6 dp | `0.075970` | `number` |
| Counts | Thousands separator | `2,363` | `number(x, 0)` |
| Missing or unreliable | En dash, with the reason in help | `–` | `placeholder="–"` in grids |

**Edge convention**, in vol space, used on every tab:

- `Mid vs fit (vp)` = σ_mid − σ_theo. Positive means the market is rich against the model.
- `Buy edge (vp)` = σ_theo − σ_ask. Positive means buying at the ask is cheap against theo.
- `Sell edge (vp)` = σ_bid − σ_theo. Positive means selling at the bid is rich against theo.
- `in band`: both edges ≤ 0, i.e. theo lies inside [σ_bid, σ_ask]. It is a flag next to the signed numbers, never a replacement for them.
- Packages use the same signs in $ per share and in vp of the package's vega.
- The Surface residual panel is the same sign, (σ_mid − σ_fit) ÷ the half-spread in vol.

Rules (principles-streamlit 1.7 [J]; Few's pitfall #3 [S]):

- Numbers and their headers are right-aligned. One precision per column.
- Units go in headers: `Vega ($/vp)`, `Θ ($/day)`, `IV (%)`, `Mid vs fit (vp)`.
- No negative zero, and no sign on a value that rounds to zero (`0.00`).
- One minus sign: U+2212 in text, tiles, `st.table` and chart labels. Inside `st.dataframe` number columns the printf hyphen-minus remains (2.8); it aligns right [V-probe].
- Shortened values (`$1.23M`) only in tiles, with the exact value in help.
- Never `nan`, `None`, `n/a` or `1.00` for a quantity.
- The Methodology "Conventions and units" table is generated from this registry, including the time-basis rules: Θ per calendar day, T+N in trading sessions, holidays skipped by `clock.is_trading_day`, 0DTE in hours.

### 2.6 Colour standard (DK-1, DK-3 tokens in `theme.py`; UI-22)

| Role | Dark | Light | Used for | Never used for |
|---|---|---|---|---|
| `profit` / `loss` (= theme `greenColor` / `redColor` [V-pr]) | #26A69A / #EF5350 | #0B7F73 / #D32F2F | P&L fills, cells, bars; `st.metric` deltas through `delta_color="green"` or `"red"` chosen from the sign | Greeks; option identity; alerts |
| `call` / `put` | #3987E5 / #D95926 | #2A78D6 / #EB6834 | Call and put series, smile markers | Warnings |
| `accent` | #9085E9 | #6250D6 | Forward, breakevens, the T+0 curve, marks | Profit or loss |
| `muted` | #8B95A7 | #5B6577 | Spot line, strikes, reference text, extrapolated segments, RN ranges | Data series |
| primary (widget chrome) | #6DA6EE (`linkColor`, 7.62:1 on the background [V-me]) | #2061B1 (6.01:1 [V-me]) | Selected pills and segments, sliders, focus rings, the tab underline: anything interactive | Data; user state |
| amber (`yellowColor`) | #E8A33D | #9E5F00 | User state only: OVERRIDE, FIT and MARKS chips, override rules and asterisks, the active what-if strip | Widget chrome; data; buttons |
| diverging ramp | `chartDivergingColors` | same | Signed heatmaps (P&L, vega buckets); `zmid = 0`, symmetric range | Unsigned data |
| sequential ramp | `chartSequentialColors` | same | Unsigned magnitudes (IV grid, quote counts) | Signed data |

Rules:

- Today `primaryColor = yellowColor`, so every selected widget is amber and "amber means user state" is lost [V-pr]. UI-22 sets `primaryColor` to the link colour in both `config.toml` copies.
- No primary buttons: white on #6DA6EE is about 2.5:1 [V-me formula], as it was 2.16:1 on amber.
- Profit and loss colours always come with a sign, and on charts with position relative to the zero line [S: WCAG 1.4.1].
- The teal and red pair passes the simulated protanopia and deuteranopia checks in `tests/test_theme.py` [V].
- Data-quality states (extrapolated, stale, wide, excluded, one-sided, unlisted) are grey with a dash or hollow marker plus a text tag, following Jane Street's "one quiet accent" [S: Signals and Threads].
- Blocking errors use `st.error`. Nothing else is red except losses.
- Contrast:
  - Stress-bar labels inside a bar use the `surface` colour (5.51:1 on #EF5350, 6.40:1 on #26A69A); outside a bar the `text` colour (15.92:1) [V-me]. Today's white-on-red labels are 3.49:1 [V-audit].
  - Input boundaries need 3:1 (WCAG 1.4.11). Today's #232C3A border is 1.36:1 [V-me]. Inputs get #5B6577 borders in both modes: 3.27:1 dark, 5.73:1 light [V-me]. Cards and tables keep the hairline.

### 2.7 Chart house style (DK-3; `charts.py` owns the builders and `show()`)

The `desk` template stays: unified hover, a crosshair spike, outside ticks, the legend at the top left [V: `charts.py`]. Additions and fixes:

1. **Sizes.** Primary chart 360 px; secondary 260 px. A shared-x Greek strip under a P&L chart is 140 px in the same figure (`make_subplots(rows=2, shared_xaxes=True, row_heights=[0.72, 0.28])`), so the crosshair spans both [V-probe 2.7]. Small multiples 180 px.
2. **Axis titles carry units**: `P&L ($)`, `IV (%)`, `K/F (%)`, `Call delta`, `Strike`, `Expiry`. Ticks use the number standard.
3. **Reference lines.**

   | Line | Style | Label |
   |---|---|---|
   | Spot | Dashed, `muted` | `S 5,800` at the top |
   | Forward | Solid, 1 px, `accent` | `F` |
   | Strikes | Dotted, `muted` | `K 5,400` at the top in 10 px, only when at most 6 strikes are in the visible range; strikes closer than 3% of the range share one label (`K 5,300/5,400`); otherwise in hover and the blotter. The server does not know the plot's pixel width [V-pr; 5 item 6]. |
   | Zero | Solid, `rule` | none |
   | Breakevens | `accent` diamond on y = 0 | `BE 5,378 (−7.3%)` |

4. **Time encodes as dash.** Expiry is solid in the `text` colour. T+0 is solid 2 px in `accent`. Dated curves are `muted`, dashed then dotted. Every curve is labelled at its right end (`exp 18‑Dec`, `T+0`, `T+13 · 11‑Nov`). The same mapping holds on Strategy and Risk.
5. **Fills.** P&L areas only under the expiry curve, at opacity 0.10. Today `FILL_OPACITY` is 0.18 and the loss fill dominates [V-audit S4].
6. **Risk-neutral ranges, not σ bands.**
   - The 68.3% and 95.4% risk-neutral ranges to the front expiry come from the published surface's CDF: P(S_T < K) = 1 + ∂C/∂K ÷ DF, Breeden-Litzenberger [S: [Breeden and Litzenberger 1978](https://doi.org/10.1086/296025)], through `surface_state.current().cdf(K, expiry)` (SF-11).
   - Drawn `muted` at opacity 0.06 and 0.03. The legend names the basis: `RN 68% to 18‑Dec: 5,492–6,167 (smile)`.
   - On the default chain the flat ±1σ band would read 5,459 to 6,163 [V-me]. Under skew, N(d₂) misstates tail probabilities: N(−d₂) gives 16.85% below the 18‑Dec 16Δ put strike, against 11.91% consistent with the smile [V-mm]. The Flat source falls back to lognormal ranges labelled `flat vol`.
7. **Heatmaps.**
   - The diverging ramp with `zmid=0` and a symmetric `zmin`/`zmax`.
   - Cell values are printed by a `go.Scatter(mode="markers+text")` overlay of square, near-transparent markers with `hoverinfo="none"`. Its per-point text colours are chosen for at least 4.5:1 against each cell's fill. Plotly's `heatmap.textfont.color` rejects per-cell arrays, while scatter accepts them [V-mm, V-pr].
   - The same overlay is the click target: `on_select="rerun"`, `selection_mode="points"`. A click on a heatmap cell returns no points [V-pr].
   - Axis labels show level and change: `5,655 (−2.5%)` and `+2.0 vp`.
8. **Annotations go inside the plot area** at the top left, in 11 px `muted` text (`20‑Nov‑26 · 27d · RMSE 0.05 vp · 100% in band · 140 used`). No floating notes above a chart [V-audit R6].
9. **No 3-D except** the Surface tab's opt-in `3D` implied view.
10. **Toolbar.**
    - Remove `"displayModeBar": "hover"`, which is not a documented Plotly value [S: [Plotly configuration](https://plotly.com/javascript/configuration-options/)].
    - `charts.show` wraps each chart in `st.container(key=f"chart_{key}")`. `desk.css` hides `.modebar-container` and Streamlit's element toolbar inside `[class*='st-key-chart_']` under `@media (hover: none)`; DK-7's selector allowlist gains this one selector [V-audit G4].
    - Downloads move to the export menu (2.13).
11. **Range control instead of persistent zoom.** Zoom resets whenever a figure's data change, with or without `uirevision` [V-probe E9]. Every spot-axis chart gets `Range`: `±5%` · `±10%` · `±20%` · `RN 95%` · `Auto`, bound to the URL per tab, setting `xaxis.range` in Python (UI-12).
12. **Accessibility.** `alt` is mandatory (exists). Each primary chart's data is in the export menu as the table equivalent [S: WCAG 1.1.1].

### 2.8 Tables standard (`components.py`, UI-3)

- `st.dataframe(..., hide_index=True, row_height=28)` (`ROW_HEIGHT` exists); the legs editor uses 32 [S: [Carbon row sizes](https://carbondesignsystem.com/components/data-table/style/)].
- `height="content"` for up to 15 rows; it exists on 1.65 and 1.66 [V-me] and removes the two blank stress rows [V-audit R2]. Longer tables are fixed at 15 rows plus the header.
- The first column is pinned (`pinned=True`). Key columns come first, because Python cannot know the viewport (section 5 item 6).
- Every column has `help`. Header text is at most 14 characters.
- A Styler is used for colour only. Every numeric column takes its format from the registry, because a Styler alone shows 6 decimals [V-probe].
- Number columns keep the printf hyphen-minus: NumberColumn formats cannot emit U+2212, and text columns would lose right alignment and sorting [V-probe].
- `st.table(border="horizontal")` holds key-value blocks of up to 8 rows that need U+2212 (exists in `format.right_aligned`).
- **Selection as a click action.**
  - A table that drives other state (the Pricing chain, the Surface expiry table, the Scenarios table) uses `selection_mode="single-row"` or `"single-cell"` with an `on_select` callback that writes the owner key (2.0).
  - Its key includes the current value (`key=f"tbl_surface_expiries_{expiry}"`), so it remounts with no stale selection when the value changes elsewhere. The current row is marked by a Styler background and a `◂` in its key cell.
  - Keyboard row selection is unproven [V-probe E7], so every such table has a widget route.
  - Callbacks are plain module functions (`on_chain_select(event)`), unit-tested directly. AppTest cannot fire `on_select`: injected selections reach the script but call no callback [V-mm, V-pr].
- **Row actions** use an action bar above the table that acts on the selected row (Flip, Duplicate, Roll ▸, Remove). It is disabled when no row is selected. A `ButtonColumn` is one button per cell with one `on_click`, so it cannot be a menu [V-pr].

### 2.9 Headline tiles

- `components.card_row` (exists; DK-5 styles it). Labels uppercase 0.72 rem; values 1.625 rem.
- At most **6 tiles per row** on every tab; `card_row`'s tiers wrap them to 3 + 3 on laptops and phones [V: `components.py`].
- The `delta` slot holds one secondary fact (`±3.40%`, `credit`, `front expiry`), with `delta_arrow="off"`.
  - For P&L, `delta_color` is `"green"` or `"red"` from the sign, never `"normal"`, because `st.metric` reads only an ASCII "-" as negative [V-pr `metric_minus.txt`].
  - Otherwise `delta_color="off"`.
- `help` holds the exact value and a one-sentence definition.

### 2.10 Loading, stale, empty and error states (UI-5)

| State | Treatment | Mechanism |
|---|---|---|
| Rerun under 300 ms | No cue | Ström: below about 100 ms, animation slows an interface [S]. A 300 ms delay avoids flicker on the 0.35 to 0.87 s reruns measured today [V-audit; J]. |
| Rerun over 300 ms | Elements in the tab body fade to 45% opacity until replaced. The header and sidebar never fade. | `desk.css`: `.st-key-tab [data-testid='stElementContainer'][data-stale='true'] { opacity: .45; transition: opacity 0s 300ms }`. `data-stale` exists in both bundles [V-me]. CSS cannot tell which input changed, so the scope is the tab body [J]. |
| Heavy element at the end of a tab (VaR report, Engine diagnostics, surface fit views) | A skeleton of the final height, then `st.status` with named steps when over 1 s ("Inverting 2,363 quotes · Fitting 16 expiries · Checking arbitrage") | A keyed `@st.fragment` placed last, so nothing below it waits; `container.skeleton()` [V-probe 2.1]; `st.cache_resource` for fits, `st.cache_data` for frames |
| Heavy element above lighter content | A placeholder reserved in place (`slot = st.empty()`), drawn as a skeleton, filled at the end of the run | Not `@st.fragment(parallel=True)`: during a parallel run it cannot write to containers created outside the fragment [V-mm] |
| Empty | One sentence on what belongs here, plus the action that fills it | Never "no records" while loading [S: [Kaplan, NN/g](https://www.nngroup.com/articles/empty-state-interface-design/)] |
| Invalid input | `st.error` directly under the widget, one line naming the field, the value and the valid range. The last good result stays, faded, captioned "Showing the last valid result (16:00:05)". | Session-state cache of the last good inputs per tab |
| Unreliable value | `–` plus the reason in help | PR-6 `MonteCarloResult.reliable` [P] |
| Fallback | One `st.warning` line naming what fell back ("ALO refused the double-boundary case; the LR tree value is shown"). The badge names the engine actually used. | PR-2 double-boundary fallback [P] |

Alerts carry no icon: the Material ligature name leaks into the alert's text [V-audit G11].

### 2.11 Phones (390 × 844 reference)

- Header at most 2 lines (2.2). With UI-1 the tab bar's top is at most 170 px. Today it is about 483 px at `0e762c3` [V-audit] and 292 to 358 px on the DK-5 branch [V].
- One chart and one control set per screen [S: the tastytrade, Sensibull and OptionStrat pattern, trader-tools 5.10]. Tiles wrap 3 per row.
- The what-if strip is the way to bump spot, vol and date without opening the full-screen sidebar [V-audit G8].
- Toolbars are hidden on touch (2.7 item 10).
- Acceptance (DK-7, `tests/e2e/test_style.py`), on all six tabs:
  - at 390 × 844 with touch, `document.scrollingElement.scrollWidth <= innerWidth` [S: WCAG 1.4.10];
  - no visible `.modebar-container` or `[data-testid='stElementToolbar']` after a tap on any chart or table;
  - the tab bar's top ≤ 170 px.

  Per-tab tile positions on phones are not gated.

### 2.12 Performance budgets (DK-7, `tests/e2e/test_performance.py`)

Budgets are tracked, not PR gates: the suite runs nightly on Streamlit 1.66, takes the median of 5 runs, records it in `docs/scorecard`, and alerts when a median exceeds its budget by more than 25%. Shared CI runners are too noisy to block a merge on [V-mm; J].

| Interaction | Budget | Basis |
|---|---|---|
| Cold load, cache warm (DOM quiet) | ≤ 3.0 s | Today 1.0 to 3.6 s [V-audit] |
| Cold load, cache cold (chain plus fit) | ≤ 4.5 s | Chain, parse and fit take 0.77 to 0.94 s before SF-3 and SF-4 [V-pr] |
| New `asof` or source (refit) | ≤ 3.0 s, behind a skeleton | as above |
| Strike or focus-expiry change on Pricing, BSM | ≤ 0.6 s | |
| Same, ALO: chain of about 28 rows × 2 sides at about 5 ms per solve on a smile [P: PR-2 ACC; J] | ≤ 0.8 s, chain in fragment `pricing_chain` | The B/K boundary cache is keyed on σ, so a smile needs a solve per strike [V-mm] |
| Profile x = Spot (BSM or ALO, one boundary) | ≤ 0.3 s, fragment `pricing_profile` | |
| Profile x = Vol or Time under ALO, 41 points × 2 | ≤ 1.0 s behind a skeleton | The boundary changes with σ and T [V-pr] |
| Engine diagnostics opened (tree 2,001, CRR, MC 2¹⁷) | ≤ 2.0 s behind a skeleton | Tree 16 ms per price, 129 ms with bumped Greeks; MC 2¹⁷ 37 ms [V-pr] |
| What-if step on Strategy (BSM legs) | ≤ 0.5 s, fragment `strategy_analysis` | |
| What-if step on Risk, tiles | ≤ 1.0 s | Today about 1.0 s with VaR [V-audit] |
| VaR report (FHS, BSM legs) | ≤ 2.5 s, in a fragment | |
| Tab switch | ≤ 0.6 s | P0-1 |
| Websocket per rerun | ≤ 60 KB | P0-1 target |

UI-21 measures `server.enableWebsocketCompression = true` before any other payload work [V-probe 2.1: off by default].

### 2.13 Exports and provenance (UI-28, new)

Every export carries its provenance inside the file. A CSV of SPX theos without the header chip reads as market data [V-mm; DT-6].

- **CSV**: leading comment lines, which pandas reads with `comment="#"`, then the header row:
  ```
  # source: SYNTHETIC — not market data
  # as_of: 2026-10-23T16:00-04:00 (cash close)
  # model: BSM European; dynamics: sticky strike; forwards: parity; fit: SVI arb-free
  # app: v3.0.0 (abc1234); seed: 0
  ```
- **XLSX**: a `Provenance` sheet with the same rows. **JSON**: a top-level `provenance` object.
- **File names** carry the source and time: `spx_pricing_chain_SYNTHETIC_20261023T1600.csv`.
- Downloads use `st.download_button(data=callable)`, so a file is built only on click [V-probe].
- Test: an AppTest calls each export builder and asserts the four lines. DT-6's policy test asserts no export lacks `source:`.

---

## 3. Tabs

Each tab section gives: its job; the layout from top to bottom with component, key and URL parameter; the default state; interactions; conventions; what is removed; and acceptance tests. AppTests live in the owner's `tests/test_app_<tab>.py`, callback unit tests in `tests/test_callbacks_<tab>.py`, and Playwright tests in `tests/e2e/` (DK-7). Every AppTest pins `?asof=2026-10-23T16:00&preset=…` so results do not depend on the day.

### 3.1 Pricing (PR-3, PR-5, PR-7; owner WS-PRICING: `tab_pricing.py`, `charts_pricing.py`, `cache.py`, `model_config.py`)

**Job.** Price the Pricing contract, call and put, with the selected engine. Show its Greeks in desk units next to its market quote, and prove the number.

```
CONTRACT  SPX 20‑Nov‑26 AM 5,815 · last trade Thu 19‑Nov · settles at the SOQ                 (scenario note when active)
TILES     CALL 98.57 | PUT 98.35 | STRADDLE 196.92 ±3.40% | σ 15.46% surface | FORWARD 5,815.22 pair +0.05 bp | TIME 27d 17h
CONTRACT  Type  Bid    Ask    Mid    Theo   IV bid  IV ask  Mid vs fit(vp)  Δ       Γ 1%     Vega($/vp)  Θ($/day)  ρ($/1%)
          Call  98.20  99.00  98.60  98.57  15.40%  15.52%  0.00 in band    +0.508  +0.0936  6.37        −2.04     +2.17
          Put   97.70  99.00  98.35  98.35  15.35%  15.56%  0.00 in band    −0.491  +0.0936  6.37        −1.49     −2.24
VIEW      [Value Δ Γ Vega Θ]  x:[Spot Vol Time]  Range:[±5 ±10 ±20 RN95 Auto]   (360 px chart)
CHAIN     Window [10Δ–10Δ ▾] Stride [25 ▾]   calls: Bid Ask Theo IV Δ | STRIKE | puts: Δ IV Theo Bid Ask   (28 rows, 5,425–6,075; click = set K)
GREEKS    Order:[1st +2nd +3rd]  rows Δ · Δ fwd · Γ 1% · Vega · Θ · ρ · ε · … · P(ITM) RN smile   cols Call Put (+ BSM ref under ALO)
▸ Engine diagnostics (collapsed): engine comparison · convergence · exercise boundary (American)
footer    BSM · European · T to 20‑Nov 09:30 ET (AM) = minutes/525,600 · σ from the surface at K/F 100.0% · per share · Methodology →
```

**Layout, top to bottom.**

1. **Context line.** `st.caption` naming the contract, its last trading day and its settlement. AM monthlies stop trading on the business day before, usually a Thursday; SPXW stops at 16:00 ET on expiry day [S: Cboe].
2. **Headline tiles.** `card_row("pricing_tiles", …)`, 6 tiles. They replace the global price cards (UI-1).

   | Tile | Value | Delta slot | Help |
   |---|---|---|---|
   | `CALL` | Theo, 2 dp | `1.70% of S` | Engine and exact value to 6 dp. Under ALO it also gives the difference to LR 2,001 in scientific notation (format `ALO vs LR 2,001: −3.1e−7`, illustrative). |
   | `PUT` | Theo | `1.70% of S` | Same |
   | `STRADDLE` | Call + put theo | `±3.40%` (straddle ÷ S, the breakeven move to expiry) | Replaces "Call ITM probability", which the audit found low-value [V-audit P4] |
   | `σ` | Vol used | `surface` or `override +0.54 vp` | Source and K/F |
   | `FORWARD` | F of the focus expiry (`parity` or `sidebar`) | `pair +0.05 bp`: the forward implied by this strike's quoted pair, K + (C_mid − P_mid) ÷ DF, against F | A market check, replacing v0.1's MODEL CHECK, which under BSM was parity against itself, always 0 [V-mm, V-pr]. `–` under Flat or when either side is unquoted. |
   | `TIME` | `27d 17h` | `AM settle` | `T = 0.075970` (minutes/525,600, DT-1) |

3. **Contract table.** `st.dataframe` key `tbl_pricing_contract`, two rows, `height="content"`.
   - Columns: `Type · Bid · Ask · Mid · Theo · IV bid · IV ask · Mid vs fit (vp) · Δ · Γ 1% · Vega ($/vp) · Θ ($/day) · ρ ($/1%)`, plus `Early-ex ($)` under ALO.
   - With an override or marks, `Mid vs fit` is replaced by `Buy edge (vp) · Sell edge (vp)`, with `in band` in the flag.
   - Market columns come from the source chain at the focus expiry and strike. An unlisted contract (custom date, custom strike, Flat) shows `–` with help "Not a listed contract in the source".
   - `pr_basis` (`Per share` · `Per contract ×100`) multiplies the money columns.
4. **Primary view.** Controls in one horizontal container: `pr_metric` (`Value` · `Δ` · `Γ` · `Vega` · `Θ`), `pr_x` (`Spot` · `Vol` · `Time`) and `Range` `pr_range` (UI-12).
   - Then `charts_pricing.profile_chart` (360 px), in fragment `pricing_profile`.
   - Call and put from the engine are solid lines; under ALO the BSM European reference is a thin dotted line labelled `BSM ref`; a marker shows the current x.
   - Default: `Value` against `Spot`, `±10%`.
5. **Chain ladder (UI-7).** `st.dataframe` key `tbl_pricing_chain_{strike}`, mirrored the OMON way [S: [HBS](https://www.library.hbs.edu/services/help-center/bloomberg-options)], in fragment `pricing_chain`.
   - Columns: `Bid · Ask · Theo · IV · Δ | Strike | Δ · IV · Theo · Bid · Ask`.
   - Window `pr_chain_win`: `10Δ–10Δ` (default) · `25Δ–25Δ` · `RN 95%` · `All`. On the default 20‑Nov expiry, 10Δp to 10Δc spans 5,425 to 6,075: 101 listed strikes [V-me]. ±10 strikes on the 5-point grid would cover only ±0.9%, against a 1σ move of 4.26% [V-mm, V-me].
   - Stride `pr_chain_stride`: `Auto` is the smallest of 5, 10, 25, 50 and 100 points (strides that land on listed strikes) that gives at most 30 rows: 25 points here, 27 rows. The Pricing strike's row is always shown, so 28 rows [V-me].
   - Theo is the engine's value at each strike's surface vol; IV is the mid IV.
   - The Pricing strike's row carries a background and `◂`. A row click calls `on_chain_select`, which sets `st.session_state["strike"]` (2.8). The keyboard route is the sidebar strike input.
6. **Greeks table.** `st.dataframe` key `tbl_pricing_greeks`; the order control `pr_greeks` (`1st` · `+2nd` · `+3rd`).
   - Rows: Δ, Δ fwd, Γ 1%, Vega, Θ, ρ, ε (dividend), then vanna, volga, charm, veta, then speed, zomma, color, ultima, then `P(ITM)`.
   - Columns: `Call (ENGINE)` and `Put (ENGINE)`, plus `BSM ref` call/put when the engine is not BSM and `pr_bsm_ref` is on.
   - Greeks the engine does not produce show `–` with the reason.
   - `P(ITM) (RN smile)` is −∂C/∂K ÷ DF for the call and ∂P/∂K ÷ DF for the put, from the published surface. On the noise-free chain that is 57.3% for the call and 42.7% for the put, against N(d₂) of 49.2% and 50.8% [V-me]. Help gives the N(d₂) value labelled "flat vol". Under Flat the row reads `P(ITM) (flat vol)`.
7. **Engine diagnostics.** `st.expander("Engine diagnostics", key="pricing_diag", on_change="rerun")`, computed only while `.open` [V-probe 2.1], placed last.
   - Engine comparison: rows BSM; ALO fast, accurate and high; LR 501 and 2,001; CRR 2,001; `MC RQMC 2¹⁷ × 16`. Columns: Call, Put, `Δ vs reference` (scientific), `ms`. The reference is ALO high for American, BSM for European.
   - Monte Carlo settings sit in a popover inside the expander: paths offered as powers of two (2¹⁴ to 2²⁰) when RQMC is on, because scrambled Sobol needs n = 2^k [V-pr: scipy 1.17.1 warns at 100,000]; seed; antithetic; control variate.
   - One line, "Validation: ALO accurate vs QuantLib QdFp on the 4,495-put grid, RMSE <value> (commit <sha>, <date>)", read from `methodology/validation_results.json` (VL-7). QuantLib is not imported at runtime.
   - Convergence: |error| against steps, log-log, with the reference slope; for MC, the 95% interval width against paths, with a 1/√N guide. Today the error is drawn on top of its reference line [V-audit P6].
   - Exercise boundary (American): the x-axis from 0, the put labelled [V-audit P9].
8. **Footer line.** `components.footer(text, anchor="pricing")`: one muted line ending in a `Methodology` link.

**Default state.** Index preset at the last cash close:

- the focus expiry nearest 30 DTE (20‑Nov‑26 AM);
- K the listed strike nearest F (5,815);
- σ from the surface; European BSM;
- Engine diagnostics collapsed.

**Interactions.**

- A strike change from the chain or the sidebar updates the context line, the tiles and every view.
- A focus-expiry change from the sidebar or the Surface tab does the same, snapping K if needed (2.0).
- The metric, x-axis and range controls rerun only `pricing_profile`.

**Desk conventions.** Prices per share. Mid vs fit and edges in vp. Δ in the spot convention, with Δ fwd beside it. Γ per 1%, vega per vp, Θ per calendar day, ρ per 1%.

**Removed.**

- The global price cards (UI-1).
- The `Unit` and `Definition` columns and 6-decimal Greeks [V-audit P2].
- The model-switch info box.
- The Theoretical option chain expander (replaced by item 5).
- The Sensitivity explorer heading.
- From diagnostics: the MC path fan and the value grid [V-mm].
- The MODEL CHECK tile.
- "Edge (vp)" against the fit.

**Acceptance tests.**

- AppTest, ETF preset with `model=American (ALO)`:
  - the put Δ in `tbl_pricing_contract` equals the put Δ in `tbl_pricing_greeks` exactly (fixes P1);
  - no column header contains "BSM" unless `pr_bsm_ref` is on.
- AppTest, default index preset at `asof=2026-10-23T16:00` [V-me goldens]:
  - `CALL` reads `98.57`, `PUT` `98.35`, `STRADDLE` `196.92` with delta `±3.40%`;
  - `σ` reads `15.46%`, `FORWARD` `5,815.22` with delta `pair +0.05 bp`, `TIME` `27d 17h`.
- AppTest, default: both rows' `Mid vs fit (vp)` read `0.00` with `in band`, and no column header contains "Edge".
- AppTest, `vol_src=Override`, `vol=16`: the call's `Buy edge (vp)` reads `+0.48` (16.00 − 15.52) and `Sell edge (vp)` reads `−0.60` (15.40 − 16.00); the header shows OVERRIDE.
- AppTest, noise-free fixture: `P(ITM) (RN smile)` lies within 0.5 pp of 57.3% (call) and 42.7% (put), and its help contains "N(d₂) 49.2%".
- AppTest, default: `tbl_pricing_chain_*` has 28 rows from 5,425 to 6,075 and includes 5,815.
- Unit test: `tab_pricing.on_chain_select` with a selection payload for the 5,850 row sets `session_state["strike"] == 5850`. Playwright: clicking that row changes the context line to `5,850` and the CALL tile.
- AppTest: with `pricing_diag` closed, patched counters on the tree, CRR and MC calls in `cache.py` read 0.
- AppTest: every caption in the tab is at most 140 characters.
- Playwright 1440 × 900, dark (nightly also light):
  - the tiles' top ≤ 180 px;
  - the profile chart's bottom ≤ 900 px;
  - page height with diagnostics collapsed ≤ 1,800 px.
- Performance (2.12): a strike change ≤ 0.6 s under BSM and ≤ 0.8 s under ALO.

### 3.2 Strategy (BK-2, BK-3, RK-9a, BK-4; owner WS-BOOK: `tab_strategy.py`, `charts_risk.py`, new `whatif.py`)

**Job.** Build a multi-leg position from a template or the chain. Read its premium, package market, maximum P&L, breakevens and risk-neutral probability, and its P&L across spot, time and vol, before trading it.

```
BUILDER   Template [Range-bound · Iron Condor ▾]  Strikes by [Δ|Width]  Shorts [16Δ]  Wings [100 pts]  Expiry [18‑Dec‑26 AM · 55d ▾]  [1 × 100 · $0.00/ct · RN · sticky strike ▾]
TILES     NET PREMIUM $2,240 cr | PACKAGE 20.80 / 24.00 cr | MAX PROFIT $2,240 | MAX LOSS −$7,760 | BREAKEVENS 5,378 / 6,172 | P(PROFIT) 73.5% RN smile
WHAT-IF   Spot [−][0.0%][+]  Vol [−][+0.00 vp][+]  Date [T+0 · Fri 23‑Oct ▾]  Rate [+0 bp]   [Reset] [Undo]     sticky strike
VIEW      [Chart|Table]  Range[…]  Curves: exp · T+0 · ⅓ · ⅔     Greek strip: [Δ Γ Vega Θ]
          (P&L chart 360 px + 140 px Greek strip, one figure)
BLOTTER   [View|Edit]  [Shift −/+] [Width −/+] [Roll ◂ ▸] [Flip] [Δ hedge ▾]          selected row: [Flip] [Duplicate] [Roll ▸] [Remove]
          On Qty  Type Expiry        Strike  Bid    Ask    Mid    IV      IV src   Mid vs fit  $Δ        Γ P&L 1%  Vega($/vp)  Θ($/day)
          ✓  +1   Put  18‑Dec‑26 AM  5,300   25.30  26.20  25.75  20.59%  surface  …
          ✓  −1   Put  18‑Dec‑26 AM  5,400   35.20  36.10  35.65  19.66%  surface  …
          ✓  −1   Call 18‑Dec‑26 AM  6,150   21.00  21.70  21.35  12.72%  surface  …
          ✓  +1   Call 18‑Dec‑26 AM  6,250    8.50   9.20   8.85  12.08%  surface  …
          Total                                                                      −$19,525  −$68      −$306       +$41
footer    BSM European legs at surface IVs · sticky strike · entry at mid · P(profit) RN from the smile at 18‑Dec · expired legs at intrinsic · Methodology →
```

**Layout, top to bottom.**

1. **Builder bar.** `st.container(horizontal=True, key="strategy_builder", wrap=True)`.
   - **Template**: one `st.selectbox` key `strategy`. Options carry a category prefix (`Range-bound · Iron Condor`), grouped in `strategies.TEMPLATE_CATEGORIES` order [V-me: 8 categories, 26 templates]. `_read_link` maps the old `strategy_category` and `strategy_<cat>` params.
   - **Strikes by** `strategy_strikes_by`: `Δ` · `Width`.
     - With Δ: `strategy_short_delta` pills (`10Δ` · `16Δ` · `20Δ` · `25Δ` · `30Δ`) and `strategy_wing_pts`, the wing width in points (pills at 1, 2, 3 and 4 × the local strike interval: `50` · `100` · `150` · `200` for SPX at these strikes), equal on both sides.
     - Short strikes are the listed strikes whose own delta at their surface IV is nearest the target (RK-9a). `build_template` today solves delta at the flat `market.volatility` [V-me].
     - Wings by points avoid the asymmetric 5Δ wings of v0.1, which gave 350- and 150-point wings and a $30,921 maximum loss [V-mm].
     - With Width: `strategy_width_pct` (exists).
   - **Expiry** `strategy_expiry`: listed expiries; default the monthly (AM) nearest 45 DTE: 18‑Dec‑26 at 55.7 d, against 20‑Nov‑26 at 27.7 d [V-me]; RK-9's acceptance uses 45 d [P].
   - **Position settings** popover, one for both, its label a summary (`1 × 100 · $0.00/ct · RN · sticky strike`):
     - `contracts`, `multiplier`, `commission` (exist);
     - drift (risk-neutral or the user's view) and premium financing (exists);
     - smile dynamics `dyn`: `Sticky strike` · `Sticky moneyness` (RK-7 [P]; Derman 1999 regimes [S: [Derman](https://hpirotte.ulb.be/INGESTriskmgt/readings/Derman%20(1999)%20-%20Volatility%20regimes.pdf)]).

     This replaces the row of three steppers [V-audit S7] and the separate Assumptions popover.
2. **Headline tiles.** `card_row("strategy_tiles", …)`, 6 tiles.

   | Tile | Value | Delta slot / help |
   |---|---|---|
   | `NET PREMIUM` | `$2,240` | `credit` or `debit`. Help: 22.40 per share at mid; the theo credit is 22.64, so entry at mid is −$24 against theo (the T+0 P&L at today's spot). |
   | `PACKAGE` | `20.80 / 24.00` | The structure's natural market: the bid is Σ shorts' bids − Σ longs' asks, the ask Σ shorts' asks − Σ longs' bids, per share, with a `cr` or `db` tag. Delta slot: `theo 22.64`. Help: package mid vs fit in vp of the package vega, cost to cross ($160 per side here), and close-out value at the market against mark-to-mid. This is how a desk quotes a spread [V-pr; J]. |
   | `MAX PROFIT` | `$2,240` | Basis: front expiry |
   | `MAX LOSS` | `−$7,760` | For multi-expiry structures the delta slot reads `front expiry`, and help says later losses can exceed it [S: [IBKR Performance Profile](https://www.ibkrguides.com/traderworkstation/performance-profile.htm)]. |
   | `BREAKEVENS` | `5,378 / 6,172` | `−7.3% / +6.4%` from S. With more than two: the two nearest S plus `+1 more`, all listed in help. Basis: front expiry. |
   | `P(PROFIT)` | `73.5%` | `RN smile` or `view`. Help: expected P&L and its standard deviation under that basis. |

   Position Greeks are in the blotter's Total row, the Greek strip and the Risk tiles; there is no second tile row [V-mm].
3. **What-if strip (UI-4).** `whatif.strip(key="whatif")`, shared with Risk. One line while every shift is zero; it wraps on narrow screens.

   | Control | Key / URL | Label (the semantics are written in the label) |
   |---|---|---|
   | Spot shift | `ws` | %, step 0.5: `Spot −2.0% → 5,684.00` |
   | Vol shift | `wv` | Absolute vp by default: `+3.00 vp: ATMF 15.52% → 18.52%`. The mode `wvmode` (`absolute` · `relative` · `1M-scaled`) is RK-1's `vol_mode` [P]. Relative reads `+10%: 15.52% → 17.07%`. |
   | Date | `wd`, an integer T+N in trading sessions | `st.select_slider(options=range(0, N + 1), format_func=session_label)`: `T+10 · Fri 06‑Nov‑26 (41d left)`. A bound select_slider serialises its options, so they are integers [V-pr]. |
   | Rate shift | `wr` | bp: `+25 bp: 4.75% → 5.00%` (2.0 rule) |
   | Reset | button `whatif_reset` (Alt+Shift+0) | Clears all four |
   | Undo | button (UI-24) | |

   - While any shift is non-zero, the strip leads with an amber `SCENARIO` label, its container gets the amber left rule `.st-key-whatif.active`, and the other tabs show the muted note (2.0).
   - A caption names the dynamics: `sticky strike: leg vols fixed by strike`.
   - IBKR's "+15%" is relative and Deribit's "+10%" absolute, so a label without a worked example is ambiguous [S: Deribit, IBKR above; trader-tools mistakes 2].
4. **Primary view.** `st_view` (`Chart` · `Table`), OptionStrat's Table/Graph toggle [S: [OptionStrat tutorial](https://optionstrat.com/blog/options-profit-calculator-tutorial)]; `Range` `st_range`; `Auto` keeps `payoff_range`.
   - **Chart** (`charts_risk.payoff_chart`, key `chart_payoff`), one figure, two rows.
     - Row 1, P&L ($) against spot:
       - the expiry curve;
       - T+0 at the what-if date and shifts;
       - up to two dated curves set by `st_dates` pills (⅓ and ⅔ of the front DTE by default, snapped to trading dates, or picked dates). ONE shows several labelled dated curves at once [S: [ONE](https://optionstradingiq.com/using-optionnet-explorer-to-study-options-strategies/)];
       - strikes, breakevens, spot and forward lines;
       - the RN 68% and 95% ranges to the front expiry (UI-13; 2.7 item 6);
       - the straddle-implied move as two `accent` ticks on the x-axis, `straddle ±4.83%` (the 18‑Dec 5,830 straddle at mid, 280.30 [V-me]) [S: [LiveVol earnings analysis](https://optionstradingiq.com/livevol-review/)].
     - Row 2 is the Greek strip: `st_greek` pills (`Δ` · `Γ` · `Vega` · `Θ`) at the T+0 date, so Greeks show for a future date [S: trader-tools mistakes 8].
   - **Table** (UI-9): a date × spot P&L heatmap, key `chart_pnl_table`, drawn per 2.7 item 7.
     - Rows: round spot levels on the chain's strike grid, each with its % change; the spot row is ruled.
     - Columns: trading dates from today to expiry (weekly, then the last 5 days daily), headed `Fri` / `06‑Nov`.
     - Cell values by `st_units`: `$` · `% of max loss` · `% of premium`.
     - On phones the default columns are today, the midpoint and expiry; "All dates" is in a popover [S: OptionStrat, Sensibull].
5. **Blotter.** One view of the legs, replacing both "Legs" and "Legs at inception" [V-audit S2]. `st_blotter` switches `View` · `Edit`.
   - **View**: `st.dataframe` key `tbl_blotter`, `selection_mode="single-row"`.
     - Columns: `On · Qty · Type · Expiry · Strike · Bid · Ask · Mid · IV (%) · IV src · Mid vs fit (vp) · $Δ · Γ P&L 1% · Vega ($/vp) · Θ ($/day)`.
     - Rows: one per leg; for positions with more than one expiry a subtotal row per expiry (`Σ 18‑Dec‑26`, with $Δ, vega and Θ) [V-mm]; then `Total`.
     - `IV src` reads `surface` (BK-3), `override` (help gives the gap to the surface), `extrap.` beyond the quoted wing (SF-8 [P]) or `unlisted`.
     - The action bar above the table (2.8) applies Flip, Duplicate, Roll ▸ and Remove to the selected row.
   - **Edit**: `st.data_editor` key `legs_editor_*`, keeping the existing session-state machinery [V: `tab_strategy._legs_state`].
     - Input columns only: `On`, `Qty` (step 1, `%+d`), `Type`, `Expiry` (selectbox of listed dates with AM/PM), `Strike` (step = interval), `IV override (%)` (blank = surface) and `Entry` (blank = mid).
     - `num_rows="dynamic"`. `Reset legs to template` lives here.
   - **Structure steppers**, one horizontal container of small buttons:
     - `Shift −/+` moves every strike one interval;
     - `Width −/+` moves the wings only, never crossing the body;
     - `Roll ◂ ▸` moves every leg to the previous or next listed expiry, using the dates held in the legs state (2.0);
     - `Flip` negates every quantity;
     - `Δ hedge` opens a hedge ticket (UI-8, needs RK-11). For index presets the instrument is `ES` or `MES`, priced at the forward of its expiry, the 18‑Dec ES at F 5,830.63. For the ETF it is shares. The ticket shows the quantity, rounded so the residual is within ±0.5 contract (or share), the residual $Δ, and "Add hedge leg". On the default condor: −0.07 ES and −0.67 MES, so 0 ES or +1 MES [V-me].

     These are Sensibull's Shift/Width/Hedge and IBKR's Make Delta Neutral [S: [Sensibull strike controls](https://blog.sensibull.com/2025/12/17/introducing-quicker-strike-controls/); [IBKR OptionTrader](https://www.ibkrguides.com/traderworkstation/optiontrader.htm)]. SPX itself cannot be traded, so a stock leg is not a hedge [V-mm, V-pr].
   - `On` excludes a leg from every number without deleting it, as OptionStrat, TradingView and Deribit do [S].
6. **Footer line**: the engine, the dynamics, the entry basis, the probability basis and the treatment of expired legs ("expired legs at intrinsic at the scenario spot").

**Default state.**

- Iron Condor on 18‑Dec‑26 AM: +1 5,300P / −1 5,400P / −1 6,150C / +1 6,250C. The shorts are the listed strikes nearest 16Δ by their own surface delta (−0.150 and +0.147); the wings are 100 points wide [V-me].
- 1 × 100, commission $0, entry at mid.
- What-if zero; Chart view; dated curves at ⅓ and ⅔ of the DTE; Greek strip on Δ; blotter in View.

**Empty state.** With no legs, or every leg off: the tiles show `–`, and the chart area reads "No legs are on. Pick a template or switch a leg on in the blotter."

**Error state.** Blotter validation errors (zero quantity, strike ≤ 0, an expiry after the last listed date under Synthetic) appear one line each under the editor. The last valid analysis stays, faded (2.10).

**Desk conventions.**

- $ Greeks for the whole position; quantities signed; premiums per share in price columns.
- Breakeven % against the current S.
- P(profit) is risk-neutral from the smile unless the Position settings set a view, and the tile says which.
- Θ per calendar day; Risk shows Θ to the next session.

**Removed.**

- Category plus strategy as two selects.
- The three steppers on their own row.
- The `Expected P&L`, `P&L std` and `Drift used` tiles [V-audit S5].
- The RETURN/RISK tile, a retail ratio [V-mm, V-pr].
- The Greek tile row.
- The breakeven prose line and the second legs table.
- The `Download P&L profile` button (moves to the export menu).
- Quantities shown as `1.00`.
- 5Δ wings by default.

**Acceptance tests.**

- Unit (RK-9a), default chain: the listed short strikes are 5,400 (put) and 6,150 (call); no other listed strike's own-vol delta is nearer ∓0.16; the unsnapped strikes solve Δ = ∓0.16 to 1e−8 at their own smile vols.
- AppTest, default [V-me goldens]:
  - the legs are exactly the four above;
  - `NET PREMIUM` reads `$2,240` with `credit`;
  - `PACKAGE` reads `20.80 / 24.00`;
  - `MAX LOSS` reads `−$7,760`;
  - `BREAKEVENS` reads `5,378 / 6,172`;
  - `P(PROFIT)` lies within 1.0 pp of 73.5% and carries `RN smile`.
- AppTest: `tbl_blotter` has `len(legs) + 1` rows, the last named `Total`. Every `Qty` is an integer. The `Total` row's `$Δ` equals the Risk `$Δ` tile to the dollar.
- AppTest: switching a leg off changes `NET PREMIUM` by exactly that leg's premium, and the chart's strike lines drop by one.
- AppTest: `Shift +` moves every strike one interval; `Width +` keeps the body and moves each wing out one interval; after 10 clicks of `Width −` no strikes have crossed.
- AppTest, `?wv=3`:
  - the strip shows `SCENARIO`, and the T+0 legend entry contains `+3.00 vp`;
  - on Pricing, IV and Surface the context line contains `Scenario on Strategy, Risk` and every tile equals its value without `wv`;
  - `whatif_reset` sets `ws`, `wv`, `wd` and `wr` to 0 and removes the note everywhere.
- AppTest, `?ws=-2&wv=3&wd=5`: the date label reads `T+5 · Fri 30‑Oct‑26`, and the same state reproduces after a reload.
- AppTest, Table view: the cell (S, valuation date) equals the T+0 P&L at S (−$24 by default) to $0.01, and the expiry column equals the expiry curve.
- AppTest, Long Call Calendar: the `MAX LOSS` help contains "front expiry", and the blotter has one subtotal row per expiry.
- AppTest: leg edits survive Strategy → Risk → Strategy (P0-1, kept).
- Playwright 1440 × 900, sidebar open: the tiles' top ≤ 320 px, the payoff chart's top ≤ 480 px, and its P&L row's bottom ≤ 900 px (the Greek strip may fall below the fold). Today the chart's top is about 1,100 px [V-audit S1].
- Performance: a what-if step reruns in ≤ 0.5 s (fragment `strategy_analysis`).

### 3.3 Risk (BK-1, RK-1 to RK-8; owner WS-BOOK: `tab_risk.py`, `charts_risk.py`, `risk_cache.py`)

**Job.** Show what the position can lose and why: a pinned headline of risk numbers, then one report at a time.

```
POSITION  Iron Condor · +1 5,300P / −1 5,400P / −1 6,150C / +1 6,250C · 18‑Dec‑26 AM (55d) · 1 × 100 · max loss −$7,760    [what-if strip, shared]
TILES     $Δ −$19,525 −0.07 ES | Γ P&L 1% −$68 | VEGA −$306/vp 1M −$225 | Θ +$41/day BE move 0.77% vs 0.81% | WORST GRID −$7,663 +10% · −10 vp | WORST STRESS −$7,728 Dot‑com
REPORT    [Ladder] [Scenarios] [Stress] [VaR] [Vega] [Explain]
          … one report body …
footer    full revaluation · BSM European legs at surface IVs · sticky strike · Methodology →
```

**Layout, top to bottom.**

1. **Position line.** `st.markdown` with the template name, every leg with its expiry date (the audit's R8), the size and the maximum loss; then the what-if strip, one line while all shifts are zero.
2. **Headline strip (BK-1).** `card_row("risk_tiles", …)`, 6 tiles at the what-if state, from `position_risk`, `weighted_vega`, `scenario_cube` and `run_stress_tests` (RK-1, RK-2 [P]).

   | Tile | Value | Delta slot | Help |
   |---|---|---|---|
   | `$Δ` | `−$19,525` | `−0.07 ES` (index) or shares | Δ in index units (−3.37), the dynamics (`sticky strike`), and the ES and MES conversion |
   | `Γ P&L 1%` | `−$68` | | ½Γ(0.01S)²; ΓS²/100 = −$13,694 |
   | `VEGA ($/VP)` | `−$306` | `1M −$225` (√T-weighted, RK-2) | Raw and weighted definitions |
   | `Θ ($/DAY)` | `+$41` | `BE move 0.77% vs 0.81%` | The breakeven daily move 1% × √(\|Θ\| ÷ \|Γ P&L 1%\|) against the implied daily move σ_ATMF ÷ √365. Both are per calendar day to match Θ [J]. Θ to the next session (Mon 26‑Oct 16:00, 3 calendar days): +$126 [V-me]. |
   | `WORST GRID` | `−$7,663` | `+10% · −10 vp` | The worst cell of the desk 9 × 7 grid today |
   | `WORST STRESS` | `−$7,728` | `Dot‑com 2000–02` | The worst named stress |

   - This is IBKR's pinned Risk Dashboard [S: [IBKR risk dashboard](https://www.ibkrguides.com/traderworkstation/risk-dashboard.htm)], without statistics from synthetic history at eye level [V-mm, V-pr].
   - Every tile is computed in the main run in ≤ 1.0 s; the grid and stress cells are a 63-cell and a 14-scenario BSM revaluation.
3. **Report selector (BK-1).** `st.pills("Report", REPORTS, key="report", bind="query-params", required=True)`, default `Ladder`. It wraps on phones. Only the selected report computes.
4. **Reports.** In each report the table and the chart show the same grid, as Bloomberg OSA does [S: [Xavier guide, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/11_derivativesoptionsfuturesswaps.pdf)].

   | Report | Controls (keys) | Table | Chart | Engine |
   |---|---|---|---|---|
   | **Ladder** | `lad_axis`: `%` · `σ`; `lad_roll`: `Today` · `Next session` · `+1w` · date; `lad_order`: `1st` · `+2nd` | Rows: spot shocks −20, −10, −5, −3, −2, −1, 0, +1, +2, +3, +5, +10, +20 %, or ±0.5/1/1.5/2/3σ. Columns: `Spot · Move · P&L ($) · $Δ · Γ P&L 1% · Vega ($/vp) · Θ ($/day)`, plus `Vanna · Volga · Charm` with `+2nd`. The 0 row is bold and ruled. Default rows: −5% `5,510.00 · −$581 · +$35,573`; 0% `5,800.00 · 0 · −$19,525`; +2% `5,916.00 · −$658 · −$46,565` [V-me]. | P&L against spot through the ladder points, with a metric selector | `scenarios.greek_ladder` (RK-1) |
   | **Scenarios** | `sc_axes`: `Spot × Vol` · `Spot × Date`; `sc_grid`: `Desk 9×7` · `Portfolio margin (FINRA 4210(g) style)`; `sc_volmode`: `vp` · `relative` · `1M-scaled`; `sc_date` | Desk 9×7: spot −10, −7.5, −5, −2.5, 0, +2.5, +5, +7.5, +10 % by vol −10, −5, −2, 0, +2, +5, +10 vp. Spot × Date: round levels by real dates. Today the vol axis is relative and uneven and the dates read "Day 52" [V-audit R3, R4]. The FINRA grid uses ten equidistant points over the product class's range: +6%/−8% for a high-cap broad-based index such as SPX, ±10% for other broad-based indexes, ±15% for equities and narrow-based indexes [S: [FINRA 4210(g)(2)(F)](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210)]. The ETF preset's class is set by the RK-8 owner from the rule text [J]. Under the FINRA grid, one line: "Requirement (illustrative, not a broker's): worst of 10 points −$3,160; minimum $0.375 × 100 per contract (4210(g)(7)(B))" [V-me; S]. | Diverging heatmap with printed values (2.7 item 7), worst cell outlined | `scenarios.scenario_cube` (RK-1); `margin` arrays (RK-8a) |
   | **Stress** | Groups: `Historical` · `Hypothetical` · `Custom` (popover); `IV damped by tenor` as a footer chip | Sorted worst first. Columns: `Scenario · Source · Spot (%) · ATM vol (vp, 1M) · Skew (vp, RK-7) · Rate (bp) · Days · P&L ($)`. `height="content"`. | Horizontal bars by P&L, labels per 2.6 | `run_stress_tests`; RK-7 named shocks |
   | **VaR** | `var_model`: `FHS-GARCH` · `HS` · `Student-t MC` · `GARCH-t MC` · `Normal MC`; confidence 99% (ES 97.5% alongside); horizon 1d; window 500d; paths (MC only); seed in an `Advanced` popover before the results | `st.table` of statistics, right-aligned | Histogram, y-axis `Probability (%)`, with VaR and ES rules and the tail shaded in `loss` | `risk.historical_var` (RK-5), `delta_gamma_var` (RK-6), MC VaR (exists) |
   | **Vega** | `vg_axis`: `Δ` · `Moneyness` | Expiry buckets 7, 14, 30, 60, 91, 182, 365, 730 d by 10Δp · 25Δp · ATM · 25Δc · 10Δc, with row and column totals | Diverging heatmap with values; tiles raw, 1M-weighted and √T-weighted | `vega_buckets`, `weighted_vega` (RK-2) |
   | **Explain** | `ex_to`: `What-if` · a stress scenario · `Next session (sample)`; `ex_method`: `Taylor` · `Shapley` | Taylor terms Δ, Γ, vega, Θ, ρ, vanna, volga, charm, veta and unexplained; or Shapley spot, vol, time and rate; then full revaluation and a quality line comparing the two methods | One waterfall (FT part-to-whole) for the selected method | `attribution.taylor_explain`, `shapley_explain` (RK-3) |

   Report-specific rules:

   - **VaR**:
     - Tiles: `VaR 99% 1D · SYN HIST`, `ES 97.5% · SYN HIST`, `DGT VaR` (the analytic comparator) and `Non-linearity` (VaR − DGT). VaR, ES and DGT VaR are signed P&L quantiles, so they read negative; Non-linearity is their signed difference.
     - The report runs in a fragment placed last, behind a skeleton (2.10), and reads the cached result `risk_cache.var(position_hash, market_hash, config)`.
     - The default model follows RK-5; the old default produced 101 exceptions against 20.1 expected [P].
     - One line links to the backtest in Methodology: "Backtest of this model on 1,000 synthetic days: <zone>, <n> exceptions vs <e> expected (SYN HIST) · Methodology →" (VL-7a).
     - FRTB uses ES at 97.5% [S: [BCBS MAR33](https://www.bis.org/basel_framework/chapter/MAR/33.htm), chapter title confirmed, body loads dynamically].
   - **Explain**: Vola compares greek-based with scenario-based attribution as model validation [S: [Vola PnL](https://www.voladynamics.com/products/vola-pnl)].
   - **Scenarios, apply a cell (UI-10)**:
     - A click on a heatmap cell (the scatter overlay, 2.7 item 7) or a `single-cell` selection in the Scenarios table calls `on_scenario_select`. It copies the cell's spot, vol and date into the what-if strip, and the strip shows SCENARIO.
     - This is IBKR's drill-through [S: [IBKR scenario report](https://www.interactivebrokers.com.hk/php/webhelp/navigator/reportMeasurePricevol.htm)].
     - The keyboard route is the strip itself.
5. **Footer line**: the revaluation engine, the dynamics, the history source and the position count.

**Default state.** The Ladder report at the valuation date with no shifts.

**Empty state.** If `build_position` returns `None` (invalid legs): "The Strategy tab's legs are invalid: <first error>", plus a button that opens Strategy [V: `app.py`].

**Desk conventions.**

- Every $ figure is for the whole position; losses are negative.
- Spot moves in % with levels; vol shocks in vp by default.
- VaR at 99% over 1 day and ES at 97.5%.
- `Next session` is the next trading session's close (Mon 26‑Oct‑26 16:00 from a Friday valuation), not +1 calendar day. Dates are trading dates.

**Removed.**

- The 6 stacked sections on one 5,106 px page [V-audit].
- The time-decay and Greek-profile charts [V-audit R7].
- The `IV damped` checkbox column, the relative uneven vol axis, `Day N` columns, the `Density` y-axis, and the seed input after the results.
- VaR and ES from the pinned tiles.
- The Backtest report (moved to Methodology › Validation) and the Margin report (one line under the FINRA grid).
- The SPAN-like 16 preset: CME SPAN margins futures and futures options, while SPX options clear at OCC [J; V-mm, V-pr].

**Acceptance tests.**

- AppTest: the selector has exactly 6 options. `?report=VaR` computes VaR once, and patched counters on `scenario_cube` (beyond the tile grid), `greek_ladder` and `rolling_backtest` read 0.
- AppTest, default [V-me goldens]:
  - the tiles read `−$19,525`, `−$68`, `−$306`, `+$41`, `−$7,663` and `−$7,728`;
  - the `$Δ` delta slot reads `−0.07 ES`;
  - the Θ delta slot reads `BE move 0.77% vs 0.81%`.
- AppTest: the Ladder's 0% row P&L reads `0`, and its $Δ, Γ P&L 1%, vega and Θ equal the tiles to the dollar (RK-1 acc 1 [P]). With `lad_roll=Next session`, the 0% row's P&L reads `+$126` ± $1.
- AppTest: the Stress table is sorted ascending. Playwright: its height is 28 × (rows + 1) ± 4 px.
- AppTest, Scenarios:
  - the y tick labels end in `vp`; the x tick labels match `^[0-9,]+ \([+−]?[0-9.]+%\)$`; the cell (0 vp, 0%, today) is 0;
  - with the FINRA grid under the index preset, the x-axis has 10 points from −8% to +6%, and the requirement line reads `−$3,160` and contains "illustrative".
- AppTest, VaR: the histogram's y-axis title is `Probability (%)`; both VaR and ES tile labels contain `SYN HIST`; the VaR value starts with `−`.
- Unit: `tab_risk.on_scenario_select` sets `ws`, `wv` and `wd` from both payload shapes (a scatter point, a dataframe cell). Playwright: clicking the (+5%, +2 vp) heatmap cell shows the SCENARIO strip with `Spot +5.0%` and `+2.00 vp`.
- Playwright 1440 × 900: the tiles' top ≤ 260 px; the report body's top ≤ 480 px; page height with the Ladder ≤ 1,800 px (today 5,106 [V-audit]).
- Playwright, dark: every stress-bar label has at least 4.5:1 contrast with its background (today 3.49:1 [V-audit]).
- Performance: a what-if step updates the tiles in ≤ 1.0 s; the VaR report renders in ≤ 2.5 s.

### 3.4 Surface (vol surface) (SF-3 to SF-11; owner WS-SURFACE: `tab_surface.py`, `charts_surface.py`, `surface_state.py`)

**Job.** Turn the chain into a forward-based, arbitrage-checked implied-vol surface. Show the fit against the market quote by quote, and publish it to every other tab.

```
DATA      Synthetic chain · as of 23‑Oct‑26 16:00 ET · 5,020 quoted → 2,363 used (2,510 ITM side · 147 zero bid · 0 wide) · forward: parity · SVI arb‑free · mid   [Filters ▾]
TILES     ATMF 30D 15.38% VIX‑style 16.43% | 25Δ RR 30D −4.21 vp | 25Δ BF 30D +0.23 vp | TERM 30→365D +1.62 vp | FIT 0.03 vp vs ½‑spr 0.09 | ARB 0 / 0
VIEW      [Smile|Quotes|Term|Grid|3D]   ◂ 20‑Nov‑26 AM · 27d ▸   x:[K/F|k|Δ|Strike]   Range[…]   [Marks ▾]
          Smile 20‑Nov‑26 (27d): bid–ask bars, mid ticks, fit line, F line, quoted range, extrapolated dashed; residual panel below
EXPIRIES  Expiry  DTE  F  Carry gap(bp)  ATMF  25ΔRR  25ΔBF  RMSE  In‑band  Used  Conf  Flags   (click = focus expiry)
▸ Arbitrage & forwards (collapsed)
footer    SVI per expiry (Gatheral 2004), Durrleman g ≥ 0, calendar w(k,T) non‑decreasing; wing cap |k| ≤ 1.5 (SF-8) · Methodology →
```

**Layout, top to bottom.**

1. **Data panel (SF-11).** `st.container(border=True, key="surface_data")`, one line while healthy.
   - It shows the sidebar's source read-only, the counts, the forward method, the calibration and the price side.
   - For an upload it also shows the quote time, the age of the quotes at the valuation time, the underlying's time and the gap between the file's S and the parity-implied S. Cboe data is delayed, and Yahoo snapshots are not synchronous [V-mm; J].
   - The `Filters` popover (`key="surface_data_edit"`, `on_change="rerun"`) holds:
     - the uploader: one `st.file_uploader`, the three DT-5 layouts, the kept-file row with `Remove` (exists);
     - filters, with defaults modelled on LSEG IPA's [S: [LSEG IPA ETI](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti)]: OTM only ✓; max spread 4 × the expiry's median; minimum 6 strikes per expiry; minimum maturity 2 d; weeklies on; drop zero bids ✓;
     - forward `sf_fwd`: `Put-call parity` · `Sidebar r, q` (SF-3 [P]);
     - calibration `sf_cal`: `Arbitrage-free` · `Unconstrained` (SF-4 [P]);
     - time `sf_time`: `Calendar` · `Trading`, shown only after SF-9 merges;
     - the wing cap |k| for extrapolation (SF-8).
   - Fixed and documented in Methodology, not toggles: the interpolation (SVI slices; SF-7's Andreasen-Huge is used only where its owner sets it as the default once merged) and the price side (mid).
   - Any non-default setting shows the FIT chip on every tab (2.2).
   - The counts always reconcile: quoted = used + Σ exclusions by reason (institutional pattern 5 [J]).
2. **Headline tiles.** `card_row("surface_tiles", …)`, 6 tiles.

   | Tile | Value | Definition |
   |---|---|---|
   | `ATMF 30D` | `15.38%` | Constant-maturity ATMF vol, interpolated in total variance [S: [Cboe VIX method](https://www.asx.com.au/content/dam/asx/investors/investment-options/vix/cboe-vix-white-paper.pdf)]. Delta slot (UI-26): the model-free 30-day vol by the VIX method, `VIX‑style 16.43%`, 1.05 vp above ATMF on the default chain [V-me `vix_style.py`]. |
   | `25Δ RR 30D` | `−4.21 vp` | σ(25Δc) − σ(25Δp). Help: negative means put skew (FX convention [S: [Reiswich and Wystup](https://ideas.repec.org/p/zbw/cpqfwp/20.html)]). |
   | `25Δ BF 30D` | `+0.23 vp` | ½(σ25Δc + σ25Δp) − σATMF |
   | `TERM 30→365D` | `+1.62 vp` | ATMF 365D (17.00%) − ATMF 30D |
   | `FIT` | `0.03 vp vs ½‑spr 0.09` | Median RMSE against the median half-spread in vol; delta slot `100% in band` [V-me `fit_numbers.py`, before SF-4] |
   | `ARB` | `0 / 0` (illustrative) | Calendar / butterfly violation counts; help "0 / 0: passes" |

   The quote count moved to the data panel, so the row stays at 6 tiles.
3. **Primary view.** Before the expiry table, so that the smile is above the fold.
   - Controls: `sf_view` (`Smile` · `Quotes` · `Term` · `Grid` · `3D`); the focus expiry shown as `◂ 20‑Nov‑26 AM · 27d ▸`, two buttons that step it in `on_click` (the keyboard route); x-axis `sf_x` (`K/F (%)` · `k = ln K/F` · `Call Δ` · `Strike`); `Range`; and `Marks` (UI-23).
   - **Smile** (`charts_surface.smile_fit`, key `chart_smile`; Vola's example charts [S: [Vola examples](https://www.voladynamics.com/examples/can-your-fitter-do-this)]; IBKR Model Navigator [S: [IBKR](https://www.ibkrguides.com/traderworkstation/model-navigator.htm)]):
     - each used quote as a vertical bid-to-ask IV bar (OTM side; calls in `call`, puts in `put`) with the mid as a tick;
     - excluded quotes as hollow `muted` markers; one-sided quotes as a half bar;
     - the fit as one `text`-colour line, dashed and `muted` beyond the last quote out to the wing cap (SF-8);
     - marks, when set, as an `accent` dashed line over the fit;
     - a solid `accent` vertical at F; dotted verticals at the quoted range;
     - inside the plot at the top left: `20‑Nov‑26 · 27d · RMSE 0.05 vp · 100% in band · 140 used` [V-me];
     - row 2 of the same figure (100 px): the residual (σ_mid − σ_fit) ÷ half-spread as bars, with a ±1 band;
     - violations marked at their k with `▲` and `cal` or `bfly`;
     - on the `Call Δ` axis: 0.90 to 0.10 call delta (puts on the left) with a 5Δ grid [S: [IvyDB flyer](https://optionmetrics.com/wp-content/uploads/2024/03/OM_IvyDB-US_Flyer_WEB_REV.pdf); [ORATS](https://orats.com/blog/describing-the-implied-volatility-options-surface)].
   - **Quotes** (replaces the IV tab's chain mode): `st.dataframe` key `tbl_surface_quotes` for the focus expiry.
     - Columns: `Strike · Side · Bid · Ask · IV bid · IV mid · IV ask · Fit · Mid vs fit (vp) · Status · Vendor IV`.
     - `Status` is `used` or the exclusion reason: PR-1's codes as words (`below floor`, `above cap`, `crossed`, `zero bid`) plus the filter reasons (`ITM side`, `wide`, `thin expiry`).
     - `Vendor IV` appears for uploads only, as a diagnostic: DT-5 measured Yahoo's IV more than 1 vp from Cboe's on 45% of OTM SPY quotes [P].
     - Caption: `140 quotes inverted in <n> ms (batch LBR)`.
   - **Term**: two rows in one figure.
     - Row 1: ATMF vol at constant maturities of 7, 14, 30, 60, 91, 182, 365 and 730 d (lines), with listed expiries as markers.
     - Row 2: 25Δ RR and BF by expiry.
     - Expiries whose variance window holds a holiday or extra weekend days are marked (help "variance-time weighting 0.1 on weekends and holidays", from `synthetic.DEFAULT_CLOCK` [V]). This explains, for example, 27‑Nov at 15.25% against 20‑Nov at 15.45% [V-mm].
     - Below, a forward-vol strip as a one-row `st.dataframe`: `7→30`, `30→60`, `60→91`, `91→182`, `182→365` [S: [ORATS forward vols](https://orats.com/docs/definitions); [Moontower](https://moontower.substack.com/p/primer-9-drilldown-a-finer-look-into)].
     - Later overlays: events with the implied event move (UI-25) and realized vol (UI-27).
   - **Grid**: an IV table, rows the constant maturities, columns 10Δp … 50Δ … 10Δc in steps of 5Δ, as a sequential heatmap with printed values (2.7 item 7). `sf_grid_kind` switches `Implied` · `Local`; local vol is blank where undefined.
   - **3D**: one opt-in chart, implied only, with the fixed sequential scale.
4. **Expiry table (SF-6).** `st.dataframe` key `tbl_surface_expiries_{expiry}`, one row per expiry (16 for the default chain [V-me]), with the 2.8 selection pattern. A row click calls `on_expiry_select`, which sets the focus expiry.
   - 12 columns: `Expiry` (`20‑Nov‑26 AM`) · `DTE` · `F` · `Carry gap (bp)` (implied minus sidebar carry; SF-3) · `ATMF (%)` · `25Δ RR (vp)` · `25Δ BF (vp)` · `RMSE (vp)` · `In band (%)` · `Used` · `Conf. (%)` · `Flags`.
   - `Conf. (%)` is the share of the call-delta interval [0.10, 0.90] spanned by used quotes. It adapts ORATS's `confidence` [S: [ORATS definitions](https://orats.com/docs/definitions); J for our definition].
   - `Flags` is text: `carry`, `thin`, `cal`, `bfly`, `extrap`. A carry gap above max(5 bp, half the ATM spread) sets `carry` (SF-3's warning rule [P]).
   - DF, 90–110, normalised skew and the 10Δ columns live in the expander or the Grid view.
5. **Arbitrage and forwards.** `st.expander(key="surface_diag", on_change="rerun")`:
   - the SF-3 table: `F · DF · implied r · implied q · pairs used · residual · SE · gap to sidebar (bp)`;
   - the violation list;
   - Durrleman g(k) for the focus expiry;
   - local-vol coverage;
   - 90–110 skew and Mixon's normalised skew per expiry [S: [Mixon 2011](https://businessweek.ivolatility.com/doc/JOD-MIXON_(2011)_+_Appendix.pdf)];
   - quote-level notes.
6. **Contract readout.** One line: `Pricing contract K 5,815 · 20‑Nov‑26 → σ 15.46% (K/F 100.0%, inside quotes)`, or `… (extrapolated, |k| 0.62)`.
7. **Export menu** (2.13): surface grid CSV and XLSX (exists), used quotes CSV, SVI parameters JSON, each with provenance.
8. **Footer line.**

**Default state.** Synthetic chain at the valuation time, arbitrage-free SVI and parity forwards, always published. Smile view of the focus expiry on the `K/F` axis.

**Empty state.** With `Upload` and no file, the data panel opens with the uploader, the three accepted layouts as one line each, and a `Use synthetic` button. The tiles show `–`.

**Error state.**

- Kept: the malformed-file error names the columns found and needed [V-audit SF7].
- The previous surface stays published, with the source chip `UPLOAD (failed) · last good: synthetic` until the user acts.

**Desk conventions.**

- ATMF and moneyness K/F, as Vola, LSEG and SVI's log-forward moneyness use [S: Vola, LSEG, [Gatheral and Jacquier](https://arxiv.org/abs/1204.0646)].
- Vols in %; RR, BF and skew in vp.
- Expiries as dates with DTE and AM/PM.
- Mid by default.

**Removed.**

- The download-then-upload demo path.
- The parametric SSVI as a source (it moves to Methodology as a worked example).
- Two side-by-side 3-D charts, 3-D local vol, and the colour-scale selectbox.
- The `Grid 30 × 41` tile; "ATM means K = S"; maturities in years; a term table that scrolls in its own box [V-audit SF1, SF2, SF4, SF6].
- The `Publish to all tabs` toggle, because an unpublished surface breaks principle 1 [V-mm].
- The interpolation and price-side toggles.
- The ▲/▼ best/worst marks.

**Acceptance tests.**

- AppTest, default:
  - the data panel contains "Synthetic — not market data";
  - on the noise-free fixture every `ATMF (%)` is within 0.1 vp of `SyntheticChain.expiries.atm_vol` and every `F` within 2e−4 relative (DT-4 acc 3, SF-3 [P]).
- AppTest: no column, axis title or tile label contains "years"; `Expiry` cells are `format.date` plus ` AM` or ` PM`.
- AppTest: `quoted == used + sum(excluded by reason)`; the default chain gives 5,020 quoted; lowering max spread to 1× the median lowers `used`.
- AppTest: the Pricing `σ` tile equals the surface vol at (K, T) to 1e−10, every blotter `IV src` reads `surface` (BK-3), and no widget key contains `publish`.
- AppTest: `sf_cal=Unconstrained` shows the FIT chip on all six tabs and changes no number on Pricing until the fit reruns.
- AppTest: `sf_x=Call Δ` gives the x-axis title `Call delta` and the range 0.90 to 0.10.
- AppTest, Quotes view: every row's `Status` is `used` or in the reason set, and the count of `used` rows equals the expiry table's `Used`.
- Unit: `tab_surface.on_expiry_select` sets `session_state["expiry"]`. Playwright: a row click changes the smile annotation and the Pricing context line's expiry.
- Playwright:
  - the number of bid/ask bars on the smile equals `Used` for that expiry;
  - the extrapolated segment has `dash != "solid"`.
- Playwright 1440 × 900: the tiles' top ≤ 260 px; the smile chart's top ≤ 560 px; page height in the Smile view ≤ 2,000 px (today 3,111 [V-audit]).
- Performance:
  - a cold fit of the default chain ≤ 2.5 s, then cached (`st.cache_resource`);
  - a focus-expiry change ≤ 0.4 s (fragment `surface_view`).

### 3.5 IV (owner WS-PRICING: `tab_iv.py`, `charts_pricing.py`; PR-1, PR-4)

**Job.** Invert one market quote to implied vol, with the no-arbitrage bounds and its distance from the published surface visible. Chain-scale inversion lives in Surface › Quotes.

```
CONTRACT  SPX 20‑Nov‑26 AM 5,815 (from the sidebar)                                         (scenario note when active)
SINGLE    Type [Call|Put]  Exercise [European|American]  Bid [ 98.20 ]  Ask [ 99.00 ]
TILES     IV BID 15.40% | IV MID 15.46% | IV ASK 15.52% | SURFACE 15.46% mid vs fit 0.00 vp | VEGA 6.37 $/vp | BOUNDS inside
CHART     model price vs σ, with bid/mid/ask lines, the IVs marked, the no‑arbitrage band shaded
footer    Let's Be Rational (Jäckel 2015, code revision 1520); American via de‑Americanisation (PR‑4) · Methodology →
```

**Layout.**

1. **Context line**: the contract from the sidebar.
2. **Input row** (horizontal container): `iv_type`, `iv_style`, `iv_bid` and `iv_ask`.
   - Their step is the source's tick: for the index $0.05 below $3 and $0.10 from $3; for the ETF $0.01 (DT-4 [P]; `synthetic._MIN_TICK` [V]).
   - Defaults come from the source chain at the contract. `iv_single` switches to a single `Price` input.
3. **Tiles**: `IV BID`, `IV MID`, `IV ASK`, `SURFACE` (delta slot `mid vs fit` in vp, per 2.5), `VEGA ($/VP)`, and `BOUNDS` (`inside`, or `below floor $4.88`).
4. **Chart** (key `chart_iv_price`): the model price against σ, with lines at bid, mid and ask, vertical lines at the three IVs, and the arbitrage-free price band shaded `muted`. American quotes also show the de-Americanised European price (PR-4 [P]).

**Default state.** The contract's call, with bid and ask from the chain.

**Error state.** A price outside the bounds:

- the affected tiles read `–`, with help naming the bound (`below the lower bound $4.88`);
- the input gets an `st.error` line under it;
- the chart stays, with the quote line drawn outside the band.

Today the cards and the chart disappear and the numbers are raw (`4.87706`) [V-audit I2].

**Desk conventions.** IV in % to 2 dp; mid vs fit in vp; vega $/vp per share; price per share.

**Removed.**

- The "Model check" card (it always equals the typed price [V-audit I1]).
- The 3-decimal IV [V-audit I3] and raw numbers in messages.
- The Expiry chain mode, which duplicated the Pricing chain and the Surface smile (now Surface › Quotes).

**Acceptance tests.**

- AppTest, noise-free fixture: `IV BID ≤ IV MID ≤ IV ASK`, and `SURFACE` lies between `IV BID` and `IV ASK`.
- AppTest, default: the IV tiles read `15.40%`, `15.46%` and `15.52%`, and `SURFACE` reads `15.46%` with `mid vs fit 0.00 vp` [V-me: `implied_volatility` on 98.20, 98.60 and 99.00].
- AppTest, `?preset=textbook`, price 3.00:
  - no exception; three tiles read `–`;
  - the error contains `$4.88` and no number with more than 2 dp;
  - `chart_iv_price` is present.
- AppTest: the tab renders no expiry-chain table; the chain-scale inversion is tested in Surface › Quotes.
- Performance: one inversion ≤ 1 ms server-side (PR-1: 10,000 quotes ≤ 50 ms [P]); the rerun ≤ 0.5 s.

### 3.6 Methodology (DK-6; owner WS-DESK: `tab_methodology.py`; content from each workstream's `methodology/<area>.md`)

**Job.** A reference that proves every number: models, conventions, data policy and validation results, readable in a drawer without leaving the tab in use.

**Layout.**

1. Contents with anchors (in the tab, for links and printing).
2. Collapsed sections in this order:
   - Data and time (Cboe sessions; NYSE holidays; GTH exceptions not modelled);
   - Pricing engines;
   - Implied vol;
   - Surface (including the parametric SSVI worked example, the fixed interpolation and the price side);
   - Risk;
   - **Conventions and units**, generated from `format.REGISTRY` (UI-3), with the time-basis and edge definitions;
   - **Validation**: tables from `validation_results.json` with commit SHA and date (VL-7), now including the VaR backtest (VL-7a: model, exceptions / expected, Kupiec p, Christoffersen p, zone as text, Z2, power at 2%, on 250, 500 and 1,000 synthetic days, tagged `SYN HIST`; Kupiec at 250 days has 24% power [P: RK-4]; zones per MAR32 [S: [BCBS MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm)]);
   - References: each with a year and a DOI, arXiv ID or URL;
   - Known limitations.
3. The drawer: `@st.dialog(position="right", width="large")` [V-probe E5], opened by the header button, Alt+Shift+M and every footer link. `st.dialog` has no scroll or anchor API [V-mm, V-pr], so the drawer renders the open tab's section first, already expanded, followed by the contents. One renderer serves the tab and the drawer.

**Default state.** Every section collapsed except the contents (tab) or the open tab's section (drawer).

**Removed.** The single long page (3,209 px desktop, 6,546 px phone [V-audit]); "no same-day options" in Limitations, which is stale once DT-1/DT-2 land.

**Acceptance tests.**

- DK-6's tests stand [P]: every file renders; no raw `$$`; at most 1,500 px tall at 1440 px and 2,500 px at 390 px when collapsed; the drawer anchored right at full height; links checked nightly.
- AppTest: opened from Risk, the drawer's first expander is labelled "Risk" and is expanded.
- AppTest: the "Conventions and units" rows equal `format.REGISTRY` item for item.
- AppTest: the Validation section holds a VaR backtest table whose zone column is text and whose rows carry `SYN HIST`.

---

## 4. New features, ranked

Every item maps to a plan item, an amendment (`-a` suffix) or a new item. Each names its files and owners (the plan's ownership rules [P]), its acceptance tests and the capability-table rows it relies on (principles-streamlit Part 3; rows marked "No" are never used). Size is my estimate [J]: S under 2 days, M 2 to 5 days, L over 5 days.

### 4.1 Must

| ID | Feature | Plan link | Files (owner) | Capability rows | Size | Acceptance |
|---|---|---|---|---|---|---|
| UI-1 | Tab-owned headline tiles; the global price cards leave the shell | Changes P0-1 step 5; extends DK-5, PR-7 | `app.py` (WS-DESK); `tab_pricing.py` (WS-PRICING) | #1, #31 | S | Playwright: no element keyed `price_cards*` on any tab but Pricing; tab bar top ≤ 170 px at 390 × 844 and ≤ 120 px at 1440 × 900 (DK-5 branch today: 292 to 358 and 173 to 211 [V]) |
| UI-2 | Market-level header: valuation time, underlying, S, source chip, the amber OVERRIDE, FIT and MARKS chips, the per-tab engine badge with validated error; contract items, r and q removed | Extends DK-5, SF-11, DT-2 | `header.py`, `app.py` (WS-DESK); reads `surface_state.current_source()`, `surface_state.state_flags()` (WS-SURFACE) and `model_config.badge(tab)` (WS-PRICING) | #1, #5 | M | AppTest: each of `source=Upload`, `vol_src=Override`, `sf_cal=Unconstrained` changes exactly its chip; the ticker holds exactly the valuation time and `SPX 5,800.00`. Playwright: ≤ 40 px at 1440 px with the sidebar open in the worst state (2.2); ≤ 2 lines at 390 px. |
| DK-4 (promoted) and UI-3 | Number standard (2.5) plus the column registry `components.col(kind, label, help)` and `format.REGISTRY` | DK-4 | `format.py`, `components.py`, `tests/test_format.py`, `tests/test_format_lint.py` (WS-DESK) | #10, #11 | M | A golden table of every kind, including signed losses, `Γ P&L 1%`, ES equivalents and the three edge kinds; the lint fails on `:.Nf`, `%.Nf` or `f"{x:,` in `ui/tab_*.py`; Playwright renders a probe grid and checks each kind against the goldens |
| UI-4 | Shared what-if strip, Strategy and Risk only; semantics in its labels; URL-bound with integer `wd`; Reset; dynamics caption; muted note on other tabs | Uses RK-1 `vol_mode`, RK-7 dynamics; part of BK-1, BK-2 | new `ui/whatif.py`, `tests/test_app_whatif.py` (WS-BOOK); the note via `components.context_line` (WS-DESK) | #25, #28 | M | Section 3.2's what-if tests; Strategy and Risk read identical shifts; Pricing, IV and Surface numbers are unchanged by any shift |
| UI-5 | Stale marking in the tab body after 300 ms; fragments, placeholders and skeletons for heavy elements | Supports P0-1 budgets, DK-7 | `static/desk.css` (WS-DESK); fragments in each tab (owners) | #24, #26, #27 | S | Playwright: during a rerun slowed to 900 ms by a test-only patch, tab-body `[data-stale=true]` elements have opacity ≤ 0.5 at 400 ms and header elements have opacity 1; the VaR report and Engine diagnostics show `stSkeleton` before content |
| UI-6 | Phone pass: toolbars hidden on touch, pinned columns, no horizontal scroll, the invalid modebar value removed | Extends DK-7 | `charts.py`, `static/desk.css`, `tests/e2e/test_style.py` (WS-DESK) | #39 | S | Section 2.11 on all six tabs |
| UI-22 | Primary colour to the link blue in both modes; amber for user state only; input borders 3:1 | Extends DK-1 | both `config.toml` copies, `theme.py`, `static/desk.css`, `tests/test_theme.py` (WS-DESK) | #32 | S | `test_theme`: `primaryColor == linkColor != yellowColor` in both modes, and primary on the background ≥ 4.5:1. Playwright: each input's computed border has ≥ 3:1 against its background in both modes. |
| UI-28 | Provenance inside every export | Extends DT-6, DK-8 | `components.export_*` (WS-DESK); builders in each tab (owners) | #30 | S | Section 2.13 |
| DK-7 (promoted) | Browser regression suite: the Playwright half of every acceptance test here; a selector-allowlist check after upgrades; performance tracking (2.12) | DK-7 | `tests/e2e/*`, `.github/workflows/e2e.yml`, `scripts/screenshots.py` (WS-DESK) | #2, #34, #35 | L | PR CI runs every Playwright test at 1440 × 900 and 390 × 844, dark, on the floor version. Nightly adds light mode, Streamlit 1.65 (until the owner raises the floor) and the performance medians. A deliberately broken `desk.css` selector fails the allowlist check. Headed Chrome and Firefox on Linux run the shortcut tests. |
| PR-7 (promoted) | Pricing and IV in the desk layout: STRADDLE and the pair-gap check; the contract table with Mid vs fit and per-side edges; the profile chart; RN-smile P(ITM); diagnostics with the tree and MC; IV as a single-quote inverter | PR-7 [P: should] | `tab_pricing.py`, `charts_pricing.py`, `tab_iv.py`, `model_config.py` (WS-PRICING) | #10, #14, #19, #31 | L | Sections 3.1 and 3.5 |
| BK-1 | Risk headline strip (6 tiles, no VaR) and report selector (6 reports) | BK-1 | `tab_risk.py`, `charts_risk.py`, `risk_cache.py` (WS-BOOK) | #26, #28, #31 | L | Section 3.3 |
| BK-2 (promoted) | Strategy desk layout: builder bar with shorts by smile delta and wings by points; 6 tiles with the package market; blotter with subtotals and the action bar; dated curves; Greek strip; P(profit) from the smile | BK-2 [P: should] | `tab_strategy.py`, `charts_risk.py` (WS-BOOK) | #15, #16, #22 (within one figure), #31 | L | Section 3.2 |
| BK-3 | Leg IVs from the published surface, with override, `extrap.` and `unlisted` tags | BK-3 | `tab_strategy.py` (WS-BOOK) | #10 | S | Section 3.4's publish test |
| SF-6 and SF-11 | Desk smile, Quotes view, Term, Grid; data panel with counts, filters, quote times and the FIT flags; always-published surface with `cdf(K, expiry)` and `state_flags()` | SF-6, SF-11 | `tab_surface.py`, `charts_surface.py`, `surface_state.py` (WS-SURFACE) | #14, #19, #29 | L | Section 3.4; unit: `cdf` is monotone in K, in [0, 1], and matches 1 + ∂C/∂K ÷ DF of the generator's prices to 0.5 pp on the noise-free fixture |
| DT-2 | Sidebar: presets, one `asof` datetime input, read-only spot under chain sources, focus expiry, ATMF strike, vol source, BSM/ALO model | DT-2 | `inputs.py`, `tests/test_app_inputs.py` (WS-DATA); `model_config.py` (WS-PRICING) | #28, #37 | M | AppTest: `?preset=textbook` reproduces v2's call and put to 1e−12 (10.4506 and 5.5735 at 4 dp; tiles `10.45` and `5.57`); the default `asof` is a cash close; `?days=30` maps to the listed expiry nearest 30 days with a note; under Synthetic no `spot` widget exists; under the index preset American is disabled |
| DT-2a | `clock.session(product, ts)`: Cboe SPX regular, curb and GTH sessions and the status caption; AM last-trading-day rule | Amends DT-2 (DT-1's `clock.py`) | `clock.py`, `tests/test_clock.py` (WS-DATA) | none (core) | S | Unit: Fri 23‑Oct‑26 16:30 ET is `curb`; 21:00 is `closed` with GTH next on Sun 20:15; Mon 10:00 is `regular`; the 20‑Nov‑26 AM expiry's last trading day is Thu 19‑Nov [S: Cboe] |
| DT-5a | `read_chain` returns the file's quote time and underlying quote when the layout carries them (Cboe `Date:` line); a last-trade summary for the yfinance layout | Amends DT-5 | `market_data/adapters.py`, `tests/test_market_data.py` (WS-DATA) | none (core) | S | Unit: `chain_cboe_wide.csv` gives 2026-10-26 10:00 ET and underlying bid 7,914.70 / ask 7,915.30 [V-me fixture]; the yfinance fixture reports "quote time unknown" plus its latest `lastTradeDate` |
| RK-9a | Strike by delta at each strike's own smile vol, snapped to the listed strike nearest the target delta; wings by points | Amends RK-9 [P: should] | `strategies.py`, `tests/test_strategies.py` (WS-RISK) | none (core) | S | Section 3.2's unit test |
| VL-7a | The RK-4 backtest of the reference position's VaR in `validation_results.json`, rendered in Methodology › Validation | Amends VL-7 | `scripts/validation_report.py` (WS-VALIDATION); `tab_methodology.py` (WS-DESK) | #8 | S | Section 3.6 |

### 4.2 Should (in build order)

| ID | Feature | Plan link | Files (owner) | Capability rows | Acceptance |
|---|---|---|---|---|---|
| UI-23 and SF-12 | Trader marks: per-expiry nudges on the SVI slice (ATM level in vp, skew in vp per 10% K/F, put wing, call wing); `Reset to fit`; Durrleman re-check on the marked slice; theo from the marks on every tab; per-side edges appear; the MARKS chip | New; a market maker's core loop [V-mm; S: Vola Fitter] | `vol_surface/marks.py` (WS-SURFACE, SF-12); `tab_surface.py` (WS-SURFACE) | #28 | Unit: zero marks reproduce the fit exactly; an ATM +0.50 vp mark raises σ at K/F 100% by 0.50 vp to 1e−10; a mark that creates butterfly arbitrage is flagged. AppTest: with a mark the Pricing table shows `Buy edge (vp)` and `Sell edge (vp)`, and `?marks=` round-trips. |
| UI-7 | Chain ladder on Pricing with delta window, stride and row click | PR-7 | `tab_pricing.py` (WS-PRICING) | #14 | Section 3.1 chain tests |
| RK-11 | `Leg` gains `exercise` (`european` · `american`) and a `future` kind priced at the forward of its expiry with its own multiplier | New, a dependency of UI-8 | `strategies.py` (WS-RISK) | none (core) | Unit: a future leg's value at S is (F(S) − F₀) × multiplier, its Δ is e^{(r−q)T} × multiplier, it has no gamma or vega, and `position_risk` adds it exactly |
| UI-8 | Structure steppers (Shift, Width, Roll, Flip), the `On` column, the row action bar, and the Δ hedge ticket (ES or MES for the index, shares for the ETF) | BK-2; needs RK-11 | `tab_strategy.py` (WS-BOOK) | #15, #16 | Section 3.2 stepper tests; the hedge leaves \|residual\| ≤ 0.5 contract; the default condor's ticket reads 0 ES or +1 MES |
| RK-12 | American revaluation in Strategy and Risk through a per-expiry ALO price grid (S/K × σ at each scenario date, interpolated), labelled `ALO grid` | New | `strategies.py`, `scenarios.py` (WS-RISK); `risk_cache.py` (WS-BOOK) | none (core) | Unit: the grid matches direct ALO accurate to 1e−4 × K over the scenario range; the 9 × 7 cube for 4 American legs takes ≤ 0.3 s; the Strategy badge reads `ALO grid · American` |
| UI-24 | Undo: a stack of the last 20 legs and what-if states; an Undo button and Alt+Shift+Z; it covers steppers, presets, Reset legs and apply-scenario | New [V-pr] | `tab_strategy.py`, `whatif.py` (WS-BOOK); the shortcut in `app.py` (WS-DESK) | #5 | AppTest: Flip then Undo restores the legs exactly; apply-scenario then Undo restores the shifts; the 21st change drops the oldest |
| UI-27 | Realized vol (typed, or from the synthetic history tagged `SYN HIST`), IV − RV, and the breakeven move against the implied move in the Risk Θ tile and the Term view | New [V-mm] | `tab_risk.py` (WS-BOOK); `charts_surface.py` (WS-SURFACE) | #19 | Unit: on a synthetic path of known vol, RV 20d is within its sampling error; AppTest: the Θ tile's delta slot reads `BE move 0.77% vs 0.81%` by default |
| UI-26 | Model-free 30D vol (Cboe VIX method) beside ATMF 30D | New [V-mm; S: Cboe] | `vol_surface/variance.py`, `tab_surface.py` (WS-SURFACE) | none | Unit: on the default chain 16.43% ± 0.05 [V-me `vix_style.py`]; on a flat-vol chain it equals ATMF to 0.1 vp |
| DT-10 | SPX listing realism: SPXW on every weekday, an SPXW PM expiry beside the AM SPX on the third Friday [S: [Cboe Tue/Thu SPXW](https://ir.cboe.com/news/news-details/2022/Cboe-to-Add-Tuesday-and-Thursday-Expirations-for-SPX-Weeklys-Options-04-13-2022/)]. Today `_schedule` lists SPXW only on non-monthly Fridays [V-mm]. | Amends DT-4 | `market_data/synthetic.py` (WS-DATA) | none (core) | Unit: from Fri 23‑Oct‑26 the first listed SPXW expiry is Mon 26‑Oct; 20‑Nov‑26 lists both SPX AM and SPXW PM; an intraday `asof` on an expiry day shows `0DTE` |
| DT-9 and UI-25 | A synthetic single-stock style with an earnings event (DT-9); an events table with the implied event move from the term structure, √(w(T₂) − w(T₁) − base variance), and the ex-event vol in the Term view (UI-25). Enables the `Single stock with earnings` preset. | New; DT-1's `VarianceClock` already models events [P] | `market_data/synthetic.py` (WS-DATA); `tab_surface.py`, `charts_surface.py` (WS-SURFACE) | #19 | Unit: with one injected event of known size, the implied event move recovers it to 5% relative; AppTest: the preset exists only when DT-9 is present |
| UI-9 | Date × spot P&L table with a unit toggle | BK-2 | `tab_strategy.py`, `charts_risk.py` (WS-BOOK) | #12 | Section 3.2 table test |
| UI-10 | Apply a scenario cell to the what-if strip (scatter overlay or table cell) | BK-1 with RK-1 | `tab_risk.py` (WS-BOOK) | #14 (single-cell), #19 (points) | Section 3.3 drill-through tests |
| UI-11 | Legs, marks, commit SHA and seed in the share link: `legs=P+1@5300:2026-12-18AM;…`, at most 1,500 characters, written at the end of the run and read in `_read_link` | Extends DK-5's Share | `app.py` (WS-DESK); `tab_strategy.encode_legs` / `decode_legs` (WS-BOOK) | #4 | AppTest: an edited 5-leg position round-trips exactly; a malformed `legs` gives a note and the template's legs; a different `v=` shows the commit note |
| UI-12 | Range presets on every spot-axis chart, URL-bound | Extends PR-7, BK-2, SF-6 | `components.range_control` (WS-DESK) | #21 (the workaround), #28 | AppTest: each preset sets `xaxis.range` to S·(1 ± x), or to the RN 95% range from `cdf` |
| UI-13 | RN ranges from the smile with their basis, plus the straddle-implied move | BK-2 | `charts_risk.py` (WS-BOOK) | #22 | AppTest: the legend contains `RN 68% to 18‑Dec` and `(smile)`; the bounds equal `cdf⁻¹(0.1587)` and `cdf⁻¹(0.8413)` (5,492 and 6,167 by default [V-me]) |
| DK-8 (promoted, core only) | Shortcut layer (2.4) and export menus | DK-8 | `components.py`, `app.py` (WS-DESK) | #5, #30 | Playwright, headed Chrome and Firefox on Linux: Alt+Shift+3 opens Risk and writes `?tab=Risk`; Alt+Shift+0 clears the SCENARIO strip; with focus in a number input no shortcut changes its value; the export menu downloads a CSV whose first line starts with `# source:` |
| RK-7 and RK-7a | Smile dynamics selector and named vol-surface shocks; the shadow delta Δ + vega · ∂σ/∂S under sticky moneyness, used by the `$Δ` tile | RK-7 | `surface_risk.py` (WS-RISK); UI in BK items | none (core) | RK-7's own tests; AppTest: sticky moneyness changes the T+0 curve for a spot shift, leaves it unchanged at zero shift, and relabels the delta `Δ (sticky moneyness)` |
| RK-8a | FINRA 4210(g) arrays with the product-class range and the per-contract minimum; the SPAN-like array stays core-only | Amends RK-8 | `margin.py` (WS-RISK); `tab_risk.py` (WS-BOOK) | #12 | Unit: the SPX class gives 10 points from −8% to +6%; equities ±15%; the requirement is max(worst loss, the minimum) |
| DK-6 | Methodology drawer, open tab's section first; one renderer | DK-6 | `tab_methodology.py`, `app.py` (WS-DESK) | #8 | Section 3.6 |
| UI-21 | Measure websocket compression and turn it on if it pays | Supports DK-7 | both `config.toml` copies (WS-DESK) | #38 | Per-rerun bytes before and after recorded in `docs/scorecard`; on if it saves ≥ 40% |

### 4.3 Could

| ID | Feature | Plan link | Files (owner) | Capability rows | Acceptance |
|---|---|---|---|---|---|
| UI-17 | Add legs from the Pricing chain: a single-cell selection on Bid sells 1 and on Ask buys 1 | BK-4 adjacent | `tab_strategy.py`, `tab_pricing.py` | #14 (single-cell) | Unit: the callback with an Ask cell at K adds `+1` at K with entry = ask. Playwright: the click adds the leg. |
| UI-18 | Smile history overlays (T−1d, T−1w) and percentile ranks of ATMF, RR and BF from the RK-4 synthetic history, tagged `SYN HIST` | RK-4 | `tab_surface.py`, `charts_surface.py` (WS-SURFACE) | #19 | AppTest: overlays labelled with dates; percentiles in [0, 100] |
| UI-20 | Small-multiple smiles for all expiries | SF-6 | `charts_surface.py` (WS-SURFACE) | #22 | Each panel 180 px; one y range |
| BK-4 | Position CSV import, expiry ladder, pin risk | BK-4 | `tab_strategy.py` (WS-BOOK) | #16 | BK-4's own tests |
| SF-10 | Quote-level arbitrage with L1 repair | SF-10 | WS-SURFACE | none | SF-10's own tests |
| PR-8 | Heston in Engine diagnostics | PR-8 | WS-PRICING | none | PR-8's own tests |

### 4.4 Feasibility check [J]

- Every item uses only capability rows marked Yes or Partly in the principles-streamlit table.
- No item needs native drag (#18), tree tables (#17), persistent zoom (#21; UI-12 is the workaround), crosshairs across separate charts (#22 is used only within one figure), keyboard selection on charts (#36), or heatmap cell events (the scatter overlay replaces them [V-pr]).
- No test asks AppTest to fire a selection callback or to read a Plotly element: callbacks are unit-tested and wiring is tested in Playwright [V-mm, V-pr].
- No custom JavaScript ships in v3.
- Everything works on Streamlit 1.65. Raising the floor to 1.66 would add `on_change="ignore"` to the legs editor and the segmented controls (1.65's `data_editor.on_change` accepts only a callable [V-me]), plus `theme.paddingTop` and AppTest lookups of tabs by key.

### 4.5 Cut line (proposed, replacing the plan's)

- **Ship:** every plan "must"; the promoted DK-4, DK-7, PR-7 and BK-2; the new musts UI-1 to UI-6, UI-22, UI-28, DT-2a, DT-5a, RK-9a and VL-7a; INT-1 and INT-2.
- **If that does not fit, trim inside the musts in this order.** Each step leaves a correct but plainer app [J]:
  1. PR-7's profile x = Vol and x = Time (keep Spot).
  2. BK-2's dated curves beyond T+0 and expiry.
  3. SF-6's Term and Grid views (Smile and Quotes stay).
  4. BK-1's Explain report.
  5. DK-7's nightly light-mode and 1.65 runs.
- **Never trim:** the context model (2.0), UI-1 to UI-4, UI-28, the number standard, or the acceptance tests of whatever ships.

---

## 5. What we deliberately do not do, and why

1. **No dragging strikes on the chart, and no right-click menus.** Streamlit has no native drag or context menu (capability #18 "No" [V-probe]). Steppers and the row action bar give structure-preserving edits that the keyboard can reach.
2. **No streaming or scraped market data.** Synthetic data and uploads only. Cboe and Yahoo terms forbid automated extraction (DT-6 [P]; capability #40).
3. **No live clock in v3.** `@st.fragment(run_every=60)` redraws only its own elements, so it cannot advance the header or the valuation time, and a new valuation time needs a refit of about 0.8 s every minute [V-pr]. The `Now` button and the static session caption stay.
4. **No 3-D by default, no 3-D local vol, no 3-D P&L.** A 3-D surface has no readable values [S: [AlgoTest](https://algotest.in/blog/opstra-strategy-builder-vs-algotest.md)]; local vol from fitted SVI is noisy [J].
5. **No persisted chart zoom, and no cross-chart crosshair** [V-probe E9, 2.7]. Range presets and shared-axis subplots replace them.
6. **No viewport sniffing in Python.** `st.context` has no viewport width [V-probe 2.8], and the server does not know a plot's pixel width, so label rules count items instead of measuring pixels (2.7 item 3).
7. **No colour without a second cue, and no colouring Greeks by sign** [J; S: WCAG 1.4.1].
8. **No per-user alert thresholds** [S: Signals and Threads].
9. **No probability without its basis, and no N(d₂) on a smile.** Every probability carries `RN smile`, `flat vol` or `view` [S: [TradingQnA thread](https://tradingqna.com/t/sensibull-payoff-graphs-wrong-on-target-date/92019); V-mm].
10. **No auto-refresh.** It is calm, it follows WCAG 2.2.2, and it keeps the end-of-day default cacheable.
11. **No multi-page app** (principles-streamlit 2.4 [J]).
12. **No primary (filled) buttons**: white on the primary blue is about 2.5:1 [V-me formula].
13. **No emoji, and no icons inside alerts** [V-audit G11].
14. **No strategy optimiser** (trader-tools 1.11 [J]).
15. **No broker-grade margin claims, and no SPAN-like array for SPX.** The FINRA-style requirement is labelled illustrative; SPAN is CME's method for futures and futures options [J; V-mm, V-pr].
16. **No second chart library** (principles-streamlit 2.7 [J]).
17. **No Methodology text in captions** (principle 10).
18. **No Tree or Monte Carlo as the app-wide engine** (2.0 engine policy) [V-mm, V-pr].
19. **No VaR or other synthetic-history statistic in a pinned tile**, and no backtest report on the Risk tab (3.3, 3.6).
20. **No "Edge" against the fit**, and no number forced to 0.00 (principle 4).
21. **Out of v3**: the command palette, punctuation hotkeys (they need custom JavaScript), a colour-palette preference (the theme cannot change per session, so charts and widgets would disagree [V-probe 2.9]), and the right-edge table fade (CSS cannot detect overflow inside Glide's canvas [V-pr; J]).

---

## 6. Build order and traceability

**Order [J]**, consistent with the plan's phases [P]:

1. **WS-DESK first**: DK-4 with UI-3, UI-1, UI-2, UI-5, UI-6, UI-22, UI-28 and DK-7. They change files every tab imports.
2. **WS-DATA and WS-RISK core**: DT-2 with DT-2a, DT-5a, RK-9a.
3. **WS-SURFACE**: SF-6 with SF-11, including `cdf` and `state_flags`.
4. **In parallel**: PR-7 with UI-7; BK-3; BK-2 with UI-4, UI-9 and UI-13; BK-1 with UI-10; VL-7a.
5. **Should, in 4.2's order**: UI-23 and SF-12 first, because until marks exist every comparison is "Mid vs fit".

**Audit finding → where this spec fixes it.**

| Audit | Fix |
|---|---|
| G1 price cards everywhere | UI-1, UI-2, 3.1 |
| G2 formats | DK-4, UI-3, 2.5 |
| G3 textbook defaults | DT-2, 2.3 |
| G4 phone toolbars | UI-6, 2.7 item 10 |
| G5 prose | Principle 10 |
| G6 hierarchy | 2.1 |
| G7 contrast | 2.6, UI-22 |
| G8 phone inputs | UI-4, 2.11 |
| G9 stale | UI-5 |
| G10 silent link | 2.4 |
| G11 icon leak | 2.10 |
| P1 two deltas | Principle 1, 3.1 |
| P2 academic Greeks | 3.1 item 6 |
| P3 hidden chain | UI-7 |
| P4 weak card | STRADDLE and the pair-gap check (3.1) |
| P5 rainbow paths | Removed from diagnostics |
| P6 convergence | 3.1 item 7 |
| P7 cut tables | 2.8 |
| S1 chart below the fold | 3.2 |
| S2 two leg tables | Blotter |
| S3 Greek units | 2.5, blotter Total |
| S4 heavy chart | 2.7 items 4 to 6 |
| S5 drift card | 3.2 Removed |
| R1 no strip | BK-1 |
| R2 blank rows | 2.8 |
| R3 and R4 grid axes | 3.3 Scenarios |
| R5 stress chart | 3.3 Stress, 2.6 |
| R6 VaR trust | 3.3 VaR (SYN HIST, signed, backtest in Methodology) |
| SF1 to SF6 | 3.4 |
| I1 to I4 | 3.5 |

**Open questions for the owner.**

1. Accept UI-1 and UI-2, which move the price cards into Pricing and the contract items out of the header, reversing P0-1 step 5 and part of DK-5?
2. Raise the Streamlit floor to 1.66? Recommended; until then DK-7 runs 1.65 nightly.
3. Default preset: index (synthetic) as proposed, or textbook?
4. Is `legs=` in the share link acceptable at up to 1,500 characters (UI-11)?
5. Should the 36 items without full text get their DETAIL and ACC written back (VL-9 or INT-2)? This spec can seed them.
6. Promote PR-7 and BK-2 to must, and adopt the cut line in 4.5?
7. Change `primaryColor` from amber to the link blue (UI-22, DK-1's file)?
8. Limit the global model control to BSM and ALO, moving the tree and Monte Carlo to diagnostics (PR-3 and PR-6 owners)?

---

## Appendix A. Review decisions

Every review point not listed here was applied as written. Each line below is a point I rejected, wholly or in part, with the reason.

1. **MM, header contents ("front F, r, q" in the global header): partly rejected.** Under parity forwards the carry is per expiry, so one r and q in the header would be a third carry source; F appears in Pricing's FORWARD tile and the sidebar's computed line. The SCENARIO chip went into the what-if strip (the principal review's option), because the shifts apply only on Strategy and Risk.
2. **MM, clock: took the option "label it cash close 16:00" rather than a 16:15 default.** The index level is fixed at 16:00, and the verified example numbers stay reproducible; the caption names the Cboe session.
3. **MM, signs ("one way to write a negative"): partly rejected for `st.dataframe` number columns.** They keep the printf hyphen-minus, because NumberColumn formats cannot emit U+2212 and text columns would lose right alignment and sorting [V-probe]. Parentheses are gone everywhere.
4. **MM, Δ-neutral hedge "with ES (or SPY)": SPY not offered for the index preset.** The SPX preset has no SPY quote, and an ETF hedge adds dividend and tracking basis the synthetic data cannot represent; ES and MES cover small and large sizes.
5. **MM, "futures or forward delta as the default for index presets": partly rejected.** The per-option Δ stays the spot convention, because RK-9's strike-by-delta and RK-1's Greeks use it and the difference at the defaults is e^{(r−q)T} ≈ 0.5%. The Greeks table adds `Δ fwd`, and the position delta is shown in ES contracts, where the hedge convention matters.
6. **MM, implied daily move as σ/√252: used σ/√365.** Θ is per calendar day (RK-1), so the breakeven move and the implied move must use the same day count.
7. **MM, trader marks as a core gap: applied as the first Should (UI-23, SF-12) rather than a Must.** It needs a new core item and the Must set already exceeds the plan's cut line. Until it lands every comparison is labelled "Mid vs fit", so nothing claims an edge.
8. **MM, fold the IV tab's chain into the Pricing chain, or fold IV into Pricing: rejected in favour of the principal review's Surface › Quotes view.** The Pricing chain already has 11 columns; per-side IVs and exclusion reasons belong with quote filtering. The single-quote inverter is a distinct job, and `?tab=IV` is part of the URL contract.
9. **MM, phones (drop the per-tab tile budgets): partly rejected.** The per-tab tile budgets are dropped, but UI-1's tab-bar-top bound at 390 px stays: it is one number, and it is the audit's largest phone finding.
10. **MM, DK-7 ("set 1.66 as the floor"): kept as a recommendation (open question 2).** The config documents 1.65 as the oldest supported release, so the floor is the owner's call; until then 1.65 runs nightly, not on every PR.
11. **Principal, MODEL CHECK replacement ("keep ALO vs LR 2,001 for ALO, and SE for MC" in the tile): partly rejected.** The ALO check moved to the CALL/PUT tile help and Engine diagnostics, because the sixth tile is now STRADDLE and Monte Carlo is no longer a global engine. The market check (the pair-implied forward gap) is in the FORWARD tile.
12. **Principal, undo on Alt+Z: used Alt+Shift+Z.** All shell shortcuts use Alt+Shift (2.4), and the reason for avoiding bare Alt applies to letters as well as digits on some layouts [J].
13. **Principal, "the upload should set and lock the valuation time": set, not lock.** Desks deliberately value an uploaded close at a later time (next-morning risk), so `asof` stays editable and the gap is flagged in the data panel [J].
14. **Principal, a WS-RISK item for `Leg` expiry dates and settlement: partly rejected.** Dates and AM/PM are held in BK-2's legs state and converted with `clock.time_to_expiry` on each run, which keeps the core's year-fraction API. Only the exercise style and the `future` kind need RK-11.
15. **Principal, stale cue "dim only the containers downstream of the changed input": partly rejected.** CSS cannot know which input changed, so the rule dims the whole tab body after 300 ms and never the header or sidebar.
16. **Principal, `asof` from two bound widgets via hand-written query sync: solved differently.** One `st.datetime_input` with `bind` exists on 1.65.0 and 1.66.1 [V-me], so there is one widget and one parameter.
17. **MM and principal, Risk tiles: MM's "maximum loss" tile went to the position line.** The six-tile cap (principal) holds, and the six pinned tiles are the Greeks, the worst grid cell and the worst stress, as MM listed.
