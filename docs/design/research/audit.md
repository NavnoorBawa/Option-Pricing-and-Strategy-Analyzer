# Visual audit: Option-Pricing-and-Strategy-Analyzer

I reviewed the integration branch `claude/laughing-mendel-5j6iya` at commit `0e762c3`, as a head of trading-tools UX would. Date: 2026-10-26.

The audit asks one question: would a Citadel Securities market maker or a volatility PM trust this screen and work fast on it? The header and Methodology are still being built on another branch (DK-5, DK-6), so I judged them lightly.

## How to read this

Every screenshot is a PNG under `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/`. Each one is cited by its full path.

File names follow `<state>_<tab>_<d|m>_<dark|light>`:
- `d` means 1440x900 and `m` means 390x844 (phone emulation with touch).
- `__fold.png` is the first screen.
- `__full_NN.png` is the full page, cut into 1100 px tiles on desktop and 1300 px tiles on phone.

Each claim is marked with one of three labels:
- **[measured]**: I measured it in the browser during this audit.
- **[source]**: a cited external fact.
- **[judgement]**: my opinion as a desk-tools designer.

## Method

- **Export.** `git archive claude/laughing-mendel-5j6iya` was unpacked into `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/app`. The main checkout was not touched; `git status` is clean.
- **Server.** `/usr/local/bin/python -m streamlit run main.py --server.headless true --server.port 8791` ran from that folder (Streamlit 1.66.1). I stopped it by PID afterwards: 12091 was the server and 12089 its shell wrapper.
- **Browser.** Playwright 1.64.1 with `executable_path=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome`.
  - Every state was captured at 1440x900 and at 390x844, in dark and in light (`color_scheme`).
  - For the full page, the viewport was grown to the height of `stMain` and then clipped into tiles.
  - Detail shots used DPR 2.
- **States covered:**
  - all 6 tabs with default inputs (24 combinations);
  - an iron condor on Strategy and Risk (`?tab=…&strategy=Iron+Condor`);
  - an American Leisen-Reimer tree on Pricing and IV (`?model=Binomial+tree&style=American`);
  - Monte Carlo on Pricing (`?model=Monte+Carlo`);
  - the Surface tab: empty state, the sample-prices CSV, the sample-IV CSV, and a malformed CSV;
  - the IV tab: a price of 12.50, and a price of 3.00, which is below the no-arbitrage floor;
  - the phone sidebar opened.
- **Volume.** 222 PNGs in total. The driver scripts are in `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/tools/` (`shoot.py`, `detail.py`, `rerun2.py`).
- **Two runs I discarded:**
  - The first American and Monte Carlo pass used `model=binomial`, which the app silently ignores (finding G10). I deleted it and re-ran into `shots_models/`.
  - The Surface pass in `shots/` failed on a selector, so I re-ran it into `shots_surface/`.

## Key measurements [measured]

| Metric | Desktop 1440 | Phone 390 |
|---|---|---|
| Tab bar top / first KPI top | 303 px / 174 px | 483 px / 239 px (502 px with the American model) |
| Tab content starts at | ~350 px | ~530 px of an 844 px screen (63% of the first screen is header plus price cards) |
| Page height: Pricing / Strategy / Risk | 1,573 / 2,108 (condor 2,232) / 5,106 px | 1,908 / 3,180 (condor 3,324) / 6,322 px |
| Page height: Surface / IV / Methodology | 3,111 / 1,233 / 3,209 px | 4,525 / 1,778 / 6,546 px |
| Cold load to idle (default tabs) | 1.0-3.6 s (Surface 3.4-3.6 s) | 0.9-3.4 s |
| Deep link to Iron Condor, Strategy tab | 4.1 s | 3.2-4.3 s |
| American Pricing load | 1.5-3.6 s | up to 5.1 s |
| Spot 100→105: until the KPI values change | Pricing 0.35-0.68 s; Strategy 0.35-0.55 s; Risk 0.36-0.50 s (VaR settles at 0.90-1.03 s); American 0.67-0.87 s | n/a |
| Tab click until idle | 0.7-1.4 s (Surface slowest) | n/a |
| Type sizes | tab labels, metric labels and captions are all 12.25 px; KPI values are 22.75 px | same |

Two more measurements matter for trust:
- **Stale numbers.** During a Risk rerun, my probe saw no running indicator and no stale styling. Old numbers stay on screen at full opacity for up to about 1 s. Playwright `evaluate` calls were also blocked from 0.1 s to 1.08 s, which suggests the browser main thread was busy; that last part is an inference.
- **Dark-mode bar-label contrast** in the stress chart. I read the colours from the SVG and computed the WCAG ratios. All are below the 4.5:1 that WCAG 1.4.3 requires for small text [source: https://www.w3.org/TR/WCAG22/#contrast-minimum]:
  - white on `#EF5350`: 3.49:1;
  - `#444` on `#26A69A`: 3.25:1;
  - white on `#26A69A` would be 3.0:1.
  - Light mode passes: white on `#0B7F73` is 4.89:1 and white on `#D32F2F` is 4.98:1.
- **Phone overlays.** On touch devices, the Plotly modebar and the `st.dataframe` toolbar are always visible (opacity 1).

Raw data:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/detail.json`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_rerun/timing_rerun2.json`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/timing_default.json`

---

## Global and shell findings (ranked)

### G1. The price cards for the sidebar contract sit above every tab, including tabs where they are the wrong subject. Severity: high.

The four cards are Call, Put, Forward and Call ITM probability. They describe one option. Yet they appear above Strategy (a multi-leg position), Risk, Surface and Methodology.

[measured] They push the tab bar to 303 px on desktop and 483 px on a phone. On a phone, 63% of the first screen goes to the header and these cards before any tab content.

[judgement] A trader reads the first card row as "my position". On Strategy and Risk it is not.

Note: the plan's P0-1 deliberately keeps the cards outside the tabs. DK-5 shrinks the header but does not change which subject the cards describe.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_m_dark__full_00.png`

### G2. Numbers are formatted inconsistently across and within tabs. Severity: high.

[measured, from screenshots] Examples:
- **Delta has four formats:**
  - 6 decimals in the Pricing Greeks table (`0.636831`);
  - 4 decimals in the tree Greeks;
  - shares in the legs table (`63.68`);
  - a card reading `18.72` labelled "Delta (shares)".
- **Implied vol** is `20.000%` and `25.433%` (3 decimals) on IV, but 2 decimals everywhere else. The Surface "Fit RMSE" column uses 3 decimals while its neighbours use 2.
- **Stress bar labels** mix 3 significant figures (`−$211`) with cents (`−$72.26`).
- **Stress table "Spot change"** shows `−20.5%`, `15%` and `−58%` in one column. "Rate Δ (bp)" shows `+0`.
- **Signs:** "Max loss $441.05" has no sign, while "Expected P&L −$22.64" is signed. The legs "Cost" column is signed the opposite way to the "debit/credit" chip.
- **Quantities:** the legs editor shows `1.00` / `-1.00`; the legs table shows `1` / `-1`.
- **The IV error message** prints `price 3 … 4.87706`, directly under a caption that says `$4.8771`.
- **The VaR statistics table** left-aligns numbers because they are text.

[judgement] This is the single biggest "amateur" signal for a desk user. DK-4 in the plan is the fix; it should become a must.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_strategy_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_03.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_error_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_04.png`

### G3. The defaults look like a textbook calculator, not a desk. Severity: high.

[measured] The defaults are S = K = 100, 365 days, 20% flat vol, r = 5% and q = 0. The iron condor therefore opens at 365 DTE with strikes 80/90/110/120. The Surface tab opens on a parametric SSVI toy.

The DT-4 synthetic chain is not wired into any tab: `grep` finds no `synthetic`, `adapters` or `clock` import under `ui/`. Neither is DT-1's valuation clock. There is no valuation date or time, and expiries are a day count, not a date.

[judgement] A vol PM would expect:
- an underlying with a skewed synthetic chain;
- 30-45 DTE;
- strikes picked by delta (RK-9);
- expiry dates as dates.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_00.png`

### G4. On phones, toolbars cover chart and table content. Severity: high on mobile.

[measured] On touch devices the Plotly modebar is always visible. The config uses `displayModeBar: "hover"`, but Plotly's documentation only describes `true`, `false` and the hover-only default [source: https://plotly.com/javascript/configuration-options/]. The modebar sits on top of chart titles, legends and annotations:
- the payoff chart's "spot" label;
- the convergence-chart legends;
- the histogram's "outcomes below the plotted range" text;
- the titles of both 3D surfaces;
- the IV chart legend.

The `st.dataframe` toolbar is also always visible on a phone. It covers the text of the element above it: the "Legs —" header, the "Log-normal density" legend, the "ATM local vol" legend and the VaR axis title.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_phone_payoff_dark.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_m_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_m_dark__full_04.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_m_dark__full_02.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_m_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_iv_m_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_strategy_m_dark__full_00.png`

### G5. Prose captions under almost every element. Severity: medium-high.

[measured] Examples:
- the Monte Carlo caption is 8 lines at 1440 px;
- the spot × date heatmap caption is 3 lines;
- the IV tab opens with a 2-line explanation of how the widget keeps its state;
- Risk has 7 explanatory paragraphs;
- "Every scenario … re-prices with Black-Scholes-Merton, whatever the sidebar pricing model" is repeated on Strategy and Risk.

[judgement] Desk tools keep definitions in tooltips or a reference drawer, and keep one terse caveat line per section. The prose makes the app read like a thesis and doubles the scroll length.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_01.png`

### G6. Navigation and labels have almost no hierarchy. Severity: medium.

[measured] The tab labels are 12.25 px plain text, the same size as captions and metric labels. Only an amber underline and colour mark the active tab.

[judgement] The primary navigation is the least prominent text on the page. Section headers, metric labels and captions all compete at about 12-14 px.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_price_cards_dark.png`

### G7. In dark mode, the bar labels in the stress chart fail contrast. Severity: medium.

[measured] See the contrast figures under "Key measurements": 3.49:1 and 3.25:1, against the 4.5:1 that WCAG 1.4.3 requires [source: https://www.w3.org/TR/WCAG22/#contrast-minimum]. Light mode passes.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_stress_dark.png`
- for comparison: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_stress_light.png`

### G8. Phone input flow. Severity: medium on mobile.

[measured] On a phone, the contract inputs live only in a sidebar drawer that covers the screen (six steppers of about 74 px each). You cannot see any result while editing, and the contract is not shown anywhere on the main page. DK-5's ticker will help.

[judgement] Changing spot or strike on a phone takes three taps and a context switch.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/sidebar_open_m_dark__fold.png`

### G9. Stale results are not marked. Severity: medium.

[measured] After a spot change, the KPI values change 0.35-0.87 s later, and the VaR settles about 1.0 s later. My probe saw no "running" indicator and no fading in that window (see Key measurements).

[judgement] A trader cannot tell a stale number from a fresh one. A small "recalculating" state on the KPI strip would fix this.

### G10. Bad link values for the model are silently ignored. Severity: low.

[measured] `?model=binomial&style=American` opens Black-Scholes-Merton with no note. A bad `tab=` value does get a note (in `app._read_link`).

Evidence:
- The log of the discarded run, `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shoot_scen.log`. Its line `american_pricing_d_dark … mainScrollHeight 1573` matches the Black-Scholes page exactly. I deleted that run's screenshots so they could not be mistaken for the American model.
- For comparison, the page from a correct link: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__fold.png`. That page is 2,694 px tall.

### G11. A Material icon's name leaks into alert text. Severity: low.

[measured] The innerText of the surface info alert starts with `my_location`, the ligature name of the icon. [judgement] A screen reader may read it aloud.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_00.png`

### Header (judged lightly; DK-5 in progress)

The 44 px title plus the feature-list caption costs about 130 px and says nothing about the contract.

The sidebar footer ("Built by …", "Educational tool") is fine for a portfolio app. [judgement] On a desk tool it belongs in the About menu.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_pricing_d_dark__fold.png`

---

## Pricing tab (ranked)

### P1. The page shows two different put deltas when the model is American. Severity: high, a trust killer.

[measured] With the American LR tree selected:
- the headline put is **$6.0898** (American);
- the Greeks table directly below is European Black-Scholes-Merton, with put **Δ −0.363169**;
- the tree Greeks, put **Δ −0.4111**, are about 1,000 px further down under "Model diagnostics";
- the sensitivity chart is also European, so at spot it reads about $5.57 against the $6.09 headline.

A caption explains this, but the hierarchy shows the wrong numbers first.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__full_01.png`

### P2. The Greeks table is academic, not desk units. Severity: high.

[measured] Values have 6 decimals. The "Unit" and "Definition" (`∂V/∂S`) columns take about 45% of the width. Everything is per share, with no contract or position view.

[judgement] A market maker reads:
- delta in shares or as 63.7;
- gamma per 1% move;
- vega in $ per vol point per contract;
- theta in $ per day.

Show those, with a ×multiplier toggle. Move the definitions into tooltips.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_pricing_d_dark__fold.png`

### P3. The most desk-native view is hidden. Severity: medium-high.

[measured] "Theoretical option chain" and "Price heatmaps" are collapsed expanders. The chain uses flat vol, 9 strikes (80-120%), and 4-decimal values.

With the default Black-Scholes-Merton model, the bottom of the page is an info box saying "switch the sidebar model".

[judgement] A strike ladder (calls | strike | puts with value, Δ, IV and vega) is the home screen of every options desk. It should be visible, not folded.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_pricing_d_dark__full_01.png`

### P4. The fourth headline card shows a low-value number. Severity: medium.

The card is "Call ITM probability 55.96%" (N(d₂), risk-neutral), and the caption under the cards repeats it.

[judgement] Delta, vega or the at-the-money straddle are worth more in that slot.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_pricing_d_dark__fold.png`

### P5. The Monte Carlo sample paths are rainbow spaghetti. Severity: medium.

[measured] 60 paths cycle through the 10-colour categorical palette, and the "strike" label collides with the paths at the right edge.

[judgement] Use one muted hue at low alpha, add 5/50/95% quantile bands, and draw the forward on top.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_d_light__full_01.png`

### P6. The convergence charts hide the thing they diagnose. Severity: medium.

[measured] The LR line sits on top of the dashed "Converged" reference. The "Converged 10.4506" label overlaps the series. The log x-axis has no tick at the sidebar's 501 steps (the last tick is 200).

[judgement] Plot |error| against steps on log-log axes, with the expected slope, and report the error at the selected step count.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__full_01.png`

### P7. Tables on phones are cut with no affordance. Severity: medium on mobile.

[measured] On phones:
- the Greeks "Unit" column is cut at the right edge;
- the tree-Greeks columns are cut;
- the Monte Carlo table is cut after "95% CI high".

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_pricing_m_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_m_dark__full_02.png`

### P8. Sidebar model controls wrap awkwardly. Severity: low.

[measured] The horizontal "Lattice" radio wraps into two rows with a gap. "Control variate (discounted spot)" wraps. The steps slider runs 10-2000 with a floating value label.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/mc_pricing_d_dark__fold.png`

### P9. The early-exercise boundary chart is a good diagnostic. Severity: low.

[judgement] It is solid. Two small gaps: the x-axis starts at about day 20, and the chart itself carries no "put" label; the caption carries it instead.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_pricing_d_dark__full_02.png`

---

## Strategy tab (ranked)

### S1. The outputs sit below the inputs, and the payoff chart is below the fold. Severity: high.

[measured] The P&L chart starts at about 1,100 px on desktop (after the global cards, selectors, size inputs, the legs editor and two card rows) and at about 2,000 px on a phone.

[judgement] The payoff and the P&L-by-date view are the reason this tab exists, so they belong in the first screen.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_strategy_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_m_dark__full_01.png`

### S2. Two tables describe the same legs, and neither is a blotter. Severity: high.

[measured] The editable "Legs" grid has Type, Quantity (`1.00`), Strike, Days, IV, and Premium shown as "auto". The read-only "Legs at inception" table repeats the legs with Model value, Cost, Δ, Γ, θ and vega.

Neither table has expiry dates, bid/ask/mid, leg IV from a surface, or per-leg dollar Greeks.

[judgement] Merge them into one blotter: editable inputs on the left, computed columns on the right, and a totals row.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_01.png`

### S3. Position Greeks are missing units or use the wrong ones. Severity: high.

[measured]
- "Gamma −0.9519" has no unit; it is shares per $1.
- "Delta (shares) −4.35".
- "Vega / vol pt −$19.04".
- In the legs table, "Delta 63.68" and "Gamma 1.8762" are in shares, but the headers do not say so.

[judgement] Desk units are $Δ, Γ per 1% (shares and $), vega in $ per vol point, and θ in $ per day. A trader at Citadel would ask "gamma per what?" within seconds. RK-1 provides these values.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_00.png`

### S4. The P&L chart is heavy and under-labelled. Severity: medium-high.

[measured]
- A red loss fill runs the full width under zero for the condor.
- A grey ±1σ band overlaps both the profit and loss fills.
- The strike lines are unlabelled dotted lines.
- The legend sits below the plot, while the house template says top-left.
- The y-axis ticks are bare numbers.
- The "Today" curve is orange dashed, close to the loss colour.

[judgement] Make these changes:
- lighter fills, or only the expiry line coloured by sign;
- strike labels on the axis;
- a marker for today's P&L at spot;
- a price-slice table beside the chart.

thinkorswim's Risk Profile pairs the chart with a table of P&L and Greeks at chosen price offsets (%, $ or σ) [source: https://toslc.thinkorswim.com/center/howToTos/thinkManual/Analyze]. OptionStrat shows P&L by date and price as a table [source: https://optionstrat.com/blog/options-profit-calculator-tutorial].

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_light__full_01.png`

### S5. Risk-neutral "expectations" are presented as headline KPIs. Severity: medium.

[measured] The cards read "Expected P&L at expiry −$22.64", "P&L standard deviation $446.54" and "Drift used 5.00%". The last one spends a whole card on an assumption. On phones it is an orphan card on its own row.

[judgement] Put the drift in a chip on the probability card. Default to showing probability of profit with a clear "risk-neutral" tag.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_m_dark__full_02.png`

### S6. Breakevens and reward/risk are rendered as a prose line. Severity: low.

[measured] The line between the two card rows reads "Breakevens at expiry: $85.58 (−14.4%) and $114.42 (+14.4%) · Reward / risk: 0.79 : 1".

[judgement] Make them cards, or annotate them on the chart.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_strategy_d_dark__full_00.png`

### S7. Position-size inputs take a full row of steppers. Severity: low.

[measured] Contracts, Shares per contract and Commission each get a large stepper, followed by an "Assumptions" popover.

[judgement] Fold them into the blotter header, as "1 × 100 @ $0.00".

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_strategy_d_dark__full_00.png`

---

## Risk tab (ranked)

### R1. There is no headline risk strip, and VaR is about 4,000 px down. Severity: high.

[measured] The page is 5,106 px long on desktop and 6,322 px on a phone. The order is:
1. spot × IV grid;
2. spot × date grid;
3. time decay;
4. Greek profiles;
5. stress tests;
6. VaR and ES.

The first number a risk manager wants is at the bottom.

[source] IBKR Risk Navigator puts a risk dashboard across the top (net liquidation value, P&L, margin, VaR) above the detail views [https://www.interactivebrokers.co.uk/en/software/tws.bak/usersguidebook/priceriskanalytics/create_a_what-if_portfolio.htm; summary via IBKR Campus]. BK-1 in the plan addresses this.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_03.png`

### R2. The stress table shows two blank rows. Severity: high, it looks unfinished.

[measured] The table height is coded as `35·(n+1)+3` px in `tab_risk._stress_tests`, but rows render at 31 px. Fourteen scenarios therefore leave 62 px of empty rows.

The "IV damped" column also uses disabled checkboxes, which look clickable.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_03.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_m_dark__full_03.png`

Streamlit supports `height="content"` and an explicit `row_height` [source: https://docs.streamlit.io/develop/api-reference/data/st.dataframe].

### R3. The spot × IV grid uses non-desk axes. Severity: medium-high.

[measured]
- **IV shocks are relative multipliers** at −50, −25, 0, +25, +50 and +100%, which are unevenly spaced: the last step is twice the others.
- **Spot rows are ±30% in 5% steps.** That is coarse: a condor with ±10% shorts has only 3 rows inside its tent.
- **No absolute levels.** No spot levels and no $ units appear in the cells.

[judgement] Desks shock vol in absolute vol points (±1, 2, 5, 10), or as 1M-equivalent shocks (RK-2). They want round spot levels plus %, and a finer grid near the money.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_heatmap_dark.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_d_dark__full_00.png`

### R4. The spot × date grid has odd prices and placeholder dates. Severity: medium.

[measured] The rows are σ steps, so the levels are 149.18, 139.56, 130.56 … 67.03. The columns are labelled "Day 52, Day 104 …".

[judgement] Use round levels or % moves, and calendar dates (or DTE).

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_d_dark__full_01.png`

### R5. The stress chart has several problems. Severity: medium.

[measured]
- **Label precision is mixed** (see G2).
- **Dark-mode labels fail contrast** (see G7).
- **Order:** the bars follow definition order, not P&L, and the worst scenario is not highlighted.
- **Phone:** the labels overflow or clip (`+$19…`, `+$190` at the edge), and the category labels take more than half the width.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/detail/zoom_stress_dark.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_m_dark__full_02.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_m_dark__full_03.png`

### R6. The VaR section's presentation and defaults undermine trust. Severity: medium-high.

[measured]
- **Histogram.** The y-axis is "Density", and the loss tail is barely visible. The "N outcomes below the plotted range" annotation floats above the plot and is clipped on phones.
- **Statistics table.** Numbers are left-aligned.
- **Seed control.** The "Random seed" input sits after the results.
- **Defaults.** Returns are Normal and "Simulate implied-vol moves" is off.

[source] The plan's RK-5 records that the current defaults gave 101 VaR exceptions against 20.1 expected on 10 years of S&P 500 / VIX data. That figure is from the project's plan, not from my own test.

[judgement] The headline VaR must state its horizon, confidence and model, for example "1d 99% VaR, Normal, no vol moves". The default should be the honest model.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_d_dark__full_04.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_d_dark__full_04.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_risk_m_dark__full_04.png`

### R7. The time-decay and Greek-profile charts are adequate but redundant. Severity: low.

[judgement] They repeat information from the grids and carry no strike markers. They belong behind a report selector (BK-1).

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/condor_risk_d_dark__full_02.png`

### R8. Good: the position context line

"Analysing Iron Condor — Long 1 × 80 Put, Short 1 × 90 Put, …" is the right idea. It lacks expiry dates.

---

## Surface tab (ranked)

### SF1. The demo path is download-then-upload of a toy file, and the default view is a toy. Severity: high.

[measured]
- The default view is a parametric SSVI surface driven by 6 sliders.
- "Market quotes" asks you to download a sample and upload it again. The sample has 104 mid prices on a 70-130 strike grid in 5-point steps, with no bid/ask.
- The DT-4 synthetic chain and the DT-5 upload adapters exist but are not used.

[judgement] The tab should open on a fitted synthetic chain with bid/ask, labelled "Synthetic — not market data". Uploads should be secondary.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_empty_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_00.png`

### SF2. The axes and "ATM" are spot-based. Severity: high for a vol PM.

[measured]
- Smiles and the 3D surfaces use moneyness K/S.
- Maturities are years (`0.082`, `0.164`, …).
- The table says "ATM means K = S".
- The skew is defined as 90-110 moneyness.

With r = 5%, the 2-year forward is about 10.5% above spot, so the "ATM" line is far from at-the-money-forward.

[source] The traditional desk skew measure is the 25-delta risk reversal [https://www.cmegroup.com/education/courses/introduction-to-cvol/introduction-to-cvol-skew.html]. SVI is parameterised in log-moneyness relative to the forward (Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces", arXiv:1204.0646 [https://arxiv.org/abs/1204.0646]).

[judgement] Offer delta (10/25Δ put, ATMF, 25/10Δ call) and ln(K/F) axes, with tenors as dates or DTE.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_prices_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_prices_d_dark__full_02.png`

### SF3. The surface does not feed pricing. Severity: high for trust.

[measured] The info box reads: "the surface's implied vol is 23.65% (sidebar volatility 20.00%)". Nothing in Pricing, Strategy or Risk uses the surface, and there is no "use surface vol" action.

[judgement] Showing two vols with no link between them invites the question "which one is the price on?". SF-11 and BK-3 in the plan fix this.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_00.png`

### SF4. Two 3D surfaces dominate the page. Severity: medium.

[measured] Each 3D chart is about 600 px tall with large empty margins. On phones they stack, and the Surface page reaches 4,525 px. A full-width (1,037 px) selectbox chooses a cosmetic colour scale.

[judgement] Keep one 3D view behind a toggle. Lead with a tenor × delta heatmap, the smiles, and an ATM / RR / BF term-structure panel.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_01.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_m_light__full_01.png`

### SF5. The smile fit cannot be judged against the market. Severity: medium.

[measured] Quotes are drawn as circles with no bid/ask bars and no residual panel, and the legend has 9 entries.

[judgement] Draw bid/ask vol bars, fit residuals in units of half-spread, and a per-expiry "inside bid/ask %" (SF-4 plans this metric).

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_prices_d_dark__full_01.png`

### SF6. The KPIs and the table mix dev metrics with desk metrics. Severity: low-medium.

[measured]
- "Grid 30 × 41" is a headline card, which is a developer metric.
- The arbitrage counts are good, but they are styled the same as everything else.
- The term-structure table scrolls inside its own frame (30 rows in a 10-row box), which nests scrolling.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_00.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_surface_d_dark__full_02.png`

### SF7. Good: the error and empty states

The malformed-CSV error names the columns it found and the columns it needs, and the parse notes ("Read strike from 'Strike' …") are transparent. These are good. A minor issue: the uploaded file name is truncated ("sample_...prices.csv").

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_garbage_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_surface/surface_ivs_d_dark__fold.png`

---

## IV tab (ranked)

### I1. It inverts one price for the sidebar contract only. Severity: medium-high.

[measured] There is no bid/ask input and no chain-level inversion, although PR-1's batch API exists. The IV result card has the same weight as "Model check", which always equals the input price, and "Time value".

[judgement] A market maker inverts bid, mid and ask across a strip. Show the bid/mid/ask vols, the spread in vol points, vega, and the vol change against the surface.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_calc_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots_models/american_iv_d_dark__fold.png`

### I2. The error state discards context and formats numbers raw. Severity: medium.

[measured]
- The message prints the price and the bound unformatted (`3`, `4.87706`).
- The cards and the chart disappear.
- The input is not marked invalid.

[judgement] Keep the chart and show the price line below the floor. Format the bound as `$4.8771`.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_error_d_dark__fold.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_error_m_dark__fold.png`

### I3. The IV precision is inconsistent. Severity: low.

[measured] The result shows 3 decimals (`25.433%`); see G2. The chip "+5.43 vol pts vs sidebar" is good.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_calc_d_dark__fold.png`

### I4. On phones, the chart is clipped and covered. Severity: low-medium.

[measured] The y-axis title is clipped to "Option valu", and the modebar covers the legend.

Screenshot: `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/iv_calc_m_light__full_00.png`

---

## Methodology (judged lightly; DK-6 in progress)

- [measured] It is a wall of text, 3,209 px on desktop and 6,546 px on a phone, with paragraphs over 20 lines long.
- The References section has no DOIs or URLs.
- "Limitations" says "whole-day expiries of at least one day (no same-day options)". [judgement] That will be stale once the DT-1 clock is wired in.

Screenshots:
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_methodology_d_dark__full_02.png`
- `/tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/audit/shots/default_methodology_m_dark__fold.png`

---

## The 10 changes that would most improve the app (ranked)

Most of these are [judgement]. I note the plan item each one maps to, or the gap it fills.

1. **Give each tab its own headline row; drop the global price cards from tabs where they are the wrong subject (fixes G1, R1, S1).**
   - Pricing: price, Δ, Γ, vega and θ from the selected engine.
   - Strategy: net premium, max profit and loss, breakevens, $Δ, Γ per 1%, vega and θ.
   - Risk: VaR and ES with horizon, confidence and model; worst stress; net $Greeks.
   - Pair it with the DK-5 contract ticker.
   - Target: content starts at 200 px or less on desktop and 330 px or less on a phone.
   - This changes the P0-1 decision to keep the cards outside the tabs. It extends DK-5 and BK-1.
2. **Make every number on Pricing come from the selected model (fixes P1).**
   - The tree, ALO or Monte Carlo Greeks become the main table.
   - Black-Scholes-Merton becomes a reference column, never the default view when another model is selected.
   - Every table and chart names its engine.
   - This extends PR-3 and PR-7.
3. **Adopt and enforce the desk number-format standard (fixes G2, I2, I3, S3, R2).**
   - Fixed decimals per quantity type.
   - Signed P&L everywhere.
   - Explicit units in headers (per share, per contract, position).
   - Vol in vol points with 2 decimals.
   - Right-aligned numbers; no `+0`; no unformatted numbers in messages.
   - Add a browser test that greps the rendered text for violations.
   - This is DK-4; promote it from "should" to "must".
4. **Strategy: outputs first, with one blotter (fixes S1, S2, S4).**
   - Put the payoff chart and a price-slice table (P&L, Δ, Γ, θ, vega at chosen spot offsets, as in the thinkorswim Risk Profile) in the first screen.
   - One legs blotter: editable inputs, computed per-leg dollar Greeks, a totals row, and expiry dates.
   - Label strikes on the chart, lighten the fills, and mark today's P&L at spot.
   - This is BK-2.
5. **Risk console restructure (fixes R1-R6).**
   - Headline strip and a report selector, with sections collapsed or tabbed.
   - Vol shocks in absolute vol points or 1M-equivalent.
   - Round spot levels with a finer grid near the money; dates, not "Day 104".
   - Sort the stress chart by P&L and highlight the worst scenario.
   - Fix the blank table rows: use `height="content"` or the correct row height.
   - Default to the honest VaR model, and label horizon, confidence and model on the number.
   - This is BK-1, RK-2 and RK-5.
6. **Desk-realistic defaults and live synthetic data (fixes G3, SF1).**
   - Wire the DT-4 chain and the DT-1 clock into the app.
   - Valuation date and expiry dates; 30-45 DTE default.
   - Strikes by delta (RK-9).
   - The Surface tab opens on a fitted synthetic chain with bid/ask; leg IVs come from it (BK-3).
   - "Synthetic — not market data" is visible.
7. **Vol-surface conventions (fixes SF2, SF4, SF5, SF3).**
   - Delta and ln(K/F) axes; tenor labels as dates or DTE; an at-the-money-forward line.
   - Bid/ask vol bars and a residual panel.
   - A 25Δ RR and BF term structure.
   - 3D behind a toggle.
   - Publish the fitted surface to Pricing, Strategy and Risk with a visible "vol source" badge.
   - This is SF-3, SF-6 and SF-11.
8. **A phone pass (fixes G4, G8, P7, R5).**
   - Turn the Plotly modebar off on touch, or move it into a menu.
   - Stop the dataframe toolbar overlapping the element above.
   - Tables collapse to priority columns, or to cards.
   - Stress labels stay inside the plot area.
   - A compact inline contract editor instead of the full-screen drawer.
   - The browser regression in DK-7 should assert "no toolbar overlaps text".
9. **Chart house-style fixes (fixes G7, P5, P6, R6).**
   - Dark-mode bar labels at 4.5:1 or better: darker fills or labels outside the bars.
   - Monte Carlo paths in a single hue with quantile bands.
   - Convergence shown as an error plot on log-log axes.
   - VaR histogram showing probability, with a loss-tail zoom.
   - Annotations inside the plot area, never floating text.
   - Legends top-left, as the house template says.
   - Owned by WS-DESK (charts.py / DK-3).
10. **Cut the prose, and add trust signals for speed (fixes G5, G9, G6).**
    - One-line captions at most. Definitions go to tooltips and the Methodology drawer (DK-6).
    - Larger, stronger tab labels.
    - A visible "recalculating" state on the KPI strip during reruns of about 0.4-1.0 s, so stale numbers are never shown as current.
    - Keep the measured budgets: under 1 s for a spot change on Risk, and under 3.5 s cold load, per the P0-1 acceptance criteria.

## Sources

- WCAG 2.2, contrast minimum: https://www.w3.org/TR/WCAG22/#contrast-minimum
- Plotly.js configuration (modebar display): https://plotly.com/javascript/configuration-options/
- Streamlit `st.dataframe` (`height`, `row_height`): https://docs.streamlit.io/develop/api-reference/data/st.dataframe
- CME Group, introduction to CVOL skew (25-delta risk reversal as the traditional measure): https://www.cmegroup.com/education/courses/introduction-to-cvol/introduction-to-cvol-skew.html
- Gatheral & Jacquier, Arbitrage-free SVI volatility surfaces: https://arxiv.org/abs/1204.0646
- thinkorswim Analyze and Risk Profile (price slices): https://toslc.thinkorswim.com/center/howToTos/thinkManual/Analyze
- OptionStrat profit calculator (P&L by date and price): https://optionstrat.com/blog/options-profit-calculator-tutorial
- IBKR Risk Navigator, what-if portfolios and risk dashboard: https://www.interactivebrokers.co.uk/en/software/tws.bak/usersguidebook/priceriskanalytics/create_a_what-if_portfolio.htm
- The project's plan of record (RK-5 VaR exception figures; P0-1, BK-1, BK-2, DK-4 to DK-7, SF-3/4/6/11): /home/user/Option-Pricing-and-Strategy-Analyzer/docs/roadmap/plan-items.md
