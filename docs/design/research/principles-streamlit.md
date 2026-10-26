# Design principles for a dense options desk, and what Streamlit 1.65 / 1.66 can do

Research task `principles-streamlit`. Written 2026-10-26.

How to read the tags:
- **[S]**: a sourced fact. The URL is given.
- **[V]**: checked by me against the installed packages (Streamlit 1.66.1 at /usr/local/bin/python, and the 1.65.0 venv at scratchpad/shell/venv165) or in a browser experiment (Playwright with Chromium, scripts under `principles-streamlit/scripts/`).
- **[J]**: my own judgement.

Nothing under the repository was modified.

Files written:
- this file;
- `dense-table-dark.png` (dataframe and Plotly probe in the project theme);
- `drawer-right.png` (st.dialog with position="right");
- the probe apps (`exp/`, `exp2/`, `exp3/`) and drivers (`scripts/probe*.py`, `scripts/sigs.py`, `scripts/api.py`).

---

## Part 1. Design principles for dense, professional financial interfaces

### 1.1 The quality bar, restated as testable properties [J]

"Precise, fast, dense, calm, trustworthy" maps to properties that can each be checked:

| Property | Test it can be held to | Main sources |
|---|---|---|
| Precise | Every number has a unit, sign convention and stated precision. Columns use one precision each. No `-0.00`. | Few pitfall #2/#3; typography refs below |
| Fast | A rerun after an input change settles in under 1 s; tab switches and selections land in about 0.1-0.4 s. | Nielsen 0.1/1/10 s |
| Dense | High value per pixel and per second: compact rows (24-32 px), no decorative chrome, small multiples. | Tufte; Ström; Carbon row sizes |
| Calm | No motion or colour without meaning. Colour is reserved for P&L sign, alerts and series identity. Nothing auto-refreshes unless it must, and that can be paused. | Few #11/#12; WCAG 2.2.2 |
| Trustworthy | Provenance (synthetic or uploaded), as-of time, model and engine shown. Extrapolated, flagged and unreliable values are marked as such, never silently blended in. | Few #2; plan items SF-8, PR-6, DT-6 |

### 1.2 Tufte: data-ink, chartjunk, small multiples, sparklines

- **[S]** Tufte's *The Visual Display of Quantitative Information* (1983; 2nd ed. 2001) covers the data-ink ratio, small multiples and high-resolution displays: https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/
- **[S]** Ström's essay on density quotes him: "Every bit of ink on a graphic requires reason" and "Graphics can be shrunk way down": https://mattstromawn.com/writing/ui-density/

**Implications [J]:**
- Remove gridlines that carry no reading, chart titles that repeat the tab name, legends where direct labels fit, and borders around every card.
- Use small multiples for "the same chart across expiries": smile per expiry, P&L at several horizons.
- Sparklines belong inside KPI tiles and tables (`st.metric(chart_data=…)`, `LineChartColumn`), not as full charts.

### 1.3 Few: what a dashboard is, and the 13 pitfalls

- **[S]** Few, *Common Pitfalls in Dashboard Design* (Perceptual Edge, February 2006), https://www.perceptualedge.com/articles/Whitepapers/Common_Pitfalls.pdf. His definition: "a visual display of the most important information needed to achieve one or more objectives, consolidated and arranged on a single screen so the information can be monitored at a glance." Users must be able to (1) see the big picture, (2) focus on what needs attention, and (3) drill into what is needed to act.
- **[S]** The 13 pitfalls (same PDF):
  1. exceeding a single screen;
  2. inadequate context;
  3. excessive detail or precision;
  4. indirect measures;
  5. inappropriate display media;
  6. meaningless variety;
  7. poorly designed media;
  8. inaccurate quantitative encoding;
  9. poor arrangement;
  10. ineffective highlighting;
  11. useless decoration;
  12. misused or overused colour;
  13. an unappealing display.

**Mapping to this tool [J]:**
- **#1:** Each tab's headline (price tiles, risk strip) must fit above the fold at 1440x900. DK-5's targets (KPI top under 200 px, tab bar at most 260 px) are the right kind of acceptance test.
- **#2:** Every Greek states its scaling (per 1 vol pt, per day, per 1%) in the label or in `help`. A value always comes with its comparator: VaR against the expected exception count; fit RMSE against the bid/ask half-spread.
- **#3:** Probe finding [V]: passing a pandas Styler to `st.dataframe` makes every column without a `column_config` format show 6 decimals ("12.100000" in `dense-table-dark.png`). Any Styler use needs a format for every numeric column.
- **#8:** Diverging scales centred exactly at zero (Plotly `zmid=0`). Area fills under P&L curves start at zero.
- **#12:** See 1.6.

### 1.4 Density versus clutter

- **[S]** Ström defines UI density as "the value a user gets from the interface divided by the time and space the interface occupies". He separates visual, information, design and temporal density. He cites the Bloomberg Terminal as dense both visually and temporally, because data loads almost instantly: https://mattstromawn.com/writing/ui-density/
- **[S]** IBM Carbon data-table row sizes are xs 24 px, sm 32 px, md 40 px, lg 48 px and xl 64 px. "The column header row should always match the row size of the table." The small toolbar (32 px) goes with sm/xs rows: https://carbondesignsystem.com/components/data-table/style/
- **[J]** Dense is not cramped. The tool should use:
  - 24-28 px rows for chains and ladders (`st.dataframe(row_height=26)` works, see the probe screenshot), 32 px for editable legs;
  - one type size for numbers (14 px at baseFontSize 14), plus one larger size only for headline values;
  - grouping by proximity and alignment rather than boxes.

  Temporal density matters as much as pixels: a 9.5 s page load (the P0-4 baseline) costs more density than any layout choice.

### 1.5 Choosing the chart: FT Visual Vocabulary

**[S]** The FT's categories are deviation, correlation, ranking, distribution, change over time, part-to-whole, magnitude, spatial and flow: https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary

**Mapping to the desk's views [J]:**

| Desk view | FT category | Recommended form |
|---|---|---|
| P&L vs spot (today, at expiry, at horizons) | Deviation (from 0) | Line with profit/loss fill split at 0. Spot, strikes and breakevens as muted reference lines. |
| Scenario grid spot x vol (x days) | Deviation, 2-D | Diverging heatmap centred at 0 with cell labels. Small multiples over days. |
| Greek ladder | Deviation + magnitude | Table with signed numbers. Optional in-cell bars (`BarChartColumn` or Styler bars). |
| Smile per expiry; fit vs quotes | Correlation | Bid/ask as vertical ranges, mid as dots, fit as a line. Extrapolated part dashed (SF-8). |
| Term structure (ATM vol, forwards) | Change over "time" (tenor) | Line with markers at quoted expiries only. |
| VaR/ES distribution | Distribution | Histogram with VaR and ES rules and the tail shaded. |
| VaR backtest exceptions | Change over time | P&L series with the VaR line and exception markers. Traffic-light zone shown as text, not only colour. |
| P&L explain (Taylor/Shapley) | Part-to-whole | Waterfall (FT lists it under part-to-whole). |
| Vega buckets expiry x strike | Magnitude, 2-D | Diverging heatmap (signed) or a table with signed numbers. |

### 1.6 Colour: semantic, colour-blind-safe, never alone

- **[S]** The Bloomberg UX team estimates about 20,000 Terminal users have colour vision deficiency (CVD). It notes that "red and green are key visual indicators" with positive and negative meaning in Western conventions. The Terminal offers alternate schemes for deuteranopia and protanomaly via `PDFU COLORS <GO>`. Sources: https://www.techatbloomberg.com/blog/designing-the-terminal-for-color-accessibility/ (it returned 403 to my fetcher; content seen through a search summary) and https://www.bloomberg.com/ux/accessibility/
- **[S]** Okabe and Ito, Color Universal Design: don't convey information through colour alone; label directly on the graph; use thick lines and large symbols; avoid colours that differ only in hue at similar brightness; check designs with colour-blind people, not only simulators. https://jfly.uni-koeln.de/color/
- **[S]** WCAG 2.2:
  - SC 1.4.1 Use of Color: colour is not the only visual means of conveying information (https://www.w3.org/TR/WCAG22/#use-of-color);
  - SC 1.4.11 Non-text Contrast: 3:1 for graphical objects and UI component boundaries needed to understand them (https://www.w3.org/TR/WCAG22/#non-text-contrast).
- **[S]** IBM Carbon:
  - apply the categorical palette strictly in sequence, which maximises contrast between neighbours;
  - in dark themes the lightest sequential colour marks the largest values;
  - "Never use a gradient in place of a sequential palette";
  - the alert palette maps red to danger, orange to serious warning, yellow to warning and green to success.

  https://carbondesignsystem.com/data-visualization/color-palettes/
- **[S]** Market colour convention is regional. In mainland China and Taiwan red means up and green means down. Evidence for Japan and Korea is mixed. Sources:
  - https://www.benzinga.com/general/education/21/10/23258001/did-you-know-you-dont-want-to-be-in-the-green-if-youre-trading-in-china-japan-or-taiwan
  - the EU data-visualisation guide on colour connotations: https://data.europa.eu/apps/data-visualisation-guide/colour-connotations
- **[V]** Where the project stands:
  - `theme.py` and `config.toml` already separate P&L colours (teal #26A69A / red #EF5350 in dark) from call/put identity colours (blue/orange) and the accent (violet).
  - The comment in `config.toml` says the palettes are checked under simulated protanopia and deuteranopia.
  - Contrast against the dark background (#0B0F16), computed by me: text 15.9:1, profit 6.4:1, loss 5.5:1, call 5.3:1, put 4.9:1, amber primary 8.9:1, muted grey #8B95A7 6.4:1, grey #5B6577 3.3:1 (fine for graphics, too low for small text).
  - The borders #232C3A (dark) and #D5DBE3 (light) are 1.36:1. That is a possible SC 1.4.11 issue for input boundaries, which should be audited [J].
- **[J] Recommendations:**
  - (a) Keep teal/red as the default. Always pair it with the sign (`+`/`−`), accounting parentheses in money columns, and position on the deviation axis.
  - (b) Offer a "Colours: standard / blue-orange (CVD) / red-up" setting, like Bloomberg's PDFU COLORS. It is feasible because P&L colours are app-side: `theme.SEMANTIC`, Plotly traces, Styler cells, and `st.metric(delta_color=<named colour>)` chosen per sign. Theme tokens such as `greenColor` cannot change per session (see 2.9).
  - (c) Never colour Greeks by sign: a negative vega is not "bad".
  - (d) Use the alert hues only for state (stale, flagged, extrapolated, breach), never for decoration.

### 1.7 Numbers: tabular figures, true minus, alignment, precision

- **[S]** CSS `font-variant-numeric: tabular-nums` selects the OpenType `tnum` feature (equal-width digits): https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric. Butterick recommends tabular figures for tables: https://practicaltypography.com/alternate-figures.html
- **[V]** I checked the self-hosted IBM Plex Sans and Mono woff2 files with fontTools:
  - every digit is 600/1000 em wide, so digits are tabular by default and no `tnum` is needed;
  - U+2212 MINUS SIGN is present and also 600 units wide, as is `+`;
  - there is a `zero` feature (slashed zero) but no `pnum`.

  This matters because `st.dataframe` draws on a canvas (Glide Data Grid), where CSS `font-variant-numeric` cannot apply. A font with tabular default digits is the only way to get aligned figures there. The Streamlit frontend bundle contains no `tabular-nums` at all [V].
- **[V]** `st.column_config.NumberColumn(format=…)` accepts:
  - printf through sprintf-js (`"%+.3f"` gives `+0.920`, verified);
  - `"accounting"` (gives `(1,234.50)`, verified);
  - `"percent"`, `"dollar"`, `"compact"`, `"scientific"` and `"engineering"`.

  It prints the ASCII hyphen-minus, not U+2212 (the metric delta also showed `-$12.30`). In a right-aligned numeric column that is acceptable: the digits align on the right edge and only the sign column differs [J]. Where the true minus matters (tiles, the header ticker, `st.table`, Markdown), use pre-formatted strings from `format.py`, which the project already does.
- **[J] Desk precision standard** (with Few #3: show the precision the decision needs):

| Field | Display | Notes |
|---|---|---|
| Spot, strike | 2 dp (strike 1 dp when every strike in the chain is a multiple of 0.5) | |
| Option price | 2 dp; tick-aware (0.05 below $3, 0.10 above, per DT-4) | |
| Implied vol | % with 2 dp (vol points) | Differences in vol pts, never relative % |
| Delta | 3 dp signed, or 0-100 "delta" in strike pickers (25Δ) | |
| Gamma | 3 significant digits; $Γ per 1% stated | |
| Vega / theta / rho | $ with 2 dp per 1 vol pt / per calendar day / per 1% | The scaling goes in the column header |
| Second-order Greeks | 3 significant digits | `format.greek` already does this |
| P&L | $ accounting style in columns; whole dollars at position level beyond $1,000 | |
| Rates, carry | % with 2-3 dp; differences in bp | |
| Moneyness | k = ln(K/F) to 3 dp, or delta buckets | |

- **[J]** Rules:
  - right-align numbers and their headers;
  - one precision per column;
  - units in headers, not in cells;
  - abbreviated values (`$1.23M`) always carry the exact value in `help`;
  - no negative zero;
  - use `placeholder="–"` for missing values (an `st.dataframe` parameter [V]), never "None" or "nan".

### 1.8 Dark-mode trading UI conventions

- **[S, partly from memory]** Material Design's dark-theme guidance recommends a dark grey surface (#121212) rather than pure black, lighter surfaces for elevation, and desaturated accent colours: https://m2.material.io/design/color/dark-theme.html. The page did not render for my fetcher, so I have not re-checked its exact numbers. White on #121212 is 18.7:1 (my computation).
- **[J]** Desk convention, as seen on Bloomberg, TradingView and broker platforms; this is not a cited standard:
  - near-black, slightly blue background (the project's #0B0F16);
  - off-white text (#E6EAF2, not #FFFFFF, to reduce halation);
  - hairline separators;
  - one warm primary (amber, Bloomberg's heritage) used sparingly;
  - saturated colour only on data.
- **[J]** Keep light mode first-class for daylight use and printing. The theme already has both modes.
- **[S]** The bundled Streamlit theme guide says: keep `secondaryBackgroundColor` distinct (lighter on dark themes); give `primaryColor` enough contrast for white button text; aim for WCAG AA 4.5:1 for body text. (Installed package: `streamlit/.agents/skills/developing-with-streamlit/references/theme.md`.)
- **[V]** The project avoids primary buttons because Streamlit puts white text on the amber (2.16:1). That is the right workaround.

### 1.9 Progressive disclosure

- **[S]** Nielsen (2006), https://www.nngroup.com/articles/progressive-disclosure/:
  - progressive disclosure "defers advanced or rarely used features to a secondary screen";
  - two levels is the practical maximum;
  - the primary level must hold everything frequently needed;
  - the path to the secondary level must be obvious and its label must set expectations.
- **[J] Mapping:**
  - **Level 1**, always visible: header ticker, headline tiles, the main chart and table of the tab.
  - **Level 2**, one click, never deeper:
    - methodology and validation in a right drawer (`st.dialog(position="right")`, DK-6);
    - chart options and units in a popover;
    - diagnostics (fit residuals, arbitrage flags, MC convergence) in expanders that are rendered only when open;
    - per-leg detail in a drawer opened from a `ButtonColumn`.

  No popover inside a drawer inside an expander.

### 1.10 Empty, loading, stale and error states

- **[S]** Kaplan (NN/g, 2021), https://www.nngroup.com/articles/empty-state-interface-design/:
  - an empty state communicates system status, increases learnability and provides a direct pathway to filling the space;
  - never leave the area blank;
  - never show "no records" while still loading.
- **[S]** Nielsen's limits (1993), https://www.nngroup.com/articles/response-times-3-important-limits/:
  - 0.1 s feels instantaneous;
  - at 1 s the train of thought is kept, but the system should show it is working;
  - past 10 s, show percent done and a way to interrupt.
- **[S]** Bundled Streamlit performance guide (installed `references/performance.md`):
  - during a rerun the previous elements go stale (faded to about 33% opacity after a delay);
  - render stable chrome first and reserve slots (`st.container()` plus `container.skeleton()`) before slow work.
- **[J] Recipes for this tool:**
  - **Empty:**
    - no uploaded chain: the Surface tab shows the synthetic demo with its "Synthetic — not market data" label and an "Upload a chain" action;
    - a strategy with no legs shows a one-line explanation and the template picker.
  - **Loading:** skeleton in the chart slot for anything over 300 ms. Use `st.status` for multi-step work (fit, then arbitrage checks, then local vol) and `st.progress` for MC paths.
  - **Unreliable:** PR-6's `reliable=False` renders the MC price with "–" and the reason, not as a number.
  - **Errors:**
    - put the message beside the input it concerns, in plain words, naming the field and the valid range;
    - keep the last good result on screen, marked stale;
    - `client.showErrorDetails = "type"` is already set.

### 1.11 Accessibility (WCAG 2.2 AA) checklist for a desk tool

**[S]** WCAG 2.2, https://www.w3.org/TR/WCAG22/. The relevant success criteria, with [J] notes:
- 1.4.3 Contrast (Minimum): 4.5:1 for text, 3:1 for large text.
- 1.4.11 Non-text Contrast: 3:1 for chart lines and marks against the background, and for component boundaries.
- 1.4.1 Use of Color.
- 1.4.10 Reflow at 320 CSS px: tables may scroll in two dimensions, but the page must not.
- 1.4.13 Content on Hover or Focus: Plotly tooltips must be dismissable and hoverable. This is Plotly's behaviour, not something we control.
- 2.1.1 Keyboard: everything operable by keyboard.
- 2.4.7 Focus Visible and 2.4.11 Focus Not Obscured: a sticky header must not cover focused elements.
- 2.5.8 Target Size (Minimum): 24x24 CSS px. Streamlit 1.65 fixed toolbar buttons to a 24 px minimum [S, release notes].
- 2.2.2 Pause, Stop, Hide: auto-updating information that starts automatically and lasts more than 5 s needs a pause mechanism. This applies to a ticking market clock: give it a pause, or make it update only on interaction.
- 4.1.3 Status Messages: toasts and status changes must be announced.
- 1.1.1 Non-text content: charts need a text alternative. Use `alt=` on `st.plotly_chart` and `st.dataframe` (1.65+ [V]) and give a data-table equivalent for key charts.

**[V]** Streamlit 1.66 adds a "Skip to main content" link and `<main>`/`<nav>` landmarks (I confirmed `<main>` and `stSkipToContent` in the DOM). Plotly charts are not keyboard-operable for selection [J], so every chart-driven selection, such as clicking a smile point, needs a keyboard path: a selectbox or a table row.

---

## Part 2. What Streamlit 1.65 and 1.66 can and cannot do

**Method:**
- I read the installed 1.66.1 source and docstrings.
- I diffed every public signature and config option against an installed 1.65.0. All the APIs below exist in both unless marked "1.66 only".
- I read the agent references bundled in the wheel (`streamlit/.agents/skills/developing-with-streamlit/references/*.md`, which ship with the package).
- I read the release notes: https://docs.streamlit.io/develop/quick-reference/release-notes (1.66.0, 14 Oct 2026) and https://docs.streamlit.io/develop/quick-reference/release-notes/2026 (1.65.0, 2 Oct 2026; 1.64.0; 1.63.0).
- I ran three probe apps in Chromium on both versions.

### 2.0 What changed between 1.65.0 and 1.66.1 [V, S]

- **`on_change="ignore"`** (store the value without a rerun) is available on every stateful widget only from 1.66. It was added there for `st.pills`, `st.segmented_control`, `st.feedback`, `st.color_picker`, `st.datetime_input`, the camera and audio inputs, `st.data_editor` and `st.pagination`. Earlier releases covered the rest:
  - 1.63: slider, text_input;
  - 1.64: number_input, select_slider, selectbox;
  - 1.65: checkbox, toggle, radio, multiselect, text_area, date and time inputs, file_uploader.
- **`theme.paddingTop` / `theme.paddingBottom`**, main and sidebar, per mode: 1.66 only. Before that, the gap between Streamlit's header and the content needs CSS on `stMainBlockContainer`.
- **Accessibility changes in 1.66:**
  - "Skip to main content" link and `<main>`/`<nav>` landmarks. The release notes say selectors like `section[data-testid="stMain"]` should drop the tag.
  - The sidebar toggle uses `data-collapsed` instead of `aria-expanded` for custom CSS and JS.

  Both are evidence that `data-testid` and tag selectors churn between minor releases.
- **AppTest in 1.66:**
  - `at.tabs` and `at.expander` can be looked up by key;
  - `persist_state="page"` values and selected tabs survive AppTest reruns;
  - adds `at.html`, `at.link_button` and `at.page_link`.
- **[J]** The floor is `streamlit>=1.65,<2`. Raising it to 1.66 would buy:
  - `on_change="ignore"` on the legs editor and segmented controls;
  - `paddingTop` instead of CSS;
  - landmarks;
  - better AppTest coverage of tabs.

  The public demo pins its own version, so the cost is small. Worth deciding explicitly.

### 2.1 Execution model and performance

- **Rerun model:** every widget interaction reruns the whole script top to bottom unless it is scoped. The levers, in the order the bundled guide recommends: caching, fragments, forms, then `on_change="ignore"`.
- **Caching** (https://docs.streamlit.io/develop/concepts/architecture/caching):
  - `st.cache_data` pickles and returns a copy, so it is safe for mutable data but costs a copy of large arrays;
  - `st.cache_resource` returns the shared object, which suits immutable heavy objects such as a fitted surface, provided no one mutates it [S/J];
  - bound caches with `max_entries` or `ttl`;
  - `refresh_mode="background"` with `ttl` serves stale values while recomputing [V, bundled guide].
- **`@st.fragment`** (https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment) [V, docstring]:
  - widgets inside a fragment rerun only the fragment;
  - `run_every` sets a periodic rerun;
  - `key=` names a fragment so that a widget callback outside it can call `st.rerun("<key>")` or `st.rerun([...])` to rerun only that fragment. Since 1.63, only from callbacks, and only for fragments drawn in the last full run;
  - `parallel=True` runs independent fragments concurrently on a thread pool during full reruns. They cannot write to outside containers or open dialogs on that run.
  - Elements a fragment writes are redrawn on each fragment rerun; the rest of the page stays.
- **Experiment E6 [V]:** `@st.fragment(run_every=1.0)` ticked 4 times in 4 s with zero full-app reruns, on both 1.65.0 and 1.66.1.
- **`on_change="ignore"`, experiment E3 [V]:** editing a `number_input(on_change="ignore")` caused no rerun (the run counter stayed at 1). The pending value (105) reached Python on the next rerun, which came from an unrelated shortcut. Same on 1.65.0 and 1.66.1. This allows a "stage several inputs, then Apply" bump bar without `st.form`'s limits. Caveat from the bundled guide: pending values are lost on page refresh unless the widget is bound to the URL.
- **Lazy containers [V]:**
  - `st.tabs`, `st.expander` and `st.popover` with `on_change="rerun"` (or a callable) expose `.open`, so hidden content can be skipped;
  - tabs and expanders also do this with `bind="query-params"`.

  The app already uses lazy tabs (P0-1).
- **Transport [V, config]:**
  - `server.enableWebsocketCompression` defaults to **False**. The P0-1 baseline measured 274-344 KB per rerun, mostly Plotly JSON, which compresses well.
  - **[J] Measure turning compression on.** It is a one-line config change and may cut payloads several-fold. Check that Community Cloud honours it.
  - ForwardMsg caching exists already: messages of at least `global.minCachedMessageSize` = 10 kB are cached by hash for `maxCachedMessageAge` = 2 runs.
  - `st.dataframe` loads rows lazily above 150,000 rows. Lazy mode disables selections, Styler and search.
- **Loading feedback [V]:**
  - `st.skeleton(height=…)` works standalone or as a context manager;
  - `container.skeleton()` reserves a slot and keeps the element mounted (bundled guide);
  - `st.spinner(show_time=True)`, `st.status` and `st.progress` are available.

### 2.2 `st.tabs` with `key`, `on_change`, `bind`, `default` [V, docstring + E1/E2]

- Signature: `st.tabs(tabs, *, width, height, default, key, on_change="ignore"|"rerun"|callable, args, kwargs, bind="query-params"|None)`. `.open` is True or False once state is tracked.
- **The active tab can be set programmatically through `st.session_state[key]`.** A bound query parameter cannot be set through `st.query_params`. The default tab removes the parameter from the URL, and invalid values are dropped.
- Labels must be unique and non-empty when bound. The URL carries the label verbatim, so Markdown or icons in labels leak into URLs. The project already avoids this.
- Content can scroll inside the panel with `height=int` or `"stretch"`.
- **Experiment E1 [V]:** a hidden button with `shortcut="Alt+3"` and `on_click=lambda: ss.update(tab="Risk")` switched the bound tab, updated the URL to `?tab=Risk`, and the deep link `?tab=Strategy` opened that tab. Same on both versions.
- **Limits:**
  - no per-tab badges or counts, other than Markdown in the label, which leaks into the URL;
  - no closable or reorderable tabs;
  - no vertical tabs.

### 2.3 `st.dialog`: modal, right/left drawer [V, docstring + E5]

- Signature: `@st.dialog(title, *, width="small"|"medium"|"large", position="center"|"left"|"right", dismissible=True, icon, on_dismiss="ignore"|"rerun"|callable)`.
  - Widths: small up to 500 px, medium up to 750 px, large up to 1280 px.
  - Left and right are "user-resizable full-height" drawers.
- **Experiment E5 [V]:** `position="right", width="medium"` rendered as a 752x800 drawer at x = 528 in a 1280x800 viewport (`drawer-right.png`), with KaTeX maths inside. Same on both versions.
- **Limits:**
  - "Side drawers stay modal: the rest of the app is inert." Only one dialog per script run.
  - `st.sidebar` cannot be used inside.
  - Widgets inside rerun only the dialog (fragment semantics).
  - **There is no non-modal inspector panel.** A side-by-side inspector must be built with `st.columns` or the sidebar. It is fine for Methodology (DK-6) and leg details; it is wrong for anything users need to watch while editing [J].

### 2.4 `st.navigation` and pages [V]

- Signature: `st.navigation(pages, *, position="sidebar"|"hidden"|"top", expanded)`, with `st.Page(…, url_path=, default=, visibility=)`, plus `st.switch_page(page, query_params=)` and `st.page_link`.
- **[J]** For this tool, bound tabs on one page are the better fit:
  - one shared header and sidebar inputs;
  - one script whose position builder serves all tabs.

  Pages would put each view on its own path and make shared state depend on `persist_state="session"` or URL binding. Reconsider only if the app grows a second workspace, for example a separate "Book" or "Backtest" area.

### 2.5 Keyboard shortcuts

- **[V, docstring + frontend bundle]** `shortcut=` exists on `st.button`, `st.download_button`, `st.link_button` and `st.form_submit_button`. There is none on tabs, popovers, `menu_button` or selectboxes. Rules:
  - accepted keys: a single letter or digit, F-keys, the special keys (Enter, Esc, Tab, Space, arrows, Home, End, PageUp, PageDown, Backspace, Delete), and any of these with Ctrl, Cmd, Alt or Shift (Mod is an alias of Ctrl/Cmd);
  - **C and R are reserved even with modifiers**;
  - punctuation (`.` `,` `/` `?`) is not supported;
  - browser combinations such as Ctrl+T, Ctrl+W and Ctrl+PageUp may never arrive.

  The button shows the shortcut as a hint ("Ctrl + K"). The listener is global (a Mousetrap-style binding on the document).
- **[V, code + E1]** Plain-key shortcuts are suppressed while focus is in an input. Shortcuts with a modifier still fire there: Alt+2 and Ctrl+K both fired with focus in a number input. Choose modifier shortcuts that do not fight typing [J].
- **[V, E1]** A shortcut still works when its button is hidden with CSS (`.st-key-hidden_keys {display:none}`). This gives an invisible layer of shell shortcuts on both versions.
- **[V, E4]** For keys Streamlit refuses (`/` for search, `?` for help, `[` and `]` for previous and next expiry), a ~15-line inline Custom Components v2 module works:
  - `st.components.v2.component(name, html=…, js=…)`, with no iframe and no build step;
  - it adds one `document` keydown listener that ignores events from inputs and calls `setTriggerValue("key", e.key)`;
  - Python receives `/` and `]`, and keys typed inside inputs are ignored.

  Verified on both versions.

### 2.6 Tables: `st.dataframe`, `column_config`, selection, `st.data_editor`, `st.table`

**`st.dataframe` signature [V]:** `(data, width, height, *, hide_index, column_order, column_config, key, on_select, selection_mode, selection_default, row_height, placeholder, lazy, alt)`.

- **Selection modes [V]:**
  - `single-row`, **`single-row-required`** (radio-like; auto-selects the first row), `multi-row`;
  - `single-column`, `multi-column` (these disable sorting);
  - `single-cell`, `multi-cell`.

  `selection_default` presets the selection on the first render.
- **Experiment E7 [V]:** `single-row-required` started at `[0]`, and a mouse click on row 4 gave `[3]`. ArrowDown then Space did not change the selection in my probe. Keyboard row selection is unproven, so give a selectbox as the keyboard path.
- **Column types [V]:**
  - Number, Text, Checkbox, Selectbox, Multiselect, Date, Time, Datetime, Link, Image, List, Json;
  - Line/Bar/Area chart columns (sparklines), Progress;
  - **ButtonColumn** (per-row actions; a list value gives a dropdown menu; needs `key`; transient click state);
  - Markdown (detail overlay only), Audio, Video.

  Common parameters: `width`, `help` (header tooltip), `pinned`, `alignment`, `disabled`, `required`. NumberColumn adds `format`, `min_value`, `max_value`, `step`.
- **Styler [V]:** cell `color` and `background` work on the canvas grid (P&L coloured in `dense-table-dark.png`).
  - **Gotcha:** unformatted columns show 6 decimals.
  - Formatting must come from `column_config`, which takes precedence over Styler formatting.
  - Lazy mode disables Styler.
- **`st.data_editor`:**
  - `num_rows` ("fixed", "dynamic", "add", "delete");
  - `disabled` per column;
  - client-side validation through the column limits;
  - edits survive data refreshes when keyed and the structure is unchanged (bundled guide);
  - `on_change="ignore"` is 1.66 only;
  - commits a cell on blur. The project already handles edits that arrive with a tab click.
- **`st.table`:** static, renders Markdown in cells, `border="horizontal"`, `hide_header`. The right choice for small key-value blocks with U+2212 strings.
- **Limits [J]:**
  - no row grouping or tree, so expiries cannot be folded within one grid (use one grid per expiry, or a selectbox);
  - no custom cell renderers beyond the column types;
  - no right-click menu (ButtonColumn is the substitute);
  - no synchronised scrolling between grids;
  - the grid takes its font from the theme, and CSS cannot reach the canvas.

### 2.7 Charts: `st.plotly_chart` (and `st.echarts_chart`)

**Signature [V]:** `(fig, *, width, height, theme="streamlit"|None, key, on_select="ignore"|"rerun"|callable, selection_mode=("points","box","lasso"), config, alt)`.

- **Selection state [V]:** `points`, `point_indices`, `box` and `lasso`. It is read-only and cannot be set programmatically.
- **Experiment E8 [V]:** a click on a point with `selection_mode="points"` returned that point (x = 97.5).
- **Not available to Python [V, PlotlyState schema]:**
  - hover events;
  - zoom or pan (relayout) events;
  - legend clicks;
  - double-clicks.

  Consequence: no hover-linked readouts in Python and no cross-chart crosshair sync between separate `st.plotly_chart` elements. Within one figure, Plotly's own shared axes and `hovermode="x unified"` work [J].
- **Experiment E9 [V], important for desk use:** user zoom survives reruns that do not change the figure. **It resets whenever the figure's data change**, for example after a spot change. This happened with `layout.uirevision="keep"` and without it, with `theme="streamlit"` and `theme=None`, on 1.66.1 (Plotly 7.1.0) and on 1.65.0 (Plotly 5.24.1).

  Plotly's documented mechanism (https://plotly.com/python/reference/layout/#layout-uirevision) does not survive Streamlit's re-render. Workarounds [J]:
  - an explicit, URL-bound range control (`segmented_control`: ±5%, ±10%, ±1σ, ±2σ, Full) that sets `xaxis.range` in Python;
  - or box-select-to-zoom (`selection_mode="box"` returns the box coordinates, which Python turns into axis ranges).
- **Theme [V]:**
  - `theme="streamlit"` applies `chartCategoricalColors` and `chartSequentialColors`, and fonts;
  - the docstring does not list diverging colours for Plotly, while the bundled theme guide says Plotly reads them. Set `colorscale` explicitly for P&L heatmaps [J];
  - the project's finding that exactly ten categorical colours are needed (otherwise Plotly's placeholders are not replaced) is recorded in `config.toml`. I did not re-test it.
- **`config` [V]:** passes through Plotly config, for example `{"displayModeBar": False}`. The `alt` parameter exists since 1.65.
- **`st.echarts_chart`** (1.64+) is an alternative with JSON-only options: no JS formatters, no custom series, no GL. It reads categorical and sequential palettes but not diverging. **[J]** Not worth a second chart library here.

### 2.8 Layout and controls

- **`st.container`** [V]: `border`, `key`, `width`, `height`, `horizontal`, `wrap`, `horizontal_alignment` ("left", "center", "right", "distribute"), `vertical_alignment` (including "distribute"), `gap` (from "xxsmall" = 0.25rem up to "xxlarge", an int in px, or None) and `autoscroll`.
  - With `horizontal=True, wrap=False`: a single scrolling row.
  - A fixed `height` scrolls.
- **`st.columns`:** `wrap=True` (the default) stacks at a viewport of 640 px or less; `wrap=False` keeps one row. Controls directly in columns or horizontal containers default to no-wrap, with labels ellipsised (bundled guide).
- **`st.metric`** [V]:
  - `format`: the same presets as NumberColumn, plus printf;
  - `delta_color`: normal, inverse, off, or a named colour;
  - `delta_arrow`: auto, up, down, off;
  - `delta_description`, `chart_data`/`chart_type` (sparkline), `border`, `height="stretch"`.

  Probe: `format="%+.4f"` printed `-0.4321`; the delta printed `-$12.30` with a hyphen.
- **`st.popover`** [V]: `key`, plus `on_change` for lazy content; `type`, `icon`, `width`. Don't nest. Contents rerun the app while keeping the popover open.
- **`st.menu_button`** [V]: a button with one-shot actions, returning the chosen option. This is the export menu (DK-8): CSV, XLSX, PNG.
- **`st.download_button(data=callable)`** [V]: the file is generated only on click, on a separate thread. Also takes `shortcut=`.
- **`st.segmented_control` and `st.pills`** [V]: single or multi selection, `required`, `format_func`, `bind`, `persist_state`, `width`, `wrap`; `on_change="ignore"` from 1.66. Use them for Call/Put, Long/Short, vol mode, units and range presets.
- **Other elements:**
  - `st.badge(color=…)` for status chips (model badge, "Synthetic", "Extrapolated");
  - `st.toast(duration=)`;
  - `st.space(size)`;
  - `st.pagination`;
  - `st.logo`.
- **`persist_state`** [V]: "page" or "session" keeps a widget's value while it is not drawn; it needs `key`. URL binding takes precedence.

### 2.9 Theming keys (config.toml)

**[V]** Sections: `[theme]`, `[theme.light]`, `[theme.dark]`, `[theme.sidebar]`, `[theme.light.sidebar]`, `[theme.dark.sidebar]`.

**Keys:**
- `base` (a built-in name or a shared TOML path/URL), `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `linkColor`, `linkUnderline`, `borderColor`, `showWidgetBorder`, `showSidebarBorder`;
- `baseRadius`, `buttonRadius`;
- `font`, `headingFont`, `codeFont`, `codeFontSize`, `codeFontWeight`, `baseFontSize`, `baseFontWeight`, `headingFontSizes`, `headingFontWeights`, `metricValueFontSize`, `metricValueFontWeight`;
- `[[theme.fontFaces]]` (with `server.enableStaticServing`);
- `red|orange|yellow|green|blue|violet|grayColor` and their `*BackgroundColor` / `*TextColor` variants;
- `codeBackgroundColor`, `codeTextColor`;
- `dataframeBorderColor`, `dataframeHeaderBackgroundColor`, `dataframeHeaderTextColor` (1.65);
- `chartCategoricalColors`, `chartSequentialColors`, `chartDivergingColors` (sequential and diverging need exactly 10);
- `paddingTop`, `paddingBottom` (1.66).

**Constraints:**
- [S, bundled guide] Theme options cannot be changed from the script: `st.set_option` rejects them.
- [V, project] `st.context.theme.type` reports the theme only as of the start of the script run. It is stale right after the user toggles the theme in the menu, until the next rerun.
- [V] The app's root element (`stApp`) carries `color-scheme: dark` or `light`. CSS `light-dark()` therefore switches instantly; the project already relies on this.

### 2.10 CSS hooks: what is stable

- **`st.html` [V, docstring]:**
  - content is sanitised with DOMPurify and not iframed;
  - **HTML that contains only `<style>` goes to the event container and takes no layout space**;
  - a `Path` to a `.css` file is wrapped in `<style>` automatically;
  - `unsafe_allow_javascript=True` runs scripts (see 2.11).
- **Documented hook:** `key=` on containers, widgets, charts, tabs, popovers and expanders adds the class `st-key-<key>` (non-alphanumerics are replaced by `-`; frontend function seen in the bundle [V]). This is the only CSS hook the docs describe.
- **Undocumented but widely used:** `data-testid` values. The 1.66.1 bundle has about 280 of them, for example `stMetricLabel`, `stMetricValue`, `stMainBlockContainer`, `stTabs`, `stDialog`, `stDataFrame`, `stSkipToContent`. They are not a public API.
  - Release 1.66 changed `section[data-testid="stMain"]` to `<main>`, and the sidebar's `aria-expanded` to `data-collapsed`.
  - DK-5's allowlist (`.st-key-*` plus three test ids, with no tag-qualified selectors) is the right defensive policy.
  - **[J]** Add a CI check that every allowlisted selector still matches an element on each Streamlit upgrade. DK-7's browser suite is the natural home.
- **[V]** The `--st-*` theme variables exist only inside CCv2 components. They are not on `:root` in the main page (`getComputedStyle(documentElement)` returned none). Page CSS must hard-code tokens (mirrored in `theme.py`, as the project does) or use `light-dark()`.

### 2.11 Custom JavaScript: what "no custom JS without components" really means now

- **[V]** `st.html(..., unsafe_allow_javascript=True)` (present in 1.65) runs JS in the page with no iframe. It is one-way: it cannot return values to Python except by clicking a Streamlit button in the DOM, which is a fragile hack.
- **[V]** Custom Components v2 (`st.components.v2.component(name, html=, css=, js=, isolate_styles=True)`, present in 1.65):
  - inline strings, so no npm and no build;
  - rendered in a shadow root in the page, not an iframe;
  - two-way through `setStateValue` (persistent) and `setTriggerValue` (one rerun);
  - `--st-*` theme variables are injected.

  Experiment E4 proves the pattern for global hotkeys. Same mechanism would cover:
  - a client-side clock with no server traffic per tick;
  - localStorage-backed preferences;
  - copy-to-clipboard of the share URL;
  - a focus manager ("/" focuses search).
- **[S, bundled guide]** v1 components (`components.v1.html`, iframes, `Streamlit.setComponentValue`) are legacy. Use `st.iframe` or CCv2.
- **[J] Cost and risk:**
  - CCv2 code is the project's own JS. It needs its own tests (Playwright) and must be idempotent across reruns: assign handlers, guard global listeners, as the probe did with `window.__deskKeys`;
  - keep each module under ~50 lines;
  - put nothing in CCv2 that a native widget can do.

### 2.12 Testing hooks

- **[V]** AppTest supports keyed lookups and more elements in 1.66 (see 2.0).
- **[J]** Browser behaviour cannot be covered by AppTest: shortcuts, CSS, Plotly zoom, drawers. It belongs in DK-7's Playwright suite. The probe scripts here (`scripts/probe.py`, `probe2.py`, `probe3.py`) are small templates for that.

---

## Part 3. Capability table: desk UI patterns in Streamlit 1.65+

Cost key:
- **S**: under half a day, native API.
- **M**: one to two days, or a CSS/CCv2 shim.
- **L**: several days, or a custom component with real JS.
- **No**: not feasible without leaving Streamlit.

Evidence uses the tags above.

| # | Desk UI pattern | Feasible in 1.65+? | How | Cost | Evidence and caveats |
|---|---|---|---|---|---|
| 1 | Compact desk header (ticker, model badge, actions) | Yes | `st.container(horizontal=True, key="desk_header")`, `st.badge`, `st.popover`, CSS on `.st-key-desk_header` | S (DK-5 in progress) | [V] |
| 2 | Sticky header while scrolling | Partly | CSS `position: sticky` on `.st-key-desk_header`; no native sticky container (only `st.bottom`) | M | [J] Must not obscure focus (WCAG 2.4.11). Test on each upgrade. |
| 3 | Lazy tabs with deep links | Yes | `st.tabs(key, on_change="rerun", bind="query-params")` plus `.open` | Done (P0-1) | [V] |
| 4 | Exact-state share link | Yes | `bind="query-params"` on inputs; `st.context.url` shown in a popover with `st.code` (copy button) | S | [V] |
| 5 | Keyboard shortcuts for actions and tab switching | Yes | `st.button(shortcut=…)` in a hidden `.st-key` container; tab switch through `session_state` in `on_click` | S | [V] E1, both versions. C and R reserved; no punctuation; modifier shortcuts fire inside inputs. |
| 6 | Punctuation hotkeys (`/`, `?`, `[`, `]`) and a "?" cheat-sheet | Yes | Inline CCv2 keydown listener plus `st.dialog` for the sheet | M | [V] E4. Own JS; must ignore inputs (done in the probe). |
| 7 | Command palette (Ctrl+K: jump to strategy, tab, expiry) | Yes, basic | `shortcut="Ctrl+K"` opens an `st.dialog` with `st.selectbox(filter_mode="fuzzy")` | M | [J] Autofocus of the selectbox in the dialog is not verified. CCv2 can force focus. |
| 8 | Right-side methodology or detail drawer | Yes (modal) | `@st.dialog(position="right", width="medium"/"large")` | S | [V] E5. Modal, one at a time. |
| 9 | Non-modal side inspector beside the main view | Partly | `st.columns([3, 1])` or the sidebar; no resizable split panes (only the sidebar and the drawer resize) | M | [J] |
| 10 | Dense numeric tables (chain, ladder, legs) | Yes | `st.dataframe(row_height=24-28, hide_index, column_config=NumberColumn(format=…, pinned=…, alignment=…), placeholder="–")` | S | [V] E7 screenshot. Hyphen-minus, not U+2212. Styler 6-dp gotcha. |
| 11 | Tabular figures everywhere | Yes | Font choice: Plex digits are tabular by default (canvas grids cannot use CSS); CSS `tabular-nums` for DOM text | Done (DK-2) | [V] fontTools |
| 12 | Conditional cell colour (P&L sign, flags) | Yes | pandas Styler `.map` for colour only, plus `column_config` for formats | S | [V]. Not with `lazy=True`. |
| 13 | In-cell bars and sparklines (vega by bucket, term structure) | Yes | `BarChartColumn`, `LineChartColumn`, `ProgressColumn`; or Styler bars | S | [V] signatures |
| 14 | Click a chain row or cell to drive other views | Yes | `on_select="rerun"`, `selection_mode="single-row-required"` or `"single-cell"`, `selection_default` | S | [V] E7 (mouse). Keyboard selection unproven: add a selectbox. Reruns everything unless the dependent views are a keyed fragment. |
| 15 | Per-row actions (close, flip, duplicate leg; open detail) | Yes | `ButtonColumn` (a list value gives a dropdown) plus a callback opening a dialog | S | [V] signature, [S] bundled guide |
| 16 | Editable legs with validation | Yes | `st.data_editor` with `SelectboxColumn`, NumberColumn min/max/step, `num_rows="dynamic"`, `disabled=[…]` | Exists | [V]. `on_change="ignore"` 1.66 only. |
| 17 | Row grouping, tree tables, column groups | No (native) | One grid per expiry; or `st.table` with a MultiIndex for static views | M / L | [J] |
| 18 | Right-click context menus, drag-and-drop | No (native) | Substitute ButtonColumn menus; drag needs CCv2 | L | [J] |
| 19 | Click a chart point to select (smile point, P&L curve) | Yes | `st.plotly_chart(on_select="rerun", selection_mode="points")` | S | [V] E8 |
| 20 | Box-select a region (zoom-to-range, pick strikes) | Yes | `selection_mode="box"` returns box coordinates; Python sets the axis range | M | [V] schema, [J] pattern |
| 21 | Zoom that survives input changes | No (native) | Explicit URL-bound range presets (`segmented_control`) set `xaxis.range` | S | [V] E9: zoom resets on any data change, `uirevision` ignored, both versions |
| 22 | Hover readouts in Python; crosshair sync across charts | No across elements; Yes inside one figure | Plotly subplots with shared axes and `hovermode="x unified"`; across charts only by a CCv2 Plotly wrapper | S / L | [V] no hover events in PlotlyState |
| 23 | Live market clock and session status | Yes | `@st.fragment(run_every=1)` (server tick) or a CCv2/`st.html` JS clock (client tick, no traffic) | S | [V] E6. WCAG 2.2.2: add a pause, or tick only the minute. |
| 24 | Recompute only the affected panel | Yes | Keyed fragments plus `st.rerun("<key>")` from callbacks; widgets inside fragments | M | [V] docstring. Sidebar inputs that feed everything still need full reruns. |
| 25 | Stage several bumps, then apply (a "bump bar") | Yes | `on_change="ignore"` on the inputs plus an Apply button; or `st.form` | S | [V] E3. On 1.65 this works for number, select, slider and text inputs, not `data_editor`, pills or segmented. |
| 26 | Parallel heavy panels (MC VaR, surface fit) | Yes | `@st.fragment(parallel=True)` with `container.skeleton()` | M | [S] docstring. No outside writes or dialogs on the parallel run; thread-safety is ours. |
| 27 | Loading, stale, empty and error states | Yes | `st.skeleton`, `st.status`, `st.progress`, `st.spinner(show_time=True)`, captions plus action buttons; `placeholder=` in grids | S | [V]. No built-in empty-state element (compose it). |
| 28 | Mode switches (Call/Put, Long/Short, units, vol mode) | Yes | `st.segmented_control` / `st.pills` with `bind` and `persist_state` | S | [V] |
| 29 | Secondary settings without leaving the view | Yes | `st.popover(key, on_change="rerun")` for lazy content | S | [V] |
| 30 | Export menu (CSV, XLSX, PNG) | Yes | `st.menu_button` plus `st.download_button(data=callable)` (lazy) | S | [V]. Static chart PNG export needs Kaleido, or the Plotly modebar's own download. |
| 31 | KPI tiles with deltas and sparklines | Yes | `st.metric(format, delta_color, delta_arrow, delta_description, chart_data, border, height="stretch")`; the project's `card_row` CSS for wrapping | S | [V] |
| 32 | Light and dark with per-mode palettes; self-hosted fonts | Yes | `[theme.light]`/`[theme.dark]` plus 10-colour chart ramps plus `fontFaces` | Done (DK-1/2) | [V] |
| 33 | Per-user CVD or "red-up" colour setting | Partly | App-side semantic palette switched by a setting (charts, Styler, CSS via `.st-key`, metric `delta_color`); theme tokens fixed | M | [V] theme options cannot be set from the script |
| 34 | Tighter page padding | Yes | 1.66: `theme.paddingTop`; 1.65: CSS on `stMainBlockContainer` (allowlisted) | S | [V] |
| 35 | Accessible names and landmarks | Yes | `alt=` on charts and grids (1.65); skip link and `<main>` (1.66 built in) | S | [V] |
| 36 | Keyboard operability of charts | No (Plotly) | Every chart selection mirrored by a selectbox or table; "View data" expander | S | [J] |
| 37 | Remember preferences across visits | Partly | URL binding (shareable) or CCv2 localStorage; `st.context.cookies` is read-only | M | [V] st.context fields, [J] |
| 38 | Smaller rerun payloads | Yes | Lazy tabs (done); `server.enableWebsocketCompression=true`; send fewer points; stable keys | S | [V] config default False. Measure it. |
| 39 | Phone layout | Yes, with care | Columns stack at 640 px or less; container-query CSS (`card_row`); short tab labels | M (largely done) | [V] |
| 40 | Real-time streaming quotes | Out of scope | The policy forbids scraped data. `run_every` could poll uploads but nothing needs it. | n/a | DT-6 |

---

## Part 4. Implications for the plan of record [J]

1. **DK-8 (shortcuts and export menus) is fully feasible on 1.65.** Use:
   - hidden `st.button(shortcut=…)` for tab switching and actions;
   - one small CCv2 module for `/`, `?`, `[` and `]`;
   - `st.menu_button` plus lazy `download_button` for exports.

   Avoid Alt+letter combinations that clash with OS menus, and the reserved C and R.
2. **Charts need explicit range controls.** User zoom is lost on every input change (E9). This affects the payoff, smile and term-structure charts, and arguably deserves a line in PR-7, BK-2 and SF-6.
3. **Styler rule:** whenever a Styler is passed to `st.dataframe`, every numeric column needs a `column_config` format. This could be enforced in `charts.show` or a `components.table()` helper.
4. **Measure websocket compression** before further payload work. It is one config key.
5. **Decide the Streamlit floor.** Version 1.66 buys `on_change="ignore"` on the legs editor and segmented controls, `paddingTop`, landmarks and better AppTest coverage of tabs.
6. **Accessibility follow-ups:**
   - audit input and border contrast (1.36:1 borders) against SC 1.4.11;
   - give the market clock a pause (SC 2.2.2);
   - add keyboard paths for chart and grid selections;
   - keep `alt=` on every chart via `charts.show`.
7. **Colour setting:** a "standard / CVD blue-orange / red-up" preference is cheap because P&L colour is app-side. It would answer the Bloomberg-style CVD expectation of a professional reviewer.

---

## Sources

**Design and data visualisation:**
- Tufte, *The Visual Display of Quantitative Information*: https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/
- Few, *Common Pitfalls in Dashboard Design* (2006): https://www.perceptualedge.com/articles/Whitepapers/Common_Pitfalls.pdf
- Ström, "UI density": https://mattstromawn.com/writing/ui-density/
- IBM Carbon, data-table style: https://carbondesignsystem.com/components/data-table/style/
- IBM Carbon, data-visualisation colour palettes: https://carbondesignsystem.com/data-visualization/color-palettes/
- FT Visual Vocabulary: https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary
- EU data-visualisation guide, colour connotations: https://data.europa.eu/apps/data-visualisation-guide/colour-connotations
- Material Design dark theme (did not render for me): https://m2.material.io/design/color/dark-theme.html

**Colour and colour-blindness:**
- Okabe and Ito, Color Universal Design: https://jfly.uni-koeln.de/color/
- Bloomberg, designing the Terminal for colour accessibility: https://www.techatbloomberg.com/blog/designing-the-terminal-for-color-accessibility/
- Bloomberg UX accessibility: https://www.bloomberg.com/ux/accessibility/
- Benzinga on red-up markets: https://www.benzinga.com/general/education/21/10/23258001/did-you-know-you-dont-want-to-be-in-the-green-if-youre-trading-in-china-japan-or-taiwan

**Usability:**
- Nielsen, progressive disclosure: https://www.nngroup.com/articles/progressive-disclosure/
- Kaplan, empty states: https://www.nngroup.com/articles/empty-state-interface-design/
- Nielsen, response times: https://www.nngroup.com/articles/response-times-3-important-limits/

**Accessibility and typography:**
- WCAG 2.2: https://www.w3.org/TR/WCAG22/
- MDN, `font-variant-numeric`: https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric
- Butterick, alternate figures: https://practicaltypography.com/alternate-figures.html
- IBM Plex: https://github.com/IBM/plex

**Streamlit:**
- Release notes: https://docs.streamlit.io/develop/quick-reference/release-notes and https://docs.streamlit.io/develop/quick-reference/release-notes/2026
- API reference:
  - st.fragment: https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment
  - st.dialog: https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog
  - st.tabs: https://docs.streamlit.io/develop/api-reference/layout/st.tabs
  - st.navigation: https://docs.streamlit.io/develop/api-reference/navigation/st.navigation
  - st.button: https://docs.streamlit.io/develop/api-reference/widgets/st.button
  - st.dataframe: https://docs.streamlit.io/develop/api-reference/data/st.dataframe
  - st.column_config: https://docs.streamlit.io/develop/api-reference/data/st.column_config
  - st.plotly_chart: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
  - st.container: https://docs.streamlit.io/develop/api-reference/layout/st.container
  - st.popover: https://docs.streamlit.io/develop/api-reference/layout/st.popover
  - st.html: https://docs.streamlit.io/develop/api-reference/text/st.html
  - st.components.v2.component: https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component
  - config.toml: https://docs.streamlit.io/develop/api-reference/configuration/config.toml
  - caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- Installed package docs (the agent references bundled with Streamlit 1.66.1): `/usr/local/lib/python3.11/dist-packages/streamlit/.agents/skills/developing-with-streamlit/references/` (performance.md, layouts.md, data-display.md, theme.md, design.md, custom-components-v2.md, ccv2-theme-css-variables.md)

**Plotly:**
- uirevision: https://plotly.com/python/reference/layout/#layout-uirevision
