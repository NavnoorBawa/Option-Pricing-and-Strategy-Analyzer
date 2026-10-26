# Trader-facing options analytics tools: what the best ones do, and what to take

Research task `trader-tools`, 26 Oct 2026. Scope: OptionStrat, thinkorswim (Analyze tab: Risk Profile and Probability Analysis), Interactive Brokers (Risk Navigator, Probability Lab, OptionTrader and Performance Profile), tastytrade (curve and Analysis), OptionNet Explorer, Option Omega, Market Chameleon, Sensibull, Opstra, Deribit (Position Builder and Greeks) and Cboe (its strategy tools).

How to read the tags:

- **[S]** is a sourced fact, with the URL inline or in the source list (section 8).
- **[O]** is something I saw in a vendor screenshot saved under `screenshots/` (section 9).
- **[J]** is my own judgement. It is not a sourced fact.

Limits of the evidence:

- Market Chameleon blocks automated access, and its Terms of Use forbid crawling, so I did not get past its access-denied page. Its section relies on third-party reviews and search-index summaries only.
- The current thinkorswim thinkManual is short, and the older Risk Profile and Probability Analysis subpages now return 404. The ToS section uses the current manual, the FAQ and broker articles.
- tastytrade has no public help-centre article on curve or Analysis mode that I could reach. Its section uses an independent hands-on review and tastylive help text as the search engine indexed it. The live tastylive URL now serves a backtesting page.
- Cboe no longer has a public "Option Strategy Builder". What exists is listed honestly in section 1.11.

---

## 0. Summary

1. **The best tools put three views of one position side by side** [J]:
   - a curve (P&L against spot for several dates);
   - a grid (P&L by spot and date, or by spot and vol);
   - a stat strip (debit or credit, max P/L, breakevens, probability of profit, net Greeks).
   All three are driven by the same small set of what-if controls: date, IV, spot and rate. OptionStrat [S], Deribit [S], IBKR [S] and Sensibull [S] all converge on this.
2. **The professional tools add things the retail ones lack.** IBKR Risk Navigator and Deribit Position Builder add:
   - explicit shift semantics (absolute vs relative vol, rates scaled by T);
   - a 3-D scenario cube viewed one 2-D slice at a time;
   - position Greeks in dollar terms;
   - a pinned risk dashboard (P&L, margin, VaR, ES);
   - a visible "this is hypothetical" state, such as a red border, edits shown in red until applied, and a Reset Scenario button [S].

   The rest of the tools do not stand out on depth. They stand out on ease:
   - OptionStrat on direct manipulation (drag strikes, a heatmap table);
   - Sensibull on strike controls that keep the structure intact (Shift, Width, Hedge);
   - Option Omega on honest statistics (sample-size dimming, a per-lot vs actual-sizing badge, a health verdict).
3. **The most common failure is hidden assumptions.** Users do not know what a T+n curve or a probability assumes, so they decide the tool is "wrong". Sensibull user threads show this directly [S]. OptionStrat and Deribit avoid it by writing the assumptions down [S].
4. **What would make this app the best at what it does** [J]: keep the retail tools' direct, dense P&L views, and add the desk-grade honesty of IBKR and Deribit: units, shift semantics, a scenario cube, a modified-state indicator, model disclosure and per-row provenance. The app's engines already exceed every tool reviewed here on model depth (third-order Greeks, Let's Be Rational, SVI/SSVI arbitrage checks). The UI has to show that precision.

---

## 1. Per-tool findings

### 1.1 OptionStrat (web, iOS, Android)

Sources: the builder tutorial https://optionstrat.com/tutorials/options-builder, the calculator tutorial https://optionstrat.com/blog/options-profit-calculator-tutorial, the FAQ https://optionstrat.com/faq and the App Store page https://apps.apple.com/us/app/optionstrat-options-toolkit/id1541714905.

**Building a strategy**
- A menu of 50+ templates, grouped by level (Novice, Intermediate, Advanced, Expert) and by outlook. Hovering over one shows a description and a small payoff shape [S].
- An expiry ribbon grouped by month, with a DTE label such as "EXPIRATION: 30d" [O].
- A horizontal strike ruler:
  - long legs sit above the axis and short legs below; calls are green and puts red [S];
  - you drag a leg's chip to change its strike [S];
  - Shift-drag moves all strikes at once; for condors and butterflies, Shift-drag on one leg moves the opposing leg the opposite way [S];
  - premium users see volume bars per strike: calls above in green, puts below in red [S][O].
- Leg menu (right-click on desktop, tap on mobile) [S]:
  - quantity (legs are linked by default; a link/unlink icon allows unbalanced ratios);
  - type;
  - custom cost basis, flagged with a blue "$" icon;
  - Include/Exclude, which keeps the leg but drops it from the calculation (crossed-out eye, dimmed);
  - liquidity warnings for low volume, low open interest or a wide bid/ask.
- Entry cost defaults to mid. A "bid/ask" option uses the ask for longs and the bid for shorts [S].

**The P&L diagram**
- A Table/Graph toggle [O].
- The table:
  - rows are underlying prices, each with its % move from spot; a dashed rule marks the current spot [O];
  - columns are dates, with month headers and weekday letters [O];
  - expiry gets two columns, open and close, if there is room [S];
  - cells are a diverging green/red heatmap graded by magnitude [O];
  - a "Chart Range ±x%" slider zooms the table [S][O];
  - a unit toggle offers P/L $, P/L %, Contract Value and % of Max Risk; % of entry cost exists only for net-debit trades and % of max risk only for defined-risk trades [S][O];
  - premium users see ex-dividend and earnings icons in the column headers [S][O].
- The graph shows one day at a time [S]:
  - a "Days until expiration" slider with the calendar date beside it;
  - hover readout, and the breakeven as a labelled blue vertical line;
  - spot as a dashed vertical line [O];
  - profit and loss areas filled green and red;
  - premium users also get a shaded probability distribution for the selected day, and arrows giving P(below) and P(above) the hovered price [S][O].

**Stat strip** [S][O]
- Net debit or credit, max loss, max profit (or "Infinite"), chance of profit, and breakeven with its % from spot.
- Unrealized and realized gain for saved trades.
- Swipe to a second page for net Delta, Theta, Gamma, Vega and Rho, each with its Greek letter in a distinct colour.

**Assumptions disclosed** [S]
- Constant IV.
- Mid prices, with no fees.
- Events are shown but not modelled.
- Table times: market open for future days, the current time for today, and market close on expiry day.
- Model: Bjerksund-Stensland 2002 American, chosen over Black-Scholes because of early exercise.
- Max loss and max profit are "at expiration" and ignore assignment and dividend risk.

**What-if controls** [S][O]
- An IV slider shows a single absolute IV. For multi-leg trades it is not clear how per-leg IVs and skew move [J].
- The DTE slider.
- No spot or rate shift beyond the table's rows.

**Comparison** [S]
- When editing a saved trade, a toggle switches between the old and new strategy, and a grey line on the graph shows the old payoff.

**Optimizer** [S]
- You enter a target price and date. It searches thousands of trades and ranks them from maximum return to maximum chance of profit.

**Mobile** [S]
- Native apps, with saved trades synced to the web.
- Tap replaces right-click, and stats move to swipeable pages.

**What to take** [J]: the date × price heatmap with a % move column and spot rule; the unit toggle; leg exclude; the cost-basis flag; ghost-line compare; the written list of assumptions.

### 1.2 thinkorswim (Schwab): Analyze tab

Sources: the thinkManual Analyze page https://toslc.thinkorswim.com/center/howToTos/thinkManual/Analyze, the FAQ https://toslc.thinkorswim.com/center/faq/Analyze, the Risk Profile video page https://www.schwab.com/learn/story/analyze-trades-using-risk-profile-on-thinkorswim, an old Ticker Tape article (domain offline) https://tickertape.tdameritrade.com/amp/trading/thinkorswim-options-risk-profile-tool-18671, and https://bearbulltraders.com/wp-content/uploads/2023/09/Unlocking-Your-Options-Trading-Potential-using-ToS.pdf.

**Building a strategy** [S]
- In Add Simulated Trades, click Ask to simulate a buy or Bid to simulate a sell. The chain looks like the Trade tab.
- Right-click a strike to get spread templates split by buy and sell side.
- Any order can be sent to analysis with "Analyze trade".

**The Risk Profile diagram** [S]
- X is the underlying price and Y is P&L.
- Two default curves: expiration, and today (using IV).
- The documentation names the colours inconsistently by scheme: blue/purple, cyan/magenta, or green on the light theme.
- Changing the date in the Positions and Simulated Trades pane moves the "today" curve.
- More plot lines can show several points in time, or changes in IV and the underlying (Ticker Tape).
- Price slices from the Add Simulated Trades pane appear on the graph with their probability at expiration.
- Breakeven slices can be set (Set Slices > Breakeven).

**Price slices** [S]
- Price targets offset from Mark or Theo, set in %, $, sigma or estimated price range.
- The default is ±10%.
- They give P/L at each slice for the chosen date. This is effectively a one-dimensional scenario table.

**Probability Analysis** [S]
- A price cone over future days, defaulting to 68.27% (1σ).
- A "Prob range" box sets another probability.
- A related chart study, ProbabilityOfExpiringCone, draws the same cone on price charts (thinkorswim blog, via search).

**Portfolio** [S]
- Beta-weighted risk profile of the whole portfolio against a reference symbol such as SPX.
- "Portfolio, Uniform Stress Test" mode stresses positions at price slices with Sticky Ratio vol and an extra vol adjustment. It is off by default; you must ask chat support to enable it.

**Volatility modes** [S, secondary]
- Application Settings offer individual IV (the default), a volatility smile approximation, or a fixed vol per expiry (search-result summary of third-party guides).

**Mobile** [S]
- I found no documentation that Risk Profile exists in the mobile app. Schwab's web and mobile pages mention "risk and reward graphs" only for the web platform.

**What to take** [J]: price slices as a compact scenario strip with probabilities; the probability cone with an editable probability; beta-weighting. **What to avoid:** colour meanings that change with the theme, and power features gated behind support.

### 1.3 Interactive Brokers: Risk Navigator, Probability Lab, OptionTrader and Performance Profile

Sources: the TWS guide pages under https://www.ibkrguides.com/traderworkstation/ (risk-navigator.htm, what-if-portfolios.htm, risk-dashboard.htm, portfolio-relative-pnl-graph.htm, scenario-analysis.htm, risk-nav-reports.htm, risk-nav-trader-personalities.htm, model-navigator.htm, optiontrader.htm, performance-profile.htm, strategy-performance-graph.htm, probability-labsm.htm); the Risk Navigator webhelp at https://www.interactivebrokers.com.hk/php/webhelp/navigator/ (reportMeasurePricevol.htm, reportUnderlying.htm, reportMeasureUnderlying.htm, reportPortfolio.htm); and https://www.interactivebrokers.com/en/trading/options-trading-tools.php.

**Building a strategy** [S]
- OptionTrader is a standalone screen with the underlying's quote, chains (tabbed by expiry or a list), orders, executions and portfolio.
- Strategy Builder:
  - click the bid to add a sell leg, or the ask to add a buy leg; calls on the left, puts on the right;
  - drop-downs change action, expiry, strike or type;
  - with a predefined strategy selected, hovering over the first leg highlights the other legs, and the scroll wheel widens or narrows the spread before you click;
  - "Make Delta Neutral" adds a stock leg sized from the system delta, and editing either the delta or the stock quantity updates the other;
  - "Add to Quote Panel" and "Transmit" are available.
- OptionTrader can also quote and work orders in volatility terms (options-trading-tools page).
- Probability Lab [S][O]:
  - shows a histogram of the market-implied terminal distribution (blue), with forward and vol;
  - you drag red bars to make your own distribution, and the others renormalise to 1.00;
  - undo, redo and reset are available;
  - it then builds up to 4-leg strategies, with options for delta-neutral only, include stock, and roll or incorporate existing positions;
  - each strategy shows expected profit, Sharpe, probability of profit, max return and loss with their probabilities, margin and breakevens;
  - the P&L is plotted over the distribution on dual axes, and a dropdown switches the line to Delta, Gamma, Vega, Theta or Rho.

**The P&L diagram: Performance Profile** [S][O]
- A Quote Details panel shows return/risk, profit probability, max return and max loss (each with "(<1% chance)" or a % probability), aggressiveness, min invest, breakevens, commission and margin impact.
- The Performance Graph shows P&L or any Greek against spot.
- A Scenarios window gives P&L and the instrument or position Greeks at spot moves of ±10/20/30%.
- **An honest note:** for multi-expiry combos, P&L "is computed as of the earliest expiration date", and losses after that expiry can exceed the shown Max Loss. The documented example goes from $411 to $2,545.

**Risk Navigator** [S]
- What-if portfolios:
  - outlined in red;
  - double-click a position to edit it; "New" or drag-and-drop adds contracts;
  - Save As, Reload from file, and add or subtract other portfolios;
  - an asterisk marks unsaved changes;
  - margin figures are stale until recalculated, and a refresh icon appears.
- Risk Dashboard, pinned above every tab:
  - Net Liquidation, daily P&L, maintenance and initial margin, VaR, Expected Shortfall, average beta and portfolio beta;
  - a warning icon lists positions excluded from VaR and ES;
  - a refresh icon appears only when results are stale.
- Reports:
  - Risk by Underlying, Portfolio, Risk by Position, Measure by Underlying and Maturity (an expiry ladder), Portfolio Statistics, VaR report and P&L pie;
  - a Report Designer configures columns, and "trader personalities" (Derivatives, Equity Hedged, Equity Long/Short) set default columns.
- Position Greeks:
  - capital letters (D, G, V, T) mean position Greeks (Greek × position);
  - Exposure = position × delta × multiplier × underlying price, converted to share-equivalent;
  - VaR is one-day at 99.5%, and "N/A on a violet background" where it cannot be computed.
- Measure by Price Change and Volatility Change, the scenario cube:
  - price −30…+30% (7 steps) × vol −30…+30% (5) × rate ±100 bp (5) = 175 scenarios per contract;
  - time is advanced to the next close and prices come from the previous close;
  - the screen shows "a flat slice" at a time, picked through Dimensions > Scenario Risk Matrix;
  - you can then **drill through**: click a cell, right-click "Apply Scenario to" another report, and that report shows the scenario at its top until you press Reset Scenario.
- Portfolio Relative P&L graph [S][O]:
  - ±30% default range, with absolute price or % on the x-axis;
  - beta-weighted aggregation;
  - right-click to add date scenarios, toggle "Vol-up by 15%", "Vol-down by 15%" (relative: 10% → 11.5%) and "reduced time to expiry by 1 day", turn on a vertical crosshair, and set slicers;
  - a one-day 99.5% (or 95%, 99%) confidence interval drawn as yellow vertical lines, and a "Historical VAR" marker on the curve.
- Vol-coordinated scenarios [S]:
  - for equities, the nominal vol shock is Y = −X when X > 0 and −10X when X < 0;
  - it is scaled by a decreasing term-structure response VR(t) with VR(0) = 1;
  - the curves are labelled "Vol.Coord." with a tooltip listing the scenarios;
  - the documentation states plainly that scenarios "are not predictive".
- Model Navigator [S][O]:
  - an IV-by-strike plot per expiry;
  - calls and puts as blue and red markers, with darker markers for half quotes;
  - **error bars from the bid/ask**, with the ± range in the tooltip, and an open circle when IV comes from the previous close;
  - a fitted curve, manual curve editing, a price offset, and a T / rate / forward-dividend readout.

**Mobile** [S]
- Option Wizard: you state a view on price or vol, and it lists strategies sortable by aggressiveness, probability of profit, max gain and price. Tapping one shows its Performance Profile and probability of profit. https://www.interactivebrokers.com/campus/trading-lessons/trading-options-using-the-option-wizard-in-impact-app/

**What to take** [J]:
- the scenario cube with slice selection and drill-through;
- the risk headline strip with stale and excluded warnings;
- capital-letter position Greeks and exposure in dollars;
- relative vs absolute vol shifts stated with a worked example;
- the multi-expiry caveat;
- IV error bars from the bid/ask;
- the red what-if frame.

### 1.4 tastytrade: curve and Analysis

Sources: the StockBrokers.com 2026 hands-on review https://www.stockbrokers.com/review/tastytrade; tastylive help text as the search engine indexed it at https://www.tastylive.com/backtest/analysis (the live URL now shows a backtesting page); the 2022 press release https://www.businesswire.com/news/home/20220629005370/en/tastyworks-Adds-Risk-Analysis-Tool-to-Help-Traders-Better-Manage-Risks-in-Their-Positions; and the technology page as indexed at https://tastytrade.com/en-au/technology/ (the live page now shows an Australian closure notice).

**Building a strategy** [S]
- The chain toggles between a table and a "curve" mode, where you drag strikes and expirations on the chart and watch profit and loss zones and probability of profit update.
- The ticket recomputes probability of profit, net Greeks and buying-power effect as you adjust width or roll strikes.
- The chain shows standard-deviation and implied-move markers.

**The P&L diagram** [S]
- The Analysis tab shows a theoretical P/L graph with P/L at expiration and P/L "on the day".
- A red slider moves the date. "Slide by Price" switches it to show how delta, theta and day P/L change with price.
- Checkboxes add delta and theta lines.
- "Advanced" shows cumulative-probability lines and the expected standard deviation of P/L to expiry.

**Portfolio** [S]
- Beta-weighted delta against SPY on the positions page.
- A Portfolio Risk Analysis tool (2022) lets you set the underlying price and IV, change time and vol, and build pricing scenarios. It sits under the portfolio report, reached from CAP REQ.

**Mobile** [S]
- Reviewed as "the gold standard for mobile derivatives trading". It has the chain with curve mode, multi-leg building, POP, net Greeks and buying-power impact on the ticket, and customisable chain columns.
- Mobile charts have no technical indicators.

**What to take** [J]: implied-move markers on the chain and the curve; "slide by date" vs "slide by price" as two modes of one control; POP and Greeks updating live as legs move.

### 1.5 OptionNet Explorer (ONE), Windows desktop

Sources: https://www.optionnetexplorer.com/Explorer.aspx, the walkthrough https://optionstradingiq.com/using-optionnet-explorer-to-study-options-strategies/, the review https://optionstradingiq.com/optionnet-explorer-review/, and user forum threads https://forums.aeromir.com/threads/how-to-look-at-greeks-for-day-steps.1911/latest (via search summary; the forum blocks fetching).

**Building a strategy** [S]
- The chain has a "Model" column where you type quantities (−2 or +2) beside each strike's delta and DTE.
- Position totals under the table: Delta, Gamma, Theta, Vega, and the Theta/Delta ratio.

**The P&L diagram** [S]
- Expiration and T+0 curves.
- "5-line projections" add evenly spaced T+n curves. In the example these are T+7 and T+28, and the "Projection" field sets the last date.
- A Greeks histogram under the risk graph shows Delta and Theta at each underlying price.
- An info panel on the graph lists the options in the position.

**What-if and time** [S]
- Historical "back-trading" from 5-minute snapshots, with buttons that step time forward by hour or day, and auto-play.
- The P&L is tracked across adjustments and commissions.

**Before and after an adjustment** [S]
- The Trade Analysis panel shows the risk change of a proposed adjustment. In the example, delta goes from 16.22 to 7.25 and theta from 3.94 to 8.73.
- P&L is shown as % of max capital at risk.

**Pain points** [S, forum via search]
- Reading the Greeks at a future day step needs a workaround: set T+0 to T+3 and the projection line count to 1.
- Clicking individual curves does nothing.
- Intraday T+0 shows as dots.

**Mobile** [S]: none. It is Windows-only, and on a Mac needs virtualisation.

**What to take** [J]: several labelled T+n projection lines; a Greeks-vs-price strip aligned under the P&L x-axis; before/after deltas for an adjustment; the Theta/Delta ratio as a headline number.

### 1.6 Option Omega (web backtester and modelling)

Sources: the documentation https://docs.optionomega.com/backtesting/backtest-results and https://docs.optionomega.com/modeling/setup-and-use, and the review https://optionstradingiq.com/option-omega-review/.

**Building a strategy** [S]
- Backtest legs are defined by DTE and delta or width.
- Entry conditions: time of day, VIX bands, technical filters.
- Exit conditions: profit target and stop as % or $ of premium, "require two consecutive hits", and timed exits.
- An option to "Ignore Trades with Wide Bid-Ask Spread".
- The Modeling tool builds trades by clicking "Mid (Buy)" or "Mid (Sell)" on the chain.

**The P&L diagram (Modeling)** [S]
- A chart with Greek overlays; the example adds Delta and Charm.
- A 3D toggle.
- In 2D you adjust time, range and IV, and the Greeks show along the bottom.

**Results screen** [S]
- Strategy setup echo, then P/L, CAGR, max drawdown (at end of day, explicitly different from the intraday max loss), MAR, Sharpe, Sortino, win %, capture rate, and so on.
- Tabs: Summary, Trade log, Analyze and Optimizations.
- Trade Replay plots trade value, underlying and VIX on toggleable legends.

**Trust patterns in the Analyze tab** [S]
- A **Sizing Mode** toggle, with every panel badged "PER 1 LOT" or "ACTUAL SIZING".
- A **Strategy Health Verdict**: cards graded green, gold or red (sample size, profit factor, Sharpe, max drawdown, tail risk, recovery factor, edge margin), each with its grading rule on hover.
- Probabilistic Sharpe, labelled "supporting evidence, not proof".
- The Edge Trend tile carries a bootstrap interval.
- The monthly P/L heatmap **dims months with fewer than five trades**.
- Monte Carlo resampling is described as "a stress test of sequencing — not a forecast".

**Mobile**: not documented.

**What to take** [J]: these trust patterns transfer directly to the planned VaR backtests (RK-4): basis badges, sample-size dimming, graded verdict cards with the grading rule on hover, and caveats placed next to the number.

### 1.7 Market Chameleon (web research site)

Sources (secondary only; see the limits above): search-index summaries of https://marketchameleon.com/Learn/option-contract-multi-leg-trades and the reviews https://tradersunion.com/interesting-articles/market-chameleon-review/ and https://thestockdork.com/market-chameleon-review/ (via search).

- It is mainly a research and screening site, not a builder [S, secondary]:
  - a multi-leg trade analyzer that detects and labels spreads traded during the day;
  - "payout diagram" glossary entries;
  - strategy backtests (options history from 2014, per reviews);
  - earnings tools comparing implied and realised moves over many cycles;
  - covered-call and naked-put screeners with 15+ filters.
- I could not verify a P&L curve, T+n lines, what-if sliders or mobile behaviour, so I make no claims about them.

**What to take** [J]: show the implied move next to the historical realised move, as context for any expected-move band.

### 1.8 Sensibull (India, web and mobile)

Sources:
- strike controls: https://blog.sensibull.com/2025/12/17/introducing-quicker-strike-controls/
- payoff table: https://blog.sensibull.com/2023/07/06/payoff-table-on-strategy-builder-analyse-widgets/
- standard-deviation bands: https://blog.sensibull.com/2022/05/10/did-you-know-we-had-standard-deviation-on-strategy-builder/
- user deck: https://mostguide.motilaloswal.com/pdf/client-sensibull/Sensibull_PPT_for_Users.pdf
- user threads: https://tradingqna.com/t/sensibull-payoff-graphs-wrong-on-target-date/92019 and https://tradingqna.com/t/payoff-graph-on-sensibull-is-not-reliable/190141

**Building a strategy** [S]
- Ready-made strategies and a "Strategy Wizard": state a view in plain terms ("NIFTY will not go below 10300") and get strategies to compare.
- **Strike controls** that keep the structure intact:
  - **Shift** moves every leg along the strike ladder, so the payoff keeps its shape;
  - **Width** sets the distance of the body or wings from ATM for straddles, strangles, condors and butterflies, and "does not allow strikes to cross over";
  - **Hedge** moves only the hedge legs ("primary option legs do not move").
  - All three work on mobile.

**The P&L diagram** [S]
- Expiry and target-date payoff, with a Target Date slider for date and time.
- A payoff table with P&L at target levels for the target day and for expiry, a target-interval dropdown, a manual target, and a "Show %" toggle for % moves.
- The same widget appears in the Strategy Builder, the option chain's Analyse widget and the positions page.

**Expected-move bands** [S]
- **Fixed** mode: ±1 and ±2 SD over N days you choose.
- **Dynamic** mode: ±1 and ±2 SD to the selected target date and time.
- The SD values are printed beside the Greeks.
- The probabilities are given as "66%" and 95%, which is inconsistent with 68.27% [J]. The method used to compute SD is not stated [S].

**Position risk summary** [S]
- Strategy and overall Greeks, IV and IV percentile, max pain, PCR, probability of profit.
- **Warnings and insights**: event alerts, high time decay over long weekends and holidays, physical-settlement risk.

**Pain points** [S]
- Users report that the target-date curve "is wrong" when realised P&L differs.
- Sensibull's answer is that the projections are right only if the IV assumptions are right, but the assumptions are not shown on the chart.
- Users also report stale prices and a slow refresh, and a calendar spread whose max loss grew from Rs 300 to Rs 30,000.

**What to take** [J]: Shift, Width and Hedge as three steppers for this app's 26 templates; fixed vs dynamic σ bands; a payoff table with a target interval; plain-language warnings about events and holidays. **What to avoid:** unlabelled assumptions behind the target-date curve.

### 1.9 Opstra (Definedge, India)

Sources (mostly secondary and partly competitor-written): https://www.strike.money/reviews/definedge-opstra, https://algotest.in/blog/best-strategy-builders-for-options-trading-in-india.md and https://algotest.in/blog/opstra-strategy-builder-vs-algotest.md.

- A strategy builder with payoffs, probability outcomes and "break-even zones" [S, secondary].
- Per-leg IV assumptions for multi-expiry structures [S, competitor claim].
- Volatility skew, IV charts and a **surface chart that a competitor calls "difficult to interpret"**, with no fixed-delta view and no clean 2D skew comparison [S, competitor claim].
- An options simulator with intraday replay [S, secondary; disputed by a competitor].
- Described as "functional, but feels dated and requires more clicks" [S, competitor].
- Mobile app rated 2.3/5, citing instability [S, strike.money].
- The site is a single-page app that would not render for fetching, so I could not verify its controls directly.

**What to take** [J]: the per-leg IV override is worth having. The lesson against copying is that a 3-D surface on its own is not readable; pair it with 2-D smile and term-structure slices.

### 1.10 Deribit: Position Builder and Greeks

Sources: the Position Builder article https://support.deribit.com/hc/en-us/articles/31238900906781-Position-Builder (read through the Zendesk API, updated 12 Oct 2026), Portfolio Margin https://support.deribit.com/hc/en-us/articles/25944756247837-Portfolio-Margin, IV orders https://support.deribit.com/hc/en-us/articles/25944809696029-IV-Order-Options and inverse options https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options.

**Building a strategy** [S]
- Add simulated options, futures and equity.
- Log in read-only to load live positions and test hypothetical trades on top of them.
- A Cross vs Segregated (portfolio) margin toggle that "does not change your real margin model".
- It does not trade.

**The P&L diagram** [S]
- P&L against index price change.
- A **chart-metric dropdown**: PnL, Equity, Margin Balance, IM, MM, or any Greek.

**What-if controls, each with its semantics written down** [S]
- **Index Shift**: altcoin indices shift by the same amount.
- **Time Shift**: a slider with start and end dates; options decay and futures converge to the index.
- **IV Shift**: an **absolute** shift for all options, with the example "+10% turns 50% IV into 60%".
- **Annualised Rate Shift**: moves forwards, scaled by T (a +10% shift moves a 30-day forward by 10% × 30/365).
- **"Remove all Overrides"** in one click, or remove them one by one.

**Expired legs** [S]
- Expired options stay in the P&L at their expiry value when the time shift passes their expiry, "by design", so that calendars and rolls stay correct.
- You untick them to exclude them.

**Risk summary** [S]
- Net Delta, Gamma, Vega and Theta under the chart.
- The portfolio-margin risk matrix: 9 price buckets × vol up, same and down, plus an extended table for large moves.

**Greeks and units** [S]
- Inverse options are priced in BTC or ETH with a USD equivalent.
- Displayed IV uses **the forward, not the index**.
- Orders can be entered as a price, a USD value or an **IV**. IV orders requote about once a second and are cancelled 15 minutes before expiry.

**Transparency** [S]
- The help page gives the rate-shift and P&L formulas, and an experimental Position Builder API (https://pb.deribit.com/docs).

**Mobile**: the Deribit app exists, but I did not verify Position Builder on mobile.

**What to take** [J]:
- one control strip with the semantics written beside each control (absolute IV, T-scaled rate, how expired legs are handled);
- a chart-metric dropdown;
- "Remove all overrides";
- IV against the forward;
- prices and orders entered in vol.

### 1.11 Cboe

Sources: https://www.cboe.com/education/tools, https://www.cboe.com/education/tools/options-calculator, https://www.cboe.com/education/tools/trade-optimizer and https://www.optionseducation.org/news/cboe-vest-technologies-and-the-options-industry-co.

- The current free tools are the **Options Calculator** (theoretical price and Greeks from inputs or a symbol, using Cboe data) and the **LiveVol Trade Optimizer**. The Optimizer takes a target price and date and ranks strategies by return on investment [S].
- The **"Options Strategy Builders"** (CBOE Vest Technologies with OIC, August 2017) covered only collars and covered calls, priced off the previous close: you pick the builder, symbol and expiry, and the tool "displays the potential results". It looks retired [S, 2017 release].
- I found no current, documented Cboe multi-leg strategy builder with T+n curves or what-if sliders. I make no claims beyond this.

**What to take** [J]: the "target price and date → ranked strategies" pattern (shared with the OptionStrat Optimizer and IBKR Probability Lab) is a possible later feature, not a core one, for a desk tool.

### 1.12 Adjacent references (not in scope, but useful evidence)

- **moomoo** P/L curve [S] https://www.moomoo.com/us/learn/options-p-l-curve-tutorial-simulate-your-options-trade-to-visualize-potential-profit-and-loss:
  - each overlay is labelled **"fixed" or "variable"**: Expiration P/L and Expected Move are fixed; Current P/L, Greeks and Price Probability vary with the date and IV sliders;
  - the IV basis can be the option IV, the underlying IV or historical volatility;
  - legends toggle overlays.
- **TradingView** options builder [S] https://tradingview.com/support/solutions/43000707214-options-strategy-builder-overview:
  - P&L on the left y-axis and Greeks on the right;
  - an eye icon excludes a leg;
  - the Adjust menu offers Set prices (bid, ask, mid or theoretical), Scale (a negative factor inverts the strategy) and Shift by N strikes.

---

## 2. Comparison matrix

Y = documented, P = partial or premium-only, - = absent or not found, ? = could not verify.

| Capability | OptionStrat | ToS | IBKR | tasty | ONE | Opt.Omega | MktCham | Sensibull | Opstra | Deribit PB | Cboe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Templates gallery with shapes | Y | P (right-click spreads) | Y (predefined, Wizard) | ? | - | P | ? | Y | Y | - | P (2 strategies) |
| Add legs by clicking chain bid/ask | P (strike ruler) | Y | Y | Y | Y (Model column) | Y (Mid Buy/Sell) | ? | P (Analyse widget on chain) | ? | P (instrument picker) | - |
| Drag strikes / keep structure | Y (drag, Shift-drag) | - | P (scroll-wheel width) | Y (curve mode) | - | - | ? | Y (Shift/Width/Hedge) | ? | - | - |
| Expiry + T+0 curves | Y (one day at a time) | Y | Y (earliest expiry) | Y | Y | P (2D time control) | ? | Y | ? | P (time-shift slider) | ? |
| Several T+n curves at once | - | Y | Y (date scenarios) | - | Y (projections) | ? | ? | - | ? | - (slider) | - |
| Date × price P&L table | Y (heatmap) | P (price slices) | P (Scenarios window) | - | - | - | ? | Y (payoff table) | ? | - | - |
| Spot × vol scenario grid | - | P (Uniform Stress Test) | Y (7×5×5 cube) | - | - | - | ? | - | ? | Y (PM risk matrix) | - |
| Greek-vs-spot chart | - | ? | Y (dropdown) | Y (delta and theta lines) | Y (histogram) | Y (overlays, 3D) | ? | - | ? | Y (metric dropdown) | - |
| Expected-move or σ bands | Y (distribution overlay, premium) | Y (cone, slices by σ) | Y (CI brackets, Prob Lab) | Y (chain markers) | ? | - | Y (implied vs realised) | Y (fixed and dynamic SD) | ? | - | - |
| Probability of profit | P (premium) | Y | Y | Y | ? | - (win % ex post) | ? | Y | P ("probability outcomes", secondary) | - | - |
| IV shift semantics stated | P (one absolute IV) | P | Y (relative 15%) | ? | ? | ? | ? | ? | ? | Y (absolute, example) | - |
| Rate shift | - | - | Y (±100 bp) | - | - | - | - | - | - | Y (T-scaled) | - |
| Position Greeks in $ / exposure | - | P (beta-weighted) | Y | Y (beta-weighted delta) | P | - | - | P | ? | P (coin) | - |
| Risk headline (VaR, ES, margin) | - | - | Y | P (BP) | P (margin) | Y (backtest stats) | - | P (ROI, max loss) | ? | Y (IM, MM) | - |
| Modified or hypothetical state visible | P (edit toggle) | - | Y (red frame, red edits) | - | - | - | - | - | - | Y (Remove overrides) | - |
| Assumptions written down | Y (FAQ) | P | Y (scenario method) | - | - | Y (metric defs) | - | - | - | Y (formulas) | - |
| Mobile analytics | Y | ? | P (Wizard) | Y | - (Windows) | ? | ? | Y | P (2.3/5) | ? | - |

---

## 3. The app today, in brief (for the mapping in sections 4 and 6)

From `options_analyzer/ui/tab_strategy.py` and the plan of record (read only):

- The Strategy tab already has the following [S, code]:
  - a template selector and a legs `data_editor`;
  - stat and Greek cards;
  - a payoff chart with "At expiry", "Today" and one "In N days" curve;
  - an IV-shift slider in absolute vol points, whose help text says it applies to the pre-expiry curves only;
  - a ±1σ log-normal band, breakevens, and a CSV download.
- Planned work includes:
  - RK-1: `position_risk`, `greek_ladder` and `scenario_cube` with `vol_mode='relative'|'absolute'|'1m'`;
  - RK-2: vega buckets and 1M-weighted vega;
  - RK-3: P&L explain;
  - RK-4: VaR backtests;
  - BK-1: risk headline strip and report selector;
  - BK-2: Strategy Builder desk layout;
  - BK-3: leg IVs from the fitted surface;
  - SF-8: flagged wing extrapolation;
  - DK-4: number formats.

So most of the ranked features below are UI on top of planned engines [J].

---

## 4. The 25 most valuable features and interaction patterns for this app, ranked

The ranking is my judgement [J]. It weighs what a market maker or vol PM needs (precision, density, trust) against how feasible it is in Streamlit (Plotly plus native widgets; no native drag-on-chart). Each item gives its source evidence and where it would live in the app.

1. **One what-if strip whose semantics are written beside each control.** It covers spot shift, date (T+n), vol shift (absolute vol points or relative %, toggled and labelled with a worked example, such as "+10 pts: 25% → 35%") and rate shift (bp, scaled by T). It has a "Reset all overrides" button. While any override is non-zero, a visible "SCENARIO" state shows (a tinted frame or header chip).
   - Sources: Deribit Index, Time, IV and Rate Shift plus "Remove all Overrides" [S]; IBKR red what-if frame, edits red until Apply, Reset Scenario [S]; IBKR relative "Vol-up 15%" with its 10% → 11.5% example [S].
   - App: Strategy and Risk tabs (BK-2, RK-1 `vol_mode`).
2. **A payoff chart with expiry, T+0 and several labelled T+n lines.** Each line is labelled with its date and DTE. The legend marks which lines move with the sliders ("fixed" vs "variable"). Spot is a dashed vertical line, and breakevens are labelled verticals with their % from spot.
   - Sources: ONE projection lines [S]; ToS lines for several dates [S]; moomoo fixed/variable labels [S]; OptionStrat breakeven and spot rules [O].
   - App: the existing payoff chart, extended from one dated curve to N.
3. **A date × spot P&L heatmap table.**
   - Rows: spot levels, each with its % move; a rule at today's spot.
   - Columns: dates with weekday letters; an expiry-open and an expiry-close column; event icons.
   - Cells: diverging, magnitude-graded colour.
   - A range control (±x% or ±kσ) and a unit toggle: $ P&L, % of debit, % of max risk, value.
   - Sources: OptionStrat table and unit toggle [S][O]; Sensibull payoff table with target interval and "Show %" [S].
   - App: Strategy (BK-2's P&L table and Range control).
4. **A spot × vol scenario matrix from a 3-D cube.**
   - Pick which two axes to show, and fix the third (days, or rate).
   - Click a cell to "apply this scenario" to the other views. They then show a scenario banner until reset.
   - Grade cells by magnitude, not sign alone.
   - Sources: IBKR Measure by Price and Vol Change (7×5×5) with drill-through and Reset Scenario [S]; IBKR binary red/green fill [O] as the thing not to copy.
   - App: Risk (RK-1 `scenario_cube`, BK-1 report selector).
5. **A metric selector on the spot-axis chart, P&L or any Greek.** Use dual axes, or a Greek strip under the P&L x-axis. Greeks at the selected T+n date, not only today.
   - Sources: IBKR Performance Graph and Probability Lab dropdowns [S]; Deribit chart metric including IM, MM and Greeks [S]; ONE Greeks histogram [S]; tastytrade delta and theta lines with "Slide by Price" [S]; TradingView right axis [S]; Option Omega Greek overlays, Delta and Charm [S].
   - App: Strategy and Risk (RK-1 Greek ladder, third-order Greeks).
6. **Position Greeks in desk units, named consistently.**
   - $Δ (and share-equivalent), $Γ per 1%, vega per vol point, 1M-weighted vega, theta per day.
   - A capital-letter or "position" naming convention to tell them from per-option Greeks.
   - Units printed in every header.
   - Sources: IBKR position D, G, V, T and Exposure = pos × Δ × mult × S [S]; Deribit coin vs USD [S].
   - App: cards and ladders (RK-1, RK-2, DK-4).
7. **A risk headline strip pinned above every Risk report.**
   - P&L, net $Δ, $Γ, vega (raw and 1M), θ, VaR, ES and margin, if margin is ever added.
   - A stale or recompute indicator, and a warning chip listing legs excluded from a measure.
   - Sources: IBKR Risk Dashboard [S].
   - App: Risk (BK-1).
8. **Expected-move overlays with their basis stated.**
   - ±1σ and ±2σ bands, either to a chosen date ("dynamic") or over N days ("fixed").
   - Optionally the terminal distribution shaded under the curve, with P(below) and P(above) at the hover point.
   - Label the vol used (ATM IV, surface, or a user value) and use 68.3%, not "66%".
   - Sources: Sensibull fixed and dynamic SD [S]; OptionStrat distribution overlay [S][O]; ToS probability cone with an editable probability [S]; IBKR confidence-interval brackets [S].
   - App: Strategy payoff (it already has a 1σ band).
9. **A stat strip carrying probabilities and caveats.**
   - Net debit or credit, max profit and max loss each with its probability (IBKR "Max Return … (<1% chance)"), breakevens with % from spot, POP, return/risk.
   - For multi-expiry structures, a visible caveat that the stats are at the front expiry and later losses can exceed them.
   - Sources: IBKR Performance Profile [S][O]; OptionStrat stat strip and FAQ caveats [S][O].
   - App: Strategy cards.
10. **Strike controls that keep the structure intact.** Shift (all legs ±N strikes), Width (body or wing distance, never crossing) and Hedge (wings only), as compact steppers, plus strike-by-delta entry.
    - Sources: Sensibull Shift, Width and Hedge [S]; OptionStrat Shift-drag [S]; IBKR scroll-wheel width [S]; TradingView Shift by N strikes [S].
    - App: Strategy (BK-2, RK-9 strike by delta).
11. **A template gallery with a mini payoff for each template.** Group by outlook (bullish, bearish, neutral, vol long or short) and by risk (defined or undefined), with a one-line description.
    - Sources: OptionStrat hover shapes and grouping [S]; Sensibull ready-mades [S].
    - App: Strategy selector over the 26 templates.
12. **Build legs from the chain.** On the synthetic or uploaded chain, a click on bid adds a sell and a click on ask adds a buy. Show bid, ask, mid, IV and delta per strike, and highlight legs already in the position.
    - Sources: ToS Add Simulated Trades [S]; IBKR Strategy Builder [S]; Option Omega "Mid (Buy)/Mid (Sell)" [S]; TradingView B/S on hover [S].
    - App: Strategy and the Data chain (DT-4, DT-5). Streamlit can use `st.dataframe` row selection plus Buy/Sell buttons [J].
13. **An include/exclude toggle per leg.** It drops the leg from the calculations without deleting it, and dims it.
    - Sources: OptionStrat Include/Exclude [S]; TradingView eye [S]; Deribit unticking expired legs [S].
    - App: an "On" column in the legs editor.
14. **Before → after for any edit.** Show a ghost line of the previous payoff, and a delta table of the risk change (Δ, Γ, vega, θ, VaR: before, change, after).
    - Sources: OptionStrat grey old-trade line and toggle [S]; ONE Trade Analysis 16.22 → 7.25 [S]; IBKR Order Preview Current, Change and Post-trade [O].
    - App: Strategy and Risk.
15. **Price basis for entry cost.** Mid, natural (ask for buys, bid for sells), theoretical or custom per leg, with custom prices flagged. Show the spread cost in $ next to the net premium.
    - Sources: OptionStrat mid vs bid/ask and the cost-basis "$" flag [S][O]; TradingView Set prices [S]; ONE fill slider between bid and mid [S].
    - App: legs editor.
16. **Vol-coordinated scenarios beside "vol unchanged" scenarios.** Spot-down/vol-up rules with a term-structure decay, plotted as separately labelled curves, with the rule in a tooltip.
    - Sources: IBKR Vol.Coord. (Y = −X / −10X, VR(t)) and its "not predictive" caveat [S].
    - App: Risk stress tests (RK-1, RK-7).
17. **An IV smile per expiry with bid/ask error bars.** Call and put markers, distinct markers for half quotes and stale quotes, and the fitted SVI curve over them. Extrapolated wings dashed.
    - Sources: IBKR Model Navigator [S][O]; the competitor criticism of Opstra's surface-only view [S].
    - App: the Vol surface and IV tabs (SF-3, SF-4, SF-8).
18. **Model and convention disclosure next to the output.**
    - The model name and why it was chosen; the time-of-day convention for each date column; constant IV vs surface-implied leg IVs; how expired legs are treated.
    - Keep it to one line under the chart, with the Methodology tab linked.
    - Sources: OptionStrat FAQ (Bjerksund-Stensland; open, now and close times; constant IV) [S]; Deribit formulas and expired-leg policy [S]; IBKR scenario methodology [S]; IBKR's status footer under the P&L graph ("Reference currency: USD, Beta Reference: SPX Index, Included Equity Positions: 11") [O].
    - App: every chart (DK-6 methodology drawer, DT-1 clock).
19. **Event markers on every date axis.** Ex-dividend, expiries, holidays and weekends on the table columns and the date slider, with a statement of whether the model includes them.
    - Sources: OptionStrat dividend and earnings icons, and "not accounted for" [S][O]; Sensibull long-weekend and holiday warnings [S].
    - App: Strategy table and date slider (DT-1 clock, dividend schedule).
20. **Plain-language risk warnings on the position.** Undefined risk, early-exercise or assignment risk on short ITM American legs near ex-dividend, pin risk at expiry, a wide-spread leg, wing extrapolation in use.
    - Sources: Sensibull Warnings and Insights [S]; OptionStrat liquidity warnings [S]; IBKR multi-expiry caveat [S].
    - App: a Strategy warnings row (BK-4 pin risk, SF-8 extrapolation flag).
21. **Make delta neutral.** One button adds or resizes an underlying leg from the position delta. Editing the delta target or the stock quantity updates the other.
    - Sources: IBKR "Make Delta Neutral" [S]; Probability Lab "Delta Neutral" [S].
    - App: legs editor.
22. **Entry and readout in vol as well as price.** Every option price input has a twin IV field (via Let's Be Rational), and readouts show both. Note which underlying the IV uses (forward vs spot).
    - Sources: IBKR OptionTrader quoting in volatility terms [S]; Deribit IV orders and IV against the forward [S].
    - App: Pricing and IV tabs, legs editor (PR-1, SF-3).
23. **Honest statistics for backtests.** A badge on every panel stating the basis (per lot vs actual size, or the confidence level). Cells with too few observations dimmed. Graded verdict cards (green, gold, red) with the grading rule on hover. Uncertainty intervals on the headline numbers.
    - Sources: Option Omega Sizing Mode badges, months with fewer than 5 trades dimmed, Strategy Health Verdict, bootstrap interval [S].
    - App: VaR backtests (RK-4) with Kupiec and Christoffersen traffic lights.
24. **Shareable, reproducible state.** URL-bound position and scenario, a "copy link" action, and a CSV or JSON export of the exact inputs and outputs. Optionally, saved positions with a compare toggle.
    - Sources: OptionStrat share and save via URL and the compare toggle [S]; Option Omega share link and saved backtests [S]; IBKR Save As and reload of what-if files [S].
    - App: the header Share link (it exists) and the CSV download (it exists for the payoff).
25. **Time-shift handling of expired legs, and expiry-day resolution.**
    - When the date passes a leg's expiry, keep its settled P&L, mark it "expired", and allow excluding it.
    - On expiry day, show open and close (or intraday) columns, not one point.
    - Sources: Deribit expired-leg policy [S]; OptionStrat extra expiry-open column [S]; ONE users missing intraday T+0 on expiry day [S, forum via search].
    - App: Strategy calendars and diagonals (DT-1 settlement times).

Close but not in the top 25 [J]:

- a "Probability Lab" style custom distribution;
- an optimizer that ranks strategies for a target price and date (OptionStrat, Cboe Trade Optimizer, IBKR Wizard);
- beta-weighting across underlyings (ToS, IBKR, tastytrade). It matters little in a single-underlying tool.

---

## 5. The 10 most common UX mistakes these tools make

1. **T+n curves whose assumptions are not shown.**
   - Constant IV, sticky strike or sticky delta, time of day, and dividend treatment are missing from the chart. Users then read a model projection as a forecast and call it wrong.
   - Evidence: Sensibull user threads ("payoff graphs wrong on target date", "not reliable"), where the answer is "correct only if IV assumptions are right" [S]. OptionStrat counters this by writing the assumptions in its FAQ, but not on the chart [S].
2. **Vol shifts with unstated semantics.**
   - Absolute vs relative, one IV for a multi-leg position, and whether skew or term structure moves are often not said.
   - Two professional tools pick opposite conventions: Deribit uses absolute (+10%: 50 → 60) and IBKR relative (+15%: 10 → 11.5) [S]. Both have to explain them with worked examples.
   - OptionStrat shows one absolute IV for the whole position [O]. For a multi-leg trade that position is ambiguous [J].
3. **Expiry-only or front-expiry statistics for multi-expiry structures.**
   - Max loss and breakevens computed at the front expiry, under clean lines, mislead on calendars and diagonals.
   - Evidence: IBKR's own warning ($411 shown, $2,545 after the front expiry) [S]; a Sensibull calendar max loss growing from Rs 300 to Rs 30,000 [S].
4. **Probabilities with no stated basis.**
   - POP, chance of profit and σ bands appear without the distribution or vol they use. Some show "66%" for ±1σ (Sensibull) while others show 68.27% (ToS) [S].
   - OptionStrat's chance of profit is only "based on the implied move" and premium-only [S]. IBKR shows "Market Implied" vs "My Forecast" explicitly, which is the counter-example [S].
5. **Staleness and hypothetical state that cannot be seen.**
   - Stale prices or recalculations with no indicator: Sensibull users report stale LTP and slow refresh [S]; Deribit IV orders can be stale in fast markets [S].
   - The good counter-examples are IBKR's refresh icon "only when … not reflected", the asterisk on unsaved what-ifs, and the red frame [S].
6. **Ambiguous units and scale.**
   - Per-share vs per-contract vs whole position, coin vs USD, Greeks per 1.00 vs per 1%.
   - ONE quotes "a credit of $1.03 for each contract on a per-share basis" [S]. OptionStrat's history chart shows the per-unit contract price, not the position [S]. IBKR needs a capital-letter convention to tell position Greeks from option Greeks [S].
   - Mixed units on one screen are a recurring source of misreads [J].
7. **Colour carrying meaning on its own, or carrying only the sign.**
   - The IBKR scenario matrix fills cells binary red or green by sign, with no magnitude [O].
   - The IBKR vol-up and vol-down curves can only be told apart by colour, and they overlap [O].
   - ToS's own documentation contradicts itself on line colours across themes (blue vs cyan vs green; purple vs magenta) [S].
   - Red/green alone also fails colour-blind users [J].
8. **Greeks shown only for "now" and "here".**
   - Most tools show net Greeks at today and current spot. Reading Greeks at a future date or another spot takes workarounds.
   - Evidence: ONE users changing the T+0 offset and projection count to read Greeks at T+3, with clicks on curves doing nothing [S, forum via search]; OptionStrat's Greeks shown only as a second page of net numbers [S].
   - tastytrade "Slide by Price" and the IBKR, Deribit and Option Omega Greek charts are the exceptions [S].
9. **3-D charts with no readable 2-D slices.**
   - A surface-only IV view is "difficult to interpret" (competitor review of Opstra) [S], and 3-D P&L toggles (Option Omega) look impressive but cannot be read precisely [J].
   - Desks read smiles per expiry and term structure per delta or moneyness. 3-D should be the overview, never the only view [J].
10. **Mobile treated as an afterthought, or crammed.**
    - ONE is Windows-only [S]. ToS mobile does not document Risk Profile [S]. Opstra's app is rated 2.3/5 for instability [S]. tastytrade mobile charts have no indicators [S].
    - The good mobile tools (tastytrade curve mode, Sensibull strike controls, OptionStrat swipe pages) keep **one** chart and **one** control set per screen and move secondary stats to swipeable pages [S].

Close but not in the top 10 [S]:

- powerful features hidden behind support or settings: ToS Uniform Stress Test needs a chat request, and some Analyze tools need "Advanced Features" enabled;
- "too many clicks" (Opstra, per a competitor).

---

## 6. What this means for this app (short, judgement)

All items in this section are [J].

- **Strategy tab (BK-2):**
  - extend the payoff chart to N labelled dated curves;
  - put the date × spot heatmap table (feature 3) beside the chart, with one Range control driving both;
  - add Shift, Width and Hedge steppers (feature 10) and an include/exclude column (feature 13);
  - write the vol-shift semantics into the slider label itself, not only its help text. Today the help text does say absolute points and pre-expiry curves only, which is good;
  - make the 1σ band's basis explicit in the legend ("±1σ, ATM IV 21.4%, to 15-Nov").
- **Risk tab (BK-1):**
  - a pinned headline strip (feature 7);
  - a report selector;
  - the scenario-cube matrix with axis pickers and an "apply scenario" banner (feature 4);
  - vol-coordinated alongside vol-unchanged stress curves (feature 16);
  - magnitude-graded diverging colour with the numbers always printed.
- **Pricing and IV tabs:** price ↔ vol twin inputs (feature 22). Show IV against the forward once SF-3 gives forwards.
- **Vol surface:** a per-expiry smile with bid/ask error bars, fitted and extrapolated segments styled differently, and a term-structure slice. 3-D is secondary (feature 17, mistake 9).
- **Everywhere:**
  - a one-line assumptions footer under each chart (feature 18);
  - units in every header (feature 6, DK-4);
  - a visible "SCENARIO" state with Reset (feature 1);
  - URL-bound state for sharing (feature 24).
- **Phones:** one chart and one control set per screen; the stat strip as a horizontally scrolling or wrapping row; the heatmap table reduced to fewer date columns (front, mid and expiry) with a "more dates" disclosure.

---

## 7. Notes on method

- Fetched pages and extracted text are kept under `raw/` for traceability. Helper scripts are in `bin/`.
- Pages that blocked automated fetching were not worked around:
  - Market Chameleon (access denied, and crawling is prohibited by its Terms of Use);
  - Sensibull blog and Deribit support HTML (read through WebFetch and the Zendesk API instead);
  - tradersfly, glasp, aeromir (left as search-summary evidence, marked as such).
- Where a claim comes only from a search-engine summary or a competitor, it is marked as such inline.

---

## 8. Sources

OptionStrat
- https://optionstrat.com/tutorials/options-builder
- https://optionstrat.com/blog/options-profit-calculator-tutorial
- https://optionstrat.com/faq
- https://apps.apple.com/us/app/optionstrat-options-toolkit/id1541714905

thinkorswim / Schwab
- https://toslc.thinkorswim.com/center/howToTos/thinkManual/Analyze
- https://toslc.thinkorswim.com/center/faq/Analyze
- https://www.schwab.com/learn/story/analyze-trades-using-risk-profile-on-thinkorswim
- https://tickertape.tdameritrade.com/amp/trading/thinkorswim-options-risk-profile-tool-18671 (search-index summary; domain offline)
- https://bearbulltraders.com/wp-content/uploads/2023/09/Unlocking-Your-Options-Trading-Potential-using-ToS.pdf

Interactive Brokers
- https://www.ibkrguides.com/traderworkstation/risk-navigator.htm
- https://www.ibkrguides.com/traderworkstation/what-if-portfolios.htm
- https://www.ibkrguides.com/traderworkstation/risk-dashboard.htm
- https://www.ibkrguides.com/traderworkstation/portfolio-relative-pnl-graph.htm
- https://www.ibkrguides.com/traderworkstation/scenario-analysis.htm
- https://www.ibkrguides.com/traderworkstation/risk-nav-reports.htm
- https://www.ibkrguides.com/traderworkstation/risk-nav-trader-personalities.htm
- https://www.ibkrguides.com/traderworkstation/understanding-risk-navigator.htm
- https://www.ibkrguides.com/traderworkstation/model-navigator.htm
- https://www.ibkrguides.com/traderworkstation/optiontrader.htm
- https://www.ibkrguides.com/traderworkstation/performance-profile.htm
- https://www.ibkrguides.com/traderworkstation/strategy-performance-graph.htm
- https://www.ibkrguides.com/traderworkstation/probability-labsm.htm
- https://www.interactivebrokers.com.hk/php/webhelp/navigator/reportMeasurePricevol.htm
- https://www.interactivebrokers.com.hk/php/webhelp/navigator/reportUnderlying.htm
- https://www.interactivebrokers.com.hk/php/webhelp/navigator/reportMeasureUnderlying.htm
- https://www.interactivebrokers.com.hk/php/webhelp/navigator/reportPortfolio.htm
- https://www.interactivebrokers.co.uk/en/software/tws.bak/usersguidebook/priceriskanalytics/open_the_ib_risk_navigator.htm
- https://www.interactivebrokers.com/en/trading/options-trading-tools.php
- https://www.interactivebrokers.com/campus/trading-lessons/trading-options-using-the-option-wizard-in-impact-app/ (search-index summary)

tastytrade
- https://www.stockbrokers.com/review/tastytrade
- https://www.tastylive.com/backtest/analysis (search-index text of the former Analysis help page)
- https://www.businesswire.com/news/home/20220629005370/en/tastyworks-Adds-Risk-Analysis-Tool-to-Help-Traders-Better-Manage-Risks-in-Their-Positions (search-index summary)
- https://tastytrade.com/en-au/technology/ (search-index summary; the live page has changed)

OptionNet Explorer
- https://www.optionnetexplorer.com/Explorer.aspx
- https://optionstradingiq.com/using-optionnet-explorer-to-study-options-strategies/
- https://optionstradingiq.com/optionnet-explorer-review/ (search-index summary)
- https://forums.aeromir.com/threads/how-to-look-at-greeks-for-day-steps.1911/latest (search-index summary)

Option Omega
- https://docs.optionomega.com/backtesting/backtest-results
- https://docs.optionomega.com/modeling/setup-and-use
- https://optionstradingiq.com/option-omega-review/

Market Chameleon (secondary only)
- https://marketchameleon.com/Learn/option-contract-multi-leg-trades (search-index summary; direct access denied)
- https://tradersunion.com/interesting-articles/market-chameleon-review/ (search-index summary)
- https://www.thestockdork.com/market-chameleon-review/ (search-index summary)

Sensibull
- https://blog.sensibull.com/2025/12/17/introducing-quicker-strike-controls/
- https://blog.sensibull.com/2023/07/06/payoff-table-on-strategy-builder-analyse-widgets/
- https://blog.sensibull.com/2022/05/10/did-you-know-we-had-standard-deviation-on-strategy-builder/
- https://mostguide.motilaloswal.com/pdf/client-sensibull/Sensibull_PPT_for_Users.pdf
- https://tradingqna.com/t/sensibull-payoff-graphs-wrong-on-target-date/92019
- https://tradingqna.com/t/payoff-graph-on-sensibull-is-not-reliable/190141

Opstra
- https://www.strike.money/reviews/definedge-opstra
- https://algotest.in/blog/best-strategy-builders-for-options-trading-in-india.md
- https://algotest.in/blog/opstra-strategy-builder-vs-algotest.md (competitor)

Deribit
- https://support.deribit.com/hc/en-us/articles/31238900906781-Position-Builder
- https://support.deribit.com/hc/en-us/articles/25944756247837-Portfolio-Margin
- https://support.deribit.com/hc/en-us/articles/25944809696029-IV-Order-Options
- https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options

Cboe / OIC
- https://www.cboe.com/education/tools
- https://www.cboe.com/education/tools/options-calculator
- https://www.cboe.com/education/tools/trade-optimizer
- https://www.livevol.com/analytics-platforms/tutorials/
- https://www.optionseducation.org/news/cboe-vest-technologies-and-the-options-industry-co

Adjacent
- https://www.moomoo.com/us/learn/options-p-l-curve-tutorial-simulate-your-options-trade-to-visualize-potential-profit-and-loss
- https://tradingview.com/support/solutions/43000707214-options-strategy-builder-overview

---

## 9. Screenshots

Vendor images downloaded from public help pages, kept for internal reference only. They are in `screenshots/` next to this file.

| File | What it shows |
|---|---|
| `optionstrat_tutorial_12.png` | The whole builder: strike ruler, stat strip, date × price heatmap with % column and spot rule, expiry open and close columns, Chart Range ±%, single IV slider, Table/Graph toggle, unit toggle |
| `optionstrat_tutorial_11.png` | Header: symbol, expiry ribbon by month with DTE, strike chip with cost-basis "$" flag, unrealized gain |
| `optionstrat_tutorial_5.png` | Heatmap table close-up: magnitude-graded diverging colour, % move column, dividend icon on a date |
| `optionstrat_tutorial_6.png` | Single-day graph: distribution overlay, spot dashed, breakeven labelled, DTE slider with date |
| `optionstrat_tutorial_3.png` | Strike ruler: long above and short below, call and put volume bars, spot triangle |
| `optionstrat_tutorial_4.png` | Net Greeks page with coloured Greek letters |
| `optionstrat_tutorial_1.png`, `_2`, `_8`, `_9`, `_10` | Other tutorial frames: overview, symbol search, leg menu, history candles, saved trade |
| `ibkr_performance_profile.png` | Strategy Builder, Quote Details with probabilities of max return and max loss, Performance Graph, Order Preview with Current, Change and Post-trade margin |
| `ibkr_probability_lab.png` | Market vs custom distribution, strategy scanner, P&L over the distribution on dual axes |
| `ibkr_measure_price_vol.png` | Spot × vol scenario matrix with binary red/green fill (shows mistake 7) |
| `ibkr_relative_pnl_volupdown.png` | Vol +15%, vol −15% and vol-coordinated curves, Historical VaR marker, crosshair tooltip |
| `ibkr_relative_pnl_volcoord.png` | Vol-coordinated curve with a Historical VaR marker, and a provenance footer ("Reference currency: USD, Beta Reference: SPX Index, Included Equity Positions: 11") |
| `ibkr_model_navigator_volcurve.png` | IV smile with bid/ask error bars, call and put markers, IV table |
