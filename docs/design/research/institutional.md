# Institutional options and volatility tools: interface research ("institutional")

Research for the Option-Pricing-and-Strategy-Analyzer UI. The goal is an options pricing, strategy and risk workstation that a market maker or volatility PM would find precise, fast, dense, calm and trustworthy. Researched 2026-10-26. Nothing under the repository was modified.

## How to read these notes

- **[S]** marks a sourced fact. The link next to it is where it comes from.
- **[S, excerpt]** means the page refused automated fetches (bloomberg.com, techatbloomberg.com, studylib, academia.edu and the IBKR help pages all returned 403 or a bot check). The claim rests on the search engine's excerpt of that page, not on a full read.
- **[J]** marks my own judgement, or my recollection of a product that I could not check against a public source. Treat [J] items about Bloomberg screen layouts as unverified.
- Screenshots: Bloomberg, LiveVol, ORATS and LSEG screens sit behind logins, so I could not capture them. I saved two public example fit charts from Vola Dynamics' examples page for reference. They are in /tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/institutional/screenshots/ (`vola_SPX_20180227-160000_i3_C14pm_K.png` and `vola_NVDA_20250403-094500_i2_C15MW_NS.png`). They are third-party images kept for internal reference only.
- Primary documents I downloaded and read are in /tmp/claude-0/-home-user-Option-Pricing-and-Strategy-Analyzer/c229f07d-6d3a-5829-b5e9-faaf587d7e00/scratchpad/design/institutional/dl/: the Bloomberg MARS brochure, the OptionMetrics IvyDB flyers, Derman (1999) and Mixon (2011).

---

## 1. Bloomberg Terminal

### 1.1 Interaction model and design writing

- **Function-per-screen model.** Every screen is a security context plus a mnemonic: `<ticker> <yellow key> <FUNCTION> <GO>`. The option functions hang off a contract's menu. [S] The HBS library says to "Left-click a contract ticker to see a list of functions, then select OSA" ([HBS](https://www.library.hbs.edu/services/help-center/bloomberg-options)). The Xavier guide says OV is reached from an option's menu ([Xavier guide, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/11_derivativesoptionsfuturesswaps.pdf)).
- **Input fields are amber.** [S] OSA "allows 'what-if' analyses by changing assumptions in amber fields" ([HBS](https://www.library.hbs.edu/services/help-center/bloomberg-options)). "Amber is our base font color", and the black-and-amber look is a deliberate identity ([Bloomberg, customer-centric design ethos, excerpt](https://www.bloomberg.com/company/stories/bloombergs-customer-centric-design-ethos)).
  - [J] This is the convention that matters most for us: the user can tell at a glance which numbers they typed or can override, and which the system computed.
- **Hierarchy within density.** [S] Bloomberg's UX staff state the aim as "creating a clear hierarchy, making sure users can find and parse information as quickly as possible". They also say "the tight, information-dense screen makes clarifying the hierarchy challenging" ([Bloomberg design ethos, excerpt](https://www.bloomberg.com/company/stories/bloombergs-customer-centric-design-ethos)).
- **Concealing complexity, and change management.** [S] In 2022 the CTO said "We're hiding complexity". The same article notes that even a font swap or a moved button can disrupt users ([How Bloomberg Terminal UX designers conceal complexity, excerpt](https://www.bloomberg.com/company/stories/how-bloomberg-terminal-ux-designers-conceal-complexity)).
  - [S] An older outside account says Bloomberg's leadership insisted on being "religiously consistent" in look and feel, and turned down a 2007 IDEO redesign ([UX Magazine](https://uxmag.com/articles/the-impossible-bloomberg-makeover)).
- **Colour accessibility.** [S] Bloomberg estimates "20,000 or more" Terminal users have red-green colour vision deficiency (CVD) ([Designing the Terminal for color accessibility, Oct 2021, excerpt](https://www.bloomberg.com/ux/2021/10/14/designing-the-terminal-for-color-accessibility/)). Further findings from the same article:
  - It used a deuteranopia simulator, which showed that the red and green up/down indicators "appear very similar in hue".
  - Users with CVD still read colour as direction: "Greens and blues tended to be self-reported as denoting an 'up' market status, while reds, yellows, and oranges tended to denote a 'down' market status."
  - The Terminal ships alternate schemes for deuteranopia and protanomaly, switched with `PDFU COLORS <GO>` ([Bloomberg UX accessibility, excerpt](https://www.bloomberg.com/ux/accessibility/)).
- **Speed is the real density.** [S] Strom's essay on UI density names the Terminal as the standard example of visual density, but argues its "real superpower" is loading with no latency. It defines UI density as "the value a user gets from the interface divided by the time and space the interface occupies". Its time thresholds:

  | Wait | Treatment |
  |---|---|
  | Under 100 ms | Feels instant; extra animation slows it down |
  | 0.1 to 1 s | Visual cues |
  | 1 to 10 s | Indeterminate loader |
  | Over 10 s | Progress bar |

  Source: [Strom, UI Density](https://mattstromawn.com/writing/ui-density).

### 1.2 OVME (Option Valuation, Equity)

- **What it is.** [S] Bloomberg's own webinar copy calls it "Options Pricer – Price volatility your way. Customizable models, real-time Greeks, and scenario testing turn complexity into control" ([Bloomberg webinar page](https://professional.content.cirrus.bloomberg.com/professional2023/insights/webinar/inside-volatility-trading-hedging-positioning-winning-with-options/)).
  - [S] For underlyings with a listed options market, OVME uses "interpolated market-implied volatilities" from the surface, so a single quote is priced off the same surface as everything else ([Bloomberg insights, excerpt](https://www.bloomberg.com/professional/insights/trading/navigating-derivatives-market-sentiment-with-volatility-and-correlation-analysis/)).
  - [S] It is the equity single-security calculator that feeds MARS Front Office, alongside SWPM for rates and OVML for FX ([MARS brochure](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf)).
- **Layout.** [J, recollection, unverified] One screen holds three things:
  - deal terms per leg (multi-leg is native);
  - a market block: spot, the vol read off the surface (each leg overridable), rate curve, dividend projection, pricing date and time;
  - a results block: premium in currency and as a % of spot or notional, and Greeks per unit and in total.
  Scenario analysis sits on a separate tab. Every market input is an amber, overridable field, and an overridden value is visibly marked.
- **What builds trust.** [J] The surface vol, rates and dividends are pulled automatically, yet each is visible and overridable. Models can be switched and compared. The methodology documents are reachable from inside the Terminal.
  - [S] An FX cheat sheet points users to a `DOCS` number for the OVML documentation, and notes that OVML can send a customised option link through instant chat ([Bloomberg FX cheat sheet, excerpt](https://cbpm.wku.edu.cn/sites/business.prod.dpmgr.wku.edu.cn/files/2023-07/tools_for_foreign_exchange_market_cheat_sheet_0.pdf)). That link is the equivalent of our Share URL.

### 1.3 OSA (Option Scenario Analysis)

- [S] OSA evaluates "an option portfolio formed with the options and its underlying stock". It shows profit against stock price "for different holding periods and expirations" as both a profit graph and a profit table ([Xavier guide, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/11_derivativesoptionsfuturesswaps.pdf)).
- [S] Assumptions sit in amber fields for what-if analysis ([HBS](https://www.library.hbs.edu/services/help-center/bloomberg-options)).
- [S] Positions can be loaded from a saved portfolio (PRTU) ([Xavier exercises, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/15_bloombergderivativeexercises.pdf)).
- [J] Hierarchy:
  1. the position, as legs plus stock;
  2. the scenario axes (underlying price, horizon date, vol shift);
  3. P&L as a curve per horizon, with the same numbers as a table underneath.
  The table and the chart always show the same grid.

### 1.4 OVDV (volatility surface) and OVML context

- **Source selection is explicit.** [S] Bloomberg's webinar agenda lists "Volatility Sources – BVOL vs LIVE vs Custom: {OVDV}" under the tagline "Every volatility tells a story … Knowing which one to trust — and when — is the key" ([Bloomberg webinar page](https://professional.content.cirrus.bloomberg.com/professional2023/insights/webinar/inside-volatility-trading-hedging-positioning-winning-with-options/)). The surface's provenance is a first-class control.
- **FX quote set.** [S] OVDV provides the standard FX quotes: ATM, 25-delta and 10-delta risk reversals, and 25-delta and 10-delta butterflies. Where 10-delta quotes are missing, providers fill them with a constant-ratio approach ([MathFinance editorial, excerpt](https://www.mathfinance.com/?p=3118)).
- **Construction.** [S] Bloomberg's equity surface document describes two stages: imply the forward curve from option prices, then calibrate a smile model (a lognormal mixture) with explicit interpolation and extrapolation ([Bloomberg "Equity Implied Volatility Surface", studylib copy, excerpt](https://studylib.net/doc/27699733/surface-ovdv--1-)).
  - [S] An academic user of Bloomberg surface data says Bloomberg backs out implied vols, builds out-of-the-money (OTM) matrices on each side of 100% moneyness, stitches them and smooths away arbitrage ([arXiv 1912.11059](https://arxiv.org/pdf/1912.11059)).
  - [S] A New York Fed report says Bloomberg extracts European implied vols from American prices, then interpolates across strike and tenor ([NY Fed SR 677, excerpt](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr677.pdf)).
- **Layout.** [J, recollection, unverified]
  - FX OVDV is a grid with tenors as rows (overnight to multi-year) and ATM / 25D RR / 25D BF / 10D RR / 10D BF as columns, with bid/mid/ask toggles. Smile, term-structure and 3-D views are a tab away.
  - Equity OVDV uses expiries or constant tenors as rows, and moneyness % or delta as columns.
- **Bloomberg data fields.** [S] Users' field lists include moneyness-grid fields such as `1ST_MTH_IMPVOL_90.0%MNY_DF` … `110.0%MNY` and delta fields such as `1M_CALL_IMP_VOL_50DELTA_DFLT` ([QuantNet thread, excerpt](https://quantnet.com/threads/help-interpreting-bloomberg-implied-volatilities.7596/post-71306)). The house grids are moneyness % and delta.

### 1.5 SKEW

- [S] SKEW is entered as `<ticker> <EQUITY> SKEW` ([Xavier exercises, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/15_bloombergderivativeexercises.pdf)).
- [S] Practitioners chart it in percent moneyness: "using percentages of moneyness rather than outright strike prices greatly simplifies comparisons of historical skews". The example overlays today's 1-month skew on earlier dates ([IBKR Traders' Insight, excerpt](https://www.interactivebrokers.com/campus/traders-insight/securities/options/options-skews-are-well-skewed/)).
- [S] Bloomberg's own example skew measure is the "90% (volatility on puts) and 110% moneyness (volatility on calls)" difference ([Bloomberg insights, excerpt](https://www.bloomberg.com/professional/insights/trading/navigating-derivatives-market-sentiment-with-volatility-and-correlation-analysis/)).

### 1.6 OMON and the monitors

- [S] OMON lists calls on the left and puts on the right, grouped by expiry ([HBS](https://www.library.hbs.edu/services/help-center/bloomberg-options)).
- [S] Bloomberg describes OMON, OMST, OVI and OPX together as "Real-time intelligence on flow, skew, and positioning" ([Bloomberg webinar page](https://professional.content.cirrus.bloomberg.com/professional2023/insights/webinar/inside-volatility-trading-hedging-positioning-winning-with-options/)).

### 1.7 MARS Front Office (portfolio risk on the same library)

[S] All of the following is from the [MARS brochure](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf) (downloaded and read):

- **Greeks and P&L.** It offers "Full Term-structure Greeks" (including an "IR Vega matrix" and "FX Vega, Rega and Sega term structure"). P&L Explain is Greeks-based "with transparent drill-down capabilities to security level".
- **Stress testing.** It has "pre-canned" stress tests covering "Black Monday, Asian Crisis and Lehman Default 2008 to recent Brexit", plus custom and historical scenarios. Pre-trade "what-if" shows the incremental VaR of a new trade.
- **Consistency.** MARS runs on the same pricing library as the single-security calculators. The single-trade pricer and the book risk always agree.

---

## 2. Cboe LiveVol (Core and Pro)

- **Layout.** [S] LiveVol is a web-based multi-window console. Users choose which windows appear and their position and size, and windows can move across monitors ([Rankia review, Spanish](https://www.rankia.com/blog/meff-eurex/3468921-analizando-herramientas-para-trading-opciones-livevol-capitulo-5)). The user guide groups components into three menus ([LiveVol user guide](https://www.livevol.com/user-guide/)):

  | Menu | Components |
  |---|---|
  | Market | Symbol Summary, Options, Futures, Trade Tape, Watchlist, Calendar, Alerts |
  | Analytics | Time and Sales, Flow Analytics, Historical Snapshots, Options Calculator, Pro Scanner, Earnings Analysis, Statistics, Trade Optimizer |
  | Chart | Stock Chart, Futures Term Structure, Skew (live and historical) |

- **Features.** [S] The comparison table lists Live Skew, Historical Skew, Option Montage, Stats Window, Volatility Studies, Futures Term Structure and Earnings Analysis ([LiveVol platforms](https://www.livevol.com/analytics-platforms/)). Pro adds "time and sales data from 2011 to present, … scrolling trades, volatility charting, earnings event analysis, alerts, watchlists, scanning and filtering" and real-time Excel RTD ([Cboe DataShop](https://datashop.cboe.com/livevol-pro)).
- **Default chart.** [S] One chart per underlying stacks price, volume, IV and HV, with selectable vol windows of 10, 30, 60 and 90 days. Clicking a day shows that day's data ([Rankia](https://www.rankia.com/blog/meff-eurex/3468921-analizando-herramientas-para-trading-opciones-livevol-capitulo-5)). A review describes the details ([OptionsTradingIQ](https://optionstradingiq.com/livevol-review/)):
  - the lower pane shows IV30 in red and HV20 in blue, with a separate IV−HV spread pane;
  - earnings dates are marked with an "E";
  - Earnings Analysis shows the expected move in blue against the actual move in green or red, with the ATM straddle price beside it.
- **What builds trust.** [J] Every number is backed by Cboe's own trade-level history (time and sales back to 2011), so a user can drill from a vol print to the trades behind it. An exchange operates the product.

## 3. ORATS

- **Conventions.** [S] Delta is the preferred horizontal axis for the smile: "it normalizes the skew so you can compare different expirations and different stocks" ([ORATS, describing the surface](https://orats.com/blog/describing-the-implied-volatility-options-surface)). ORATS summarises the smile with ATM vol, a slope and a derivative:
  - **Slope:** "the amount that implied volatility changes for every increase of 10 call delta points".
  - **Derivative:** the change in that slope, so a measure of curvature.
  - Calls and puts are aligned with a residual yield before fitting.
- **Delta buckets.** [S] "21 volatility buckets from 100 delta to 0 delta every 5 deltas" (the "Monies" endpoint) ([ORATS buckets](https://orats.com/blog/volatility-skew-and-buckets)).
- **Term structure in three numbers.** [S] The curve is summarised as 30-day IV, 2-year IV and an implied earnings effect. In the NVDA example the earnings effect is "277% of its implied volatility"; the 30-day IV is quoted ex-earnings ([ORATS term structure](https://orats.com/blog/implied-volatility-term-structures-three-parameters)).
- **Data fields.** [S] All from the [ORATS definitions](https://orats.com/docs/definitions):

  | Field | Meaning |
  |---|---|
  | `iv30d` | 30-day constant-maturity IV |
  | `fwd30_20`, `fwd60_30`, … | Forward vols |
  | `ffwd30_20` | Flat-forward vol |
  | `fexErn30_20` | Forward ex-earnings vol |
  | `exErnIv30d` | Ex-earnings IV |
  | `impliedEarningsMove` | % move priced for earnings |
  | `dlt5Iv30d` / `dlt25Iv30d` / `dlt75Iv30d` / `dlt95Iv30d` | 30-day IV at fixed deltas |
  | `borrow30` | "implied hard-to-borrow interest rate … given options prices put call parity" |
  | `residualYieldRate` | Dividend-yield adjustment at the 50 delta |
  | `confidence` | "portion of the delta range 'covered' by market data" |
  | `mwVol` | "ATM weighted market width in implied volatility terms" |

- **Rich/cheap framing.** [S] Smooth vols, called SMV, are compared against market bid/ask to judge rich or cheap. The Strikes report shows bid/ask, bid/mid/ask IVs, smooth vol, theoretical values, Greeks, rates and dividends ([ORATS SMV](https://orats.com/blog/smoothing-options-implied-volatilities-using-orats-smv-system)).
  - [S] Skew richness is judged against the name's own one-year percentile ("87th percentile for the year") and against its best ETF ("slope divided by the best ETF … is 1.7") ([ORATS, is skew cheap](https://orats.com/blog/is-skew-cheap-or-expensive)).
- **UI.** [S] All from ORATS' [Ticker Analysis](https://orats.com/ticker-analysis) and [Trade Builder](https://orats.com/trade-builder) pages:
  - Ticker Analysis overlays "call and put mid IVs for each strike … on top of the smoothed IV skew for each expiration".
  - A line through each expiry's ATM IV shows the direction of the term structure.
  - "Highlighted are the highest and lowest slope, IV, confidence, and mwVol … of all expirations."
  - Time and sales shows "theoretical edge (compared to our theoretical price)" and an intraday dollar-delta chart.
  - Trade Builder shows "68% and 95% expected range" bands from ATM IV, and colours forecast vol as undervalued (green) or overvalued (red).
- **What builds trust.** [J] The confidence and market-width numbers are published next to the fitted values. ORATS says when its fit is thin.

## 4. Vola Dynamics

- **Fitter.** [S] "Output error bars derived from input bid-ask spreads quantify uncertainty in each fitted vol". "If the market price is within the error band, there is no statistical confidence that edge exists". Error bars "widen automatically" as data degrades. Fits are Bayesian and filtered "across strikes, expiries and time". The vendor claims it holds up through COVID, GameStop, negative oil and 0DTE "with no manual intervention" ([Vola Fitter](https://www.voladynamics.com/products/vola-fitter)).
- **Curves.** [S] Vola uses its nested "C*" curve family, plus SVI, SSVI and SABR. Users adjust "overall vol level, at-the-money skew and curvature, and each wing independently" ([Vola home](https://www.voladynamics.com/)).
- **Dynamics.** [S] Spot-vol dynamics come from a "Skew Stickiness Ratio (SSR)" that drives "smart delta and gamma, spot scenarios, and surface adjustments between fits" ([Vola home](https://www.voladynamics.com/)).
- **P&L explain.** [S] Explanation works by greeks ("Smart greeks produce a cleaner decomposition because they account for the spot-vol relationship") or by exact scenario repricing. Vol P&L splits into "ATF (level), skew (slope), curvature, and unexplained". Comparing greek-based with scenario-based attribution is offered as model validation ([Vola PnL](https://www.voladynamics.com/products/vola-pnl)).
  - [S] There are separate modules for discount curves, dividends and event variance ([Vola home](https://www.voladynamics.com/)).
- **Chart conventions.** [S] I read these off the two saved example charts ([examples page](https://www.voladynamics.com/examples/can-your-fitter-do-this)):
  - Market quotes are vertical bid-to-ask bars; the fit is a single line.
  - A solid vertical line marks the forward; dotted vertical lines mark the quoted strike range.
  - The x-axis is strike, or "normalized strike space" for single stocks; the y-axis is vol as a decimal.
  - The title carries the fit statistics (`T=0.2950, i=3, chi=0.029, avE5=3.8`).
  - Far-wing quotes with huge spreads show as tall bars that the fit is free to ignore.
- **What builds trust.** [J] The uncertainty of each fitted point is visible, and a trader can see at once that the curve passes through the bars.

## 5. OptionMetrics IvyDB

- [S] The volatility surface is a "kernel-smoothed constant expiration volatility surface … Deltas between 10 and 90 … with increments of 5 (negative for puts), and expirations from 10 days to 2 years" ([IvyDB US flyer](https://optionmetrics.com/wp-content/uploads/2024/03/OM_IvyDB-US_Flyer_WEB_REV.pdf)).
  - [S] The earlier grid was 20–80 delta and 30 days to 2 years ([2023 brochure](https://optionmetrics.com/wp-content/uploads/2023/03/IvyDB-US-Brochure.pdf)). The 10-day point was added "to approximate weekly contracts" ([BusinessWire 2020](https://www.businesswire.com/news/home/20200107005265/en/OptionMetrics-Announces-Updates-to-U.S.-Options-Database-to-Address-Evolving-Markets-Increased-Need-for-Accuracy)).
  - [S, excerpt] A search excerpt from an academic paper that uses IvyDB lists the tenors as 10, 30, 60, 91, 122, 152, 182, 273, 365, 547 and 730 calendar days. The excerpt is probably from [arXiv 2608.22478](https://arxiv.org/pdf/2608.22478), but I did not open the paper, so the attribution is uncertain.
- [S] Methods: "Correct implied volatility models (a multi-thousand step binomial tree for American options; discrete dividend projections)". Daily zero curves are included ([flyer](https://optionmetrics.com/wp-content/uploads/2024/03/OM_IvyDB-US_Flyer_WEB_REV.pdf)).
- [S] A "daily patch file … contains corrections to previous prices or calculations when needed" ([IvyDB US](https://optionmetrics.com/united-states/)).
- **What builds trust.** [J] The model is named, corrections are published, and the grid is fixed so researchers can compare across time.

## 6. LSEG / Refinitiv (Workspace, IPA)

- **Apps.** [S] Workspace equity calculators include "Option Watch, Option Pricer, Surface Volatility" ([LSEG Workspace equities, excerpt](https://lseg.com/en/data-analytics/products/workspace/equities)). The surface app is SURF, and its ranges can be adjusted by the user ([LSEG volatility surfaces, excerpt](https://lseg.com/en/data-analytics/financial-data/pricing-and-market-data/fixed-income-pricing-data/futures-options-derivatives/volatility-surface)).
- **Published grids.** [S] Same LSEG source ([excerpt](https://lseg.com/en/data-analytics/financial-data/pricing-and-market-data/fixed-income-pricing-data/futures-options-derivatives/volatility-surface)):
  - by delta, "from 10d – 45d for puts and calls with an ATM, intervals of 5 deltas";
  - by moneyness, in 10% steps over 50–70% and 130–150%, and 5% steps over 75–125%;
  - in constant maturities up to 24 months, or by actual expiries;
  - fitted with SVI.
- **Equity (ETI) surface defaults.** [S] All from the [IPA ETI docs](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti):

  | Setting | Values (default first where documented) |
  |---|---|
  | Axes | x = Date, y = Strike; also Delta, Tenor, Moneyness |
  | `moneynessType` | Spot / Fwd / Sigma |
  | `priceSide` | Bid / Ask / Mid / Last (default Mid) |
  | `volatilityModel` | SVI / SSVI |
  | `smileBySmileArbitrageCheck` | True |
  | `sviAlphaExtrapolation` | True |
  | Optional outputs | Discount curve, dividends (an `ImpliedYield` type for indices), underlying spot |

  The quote filters, with their defaults:

  | Filter | Default |
  |---|---|
  | `maxStalenessDays` | 5 |
  | `maxOfMedianBidAskSpread` | 4.0 |
  | `atmToleranceIntervalPercent` | 1.0 |
  | `ensurePricesMonotonicity` | True |
  | `useWeeklyOptions` | False |
  | `minMaturity` | 7D |
  | Strike range: `minOfMedianImpliedVolPercent` / `maxOfMedianImpliedVolPercent` | 30 / 500 |
  | `minOfMedianNbOfStrikesPercent` | 20 |

- **FX surface.** [S] Per the [IPA FX docs](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-fx):
  - `deltaConvention` is Spot or Fwd;
  - `deltaPremiumConvention` defaults to PA (premium-adjusted);
  - the y-axis defaults to Delta;
  - models include SVI, SABR, VannaVolga10D and VannaVolga25D;
  - the ATM and RR/BF sources can be overridden.
- **Plots.** [S] LSEG's tutorial plots smiles of several names at one expiry, a term structure at a chosen moneyness, and a forward curve ([LSEG article](https://developers.lseg.com/en/article-catalog/article/instrument-pricing-analytics-volatility-surfaces-and-curves)).
- **What builds trust.** [J] The filters are documented parameters with published defaults, so two users can reproduce the same surface.

## 7. QuantLib-based and open-source desk tools

- **ORE (Open Source Risk Engine, built on QuantLib).** [S] The 12th release added historical-simulation VaR, backtesting, "P&L and P&L explain", stress testing and AAD sensitivities. Launchers exist for Excel, Python and Jupyter ([LSEG ORE release](https://www.lseg.com/en/media-centre/press-releases/acadia/2024/acadia-12th-update-open-source-risk-engine); [ORE GitHub](https://github.com/OpenSourceRisk/Engine)). ORE ships no GUI; its "interface" is reports and notebooks.
- **OpenGamma Platform (2011–13).** [S] Version 1.0 let users "view risk figures alongside the market data used in the calculations". It had a dependency-graph view (including a P&L series for historical VaR), market-data snapshots as a data source, and position-detail pop-ups ([WatersTechnology, excerpt](https://waterstechnology.com/node/2165150); [OpenGamma forum, excerpt](https://forums.opengamma.com/t/platform-1-0-0-release/79)).
- [J] The lesson from both is lineage: a reviewer trusts a number they can trace to its inputs and model, and an independent library (QuantLib) agreeing to many decimals is persuasive.

## 8. Market-making and vol-desk tooling: what is public

- **Jane Street, "Building Tools for Traders"** ([Signals and Threads, Ian Henry, 2025](https://signalsandthreads.com/building-tools-for-traders/)). This is the most direct public account of options-desk tooling. [S]
  - The tools serve two jobs: exploration, and "attention management" through alerts.
  - Views cover "an entire und and expiration date", across "hundreds of different strikes and exps".
  - Density is extreme ("I want this to be about six pixels high because then I can see more of it on my screen").
  - "Everything is accessible from the keyboard", so "You can't hover to see more".
  - Latency is measured in microseconds. Incremental re-render (Bonsai) updates only what changed.
  - Loading, loaded and errored are explicit states.
  - On alert fatigue, desk-wide thresholds are preferred over per-user knobs.
  - Senior traders shape the UX, and feedback loops are same-day.
- **Optiver.** [S] "Traders want to see a lot of data at the same time in a compact format" ([Optiver data visualisation](https://optiver.com/working-at-optiver/career-hub/data-visualisation-at-optiver/)). GUIs "must be reliable and responsive"; "unresponsive or crashing applications … could lead to significant monetary loss". The fix described: stream only the data the trader can see ([Optiver C# blog](https://www.optiver.com/insights/technology-blog/navigating-performance-challenges-as-a-c-software-engineer-at-optiver/)).
- **IMC.** [S] Job posts describe a tooling team making "large datasets explorable", and engineers building "options pricing, volatility calculations, simulation frameworks, and data analysis & visualization tools for traders" ([IMC job post, excerpt](https://www.imc.com/eu/careers/jobs/4628622101)). No public UI write-up.
- **Citadel Securities.** [S] Its public material stresses robustness: during the 2023 rates volatility there were "only a handful of situations in which the firm was unable to quote" ([Risk.net award, excerpt](https://www.risk.net/awards/7958404/flow-market-maker-of-the-year-citadel-securities)). I found no public UI or tooling write-up.
- **SIG.** [S] Options trading is SIG's core business ([sig.com, excerpt](https://sig.com/what-we-do/)). I found no public tooling material.
- **Moontower (Kris Abdelmessih, a former options market maker).** [S] His surface-drilldown primer describes these views ([Moontower primer #9](https://moontower.substack.com/p/primer-9-drilldown-a-finer-look-into)):
  - a constant-maturity term structure;
  - a "forward IV matrix" of every pairwise forward vol;
  - a vol cone with the IV term structure overlaid;
  - normalised skew and its percentile at the 10-delta call and 25-delta put;
  - a fixed-strike vol-change scanner by delta bucket, with changes optionally normalised by √T;
  - a relative term structure (each tenor divided by one chosen tenor).

  On spot-vol dynamics: "a slope of -3.0 means a 1% rally drops ATM vol by 3%" ([Moontower, sticky vs floating](https://moontower.substack.com/p/sticky-vs-floating-strike)).
- **Rival Systems (market-maker front end).** [S] Its features include an "auto-fitter to fit traders' curves to real-time implied volatility", quoting with "dynamic edge formulas", "real-time risk and margin" and "custom risk scenarios and VaR" ([Rival ecosystem, excerpt](https://www.rivalsystems.com/news/archive/2020/01/the-rival-ecosystem/)).
- **Peter Carr, "Vol, Skew, and Smile Trading".** [S] Level, slope and curvature of the implied variance curve each map to a trading strategy, which "help[s] explain standard quotation conventions" (ATM, risk reversal, butterfly) ([Fields Institute, 2017](https://www.fields.utoronto.ca/talks/Vol-Skew-and-Smile-Trading)).

---

## 9. Cross-cutting conventions (desk standard)

| Topic | Convention | Source |
|---|---|---|
| Vol units | Vol in % or vol points. Changes in vol points; small biases in vol bp (0.01 vol pt). Market width quoted in vol (ORATS `mwVol`). | [S] [ORATS defs](https://orats.com/docs/definitions); [J] for vol bp |
| Vega | $ per 1 vol point | [S] [TradeStation, excerpt](https://help.tradestation.com/10_00/eng/tradestationhelp/options_positions/about_gamma_vega_theta_rho.htm) |
| Theta | $ per calendar day | [S] same |
| Gamma | Dollar gamma = 0.5 × Γ × (S × 1%)², the P&L of a 1% move | [S] [IBKR Risk Navigator metrics, excerpt](https://www.interactivebrokers.com.hk/php/webhelp/navigator/metrics.htm) |
| Delta | Cash delta = position × Δ × multiplier × S | [S] same |
| Rates, carry | Basis points; implied borrow and residual yield as rates | [S] [ORATS defs](https://orats.com/docs/definitions) |
| Time | Minutes / 525,600 for short-dated work (Cboe VIX); constant-maturity points interpolated in variance | [S] [Cboe VIX white paper](https://www.asx.com.au/content/dam/asx/investors/investment-options/vix/cboe-vix-white-paper.pdf) |
| Strike axes | Strike; moneyness K/S or K/F (LSEG `moneynessType` Spot/Fwd/Sigma); log or normalised strike (Vola NS); delta (ORATS, IvyDB, LSEG). Some tools let the user switch x-axis between moneyness, fixed strike and delta. | [S] [LSEG ETI](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti); [SpotGamma, excerpt](https://support.spotgamma.com/hc/en-us/articles/23982082962195) |
| Delta grids | 10–90 by 5 (IvyDB); 10–45 plus ATM by 5 (LSEG); 0–100 by 5 (ORATS) | [S] as above |
| Moneyness grids | Bloomberg-style 90/95/100/105/110 (from field names); LSEG 50–150 | [S] as above |
| ATM | ATM-forward (Vola "ATF"); FX: delta-neutral straddle, by market convention | [S] Vola; [J] for the FX DNS detail |
| Skew | Equity: 90%−110% moneyness vol (Bloomberg). 25Δ RR = σ(25Δc) − σ(25Δp); sign varies by desk. Normalised skew (25Δp − 25Δc)/50Δ vol, "most descriptive and least redundant" (Mixon 2011). ORATS slope per 10 call-delta points. | [S] [Bloomberg, excerpt](https://www.bloomberg.com/professional/insights/trading/navigating-derivatives-market-sentiment-with-volatility-and-correlation-analysis/); [Mixon](https://businessweek.ivolatility.com/doc/JOD-MIXON_(2011)_+_Appendix.pdf); [ORATS](https://orats.com/blog/describing-the-implied-volatility-options-surface) |
| Butterfly | Wings over ATM, in the simple form: σ(25Δc) = ATM + ½RR + BF; σ(25Δp) = ATM − ½RR + BF. The FX market-strangle convention differs (see Reiswich & Wystup). | [S] [AQH](https://algoquanthub.beehiiv.com/p/fx-volatility-smiles-market-quotes); [Reiswich & Wystup](https://ideas.repec.org/p/zbw/cpqfwp/20.html) |
| Term structure | Constant-maturity grid (IvyDB 10d–2y; LSEG to 24M); forward vols (ORATS `fwd30_20`); ex-earnings and earnings effect (ORATS); vol cone (Moontower) | [S] as above |
| Forwards and carry | Forward implied from option prices (Bloomberg); residual yield and implied borrow from put-call parity (ORATS); implied dividend yield for indices (LSEG); dividend and discount fitters (Vola) | [S] as above |
| Dynamics | Sticky strike: fixed-strike vol independent of spot; ATM vol falls as spot rises; Δ = Δ_BS. Sticky delta: ATM vol independent of spot; fixed-strike vol rises with spot; Δ > Δ_BS. Sticky implied tree: all vols inversely related to spot; Δ < Δ_BS. SSR 1 = sticky strike, 0 = sticky delta. | [S] [Derman 1999, Tables 3, 5, 8](https://hpirotte.ulb.be/INGESTriskmgt/readings/Derman%20(1999)%20-%20Volatility%20regimes.pdf); [Risk.net, excerpt](https://www.risk.net/ja/node/7904581) |
| Scenario grids | SPAN: 16 scenarios (price at 0, ±1/3, ±2/3, ±3/3 of scan range × vol up/down, plus two extremes at 3× range covering 30% of the loss). FINRA 4210(g): 10 equidistant points over ±15% for equity options. OSA: P&L vs price per horizon. | [S] [CME SPAN](https://www.cmegroup.com/clearing/files/span-methodology.pdf), [CME risk arrays](https://cmegroupclientsite.atlassian.net/wiki/spaces/pubsub/pages/457083458/Risk+Arrays); [FINRA 07-11](https://www.finra.org/rules-guidance/notices/07-11) |
| P&L explain | Total, explained by factor, unexplained residual | [S] [Wikipedia](https://en.wikipedia.org/wiki/PnL_explained); [Vola PnL](https://www.voladynamics.com/products/vola-pnl); [MARS](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf) |

## 10. What makes professionals trust these tools (synthesis)

1. **The market is drawn next to the model.** Bid/ask bars under the fit (Vola), mid IVs over the smooth curve (ORATS), and theo against bid/ask in a table (ORATS SMV). [S]
2. **Uncertainty is quantified and published.** Error bars and "minimum edge" (Vola); confidence and `mwVol` (ORATS); explicit quote filters with defaults (LSEG). [S]
3. **Provenance is a control, not a footnote.** BVOL vs LIVE vs Custom (Bloomberg OVDV); `priceSide` and `timeStamp` (LSEG); risk next to its market data (OpenGamma). [S]
4. **One library, one surface, everywhere.** OVME reads the same surface as OVDV, and MARS uses the same library as OVME. [S] [J] Consistency across views is itself evidence of correctness.
5. **Inputs and outputs are visually distinct, and overrides are visible.** Amber fields (Bloomberg). [S]
6. **Named, documented models with published corrections.** IvyDB's model statement and daily patch file; Bloomberg's DOCS. [S]
7. **Explainable P&L with a residual.** Unexplained P&L is how spot errors and model gaps surface (Vola PnL use cases). [S]
8. **Speed and stability.** "Loading it with no latency is Terminal's real superpower" (Strom); "unresponsive … could lead to significant monetary loss" (Optiver); explicit error states (Jane Street). [S]
9. **Conservative, consistent change.** Even small visual changes are disruptive (Bloomberg). [S]

---

## 11. Ranked: the 25 most valuable patterns for this app

The ranking is my judgement [J]: value to a market maker or vol PM reviewing the app, weighted by feasibility in Streamlit with free resources and synthetic or uploaded data. Each entry gives the pattern, where it goes in the app (with plan items from `docs/roadmap/plan-items.md` where they exist), and its source.

1. **Bid/ask vol bars with the fitted smile, the forward marked and fit statistics in the title.** Per expiry, draw each quote as a vertical bid-to-ask IV bar (mid as a tick), the fitted curve as one line, a vertical rule at the forward, and dotted rules at the quoted-strike range. Put the fit statistics in the title or caption: RMSE in vol pts, share of mids inside the band, N quotes used out of N quoted.
   - *Where:* Vol surface tab smile view (SF-4, SF-6).
   - *Source:* [Vola Dynamics examples](https://www.voladynamics.com/examples/can-your-fitter-do-this) (saved screenshots); [ORATS Ticker Analysis](https://orats.com/ticker-analysis). [S]
2. **A forward-centric surface: implied forward and carry per expiry, ATM-forward vol and moneyness in K/F.** Show for each expiry:
   - the parity-implied forward;
   - the implied carry (rate − dividend − borrow) in bp;
   - the gap to the sidebar carry in bp, flagged above a threshold.

   Default moneyness to K/F and label ATM as ATMF. Today the Surface table says "ATM means K = S".
   - *Where:* Vol surface tab and header ticker (SF-3, SF-11).
   - *Source:* [Bloomberg surface doc, excerpt](https://studylib.net/doc/27699733/surface-ovdv--1-); [ORATS `borrow30`, `residualYieldRate`](https://orats.com/docs/definitions); [LSEG `moneynessType` Fwd](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti); [Vola "ATF"](https://www.voladynamics.com/products/vola-pnl). [S]
3. **Desk smile summary per expiry, in vol points.** Rows are expiries. Columns:
   - ATMF vol;
   - 25Δ RR and 25Δ BF, with 10Δ RR and BF;
   - 90–110 skew;
   - normalised skew (25Δp − 25Δc)/50Δ vol;
   - fit RMSE and confidence.

   Say which sign convention the RR uses in the column help.
   - *Where:* Vol surface tab (SF-6), replacing or extending today's "Skew 90-110 / Butterfly" table.
   - *Source:* [MathFinance on OVDV quotes, excerpt](https://www.mathfinance.com/?p=3118); [AQH formulas](https://algoquanthub.beehiiv.com/p/fx-volatility-smiles-market-quotes); [Carr, Fields 2017](https://www.fields.utoronto.ca/talks/Vol-Skew-and-Smile-Trading); [Mixon 2011](https://businessweek.ivolatility.com/doc/JOD-MIXON_(2011)_+_Appendix.pdf). [S]
4. **A switchable strike axis (strike, K/F, log-moneyness or normalised strike, delta) on standard grids.** Use 10–90Δ in steps of 5 and a 90/95/100/105/110 moneyness grid, with the same switch on every smile, surface and vega-bucket view.
   - *Where:* Vol surface tab, IV tab, Risk vega buckets (SF-6, RK-2).
   - *Source:* [ORATS, delta "normalizes the skew"](https://orats.com/blog/describing-the-implied-volatility-options-surface); [IvyDB 10–90 by 5](https://optionmetrics.com/wp-content/uploads/2024/03/OM_IvyDB-US_Flyer_WEB_REV.pdf); [LSEG grids, excerpt](https://lseg.com/en/data-analytics/financial-data/pricing-and-market-data/fixed-income-pricing-data/futures-options-derivatives/volatility-surface). [S]
5. **Quote filters with visible counts.** Make the filters explicit, adjustable and counted ("212 quoted → 171 used: 23 wide, 12 stale, 6 ITM"):
   - OTM-only;
   - max spread relative to the median spread;
   - minimum strikes per expiry;
   - minimum maturity;
   - weeklies on or off;
   - staleness (for uploads).
   - *Where:* Vol surface data-source panel (SF-11, DT-5).
   - *Source:* [LSEG IPA filters and defaults](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti); [arXiv 1912.11059 on Bloomberg using OTM only](https://arxiv.org/pdf/1912.11059). [S]
6. **Fit-confidence numbers next to every fitted value.** Show per-expiry confidence (share of the delta range covered by quotes), ATM market width in vol points, and the bid/ask error band of each fitted vol. Highlight the best and worst expiry on each metric.
   - *Where:* Vol surface tab, and BK-3 leg IVs from the surface.
   - *Source:* [ORATS `confidence`, `mwVol`, highlighting](https://orats.com/ticker-analysis); [Vola error bars](https://www.voladynamics.com/products/vola-fitter). [S]
7. **Edge in vol points against the error band.** For any priced leg or chain row, show theo − mid in vol points and in price ticks. Mark "no edge" when the mid sits inside the fitted vol's bid/ask-derived band.
   - *Where:* Pricing and IV tabs, Strategy leg table (BK-3).
   - *Source:* [Vola "minimum edge"](https://www.voladynamics.com/products/vola-fitter); [ORATS SMV vs bid/ask](https://orats.com/blog/smoothing-options-implied-volatilities-using-orats-smv-system); [ORATS "theoretical edge"](https://orats.com/ticker-analysis). [S]
8. **Spot × vol × time scenario grid with presets.**
   - Default to OSA-style P&L against spot, one curve per horizon, with the identical table underneath.
   - Add a "SPAN-like 16" preset (0, ±⅓, ±⅔, ±1 of range × vol up/down, plus ±3× extremes) and a "FINRA 4210(g)" preset (10 points over ±15%), with the worst cell highlighted.
   - *Where:* Risk tab (RK-1 scenario cube, RK-8).
   - *Source:* [Xavier OSA, excerpt](https://www.xavier.edu/fifth-third-trading-center/bloomberg/guides/11_derivativesoptionsfuturesswaps.pdf); [CME SPAN](https://www.cmegroup.com/clearing/files/span-methodology.pdf); [CME risk arrays](https://cmegroupclientsite.atlassian.net/wiki/spaces/pubsub/pages/457083458/Risk+Arrays); [FINRA 07-11](https://www.finra.org/rules-guidance/notices/07-11). [S]
9. **A spot-vol dynamics selector that drives scenarios and deltas.** The choices are sticky strike, sticky delta (moneyness) and an SSR slider. Show the BS delta and the dynamics-adjusted ("smart") delta side by side, and print the regime under every scenario chart.
   - *Where:* Risk tab and Strategy tab (RK-7).
   - *Source:* [Derman 1999, Tables 3/5/8](https://hpirotte.ulb.be/INGESTriskmgt/readings/Derman%20(1999)%20-%20Volatility%20regimes.pdf); [Vola SSR, smart delta](https://www.voladynamics.com/); [Moontower vol path](https://moontower.substack.com/p/sticky-vs-floating-strike). [S]
10. **Term-structure panel with forward vols and event variance.** Show:
    - ATMF vol at constant maturities (7, 14, 30, 60, 91, 182, 365, 730 d) interpolated in total variance;
    - the actual expiries as markers;
    - a forward-vol matrix (or a simple forward-vol strip);
    - ex-event vol beside the event-implied move when an event is set.
    - *Where:* Vol surface tab (SF-6, DT-1 event variance).
    - *Source:* [ORATS fwd and ex-earnings fields](https://orats.com/docs/definitions); [ORATS three-parameter term structure](https://orats.com/blog/implied-volatility-term-structures-three-parameters); [Moontower forward IV matrix](https://moontower.substack.com/p/primer-9-drilldown-a-finer-look-into); [IvyDB tenors](https://optionmetrics.com/wp-content/uploads/2024/03/OM_IvyDB-US_Flyer_WEB_REV.pdf); [Cboe VIX variance interpolation](https://www.asx.com.au/content/dam/asx/investors/investment-options/vix/cboe-vix-white-paper.pdf). [S]
11. **P&L explain waterfall with a visible residual.** The bars are delta, gamma, theta, vega (split into ATF level, skew and curvature), rates and unexplained. Show greek-based and full-revaluation results side by side as a self-check.
    - *Where:* Risk tab (RK-3).
    - *Source:* [Vola PnL](https://www.voladynamics.com/products/vola-pnl); [MARS P&L Explain with drill-down](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf); [Wikipedia](https://en.wikipedia.org/wiki/PnL_explained). [S]
12. **Vega by expiry × delta bucket, with 1M-equivalent vega as the headline.** Raw net vega misleads; the plan measured +162.52 raw against +31.94 1M-weighted. Show row and column totals.
    - *Where:* Risk tab headline strip (RK-2, BK-1).
    - *Source:* [MARS "IR Vega matrix", "Full Term-structure Greeks"](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf); [ORATS delta buckets](https://orats.com/blog/volatility-skew-and-buckets). [S]
13. **Provenance badge and source selector everywhere.** Every surface, price and risk number names its source (synthetic demo / uploaded file name / user-overridden / model-only), its timestamp, and its price side (mid, bid, ask). The source is switchable like Bloomberg's BVOL / LIVE / Custom.
    - *Where:* Header strip and Vol surface panel (SF-11, DK-5, DT-6).
    - *Source:* [Bloomberg OVDV "Volatility Sources"](https://professional.content.cirrus.bloomberg.com/professional2023/insights/webinar/inside-volatility-trading-hedging-positioning-winning-with-options/); [LSEG `priceSide`, `timeStamp`](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti); [OpenGamma, excerpt](https://waterstechnology.com/node/2165150). [S]
14. **Arbitrage and extrapolation flags at the point where they occur.**
    - Mark butterfly violations on the smile.
    - Mark calendar violations on the term structure.
    - Shade the wing extrapolation region (beyond the last quote, held to Lee's bound) on every chart.
    - Show a pass/fail count in the summary table.
    - *Where:* Vol surface tab (SF-4, SF-8, SF-10).
    - *Source:* [LSEG `smileBySmileArbitrageCheck`, `sviAlphaExtrapolation`](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti); [Vola "arbitrage-free … in the far wings"](https://www.voladynamics.com/products/vola-fitter). [S]
15. **Desk units in every label.** Use vega per vol pt, theta per calendar day, $gamma per 1%, cash delta, vol changes in vol pts, rates in bp, and time to expiry as days plus hours for anything under 2 days. The app already uses "Vega / vol pt" and bp in places; make it universal.
    - *Where:* All tabs (DK-4, DT-1).
    - *Source:* [IBKR Risk Navigator, excerpt](https://www.interactivebrokers.com.hk/php/webhelp/navigator/metrics.htm); [TradeStation, excerpt](https://help.tradestation.com/10_00/eng/tradestationhelp/options_positions/about_gamma_vega_theta_rho.htm); [Cboe VIX minutes/525,600](https://www.asx.com.au/content/dam/asx/investors/investment-options/vix/cboe-vix-white-paper.pdf). [S]
16. **Inputs look different from outputs, and overrides look different from both.** Give typed inputs one visual treatment (Bloomberg's amber field; a tokenised accent here). Mark a leg vol that overrides the surface with an "override" tag and its gap to the surface in vol pts.
    - *Where:* Sidebar, Strategy leg editor, Pricing (DK-3, BK-3).
    - *Source:* [HBS, OSA amber fields](https://www.library.hbs.edu/services/help-center/bloomberg-options); [Bloomberg "Amber is our base font color", excerpt](https://www.bloomberg.com/company/stories/bloombergs-customer-centric-design-ethos). [S]
17. **Independent cross-check and named model on every price.** Show the model badge, the engine settings, and an "agrees with QuantLib to x" or "ALO vs binomial differ by y" readout. Link to the methodology section and the validation report.
    - *Where:* Pricing tab, header model badge, Methodology drawer (VL-1, VL-7, DK-6).
    - *Source:* [Vola "Comparing greek-based and scenario-based attribution to validate models"](https://www.voladynamics.com/products/vola-pnl); [IvyDB names its models and publishes a patch file](https://optionmetrics.com/united-states/); [Bloomberg OVME "Customizable models"](https://professional.content.cirrus.bloomberg.com/professional2023/insights/webinar/inside-volatility-trading-hedging-positioning-winning-with-options/). [S] for the precedent; [J] for the QuantLib badge.
18. **One contract context drives every view, and the exact state is shareable.** The header ticker (S, K, expiry, DTE, σ, r, q, F) is the single context; every tab follows it. The Share link reproduces the exact state, like Bloomberg's ticker-plus-function model and OVML's shareable option link.
    - *Where:* Header (DK-5, P0-1).
    - *Source:* [HBS, functions from the contract menu](https://www.library.hbs.edu/services/help-center/bloomberg-options); [Bloomberg FX cheat sheet, excerpt](https://cbpm.wku.edu.cn/sites/business.prod.dpmgr.wku.edu.cn/files/2023-07/tools_for_foreign_exchange_market_cheat_sheet_0.pdf); [LiveVol linked multi-window console](https://www.rankia.com/blog/meff-eurex/3468921-analizando-herramientas-para-trading-opciones-livevol-capitulo-5). [S]
19. **History overlays and percentiles.**
    - Overlay today's smile on T−1d and T−1w.
    - Show percentile ranks of ATMF vol, 25Δ RR and slope over a lookback.
    - With synthetic history from the RK-4 generator or user uploads, this is free.
    - *Where:* Vol surface and IV tabs (RK-4).
    - *Source:* [Bloomberg SKEW overlays, excerpt](https://www.interactivebrokers.com/campus/traders-insight/securities/options/options-skews-are-well-skewed/); [ORATS "87th percentile"](https://orats.com/blog/is-skew-cheap-or-expensive); [Moontower skew percentile](https://moontower.substack.com/p/primer-9-drilldown-a-finer-look-into). [S]
20. **Expected-move bands from ATM IV and the straddle.** Draw ±1σ (68%) and ±2σ (95%) bands from ATMF vol to the horizon on payoff and scenario charts, and print the straddle-implied move.
    - *Where:* Strategy and Risk tabs (BK-2).
    - *Source:* [ORATS Trade Builder "68% and 95% expected range"](https://orats.com/trade-builder); [LiveVol earnings analysis with ATM straddle](https://optionstradingiq.com/livevol-review/). [S]
21. **A named historical stress library with spot, vol and skew shocks.** Include Black Monday 1987, 2008 Lehman, Feb-2018 vol spike, Mar-2020 COVID and Aug-2024 unwind, each with its spot move, ATM vol shock and skew steepening. Allow custom scenarios too.
    - *Where:* Risk tab (RK-7).
    - *Source:* [MARS "pre-canned stress test library … Black Monday, Asian Crisis and Lehman … Brexit"](https://data.bloomberglp.com/professional/sites/10/MARS-Front-Office-Risk-Brochure.pdf). [S] for the pattern; [J] for the scenario list.
22. **Attention management: highlight what is unusual, not everything.**
    - Mark quotes whose mid sits outside the fitted error band.
    - Mark the expiry with the extreme slope or confidence.
    - Flag carry gaps.
    - Use one quiet accent rather than many alerts; thresholds are desk-wide, not per-user.
    - *Where:* Vol surface and IV tabs.
    - *Source:* [Jane Street, Signals and Threads](https://signalsandthreads.com/building-tools-for-traders/); [ORATS highlighting](https://orats.com/ticker-analysis). [S]
23. **Keyboard-first, no information that only exists on hover, and fast updates.**
    - Every value that matters is printed, not hover-only.
    - Shortcuts switch tabs and focus the spot and vol inputs.
    - Reruns stay under about 1 s, with an explicit loading or error state rather than silent stale output.
    - *Where:* Shell (DK-8, DK-7, P0-1).
    - *Source:* [Jane Street: "Everything is accessible from the keyboard … You can't hover to see more"](https://signalsandthreads.com/building-tools-for-traders/); [Strom thresholds](https://mattstromawn.com/writing/ui-density); [Optiver responsiveness](https://www.optiver.com/insights/technology-blog/navigating-performance-challenges-as-a-c-software-engineer-at-optiver/). [S]
24. **Dense but hierarchical layout.** Use compact tables with tabular figures and small labels, many numbers per screen, and one clear primary number per panel. Hide secondary detail behind expanders rather than removing it.
    - *Where:* All tabs (DK-5, DK-3).
    - *Source:* [Optiver "a lot of data at the same time in a compact format"](https://optiver.com/working-at-optiver/career-hub/data-visualisation-at-optiver/); [Jane Street "six pixels high"](https://signalsandthreads.com/building-tools-for-traders/); [Bloomberg "clear hierarchy", "hiding complexity", excerpts](https://www.bloomberg.com/company/stories/how-bloomberg-terminal-ux-designers-conceal-complexity). [S]
25. **Colour-vision-safe direction colours, an alternate palette, and non-colour cues.**
    - Gains and losses carry a sign and a glyph as well as colour.
    - Use a blue-for-up / orange-for-down palette, or offer it as an alternate.
    - Check charts with a deuteranopia simulation.
    - Keep the scheme consistent across tabs and over time.
    - *Where:* Theme (DK-1, DK-3).
    - *Source:* [Bloomberg color accessibility, excerpt](https://www.bloomberg.com/ux/2021/10/14/designing-the-terminal-for-color-accessibility/); [Bloomberg PDFU COLORS, excerpt](https://www.bloomberg.com/ux/accessibility/); [UX Magazine on consistency](https://uxmag.com/articles/the-impossible-bloomberg-makeover). [S]

## 12. Gaps and caveats

- **Bloomberg screen details are unverified.** Exact OVME, OVDV, SKEW and OSA layouts, tab names and default grids are not publicly documented, and bloomberg.com refused automated reads. Everything marked [J, recollection] should be checked on a Terminal before anyone quotes it.
- **The following have no public UI write-ups:** Citadel Securities, SIG and IMC. Optiver's posts are general. Jane Street's podcast is the only first-hand account of options-desk tooling I found.
- **Cboe LiveVol window-level documentation** (Option Montage, Live Skew) is behind the product; the review sources may be dated.
- **Vendor claims are vendor claims:** Vola's robustness, ORATS testimonials and IvyDB's "industry standard".
- **Sign conventions for the 25Δ risk reversal differ** between FX (call − put) and some equity screens (put − call). The app should state its convention in the column help.
