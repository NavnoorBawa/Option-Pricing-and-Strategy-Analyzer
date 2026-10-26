# Data and licensing policy

The analyzer is a public demo (Streamlit Community Cloud) and an open-source library. It may only show and ship data
that anyone is free to use. This page lists every data source the project touches or was asked about, what its owner
allows, and how the project uses it. The terms were read on 2026-10-26; the quotes are verbatim.

## The public-demo rule

1. **Default data is synthetic.** The demo prices a synthetic chain from a known model
   (`options_analyzer.market_data.sample_chain`), labelled `Synthetic — not market data` wherever it appears.
2. **Market data comes only from the visitor.** A visitor may upload a chain they downloaded themselves, under the
   terms they accepted with that source. The adapters (`options_analyzer.market_data.read_chain`) read the file in
   memory for that session; the project never redistributes it.
3. **No server-side fetching or scraping.** The app and the package fetch no quotes, index levels or price histories
   from third-party servers. Public-domain reference data (US Treasury yields) may be bundled with a citation.
4. **Fixtures say where they come from.** Every file in `tests/fixtures/` starts with a provenance header (below). Option
   chains and price histories in the repository are fabricated or synthetic; none is a copy of real third-party data.
5. **Local-only extras stay off the default path.** The optional `data` extra (yfinance) is for a user's own machine.
   No module that the app or the package imports by default may fetch from Cboe or Yahoo.

`tests/test_data_policy.py` enforces rules 3 to 5: no `cdn.cboe.com` or `finance.yahoo.com` URL and no `yfinance`
import anywhere in the package or the entry point, no opt-in module on the default import path, the default app
running with the network unreachable through a dead proxy, and the provenance header of every fixture.

## Sources

| Source | Terms | How the project uses it |
| --- | --- | --- |
| Cboe delayed quotes ([www.cboe.com/delayed_quotes](https://www.cboe.com/delayed_quotes/spx/quote_table), and the JSON behind the page on `cdn.cboe.com`) | Notice on every quote page: "IT IS STRICTLY PROHIBITED TO DOWNLOAD DELAYED QUOTE TABLE DATA FROM THIS WEB SITE BY USING AUTO-EXTRACTION PROGRAMS/QUERIES AND/OR SOFTWARE. CBOE WILL BLOCK IP ADDRESSES OF ALL PARTIES WHO ATTEMPT TO DO SO. THIS DATA IS PROPERTY OF CBOE LIVEVOL OR ITS DATA PROVIDERS. DOWNLOADING THIS DATA IN ANY OTHER WAY THAN BY MANUAL TICKER SYMBOL ENTRY IS STRICTLY PROHIBITED." | Never fetched. The notice also rules out an automated local adapter for these pages or their JSON (roadmap item DT-7). A visitor may upload the CSV they downloaded by hand; `read_chain` reads its layout (`tests/fixtures/chain_cboe_wide.csv` holds fabricated values in that layout). |
| Cboe website terms ([www.cboe.com/terms](https://www.cboe.com/terms), last updated 2022-11-16) | "You may view, print and download one copy of the Materials for your personal non-commercial use in connection with products and services offered by Cboe, provided that you maintain all copyright, trademark and other notices contained on the Materials. You may not otherwise copy, reproduce, alter, store either in hard copy or in an electronic retrieval system, license, transmit, display, broadcast, create a derivative work (for example, a financial product, service or index) from, use to verify or correct other data or information, publish, rent, sublicense, distribute, or otherwise use in whole or in part in any other manner the Materials without Cboe's prior written consent ..." | Nothing from cboe.com is stored or redistributed. Contract specifications (settlement times, tick sizes, the AM/PM roots) are used as facts. |
| Yahoo Finance ([Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), last updated 2026-08-04) | Section 2(d)(ix): you agree not to "access or collect data, or attempt to access or collect data, from our Services using any automated means, devices, programs, algorithms or methodologies, including but not limited to robots, spiders, scrapers, data mining tools, or data gathering or extraction tools, for any purpose without our express, prior permission." | Never fetched by the app or the package. A visitor may upload a chain they exported themselves; `read_chain` reads the yfinance layout (`tests/fixtures/chain_yfinance.csv` is fabricated). |
| yfinance library ([github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance), Apache License 2.0) | Its README: yfinance "is intended for research and educational purposes. You should refer to Yahoo!'s terms of use ... for details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only." | The optional `data` extra installs it for local use; no module of the package imports it. |
| FRED, Federal Reserve Bank of St. Louis ([API terms of use](https://fred.stlouisfed.org/docs/api/terms_of_use.html)) | An API key is required ("In order to use the FRED® API, you must have register for an API Key"). "Data series available through the FRED® API, may be owned by third parties and subject to copyright restrictions." Using such series beyond personal use needs the owner's permission. The [S&P 500 series](https://fred.stlouisfed.org/series/SP500): "Copyright © 2016, S&P Dow Jones Indices LLC. All rights reserved. Reproduction of S&P 500 in any form is prohibited except with the prior written permission of S&P Dow Jones Indices LLC". The [VIX series](https://fred.stlouisfed.org/series/VIXCLS): "Copyright, 2016, Chicago Board Options Exchange, Inc. Reprinted with permission." | Not used, and no index history is stored. The historical stress scenarios (`options_analyzer/scenarios.py`) quote a handful of published index moves and VIX levels as facts. |
| Federal Reserve Board, H.15 Selected Interest Rates ([www.federalreserve.gov/releases/h15](https://www.federalreserve.gov/releases/h15/)) | [Website disclaimer](https://www.federalreserve.gov/disclaimer.htm): "Unless otherwise indicated, information on Board's website is in the public domain and may be copied and distributed without permission. Please cite to the Board as the source of the information." | The planned bundled Treasury curve (roadmap item DT-3) may use it, cited as "Board of Governors of the Federal Reserve System, H.15 Selected Interest Rates". Nothing is bundled yet. |
| US Treasury, Daily Treasury Par Yield Curve Rates ([home.treasury.gov interest-rate statistics](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics)) | A work of the US Government, which has no copyright protection in the United States (17 U.S.C. § 105). The site's policy page stated no reuse terms when read. | A planned public-domain fallback for the curve (DT-3), cited to the Treasury. Nothing is bundled yet. |
| NYSE holidays and trading hours ([www.nyse.com/markets/hours-calendars](https://www.nyse.com/markets/hours-calendars); NYSE Group news release of 2025-12-23 for 2026-2028) | The dates are facts. The pages themselves are © Intercontinental Exchange, Inc. | `options_analyzer/clock.py` derives the holidays from NYSE's rules; `tests/fixtures/nyse_holidays_2026_2028.csv` transcribes the published dates, with source and retrieval date, to check them. |
| OCC option symbology | A naming format: root, expiry as YYMMDD, C or P, strike x 1000. | Parsed and written by `options_analyzer/market_data/schema.py`; no data. |

## Provenance header for `tests/fixtures`

Every fixture begins with comment lines (`#`). The first one names its provenance:

```text
# Provenance: fabricated | synthetic | published-facts | public-domain
# Source: what it is and who published it   (required for published-facts and public-domain)
# URL: https://...                           (one or more; required for published-facts and public-domain)
# Retrieved: YYYY-MM-DD                      (required for published-facts and public-domain)
# Any further notes, e.g. the layout the values are written in.
```

- `fabricated`: values invented by hand or by a script, written in a real layout.
- `synthetic`: values generated by the package's own models (for example `sample_chain`).
- `published-facts`: facts transcribed from a public source, such as exchange holiday dates.
- `public-domain`: data in the public domain, such as US Government yields.

Option quotes, prices and index histories must be `fabricated` or `synthetic`. Data files belong in `tests/fixtures/`
only; `.gitignore` keeps local data and caches out of the repository.

## Bundled assets other than data

The IBM Plex fonts are under the SIL Open Font License 1.1, and the port of Let's Be Rational keeps its author's notice;
see `THIRD_PARTY_NOTICES.md`.
