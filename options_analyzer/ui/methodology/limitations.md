## Known limitations

* Sticky strike: Strategy and Risk value every leg with Black–Scholes–Merton at its own surface vol and expiry, and
  the smile does not move with spot in scenarios.
* European exercise outside the Pricing and IV tabs: the textbook market's American engine prices those two, and
  the Strategy tab says what early exercise adds to the sidebar contract.
* Legs on a later expiry than the front one take the front expiry's rate and dividend yield (on the default call
  calendar about 2 bp off the later expiry's own discount rate, \$2.69 a contract).
* The portfolio-margin array is illustrative (FINRA 4210(g)-style), not a broker's requirement.
* Transaction costs beyond a flat commission per contract are not modelled, nor are discrete dividends: the index
  pays a continuous yield.
* The VaR model's backtest runs on a synthetic history: there is no market history in the public app.

This tool is for education and analysis only and is not investment advice.
