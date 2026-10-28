## Conventions and units

* Rates $r$ and dividend yield $q$ are continuously compounded.
* Greeks are shown per calendar day (theta, charm, veta, color), per volatility point (vega, vanna, zomma),
  per 1 % rate move (rho) and per \$1 move in the underlying (delta, gamma, speed). Every time Greek is the
  change as calendar time *passes*: theta is usually negative for a long option (deep in-the-money puts and
  calls on high-dividend stocks can gain value as time passes).
* Theta per day is theta / 365, the instantaneous rate. The *1-day decay* reprices instead: the value with one
  calendar day less to expiry, spot and implied volatility unchanged, minus the value now. An option that expires
  within the day is worth its intrinsic value at the current spot, so it loses its whole time value: the
  *decay to expiry*. The two agree while options have weeks to run and part near expiry: a 1-day at-the-money
  put worth \$21.56 (spot 5,800, volatility 18 %, rate 4.3 %, dividend yield 1.3 %) has a theta of −\$10.66 per
  day and loses all \$21.56. The decay is the headline (the Strategy card, the first of the two rows in the Greeks
  table) when an option expires within the day or when a decay $D$ and theta differ by more than 10 % of the
  larger, ${|D - \theta/365| > 0.1\,\max(|D|, |\theta/365|)}$; the card's help gives both figures.
* Strategy quantities are signed (+ long, − short); one unit is an option on one share. The contract
  multiplier converts to dollar P&L.
