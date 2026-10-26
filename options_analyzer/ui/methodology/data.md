### Conventions
* Time to expiry $T$ = calendar days / 365. Rates $r$ and dividend yield $q$ are continuously compounded.
* Greeks are shown per calendar day (theta, charm, veta, color), per volatility point (vega, vanna, zomma),
  per 1 % rate move (rho) and per \$1 move in the underlying (delta, gamma, speed). Every time Greek is the
  change as calendar time *passes*: theta is usually negative for a long option (deep in-the-money puts and
  calls on high-dividend stocks can gain value as time passes).
* Value-at-Risk horizons are in trading days (252 per year), because volatility accrues with trading time.
* Strategy quantities are signed (+ long, − short); one unit is an option on one share. The contract
  multiplier converts to dollar P&L.
