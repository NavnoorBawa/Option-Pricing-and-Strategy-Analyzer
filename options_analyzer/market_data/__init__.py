"""Option-chain data: a realistic synthetic chain and adapters for chain files that users download themselves.

The public demo runs on synthetic data and on files that visitors upload; it never fetches quotes from a server (see
``docs/data_policy.md``). Every source produces the same canonical chain table (:mod:`.schema`):

* :func:`sample_chain` (:mod:`.synthetic`) prices a chain from a known model, labelled
  :data:`SYNTHETIC_LABEL`, for the demo and as a test oracle;
* :func:`read_chain` (:mod:`.adapters`) reads the app's long layout, a yfinance-style export or Cboe's
  manual-download layout;
* :func:`clean_chain` drops the contracts that cannot be quoted, and :func:`to_quotes` turns a chain into the table
  that :func:`options_analyzer.vol_surface.parse_quotes` reads.
"""

from .adapters import LAYOUTS, clean_chain, read_chain, to_quotes
from .schema import CHAIN_COLUMNS, ChainSchemaError, OccSymbol, occ_symbol, parse_occ_symbol, validate_chain
from .synthetic import SYNTHETIC_LABEL, SyntheticChain, sample_chain

__all__ = [
    "CHAIN_COLUMNS",
    "LAYOUTS",
    "SYNTHETIC_LABEL",
    "ChainSchemaError",
    "OccSymbol",
    "SyntheticChain",
    "clean_chain",
    "occ_symbol",
    "parse_occ_symbol",
    "read_chain",
    "sample_chain",
    "to_quotes",
    "validate_chain",
]
