"""The independent libraries the oracle tests compare against.

A plain module rather than part of ``conftest.py``: a ``conftest.py`` in a subdirectory (``tests/oracle/``) takes over
the module name ``conftest``, so test modules must never import from it.
"""

#: Distribution name (as pinned in requirements-oracle.txt or requirements-dev.txt) -> import name.
ORACLE_LIBRARIES = {"QuantLib": "QuantLib", "lets-be-rational": "py_lets_be_rational", "mpmath": "mpmath"}

#: Pinned in requirements-oracle.txt only because an oracle's numbers depend on them; they are not imported directly.
ORACLE_DEPENDENCIES = frozenset({"cody-special", "piecewise-rational"})
