"""Exports that carry their provenance inside the file (UI spec 2.13, UI-28).

A CSV of SPX theos without the header's SYNTHETIC chip reads as market data, so every file the app hands out says
where its numbers come from, in the file itself:

* CSV: comment lines before the header row, which ``pandas.read_csv(..., comment="#")`` skips (and so does the
  app's own reader, :func:`read_csv`)::

    # source: SYNTHETIC — not market data
    # as_of: 2026-10-23T16:00-04:00 (cash close)
    # model: BSM · closed form; dynamics: sticky strike; forwards: parity; fit: SVI arb-free
    # app: v3.0.0 (abc1234); seed: 0

* XLSX: a ``Provenance`` sheet, first, with the same fields;
* JSON: a top-level ``provenance`` object;
* file names that carry the source and the time: ``spx_chain_SYNTHETIC_20261023T1600.csv``.

Every ``st.download_button`` of the app builds its file through this module (``tests/test_data_policy.py`` checks
both), and passes the builder itself, so that a file is built only when it is clicked.
"""

from __future__ import annotations

import functools
import io
import json as _json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__, clock
from ..market_data.synthetic import SYNTHETIC_LABEL
from . import market as mk
from .header import model_badge
from .inputs import Inputs

#: The seed of the synthetic chain the index preset prices (``market_data.sample_chain``'s default).
SYNTHETIC_SEED = 0
#: Sources, by the tag that file names carry.
SYNTHETIC, UPLOAD, TEXTBOOK, PARAMETRIC = "SYNTHETIC", "UPLOAD", "TEXTBOOK", "PARAMETRIC"
#: The synthetic chain's source line: its label (``synthetic.SYNTHETIC_LABEL``) led by the tag in capitals.
SYNTHETIC_SOURCE = SYNTHETIC_LABEL.replace("Synthetic", SYNTHETIC, 1)
_REPOSITORY = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def commit() -> str | None:
    """The short hash of the checked-out commit, or None outside a git checkout (an installed package)."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = done.stdout.strip()
    return sha if sha and all(c in "0123456789abcdef" for c in sha) else None


@dataclass(frozen=True)
class Provenance:
    """Where an export's numbers come from.

    Attributes
    ----------
    tag : str
        ``SYNTHETIC``, ``UPLOAD``, ``TEXTBOOK`` or ``PARAMETRIC``: the source in one word, for file names.
    source : str
        The source line: ``SYNTHETIC — not market data``, ``UPLOAD · quotes.csv — the user's file``, ...
    as_of : datetime or None
        The valuation time (New York wall time when naive); None for the textbook market, which has none.
    model, dynamics, forwards, fit : str
        The pricing engine, the smile dynamics, where forwards come from, and the vol fit.
    seed : int or None
        The synthetic chain's seed; None when no random numbers made the data.
    """

    tag: str
    source: str
    as_of: datetime | None
    model: str
    dynamics: str = "sticky strike"
    forwards: str = "parity"
    fit: str = "SVI arb-free"
    seed: int | None = None

    @property
    def app(self) -> str:
        """``v3.0.0 (abc1234)``, without the hash outside a git checkout."""
        sha = commit()
        return f"v{__version__}" + (f" ({sha})" if sha else "")

    @property
    def as_of_text(self) -> str:
        """``2026-10-23T16:00-04:00 (cash close)``, or ``none (textbook inputs)``."""
        if self.as_of is None:
            return "none (textbook inputs: time to expiry from the typed days)"
        wall = clock.to_new_york(self.as_of)
        return f"{wall.isoformat(timespec='minutes')} ({mk.session_note(wall.replace(tzinfo=None))})"

    def fields(self) -> dict[str, str]:
        """The provenance as ordered text fields: source, as_of, model, dynamics, forwards, fit, app, seed."""
        return {
            "source": self.source,
            "as_of": self.as_of_text,
            "model": self.model,
            "dynamics": self.dynamics,
            "forwards": self.forwards,
            "fit": self.fit,
            "app": self.app,
            "seed": "none" if self.seed is None else str(self.seed),
        }

    def lines(self) -> list[str]:
        """The four comment lines of a CSV, without the ``#``.

        Examples
        --------
        >>> p = Provenance(SYNTHETIC, SYNTHETIC_SOURCE, datetime(2026, 10, 23, 16), "BSM · closed form", seed=0)
        >>> print("\n".join(p.lines()[:3]))
        source: SYNTHETIC — not market data
        as_of: 2026-10-23T16:00-04:00 (cash close)
        model: BSM · closed form; dynamics: sticky strike; forwards: parity; fit: SVI arb-free
        """
        f = self.fields()
        return [
            f"source: {f['source']}",
            f"as_of: {f['as_of']}",
            f"model: {f['model']}; dynamics: {f['dynamics']}; forwards: {f['forwards']}; fit: {f['fit']}",
            f"app: {f['app']}; seed: {f['seed']}",
        ]

    def file_name(self, stem: str, extension: str) -> str:
        """``<stem>_<TAG>_<YYYYMMDDTHHMM>.<extension>``: the valuation time, or the export's time (New York) for the
        textbook market.

        Examples
        --------
        >>> Provenance(SYNTHETIC, SYNTHETIC_SOURCE, datetime(2026, 10, 23, 16), "BSM").file_name("spx_chain", "csv")
        'spx_chain_SYNTHETIC_20261023T1600.csv'
        """
        when = clock.to_new_york(self.as_of or datetime.now(clock.NEW_YORK))
        return f"{stem}_{self.tag}_{when:%Y%m%dT%H%M}.{extension}"


def provenance(
    inputs: Inputs,
    *,
    tag: str | None = None,
    source: str | None = None,
    model: str | None = None,
    dynamics: str = "sticky strike",
    forwards: str | None = None,
    fit: str | None = None,
) -> Provenance:
    """The provenance of an export made from the sidebar's market, with any field the export sets itself.

    Under the synthetic index: ``SYNTHETIC — not market data`` at the valuation time, parity forwards, the
    arbitrage-free SVI fit and the chain's seed. Under the textbook market: ``TEXTBOOK``, no valuation time, forwards
    from the typed r and q and no fit.

    Parameters
    ----------
    inputs : Inputs
        The sidebar's market.
    tag, source : str, optional
        Another source (an upload, a parametric surface): its one-word tag and its line.
    model : str, optional
        The engine; default the sidebar model's badge (``BSM · closed form``).
    dynamics : str, default "sticky strike"
        The smile dynamics of a scenario export.
    forwards, fit : str, optional
        Where forwards come from and how vols are fitted, when the export's differ from the market's.

    Returns
    -------
    Provenance
    """
    index = inputs.context is not None
    default_tag = SYNTHETIC if index else TEXTBOOK
    default_source = SYNTHETIC_SOURCE if index else "TEXTBOOK — typed inputs, not market data"
    return Provenance(
        tag=tag or default_tag,
        source=source or default_source,
        as_of=inputs.context.as_of if index else None,
        model=model or model_badge(inputs.model),
        dynamics=dynamics,
        forwards=forwards or ("parity" if index else "S·e^((r−q)T) from the typed r and q"),
        fit=fit or ("SVI arb-free, spread-weighted" if index else "none (one typed vol)"),
        seed=SYNTHETIC_SEED if index and (tag or default_tag) == SYNTHETIC else None,
    )


def csv(frame: pd.DataFrame, prov: Provenance, **to_csv: Any) -> bytes:
    """``frame`` as CSV after the provenance comment lines (UTF-8).

    Examples
    --------
    >>> p = Provenance(SYNTHETIC, SYNTHETIC_SOURCE, datetime(2026, 10, 23, 16), "BSM", seed=0)
    >>> print(csv(pd.DataFrame({"Strike": [5800.0]}), p, index=False).decode().splitlines()[0])
    # source: SYNTHETIC — not market data
    >>> pd.read_csv(io.BytesIO(csv(pd.DataFrame({"Strike": [5800.0]}), p, index=False)), comment="#")
       Strike
    0  5800.0
    """
    header = "".join(f"# {line}\n" for line in prov.lines())
    return (header + frame.to_csv(**{"index": False} | to_csv)).encode("utf-8")


def xlsx(sheets: Mapping[str, pd.DataFrame], prov: Provenance, *, index: bool = False) -> bytes:
    """A workbook with a ``Provenance`` sheet first (field, value), then ``sheets`` in order."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        table = pd.DataFrame({"Field": list(prov.fields()), "Value": list(prov.fields().values())})
        table.to_excel(writer, sheet_name="Provenance", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=index)
    return buffer.getvalue()


def json(payload: Mapping[str, Any], prov: Provenance) -> bytes:
    """``payload`` as JSON with a top-level ``provenance`` object first (UTF-8, two-space indent).

    Examples
    --------
    >>> p = Provenance(SYNTHETIC, SYNTHETIC_SOURCE, None, "SVI")
    >>> _json.loads(json({"slices": []}, p))["provenance"]["source"]
    'SYNTHETIC — not market data'
    """
    body = {"provenance": prov.fields(), **payload}
    return _json.dumps(body, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def builder(function: Callable[..., bytes], *args: Any, **kwargs: Any) -> Callable[[], bytes]:
    """The export as a callable for ``st.download_button(data=...)``, so the file is built only on a click."""
    return functools.partial(function, *args, **kwargs)


def read_csv(data: bytes, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV that may start with provenance comment lines (an export of this app fed back to it): the leading
    lines that start with ``#`` are skipped, and nothing else is touched (a ``#`` inside a cell stays)."""
    lines = data.splitlines(keepends=True)
    start = 0
    while start < len(lines) and lines[start].lstrip().startswith(b"#"):
        start += 1
    return pd.read_csv(io.BytesIO(b"".join(lines[start:])), **kwargs)
