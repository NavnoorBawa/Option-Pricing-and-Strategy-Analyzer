# Contributing

Thanks for helping improve the Options Pricing & Strategy Analyzer.

## Set-up

```bash
git clone https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer.git
cd Option-Pricing-and-Strategy-Analyzer
pip install -r requirements-dev.txt          # the app, plus the pinned test and lint tools
streamlit run main.py --client.showErrorDetails full
```

`requirements-dev.txt` is the one place the development tools are pinned (CI and the dev container install from it).
The independent libraries that the oracle tests compare against (QuantLib, Let's Be Rational) are test-only and live
in `requirements-oracle.txt`, which includes the dev tools:

```bash
pip install -r requirements-oracle.txt
REQUIRE_ORACLES=1 pytest -m oracle           # as CI does: a missing library or a skipped oracle test fails
pytest -m oracle                             # without REQUIRE_ORACLES, missing libraries skip the oracle tests
```

Running the tests:

```bash
pytest -n auto --dist loadfile               # in parallel, as CI does (about 75 s on 4 cores, 3.5 min serially)
pytest -m "not slow"                         # markers: oracle, slow, network, e2e (see pyproject.toml)
HYPOTHESIS_PROFILE=ci pytest                 # Hypothesis profiles: dev (default), ci, nightly
```

Tests run with outbound network access blocked (`tests/conftest.py`); a test that really needs the internet must be
marked `@pytest.mark.network`. Proxy settings (`HTTPS_PROXY` and the like) are switched off inside unmarked tests,
so a proxy on your machine cannot fetch pages for them and a test that passes locally also passes in CI, which has no
proxy. The shared Streamlit fixtures `app` and `new_app` are also in `tests/conftest.py`.

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest
```

Continuous integration runs the same commands on every supported Python version and against the oldest declared
dependency versions. Development tools are pinned in `requirements-dev.txt` and updated by Dependabot.

## Guidelines

- Keep the numerical core (`options_analyzer/*.py`) free of Streamlit imports; the interface lives in
  `options_analyzer/ui/`.
- Every numerical change needs a test against an independent reference: a closed form, a high-precision
  (`mpmath`) computation, a published value or an invariant such as put-call parity.
- Adding a strategy template: add a `StrategyTemplate` to `TEMPLATES` in `options_analyzer/strategies.py` (with its
  `structure` text and `lowest_offset`); `tests/test_strategies.py` and `tests/test_app.py` then exercise it
  automatically.
- Use plain units in the interface and say which convention a number follows (per day, per volatility point, per
  share or per position).
- After an interface change, refresh the README screenshots with `python scripts/screenshots.py` (needs
  `pip install playwright pillow` and `playwright install chromium`).

## Reporting bugs

Open an issue with the inputs that reproduce the problem (the sidebar settings are in the page URL), what you
expected and what you saw.
