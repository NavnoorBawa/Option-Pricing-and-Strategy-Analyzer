# Contributing

Thanks for helping improve the Options Pricing & Strategy Analyzer.

## Set-up

```bash
git clone https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer.git
cd Option-Pricing-and-Strategy-Analyzer
pip install -r requirements-dev.txt
streamlit run main.py --client.showErrorDetails full
```

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

## Reporting bugs

Open an issue with the inputs that reproduce the problem (the sidebar settings are in the page URL), what you
expected and what you saw.
