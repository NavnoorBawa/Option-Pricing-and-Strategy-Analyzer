# Third-party notices

The Options Pricing & Strategy Analyzer is released under the MIT License (see `LICENSE`). It contains, or is
adapted from, the third-party work below. The notices are reproduced as their licences require.

## Let's Be Rational, by Peter Jäckel

`options_analyzer/lets_be_rational.py` is a port to Python and NumPy of the reference implementation of
"Let's Be Rational", revision 1520 of 2024-02-16, downloaded from www.jaeckel.org/LetsBeRational.7z. The
algorithm is described in P. Jäckel, "Let's Be Rational", *Wilmott*, January 2015, pp. 40-53
(www.jaeckel.org/LetsBeRational.pdf). The module header carries the same notice. The reference implementation's
notice:

```text
======================================================================================
Copyright © 2013-2024 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

WARRANTY DISCLAIMER
The Software is provided "as is" without warranty of any kind, either express or implied,
including without limitation any implied warranties of condition, uninterrupted use,
merchantability, fitness for a particular purpose, or non-infringement.
======================================================================================
```

## IBM Plex fonts

`options_analyzer/ui/static/fonts/` (copied to `static/fonts/` for `streamlit run main.py`) contains IBM Plex Sans
and IBM Plex Mono, Copyright © 2017 IBM Corp. with Reserved Font Name "Plex", licensed under the SIL Open Font
License, Version 1.1. The licence text is in `LICENSE.txt` in each of those folders.

## Used only by the tests (not distributed)

These libraries are installed from `requirements-oracle.txt` and `requirements-dev.txt` to check the numbers. The
package never imports them.

| Library | Licence | Used for |
| --- | --- | --- |
| lets-be-rational 1.1.2 (`py_lets_be_rational`), © 2017 Gammon Capital LLC | MIT | an independent port of Let's Be Rational (revision of 2013) to compare with |
| cody-special 1.0.0, piecewise-rational 1.0.0 | MIT | dependencies of lets-be-rational |
| QuantLib 1.44 | BSD-3-Clause | reference prices, trees and finite-difference engines |
| mpmath 1.4.1 | BSD-3-Clause | high-precision reference values |
