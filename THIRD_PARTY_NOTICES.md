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

## QuantLib's QdFp American engine (design reference and reference prices)

`options_analyzer/american.py` follows the design of QuantLib's `QdFpAmericanEngine` and `QdPlusAmericanEngine`
(`ql/pricingengines/vanilla/qdfpamericanengine.cpp` and `qdplusamericanengine.cpp` in QuantLib 1.44, Copyright (C)
2022 Klaus Spanderen): the fixed-point equations of Andersen, Lake and Offengenden (2016) on Chebyshev nodes in the
square root of the time to expiry, the QD+ initial guess of Li (2010) solved by Halley's method, one Jacobi-Newton
step followed by plain fixed-point steps, and the density form of the equation when |r - q| < 1e-3. The module is
written from the papers; no QuantLib source code is copied into it. The reference prices in
`tests/fixtures/alo_grid_qdfp.npz` were computed with QuantLib 1.44 by `scripts/gen_alo_reference.py`. QuantLib is
distributed under the following licence (from its `LICENSE.TXT`, which lists the copyright holders):

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    Neither the names of the copyright holders nor the names of the QuantLib
    Group and its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
