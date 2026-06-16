"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_fourier_adder import (
    jasp_fourier_adder,
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
    BigInteger,
    bi_modinv,
    bi_montgomery_encode,
    bi_montgomery_decode,
    bi_extended_euclidean,
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
    best_montgomery_shift,
)


def __getattr__(name):
    import importlib
    lazy = {
        "jasp_mod_adder": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_adder",
        "jasp_controlling_multiplyer": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_multiplyers",
        "jasp_squaring": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_multiplyers",
        "jasp_multiplyer": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_multiplyers",
        "q_montgomery_reduction": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery",
        "qq_montgomery_multiply": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery",
        "cq_montgomery_multiply": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery",
        "cq_montgomery_multiply_inplace": "qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery",
    }
    if name in lazy:
        mod = importlib.import_module(lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
