"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

import numpy as np
from qrisp import (
    QuantumFloat,
    QuantumBool,
    conjugate,
    x,
    invert,
    control
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly

def GQSVT(qarg, H, p, kind="Polynomial"):
    
    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )

    # Rescaling of the polynomial to account for scaling factor alpha of pauli block-encoding
    H = H.hermitize().to_pauli()
    _, coeffs = H.unitaries()
    alpha = np.sum(np.abs(coeffs))
    scaling_exponents = np.arange(len(p))
    scaling_factors = np.power(alpha, scaling_exponents)

    # Convert to Polynomial for rescaling
    if kind=="Chebyshev":
        p = cheb2poly(p)

    p = p * scaling_factors

    p = poly2cheb(p)

    U, state_prep, n = H.pauli_block_encoding()

    # Qubitization step based on the Hermitization GQSVT approach
    def RU_tilde(case, aux, operand):
        x(aux)
        with control(aux, ctrl_state=0):
            U(case, operand)
            reflection(case, state_function=state_prep)
        
        with control(aux, ctrl_state=1):
            with invert():
                U(case, operand)
            reflection(case, state_function=state_prep)

    case = QuantumFloat(n)
    aux = QuantumBool()

    with conjugate(state_prep)(case):
        qbl = GQSP([case, aux, qarg], RU_tilde, p, k=0)

    return qbl, aux, case