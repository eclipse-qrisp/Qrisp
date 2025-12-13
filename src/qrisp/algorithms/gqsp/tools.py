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
    QuantumArray,
    QuantumVariable,
    QuantumBool,
    QuantumFloat,
    h,
    u3,
    z,
    conjugate,
    control,
    invert,
    gphase,
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.algorithms.gqsp.gqsp import GQSP, polynomial_to_chebyshev
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp


def apply(qarg, H, p, basis="polynomial"):
    r"""
    Applies are polynomial transformation of a Hamiltonian to a quantum state.

    Parameters
    ----------
    qarg : QuantumVariable

    H : QubitOperator

    p : ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.


    Returns
    -------
    QuantumBool

    Examples
    --------

    """

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    #_, coeffs = H.unitaries()
    #alpha = np.sum(coeffs)
    #scaling_exponents = np.arange(len(p))
    #scaling_factors = np.power(alpha, scaling_exponents)

    #p = p * scaling_factors

    if basis=="polynomial":
        print("polynomial")
        p = polynomial_to_chebyshev(p)

    U, state_prep, n = H.pauli_block_encoding()

    # Qubitization step: RU^k is a block-encoding of T_k(H)
    def RU(case, operand):
        U(case, operand)
        reflection(case, state_function=state_prep)

    case = QuantumFloat(n)

    with conjugate(state_prep)(case):
        qbl = GQSP([case, qarg], RU, p, k=0)

    return qbl, case
