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
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.gqet import GQET
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp


def hamiltonian_simulation(qarg, H, t):
    r"""
    Performs Hamiltonian simulation.

    .. math ::

        e^{-itz} = J_0(t) + 2\sum{m=0}^{\infty}(-i)^mJ_m(t)T_m(z)

    where $T_m(z)$ are Chebyshev polynomials of the first kind, and $J_m(x)$ are Bessel functions of the first kind.

    """

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    _, coeffs = H.unitaries()
    alpha = np.sum(coeffs)

    from scipy.special import jv

    #N = int(2 * np.ceil(alpha))
    N = 5

    J_values = np.array([jv(m, t) for m in range(N)])
    factors = np.array([2*(-1.j)**m for m in range(N)])
    factors[0] = 1

    # Coefficients of Chebyshev series
    cheb = factors * J_values

    qbl, case = GQET(qarg, H, cheb, kind="Chebyshev")

    return qbl, case