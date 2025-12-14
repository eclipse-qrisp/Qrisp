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
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp


def apply(qarg, H, p, kind="Polynomial"):
    r"""
    Applies are polynomial transformation of a Hamiltonian to a quantum state.

    Parameters
    ----------
    qarg : QuantumVariable

    H : QubitOperator

    p : ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.

    kind : str, optinal
        ``"Polynomial"``or ``"Chebyshev"``.
        

    Returns
    -------
    qbl : QuantumBool
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.
    case : QuantumFloat
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.

    Examples
    --------

    ::

        from qrisp import *
        from qrisp.gqsp import *
        from qrisp.operators import X, Y, Z
        from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function
        import numpy as np
        import networkx as nx


        def generate_1D_chain_graph(L):
            graph = nx.Graph()
            graph.add_edges_from([(k, (k+1)%L) for k in range(L-1)]) 
            return graph


        # Define Heisenberg Hamiltonian 
        L = 10
        G = generate_1D_chain_graph(L)
        H = sum((X(i)*X(j) + Y(i)*Y(j) + Z(i)*Z(j)) for i,j in G.edges())

        M = nx.maximal_matching(G)
        U0 = create_heisenberg_init_function(M)


        # Define initial state preparation function
        def psi_prep():
            operand = QuantumVariable(H.find_minimal_qubit_amount())
            U0(operand)
            return operand


        # Calculate the energy
        E = H.expectation_value(psi_prep, precision=0.001)()
        print(E)

    ::

        poly = np.array([1., 2., 1.])

        @RUS
        def filtered_psi_prep():

            operand = psi_prep()

            qbl, case = apply(operand, H, poly, kind="Polynomial")

            success_bool = (measure(qbl) == 0) & (measure(case) == 0)
            return success_bool, operand

    ::

        @jaspify(terminal_sampling=True)
        def main(): 

            E = H.expectation_value(filtered_psi_prep, precision=0.001)()
            return E

        print(main())
        # -16.67236953920006 

    Finally, we compare the quantum simulation to the corresponding numpy calculation:

    ::

        # Calculate energy for |psi_0>
        H_arr = H.to_array()
        psi_0 = psi_prep().qs.statevector_array()
        E_0 = (psi_0.conj() @ H_arr @ psi_0).real
        print("E_0", E_0)

        # Calculate energy for |psi> = poly(H) |psi0>
        I = np.eye(H_arr.shape[0])
        psi = (I + H_arr) @ (I + H_arr) @ psi_0
        psi = psi / np.linalg.norm(psi)
        E = (psi.conj() @ H_arr @ psi).real
        print("E", E)

    """

    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    _, coeffs = H.unitaries()
    alpha = np.sum(coeffs)
    scaling_exponents = np.arange(len(p))
    scaling_factors = np.power(alpha, scaling_exponents)

    # Convert to Polynomial for rescaling
    if kind=="Chebyshev":
        p = cheb2poly(p)

    p = p * scaling_factors

    p = poly2cheb(p)

    U, state_prep, n = H.pauli_block_encoding()

    # Qubitization step: RU^k is a block-encoding of T_k(H)
    def RU(case, operand):
        U(case, operand)
        reflection(case, state_function=state_prep)

    case = QuantumFloat(n)

    with conjugate(state_prep)(case):
        qbl = GQSP([case, qarg], RU, p, k=0)

    return qbl, case


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

    qbl, case = apply(qarg, H, cheb, kind="Chebyshev")

    return qbl, case