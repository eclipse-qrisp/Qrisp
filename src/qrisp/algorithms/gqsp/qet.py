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
import jax.numpy as jnp
from qrisp import QuantumBool
from qrisp.core.gate_application_functions import rx
from qrisp.environments import conjugate, control
from qrisp.alg_primitives.reflection import reflection
from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import QubitOperator
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def QET(H: BlockEncoding | QubitOperator, p: "ArrayLike", kind: Literal["Polynomial", "Chebyshev"] = "Polynomial") -> BlockEncoding:
    r"""
    Performs `Quantum Eigenvalue Transform <https://arxiv.org/pdf/2312.00723>`_.
    Applies **real, fixed parity** polynomial transformations on the eigenvalues of a Hermitian operator.

    The Quantum Eigenvalue Transform is described as follows:
    
    * Given a Hermitian operator $H=\sum_i\lambda_i\ket{\lambda_i}\bra{\lambda_i}$ where $\lambda_i\in\mathbb R$ are the eigenvalues for the eigenstates $\ket{\lambda_i}$, 
    * A quantum state $\ket{\psi}=\sum_i\alpha_i\ket{\lambda_i}$ where $\alpha_i\in\mathbb C$ are the amplitudes for the eigenstates $\ket{\lambda_i}$, 
    * A (complex) polynomial $p(z)$,

    this transformation prepares a state proportional to

    .. math::

        p(H)\ket{\psi}=\sum_i p(\lambda_i)\ket{\lambda_i}\bra{\lambda_i}\sum_j\alpha_j\ket{\lambda_j}=\sum_i p(\lambda_i)\alpha_i\ket{\lambda_i}

    Parameters
    ----------
    H : BlockEncoding | QubitOperator
        The Hermitian operator.
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
        Polynomial must be real and have even or odd parity.
    kind : {"Polynomial", "Chebyshev"}
        The kind of ``p``. The default is ``"Polynomial"``.

    Returns
    -------
    BlockEncoding
        A block encoding of $p(H)$.

    Examples
    --------

    Define a Hermitian matrix $A$ and a vector $\vec{b}$.

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                    [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                    [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                    [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

    Generate a BlockEncoding of $A$ and use QET to obtain a BlockEncoding of $p(H)$ 
    for a real polynomial of even parity.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import QET

        BA = BlockEncoding.from_array(A)

        BA_poly = QET(BA, np.array([1.,0.,1.]))

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumVariable(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():
            operand = BA_poly.apply_rus(prep_b)()
            return operand

        res_dict = main()

        # Convert measurement probabilities to (absolute values of) amplitudes
        for k, v in res_dict.items():
            res_dict[k] = v**0.5
        q = np.array([res_dict.get(key, 0) for key in range(len(b))])

    Finally, compare the quantum simulation result with the classical solution:
            
    ::

        c = (np.eye(4) + A @ A) @ b
        c = c / np.linalg.norm(c)

        print("QUANTUM SIMULATION\n", q, "\nCLASSICAL SOLUTION\n", c)
        #QUANTUM SIMULATION
        # [0.02405799 0.57657687 0.61493257 0.53743675] 
        #CLASSICAL SOLUTION
        # [-0.024058    0.57657692  0.61493253  0.53743673]

    """

    #is_real = jnp.allclose(p.imag, 0, atol=1e-6)
    #is_even = jnp.allclose(p[1::2], 0, atol=1e-6)
    is_odd = jnp.allclose(p[0::2], 0, atol=1e-6)

    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )

    if isinstance(H, QubitOperator):
        H = H.pauli_block_encoding()    

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    alpha = H.alpha
    scaling_exponents = jnp.arange(len(p))
    scaling_factors = jnp.power(alpha, scaling_exponents)

    # Convert to Polynomial for rescaling
    if kind=="Chebyshev":
        p = cheb2poly(p)

    p = p * scaling_factors

    p = poly2cheb(p)

    m = len(H.anc_templates)
    d = len(p)
    # Angles theta and lambda vanish for real polynomials.
    # Implementation based on conjecture: phi has fixed parity iff p has fixed parity.
    # Combine two consecutive walk operators: (R U) (R U) = (R U R U ) = (R U_dg R U) = T_2
    # https://math.berkeley.edu/~linlin/qasc/qasc_notes.pdf
    angles, alpha = gqsp_angles(p)
    phi = angles[1][::-1]

    def T1(*args):
        H.unitary(*args)
        reflection(args[:m])

    def T2(*args):
        with conjugate(H.unitary)(*args):
            reflection(args[:m])
        reflection(args[:m])

    def new_unitary(*args):

        rx(-2*phi[0], args[0])
        for i in range(1, (d + 1) // 2):
            with control(args[0], ctrl_state=0):
                T2(*args[1:])
            rx(-2 * phi[2*i], args[0])

        with control(is_odd):
            with control(args[0], ctrl_state=0):
                T1(*args[1:])

    new_anc_templates = [QuantumBool().template()] + H.anc_templates
    return BlockEncoding(alpha, new_anc_templates, new_unitary)