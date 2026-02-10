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
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange
from qrisp.operators import QubitOperator, FermionicOperator
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def QET(H: BlockEncoding | FermionicOperator | QubitOperator, p: "ArrayLike", kind: Literal["Polynomial", "Chebyshev"] = "Polynomial", rescale: bool = True) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator via `Quantum Eigenvalue Transform <https://arxiv.org/pdf/2312.00723>`_.

    For a block-encoded operator $H$ and a **real, fixed parity** polynomial $p(x)$, this method returns 
    a BlockEncoding of the operator $p(H)$.

    The Quantum Eigenvalue Transform is described as follows:
    
    * Given a Hermitian operator $H=\sum_i\lambda_i\ket{\lambda_i}\bra{\lambda_i}$ where $\lambda_i\in\mathbb R$ are the eigenvalues for the eigenstates $\ket{\lambda_i}$, 
    * A quantum state $\ket{\psi}=\sum_i\alpha_i\ket{\lambda_i}$ where $\alpha_i\in\mathbb C$ are the amplitudes for the eigenstates $\ket{\lambda_i}$, 
    * A (complex) polynomial $p(z)$,

    this transformation prepares a state proportional to

    .. math::

        p(H)\ket{\psi}=\sum_i p(\lambda_i)\ket{\lambda_i}\bra{\lambda_i}\sum_j\alpha_j\ket{\lambda_j}=\sum_i p(\lambda_i)\alpha_i\ket{\lambda_i}

    Parameters
    ----------
    H : BlockEncoding | FermionicOperator | QubitOperator
        The Hermitian operator to be transformed.
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : {"Polynomial", "Chebyshev"}
        The basis in which the coefficients are defined. 

        - ``"Polynomial"``: $p(x) = \sum c_i x^i$

        - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

        Default is ``"Polynomial"``.
    rescale : bool
        If True (default), the method returns a block-encoding of $p(H)$.
        If False, the method returns a block-encoding of $p(H/\alpha)$ where $\alpha$ is the normalization factor for the block-encoding of the operator $H$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the transformed operator $p(H)$.

    Notes
    -----
    - Improved efficiency compared to GQET for a real, fixed parity polynomial $p(x)$.

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
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    Finally, compare the quantum simulation result with the classical solution:
            
    ::

        c = (np.eye(4) + A @ A) @ b
        c = c / np.linalg.norm(c)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
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

    if isinstance(H, (QubitOperator, FermionicOperator)):
        H = BlockEncoding.from_operator(H)  

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    if rescale:
        p = _rescale_poly(H.alpha, p, kind=kind)
    if kind=="Polynomial":
        p = poly2cheb(p)

    m = len(H._anc_templates)
    d = len(p)
    # Angles theta and lambda vanish for real polynomials https://arxiv.org/abs/2503.03026.
    # Implementation based on conjecture: phi has fixed parity iff p has fixed parity.
    # Combine two consecutive walk operators: (R U) (R U) = (R U R U ) = (R U_dg R U) = T_2
    # This is qubitization step even if the block encoding unitary is not Hemitian.
    # https://math.berkeley.edu/~linlin/qasc/qasc_notes.pdf
    # If the parity is odd, there is a single (R U) at the end which is not followed by a rotation.
    # Since the QET is only successful if all ancillas are |0>, there is no need to control-(R U) 
    # and the refelction acts as identity.
    
    angles, alpha = gqsp_angles(p)
    phi = angles[1][::-1]

    def T2(*args):
        with conjugate(H.unitary)(*args):
            reflection(args[:m])
        reflection(args[:m])

    def new_unitary(*args):

        rx(-2*phi[0], args[0])
        for i in jrange(1, (d + 1) // 2):
            with control(args[0], ctrl_state=0):
                T2(*args[1:])
            rx(-2 * phi[2*i], args[0])

        with control(is_odd):
            H.unitary(*args[1:])

    new_anc_templates = [QuantumBool().template()] + H._anc_templates
    return BlockEncoding(alpha, new_anc_templates, new_unitary, num_ops=H.num_ops, is_hermitian=False)