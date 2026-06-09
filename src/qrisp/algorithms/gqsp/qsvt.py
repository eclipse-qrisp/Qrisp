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

from qrisp import (
    QuantumBool,
    invert,
    rz,
    h,
    mcx,
    x,
)
from qrisp.environments import conjugate, control
from qrisp.algorithms.gqsp.gqsp_angles import qsvt_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange, q_cond
from qrisp.operators import QubitOperator, FermionicOperator
from jax import numpy as jnp
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def QSVT(
    H: BlockEncoding | FermionicOperator | QubitOperator,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    parity: Literal["odd", "even"] = "odd",
    rescale: bool = True,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator via `Quantum Singular Value Transformation <https://arxiv.org/abs/1806.01838>`_.

    For a block-encoded operator $H$ with `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_ $H = U \Sigma V^{\dagger}$ for unitaries $U, V$,
    and a (complex) polynomial $p(z)$, this method returns a BlockEncoding of either operator:

    - $p_{odd}(H)=U p_{odd}(\Sigma) V^{\dagger}$

    - $p_{even}(H)=V p_{even}(\Sigma) V^{\dagger}$

    where $p=p_{odd}+p_{even}$ is decomposed into odd and even parity parts.

    Parameters
    ----------
    H : BlockEncoding | FermionicOperator | QubitOperator
        The operator to be transformed. Unlike QET, this operator is not strictly required to be Hermitian.
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : {"Polynomial", "Chebyshev"}
        The basis in which the coefficients are defined.

        - ``"Polynomial"``: $p(x) = \sum c_i x^i$

        - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

        Default is ``"Polynomial"``.
    parity : {"odd", "even"}
        The parity part of $p=p_{odd}+p_{even}$ to be applied.

        - ``"odd"``: The odd part $p_{odd}(H)$ is applied.

        - ``"even"``: The even part $p_{even}(H)$ is applied.

        Default is ``"odd"``.
    rescale : bool
        If True (default), the method returns a block-encoding of $p(H)$.
        If False, the method returns a block-encoding of $p(H/\alpha)$ where $\alpha$ is the normalization factor for the block-encoding of the operator $H$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the transformed operator $p_{odd}(H)$ or $p_{even}(H)$.

    Examples
    --------

    Define a Hermitian matrix $H\_mat$ and a vector $\vec{b}$.

    ::

        import numpy as np

        H = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                    [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                    [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                    [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

    Generate a BlockEncoding of $H$ and use QSVT to obtain a BlockEncoding of $p(H)$
    for an odd parity polynomial ($p(x) = x + x^3$).

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import QSVT

        BH = BlockEncoding.from_array(H)

        # Applying polynomial p(x) = 0*x^0 + 1*x^1 + 0*x^2 + 1*x^3
        BH_poly = QSVT(BH, np.array([0., 1., 0., 1.]), parity="odd")

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumVariable(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():
            operand = BH_poly.apply_rus(prep_b)()
            return operand

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    Finally, compare the quantum simulation result with the classical Singular Value polynomial transformation:

    ::

        # Compute the SVD
        U, S, Vh = np.linalg.svd(H)

        # Apply odd polynomial z + z^3 to singular values
        S_poly = S + S ** 3

        # Reconstruct transformed matrix
        H_poly = (U @ np.diag(S_poly) @ Vh).conj().T

        c = H_poly @ b
        c = c / np.linalg.norm(c)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.07018199 0.56676065 0.67904296 0.46125645]
        # CLASSICAL SOLUTION
        # [-0.07018194  0.56676073  0.67904288  0.46125647]

    """

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
    if kind == "Polynomial":
        p = poly2cheb(p)

    phi, new_alpha = qsvt_angles(p, parity=parity)

    m = len(H._anc_templates)

    def reflection(args, phase):
        qubits = sum([arg.reg for arg in args[1 : m + 1]], [])
        with conjugate(mcx)(qubits, args[0], ctrl_state=0):
            rz(phase, args[0])

    def even(args):
        H.unitary(*args[1:])

    def odd(args):
        with invert():
            H.unitary(*args[1:])

    def new_unitary(*args):
        h(args[0])

        d = len(phi) - 1

        for i in jrange(0, d):
            reflection(args, phase=2 * phi[d - i])
            q_cond(i % 2 == 0, even, odd, args)
        reflection(args, phase=2 * phi[0])

        h(args[0])

    new_anc_templates = [QuantumBool().template()] + H._anc_templates
    return BlockEncoding(new_alpha, new_anc_templates, new_unitary, is_hermitian=False)
