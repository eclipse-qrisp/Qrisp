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

from typing import Literal, TYPE_CHECKING

from qrisp import (
    QuantumBool,
    invert,
    rz,
    h,
    mcx
)
from qrisp.environments import conjugate
from qrisp.algorithms.gqsp.gqsp_angles import qsvt_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange, q_cond
from qrisp.operators import QubitOperator, FermionicOperator

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def QSVT(
    A: BlockEncoding | FermionicOperator | QubitOperator,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    parity: Literal["odd", "even"] = "odd",
    rescale: bool = True,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator via `Quantum Singular Value Transformation <https://arxiv.org/abs/1806.01838>`_.

    For a block-encoded operator $A$ with `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_ $A = U \Sigma V^{\dagger}$ for unitaries $U, V$,
    and a (real) polynomial $p(x)$, this method returns a BlockEncoding of either operator:

    - $p_{odd}(A)=U p_{odd}(\Sigma) V^{\dagger}$

    - $p_{even}(A)=V p_{even}(\Sigma) V^{\dagger}$

    where $p=p_{odd}+p_{even}$ is decomposed into odd and even parity parts.

    .. warning::
        If the parity is odd, this deviates from :func:`qrisp.algorithms.gqsp.gqsvt.GQSVT`,
        which returns a BlockEncoding of $p_{odd}(A)=V p_{odd}(\Sigma) U^{\dagger}$, i.e., the Hermitian conjugate.

    Parameters
    ----------
    A : BlockEncoding | FermionicOperator | QubitOperator
        The operator to be transformed. Unlike in (G)QET, this operator does not need to be Hermitian.
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : {"Polynomial", "Chebyshev"}
        The basis in which the coefficients are defined.

        - ``"Polynomial"``: $p(x) = \sum c_i x^i$

        - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

        Default is ``"Polynomial"``.
    parity : {"odd", "even"}
        The parity part of $p=p_{odd}+p_{even}$ to be applied.

        - ``"odd"``: The odd part $p_{odd}(A)$ is applied.

        - ``"even"``: The even part $p_{even}(A)$ is applied.

        Default is ``"odd"``.
    rescale : bool
        If True (default), the method returns a block-encoding of $p(A)$.
        If False, the method returns a block-encoding of $p(A/\alpha)$ where $\alpha$ is the normalization factor for the block-encoding of the operator $A$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the transformed operator $p_{odd}(A)$ or $p_{even}(A)$.

    Examples
    --------

    Define a non-Hermitian matrix $A$ and a vector $\vec{b}$. The matrix $A$ has singular value decomposition
    $A = U \Sigma V^{\dagger}$ for unitary matrices $U, V$.

    ::

        import numpy as np

        N = 4
        A = np.eye(N, k=1) + 3 * np.eye(N)
        A[N-1,0] = 1

        b = np.array([1,0,0,0])

        print(A)
        # [[3. 1. 0. 0.]
        # [0. 3. 1. 0.]
        # [0. 0. 3. 1.]
        # [1. 0. 0. 3.]]

    Generate a BlockEncoding of $A$ and use QSVT to obtain a BlockEncoding of $p(A)=U p(\Sigma) V^{\dagger}$
    for an odd parity polynomial.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import QSVT

        def U0(qv): pass
        def U1(qv): qv-=1
        BE = BlockEncoding.from_lcu(np.array([3,1]), [U0,U1])

        BE_poly = QSVT(BE, np.array([0.,1.,0.,1.]), parity="odd")

        # Prepare initial system state |b>
        def operand_prep():
            qv = QuantumFloat(2)
            prepare(qv, b)
            return qv

        @terminal_sampling
        def main():
            operand = BE_poly.apply_rus(operand_prep)()
            return operand

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
        print(amps)
        # [0.85184732 0.21296187 0.0709873  0.47324855]

    Finally, compare the quantum simulation result with the classical solution:

    ::

        # Compute the SVD
        U, S, Vh = np.linalg.svd(A)

        # Apply polynomial z + z^3 to singular values
        S_poly = S + S ** 3

        # Reconstruct transformed matrix
        A_poly = U @ np.diag(S_poly) @ Vh

        res = A_poly @ b / np.linalg.norm(A_poly @ b)
        print(res)
        # [0.85184734 0.21296184 0.07098728 0.47324852]

    .. warning::

        For non-Hermitian matrices performing Singular Value Transform
        is not the same as applying a matrix polynomial.

    ::

        A_poly = A + A @ A @ A
        res = A_poly @ b / np.linalg.norm(A_poly @ b)
        print(res)
        # [0.71388113 0.02379604 0.21416434 0.66628906]

    """

    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )

    if isinstance(A, (QubitOperator, FermionicOperator)):
        A = BlockEncoding.from_operator(A)

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    # If rescale=False, the returned block-encoding will implement p(A/alpha) instead of p(A),
    # where alpha is the normalization factor of the input block-encoding A. 
    if rescale:
        p = _rescale_poly(A.alpha, p, kind=kind)

    # If the coefficients are given in the standard polynomial basis, convert them to the Chebyshev basis,
    # which is used internally for the angle calculation.
    if kind == "Polynomial":
        p = poly2cheb(p)

    phi, new_alpha = qsvt_angles(p, parity=parity)

    m = len(A._anc_templates)

    def reflection(args, phase):
        qubits = sum([arg.reg for arg in args[1 : m + 1]], [])
        with conjugate(mcx)(qubits, args[0], ctrl_state=0):
            rz(phase, args[0])

    def even(args):
        A.unitary(*args[1:])

    def odd(args):
        with invert():
            A.unitary(*args[1:])

    def new_unitary(*args):
        h(args[0])

        d = len(phi) - 1

        for i in jrange(0, d):
            reflection(args, phase=2 * phi[d - i])
            q_cond(i % 2 == 0, even, odd, args)
        reflection(args, phase=2 * phi[0])

        h(args[0])

    new_anc_templates = [QuantumBool().template()] + A._anc_templates
    return BlockEncoding(
        new_alpha, new_anc_templates, new_unitary, num_ops=A.num_ops, is_hermitian=False
    )
