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
import numpy as np

from qrisp.block_encodings.block_encoding_base import BlockEncoding

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def apply_svt(
    self,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    parity: Literal["even", "odd"] = "odd",
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a singular value transformation (SVT) of the operator.

    For a block-encoded operator $A$ with `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_ $A = U \Sigma V^{\dagger}$ for unitaries $U, V$,
    and a (real) polynomial $p(x)$, this method returns a BlockEncoding of either operator:

    - $p_{odd}(A)=U p_{odd}(\Sigma) V^{\dagger}$

    - $p_{even}(A)=V p_{even}(\Sigma) V^{\dagger}$

    where $p=p_{odd}+p_{even}$ is decomposed into odd and even parity parts.
    This is achieved using the `Quantum Singular Value Transform (QSVT) <https://arxiv.org/abs/1806.01838>`_.

    Parameters
    ----------
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

    Generate a BlockEncoding of $A$ and use :meth:`svt` to obtain a BlockEncoding of $p(A)=U p(\Sigma) V^{\dagger}$
    for an odd parity polynomial.

    ::

        from qrisp import QuantumFloat, prepare, terminal_sampling
        from qrisp.block_encodings import BlockEncoding

        def U0(qv): pass
        def U1(qv): qv-=1
        BE = BlockEncoding.from_lcu(np.array([3,1]), [U0,U1])

        BE_poly = BE.svt(np.array([0.,1.,0.,1.]), parity="odd")

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
    from qrisp.algorithms.gqsp import QSVT

    if isinstance(p, list):
        p = np.array(p)
    return QSVT(self, p, kind=kind, parity=parity)
