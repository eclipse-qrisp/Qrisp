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

import numpy as np
import jax.numpy as jnp
from qrisp import QuantumBool, x
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def GQSVT(
    A: BlockEncoding,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    parity: Literal["odd", "even"] = "odd",
    rescale: bool = True,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator via `Generalized Quantum Singular Value Transform <https://arxiv.org/pdf/2312.00723>`_.

    For a block-encoded operator $A$ with Singular Value Decomposition $A = U \Sigma V^{\dagger}$ for unitaries $U, V$, 
    and a (complex) polynomial $p(z)$, this method returns a BlockEncoding of either operator:

    - $p_{odd}(A)=V p_{odd}(\Sigma) U^{\dagger}$ 

    - $p_{even}(A)=V p_{even}(\Sigma) V^{\dagger}$

    where $p=p_{odd}+p_{even}$ is decomposed into odd and even parity parts.

    Parameters
    ----------
    A : BlockEncoding
        The (**not** necessarily Hermitian) operator to be transformed.
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
        A new BlockEncoding instance representing the transformed operator $p(A)$.

    Examples
    --------

    Define a non-Hermitian matrix $A$ and a vector $\vec{b}$. The matris $A$ has singular value decomposition
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

    Generate a BlockEncoding of $A$ and use GQSVT to obtain a BlockEncoding of $p(A)=U p(\Sigma) V^{\dagger}$
    for an odd parity polynomial.
    
    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import GQSVT

        def U0(qv): pass
        def U1(qv): qv-=1
        BE = BlockEncoding.from_lcu(np.array([3,1]), [U0,U1])

        BE_poly = GQSVT(BE, np.array([0.,1.,0.,1.]), parity="odd")

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
        # [0.85184732 0.21296187 0.07098729 0.47324855]

    Finally, compare the quantum simulation result with the classical solution:

    ::

        # Compute the SVD
        U, S, Vh = np.linalg.svd(A)

        # Apply polynomial z + z^3 to singular values
        S_poly = S + S ** 3

        # Reconstruct transformed matrix
        A_poly = (U @ np.diag(S_poly) @ Vh).conj().T

        res = A_poly @ b / np.linalg.norm(A_poly @ b)
        print(res)
        # [0.85184734, 0.21296184, 0.07098728, 0.47324852]

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

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    if rescale:
        p = _rescale_poly(A.alpha, p, kind=kind)
    if kind == "Polynomial":
        p = poly2cheb(p)

    BE_herm = A._hermitianization()

    angles, new_alpha = gqsp_angles(p)

    def new_unitary(*args):
        if parity == "even":
            x(args[1])
        GQSP(args[0], *args[1:], unitary=BE_herm.unitary, angles=angles)
        x(args[1]) # Ensure measuring ancilla in |0> yields correct result

    new_anc_templates = [QuantumBool().template()] + BE_herm._anc_templates
    return BlockEncoding(
        new_alpha,
        new_anc_templates,
        new_unitary,
        num_ops=BE_herm.num_ops,
        is_hermitian=False,
    )
