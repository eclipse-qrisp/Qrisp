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


def apply_poly(
    self, p: "ArrayLike", kind: Literal["Polynomial", "Chebyshev"] = "Polynomial"
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator.

    For a block-encoded **Hermitian** matrix $A$ and a (complex) polynomial $p(z)$, this method returns
    a BlockEncoding of the operator $p(A)$. This is achieved using
    Generalized Quantum Eigenvalue Transformation (GQET).

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : {"Polynomial", "Chebyshev"}
        The basis in which the coefficients are defined.

        - ``"Polynomial"``: $p(x) = \sum c_i x^i$

        - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

        Default is ``"Polynomial"``.

    Returns
    -------
    BlockEncoding
        A new Block-Encoding instance representing the transformed operator $p(A)$.

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

    Generate a block-encoding $A$ of and use :meth:`poly` to find a block-encoding of $p(A)$.

    ::

        from qrisp import QuantumVariable, prepare, terminal_sampling
        from qrisp.block_encodings import BlockEncoding

        BA = BlockEncoding.from_array(A)

        BA_poly = BA.poly(np.array([1.,2.,1.]))

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

        c = (np.eye(4) + 2 * A + A @ A) @ b
        c = c / np.linalg.norm(c)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.02986315 0.57992481 0.62416743 0.52269535]
        # CLASSICAL SOLUTION
        # [-0.02986321  0.57992485  0.6241675   0.52269522]

    """
    from qrisp.algorithms.gqsp import GQET
    
    if isinstance(p, list):
        p = np.array(p)
    return GQET(self, p, kind=kind)
