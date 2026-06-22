"""********************************************************************************
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

from typing import Literal

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.environments import invert


def apply_inv(self, eps: float, kappa: float, method: Literal["QET", "QSVT", "GQSVT"] = "QSVT") -> BlockEncoding:
    r"""Returns a BlockEncoding approximating the matrix inversion of the operator.

    For a block-encoded matrix $A$ with normalization factor $\alpha$, this function returns a BlockEncoding of an
    operator $\tilde{A}^{-1}$ such that $\|\tilde{A}^{-1} - A^{-1}\| \leq \epsilon$.

    The inversion is implemented via

    - Quantum Eigenvalue Transformation (QET) ($A$ must be **Hermitian**)

    - Quantum Singular Value Transform (QSVT)

    - Generalized Quantum Singular Value Transform (GQSVT)

    using a polynomial approximation of $1/x$ over the domain $D_{\kappa} = [-1, -1/\kappa] \cup [1/\kappa, 1]$.

    Parameters
    ----------
    eps : float
        The target precision $\epsilon$.
    kappa : float
        An upper bound for the condition number $\kappa$ of $A$.
        This value defines the "gap" around zero where the function $1/x$ is not approximated.
    method : {"QET", "QSVT", "GQSVT"}
        The method for implementing the inversion.

        - ``"QET"``: Quantum Eigenvalue Transform ($A$ must be Hermitian)

        - ``"QSVT"``: Quantum Singular Value Transform

        - ``"GQSVT"``: Generalized Quantum Singular Value Transform

        Default is ``"QSVT"``.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing an approximation of the inverse $A^{-1}$.

    Notes
    -----
    - **Complexity**: The query complexity of the algorithm scales as :math:`\mathcal{O}(\kappa^2 \log(\kappa/\epsilon))`:
      Guaranteeing successful inversion with high probability requires repeating the procedure :math:`\mathcal{O}(\kappa)` times,
      and each application of the polynomial requires :math:`\mathcal{O}(\kappa \log(\kappa/\epsilon))` (the polynomial degree) queries to the block-encoding of $A$.
    - It is assumed that the eigenvalues of $A/\alpha$ lie within $D_{\kappa}$.

    References
    ----------
    - Childs et. al (2017) `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/pdf/1511.02306>`_.

    Examples
    --------
    Define a QSLP and solve it using :meth:`inv`.

    First, define a Hermitian matrix $A$ and a right-hand side vector $\vec{b}$.

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                    [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                    [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                    [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

        kappa = np.linalg.cond(A)
        print("Condition number of A: ", kappa)
        # Condition number of A:  1.8448536035491883

    Generate a block-encoding of $A$ and use :meth:`inv` to find a block-encoding approximating $A^{-1}$.

    ::

        from qrisp import QuantumVariable, prepare, terminal_sampling
        from qrisp.block_encodings import BlockEncoding

        BA = BlockEncoding.from_array(A)

        BA_inv = BA.inv(0.01, 2)

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumVariable(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():
            operand = BA_inv.apply_rus(prep_b)()
            return operand

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    Finally, compare the quantum simulation result with the classical solution:

    ::

        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.02844496 0.55538449 0.53010186 0.64010231]
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    """
    from qrisp.algorithms.gqsp import inversion

    # The operator is unitary (up to scaling).
    if self.num_ancs == 0:
        if not self.is_hermitian:

            def new_unitary(*args):
                with invert():
                    self.unitary(*args)

        else:
            # The operator is a reflection (up to scaling).
            new_unitary = self.unitary

        return BlockEncoding(
            1.0 / self.alpha,
            self._anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=self.is_hermitian,
        )

    return inversion(self, eps, kappa, method=method)
