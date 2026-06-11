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

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.operators import QubitOperator
from qrisp.block_encodings.constructors.foqcs_lcu.from_foqcs_lcu_prep import build_from_foqcs_lcu_prep


def build_from_foqcs_lcu_operator(
    cls: BlockEncoding,
    O: QubitOperator,
    L: int = -1,
    tol: float = 1e-12
) -> BlockEncoding:
    r"""
    Constructs a :class:`BlockEncoding` from a compatible :class:`QubitOperator` using the
    Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) protocol.
    Based on the application of the same name in https://arxiv.org/abs/2507.20887.

    The operator is analyzed automatically and the corresponding FOQCS-LCU PREP routine is selected.
    Currently supported structures include the specialized one-dimensional nearest-neighbour Heisenberg form
    and the more general spin-glass / same-axis two-body form.

    Parameters
    ----------
    O : QubitOperator
        Operator to encode, e.g.
        ``O = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)``

    L : int = -1
        Number of interacting qubits.
        If not specified, will default to -1, and infer the number of interacting qubits from the operator

    tol : float = 1e-12
        Tolerance for considering the entry to be zero

    Returns
    -------
    BlockEncoding
        A BlockEncoding using the FOQCS-LCU protocol for a compatible QubitOperator,
        with PREP chosen automatically as either the Heisenberg or Spin-glass implementation.

    Raises
    ------
    ValueError
        When the operator is not representing spin-glass model.
    KeyError
        If method received an unsupported FOQCS-LCU PREP method

    Examples
    --------

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X

        H = X(0) + X(1)
        BE = BlockEncoding.from_foqcs_lcu_operator(H, L=2)

        @terminal_sampling
        def main():
            return BE.apply_rus(lambda: QuantumFloat(2))()

        res = main()
        print(res)
        # {1.0: 0.5, 2.0: 0.5}


    """

    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import foqcs_analyze_operator
    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import build_foqcs_lcu_prep_from_analysis
    # Analyze the Qubit operator
    aresult = foqcs_analyze_operator(O, L = L, tol = tol)        
    return build_from_foqcs_lcu_prep(cls, *build_foqcs_lcu_prep_from_analysis(aresult))
