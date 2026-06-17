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
    tol: float = 1e-12
) -> BlockEncoding:
    r"""
    Constructs a :class:`BlockEncoding` from a compatible :class:`QubitOperator` using the
    Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) algorithm
    specified in https://arxiv.org/abs/2507.20887.

    The input operator is analyzed automatically. If it matches the more
    specialized one-dimensional nearest-neighbour Heisenberg form, the
    Heisenberg PREP routine is used. Otherwise, if it matches the more general
    same-axis two-body spin-glass form, the spin-glass PREP routine is used.

    This method uses more general :meth:`from_foqcs_lcu_prep` internally,
    abstracting the partial ``PREP`` methods construction. If the operator does not
    match supported operators specified in `Notes`, it will require its own custom PREP subroutine,
    which is covered in more detail in :meth:`from_foqcs_lcu_prep` documentation.

    Parameters
    ----------
    O : QubitOperator
        Operator to encode, supported operators are covered in `Notes`, e.g.
        ``O = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)``

    tol : float, optional = 1e-12
        The tolerance used to determine if an entry is zero. 
        Defaults to 1e-12.

    Returns
    -------
    BlockEncoding
        A BlockEncoding using the FOQCS-LCU protocol for a compatible QubitOperator,
        with PREP chosen automatically as either the Heisenberg or spin-glass implementation.

    Raises
    ------
    ValueError
        When the operator is not representing spin-glass model.
    KeyError
        If method received an unsupported FOQCS-LCU PREP method

    Notes
    -----
    Let :math:`L` be the number of operand qubits and let
    :math:`P_i` denote the Pauli operator :math:`P` acting on qubit :math:`i`, where
    :math:`P \in \{X, Y, Z\}`.

    The general supported form is the same-axis one- and two-body spin-glass
    Hamiltonian

    .. math::

        H =
        \sum_{P \in \{X,Y,Z\}}
        \left(
            \sum_{i=0}^{L-1} g_i^P P_i
            +
            \sum_{0 \leq i < j < L} J_{ij}^P P_i P_j
        \right).

    Equivalently, every non-zero Pauli term must be one of

    .. math::

        X_i,\; Y_i,\; Z_i,\;
        X_i X_j,\; Y_i Y_j,\; Z_i Z_j.

    Thus, arbitrary one-body fields are allowed, and arbitrary two-body
    couplings are allowed as long as both Paulis in a two-body term use the
    same axis.

    The following terms are not supported:

    * constant / identity terms, e.g. ``c * I``;
    * mixed-axis two-body terms, e.g. ``X(i) * Z(j)``,
      ``X(i) * Y(j)``, or ``Y(i) * Z(j)``;
    * three- or higher-body terms, e.g. ``X(i) * X(j) * X(k)``.

    A specialized Heisenberg PREP routine is selected when the operator has the
    one-dimensional nearest-neighbour form

    .. math::

        H =
        \sum_{i=0}^{L-1}
        \left(
            g^X X_i + g^Y Y_i + g^Z Z_i
        \right)
        +
        \sum_{i=0}^{L-2}
        \left(
            J^X X_i X_{i+1}
            + J^Y Y_i Y_{i+1}
            + J^Z Z_i Z_{i+1}
        \right).

    Here the local field coefficients must be uniform in :math:`i` for each Pauli
    axis, and the nearest-neighbour coupling coefficients must also be uniform
    in :math:`i` for each Pauli axis. The coefficients may still differ between
    axes, e.g. :math:`J^X`, :math:`J^Y`, and :math:`J^Z` need not be equal.

    Examples of supported spin-glass terms include

    ``X(0) + 0.5 * Y(2) + 0.2 * Z(0) * Z(3)``

    and

    ``X(0) * X(2) + Y(1) * Y(4) + Z(0) * Z(1)``.

    Examples of unsupported terms include

    * ``2.0``: identity / constant term;
    * ``X(0) * Z(1)``: mixed-axis coupling;
    * ``X(0) * X(1) * X(2)``: three-body interaction.

    Examples
    --------

    A minimal one-body example:

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X

        H = X(0) + X(1)
        BE = BlockEncoding.from_foqcs_lcu_operator(H)

        @terminal_sampling
        def main():
            return BE.apply_rus(lambda: QuantumFloat(2))()

        res = main()
        print(res)
        # {1.0: 0.5, 2.0: 0.5}

    A same-axis two-body spin-glass example:

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H = (
            0.7 * X(0)
            - 0.3 * Z(2)
            + 0.5 * X(0) * X(1)
            + 1.2 * Y(1) * Y(3)
            - 0.8 * Z(0) * Z(3)
        )

        BE = BlockEncoding.from_foqcs_lcu_operator(H)

        @terminal_sampling
        def main():
            return BE.apply_rus(lambda: QuantumFloat(4))()

        res = main()
        print(res)

    """

    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import foqcs_analyze_operator
    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import build_foqcs_lcu_prep_from_analysis
    # Analyze the Qubit operator
    aresult = foqcs_analyze_operator(O, tol = tol)        
    return build_from_foqcs_lcu_prep(cls, *build_foqcs_lcu_prep_from_analysis(aresult))
