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

from typing import Callable

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.environments import conjugate, invert
from qrisp.jasp import qache
from qrisp.qtypes import QuantumVariable
from qrisp.core import cx, cz


def build_from_foqcs_lcu_prep(
    cls: BlockEncoding,
    prep: Callable[[QuantumVariable], None],
    num_q_ops: int = 1,
    unprep: Callable[[QuantumVariable], None] = None,
    is_hermitian: bool = False,
    norm: "ArrayLike" = 1
) -> BlockEncoding:
    r"""
    Constructs a :class:`BlockEncoding` using the Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) protocol.
    Based on the application of the same name in https://arxiv.org/abs/2507.20887.

    This method implements the Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) structure.
    The provided ``prep`` routine prepares the right PREP (:math:`PREP_{R}`) state on the FOQCS-LCU ancilla register.
    If ``unprep`` is provided, it is interpreted as the corresponding left PREP (:math:`PREP_{L}`) routine and is applied inversely after ``SELECT``.

    Parameters
    ----------
    prep : Callable[[QuantumVariable], None]
        Partial :math:`PREP_{R}` function with all relevant parameters passed except QuantumVariable.
        Parameters can be fixed by using :class:`functools.partial`
        
        num_q_ops : int
        Number of operand qubits, i.e. ``L`` argument for FOQCS-LCU PREP routines.
        The default is 1.

    unprep : Callable[[QuantumVariable], None] = None
        Complex conjugate transpose of :math:`PREP_{R}` with conjugated parameters (see Notes), 
        also a partial function variable with all relevant parameters passed except QuantumVariable.
        The default is None, in which case the unprep is calculated using the prep parameter.

    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian.
        The default is False.
        
    norm : "ArrayLike"
        Normalization factor.
        The default is `1` in case no normalization factor is passed.

    Returns
    -------
    BlockEncoding
    A BlockEncoding using FOQCS LCU.

    Raises
    ------
    ValueError
    When the operator is not representing spin-glass model.
    KeyError
    If method received an unsupported FOQCS-LCU PREP method
            
    Notes
    -----
    - :math:`PREP_{R}` is the PREP subcircuit for FOQCS-LCU, :math:`PREP_{L}^{\dagger}` is the unprep subcircuit and therefore has to undo
      what :math:`PREP_{R}` did to the ancilla qubits. :math:`PREP_{L}` is a version of :math:`PREP_{R}` that is prepared using
      conjugated parameters. :math:`PREP_{L}^{\dagger}` undoes the :math:`PREP_{R}` operation in its entirety.

    Examples
    --------

    ::

        import numpy as np
        from qrisp import QuantumVariable, terminal_sampling
        from qrisp.block_encodings import BlockEncoding, foqcs_prep_heisenberg
        from functools import partial

        def _prep_psi(q_num):
            # Generate state amplitudes.
            psi = np.random.uniform(-1, 1, 2 ** (q_num)) + 1j * np.random.uniform(
                -1, 1, 2 ** (q_num)
            )
            psi /= np.linalg.norm(psi)
            return psi

        # Initialize variables + their values
        L = 4
        g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
        J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

        # Normalize
        norm = np.linalg.norm(np.block([g, J]))
        g /= norm
        J /= norm

        # Calculating the normalization factor
        _g = np.zeros((3,), dtype="complex")
        _J = np.zeros((3,), dtype="complex")

        for i in range(3):
            _g[i] = np.sqrt(g[i] * L)
            _J[i] = np.sqrt(J[i] * (L - 1))

        # Correction for XZ = -iY
        _J[1] = 1j * _J[1]
        _g[1] = (1 - 1j) * _g[1] / np.sqrt(2)

        # Normalization for block encoding
        norm = np.linalg.norm(np.block([_g, _J]))

        # Construct dictionary input expected by foqcs_prep_heisenberg()
        heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
        heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

        # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
        prep = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
        )
        unprep = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
            conjugate=True
        )

        be = BlockEncoding.from_foqcs_lcu_prep(prep=prep, num_q_ops=L, unprep=unprep, norm=norm ** 2)

        psi = _prep_psi(L)

        def operand_prep(psi):
            qv = QuantumVariable(4)
            qv.init_state(psi, method="qswitch")
            return qv

        @terminal_sampling
        def main_apply_rus(BE):
            return BE.apply_rus(operand_prep)(psi)

        # Do the measurement using RUS
        result_rus = main_apply_rus(be)
        print(result_rus)


    """
    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_preps import get_foqcs_lcu_prep_num_of_ancillae
    n_anc = get_foqcs_lcu_prep_num_of_ancillae(prep, num_q_ops)

    # FOQCS-LCU SELECT
    def _select(num_q_ops: int, n_anc: int, ancillae, *operands):
        extra_anc = n_anc - num_q_ops * 2
        cx(ancillae[extra_anc:extra_anc + num_q_ops], operands[0])
        cz(ancillae[extra_anc + num_q_ops:], operands[0])

    if unprep is None:
        @qache
        def unitary(*args):
            # LCU = PREP SELECT PREP^dg
            with conjugate(prep)(args[0]):
                _select(num_q_ops, n_anc, args[0], *args[1:])
    else:
        @qache
        def unitary(*args):
            # LCU = PREP_R SELECT PREP_L^dg (note: PREP(a)^dg != PREP(a*)^dg, where PREP(a) = PREP_R, and PREP(a*) = PREP_L)
            prep(args[0])
            _select(num_q_ops, n_anc, args[0], *args[1:])
            with invert():
                unprep(args[0])

    return cls(
        norm,
        [QuantumVariable(n_anc).template()],
        unitary,
        num_ops=1,
        is_hermitian=is_hermitian,
    )
