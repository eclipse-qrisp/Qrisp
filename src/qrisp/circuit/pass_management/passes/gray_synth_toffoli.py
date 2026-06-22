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

from __future__ import annotations

import functools
import numpy as np

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.operation import ControlledOperation
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


# The gray-synthesis Toffoli circuit is built lazily (on first call) rather than
# at module level. Building it at import time triggers QuantumCircuit.append →
# convert_to_qb_list → "from qrisp import QuantumArray", which hits a circular
# import because qrisp.circuit is itself not yet fully initialised when
# qrisp.__init__ imports qrisp.circuit.passes. The lru_cache ensures the circuit
# is constructed only once and reused on every subsequent call.
@functools.lru_cache(maxsize=1)
def _get_gray_toffoli_qc() -> QuantumCircuit:
    """Build and return the cached gray-synthesis Toffoli circuit."""
    qc = QuantumCircuit(3)
    qc.t(0)
    qc.t(1)
    qc.h(2)
    qc.t(2)
    qc.cx(1, 2)
    qc.t_dg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.t_dg(2)
    qc.cx(0, 2)
    qc.h(2)
    qc.rzz(-np.pi / 4, 0, 1)
    return qc


def is_toffoli(op) -> bool:
    """Return True if *op* is a Toffoli (doubly-controlled X) gate.

    Parameters
    ----------
    op : Operation
        The operation to check.

    Returns
    -------
    bool
        ``True`` if *op* is a :class:`~qrisp.circuit.ControlledOperation`
        with two controls whose base operation is an ``x`` gate.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp import is_toffoli
    >>>
    >>> qc = QuantumCircuit(3)
    >>> qc.ccx(0, 1, 2)
    >>>
    >>> is_toffoli(qc.data[0].op)
    True
    """
    return isinstance(op, ControlledOperation) and len(op.controls) == 2 and op.base_operation.name == "x"


@CircuitPass
def gray_synth_toffoli(qc: QuantumCircuit) -> QuantumCircuit:
    """Replace Toffoli gates with a gray-synthesis decomposition.

    Qrisp's default Toffoli implementation is optimised for T-depth, since
    the majority of Qrisp algorithms target fault-tolerant execution where
    T gates dominate cost.  The default decomposition does not have a
    higher CNOT count than the gray-synthesis variant, but it requires
    more SWAP gates when routed onto linear nearest-neighbour connectivity.

    The gray-synthesis decomposition uses 6 CNOT gates and does not
    inherently satisfy linear connectivity either, but the router
    consistently resolves the gap with a single SWAP whose constituent
    CNOTs partially cancel with a Toffoli CNOT — resulting in 7 physical
    CNOTs and a displaced target qubit.

    Parameters
    ----------
    qc : QuantumCircuit
        The input quantum circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with every Toffoli replaced by the gray-synthesis
        decomposition.

    Examples
    --------

    We showcase the distinction in Toffoli decompositions.

    >>> from qrisp import QuantumCircuit, PassManager
    >>> from qrisp import gray_synth_toffoli, decompose
    >>> qc = QuantumCircuit(3)
    >>> qc.ccx(0, 1, 2)
    >>> print(qc)
    <BLANKLINE>
    qb_95: ──■──
             │
    qb_96: ──■──
           ┌─┴─┐
    qb_97: ┤ X ├
           └───┘

    >>> pm_0 = PassManager()
    >>> pm_0 += decompose()
    >>> decomposed_qc = pm_0.run(qc)
    >>> print(decomposed_qc)
           ┌─────┐
    qb_95: ┤ Tdg ├───────■─────────■────■───────────────────────■──
           ├─────┤┌───┐  │  ┌───┐┌─┴─┐  │  ┌─────┐┌───┐ ┌───┐ ┌─┴─┐
    qb_96: ┤ Tdg ├┤ X ├──┼──┤ T ├┤ X ├──┼──┤ Tdg ├┤ X ├─┤ T ├─┤ X ├
           └┬───┬┘└─┬─┘┌─┴─┐├───┤└───┘┌─┴─┐└─────┘└─┬─┘┌┴───┴┐├───┤
    qb_97: ─┤ H ├───■──┤ X ├┤ T ├─────┤ X ├─────────■──┤ Tdg ├┤ H ├
            └───┘      └───┘└───┘     └───┘            └─────┘└───┘

    While this implementation has only a T-depth of 4, the CX gates
    essentially "cycle" through the connectivity requirements. On
    a linear chain connectivity, several swaps would be required.

    >>> pm_1 = PassManager()
    >>> pm_1 += gray_synth_toffoli
    >>> pm_1 += decompose()
    >>> optimized_qc = pm_1.run(qc)
    >>> print(optimized_qc)
           ┌───┐                                                  ┌────────┐
    qb_95: ┤ T ├───────────────────■─────────────────────■────■───┤ gphase ├──■──
           ├───┤                   │                     │  ┌─┴─┐┌┴────────┤┌─┴─┐
    qb_96: ┤ T ├───────■───────────┼─────────■───────────┼──┤ X ├┤ P(-π/4) ├┤ X ├
           ├───┤┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐├───┤└─────────┘└───┘
    qb_97: ┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ H ├────────────────
           └───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘

    This implementation has T-depth 5 but the first 4 CX gates can be implemented
    swap-free on a linear chain connectivity. After this, a single SWAP (that can
    be fused with one of the CX) is suffificient to execute the remaining CX.
    """
    qc_new = qc.clearcopy()

    for instr in qc.data:
        op = instr.op

        if is_toffoli(op):
            # Imported here to avoid a circular import: alg_primitives imports
            # from qrisp.core, which is not yet available when qrisp.circuit.passes
            # is first loaded.
            from qrisp.alg_primitives.mcx_algs import ctrl_state_wrap

            assert isinstance(op, ControlledOperation)
            new_op = op.copy()
            new_op.definition = ctrl_state_wrap(_get_gray_toffoli_qc(), op.ctrl_state)
            qc_new.append(new_op, instr.qubits, instr.clbits)
        else:
            qc_new.append(instr)

    return qc_new
