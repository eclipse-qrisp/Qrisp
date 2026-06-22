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

from collections.abc import Callable

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def convert_to_cx(
    strict: bool = False,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """
    Create a pass that converts two-qubit gates to CX-based decompositions.

    This pass converts CZ, CY, and SWAP gates to their CX-based equivalents
    using single-qubit gate decompositions. CX gates are native to many
    quantum computing platforms.

    Two-qubit gates that are already CX or barrier instructions are always left
    unchanged. Any other two-qubit gate that has no known CX decomposition is
    either passed through silently (``strict=False``, the default) or causes the
    pass to raise an exception (``strict=True``).

    .. note::

        This pass does **not** decompose composite (wrapped) gates. Use the
        :func:`decompose` pass to expand composite gates into elementary
        operations before applying this pass.

    Parameters
    ----------
    strict : bool, optional
        When ``True``, an :class:`Exception` is raised if a two-qubit gate with
        no known CX decomposition is encountered. When ``False`` (the default),
        such gates are left unchanged in the output circuit.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function suitable for :meth:`PassManager.add_pass`.

    Decompositions
    --------------
    - CZ(control, target)  -> H(target), CX(control, target), H(target)
    - CY(control, target)  -> S†(target), CX(control, target), S(target)
    - SWAP(a, b)           -> three CX gates


    Examples
    --------
    Convert a CZ gate to H—CX—H::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import convert_to_cx
        >>> qc = QuantumCircuit(2)
        >>> qc.cz(0, 1)
        >>> print(qc)
        <BLANKLINE>
        qb_69: ──■──
                 │
        qb_70: ──■──
        >>> pm = PassManager()
        >>> pm += convert_to_cx()
        >>> optimized_qc = pm.run(qc)
        >>> print(optimized_qc)
        qb_69: ──────■──────
               ┌───┐ │ ┌───┐
        qb_70: ┤ H ├─■─┤ H ├
               └───┘   └───┘

    """

    @CircuitPass
    def _convert_to_cx(qc: QuantumCircuit) -> QuantumCircuit:
        qc_new = qc.clearcopy()

        for instr in qc.data:
            op = instr.op

            if op.num_qubits == 2 and op.name not in ("cx", "barrier"):
                if op.name == "cz":
                    qc_new.h(instr.qubits[1])
                    qc_new.cx(instr.qubits[0], instr.qubits[1])
                    qc_new.h(instr.qubits[1])
                elif op.name == "cy":
                    qc_new.s_dg(instr.qubits[1])
                    qc_new.cx(instr.qubits[0], instr.qubits[1])
                    qc_new.s(instr.qubits[1])
                elif op.name == "swap":
                    qc_new.cx(instr.qubits[0], instr.qubits[1])
                    qc_new.cx(instr.qubits[1], instr.qubits[0])
                    qc_new.cx(instr.qubits[0], instr.qubits[1])
                elif strict:
                    raise ValueError(f"Don't know how to convert two-qubit gate {op.name!r} to CX")
                else:
                    qc_new.append(instr)
            else:
                qc_new.append(instr)

        return qc_new

    return _convert_to_cx
