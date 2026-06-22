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

import numpy as np
import sympy as sp

from qrisp.circuit.operation import Operation
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def _collect_gphases(qc: QuantumCircuit) -> QuantumCircuit:
    """Remove all ``gphase`` gates from *qc*, accumulate their phases, and
    append a single ``gphase`` on the first qubit of the last instruction
    whose ``num_qubits > 0``.
    """
    accumulated_phase = 0.0
    kept = []
    anchor_qubit = None

    for instr in qc.data:
        if instr.op.name == "gphase":
            accumulated_phase += instr.op.params[0]
        else:
            kept.append(instr)
            if instr.op.num_qubits > 0:
                anchor_qubit = instr.qubits[0]

    # Rebuild circuit from kept instructions
    new_qc = qc.clearcopy()
    for instr in kept:
        new_qc.append(instr.op, instr.qubits, instr.clbits)

    # Emit accumulated phase
    accumulated_phase = accumulated_phase % (2 * np.pi)
    if isinstance(accumulated_phase, sp.Expr) or (abs(accumulated_phase) > 1e-10 and anchor_qubit is not None):
        new_qc.gphase(accumulated_phase, anchor_qubit)

    return new_qc


def decompose(
    level: int | float = np.inf,
    decompose_predicate: Callable[[Operation], bool] | None = None,
    collect_gphases: bool = False,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """
    Create a pass that recursively decomposes synthesized gates.

    Every gate that has a ``.definition`` (a sub-circuit) is dissolved into
    its constituent elementary gates up to the specified recursion *level*.
    Gates whose definitions themselves contain further synthesized gates are
    expanded deeper with each level increment.

    By default *level* is ``np.inf``, so **all** synthesized gates are
    decomposed all the way down to elementary gates in a single pass.

    Parameters
    ----------
    level : int or float, optional
        Maximum recursion depth for decomposition.  ``0`` leaves the circuit
        unchanged.  ``1`` dissolves one layer of synthesized gates.  Higher
        values expand deeper nested definitions.  The default is ``np.inf``,
        which decomposes all synthesized gates down to elementary gates.
    decompose_predicate : Callable[[Operation], bool], optional
        An optional predicate function that receives each :class:`Operation`
        as its only argument and returns ``True`` if the gate should be
        decomposed.  Gates for which the predicate returns ``False`` are left
        intact even if they have a ``.definition`` and are within the
        recursion *level*.  When ``None`` (the default), all synthesized
        gates are decomposed.
    collect_gphases : bool, optional
        If ``True``, all :class:`~qrisp.circuit.GPhaseGate` instructions in
        the decomposed circuit are removed and their phases accumulated into
        a single ``gphase`` gate appended to the first qubit of the last
        non-trivial instruction (one with ``num_qubits > 0``).  This ensures
        the global phase gate does not act on a freshly allocated qubit,
        avoiding interference with qubit-freshness mechanisms.  Default is
        ``False``.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function suitable for :meth:`PassManager.add_pass` or
        :meth:`PassManager.__iadd__`.

    Examples
    --------
    Decompose an MCX gate down to elementary gates::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import decompose
        >>> qc = QuantumCircuit(3)
        >>> qc.mcx([0, 1], 2)
        >>> print(qc)
        qb_71: ──■──
                 │
        qb_72: ──■──
               ┌─┴─┐
        qb_73: ┤ X ├
               └───┘

        >>> pm = PassManager()
        >>> pm += decompose()
        >>> decomposed_qc = pm.run(qc)
        >>> print(decomposed_qc)
               ┌─────┐
        qb_71: ┤ Tdg ├───────■─────────■────■───────────────────────■──
               ├─────┤┌───┐  │  ┌───┐┌─┴─┐  │  ┌─────┐┌───┐ ┌───┐ ┌─┴─┐
        qb_72: ┤ Tdg ├┤ X ├──┼──┤ T ├┤ X ├──┼──┤ Tdg ├┤ X ├─┤ T ├─┤ X ├
               └┬───┬┘└─┬─┘┌─┴─┐├───┤└───┘┌─┴─┐└─────┘└─┬─┘┌┴───┴┐├───┤
        qb_73: ─┤ H ├───■──┤ X ├┤ T ├─────┤ X ├─────────■──┤ Tdg ├┤ H ├
                └───┘      └───┘└───┘     └───┘            └─────┘└───┘

    Decompose only specific gates with a predicate::

        >>> pm2 = PassManager()
        >>> pm2 += decompose(decompose_predicate=lambda op: "cx" in op.name)

    Decompose only one layer::

        >>> pm3 = PassManager()
        >>> pm3 += decompose(level=1)
    """

    @CircuitPass
    def _decompose(qc: QuantumCircuit) -> QuantumCircuit:
        from qrisp.circuit.transpiler import transpile

        result = transpile(
            qc,
            transpilation_level=level,
            transpile_predicate=decompose_predicate,
        )

        if collect_gphases:
            result = _collect_gphases(result)

        return result

    _decompose.__name__ = f"decompose(level={level})"
    _decompose.__doc__ = f"Recursively decompose synthesized gates up to recursion depth {level}."

    return _decompose
