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

from qrisp.circuit.operation import Operation
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def decompose(
    level: int | float = np.inf,
    decompose_predicate: Callable[[Operation], bool] | None = None,
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

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function suitable for :meth:`PassManager.add_pass` or
        :meth:`PassManager.__iadd__`.

    Example
    -------
    Decompose all synthesized gates down to elementary gates::

        >>> from qrisp import PassManager, decompose
        >>> pm = PassManager()
        >>> pm += decompose()
        >>> transpiled_qc = pm.run(qc)

    Decompose only ``mcx`` gates::

        >>> pm += decompose(decompose_predicate=lambda op: op.name == "mcx")
    """

    @CircuitPass
    def _decompose(qc: QuantumCircuit) -> QuantumCircuit:
        from qrisp.circuit.transpiler import transpile

        return transpile(
            qc,
            transpilation_level=level,
            transpile_predicate=decompose_predicate,
        )

    _decompose.__name__ = f"decompose(level={level})"
    _decompose.__doc__ = (
        f"Recursively decompose synthesized gates up to recursion depth {level}."
    )

    return _decompose
