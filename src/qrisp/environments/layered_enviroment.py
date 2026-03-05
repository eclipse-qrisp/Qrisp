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

from typing import Any, List

from qrisp.environments.quantum_environments import QuantumEnvironment


class GateStack(QuantumEnvironment):
    """
    Collect operations into layers for a parent LayeredEnvironment.

    Default policy:

      - each top-level instruction inside the GateStack becomes its own layer

      - sub-environments are treated as atomic instructions (kept as objects),
        and compiled when the parent emits that layer.

    Note:

      This class intentionally does NOT emit instructions to env_qs in compile().
      The parent LayeredEnvironment is responsible for emitting.

    """

    def __init__(self, env_args=None):
        """Initialize with empty layers."""

        super().__init__(env_args=env_args)

        self.layers: List[List[Any]] = []

    def compile(self):
        """
        Prepare layers, but do not emit anything.

        The parent LayeredEnvironment will interleave and emit these layers.
        """

        for instr in self.env_data:
            self.layers.append([instr])


class LayeredEnvironment(QuantumEnvironment):
    """
    A QuantumEnvironment that interleaves layers from multiple GateStack children.

    This environment collects GateStack instances and emits their operations in a
    "brick" pattern: all layer-0 instructions from all stacks, then all layer-1
    instructions, and so on. This enables better parallelization of operations
    across different stacks.

    GateStack instances are compiled and their layers are emitted in brick order.
    Non-GateStack instructions are emitted immediately in order.
    Operations are reordered across stacks, so they must either act on
    disjoint qubits or commute to maintain semantic correctness.

    Examples
    --------

    Let's suppose we need to measure a Z-type stabilizer by entangling all source qubits
    with a target ancilla qubit using CX gates.

    We first define a set of data qubits and ancilla qubits using a ``QuantumArray``:

    ::

        from qrisp import *

        n = 3
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits = qubits[::2]
        ancilla_qubits = qubits[1::2]


    Here, we have 3 data qubits and 2 ancilla qubits interleaved in the `qubits` array.
    Ancilla qubits are at odd indices and data qubits are at even indices.

    First, let's see what happens if we call the stabilizer measurement function in a normal environment without layering:

    ::

        for i in range(n - 1):
            cx(data_qubits[i], ancilla_qubits[i])
            cx(data_qubits[i + 1], ancilla_qubits[i])

    We can see that the instructions are emitted in the original order,
    with all CXs for each stabilizer measurement grouped together:

    >>> print(qubits.qs)
        QuantumCircuit:
        ---------------
          qubits.0: ──■─────────────────
                    ┌─┴─┐┌───┐
        qubits_1.0: ┤ X ├┤ X ├──────────
                    └───┘└─┬─┘
        qubits_2.0: ───────■────■───────
                              ┌─┴─┐┌───┐
        qubits_3.0: ──────────┤ X ├┤ X ├
                              └───┘└─┬─┘
        qubits_4.0: ─────────────────■──
        ...


    Now, let's see what happens if we wrap the same code in a LayeredEnvironment and GateStack:

    ::

        qubits.qs.clear_data()

        with LayeredEnvironment():
            for i in range(n - 1):
                with GateStack():
                    cx(data_qubits[i], ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])



    We can see that the instructions are now interleaved layer-by-layer across the different GateStacks:

    >>> print(qubits.qs)
    QuantumCircuit:
    ---------------
      qubits.0: ──■───────
                ┌─┴─┐┌───┐
    qubits_1.0: ┤ X ├┤ X ├
                └───┘└─┬─┘
    qubits_2.0: ──■────■──
                ┌─┴─┐┌───┐
    qubits_3.0: ┤ X ├┤ X ├
                └───┘└─┬─┘
    qubits_4.0: ───────■──
    ...

    """

    def __init__(self, env_args=None):
        super().__init__(env_args=env_args)

    def _emit_instruction(self, instr: Any):
        """
        Emit a single instruction or compile a nested environment into this env_qs.
        """
        if isinstance(instr, QuantumEnvironment):
            instr.compile()
        else:
            self.env_qs.append(instr)

    def compile(self):
        """
        Compile by interleaving GateStack layers.

        Non-GateStack instructions are emitted immediately.
        GateStack instructions are compiled and their layers emitted in brick order.
        """
        stacks: List[GateStack] = []

        for instr in self.env_data:
            if isinstance(instr, GateStack):
                stacks.append(instr)
            else:
                self._emit_instruction(instr)

        if not stacks:
            return

        for st in stacks:
            st.compile()

        max_layers = max((len(st.layers) for st in stacks), default=0)

        for layer_idx in range(max_layers):
            for st in stacks:
                if layer_idx >= len(st.layers):
                    continue
                for instr in st.layers[layer_idx]:
                    self._emit_instruction(instr)
