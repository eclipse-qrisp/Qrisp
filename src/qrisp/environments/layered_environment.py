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

from typing import List, Tuple

from qrisp.circuit.instruction import Instruction
from qrisp.environments.quantum_environments import QuantumEnvironment


class GateStack(QuantumEnvironment):
    """
    Collect operations into layers for a parent :class:`LayeredEnvironment` to interleave.

    This class intentionally does not emit instructions to the parent environment when they are added.
    Instead, it collects them into layers, which are
    emitted by the parent :class:`LayeredEnvironment` after interleaving.

    For more details and examples, see :class:`LayeredEnvironment`.

    """

    def __init__(self, env_args=None):
        """Initialize with empty layers."""

        super().__init__(env_args=env_args)

        self.layers: List[Instruction] = []

    def compile(self):
        """
        Prepare layers, but do not emit anything.

        The parent LayeredEnvironment will interleave and emit these layers.
        """

        if self.parent is None:
            raise ValueError(
                "GateStack must have a parent LayeredEnvironment to compile"
            )

        for instr in self.env_data:
            if isinstance(instr, QuantumEnvironment):
                raise ValueError("Nested QuantumEnvironments are not supported")

            self.layers.append(instr)


class LayeredEnvironment(QuantumEnvironment):
    """
    A QuantumEnvironment that interleaves layers from consecutive ``GateStack``
    children to reduce circuit depth.

    Instructions inside each ``GateStack`` are treated as ordered layers.
    When two or more ``GateStack`` objects appear consecutively in the instruction
    stream, their layers are emitted in a "brick" pattern: all layer-0 instructions
    across every stack first, then all layer-1 instructions, and so on.  Because
    operations from different stacks are reordered, the user is responsible for
    ensuring that instructions across adjacent stacks act on disjoint qubits or
    otherwise commute.

    Any instruction emitted directly inside ``LayeredEnvironment`` and outside of a
    ``GateStack`` is passed through immediately at the position it appears,
    breaking adjacency between the stacks on either side of it. Stacks separated
    by such a bare instruction are therefore interleaved independently, not with
    each other.

    If two stacks have different numbers of layers, the shorter stack simply
    contributes nothing for the extra layers of the longer one.

    .. note::

        This environment performs no commutation or qubit-conflict checks.
        Incorrectly interleaving non-commuting operations on shared qubits will
        produce a wrong circuit.

    Parameters
    ----------
    env_args : optional
        Forwarded to the parent :class:`QuantumEnvironment` constructor.

    Examples
    --------

    **Basic usage: parallel stabilizer measurement**

    Consider measuring a Z-type stabilizer by entangling data qubits with ancilla
    qubits via CX gates.  We define interleaved data and ancilla qubits

    ::

        from qrisp import *

        n = 3
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits   = qubits[::2]   # even indices: data
        ancilla_qubits = qubits[1::2]  # odd  indices: ancilla

    Without layering, instructions are emitted in source order. That is, all CX gates for
    each stabilizer are grouped together, giving circuit depth :math:`2(n-1)`:

    ::

        for i in range(n - 1):
            cx(data_qubits[i],     ancilla_qubits[i])
            cx(data_qubits[i + 1], ancilla_qubits[i])

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

    >>> print(qubits.qs.depth())
    4

    Wrapping each stabilizer in a ``GateStack`` inside a ``LayeredEnvironment``
    interleaves the CX gates across stabilizers, reducing depth to 2:

    ::

        qubits.qs.clear_data()

        with LayeredEnvironment():
            for i in range(n - 1):
                with GateStack():
                    cx(data_qubits[i],     ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

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

    >>> print(qubits.qs.depth())
    2

    **Bare instructions break adjacency**

    A bare instruction between two ``GateStack`` objects is emitted at that exact
    position in the output, and it breaks the run of adjacent stacks. Stacks on
    either side of it are not interleaved with each other:

    ::

        N = 4
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * N - 1))
        data_qubits    = qubits[::2]
        ancilla_qubits = qubits[1::2]

        with LayeredEnvironment():
            with GateStack():
                cx(data_qubits[0], ancilla_qubits[0])
                cx(data_qubits[1], ancilla_qubits[0])
            with GateStack():
                cx(data_qubits[1], ancilla_qubits[1])
                cx(data_qubits[2], ancilla_qubits[1])

            z(data_qubits[2])

            with GateStack():
                cx(data_qubits[2], ancilla_qubits[2])
                cx(data_qubits[3], ancilla_qubits[2])

    The first two stacks are interleaved with each other. Then, the Z gate is emitted
    immediately after. Finally, the last stack is emitted sequentially on its own:

    >>> print(qubits.qs)
    QuantumCircuit:
    ---------------
      qubits.0: ──■──────────────────────
                ┌─┴─┐┌───┐
    qubits_1.0: ┤ X ├┤ X ├───────────────
                └───┘└─┬─┘
    qubits_2.0: ──■────■─────────────────
                ┌─┴─┐┌───┐
    qubits_3.0: ┤ X ├┤ X ├───────────────
                └───┘└─┬─┘┌───┐
    qubits_4.0: ───────■──┤ Z ├──■───────
                          └───┘┌─┴─┐┌───┐
    qubits_5.0: ───────────────┤ X ├┤ X ├
                               └───┘└─┬─┘
    qubits_6.0: ──────────────────────■──
    ...

    """

    def __init__(self, env_args=None) -> None:
        super().__init__(env_args=env_args)

    def _emit_instruction(self, instr: Instruction) -> None:
        """Emit a single instruction into env_qs."""

        if isinstance(instr, QuantumEnvironment):
            raise ValueError("Nested QuantumEnvironments are not supported")

        self.env_qs.append(instr)

    # We interleave by grouping layers by their index across all stacks.
    #
    # For example, if we have:
    #
    # Stack A: layers [A1, A2, A3]
    # Stack B: layers [B1]
    # Stack C: layers [C1, C2]
    #
    # The interleaved order would be: A1, B1, C1, A2, C2, A3
    def _interleave_layers(self, stacks: List[GateStack]) -> None:
        """Interleave layers from multiple GateStacks in brick order."""

        if not stacks:
            return

        max_layers = max((len(st.layers) for st in stacks), default=0)

        for layer_idx in range(max_layers):
            for st in stacks:
                if layer_idx < len(st.layers):
                    instr = st.layers[layer_idx]
                    self._emit_instruction(instr)

    # Each segment is a tuple of the form (segment_type, items), where segment_type is either "bare" or "stacks":
    #
    # - "bare": items is a list of instructions that are not GateStacks.
    # - "stacks": items is a list of consecutive GateStack objects.
    #
    # For example, if self.env_data contains the following sequence of instructions:
    #
    # [instr1, instr2, GateStack1, GateStack2, instr3, GateStack3]
    #
    # The resulting segments would be:
    #
    # [
    #     ("bare", [instr1, instr2]),
    #     ("stacks", [GateStack1, GateStack2]),
    #     ("bare", [instr3]),
    #     ("stacks", [GateStack3])
    # ]
    def _prepare_segment(self) -> List[Tuple[str, List]]:
        """
        Prepare a list of segments from self.env_data, grouping consecutive GateStacks together.
        """

        segments: List[Tuple[str, List]] = []

        for instr in self.env_data:
            is_gatestack = isinstance(instr, GateStack)
            segment_type = "stacks" if is_gatestack else "bare"

            if segments and segments[-1][0] == segment_type:
                segments[-1][1].append(instr)
            else:
                segments.append((segment_type, [instr]))

        return segments

    def compile(self) -> None:
        """Compile by interleaving GateStack layers in brick order."""

        segments = self._prepare_segment()

        for segment_type, items in segments:
            if segment_type == "bare":
                for instr in items:
                    self._emit_instruction(instr)
            else:
                for stack in items:
                    stack.compile()
                self._interleave_layers(items)
