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

    The default policy is that each top-level instruction inside the GateStack becomes its own layer.
    Sub-environments are treated as atomic instructions (kept as objects), and compiled when the parent emits that layer.

    This class intentionally does not emit instructions to env_qs in compile().
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

        if self.parent is None:
            raise ValueError(
                "GateStack must have a parent LayeredEnvironment to compile"
            )

        for instr in self.env_data:
            if isinstance(instr, QuantumEnvironment):
                raise ValueError("Nested QuantumEnvironments are not supported")

            self.layers.append([instr])


class LayeredEnvironment(QuantumEnvironment):
    """
    A QuantumEnvironment that interleaves layers from consecutive ``GateStack``
    children to reduce circuit depth.

    Instructions inside each ``GateStack`` are treated as ordered layers.
    When two or more ``GateStack`` objects appear consecutively in the instruction
    stream, their layers are emitted in a "brick" pattern: all layer-0 instructions
    across every stack first, then all layer-1 instructions, and so on.  Because
    operations from different stacks are reordered, **the user is responsible for
    ensuring that instructions across adjacent stacks act on disjoint qubits or
    otherwise commute**.

    Any instruction emitted directly inside ``LayeredEnvironment`` — outside of a
    ``GateStack`` — is passed through immediately at the position it appears,
    breaking adjacency between the stacks on either side of it.  Stacks separated
    by such a bare instruction are therefore interleaved independently, not with
    each other.

    If two stacks have different numbers of layers, the shorter stack simply
    contributes nothing for the extra layers of the longer one. # TODO: there must be a test for this

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
    qubits via CX gates.  We define interleaved data and ancilla qubits::

        from qrisp import *

        n = 3
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits   = qubits[::2]   # even indices: data
        ancilla_qubits = qubits[1::2]  # odd  indices: ancilla

    Without layering, instructions are emitted in source order. That is, all CX gates for
    each stabilizer are grouped together, giving circuit depth :math:`2(n-1)`::

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
    interleaves the CX gates across stabilizers, reducing depth to 2::

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
    position in the output, and it breaks the run of adjacent stacks.  Stacks on
    either side of it are interleaved only among themselves::

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

    def __init__(self, env_args=None):
        super().__init__(env_args=env_args)

    def _emit_instruction(self, instr: Any):
        """Emit a single instruction into env_qs."""

        if isinstance(instr, QuantumEnvironment):
            raise ValueError("Nested QuantumEnvironments are not supported")

        self.env_qs.append(instr)

    def _interleave_layers(self, stacks: List[GateStack]):
        """Interleave layers from multiple GateStacks in brick order."""

        max_layers = max(len(st.layers) for st in stacks)

        for layer_idx in range(max_layers):
            for st in stacks:
                if layer_idx >= len(st.layers):
                    continue
                for instr in st.layers[layer_idx]:
                    self._emit_instruction(instr)

    def compile(self):
        """Compile by interleaving GateStack layers in brick order."""

        # Each segment is either:
        #   ("bare",  [instr, ...])        – plain instructions
        #   ("stacks", [GateStack, ...])   – a consecutive run of GateStacks

        segments: List[tuple] = []

        for instr in self.env_data:
            if isinstance(instr, GateStack):
                if segments and segments[-1][0] == "stacks":
                    segments[-1][1].append(instr)
                else:
                    segments.append(("stacks", [instr]))
            else:
                if segments and segments[-1][0] == "bare":
                    segments[-1][1].append(instr)
                else:
                    segments.append(("bare", [instr]))

        for kind, items in segments:

            if kind == "bare":

                for instr in items:
                    self._emit_instruction(instr)
            else:

                stacks: List[GateStack] = items

                for st in stacks:
                    st.compile()

                self._interleave_layers(stacks)
