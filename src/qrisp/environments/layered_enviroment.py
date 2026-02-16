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

from qrisp import QuantumEnvironment
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
        super().__init__(env_args=env_args)
        self.layers: List[List[Any]] = []

    def _build_layers(self) -> List[List[Any]]:
        """
        Build layered representation from self.env_data.

        Default: one instruction per layer.
        """
        layers: List[List[Any]] = []

        for instr in self.env_data:
            layers.append([instr])

        return layers

    def compile(self):
        """
        Prepare layers, but do not emit anything.

        The parent LayeredEnvironment will interleave and emit these layers.
        """
        self.layers = self._build_layers()


class LayeredEnvironment(QuantumEnvironment):
    """
    Interleave multiple GateStacks layer-by-layer.

    Behavior:
      - Collect all GateStack children that appear directly in env_data.
      - Compile each GateStack (which builds `stack.layers` but emits nothing).
      - Emit: layer 0 from all stacks, then layer 1 from all stacks, etc.

    Non-GateStack instructions:
      - Default policy below: emit them immediately in the original order,
        but when a GateStack is encountered we buffer it for the final brick emission.

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
