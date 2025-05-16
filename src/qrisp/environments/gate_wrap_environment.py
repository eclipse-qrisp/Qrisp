"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from qrisp.circuit import QuantumCircuit, QubitAlloc, QubitDealloc, XGate
from qrisp.environments import QuantumEnvironment


class GateWrapEnvironment(QuantumEnvironment):
    """
    This environment allows to hide complexity in the circuit visualisation.
    Operations appended inside this environment are bundled into a single
    :ref:`Instruction` object.

    The functionality of this :ref:`QuantumEnvironment` can also be used with the
    :meth:`gate_wrap <qrisp.gate_wrap>` decorator.

    After compiling, the wrapped instruction can be retrieved using the
    ``.instruction`` attribute.

    Parameters
    ----------
    name : string, optional
        The name of the resulting gate. The default is None.

    Examples
    --------

    We create some :ref:`QuantumVariable` and execute some gates inside
    a GateWrapEnvironment: ::

        from qrisp import QuantumVariable, GateWrapEnvironment, x, y, z

        qv = QuantumVariable(3)

        gwe = GateWrapEnvironment("example")


        with gwe:
            x(qv[0])
            y(qv[1])

        z(qv[2])


    >>> print(qv.qs)

    ::

        QuantumCircuit:
        --------------
              ┌──────────┐
        qv.0: ┤0         ├
              │  example │
        qv.1: ┤1         ├
              └──┬───┬───┘
        qv.2: ───┤ Z ├────
                 └───┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    We can access the instruction, which has been appended using the
    ``.instruction`` attribute:

    >>> instruction = gwe.instruction
    >>> print(instruction.op.definition)

    ::

               ┌───┐
        qb_41: ┤ X ├
               ├───┤
        qb_42: ┤ Y ├
               └───┘

    Using the :meth:`gate_wrap <qrisp.gate_wrap>` decorator we can quickly gate wrap
    functions: ::

        from qrisp import gate_wrap

        @gate_wrap
        def example_function(qv):
            x(qv[0])
            y(qv[1])
            z(qv[2])


        example_function(qv)

    >>> print(qv.qs)

    ::

        QuantumCircuit:
        --------------
              ┌──────────┐┌───────────────────┐
        qv.0: ┤0         ├┤0                  ├
              │  example ││                   │
        qv.1: ┤1         ├┤1 example_function ├
              └──┬───┬───┘│                   │
        qv.2: ───┤ Z ├────┤2                  ├
                 └───┘    └───────────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv

    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

        self.manual_allocation_management = True

    def compile(self):
        temp_data_list = list(self.env_qs.data)

        self.env_qs.data = []
        super().compile()

        compiled_qc = self.env_qs.clearcopy()

        compiled_qc.data = list(self.env_qs.data)

        self.env_qs.clear_data()
        self.env_qs.data.extend(temp_data_list)

        if len(compiled_qc.data) == 0:
            self.instruction = None
            return None

        qc = QuantumCircuit(len(self.env_qs.qubits), len(self.env_qs.clbits))

        translation_dic = {
            self.env_qs.qubits[i]: qc.qubits[i] for i in range(len(qc.qubits))
        }
        translation_dic.update(
            {self.env_qs.clbits[i]: qc.clbits[i] for i in range(len(qc.clbits))}
        )

        qubit_set = set([])

        dealloc_list = []
        alloc_list = []

        for instr in compiled_qc.data:
            qubit_set = qubit_set.union(
                set([translation_dic[qb] for qb in instr.qubits])
            )
            if instr.op.name == "qb_dealloc":
                instr.qubits[0].allocated = True
                # dealloc_list.append(instr)
                dealloc_list.append(instr.qubits[0])
                continue
            if instr.op.name == "qb_alloc":
                alloc_list.append(instr.qubits[0])
                try:
                    dealloc_list.remove(instr.qubits[0])
                except ValueError:
                    pass
                continue

            qc.append(
                instr.op,
                [translation_dic[qb] for qb in instr.qubits],
                [translation_dic[cb] for cb in instr.clbits],
            )

        idle_qubit_list = list(set(qc.qubits) - qubit_set)

        for j in range(len(idle_qubit_list)):
            for i in range(len(qc.qubits)):
                if qc.qubits[i].identifier == idle_qubit_list[j].identifier:
                    qc.qubits.pop(i)
                    break

        translation_dic_inv = {
            translation_dic[key]: key for key in translation_dic.keys()
        }

        gate = qc.to_gate(self.name)

        alloc_list = list(set(alloc_list))
        for qb in alloc_list:
            self.env_qs.append(QubitAlloc(), [qb])

        self.env_qs.append(
            gate,
            [translation_dic_inv[qb] for qb in qc.qubits],
            [translation_dic_inv[cb] for cb in qc.clbits],
        )
        self.instruction = self.env_qs.data[-1]

        dealloc_list = list(set(dealloc_list))
        for qb in dealloc_list:
            self.env_qs.append(QubitDealloc(), [qb])
            qb.allocated = False
