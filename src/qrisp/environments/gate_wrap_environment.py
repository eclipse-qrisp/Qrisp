# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines GateWrapEnvironment, which bundles contained operations into a single wrapped gate instruction."""

import jax
from jax.extend.core import Var

from qrisp.circuit import QuantumCircuit, QubitAlloc, QubitDealloc
from qrisp.environments import QuantumEnvironment


class GateWrapEnvironment(QuantumEnvironment):
    """This environment allows to hide complexity in the circuit visualisation.
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

    .. code-block:: none

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

    .. code-block:: none

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

    .. code-block:: none

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

    def __init__(self, name=None, arg_qubits=(), arg_permeability=(), result_permeability=None, is_qfree=None):
        # In Jasp mode, the qubits of the wrapped function's arguments are passed
        # as environment arguments, so that jcompile can identify them in the
        # collected body.
        super().__init__(env_args=list(arg_qubits))
        self.gate_name = name

        self.manual_allocation_management = True

        self.arg_permeability = list(arg_permeability)
        self.result_permeability = result_permeability
        self.is_qfree = is_qfree

    def jcompile(self, eqn, context_dic):
        """Emit the collected body as a ``jit`` equation whose Jaspr carries the gate_wrap specification.

        The collected equation's invars start with the environment arguments
        (the argument qubits), followed by the outer variables that the body
        reads, which line up with the body's invars.
        """
        from qrisp.jasp import AbstractQubit, AbstractQubitArray, extract_invalues, get_last_equation, insert_outvalues

        num_env_args = len(self.env_args)
        args = extract_invalues(eqn, context_dic)[num_env_args:]
        body_jaspr = eqn.params["jaspr"].flatten_environments().copy()

        # A qubit passed in several arguments is only permeable if all of them are.
        qubit_permeability = {}
        for var, permeable in zip(eqn.invars[:num_env_args], self.arg_permeability):
            if isinstance(var, Var) and permeable is not None:
                qubit_permeability[var] = qubit_permeability.get(var, True) and permeable

        for outer_var, var in zip(eqn.invars[num_env_args:-1], body_jaspr.invars[:-1]):
            if isinstance(outer_var, Var) and outer_var in qubit_permeability:
                body_jaspr.permeability[var] = qubit_permeability[outer_var]

        if self.result_permeability is not None:
            for var in body_jaspr.outvars[:-1]:
                if isinstance(getattr(var, "aval", None), (AbstractQubit, AbstractQubitArray)):
                    body_jaspr.permeability[var] = self.result_permeability

        body_jaspr.isqfree = self.is_qfree

        res = jax.jit(body_jaspr.eval)(*args)

        jit_eqn = get_last_equation()
        jit_eqn.params["jaxpr"] = body_jaspr
        jit_eqn.params["name"] = self.gate_name

        if not isinstance(res, tuple):
            res = (res,)

        insert_outvalues(eqn, context_dic, res)

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

        translation_dic = {self.env_qs.qubits[i]: qc.qubits[i] for i in range(len(qc.qubits))}
        translation_dic.update({self.env_qs.clbits[i]: qc.clbits[i] for i in range(len(qc.clbits))})

        qubit_set = set([])

        dealloc_list = []
        alloc_list = []

        for instr in compiled_qc.data:
            qubit_set = qubit_set.union(set([translation_dic[qb] for qb in instr.qubits]))
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

        translation_dic_inv = {translation_dic[key]: key for key in translation_dic.keys()}

        gate = qc.to_gate(self.gate_name)

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
