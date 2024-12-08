"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""


from qrisp.circuit import QubitAlloc, QubitDealloc
from qrisp.environments.quantum_environments import QuantumEnvironment


# Environment inheritor where the environment content is appended as the inverse
class InversionEnvironment(QuantumEnvironment):
    """
    This QuantumEnvironment can be used to invert (i.e. "dagger") a block of operations.

    An alias for this is ``invert``.

    Examples
    --------

    We increment a :ref:`QuantumFloat` and afterwards revert using the
    InversionEnvironment: ::

        from qrisp import QuantumFloat, invert

        qf = QuantumFloat(4)

        qf += 3

        with invert():
            qf += 3


    >>> print(qf)
    {0: 1.0}
    >>> print(qf.qs)
    
    ::
    
        QuantumCircuit:
        --------------
              ┌───────────┐┌──────────────┐
        qf.0: ┤0          ├┤0             ├
              │           ││              │
        qf.1: ┤1          ├┤1             ├
              │  __iadd__ ││  __iadd___dg │
        qf.2: ┤2          ├┤2             ├
              │           ││              │
        qf.3: ┤3          ├┤3             ├
              └───────────┘└──────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumFloat qf

    In the next example, we create a :ref:`QuantumFloat` and bring it into uniform
    superposition. We calculate the square and set a :ref:`QuantumBool` to ``True``,
    based on if the result is less than 10. Finally, we use the InversionEnvironment
    to uncompute the result of the multiplication. ::

        from qrisp import QuantumBool, h, q_mult, multi_measurement

        qf = QuantumFloat(3)

        h(qf)

        mult_res = q_mult(qf, qf)

        q_bool = QuantumBool()

        with mult_res < 10:
            q_bool.flip()


        with invert():
            q_mult(qf, qf, target = mult_res)


        mult_res.delete(verify = True)

    >>> print(multi_measurement([qf, q_bool]))
    {(0, True): 0.125, (1, True): 0.125, (2, True): 0.125, (3, True): 0.125,
    (4, False): 0.125, (5, False): 0.125, (6, False): 0.125, (7, False): 0.125}

    .. note::
        In many cases, this way of manually uncomputing only works if the uncomputed
        function (in this case ``q_mult``) allows specifying the target variable.
        Using the :meth:`redirect_qfunction <qrisp.redirect_qfunction>` decorator, you
        can turn any quantum function into it's target specifiable version.

    """

    def __enter__(self):
        self.manual_allocation_management = True

        QuantumEnvironment.__enter__(self)

    def compile(self):
        # Save original circuit
        original_circuit = self.env_qs.copy()

        self.env_qs.clear_data()

        # Compile environment
        for instruction in self.env_data:
            # If the instruction is an environment, compile the environment
            if isinstance(instruction, QuantumEnvironment):
                instruction.compile()
                continue

            self.env_qs.append(instruction)

        # We are now faced with the challenge of handling the (De)Allocation gates.
        # Allocation gates are inverted to Deallocation and vice versa.
        # Implying no treatment at all can lead to the situation that a QuantumVariable
        # created inside this environment is unintentionally deallocated after
        # compilation and has no prior allocation.

        # from qrisp import invert, QuantumBool, QuantumFloat, cx

        # qf = QuantumFloat(3)

        # with invert():
        #     qf_res = qf*qf

        # print(qf.qs)

        # QuantumCircuit:
        # ---------------
        #              ┌──────────┐┌──────────────┐
        #        qf.0: ┤ qb_alloc ├┤0             ├──────────────
        #              ├──────────┤│              │
        #        qf.1: ┤ qb_alloc ├┤1             ├──────────────
        #              ├──────────┤│              │
        #        qf.2: ┤ qb_alloc ├┤2             ├──────────────
        #              └──────────┘│              │┌────────────┐
        # mul_res_1.0: ────────────┤3             ├┤ qb_dealloc ├
        #                          │              │├────────────┤
        # mul_res_1.1: ────────────┤4             ├┤ qb_dealloc ├
        #                          │              │├────────────┤
        # mul_res_1.2: ────────────┤5             ├┤ qb_dealloc ├
        #                          │   __mul___dg │├────────────┤
        # mul_res_1.3: ────────────┤6             ├┤ qb_dealloc ├
        #                          │              │├────────────┤
        # mul_res_1.4: ────────────┤7             ├┤ qb_dealloc ├
        #                          │              │├────────────┤
        # mul_res_1.5: ────────────┤8             ├┤ qb_dealloc ├
        #              ┌──────────┐│              │├────────────┤
        # sbp_anc_3.0: ┤ qb_alloc ├┤9             ├┤ qb_dealloc ├
        #              ├──────────┤│              │├────────────┤
        # sbp_anc_4.0: ┤ qb_alloc ├┤10            ├┤ qb_dealloc ├
        #              ├──────────┤│              │├────────────┤
        # sbp_anc_5.0: ┤ qb_alloc ├┤11            ├┤ qb_dealloc ├
        #              └──────────┘└──────────────┘└────────────┘
        # Live QuantumVariables:
        # ----------------------
        # QuantumFloat mul_res_1
        # QuantumFloat qf

        # A naive solution would be to collect all (De)allocation gates and execute them
        # after and before the inverted instructions are appended.

        # This however results in the problem that automatic recomputation ancilla
        # management is no longer available within this environment because a function
        # using ancillae being uncomputed, will recompute it's ancillae on the same
        # qubits because there are no longer any (De)Allocation gates in between.

        # from qrisp import invert, QuantumBool, QuantumFloat, cx

        # qf = QuantumFloat(3)

        # with invert():
        #     qf_res = qf*qf
        #     qb = QuantumBool()
        #     cx(qf_res[0], qb)
        #     qf_res.uncompute()

        # print(qf.qs)

        # QuantumCircuit:
        # ---------------
        #              ┌──────────┐┌─────────────────┐     ┌──────────────┐
        #        qf.0: ┤ qb_alloc ├┤0                ├─────┤0             ├──────────────
        #              ├──────────┤│                 │     │              │
        #        qf.1: ┤ qb_alloc ├┤1                ├─────┤1             ├──────────────
        #              ├──────────┤│                 │     │              │
        #        qf.2: ┤ qb_alloc ├┤2                ├─────┤2             ├──────────────
        #              ├──────────┤│                 │     │              │┌────────────┐
        # mul_res_1.0: ┤ qb_alloc ├┤3                ├──■──┤3             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # mul_res_1.1: ┤ qb_alloc ├┤4                ├──┼──┤4             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # mul_res_1.2: ┤ qb_alloc ├┤5                ├──┼──┤5             ├┤ qb_dealloc ├
        #              ├──────────┤│   __mul___dg_dg │  │  │   __mul___dg │├────────────┤
        # mul_res_1.3: ┤ qb_alloc ├┤6                ├──┼──┤6             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # mul_res_1.4: ┤ qb_alloc ├┤7                ├──┼──┤7             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # mul_res_1.5: ┤ qb_alloc ├┤8                ├──┼──┤8             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # sbp_anc_3.0: ┤ qb_alloc ├┤9                ├──┼──┤9             ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # sbp_anc_4.0: ┤ qb_alloc ├┤10               ├──┼──┤10            ├┤ qb_dealloc ├
        #              ├──────────┤│                 │  │  │              │├────────────┤
        # sbp_anc_5.0: ┤ qb_alloc ├┤11               ├──┼──┤11            ├┤ qb_dealloc ├
        #              ├──────────┤└─────────────────┘┌─┴─┐└──────────────┘└────────────┘
        #        qb.0: ┤ qb_alloc ├───────────────────┤ X ├──────────────────────────────
        #              └──────────┘                   └───┘
        # Live QuantumVariables:
        # ----------------------
        # QuantumBool qb
        # QuantumFloat qf

        # Our solution is therefore to collect all the INITAL and FINAL allocation gates
        # and execute them separately while keeping the inner (De)Allocation gates
        # where they are.

        # from qrisp import invert, QuantumBool, QuantumFloat, cx

        # qf = QuantumFloat(3)

        # with invert():
        #     qf_res = qf*qf
        #     qb = QuantumBool()
        #     cx(qf_res[0], qb)
        #     qf_res.uncompute()

        # print(qf.qs)

        # QuantumCircuit:
        # ---------------
        #              ┌──────────┐┌─────────────────┐                               »
        #        qf.0: ┤ qb_alloc ├┤0                ├───────────────────────────────»
        #              ├──────────┤│                 │                               »
        #        qf.1: ┤ qb_alloc ├┤1                ├───────────────────────────────»
        #              ├──────────┤│                 │                               »
        #        qf.2: ┤ qb_alloc ├┤2                ├───────────────────────────────»
        #              ├──────────┤│                 │                               »
        # mul_res_2.0: ┤ qb_alloc ├┤3                ├────────────────■──────────────»
        #              ├──────────┤│                 │                │              »
        # mul_res_2.1: ┤ qb_alloc ├┤4                ├────────────────┼──────────────»
        #              ├──────────┤│                 │                │              »
        # mul_res_2.2: ┤ qb_alloc ├┤5                ├────────────────┼──────────────»
        #              ├──────────┤│   __mul___dg_dg │                │              »
        # mul_res_2.3: ┤ qb_alloc ├┤6                ├────────────────┼──────────────»
        #              ├──────────┤│                 │                │              »
        # mul_res_2.4: ┤ qb_alloc ├┤7                ├────────────────┼──────────────»
        #              ├──────────┤│                 │                │              »
        # mul_res_2.5: ┤ qb_alloc ├┤8                ├────────────────┼──────────────»
        #              ├──────────┤│                 │┌────────────┐  │  ┌──────────┐»
        # sbp_anc_6.0: ┤ qb_alloc ├┤9                ├┤ qb_dealloc ├──┼──┤ qb_alloc ├»
        #              ├──────────┤│                 │├────────────┤  │  ├──────────┤»
        # sbp_anc_7.0: ┤ qb_alloc ├┤10               ├┤ qb_dealloc ├──┼──┤ qb_alloc ├»
        #              ├──────────┤│                 │├────────────┤  │  ├──────────┤»
        # sbp_anc_8.0: ┤ qb_alloc ├┤11               ├┤ qb_dealloc ├──┼──┤ qb_alloc ├»
        #              ├──────────┤└─────────────────┘└────────────┘┌─┴─┐└──────────┘»
        #        qb.0: ┤ qb_alloc ├─────────────────────────────────┤ X ├────────────»
        #              └──────────┘                                 └───┘            »
        # «             ┌──────────────┐
        # «       qf.0: ┤0             ├──────────────
        # «             │              │
        # «       qf.1: ┤1             ├──────────────
        # «             │              │
        # «       qf.2: ┤2             ├──────────────
        # «             │              │┌────────────┐
        # «mul_res_2.0: ┤3             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «mul_res_2.1: ┤4             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «mul_res_2.2: ┤5             ├┤ qb_dealloc ├
        # «             │   __mul___dg │├────────────┤
        # «mul_res_2.3: ┤6             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «mul_res_2.4: ┤7             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «mul_res_2.5: ┤8             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «sbp_anc_6.0: ┤9             ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «sbp_anc_7.0: ┤10            ├┤ qb_dealloc ├
        # «             │              │├────────────┤
        # «sbp_anc_8.0: ┤11            ├┤ qb_dealloc ├
        # «             └──────────────┘└────────────┘
        # «       qb.0: ──────────────────────────────
        # «
        # Live QuantumVariables:
        # ----------------------
        # QuantumBool qb
        # QuantumFloat qf

        initially_allocated_qubits = []
        i = 0
        while i < len(self.env_qs.data):
            instr = self.env_qs.data[i]
            if (
                instr.op.name == "qb_alloc"
                and not instr.qubits[0] in initially_allocated_qubits
            ):
                initially_allocated_qubits.append(self.env_qs.data.pop(i).qubits[0])
                continue
            i += 1

        deallocated_qubits = []
        i = 0

        self.env_qs.data.reverse()
        while i < len(self.env_qs.data):
            instr = self.env_qs.data[i]
            if (
                instr.op.name == "qb_dealloc"
                and not instr.qubits[0] in deallocated_qubits
            ):
                deallocated_qubits.append(self.env_qs.data.pop(i).qubits[0])
                continue
            i += 1

        deallocated_qubits = list(set(deallocated_qubits))

        self.env_qs.data.reverse()

        # print(transpile(self.env_qs.inverse()))
        # print(transpile(self.env_qs))
        # Merge the original circuit with the inverse of the environment content

        original_circuit.qubits = self.env_qs.qubits
        original_circuit.clbits = self.env_qs.clbits

        for qubit in initially_allocated_qubits:
            original_circuit.append(QubitAlloc(), [qubit])

        original_circuit.extend(self.env_qs.inverse())

        for qubit in deallocated_qubits:
            original_circuit.append(QubitDealloc(), [qubit])

        # Reinstate the resulting circuit in the quantum session circuit
        self.env_qs.data = original_circuit.data
        
    def jcompile(self, eqn, context_dic):
        
        from qrisp.jasp import extract_invalues, insert_outvalues
        args = extract_invalues(eqn, context_dic)
        body_jaspr = eqn.params["jaspr"]
        
        inverted_jaspr = body_jaspr.flatten_environments().inverse()
        
        res = inverted_jaspr.eval(*args)
        insert_outvalues(eqn, context_dic, res)


# Shortcut to quickly initiate inversion environments
def invert():
    return InversionEnvironment()
