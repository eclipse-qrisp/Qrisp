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


class Instruction:
    """
    This class combines Operation objects with their operands (ie. qubits and classical
    bits). The data attribut of the QuantumCircuit class consists of a list of
    Instructions.

    Instructions can be added to QuantumCircuits using the
    `append <qrisp.QuantumCircuit.append>` method without any
    qubit/ classical bit arguments.

    Parameters
    ----------
    op : Operation
        The Operation object.
    qubits : list[Qubit], optional
        The list of Qubits on which op should take place. The default is [].
    clbits : list[Clbit], optional
        The list of Clbits on which op should take place. The default is [].


    Examples
    --------

    We create two Instruction objects, merge them and append the result to a
    QuantumCircuit.

    >>> from qrisp import Instruction, QuantumCircuit, CXGate, XGate
    >>> qc = QuantumCircuit(2)
    >>> ins_0 = Instruction(XGate(), [qc.qubits[0]])
    >>> ins_1 = Instruction(CXGate(), qc.qubits)
    >>> merged_ins = ins_0.merge(ins_1)
    >>> qc.append(merged_ins)
    >>> qc.measure(qc.qubits[1])
    >>> qc.run()
    {'1': 10000}
    >>> print(qc.transpile())
          ┌───┐
    qb_0: ┤ X ├──■─────
          └───┘┌─┴─┐┌─┐
    qb_1: ─────┤ X ├┤M├
               └───┘└╥┘
    cb_0: ═══════════╩═

    """

    def __init__(self, op, qubits=[], clbits=[]):
        self.op = op
        self.qubits = qubits
        self.clbits = clbits

    def merge(self, other):
        """
        Merges two instructions into one.

        Parameters
        ----------
        other : Instruction
            The second instruction.

        Returns
        -------
        res : Instruction
            The merged instruction (self is executed first).

        """

        from qrisp.circuit import QuantumCircuit

        qubit_list = list(set(self.qubits + other.qubits))
        clbit_list = list(set(self.clbits + other.clbits))

        qubit_list.sort(key=lambda x: x.identifier)
        clbit_list.sort(key=lambda x: x.identifier)
        qc = QuantumCircuit()

        for i in range(len(qubit_list)):
            qc.add_qubit(qubit_list[i])

        for i in range(len(clbit_list)):
            qc.add_clbit(clbit_list[i])

        qc.data = [self, other]

        res = Instruction(qc.to_op(), qubit_list, clbit_list)
        return res

    def copy(self):
        """
        Returns a copy of the Instruction.

        Returns
        -------
        Instruction
            The copied Instruction.

        """
        return Instruction(self.op.copy(), list(self.qubits), list(self.clbits))

    def inverse(self):
        res = self.copy()
        res.op = res.op.inverse()

        return res

    def __str__(self):
        if len(self.clbits):
            return (
                self.op.name
                + "("
                + str(self.qubits)[1:-1]
                + ", "
                + str(self.clbits)[1:-1]
                + ")"
            )
        else:
            return self.op.name + "(" + str(self.qubits)[1:-1] + ")"

    def __repr__(self):
        return self.__str__()
