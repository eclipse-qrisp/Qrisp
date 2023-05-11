"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""


from qrisp import QuantumVariable, x
from qrisp.misc import gate_wrap


@gate_wrap(is_qfree=True)
def increment(qf, amount):
    if amount == 0:
        return

    if not abs(amount) in [qf.decoder(i) for i in range(2**qf.size)]:
        raise Exception("Tried to increment with invalid value")

    # Convert amount to integer reesentation
    amount = amount / 2**qf.exponent

    if amount < 0:
        neg_value = True
        amount = -amount
    else:
        neg_value = False

    # Factorize amount such that amount = amount_tilde 2**power_offset
    # This implies that amount has a binary representation like
    # 10101100000000000
    # where the amount of zeros is the power offset
    # Addition on these significances dont need any further treatment
    power_offset = 0

    while amount % 2 == 0:
        power_offset += 1
        amount = amount / 2

    # Prepare qubit list
    qubit_list = qf.reg[power_offset:]

    if neg_value:
        x(qf.reg)

    amount = int(amount)
    if amount == 1:
        increment_qb_list(qf.qs, qubit_list)
    else:
        increment_arbitrary_constant_qubit_list(qf.qs, qubit_list, amount)

    if neg_value:
        x(qf.reg)


# Incrementation method presented at
# https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html
def increment_qb_list(qs, qubit_list):
    n = len(qubit_list)

    if n == 1:
        qs.x(qubit_list[0])
        return
    if n == 2:
        qs.cx(qubit_list[0], qubit_list[1])
        qs.x(qubit_list[0])
        return

    incr_anc = QuantumVariable(n - 2)
    qs.mcx([qubit_list[0], qubit_list[1]], incr_anc[0], method="gray_pt")

    for i in range(2, n - 1):
        qs.mcx([qubit_list[i], incr_anc[i - 2]], incr_anc[i - 1], method="gray_pt")

    qs.cx(incr_anc[-1], qubit_list[-1])

    for i in range(n - 2, 1, -1):
        qs.mcx([qubit_list[i], incr_anc[i - 2]], incr_anc[i - 1], method="gray_pt_inv")
        qs.cx(incr_anc[i - 2], qubit_list[i])

    qs.mcx([qubit_list[0], qubit_list[1]], incr_anc[0], method="gray_pt_inv")
    qs.cx(qubit_list[0], qubit_list[1])
    qs.x(qubit_list[0])

    incr_anc.delete()


# Incrementation circuit based on the Cuccaro adder
# The idea is to perform the Cuccaro adder but instead of performing
# CNOTs conditioned on the classically known incrementation qubits, we perform NOT gates
# classically conditioned on the incrementation


# For this we define the semi-classical versions of the MAJ and UMA gates depending
# on the (classically known) value of the a variables from the Cuccaro construction
def cMAJ_gate(val):
    from qrisp import QuantumCircuit

    qc = QuantumCircuit(3)

    if val:
        qc.x(0)
        qc.x(1)

    # qc.mcx([0,1], 2, method = "gray_pt")
    qc.mcx([0, 1], 2)

    result = qc.to_gate()

    result.name = "cMAJ"

    return result


def cUMA_gate(val):
    from qrisp import QuantumCircuit

    qc = QuantumCircuit(3)

    # qc.mcx([0,1], 2, method = "gray_pt_inv")
    qc.mcx([0, 1], 2)
    if val:
        qc.x(0)
    qc.cx(0, 1)

    result = qc.to_gate()
    result.name = "cUMA" + str(val)
    return result


# Incrementation by an arbitrary constant using the Cuccaro Adder
# ie. this performs
# qf += x
def increment_arbitrary_constant_qubit_list(qs, qubit_list, c):
    n = len(qubit_list)

    # Initialize the a variable
    from qrisp.core import QuantumVariable

    incr_ancilla = QuantumVariable(n)

    from qrisp.misc import int_as_array

    # Calculate the (classically known) bitstring of c

    a = int_as_array(c, n)[::-1]

    for i in range(n):
        if a[i]:
            x(incr_ancilla[i])

    incr_carry_ancilla = QuantumVariable(1)

    # Perform modular Cuccaro Addition

    qubit_list_1 = incr_ancilla.reg[:-1]
    qubit_list_2 = qubit_list[:-1]

    # Prepare MAJ/UMA gate qubits
    slot_1_qbs = [incr_carry_ancilla.reg[0]] + qubit_list_1[:-1]
    slot_2_qbs = qubit_list_2
    slot_3_qbs = qubit_list_1

    output_qubit = qubit_list[-1]

    iterations = len(slot_1_qbs)

    # Perform 1st step of the modular addition section
    for i in range(iterations):
        qbits = [slot_1_qbs[i], slot_2_qbs[i], slot_3_qbs[i]]
        qs.append(cMAJ_gate(a[i]), qbits)

    # Calculate output qbit
    qs.cx(qbits[2], output_qubit)

    # Perform UMA iterations
    for i in range(iterations - 1, -1, -1):
        qbits = [slot_1_qbs[i], slot_2_qbs[i], slot_3_qbs[i]]
        qs.append(cUMA_gate(a[i]), qbits)

    # Perform the 2nd step for achieving modular arithmetic
    qs.cx(incr_ancilla.reg[-1], output_qubit)

    # Uncompute the ancilla variable
    for i in range(n):
        if a[i]:
            x(incr_ancilla[i])

    # Delete ancilla

    incr_ancilla.delete()
    incr_carry_ancilla.delete()
