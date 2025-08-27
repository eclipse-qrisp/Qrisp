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

from qrisp import QuantumCircuit
from qrisp.alg_primitives.arithmetic.adders.adder_tools import ammend_inpl_adder

# Implementation of the modular Cuccaro-Adder https://arxiv.org/pdf/quant-ph/0410184.pdf


def MAJ_gate():
    qc = QuantumCircuit(3)

    qc.cx(2, 1)
    qc.cx(2, 0)
    qc.mcx([0, 1], 2)

    result = qc.to_gate()

    result.name = "MAJ"

    return result


def UMA_gate(mode=2):
    qc = QuantumCircuit(3)
    if mode == 2:
        qc.mcx([0, 1], 2)
        qc.cx(2, 0)
        qc.cx(0, 1)

    if mode == 3:
        qc.x(1)
        qc.cx(0, 1)
        qc.mcx([0, 1], 2)
        qc.x(1)
        qc.cx(2, 0)
        qc.cx(2, 1)

    result = qc.to_gate()
    result.name = "UMA"
    return result


# Performs in-place addition on qv2, i.e.
# qv1 += qv2


# TO-DO make this less circuity and more qrispy
def cuccaro_procedure(qs, qubit_list_1, qubit_list_2, output_qubit=None, carry_in=None):
    if len(qubit_list_1) != len(qubit_list_2):
        print(len(qubit_list_1))
        print(len(qubit_list_2))
        raise Exception(
            "Tried to call Cuccaro-procedure with qubit lists of unequal length"
        )

    from qrisp.core import QuantumVariable

    if carry_in is None:
        # Request ancilla Qubit
        ancilla = QuantumVariable(1)
    else:
        ancilla = [carry_in]

    # Prepare MAJ/UMA gate qubits
    slot_1_qbs = list(ancilla) + qubit_list_1[:-1]
    slot_2_qbs = qubit_list_2
    slot_3_qbs = qubit_list_1

    iterations = len(slot_1_qbs)

    # Perform 1st step of the modular addition section
    for i in range(iterations):
        qbits = [slot_1_qbs[i], slot_2_qbs[i], slot_3_qbs[i]]
        qs.append(MAJ_gate(), qbits)

    # Calculate output Qubit
    if output_qubit:
        qs.cx(qbits[2], output_qubit)

    # Perform UMA iterations
    for i in range(iterations - 1, -1, -1):
        qbits = [slot_1_qbs[i], slot_2_qbs[i], slot_3_qbs[i]]

        qs.append(UMA_gate(), qbits)

    if carry_in is None:
        # Delete ancilla
        ancilla.delete()


def cuccaro_adder(a, b, c_in=None, c_out=None):
    """
    In-place adder function based on `this paper <https://arxiv.org/abs/quant-ph/0410184>`__ .
    Performs the addition:

    ::

        b += a


    Parameters
    ----------
    a : int or QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_in : Qubit, optional
        An optional carry in value. The default is None.
    c_out : Qubit, optional
        An optional carry out value. The default is None.

    Examples
    --------

    We add two integers:

    >>> from qrisp import QuantumFloat, cuccaro_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> cuccaro_adder(a,b)
    >>> print(b)
    {9: 1.0}

    """

    cuccaro_procedure(b[0].qs(), a, b, carry_in=c_in, output_qubit=c_out)


temp = cuccaro_adder.__doc__
cuccaro_adder = ammend_inpl_adder(cuccaro_adder)
cuccaro_adder.__doc__ = temp
