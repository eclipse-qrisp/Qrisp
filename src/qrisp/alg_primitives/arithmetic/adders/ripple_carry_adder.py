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

import numpy as np

from qrisp import QuantumCircuit
from qrisp.misc import gate_wrap
from qrisp import cdkpm_adder



@gate_wrap(is_qfree=True, permeability=[1])
def inpl_add(
    qf1,
    qf2,
    ignore_rounding_error=False,
    ignore_overflow_error=False,
    adder="thapliyal",
):
    """
    Performs in-place addition of the second argument onto the first.
    In Python syntax: ::

        qf1 += qf2

    There are two different algorithms available:
    The `Thapliyal adder <https://arxiv.org/abs/1712.02630>`_
    and the `Cuccaro adder <https://arxiv.org/abs/quant-ph/0410184>`_

    Parameters
    ----------
    qf1 : QuantumFloat
        The QuantumFloat that will be in-place modified.
    qf2 : QuantumFloat
        The QuantumFloat that is being added.
    ignore_rounding_error : bool, optional
        If set to False, an Exception will be raised if qf2 has higher precision
        than qf1. The default is False.
    ignore_overflow_error : bool, optional
        If set to False, an Exception will be raised if qf2 has higher maximum
        significance than qf2. The default is False.
    adder : str, optional
        Specifies the adder. Available are "thapliyal" and "cuccaro".
        The default is "thapliyal".

    Raises
    ------
    Exception
        Tried to add signed QuantumFloat onto non signed QuantumFloat.

    Examples
    --------

    We create two QuantumFloats and apply the inplace adder

    >>> from qrisp import QuantumFloat, inpl_add
    >>> qf_0 = QuantumFloat(5)
    >>> qf_1 = QuantumFloat(5)
    >>> qf_0[:] = 4
    >>> qf_1[:] = 3
    >>> inpl_add(qf_0, qf_1)
    >>> print(qf_0)
    {7.0: 1.0}
    """

    qs = qf1.qs

    # If qf2 has lower exponent, the qf2 bits with less significance than all of qf1
    # can not be added to qf1 (rounding error)
    if not ignore_rounding_error and qf1.exponent > qf2.exponent:
        raise Exception(
            "Tried to add QuantumFloat to QuantumFloat of lower precision"
            " (set ignore_rounding_error = True)"
        )

    # If qf2 has higher maximum significance than qf1, the qf2 bits with higher
    # significance than all of qf1 can not be added to qf1 (overflow error)
    if not ignore_overflow_error and qf1.mshape[1] < qf2.mshape[1]:
        raise Exception(
            "Tried to add QuantumFloat to QuantumFloat of lower precision"
            " (set ignore_overflow_error = True)"
        )

    # Determine the significance range
    # For instance for a QuantumFloat with datashape [-2,3]
    # has significance range [-2,-1,0,1,2,3]
    significance_range_qf1 = list(range(qf1.mshape[0], qf1.mshape[1] + 1))
    significance_range_qf2 = list(range(qf2.mshape[0], qf2.mshape[1] + 1))

    # Determine the intersection of the significance ranges
    signficance_range_intersetion = list(
        set(significance_range_qf1).intersection(significance_range_qf2)
    )

    # Determine maximum and minimum significance of the addition
    # The maximum significance is the maximum significance of qf1
    # (since we are adding in-place)
    # The minimum significance is the minimum of the intersection because
    # if qf2.exponent < qf1.exponent
    # we are allowed to ignore possible rounding errors
    # (skip all significances lower than qf1.exponent)
    # if qf2.exponent > qf1.exponent
    # The qbits of qf1 with lower significance than all of qf2
    # stay unchanged during addition

    max_sig = max(significance_range_qf1)
    min_sig = min(signficance_range_intersetion)

    # If the maximum significance is higher than the maximum signficance of qf2, we need
    # to augment some ancilla qubits because the Cuccaro-procedure requires equal
    # amount of input qubits

    from qrisp.core import QuantumVariable

    if max_sig > max(significance_range_qf2):
        # print(max_sig-max(significance_range_qf2))
        ancilla_var = QuantumVariable(
            max_sig - max(significance_range_qf2) - int(qf2.signed)
        )
        augmented_qf2_qbs = qf2.reg + ancilla_var.reg
    else:
        augmented_qf2_qbs = qf2.reg

    # Determine the bit window of which bits should participate in the Cuccaro-procedure
    bit_window_1 = [
        significance_range_qf1.index(min_sig),
        significance_range_qf1.index(max_sig),
    ]

    if max_sig > max(significance_range_qf2):
        bit_window_2 = [significance_range_qf2.index(min_sig), len(augmented_qf2_qbs)]
    else:
        bit_window_2 = [
            significance_range_qf2.index(min_sig),
            significance_range_qf2.index(max_sig),
        ]

    # print(max_sig)
    # print(ancilla_var_.size)
    # print(significance_range_qf1)
    # print(significance_range_qf2)
    # print(bit_window_1)
    # print(bit_window_2)

    # Determine qubit lists
    qubit_list_1 = qf1.reg[bit_window_1[0] : bit_window_1[1]]
    qubit_list_2 = augmented_qf2_qbs[bit_window_2[0] : bit_window_2[1]]

    # print(len(qubit_list_1))
    # print(len(qubit_list_2))

    # Now we treat the sign bits
    if qf2.signed and not qf1.signed:
        raise Exception("Tried to add signed QuantumFloat to non signed QuantumFloat")

    # The signs are handled via modular arithmetic
    # According to the Cuccaro-Paper this is done via manipulating
    # the final qubits of the input list
    # if True:
    if qf1.signed:
        qubit_list_1.append(qf1[-1])

        if qf2.signed:
            qubit_list_2.append(qf2[-1])
        else:
            ancilla_var_2 = QuantumVariable(1)
            qubit_list_2.append(ancilla_var_2[0])

    # If qf2 is signed and we augmented Qubits, we need to make sure
    # the augmented qubits are set to 1 if qf2 is in a negative state
    # (for details on this check how negative numbers are encoded)
    if qf2.signed and max_sig > max(significance_range_qf2):
        for i in range(ancilla_var.size):
            qs.cx(qf2[-1], ancilla_var[i])

    if adder == "thapliyal":
        thapliyal_procedure(
            qs, qubit_list_2[:-1], qubit_list_1[:-1], output_qubit=qubit_list_1[-1]
        )
    else:
        raise Exception("Adder " + adder + " not implemented")

    # Perform 2nd step of the modular addition section
    # qs.cx(qubit_list_2[-2], qubit_list_1[-1])
    qs.cx(qubit_list_2[-1], qubit_list_1[-1])

    # Remove the 1s from the augmented qubits and delete
    if len(augmented_qf2_qbs) != len(qf2.reg):
        if qf2.signed:
            for i in range(ancilla_var.size):
                qs.cx(qf2[-1], ancilla_var[i])

    try:
        ancilla_var.delete()
    except NameError:
        pass

    if qf1.signed and not qf2.signed:
        ancilla_var_2.delete()


# Workaround for protecting the docstring from the decorator
temp = inpl_add.__doc__
inpl_add = gate_wrap(inpl_add)
inpl_add.__doc__ = temp


# Adder based on https://arxiv.org/abs/1712.02630


def TR_gate():
    qc = QuantumCircuit(3)
    qc.crx(-np.pi / 2, 1, 2)
    qc.p(-np.pi / 4, 1)
    qc.cx(0, 1)

    qc.p(np.pi / 4, 0)
    qc.crx(np.pi / 2, 0, 2)

    qc.p(np.pi / 4, 1)
    qc.crx(np.pi / 2, 1, 2)

    # Error in Thapliyal paper? Doesnt work if there is no inverse here
    result = qc.to_gate().inverse()
    result.name = "TR"
    return result


def thapliyal_procedure(qc, qubit_list_1, qubit_list_2, output_qubit):
    if len(qubit_list_1) != len(qubit_list_2):
        raise Exception(
            "Tried to call Thapliyal-procedure with qubit lists of unequal length"
        )

    n = len(qubit_list_1)

    # Step 1
    for i in range(1, n):
        qc.cx(qubit_list_1[i], qubit_list_2[i])

    # Step 2
    qc.cx(qubit_list_1[-1], output_qubit)

    for i in range(n - 2, 0, -1):
        qc.cx(qubit_list_1[i], qubit_list_1[i + 1])

    # Step 3
    for i in range(n - 1):
        qc.mcx([qubit_list_1[i], qubit_list_2[i]], qubit_list_1[i + 1])

    # Step 4
    qc.append(TR_gate(), [qubit_list_1[-1], qubit_list_2[-1], output_qubit])

    for i in range(n - 2, -1, -1):
        qc.append(TR_gate(), [qubit_list_1[i], qubit_list_2[i], qubit_list_1[i + 1]])

    # Step 5
    for i in range(1, n - 1):
        qc.cx(qubit_list_1[i], qubit_list_1[i + 1])

    # Step 6
    for i in range(1, n):
        qc.cx(qubit_list_1[i], qubit_list_2[i])
