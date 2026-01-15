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
