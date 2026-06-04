"""
********************************************************************************
* Copyright (c) 2026 the Qrisp Authors
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
from qrisp.core import cx, mcx
from qrisp.core.gate_application_functions import append_operation

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


def thapliyal_adder(a, b):
    """
    In-place adder based on https://arxiv.org/abs/1712.02630

    Performs ``b += a`` in-place. The first ``size - 1`` qubits of each register
    hold the values being summed, and ``b[-1]`` serves as the carry-out qubit.
    The last qubit of ``a`` is not touched by the algorithm and is available
    for post-processing by the caller.

    Parameters
    ----------
    a : QuantumVariable
        The source register (value to add).
    b : QuantumVariable
        The target register (modified in-place). Must have the same size as ``a``.
        The last qubit of ``b`` functions as the carry-out.

    Raises
    ------
    Exception
        If ``a`` and ``b`` have different sizes.

    Examples
    --------
    >>> from qrisp import QuantumFloat
    >>> from qrisp.alg_primitives.arithmetic.adders.thapliyal_adder import thapliyal_adder
    >>> a = QuantumFloat(3)
    >>> b = QuantumFloat(3)
    >>> a[:] = 2
    >>> b[:] = 4
    >>> thapliyal_adder(a, b)
    >>> print(b)
    {6: 1.0}
    """
    n = a.size if hasattr(a, "size") else len(a)

    if hasattr(b, "size"):
        if b.size != n:
            raise Exception("a and b must have the same size")
    elif len(b) != n:
        raise Exception("a and b must have the same size")

    N = n - 1

    if N <= 0:
        return

    # Step 1
    for i in range(1, N):
        cx(a[i], b[i])

    # Step 2
    cx(a[N - 1], b[-1])
    for i in range(N - 2, 0, -1):
        cx(a[i], a[i + 1])

    # Step 3
    for i in range(N - 1):
        mcx([a[i], b[i]], a[i + 1])

    # Step 4
    append_operation(TR_gate(), [a[N - 1], b[N - 1], b[-1]])
    for i in range(N - 2, -1, -1):
        append_operation(TR_gate(), [a[i], b[i], a[i + 1]])

    # Step 5
    for i in range(1, N - 1):
        cx(a[i], a[i + 1])

    # Step 6
    for i in range(1, N):
        cx(a[i], b[i])
