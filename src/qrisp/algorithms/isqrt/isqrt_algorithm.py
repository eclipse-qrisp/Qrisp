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

from qrisp import *
from qrisp.algorithms.isqrt.tools import *

def initial_subtraction(qc: QuantumCircuit, n: int):
    R = [i for i in range(n)]
    F = [i for i in range(n, 2 * n)]
    z = 2 * n

    # Step 1
    qc.x(R[n-2])

    # Step 2
    qc.cx(R[n-2], R[n-1])

    # Step 3
    qc.cx(R[n-1], F[1])

    # Step 4
    zcx(qc, (R[n-1], z))
    
    # Step 5
    zcx(qc, (R[n-1], F[2]))

    # Step 6
    control_add_sub(qc, 4, z, [R[n-4], R[n-3], R[n-2], R[n-1]], [F[0], F[1], F[2], F[3]])

def conditional_addition_or_subtraction(qc: QuantumCircuit, n: int):
    R = [i for i in range(n)]
    F = [i for i in range(n, 2 * n)]
    z = 2 * n

    for i in range(2, n // 2):
        # Step 1
        zcx(qc, (z, F[1]))

        # Step 2
        qc.cx(F[2], z)

        # Step 3
        qc.cx(R[n-1], F[1])

        # Step 4
        zcx(qc, (R[n-1], z))

        # Step 5
        zcx(qc, (R[n-1], F[i+1]))

        # Step 6
        for j in range(i + 1, 2, -1):
            qc.swap(F[j], F[j-1])

        # Step 7
        R_sum_qubits = [R[j] for j in range(n - 2 * i - 2, n)]
        F_sum_qubits = [F[j] for j in range(0, 2 * i + 2)]
        control_add_sub(qc, len(R_sum_qubits), z, R_sum_qubits, F_sum_qubits)

def remainder_restoration(qc: QuantumCircuit, n: int):
    R = [i for i in range(n)]
    F = [i for i in range(n, 2 * n)]
    z = 2 * n

    # Step 1
    zcx(qc, (z, F[1]))

    # Step 2
    qc.cx(F[2], z)

    # Step 3
    zcx(qc, (R[n-1], z))

    # Step 4
    zcx(qc, (R[n-1], F[n//2+1]))

    # Step 5
    qc.x(z)

    # Step 6
    control_add(qc, n, z, R, F)

    # Step 7
    qc.x(z)

    # Step 8
    for j in range(n//2 + 1, 2, -1):
        qc.swap(F[j], F[j-1])

    # Step 9
    qc.cx(F[2], z)

def isqrt_circuit(n: int) -> Operation:
    """
    Builds the quantum circuit for the integer square root algorithm.

    Parameters
    ----------
    n : int
        The number of qubits in the input number.

    Returns
    -------
    qc : QuantumCircuit
        The quantum circuit for the integer square root algorithm.
    """
    qc = QuantumCircuit(2 * n + 1)

    initial_subtraction(qc, n)
    conditional_addition_or_subtraction(qc, n)
    remainder_restoration(qc, n)

    return qc.to_gate(name="ISQRT")
    

def isqrt_alg(R: QuantumFloat) -> QuantumFloat:
    """
    Computes the integer square root of a QuantumFloat R, as well as the remainder using algorithm `<https://arxiv.org/abs/1712.08254>`.
    The input value R should be a positive binary value in 2's complement representation with even number of bits.

    Parameters
    ----------
    R : QuantumFloat
        Positive value in 2's complement representation with even number of bits which square root is to be computed.
    
    Returns
    -------
    res : QuantumFloat
        The integer square root of R.
    Also transforms input value R into the remainder.

    Examples
    --------
    Calculate the integer square root of 131:
    >>> from qrisp.isqrt import isqrt_alg
    >>> R = QuantumFloat(10, 0) # 131 has 8 bits, but since msb is 1, we need to add 1 qubit, so its 2's complement and another 1 qubit, so it has even number of qubits.
    >>> R[:] = 131
    >>> res = isqrt_alg(R)
    >>> print(res.get_measurement())
    {11: 1.0}
    >>> print(R.get_measurement())
    {10: 1.0}
    """
    n = R.size
    
    F = QuantumFloat(n, 0, name="F")
    z = QuantumFloat(1, 0, name="z")

    F[:] = 1
    z[:] = 0

    qs = QuantumSession()
    qs.append(isqrt_circuit(n), R[:] + F[:] + z[:])
    qs.x(F[0])
    for i in range(2, n // 2 + 2):
        qs.swap(F[i], F[i - 2])

    return F