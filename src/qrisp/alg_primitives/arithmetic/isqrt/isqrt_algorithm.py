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

from qrisp.circuit import Qubit
from qrisp.qtypes import QuantumFloat
from qrisp.core.gate_application_functions import x, cx, mcx, swap


def peres(a: Qubit, b: Qubit, c: Qubit):
    mcx([a, b], c)
    cx(a, b)


def add(A: list[Qubit], B: list[Qubit]):
    n = len(A)
    # Step 1
    for i in range(1, n, 1):
        cx(B[i], A[i])

    # Step 2
    for i in range(n - 2, 0, -1):
        cx(B[i], B[i + 1])

    # Step 3
    for i in range(0, n - 1, 1):
        mcx([B[i], A[i]], B[i + 1])

    # Step 4
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            cx(B[i], A[i])
        else:
            peres(B[i], A[i], B[i + 1])

    # Step 5
    for i in range(1, n - 1, 1):
        cx(B[i], B[i + 1])

    # Step 6
    for i in range(1, n, 1):
        cx(B[i], A[i])


def control_add(z: Qubit, B: list[Qubit], A: list[Qubit]):
    n = len(B)

    # Step 1
    for i in range(1, n):
        cx(A[i], B[i])

    # Step 2
    for i in range(n - 2, 0, -1):
        cx(A[i], A[i + 1])

    # Step 3
    for i in range(0, n - 1):
        mcx([B[i], A[i]], A[i + 1])

    # Step 4
    mcx([z, A[n - 1]], B[n - 1])

    # Step 5
    for i in range(n - 2, -1, -1):
        mcx([B[i], A[i]], A[i + 1])
        mcx([z, A[i]], B[i])

    # Step 6
    for i in range(1, n - 1):
        cx(A[i], A[i + 1])

    # Step 7
    for i in range(1, n):
        cx(A[i], B[i])


def control_add_sub(z: Qubit, A: list[Qubit], B: list[Qubit]):
    for i in A:
        cx(z, i)

    add(A, B)

    for i in A:
        cx(z, i)


def initial_subtraction(R: QuantumFloat, F: QuantumFloat, z: QuantumFloat):
    n = R.size

    # Step 1
    x(R[n - 2])

    # Step 2
    cx(R[n - 2], R[n - 1])

    # Step 3
    cx(R[n - 1], F[1])

    # Step 4
    mcx([R[n - 1]], z[0], ctrl_state=0)

    # Step 5
    mcx([R[n - 1]], F[2], ctrl_state=0)

    # Step 6
    control_add_sub(
        z, [R[n - 4], R[n - 3], R[n - 2], R[n - 1]], [F[0], F[1], F[2], F[3]]
    )


def conditional_addition_or_subtraction(
    R: QuantumFloat, F: QuantumFloat, z: QuantumFloat
):
    n = R.size

    for i in range(2, n // 2):
        # Step 1
        mcx(z, F[1], ctrl_state=0)

        # Step 2
        cx(F[2], z)

        # Step 3
        cx(R[n - 1], F[1])

        # Step 4
        mcx([R[n - 1]], z[0], ctrl_state=0)

        # Step 5
        mcx([R[n - 1]], F[i + 1], ctrl_state=0)

        # Step 6
        for j in range(i + 1, 2, -1):
            swap(F[j], F[j - 1])

        # Step 7
        R_sum_qubits = [R[j] for j in range(n - 2 * i - 2, n)]
        F_sum_qubits = [F[j] for j in range(0, 2 * i + 2)]
        control_add_sub(z, R_sum_qubits, F_sum_qubits)


def remainder_restoration(R: QuantumFloat, F: QuantumFloat, z: QuantumFloat):
    n = R.size

    # Step 1
    mcx(z, F[1], ctrl_state=0)

    # Step 2
    cx(F[2], z)

    # Step 3
    mcx([R[n - 1]], z[0], ctrl_state=0)

    # Step 4
    mcx([R[n - 1]], F[n // 2 + 1], ctrl_state=0)

    # Step 5
    x(z)

    # Step 6
    control_add(z, R, F)

    # Step 7
    x(z)

    # Step 8
    for j in range(n // 2 + 1, 2, -1):
        swap(F[j], F[j - 1])

    # Step 9
    cx(F[2], z)


def q_isqrt(R: QuantumFloat) -> QuantumFloat:
    """
    Computes the integer square root of a QuantumFloat, as well as the remainder using the `non-Restoring square root algorithm <https://arxiv.org/abs/1712.08254>`_.
    Does not yield plausible output when applied to negative integers.

    Parameters
    ----------
    R : QuantumFloat
        QuantumFloat value to compute the integer square root of.

    Returns
    -------
    QuantumFloat
        The integer square root of R.
    Also transforms input value R into the remainder.

    Examples
    --------
    Calculate the integer square root of 131:

    >>> from qrisp import QuantumFloat, q_isqrt
    >>> R = QuantumFloat(8, 0)
    >>> R[:] = 131
    >>> res = q_isqrt(R)
    >>> print(res.get_measurement())
    >>> print(R.get_measurement())
    {11: 1.0}
    {10: 1.0}
    """
    n = R.size
    e = R.exponent

    if e != 0:
        raise Exception(
            "Tried to compute integer square root for QuantumFloat with non-zero exponent"
        )

    if n == 1:
        F = QuantumFloat(1, 0)
        cx(R[0], F[0])
        cx(F[0], R[0])
        return F

    # To ensure that first qubit is 0 and total amount of qubits is even we extend the input variable
    delta = 1 if n % 2 == 1 else 2
    n += delta
    R.extend(delta)

    F = QuantumFloat(n, 0)
    z = QuantumFloat(1, 0)

    F[:] = 1
    z[:] = 0

    initial_subtraction(R, F, z)
    conditional_addition_or_subtraction(R, F, z)
    remainder_restoration(R, F, z)

    x(F[0])
    for i in range(2, n // 2 + 2):
        swap(F[i], F[i - 2])

    R.reduce(R[-delta:])
    F.reduce(F[-delta:])
    z.delete()

    return F
