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

def zcx(qc: QuantumCircuit, qubits: tuple[int, int]):
    """
    Applies a controlled-X gate with control state 0 to the qubits.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to apply the gate to.
    qubits : tuple[int, int]
        The qubits to apply the controlled-X gate to.
    """
    qc.append(XGate().control(ctrl_state=0), [qubits[0], qubits[1]])

def peres(qc: QuantumCircuit, qubits: tuple[int, int, int]):
    """
    Applies the Peres gate to the qubits.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to apply the gate to.
    qubits : tuple[int, int, int]
        The qubits to apply the Peres gate to.
    """
    qc.ccx(qubits[0], qubits[1], qubits[2])
    qc.cx(qubits[0], qubits[1])

def add(qc: QuantumCircuit, n: int, A: list[int], B: list[int]):
    """
    Performs the addition of two numbers A and B.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to apply the gate to.
    n : int
        The number of qubits in the input numbers.
    A : list[int]
        The n qubits of the first number.
    B : list[int]
        The n qubits of the second number.
    """
    # Step 1
    for i in range(1, n, 1):
        qc.cx(B[i], A[i])
        
    # Step 2
    for i in range(n-2, 0, -1):
        qc.cx(B[i], B[i+1])

    # Step 3
    for i in range(0, n-1, 1):
        qc.ccx(B[i], A[i], B[i+1])

    # Step 4
    for i in range(n-1, -1, -1):
        if(i == n - 1):
            qc.cx(B[i], A[i])
        else:
            peres(qc, (B[i], A[i], B[i+1]))

    # Step 5
    for i in range(1, n-1, 1):
        qc.cx(B[i], B[i+1])

    # Step 6
    for i in range(1, n, 1):
        qc.cx(B[i], A[i])

def control_add(qc: QuantumCircuit, n: int, z: int, B: list[int], A: list[int]):
    """
    Performs the controlled addition of two numbers A and B.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to apply the gate to.
    n : int
        The number of qubits in the input numbers.
    z : int
        The qubit to control the addition.
    B : list[int]
        The n qubits of the first number.
    A : list[int]
        The n qubits of the second number.
    """
    # Step 1
    for i in range(1, n):
        qc.cx(A[i], B[i])

    # Step 2
    for i in range(n - 2, 0, -1):
        qc.cx(A[i], A[i+1])

    # Step 3
    for i in range(0, n - 1):
        qc.ccx(B[i], A[i], A[i+1])

    # Step 4
    qc.ccx(z, A[n-1], B[n-1])

    # Step 5
    for i in range(n-2, -1, -1):
        qc.ccx(B[i], A[i], A[i+1])
        qc.ccx(z, A[i], B[i])

    # Step 6
    for i in range(1, n-1):
        qc.cx(A[i], A[i+1])

    # Step 7
    for i in range(1, n):
        qc.cx(A[i], B[i])

def control_add_sub(qc: QuantumCircuit, n: int, z: int, A: list[int], B: list[int]):
    """
    Performs the controlled addition or subtraction of two numbers A and B, depending on the control qubit z.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to apply the gate to.
    n : int
        The number of qubits in the input numbers.
    z : int
        The qubit to control the addition or subtraction.
    A : list[int]
        The n qubits of the first number.
    B : list[int]
        The n qubits of the second number.
    """
    for i in A:
        qc.cx(z, i)

    add(qc, n, A, B)

    for i in A:
        qc.cx(z, i)