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


# This file uses the gate constructor from operation.py to define
# some standard operations

import types

import numpy as np

from qrisp.circuit import PauliGate, U3Gate
from qrisp.circuit.operation import Operation

# TODO: properly treat all gates
# from Aer.get_backend('qasm_simulator').configuration().basis_gates


def Measurement():
    res = Operation(name="measure", num_qubits=1, num_clbits=1)
    res.permeability = {0: True}
    return res


def XGate():
    return PauliGate(name="x")


def YGate():
    return PauliGate(name="y")


def ZGate():
    return PauliGate(name="z")


def CXGate():
    return XGate().control()


def CYGate():
    return YGate().control()


def CZGate():
    return ZGate().control()


def MCXGate(control_amount=1, ctrl_state=-1, method="gray"):
    return XGate().control(control_amount, method=method, ctrl_state=ctrl_state)


def PGate(phi=0):
    res = U3Gate(0, 0, phi, name="p")
    return res


def CPGate(phi=0):
    if isinstance(phi, (int, float)):
        phi = phi % (2 * np.pi)
        if np.abs(phi - np.pi) < 1e-8:
            return CZGate()
        elif np.abs(phi) < 1e-8:
            from qrisp.circuit.quantum_circuit import QuantumCircuit

            temp_circ = QuantumCircuit(2)
            res = temp_circ.to_gate(name="cp")
            res.params = [0]
            return res

        else:
            return PGate(phi).control()
    else:
        return PGate(phi).control()


def HGate():
    res = U3Gate(np.pi / 2, 0, np.pi, name="h")
    return res


def RXGate(phi=0):
    res = U3Gate(phi, -np.pi / 2, np.pi / 2, name="rx")
    return res


def RYGate(phi=0):
    res = U3Gate(phi, 0, 0, name="ry")
    return res


def RZGate(phi=0):
    res = U3Gate(0, phi, 0, name="rz", global_phase=-phi / 2)
    return res


def GPhaseGate(phi=0):
    res = U3Gate(0, 0, 0, name="gphase", global_phase=phi)
    res.params = [phi]
    res.permeability[0] = True
    res.is_qfree = True
    return res


def MCRXGate(phi=0, control_amount=0):
    res = RXGate(phi).control(control_amount)
    res.name = "mcrx"
    return res


def SGate():
    res = PGate(np.pi / 2)
    res.params = []
    res.name = "s"
    return res


def TGate():
    res = PGate(np.pi / 4)
    res.params = []
    res.name = "t"
    return res


def IDGate():
    res = U3Gate(0, 0, 0, name="id")
    return res


def RXXGate(phi=0):
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.gphase(-phi / 2, 0)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.p(phi, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)

    return Operation(
        name="rxx", num_qubits=2, num_clbits=0, params=[phi], definition=qc
    )


def RZZGate(phi=0):
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.gphase(-phi / 2, 0)
    qc.cx(0, 1)
    qc.p(phi, 1)
    qc.cx(0, 1)

    return Operation(
        name="rzz", num_qubits=2, num_clbits=0, params=[phi], definition=qc
    )

def XXYYGate(phi=0, beta=0):
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.rz(beta, 0)
    qc.rz(-np.pi/2, 1)
    qc.sx(1)
    qc.rz(np.pi/2, 1)
    qc.s(0)
    qc.cx(1,0)
    qc.ry(-phi/2, 1)
    qc.ry(-phi/2, 0)
    qc.cx(1,0)
    qc.s_dg(0)
    qc.rz(-np.pi/2,1)
    qc.sx_dg(1)
    qc.rz(np.pi/2,1)
    qc.rz(-beta,0)

    return Operation(
        name="xxyy", num_qubits=2, num_clbits=0, params=[phi,beta], definition=qc
    )

def Barrier(num_qubits=1):
    res = Operation(num_qubits=num_qubits, name="barrier")

    res.permeability = {i: False for i in range(num_qubits)}
    res.is_qfree = True

    return res


def u3Gate(theta=0, phi=0, lam=0):
    return U3Gate(theta, phi, lam)


def U1Gate(phi=0):
    res = RZGate(phi)
    res.name = "u1"
    res.params = [phi]
    return res


def Reset():
    res = Operation("reset", 1)
    res.permeability = {0: False}
    return res


def SXGate():
    res = RXGate(np.pi / 2)
    res.name = "sx"
    res.params = []
    return res


def SXDGGate():
    res = RXGate(-np.pi / 2)
    res.name = "sx_dg"
    res.params = []
    return res


def SwapGate():
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    temp_qc = QuantumCircuit(2)
    temp_qc.cx(0, 1)
    temp_qc.cx(1, 0)
    temp_qc.cx(0, 1)

    res = temp_qc.to_gate(name="swap")

    res.inverse = types.MethodType(lambda self: self.copy(), res)

    return res


def QubitAlloc():
    res = Operation("qb_alloc", 1)
    res.unitary = np.eye(2, dtype=np.complex64)
    res.qfree = True
    res.permeability[0] = False
    return res


def QubitDealloc():
    res = Operation("qb_dealloc", 1)
    res.unitary = np.eye(2, dtype=np.complex64)
    res.qfree = True
    res.permeability[0] = False
    return res


op_list = [
    XGate,
    YGate,
    ZGate,
    CXGate,
    CYGate,
    CZGate,
    MCXGate,
    PGate,
    CPGate,
    u3Gate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    MCRXGate,
    SGate,
    TGate,
    RXXGate,
    RZZGate,
    SXGate,
    SXDGGate,
    XXYYGate,
    Barrier,
    Measurement,
    Reset,
    QubitAlloc,
    QubitDealloc,
    GPhaseGate,
    SwapGate,
    U1Gate,
    IDGate,
]
