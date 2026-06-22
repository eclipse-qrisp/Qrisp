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

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np

from qrisp.circuit import PauliGate, U3Gate
from qrisp.circuit.operation import Operation

if TYPE_CHECKING:
    from qrisp.typing import FloatLike

# TODO: properly treat all gates
# from Aer.get_backend('qasm_simulator').configuration().basis_gates


def Measurement():
    """Return a measurement :class:`.Operation`.

    Returns
    -------
    :class:`.Operation`
        A measurement operation with 1 qubit and 1 classical bit.
    """
    res = Operation(name="measure", num_qubits=1, num_clbits=1)
    res.permeability = {0: True}
    return res


def XGate():
    """Return the Pauli-X (bit-flip / NOT) gate.

    Returns
    -------
    :class:`.Operation`
        The Pauli-X gate operation.
    """
    return PauliGate(name="x")


def YGate():
    """Return the Pauli-Y gate.

    Returns
    -------
    :class:`.Operation`
        The Pauli-Y gate operation.
    """
    return PauliGate(name="y")


def ZGate():
    """Return the Pauli-Z (phase-flip) gate.

    Returns
    -------
    :class:`.Operation`
        The Pauli-Z gate operation.
    """
    return PauliGate(name="z")


def CXGate():
    """Return the controlled-X (CNOT) gate.

    Returns
    -------
    :class:`.Operation`
        The controlled-X gate operation.
    """
    return XGate().control()


def CYGate():
    """Return the controlled-Y gate.

    Returns
    -------
    :class:`.Operation`
        The controlled-Y gate operation.
    """
    return YGate().control()


def CZGate():
    """Return the controlled-Z gate.

    Returns
    -------
    :class:`.Operation`
        The controlled-Z gate operation.
    """
    return ZGate().control()


def MCXGate(control_amount=1, ctrl_state=-1, method="gray"):
    """Return a multi-controlled X gate.

    Parameters
    ----------
    control_amount : int, optional
        Number of control qubits. The default is 1.
    ctrl_state : int, optional
        Integer that represents the control state in binary.
        The default is ``-1`` (all controls active when in state ``|1⟩``).
    method : str, optional
        Synthesis method. The default is ``"gray"``.

    Returns
    -------
    :class:`.Operation`
        The multi-controlled X gate operation.
    """
    return XGate().control(control_amount, method=method, ctrl_state=ctrl_state)


def PGate(phi: FloatLike = 0):
    """Return a phase gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The phase angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The phase gate operation.
    """
    res = U3Gate(0, 0, phi, name="p")
    return res


def CPGate(phi: FloatLike = 0):
    """Return a controlled phase gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The phase angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The controlled phase gate operation.
    """
    if isinstance(phi, (int, float)):
        phi = phi % (2 * np.pi)
        if np.abs(phi - np.pi) < 1e-8:
            return CZGate()
        if np.abs(phi) < 1e-8:
            from qrisp.circuit.quantum_circuit import QuantumCircuit

            temp_circ = QuantumCircuit(2)
            res = temp_circ.to_gate(name="cp")
            res.params = [0]
            return res
        return PGate(phi).control()
    return PGate(phi).control()


def HGate():
    """Return the Hadamard gate.

    Returns
    -------
    :class:`.Operation`
        The Hadamard gate operation.
    """
    res = U3Gate(np.pi / 2, 0, np.pi, name="h")
    return res


def RXGate(phi: FloatLike = 0):
    """Return a rotation-around-X gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The rotation angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The :math:`R_X` gate operation.
    """
    res = U3Gate(phi, -np.pi / 2, np.pi / 2, name="rx")
    return res


def RYGate(phi: FloatLike = 0):
    """Return a rotation-around-Y gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The rotation angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The :math:`R_Y` gate operation.
    """
    res = U3Gate(phi, 0, 0, name="ry")
    return res


def RZGate(phi: FloatLike = 0):
    """Return a rotation-around-Z gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The rotation angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The :math:`R_Z` gate operation.
    """
    res = U3Gate(0, phi, 0, name="rz", global_phase=-phi / 2)  # type: ignore[operator]
    return res


def RGate(theta: FloatLike = 0, phi: FloatLike = 0):
    """Return an R gate.

    Parameters
    ----------
    theta : FloatLike, optional
        The rotation angle in radians. The default is 0.
    phi : FloatLike, optional
        The axis angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The R gate operation.
    """
    res = U3Gate(-theta, phi, -phi, name="r", global_phase=0)
    res.params = [theta, phi]
    return res


def GPhaseGate(phi: FloatLike = 0):
    """Return a global phase gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The global phase angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The global phase gate operation.
    """
    res = U3Gate(0, 0, 0, name="gphase", global_phase=phi)
    return res


def MCRXGate(phi: FloatLike = 0, control_amount: int = 0):
    """Return a multi-controlled :func:`RXGate`.

    Parameters
    ----------
    phi : FloatLike, optional
        The rotation angle in radians. The default is 0.
    control_amount : int, optional
        Number of control qubits. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The multi-controlled :math:`R_X` gate operation.
    """
    res = RXGate(phi).control(control_amount)
    res.name = "mcrx"
    return res


def SGate():
    """Return the S gate.

    Returns
    -------
    :class:`.Operation`
        The S gate operation.
    """
    res = PGate(np.pi / 2)
    res.params = []
    res.name = "s"
    return res


def TGate():
    """Return the T gate.

    Returns
    -------
    :class:`.Operation`
        The T gate operation.
    """
    res = PGate(np.pi / 4)
    res.params = []
    res.name = "t"
    return res


def IDGate():
    """Return the single-qubit identity gate.

    Returns
    -------
    :class:`.Operation`
        The identity gate operation.
    """
    res = U3Gate(0, 0, 0, name="id")
    return res


def RXXGate(phi: FloatLike = 0):
    """Return an Ising XX-coupling gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The coupling angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The :math:`R_{XX}` gate operation.
    """
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.gphase(-phi / 2, qc.qubits[0])
    qc.h(qc.qubits[0])
    qc.h(qc.qubits[1])
    qc.cx(qc.qubits[0], qc.qubits[1])
    qc.p(phi, qc.qubits[1])
    qc.cx(qc.qubits[0], qc.qubits[1])
    qc.h(qc.qubits[0])
    qc.h(qc.qubits[1])

    res = Operation(name="rxx", num_qubits=2, num_clbits=0, params=[phi], definition=qc)

    res.permeability = {0: False, 1: False}
    res.is_qfree = False

    return res


def RZZGate(phi: FloatLike = 0):
    """Return an Ising ZZ-coupling gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The coupling angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The :math:`R_{ZZ}` gate operation.
    """
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.cx(qc.qubits[0], qc.qubits[1])
    qc.gphase(-phi / 2, qc.qubits[0])
    qc.p(phi, qc.qubits[1])
    qc.cx(qc.qubits[0], qc.qubits[1])

    res = Operation(name="rzz", num_qubits=2, num_clbits=0, params=[phi], definition=qc)

    res.permeability = {0: True, 1: True}
    res.is_qfree = True

    return res


def XXYYGate(phi: FloatLike = 0, beta: FloatLike = 0):
    """Return an XX+YY interaction gate.

    Parameters
    ----------
    phi : FloatLike, optional
        The rotation angle in radians. The default is 0.
    beta : FloatLike, optional
        The phase angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The XX+YY gate operation.
    """
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    qc = QuantumCircuit(2)

    qc.rz(beta, qc.qubits[0])
    qc.rz(-np.pi / 2, qc.qubits[1])
    qc.sx(qc.qubits[1])
    qc.rz(np.pi / 2, qc.qubits[1])
    qc.s(qc.qubits[0])
    qc.cx(qc.qubits[1], qc.qubits[0])
    qc.ry(-phi / 2, qc.qubits[1])
    qc.ry(-phi / 2, qc.qubits[0])
    qc.cx(qc.qubits[1], qc.qubits[0])
    qc.s_dg(qc.qubits[0])
    qc.rz(-np.pi / 2, qc.qubits[1])
    qc.sx_dg(qc.qubits[1])
    qc.rz(np.pi / 2, qc.qubits[1])
    qc.rz(-beta, qc.qubits[0])

    return Operation(
        name="xxyy", num_qubits=2, num_clbits=0, params=[phi, beta], definition=qc
    )


def Barrier(num_qubits=1):
    """Return a barrier.

    Parameters
    ----------
    num_qubits : int, optional
        The number of qubits the barrier spans. The default is 1.

    Returns
    -------
    :class:`.Operation`
        The barrier operation spanning ``num_qubits`` qubits.
    """
    res = Operation(num_qubits=num_qubits, name="barrier")

    res.permeability = {i: False for i in range(num_qubits)}
    res.is_qfree = True

    return res


def u3Gate(theta: FloatLike = 0, phi: FloatLike = 0, lam: FloatLike = 0):
    """Return a U3 gate with Euler angles ``theta``, ``phi``, ``lam``.

    This is a thin alias for :class:`.U3Gate`.

    Parameters
    ----------
    theta : FloatLike, optional
        The theta Euler angle in radians. The default is 0.
    phi : FloatLike, optional
        The phi Euler angle in radians. The default is 0.
    lam : FloatLike, optional
        The lambda Euler angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The U3 gate operation.
    """
    return U3Gate(theta, phi, lam)


def U1Gate(phi: FloatLike = 0):
    """Return a U1 gate, equivalent to :func:`RZGate` up to a global phase.

    Parameters
    ----------
    phi : FloatLike, optional
        The phase angle in radians. The default is 0.

    Returns
    -------
    :class:`.Operation`
        The U1 gate operation.
    """
    res = RZGate(phi)
    res.name = "u1"
    res.params = [phi]

    res.permeability = {0: True}
    res.is_qfree = True

    return res


def Reset():
    """Return a reset gate.

    Returns
    -------
    :class:`.Operation`
        The reset operation.
    """
    res = Operation("reset", 1)
    res.permeability = {0: False}
    return res


def SXGate():
    """Return the square-root-of-X gate.

    Returns
    -------
    :class:`.Operation`
        The :math:`\\sqrt{X}` gate operation.
    """
    res = RXGate(np.pi / 2)
    res.name = "sx"
    res.params = []

    res.permeability = {0: False}
    res.is_qfree = False

    return res


def SXDGGate():
    """Return the adjoint of the square-root-of-X gate.

    Returns
    -------
    :class:`.Operation`
        The adjoint of the square-root-of-X gate operation.
    """
    res = RXGate(-np.pi / 2)
    res.name = "sx_dg"
    res.params = []

    res.permeability = {0: False}
    res.is_qfree = False

    return res


def SwapGate():
    """Return a SWAP gate.

    Returns
    -------
    :class:`.Operation`
        The SWAP gate operation.
    """
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    temp_qc = QuantumCircuit(2)
    temp_qc.cx(temp_qc.qubits[0], temp_qc.qubits[1])
    temp_qc.cx(temp_qc.qubits[1], temp_qc.qubits[0])
    temp_qc.cx(temp_qc.qubits[0], temp_qc.qubits[1])

    res = temp_qc.to_gate(name="swap")

    res.inverse = types.MethodType(lambda self: self.copy(), res)

    res.permeability = {0: False, 1: False}
    res.is_qfree = True

    return res


def QubitAlloc():
    """Return an internal qubit-allocation marker.

    Returns
    -------
    :class:`.Operation`
        The qubit-allocation marker operation.
    """
    res = Operation("qb_alloc", 1)
    res.unitary = np.eye(2, dtype=np.complex64)
    res.qfree = True
    res.permeability[0] = False
    return res


def QubitDealloc():
    """Return an internal qubit-deallocation marker.

    Returns
    -------
    :class:`.Operation`
        The qubit-deallocation marker operation.
    """
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
    RGate,
]
