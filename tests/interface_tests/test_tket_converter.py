# from qrisp.circuit.standard_operations import op_list
import numpy as np
import pytest

np.random.seed(42)  # Deterministic for reproducible test results

from qrisp import QuantumCircuit, QuantumVariable
from qrisp.circuit.standard_operations import (
    CPGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    MCRXGate,
    MCXGate,
    PGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    SGate,
    SwapGate,
    SXGate,
    U1Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)


def test_pytket_rand_test():
    # Needs pytket-qiskit's AerBackend; skips cleanly where it is not installed
    # (e.g. CI), so the test is collected without erroring.
    pytest.importorskip("pytket.extensions.qiskit")
    qvRand = QuantumVariable(10)
    qcRand = qvRand.qs
    rotation = np.pi / 2
    single_gates = [XGate(), YGate(), ZGate(), HGate(), SXGate(), SGate()]

    rot_gates = [RXGate, RYGate, RZGate, PGate]

    c_gates = [CXGate(), CYGate(), CZGate()]

    mc_gates = [
        SwapGate(),
    ]

    mc_rot_gates = [RXXGate, RZZGate]

    special_gates = [MCXGate(control_amount=3)]

    op_list = [
        *mc_gates,
        *c_gates,
        *single_gates,
        *rot_gates,
        *special_gates,
    ]

    used_ops = []
    for index in range(30):
        randInteg = np.random.randint(0, len(op_list) + 1)
        if randInteg == len(op_list):
            used_ops.append("mcrxxx")
        else:
            used_ops.append(op_list[randInteg])

    for op in used_ops:
        qubit_1 = qvRand[np.random.randint(7, 9)]
        qubit_2 = qvRand[np.random.randint(3, 6)]

        if op == "mcrxxx":
            qcRand.append(MCRXGate(rotation, control_amount=3), [qvRand[0], qvRand[2], qubit_2, qubit_1])
        elif op in single_gates:
            qcRand.append(op, qubit_1)
        elif op in rot_gates:
            qcRand.append(op(rotation), qubit_1)
        # this is being called first due to mcx is subclass of cx reasons
        elif op in special_gates:
            qcRand.append(op, [qvRand[0], qvRand[2], qubit_2, qubit_1])
        elif op in mc_rot_gates:
            # elif op in c_gates:
            qcRand.append(op(rotation), [qubit_1, qubit_2])
        elif op in c_gates or mc_gates:
            # elif op in c_gates:
            qcRand.append(op, [qubit_1, qubit_2])

    # from qrisp.interface.converter.PyTket.convert_from_tket import convert_to_tket

    tket_qcRand = qcRand.to_pytket()
    # tket_qcRand = convert_to_tket(qc=qcRand)
    from pytket.extensions.qiskit import AerBackend

    tket_qcRand.measure_all()

    backend = AerBackend()
    if not backend.valid_circuit(tket_qcRand):
        compiled_circ = backend.get_compiled_circuit(tket_qcRand)
        assert backend.valid_circuit(compiled_circ)
    else:
        compiled_circ = backend.get_compiled_circuit(tket_qcRand)
    # compiled_circ = tket_qcRand
    handle = backend.process_circuit(compiled_circ, n_shots=1000)
    result = backend.get_result(handle)
    cnt = result.get_counts()
    d = {}
    for key, value in cnt.items():
        Key = (str(index) for index in key)
        converted2 = "".join(Key)
        # reverse string because of pytket specific return??
        # converted = converted2[::-1]
        converted = converted2
        d.setdefault(converted, value / 1000)

    print(d)

    # print(result.get_counts(basis=BasisOrder.dlo))
    theRes = qvRand.get_measurement()
    print(theRes)

    for index4 in list(d.keys()):
        if index4 not in list(theRes.keys()):
            print("NOT IN THERE")
            print(index4)
        assert index4 in list(theRes.keys())
        if not theRes[index4] * 0.6 <= d[index4] <= theRes[index4] * 1.4:
            print(index4)
            print(theRes[index4])
            print(d[index4])
        if not d[index4] < 0.05:
            assert theRes[index4] * 0.6 <= d[index4] <= theRes[index4] * 1.4


def _matches_up_to_global_phase(actual, expected, atol=1e-6):
    """Return True if two unitaries are equal up to a global phase factor."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    def _strip_phase(unitary):
        flat = unitary.flatten()
        pivot = flat[np.argmax(np.abs(flat))]
        return unitary * np.exp(-1j * np.angle(pivot))

    return np.allclose(_strip_phase(actual), _strip_phase(expected), atol=atol)


def test_to_pytket_u1_angle():
    """Regression for #631: the u1 angle must not be divided by pi twice.

    u1(theta) = diag(1, e^{i*theta}). It maps to a pytket Rz, which differs
    only by a global phase, so the converted unitary must equal diag(1,
    e^{i*theta}) up to a global phase. Before the fix the angle was divided by
    pi a second time, producing the wrong rotation.
    """
    pytest.importorskip("pytket")

    theta = 0.7
    qc = QuantumCircuit(1)
    qc.append(U1Gate(theta), [0])

    unitary = qc.to_pytket().get_unitary()
    expected = np.diag([1, np.exp(1j * theta)])

    assert _matches_up_to_global_phase(unitary, expected)


def test_to_pytket_cp_controlled_phase():
    """Regression for #630: cp must map to controlled-phase, not CRz.

    cp(theta) = diag(1, 1, 1, e^{i*theta}). Before the fix it was emitted as a
    CRz, which splits the phase across the two target states and is a different
    gate. The converted unitary must equal the controlled-phase matrix up to a
    global phase.
    """
    pytest.importorskip("pytket")

    theta = 0.7
    qc = QuantumCircuit(2)
    qc.append(CPGate(theta), [0, 1])

    unitary = qc.to_pytket().get_unitary()
    expected = np.diag([1, 1, 1, np.exp(1j * theta)])

    assert _matches_up_to_global_phase(unitary, expected)


_GATE_THETA = 0.7

# Gate builders whose pytket conversion must reproduce Qrisp's own unitary.
# Comparing to_pytket().get_unitary() against QuantumCircuit.get_unitary() checks
# that the converter preserves each gate's action (up to a global phase).
_SINGLE_QUBIT_GATE_BUILDERS = {
    "h": lambda qc: qc.h(0),
    "x": lambda qc: qc.x(0),
    "y": lambda qc: qc.y(0),
    "z": lambda qc: qc.z(0),
    "s": lambda qc: qc.s(0),
    "s_dg": lambda qc: qc.s_dg(0),
    "t": lambda qc: qc.t(0),
    "t_dg": lambda qc: qc.t_dg(0),
    "sx": lambda qc: qc.append(SXGate(), [0]),
    "rx": lambda qc: qc.rx(_GATE_THETA, 0),
    "ry": lambda qc: qc.ry(_GATE_THETA, 0),
    "rz": lambda qc: qc.rz(_GATE_THETA, 0),
    "p": lambda qc: qc.p(_GATE_THETA, 0),
    "u3": lambda qc: qc.append(U3Gate(0.4, 0.5, 0.6), [0]),
}

_TWO_QUBIT_GATE_BUILDERS = {
    "cx": lambda qc: qc.cx(0, 1),
    "cy": lambda qc: qc.cy(0, 1),
    "cz": lambda qc: qc.cz(0, 1),
    "cp": lambda qc: qc.cp(_GATE_THETA, 0, 1),
    "swap": lambda qc: qc.swap(0, 1),
    "rxx": lambda qc: qc.rxx(_GATE_THETA, 0, 1),
    "rzz": lambda qc: qc.rzz(_GATE_THETA, 0, 1),
}


@pytest.mark.parametrize("gate", sorted(_SINGLE_QUBIT_GATE_BUILDERS))
def test_to_pytket_single_qubit_gate_unitary(gate):
    """to_pytket() reproduces each single-qubit gate's unitary.

    Compares the converted pytket unitary against Qrisp's own get_unitary() up to
    a global phase, giving the converter deterministic per-gate coverage.
    """
    pytest.importorskip("pytket")

    qc = QuantumCircuit(1)
    _SINGLE_QUBIT_GATE_BUILDERS[gate](qc)

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())


@pytest.mark.parametrize("gate", sorted(_TWO_QUBIT_GATE_BUILDERS))
def test_to_pytket_two_qubit_gate_unitary(gate):
    """to_pytket() reproduces each two-qubit gate's unitary.

    Compares the converted pytket unitary against Qrisp's own get_unitary() up to
    a global phase.
    """
    pytest.importorskip("pytket")

    qc = QuantumCircuit(2)
    _TWO_QUBIT_GATE_BUILDERS[gate](qc)

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())
