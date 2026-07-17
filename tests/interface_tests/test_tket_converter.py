# from qrisp.circuit.standard_operations import op_list
from unittest.mock import patch

import numpy as np
import pytest

np.random.seed(42)  # Deterministic for reproducible test results

from qrisp import QuantumCircuit, QuantumVariable
from qrisp.circuit import Operation
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
from qrisp.interface.converter.pytket_converter import (
    create_tket_instruction,
    pytket_converter,
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
        converted = "".join(str(index) for index in key)
        d.setdefault(converted, value / 1000)

    theRes = qvRand.get_measurement()

    for index4 in list(d.keys()):
        assert index4 in list(theRes.keys())
        # Ignore low-probability outcomes that are dominated by shot noise.
        if not d[index4] < 0.05:
            assert theRes[index4] * 0.6 <= d[index4] <= theRes[index4] * 1.4


def _matches_up_to_global_phase(actual, expected, atol=1e-6):
    """Return True if two unitaries are equal up to a global phase factor.

    The global phase is recovered from the Frobenius inner product of the two
    matrices. This is robust even when several entries share the largest
    magnitude: normalising against a single pivot entry is not, because
    floating-point noise can make ``argmax`` select different entries for the
    two unitaries, so they end up referenced to different phases.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    overlap = np.vdot(expected, actual)
    if np.abs(overlap) < atol:
        return False

    phase = overlap / np.abs(overlap)
    return np.allclose(actual, expected * phase, atol=atol)


_GATE_THETA = 0.7

# Regression cases for the two mapping fixes. Each converted unitary must equal
# the gate's textbook matrix up to a global phase, independently of Qrisp's own
# get_unitary().
_REGRESSION_CASES = {
    # #631: the u1 angle must not be divided by pi twice. u1(theta) maps to a
    # pytket Rz, which differs only by a global phase, so the converted unitary
    # must equal diag(1, e^{i*theta}).
    "u1": (
        1,
        lambda qc: qc.append(U1Gate(_GATE_THETA), [0]),
        np.diag([1, np.exp(1j * _GATE_THETA)]),
    ),
    # #630: cp must map to controlled-phase, not CRz. cp(theta) is
    # diag(1, 1, 1, e^{i*theta}); CRz splits the phase across the two target
    # states and is a different gate.
    "cp": (
        2,
        lambda qc: qc.append(CPGate(_GATE_THETA), [0, 1]),
        np.diag([1, 1, 1, np.exp(1j * _GATE_THETA)]),
    ),
}


@pytest.mark.parametrize("gate", sorted(_REGRESSION_CASES))
def test_to_pytket_regression(gate):
    """Regression tests for the #630 (cp) and #631 (u1) gate-mapping fixes."""
    pytest.importorskip("pytket")

    num_qubits, build, expected = _REGRESSION_CASES[gate]
    qc = QuantumCircuit(num_qubits)
    build(qc)

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), expected)


# Gate builders whose pytket conversion must reproduce Qrisp's own unitary.
# Comparing to_pytket().get_unitary() against QuantumCircuit.get_unitary() checks
# that the converter preserves each gate's action (up to a global phase).
_SINGLE_QUBIT_GATE_BUILDERS = {
    "h": lambda qc: qc.h(0),
    "x": lambda qc: qc.x(0),
    "y": lambda qc: qc.y(0),
    "z": lambda qc: qc.z(0),
    "id": lambda qc: qc.id(0),
    "s": lambda qc: qc.s(0),
    "s_dg": lambda qc: qc.s_dg(0),
    "t": lambda qc: qc.t(0),
    "t_dg": lambda qc: qc.t_dg(0),
    "sx": lambda qc: qc.append(SXGate(), [0]),
    "sx_dg": lambda qc: qc.sx_dg(0),
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


def test_to_pytket_measure():
    """to_pytket() maps measurements to pytket Measure ops, preserving wiring.

    Measurement is not unitary, so this checks the converted circuit
    structurally: every qubit is measured into the classical bit of the same
    index.
    """
    pytest.importorskip("pytket")
    from pytket import OpType

    num_qubits = 2
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    tket_qc = qc.to_pytket()
    measurements = [cmd for cmd in tket_qc.get_commands() if cmd.op.type == OpType.Measure]

    assert tket_qc.n_bits == num_qubits
    assert len(measurements) == num_qubits
    # Each measurement wires qubit k to the classical bit of the same index.
    expected_wiring = {(k, k) for k in range(num_qubits)}
    assert {(cmd.qubits[0].index[0], cmd.bits[0].index[0]) for cmd in measurements} == expected_wiring


def test_to_pytket_gphase():
    """to_pytket() preserves a global-phase gate exactly.

    ``x, x`` is the identity, so ``gphase(theta)`` leaves the unitary as
    ``e^{i*theta} * I``. Global phase is unobservable up to a phase factor, so
    this compares the converted unitary against Qrisp's own get_unitary()
    *exactly* to prove the phase itself is carried across.
    """
    pytest.importorskip("pytket")

    theta = 0.5
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.gphase(theta, 0)

    np.testing.assert_array_almost_equal(qc.to_pytket().get_unitary(), qc.get_unitary())


# Multi-controlled gates route through the ControlledOperation branch and are
# emitted as abstract CircBoxes. MCX (base name "x") and MCRX (base name "rx")
# also exercise both sides of the single-character base-name check.
_CONTROLLED_GATE_BUILDERS = {
    "mcx": (4, lambda qc: qc.append(MCXGate(control_amount=3), [0, 1, 2, 3])),
    "mcrx": (3, lambda qc: qc.append(MCRXGate(_GATE_THETA, control_amount=2), [0, 1, 2])),
}


@pytest.mark.parametrize("gate", sorted(_CONTROLLED_GATE_BUILDERS))
def test_to_pytket_controlled_gate_unitary(gate):
    """to_pytket() reproduces multi-controlled gates (ControlledOperation path).

    Each is emitted as an abstract CircBox of its definition; the converted
    unitary must match Qrisp's own up to a global phase.
    """
    pytest.importorskip("pytket")

    num_qubits, build = _CONTROLLED_GATE_BUILDERS[gate]
    qc = QuantumCircuit(num_qubits)
    build(qc)

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())


def test_to_pytket_composite_gate():
    """to_pytket() reproduces a composite (non-elementary) gate's unitary.

    A gate built from a sub-circuit has a ``definition`` but no elementary pytket
    equivalent, so it exercises the CircBox path in ``create_tket_instruction``.
    """
    pytest.importorskip("pytket")

    sub = QuantumCircuit(2)
    sub.h(0)
    sub.cx(0, 1)

    qc = QuantumCircuit(2)
    qc.append(sub.to_gate(name="myblock"), [0, 1])

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())


def test_to_pytket_grover():
    """to_pytket() reproduces a full Grover search circuit.

    This is the converter's end-to-end regression: a compiled Grover session mixes
    composite/controlled gates (mapped to CircBoxes), qubit (de)allocation, and the
    diffuser's global-phase gate. The instance is deliberately small
    (``QuantumFloat(2, -1)`` -> 9 qubits) so it stays within pytket's statevector
    simulation limit. The converted statevector must match Qrisp's own up to a
    global phase.
    """
    pytest.importorskip("pytket")

    from math import pi

    from qrisp.grover import diffuser

    from qrisp import QuantumFloat, auto_uncompute, h, z

    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = qf * qf == 0.25
        z(temp_qbool)

    qf = QuantumFloat(2, -1, signed=True)
    n = qf.size
    iterations = max(1, int(0.25 * pi * (2**n / 2) ** 0.5))

    h(qf)
    for _ in range(iterations):
        sqrt_oracle(qf)
        diffuser(qf)

    qc = qf.qs.compile()

    assert _matches_up_to_global_phase(qc.to_pytket().get_statevector(), qc.statevector_array())


def test_create_tket_instruction_unknown_raises():
    """create_tket_instruction() raises for an op with no pytket equivalent.

    An operation whose name is not in the gate map and that has no ``definition``
    cannot be converted, hitting the final ``else`` branch.
    """
    pytest.importorskip("pytket")

    op = Operation(name="totally_unknown", num_qubits=1)

    with pytest.raises(Exception, match="Could not convert"):
        create_tket_instruction(op)


def _make_import_fail():
    """Patch ``__import__`` so importing pytket fails, simulating its absence."""
    import builtins

    real = builtins.__import__

    def mock(name, *args, **kwargs):
        if name == "pytket" or name.startswith("pytket."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real(name, *args, **kwargs)

    return patch("builtins.__import__", mock)


def test_pytket_converter_import_error():
    """pytket_converter() raises a clear ImportError when pytket is missing."""
    with _make_import_fail():
        with pytest.raises(ImportError, match="PyTket must be installed"):
            pytket_converter(QuantumCircuit(1))


def test_create_tket_instruction_import_error():
    """create_tket_instruction() raises a clear ImportError when pytket is missing."""
    with _make_import_fail():
        with pytest.raises(ImportError, match="PyTket must be installed"):
            create_tket_instruction(XGate())
