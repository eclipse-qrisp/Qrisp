# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""This module provides functionalities for converting between Qrisp and PyZX quantum circuits."""

from __future__ import annotations

import copy
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from qrisp import HGate, QuantumCircuit, RYGate, RZGate, SwapGate, SXGate, ZGate, u3Gate

if TYPE_CHECKING:
    from pyzx import Circuit
    from pyzx.circuit.gates import Gate

    from qrisp import Clbit, Operation, Qubit


MIN_PYZX_VERSION = "0.10.1"
_REQUIRED_PYZX_GATES = frozenset({"Y", "CY", "RXX", "RZZ", "Reset", "U2", "U3"})


def _check_pyzx_compatibility(pyzx_module: object, circuit_cls: type, gate_types: dict[str, object]) -> None:
    """Raise a clear ``ImportError`` if the installed PyZX lacks required features.

    Parameters
    ----------
    pyzx_module : object
        The imported ``pyzx`` module.
    circuit_cls : type
        The PyZX ``Circuit`` class.
    gate_types : dict[str, object]
        Mapping of PyZX gate names to gate classes.

    Raises
    ------
    ImportError
        If required gate types or ``Circuit.bits`` are unavailable.

    """
    version = getattr(pyzx_module, "__version__", "unknown")
    missing = sorted(_REQUIRED_PYZX_GATES.difference(gate_types))
    if not missing and hasattr(circuit_cls(0), "bits"):
        return

    details = f"; missing gate types: {', '.join(missing)}" if missing else ""
    raise ImportError(f"PyZX >= {MIN_PYZX_VERSION} is required by the Qrisp PyZX converter (found {version}{details}).")


def _import_pyzx() -> tuple[object, type, dict[str, object]]:
    """Import PyZX, verify compatibility, and return its module, Circuit, and gate types.

    Returns
    -------
    tuple[object, type, dict[str, object]]
        The ``pyzx`` module, the ``Circuit`` class, and the gate-type mapping.

    Raises
    ------
    ImportError
        If PyZX is not installed or lacks required features.

    """
    try:
        import pyzx
        from pyzx import Circuit
        from pyzx.circuit import gates as pyzx_gates
    except ImportError as exc:
        raise ImportError("PyZX must be installed to be able to use the PyZX converter.") from exc

    _check_pyzx_compatibility(pyzx, Circuit, pyzx_gates.gate_types)
    return pyzx, Circuit, pyzx_gates.gate_types


def _to_float(value: object, description: str = "parameter") -> float:
    """Convert a numeric value to float, raising a clear error for symbolic values.

    Parameters
    ----------
    value : object
        The value to convert.
    description : str, optional
        A human-readable description used in the error message. The default is
        ``"parameter"``.

    Returns
    -------
    float
        The converted value.

    Raises
    ------
    ValueError
        If ``value`` is symbolic or otherwise not convertible to ``float``.

    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert symbolic {description} {value!r} to a float. Bind symbolic parameters before converting."
        ) from exc


def _phase(angle: float, max_denominator: int = 1024, tol: float = 1e-9) -> Fraction:
    """Convert an angle in radians to a PyZX phase in units of pi.

    PyZX simplification rules key on small rational phases such as ``1/4`` or
    ``1/2``. Since most angles are not exactly representable as binary floats,
    the result is snapped to a small denominator whenever the approximation is
    within ``tol``. Angles that are not close to such a rational are kept as
    their exact float value.

    Parameters
    ----------
    angle : float
        The angle in radians.
    max_denominator : int, optional
        The largest denominator allowed when snapping to a rational phase.
        The default is 1024.
    tol : float, optional
        The maximum absolute error (in units of pi) for accepting a snapped
        fraction. The default is 1e-9.

    Returns
    -------
    Fraction
        The angle expressed in units of pi.

    Examples
    --------
    >>> _phase(np.pi / 3)
    Fraction(1, 3)
    >>> _phase(np.pi / 4) == Fraction(1, 4)
    True

    """
    x = _to_float(angle, "phase") / np.pi
    fraction = Fraction(x).limit_denominator(max_denominator)
    return fraction if abs(float(fraction) - x) < tol else Fraction(x)


def _ungadget(gate: Gate) -> Gate:
    """Return a copy of a phase gadget with its gadget flag disabled.

    A copy is made so that converting a PyZX circuit does not mutate the
    caller's gate objects. Gates that are not gadgets are returned unchanged.

    Parameters
    ----------
    gate : pyzx.circuit.gates.Gate
        The gate to inspect.

    Returns
    -------
    pyzx.circuit.gates.Gate
        The original gate, or an un-gadgetted copy.

    """
    if not getattr(gate, "as_gadget", False):
        return gate
    result = copy.copy(gate)
    result.as_gadget = False
    return result


def _measurement_clbit(qc: QuantumCircuit, result_bit: int | None) -> Clbit:
    """Return the Qrisp clbit for a PyZX measurement destination.

    A measurement without a destination is assigned a freshly allocated
    classical bit. An explicit destination that exceeds the circuit's classical
    bit count raises a ``ValueError``.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp circuit that receives the measurement.
    result_bit : int or None
        The measurement's destination as reported by PyZX.

    Returns
    -------
    qrisp.Clbit
        The classical bit the measurement should write to.

    Raises
    ------
    ValueError
        If ``result_bit`` is an index beyond the circuit's classical bits.

    """
    if result_bit is None:
        return qc.add_clbit()
    if result_bit >= len(qc.clbits):
        raise ValueError(
            f"Measurement targets result bit {result_bit}, but the PyZX circuit "
            f"only declares {len(qc.clbits)} classical bits."
        )
    return qc.clbits[result_bit]


def _circuit_signature(circuit: QuantumCircuit) -> tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]:
    """Return a comparable signature of a circuit's instructions."""
    return tuple(
        (
            instr.op.name,
            repr(getattr(instr.op, "params", ())),
            tuple(qubit.identifier for qubit in instr.qubits),
            tuple(clbit.identifier for clbit in instr.clbits),
        )
        for instr in circuit.data
    )


def _transpile(
    qrisp_circuit: QuantumCircuit,
    gate_map: dict[str, str | Callable[..., None]],
    max_iterations: int = 100,
) -> QuantumCircuit:
    """Transpile a circuit until only operations known to ``gate_map`` remain.

    Unknown operations are repeatedly transpiled until no further progress can
    be made. The code was adapted from the Cirq converter.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The circuit to transpile.
    gate_map : dict[str, str | Callable[..., None]]
        Mapping from supported Qrisp operation names to their PyZX gate names.
        Entries without a direct PyZX equivalent may map to a handler callable.
    max_iterations : int, optional
        The maximum number of transpilation rounds before giving up. The
        default is 100.

    Returns
    -------
    QuantumCircuit
        A circuit containing only operations whose names occur in ``gate_map``.

    Raises
    ------
    ValueError
        If an unknown operation cannot be transpiled, if transpilation makes no
        further progress, or if ``max_iterations`` is exceeded.

    """

    def _unknown_names(circuit: QuantumCircuit) -> set[str]:
        """Return the names of all operations that are not keys of ``gate_map``."""
        return {instr.op.name for instr in circuit.data if instr.op.name not in gate_map}

    for _ in range(max_iterations):
        unknown = _unknown_names(qrisp_circuit)
        if not unknown:
            return qrisp_circuit

        def _transpile_predicate(op: Operation, _unknown: set[str] = unknown) -> bool:
            """Return whether ``op`` belongs to the set of unknown operations."""
            return op.name in _unknown

        try:
            transpiled = qrisp_circuit.transpile(transpile_predicate=_transpile_predicate)
        except (KeyError, TypeError, ValueError, AttributeError, RuntimeError, NotImplementedError) as exc:
            raise ValueError(
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to PyZX converter."
            ) from exc

        if _circuit_signature(transpiled) == _circuit_signature(qrisp_circuit):
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {names}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_pyzx(), or "
                f"use only gates supported natively by the converter."
            )

        qrisp_circuit = transpiled

    if not _unknown_names(qrisp_circuit):
        return qrisp_circuit

    names = ", ".join(sorted(_unknown_names(qrisp_circuit)))
    raise ValueError(
        f"Transpilation did not converge after {max_iterations} iterations. Remaining unsupported gates: {names}."
    )


def _check_ctrl_state(op: Operation) -> None:
    """Raise if a controlled operation uses non-default (``|0>``) controls.

    Qrisp keeps the operation name (for example ``"cx"``) when a control is
    inverted via ``ctrl_state``. Since PyZX has no equivalent of an inverted
    control, such operations must be rejected instead of silently converted as
    if they triggered on ``|1>``.

    Parameters
    ----------
    op : qrisp.Operation
        The operation to inspect.

    Raises
    ------
    ValueError
        If the operation has a control state other than all ones.

    """
    ctrl_state = getattr(op, "ctrl_state", None)
    controls = getattr(op, "controls", None)
    if ctrl_state is None or controls is None:
        return

    if isinstance(ctrl_state, int):
        is_all_ones = ctrl_state in (-1, (1 << len(controls)) - 1)
    else:
        state = str(ctrl_state)
        is_all_ones = len(state) == len(controls) and set(state) == {"1"}

    if not is_all_ones:
        raise ValueError(
            f"Controlled operation {op.name!r} with ctrl_state={ctrl_state!r} is not supported. "
            "Only controls that trigger on |1> are supported by the PyZX converter."
        )


def _skip_operation(pyzx_circuit: Circuit, qubits: list[int], params: list[float]) -> None:
    """Drop an operation that has no PyZX representation."""
    return None


def _emit_s_dg(pyzx_circuit: Circuit, qubits: list[int], params: list[float]) -> None:
    """Emit the adjoint of S as a PyZX ZPhase gate."""
    pyzx_circuit.add_gate("ZPhase", qubits[0], _phase(-np.pi / 2))


def _emit_t_dg(pyzx_circuit: Circuit, qubits: list[int], params: list[float]) -> None:
    """Emit the adjoint of T as a PyZX ZPhase gate."""
    pyzx_circuit.add_gate("ZPhase", qubits[0], _phase(-np.pi / 4))


def _emit_p(pyzx_circuit: Circuit, qubits: list[int], params: list[float]) -> None:
    """Emit Qrisp's phase gate as a PyZX ZPhase gate."""
    pyzx_circuit.add_gate("ZPhase", qubits[0], _phase(params[0]))


def _emit_sx_dg(pyzx_circuit: Circuit, qubits: list[int], params: list[float]) -> None:
    """Emit the adjoint of SX as a PyZX XPhase gate."""
    pyzx_circuit.add_gate("XPhase", qubits[0], _phase(-np.pi / 2))


def convert_to_pyzx(qrisp_circuit: QuantumCircuit) -> Circuit:
    """Convert a Qrisp QuantumCircuit to a PyZX Circuit.

    Parameters
    ----------
    qrisp_circuit : qrisp.QuantumCircuit
        The Qrisp QuantumCircuit to convert.

    Returns
    -------
    pyzx.Circuit
        A pyzx.Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If PyZX is not installed or older than ``0.10.1``.
    ValueError
        If a gate is not supported by the converter or a parameter is symbolic.

    Notes
    -----
    Requires PyZX >= 0.10.1.

    Unknown gates are decomposed by Qrisp's transpiler before conversion.
    The converter transpiles all unknown gates together, then checks if any
    new unknown gates appeared.  This repeats until all gates are known
    or transpilation makes no progress.

    Because PyZX does not track global phases, global-phase gates (``gphase``)
    are dropped. Conversions are therefore only equivalent up to global phase.
    Symbolic parameters are not supported and must be bound before conversion.

    Examples
    --------
    Convert a Qrisp Bell-state circuit to PyZX and back:

    >>> from qrisp import QuantumCircuit, Qubit
    >>> from qrisp.interface import convert_from_pyzx, convert_to_pyzx
    >>> qc = QuantumCircuit()
    >>> _ = qc.add_qubit(Qubit("q0"))
    >>> _ = qc.add_qubit(Qubit("q1"))
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> print(qc)  # doctest: +NORMALIZE_WHITESPACE
        ┌───┐
    q0: ┤ H ├──■──
        └───┘┌─┴─┐
    q1: ─────┤ X ├
             └───┘
    >>> pyzx_circuit = convert_to_pyzx(qc)
    >>> print(pyzx_circuit)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> [(gate.name, gate.target) for gate in pyzx_circuit.gates]
    [('HAD', 0), ('CNOT', 1)]
    >>> qc_roundtrip = convert_from_pyzx(pyzx_circuit)
    >>> [(ins.op.name, [qc_roundtrip.qubits.index(q) for q in ins.qubits]) for ins in qc_roundtrip.data]
    [('h', [0]), ('cx', [0, 1])]

    Measurement destinations are preserved as PyZX result bits:

    >>> qc = QuantumCircuit(1, 1)
    >>> qc.measure(0, 0)
    >>> convert_to_pyzx(qc).gates[0].result_bit
    0

    Gates without a native PyZX counterpart are transpiled automatically:

    >>> qc = QuantumCircuit(2)
    >>> qc.ryy(0.1, 0, 1)
    >>> len(convert_to_pyzx(qc).gates) > 0
    True

    """
    _, Circuit, _ = _import_pyzx()

    gate_map: dict[str, str | Callable[..., None]] = {
        "cx": "CNOT",
        "cy": "CY",
        "cz": "CZ",
        "cp": "CPhase",
        "swap": "SWAP",
        "2cx": "TOF",
        "2cz": "CCZ",
        "ch": "CHAD",
        "h": "HAD",
        "x": "NOT",
        "y": "Y",
        "z": "Z",
        "rx": "XPhase",
        "ry": "YPhase",
        "rz": "ZPhase",
        "u1": "ZPhase",
        "u3": "U3",
        "prx": "U3",
        "s": "S",
        "t": "T",
        "sx": "SX",
        "rxx": "RXX",
        "rzz": "RZZ",
        "measure": "Measurement",
        "reset": "Reset",
        # Gates without a direct PyZX equivalent are handled by these callables.
        "s_dg": _emit_s_dg,
        "t_dg": _emit_t_dg,
        "p": _emit_p,
        "sx_dg": _emit_sx_dg,
        # Skip allocation/deallocation, identity, global phase, and barriers.
        "qb_alloc": _skip_operation,
        "qb_dealloc": _skip_operation,
        "gphase": _skip_operation,
        "id": _skip_operation,
        "barrier": _skip_operation,
    }

    qrisp_circuit = _transpile(qrisp_circuit, gate_map)

    num_qubits = qrisp_circuit.num_qubits()
    num_clbits = len(qrisp_circuit.clbits)
    pyzx_circuit = Circuit(qubit_amount=num_qubits, bit_amount=num_clbits)

    qubit_map: dict[Qubit, int] = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = i
    clbit_map: dict[Clbit, int] = {}
    for i, c in enumerate(qrisp_circuit.clbits):
        clbit_map[c] = i

    for instr in qrisp_circuit.data:
        _check_ctrl_state(instr.op)

        name = instr.op.name
        params = instr.op.params if hasattr(instr.op, "params") else []
        pyzx_op_qubits = [qubit_map[qubit] for qubit in instr.qubits]

        mapping = gate_map[name]
        if not isinstance(mapping, str):
            mapping(pyzx_circuit, pyzx_op_qubits, params)
            continue

        if name == "measure" and instr.clbits:
            pyzx_circuit.add_gate("Measurement", pyzx_op_qubits[0], clbit_map[instr.clbits[0]])
        elif params:
            pyzx_circuit.add_gate(mapping, *pyzx_op_qubits, *[_phase(p) for p in params])
        else:
            pyzx_circuit.add_gate(mapping, *pyzx_op_qubits)

    return pyzx_circuit


def _append_controlled_sx(qc: QuantumCircuit, x: Qubit, y: Qubit) -> None:
    """Append a controlled SX gate acting on ``x`` (control) and ``y`` (target)."""
    qc.append(SXGate().control(), [x, y])


def _handle_measurement(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a measurement, allocating a clbit when PyZX has no destination."""
    method(gate.target, _measurement_clbit(qc, gate.result_bit))


def _handle_single_qubit_parameter_free(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a parameter-free single-qubit gate or non-unitary operation."""
    method(gate.target)


def _handle_single_qubit_with_adjoint_version(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a single-qubit gate, honoring its ``adjoint`` flag."""
    method(gate.adjoint, gate.target)


def _handle_single_qubit_one_parameter(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a single-qubit gate with one phase parameter (in units of pi)."""
    method(_to_float(gate.phase, "phase") * np.pi, gate.target)


def _handle_single_qubit_multi_parameter(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a single-qubit gate with multiple phase parameters (in units of pi)."""
    method(*[_to_float(phase, "phase") * np.pi for phase in gate.phases], gate.target)


def _handle_two_qubit_parameter_free(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a parameter-free two-qubit gate."""
    method(gate.control, gate.target)


def _handle_two_qubit_one_parameter(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a two-qubit gate with one phase parameter (in units of pi)."""
    method(_to_float(gate.phase, "phase") * np.pi, gate.control, gate.target)


def _handle_two_qubit_multi_parameter(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a two-qubit gate with multiple phase parameters (in units of pi)."""
    method(*[_to_float(phase, "phase") * np.pi for phase in gate.phases], gate.control, gate.target)


def _handle_multi_qubit_parameter_free(qc: QuantumCircuit, method: Callable[..., None], gate: Gate) -> None:
    """Append a parameter-free three-qubit gate."""
    method(gate.ctrl1, gate.ctrl2, gate.target)


def convert_from_pyzx(pyzx_circuit: Circuit) -> QuantumCircuit:
    """Convert a PyZX Circuit to a Qrisp QuantumCircuit.

    Parameters
    ----------
    pyzx_circuit : pyzx.Circuit
        The PyZX Circuit to convert.

    Returns
    -------
    qrisp.QuantumCircuit
        A qrisp.QuantumCircuit equivalent to the input PyZX circuit.

    Raises
    ------
    ImportError
        If PyZX is not installed or older than ``0.10.1``.
    ValueError
        If a gate is not supported by the converter or a parameter is symbolic.

    Notes
    -----
    Requires PyZX >= 0.10.1.

    Gates that exist in PyZX but not in Qrisp are either substituted directly if they
    have a straightforward equivalent (which applies to U2 and several controlled
    gates), or PyZX's ``to_basic_gates()`` method is used to decompose them.

    Because PyZX does not track global phases, round-trips are only equivalent
    up to global phase. Symbolic PyZX phases are not supported and must be
    replaced by numeric values before conversion.

    Examples
    --------
    Convert a PyZX Bell-state circuit to Qrisp and back:

    >>> from pyzx import Circuit
    >>> from qrisp.interface import convert_from_pyzx, convert_to_pyzx
    >>> pyzx_circuit = Circuit(2)
    >>> pyzx_circuit.add_gate("HAD", 0)
    >>> pyzx_circuit.add_gate("CNOT", 0, 1)
    >>> print(pyzx_circuit)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> qc = convert_from_pyzx(pyzx_circuit)
    >>> for i, qubit in enumerate(qc.qubits):
    ...     qubit.identifier = f"q{i}"
    >>> print(qc)  # doctest: +NORMALIZE_WHITESPACE
        ┌───┐
    q0: ┤ H ├──■──
        └───┘┌─┴─┐
    q1: ─────┤ X ├
             └───┘
    >>> pyzx_roundtrip = convert_to_pyzx(qc)
    >>> print(pyzx_roundtrip)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> [(gate.name, gate.target) for gate in pyzx_roundtrip.gates]
    [('HAD', 0), ('CNOT', 1)]

    Measurement result bits are mapped to the corresponding classical bits:

    >>> pyzx_circuit = Circuit(1, bit_amount=1)
    >>> pyzx_circuit.add_gate("Measurement", 0, result_bit=0)
    >>> qc = convert_from_pyzx(pyzx_circuit)
    >>> [(ins.op.name, qc.clbits.index(ins.clbits[0])) for ins in qc.data]
    [('measure', 0)]

    Gates without a direct Qrisp equivalent are decomposed by PyZX first:

    >>> from fractions import Fraction
    >>> pyzx_circuit = Circuit(3)
    >>> pyzx_circuit.add_gate("ParityPhase", Fraction(1, 2), 0, 1, 2)
    >>> [ins.op.name for ins in convert_from_pyzx(pyzx_circuit).data]
    ['cx', 'cx', 'rz', 'cx', 'cx']

    """
    _import_pyzx()
    qc = QuantumCircuit(pyzx_circuit.qubits, pyzx_circuit.bits)

    gate_table: dict[str, tuple[Callable[..., None] | None, Callable[..., None] | None]] = {
        # single-qubit gates
        "NOT": (qc.x, _handle_single_qubit_parameter_free),
        "Y": (qc.y, _handle_single_qubit_parameter_free),
        "Z": (qc.z, _handle_single_qubit_parameter_free),
        "HAD": (qc.h, _handle_single_qubit_parameter_free),
        "XPhase": (qc.rx, _handle_single_qubit_one_parameter),
        "YPhase": (qc.ry, _handle_single_qubit_one_parameter),
        "ZPhase": (qc.rz, _handle_single_qubit_one_parameter),
        "U2": (partial(qc.u3, np.pi / 2), _handle_single_qubit_multi_parameter),
        "U3": (qc.u3, _handle_single_qubit_multi_parameter),
        "SX": (lambda adj, x: qc.sx_dg(x) if adj else qc.sx(x), _handle_single_qubit_with_adjoint_version),
        "S": (lambda adj, x: qc.s_dg(x) if adj else qc.s(x), _handle_single_qubit_with_adjoint_version),
        "T": (lambda adj, x: qc.t_dg(x) if adj else qc.t(x), _handle_single_qubit_with_adjoint_version),
        # multi-qubit gates
        "CNOT": (qc.cx, _handle_two_qubit_parameter_free),
        "CY": (qc.cy, _handle_two_qubit_parameter_free),
        "CZ": (qc.cz, _handle_two_qubit_parameter_free),
        "CRX": (qc.crx, _handle_two_qubit_one_parameter),
        "CRY": (lambda phase, x, y: qc.append(RYGate(phase).control(), [x, y]), _handle_two_qubit_one_parameter),
        "CRZ": (lambda phase, x, y: qc.append(RZGate(phase).control(), [x, y]), _handle_two_qubit_one_parameter),
        "CSX": (partial(_append_controlled_sx, qc), _handle_two_qubit_parameter_free),
        "CPhase": (qc.cp, _handle_two_qubit_one_parameter),
        "ParityPhase": (None, None),
        "XCX": (None, None),
        "SWAP": (qc.swap, _handle_two_qubit_parameter_free),
        "CSWAP": (
            lambda x, y, z: qc.append(SwapGate().control(), [x, y, z]),
            _handle_multi_qubit_parameter_free,
        ),
        "CHAD": (lambda x, y: qc.append(HGate().control(), [x, y]), _handle_two_qubit_parameter_free),
        "Tof": (qc.ccx, _handle_multi_qubit_parameter_free),
        "CCZ": (lambda x, y, z: qc.append(ZGate().control(2), [x, y, z]), _handle_multi_qubit_parameter_free),
        "CU3": (
            lambda theta, phi, lam, x, y: qc.append(u3Gate(theta, phi, lam).control(), [x, y]),
            _handle_two_qubit_multi_parameter,
        ),
        "CU": (None, None),
        "RZZ": (qc.rzz, _handle_two_qubit_one_parameter),
        "RXX": (qc.rxx, _handle_two_qubit_one_parameter),
        "FSim": (None, None),
        # non-unitary operations
        "Measurement": (qc.measure, _handle_measurement),
        "Reset": (qc.reset, _handle_single_qubit_parameter_free),
        "InitAncilla": (None, None),
        "PostSelect": (None, None),
        "DiscardBit": (None, None),
        "ConditionalGate": (None, None),
    }

    for gate in pyzx_circuit.gates:
        entry = gate_table.get(gate.name)
        if entry is None:
            raise ValueError(f"{gate.name} of PyZX is unknown and thus cannot be converted to Qrisp.")

        method, handler = entry
        if method is not None and handler is not None:
            handler(qc, method, gate)
            continue

        # Try with PyZX's basic gate decomposition.
        for _gate in _ungadget(gate).to_basic_gates():
            sub_entry = gate_table.get(_gate.name)
            if sub_entry is None or sub_entry[0] is None or sub_entry[1] is None:
                raise ValueError(f"{_gate.name} gate has no Qrisp equivalent and cannot be decomposed either.")
            sub_method, sub_handler = sub_entry
            sub_handler(qc, sub_method, _gate)

    return qc
