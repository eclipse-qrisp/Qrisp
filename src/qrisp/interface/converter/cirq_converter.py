"""********************************************************************************
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
"""  # noqa: D205, D208

import numpy as np

try:
    import cirq
except ImportError:
    cirq = None

from qrisp.circuit import ControlledOperation, Operation
from qrisp.circuit import standard_operations as ops
from qrisp.circuit.quantum_circuit import QuantumCircuit


def _transpile_to_known_gates(qrisp_circuit, gate_map):
    """Transpile unknown gates until all gates are in gate_map.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The Qrisp circuit whose unknown gates will be transpiled.
    gate_map : dict
        Mapping of gate names to Cirq gates. Any gate whose name is not a
        key is considered unknown and will be transpiled.

    Returns
    -------
    QuantumCircuit
        A transpiled circuit where every gate name is a key in gate_map.

    Raises
    ------
    ValueError
        If unknown gates cannot be fully decomposed.

    """

    def _unknown_names(circ):
        return {instr.op.name for instr in circ.data if instr.op.name not in gate_map}

    while True:
        unknown = _unknown_names(qrisp_circuit)
        if not unknown:
            break

        def _transpile_predicate(op, _unknown=unknown):
            return op.name in _unknown

        try:
            transpiled = qrisp_circuit.transpile(transpile_predicate=_transpile_predicate)
        except Exception as exc:
            raise ValueError(
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to Cirq converter."
            ) from exc

        new_unknown = _unknown_names(transpiled)
        if new_unknown == unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {names}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_cirq(), or "
                f"use only gates supported natively by the converter."
            )

        qrisp_circuit = transpiled

    return qrisp_circuit


def _ctrl_state_to_cirq_values(ctrl_state):
    """Convert Qrisp ctrl_state string (e.g. '101') to cirq control_values (e.g. [1, 0, 1]).

    Used by convert_to_cirq to pass control values to cirq's base.controlled().
    """
    return [int(c) for c in ctrl_state]


def _cirq_values_to_ctrl_str(control_values, obj_name):
    """Convert cirq control_values (e.g. [[1],[0],[1]]) to Qrisp ctrl_state string (e.g. '101').

    Used by:
    - convert_from_cirq via _unpack_cirq_control_layers to unwrap cirq.ControlledGate layers.
    - convert_from_cirq directly to extract the ctrl_state from a cirq.ControlledOperation's
      control_values and re-apply it as a Qrisp ControlledOperation.
    """
    result = []
    for v in control_values:
        if len(v) != 1:
            raise ValueError(f"Multi-valued control {v} in {obj_name} not supported.")
        val = v[0]
        if val not in (0, 1):
            raise ValueError(f"Unsupported control value {val} in {obj_name}.")
        result.append(str(val))
    return "".join(result)


def _unpack_cirq_control_layers(gate):
    """Unwrap cirq.ControlledGate layers from a gate.

    Used by convert_from_cirq to peel off all cirq.ControlledGate wrappers,
    yielding the inner (non-controlled) gate and an ordered list of control layers
    that were applied outermost-first.
    """
    ctrl_layers = []
    inner_gate = gate
    while isinstance(inner_gate, cirq.ControlledGate):
        ctrl_state = _cirq_values_to_ctrl_str(inner_gate.control_values, inner_gate)
        ctrl_layers.append((inner_gate.num_controls(), ctrl_state))
        inner_gate = inner_gate.sub_gate
    return inner_gate, ctrl_layers


def _apply_ctrl_layers(qrisp_op, ctrl_layers):
    """Re-apply (num_controls, ctrl_state) layers to a Qrisp operation inside-out.

    Used by convert_from_cirq after converting the inner gate: wraps it in
    ControlledOperation layers in the reverse order they were peeled off
    (outermost control layer first).
    """
    for num_ctrl, ctrl_state in reversed(ctrl_layers):
        qrisp_op = ControlledOperation(
            base_operation=qrisp_op,
            num_ctrl_qubits=num_ctrl,
            ctrl_state=ctrl_state,
        )
    return qrisp_op


def convert_to_cirq(qrisp_circuit, cirq_qubits=None):
    """Convert a Qrisp QuantumCircuit to a Cirq Circuit.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The Qrisp QuantumCircuit to convert.
    cirq_qubits : list[cirq.LineQubit], optional
        List of Cirq qubits to map to. If None, LineQubits are created
        automatically. The default is None.

    Returns
    -------
    cirq.Circuit
        A cirq.Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If Cirq is not installed.
    ValueError
        If a gate is not supported by the converter.

    Notes
    -----
    Unknown gates are decomposed by Qrisp's transpiler before conversion.
    The converter transpiles all unknown gates together, then checks if any
    new unknown gates appeared.  This repeats until all gates are known
    or transpilation makes no progress.

    """
    try:
        from cirq import Circuit, LineQubit  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Qrisp to Cirq converter.") from exc

    from cirq import (  # noqa: PLC0415
        CNOT,
        CZ,
        SWAP,
        GlobalPhaseGate,
        H,
        I,
        M,
        R,
        S,
        T,
        X,
        XPowGate,
        Y,
        Z,
        ZPowGate,
        rx,
        ry,
        rz,
    )

    def _build_controlled_base(name, params, cirq_gate):
        """Build the base cirq gate for a ControlledOperation.

        Used by convert_to_cirq to handle Qrisp gates whose cirq equivalent
        differs when used as the base of a controlled gate:
        - sx/sx_dg -> XPowGate with explicit exponent (the gate_map entry
          is the class, not an instance)
        - p -> ZPowGate with exponent from params
        - cx/cz -> X/Z instead of CNOT/CZ (controlled-CX uses X as base)
        """
        known = {
            "sx": XPowGate(exponent=0.5),
            "sx_dg": XPowGate(exponent=-0.5),
            "cx": X,
            "cz": Z,
        }
        if name in known:
            return known[name]
        if name == "p" and params:
            return ZPowGate(exponent=params[0] / np.pi)
        if params:
            return cirq_gate(*params)
        return cirq_gate

    def _build_simple_gate(name, params, cirq_gate):
        """Build the cirq gate for a non-controlled Qrisp instruction.

        Used by convert_to_cirq to handle the same sx/sx_dg/p
        special cases as _build_controlled_base but without the cx/cz
        overrides (non-controlled CNOT/CZ from gate_map are already correct).
        """
        known = {
            "sx": XPowGate(exponent=0.5),
            "sx_dg": XPowGate(exponent=-0.5),
        }
        if name in known:
            return known[name]
        if name == "p":
            return ZPowGate(exponent=params[0] / np.pi) if params else cirq_gate
        if params and name != "id":
            return cirq_gate(*params)
        return cirq_gate

    # known gate mapping
    gate_map = {
        "cx": CNOT,
        "cy": Y,
        "cz": CZ,
        "swap": SWAP,
        "h": H,
        "x": X,
        "y": Y,
        "z": Z,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "s": S,
        "t": T,
        "s_dg": S**-1,
        "t_dg": T**-1,
        "measure": M,
        "reset": R,
        "id": I,
        "p": ZPowGate,
        "sx": XPowGate,
        "sx_dg": XPowGate,
        "gphase": None,
        "xxyy": None,
        "rxx": None,
        "rzz": None,
        # skip qubit allocation and deallocation ops in the converter
        "qb_alloc": None,
        "qb_dealloc": None,
    }

    qrisp_circuit = _transpile_to_known_gates(qrisp_circuit, gate_map)

    cirq_circ = Circuit()

    if cirq_qubits is None:
        cirq_qubits = [LineQubit(i) for i in range(len(qrisp_circuit.qubits))]

    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    for instr in qrisp_circuit.data:
        name = instr.op.name
        qubits = instr.qubits
        params = instr.op.params if hasattr(instr.op, "params") else []

        if name == "gphase":
            cirq_circ.append(GlobalPhaseGate(np.exp(1j * params[0]))())
            continue

        cirq_gate = gate_map[name]
        cirq_op_qubits = [qubit_map[q] for q in qubits]

        if cirq_gate is None:
            if instr.op.definition:
                cirq_circ.append(convert_to_cirq(instr.op.definition, cirq_op_qubits))
                continue
            if name not in ("qb_alloc", "qb_dealloc"):
                raise ValueError(f"{name} gate has no Cirq equivalent and no definition to decompose.")
            continue

        if isinstance(instr.op, ControlledOperation):
            base = _build_controlled_base(name, params, cirq_gate)
            control_values = _ctrl_state_to_cirq_values(instr.op.ctrl_state)
            controlled = base.controlled(num_controls=len(instr.op.ctrl_state), control_values=control_values)
            cirq_circ.append(controlled(*cirq_op_qubits))
            continue

        gate = _build_simple_gate(name, params, cirq_gate)
        cirq_circ.append(gate(*cirq_op_qubits))

    return cirq_circ


def _fractional_h_gate(exp):
    """Build H^t as a custom gate using eigenvalue decomposition."""
    h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    vals, vecs = np.linalg.eigh(h_mat)
    h_pow = vecs @ np.diag([1, np.exp(1j * np.pi * exp)]) @ vecs.conj().T
    op = Operation(name="h", num_qubits=1)
    op.unitary = h_pow
    return op


def _conv_h(inner_gate):
    """Convert cirq.HPowGate to a Qrisp HGate (or fractional power)."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_h_gate(inner_gate.exponent)
    return ops.HGate()


def _fractional_cx_gate(exp):
    """Build CX^t = controlled RX(t*pi)."""
    return ops.RXGate(exp * np.pi).control(1)


def _conv_cx(inner_gate):
    """Convert cirq.CXPowGate to a Qrisp CXGate (or fractional power)."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_cx_gate(inner_gate.exponent)
    return ops.CXGate()


def _fractional_cz_gate(exp):
    """Build CZ^t = CP(t*pi)."""
    return ops.CPGate(exp * np.pi)


def _conv_cz(inner_gate):
    """Convert cirq.CZPowGate to a Qrisp CZGate (or fractional power)."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_cz_gate(inner_gate.exponent)
    return ops.CZGate()


def _fractional_swap_gate(exp):
    """Build SWAP^t as a custom gate using eigenvalue decomposition."""
    swap_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    vals, vecs = np.linalg.eigh(swap_mat)
    vals_t = np.array([np.exp(1j * np.pi * exp), 1.0, 1.0, 1.0])
    swap_pow = (vecs * vals_t) @ vecs.conj().T
    op = Operation(name="swap", num_qubits=2)
    op.unitary = swap_pow
    return op


def _conv_swap(inner_gate):
    """Convert cirq.SwapPowGate to a Qrisp SwapGate (or fractional power)."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_swap_gate(inner_gate.exponent)
    return ops.SwapGate()


def _conv_id(inner_gate):
    """Convert cirq.IdentityGate to a Qrisp IDGate."""
    return ops.IDGate()


def _conv_reset(inner_gate):
    """Convert cirq.ResetChannel to a Qrisp Reset."""
    return ops.Reset()


def _convert_pauli_pow(inner_gate, rotation_cls, rotation_fn, specials, fallback_fn):
    """Convert a Cirq Pauli power gate to a Qrisp operation.

    Handles the common pattern shared by XPowGate, YPowGate, and
    ZPowGate: subclass check for the explicit rotation variant
    (Rx/Ry/Rz), exponent-based dispatch for standard Pauli powers
    (X, SX, S, T, ...), and a generic fallback.

    Parameters
    ----------
    inner_gate : cirq.XPowGate | cirq.YPowGate | cirq.ZPowGate
        The inner (non-controlled) gate to convert.
    rotation_cls : type
        The rotation subclass to check via isinstance (e.g. cirq.Rx).
    rotation_fn : Callable[[float], Operation]
        Factory for the rotation gate (e.g. ops.RXGate).
    specials : list[tuple[float, Callable[[], Operation]]]
        List of (exponent_mod_4, factory) pairs for standard powers.
    fallback_fn : Callable[[float], Operation]
        Factory for the generic fallback (e.g. ops.PGate).

    Returns
    -------
    Operation

    """
    exp = inner_gate.exponent
    if isinstance(inner_gate, rotation_cls):
        return rotation_fn(exp * np.pi)
    for modulus, factory in specials:
        if np.isclose(exp % 4, modulus):
            return factory()
    return fallback_fn(exp * np.pi)


def _conv_x(inner_gate):
    """Convert cirq.XPowGate to X, SX, SX-dagger, or RX."""
    from cirq import Rx  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Rx,
        ops.RXGate,
        [
            (1.0, ops.XGate),
            (0.5, ops.SXGate),
            (3.5, ops.SXDGGate),
        ],
        ops.RXGate,
    )


def _conv_y(inner_gate):
    """Convert cirq.YPowGate to Y or RY."""
    from cirq import Ry  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Ry,
        ops.RYGate,
        [
            (1.0, ops.YGate),
        ],
        ops.RYGate,
    )


def _conv_z(inner_gate):
    """Convert cirq.ZPowGate to Z, S, S-dagger, T, T-dagger, or P."""
    from cirq import Rz  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Rz,
        ops.RZGate,
        [
            (1.0, ops.ZGate),
            (0.5, ops.SGate),
            (3.5, lambda: ops.SGate().inverse()),
            (0.25, ops.TGate),
            (3.75, lambda: ops.TGate().inverse()),
        ],
        ops.PGate,
    )


def _conv_iswap(inner_gate):
    """Convert cirq.ISwapPowGate to a Qrisp iSWAP (or its adjoint)."""
    exp = inner_gate.exponent
    if not np.isclose(exp % 2, 1.0):
        raise ValueError(
            f"Only the full ISwapPowGate "
            f"(exponent 1, -1, 3, ...) can be converted, "
            f"not the fractional-power variant with exponent "
            f"{exp}."
        )
    tmp = QuantumCircuit(2)
    tmp.cx(tmp.qubits[0], tmp.qubits[1])
    tmp.h(tmp.qubits[0])
    tmp.cx(tmp.qubits[1], tmp.qubits[0])
    tmp.s(tmp.qubits[0])
    tmp.cx(tmp.qubits[1], tmp.qubits[0])
    tmp.s_dg(tmp.qubits[0])
    tmp.h(tmp.qubits[0])
    tmp.cx(tmp.qubits[0], tmp.qubits[1])
    op = tmp.to_gate()
    op.name = "iswap"
    if np.isclose(exp % 4, 3.0):
        op = op.inverse()
    return op


def _conv_ccx(inner_gate):
    """Convert cirq.CCXPowGate to a Qrisp Toffoli (MCXGate with 2 controls)."""
    exp = inner_gate.exponent
    if not np.isclose(exp % 2, 1.0):
        raise ValueError(
            f"Only the full CCXPowGate (Toffoli) "
            f"(exponent 1, -1, 3, ...) can be converted, "
            f"not the fractional-power variant with exponent "
            f"{exp}."
        )
    return ops.MCXGate(control_amount=2)


def _conv_ccz(inner_gate):
    """Convert cirq.CCZPowGate to a doubly-controlled Z gate."""
    exp = inner_gate.exponent
    if not np.isclose(exp % 2, 1.0):
        raise ValueError(
            f"Only the full CCZPowGate "
            f"(exponent 1, -1, 3, ...) can be converted, "
            f"not the fractional-power variant with exponent "
            f"{exp}."
        )
    return ops.ZGate().control(2)


_GATE_CONVERTERS = None


def _get_gate_converters():
    """Lazy-initialize and return the cirq-to-Qrisp gate converter registry."""
    global _GATE_CONVERTERS  # noqa: PLW0603
    if _GATE_CONVERTERS is None:
        _GATE_CONVERTERS = [
            (cirq.HPowGate, _conv_h),
            (cirq.CXPowGate, _conv_cx),
            (cirq.CZPowGate, _conv_cz),
            (cirq.SwapPowGate, _conv_swap),
            (cirq.IdentityGate, _conv_id),
            (cirq.ResetChannel, _conv_reset),
            (cirq.XPowGate, _conv_x),
            (cirq.YPowGate, _conv_y),
            (cirq.ZPowGate, _conv_z),
            (cirq.ISwapPowGate, _conv_iswap),
            (cirq.CCXPowGate, _conv_ccx),
            (cirq.CCZPowGate, _conv_ccz),
        ]
    return _GATE_CONVERTERS


def convert_from_cirq(cirq_circuit):  # noqa: PLR0912, PLR0915
    """Convert a Cirq Circuit to a Qrisp QuantumCircuit.

    Parameters
    ----------
    cirq_circuit : cirq.Circuit
        The Cirq Circuit to convert.

    Returns
    -------
    QuantumCircuit
        A Qrisp QuantumCircuit equivalent to the input Cirq circuit.

    Raises
    ------
    ImportError
        If Cirq is not installed.
    ValueError
        If a gate is not supported by the converter.

    Notes
    -----
    Measurement key names from the source Cirq circuit are **not** preserved
    during conversion.  Cirq's automatic key generation is used instead, so
    a round-trip (Qrisp -> Cirq -> Qrisp) will lose the original
    classical-bit associations.

    """
    try:
        import cirq  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Cirq to Qrisp converter.") from exc

    all_qs = cirq_circuit.all_qubits()
    qc = QuantumCircuit(len(all_qs))

    try:
        cirq_qubits = sorted(all_qs)
    except TypeError as exc:
        types = {type(q).__name__ for q in cirq_circuit.all_qubits()}
        raise ValueError(
            f"Mixed qubit types {types} found in the circuit. The converter "
            f"requires all qubits to be of the same type (e.g. all LineQubit)."
        ) from exc
    qubit_map = {q: qc.qubits[i] for i, q in enumerate(cirq_qubits)}

    for op in cirq_circuit.all_operations():
        if isinstance(op, cirq.ControlledOperation):
            sub_op = op.sub_operation
            gate = getattr(sub_op, "gate", None)
            if gate is None:
                raise ValueError(f"Controlled sub-operation {sub_op} is not supported by the Cirq to Qrisp converter.")
            orig_controls = (list(op.controls), list(sub_op.qubits), op.control_values)
        else:
            gate = getattr(op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Operation {op} without gate attribute is not supported by the Cirq to Qrisp converter."
                )
            orig_controls = None

        if isinstance(gate, cirq.GlobalPhaseGate):
            phi = np.angle(gate.coefficient)
            if cirq_qubits:
                qc.append(ops.GPhaseGate(phi), [qubit_map[cirq_qubits[0]]])
            continue

        if isinstance(gate, cirq.MeasurementGate):
            qrisp_qubits = [qubit_map[q] for q in op.qubits]
            if len(qrisp_qubits) == 1:
                qc.measure(qrisp_qubits[0])
            else:
                qc.measure(qrisp_qubits)
            continue

        inner_gate, ctrl_layers = _unpack_cirq_control_layers(gate)

        qrisp_op = None
        for gate_type, converter in _get_gate_converters():
            if isinstance(inner_gate, gate_type):
                qrisp_op = converter(inner_gate)
                break
        if qrisp_op is None:
            raise ValueError(f"Gate {gate} is not supported by the Cirq to Qrisp converter.")

        qrisp_op = _apply_ctrl_layers(qrisp_op, ctrl_layers)

        if orig_controls is not None:
            controls, sub_qubits, cv = orig_controls
            ctrl_state = _cirq_values_to_ctrl_str(cv, op)
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=len(controls),
                ctrl_state=ctrl_state,
            )
            qrisp_qubits = [qubit_map[q] for q in controls + sub_qubits]
        else:
            qrisp_qubits = [qubit_map[q] for q in op.qubits]

        qc.append(qrisp_op, qrisp_qubits)

    return qc
