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

from qrisp.circuit import ControlledOperation
import numpy as np


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
        from cirq import Circuit, LineQubit
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Qrisp to Cirq converter.") from exc

    from cirq import (
        CNOT,
        H,
        X,
        Y,
        Z,
        S,
        T,
        SWAP,
        rx,
        ry,
        rz,
        I,
        M,
        R,
        CZ,
        ZPowGate,
        XPowGate,
        GlobalPhaseGate,
    )

    # known gate mapping
    gate_map = {
        "cx": CNOT,
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

    # repeatedly transpile unknown gates until only known ones remain
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

    # create an empty Cirq circuit
    cirq_circ = Circuit()

    if cirq_qubits is None:
        cirq_qubits = [LineQubit(i) for i in range(len(qrisp_circuit.qubits))]

    # create a mapping of Qrisp qubits to Cirq qubits
    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    for instr in qrisp_circuit.data:
        name = instr.op.name
        qubits = instr.qubits
        params = instr.op.params if hasattr(instr.op, "params") else []

        # global phase
        if name == "gphase":
            cirq_circ.append(GlobalPhaseGate(np.exp(1j * params[0]))())
            continue

        cirq_gate = gate_map[name]
        cirq_op_qubits = [qubit_map[q] for q in qubits]

        # gate with no direct Cirq equivalent
        if cirq_gate is None:
            # decompose via its .definition circuit (e.g. xxyy, rxx, rzz)
            if instr.op.definition:
                cirq_circ.append(convert_to_cirq(instr.op.definition, cirq_op_qubits))
                continue
            # qb_alloc/qb_dealloc are bookkeeping ops with no circuit effect
            if name not in ("qb_alloc", "qb_dealloc"):
                raise ValueError(f"{name} gate has no Cirq equivalent and no definition to decompose.")
            continue

        # controlled operations (multi-qubit)
        if isinstance(instr.op, ControlledOperation):
            cs = instr.op.ctrl_state
            n_ctrl = len(cs)
            control_qb = qubits[:n_ctrl]
            target_qb = qubits[n_ctrl:]
            cirq_ctrl = [qubit_map[q] for q in control_qb]
            cirq_target = [qubit_map[q] for q in target_qb]

            # Build the base gate.  cx/cz are unwrapped to X/Z because
            # CNOT.controlled() would produce CCNOT instead of CNOT.
            if name == "sx":
                base = XPowGate(exponent=0.5)
            elif name == "sx_dg":
                base = XPowGate(exponent=-0.5)
            elif name == "p" and params:
                base = ZPowGate(exponent=params[0] / np.pi)
            elif name == "cx":
                base = X
            elif name == "cz":
                base = Z
            elif params:
                base = cirq_gate(*params)
            else:
                base = cirq_gate

            # Convert ctrl_state string ("101") to Cirq control_values list
            ctrl_vals = [int(c) for c in cs]

            controlled = base.controlled(num_controls=n_ctrl, control_values=ctrl_vals)
            cirq_circ.append(controlled(*cirq_ctrl, *cirq_target))
            continue

        # sx / sx_dg (non-controlled path; controlled is handled above)
        if name == "sx":
            cirq_circ.append(XPowGate(exponent=0.5)(*cirq_op_qubits))
            continue
        if name == "sx_dg":
            cirq_circ.append(XPowGate(exponent=-0.5)(*cirq_op_qubits))
            continue

        # parametrized single-qubit gates
        if params and name != "id":
            if name == "p":
                cirq_circ.append(ZPowGate(exponent=params[0] / np.pi)(*cirq_op_qubits))
            else:
                cirq_circ.append(cirq_gate(*params)(*cirq_op_qubits))
            continue

        # simple gates (no parameters)
        cirq_circ.append(cirq_gate(*cirq_op_qubits))

    return cirq_circ


def convert_from_cirq(cirq_circuit):
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
        import cirq
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Cirq to Qrisp converter.") from exc
    from qrisp import QuantumCircuit
    from qrisp.circuit import standard_operations as ops

    # setup: qubit map and gate lookup table
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

    # Maps Cirq gate types to Qrisp gate *callables* (not instances), so
    # each use gets a fresh object and no two operations share a gate.
    gate_map = {
        cirq.HPowGate: ops.HGate,
        cirq.CXPowGate: ops.CXGate,
        cirq.CZPowGate: ops.CZGate,
        cirq.SwapPowGate: ops.SwapGate,
        cirq.IdentityGate: ops.IDGate,
        cirq.ResetChannel: ops.Reset,
    }
    # these gate types have no Qrisp equivalent for fractional exponents;
    # only the full gate (exponent 1, -1, 3, …) can be converted
    exponent_guarded = {cirq.HPowGate, cirq.CXPowGate, cirq.CZPowGate, cirq.SwapPowGate}

    # main conversion loop
    for op in cirq_circuit.all_operations():
        # Extract the gate, unwrapping ControlledOperation if present
        extra_controls = None
        if isinstance(op, cirq.ControlledOperation):
            sub_op = op.sub_operation
            gate = getattr(sub_op, "gate", None)
            if gate is None:
                raise ValueError(f"Controlled sub-operation {sub_op} is not supported by the Cirq to Qrisp converter.")
            extra_controls = (list(op.controls), list(sub_op.qubits))
        else:
            gate = getattr(op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Operation {op} without gate attribute is not supported by the Cirq to Qrisp converter."
                )

        # Global phase in Cirq acts on 0 qubits; Qrisp requires a qubit
        # argument, so map it to the first available qubit.  Handled
        # outside the dict because of this special qubit-mapping logic.
        if isinstance(gate, cirq.GlobalPhaseGate):
            phi = np.angle(gate.coefficient)
            if cirq_qubits:
                qc.append(ops.GPhaseGate(phi), [qubit_map[cirq_qubits[0]]])
            continue

        # Measurement uses the circuit's measure() method for automatic
        # classical-bit allocation.  Cannot go through the gate dict.
        if isinstance(gate, cirq.MeasurementGate):
            qrisp_qubits = [qubit_map[q] for q in op.qubits]
            if len(qrisp_qubits) == 1:
                qc.measure(qrisp_qubits[0])
            else:
                qc.measure(qrisp_qubits)
            continue

        # Unwrap nested ControlledGate layers (e.g. cirq.X.controlled()).
        # Cannot live in the dict because it needs to recursively peel
        # control layers off before converting the innermost gate.
        ctrl_layers = []
        inner_gate = gate
        while isinstance(inner_gate, cirq.ControlledGate):
            cv = inner_gate.control_values
            ctrl_state = ""
            for v in cv:
                if len(v) != 1:
                    raise ValueError(f"Multi-valued control {v} in {inner_gate} not supported.")
                val = v[0]
                if val not in (0, 1):
                    raise ValueError(f"Unsupported control value {val} in {inner_gate}.")
                ctrl_state += str(val)
            ctrl_layers.append((inner_gate.num_controls(), ctrl_state))
            inner_gate = inner_gate.sub_gate

        # Convert the innermost gate to a Qrisp operation
        qrisp_op = None

        # Dict lookup by exact type for gates whose Qrisp equivalent is
        # fixed or follows a simple (attr, multiplier) pattern.
        converter = gate_map.get(type(inner_gate))
        if converter is not None:
            if type(inner_gate) in exponent_guarded:
                if not np.isclose(inner_gate.exponent % 2, 1.0):
                    raise ValueError(
                        f"Only the full {type(inner_gate).__name__} "
                        f"(exponent 1, -1, 3, ...) can be converted, "
                        f"not the fractional-power variant with exponent "
                        f"{inner_gate.exponent} ({inner_gate})."
                    )
            qrisp_op = converter()

        # XPowGate uses isinstance to catch cirq.X (type _PauliX, a
        # subclass).  The exponent selects X, SX, SX†, or generic RX.
        elif isinstance(inner_gate, cirq.XPowGate):
            exp = inner_gate.exponent
            if isinstance(inner_gate, cirq.Rx):
                qrisp_op = ops.RXGate(exp * np.pi)
            elif np.isclose(exp % 4, 1.0):
                qrisp_op = ops.XGate()
            elif np.isclose(exp % 4, 0.5):
                qrisp_op = ops.SXGate()
            elif np.isclose(exp % 4, 3.5):
                qrisp_op = ops.SXDGGate()
            else:
                qrisp_op = ops.RXGate(exp * np.pi)

        # YPowGate: same subclass issue with cirq.Y (type _PauliY).
        elif isinstance(inner_gate, cirq.YPowGate):
            exp = inner_gate.exponent
            if isinstance(inner_gate, cirq.Ry):
                qrisp_op = ops.RYGate(exp * np.pi)
            elif np.isclose(exp % 4, 1.0):
                qrisp_op = ops.YGate()
            else:
                qrisp_op = ops.RYGate(exp * np.pi)

        # ZPowGate: catches cirq.Z (type _PauliZ) and exponent-based
        # variants like S (exp=0.5), T (exp=0.25), etc.
        elif isinstance(inner_gate, cirq.ZPowGate):
            exp = inner_gate.exponent
            if isinstance(inner_gate, cirq.Rz):
                qrisp_op = ops.RZGate(exp * np.pi)
            elif np.isclose(exp % 4, 1.0):
                qrisp_op = ops.ZGate()
            elif np.isclose(exp % 4, 0.5):
                qrisp_op = ops.SGate()
            elif np.isclose(exp % 4, 3.5):
                qrisp_op = ops.SGate().inverse()
            elif np.isclose(exp % 4, 0.25):
                qrisp_op = ops.TGate()
            elif np.isclose(exp % 4, 3.75):
                qrisp_op = ops.TGate().inverse()
            else:
                qrisp_op = ops.PGate(exp * np.pi)

        else:
            raise ValueError(f"Gate {gate} is not supported by the Cirq to Qrisp converter.")

        # Wrap any ControlledGate layers (inside-out)
        for num_ctrl, ctrl_state in reversed(ctrl_layers):
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=num_ctrl,
                ctrl_state=ctrl_state,
            )

        # Wrap any outer ControlledOperation and append
        if extra_controls is not None:
            controls, sub_qubits = extra_controls
            cv = op.control_values
            ctrl_state = ""
            for v in cv:
                if len(v) != 1:
                    raise ValueError(f"Multi-valued control {v} in {op} not supported.")
                val = v[0]
                if val not in (0, 1):
                    raise ValueError(f"Unsupported control value {val} in {op}.")
                ctrl_state += str(val)
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
