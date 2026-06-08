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
    Gates that are not directly mapped to a Cirq equivalent are decomposed
    by Qrisp's transpiler before conversion.  The converter collects all
    unknown gate names in the circuit and transpiles them together in a
    single pass, then repeats on the result; the loop always terminates
    because repeated passes either shrink the unknown set to empty, detect
    a cycle (same set appearing twice), or detect stagnation (transpilation
    made no progress).
    """
    try:
        from cirq import Circuit, LineQubit
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Cirq must be installed to be able to use the Qrisp to Cirq converter."
        )

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

    # map from Qrisp gate name to Cirq gate callable/class
    qrisp_cirq_ops_dict = {
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
        "sx_dg": XPowGate,  # dagger handled in the sx_dg special-case block below
        # entries with None are handled below: gphase via early return,
        # xxyy/rxx/rzz via their .definition, qb_alloc/qb_dealloc silently skipped
        "gphase": None,
        "xxyy": None,
        "rxx": None,
        "rzz": None,
        "qb_alloc": None,
        "qb_dealloc": None,
    }
    # repeatedly transpile unknown gates until only known ones remain.
    # unknown gates are all transpiled together in a single pass so that
    # decomposing one gate cannot reintroduce a previously-seen gate name
    # across recursive frames (which would cause infinite alternation).
    # Track every distinct set of unknown names to detect cycles.
    _unknown_sets_seen = []
    while True:
        unknown = {instr.op.name for instr in qrisp_circuit.data
                   if instr.op.name not in qrisp_cirq_ops_dict
                   and instr.op.name not in ("gphase", "qb_alloc", "qb_dealloc")}
        if not unknown:
            break

        if unknown in _unknown_sets_seen:
            gates = ", ".join(sorted(unknown))
            raise ValueError(
                f"Decomposition cycle detected for gates: {gates}. "
                f"Each pass reintroduces gate names that were previously "
                f"eliminated. Try transpiling the circuit manually with "
                f"Qrisp's transpile() method before calling to_cirq()."
            )
        _unknown_sets_seen.append(unknown)

        # capture unknown by value so the closure works even if
        # transpile() were ever called lazily
        def _transpile_predicate(op, _unknown=unknown):
            return op.name in _unknown

        try:
            transpiled = qrisp_circuit.transpile(
                transpile_predicate=_transpile_predicate
            )
        except Exception:
            raise ValueError(
                f"Gates {unknown} could not be transpiled and are not supported "
                "by the Qrisp to Cirq converter."
            )
        # re-check: if the same unknown gate names persist, transpile
        # made no progress and we'd loop forever
        new_unknown = {instr.op.name for instr in transpiled.data
                       if instr.op.name not in qrisp_cirq_ops_dict
                       and instr.op.name not in ("gphase", "qb_alloc", "qb_dealloc")}
        if new_unknown == unknown:
            gates = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {gates}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_cirq(), or "
                f"use only gates supported natively by the converter."
            )
        qrisp_circuit = transpiled

    # get data from Qrisp circuit
    qrisp_circ_ops_data = qrisp_circuit.data

    # create an empty Cirq circuit
    cirq_circuit = Circuit()

    if cirq_qubits is None:
        cirq_qubits = [LineQubit(i) for i in range(len(qrisp_circuit.qubits))]

    # create a mapping of Qrisp qubits to Cirq qubits
    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    for instr in qrisp_circ_ops_data:
        op_i = instr.op.name
        op_qubits_i = instr.qubits
        params = instr.op.params if hasattr(instr.op, "params") else []

        if op_i == "gphase":
            # Qrisp's gphase param is a radian angle; GlobalPhaseGate
            # takes the complex coefficient exp(i * angle).
            cirq_circuit.append(GlobalPhaseGate(np.exp(1j * params[0]))())
            continue

        cirq_op_qubits = [qubit_map[q] for q in op_qubits_i]
        cirq_gate = qrisp_cirq_ops_dict[op_i]

        if cirq_gate is None:
            if instr.op.definition:
                new_circ = instr.op.definition
                cirq_circuit.append(convert_to_cirq(new_circ, cirq_op_qubits))
                continue
            if op_i not in ("qb_alloc", "qb_dealloc"):
                raise ValueError(
                    f"{op_i} gate has no Cirq equivalent and no definition to decompose."
                )
            continue

        if isinstance(instr.op, ControlledOperation):
            # control and target qubits from qrisp
            cs = instr.op.ctrl_state
            if isinstance(cs, int):
                n_ctrl = instr.op.num_qubits - instr.op.base_operation.num_qubits
            else:
                n_ctrl = len(cs)
            control_qubits = instr.qubits[:n_ctrl]
            target_qubits = instr.qubits[n_ctrl:]

            cirq_ctrl_qubits = [qubit_map[q] for q in control_qubits]
            cirq_target_qubits = [qubit_map[q] for q in target_qubits]

            # verify the control and target qubit mapping worked
            if cirq_op_qubits != cirq_ctrl_qubits + cirq_target_qubits:
                raise RuntimeError("Qubit mapping invariant violated.")
            if op_qubits_i != control_qubits + target_qubits:
                raise RuntimeError("Qubit mapping invariant violated.")

            # build the base gate for .controlled().
            # cx/cz are unwrapped to X/Z because CNOT.controlled() would
            # produce CCNOT (Toffoli) instead of CNOT.
            if op_i == "sx":
                base = XPowGate(exponent=0.5)
            elif op_i == "sx_dg":
                base = XPowGate(exponent=-0.5)
            elif op_i == "p" and params:
                base = ZPowGate(exponent=params[0] / np.pi)
            elif op_i == "cx":
                base = X
            elif op_i == "cz":
                base = Z
            elif params:
                base = cirq_gate(*params)
            else:
                base = cirq_gate

            # convert ctrl_state for Cirq's .controlled(control_values=...)
            if isinstance(cs, str):
                ctrl_state = [int(c) for c in cs]
            elif isinstance(cs, int) and cs != -1:
                ctrl_state = [int(b) for b in bin(cs)[2:].zfill(n_ctrl)]
            else:
                # cs == -1 means all controls on; None tells Cirq the same
                ctrl_state = None
            controlled = base.controlled(
                num_controls=n_ctrl, control_values=ctrl_state
            )
            cirq_circuit.append(controlled(*cirq_ctrl_qubits, *cirq_target_qubits))
            continue

        # for single qubit parametrized gates
        if params and op_i != "id":
            if op_i == "p":
                # Cirq does not have a phase gate
                # for this reason, it has to be dealt with as a special case.
                # the ZPowGate has a global phase in addition to the
                # phase exponent. The default is to assume global_shift = 0 in cirq
                exp_param = params[0]
                cirq_circuit.append(
                    ZPowGate(exponent=exp_param / np.pi)(*cirq_op_qubits)
                )
            else:
                gate_instance = cirq_gate(*params)
                cirq_circuit.append(gate_instance(*cirq_op_qubits))
            continue

        # for simple single qubit gates
        if op_i == "sx":
            cirq_circuit.append(XPowGate(exponent=0.5)(*cirq_op_qubits))
        elif op_i == "sx_dg":
            cirq_circuit.append(XPowGate(exponent=-0.5)(*cirq_op_qubits))
        else:
            cirq_circuit.append(cirq_gate(*cirq_op_qubits))

    return cirq_circuit


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
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Cirq must be installed to be able to use the Cirq to Qrisp converter."
        )

    from qrisp import QuantumCircuit
    from qrisp.circuit import standard_operations as ops

    qc = QuantumCircuit(len(cirq_circuit.all_qubits()))

    # Cirq supports mixed qubit types (LineQubit, GridQubit, etc.)
    # in the same circuit; sorting them raises TypeError.
    try:
        cirq_qubits = sorted(cirq_circuit.all_qubits())
    except TypeError:
        raise ValueError(
            "The Cirq to Qrisp converter requires homogeneous qubit types "
            "(all qubits must be comparable, e.g. all LineQubit)."
        )
    qubit_map = {q: qc.qubits[i] for i, q in enumerate(cirq_qubits)}

    # Build a lookup table once, outside the loop.
    # Maps Cirq gate types to Qrisp gate callables; fresh instances
    # are created on each use so that no two circuit operations share
    # the same mutable gate object.
    cirq_qrisp_gate_map = {
        cirq.HPowGate: ops.HGate,
        cirq.CXPowGate: ops.CXGate,
        cirq.CZPowGate: ops.CZGate,
        cirq.SwapPowGate: ops.SwapGate,
        cirq.IdentityGate: ops.IDGate,
        cirq.ResetChannel: ops.Reset,
    }

    exponent_guarded = {cirq.HPowGate, cirq.CXPowGate,
                        cirq.CZPowGate, cirq.SwapPowGate}

    for op in cirq_circuit.all_operations():
        # Extract the gate, unwrapping ControlledOperation if present
        extra_controls = None
        if isinstance(op, cirq.ControlledOperation):
            sub_op = op.sub_operation
            gate = getattr(sub_op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Controlled sub-operation {sub_op} is not supported "
                    "by the Cirq to Qrisp converter."
                )
            extra_controls = (list(op.controls), list(sub_op.qubits))
        else:
            gate = getattr(op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Operation {op} without gate attribute is not supported "
                    "by the Cirq to Qrisp converter."
                )

        # Global phase in Cirq acts on 0 qubits; Qrisp requires a qubit
        # argument, so map it to the first available qubit.  If the
        # circuit has no qubits the phase is a scalar and is skipped.
        if isinstance(gate, cirq.GlobalPhaseGate):
            coeff = gate.coefficient
            phi = np.angle(coeff)
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
            ctrl_values = inner_gate.control_values
            if ctrl_values is not None:
                # Cirq yields tuples like (1,) per control; take first element.
                # Reject multi-valued controls (e.g. (0, 1)) which Qrisp cannot
                # represent.
                for v in ctrl_values:
                    if isinstance(v, int):
                        if v not in (0, 1):
                            raise ValueError(
                                f"Invalid control value {v} in gate "
                                f"{inner_gate} is not supported by the "
                                "Cirq to Qrisp converter."
                            )
                    elif len(v) != 1:
                        raise ValueError(
                            f"Multi-valued control qubit {v} in gate "
                            f"{inner_gate} is not supported by the Cirq "
                            "to Qrisp converter."
                        )
                ctrl_state = "".join(
                    str(int(v)) if isinstance(v, int) else str(int(v[0]))
                    for v in ctrl_values
                )
            else:
                ctrl_state = -1
            ctrl_layers.append((inner_gate.num_controls(), ctrl_state))
            inner_gate = inner_gate.sub_gate

        # Dict lookup by exact type for gates whose Qrisp equivalent is
        # fixed or follows a simple (attr, multiplier) pattern.
        converter = cirq_qrisp_gate_map.get(type(inner_gate))
        if converter is not None:
            if type(inner_gate) in exponent_guarded:
                if not np.isclose(inner_gate.exponent % 2, 1.0):
                    raise ValueError(
                        f"Only the full {type(inner_gate).__name__} "
                        f"(exponent 1, -1, 3, etc.) can be converted, "
                        f"not the fractional-power variant with exponent "
                        f"{inner_gate.exponent} ({inner_gate})."
                    )
            qrisp_op = converter()

        # XPowGate uses isinstance to catch cirq.X (type _PauliX, a
        # subclass).  The exponent selects X, SX, SX†, or generic RX.
        # Python % always yields [0, 4), so only the positive branch of
        # each modulus pair can fire; the negative equivalents are listed
        # here for readability (matching the Cirq convention where e.g.
        # exp=-0.5 and exp=3.5 are the same gate).
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
            raise ValueError(
                f"Gate {gate} is not supported by the Cirq to Qrisp converter."
            )

        # Wrap any ControlledGate layers (inside-out)
        for num_ctrl, ctrl_state in reversed(ctrl_layers):
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=num_ctrl,
                ctrl_state=ctrl_state,
            )

        # Wrap any outer ControlledOperation and append.
        # op.control_values is used directly rather than going through
        # op.gate because ControlledOperation.gate may be None depending
        # on the Cirq version.
        if extra_controls is not None:
            controls, sub_qubits = extra_controls
            cv = op.control_values
            if cv is not None:
                # reject multi-valued controls that Qrisp cannot represent
                for v in cv:
                    if isinstance(v, int):
                        if v not in (0, 1):
                            raise ValueError(
                                f"Invalid control value {v} in gate "
                                f"{op} is not supported by the Cirq "
                                "to Qrisp converter."
                            )
                    elif len(v) != 1:
                        raise ValueError(
                            f"Multi-valued control qubit {v} in gate "
                            f"{op} is not supported by the Cirq "
                            "to Qrisp converter."
                        )
                ctrl_state = "".join(
                    str(int(v)) if isinstance(v, int) else str(int(v[0]))
                    for v in cv
                )
            else:
                ctrl_state = -1
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
