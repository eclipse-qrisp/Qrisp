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
"""

from fractions import Fraction
from functools import partial

import numpy as np

from qrisp import HGate, QuantumCircuit, RYGate, RZGate, SwapGate, SXGate, ZGate, u3Gate


def _transpile(qrisp_circuit, gate_map):
    # repeatedly transpile unknown gates until only known ones remain
    def _unknown_names(circuit):
        return {instr.op.name for instr in circuit.data if instr.op.name not in gate_map}

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
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to PyZX converter."
            ) from exc

        new_unknown = _unknown_names(transpiled)
        if new_unknown == unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {names}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_pyzx(), or "
                f"use only gates supported natively by the converter."
            )

        qrisp_circuit = transpiled

    return qrisp_circuit


def convert_to_pyzx(qrisp_circuit: QuantumCircuit):
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
        If PyZX is not installed.
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
        from pyzx import Circuit
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("PyZX must be installed to be able to use the Qrisp to PyZX converter.") from exc
    from pyzx import settings

    settings.strict_phase_types = False  # this enables PyZX to (approximately) convert float phases to a Fraction

    gate_map = {
        "cx": "CNOT",
        "cy": "CY",
        "cz": "CZ",
        "swap": "SWAP",
        "h": "HAD",
        "x": "NOT",
        "y": "Y",
        "z": "Z",
        "rx": "XPhase",
        "ry": "YPhase",
        "rz": "ZPhase",
        "u3": "U3",
        "s": "S",
        "t": "T",
        "s_dg": None,
        "t_dg": None,
        "p": None,
        "sx": "SX",
        "sx_dg": None,
        "xxyy": None,
        "rxx": "RXX",
        "rzz": "RZZ",
        "measure": "Measurement",
        "reset": "Reset",
        # skip qubit allocation and deallocation ops in the converter, as well as identity and global phases
        "qb_alloc": None,
        "qb_dealloc": None,
        "gphase": None,
        "id": None,
    }

    qrisp_circuit = _transpile(qrisp_circuit, gate_map)

    num_qubits = qrisp_circuit.num_qubits()
    pyzx_circuit = Circuit(num_qubits)

    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = i

    for instr in qrisp_circuit.data:
        name = instr.op.name
        qubits = instr.qubits
        params = instr.op.params if hasattr(instr.op, "params") else []

        pyxz_gate = gate_map[name]
        pyxz_op_qubits = [qubit_map[q] for q in qubits]

        special_gate_actions = {
            **dict.fromkeys(["id", "gphase", "qb_alloc", "qb_dealloc"], lambda: None),
            "s_dg": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, Fraction(-1, 2)),
            "t_dg": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, Fraction(-1, 4)),
            "p": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, params[0] / np.pi),
            "sx_dg": lambda: pyzx_circuit.add_gate("XPhase", *pyxz_op_qubits, Fraction(-1, 2)),
        }

        # gate with no direct PyXZ equivalent
        if pyxz_gate is None:
            if name in special_gate_actions:
                special_gate_actions[name]()
            # decompose via its .definition circuit (e.g. xxyy)
            elif instr.op.definition:
                pyzx_circuit.add_circuit(convert_to_pyzx(instr.op.definition), mask=pyxz_op_qubits)
            else:
                raise ValueError(f"{name} gate has no PyZX equivalent and no definition to decompose.")
            continue

        if params:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits, *[p / np.pi for p in params])
        else:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits)

    return pyzx_circuit


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyzx import Circuit


def convert_from_pyzx(pyzx_circuit: "Circuit"):
    """Convert a PyZX QuantumCircuit to a Qrisp Circuit.

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
    ValueError
        If a gate is not supported by the converter.

    Notes
    -----
    Gates that exist in PyZX but not in Qrisp are either substituted directly if they
    have a straightforward equivalent (applies to U2 and several controlled gates),
    or it is used PyZX's to_basic_gates() method to decompose those gates.

    """
    qc = QuantumCircuit(pyzx_circuit.qubits)

    gate_map = {
        # single-qubit gates
        "NOT": qc.x,
        "Y": qc.y,
        "Z": qc.z,
        "HAD": qc.h,
        "XPhase": qc.rx,
        "YPhase": qc.ry,
        "ZPhase": qc.rz,
        "U2": partial(qc.u3, np.pi / 2),
        "U3": qc.u3,
        "SX": lambda adj, x: qc.sx_dg(x) if adj else qc.sx(x),
        "S": lambda adj, x: qc.s_dg(x) if adj else qc.s(x),
        "T": lambda adj, x: qc.t_dg(x) if adj else qc.t(x),
        # multi-qubits gates
        "CNOT": qc.cx,
        "CY": qc.cy,
        "CZ": qc.cz,
        "CRX": qc.crx,
        "CRY": lambda phase, x, y: qc.append(RYGate(phase).control(), [x, y]),
        "CRZ": lambda phase, x, y: qc.append(RZGate(phase).control(), [x, y]),
        "CSX": lambda x, y: qc.append(SXGate().control(), [x, y]),
        "CPhase": qc.cp,
        "ParityPhase": None,
        "PhaseGadget": None,
        "XCX": None,
        "SWAP": qc.swap,
        "CSWAP": lambda x, y, z: qc.append(SwapGate().control(), [x, y, z]),
        "CHAD": lambda x, y: qc.append(HGate().control(), [x, y]),
        "Tof": qc.ccx,
        "CCZ": lambda x, y, z: qc.append(ZGate().control(2), [x, y, z]),
        "CU3": lambda theta, phi, lam, x, y: qc.append(u3Gate(theta, phi, lam).control(), [x, y]),
        "CU": None,
        "RZZ": qc.rzz,
        "RXX": qc.rxx,
        "FSim": None,
        # non-unitary operations
        "Measurement": qc.measure,
        "Reset": qc.reset,
        "InitAncilla": None,
        "PostSelect": None,
        "DiscardBit": None,
        "ConditionalGate": None,
    }

    def add_gate(gate):
        # single-qubit, parameter-free gates without adjoint version and non-unitary operations
        def _single_qubit_parameter_free(gate):
            gate_map[gate.name](gate.target)

        # single-qubit gates with adjoint version
        def _single_qubit_with_adjoint_version(gate):
            gate_map[gate.name](gate.adjoint, gate.target)

        # single-qubit, one-parameter gates
        def _single_qubit_one_parameter(gate):
            gate_map[gate.name](float(gate.phase) * np.pi, gate.target)

        # single-qubit, multi-parameter gates
        def _single_qubit_multi_parameter(gate):
            gate_map[gate.name](*[float(p) * np.pi for p in gate.phases], gate.target)

        # two-qubit, paratemeter-free gates
        def _two_qubit_parameter_free(gate):
            gate_map[gate.name](gate.control, gate.target)

        # two-qubit, one-paratemeter gates
        def _two_qubit_one_parameter(gate):
            gate_map[gate.name](float(gate.phase) * np.pi, gate.control, gate.target)

        # two-qubit, multi-paratemeter gates
        def _two_qubit_multi_parameter(gate):
            gate_map[gate.name](*[float(p) * np.pi for p in gate.phases], gate.control, gate.target)

        # multi-qubit, paratemeter-free gates
        def _multi_qubit_parameter_free(gate):
            gate_map[gate.name](gate.ctrl1, gate.ctrl2, gate.target)

        function_map = {
            **dict.fromkeys(["NOT", "Y", "Z", "HAD", "Measurement", "Reset"], _single_qubit_parameter_free),
            **dict.fromkeys(["SX", "S", "T"], _single_qubit_with_adjoint_version),
            **dict.fromkeys(["XPhase", "YPhase", "ZPhase"], _single_qubit_one_parameter),
            **dict.fromkeys(["U2", "U3"], _single_qubit_multi_parameter),
            **dict.fromkeys(["CNOT", "CY", "CZ", "CSX", "SWAP", "CHAD"], _two_qubit_parameter_free),
            **dict.fromkeys(["CRX", "CRY", "CRZ", "CPhase", "RZZ", "RXX"], _two_qubit_one_parameter),
            **dict.fromkeys(["CU3"], _two_qubit_multi_parameter),
            **dict.fromkeys(["CSWAP", "Tof", "CCZ"], _multi_qubit_parameter_free),
        }

        function_map[gate.name](gate)

    for gate in pyzx_circuit.gates:
        if gate.name in gate_map:
            if gate_map[gate.name] is not None:
                add_gate(gate)
            else:
                # try with pyzx's basic gate decomposition
                for _gate in gate.to_basic_gates():
                    if _gate.name in gate_map and gate_map[_gate.name] is not None:
                        add_gate(_gate)
                    else:
                        raise ValueError(f"{_gate.name} gate has no Qrisp equivalent and cannot be decomposed either.")
        else:
            raise ValueError(f"{gate.name} of PyZX is unknown and thus cannot be converted to Qrisp.")

    return qc
