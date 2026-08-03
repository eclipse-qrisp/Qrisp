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

from qrisp import QuantumCircuit, RYGate, RZGate, SXGate, SwapGate, HGate, ZGate, u3Gate
import numpy as np
from functools import partial
from fractions import Fraction


def convert_to_pyzx(qrisp_circuit):
    """Convert a Qrisp QuantumCircuit to a PyZX Circuit.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The Qrisp QuantumCircuit to convert.

    Returns
    -------
    pyzx.Circuit
        A pyzx.Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If PyZ is not installed.
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
    settings.strict_phase_types = False     #this enables PyZX to (approximately) convert float phases to a Fraction

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
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to Cirq converter."
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

        # gate with no direct PyXZ equivalent
        if pyxz_gate is None:
            if name in ["id", "gphase", "qb_alloc", "qb_dealloc"]:
                pass
            elif name == "s_dg":
                pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, Fraction(-1,2))
            elif name == "t_dg":
                pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, Fraction(-1,4))
            elif name == "p":
                pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, params[0]/np.pi)
            elif name == "sx_dg":
                pyzx_circuit.add_gate("XPhase", *pyxz_op_qubits, Fraction(-1,2))
            # decompose via its .definition circuit (e.g. xxyy)
            elif instr.op.definition:
                pyzx_circuit.append(convert_to_pyzx(instr.op.definition), mask=pyxz_op_qubits)
            else:
                raise ValueError(f"{name} gate has no PyXZ equivalent and no definition to decompose.")
            continue

        if params:
            if name in ["rx", "ry", "rz", "u3", "rxx", "rzz"]:
                pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits, *[p/np.pi for p in params])
            else:
                raise ValueError(f"{name} gate has a parameter but is not in rx, ry, rz, u3, rxx, rzz.")
        else:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits)


    return pyzx_circuit
        
    
def convert_from_pyzx(pyzx_circuit):
    """Convert a PyZX QuantumCircuit to a Qrisp Circuit.

    Parameters
    ----------
    pyzx_circuit : QuantumCircuit
        The Qrisp QuantumCircuit to convert.

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
        #single-qubit gates
        "NOT": qc.x,
        "Y": qc.y,
        "Z": qc.z,
        "HAD": qc.h,
        "XPhase": qc.rx,
        "YPhase": qc.ry,
        "ZPhase": qc.rz,
        "U2": partial(qc.u3, np.pi/2),
        "U3": qc.u3,
        "SX": qc.sx,
        "S": qc.s,
        "T": qc.t,

        #multi-qubits gates
        "CNOT": qc.cx,
        "CY": qc.cy,
        "CZ": qc.cz,
        "CRX": qc.crx,
        "CRY": lambda phase,x,y: qc.append(RYGate(phase).control(), [x,y]),
        "CRZ": lambda phase,x,y: qc.append(RZGate(phase).control(), [x,y]),
        "CSX": lambda x,y: qc.append(SXGate().control(), [x,y]),
        "CPhase": qc.cp,
        "ParityPhase": None,
        "PhaseGadget": None,
        "XCX": None,
        "SWAP": qc.swap,
        "CSWAP": lambda x,y,z: qc.append(SwapGate().control(), [x,y,z]),
        "CHAD": lambda x,y: qc.append(HGate().control(), [x,y]),
        "Tof": qc.ccx,
        "CCZ": lambda x,y,z: qc.append(ZGate().control(2), [x,y,z]),
        "CU3": lambda theta,phi,lam,x,y: qc.append(u3Gate(theta,phi,lam).control(), [x,y]),
        "RZZ": qc.rzz,
        "RXX": qc.rxx,
        "FSim": None,

        #non-unitary operations
        "Measurement": qc.measure,
        "Reset": qc.reset,
        "InitAncilla": None,
        "PostSelect": None,
        "DiscardBit": None,
        "ConditionalGate": None,
    }


    def add_gate(gate):
        #single-qubit, parameter-free gates and non-unitary operations
        if gate.name in ["NOT", "Y", "Z", "HAD", "SX", "S", "T", "Measurement", "Reset"]:
            gate_map[gate.name](gate.target)
        #single-qubit, one-parameter gates
        elif gate.name in ["XPhase", "YPhase", "ZPhase"]:
            gate_map[gate.name](float(gate.phase)*np.pi, gate.target)
        #single-qubit, multi-parameter gates
        elif gate.name in ["U2", "U3"]:
            gate_map[gate.name](*[float(p)*np.pi for p in gate.phases], gate.target)

        #two-qubit, paratemeter-free gates
        elif gate.name in ["CNOT", "CY", "CZ", "CSX", "SWAP", "CHAD"]:
            gate_map[gate.name](gate.control, gate.target)
        #two-qubit, one-paratemeter gates
        elif gate.name in ["CRX", "CRY", "CRZ", "CPhase", "RZZ", "RXX"]:
            gate_map[gate.name](float(gate.phase)*np.pi, gate.control, gate.target)
        #two-qubit, multi-paratemeter gates
        elif gate.name in ["CU3"]:
            gate_map[gate.name](*[float(p)*np.pi for p in gate.phases], gate.control, gate.target)
        #multi-qubit, paratemeter-free gates
        elif gate.name in ["CSWAP", "Tof", "CCZ"]:
            gate_map[gate.name](gate.ctrl1, gate.ctrl2, gate.target)

    for gate in pyzx_circuit.gates:
        if gate_map[gate.name] is not None:
            add_gate(gate)
        else:
            #try with pyzx's basic gate decomposition
            for _gate in gate.to_basic_gates():
                if gate_map[_gate.name] is not None:
                    add_gate(_gate)
                else:
                    raise ValueError(f"{_gate.name} gate has no Qrisp equivalent and cannot be decomposed either.")

    return qc
