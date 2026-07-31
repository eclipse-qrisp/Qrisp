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

import numpy as np


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
    """    
    from pyzx.circuit import (
        NOT,
        Y, 
        Z, 
        HAD, 
        XPhase, 
        YPhase, 
        ZPhase, 
        U2, 
        U3, 
        S, 
        T, 
        SX, 
        SWAP, 
        RXX, 
        RZZ, 
        CNOT,
        CY, 
        CZ, 
        CHAD, 
        CSX, 
        XCX, 
        CRX, 
        CRY, 
        CRZ, 
        CPhase, 
        CU3, 
        CU, 
        CSWAP, 
        Tofolli, 
        CCZ, 
        ParityPhase, 
        FSim,
        Measurement,
        PhaseGadget, 
        ConditionalGate
    )
    """
    gate_map = {
        "cx": "CNOT",
        "cz": "CZ",
        "swap": "SWAP",
        "h": "HAD",
        "x": "NOT",
        "y": "Y",
        "z": "Z",
        "rx": "XPhase",
        "ry": "YPhase",
        "rz": "ZPhase",
        "s": "S",
        "t": "T",
        "s_dg": None,
        "t_dg": None,
        "measure": "Measurement",
        "reset": "Reset",
        "id": None,
        "p": None,
        "sx": "SX",
        "sx_dg": None,
        "gphase": None,
        "xxyy": None,
        "rxx": "RXX",
        "rzz": "RZZ",
        # skip qubit allocation and deallocation ops in the converter
        "qb_alloc": None,
        "qb_dealloc": None,
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
            if name in ["id", "qb_alloc", "qb_dealloc"]:
                pass
            # decompose via its .definition circuit (e.g. xxyy, rxx, rzz)
            elif instr.op.definition:
                pyzx_circuit.append(convert_to_pyzx(instr.op.definition), mask=pyxz_op_qubits)
            else:
                raise ValueError(f"{name} gate has no PyXZ equivalent and no definition to decompose.")
            continue
        
        
        if params:
            if name in ["rx", "ry", "rz", "rxx", "rzz"]:
                pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits, phase=params[0])
            else:
                raise ValueError(f"{name} gate has a parameter but is not in rx, ry, rz, rxx, rzz.")
        else:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits)
        
        
    return pyzx_circuit
        
    
