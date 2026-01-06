"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

def qrisp_to_stim(qc, return_clbit_map = False):
    """
    Convert a Qrisp quantum circuit to a Stim circuit.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp quantum circuit to convert. The circuit will be automatically
        transpiled to decompose composite gates into Clifford basis gates.
    return_clbit_map : bool, optional
        If set to True, the function returns the clbit_map, as described below.
        The default is False.
    
    Returns
    -------
    stim_circuit : stim.Circuit
        The converted Stim circuit.
    clbit_map : dict
        A dictionary mapping Qrisp Clbit objects to Stim measurement record indices. 
        For example, {clbit_obj_0: 2, clbit_obj_1: 0} means the first Clbit object 
        corresponds to the 3rd measurement (index 2) in Stim's measurement record.
    
    Notes
    -----
    Stim only supports Clifford gates. Non-Clifford gates will raise an exception.
    Supported gates include: H, X, Y, Z, S, S_DAG, SQRT_X, SQRT_X_DAG, SQRT_Y, 
    SQRT_Y_DAG, CX (CNOT), CY, CZ, SWAP, ISWAP, and measurements.
    
    Mid-circuit measurements are preserved in their original positions in the circuit.
    The classical bit mapping dictionary allows you to reorder Stim's measurement 
    results to match Qrisp's classical bit ordering.
    
    Examples
    --------
    Basic conversion:
    
    >>> from qrisp import QuantumCircuit
    >>> from qrisp_to_stim_converter import qrisp_to_stim
    >>> qc = QuantumCircuit(2, 2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1])
    >>> stim_circuit, clbit_map = qrisp_to_stim(qc, True)
    >>> print(stim_circuit)
    H 0
    CX 0 1
    M 0 1
    >>> print(clbit_map)  # Maps Clbit objects to measurement indices
    {cb_0: 0, cb_1: 1}
    
    Handling non-sequential classical bit mapping:
    
    >>> qc = QuantumCircuit(3, 3)
    >>> qc.h(0)
    >>> qc.x(1)
    >>> qc.measure(qc.qubits[0], qc.clbits[2])  # qubit 0 -> clbit 2
    >>> qc.measure(qc.qubits[1], qc.clbits[0])  # qubit 1 -> clbit 0
    >>> stim_circuit, clbit_map = qrisp_to_stim(qc, True)
    >>> # clbit_map maps Clbit objects to Stim measurement record indices
    >>> sampler = stim_circuit.compile_sampler()
    >>> samples = sampler.sample(100)
    >>> # Reorder to match Qrisp's classical bit order:
    >>> clbit_indices = {clbit: qc.clbits.index(clbit) for clbit in clbit_map}
    >>> sorted_clbits = sorted(clbit_map.keys(), key=lambda cb: clbit_indices[cb])
    >>> reordered = samples[:, [clbit_map[cb] for cb in sorted_clbits]]
    """
    import stim
    from qrisp.stim_noise import StimError
    
    qc = qc.transpile()
    
    # Create Stim circuit
    stim_circuit = stim.Circuit()
    
    # Track measurements to build the classical bit mapping
    # Stim records measurements sequentially in a measurement record
    # We need to map Qrisp Clbit objects to Stim measurement record indices
    clbit_to_measurement_idx = {}  # Maps Clbit object -> measurement_record_idx
    measurement_counter = 0  # Tracks the current position in Stim's measurement record
    
    # Gate mapping from Qrisp to Stim
    # Stim gate names are usually uppercase
    single_qubit_gates = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "s_dg": "S_DAG",
        "sdg": "S_DAG",  # Alternative name from transpiler
        "sx": "SQRT_X",
        "sx_dg": "SQRT_X_DAG",
        "sxdg": "SQRT_X_DAG",  # Alternative name from transpiler
        "sy": "SQRT_Y",
        "sy_dg": "SQRT_Y_DAG",
        "sydg": "SQRT_Y_DAG",  # Alternative name from transpiler
        "id": "I",
        "i": "I",
    }
    
    two_qubit_gates = {
        "cx": "CX",
        "cnot": "CX",
        "cy": "CY",
        "cz": "CZ",
        "swap": "SWAP",
        "iswap": "ISWAP",
    }
    
    # Process each instruction in the Qrisp circuit
    for instr in qc.data:
        op = instr.op
        op_name = op.name.lower()
        qubits = instr.qubits
        
        # Skip allocation/deallocation operations
        if op_name in ["qb_alloc", "qb_dealloc", "barrier"]:
            continue
        
        # Get qubit indices
        qubit_indices = [qc.qubits.index(q) for q in qubits]
        
        # Handle measurement
        if op_name == "measure" or op_name == "measurement":
            # Qrisp measurements have both qubits and clbits
            clbits = instr.clbits
            if len(qubit_indices) != len(clbits):
                raise ValueError(
                    f"Measurement has {len(qubit_indices)} qubits but {len(clbits)} classical bits"
                )
            
            # Add each measurement and track the mapping
            for qubit_idx, clbit in zip(qubit_indices, clbits):
                # Add measurement to Stim circuit
                stim_circuit.append("M", [qubit_idx])
                
                # Map this Qrisp Clbit object to the current position in Stim's measurement record
                clbit_to_measurement_idx[clbit] = measurement_counter
                measurement_counter += 1
        
        # Handle single-qubit gates
        elif op_name in single_qubit_gates:
            stim_gate = single_qubit_gates[op_name]
            for qubit_idx in qubit_indices:
                stim_circuit.append(stim_gate, [qubit_idx])
        
        # Handle two-qubit gates
        elif op_name in two_qubit_gates:
            if len(qubit_indices) != 2:
                raise ValueError(f"Gate {op_name} requires exactly 2 qubits, got {len(qubit_indices)}")
            stim_gate = two_qubit_gates[op_name]
            stim_circuit.append(stim_gate, qubit_indices)
        
        # Handle reset operations
        elif op_name == "reset":
            # R is reset to |0⟩ state
            stim_circuit.append("R", qubit_indices)
        
        elif isinstance(op, StimError):
            if op.pauli_string is None:
                stim_circuit.append(op.stim_name, qubit_indices, op.params)
            else:
                targets = []
                for i in range(len(qubit_indices)):
                    char = op.pauli_string[i].upper()
                    idx = qubit_indices[i]
                    if char == "I":
                        continue
                    elif char == "X":
                        targets.append(stim.target_x(idx))
                    elif char == "Y":
                        targets.append(stim.target_y(idx))
                    elif char == "Z":
                        targets.append(stim.target_z(idx))
                    else:
                        raise ValueError(f"Unknown Pauli char: {char}")
                
                stim_circuit.append(op.stim_name, targets, op.params)

        # Handle T gate (not a Clifford gate, but check for it)
        elif op_name in ["t", "t_dg"]:
            raise NotImplementedError(
                f"Gate '{op_name}' is not a Clifford gate and cannot be simulated by Stim. "
                "Stim only supports Clifford operations."
            )
        
        # Handle parametric gates (not Clifford)
        elif op_name in ["rx", "ry", "rz", "p", "u1", "u2", "u3", "u"]:
            raise NotImplementedError(
                f"Parametric gate '{op_name}' is not a Clifford gate and cannot be simulated by Stim. "
                "Stim only supports Clifford operations."
            )
        
        # Unknown gate
        else:
            raise ValueError(
                f"Unknown or unsupported gate: {op_name}. "
                f"Stim only supports Clifford gates."
            )
    
    if return_clbit_map:
        # Return both the circuit and the classical bit mapping
        return stim_circuit, clbit_to_measurement_idx
    else:
        return stim_circuit
