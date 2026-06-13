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

from qrisp import QuantumCircuit, Qubit, U3Gate, PRYGate, convert_to_pry
import numpy as np
def test_convert_to_pry():
    """Test the convert_to_pry function with various quantum circuits."""
    
    print("Testing convert_to_pry...")
    
    # Test case 1: Simple single-qubit gate conversion
    qc = QuantumCircuit(2)
    
    # Add a U3 gate that needs conversion
    theta, phi, lam = np.pi/2, np.pi/4, np.pi/8
    u3_gate = U3Gate(theta, phi, lam)
    qc.append(u3_gate, 0)
    qc.reset(0)
    qc.measure(0)
    qc.cz(0, 1)
    qc.barrier([0,1])
    
    qc_converted = convert_to_pry(qc)
    
    assert hasattr(qc_converted, 'num_qubits'), "Result should be a QuantumCircuit"
    assert qc_converted.num_qubits() == qc.num_qubits(), "Should preserve number of qubits"
    
    # Check that the conversion happened
    converted_ops = [instr.op for instr in qc_converted.data]
    pry_ops = [op for op in converted_ops if isinstance(op, PRYGate)]
    
    assert len(pry_ops)
    for op in converted_ops:
        if isinstance(op, U3Gate) and not isinstance(op, PRYGate):
            assert False, "Left non-converted U3 Gate"

    # For a general U3 gate where lam ≠ -phi, we expect 2 PRY gates
    if abs(lam + phi) >= 1E-5:
        assert len(pry_ops) >= 1, "Should have at least one PRYGate for general U3"
    
    print(f"✓ Single-qubit conversion: {len(converted_ops)} operations, {len(pry_ops)} PRY gates")
    
    # Test case 2: U3 gate with lam = -phi (special case)
    qc = QuantumCircuit()
    q0 = Qubit("q0")
    qc.add_qubit(q0)
    
    theta, phi = np.pi/3, np.pi/6
    lam = -phi  # Special case where lam = -phi
    u3_special = U3Gate(theta, phi, lam)
    qc.append(u3_special, [q0])
    
    qc_converted = convert_to_pry(qc)
    
    converted_ops = [instr.op for instr in qc_converted.data]
    pry_ops = [op for op in converted_ops if isinstance(op, PRYGate)]
    
    # Should have exactly one PRY gate for this special case
    assert len(pry_ops) == 1, f"Special case should have 1 PRY gate, got {len(pry_ops)}"
    assert abs(pry_ops[0].theta - theta) < 1E-10, "PRY theta should match original"
    assert abs(pry_ops[0].phi - phi) < 1E-10, "PRY phi should match original"
    
    print("✓ Special case (lam=-phi) conversion passed")
    
    # Test case 3: Two-qubit gates should be preserved unchanged
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(2)]
    for q in qubits:
        qc.add_qubit(q)
    
    # Add two-qubit gate
    qc.cx(qubits[0], qubits[1])
    
    # Add single-qubit gate that will be converted
    qc.h(qubits[0])
    
    qc_converted = convert_to_pry(qc)
    
    # Should have the CX gate unchanged plus PRY conversion of H gate
    cx_gates = [instr for instr in qc_converted.data if instr.op.name == "cx"]
    assert len(cx_gates) == 1, "CX gate should be preserved"
    
    print("✓ Two-qubit gate preservation test passed")
    
    # Test case 4: Empty circuit
    empty_qc = QuantumCircuit()
    q0 = Qubit("q0")
    empty_qc.add_qubit(q0)
    
    qc_converted = convert_to_pry(empty_qc)
    
    assert len(qc_converted.data) == 0, "Empty circuit should remain empty"
    assert qc_converted.num_qubits() == 1, "Should preserve qubits"
    
    print("✓ Empty circuit test passed")
    
    print("✓ All convert_to_pry tests passed!")


if __name__ == "__main__":
    test_convert_to_pry()