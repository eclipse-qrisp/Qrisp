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

from qrisp import QuantumCircuit, Qubit, U3Gate, convert_to_prx
from qrisp.circuit import PRXGate
import numpy as np


def test_convert_to_prx():
    """Test the convert_to_prx function with various quantum circuits."""

    print("Testing convert_to_prx...")

    # Test case 1: General U3 -> two PRX gates
    qc = QuantumCircuit(2)

    theta, phi, lam = np.pi / 2, np.pi / 4, np.pi / 8
    u3_gate = U3Gate(theta, phi, lam)
    qc.append(u3_gate, 0)
    qc.cz(0, 1)
    qc.barrier([0, 1])

    qc_converted = convert_to_prx(qc)

    assert hasattr(qc_converted, 'num_qubits'), "Result should be a QuantumCircuit"
    assert qc_converted.num_qubits() == qc.num_qubits(), "Should preserve number of qubits"

    converted_ops = [instr.op for instr in qc_converted.data]
    prx_ops = [op for op in converted_ops if isinstance(op, PRXGate)]

    assert len(prx_ops) >= 1, f"Should have at least one PRXGate, got {len(prx_ops)}"

    # No leftover non-PRX U3 gates (gphase is allowed — it tracks global phase)
    for op in converted_ops:
        if isinstance(op, U3Gate) and not isinstance(op, PRXGate) and op.name != "gphase":
            assert False, f"Leftover non-PRX U3Gate: {op.name}"

    # Unitary must match exactly (global phase is tracked)
    assert np.allclose(qc.get_unitary(), qc_converted.get_unitary(), atol=1E-6), \
        "Converted circuit unitary must match original"

    print(f"  General U3 conversion: {len(converted_ops)} operations, {len(prx_ops)} PRX gates")

    # Test case 2: Single PRX case (lam = -phi)
    qc = QuantumCircuit()
    q0 = Qubit("q0")
    qc.add_qubit(q0)

    theta, phi = np.pi / 3, np.pi / 6
    lam = -phi  # Special case: lam = -phi
    u3_special = U3Gate(theta, phi, lam)
    qc.append(u3_special, [q0])

    qc_converted = convert_to_prx(qc)

    converted_ops = [instr.op for instr in qc_converted.data]
    prx_ops = [op for op in converted_ops if isinstance(op, PRXGate)]

    # Should have exactly one PRX gate (plus possibly a gphase gate)
    assert len(prx_ops) == 1, f"Special case should have 1 PRX gate, got {len(prx_ops)}"
    assert abs(prx_ops[0].alpha - theta) < 1E-10, "PRX alpha should match theta"
    assert abs(prx_ops[0].beta - (phi + np.pi / 2)) < 1E-10, \
        f"PRX beta should be phi + pi/2 = {phi + np.pi/2}, got {prx_ops[0].beta}"

    # No spurious gphase for this case (no global phase introduced)
    gphase_ops = [op for op in converted_ops if op.name == "gphase"]
    assert len(gphase_ops) == 0, "Single PRX case should not emit a gphase gate"

    assert np.allclose(qc.get_unitary(), qc_converted.get_unitary(), atol=1E-6)

    print("  Single PRX case (lam=-phi) test passed")

    # Test case 3: Two-qubit gates preserved unchanged
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(2)]
    for q in qubits:
        qc.add_qubit(q)

    qc.cx(qubits[0], qubits[1])
    qc.h(qubits[0])

    qc_converted = convert_to_prx(qc)

    # CX gate should be preserved
    cx_gates = [instr for instr in qc_converted.data if instr.op.name == "cx"]
    assert len(cx_gates) == 1, "CX gate should be preserved"

    # H gate should be converted to PRX
    prx_ops = [instr.op for instr in qc_converted.data if isinstance(instr.op, PRXGate)]
    assert len(prx_ops) >= 1, "H gate should be converted to PRX gate(s)"

    assert np.allclose(qc.get_unitary(), qc_converted.get_unitary(), atol=1E-6)

    print("  Two-qubit gate preservation test passed")

    # Test case 4: Empty circuit
    empty_qc = QuantumCircuit()
    q0 = Qubit("q0")
    empty_qc.add_qubit(q0)

    qc_converted = convert_to_prx(empty_qc)

    # Empty circuit stays empty (no gphase either since no decompositions happened)
    assert len(qc_converted.data) == 0, "Empty circuit should remain empty"
    assert qc_converted.num_qubits() == 1, "Should preserve qubits"

    print("  Empty circuit test passed")

    # Test case 5: Global phase tracking
    qc_g = QuantumCircuit(1)
    # U3 with explicit global phase
    op_with_phase = U3Gate(0.5, 0.3, 0.7, global_phase=1.2)
    qc_g.append(op_with_phase, 0)

    qc_converted = convert_to_prx(qc_g)

    # Should have a gphase gate
    gphase_ops = [instr for instr in qc_converted.data if instr.op.name == "gphase"]
    assert len(gphase_ops) == 1, "Should emit a gphase gate for global phase"
    assert abs(gphase_ops[0].op.params[0]) > 1E-10, "Gphase should be non-trivial"

    # Unitary must match exactly
    assert np.allclose(qc_g.get_unitary(), qc_converted.get_unitary(), atol=1E-6)

    print("  Global phase tracking test passed")

    # Test case 6: Identity U3 with global phase
    qc_id = QuantumCircuit(1)
    op_id = U3Gate(0, 0.3, -0.3, global_phase=1.5)
    qc_id.append(op_id, 0)

    qc_converted = convert_to_prx(qc_id)

    # Should have a gphase gate but no PRX gates (since theta=0 => identity)
    gphase_ops = [instr for instr in qc_converted.data if instr.op.name == "gphase"]
    prx_ops = [instr.op for instr in qc_converted.data if isinstance(instr.op, PRXGate)]

    assert len(gphase_ops) == 1, "Identity with phase should emit gphase"
    assert len(prx_ops) == 0, "Identity U3 should not emit PRX gates"

    assert np.allclose(qc_id.get_unitary(), qc_converted.get_unitary(), atol=1E-6)

    print("  Identity U3 with global phase test passed")

    # Test case 7: Random circuit unitary fidelity
    np.random.seed(123)
    for _ in range(50):
        n_qubits = np.random.randint(1, 4)
        qc_r = QuantumCircuit(n_qubits)

        for _ in range(np.random.randint(1, 10)):
            qb = np.random.randint(0, n_qubits)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            lam = np.random.uniform(0, 2 * np.pi)
            qc_r.u3(theta, phi, lam, qb)

            if n_qubits >= 2:
                qb2 = (qb + 1) % n_qubits
                qc_r.cz(qb, qb2)

        result_r = convert_to_prx(qc_r)
        assert np.allclose(qc_r.get_unitary(), result_r.get_unitary(), atol=1E-6), \
            f"Random circuit unitary mismatch"

        # Verify all single-qubit gates are PRXGate (or gphase)
        for instr in result_r.data:
            op = instr.op
            if op.name in ("qb_alloc", "qb_dealloc", "barrier"):
                continue
            if op.num_qubits == 1 and op.num_clbits == 0:
                assert isinstance(op, PRXGate) or op.name == "gphase", \
                    f"Non-PRX single-qubit gate found: {op.name}"

    print("  Random circuit fidelity test passed (50 circuits)")

    print("All convert_to_prx tests passed!")
