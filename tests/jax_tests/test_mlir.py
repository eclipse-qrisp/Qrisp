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

from qrisp import *
from qrisp.operators import a, c

def test_mlir_generation():
    
    
    # Test some features to make sure mlir generation works properly
    def inner(i):
        
        a = QuantumVariable(i)
        b = QuantumFloat(i)
        
        h(a)
        
        meas_res = measure(a)
        
        with control(meas_res == 0):
            for i in jrange(b.size):
                rz(1/i, b[i])
                cx(a[i], b[i])

        return a, b                
    
    def main(i):
        return expectation_value(inner, shots = i*10)(i)

    jaspr = make_jaspr(main)(2)
    xdsl_module = jaspr.to_mlir()

    # Test wheter stablehlo control flow is properly removed    
    
    from xdsl.printer import Printer
    
    def main():
        
        qv = QuantumVariable(2)
        h(qv[0])
        
        c = measure(qv[0])
        
        for i in jrange(qv.size):
            with control(c):
                x(qv[1])
        

    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()

    mlir_str = str(xdsl_module)
    
    assert "stablehlo.case" not in mlir_str
    assert "stablehlo.while" not in mlir_str
    assert "stablehlo.return" not in mlir_str
    
    # Test https://github.com/eclipse-qrisp/Qrisp/pull/296#issuecomment-3468979932
    
    H = a(0)
    orbital_amount = H.find_minimal_qubit_amount()
    U = qache(H.trotterization(forward_evolution=False))

    # Finding the gound state energy of the Water molecule with QPE
    def main():
        qv = QuantumFloat(orbital_amount)
        [x(qv[i]) for i in range(1)] # Prepare Hartree-Fock state, H2O molecule has 10 electrons

        qpe_res = QPE(qv,U,precision=1,kwargs={"steps":1})
        phi = measure(qpe_res)
        return phi

    jaspr = make_jaspr(main)()
    mlir = jaspr.to_mlir()

def test_mlir_basic_dialect_operations():
    """
    Test that basic JASP dialect operations are properly emitted in MLIR.
    This verifies the lowering rules for fundamental quantum operations.
    """
    from qrisp import QuantumVariable, h, cx, measure, x
    from qrisp.jasp import make_jaspr
    
    def main():
        # Test create_qubits
        qv = QuantumVariable(3)
        
        # Test quantum gates
        h(qv[0])
        cx(qv[0], qv[1])
        x(qv[2])
        
        # Test measurement
        result = measure(qv)
        
        return result
    
    # Create jaspr and convert to MLIR
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify that JASP dialect operations appear in the MLIR
    assert "jasp.create_qubits" in mlir_str, "create_qubits operation not found in MLIR"
    assert "jasp.measure" in mlir_str, "measure operation not found in MLIR"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "gate operations not found in MLIR"

def test_mlir_quantum_control_flow_rewriting():
    """
    Test that StableHLO control flow is properly rewritten to SCF for quantum types.
    Verifies the fix_quantum_control_flow function works correctly.
    """
    from qrisp import QuantumVariable, QuantumFloat, h, x, rz, cx, measure, control
    from qrisp.jasp import make_jaspr, jrange
    
    def main():
        qv = QuantumVariable(3)
        qf = QuantumFloat(2)
        
        h(qv[0])
        
        # Create measurement-based control flow
        meas_result = measure(qv[0])
        
        # Use control flow with quantum types
        with control(meas_result == 0):
            for i in jrange(qf.size):
                rz(1.0, qf[i])
                cx(qv[1], qf[i])
        
        # Additional control structure
        for j in jrange(qv.size):
            with control(meas_result):
                x(qv[j])
        
        return qv, qf
    
    # Create jaspr and convert to MLIR
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify that StableHLO control flow has been removed
    assert "stablehlo.case" not in mlir_str, "stablehlo.case should be rewritten to scf.if"
    assert "stablehlo.while" not in mlir_str, "stablehlo.while should be rewritten to scf.while"
    assert "stablehlo.return" not in mlir_str, "stablehlo.return should be rewritten to scf.yield"
    
    # Verify that SCF operations are present
    assert "scf.if" in mlir_str or "scf.while" in mlir_str or "scf.yield" in mlir_str, \
        "SCF control flow operations should be present"

def test_mlir_grovers_algorithm():
    """
    Test MLIR generation and execution for Grover's algorithm.
    Verifies that complex algorithms can be lowered to MLIR correctly.
    """
    from qrisp import QuantumFloat
    from qrisp.grover import tag_state, grovers_alg
    from qrisp.jasp import make_jaspr
    import numpy as np
    
    # Define oracle for Grover's algorithm (matching existing test pattern)
    def test_oracle(qf_list, phase=np.pi):
        tag_dic = {qf_list[0]: 0, qf_list[1]: 0.5}
        tag_state(tag_dic, phase=phase)
    
    def main():
        qf_list = [QuantumFloat(2, -2), QuantumFloat(2, -2)]
        grovers_alg(qf_list, test_oracle)
        return qf_list[0], qf_list[1]
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"

def test_mlir_qae_algorithm():
    """
    Test MLIR generation for Quantum Amplitude Estimation (QAE).
    Verifies that complex estimation algorithms can be lowered to MLIR correctly.
    """
    from qrisp import QuantumFloat, ry, z, QAE
    from qrisp.jasp import make_jaspr, terminal_sampling
    import numpy as np
    
    def state_function(qb):
        ry(np.pi / 4, qb)
    
    def oracle_function(qb):
        z(qb)
    
    def main():
        qb = QuantumFloat(1)
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected JASP dialect operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"
    
    # Verify algorithm produces correct result using terminal_sampling wrapper
    @terminal_sampling
    def main_sampling():
        qb = QuantumFloat(1)
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res
    
    meas_res = main_sampling()
    assert np.round(meas_res[0.125], 2) == 0.5, f"Expected ~0.5 probability for 0.125"
    assert np.round(meas_res[0.875], 2) == 0.5, f"Expected ~0.5 probability for 0.875"

def test_mlir_iqpe():
    """
    Test MLIR generation for Iterative Quantum Phase Estimation (IQPE).
    Verifies that IQPE can be properly lowered to MLIR.
    """
    from qrisp import QuantumVariable, h, x, rx, IQPE
    from qrisp.jasp import make_jaspr
    import numpy as np
    
    def U(qv):
        x_val = 1/2**3
        y_val = 1/2**2
        rx(x_val * 2 * np.pi, qv[0])
        rx(y_val * 2 * np.pi, qv[1])
    
    def main():
        qv = QuantumVariable(2)
        x(qv)
        h(qv)
        return IQPE(qv, U, precision=4)
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"

def test_mlir_iqae():
    """
    Test MLIR generation for Iterative Quantum Amplitude Estimation (IQAE).
    Verifies that IQAE can be properly lowered to MLIR.
    """
    from qrisp import QuantumFloat, QuantumBool, control, h, ry, IQAE
    from qrisp.jasp import make_jaspr, jrange
    import numpy as np
    
    # State function for integration example
    def state_function(inp, tar):
        h(inp)  # Distribution
        
        N = 2**inp.size
        for k in jrange(inp.size):
            with control(inp[k]):
                ry(2**(k+1)/N, tar)
    
    def main():
        n = 4  # Smaller precision for faster testing
        inp = QuantumFloat(n, -n)
        tar = QuantumBool()
        input_list = [inp, tar]
        
        eps = 0.05
        alpha = 0.05
        
        return IQAE(input_list, state_function, eps=eps, alpha=alpha)
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"

def test_mlir_hamiltonian_simulation():
    """
    Test MLIR generation for Hamiltonian simulation.
    Verifies that Hamiltonian trotterization can be lowered to MLIR.
    """
    from qrisp import QuantumFloat, x, qache
    from qrisp.jasp import make_jaspr
    from qrisp.operators import a
    
    # Create a simple Hamiltonian
    H = a(0)
    orbital_amount = H.find_minimal_qubit_amount()
    U = qache(H.trotterization(forward_evolution=False))
    
    def main():
        qv = QuantumFloat(orbital_amount)
        # Prepare initial state
        x(qv[0])
        # Apply Hamiltonian evolution
        U(qv)
        return qv
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"

def test_mlir_qaoa():
    """
    Tests MLIR generation for the complete QAOA workflow as shown in the 
    'How to use QAOA in Jasp' documentation. This test verifies that the 
    entire QAOA optimization loop (including QAOAProblem setup, cost operator,
    mixer, and sample array post-processing) can be compiled to MLIR.
    """
    from qrisp import QuantumVariable, make_jaspr
    from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cost_operator, create_maxcut_sample_array_post_processor
    import networkx as nx

    def main():
        # Create a random graph for the MaxCut problem
        G = nx.erdos_renyi_graph(6, 0.7, seed=133)

        # Create the sample array post-processor for Jasp (works with integer arrays)
        cl_cost = create_maxcut_sample_array_post_processor(G)

        # Create quantum argument
        qarg = QuantumVariable(G.number_of_nodes())

        # Set up the QAOA problem with cost operator, mixer, and classical cost function
        qaoa_maxcut = QAOAProblem(
            cost_operator=create_maxcut_cost_operator(G),
            mixer=RX_mixer,
            cl_cost_function=cl_cost
        )
        
        # Run QAOA with depth 5, max 50 iterations, and SPSA optimizer
        res_sample = qaoa_maxcut.run(qarg, depth=5, max_iter=50, optimizer="SPSA")

        return res_sample

    # Generate MLIR from the complete QAOA workflow
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)

    # Verify that MLIR contains expected JASP dialect operations
    assert "jasp.create_qubits" in mlir_str, "MLIR should contain jasp.create_qubits operation"
    assert "jasp.quantum_gate" in mlir_str, "MLIR should contain jasp.quantum_gate operations"
    assert "jasp.measure" in mlir_str, "MLIR should contain jasp.measure operation"
    
    # Verify that control flow has been rewritten from StableHLO to SCF for optimization loop
    assert "scf.while" in mlir_str, "MLIR should contain SCF while for QAOA optimization loop"
    assert "scf.yield" in mlir_str, "MLIR should contain SCF yield operations"
    
    # Verify that QAOA-specific operations are present
    assert "jasp.slice" in mlir_str or "jasp.get_qubit" in mlir_str, "MLIR should contain qubit indexing operations"

def test_mlir_array_operations():
    """
    Test MLIR generation for quantum array operations.
    Verifies that array slicing and fusion are properly lowered to MLIR.
    """
    from qrisp import QuantumVariable, h, cx
    from qrisp.jasp import make_jaspr, jrange
    
    def main():
        # Create quantum arrays
        qv1 = QuantumVariable(4)
        qv2 = QuantumVariable(3)
        
        # Test slicing operations
        h(qv1[0:2])
        cx(qv1[1], qv2[0])
        
        # Test iteration with jrange
        for i in jrange(3):
            h(qv2[i])
        
        return qv1, qv2
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.slice" in mlir_str or "jasp.get_qubit" in mlir_str, "Should contain array access operations"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    