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

from jax import make_jaxpr
from qrisp import QuantumVariable, cx, QuantumCircuit, QuantumFloat, x, rz, measure, control, QuantumBool
from qrisp.jasp import qache, flatten_pjit, make_jaspr, ProcessedMeasurement

def test_qc_converter():
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        cx(qv[1], qv[2])
    
    for i in range(3, 7):
        
        jaspr = make_jaspr(test_function)(i)
        qc = jaspr.to_qc(i)
        
        comparison_qc = QuantumCircuit(i)
        comparison_qc.cx(0, 1)
        comparison_qc.cx(1, 2)
        comparison_qc.cx(1, 2)
        
        assert qc.compare_unitary(comparison_qc)
    
    @qache
    def inner_function(qv, i):
        cx(qv[i], qv[i+1])
        cx(qv[i+2], qv[i+1])
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        inner_function(qv, 0)
        inner_function(qv, 1)
        inner_function(qv, 2)
        
    
    jaspr = make_jaspr(test_function)(5)
    qc = jaspr.to_qc(5)
    
    assert len(qc.data) == 3 + len(qc.qubits)
    print(qc)
    comparison_qc = QuantumCircuit(5)
    
    comparison_qc.cx(0, 1)
    comparison_qc.cx(2, 1)
    comparison_qc.cx(1, 2)
    comparison_qc.cx(3, 2)
    comparison_qc.cx(2, 3)
    comparison_qc.cx(4, 3)
    
    print(comparison_qc)
    
    assert qc.compare_unitary(comparison_qc)
    
    jaspr = make_jaspr(test_function)(5)
    flattened_jaspr = flatten_pjit(jaspr)
    
    qc = flattened_jaspr.to_qc(5)
    
    assert qc.compare_unitary(comparison_qc)
    
    ######
    # Test classically controlled Operations
    
    def main():
        
        qf = QuantumFloat(5)
        bl = measure(qf[0])
        
        with control(bl):
            rz(0.5, qf[1])
            x(qf[1])
        
        return

    jaspr = make_jaspr(main)()

    qrisp_qc = jaspr.to_qc()
    qiskit_qc = qrisp_qc.to_qiskit()
    qasm_str = qrisp_qc.to_qasm3()

    assert qasm_str.find("if (cb_0[0]) {") != -1
    
    def main():
        
        qv = QuantumFloat(3)
        qv += 4

    jaspr = make_jaspr(main)()
    str(jaspr.to_qc())
    
    def main():
        
        qv = QuantumFloat(5)
        qbl = QuantumBool()
        with control(qbl):
            qv += 4
        
        return measure(qv) + 5
        
    jaspr = make_jaspr(main)()
    
    assert isinstance(jaspr.to_qc()[0], ProcessedMeasurement)


def test_parity_to_qc():
    """Test parity primitive with to_qc (cl_func_interpreter/qc_extraction_interpreter)."""
    from qrisp.jasp import parity
    
    # Test that parity creates ProcessedMeasurement placeholders in QC extraction
    # (since classical post-processing can't be represented in QuantumCircuit)
    def parity_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])
        
        result = parity(m1, m2, m3)
        return result
    
    jaspr = make_jaspr(parity_test)()
    result, qc = jaspr.to_qc()
    
    # Parity is a post-processing operation, so should return ProcessedMeasurement
    assert isinstance(result, ProcessedMeasurement), f"Expected ProcessedMeasurement, got {type(result)}"
    
    # The quantum circuit should contain the measurements
    assert len(qc.clbits) == 3, f"Expected 3 classical bits, got {len(qc.clbits)}"
    