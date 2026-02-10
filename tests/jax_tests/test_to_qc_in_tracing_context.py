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

Tests for calling Jaspr.to_qc inside a JAX tracing context.

The eval_context() wrapper in jaspr_to_qc ensures that the qc_extraction
interpreter evaluates primitives concretely even when called from within
an outer make_jaspr or jit tracing context. Without this, classical JAX
primitives like slice, squeeze, mul, etc. would be captured by the outer
trace instead of being evaluated, leading to TracerIntegerConversionError.
"""

import numpy as np
from qrisp import (
    QuantumVariable, QuantumFloat, QuantumBool, QuantumArray,
    QuantumCircuit, cx, x, h, measure, Clbit
)
from qrisp.jasp import make_jaspr, qache
from qrisp.jasp.interpreter_tools.interpreters import jaspr_to_qc
from qrisp.circuit import QuantumCircuit as QC


def _build_to_qc_args(inner_jaspr, num_qubits):
    """
    Build concrete arguments for calling to_qc on an inner Jaspr.
    
    Inspects the Jaspr's invars (excluding the trailing QuantumCircuit)
    to build appropriate concrete arguments. Handles both QuantumVariable
    (invars: [QubitArray, QC]) and QuantumArray (invars: [int, int[n],
    QubitArray, int, QC]) patterns.
    """
    analysis_qc = QC(num_qubits)
    qubit_list = list(analysis_qc.qubits)
    
    # invars minus the trailing QuantumCircuit (added by to_qc)
    invars = inner_jaspr.invars[:-1]
    
    args = []
    for v in invars:
        aval = v.aval
        aval_str = str(aval)
        if aval_str == "QubitArray":
            args.append(qubit_list)
        elif "int" in aval_str and hasattr(aval, "shape") and aval.shape:
            # int64[n] array — the index array
            args.append(np.arange(num_qubits))
        elif "int" in aval_str:
            # scalar int — qtype_size or template_child (both 1 for QuantumBool)
            args.append(1)
        else:
            raise ValueError(f"Unexpected invar type: {aval_str}")
    
    return args


def test_to_qc_standalone():
    """Baseline: to_qc works outside any tracing context (unchanged behavior)."""
    
    def bell_pair(n):
        qv = QuantumVariable(n)
        h(qv[0])
        cx(qv[0], qv[1])
        return qv
    
    jaspr = make_jaspr(bell_pair)(2)
    qubits, qc = jaspr.to_qc(2)
    
    # Should produce H on qubit 0, CX on (0, 1)
    assert len(qc.qubits) == 2
    gate_names = [instr.op.name for instr in qc.data]
    assert "h" in gate_names
    assert "cx" in gate_names


def test_to_qc_inside_make_jaspr():
    """
    Call to_qc on a sub-jaspr from within a make_jaspr tracing context.
    
    This is the core use case: tracing an inner function to get its Jaspr,
    then calling to_qc to analyze its circuit structure, all while
    the outer function is itself being traced by make_jaspr.
    """
    
    def inner_func(qv):
        cx(qv[0], qv[1])
        h(qv[1])
        return measure(qv[1])
    
    @make_jaspr
    def outer_func():
        qv = QuantumVariable(3)
        
        # Trace inner function to get its Jaspr
        inner_jaspr = make_jaspr(inner_func)(qv)
        
        # Build concrete analysis args
        analysis_qc = QC(3)
        qubit_list = list(analysis_qc.qubits)
        
        # QuantumVariable jaspr invars: [QubitArray, QuantumCircuit]
        # to_qc adds the QC, so we pass just the QubitArray
        result = inner_jaspr.to_qc(qubit_list)
        
        # result = (*return_values, qc)
        analysis_qc_out = result[-1]
        
        # Verify analysis circuit has the right gates
        gate_names = [instr.op.name for instr in analysis_qc_out.data]
        assert "cx" in gate_names, f"Expected cx in {gate_names}"
        assert "h" in gate_names, f"Expected h in {gate_names}"
        
        # Now actually run the inner function in the outer trace
        res = inner_func(qv)
        return res
    
    # This should succeed without TracerIntegerConversionError
    jaspr = outer_func()
    assert jaspr is not None


def test_to_qc_inside_make_jaspr_with_measurements():
    """
    Call to_qc inside tracing and verify measurement Clbit objects
    are returned correctly from the analysis.
    """
    
    def measure_func(qv):
        h(qv[0])
        m = measure(qv[0])
        return m
    
    @make_jaspr
    def outer_func():
        qv = QuantumVariable(2)
        
        inner_jaspr = make_jaspr(measure_func)(qv)
        
        analysis_qc = QC(2)
        qubit_list = list(analysis_qc.qubits)
        
        # QuantumVariable: invars = [QubitArray, QC]
        result = inner_jaspr.to_qc(qubit_list)
        
        analysis_qc_out = result[-1]
        returned_clbit = result[0]
        
        # The returned measurement should be a Clbit
        assert isinstance(returned_clbit, Clbit), (
            f"Expected Clbit, got {type(returned_clbit)}"
        )
        
        # analysis_qc_out should have a measurement
        gate_names = [instr.op.name for instr in analysis_qc_out.data]
        assert "measure" in gate_names, f"Expected measure in {gate_names}"
        
        # Actually run the function for the outer trace
        res = measure_func(qv)
        return res
    
    jaspr = outer_func()
    assert jaspr is not None


def test_to_qc_inside_make_jaspr_quantum_array():
    """
    Call to_qc with QuantumArray arguments inside tracing context.
    
    This tests the full pattern with QuantumArray (multiple QuantumBool),
    which produces more complex Jaspr invars including ind_array and
    qtype_size parameters.
    """
    
    def syndrome_circuit(qa):
        """Simple syndrome extraction: 2 data + 1 ancilla."""
        cx(qa[0], qa[2])
        cx(qa[1], qa[2])
        return measure(qa[2])
    
    @make_jaspr
    def outer_func():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        
        inner_jaspr = make_jaspr(syndrome_circuit)(qa)
        
        num_qubits = 3  # qa.size * qa.qtype.size = 3 * 1
        args = _build_to_qc_args(inner_jaspr, num_qubits)
        result = inner_jaspr.to_qc(*args)
        
        analysis_qc_out = result[-1]
        
        # Should have 2 CX gates and 1 measurement
        gate_names = [instr.op.name for instr in analysis_qc_out.data]
        cx_count = gate_names.count("cx")
        assert cx_count == 2, f"Expected 2 cx gates, got {cx_count}"
        assert "measure" in gate_names
        
        # Actually call the function
        res = syndrome_circuit(qa)
        return res
    
    jaspr = outer_func()
    assert jaspr is not None


def test_to_qc_stim_conversion_inside_tracing():
    """
    Full pipeline: to_qc -> to_stim inside tracing context.
    
    Verifies that the QuantumCircuit produced by to_qc can be further
    converted to a stim circuit, all within a make_jaspr tracing context.
    """
    
    def simple_circuit(qv):
        h(qv[0])
        cx(qv[0], qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        return m0, m1
    
    @make_jaspr
    def outer_func():
        qv = QuantumVariable(2)
        
        inner_jaspr = make_jaspr(simple_circuit)(qv)
        
        analysis_qc = QC(2)
        qubit_list = list(analysis_qc.qubits)
        
        result = inner_jaspr.to_qc(qubit_list)
        analysis_qc_out = result[-1]
        
        # Convert to stim
        stim_circ = analysis_qc_out.to_stim()
        
        # Should have 2 measurements
        assert stim_circ.num_measurements == 2, (
            f"Expected 2 measurements, got {stim_circ.num_measurements}"
        )
        
        # Run the actual function
        res = simple_circuit(qv)
        return res
    
    jaspr = outer_func()
    assert jaspr is not None


def test_to_qc_existing_behavior_unchanged():
    """
    Verify that normal (non-tracing) to_qc behavior is unchanged after
    the eval_context addition.
    """
    
    @qache
    def inner_function(qv, i):
        cx(qv[i], qv[i+1])
    
    def test_function(n):
        qv = QuantumVariable(n)
        inner_function(qv, 0)
        inner_function(qv, 1)
        return qv
    
    for n in range(3, 6):
        jaspr = make_jaspr(test_function)(n)
        result = jaspr.to_qc(n)
        qc = result[-1] if isinstance(result, tuple) else result
        
        # Should have n qubits
        assert len(qc.qubits) == n
        
        # Should have 2 CX gates
        gate_names = [instr.op.name for instr in qc.data]
        cx_count = gate_names.count("cx")
        assert cx_count == 2, f"Expected 2 cx, got {cx_count} for n={n}"