// Reference module exercising the custom assembly format of every operation in
// the JASP dialect, with one instance per syntactic shape.
//
// This file is the textual contract of the dialect.  The same syntax is defined
// twice: by `assembly_format` in src/qrisp/jasp/mlir/xdsl_dialect.py (which is
// what Qrisp prints) and by `assemblyFormat` in
// src/qrisp/jasp/mlir/dialect_definition/JaspOps.td (which is what an MLIR-based
// consumer builds its parser from).  Whenever either is touched, update this
// file to match -- tests/jax_tests/test_mlir_roundtrip.py parses it back with
// xDSL, and it is written so that it can also be fed to `mlir-opt` by a
// consumer that has built the dialect from the TableGen files.

builtin.module {
  func.func private @jasp_dialect_syntax(%n: tensor<i64>, %i: tensor<i64>, %j: tensor<i64>, %theta: tensor<f64>) -> (tensor<i1>, tensor<i1>) {
    %qst0 = jasp.create_quantum_kernel -> !jasp.QuantumState
    %qb_array, %qst1 = jasp.create_qubits %n, %qst0 : tensor<i64>, !jasp.QuantumState -> !jasp.QubitArray, !jasp.QuantumState
    %size = jasp.get_size %qb_array : !jasp.QubitArray -> tensor<i64>
    %q0 = jasp.get_qubit %qb_array, %i : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %q1 = jasp.get_qubit %qb_array, %j : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %sliced = jasp.slice %qb_array, %i, %j : !jasp.QubitArray, tensor<i64>, tensor<i64> -> !jasp.QubitArray
    %fused = jasp.fuse %q0, %sliced : !jasp.Qubit, !jasp.QubitArray -> !jasp.QubitArray

    // Gate operands are the qubits the gate acts on, followed by the float
    // parameters of parametric gates.
    %qst2 = jasp.quantum_gate "h"(%q0), %qst1 : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    %qst3 = jasp.quantum_gate "cx"(%q0, %q1), %qst2 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    %qst4 = jasp.quantum_gate "rz"(%q0, %theta), %qst3 : (!jasp.Qubit, tensor<f64>), !jasp.QuantumState -> !jasp.QuantumState

    %qst5 = jasp.reset %fused, %qst4 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState

    // Measuring a single Qubit yields tensor<i1>, measuring a QubitArray
    // yields tensor<i64>.
    %m0, %qst6 = jasp.measure %q0, %qst5 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    %m1, %qst7 = jasp.measure %q1, %qst6 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    %m2, %qst8 = jasp.measure %qb_array, %qst7 : !jasp.QubitArray, !jasp.QuantumState -> tensor<i64>, !jasp.QuantumState

    %parity = jasp.parity %m0, %m1 {expectation = 1 : i64, observable = 0 : i64} : tensor<i1>, tensor<i1> -> tensor<i1>

    %qst9 = jasp.delete_qubits %qb_array, %qst8 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState
    %success = jasp.consume_quantum_kernel %qst9 : !jasp.QuantumState -> tensor<i1>

    func.return %parity, %success : tensor<i1>, tensor<i1>
  }
}
