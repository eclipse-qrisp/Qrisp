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

import numpy as np
from sympy import Add, Expr, Mul, Pow

from qrisp.jasp.primitives import (
    create_qubits_p,
    get_qubit_p,
    get_size_p,
    slice_p,
    fuse_p,
    reset_p,
    Measurement_p,
    delete_qubits_p,
    create_quantum_kernel_p,
    consume_quantum_kernel_p,
    quantum_gate_p,
    parity_p,
    AbstractQuantumState,
    AbstractQubitArray,
    AbstractQubit,
)
from qrisp.jasp.primitives.operation_primitive import greek_letters as _greek_letters


import qrisp.jasp.mlir.dialect_implementation._jasp_ops_gen as jasp_dialect

from jax.interpreters import mlir as jax_mlir
from jax.interpreters.mlir import ir_type_handlers
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

##########################
# Register type lowering #
##########################


# These types need to be callable since the can only be
# created when inside an MLIR Context manager
def get_ir_qst_type():
    return ir.Type.parse("!jasp.QuantumState")


def get_ir_qa_type():
    return ir.Type.parse("!jasp.QubitArray")


def get_ir_qb_type():
    return ir.Type.parse("!jasp.Qubit")


# Define builtin integer type functions
def get_i64_type():
    return ir.IntegerType.get_signless(64)


def get_i1_type():
    return ir.IntegerType.get_signless(1)


# Register in ir_type_handlers


# Register type lowering (like Catalyst does)
def _aqc_lowering(aval):
    assert isinstance(aval, AbstractQuantumState)
    return get_ir_qst_type()


def _aqa_lowering(aval):
    assert isinstance(aval, AbstractQubitArray)
    return get_ir_qa_type()


def _aqb_lowering(aval):
    assert isinstance(aval, AbstractQubit)
    return get_ir_qb_type()


# Register in ir_type_handlers
ir_type_handlers[AbstractQuantumState] = _aqc_lowering
ir_type_handlers[AbstractQubitArray] = _aqa_lowering
ir_type_handlers[AbstractQubit] = _aqb_lowering


_greek_letter_order = {sym: i for i, sym in enumerate(_greek_letters)}


def _as_ir_value(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"Expected a single MLIR value, got {len(value)} values")
        return value[0]
    return value


def _sorted_param_symbols(param_exprs):
    symbols = set()
    for expr in param_exprs:
        if isinstance(expr, Expr):
            symbols.update(expr.free_symbols)
    return sorted(symbols, key=lambda sym: _greek_letter_order.get(sym, len(_greek_letters)))


def _lower_f64_constant(ctx, value):
    return _as_ir_value(
        jax_mlir.ir_constant(
            np.asarray(float(value), dtype=np.float64),
            const_lowering=ctx.const_lowering,
        )
    )


def _lower_gate_param_expr(ctx, expr, symbol_bindings):
    """Lower a sympy gate-parameter expression to a tensor<f64> SSA value.

    quantum_gate_p stores gate parameters symbolically in ``gate.params`` while the
    runtime tracer operands are provided positionally. MLIR lowering must rebuild
    the symbolic expression explicitly; otherwise a gate such as ``gphase(-alpha/2)``
    is emitted as if its parameter were just ``alpha``.
    """
    if isinstance(expr, np.number):
        expr = expr.item()

    if isinstance(expr, (int, float)):
        return _lower_f64_constant(ctx, expr)

    if isinstance(expr, Expr):
        if expr.is_Symbol:
            try:
                return symbol_bindings[expr]
            except KeyError as exc:
                raise KeyError(f"Missing MLIR operand for symbolic gate parameter {expr}") from exc

        if not expr.free_symbols:
            return _lower_f64_constant(ctx, expr)

        if isinstance(expr, Add):
            values = [_lower_gate_param_expr(ctx, arg, symbol_bindings) for arg in expr.args]
            acc = values[0]
            for value in values[1:]:
                acc = stablehlo.AddOp(acc, value).result
            return acc

        if isinstance(expr, Mul):
            values = [_lower_gate_param_expr(ctx, arg, symbol_bindings) for arg in expr.args]
            acc = values[0]
            for value in values[1:]:
                acc = stablehlo.MulOp(acc, value).result
            return acc

        if isinstance(expr, Pow):
            base_expr, exponent_expr = expr.args

            if isinstance(exponent_expr, Expr) and exponent_expr.free_symbols:
                raise NotImplementedError(
                    f"Dynamic exponent in gate parameter expression is not supported: {expr}"
                )

            exponent_value = float(exponent_expr)
            if not exponent_value.is_integer():
                raise NotImplementedError(
                    f"Non-integer powers in gate parameter expressions are not supported: {expr}"
                )

            exponent = int(exponent_value)
            if exponent == 0:
                return _lower_f64_constant(ctx, 1.0)

            base_value = _lower_gate_param_expr(ctx, base_expr, symbol_bindings)
            power_value = base_value
            for _ in range(abs(exponent) - 1):
                power_value = stablehlo.MulOp(power_value, base_value).result

            if exponent < 0:
                return stablehlo.DivOp(_lower_f64_constant(ctx, 1.0), power_value).result
            return power_value

    raise NotImplementedError(
        f"Unsupported gate parameter expression for MLIR lowering: {expr!r}"
    )


###############################
# Register Primitive lowering #
###############################

lowering_rules = []


def create_qubits_lowering(ctx, amount, qst_in):
    """
    Lowering rule that emits our CreateQubits dialect operation.
    """
    # Create our create_qubits operation using the generated class
    create_qubits_op = jasp_dialect.CreateQubitsOp(
        get_ir_qa_type(), get_ir_qst_type(), amount, qst_in
    )
    # Return both results: QubitArray and QuantumState as list
    return [create_qubits_op.results[0], create_qubits_op.results[1]]


lowering_rules.append((create_qubits_p, create_qubits_lowering))


def get_qubit_lowering(ctx, qb_array, position):
    """
    Lowering rule that emits our GetQubit dialect operation.
    """
    # Create our get_qubit operation using the generated class
    get_qubit_op = jasp_dialect.GetQubitOp(get_ir_qb_type(), qb_array, position)
    # Return the result of our operation
    return [get_qubit_op.results[0]]


lowering_rules.append((get_qubit_p, get_qubit_lowering))


def get_size_lowering(ctx, qb_array):
    """
    Lowering rule that emits our GetSize dialect operation.
    """
    # Create scalar tensor type for i64 result
    i64_result_type = ir.RankedTensorType.get(
        [], get_i64_type()
    )  # scalar tensor of i64

    # Create our get_size operation using the generated class
    get_size_op = jasp_dialect.GetSizeOp(i64_result_type, qb_array)
    # Return the result of our operation
    return [get_size_op.results[0]]


lowering_rules.append((get_size_p, get_size_lowering))


def slice_lowering(ctx, qb_array, start, end):
    """
    Lowering rule that emits our Slice dialect operation.
    """
    # Create our slice operation using the generated class
    slice_op = jasp_dialect.SliceOp(get_ir_qa_type(), qb_array, start, end)
    # Return the result QubitArray
    return [slice_op.results[0]]


lowering_rules.append((slice_p, slice_lowering))


def fuse_lowering(ctx, operand1, operand2):
    """
    Lowering rule that emits our Fuse dialect operation.
    """
    # Create our fuse operation using the generated class
    fuse_op = jasp_dialect.FuseOp(get_ir_qa_type(), operand1, operand2)
    # Return the result QubitArray
    return [fuse_op.results[0]]


lowering_rules.append((fuse_p, fuse_lowering))


def reset_lowering(ctx, qubits, in_qst):
    """
    Lowering rule that emits our Reset dialect operation.
    """
    # Create our reset operation using the generated class
    reset_op = jasp_dialect.ResetOp(get_ir_qst_type(), qubits, in_qst)
    # Return the result quantum state
    return [reset_op.results[0]]


lowering_rules.append((reset_p, reset_lowering))


def measure_lowering(ctx, meas_q, in_qst):
    """
    Lowering rule that emits our Measure dialect operation.
    """
    if meas_q.type == get_ir_qa_type():
        # For QubitArray measurement, result is tensor<i64>
        res_type = ir.RankedTensorType.get([], get_i64_type())  # scalar tensor of i64
    elif meas_q.type == get_ir_qb_type():
        # For single Qubit measurement, result is tensor<i1>
        res_type = ir.RankedTensorType.get([], get_i1_type())  # scalar tensor of i1
    else:
        raise Exception(f"Unknown qubit type: {meas_q.type}")

    # Create our measure operation using the generated class
    measure_op = jasp_dialect.MeasureOp(res_type, get_ir_qst_type(), meas_q, in_qst)
    # Return both results: measurement result and quantum state
    return [measure_op.results[0], measure_op.results[1]]


lowering_rules.append((Measurement_p, measure_lowering))


def delete_qubits_lowering(ctx, qubits, in_qst):
    """
    Lowering rule that emits our DeleteQubits dialect operation.
    """
    # Create our delete_qubits operation using the generated class
    delete_qubits_op = jasp_dialect.DeleteQubitsOp(get_ir_qst_type(), qubits, in_qst)
    # Return the result quantum state
    return [delete_qubits_op.results[0]]


lowering_rules.append((delete_qubits_p, delete_qubits_lowering))


def create_quantum_kernel_lowering(ctx):
    """
    Lowering rule that emits our CreateQuantumKernel dialect operation.
    """
    # Create our create_quantum_kernel operation using the generated class
    create_quantum_kernel_op = jasp_dialect.CreateQuantumKernelOp(get_ir_qst_type())
    # Return the result quantum state
    return [create_quantum_kernel_op.results[0]]


lowering_rules.append((create_quantum_kernel_p, create_quantum_kernel_lowering))


def consume_quantum_kernel_lowering(ctx, qst):
    """
    Lowering rule that emits our ConsumeQuantumKernel dialect operation.
    """
    # Create scalar tensor type for boolean result
    bool_result_type = ir.RankedTensorType.get([], get_i1_type())  # scalar tensor of i1

    # Create our consume_quantum_kernel operation using the generated class
    consume_quantum_kernel_op = jasp_dialect.ConsumeQuantumKernelOp(
        bool_result_type, qst
    )
    # Return the boolean result indicating success
    return [consume_quantum_kernel_op.results[0]]


lowering_rules.append((consume_quantum_kernel_p, consume_quantum_kernel_lowering))


def quantum_gate_lowering(ctx, *args, **params):
    """
    Lowering rule that emits our QuantumGate dialect operation.
    """

    # Extract gate type from params (JAX primitive parameters)
    op = params.get("gate", None)

    # Convert gate_type string to MLIR string attribute
    gate_type_attr = ir.StringAttr.get(op.name)

    # args contains: [qubits..., raw_param_operands..., quantum_state]. Rebuild
    # the gate's symbolic parameter expressions from gate.params so expressions
    # like -0.5 * alpha survive into MLIR as StableHLO arithmetic.
    qubit_operands = list(args[: op.num_qubits])
    raw_param_operands = list(args[op.num_qubits:-1])
    quantum_state = args[-1]

    sorted_symbols = _sorted_param_symbols(op.params)
    if len(raw_param_operands) != len(sorted_symbols):
        raise ValueError(
            f"Gate '{op.name}' expected {len(sorted_symbols)} raw parameter operand(s) "
            f"for symbols {sorted_symbols}, got {len(raw_param_operands)}"
        )

    symbol_bindings = {
        sym: raw_param_operands[i] for i, sym in enumerate(sorted_symbols)
    }
    lowered_param_operands = [
        _lower_gate_param_expr(ctx, expr, symbol_bindings) for expr in op.params
    ]
    gate_operands = qubit_operands + lowered_param_operands

    # Create our quantum_gate operation using the generated class
    quantum_gate_op = jasp_dialect.QuantumGateOp(
        get_ir_qst_type(),  # Result type
        gate_type_attr,  # Gate type as string attribute
        gate_operands,  # Variadic gate operands (parameters + qubits)
        quantum_state,  # Input quantum state
    )
    # Return the result quantum state
    return [quantum_gate_op.results[0]]


lowering_rules.append((quantum_gate_p, quantum_gate_lowering))


def parity_lowering(ctx, *measurements, expectation, observable):
    """
    Lowering rule that emits our Parity dialect operation.
    """
    # Measurements is a list of booleans, we pass them as variadic arguments
    # Expectation is an integer attribute, observable is an integer attribute (0 or 1)

    # Create the result type: scalar tensor of i1 (0-rank tensor)
    result_type = ir.RankedTensorType.get([], get_i1_type())

    # Create our parity operation using the generated class
    parity_op = jasp_dialect.ParityOp(
        result_type, list(measurements), expectation, observable
    )
    # Return the boolean result
    return [parity_op.results[0]]


lowering_rules.append((parity_p, parity_lowering))

jasp_lowering_rules = tuple(lowering_rules)
