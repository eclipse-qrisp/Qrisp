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
    AbstractQuantumCircuit,
    AbstractQubitArray,
    AbstractQubit)

import qrisp.jasp.mlir.dialect_implementation._jasp_ops_gen as jasp_dialect

from jax.interpreters.mlir import LoweringParameters, ModuleContext, lower_jaxpr_to_fun, ir_type_handlers
from jaxlib.mlir import ir


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
    assert isinstance(aval, AbstractQuantumCircuit)
    return get_ir_qst_type()
def _aqa_lowering(aval):
    assert isinstance(aval, AbstractQubitArray)
    return get_ir_qa_type()
def _aqb_lowering(aval):
    assert isinstance(aval, AbstractQubit)
    return get_ir_qb_type()

# Register in ir_type_handlers
ir_type_handlers[AbstractQuantumCircuit] = _aqc_lowering
ir_type_handlers[AbstractQubitArray] = _aqa_lowering
ir_type_handlers[AbstractQubit] = _aqb_lowering


###############################
# Register Primitive lowering #
###############################

lowering_rules = []

def create_qubits_lowering(ctx, amount, qst_in):
    """
    Lowering rule that emits our CreateQubits dialect operation.
    """
    # Enable unregistered dialects for our dialect
    ctx.module_context.context.allow_unregistered_dialects = True
    
    # Create our create_qubits operation using the generated class
    create_qubits_op = jasp_dialect.CreateQubitsOp(get_ir_qa_type(), get_ir_qst_type(), amount, qst_in)
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
    i64_result_type = ir.RankedTensorType.get([], get_i64_type())  # scalar tensor of i64
    
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
        res_type = ir.RankedTensorType.get([], get_i1_type())   # scalar tensor of i1
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
    consume_quantum_kernel_op = jasp_dialect.ConsumeQuantumKernelOp(bool_result_type, qst)
    # Return the boolean result indicating success
    return [consume_quantum_kernel_op.results[0]]

lowering_rules.append((consume_quantum_kernel_p, consume_quantum_kernel_lowering))

def quantum_gate_lowering(ctx, *args, **params):
    """
    Lowering rule that emits our QuantumGate dialect operation.
    """
    
    # Extract gate type from params (JAX primitive parameters)
    op = params.get('gate', None)
    
    
    # Convert gate_type string to MLIR string attribute
    gate_type_attr = ir.StringAttr.get(op.name)
    
    
    # args contains: [gate_operands..., quantum_state]
    # Separate gate operands from the quantum state (last argument)
    gate_operands = args[:-1]  # All arguments except the last
    quantum_state = args[-1]   # Last argument is the quantum state
    
    # Create our quantum_gate operation using the generated class
    quantum_gate_op = jasp_dialect.QuantumGateOp(
        get_ir_qst_type(),           # Result type
        gate_type_attr,          # Gate type as string attribute
        gate_operands,           # Variadic gate operands (parameters + qubits)
        quantum_state            # Input quantum state
    )
    # Return the result quantum state
    return [quantum_gate_op.results[0]]

lowering_rules.append((quantum_gate_p, quantum_gate_lowering))

CUSTOM_LOWERING_RULES = tuple(lowering_rules)
# Use LoweringParameters with override_lowering_rules
lowering_params = LoweringParameters(override_lowering_rules=CUSTOM_LOWERING_RULES)

def lower_jaspr_to_MLIR_raw(jaspr):
    
    # Create the necessary components for ModuleContext
    keepalives = []
    host_callbacks = []
    channel_iter = 1
    
    ctx = ModuleContext(
        backend=None,
        platforms=["cpu"],
        axis_context=None,
        keepalives=keepalives,
        channel_iterator=channel_iter,
        host_callbacks=host_callbacks,
        lowering_parameters=lowering_params,
    )
    
    # Enable unregistered dialects
    ctx.context.allow_unregistered_dialects = True
    
    # Lower JAXPR to MLIR using Catalyst's method
    with ctx.context, ir.Location.unknown(ctx.context):
        
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get("jasp_module")
        
        from jax._src.source_info_util import NameStack
        
        try:
            lower_jaxpr_to_fun(
                ctx,
                "main",
                jaspr,  # Pass the full ClosedJaxpr object
                jaspr.effects,
                public=True,
                name_stack=NameStack(),
            )
        except Exception as e:
            print(f"Error in lower_jaxpr_to_fun: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    return ctx.module
