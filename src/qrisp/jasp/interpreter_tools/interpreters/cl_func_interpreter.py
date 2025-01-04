"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

from functools import lru_cache

from jax import make_jaxpr, jit, debug
from jax.core import ClosedJaxpr, Literal
from jax.lax import fori_loop, cond, while_loop
from jax._src.linear_util import wrap_init
import jax.numpy as jnp

from qrisp.circuit import ControlledOperation
from qrisp.jasp import (QuantumPrimitive, OperationPrimitive, AbstractQuantumCircuit, AbstractQubitArray, 
AbstractQubit, eval_jaxpr, Jaspr, extract_invalues, insert_outvalues, Measurement_p, get_qubit_p,
get_size_p, delete_qubits_p)


def cl_func_eqn_evaluator(eqn, context_dic):
    
    # If the equations primitive is a Qrisp primitive, we process it according
    # to one of the implementations below. Otherwise we return True to indicate
    # default interpretation.
    if isinstance(eqn.primitive, QuantumPrimitive):
        
        invars = eqn.invars
        outvars = eqn.outvars
        
        if eqn.primitive.name == "jasp.create_qubits":
            process_create_qubits(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.get_qubit":
            process_get_qubit(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.measure":
            process_measurement(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.get_size":
            process_get_size(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.slice":
            process_slice(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.delete_qubits":
            process_delete_qubits(eqn, context_dic)
        elif eqn.primitive.name == "jasp.reset":
            process_reset(eqn, context_dic)
        elif isinstance(eqn.primitive, OperationPrimitive):
            process_op(eqn.primitive, invars, outvars, context_dic)
        else:
            raise Exception(f"Don't know how to process QuantumPrimitive {eqn.primitive}")
    else:
        if eqn.primitive.name == "while":
            return process_while(eqn, context_dic)
        elif eqn.primitive.name == "cond":
            return process_cond(eqn, context_dic)#
        elif eqn.primitive.name == "pjit":
            process_pjit(eqn, context_dic)
        else:
            return True
    
def process_create_qubits(invars, outvars, context_dic):
    
    # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
    # which is represented by an AbstractQreg and an integer
    qreg, stack_size = context_dic[invars[0]]
    
    # We create the new QubitArray representation by putting the appropriate tuple
    # in the context_dic
    context_dic[outvars[1]] = (stack_size, context_dic[invars[1]])
    
    # Furthermore we create the updated AbstractQuantumCircuit representation.
    # The new stack size is the old stask size + the size of the QubitArray
    context_dic[outvars[0]] = (qreg, stack_size + context_dic[invars[1]])
    

def process_get_qubit(invars, outvars, context_dic):
    # The get_qubit primitive needs to retrieve the integer that indexes the 
    # AbstractQubit in the stack.
    # For that we add the Qubit index (in the QubitArray) to the QubitArray 
    # starting index.
    
    qubit_array_starting_index = context_dic[invars[0]][0]
    qubit_index = context_dic[invars[1]]
    context_dic[outvars[0]] = qubit_array_starting_index + qubit_index
    
def process_slice(invars, outvars, context_dic):
    
    base_qubit_array_starting_index = context_dic[invars[0]][0]
    base_qubit_array_ending_index = base_qubit_array_starting_index + context_dic[invars[0]][1]
    
    new_starting_index = context_dic[invars[1]] + base_qubit_array_starting_index
    
    new_max_index = jnp.min(jnp.array([base_qubit_array_ending_index, 
                                       base_qubit_array_starting_index + context_dic[invars[2]]]))
    
    new_size = new_max_index - new_starting_index
    
    context_dic[outvars[0]] = (new_starting_index, new_size)
    
    
def process_get_size(invars, outvars, context_dic):
    # The size is simply the second entry of the QubitArray representation
    context_dic[outvars[0]] = context_dic[invars[0]][1]
    
    
def process_op(op_prim, invars, outvars, context_dic):
    
    # B
    bit_array = context_dic[invars[0]][0]
    
    op = op_prim.op
    # For that we find all the integers (ie. positions) of the participating
    # qubits in the AbstractQreg
    qb_pos = []
    for i in range(op.num_qubits):
        qb_pos.append(context_dic[invars[i+1+len(op.params)]])
        
    if op.name == "x":
        ctrl_state = ""
    elif isinstance(op, ControlledOperation) and op.base_operation.name in ["x", "y", "z"]:
        if op.base_operation.name == "z":
            context_dic[outvars[0]] = context_dic[invars[0]]
            return
        ctrl_state = op.ctrl_state
    elif op.name == "z":
        context_dic[outvars[0]] = context_dic[invars[0]]
        return
    else:
        print(type(op))
        raise Exception(f"Classical function simulator can't process gate {op.name}")
    
    bit_array = cl_multi_cx(bit_array, ctrl_state, qb_pos)
    
    context_dic[outvars[0]] = (bit_array, context_dic[invars[0]][1])


def cl_multi_cx(bit_array, ctrl_state, bit_pos):
    
    flip = jnp.bool(True)
    for i in range(len(bit_pos) - 1):
        if ctrl_state[i] == "1":
            flip = flip & (bit_array[bit_pos[i]])
        else:
            flip = flip & (~bit_array[bit_pos[i]])
            
    bit_array = bit_array.at[bit_pos[-1]].set((bit_array[bit_pos[-1]] ^ flip), mode = "promise_in_bounds")
    
    return bit_array

def process_measurement(invars, outvars, context_dic):
    # This function treats measurement primitives
    
    # Retrieve the catalyst register tracer
    bit_array = context_dic[invars[0]][0]
    
    # We differentiate between the cases that the measurement is performed on a 
    # singular Qubit (returns a bool) or a QubitArray (returns an integer)
    
    # This case treats the QubitArray situation
    if isinstance(invars[1].aval, AbstractQubitArray):
        
        # Retrieve the start and the endpoint indices of the QubitArray
        qubit_array_data = context_dic[invars[1]]
        start = qubit_array_data[0]
        stop = start + qubit_array_data[1]
        
        # The multi measurement logic is outsourced into a dedicated function
        bit_array, meas_res = exec_multi_measurement(bit_array, start, stop)
    
    # The singular Qubit case
    else:
        
        # Get the position of the Qubit
        meas_res = bit_array[context_dic[invars[1]]]
    
    # Insert the result values into the context dict
    context_dic[outvars[0]] = (bit_array, context_dic[invars[0]][1])
    context_dic[outvars[1]] = meas_res
        
        
def exec_multi_measurement(bit_array, start, stop):
    # This function performs the measurement of multiple qubits at once, returning
    # an integer. The qubits to be measured sit in one consecutive interval,
    # starting at index "start" and ending at "stop".
    
    # We realize this feature using the Catalyst while loop primitive.
    
    # For that we need to define the loop body and the condition for breaking the
    # loop
    
    def loop_body(i, arg_tuple):
        acc, bit_array = arg_tuple
        res_bl = bit_array[i]
        acc = acc + (jnp.asarray(1, dtype = "int32")<<(i-start))*res_bl
        i += jnp.asarray(1, dtype = "int32")
        return (acc, bit_array)
    
    acc, bit_array = fori_loop(start, stop, loop_body, (0, bit_array))
    
    return bit_array, acc


def process_while(eqn, context_dic):
    for invar in eqn.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            break
    else:
        return True
    
    body_jaxpr = eqn.params["body_jaxpr"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    
    invalues = extract_invalues(eqn, context_dic)
    
    if isinstance(body_jaxpr.jaxpr.invars[0].aval, AbstractQuantumCircuit):
        body_jaxpr = jaspr_to_cl_func_jaxpr(body_jaxpr.jaxpr, invalues[0][0].shape[0])
    if isinstance(cond_jaxpr.jaxpr.invars[0].aval, AbstractQuantumCircuit):
        cond_jaxpr = jaspr_to_cl_func_jaxpr(cond_jaxpr.jaxpr, invalues[0][0].shape[0])
        
    def body_fun(args):
        return tuple(eval_jaxpr(body_jaxpr)(*args))
    
    def cond_fun(args):
        return eval_jaxpr(cond_jaxpr)(*args)
    
    
    
    unflattening_signature = []
    flattened_invalues = []
    for value in invalues:
        if isinstance(value, tuple):
            unflattening_signature.append(len(value))
            flattened_invalues.extend(value)
        else:
            unflattening_signature.append(None)
            flattened_invalues.append(value)
    
    # Make sure integers are properly converted to i32 
    # (under some circumstances jax seems to convert python ints
    # to i64, which raises a typing error)
    for i in range(len(flattened_invalues)):
        if isinstance(flattened_invalues[i], int):
            flattened_invalues[i] = jnp.asarray(flattened_invalues[i], dtype = "int32")
            
    outvalues = while_loop(cond_fun, body_fun, tuple(flattened_invalues))
    
    outvalues = list(outvalues)
    unflattened_outvalues = []
    for outvar in eqn.outvars:
        if isinstance(outvar.aval, AbstractQuantumCircuit):
            unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
        elif isinstance(outvar.aval, AbstractQubitArray):
            unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
        else:
            unflattened_outvalues.append(outvalues.pop(0))
            
    
    insert_outvalues(eqn, context_dic, unflattened_outvalues)

def process_cond(eqn, context_dic):
    
    false_jaxpr = eqn.params["branches"][0]
    true_jaxpr = eqn.params["branches"][1]
    
    invalues = extract_invalues(eqn, context_dic)
    
    if isinstance(false_jaxpr.jaxpr.invars[0].aval, AbstractQuantumCircuit):
        false_jaxpr = jaspr_to_cl_func_jaxpr(false_jaxpr.jaxpr, invalues[1][0].shape[0])
    if isinstance(true_jaxpr.jaxpr.invars[0].aval, AbstractQuantumCircuit):
        true_jaxpr = jaspr_to_cl_func_jaxpr(true_jaxpr.jaxpr, invalues[1][0].shape[0])

    

    def true_fun(*args):
        return eval_jaxpr(true_jaxpr)(*args)
    
    def false_fun(*args):
        return eval_jaxpr(false_jaxpr)(*args)
    
    flattened_invalues = []
    for value in invalues:
        if isinstance(value, tuple):
            flattened_invalues.extend(value)
        else:
            flattened_invalues.append(value)

    for i in range(len(flattened_invalues)):
        if isinstance(flattened_invalues[i], int):
            flattened_invalues[i] = jnp.asarray(flattened_invalues[i], dtype = "int32")
            
    outvalues = cond(flattened_invalues[0], true_fun, false_fun, *flattened_invalues[1:])
    outvalues = list(outvalues)
    
    unflattened_outvalues = []
    for outvar in eqn.outvars:
        if isinstance(outvar.aval, AbstractQuantumCircuit):
            unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
        elif isinstance(outvar.aval, AbstractQubitArray):
            unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
        else:
            unflattened_outvalues.append(outvalues.pop(0))
    
    insert_outvalues(eqn, context_dic, unflattened_outvalues)

@lru_cache(maxsize = int(1E5))
def get_traced_fun(jaxpr, bit_array_size):
    
    from jax.core import eval_jaxpr
    
    if isinstance(jaxpr, Jaspr):
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(jaxpr, bit_array_size)
    else:
        cl_func_jaxpr = jaxpr
    
    cl_func_jaxpr = ClosedJaxpr(cl_func_jaxpr, [])
    @jit
    def jitted_fun(*args):
        return eval_jaxpr(cl_func_jaxpr.jaxpr, [], *args)
     
    return jitted_fun
    

def process_pjit(eqn, context_dic):
    
    invalues = extract_invalues(eqn, context_dic)
    
    if eqn.params["name"] in ["gidney_mcx", "gidney_mcx_inv"]:
        unflattened_outvalues = [(cl_multi_cx(invalues[0][0], "11", invalues[1:]),invalues[0][1])]
    
    else:
        flattened_invalues = []
        for value in invalues:
            if isinstance(value, tuple):
                flattened_invalues.extend(value)
            else:
                flattened_invalues.append(value)

        jaxpr = eqn.params["jaxpr"]

        if isinstance(jaxpr.jaxpr, Jaspr):
            bit_array_size = invalues[0][0].shape[0]
        else:
            bit_array_size = 0
                
        
        traced_fun = get_traced_fun(jaxpr.jaxpr, bit_array_size)
                
        outvalues = traced_fun(*flattened_invalues)
    
        outvalues = list(outvalues)
        unflattened_outvalues = []
        for outvar in eqn.outvars:
            if isinstance(outvar.aval, AbstractQuantumCircuit):
                unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
            elif isinstance(outvar.aval, AbstractQubitArray):
                unflattened_outvalues.append((outvalues.pop(0), outvalues.pop(0)))
            else:
                unflattened_outvalues.append(outvalues.pop(0))

    insert_outvalues(eqn, context_dic, unflattened_outvalues)
    
def process_delete_qubits(eqn, context_dic):
    invalues = extract_invalues(eqn, context_dic)
    
    start = invalues[1][0]
    stop = start + invalues[1][1]
    
    def true_fun():
        debug.print("WARNING: Faulty uncomputation found during simulation.")
    def false_fun():
        return
    
    def loop_body(i, bit_array):
        cond(bit_array[i], true_fun, false_fun)
        return bit_array
    
    fori_loop(start, stop, loop_body, invalues[0][0])
    
    insert_outvalues(eqn, context_dic, invalues[0])
    
def process_reset(eqn, context_dic):
    
    invalues = extract_invalues(eqn, context_dic)
    
    bit_array = invalues[0][0]
    
    jnp_false = jnp.array(False, dtype = jnp.bool)
    
    if isinstance(eqn.invars[1].aval, AbstractQubitArray):
        
        start = invalues[1][0]
        stop = start + invalues[1][1]
        
        def loop_body(i, bit_array):
            bit_array = bit_array.at[i].set(jnp_false)
            return bit_array
        
        bit_array = fori_loop(start, stop, loop_body, bit_array)
        
    else:
        
        bit_array = bit_array.at[invalues[1]].set(jnp_false)
    
    outvalues = (bit_array, invalues[0][1])
    insert_outvalues(eqn, context_dic, outvalues)
    
    
    
def jaspr_to_cl_func_jaxpr(jaspr, bit_array_size):

    # Translate the input args according to the above rules.
    args = []
    for invar in jaspr.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            args.append((jnp.zeros(bit_array_size, dtype = jnp.bool), jnp.asarray(0, dtype = "int32")))
        elif isinstance(invar.aval, AbstractQubitArray):
            args.append((jnp.asarray(0, dtype = "int32"), jnp.asarray(0, dtype = "int32")))
        elif isinstance(invar.aval, AbstractQubit):
            args.append(jnp.asarray(0, dtype = "int32"))
        elif isinstance(invar, Literal):
            if isinstance(invar.val, int):    
                args.append(jnp.asarray(invar.val, dtype = "int32"))
            if isinstance(invar.val, float):
                args.append(jnp.asarray(invar.val, dtype = "f32"))
        else:
            args.append(invar.aval)
    
    # Call the Catalyst interpreter
    return make_jaxpr(eval_jaxpr(jaspr, eqn_evaluator = cl_func_eqn_evaluator))(*args).jaxpr