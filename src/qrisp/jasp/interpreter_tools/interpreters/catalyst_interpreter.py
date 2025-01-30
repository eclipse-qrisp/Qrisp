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

from sympy import lambdify, symbols

from jax import make_jaxpr, jit
from jax.core import ClosedJaxpr
from jax.lax import fori_loop, cond, while_loop
from jax._src.linear_util import wrap_init
import jax.numpy as jnp

from catalyst.jax_primitives import (AbstractQreg, qinst_p, qmeasure_p, 
                                     qextract_p, qinsert_p, while_p, cond_p, func_p)

from qrisp.jasp import (QuantumPrimitive, OperationPrimitive, AbstractQuantumCircuit, AbstractQubitArray, 
AbstractQubit, eval_jaxpr, Jaspr, extract_invalues, insert_outvalues, Measurement_p, get_qubit_p,
get_size_p, delete_qubits_p, Jlist)

greek_letters = symbols('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega')

# Name translator from Qrisp gate naming to Catalyst gate naming
op_name_translation_dic = {"cx" : "CNOT",
                           "cy" : "CY", 
                           "cz" : "CZ", 
                           "crx" : "CRX",
                           "crz" : "CRZ",
                           "swap" : "SWAP",
                           "x" : "PauliX",
                           "y" : "PauliY",
                           "z" : "PauliZ",
                           "h" : "Hadamard",
                           "rx" : "RX",
                           "ry" : "RY",
                           "rz" : "RZ",
                           "s" : "S",
                           "t" : "T",
                           "p" : "RZ",
                           "u3" : "Rot"}


def catalyst_eqn_evaluator(eqn, context_dic):
    """
    This function serves as the central interface to interpret jasp equations 
    into Catalyst primitives.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be interpreted.
    context_dic : qrisp.jasp.interpreters.ContextDict
        The ContextDict representing the current state of the program by 
        translating from variables to values.

    Raises
    ------
    Exception
        Don't know how to interpret QuantumPrimitive.

    Returns
    -------
    bool
        Like other equation processing managers, if this function returns True,
        it implies that the equation should be processed by simply binding the 
        primitive.

    """
    
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
    
# def process_create_qubits(invars, outvars, context_dic):
    
#     # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
#     # which is represented by an AbstractQreg and an integer
#     qreg, stack_size = context_dic[invars[0]]
    
#     # We create the new QubitArray representation by putting the appropriate tuple
#     # in the context_dic
#     context_dic[outvars[1]] = (stack_size, context_dic[invars[1]])
    
#     # Furthermore we create the updated AbstractQuantumCircuit representation.
#     # The new stack size is the old stask size + the size of the QubitArray
#     context_dic[outvars[0]] = (qreg, stack_size + context_dic[invars[1]])
    
def process_create_qubits(invars, outvars, context_dic):
    
    # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
    # which is represented by an AbstractQreg and an integer
    qreg, free_qubits = context_dic[invars[0]]
    
    size = context_dic[invars[1]]
    
    # We create the new QubitArray representation by putting the appropriate tuple
    # in the context_dic
    
    reg_qubits = Jlist()
    
    def loop_body(i, val_tuple):
        free_qubits, reg_qubits = val_tuple
        reg_qubits.append(free_qubits.pop())
        return free_qubits, reg_qubits
    
    @jit
    def make_tracer(x):
        return x
    
    size = make_tracer(size)
    
    
    free_qubits, reg_qubits = fori_loop(0, size, loop_body, (free_qubits, reg_qubits))
    
    context_dic[outvars[1]] = reg_qubits
    
    # Furthermore we create the updated AbstractQuantumCircuit representation.
    # The new stack size is the old stask size + the size of the QubitArray
    context_dic[outvars[0]] = (qreg, free_qubits)
    
    
def process_delete_qubits(eqn, context_dic):
    
    # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
    # which is represented by an AbstractQreg and an integer
    qreg, free_qubits = context_dic[eqn.invars[0]]
    reg_qubits = context_dic[eqn.invars[1]]
    
    # We create the new QubitArray representation by putting the appropriate tuple
    # in the context_dic
    
    def loop_body(i, val_tuple):
        free_qubits, reg_qubits = val_tuple
        free_qubits.append(reg_qubits.pop())
        return free_qubits, reg_qubits
    
    @jit
    def make_tracer(x):
        return x
    
    free_qubits, reg_qubits = fori_loop(0, reg_qubits.counter, loop_body, (free_qubits, reg_qubits))
    
    context_dic[eqn.outvars[0]] = (qreg, free_qubits)

def process_get_qubit(invars, outvars, context_dic):
    # The get_qubit primitive needs to retrieve the integer that indexes the 
    # AbstractQubit in the stack.
    # For that we add the Qubit index (in the QubitArray) to the QubitArray 
    # starting index.
    
    qubit_list = context_dic[invars[0]]
    qubit_index = qubit_list[context_dic[invars[1]]]
    context_dic[outvars[0]] = qubit_index
    
def process_slice(invars, outvars, context_dic):
    
    qubit_reg = context_dic[invars[0]]
    start = context_dic[invars[1]]
    stop = context_dic[invars[2]]
    
    context_dic[outvars[0]] = qubit_reg[start:stop]    
    
    
def process_get_size(invars, outvars, context_dic):
    # The size is simply the second entry of the QubitArray representation
    context_dic[outvars[0]] = context_dic[invars[0]].counter
    
    
def process_op(op_prim, invars, outvars, context_dic):
    
    # Alias the AbstractQreg
    catalyst_register_tracer = context_dic[invars[0]][0]
    
    # We first collect all the Catalyst qubit tracers
    # that are required for the Operation
    qb_vars = []
    
    op = op_prim.op
    # For that we find all the integers (ie. positions) of the participating
    # qubits in the AbstractQreg
    qb_pos = []
    for i in range(op.num_qubits):
        qb_vars.append(invars[i+1+len(op.params)])
        qb_pos.append(context_dic[invars[i+1+len(op.params)]])
    
    num_qubits = len(qb_pos)
    
    param_dict = {}
    if op.name == "u3":
        param_dict[op.params[0]] = context_dic[invars[3]]
        param_dict[op.params[1]] = context_dic[invars[1]]
        param_dict[op.params[2]] = context_dic[invars[2]]
    else:
        for i in range(len(op.params)):
            param_dict[op.params[i]] = context_dic[invars[i+1]]

    
    
    # Extract the catalyst qubit tracers by using the qextract primitive.
    catalyst_qb_tracers = []
    for i in range(num_qubits):
        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                             qb_pos[i])
        catalyst_qb_tracers.append(catalyst_qb_tracer)
    
    # We can now apply the gate primitive
    res_qbs = exec_qrisp_op(op, catalyst_qb_tracers, param_dict)
        
    
    # Finally, we reinsert the qubits and update the register tracer
    for i in range(num_qubits):
        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                             qb_pos[i],
                                             res_qbs[i])
        
    context_dic[outvars[0]] = (catalyst_register_tracer, context_dic[invars[0]][1])

def exec_qrisp_op(op, catalyst_qbs, param_dict):
    # This function takes a Qrisp operation and a list of catalyst qubit tracers
    # and applies the operation to these qubits.
    
    # If the Operation has a definition, we call this function recursively.
    if op.definition:
        defn = op.definition
        for instr in defn.data:
            qubits = instr.qubits
            qubit_indices = [defn.qubits.index(qb) for qb in qubits]
            temp_catalyst_qbs = [catalyst_qbs[i] for i in qubit_indices]
            res_qbs = exec_qrisp_op(instr.op, temp_catalyst_qbs, param_dict)
            
            for i in range(len(qubit_indices)):
                catalyst_qbs[qubit_indices[i]] = res_qbs[i]
                
        return catalyst_qbs
    
    # Otherwise we simply call the bind method
    else:
        
        if op.name == "gphase":
            return catalyst_qbs
        
        if op.name[-3:] == "_dg":
            op_name = op.name[:-3]
            invert = True
        else:
            invert = False
            op_name = op.name
        
        jax_values = list(param_dict.values())
        
        param_list = [lambdify(greek_letters[:len(op.abstract_params)], expr)(*jax_values) for expr in op.params]
        
        if op_name == "sx":
            op_name = "rx"
            param_list = [jnp.pi/2]
        
        # param_list = [param_dict[symb] for symb in op.params]

        catalyst_name = op_name_translation_dic[op_name]        
        
        res_qbs = qinst_p.bind(*(catalyst_qbs+param_list), 
                               op = catalyst_name, 
                               qubits_len = op.num_qubits,
                               adjoint = invert,
                               ctrl_len = 0,
                               ctrl_value_len = 0)
        return res_qbs


def process_measurement(invars, outvars, context_dic):
    # This function treats measurement primitives
    
    # Retrieve the catalyst register tracer
    catalyst_register_tracer = context_dic[invars[0]][0]
    
    # We differentiate between the cases that the measurement is performed on a 
    # singular Qubit (returns a bool) or a QubitArray (returns an integer)
    
    # This case treats the QubitArray situation
    if isinstance(invars[1].aval, AbstractQubitArray):
        
        # Retrieve the start and the endpoint indices of the QubitArray
        qubit_list = context_dic[invars[1]]
        
        # The multi measurement logic is outsourced into a dedicated function
        catalyst_register_tracer, meas_res = exec_multi_measurement(catalyst_register_tracer, qubit_list)
    
    # The singular Qubit case
    else:
        
        # Get the position of the Qubit
        qb_pos = context_dic[invars[1]]
        
        # Get the catalyst Qubit tracer
        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                             qb_pos)
        
        # Call the measurement primitive
        meas_res, res_qb = qmeasure_p.bind(catalyst_qb_tracer)
        
        # Reinsert the Qubit
        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                             qb_pos,
                                             res_qb)
    
    # Insert the result values into the context dict
    context_dic[outvars[0]] = (catalyst_register_tracer, context_dic[invars[0]][1])
    context_dic[outvars[1]] = meas_res
        
        
def exec_multi_measurement(catalyst_register, qubit_list):
    # This function performs the measurement of multiple qubits at once, returning
    # an integer. The qubits to be measured sit in one consecutive interval,
    # starting at index "start" and ending at "stop".
    
    # We realize this feature using the Catalyst while loop primitive.
    
    # For that we need to define the loop body and the condition for breaking the
    # loop
    
    def loop_body(i, acc, reg, static_qubit_array, list_size):
        
        qb_index = static_qubit_array[i]
        qb = qextract_p.bind(reg, qb_index)
        res_bl, res_qb = qmeasure_p.bind(qb)
        reg = qinsert_p.bind(reg, qb_index, res_qb)
        acc = acc + (jnp.asarray(1, dtype = "int64")<<i)*res_bl
        i += jnp.asarray(1, dtype = "int64")
        return i, acc, reg, static_qubit_array, list_size
    
    def cond_body(i, acc, reg, static_qubit_array, list_size):
        return i < list_size
    
    static_qubit_array, list_size = qubit_list.array, qubit_list.counter
    
    # Turn them into jaxpr
    zero = jnp.asarray(0, dtype = "int64")
    loop_jaxpr = make_jaxpr(loop_body)(zero, zero, catalyst_register.aval, static_qubit_array, list_size)
    cond_jaxpr = make_jaxpr(cond_body)(zero, zero, catalyst_register.aval, static_qubit_array, list_size)

    # Bind the while loop primitive
    i, acc, reg, static_qubit_array, list_size = while_p.bind(*(zero, zero, catalyst_register, static_qubit_array, list_size),
                                     cond_jaxpr = cond_jaxpr, 
                                     body_jaxpr = loop_jaxpr,
                                     cond_nconsts = 0,
                                     body_nconsts = 0,
                                     nimplicit = 0,
                                     preserve_dimensions = True,
                                     ) 
    
    return reg, acc


def process_while(eqn, context_dic):
    for invar in eqn.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            break
    else:
        return True
    
    body_jaxpr = ensure_conversion(eqn.params["body_jaxpr"].jaxpr)
    cond_jaxpr = ensure_conversion(eqn.params["cond_jaxpr"].jaxpr)
    
    invalues = extract_invalues(eqn, context_dic)
    
    flattened_invalues = flatten_signature(invalues, eqn.invars)
    
    outvalues = while_p.bind(*flattened_invalues,
                            cond_jaxpr = cond_jaxpr, 
                            body_jaxpr = body_jaxpr,
                            cond_nconsts = 0,
                            body_nconsts = 0,
                            nimplicit = 0,
                            preserve_dimensions = True,)
    
    outvalues = list(outvalues)
    unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)
    
    insert_outvalues(eqn, context_dic, unflattened_outvalues)

def process_cond(eqn, context_dic):
    
    false_jaxpr = ensure_conversion(eqn.params["branches"][0].jaxpr)
    true_jaxpr = ensure_conversion(eqn.params["branches"][1].jaxpr)
    
    invalues = extract_invalues(eqn, context_dic)

    # Contrary to the jax cond primitive, the catalyst cond primitive
    # wants a bool    
    invalues[0] = jnp.asarray(invalues[0], bool)
    
    flattened_invalues = flatten_signature(invalues, eqn.invars)
    
    outvalues = cond_p.bind(*flattened_invalues, branch_jaxprs = (true_jaxpr, false_jaxpr), nimplicit_outputs = 0)
    
    unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)
    
    insert_outvalues(eqn, context_dic, unflattened_outvalues)

@lru_cache(maxsize = int(1E5))
def get_traced_fun(jaxpr):
    
    catalyst_jaxpr = ensure_conversion(jaxpr)
    
    from jax.core import eval_jaxpr
    @jit
    def jitted_fun(*args):
        return eval_jaxpr(catalyst_jaxpr.jaxpr, [], *args)
     
    return jitted_fun
    

def process_pjit(eqn, context_dic):
    
    invalues = extract_invalues(eqn, context_dic)
    
    flattened_invalues = flatten_signature(invalues, eqn.invars)
            
    jaxpr = eqn.params["jaxpr"]
    traced_fun = get_traced_fun(jaxpr.jaxpr)
            
    outvalues = func_p.bind(wrap_init(traced_fun), *flattened_invalues, fn=traced_fun)

    outvalues = list(outvalues)
    unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)

    insert_outvalues(eqn, context_dic, unflattened_outvalues)

# Function to reset and delete a qubit array
def reset_qubit_array(abs_qc, qb_array):
    from qrisp.circuit import XGate
    
    def body_func(arg_tuple):
        
        abs_qc, qb_array, i = arg_tuple
        
        abs_qb = get_qubit_p.bind(qb_array, i)
        abs_qc, meas_bl = Measurement_p.bind(abs_qc, abs_qb)
        
        def true_fun(arg_tuple):
            abs_qc, qb = arg_tuple
            abs_qc = OperationPrimitive(XGate()).bind(abs_qc, qb)
            return (abs_qc, qb)
        
        def false_fun(arg_tuple):
            return arg_tuple
        
        abs_qc, qb = cond(meas_bl, true_fun, false_fun, (abs_qc, abs_qb))
        
        i += 1
        
        return (abs_qc, qb_array, i)
    
    def cond_fun(arg_tuple):
        return arg_tuple[-1] < get_size_p.bind(arg_tuple[1])
    
    
    abs_qc, qb_array, i = while_loop(cond_fun,
                                 body_func,
                                 (abs_qc, qb_array, jnp.array(0, dtype = jnp.int64))
                                 )
    
    return abs_qc

reset_jaxpr = make_jaxpr(reset_qubit_array)(AbstractQuantumCircuit(), AbstractQubitArray())

def process_reset(eqn, context_dic):
    
    invalues = extract_invalues(eqn, context_dic)
    outvalues = eval_jaxpr(reset_jaxpr.jaxpr, eqn_evaluator = catalyst_eqn_evaluator)(*invalues)
    insert_outvalues(eqn, context_dic, outvalues)
    

def ensure_conversion(jaxpr):
    from qrisp.jasp.evaluation_tools.catalyst_interface import jaspr_to_catalyst_jaxpr
    for invar in jaxpr.invars:
        if isinstance(invar.aval, (AbstractQuantumCircuit, AbstractQubitArray, AbstractQubit)):
            return jaspr_to_catalyst_jaxpr(jaxpr)
    return ClosedJaxpr(jaxpr, [])
    
    
def flatten_signature(values, variables):
    values = list(values)
    flattened_values = []
    for i in range(len(variables)):
        var = variables[i]
        value = values.pop(0)
        if isinstance(var.aval, AbstractQuantumCircuit):
            flattened_values.extend((value[0], *value[1].flatten()[0]))
        elif isinstance(var.aval, AbstractQubitArray):
            flattened_values.extend(value.flatten()[0])
        else:
            flattened_values.append(value)

    return flattened_values

def unflatten_signature(values, variables):
    values = list(values)
    unflattened_values = []
    for var in variables:
        if isinstance(var.aval, AbstractQuantumCircuit):
            catalyst_register_tracer = values.pop(0)
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append((catalyst_register_tracer, Jlist.unflatten([], jlist_tuple)))
        elif isinstance(var.aval, AbstractQubitArray):
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append(Jlist.unflatten([], jlist_tuple))
        else:
            unflattened_values.append(values.pop(0))
    
    return unflattened_values