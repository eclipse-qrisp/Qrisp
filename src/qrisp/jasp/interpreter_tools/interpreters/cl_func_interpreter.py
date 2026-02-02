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

from functools import lru_cache

import numpy as np

from jax import make_jaxpr, jit, debug
from jax.extend.core import ClosedJaxpr, Literal
from jax.lax import fori_loop, cond, while_loop, switch
import jax.numpy as jnp

from qrisp.circuit import PTControlledOperation
from qrisp.jasp import (
    QuantumPrimitive,
    AbstractQuantumCircuit,
    AbstractQubitArray,
    AbstractQubit,
    eval_jaxpr,
    Jaspr,
    extract_invalues,
    insert_outvalues,
    Jlist,
)


def cl_func_eqn_evaluator(eqn, context_dic):

    # If the equations primitive is a Qrisp primitive, we process it according
    # to one of the implementations below. Otherwise we return True to indicate
    # default interpretation.
    if isinstance(eqn.primitive, QuantumPrimitive):

        invars = eqn.invars
        outvars = eqn.outvars

        if eqn.primitive.name == "jasp.quantum_gate":
            process_op(eqn.params["gate"], invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.create_qubits":
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
        elif eqn.primitive.name == "jasp.fuse":
            process_fuse(eqn, context_dic)
        else:
            raise Exception(
                f"Don't know how to process QuantumPrimitive {eqn.primitive}"
            )
    else:
        if eqn.primitive.name == "while":
            return process_while(eqn, context_dic)
        elif eqn.primitive.name == "cond":
            return process_cond(eqn, context_dic)  #
        elif eqn.primitive.name == "jit":
            process_pjit(eqn, context_dic)
        else:
            return True


def process_create_qubits(invars, outvars, context_dic):

    # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
    # which is represented by an AbstractQreg and an integer
    qreg, free_qubits = context_dic[invars[1]]

    size = context_dic[invars[0]]

    # We create the new QubitArray representation by putting the appropriate tuple
    # in the context_dic

    reg_qubits = Jlist(init_val = jnp.array([], dtype = jnp.uint64),
                       max_size = 64*qreg.size//1024)

    def loop_body(i, val_tuple):
        reg_qubits, free_qubits = val_tuple
        reg_qubits.append(free_qubits.pop())
        return reg_qubits, free_qubits

    @jit
    def make_tracer(x):
        return x

    size = make_tracer(size)

    reg_qubits, free_qubits = fori_loop(0, size, loop_body, (reg_qubits, free_qubits))

    context_dic[outvars[0]] = reg_qubits

    # Furthermore we create the updated AbstractQuantumCircuit representation.
    # The new stack size is the old stask size + the size of the QubitArray
    context_dic[outvars[1]] = (qreg, free_qubits)


def process_fuse(eqn, context_dic):

    invalues = extract_invalues(eqn, context_dic)

    if isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(
        eqn.invars[1].aval, AbstractQubit
    ):
        res_qubits = Jlist(invalues, max_size=invalues[0].max_size)
    elif isinstance(eqn.invars[0].aval, AbstractQubitArray) and isinstance(
        eqn.invars[1].aval, AbstractQubit
    ):
        res_qubits = invalues[0].copy()
        res_qubits.append(invalues[1])
    elif isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(
        eqn.invars[1].aval, AbstractQubitArray
    ):
        res_qubits = invalues[1].copy()
        res_qubits.prepend(invalues[0])
    else:
        res_qubits = invalues[0].copy()
        res_qubits.extend(invalues[1])

    insert_outvalues(eqn, context_dic, res_qubits)


def process_get_qubit(invars, outvars, context_dic):
    # The get_qubit primitive needs to retrieve the integer that indexes the
    # AbstractQubit in the stack.
    # For that we add the Qubit index (in the QubitArray) to the QubitArray
    # starting index.

    reg_qubits = context_dic[invars[0]]
    index = context_dic[invars[1]]
    context_dic[outvars[0]] = reg_qubits[index]


def process_slice(invars, outvars, context_dic):

    reg_qubits = context_dic[invars[0]]
    start = context_dic[invars[1]]
    stop = context_dic[invars[2]]

    context_dic[outvars[0]] = reg_qubits[start:stop]


def process_get_size(invars, outvars, context_dic):
    # The size is simply the second entry of the QubitArray representation
    context_dic[outvars[0]] = context_dic[invars[0]].counter


def process_op(op, invars, outvars, context_dic):

    # B
    bit_array = context_dic[invars[-1]][0]

    # For that we find all the integers (ie. positions) of the participating
    # qubits in the AbstractQreg
    qb_pos = []
    for i in range(op.num_qubits):
        qb_pos.append(context_dic[invars[i]])

    if op.name in ["x", "swap"]:
        ctrl_state = ""
    elif isinstance(op, PTControlledOperation) and op.base_operation.name in [
        "x",
        "y",
        "z",
        "pt2cx",
        "swap"
    ]:
        if op.base_operation.name in ["z", "rz", "t", "s", "t_dg", "s_dg", "p"]:
            context_dic[outvars[-1]] = context_dic[invars[-1]]
            return
        ctrl_state = op.ctrl_state
    elif op.name in ["z", "rz", "t", "s", "t_dg", "s_dg", "p"]:
        context_dic[outvars[-1]] = context_dic[invars[-1]]
        return
    else:
        raise Exception(f"Classical function simulator can't process gate {op.name}")

    if "swap" in op.name:
        swapped_qb_pos = list(qb_pos)
        swapped_qb_pos[-1], swapped_qb_pos[-2] = swapped_qb_pos[-2], swapped_qb_pos[-1]
        ctrl_state = ctrl_state + "1"
        
        bit_array = cl_multi_cx(bit_array, ctrl_state, swapped_qb_pos)
        bit_array = cl_multi_cx(bit_array, ctrl_state, qb_pos)
        bit_array = cl_multi_cx(bit_array, ctrl_state, swapped_qb_pos)
    else:
        bit_array = cl_multi_cx(bit_array, ctrl_state, qb_pos)

    context_dic[outvars[-1]] = (bit_array, context_dic[invars[-1]][1])


def cl_multi_cx(bit_array, ctrl_state, bit_pos):

    flip = jnp.asarray(1, dtype=bit_array.dtype)
    for i in range(len(bit_pos) - 1):
        if ctrl_state[i] == "1":
            flip = flip & (get_bit_array(bit_array, bit_pos[i]))
        else:
            flip = flip & (~get_bit_array(bit_array, bit_pos[i]))

    bit_array = conditional_bit_flip_bit_array(bit_array, bit_pos[-1], flip)
    # bit_array = bit_array.at[bit_pos[-1]].set((bit_array[bit_pos[-1]] ^ flip), mode = "promise_in_bounds")

    return bit_array


def process_measurement(invars, outvars, context_dic):

    # This function treats measurement primitives

    # Retrieve the catalyst register tracer
    bit_array = context_dic[invars[-1]][0]

    # We differentiate between the cases that the measurement is performed on a
    # singular Qubit (returns a bool) or a QubitArray (returns an integer)

    # This case treats the QubitArray situation
    if isinstance(invars[0].aval, AbstractQubitArray):

        # Retrieve the start and the endpoint indices of the QubitArray
        qubit_reg = context_dic[invars[0]]

        # The multi measurement logic is outsourced into a dedicated function
        bit_array, meas_res = exec_multi_measurement(bit_array, qubit_reg)

    # The singular Qubit case
    else:
        # Get the boolean value of the Qubit
        meas_res = get_bit_array(bit_array, context_dic[invars[0]])
        # meas_res = bit_array[context_dic[invars[1]]]

    # Insert the result values into the context dict
    context_dic[outvars[1]] = (bit_array, context_dic[invars[1]][1])
    context_dic[outvars[0]] = meas_res


def exec_multi_measurement(bit_array, qubit_reg):
    # This function performs the measurement of multiple qubits at once, returning
    # an integer. The qubits to be measured sit in one consecutive interval,
    # starting at index "start" and ending at "stop".

    # We realize this feature using the Catalyst while loop primitive.

    # For that we need to define the loop body and the condition for breaking the
    # loop

    def loop_body(i, arg_tuple):
        acc, bit_array, qubit_reg = arg_tuple
        qb_index = qubit_reg[i]
        res_bl = get_bit_array(bit_array, qb_index)
        acc = acc + (jnp.asarray(res_bl, dtype="int64") << i)
        return (acc, bit_array, qubit_reg)

    acc, bit_array, qubit_reg = fori_loop(
        0, qubit_reg.counter, loop_body, (0, bit_array, qubit_reg)
    )

    return bit_array, acc


def process_while(eqn, context_dic):
    for invar in eqn.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            break
    else:
        return True

    body_jaxpr = eqn.params["body_jaxpr"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    overall_constant_amount= eqn.params["body_nconsts"] + eqn.params["cond_nconsts"]

    invalues = extract_invalues(eqn, context_dic)

    if isinstance(body_jaxpr.jaxpr.invars[-1].aval, AbstractQuantumCircuit):
        converted_body_jaxpr = jaspr_to_cl_func_jaxpr(
            body_jaxpr.jaxpr, invalues[-1][0].shape[0] * 64
        )
    if isinstance(cond_jaxpr.jaxpr.invars[-1].aval, AbstractQuantumCircuit):
        converted_cond_jaxpr = jaspr_to_cl_func_jaxpr(
            cond_jaxpr.jaxpr, invalues[-1][0].shape[0] * 64
        )

    def body_fun(args):
        
        constants = args[eqn.params["cond_nconsts"]:overall_constant_amount]
        carries = args[overall_constant_amount:]
        
        flattened_invalues = flatten_signature(constants + carries, body_jaxpr.jaxpr.invars)
        
        body_res = eval_jaxpr(converted_body_jaxpr)(*(flattened_invalues))
        
        unflattened_body_outvalues = unflatten_signature(body_res, body_jaxpr.jaxpr.outvars)
        
        return list(args[:overall_constant_amount]) + list(unflattened_body_outvalues)

    def cond_fun(args):
        
        constants = args[:eqn.params["cond_nconsts"]]
        carries = args[overall_constant_amount:]
        
        flattened_invalues = flatten_signature(constants + carries, cond_jaxpr.jaxpr.invars)
        
        return eval_jaxpr(converted_cond_jaxpr)(*flattened_invalues)

    outvalues = while_loop(cond_fun, body_fun, invalues)[overall_constant_amount:]
    
    insert_outvalues(eqn, context_dic, outvalues)


def process_cond(eqn, context_dic):

    invalues = extract_invalues(eqn, context_dic)

    branch_list = []

    for i in range(len(eqn.params["branches"])):
        
        converted_jaxpr = ensure_conversion(
            eqn.params["branches"][i].jaxpr, invalues[1:]
        )
        branch_list.append(eval_jaxpr(converted_jaxpr))

    flattened_invalues = flatten_signature(invalues, eqn.invars)
    outvalues = switch(flattened_invalues[0], branch_list, *flattened_invalues[1:])

    if isinstance(outvalues, tuple):
        unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)
    else:
        unflattened_outvalues = (outvalues,)

    insert_outvalues(eqn, context_dic, unflattened_outvalues)


@lru_cache(maxsize=int(1e5))
def get_traced_fun(jaxpr, bit_array_size):

    from jax.core import eval_jaxpr

    if isinstance(jaxpr, Jaspr):
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(jaxpr, bit_array_size)
    else:
        cl_func_jaxpr = jaxpr

    @jit
    def jitted_fun(*args):
        return eval_jaxpr(cl_func_jaxpr.jaxpr, cl_func_jaxpr.consts, *args)

    return jitted_fun


def process_pjit(eqn, context_dic):

    invalues = extract_invalues(eqn, context_dic)
    
    if eqn.params["name"] in ["gidney_mcx", "gidney_mcx_inv"]:
        unflattened_outvalues = [
            (cl_multi_cx(invalues[-1][0], "11", invalues[:-1]), invalues[-1][1])
        ]
    elif eqn.params["name"] in ["gidney_CCCX", "gidney_CCCX_dg"]:
        controls = [invalues[0][i] for i in range(3)]
        unflattened_outvalues = [
            (cl_multi_cx(invalues[-1][0], "111",  controls + [invalues[1]]), invalues[-1][1])
        ]
    else:
        flattened_invalues = []
        for value in invalues:
            if isinstance(value, tuple):
                flattened_invalues.extend(value)
            else:
                flattened_invalues.append(value)

        jaxpr = eqn.params["jaxpr"]

        if isinstance(jaxpr, Jaspr):
            bit_array_padding = invalues[-1][0].shape[0] * 64
        else:
            bit_array_padding = 0

        traced_fun = get_traced_fun(jaxpr, bit_array_padding)

        flattened_invalues = flatten_signature(invalues, eqn.invars)
        outvalues = traced_fun(*flattened_invalues)
        unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)

    insert_outvalues(eqn, context_dic, unflattened_outvalues)


def process_delete_qubits(eqn, context_dic):
    invalues = extract_invalues(eqn, context_dic)

    bit_array = invalues[-1][0]
    free_qubits = invalues[-1][1]
    qubit_reg = invalues[0]

    def true_fun():
        debug.print("WARNING: Faulty uncomputation found during simulation.")

    def false_fun():
        return

    def loop_body(i, value_tuple):
        qubit_reg, bit_array, free_qubits = value_tuple

        qubit = qubit_reg.pop()
        cond(get_bit_array(bit_array, qubit), true_fun, false_fun)
        free_qubits.append(qubit)

        return qubit_reg, bit_array, free_qubits

    qubit_reg, bit_array, free_qubits = fori_loop(
        0, qubit_reg.counter, loop_body, (qubit_reg, bit_array, free_qubits)
    )

    outvalues = (bit_array, free_qubits)

    insert_outvalues(eqn, context_dic, outvalues)


def process_reset(eqn, context_dic):

    invalues = extract_invalues(eqn, context_dic)

    bit_array = invalues[-1][0]

    if isinstance(eqn.invars[0].aval, AbstractQubitArray):

        start = invalues[0][0]
        stop = start + invalues[0][1]

        def loop_body(i, bit_array):
            bit_array = conditional_bit_flip_bit_array(
                bit_array, i, get_bit_array(bit_array, i)
            )
            return bit_array

        bit_array = fori_loop(start, stop, loop_body, bit_array)

    else:

        bit_array = conditional_bit_flip_bit_array(
            bit_array, invalues[0], get_bit_array(bit_array, invalues[0])
        )

    outvalues = (bit_array, invalues[-1][1])
    insert_outvalues(eqn, context_dic, outvalues)


def jaspr_to_cl_func_jaxpr(jaspr, bit_array_padding):

    args = []
    for invar in jaspr.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            args.append(
                (
                    jnp.zeros(int(np.ceil(bit_array_padding / 64)), dtype=jnp.uint64),
                    Jlist(jnp.arange(bit_array_padding), max_size=bit_array_padding),
                )
            )
        elif isinstance(invar.aval, AbstractQubitArray):
            qreg = Jlist(init_val = jnp.array([], dtype = jnp.uint64),
                                   max_size = bit_array_padding//1024)
            args.append(qreg)
        elif isinstance(invar.aval, AbstractQubit):
            args.append(jnp.asarray(0, dtype="int64"))
        elif isinstance(invar, Literal):
            if isinstance(invar.val, int):
                args.append(jnp.asarray(invar.val, dtype="int64"))
            if isinstance(invar.val, float):
                args.append(jnp.asarray(invar.val, dtype="float64"))
        else:
            args.append(invar.aval)

    return make_jaxpr(eval_jaxpr(jaspr, eqn_evaluator=cl_func_eqn_evaluator))(
        *args
    )


def conditional_bit_flip_bit_array(bit_array, index, condition):
    index = jnp.uint64(index)
    array_index = index >> 6
    int_index = index ^ (array_index << 6)
    set_value = bit_array[array_index] ^ (condition << int_index)
    # set_value = bit_array[index]
    return bit_array.at[array_index].set(set_value)


conditional_bit_flip_bit_array = jit(conditional_bit_flip_bit_array, donate_argnums=0)


def get_bit_array(bit_array, index):
    index = jnp.uint64(index)
    array_index = index >> 6
    int_index = index ^ (array_index << 6)
    return (bit_array[array_index] >> int_index) & 1


def unflatten_signature(values, variables):
    values = list(values)
    unflattened_values = []
    for var in variables:
        if isinstance(var.aval, AbstractQuantumCircuit):
            bit_array = values.pop(0)
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append((bit_array, Jlist.unflatten([], jlist_tuple)))
        elif isinstance(var.aval, AbstractQubitArray):
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append(Jlist.unflatten([], jlist_tuple))
        else:
            unflattened_values.append(values.pop(0))

    return unflattened_values


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


def ensure_conversion(jaxpr, invalues):

    bit_array_padding = 0
    convert = False
    for i in range(len(jaxpr.invars)):
        invar = jaxpr.invars[i]
        if isinstance(
            invar.aval, (AbstractQuantumCircuit, AbstractQubitArray, AbstractQubit)
        ):
            convert = True
            if isinstance(invar.aval, AbstractQuantumCircuit):
                bit_array_padding = invalues[i][0].shape[0] * 64

    if convert:
        return jaspr_to_cl_func_jaxpr(jaxpr, bit_array_padding)
    return ClosedJaxpr(jaxpr, [])
