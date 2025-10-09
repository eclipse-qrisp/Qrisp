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

"""
This file implements the tools to perform quantum resource estimation using Jasp
infrastructure. The idea here is to transform the quantum instructions within a
given Jaspr into "counting instructions". That means instead of performing some
quantum gate, we increment an index in an array, which keeps track of how many
instructions of each type have been performed.

To do this, we implement the 

qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter.py

Which handles the transformation logic of the Jaspr.
This file implements the interfaces to evaluating the transformed Jaspr.

"""


from functools import lru_cache

import numpy as np

from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    extract_invalues,
    eval_jaxpr,
    insert_outvalues
)

from qrisp.jasp.primitives import (
    QuantumPrimitive,
    OperationPrimitive,
    AbstractQubitArray,
)


import jax
import jax.numpy as jnp
from jax.random import key



# This functions takes a "profiling dic", i.e. a dictionary of the form {str : int}
# indicating what kinds of quantum gates can appear in a Jaspr.
# It returns an equation evaluator, which increments a counter in an array for
# each quantum operation.
def make_profiling_eqn_evaluator(profiling_dic, meas_behavior):
    
    def profiling_eqn_evaluator(eqn, context_dic):

        invalues = extract_invalues(eqn, context_dic)

        if isinstance(eqn.primitive, QuantumPrimitive):

            # In the case of an OperationPrimitive, we determine the array index
            # to be increment via dictionary look-up and perform the increment
            # via the Jax-given .at method.
            if isinstance(eqn.primitive, OperationPrimitive):

                counting_array = list(invalues[-1][0])
                incrementation_constants = invalues[-1][1]

                op = eqn.primitive.op

                if op.definition:
                    op_counts = op.definition.transpile().count_ops()
                else:
                    op_counts = {op.name: 1}

                for op_name, count in op_counts.items():
                    counting_index = profiling_dic[op_name]
                    
                    # It seems like at this point we can just
                    # naively increment the Jax tracer but
                    # unfortunately the XLA compiler really 
                    # doesn't like constants. (Compile time blows up)
                    # We therefore jump through a lot of hoops
                    # to make it look like we are adding variables.
                    # This is done by supplying a list of tracers
                    # that will contain the numbers from 1 to n.
                    # Here is the next problem: If n is very large
                    # this also slows down the compilation because
                    # each function call has to have n+ arguments.
                    # We therefore perform the following loop:
                    while count:
                        incrementor = min(count, len(incrementation_constants))
                        count -= incrementor
                        counting_array[counting_index] += incrementation_constants[incrementor-1]

                insert_outvalues(eqn, context_dic, (counting_array, incrementation_constants))

            elif eqn.primitive.name == "jasp.measure":

                counting_index = profiling_dic["measure"]
                counting_array = list(invalues[-1][0])
                incrementation_constants = invalues[-1][1]

                meas_number = counting_array[counting_index]

                if isinstance(eqn.invars[0].aval, AbstractQubitArray):

                    def rng_body(i, acc):
                        meas_key = key(meas_number + i)
                        meas_res = meas_behavior(meas_key)

                        if (
                            not isinstance(meas_res, bool)
                            and not meas_res.dtype == jnp.bool
                        ):
                            raise Exception(
                                f"Tried to profil Jaspr with a measurement behavior not returning a boolean (got {meas_res.dtype}) instead"
                            )
                        acc = acc + (1 << i) * meas_res
                        return acc

                    meas_res = jax.lax.fori_loop(0, invalues[0], rng_body, jnp.int64(0))
                    counting_array[counting_index] += invalues[0]
                else:
                    meas_res = meas_behavior(key(meas_number))
                    if (
                        not isinstance(meas_res, bool)
                        and not meas_res.dtype == jnp.bool
                    ):
                        raise Exception(
                            f"Tried to profil Jaspr with a measurement behavior not returning a boolean (got {meas_res.dtype}) instead"
                        )
                        
                    counting_array[counting_index] += incrementation_constants[1]

                insert_outvalues(eqn, context_dic, [meas_res, (counting_array, incrementation_constants)])

            # Since we don't need to track to which qubits a certain operation
            # is applied, we can implement a really simple behavior for most
            # Qubit/QubitArray handling methods.
            # We represent qubit arrays simply with integers (indicating their)
            # size.
            elif eqn.primitive.name == "jasp.create_qubits":
                # create_qubits has the signature (size, QuantumCircuit).
                # Since we represent QubitArrays via integers, it is sufficient
                # to simply return the input as the output for this primitive.
                insert_outvalues(eqn, context_dic, invalues)

            elif eqn.primitive.name == "jasp.get_size":
                # The QubitArray size is represented via an integer.
                insert_outvalues(eqn, context_dic, invalues[0])

            elif eqn.primitive.name == "jasp.fuse":
                # The size of the fused qubit array is the size of the two added.
                insert_outvalues(eqn, context_dic, invalues[0] + invalues[1])

            elif eqn.primitive.name == "jasp.slice":
                # For the slice operation, we need to make sure, we don't go out
                # of bounds.
                start = jnp.max(jnp.array([invalues[1], 0]))
                stop = jnp.min(jnp.array([invalues[2], invalues[0]]))

                insert_outvalues(eqn, context_dic, stop - start)

            elif eqn.primitive.name == "jasp.get_qubit":
                # Trivial behavior since we don't need qubit address information
                insert_outvalues(eqn, context_dic, None)

            elif eqn.primitive.name in ["jasp.delete_qubits", "jasp.reset"]:
                # Trivial behavior: return the last argument (the counting array).
                insert_outvalues(eqn, context_dic, invalues[-1])
            elif eqn.primitive.name == "jasp.create_quantum_kernel":
                raise Exception("Tried to perform resource estimation on a function calling calling a kernelized function")
            else:
                raise Exception(
                    f"Don't know how to perform resource estimation with quantum primitive {eqn.primitive}"
                )

        elif eqn.primitive.name == "while":
            
            overall_constant_amount= max(eqn.params["body_nconsts"], eqn.params["cond_nconsts"])
            
            # Reinterpreted body and cond function
            def body_fun(val):
                
                constants = val[:eqn.params["body_nconsts"]]
                carries = val[overall_constant_amount:]
                
                body_res = eval_jaxpr(
                    eqn.params["body_jaxpr"], eqn_evaluator=profiling_eqn_evaluator
                )(*(constants + carries))
                
                if not isinstance(body_res, tuple):
                    body_res = (body_res,)
                
                return val[:overall_constant_amount] + tuple(body_res)

            def cond_fun(val):
                
                constants = val[:eqn.params["cond_nconsts"]]
                carries = val[overall_constant_amount:]
                
                res = eval_jaxpr(
                    eqn.params["cond_jaxpr"], eqn_evaluator=profiling_eqn_evaluator
                )(*(constants + carries))
                
                return res

            outvalues = jax.lax.while_loop(cond_fun, body_fun, tuple(invalues))[overall_constant_amount:]

            insert_outvalues(eqn, context_dic, outvalues)

        elif eqn.primitive.name == "cond":

            # Reinterpret branches
            branch_list = []

            for i in range(len(eqn.params["branches"])):
                branch_list.append(
                    eval_jaxpr(
                        eqn.params["branches"][i], eqn_evaluator=profiling_eqn_evaluator
                    )
                )

            outvalues = jax.lax.switch(invalues[0], branch_list, *invalues[1:])

            if len(eqn.outvars) == 1:
                outvalues = (outvalues,)

            insert_outvalues(eqn, context_dic, outvalues)

        elif eqn.primitive.name == "pjit":

            # For qached functions, we want to make sure, the compiled function
            # contains only a single implementation per qached function.

            # Within a Jaspr, it is made sure that qached function, which is called
            # multiple times only calls the Jaspr by reference in the pjit primitive
            # this way no "copies" of the implementation appear, thus keeping
            # the size of the intermediate representation limited.

            # We want to carry on this property. For this we use the lru_cache feature
            # on the get_compiler_profiler function. This function returns a
            # jitted function, which will always return the same object if called
            # with the same object. Since identical qached function calls are
            # represented by the same jaxpr, we achieve our goal.

            zipped_profiling_dic = tuple(profiling_dic.items())

            profiler = get_compiled_profiler(
                eqn.params["jaxpr"], zipped_profiling_dic, meas_behavior
            )

            outvalues = profiler(*invalues)

            if len(eqn.outvars) == 1:
                outvalues = (outvalues,)

            insert_outvalues(eqn, context_dic, outvalues)

        else:
            return True

    return profiling_eqn_evaluator


@lru_cache(int(1e5))
def get_compiled_profiler(jaxpr, zipped_profiling_dic, meas_behavior):

    profiling_dic = dict(zipped_profiling_dic)

    profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic, meas_behavior)

    jitted_profiler = jax.jit(eval_jaxpr(jaxpr, eqn_evaluator=profiling_eqn_evaluator))
    
    call_counter = np.zeros(1)
    
    def profiler(*args):
        if call_counter[0] < 3 or len(jaxpr.eqns) < 20:
            call_counter[0] += 1
            return eval_jaxpr(jaxpr, eqn_evaluator=profiling_eqn_evaluator)(*args)
        else:
            return jitted_profiler(*args)

    return profiler
