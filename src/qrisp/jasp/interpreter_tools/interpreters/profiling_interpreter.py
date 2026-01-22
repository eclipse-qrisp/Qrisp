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
from typing import Callable

import jax
import numpy as np

from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    ContextDict,
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.primitives import (
    QuantumPrimitive,
)


# This reconstructs the metric inside the cached function so caching not keyed
# by the metric object identity.
@lru_cache(int(1e5))
def get_compiled_profiler(jaxpr, metric_cls, zipped_profiling_dic, meas_behavior):

    profiling_dic = dict(zipped_profiling_dic)
    metric = metric_cls(profiling_dic, meas_behavior)

    profiling_eqn_evaluator = make_profiling_eqn_evaluator(metric)
    jaxpr_evaluator = eval_jaxpr(jaxpr, eqn_evaluator=profiling_eqn_evaluator)
    jitted_profiler = jax.jit(jaxpr_evaluator)

    call_counter = np.zeros(1)

    def profiler(*args):
        if call_counter[0] < 3 or len(jaxpr.eqns) < 20:
            call_counter[0] += 1
            return jaxpr_evaluator(*args)
        else:
            return jitted_profiler(*args)

    return profiler


def make_profiling_eqn_evaluator(metric) -> Callable:
    """
    Build a profiling equation evaluator for a given metric.

    Parameters
    ----------
    metric
        The metric to use for profiling.

    Returns
    -------
    Callable
        The profiling equation evaluator.
    """

    # In this interpreter, context_dic is an environment mapping:
    # - keys: JAXPR variables (eqn.outvars[i], i.e. SSA names)
    # - values: whatever concrete (or traced) JAX objects every metric chooses to represent them with
    def profiling_eqn_evaluator(eqn, context_dic: ContextDict):

        invalues = extract_invalues(eqn, context_dic)
        prim_name = eqn.primitive.name

        if isinstance(eqn.primitive, QuantumPrimitive):

            # TODO: replace this with a more compact and efficient dispatch mechanism
            # after everything has been unified to use this new profiling interpreter structure.
            match prim_name:

                # create_qubits has the signature (size, QuantumCircuit)
                case "jasp.create_qubits":
                    outvalues = metric.handle_create_qubits(invalues, context_dic)
                    # Outvars are (QubitArray, QuantumCircuit)
                    insert_outvalues(eqn, context_dic, outvalues)

                # get_qubit has the signature (QubitArray, index (int))
                case "jasp.get_qubit":
                    outvalues = metric.handle_get_qubit(invalues)
                    # Outvars are (Qubit)
                    insert_outvalues(eqn, context_dic, outvalues)

                # slice has the signature (QubitArray)
                case "jasp.get_size":
                    outvalues = metric.handle_get_size(invalues)
                    # Outvars are (size)
                    insert_outvalues(eqn, context_dic, outvalues)

                # fuse has the signature (QubitArray, QubitArray)
                case "jasp.fuse":
                    outvalues = metric.handle_fuse(invalues)
                    # Outvars are (QubitArray)
                    insert_outvalues(eqn, context_dic, outvalues)

                # slice has the signature (QubitArray, start (int), stop (int))
                case "jasp.slice":
                    outvalues = metric.handle_slice(invalues)
                    # Outvars are (QubitArray)
                    insert_outvalues(eqn, context_dic, outvalues)

                # quantum_gate has the signature (Qubit, ... , QuantumCircuit)
                # (it depends on the gate how many qubits there are)
                case "jasp.quantum_gate":
                    outvalues = metric.handle_quantum_gate(invalues, eqn)
                    # Outvars is (QuantumCircuit)
                    insert_outvalues(eqn, context_dic, outvalues)

                # measure has the signature (Qubit | QubitArray, QuantumCircuit)
                case "jasp.measure":
                    outvalues = metric.handle_measure(invalues, eqn)
                    # Outvars are (meas_result, QuantumCircuit)
                    insert_outvalues(eqn, context_dic, outvalues)

                # reset has the signature (QubitArray, QuantumCircuit)
                case "jasp.reset":
                    outvalues = metric.handle_reset(invalues)
                    # Outvars are (QuantumCircuit)
                    insert_outvalues(eqn, context_dic, outvalues)

                # delete_qubits has the signature (QubitArray, QuantumCircuit)
                case "jasp.delete_qubits":
                    outvalues = metric.handle_delete_qubits(invalues)
                    # Outvars are (QuantumCircuit)
                    insert_outvalues(eqn, context_dic, outvalues)

                case "jasp.create_quantum_kernel":
                    raise NotImplementedError(
                        "Quantum kernel creation not yet supported in profiling interpreter."
                    )

                case _:
                    raise NotImplementedError(
                        f"Don't know how to handle quantum primitive {prim_name} in profiling interpreter."
                    )

        elif eqn.primitive.name == "cond":

            branch_fns = [
                eval_jaxpr(branch_jaxpr, eqn_evaluator=profiling_eqn_evaluator)
                for branch_jaxpr in eqn.params["branches"]
            ]

            # invalues[0] is the branch index/predicate encoding
            # remaining invalues are operands/carries passed to the branches
            outvalues = jax.lax.switch(invalues[0], branch_fns, *invalues[1:])
            outvalues = (outvalues,) if len(eqn.outvars) == 1 else outvalues
            insert_outvalues(eqn, context_dic, outvalues)

        elif eqn.primitive.name == "while":

            body_jaxpr = eqn.params["body_jaxpr"]
            cond_jaxpr = eqn.params["cond_jaxpr"]
            body_nconsts = eqn.params["body_nconsts"]
            cond_nconsts = eqn.params["cond_nconsts"]

            overall_constant_amount = body_nconsts + cond_nconsts

            body_eval = eval_jaxpr(body_jaxpr, eqn_evaluator=profiling_eqn_evaluator)
            cond_eval = eval_jaxpr(cond_jaxpr, eqn_evaluator=profiling_eqn_evaluator)

            def body_fun(val):
                constants = val[cond_nconsts:overall_constant_amount]
                carries = val[overall_constant_amount:]
                body_res = body_eval(*(constants + carries))
                body_res = body_res if isinstance(body_res, tuple) else (body_res,)
                return val[:overall_constant_amount] + body_res

            def cond_fun(val):
                constants = val[:cond_nconsts]
                carries = val[overall_constant_amount:]
                return cond_eval(*(constants + carries))

            outvalues = jax.lax.while_loop(cond_fun, body_fun, tuple(invalues))[
                overall_constant_amount:
            ]

            insert_outvalues(eqn, context_dic, outvalues)

        elif eqn.primitive.name == "jit":

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

            zipped_profiling_dic = tuple(metric.profiling_dic.items())
            meas_behavior = metric.meas_behavior

            profiler = get_compiled_profiler(
                eqn.params["jaxpr"],
                type(metric),  # used only to reconstruct metric inside cache
                zipped_profiling_dic,
                meas_behavior,
            )

            outvalues = profiler(*invalues)

            if len(eqn.outvars) == 1:
                outvalues = (outvalues,)

            insert_outvalues(eqn, context_dic, outvalues)

        else:
            return True

    return profiling_eqn_evaluator
