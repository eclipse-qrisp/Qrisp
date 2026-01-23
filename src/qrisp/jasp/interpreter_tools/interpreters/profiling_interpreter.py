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


from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, Dict

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


class BaseMetric(ABC):
    """Runtime-enforced base class for profiling metrics."""

    # create_qubits has the signature (size, QuantumCircuit)
    # Outvars are (QubitArray, QuantumCircuit)
    @abstractmethod
    def handle_create_qubits(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.create_qubits` primitive."""

    # get_qubit has the signature (QubitArray, index (int))
    # Outvars are (Qubit)
    @abstractmethod
    def handle_get_qubit(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.get_qubit` primitive."""

    # get_size has the signature (QubitArray)
    # Outvars are (size)
    @abstractmethod
    def handle_get_size(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.get_size` primitive."""

    # fuse has the signature (QubitArray, QubitArray)
    # Outvars are (QubitArray)
    @abstractmethod
    def handle_fuse(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.fuse` primitive."""

    # slice has the signature (QubitArray, start (int), stop (int))
    # Outvars are (QubitArray)
    @abstractmethod
    def handle_slice(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.slice` primitive."""

    # quantum_gate has the signature (Qubit, ... , QuantumCircuit)
    # Outvars is (QuantumCircuit)
    @abstractmethod
    def handle_quantum_gate(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.quantum_gate` primitive."""

    # measure has the signature (Qubit | QubitArray, QuantumCircuit)
    # Outvars are (meas_result, QuantumCircuit)
    @abstractmethod
    def handle_measure(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.measure` primitive."""

    # reset has the signature (QubitArray, QuantumCircuit)
    # Outvars are (QuantumCircuit)
    @abstractmethod
    def handle_reset(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.reset` primitive."""

    # delete_qubits has the signature (QubitArray, QuantumCircuit)
    # Outvars are (QuantumCircuit)
    @abstractmethod
    def handle_delete_qubits(self, *, invalues, eqn, context_dic: ContextDict):
        """Handle the `jasp.delete_qubits` primitive."""

    def handle_create_quantum_kernel(self, *_args, **_kwargs):
        """Handle the `jasp.create_quantum_kernel` primitive."""

        raise NotImplementedError(
            "Quantum kernel creation not yet supported in profiling interpreter."
        )

    def get_handlers(self) -> Dict[str, Callable[..., Any]]:
        """Return a mapping from primitive names to handler methods."""

        return {
            "jasp.create_qubits": self.handle_create_qubits,
            "jasp.get_qubit": self.handle_get_qubit,
            "jasp.get_size": self.handle_get_size,
            "jasp.fuse": self.handle_fuse,
            "jasp.slice": self.handle_slice,
            "jasp.quantum_gate": self.handle_quantum_gate,
            "jasp.measure": self.handle_measure,
            "jasp.reset": self.handle_reset,
            "jasp.delete_qubits": self.handle_delete_qubits,
        }


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


def make_profiling_eqn_evaluator(metric: BaseMetric) -> Callable:
    """
    Build a profiling equation evaluator for a given metric.

    Parameters
    ----------
    metric : BaseMetric
        The metric to use for profiling.

    Returns
    -------
    Callable
        The profiling equation evaluator.
    """

    # We cache once per call and use closure + O(1) lookup to handle primitives.
    handlers = dict(metric.get_handlers())

    # In this interpreter, context_dic is an environment mapping:
    # - keys: JAXPR variables (eqn.outvars[i], i.e. SSA names)
    # - values: whatever concrete (or traced) JAX objects every metric chooses to represent them with
    def profiling_eqn_evaluator(eqn, context_dic: ContextDict):

        invalues = extract_invalues(eqn, context_dic)
        prim = eqn.primitive

        if isinstance(prim, QuantumPrimitive):

            prim_handler = handlers.get(prim.name, None)

            if prim_handler is None:
                raise NotImplementedError(
                    f"Don't know how to handle quantum primitive {prim.name} in profiling interpreter."
                )

            outvalues = prim_handler(
                invalues=invalues, eqn=eqn, context_dic=context_dic
            )
            insert_outvalues(eqn, context_dic, outvalues)

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
