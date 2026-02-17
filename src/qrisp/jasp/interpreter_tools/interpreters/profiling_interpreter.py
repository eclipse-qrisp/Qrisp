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
from typing import Any, Callable, Dict, Sequence

import jax
import numpy as np
from jax._src.core import JaxprEqn
from jax.typing import ArrayLike

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
    """
    Runtime-enforced base class for profiling metrics.

    Classes inheriting from `BaseMetric` must implement handler methods
    for all quantum primitives, and define an `initial_metric` property
    that specifies the initial value of the metric before any primitives are executed.


    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    profiling_dic : dict
        A dictionary mapping quantum operation names to their profiling indices.

    """

    def __init__(self, meas_behavior: Callable, profiling_dic: dict):
        """Initialize the BaseMetric."""

        self._meas_behavior = meas_behavior
        self._profiling_dic = profiling_dic

    @property
    def meas_behavior(self) -> Callable:
        return self._meas_behavior

    @property
    def profiling_dic(self) -> dict:
        return self._profiling_dic

    @property
    @abstractmethod
    def initial_metric(self) -> Any: ...

    def _validate_measurement_result(self, meas_res: bool | jax.Array) -> None:
        """Validate that measurement result is a boolean."""

        if isinstance(meas_res, bool):
            return
        if hasattr(meas_res, "dtype") and meas_res.dtype == jax.numpy.bool_:
            return
        raise ValueError(
            f"Measurement behavior must return a boolean, got {meas_res} of type {type(meas_res)}."
        )

    def _measurement_body_fun(
        self, meas_number: ArrayLike, i: ArrayLike, acc: ArrayLike
    ) -> ArrayLike:
        """Helper function for measuring qubit arrays."""

        meas_key = jax.random.key(meas_number + i)
        meas_res = self.meas_behavior(meas_key)
        self._validate_measurement_result(meas_res)
        return acc + jax.numpy.left_shift(1, i) * meas_res

    @abstractmethod
    def handle_create_qubits(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.create_qubits` primitive.

        The `create_qubits_p` primitive has the following semantics:

        - Invars: (size, QuantumCircuit)

        - Outvars: (QubitArray, QuantumCircuit)
        """

    @abstractmethod
    def handle_get_qubit(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.get_qubit` primitive.

        The `get_qubit_p` primitive has the following semantics:

        - Invars: (QubitArray, index)

        - Outvars: (Qubit)
        """

    @abstractmethod
    def handle_get_size(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.get_size` primitive.

        The `get_size_p` primitive has the following semantics:

        - Invars: (QubitArray)

        - Outvars: (size)
        """

    @abstractmethod
    def handle_fuse(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.fuse` primitive.

        The `fuse_p` primitive has the following semantics:

        - Invars: (QubitArray, QubitArray)

        - Outvars: (QubitArray)
        """

    @abstractmethod
    def handle_slice(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.slice` primitive.

        The `slice_p` primitive has the following semantics:

        - Invars: (QubitArray, start, stop)

        - Outvars: (QubitArray)
        """

    @abstractmethod
    def handle_quantum_gate(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.quantum_gate` primitive.

        The `quantum_gate_p` primitive has the following semantics:

        - Invars: (Qubit 0, Qubit 1,  ...  , Param 0, Param 1 ... , QuantumCircuit)

        - Outvars: (QuantumCircuit)
        """

    @abstractmethod
    def handle_measure(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.measure` primitive.

        The `measure_p` primitive has the following semantics:

        - Invars: (Qubit | QubitArray, QuantumCircuit)

        - Outvars: (meas_result, QuantumCircuit)
        """

    @abstractmethod
    def handle_reset(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.reset` primitive.

        The `reset_p` primitive has the following semantics:

        - Invars: (QubitArray, QuantumCircuit)

        - Outvars: (QuantumCircuit)
        """

    @abstractmethod
    def handle_delete_qubits(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.delete_qubits` primitive.

        The `delete_qubits_p` primitive has the following semantics:

        - Invars: (QubitArray, QuantumCircuit)

        - Outvars: (QuantumCircuit)
        """

    @abstractmethod
    def handle_parity(
        self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict
    ) -> Sequence:
        """
        Handle the `jasp.parity` primitive.

        TODO: add semantics here.
        """

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
            "jasp.create_quantum_kernel": self.handle_create_quantum_kernel,
            "jasp.parity": self.handle_parity,
        }


# TODO: We should refactor the way in which parameters are passed to the compiled profiler.
#
# This reconstructs the metric inside the cached function so caching not keyed
# by the metric object identity.
@lru_cache(int(1e5))
def get_compiled_profiler(
    jaxpr,
    metric_cls,
    zipped_profiling_dic,
    meas_behavior,
    max_qubits=None,
    max_allocations=None,
):
    """Get a compiled profiler for a given Jaspr and metric."""

    profiling_dic = dict(zipped_profiling_dic)
    metric_kwargs = {
        "profiling_dic": profiling_dic,
        "meas_behavior": meas_behavior,
        **({"max_qubits": max_qubits} if max_qubits is not None else {}),
        **({"max_allocations": max_allocations} if max_allocations is not None else {}),
    }

    metric = metric_cls(**metric_kwargs)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(metric)
    jaxpr_evaluator = eval_jaxpr(jaxpr, eqn_evaluator=profiling_eqn_evaluator)
    jitted_profiler = jax.jit(jaxpr_evaluator)

    call_counter = np.zeros(1)

    def profiler(*args):
        if call_counter[0] < 3 or len(jaxpr.eqns) < 20:
            call_counter[0] += 1
            return jaxpr_evaluator(*args)

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

    prim_handlers = metric.get_handlers()

    def profiling_eqn_evaluator(eqn, context_dic: ContextDict):

        invalues = extract_invalues(eqn, context_dic)
        prim = eqn.primitive

        if isinstance(prim, QuantumPrimitive):
            prim_handler = prim_handlers.get(prim.name, None)
            if prim_handler is None:
                raise NotImplementedError(
                    f"Don't know how to handle quantum primitive {prim.name} in profiling interpreter."
                )

            outvalues = prim_handler(invalues, eqn, context_dic)
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

        elif eqn.primitive.name == "scan":

            # Reinterpret the scan body function
            scan_body = eval_jaxpr(
                eqn.params["jaxpr"], eqn_evaluator=profiling_eqn_evaluator
            )

            # Extract scan parameters
            num_consts = eqn.params["num_consts"]
            num_carry = eqn.params["num_carry"]
            length = eqn.params["length"]
            reverse = eqn.params.get("reverse", False)
            unroll = eqn.params.get("unroll", 1)

            # Separate inputs
            consts = invalues[:num_consts]
            init = invalues[num_consts : num_consts + num_carry]
            xs = invalues[num_consts + num_carry :]

            # Create a wrapper function that includes constants
            if num_consts > 0:

                def wrapped_body(carry, x):
                    args = (
                        consts + list(carry) + list(x)
                        if isinstance(x, tuple)
                        else consts + list(carry) + [x]
                    )
                    result = scan_body(*args)
                    if not isinstance(result, tuple):
                        result = (result,)
                    return result[:num_carry], result[num_carry:]

            else:

                def wrapped_body(carry, x):
                    args = list(carry) + (list(x) if isinstance(x, tuple) else [x])
                    result = scan_body(*args)
                    if not isinstance(result, tuple):
                        result = (result,)
                    return result[:num_carry], result[num_carry:]

            # Call JAX scan with the reinterpreted body
            if len(xs) == 1:
                xs_arg = xs[0]
            else:
                xs_arg = tuple(xs)

            if len(init) == 1:
                init_arg = init[0]
            else:
                init_arg = tuple(init)

            final_carry, ys = jax.lax.scan(
                wrapped_body,
                init_arg,
                xs_arg,
                length=length,
                reverse=reverse,
                unroll=unroll,
            )

            # Prepare output
            if not isinstance(final_carry, tuple):
                final_carry = (final_carry,)
            if not isinstance(ys, tuple):
                ys = (ys,)

            outvalues = final_carry + ys

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
            max_qubits = getattr(metric, "max_qubits", None)
            max_allocations = getattr(metric, "max_allocations", None)

            profiler = get_compiled_profiler(
                eqn.params["jaxpr"],
                type(metric),  # used only to reconstruct metric inside cache
                zipped_profiling_dic,
                meas_behavior,
                max_qubits,
                max_allocations,
            )

            outvalues = profiler(*invalues)

            if len(eqn.outvars) == 1:
                outvalues = (outvalues,)

            insert_outvalues(eqn, context_dic, outvalues)

        else:
            return True

    return profiling_eqn_evaluator
