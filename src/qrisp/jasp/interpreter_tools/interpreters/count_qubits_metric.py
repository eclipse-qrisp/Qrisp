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

import types
from functools import lru_cache
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.random import key
from jax.typing import ArrayLike

from qrisp.circuit.operation import Operation
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    BaseMetric,
    eval_jaxpr,
    make_profiling_eqn_evaluator,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    get_quantum_operations,
    is_abstract,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


class CountQubitsMetric(BaseMetric):
    """
    A metric implementation that computes the count of qubits in a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    profiling_dic : dict
        The profiling dictionary mapping quantum operations to indices.

    """

    def __init__(self, meas_behavior: Callable, profiling_dic: dict):
        """Initialize the CountQubitsMetric."""

        super().__init__(meas_behavior=meas_behavior, profiling_dic=profiling_dic)

        self._initial_metric = ...

    @property
    def initial_metric(self) -> ...:
        """Return the initial metric value."""
        return self._initial_metric

    def handle_create_qubits(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # QubitArray -> ...
        # QuantumCircuit -> metric_data ...
        return ...

    def handle_get_qubit(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # Qubit -> ...
        return ...

    def handle_get_size(self, invalues, eqn, context_dic):
        """Handle the `jasp.get_size` primitive."""

        ...

        # Associate the following in context_dic:
        # size -> ...
        return ...

    def handle_quantum_gate(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # QuantumCircuit -> ...
        return ...

    def handle_measure(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # meas_result -> ...
        # QuantumCircuit -> ...
        return ...

    def handle_fuse(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # QubitArray -> ...
        return ...

    def handle_slice(self, invalues, eqn, context_dic):

        ...

        ...

        # Associate the following in context_dic:
        # QubitArray -> ...
        return ...

    def handle_reset(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # QuantumCircuit -> ...
        return ...

    def handle_delete_qubits(self, invalues, eqn, context_dic):

        ...

        # Associate the following in context_dic:
        # QuantumCircuit -> ...
        return ...


def extract_count_qubits(res: Tuple, jaspr: Jaspr, _):
    """Extract count_qubits from the profiling result."""

    pass


@lru_cache(int(1e5))
def get_count_qubits_profiler(
    jaspr: Jaspr, meas_behavior: Callable
) -> Tuple[Callable, None]:
    """
    Build a count qubits profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    Returns
    -------
    Tuple[Callable, None]
        A count qubits profiler function and None as auxiliary data.

    """
    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    count_qubits_metric = CountQubitsMetric(meas_behavior, profiling_dic)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(count_qubits_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def count_qubits_profiler(*args):
        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        # Import here to avoid circular import issues
        from qrisp.operators import FermionicOperator, QubitOperator

        STATIC_TYPES = (str, QubitOperator, FermionicOperator, types.FunctionType)

        initial_metric_value = count_qubits_metric.initial_metric
        filtered_args = [
            x for x in args + (initial_metric_value,) if type(x) not in STATIC_TYPES
        ]
        return jitted_evaluator(*filtered_args)

    return count_qubits_profiler, None


def simulate_count_qubits(jaspr: Jaspr, *_, **__) -> int:
    """Simulate count_qubits metric via actual simulation."""

    raise NotImplementedError(
        "Count qubits metric via simulation is not implemented yet."
    )
