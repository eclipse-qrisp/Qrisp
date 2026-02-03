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
    ...
    """

    pass


def extract_count_qubits(res: Tuple, jaspr: Jaspr, _):
    """Extract count_qubits from the profiling result."""

    pass


@lru_cache(int(1e5))
def get_count_qubits_profiler(
    jaspr: Jaspr, meas_behavior: Callable
) -> Tuple[Callable, None]:

    pass


def simulate_count_qubits(jaspr: Jaspr, *_, **__) -> int:
    """Simulate count_qubits metric via actual simulation."""

    raise NotImplementedError(
        "Count qubits metric via simulation is not implemented yet."
    )
