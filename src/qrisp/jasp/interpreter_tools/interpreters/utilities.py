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

from typing import List

import jax
from jax.extend.core import ClosedJaxpr
from jax.typing import ArrayLike

from qrisp.jasp.jasp_expression import Jaspr


def always_zero(_):
    """Return False for all inputs, simulating measurements that always yield 0."""
    return False


def always_one(_):
    """Return True for all inputs, simulating measurements that always yield 1."""
    return True


def simulation():
    """Simulate measurements normally without any forced behavior."""
    pass


# Deterministic for every key
def meas_rng(key):
    """Simulate measurements using the provided random key."""
    return jax.numpy.bool_(jax.random.randint(key, (1,), 0, 2)[0])


def is_abstract(tensor: ArrayLike) -> bool:
    """Check if the input is an abstract JAX value.

    Parameters
    ----------
    tensor : ArrayLike
        The input to check.

    Returns
    -------
    bool
        True if the input is an abstract JAX value, False otherwise.

    """
    # Use jax.core.Tracer as base class to catch all tracer types including new ones in JAX 0.7.0+
    if isinstance(tensor, jax.core.Tracer):
        return not jax.core.is_concrete(tensor)
    return False


def get_quantum_operations(jaspr: Jaspr) -> List[str]:
    """
    Get the list of quantum operations used in a Jaspr expression.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to analyze.

    Returns
    -------
    List[str]
        A list of quantum operation names used in the Jaspr expression.

    """

    quantum_operations = set()

    for eqn in jaspr.eqns:

        if eqn.primitive.name == "jasp.quantum_gate":

            if eqn.params["gate"].definition:
                for op_name in (
                    eqn.params["gate"].definition.transpile().count_ops().keys()
                ):
                    quantum_operations.add(op_name)
            else:
                quantum_operations.add(eqn.params["gate"].name)

        if eqn.primitive.name == "cond":
            quantum_operations.update(
                get_quantum_operations(eqn.params["branches"][0].jaxpr)
            )
            quantum_operations.update(
                get_quantum_operations(eqn.params["branches"][1].jaxpr)
            )
            continue

        # Handle call primitives (like cond/pjit)
        for param in eqn.params.values():
            if isinstance(param, ClosedJaxpr):
                quantum_operations.update(get_quantum_operations(param.jaxpr))

    return list(quantum_operations)
