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

from jax.extend.core import ClosedJaxpr

from qrisp.jasp.jasp_expression import Jaspr


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
