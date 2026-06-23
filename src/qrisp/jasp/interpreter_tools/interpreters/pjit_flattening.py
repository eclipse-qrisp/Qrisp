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

from jax.core import ClosedJaxpr
from jax import jit
from qrisp.jasp.interpreter_tools import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
    reinterpret,
)


def evaluate_pjit_eqn(pjit_eqn, context_dic):

    definition_jaxpr = pjit_eqn.params["jaxpr"].jaxpr

    # Extract the invalues from the context dic
    invalues = extract_invalues(pjit_eqn, context_dic)

    res = jit(eval_jaxpr(definition_jaxpr), inline=True)(*invalues)

    if len(definition_jaxpr.outvars) == 1:
        res = [res]

    # Insert the values into the context_dic
    insert_outvalues(pjit_eqn, context_dic, res)


# Flattens/Inlines a pjit calls in a jaxpr
def flatten_pjit(jaxpr):

    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr

    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "pjit":
            evaluate_pjit_eqn(eqn, context_dic)
        else:
            return True

    return type(jaxpr)(reinterpret(jaxpr, eqn_evaluator))
