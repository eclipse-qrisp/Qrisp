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

import sys
import types

import jax

jax.config.update("jax_enable_x64", True)

from qrisp.alg_primitives import *
from qrisp.algorithms import *
from qrisp.circuit import *
from qrisp.core import *
from qrisp.environments import *
from qrisp.misc import *
from qrisp.permeability import *
from qrisp.qtypes import *

for i in [
    "shor",
    "qaoa",
    "qiro",
    "grover",
    "quantum_backtracking",
    "quantum_counting",
    "vqe",
    "qite",
    "qmci",
]:
    sys.modules["qrisp." + i] = sys.modules["qrisp.algorithms." + i]

from qrisp.default_backend import *
from qrisp.jasp import *


# Register some types as "always static" within Jasp
def register_static_types():
    from jax import tree_util

    def unflatten_function(aux_data, children):
        return aux_data

    def flatten_function(arg):
        # return the tracers and auxiliary data (structure of the object)
        return tuple(), arg

    try:
        tree_util.register_pytree_node(
            types.FunctionType, flatten_function, unflatten_function
        )
        tree_util.register_pytree_node(str, flatten_function, unflatten_function)
    except ValueError as e:
        if not "PyTree" in str(e):
            raise e

    from qrisp.operators import FermionicOperator, QubitOperator

    tree_util.register_pytree_node(QubitOperator, flatten_function, unflatten_function)
    tree_util.register_pytree_node(
        FermionicOperator, flatten_function, unflatten_function
    )


register_static_types()
