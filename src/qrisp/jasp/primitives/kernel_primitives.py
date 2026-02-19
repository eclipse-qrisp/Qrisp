"""
\********************************************************************************
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
********************************************************************************/
"""

import jax.numpy as jnp
from qrisp.jasp.primitives.quantum_primitive import QuantumPrimitive
from qrisp.jasp.primitives.abstract_quantum_state import AbstractQuantumState

create_quantum_kernel_p = QuantumPrimitive("create_quantum_kernel")
consume_quantum_kernel_p = QuantumPrimitive("consume_quantum_kernel")


@create_quantum_kernel_p.def_abstract_eval
def quantum_kernel_abstract_eval():
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """

    return AbstractQuantumState()


@consume_quantum_kernel_p.def_abstract_eval
def quantum_kernel_abstract_eval(abs_qst):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """

    return jnp.bool(False).aval
