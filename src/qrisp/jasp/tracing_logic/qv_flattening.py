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

import copy
from jax import tree_util
import jax.numpy as jnp
from qrisp.jasp.tracing_logic import (
    TracingQuantumSession,
    DynamicQubitArray,
    check_for_tracing_mode,
)


# This class has two purposes

# 1. enable flattening of typed QuantumVariables.
# The flattening/unflattening process needs to track the type of the QuantumVariable
# for reconstruction. For this matter, the QuantumVariableTemplate keeps a
# copy of the type and reconstructs it when unflattening.


# 2. The second usecase is for passing quantum type information to kernelized
# functions.
# This is a problem because passing the QuantumVariable itself to .duplicate
# results in an error because the quantum_kernel interprets this as a passed
# quantum value.
# The QuantumVariable template doesn't carry the register information (only the
# size) and can therefore be passed around like a classical value.
class QuantumVariableTemplate:

    def __init__(self, qv, size_tracked=True):
        self.duplication_counter = 0
        self.qv = copy.copy(qv)
        self.qv.reg = None
        self.size_tracked = size_tracked
        self.qv_size = None
        if size_tracked:
            self.qv_size = qv.size

    def construct(self, reg=None):
        res = copy.copy(self.qv)
        res.name = self.qv.name + "duplicate_" + str(self.duplication_counter)
        self.duplication_counter += 1

        if check_for_tracing_mode():
            qs = TracingQuantumSession.get_instance()
        else:
            from qrisp import QuantumSession

            qs = QuantumSession()

        res.qs = qs
        if reg is None:
            if not self.size_tracked:
                raise Exception(
                    "Tried to construct QuantumVariable from template lacking a size specification"
                )

            qs.register_qv(res, self.qv_size)
        else:
            res.reg = reg
        return res

    def __hash__(self):
        return hash(type(self.qv))

    def __eq__(self, other):
        return type(self.qv) == type(other.qv)


def flatten_qv(qv):
    # return the tracers and auxiliary data (structure of the object)
    children = [qv.reg.tracer]
    for traced_attribute in qv.traced_attributes:
        attr = getattr(qv, traced_attribute)
        if isinstance(attr, bool):
            attr = jnp.array(attr, jnp.dtype("bool"))
        elif isinstance(attr, int):
            attr = jnp.array(attr, jnp.dtype("int64"))
        elif isinstance(attr, float):
            attr = jnp.array(attr, jnp.dtype("float64"))
        children.append(attr)

    return tuple(children), QuantumVariableTemplate(qv, False)


def unflatten_qv(aux_data, children):
    # The unflattening procedure creates a copy of the QuantumVariable object
    # and updates the traced attributes. When calling this procedure,
    # the user has to make sure that the result of this function is
    # registered in a QuantumSession.

    qv_container = aux_data
    reg = DynamicQubitArray(children[0])
    qv = qv_container.construct(reg)
    # qv = copy.copy(qv_container.qv)
    # qv.reg = reg
    qv.qs = None

    for i in range(len(qv.traced_attributes)):
        setattr(qv, qv.traced_attributes[i], children[i + 1])
    return qv


def flatten_template(tmpl):
    children = []
    qv = tmpl.qv
    for traced_attribute in qv.traced_attributes:
        attr = getattr(qv, traced_attribute)
        if isinstance(attr, bool):
            attr = jnp.array(attr, jnp.dtype("bool"))
        elif isinstance(attr, int):
            attr = jnp.array(attr, jnp.dtype("int64"))
        elif isinstance(attr, float):
            attr = jnp.array(attr, jnp.dtype("float64"))
        children.append(attr)

    if tmpl.size_tracked:
        children.append(tmpl.qv_size)

    return tuple(children), tmpl


def unflatten_template(aux_data, children):
    res = copy.copy(aux_data)
    res.qv = copy.copy(res.qv)
    for i in range(len(res.qv.traced_attributes)):
        setattr(res.qv, res.qv.traced_attributes[i], children[i])

    if res.size_tracked:
        res.qv_size = children[-1]

    return res


tree_util.register_pytree_node(
    QuantumVariableTemplate, flatten_template, unflatten_template
)
