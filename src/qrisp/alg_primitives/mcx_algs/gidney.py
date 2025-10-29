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

from qrisp.alg_primitives.mcx_algs.circuit_library import (
    ctrl_state_wrap,
    gidney_qc,
    gidney_qc_inv,
    margolus_qc,
)
from qrisp.circuit import Operation, QuantumCircuit


class GidneyLogicalAND(Operation):

    def __init__(self, inv=False, ctrl_state="11"):

        definition = QuantumCircuit(3)

        if ctrl_state[0] == "0":
            definition.x(definition.qubits[0])
        if ctrl_state[1] == "0":
            definition.x(definition.qubits[1])

        definition.append(margolus_qc.to_gate("margolus"), definition.qubits)

        if ctrl_state[0] == "0":
            definition.x(definition.qubits[0])
        if ctrl_state[1] == "0":
            definition.x(definition.qubits[1])

        name = "uncompiled_gidney_mcx"
        if inv:
            name = name + "_inv"
        Operation.__init__(self, name=name, num_qubits=3, definition=definition)

        self.permeability = {0: True, 1: True, 2: False}
        self.inv = inv
        self.is_qfree = True
        self.ctrl_state = ctrl_state

    def inverse(self):
        return GidneyLogicalAND(inv=not self.inv, ctrl_state=self.ctrl_state)
        # Implementation from https://arxiv.org/abs/2101.04764

    def recompile(self, computation_strategy="gidney"):
        if self.inv:
            compiled_qc = gidney_qc_inv
        else:
            if computation_strategy == "gidney":
                compiled_qc = gidney_qc
            elif computation_strategy == "margolus":
                compiled_qc = margolus_qc
            else:
                raise Exception(
                    f"Don't know measurement based uncomputation strategy {computation_strategy}"
                )

        name = "compiled_gidney_mcx"
        if self.inv:
            name = name + "_inv"

        res = ctrl_state_wrap(compiled_qc, self.ctrl_state).to_op(name)
        res.is_qfree = True
        res.permeability = {0: True, 1: True, 2: False}
        res.ctrl_state = self.ctrl_state
        return res


from qrisp.core import cx, cz, h, measure, s, t, t_dg, x
from qrisp.environments import control
from qrisp.jasp import AbstractQubit, Jaspr, make_jaspr


def gidney_mcx_impl(a, b, c):

    h(c)
    t(c)

    cx(a, c)
    cx(b, c)
    cx(c, a)
    cx(c, b)

    t_dg(a)
    t_dg(b)
    t(c)

    cx(c, a)
    cx(c, b)

    h(c)
    s(c)


def gidney_mcx_inv_impl(a, b, c):
    h(c)
    bl = measure(c)

    with control(bl):
        cz(a, b)
        x(c)


class GidneyMCXJaspr(Jaspr):

    slots = ["inv"]

    def __init__(self, inv):
        self.inv = inv
        if self.inv:
            temp_jaspr = make_jaspr(gidney_mcx_inv_impl)(
                AbstractQubit(), AbstractQubit(), AbstractQubit()
            ).flatten_environments()
        else:
            temp_jaspr = make_jaspr(gidney_mcx_impl)(
                AbstractQubit(), AbstractQubit(), AbstractQubit()
            ).flatten_environments()

        Jaspr.__init__(self, temp_jaspr)
        self.envs_flattened = True

    def inverse(self):
        if self.inv:
            return gidney_mcx_jaspr
        else:
            return gidney_mcx_inv_jaspr


gidney_mcx_jaspr = GidneyMCXJaspr(False)
gidney_mcx_inv_jaspr = GidneyMCXJaspr(True)


def jasp_gidney_mcx(a, b, c):
    gidney_mcx_jaspr.embedd(a, b, c, name="gidney_mcx")


def jasp_gidney_mcx_inv(a, b, c):
    gidney_mcx_inv_jaspr.embedd(a, b, c, name="gidney_mcx_inv")
