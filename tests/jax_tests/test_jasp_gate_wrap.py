# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for gate_wrap in Jasp mode: permeability/qfree metadata on the Jasprs of qached functions."""

from qrisp import *
from qrisp.jasp import *


def jit_eqns(jaspr, name):
    return [eqn for eqn in jaspr.eqns if eqn.primitive.name == "jit" and eqn.params["name"] == name]


def invar_permeability(jaspr):
    return [jaspr.permeability.get(var) for var in jaspr.invars]


def outvar_permeability(jaspr):
    return [jaspr.permeability.get(var) for var in jaspr.outvars]


def test_gate_wrap_permeability_specification():

    @gate_wrap(permeability=[0], is_qfree=True)
    @qache
    def listed(a, b):
        cx(a[0], b[0])

    @gate_wrap(permeability="args", is_qfree=True)
    @qache
    def args_permeable(a, b):
        res = QuantumBool()
        mcx([a[0], b[0]], res[0])
        return res

    @gate_wrap(permeability="full", is_qfree=False)
    @qache
    def full(a):
        res = QuantumBool()
        cz(a[0], res[0])
        return res

    def main():
        a = QuantumVariable(1)
        b = QuantumVariable(1)
        listed(a, b)
        args_permeable(a, b)
        full(a)

    jaspr = make_jaspr(main)()

    listed_jaspr = jit_eqns(jaspr, "listed")[0].params["jaxpr"]
    assert invar_permeability(listed_jaspr) == [True, False, None]
    assert listed_jaspr.isqfree is True

    args_jaspr = jit_eqns(jaspr, "args_permeable")[0].params["jaxpr"]
    assert invar_permeability(args_jaspr) == [True, True, None]
    assert outvar_permeability(args_jaspr) == [False, None]

    full_jaspr = jit_eqns(jaspr, "full")[0].params["jaxpr"]
    assert invar_permeability(full_jaspr) == [True, None]
    assert outvar_permeability(full_jaspr) == [True, None]
    assert full_jaspr.isqfree is False


def test_gate_wrap_classical_and_static_arguments():

    @gate_wrap(permeability=[1], is_qfree=True)
    @qache(static_argnums=0)
    def fan_out(n, a, b, phi):
        for i in range(n):
            cx(a[0], b[i])
        p(phi, a[0])

    def main():
        a = QuantumVariable(1)
        b = QuantumVariable(3)
        fan_out(2, a, b, 0.5)

    jaspr = make_jaspr(main)()
    fan_out_jaspr = jit_eqns(jaspr, "fan_out")[0].params["jaxpr"]
    # The static n has no invar, the classical phi stays unspecified.
    assert invar_permeability(fan_out_jaspr) == [True, False, None, None]


def test_gate_wrap_name():

    @gate_wrap(name="custom_name")
    @qache
    def function(a):
        x(a[0])

    def main():
        function(QuantumVariable(1))

    assert len(jit_eqns(make_jaspr(main)(), "custom_name")) == 1


def test_gate_wrap_without_qache_is_inlined():

    @gate_wrap(permeability="args", is_qfree=True)
    def flip(a, indices):
        for i in indices:
            x(a[i])
        return a

    def main():
        a = QuantumVariable(3)
        a = flip(a, [0, 2])
        return measure(a)

    jaspr = make_jaspr(main)()
    assert not any(eqn.primitive.name == "jit" and eqn.params["name"] == "flip" for eqn in jaspr.eqns)
    assert jaspify(main)() == 5


def test_permeability_survives_transformations():

    @gate_wrap(permeability=[0], is_qfree=True)
    @qache
    def function(a, b):
        cx(a[0], b[0])

    def main():
        a = QuantumVariable(1)
        b = QuantumVariable(1)
        ctrl = QuantumBool()
        function(a, b)
        with invert():
            function(a, b)
        with control(ctrl):
            function(a, b)

    jaspr = make_jaspr(main)()

    base = jit_eqns(jaspr, "function")[0].params["jaxpr"]
    assert invar_permeability(base) == [True, False, None]

    # Inversion
    inverted = jit_eqns(jaspr, "function_dg")[0].params["jaxpr"]
    assert invar_permeability(inverted) == [True, False, None]
    assert inverted.isqfree is True
    assert invar_permeability(base.inverse()) == [True, False, None]

    # Control (the control qubits are permeable)
    ctrl_env = jit_eqns(jaspr, "ctrl_env")[0].params["jaxpr"]
    controlled = jit_eqns(ctrl_env, "cfunction")[0].params["jaxpr"]
    assert invar_permeability(controlled) == [True, True, False, None]
    assert controlled.isqfree is True
    assert invar_permeability(base.control(2, ctrl_state=1)) == [True, True, True, False, None]


def test_balauca_mcx_permeability():

    def find_balauca(jaspr):
        for eqn in jaspr.eqns:
            if eqn.primitive.name == "jit":
                if eqn.params["name"] == "jasp_balauca_mcx":
                    return eqn.params["jaxpr"]
                res = find_balauca(eqn.params["jaxpr"])
                if res is not None:
                    return res

    def main(n):
        ctrls = QuantumVariable(n)
        target = QuantumBool()
        mcx(ctrls, target[0])
        mcx(ctrls, target[0])
        return measure(target)

    # Static and dynamic control-register size
    for jaspr in [make_jaspr(main)(4), make_jaspr(lambda: main(4))()]:
        balauca_jaspr = find_balauca(jaspr)
        qubit_permeability = [
            balauca_jaspr.permeability[var] for var in balauca_jaspr.invars if str(var.aval) in ("Qubit", "QubitArray")
        ]
        assert qubit_permeability == [True, False]
        assert balauca_jaspr.isqfree is True

        # Repeated calls reuse the cached Jaspr
        balauca_eqns = jit_eqns(jaspr, "jasp_balauca_mcx")
        assert len(balauca_eqns) == 2
        assert balauca_eqns[0].params["jaxpr"] is balauca_eqns[1].params["jaxpr"]
