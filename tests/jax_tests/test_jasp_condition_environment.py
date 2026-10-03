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

"""Tests ConditionEnvironment in Jasp: uncomputation of intermediate results, infix conditions, control and errors."""

import jax
import numpy as np
import pytest

from qrisp import *
from qrisp.environments.jasp_condition_compilation import ConditionCompilationError
from qrisp.jasp import *


@quantum_condition
def small_or_seven(qf):
    # Creates two intermediate QuantumBools, which must be uncomputed as well
    return (qf < 3) | (qf == 7)


def run_classical(body):
    """Run ``body(qf, t, u)`` for all values of a 3-qubit QuantumFloat with boolean simulation."""

    @boolean_simulation
    def main(value):
        qf = QuantumFloat(3)
        qf[:] = value
        t = QuantumBool()
        u = QuantumBool()
        body(qf, t, u)
        return measure(qf), measure(t), measure(u)

    return [tuple(int(r) for r in main(value)) for value in range(8)]


def phase_distribution(oracle):
    """Distribution of H oracle H applied to |0>, which reveals the phases applied by oracle."""

    @terminal_sampling
    def main(n):
        qf = QuantumFloat(n)
        h(qf)
        oracle(qf)
        h(qf)
        return qf

    return {k: round(v, 6) for k, v in main(3).items() if v > 1e-9}


def reference_oracle(qf, phase=None):
    # The oracle of small_or_seven, uncomputed by hand
    f1 = QuantumBool()
    f2 = QuantumBool()
    f3 = QuantumBool()
    with conjugate(f1 << (lambda qf: qf < 3))(qf):
        with conjugate(f2 << (lambda qf: qf == 7))(qf):
            with conjugate(f3 << (lambda a, b: a | b))(f1, f2):
                if phase is None:
                    z(f3)
                else:
                    p(phase, f3)
    f1.delete()
    f2.delete()
    f3.delete()


def test_condition_with_intermediates(capsys):

    def body(qf, t, u):
        with small_or_seven(qf):
            t.flip()

    assert run_classical(body) == [(v, v < 3 or v == 7, 0) for v in range(8)]

    jax.effects_barrier()
    assert "Faulty uncomputation" not in capsys.readouterr().out


def test_infix_conditions():

    def equal(qf, t, u):
        with qf == 5:
            t.flip()

    def less_than(qf, t, u):
        with qf < 3:
            t.flip()

    assert run_classical(equal) == [(v, v == 5, 0) for v in range(8)]
    assert run_classical(less_than) == [(v, v < 3, 0) for v in range(8)]


def test_compound_condition():

    def body(qf, t, u):
        with quantum_condition(lambda qf: (qf < 3) | (qf == 7))(qf):
            t.flip()

    assert run_classical(body) == [(v, v < 3 or v == 7, 0) for v in range(8)]


def test_comparison_inside_with_line_is_not_entered():
    # The reference evaluates comparisons inside lambdas on with lines. They
    # must return QuantumBools, not be entered as conditions.
    assert phase_distribution(reference_oracle) == {4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25}


def test_flip_inverts_the_condition():

    def body(qf, t, u):
        with small_or_seven(qf) as flag:
            t.flip()
            flag.flip()
            u.flip()

    assert run_classical(body) == [(v, v < 3 or v == 7, not (v < 3 or v == 7)) for v in range(8)]


def test_phase_on_truth_value():

    def oracle(qf):
        with small_or_seven(qf) as flag:
            z(flag)

    assert phase_distribution(oracle) == phase_distribution(reference_oracle)


def test_balauca_on_truth_value():
    # mcx and mcp use the Balauca implementation for a QuantumBool, which is
    # declared permeable on its controls

    def control(qf, t, u):
        with small_or_seven(qf) as flag:
            mcx(flag, t)

    def phase(qf):
        with small_or_seven(qf) as flag:
            mcp(np.pi, flag)

    assert run_classical(control) == [(v, v < 3 or v == 7, 0) for v in range(8)]
    assert phase_distribution(phase) == phase_distribution(reference_oracle)


@qache
def flip_if_false(flag, t):
    # Permeable on flag as a whole, but not gate by gate
    x(flag)
    cx(flag, t)
    x(flag)


@gate_wrap(permeability=[0], is_qfree=True)
@qache
def declared_flip_if_false(flag, t):
    x(flag)
    cx(flag, t)
    x(flag)


def test_declared_permeability():

    def declared(qf, t, u):
        with small_or_seven(qf) as flag:
            declared_flip_if_false(flag, t)

    # Operations on the truth value are not controlled on it
    assert run_classical(declared) == [(v, not (v < 3 or v == 7), 0) for v in range(8)]

    def undeclared(qf, t):
        with small_or_seven(qf) as flag:
            flip_if_false(flag, t)

    with pytest.raises(ConditionCompilationError, match="not supported"):
        trace(undeclared)


def test_controlled():

    def body(qf, t, u):
        u.flip()
        with control(u):
            with small_or_seven(qf) as flag:
                t.flip()
                flag.flip()
                flag.flip()

    assert run_classical(body) == [(v, v < 3 or v == 7, 1) for v in range(8)]

    # Only the body is controlled, the evaluation of the condition is not
    def main():
        qf = QuantumFloat(3)
        ctrl = QuantumBool()
        with control(ctrl):
            with small_or_seven(qf) as flag:
                z(flag)

    def jit_names(jaspr):
        names = []
        for eqn in jaspr.eqns:
            if eqn.primitive.name == "jit":
                names.append(eqn.params["name"])
                names.extend(jit_names(eqn.params["jaxpr"]))
        return names

    names = jit_names(make_jaspr(main)())
    assert "ccondition_env" in names
    assert "small_or_seven" in names
    assert "small_or_seven_dg" in names
    assert "csmall_or_seven" not in names


def test_inverted():

    def oracle(qf):
        with small_or_seven(qf) as flag:
            p(0.3, flag)

    def inverted(qf):
        with invert():
            oracle(qf)

    def oracle_then_inverse(qf):
        oracle(qf)
        inverted(qf)

    assert phase_distribution(oracle_then_inverse) == {0: 1.0}
    assert phase_distribution(inverted) == phase_distribution(lambda qf: reference_oracle(qf, phase=-0.3))


def test_nested():

    def body(qf, t, u):
        with qf < 6:
            with qf > 2:
                t.flip()

    assert run_classical(body) == [(v, 2 < v < 6, 0) for v in range(8)]


def test_static_and_dynamic_size():

    def main(n):
        qf = QuantumFloat(n)
        qf[:] = 2
        t = QuantumBool()
        with small_or_seven(qf):
            t.flip()
        return measure(t)

    assert jaspify(main)(3)
    assert jaspify(lambda: main(3))()


def trace(body):
    def main():
        qf = QuantumFloat(3)
        t = QuantumBool()
        body(qf, t)
        return measure(qf)

    return make_jaspr(main)()


def test_error_truth_value_used_after():

    def body(qf, t):
        with small_or_seven(qf) as flag:
            t.flip()
        cx(flag, t)

    with pytest.raises(ConditionCompilationError, match="is used after the condition"):
        trace(body)


def test_error_unsupported_operation_on_truth_value():

    def body(qf, t):
        with small_or_seven(qf) as flag:
            h(flag)

    with pytest.raises(ConditionCompilationError, match="not supported"):
        trace(body)


def leaky_condition():
    """Return a condition that leaks its intermediate result ``qf < 3`` through a list."""
    leaked = []

    @quantum_condition
    def small_or_seven_leaky(qf):
        small = qf < 3
        leaked.append(small)
        return small | (qf == 7)

    return small_or_seven_leaky, leaked


def test_intermediate_used_as_control():
    condition, leaked = leaky_condition()

    def body(qf, t, u):
        with condition(qf):
            cx(leaked[-1], t)

    assert run_classical(body) == [(v, v < 3, 0) for v in range(8)]


def test_error_intermediate_modified():
    condition, leaked = leaky_condition()

    def body(qf, t):
        with condition(qf):
            leaked[-1].flip()

    with pytest.raises(ConditionCompilationError, match="intermediate results that is not supported"):
        trace(body)


def test_error_measurement():

    def body(qf, t):
        with small_or_seven(qf):
            measure(t)

    with pytest.raises(ConditionCompilationError, match="performs a measurement"):
        trace(body)


def test_error_temporary_in_classical_control_flow():

    @quantum_condition
    def loop_condition(qf):
        for i in jrange(2):
            tmp = QuantumBool()
            cx(qf[0], tmp[0])
        return qf == 1

    def body(qf, t):
        with loop_condition(qf):
            t.flip()

    with pytest.raises(ConditionCompilationError, match="inside classical control flow"):
        trace(body)
