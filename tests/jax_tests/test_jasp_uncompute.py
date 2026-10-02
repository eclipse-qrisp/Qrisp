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

"""Tests for the jasp_uncompute decorator: correctness against manual conjugation, control, inversion and errors."""

import jax
import pytest

from qrisp import *
from qrisp.circuit import QuantumCircuit
from qrisp.environments.jasp_uncomputation_environment import JaspUncomputationError
from qrisp.jasp import *


def oracle_body(qf):
    flag1 = qf < 3
    flag2 = qf == 7
    flag_any = flag1 | flag2
    z(flag_any)


oracle = jasp_uncompute(oracle_body)


def manual_oracle(qf, phase=None):
    # Reference: the same oracle uncomputed by hand via injection and conjugation
    flag1 = QuantumBool()
    flag2 = QuantumBool()
    flag_any = QuantumBool()
    with conjugate(flag1 << (lambda qf: qf < 3))(qf):
        with conjugate(flag2 << (lambda qf: qf == 7))(qf):
            with conjugate(flag_any << (lambda a, b: a | b))(flag1, flag2):
                if phase is None:
                    z(flag_any)
                else:
                    p(phase, flag_any)
    flag1.delete()
    flag2.delete()
    flag_any.delete()


def phase_distribution(function, size=4):
    """Distribution of H function H applied to |0>, which reveals the phases applied by function."""

    @terminal_sampling
    def main(n):
        qf = QuantumFloat(n)
        h(qf)
        function(qf)
        h(qf)
        return qf

    return {k: round(v, 6) for k, v in main(size).items() if v > 1e-9}


def test_oracle_matches_manual_uncomputation():
    expected = phase_distribution(manual_oracle)
    assert phase_distribution(oracle) == expected
    # Without uncomputation, the dirty temporaries change the result
    assert phase_distribution(oracle_body) != expected


def test_static_and_dynamic_size():

    def main(n):
        qf = QuantumFloat(n)
        h(qf)
        oracle(qf)
        h(qf)
        return qf

    expected = phase_distribution(manual_oracle)
    dynamic = terminal_sampling(main)(4)
    static = terminal_sampling(lambda: main(4))()
    for res in (dynamic, static):
        assert {k: round(v, 6) for k, v in res.items() if v > 1e-9} == expected


def test_temporaries_are_clean(capsys):

    @jasp_uncompute
    def mark(qf, target):
        flag1 = qf < 3
        flag2 = qf == 7
        flag_any = flag1 | flag2
        cx(flag_any, target)

    @boolean_simulation
    def main(value):
        qf = QuantumFloat(4)
        qf[:] = value
        target = QuantumBool()
        mark(qf, target)
        return measure(qf), measure(target)

    for value in range(16):
        qf_value, target_value = main(value)
        assert qf_value == value
        assert bool(target_value) == (value < 3 or value == 7)

    jax.effects_barrier()
    assert "Faulty uncomputation" not in capsys.readouterr().out


def test_controlled():

    def run(function):
        @terminal_sampling
        def main():
            qf = QuantumFloat(4)
            h(qf)
            ctrl = QuantumBool()
            h(ctrl)
            with control(ctrl):
                function(qf)
            h(ctrl)
            return qf, ctrl

        return {k: round(v, 6) for k, v in main().items() if v > 1e-9}

    assert run(oracle) == run(manual_oracle)

    # Only the use is controlled, the computation of the temporaries is not
    def main():
        qf = QuantumFloat(4)
        ctrl = QuantumBool()
        with control(ctrl):
            oracle(qf)

    def jit_names(jaspr):
        names = []
        for eqn in jaspr.eqns:
            if eqn.primitive.name == "jit":
                names.append(eqn.params["name"])
                names.extend(jit_names(eqn.params["jaxpr"]))
        return names

    names = jit_names(make_jaspr(main)())
    assert "coracle_body" in names
    assert "oracle_body_compute" in names
    assert "oracle_body_uncompute" in names
    assert "coracle_body_compute" not in names


def test_inversion():

    @jasp_uncompute
    def phase_oracle(qf):
        flag1 = qf < 3
        flag2 = qf == 7
        flag_any = flag1 | flag2
        p(0.3, flag_any)

    def inverted(qf):
        with invert():
            phase_oracle(qf)

    def forward_then_inverse(qf):
        phase_oracle(qf)
        inverted(qf)

    assert phase_distribution(forward_then_inverse) == {0: 1.0}
    assert phase_distribution(inverted) == phase_distribution(lambda qf: manual_oracle(qf, phase=-0.3))


def test_multi_controlled_use():
    # The MCX uses the temporaries as controls. It belongs to the use and must
    # not be cancelled by the uncomputation.

    def run(function):
        @terminal_sampling
        def main():
            qf = QuantumFloat(4)
            h(qf)
            target = QuantumBool()
            function(qf, target)
            return qf, target

        return main()

    @jasp_uncompute
    def mark(qf, target):
        flag1 = qf < 3
        flag2 = ~(qf == 1)
        mcx(flag1.reg + flag2.reg, target[0])

    res = run(mark)
    assert all(bool(target) == (value in (0, 2)) for value, target in res)


def test_returned_result():

    @jasp_uncompute
    def less_than_three(qf):
        flag = qf < 3
        res = QuantumBool()
        cx(flag, res)
        return res

    @terminal_sampling
    def main():
        qf = QuantumFloat(4)
        h(qf)
        res = less_than_three(qf)
        return qf, res

    res = main()
    assert len(res) == 16
    assert all(bool(r) == (value < 3) for value, r in res)


def test_nested():

    @jasp_uncompute
    def inner(qf):
        flag = qf == 7
        z(flag)

    @jasp_uncompute
    def outer(qf):
        flag = qf < 8
        with control(flag):
            inner(qf)

    def manual(qf):
        outer_flag = QuantumBool()
        inner_flag = QuantumBool()
        with conjugate(outer_flag << (lambda qf: qf < 8))(qf):
            with control(outer_flag):
                with conjugate(inner_flag << (lambda qf: qf == 7))(qf):
                    z(inner_flag)
        outer_flag.delete()
        inner_flag.delete()

    assert phase_distribution(outer) == phase_distribution(manual)


def test_without_temporaries():

    @jasp_uncompute
    def no_temporaries(qf):
        x(qf[0])

    assert phase_distribution(no_temporaries) == phase_distribution(lambda qf: x(qf[0]))


def test_outside_of_jasp():
    qf = QuantumFloat(4)
    h(qf)
    oracle(qf)
    assert [qv.name for qv in qf.qs.qv_list] == [qf.name]


def trace(function):
    def main():
        qf = QuantumFloat(4)
        function(qf)
        return measure(qf)

    return make_jaspr(main)()


def test_error_interleaved_use():

    @jasp_uncompute
    def interleaved(qf):
        flag1 = qf < 3
        z(flag1)
        flag2 = qf == 7
        z(flag2)

    with pytest.raises(JaspUncomputationError, match="before all temporaries have been computed"):
        trace(interleaved)


def test_error_use_modifies_input():

    @jasp_uncompute
    def modifies_input(qf):
        flag = qf < 3
        z(flag)
        x(qf[0])

    with pytest.raises(JaspUncomputationError, match="modifies qubits that the temporaries were computed from"):
        trace(modifies_input)


def test_error_measurement_in_computation():

    @jasp_uncompute
    def measures(qf):
        flag1 = qf < 3
        measure(qf[0])
        flag2 = qf == 7
        z(flag1 | flag2)

    with pytest.raises(JaspUncomputationError, match="performs a measurement while computing"):
        trace(measures)


def test_error_measured_temporary():

    @jasp_uncompute
    def measures_temporary(qf):
        flag = qf < 3
        measure(flag)

    with pytest.raises(JaspUncomputationError, match="measures a temporary"):
        trace(measures_temporary)


def test_error_temporary_in_classical_control_flow():

    @jasp_uncompute
    def loop(qf):
        for i in jrange(2):
            flag = QuantumBool()
            cx(qf[0], flag[0])

    with pytest.raises(JaspUncomputationError, match="inside classical control flow"):
        trace(loop)


def test_error_unclassifiable_operation():
    qc = QuantumCircuit(1)
    qc.z(0)
    # A gate without permeability information
    unknown_gate = qc.to_gate("unknown_z")

    @jasp_uncompute
    def unknown_use(qf):
        flag = qf < 3
        flag.qs.append(unknown_gate, [flag[0]])

    with pytest.raises(JaspUncomputationError, match="can not determine whether it computes or uses"):
        trace(unknown_use)
