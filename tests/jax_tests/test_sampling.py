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


from qrisp import QuantumBool, QuantumFloat, conjugate, control, cx, h, measure, t, x
from qrisp.jasp import expectation_value, jaspify, jrange, q_while_loop, sample


def double(*args):
    if len(args) == 1:
        return 2 * args[0]
    return tuple([2 * x for x in args])


def test_sampling():

    def inner_f(i):
        qf = QuantumFloat(4)

        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])

        return qf

    @jaspify
    def main():
        res = sample(inner_f, 500)(2)
        return res

    assert set(int(i) for i in main()) == {0, 1}

    @jaspify(terminal_sampling=True)
    def main():
        res = sample(inner_f, 500)(2)
        return res

    assert set(int(i) for i in main()) == {0, 1}

    @jaspify
    def main():
        res = sample(inner_f, 500, post_processor=double)(2)
        return res

    assert set(int(i) for i in main()) == {0, 2}

    @jaspify(terminal_sampling=True)
    def main():
        res = sample(inner_f, 500, post_processor=double)(2)
        return res

    assert set(int(i) for i in main()) == {0, 2}

    def inner_f(i):
        qf = QuantumFloat(4)
        qf_2 = QuantumFloat(4)
        qf_3 = QuantumFloat(4)
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])

        return qf, qf_2, qf_3

    @jaspify
    def main():

        res = sample(inner_f, 10)(2)

        return res

    assert main().shape == (10, 3)

    @jaspify
    def main():

        res = sample(inner_f, 10, post_processor=double)(2)

        return res

    assert main().shape == (10, 3)

    @jaspify(terminal_sampling=True)
    def main():

        res = sample(inner_f, 10)(2)

        return res

    assert main().shape == (10, 3)

    @jaspify(terminal_sampling=True)
    def main():

        res = sample(inner_f, 10, post_processor=double)(2)

        return res

    assert main().shape == (10, 3)

    @sample
    def main():

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        # Bring qbl into superposition
        h(qbl)

        # Perform a measure
        cl_bl = measure(qbl)

        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])

        return qf, qbl

    assert main() in [{(1.0, True): 0.5, (5.0, True): 0.5}, {(0.0, False): 1.0}]

    @sample
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return qf, a, qbl

    for i in range(3):
        for j in range(3):
            assert main(i, j) == {(0.0, 0.0, False): 0.5, (2**i, 2**j, True): 0.5}

    @sample(500)
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return qf, a, qbl

    assert sum(main(2, 2).values()) == 500


def test_expectation_value():

    def inner_f(i):
        qf = QuantumFloat(4)

        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])

        return qf

    @jaspify(terminal_sampling=True)
    def main():
        res = expectation_value(inner_f, 10000)(2)
        return res

    assert abs(main() - 0.5) < 0.05

    @jaspify
    def main():
        res = expectation_value(inner_f, 500)(2)
        return res

    assert abs(main() - 0.5) < 0.2

    @jaspify(terminal_sampling=True)
    def main():
        res = expectation_value(inner_f, 10000, post_processor=double)(2)
        return res

    assert abs(main() - 1) < 0.05

    @jaspify
    def main():
        res = expectation_value(inner_f, 500, post_processor=double)(2)
        return res

    assert abs(main() - 1) < 0.2

    def inner_f(i):
        a = QuantumFloat(4)
        b = QuantumFloat(4)

        with conjugate(h)(a):
            for k in jrange(i):
                t(a[0])
                x(b[0])

        return a, b

    @jaspify(terminal_sampling=True)
    def main():
        res = expectation_value(inner_f, 10000)(2)
        return res

    ev_res = main()
    assert abs(ev_res[0] - 0.5) < 0.05 and ev_res[1] == 0

    @jaspify
    def main():
        res = expectation_value(inner_f, 500)(2)
        return res

    ev_res = main()
    assert abs(ev_res[0] - 0.5) < 0.2 and ev_res[1] == 0

    @jaspify(terminal_sampling=True)
    def main():
        res = expectation_value(inner_f, 10000, post_processor=double)(2)
        return res

    ev_res = main()
    assert abs(ev_res[0] - 1) < 0.05 and ev_res[1] == 0

    @jaspify
    def main():
        res = expectation_value(inner_f, 500, post_processor=double)(2)
        return res

    ev_res = main()
    assert abs(ev_res[0] - 1) < 0.2 and ev_res[1] == 0

    def prep(k):
        qf = QuantumFloat(k)
        h(qf)
        return qf

    # Code example from https://github.com/eclipse-qrisp/Qrisp/issues/140
    @jaspify
    def test():

        def cond_fun(state):
            index, sum = state
            return index < 5

        def body_fun(state):
            index, sum = state
            a = expectation_value(prep, shots=10)(index)
            index += 1
            sum += a
            return index, sum

        index, sum = q_while_loop(cond_fun, body_fun, (1, 0))
        return sum


# ------------------------------------------------------------------
# Shared state-preparation helpers for TestClassicalAndMixedReturns
# ------------------------------------------------------------------


def _sp_classical_scalar_sample():
    qf = QuantumFloat(4)
    h(qf[0])
    h(qf[1])
    return measure(qf)


def _sp_classical_scalar_ev():
    qf = QuantumFloat(3)
    h(qf[0])
    h(qf[1])
    return measure(qf)


def _sp_classical_tuple():
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    h(a[0])
    cx(a[0], b[0])
    return measure(a), measure(b)


def _sp_pp():
    qf = QuantumFloat(3)
    h(qf[0])
    return measure(qf)


def _sp_mixed():
    qf = QuantumFloat(3)
    h(qf[0])
    mes = measure(qf[1])  # classical (always 0, no superposition on bit 1)
    return qf, mes


def _pp_sum(x, y):
    return x + y


class TestClassicalAndMixedReturns:
    """Tests for sample() and expectation_value() with classical/mixed returns and terminal-sampling rejection."""

    import jax.numpy as jnp
    import pytest

    # ==================================================================
    # sample() — classical scalar return
    # ==================================================================

    def test_sample_classical_scalar(self):
        @jaspify(terminal_sampling=False)
        def main():
            return sample(_sp_classical_scalar_sample, shots=30)()

        res = main()
        assert res.shape == (30,)
        assert len(self.jnp.unique(res)) >= 2

    def test_sample_classical_scalar_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return sample(_sp_classical_scalar_sample, shots=30)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # sample() — classical tuple return
    # ==================================================================

    def test_sample_classical_tuple(self):
        @jaspify(terminal_sampling=False)
        def main():
            return sample(_sp_classical_tuple, shots=20)()

        res = main()
        assert res.shape == (20, 2)

    def test_sample_classical_tuple_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return sample(_sp_classical_tuple, shots=20)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # sample() — classical return with post_processor
    # ==================================================================

    def test_sample_classical_with_pp(self):
        @jaspify(terminal_sampling=False)
        def main():
            return sample(_sp_pp, shots=20, post_processor=double)()

        res = main()
        assert res.shape == (20,)
        assert self.jnp.all(res % 2 == 0)

    def test_sample_classical_with_pp_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return sample(_sp_pp, shots=20, post_processor=double)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # sample() — mixed return (quantum + classical)
    # ==================================================================

    def test_sample_mixed(self):
        @jaspify(terminal_sampling=False)
        def main():
            return sample(_sp_mixed, shots=20)()

        res = main()
        assert res.shape == (20, 2)

    def test_sample_mixed_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return sample(_sp_mixed, shots=20)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # sample() — mixed return with post_processor
    # ==================================================================

    def test_sample_mixed_with_pp(self):
        @jaspify(terminal_sampling=False)
        def main():
            return sample(_sp_mixed, shots=15, post_processor=_pp_sum)()

        res = main()
        assert res.shape == (15,)

    def test_sample_mixed_with_pp_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return sample(_sp_mixed, shots=15, post_processor=_pp_sum)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # expectation_value() — classical scalar return
    # ==================================================================

    def test_ev_classical_scalar(self):
        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(_sp_classical_scalar_ev, shots=500)()

        res = main()
        assert abs(res - 1.5) < 0.3

    def test_ev_classical_scalar_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(_sp_classical_scalar_ev, shots=500)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # expectation_value() — classical tuple return
    # ==================================================================

    def test_ev_classical_tuple(self):
        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(_sp_classical_tuple, shots=500)()

        res = main()
        assert len(res) == 2
        assert abs(res[0] - 0.5) < 0.3
        assert abs(res[1] - 0.5) < 0.3

    def test_ev_classical_tuple_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(_sp_classical_tuple, shots=500)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # expectation_value() — classical return with post_processor
    # ==================================================================

    def test_ev_classical_with_pp(self):
        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(_sp_pp, shots=500, post_processor=double)()

        res = main()
        assert abs(res - 1.0) < 0.3

    def test_ev_classical_with_pp_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(_sp_pp, shots=500, post_processor=double)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # expectation_value() — mixed return (quantum + classical)
    # ==================================================================

    def test_ev_mixed(self):
        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(_sp_mixed, shots=500)()

        res = main()
        assert len(res) == 2
        assert abs(res[0] - 0.5) < 0.3
        assert res[1] == 0.0

    def test_ev_mixed_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(_sp_mixed, shots=500)()

        with self.pytest.raises(ValueError):
            main()

    # ==================================================================
    # expectation_value() — mixed return with post_processor
    # ==================================================================

    def test_ev_mixed_with_pp(self):
        @jaspify(terminal_sampling=False)
        def main():
            return expectation_value(_sp_mixed, shots=500, post_processor=_pp_sum)()

        res = main()
        assert abs(res - 0.5) < 0.3

    def test_ev_mixed_with_pp_rejected_by_ts(self):
        @jaspify(terminal_sampling=True)
        def main():
            return expectation_value(_sp_mixed, shots=500, post_processor=_pp_sum)()

        with self.pytest.raises(ValueError):
            main()
