"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import jax
import jax.numpy as jnp
import pytest

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    h,
    invert,
    mcx,
    measure,
    rx,
    ry,
    x,
)
from qrisp.jasp import backend_sampler, jaspify, q_cond, sample

# ===========================================================================
# Helpers
# ===========================================================================

_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        from qrisp.default_backend import QrispSimulatorBackend

        _backend = QrispSimulatorBackend()
    return _backend


# ===========================================================================
# Basic sampling patterns
# ===========================================================================


def test_single_return_hadamard():
    """QuantumFloat(4) with H on qubit 0 → {0, 1}"""

    def kernel(k):
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(k):
        return sample(kernel, shots=200)(k)

    res = main(1)
    assert res.shape == (200,)
    assert {float(v) for v in res} == {0.0, 1.0}


def test_single_return_uniform_superposition():
    """QuantumFloat(3) with H on all qubits → uniform 0..7"""

    def kernel():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=500)()

    res = main()
    assert res.shape == (500,)
    vals = {int(v) for v in res}
    assert len(vals) >= 6


def test_multi_return_cat_state():
    """GHZ-like: |0000⟩ + |kkkk⟩ → measure two registers"""

    def kernel(k):
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        qbl = QuantumBool()
        h(qbl)
        with control(qbl[0]):
            a[:] = k
        cx(a, b)
        return measure(a), measure(b)

    @backend_sampler(backend=_get_backend())
    def main(k):
        return sample(kernel, shots=300)(k)

    res = main(3)
    assert res.shape == (300, 2)
    pairs = {(float(r[0]), float(r[1])) for r in res}
    assert pairs == {(0.0, 0.0), (3.0, 3.0)}


def test_single_return_boolean():
    """QuantumBool with H → {True, False}"""

    def kernel():
        qbl = QuantumBool()
        h(qbl)
        return measure(qbl)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200,)
    assert {bool(v) for v in res} == {True, False}


def test_triple_return():
    """Three independent QuantumFloats measured"""

    def kernel():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        c = QuantumFloat(2)
        h(a[0])
        x(b[0])
        h(c[0])
        h(c[1])
        return measure(a), measure(b), measure(c)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300, 3)
    assert {float(r[0]) for r in res} == {0.0, 1.0}
    assert {float(r[1]) for r in res} == {1.0}
    assert len({float(r[2]) for r in res}) >= 3


# ===========================================================================
# Classical post-processing
# ===========================================================================


def test_postproc_arithmetic():
    """measure(qf) * 2 + 1 → all odd"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        h(qf[1])
        mes = measure(qf)
        return mes * 2 + 1

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert all(int(v) % 2 == 1 for v in res)
    assert {int(v) for v in res} == {1, 3, 5, 7}


def test_postproc_multi_step():
    """Multiple classical operations on measurement"""

    def kernel():
        qf = QuantumFloat(5)
        h(qf[0])
        h(qf[1])
        h(qf[2])
        a = measure(qf)
        b = a + 3
        c = b * 2
        return c

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert all(int(v) % 2 == 0 for v in res)


def test_postproc_jax_array():
    """JAX array operations on measurement results"""

    def kernel():
        qv = QuantumVariable(4)
        h(qv[0])
        h(qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        arr = jnp.array([m0, m1, m2, m3], dtype=jnp.float64)
        return jnp.sum(arr)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200,)
    vals = {int(v) for v in res}
    assert all(0 <= v <= 2 for v in vals)


def test_postproc_tuple_return():
    """Post-processing on multiple returns"""

    def post_processor(x, y):
        return x + y, x * y

    def kernel():
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        h(a[0])
        x(b[0])
        return measure(a), measure(b)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200, post_processor=post_processor)()

    res = main()
    assert res.shape == (200, 2)


# ===========================================================================
# Entanglement & controlled operations
# ===========================================================================


def test_bell_state():
    """Bell state: |00⟩ + |11⟩ → measure both qubits"""

    def kernel():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv[0]), measure(qv[1])

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300, 2)
    pairs = {(bool(r[0]), bool(r[1])) for r in res}
    assert pairs == {(False, False), (True, True)}


def test_ghz_state():
    """GHZ: |0000⟩ + |1111⟩"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        cx(qf[0], qf[1])
        cx(qf[1], qf[2])
        cx(qf[2], qf[3])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert {int(v) for v in res} == {0, 15}


def test_controlled_operation():
    """Control-X conditioned on QuantumBool"""

    def kernel():
        qbl = QuantumBool()
        qf = QuantumFloat(4)
        h(qbl)
        with control(qbl[0]):
            qf[:] = 5
        return measure(qbl), measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300, 2)
    pairs = {(bool(r[0]), float(r[1])) for r in res}
    assert pairs == {(False, 0.0), (True, 5.0)}


def test_multi_controlled_x():
    """MCX gate with 3 controls"""

    def kernel():
        qf = QuantumFloat(5)
        target = QuantumBool()
        x(qf[0])
        x(qf[1])
        x(qf[2])
        mcx(qf[:3], target[0])
        return measure(target)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200,)
    assert all(bool(v) for v in res)


def test_inversion_environment():
    """Invert a sequence of gates"""

    def kernel():
        qf = QuantumFloat(3)
        with invert():
            x(qf[0])
            x(qf[1])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200,)


# ===========================================================================
# Edge cases & robustness
# ===========================================================================


def test_dynamic_kernel_arg():
    """Kernel arg is dynamic (JAX tracer)"""

    def kernel(size):
        qf = QuantumFloat(size)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(s):
        return sample(kernel, shots=150)(s)

    res = main(4)
    assert res.shape == (150,)
    assert {float(v) for v in res} == {0.0, 1.0}


def test_large_qubit_count():
    """Kernel with 10-qubit QuantumFloat"""

    def kernel():
        qf = QuantumFloat(10)
        h(qf[0])
        h(qf[9])
        return measure(qf[0]), measure(qf[9])

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=100)()

    res = main()
    assert res.shape == (100, 2)


def test_multiple_sample_calls():
    """Two separate sample() calls in one function"""

    def kernel_a():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    def kernel_b():
        qf = QuantumFloat(3)
        x(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        res_a = sample(kernel_a, shots=100)()
        res_b = sample(kernel_b, shots=100)()
        return res_a, res_b

    res_a, res_b = main()
    assert res_a.shape == (100,) and res_b.shape == (100,)
    assert all(int(v) % 2 == 1 for v in res_b[:50])


def test_zero_shots():
    """shots=0 is rejected while tracing.

    sample() requires a static shot count, so the check in sample() covers
    every case and the error surfaces before any circuit is built.
    """

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=0)()

    with pytest.raises(ValueError, match="at least one shot is required"):
        main()


# ===========================================================================
# Backend integration
# ===========================================================================


def test_custom_backend():
    """Backend with custom options"""
    from qrisp.default_backend import QrispSimulatorBackend

    custom = QrispSimulatorBackend()
    custom.update_options(shots=500)

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=custom)
    def main():
        return sample(kernel, shots=50)()

    res = main()
    assert res.shape == (50,)
    vals = {float(v) for v in res}
    assert 0.0 in vals and 1.0 in vals


def test_default_backend():
    """Explicit default backend."""
    from qrisp.default_backend import QrispSimulatorBackend

    def kernel():
        qf = QuantumFloat(2)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=QrispSimulatorBackend())
    def main():
        return sample(kernel, shots=30)()

    res = main()
    assert res.shape == (30,)


# ===========================================================================
# Statistical distribution checks
# ===========================================================================


def test_statistical_uniformity():
    """Chi-squared-like: uniform superposition of 8 values, 2000 shots"""

    def kernel():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=2000)()

    res = main()
    assert res.shape == (2000,)

    counts = {}
    for v in res:
        iv = int(v)
        counts[iv] = counts.get(iv, 0) + 1

    expected = 2000 / 8
    # 5σ threshold: σ = sqrt(2000 * 1/8 * 7/8) ≈ 14.8, so 5σ ≈ 74.
    # Using 5 * sqrt(expected) ≈ 5 * 15 = 75 gives comfortable margin.
    assert all(abs(c - expected) < 5 * 15 for c in counts.values())
    assert len(counts) == 8


def test_compare_with_jaspify():
    """Backend sampler matches jaspify (terminal_sampling) distribution"""

    def kernel_qv():
        qf = QuantumFloat(4)
        h(qf[0])
        h(qf[1])
        return qf

    @backend_sampler(backend=_get_backend())
    def backend_main():
        def measured_kernel():
            qf = kernel_qv()
            return measure(qf)

        return sample(measured_kernel, shots=300)()

    @jaspify(terminal_sampling=True)
    def jaspify_main():
        return sample(kernel_qv, shots=300)()

    res_backend = backend_main()
    res_jaspify = jaspify_main()

    bc = {}
    for v in res_backend:
        bc[int(v)] = bc.get(int(v), 0) + 1
    jc = {}
    for v in res_jaspify:
        jc[int(v)] = jc.get(int(v), 0) + 1

    assert set(bc.keys()) == set(jc.keys())
    for k in bc:
        if k in jc:
            ratio = bc[k] / max(1, jc[k])
            assert 0.4 < ratio < 1.6


# ===========================================================================
# Corner cases
# ===========================================================================


def test_measure_all_qubits():
    """Kernel measures every qubit individually"""

    def kernel():
        qv = QuantumVariable(4)
        h(qv[0])
        h(qv[2])
        return measure(qv[0]), measure(qv[1]), measure(qv[2]), measure(qv[3])

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200, 4)
    assert {bool(r[1]) for r in res} == {False}
    assert {bool(r[3]) for r in res} == {False}


def test_parameterized_gates():
    """Rx/Ry parameterized gates in kernel"""

    def kernel(angle):
        qf = QuantumFloat(3)
        rx(angle, qf[0])
        ry(angle, qf[1])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(a):
        return sample(kernel, shots=200)(a)

    res = main(0.5)
    assert res.shape == (200,)


# ===========================================================================
# Error handling
# ===========================================================================


def test_raises_without_sample():
    """@backend_sampler raises if the function doesn't use sample()"""

    @backend_sampler(backend=_get_backend())
    def no_sample():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    with pytest.raises(RuntimeError):
        no_sample()


def test_raises_on_realtime_feedback():
    """@backend_sampler raises RuntimeError for kernels with real-time
    feedback (mid-circuit measurement whose classical post-processing
    controls subsequent quantum gates)."""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        m = measure(qf[0])
        # Arithmetic on the measurement result — this becomes a
        # ProcessedMeasurement during to_qc, which cannot be used
        # to decide over further circuit construction.
        processed = m + 1

        def true_fun(qf):
            x(qf[1])
            return qf

        def false_fun(qf):
            return qf

        qf = q_cond(processed > 0, true_fun, false_fun, qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=10)()

    with pytest.raises(RuntimeError, match="real-time feedback"):
        main()


# ===========================================================================
# Control flow propagation tests
# ===========================================================================


def test_fori_loop_around_sample():
    """fori_loop orchestrating multiple sample() calls — the outer
    evaluator must propagate through the while loop created by fori_loop
    and intercept each sample() inside."""

    def kernel():
        qf = QuantumFloat(3)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        results = jnp.zeros((3, 50), dtype=jnp.float64)

        def body(i, acc):
            samples = sample(kernel, shots=50)()
            acc = acc.at[i].set(samples)
            return acc

        return jax.lax.fori_loop(0, 3, body, results)

    res = main()
    assert res.shape == (3, 50)


def test_cond_around_sample():
    """jax.lax.cond choosing between two sampling kernels — the outer
    evaluator must propagate through the cond and intercept both
    branches."""

    def kernel_a():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    def kernel_b():
        qf = QuantumFloat(4)
        h(qf[1])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(use_a):
        def true_branch(_):
            return sample(kernel_a, shots=30)()

        def false_branch(_):
            return sample(kernel_b, shots=30)()

        return jax.lax.cond(use_a, true_branch, false_branch, None)

    res_a = main(True)
    res_b = main(False)
    assert res_a.shape == (30,)
    assert res_b.shape == (30,)


def test_while_loop_around_sample():
    """jax.lax.while_loop containing sample() calls — the outer evaluator
    must propagate through the while loop body."""

    def kernel():
        qf = QuantumFloat(4)
        qf[:] = 3
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(max_iter):
        def cond_fun(state):
            i, _ = state
            return i < max_iter

        def body_fun(state):
            i, prev = state
            samples = sample(kernel, shots=10)()
            return i + 1, samples

        _, final = jax.lax.while_loop(cond_fun, body_fun, (0, jnp.zeros(10)))
        return final

    res = main(4)
    assert res.shape == (10,)


def test_scan_around_sample():
    """jax.lax.scan over a range, calling sample() each step — the outer
    evaluator must propagate through scan."""

    def kernel(i):
        qf = QuantumFloat(3)
        qf[:] = i
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        def body(carry, i):
            samples = sample(kernel, shots=20)(i)
            return carry, samples

        _, results = jax.lax.scan(body, None, jnp.arange(4))
        return results

    res = main()
    assert res.shape == (4, 20)


def test_nested_jit_orchestrating_sample():
    """A @jax.jit helper orchestrating a sample() call — the outer
    evaluator must propagate through the nested jit to intercept
    sample() inside."""

    def kernel():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    @jax.jit
    def helper():
        return sample(kernel, shots=40)()

    @backend_sampler(backend=_get_backend())
    def main():
        return helper()

    res = main()
    assert res.shape == (40,)


def test_switch_around_sample():
    """jax.lax.switch choosing among multiple kernels — the outer
    evaluator must propagate through all branches."""

    def kernel_0():
        qf = QuantumFloat(4)
        qf[:] = 0
        return measure(qf)

    def kernel_1():
        qf = QuantumFloat(4)
        qf[:] = 1
        return measure(qf)

    def kernel_2():
        qf = QuantumFloat(4)
        qf[:] = 2
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(branch_idx):
        branches = [
            lambda _: sample(kernel_0, shots=20)(),
            lambda _: sample(kernel_1, shots=20)(),
            lambda _: sample(kernel_2, shots=20)(),
        ]
        return jax.lax.switch(branch_idx, branches, None)

    for i in range(3):
        res = main(i)
        assert res.shape == (20,)


# ===========================================================================
# Captured-value tests
#
# The loop counter is located structurally, from the sampling loop's own
# condition. An earlier implementation inferred it from avals, which broke
# once JAX prepended a captured closure value whose aval collided with the
# accumulator's or with the counter's.
# ===========================================================================


def test_captured_array_matching_accumulator_shape():
    """A captured array shaped like the accumulator must not be mistaken for it."""

    @backend_sampler(backend=_get_backend())
    def main(k):
        captured = jnp.zeros((4,), dtype=jnp.float64) + k

        def kernel():
            qf = QuantumFloat(2)
            h(qf[0])
            return measure(qf) + captured[0]

        return sample(kernel, shots=4)()

    result = main(2.0)

    assert result.shape == (4,)
    assert {float(value) for value in result} <= {2.0, 3.0}


def test_captured_int_scalar_does_not_shadow_loop_index():
    """A captured int scalar must not be mistaken for the loop counter.

    Selecting it would silently make every iteration read the same shot,
    so this asserts the shots actually vary rather than just that the call
    succeeds.
    """

    def kernel_factory(extra):
        def kernel():
            qf = QuantumFloat(3)
            h(qf[0])
            h(qf[1])
            h(qf[2])
            return measure(qf) + extra

        return kernel

    @backend_sampler(backend=_get_backend())
    def with_capture(k):
        captured_idx = jnp.int64(0) + k
        return sample(kernel_factory(captured_idx * 0), shots=50)()

    result = with_capture(0)

    assert result.shape == (50,)
    # A uniform superposition over 3 qubits: collapsing to a single repeated
    # shot is the failure mode this guards against.
    assert len(set(float(value) for value in result)) > 1
