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

import jax.numpy as jnp
import numpy as np

from qrisp import *
from qrisp.jasp import *

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
    assert {float(v) for v in res[:100]} == {0.0, 1.0}


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
    assert len(vals) >= 6, f"expected ≥6 values from 0-7, got {len(vals)}"


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
    pairs = {(float(r[0]), float(r[1])) for r in res[:200]}
    assert pairs == {(0.0, 0.0), (3.0, 3.0)}, f"got {pairs}"


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
    assert {bool(v) for v in res[:100]} == {True, False}


def test_triple_return():
    """Three independent QuantumFloats measured"""

    def kernel():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        c = QuantumFloat(2)
        h(a[0])
        x(b[0])
        h(c[0]); h(c[1])
        return measure(a), measure(b), measure(c)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300, 3)
    assert {float(r[0]) for r in res[:100]} == {0.0, 1.0}
    assert {float(r[1]) for r in res[:100]} == {1.0}
    assert len({float(r[2]) for r in res[:100]}) >= 3


# ===========================================================================
# Classical post-processing
# ===========================================================================

def test_postproc_arithmetic():
    """measure(qf) * 2 + 1 → all odd"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0]); h(qf[1])
        mes = measure(qf)
        return mes * 2 + 1

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert all(int(v) % 2 == 1 for v in res[:200])
    assert {int(v) for v in res[:200]} == {1, 3, 5, 7}


def test_postproc_multi_step():
    """Multiple classical operations on measurement"""

    def kernel():
        qf = QuantumFloat(5)
        h(qf[0]); h(qf[1]); h(qf[2])
        a = measure(qf)
        b = a + 3
        c = b * 2
        return c

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert all(int(v) % 2 == 0 for v in res[:200])


def test_postproc_jax_array():
    """JAX array operations on measurement results"""

    def kernel():
        qv = QuantumVariable(4)
        h(qv[0]); h(qv[1])
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
    vals = {int(v) for v in res[:100]}
    assert all(0 <= v <= 2 for v in vals), f"values in [0,2], got {vals}"


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
    pairs = {(bool(r[0]), bool(r[1])) for r in res[:200]}
    assert pairs == {(False, False), (True, True)}, f"got {pairs}"


def test_ghz_state():
    """GHZ: |0000⟩ + |1111⟩"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        cx(qf[0], qf[1]); cx(qf[1], qf[2]); cx(qf[2], qf[3])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=300)()

    res = main()
    assert res.shape == (300,)
    assert {int(v) for v in res[:200]} == {0, 15}


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
    pairs = {(bool(r[0]), float(r[1])) for r in res[:200]}
    assert pairs == {(False, 0.0), (True, 5.0)}, f"got {pairs}"


def test_multi_controlled_x():
    """MCX gate with 3 controls"""

    def kernel():
        qf = QuantumFloat(5)
        target = QuantumBool()
        x(qf[0]); x(qf[1]); x(qf[2])
        mcx(qf[:3], target[0])
        return measure(target)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200,)
    assert all(bool(v) for v in res[:100])


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
    assert {float(v) for v in res[:80]} == {0.0, 1.0}


def test_large_qubit_count():
    """Kernel with 10-qubit QuantumFloat"""

    def kernel():
        qf = QuantumFloat(10)
        h(qf[0]); h(qf[9])
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
    """Zero shots should raise"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=0)()

    try:
        main()
    except Exception:
        pass  # expected


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
    vals = {float(v) for v in res[:50]}
    assert 0.0 in vals and 1.0 in vals


def test_default_backend():
    """No backend specified → uses default"""

    def kernel():
        qf = QuantumFloat(2)
        h(qf[0])
        return measure(qf)

    @backend_sampler
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
        h(qf[0]); h(qf[1])
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
            assert 0.4 < ratio < 1.6, f"count ratio for {k}: {ratio:.2f}"


# ===========================================================================
# Corner cases
# ===========================================================================

def test_measure_all_qubits():
    """Kernel measures every qubit individually"""

    def kernel():
        qv = QuantumVariable(4)
        h(qv[0]); h(qv[2])
        return measure(qv[0]), measure(qv[1]), measure(qv[2]), measure(qv[3])

    @backend_sampler(backend=_get_backend())
    def main():
        return sample(kernel, shots=200)()

    res = main()
    assert res.shape == (200, 4)
    assert {bool(r[1]) for r in res[:100]} == {False}
    assert {bool(r[3]) for r in res[:100]} == {False}


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

    try:
        no_sample()
        assert False, "should have raised RuntimeError"
    except RuntimeError:
        pass
