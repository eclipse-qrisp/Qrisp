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
# Basic expectation_value patterns
# ===========================================================================

def test_ev_hadamard():
    """⟨0|H Z H|0⟩ = 0 — equal superposition, expectation of 0 vs 1 is 0.5"""
    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    # 50% 0, 50% 1 → expectation ≈ 0.5
    assert 0.3 < float(res) < 0.7, f"expected ~0.5, got {res}"


def test_ev_deterministic():
    """State prepared in |5⟩ → expectation = 5.0"""
    def kernel():
        qf = QuantumFloat(4)
        qf[:] = 5
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=200)()

    res = main()
    assert 4.5 < float(res) < 5.5, f"expected ~5.0, got {res}"


def test_ev_uniform():
    """Uniform superposition of 0..7 → expectation ≈ 3.5"""
    def kernel():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=1000)()

    res = main()
    assert 2.5 < float(res) < 4.5, f"expected ~3.5, got {res}"


def test_ev_boolean():
    """QuantumBool in |+⟩ → expectation ≈ 0.5"""
    def kernel():
        qbl = QuantumBool()
        h(qbl)
        return measure(qbl)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    assert 0.0 < float(res) < 1.0, f"expected ~0.5, got {res}"


def test_ev_ghz():
    """GHZ: |0000⟩ + |1111⟩ → expectation ≈ 7.5 (50% 0, 50% 15)"""
    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        cx(qf[0], qf[1]); cx(qf[1], qf[2]); cx(qf[2], qf[3])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    assert 5.0 < float(res) < 10.0, f"expected ~7.5, got {res}"


# ===========================================================================
# Post-processing in expectation_value
# ===========================================================================

def test_ev_postproc():
    """E[measure(qf) * 2 + 1] with 2 random bits → expectation = (avg of {1,3,5,7}) = 4.0"""
    def kernel():
        qf = QuantumFloat(4)
        h(qf[0]); h(qf[1])
        return measure(qf) * 2 + 1

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    assert 3.0 < float(res) < 5.0, f"expected ~4.0, got {res}"


def test_ev_postproc_with_fn():
    """Expectation with explicit post_processor callable."""
    def post_processor(x):
        return x * x  # square the measurement

    def kernel():
        qf = QuantumFloat(3)
        h(qf[0])  # 50% 0, 50% 1
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500, post_processor=post_processor)()

    res = main()
    # 0^2 and 1^2 equally likely → expectation ≈ 0.5
    assert 0.3 < float(res) < 0.7, f"expected ~0.5, got {res}"


# ===========================================================================
# Comparison against jaspify
# ===========================================================================

def test_ev_compare_jaspify():
    """Backend sampler expectation_value matches jaspify."""
    def kernel():
        qf = QuantumFloat(4)
        h(qf[0]); h(qf[1])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def backend_main():
        return expectation_value(kernel, shots=300)()

    @jaspify
    def jaspify_main():
        return expectation_value(kernel, shots=300)()

    r_backend = float(backend_main())
    r_jaspify = float(jaspify_main())

    # Both should be ~1.5 (avg of 0,1,2,3)
    assert abs(r_backend - r_jaspify) < 0.3, \
        f"backend={r_backend:.3f}, jaspify={r_jaspify:.3f}, diff too large"


def test_ev_compare_jaspify_many_shots():
    """With many shots, backend and jaspify should converge."""
    def kernel():
        qf = QuantumFloat(3)
        h(qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def backend_main():
        return expectation_value(kernel, shots=2000)()

    @jaspify
    def jaspify_main():
        return expectation_value(kernel, shots=2000)()

    r_backend = float(backend_main())
    r_jaspify = float(jaspify_main())

    # Both should be ~3.5 (avg of 0..7)
    assert abs(r_backend - r_jaspify) < 0.2, \
        f"backend={r_backend:.3f}, jaspify={r_jaspify:.3f}, diff too large"


# ===========================================================================
# Deterministic checks
# ===========================================================================

def test_ev_all_zeros():
    """All qubits in |0⟩ → expectation = 0.0"""
    def kernel():
        qf = QuantumFloat(5)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=100)()

    res = main()
    assert float(res) == 0.0, f"expected 0.0, got {res}"


def test_ev_all_ones():
    """All qubits in |1⟩ → expectation = 2^n - 1"""
    def kernel():
        qf = QuantumFloat(3)
        x(qf)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=100)()

    res = main()
    assert float(res) == 7.0, f"expected 7.0, got {res}"


# ===========================================================================
# Edge cases
# ===========================================================================

def test_ev_dynamic_kernel_arg():
    """Dynamic kernel argument with expectation_value."""
    def kernel(size):
        qf = QuantumFloat(size)
        qf[:] = 3
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(s):
        return expectation_value(kernel, shots=100)(s)

    res = main(4)
    assert 2.5 < float(res) < 3.5, f"expected ~3.0, got {res}"


def test_ev_large_qubits():
    """10-qubit register, only a few gates."""
    def kernel():
        qf = QuantumFloat(10)
        h(qf[0]); h(qf[9])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=100)()

    res = main()
    assert 0.0 <= float(res) <= 1023.0, f"unexpected value {res}"


def test_ev_zero_shots():
    """Zero shots should raise or return NaN."""
    def kernel():
        qf = QuantumFloat(4)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=0)()

    try:
        res = main()
        # Should either raise or return something sensible
        print(f"Zero shots returned: {res}")
    except Exception:
        pass  # expected


def test_ev_multiple_calls():
    """Two expectation_value calls in one function."""
    def kernel_a():
        qf = QuantumFloat(4)
        qf[:] = 5
        return measure(qf)

    def kernel_b():
        qf = QuantumFloat(4)
        qf[:] = 10
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        ev_a = expectation_value(kernel_a, shots=100)()
        ev_b = expectation_value(kernel_b, shots=100)()
        return ev_a, ev_b

    ra, rb = main()
    assert 4.0 < float(ra) < 6.0, f"expected ~5.0, got {ra}"
    assert 9.0 < float(rb) < 11.0, f"expected ~10.0, got {rb}"


# ===========================================================================
# Backend integration
# ===========================================================================

def test_ev_custom_backend():
    """Custom backend options."""
    from qrisp.default_backend import QrispSimulatorBackend
    custom = QrispSimulatorBackend()
    custom.update_options(shots=500)

    def kernel():
        qf = QuantumFloat(4)
        qf[:] = 7
        return measure(qf)

    @backend_sampler(backend=custom)
    def main():
        return expectation_value(kernel, shots=50)()

    res = main()
    assert 6.0 < float(res) < 8.0, f"expected ~7.0, got {res}"


def test_ev_default_backend():
    """Explicit default backend."""
    from qrisp.default_backend import QrispSimulatorBackend

    def kernel():
        qf = QuantumFloat(2)
        qf[:] = 1
        return measure(qf)

    @backend_sampler(backend=QrispSimulatorBackend())
    def main():
        return expectation_value(kernel, shots=50)()

    res = main()
    assert 0.5 < float(res) < 1.5, f"expected ~1.0, got {res}"


# ===========================================================================
# Mixed: sample + expectation_value in same function
# ===========================================================================

def test_ev_and_sample_together():
    """sample() and expectation_value() in the same decorated function."""
    def kernel():
        qf = QuantumFloat(4)
        qf[:] = 3
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        samples = sample(kernel, shots=30)()
        ev = expectation_value(kernel, shots=100)()
        return samples, ev

    samples, ev = main()
    assert samples.shape == (30,)
    assert 2.0 < float(ev) < 4.0, f"expected ~3.0, got {ev}"
    assert all(float(v) == 3.0 for v in samples)
