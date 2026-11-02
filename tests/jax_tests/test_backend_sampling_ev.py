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

import numpy as np
import pytest

from qrisp import QuantumBool, QuantumFloat, cx, h, measure, x
from qrisp.jasp import backend_sampler, expectation_value, jaspify, sample

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
    assert 0.3 < float(res) < 0.7


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
    assert np.isclose(res, 5.0)


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
    assert 2.5 < float(res) < 4.5


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
    assert 0.0 < float(res) < 1.0


def test_ev_ghz():
    """GHZ: |0000⟩ + |1111⟩ → expectation ≈ 7.5 (50% 0, 50% 15)"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        cx(qf[0], qf[1])
        cx(qf[1], qf[2])
        cx(qf[2], qf[3])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    assert 5.0 < float(res) < 10.0


# ===========================================================================
# Post-processing in expectation_value
# ===========================================================================


def test_ev_postproc():
    """E[measure(qf) * 2 + 1] with 2 random bits → expectation = (avg of {1,3,5,7}) = 4.0"""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        h(qf[1])
        return measure(qf) * 2 + 1

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=500)()

    res = main()
    assert 3.0 < float(res) < 5.0


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
    assert 0.3 < float(res) < 0.7


# ===========================================================================
# Comparison against jaspify
# ===========================================================================


def test_ev_compare_jaspify():
    """Backend sampler expectation_value matches jaspify."""

    def kernel():
        qf = QuantumFloat(4)
        h(qf[0])
        h(qf[1])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def backend_main():
        return expectation_value(kernel, shots=300)()

    @jaspify
    def jaspify_main():
        return expectation_value(kernel, shots=300)()

    r_backend = float(backend_main())
    r_jaspify = float(jaspify_main())

    # Both estimate the same mean of ~1.5 (avg of 0,1,2,3). The standard
    # error of the difference is ~0.091, so 0.5 is a ~5.5 sigma bound:
    # tight enough to catch a real divergence, loose enough not to flake.
    assert abs(r_backend - r_jaspify) < 0.5


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

    # Both estimate the same mean of ~3.5 (avg of 0..7). The standard error
    # of the difference is ~0.072, so 0.5 is a ~6.9 sigma bound.
    assert abs(r_backend - r_jaspify) < 0.5


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
    assert np.isclose(res, 0.0)


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
    assert np.isclose(res, 7.0)


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
    assert np.isclose(res, 3.0)


def test_ev_large_qubits():
    """10-qubit register, only a few gates."""

    def kernel():
        qf = QuantumFloat(10)
        h(qf[0])
        h(qf[9])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=100)()

    res = main()
    assert 0.0 <= float(res) <= 1023.0


def test_ev_zero_shots():
    """shots=0 is rejected by backend_sampler.

    Unlike sample(), expectation_value runs even a plain int through
    make_tracer, so the count is no longer inspectable while tracing. It is
    caught in the callback instead, and the decorator restores the original
    ValueError from the XlaRuntimeError that XLA raises in its place.
    """

    def kernel():
        qf = QuantumFloat(4)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=0)()

    with pytest.raises(ValueError, match="positive number of shots"):
        main()


def test_ev_zero_shots_dynamic():
    """A dynamic shots=0 is rejected once it is resolved.

    Same guard as the static case: the callback is where a traced shot count
    finally becomes concrete.
    """

    def kernel():
        qf = QuantumFloat(4)
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main(n):
        return expectation_value(kernel, shots=n)()

    with pytest.raises(ValueError, match="positive number of shots"):
        main(0)

    # a positive dynamic shot count still works
    assert np.isclose(main(20), 0.0)


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
    assert np.isclose(ra, 5.0)
    assert np.isclose(rb, 10.0)


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
    assert np.isclose(res, 7.0)


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
    assert np.isclose(res, 1.0)


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
    assert np.isclose(ev, 3.0)
    assert np.allclose(samples, 3.0)


def test_return_dict_is_rejected():
    """``return_dict=True`` marks the eval function for terminal sampling.

    ``backend_sampler`` returns through a jitted ``pure_callback``, which needs
    a static output shape and so cannot produce a dict. Left unintercepted the
    quantum state reaches the jit boundary and XLA raises an opaque aval error,
    so the decorator rejects it explicitly instead.
    """

    def kernel():
        qf = QuantumFloat(3)
        h(qf[0])
        return measure(qf)

    @backend_sampler(backend=_get_backend())
    def main():
        return expectation_value(kernel, shots=20, return_dict=True)()

    with pytest.raises(NotImplementedError, match="return_dict"):
        main()
