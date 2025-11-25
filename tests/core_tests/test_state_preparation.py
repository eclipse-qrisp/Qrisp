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

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, x
from qrisp.alg_primitives.state_preparation import _preprocess
from qrisp.jasp import terminal_sampling
from qrisp.misc.utility import jasp_bit_reverse


def _rotation_from_state_check(vec: np.ndarray) -> tuple:
    """
    Map |0> → a|0> + b|1>, with a real ≥ 0.

    This is a simpler, non-Jasp version of the rotation_from_state function for testing purposes.
    """
    a, b = vec
    a_real = np.real_if_close(a)
    if a_real < 0:
        # flip a global π phase to make 'a' non-negative real
        a_real = -a_real
        b = -b
    theta = 2.0 * np.arccos(a_real)
    phi = np.angle(b) if abs(b) > 1e-12 else 0.0
    lam = 0.0
    return theta, phi, lam


def _preprocess_check(target_array) -> tuple:
    """
    Preprocess the target statevector for state preparation.

    This is a simpler, non-Jasp version of the _preprocess function for testing purposes.
    """

    n = int(np.log2(target_array.size))
    thetas = [np.zeros(1 << l, dtype=float) for l in range(max(0, n - 1))]
    leaf_u = np.zeros((1 << (n - 1), 3), dtype=float)
    leaf_phase = np.zeros(1 << (n - 1), dtype=float)

    queue = [(target_array, 0, 0, 0.0)]

    while queue:

        subvec, level, prefix_idx, acc_phase = queue.pop(0)

        L = subvec.size
        if L == 2:
            a0 = subvec[0]
            a0_phase = np.angle(a0) if abs(a0) > 1e-12 else 0.0

            vec_n = subvec * np.exp(-1j * a0_phase)
            theta, phi, lam = _rotation_from_state_check(vec_n)

            leaf_u[prefix_idx] = (theta, phi, lam)
            leaf_phase[prefix_idx] = acc_phase + a0_phase
            continue

        half = L // 2
        v0, v1 = subvec[:half], subvec[half:]

        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)

        theta_l = 2.0 * np.arccos(min(1.0, n0))
        thetas[level][prefix_idx] = theta_l

        alpha0 = np.angle(v0[0]) if n0 > 1e-12 else 0.0
        alpha1 = np.angle(v1[0]) if n1 > 1e-12 else 0.0

        v0n = v0 / (n0 * np.exp(1j * alpha0)) if n0 > 1e-12 else v0
        v1n = v1 / (n1 * np.exp(1j * alpha1)) if n1 > 1e-12 else v1

        queue.append((v0n, level + 1, (prefix_idx << 1) | 0, acc_phase + alpha0))
        queue.append((v1n, level + 1, (prefix_idx << 1) | 1, acc_phase + alpha1))

    return thetas, leaf_u, leaf_phase


#######################################
### Test state preparation with qswitch
#######################################


def _compute_statevector_logical_qubits(qv: QuantumVariable) -> np.ndarray:
    """Compute the statevector amplitudes corresponding to the logical qubits"""

    qs_compiled = qv.qs.compile()
    sv = qs_compiled.statevector_array()
    qubits = qs_compiled.qubits

    logical_positions = [qubits.index(qv[i]) for i in range(qv.size)]

    logical_amplitudes = []
    for index in range(2**qv.size):
        bits = format(index, f"0{qv.size}b")
        full_bits = ["0"] * len(qubits)
        for pos, bit in zip(logical_positions, bits):
            full_bits[pos] = bit

        index = int("".join(full_bits), 2)
        logical_amplitudes.append(sv[index])

    return np.array(logical_amplitudes, dtype=complex)


def _gen_real_vector(n):
    """Returns a full real normalized vector."""
    v = np.random.rand(2**n)
    return v / np.linalg.norm(v)


def _gen_sparse_vector(n):
    """Returns a vector with n-1 zeros and one non-zero real entry."""
    v = np.zeros(2**n, dtype=complex)
    v[0] = 1.0
    return v


def _gen_complex_vector(n):
    """Returns a full complex normalized vector."""
    v = np.random.rand(2**n) + 1j * np.random.rand(2**n)
    return v / np.linalg.norm(v)


class TestStatePreparationQSwitch:
    """Test state preparation using the qswitch method."""

    def test_warning_non_normalized(self):
        """Test that a warning is raised when a non-normalized vector is provided."""

        qv = QuantumVariable(3)

        array = np.array([1, 1j, 0, 0, 0, 0, 0, 0], dtype=complex)

        with pytest.warns(
            UserWarning, match="The provided state vector is not normalized"
        ):
            qv.init_state_qswitch(array)

    def test_error_mismatched_size(self):
        """Test that an error is raised when a vector of mismatched size is provided."""

        qv = QuantumVariable(3)

        array = np.array([1j, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="Length of statevector must be 8 for 3 qubits, got 4",
        ):
            qv.init_state_qswitch(array)

    def test_error_fresh_qubits(self):
        """Test that an error is raised when qubits are not in the |0> state."""

        qv = QuantumVariable(2)
        x(qv[0])

        array = np.array([1j, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="Tried to initialize qubits which are not fresh anymore",
        ):
            qv.init_state_qswitch(array)

    def test_error_zero_vector(self):
        """Test that an error is raised when a zero vector is provided."""

        qv = QuantumVariable(2)

        array = np.array([0, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="The provided state vector has zero norm",
        ):
            qv.init_state_qswitch(array)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "statevector_fn",
        [_gen_real_vector, _gen_sparse_vector, _gen_complex_vector],
    )
    def test_state_prep_parametric(self, n, statevector_fn):
        """Test state preparation with qswitch for different statevectors and sizes."""

        qv = QuantumVariable(n)

        array = statevector_fn(n)
        qv.init_state_qswitch(array)

        logical_sv = _compute_statevector_logical_qubits(qv)

        assert np.allclose(logical_sv, array, atol=1e-5)

    @pytest.mark.parametrize("method", ["auto", "tree", "sequential"])
    def test_state_prep_methods(self, method):
        """Test state preparation with different methods."""

        n = 3
        qv = QuantumVariable(n)

        array = _gen_complex_vector(n)
        qv.init_state_qswitch(array, method=method)

        logical_sv = _compute_statevector_logical_qubits(qv)

        assert np.allclose(logical_sv, array, atol=1e-5)


class TestStatePreparationQswitchJasp:

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "statevector_fn",
        [_gen_real_vector, _gen_sparse_vector, _gen_complex_vector],
    )
    def test_preprocess(self, n, statevector_fn):
        """Test that the preprocessing of state preparation matches between Qrisp and Jasp."""

        statevector = statevector_fn(n)
        thetas, leaf_u, leaf_phase = _preprocess_check(statevector)
        thetas_jasp, leaf_u_jasp, leaf_phase_jasp = _preprocess(statevector)

        for l, arr in enumerate(thetas):
            k = len(arr)
            assert np.allclose(arr, thetas_jasp[l, :k])

        assert np.allclose(leaf_u, leaf_u_jasp)
        assert np.allclose(leaf_phase, leaf_phase_jasp)

    @pytest.mark.parametrize("n", [2, 3])
    def test_state_prep_jasp(self, n):
        """Test state preparation with qswitch using Jasp backend."""

        @terminal_sampling(shots=1)
        def main(idx):
            qv = QuantumFloat(n)
            state_vector = jnp.zeros(2**n, dtype=complex)
            state_vector = state_vector.at[idx].set(1.0)
            qv.init_state_qswitch(state_vector)
            return qv

        for idx in range(2**n):
            dict_res = main(idx)
            key = jasp_bit_reverse(idx, n)
            assert dict_res[float(key)] == 1


def test_state_preparation():
    """Test state preparation for QuantumFloat with a dictionary."""

    qf = QuantumFloat(4, -2, signed=True)

    state_dic = {
        2.75: 1 / 4**0.5,
        -1.5: -1 / 4**0.5,
        2: 1 / 4**0.5,
        3: 1j / 4**0.5,
    }

    qf.init_state(state_dic)

    debugger = qf.qs.statevector("function")

    print(type(debugger))

    print("Amplitude of state 2.75: ", debugger({qf: 2.75}))
    print("Amplitude of state -1.5: ", debugger({qf: -1.5}))
    print("Amplitude of state 3: ", debugger({qf: 3}))

    assert np.abs(debugger({qf: 2.75}) - 0.5) < 1e-5
    assert np.abs(debugger({qf: -1.5}) + 0.5) < 1e-5
    assert np.abs(debugger({qf: 3}) - 0.5j) < 1e-5
