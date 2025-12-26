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
from qrisp.jasp import terminal_sampling
from qrisp.misc.utility import _EPSILON, bit_reverse

########################################################
### Test state preparation with array (based on qswitch)
########################################################


def _compute_statevector_logical_qubits(qv: QuantumVariable) -> np.ndarray:
    """Compute the statevector amplitudes corresponding to the logical qubits"""

    qs_compiled = qv.qs.compile()
    sv = qs_compiled.statevector_array()
    qubits = qs_compiled.qubits

    logical_positions = [qubits.index(qv[i]) for i in range(qv.size)]

    logical_amplitudes = []
    for index in range(1 << qv.size):
        bits = format(index, f"0{qv.size}b")
        full_bits = ["0"] * len(qubits)
        for pos, bit in zip(logical_positions, bits):
            full_bits[pos] = bit

        index = int("".join(full_bits), 2)
        logical_amplitudes.append(sv[index])

    return np.array(logical_amplitudes, dtype=complex)


def _gen_real_vector(n):
    """Returns a full real normalized vector."""
    v = np.random.rand(1 << n) - 0.5
    return v / np.linalg.norm(v)


def _gen_sparse_vector(n, idx=0):
    """Returns a vector with n-1 zeros and one non-zero real entry."""
    v = jnp.zeros(1 << n, dtype=complex)
    v = v.at[idx].set(1.0)
    return v


def _gen_complex_vector(n):
    """Returns a full complex normalized vector."""
    v = (np.random.rand(1 << n) - 0.5) + 1j * (np.random.rand(1 << n) - 0.5)
    return v / np.linalg.norm(v)


def _gen_uniform_vector(n):
    """Returns a uniform superposition vector."""
    v = np.ones(1 << n, dtype=complex)
    return v / np.linalg.norm(v)


class TestStatePreparationQSwitch:
    """Test state preparation using the qswitch method."""

    def test_error_mismatched_size(self):
        """Test that an error is raised when a vector of mismatched size is provided."""

        qv = QuantumVariable(3)

        array = np.array([1j, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="Statevector length must be 8 for 3 qubits, got 4",
        ):
            qv.init_state(array, method="qswitch")

    def test_error_fresh_qubits(self):
        """Test that an error is raised when qubits are not in the |0> state."""

        qv = QuantumVariable(2)
        x(qv[0])

        array = np.array([1j, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="Tried to initialize qubits which are not fresh anymore",
        ):
            qv.init_state(array, method="qswitch")

    def test_error_zero_vector(self):
        """Test that an error is raised when a zero vector is provided."""

        qv = QuantumVariable(2)

        array = np.array([0, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="The provided statevector has zero norm",
        ):
            qv.init_state(array, method="qswitch")

    def test_error_unrecognized_method(self):
        """Test that an error is raised when an unrecognized method is provided."""

        qv = QuantumVariable(2)

        array = np.array([1j, 0, 0, 0], dtype=complex)

        with pytest.raises(
            ValueError,
            match="method must be 'auto', 'qiskit', or 'qswitch'",
        ):
            qv.init_state(array, method="unknown_method")

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "statevector_fn",
        [_gen_real_vector, _gen_sparse_vector, _gen_complex_vector],
    )
    @pytest.mark.parametrize("method", ["qswitch", "qiskit"])
    def test_state_prep_parametric(self, n, statevector_fn, method):
        """Test state preparation for various number of qubits, statevectors, and methods."""

        qv = QuantumVariable(n)

        array = statevector_fn(n)
        qv.init_state(array, method=method)

        logical_sv = _compute_statevector_logical_qubits(qv)

        assert np.allclose(logical_sv, array, atol=1e-5)

    def test_phase_varied_state(self):
        """Test state preparation of a state with varied phases."""

        qv = QuantumVariable(3)
        array = jnp.array([1, 1j, -1, -1j, 1, 1j, -1, -1j])
        array /= jnp.linalg.norm(array)
        qv.init_state(array, method="qswitch")

        logical_sv = _compute_statevector_logical_qubits(qv)
        assert np.allclose(logical_sv, array, atol=1e-5)

    def test_near_zero_amplitudes(self):
        """Test state preparation of a state with near-zero amplitudes."""

        qv = QuantumVariable(3)
        array = jnp.array([1.0, _EPSILON, 0, 0, 0, _EPSILON * 1j, -_EPSILON, 0])
        array /= jnp.linalg.norm(array)
        qv.init_state(array, method="qswitch")

        logical_sv = _compute_statevector_logical_qubits(qv)
        assert np.allclose(logical_sv, array, atol=1e-5)


class TestStatePreparationQswitchJasp:
    """Test state preparation using the qswitch method in JASP mode."""

    def test_error_if_qiskit_in_jasp(self):
        """Test that an error is raised when method 'qiskit' is used in JASP mode."""

        @terminal_sampling(shots=10)
        def main(state_vector):
            qv = QuantumFloat(2)
            qv.init_state(state_vector, method="qiskit")
            return qv

        with pytest.raises(
            ValueError,
            match="Tried to initialize dynamic jax array using state preparation method qiskit",
        ):
            state_vector = _gen_real_vector(2)
            main(state_vector)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_basis_states(self, n):
        """Test state preparation of basis states in JASP mode."""

        @terminal_sampling(shots=1)
        def main(idx):
            qv = QuantumFloat(n)
            state_vector = _gen_sparse_vector(n, idx)
            qv.init_state(state_vector)
            return qv

        for idx in range(1 << n):
            dict_res = main(idx)
            key = bit_reverse(idx, n)
            assert len(dict_res) == 1
            assert dict_res[float(key)] == 1

    # The standard deviation expected on the number of shots per state is
    # sqrt(n_shots * p * (1-p)) where p = 1 / 2^n
    # We set a tolerance of 6-sigma.
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_uniform_superposition(self, n):
        """Test state preparation of uniform superposition in JASP mode."""

        n_shots = 10000
        std = jnp.sqrt(n_shots * (1 / (1 << n)) * (1 - 1 / (1 << n)))
        tolerance = 6 * std

        @terminal_sampling(shots=n_shots)
        def main():
            qv = QuantumFloat(n)
            state_vector = _gen_uniform_vector(n)
            qv.init_state(state_vector)
            return qv

        dict_res = main()
        expected_shots = n_shots / (1 << n)

        for key in dict_res:
            n_shots = dict_res[key]
            assert np.abs(n_shots - expected_shots) < tolerance

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_sparse_k_state_superposition(self, n):
        """Test state prep for a sparse uniform superposition on several states."""

        shots = 30000
        n_qubits = 1 << n
        k = 3

        p = 1 / k
        expected = shots * p
        std = np.sqrt(shots * p * (1 - p))
        tolerance = 6 * std

        idxs = np.random.choice(n_qubits, size=k, replace=False)

        @terminal_sampling(shots=shots)
        def main():
            qv = QuantumFloat(n)
            state_vector = jnp.zeros(n_qubits, dtype=complex)
            amp = 1.0 / jnp.sqrt(k)
            for i in idxs:
                state_vector = state_vector.at[i].set(amp)
            qv.init_state(state_vector)
            return qv

        res = main()

        keys = {float(bit_reverse(i, n)) for i in idxs}
        assert set(res.keys()) == keys

        for key in keys:
            assert abs(res[key] - expected) < tolerance


############################################################
### Test state preparation with dictionary (based on qiskit)
############################################################


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
