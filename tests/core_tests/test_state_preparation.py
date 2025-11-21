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

import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, x
from qrisp.misc.utility import _preprocess, _preprocess_jasp

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


class TestStatePreparationQswitchJasp:

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "statevector_fn",
        [_gen_real_vector, _gen_sparse_vector, _gen_complex_vector],
    )
    def test_preprocess(self, n, statevector_fn):
        """Test that the preprocessing of state preparation matches between Qrisp and Jasp."""

        statevector = statevector_fn(n)
        thetas, leaf_u, leaf_phase = _preprocess(statevector)
        thetas_jasp, leaf_u_jasp, leaf_phase_jasp = _preprocess_jasp(statevector)

        for l, arr in enumerate(thetas):
            k = len(arr)
            assert np.allclose(arr, thetas_jasp[l, :k])

        assert np.allclose(leaf_u, leaf_u_jasp)
        assert np.allclose(leaf_phase, leaf_phase_jasp)


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
