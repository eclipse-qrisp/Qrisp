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

from qrisp import QuantumFloat, QuantumVariable


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_qswitch_state_preparation(n_qubits):
    """Test qswitch state preparation for various qubit counts."""

    def normalize(vec):
        """Utility to normalize any vector."""
        vec = np.asarray(vec, dtype=complex)
        return vec / np.linalg.norm(vec)

    # Generate two random test vectors
    test_vectors = [
        np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits),
        np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits),
    ]

    for sv in test_vectors:
        sv = normalize(sv)

        qv = QuantumVariable(n_qubits)
        prepared = qv.init_state_qswitch(sv)

        assert np.allclose(prepared, sv, atol=1e-6), (
            f"State preparation failed for {n_qubits} qubits.\n"
            f"Expected: {sv}\nGot: {prepared}"
        )


def test_state_preparation():
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
