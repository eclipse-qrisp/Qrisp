"""
********************************************************************************
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
from qrisp import QuantumVariable
from qrisp.core import x
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_W_state
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from qrisp.jasp import terminal_sampling
from qrisp.environments import invert

#############################################################
##################### Dicke state tests #####################
#############################################################

def test_dicke_state_balanced():
    n = 3 # Number of qubits
    k = 1 # Excitations
    # Prepare balanced Dicke state
    qv = QuantumVariable(n)
    x(qv[n - 1])
    dicke_state(qv, k)
    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state:
    # |D^3_1> = (|001> + |010> + |100>) / sqrt(3)
    expected_sv = np.zeros(2 ** n, dtype=complex)
    amp = 1 / np.sqrt(n)
    for i in range(n):
        expected_sv[2 ** i] = amp

    print(f"Prepared statevector:\n{prepared_sv}")
    print(f"Expected statevector:\n{expected_sv}")

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)

def test_dicke_state_balanced_jasp():
    n = 3 # Number of qubits
    k = 1 # Excitations

    # Prepare balanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        x(qv[n - 1])
        dicke_state(qv, k)
        return qv
    
    result = main()
    #prepared_sv = result.qs.compile().statevector_array()
    print(result)

    res_arr = np.zeros(2 ** n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2 ** n)
    amp = 1 / n
    for i in range(n):
        expected_arr[2 ** i] = amp

    print(f"Measured distribution:\n{res_arr}")
    print(f"Expected distribution:\n{expected_arr}")

    assert np.allclose(res_arr, expected_arr, atol=1e-6)

def test_dicke_state_balanced_jasp_inverse():
    n = 3 # Number of qubits
    k = 1 # Excitations

    # Prepare balanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        x(qv[n - 1])
        dicke_state(qv, k)
        with invert():
            x(qv[n - 1])
            dicke_state(qv, k)
        return qv
    
    result = main()
    #prepared_sv = result.qs.compile().statevector_array()
    print(result)

    res_arr = np.zeros(2 ** n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2 ** n)
    expected_arr[0] = 1

    print(f"Measured distribution:\n{res_arr}")
    print(f"Expected distribution:\n{expected_arr}")

    assert np.allclose(res_arr, expected_arr, atol=1e-6)

##############################################################
################## Unbalanced W state tests ##################
##############################################################

def test_unbalanced_W_state():
    n = 3 # Number of qubits
    amps = np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex)
    
    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    unbalanced_W_state(qv, amps, reversed=True)
    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state:
    # |ψ> = a0 |001> + a1 |010> + a2 |100>
    expected_sv = np.zeros(2 ** n, dtype=complex)
    norm = np.linalg.norm(amps)
    normalized_amps = amps / norm
    for i in range(n):
        expected_sv[2 ** i] = normalized_amps[i]

    print(f"Prepared statevector:\n{prepared_sv}")
    print(f"Expected statevector:\n{expected_sv}")

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)

def test_unbalanced_W_state_trailing_zeroes():
    n = 3 # Number of qubits
    amps = np.array([0.25 + 0.2j, 0, 0], dtype=complex)
    
    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    unbalanced_W_state(qv, amps, reversed=True)
    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state:
    # |ψ> = a0 |001> + a1 |010> + a2 |100>
    expected_sv = np.zeros(2 ** n, dtype=complex)
    norm = np.linalg.norm(amps)
    normalized_amps = amps / norm
    for i in range(n):
        expected_sv[2 ** i] = normalized_amps[i]

    print(f"Prepared statevector:\n{prepared_sv}")
    print(f"Expected statevector:\n{expected_sv}")

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)

def test_unbalanced_W_state_jasp():
    n = 3 # Number of qubits
    amps = np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex)
    
    # Prepare unbalanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        unbalanced_W_state(qv, amps)
        return qv
    result = main()
    # Manual expected state:
    # |ψ> = a0 |001> + a1 |010> + a2 |100>
    norm = np.linalg.norm(amps)
    normalized_amps = amps / norm
    expected = {
        2**i: float(abs(normalized_amps[i]) ** 2)
        for i in range(n)
    }

    print(f"Prepared measurement:\n{result}")
    print(f"Expected measurement:\n{expected}")

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)

def test_unbalanced_W_state_jasp_inverse():
    n = 3 # Number of qubits
    amps = np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex)
    
    # Prepare unbalanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        unbalanced_W_state(qv, amps)
        with invert():
            unbalanced_W_state(qv, amps)
        return qv
    result = main()
    # Manual expected state:
    # |ψ> = 1 |000>
    expected = {0: 1.0}

    print(f"Prepared measurement:\n{result}")
    print(f"Expected measurement:\n{expected}")

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)

def test_unbalanced_W_state_one_qubit():
    n = 1 # Number of qubits
    amps = np.array([0.25 + 0.2j], dtype=complex)
    
    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    unbalanced_W_state(qv, amps, reversed=True)
    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state
    expected_sv = np.zeros(2 ** n, dtype=complex)
    norm = np.linalg.norm(amps)
    normalized_amps = amps / norm
    expected_sv[1] = normalized_amps[0]

    print(f"Prepared statevector:\n{prepared_sv}")
    print(f"Expected statevector:\n{expected_sv}")

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)

def test_unbalanced_W_state_fail_len_check():
    n = 1 # Number of qubits
    amps = np.array([0.25 + 0.2j, 2, 3, 4, 5, 6, 7], dtype=complex)

    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    with pytest.raises(ValueError) as exc_info:
        unbalanced_W_state(qv, amps, reversed=True)

    print(exc_info.value)
    assert f"Length of amplitudes" in str(exc_info.value)

def test_unbalanced_W_state_fail_zero_vector():
    n = 3 # Number of qubits
    amps = np.array([0, 0, 0], dtype=complex)

    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    with pytest.raises(ValueError) as exc_info:
        unbalanced_W_state(qv, amps, reversed=True)

    print(exc_info.value)
    assert f"Amplitude vector must be non-zero." in str(exc_info.value)
