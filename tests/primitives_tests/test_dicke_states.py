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

import math

import numpy as np
import pytest

from qrisp import QuantumVariable
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_w_state
from qrisp.core import h, x
from qrisp.environments import control, invert
from qrisp.jasp import terminal_sampling

#############################################################
##################### Dicke state tests #####################
#############################################################

# Every Dicke state test is run with both preparation methods.
DICKE_METHODS = ["deterministic", "divide-and-conquer"]


@pytest.mark.parametrize("method", DICKE_METHODS)
@pytest.mark.parametrize(
    "n, k",
    [
        (3, 1),
        (4, 1),
        (1, 1),
    ],
)
def test_dicke_state_balanced(n, k, method):
    # n - Number of qubits
    # k - Excitations
    # Prepare balanced Dicke state
    qv = QuantumVariable(n)
    x(qv[n - 1])
    dicke_state(qv, k, method=method)
    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state:
    # |D^3_1> = (|001> + |010> + |100>) / sqrt(3)
    expected_sv = np.zeros(2**n, dtype=complex)
    amp = 1 / np.sqrt(n)
    for i in range(n):
        expected_sv[2**i] = amp

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)


@pytest.mark.parametrize("method", DICKE_METHODS)
@pytest.mark.parametrize("n, k", [(4, 2), (3, 2), (5, 3)])
def test_dicke_state_statevector(n, k, method):
    # Same as above for k > 1, where the amplitudes (and not just the
    # probabilities) of the individual branches have to line up.
    qv = QuantumVariable(n)
    for i in range(n - k, n):
        x(qv[i])
    dicke_state(qv, k, method=method)
    prepared_sv = qv.qs.compile().statevector_array()

    # The compiler allocates ancillae for the doubly controlled rotations, so
    # the statevector is not necessarily of size 2**n. The ancillae are
    # uncomputed to |0>, so every populated basis state still has Hamming
    # weight k, no matter where the ancillae ended up.
    populated = np.flatnonzero(~np.isclose(prepared_sv, 0, atol=1e-6))

    # Manual expected state:
    # |D^n_k> = equal superposition of all basis states of Hamming weight k,
    # i.e. binom(n, k) states sharing one real, positive amplitude.
    assert len(populated) == math.comb(n, k)
    assert all(bin(i).count("1") == k for i in populated)
    assert np.allclose(prepared_sv[populated], 1 / np.sqrt(math.comb(n, k)), atol=1e-6)


@pytest.mark.parametrize("method", DICKE_METHODS)
def test_dicke_state_balanced_jasp(method):
    n = 3  # Number of qubits
    k = 1  # Excitations

    # Prepare balanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        x(qv[n - 1])
        dicke_state(qv, k, method=method)
        return qv

    result = main()

    res_arr = np.zeros(2**n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2**n)
    amp = 1 / n
    for i in range(n):
        expected_arr[2**i] = amp

    assert np.allclose(res_arr, expected_arr, atol=1e-6)


@pytest.mark.parametrize("method", DICKE_METHODS)
def test_dicke_state_balanced_jasp_inverse(method):
    n = 3  # Number of qubits
    k = 1  # Excitations

    # Prepare balanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        x(qv[n - 1])
        dicke_state(qv, k, method=method)
        with invert():
            x(qv[n - 1])
            dicke_state(qv, k, method=method)
        return qv

    result = main()

    res_arr = np.zeros(2**n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2**n)
    expected_arr[0] = 1

    assert np.allclose(res_arr, expected_arr, atol=1e-6)


def test_dicke_state_jasp_large_k_divide_and_conquer(method):
    n = 4  # Number of qubits
    k = 3  # Excitations

    # For k > n/2 the state is prepared as D(n, n-k) and then inverted. Under
    # Jasp that reduction is traced, so it needs its own test.
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        for i in range(n - k, n):
            x(qv[i])
        dicke_state(qv, k, method="divide-and-conquer")
        return qv

    result = main()

    res_arr = np.zeros(2**n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2**n)
    for i in range(2**n):
        if bin(i).count("1") == k:
            expected_arr[i] = 1 / math.comb(n, k)

    assert np.allclose(res_arr, expected_arr, atol=1e-6)


@pytest.mark.parametrize("n, k", [(4, 2), (5, 3)])
def test_dicke_state_methods_agree(n, k):
    # Preparing with one method and uncomputing with the other has to return
    # the variable to its initial state. Unlike a comparison of measurement
    # probabilities, this also detects a phase mismatch between the two.
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        for i in range(n - k, n):
            x(qv[i])
        dicke_state(qv, k, method="deterministic")
        with invert():
            for i in range(n - k, n):
                x(qv[i])
            dicke_state(qv, k, method="divide-and-conquer")
        return qv

    result = main()

    res_arr = np.zeros(2**n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2**n)
    expected_arr[0] = 1

    assert np.allclose(res_arr, expected_arr, atol=1e-6)


@pytest.mark.parametrize("method", DICKE_METHODS)
@pytest.mark.parametrize("n, k", [(4, 2), (5, 3), (6, 5), (3, 3), (1, 1), (5, 0)])
def test_dicke_state_k(n, k, method):
    qv = QuantumVariable(n)

    for i in range(n - k, n):
        x(qv[i])

    dicke_state(qv, k, method=method)

    res = qv.get_measurement()

    assert len(res) == math.comb(n, k)

    expected_prob = 1 / math.comb(n, k)

    for outcome, prob in res.items():
        # get_measurement returns bitstrings and integer-like outcomes.
        # Force an n-bit string.
        if isinstance(outcome, str):
            bitstring = outcome
        else:
            bitstring = format(int(outcome), f"0{n}b")

        assert bitstring.count("1") == k
        assert np.isclose(prob, expected_prob, atol=1e-6)

@pytest.mark.parametrize(
    "n, k, l",
    [
        (2, 2, 1),
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (4, 3, 1),
        (4, 3, 2),
        (5, 3, 2),
        (5, 4, 2),
        (6, 4, 2),
        (6, 5, 3),
    ],
)
def test_dicke_state_deterministic_lower_hamming_weight(n, k, l):
    # The deterministic method applies the Dicke state unitary U_{n,k}, which
    # is defined by U_{n,k} |0...01...1> = |D^n_l> for *every* l <= k and not
    # just for l == k (arXiv:2112.12435, Eq. 2). Divide-and-conquer is a state
    # preparation circuit rather than a Dicke state unitary and assumes a
    # Hamming weight of exactly k, hence deterministic only.
    qv = QuantumVariable(n)
    for i in range(n - l, n):
        x(qv[i])

    dicke_state(qv, k, method="deterministic")

    prepared_sv = qv.qs.compile().statevector_array()
    populated = np.flatnonzero(~np.isclose(prepared_sv, 0, atol=1e-6))

    # Manual expected state:
    # |D^n_l> = equal superposition of all basis states of Hamming weight l.
    assert len(populated) == math.comb(n, l)
    assert all(bin(i).count("1") == l for i in populated)
    assert np.allclose(prepared_sv[populated], 1 / np.sqrt(math.comb(n, l)), atol=1e-6)


@pytest.mark.parametrize("n, k, l", [(4, 3, 1), (5, 4, 2), (6, 5, 3)])
def test_dicke_state_deterministic_lower_hamming_weight_jasp(n, k, l):
    # Same property under Jasp. k > n/2 combined with l < k is the case that a
    # k -> n-k reduction inside dicke_state would silently break.
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        for i in range(n - l, n):
            x(qv[i])
        dicke_state(qv, k, method="deterministic")
        return qv

    result = main()

    res_arr = np.zeros(2**n)
    for key in result:
        res_arr[int(key)] = result[key]

    # Manual expected measurement:
    expected_arr = np.zeros(2**n)
    for i in range(2**n):
        if bin(i).count("1") == l:
            expected_arr[i] = 1 / math.comb(n, l)

    assert np.allclose(res_arr, expected_arr, atol=1e-6)


@pytest.mark.parametrize("n, k, l", [(4, 2, 1), (5, 3, 2), (6, 4, 3)])
def test_dicke_state_deterministic_superposition(n, k, l):
    # U_{n,k} is a unitary, so the property above has to survive on
    # superpositions: (|0...01^l> + |0...01^(l+1)>) / sqrt(2) has to come out
    # as (|D^n_l> + |D^n_(l+1)>) / sqrt(2), relative phase included. This is
    # what a caller assembling symmetric states actually relies on.
    qv = QuantumVariable(n)
    for i in range(n - l, n):
        x(qv[i])
    h(qv[n - l - 1])

    dicke_state(qv, k, method="deterministic")

    prepared_sv = qv.qs.compile().statevector_array()
    populated = np.flatnonzero(~np.isclose(prepared_sv, 0, atol=1e-6))

    # Manual expected state:
    # every basis state of Hamming weight l or l+1, carrying a real, positive
    # amplitude of 1 / sqrt(2 binom(n, weight)).
    assert len(populated) == math.comb(n, l) + math.comb(n, l + 1)

    for i in populated:
        weight = bin(i).count("1")
        assert weight in (l, l + 1)
        assert np.isclose(
            prepared_sv[i], 1 / np.sqrt(2 * math.comb(n, weight)), atol=1e-6
        )


def test_dicke_state_fail_unknown_method():
    n = 3  # Number of qubits
    k = 1  # Excitations

    qv = QuantumVariable(n)
    x(qv[n - 1])

    with pytest.raises(ValueError) as exc_info:
        dicke_state(qv, k, method="divide_and_conquer")

    assert "Unknown `method`" in str(exc_info.value)


@pytest.mark.parametrize("n, k", [(8, 2), (12, 3)])
def test_dicke_state_divide_and_conquer_depth(n, k):
    # The divide-and-conquer method prepares the two halves of the variable on
    # disjoint qubits, so they are applied in parallel and the circuit depth is
    # roughly halved compared to the deterministic method.
    depths = {}

    for method in DICKE_METHODS:
        qv = QuantumVariable(n)
        for i in range(n - k, n):
            x(qv[i])
        dicke_state(qv, k, method=method)
        depths[method] = qv.qs.compile().depth()

    assert depths["divide-and-conquer"] < 0.8 * depths["deterministic"]


##############################################################
################## Unbalanced W state tests ##################
##############################################################


@pytest.mark.parametrize(
    "amps",
    [
        pytest.param(
            np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex),
            id="three_qubits_complex",
        ),
        pytest.param(
            np.array([0.25 + 0.2j, 0, 0], dtype=complex),
            id="trailing_zeroes",
        ),
        pytest.param(
            np.array([0.25 + 0.2j], dtype=complex),
            id="one_qubit",
        ),
    ],
)
def test_unbalanced_w_state(amps):
    n = len(amps)

    qv = QuantumVariable(n)
    unbalanced_w_state(qv, amps[::-1])

    prepared_sv = qv.qs.compile().statevector_array()

    # Manual expected state
    # e.g. |ψ> = a0 |001> + a1 |010> + a2 |100>
    expected_sv = np.zeros(2**n, dtype=complex)
    normalized_amps = amps / np.linalg.norm(amps)
    for i in range(n):
        expected_sv[2**i] = normalized_amps[i]

    assert np.allclose(prepared_sv, expected_sv, atol=1e-6)


@pytest.mark.parametrize(
    "amps",
    [
        pytest.param(
            np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex),
            id="three_qubits_complex",
        ),
        pytest.param(
            np.array([0.25 + 0.2j, 0, 0], dtype=complex),
            id="trailing_zeroes",
        ),
        pytest.param(
            np.array([0.25 + 0.2j], dtype=complex),
            id="one_qubit",
        ),
    ],
)
def test_unbalanced_w_state_measurement(amps):
    n = len(amps)
    qv = QuantumVariable(n)

    unbalanced_w_state(qv, amps)
    result = qv.get_measurement()

    normalized_amps = amps / np.linalg.norm(amps)
    expected = {
        format(2 ** (n - 1 - i), f"0{n}b"): float(abs(normalized_amps[i]) ** 2)
        for i in range(n)
        if not np.isclose(normalized_amps[i], 0, atol=1e-12)
    }

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)


def test_unbalanced_w_state_jasp():
    n = 3  # Number of qubits
    amps = np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex)

    # Prepare unbalanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        unbalanced_w_state(qv, amps)
        return qv

    result = main()
    # Manual expected state:
    # |ψ> = a0 |001> + a1 |010> + a2 |100>
    norm = np.linalg.norm(amps)
    normalized_amps = amps / norm
    expected = {2**i: float(abs(normalized_amps[i]) ** 2) for i in range(n)}

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)


def test_unbalanced_w_state_jasp_inverse():
    n = 3  # Number of qubits
    amps = np.array([0.25 + 0.2j, 0.375 + 0.18j, 0.375], dtype=complex)

    # Prepare unbalanced Dicke state
    @terminal_sampling
    def main():
        qv = QuantumVariable(n)
        unbalanced_w_state(qv, amps)
        with invert():
            unbalanced_w_state(qv, amps)
        return qv

    result = main()
    # Manual expected state:
    # |ψ> = 1 |000>
    expected = {0: 1.0}

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)


def test_unbalanced_w_state_fail_len_check():
    n = 1  # Number of qubits
    amps = np.array([0.25 + 0.2j, 2, 3, 4, 5, 6, 7], dtype=complex)

    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    with pytest.raises(ValueError) as exc_info:
        unbalanced_w_state(qv, amps)

    assert "Length of amplitudes" in str(exc_info.value)


def test_unbalanced_w_state_fail_zero_vector():
    n = 3  # Number of qubits
    amps = np.array([0, 0, 0], dtype=complex)

    # Prepare unbalanced Dicke state
    qv = QuantumVariable(n)
    with pytest.raises(ValueError) as exc_info:
        unbalanced_w_state(qv, amps)

    assert "Amplitude vector must be non-zero." in str(exc_info.value)


def test_unbalanced_w_state_one_qubit_jasp():
    phi = 0.73
    amps = np.array([np.exp(1j * phi)], dtype=complex)

    @terminal_sampling
    def main():
        ctrl = QuantumVariable(1)
        target = QuantumVariable(1)

        # Prepare (|0> + |1>) / sqrt(2) on the control.
        h(ctrl[0])

        # On the ctrl=1 branch:
        #   unbalanced_w_state(target, [exp(i phi)]) prepares exp(i phi)|1>.
        #   x(target) maps |1> back to |0>, leaving exp(i phi) as a
        #   relative phase on the control branch.
        with control(ctrl[0]):
            unbalanced_w_state(target, amps)
            x(target[0])

        # Convert the relative phase into measurement probabilities.
        h(ctrl[0])

        return ctrl

    result = main()

    expected = {
        0: float(np.cos(phi / 2) ** 2),
        1: float(np.sin(phi / 2) ** 2),
    }

    keys = sorted(set(result) | set(expected))
    result_arr = np.array([result.get(k, 0.0) for k in keys])
    expected_arr = np.array([expected.get(k, 0.0) for k in keys])

    assert np.allclose(result_arr, expected_arr, atol=1e-6)
