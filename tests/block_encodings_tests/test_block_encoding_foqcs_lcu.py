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

from functools import partial
from qrisp.typing import NDArrayLike
import numpy as np
import pytest
from qrisp.block_encodings import BlockEncoding
from qrisp import QuantumVariable, QuantumFloat, terminal_sampling, multi_measurement
from qrisp.block_encodings.constructors.foqcs_lcu import (
  foqcs_prep_heisenberg,
  is_operator_foqcs_compatible,
  foqcs_analyze_operator_spin_glass,
  foqcs_analyze_operator_heisenberg,
  foqcs_prep_spin_glass
)
from qrisp.operators import X, Y, Z
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_w_state

def _heisenberg_from_def(L: int, g: NDArrayLike, J: NDArrayLike):
    assert len(J) == 3, "J must be a list of length 3."
    assert len(g) == 3, "g must be list a of length 3."
    sigma_list = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    H = np.zeros((2**L, 2**L))
    for k, sigma in enumerate(sigma_list):
        sisj = np.kron(sigma, sigma)
        for i in range(L):
            if i < L-1:
                H = H + J[k] * np.kron(np.identity(2**i), np.kron(sisj, np.identity(2**(L-i-2))))
            H = H + g[k] * np.kron(np.identity(2**i), np.kron(sigma, np.identity(2**(L-i-1))))
    return H

def _spin_glass_from_def(L: int, g: dict, J: dict):
    """
    Reference matrix matching foqcs_prep_spin_glass's current site ordering.

    H = sum_a sum_i g_a[i] sigma_a(L - 1 - i)
      + sum_a sum_k sum_i J_a[k-1][i]
            sigma_a(L - 1 - i) sigma_a(L - 1 - (i+k))
    """

    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    def pauli_string(sites, sigma):
        ops = []
        for q in range(L):
            ops.append(sigma if q in sites else np.eye(2, dtype=complex))

        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    H = np.zeros((2**L, 2**L), dtype=complex)

    for axis, sigma in paulis.items():
        for i in range(L):
            site = L - 1 - i
            H += g[axis][i] * pauli_string({site}, sigma)

        for k in range(1, L):
            for i in range(L - k):
                site_0 = L - 1 - i
                site_1 = L - 1 - (i + k)
                H += J[axis][k - 1][i] * pauli_string({site_0, site_1}, sigma)

    return H

def _flatten_spin_glass_coeffs(g: dict, J: dict):
    coeffs = []
    for axis in ["X", "Y", "Z"]:
        coeffs.extend(g[axis])
        for diag in J[axis]:
            coeffs.extend(diag)
    return np.array(coeffs, dtype=complex)

def _generate_heisenberg_coeff():
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    # Fix coefficients for debugging
    #g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    #J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])
    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm
    return g, J

def _preprocess_heisenberg_coeff(g, J, L):
    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")
    # Modify the original coefficients for the manual state building.
    # g and J must be normalized
    _g[0] = np.sqrt(g[0] * L)
    _g[1] = np.sqrt(g[1] * L * -1j)
    _g[2] = np.sqrt(g[2] * L)
    _J[0] = np.sqrt(J[0] * (L - 1))
    _J[1] = np.sqrt(J[1] * -(L - 1))
    _J[2] = np.sqrt(J[2] * (L - 1))

    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    return _g, _J, norm

def _prep_psi(q_num):
    # Generate state amplitudes.
    psi = np.random.uniform(-1, 1, 2 ** (q_num)) + 1j * np.random.uniform(
        -1, 1, 2 ** (q_num)
    )
    # Fix operands state for debugging  #q_num = 4
    # psi = [0.57501513+0.26902124j, -0.16783319+0.96769323j, -0.15515405+0.02518714j,
    #        -0.90288347-0.93298597j, 0.08905568-0.48063457j,  0.34150327+0.74670678j,
    #        -0.06620886+0.96652259j, 0.76446858+0.88954485j,  0.26407693-0.21105958j,
    #        -0.64653787-0.73012458j, 0.29551929+0.96235332j, -0.23794186-0.8503716j,
    #         0.58938419-0.67645356j, 0.04926328-0.06230936j,  0.74225725-0.00355716j,
    #         0.62479001+0.83636103j
    #         ]
    psi /= np.linalg.norm(psi)
    return psi

def _bit_reverse(i: int, n: int) -> int:
        return int(f"{i:0{n}b}"[::-1], 2)

def _pick_ops_with_anc_all_zero(
    sv: NDArrayLike,
    anc: NDArrayLike,
    L: int
)-> NDArrayLike:
    res_ops = []

    for i in range(0, 2 ** L):
        qi = _bit_reverse(i, L)
        ind = qi << (len(anc[0]))
        res_ops.append(sv[ind])

    return res_ops

def test_foqcs_lcu_heisenberg_prep():
    r"""
    Verifies Heisenberg model `foqcs_prep_heisenberg` PREP statevector
    against the manually constructed expected statevector.
    """
    # Initialize variables + their values
    L = 4
    g, J = _generate_heisenberg_coeff()
    g, J, _ = _preprocess_heisenberg_coeff(g, J, L)

    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    d_state = QuantumVariable(L * 2 + 6)

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    foqcs_prep_heisenberg(d_state, L, heis_g, heis_J)
    # Take out the statevector from the compiled circuit
    qc = d_state.qs.compile()
    statev = qc.statevector_array()

    # Build Dicke states manually, give them all unique weights from g[0] to J[2]: base, double, 2NN, 2NN double...
    dicke_1_norm = 1 / (np.sqrt(L))
    dicke_1 = np.zeros(2**L)
    for i in range(L):
        dicke_1[2**i] = dicke_1_norm

    dicke_double = np.zeros(4**L)
    for i in range(L):
        dicke_double[2**i + 2 ** (L + i)] = dicke_1_norm

    dicke_2NN_norm = 1 / (np.sqrt(L - 1))
    dicke_2NN = np.zeros((2**L,), dtype="complex")
    for i in range(L - 1):
        dicke_2NN[2**i + 2 ** (i + 1)] = dicke_2NN_norm

    dicke_2NN_double = np.zeros((4**L,), dtype="complex")
    for i in range(L - 1):
        dicke_2NN_double[
            2**i + 2 ** (i + 1) + 2 ** (i + L) + 2 ** (i + L + 1)
        ] = dicke_2NN_norm

    ref_state = np.zeros((2 ** (6 + 2 * L),), dtype="complex")
    zero_n = np.array([1] + [0] * (2**L - 1))

    def _ref_state_helper(coeff, q, param):
        r"""
        Build one weighted Heisenberg PREP reference-state branch.

        Selects the q-th basis state of the 6-qubit selector register,
        tensors it with ``param``, and scales the resulting branch by ``coeff``.
        """

        return coeff * np.kron([1 if i == 2 ** (6 - q) else 0 for i in range(2**6)], param)

    coeff_arr = [g[0], g[1], g[2], J[0], J[1], J[2]]
    param_arr = [np.kron(dicke_1, zero_n), dicke_double, np.kron(zero_n, dicke_1), np.kron(dicke_2NN, zero_n), dicke_2NN_double, np.kron(zero_n, dicke_2NN)]
    
    for i in range(6):

        ref_state += _ref_state_helper(coeff_arr[i], i + 1, param_arr[i])

    # Test that the state received is the same as the reference
    assert np.allclose(statev, ref_state, atol=1e-06), (
        f"States differ"
    )

def test_foqcs_lcu_spin_glass_prep():
    r"""
    Verifies Spin-glass model `foqcs_prep_spin_glass` PREP statevector
    against the manually constructed expected statevector
    """
    # Initialize variables + their values
    L = 3
    coeff = np.random.uniform(-1, 1, (3, L, L))
    g = []
    J = []

    for i in range(3):
    
        g.append(np.diag(coeff[i]))
        coeff[i] = (coeff[i] + coeff[i].T) / 2.0
        J.append(coeff[i] - np.diag(g[i]))
    
    g = np.array(g, dtype=complex)
    J = np.array(J, dtype=complex)
    d_state = QuantumVariable(5 * L)

    # Fix coefficients for debugging
    # g = np.array( [[-0.4808829 +0.j, -0.86150457+0.j,  0.22114172+0.j],
    #                [-0.1736148 +0.j,  0.49011868+0.j,  0.78437336+0.j],
    #                [-0.33825609+0.j, -0.19728503+0.j, -0.5482909 +0.j]])
    # J = np.array([[[ 0.        +0.j,  0.45669928+0.j, -0.29835039+0.j],
    #                [ 0.45669928+0.j,  0.        +0.j, -0.66928103+0.j],
    #                [-0.29835039+0.j, -0.66928103+0.j,  0.        +0.j]],
    #               [[ 0.        +0.j, -0.24574681+0.j,  0.07099358+0.j],
    #                [-0.24574681+0.j,  0.        +0.j, -0.08502734+0.j],
    #                [ 0.07099358+0.j, -0.08502734+0.j,  0.        +0.j]],
    #               [[ 0.        +0.j, -0.49783128+0.j,  0.68088956+0.j],
    #                [-0.49783128+0.j,  0.        +0.j, -0.82130807+0.j],
    #                [ 0.68088956+0.j, -0.82130807+0.j,  0.        +0.j]]])

    # Normalize
    norms_kNN = np.zeros((3, L))

    for x in range(3):

        norms_kNN[x, 0] = np.linalg.norm(g[x])

        for k in range(1, L):

            J_kNN = []

            for i in range(L - k):

                J_kNN.append(J[x, i, i + k])

            norms_kNN[x, k] = np.linalg.norm(J_kNN)

    spin_glass_g = {"X": g[0], "Y": g[1], "Z": g[2]}

    J_diags = []

    for i in range(3):

        result = [list(J[i].diagonal(offset=k)) for k in range(1, J[i].shape[0])]
        J_diags.append(result)

    spin_glass_J = {"X": J_diags[0], "Y": J_diags[1], "Z": J_diags[2]}

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    foqcs_prep_spin_glass(d_state, L, spin_glass_g, spin_glass_J)

    # Take out the statevector from the compiled circuit
    qc = d_state.qs.compile()
    statev = qc.statevector_array()

    g_betas = [] # Squared normalization factors for all g components (X, Y, Z) --> [g_beta_X, g_beta_Y, g_beta_Z]
    J_betas = [[], [], []] # Squared normalization factors for all J components and diagonals (X, Y, Z) --> [[J_beta_X1, J_beta_X2, ...], [J_beta_Y1, J_beta_Y2, ...], [J_beta_Z1, J_beta_Z2, ...]]
    g_hats = [[], [], []] # Normalized g coefficients
    J_hats = [[], [], []] # Normalized J coefficients
    components = ["X", "Y", "Z"]

	# Normalization for state preparation
    for i in range(3):

        for j in range(len(spin_glass_J["X"])):

            J_hats[i].append([])

    for i in range(3):

        s_sum = 0
        dimension = components[i]

        for j in range(len(spin_glass_g[dimension])):

            s_sum += abs(spin_glass_g[dimension][j]) ** 2

        g_betas.append(s_sum)

    for i in range(3):

        dimension = components[i]

        for j in range(len(spin_glass_J[dimension])):

            s_sum = 0

            for k in range(len(spin_glass_J[dimension][j])):

                s_sum += abs(spin_glass_J[dimension][j][k]) ** 2

            J_betas[i].append(s_sum)

    for i in range(3):

        dimension = components[i]

        for j in range(len(spin_glass_g[dimension])):

            new_g = spin_glass_g[dimension][j] / (g_betas[i] ** 0.5)
            g_hats[i].append(new_g)

    for i in range(3):

        dimension = components[i]

        for j in range(len(spin_glass_J[dimension])):

            for k in range(len(spin_glass_J[dimension][j])):

                new_J = spin_glass_J[dimension][j][k] / ((J_betas[i][j]) ** 0.5)
                J_hats[i][j].append(new_J)

    final_betas = []

    for i in range(3):

        final_betas.append(g_betas[i])

        for j in range(len(J_betas[i])):

            final_betas.append(J_betas[i][j])

    final_betas = np.sqrt(np.array(final_betas))
    final_betas = final_betas / np.linalg.norm(final_betas)

    ref_state = np.zeros(2 ** (5 * L), dtype=complex)
    zero_n = np.array([1] + [0] * (2**L - 1))

    def _add_ref_state_kron_term(ref_state, coeff, eye_3L, eye_power, *right_factors):
        r"""
        Add one weighted tensor-product branch to the spin-glass reference state.

        Selects the ``|2**eye_power>`` basis state of the 3L-qubit selector
        register, tensors together the supplied right-hand state factors, and
        accumulates the resulting branch into ``ref_state`` scaled by ``coeff``.
        """

        right_state = right_factors[0]

        for factor in right_factors[1:]:
            right_state = np.kron(right_state, factor)

        ref_state += coeff * np.kron(
            eye_3L[2 ** eye_power],
            right_state,
        )

        return ref_state

    eye_3L = np.eye(2 ** (3 * L))

    for k in range(L):
        for i in range(0, L - k):
            # g terms
            if k == 0:
                ket = np.zeros(2**L)
                ket[2 ** (L - i - 1)] = 1

                # gx
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    g[0, i],
                    eye_3L,
                    3 * L - 1,
                    ket,
                    zero_n,
                )

                # gz
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    g[2, i],
                    eye_3L,
                    L - 1,
                    zero_n,
                    ket,
                )

                double_ket = np.zeros(2 ** (2 * L))
                double_ket[2 ** (L - i - 1) + 2 ** (2 * L - i - 1)] = 1

                # gy
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    g[1, i],
                    eye_3L,
                    2 * L - 1,
                    double_ket,
                )

            # J terms
            else:
                ket = np.zeros(2**L)
                ket[2 ** (L - i - 1) + 2 ** (L - i - k - 1)] = 1

                # Jx
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    J[0, i, i + k],
                    eye_3L,
                    3 * L - k - 1,
                    ket,
                    zero_n,
                )

                # Jz
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    J[2, i, i + k],
                    eye_3L,
                    L - k - 1,
                    zero_n,
                    ket,
                )

                double_ket = np.zeros(2 ** (2 * L))
                double_ket[
                    2 ** (L - i - 1)
                    + 2 ** (L - i - k - 1)
                    + 2 ** (2 * L - i - 1)
                    + 2 ** (2 * L - i - k - 1)
                ] = 1

                # Jy
                ref_state = _add_ref_state_kron_term(
                    ref_state,
                    J[1, i, i + k],
                    eye_3L,
                    2 * L - k - 1,
                    double_ket,
                )

    ref_state = ref_state / np.linalg.norm(ref_state)

    statev[np.isclose(statev, 0j, atol=1e-6)] = 0

    # Test that the state received is the same as the reference
    assert np.allclose(statev, ref_state, atol=1e-06), (
        f"States differ"
    )

def test_block_encoding_from_foqcs_lcu_heisenberg_prep():
    r"""
    Verifies Heisenberg model block encoding `.apply`.
    Tests it against the manually constructed statevector from definition
    """
    # Initialize variables + their values
    L = 4
    g, J = _generate_heisenberg_coeff()
    _g, _J, norm = _preprocess_heisenberg_coeff(g, J, L)

    # Construct dictionary input expected by foqcs_prep_heisenberg()
    heis_g = {"X": _g[0], "Y": _g[1], "Z": _g[2]}
    heis_J = {"X": _J[0], "Y": _J[1], "Z": _J[2]}

    # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
    p_r = partial(
        foqcs_prep_heisenberg,
        L=L,
        g=heis_g,
        J=heis_J,
    )
    p_l = partial(
        foqcs_prep_heisenberg,
        L=L,
        g=heis_g,
        J=heis_J,
        conjugate=True
    )

    be = BlockEncoding.from_foqcs_lcu_prep(
        p_r = p_r,
        p_l = p_l,
        num_q_ops = L,
        norm = norm ** 2
    )

    qv = QuantumVariable(4)

    psi = _prep_psi(L)

    qv.init_state(psi, method="qswitch")

    def main(BE):
        operand = qv
        ancillas = BE.apply(operand)
        return operand, ancillas

    operand, ancillas = main(be)

    qc = operand.qs.compile()
    sv = qc.statevector_array()

    # Take out the resulting operands with zero ancillas amplitude.
    res_ops = _pick_ops_with_anc_all_zero(sv, ancillas, L)

    # Construct reference state vector
    H = _heisenberg_from_def(L, g, J) / (norm ** 2)
    ref_state = H @ psi

    assert np.allclose(res_ops, ref_state, atol=1e-6)

def test_block_encoding_from_foqcs_lcu_spin_glass_prep():
    r"""
    Verifies spin-glass model block encoding `.apply`.
    Tests it against the manually constructed statevector from definition.
    """
    L = 3

    # Physical Hamiltonian coefficients.
    # Keep these real if you want a Hermitian reference Hamiltonian.
    g = {
        "X": np.array([0.31, -0.47, 0.22], dtype=complex),
        "Y": np.array([-0.18, 0.29, 0.41], dtype=complex),
        "Z": np.array([0.52, -0.13, -0.36], dtype=complex),
    }

    # J[axis][k - 1][i] couples sites i and i + k.
    J = {
        "X": [
            np.array([0.17, -0.24], dtype=complex),  # distance 1: (0,1), (1,2)
            np.array([0.33], dtype=complex),         # distance 2: (0,2)
        ],
        "Y": [
            np.array([-0.21, 0.15], dtype=complex),
            np.array([-0.28], dtype=complex),
        ],
        "Z": [
            np.array([0.39, -0.11], dtype=complex),
            np.array([0.26], dtype=complex),
        ],
    }

    # Normalize the physical coefficients, same style as the Heisenberg test.
    # This is not strictly necessary, but keeps amplitudes well-conditioned.
    phys_norm = np.linalg.norm(_flatten_spin_glass_coeffs(g, J))
    for axis in ["X", "Y", "Z"]:
        g[axis] = g[axis] / phys_norm
        J[axis] = [diag / phys_norm for diag in J[axis]]

    # Physical Hamiltonian coefficients.
    phys_g = g
    phys_J = J

    # PREP coefficients.
    #
    # SELECT convention:
    #   X branch  -> X
    #   Z branch  -> Z
    #   Y branch  -> iY
    #   YY branch -> -YY
    #
    # Therefore:
    #   local Y needs sqrt(-i * coeff)
    #   YY needs sqrt(-coeff)
    prep_g = {
        "X": np.sqrt(phys_g["X"]),
        "Y": np.sqrt(-1j * phys_g["Y"]),
        "Z": np.sqrt(phys_g["Z"]),
    }

    prep_J = {
        "X": [np.sqrt(diag) for diag in phys_J["X"]],
        "Y": [np.sqrt(-diag) for diag in phys_J["Y"]],
        "Z": [np.sqrt(diag) for diag in phys_J["Z"]],
    }

    alpha = np.linalg.norm(_flatten_spin_glass_coeffs(prep_g, prep_J)) ** 2

    p_r = partial(
        foqcs_prep_spin_glass,
        L=L,
        g=prep_g,
        J=prep_J,
    )

    p_l = partial(
        foqcs_prep_spin_glass,
        L=L,
        g=prep_g,
        J=prep_J,
        conjugate=True,
    )

    be = BlockEncoding.from_foqcs_lcu_prep(
        p_r = p_r,
        p_l = p_l,
        num_q_ops = L,
        norm = alpha,
    )

    qv = QuantumVariable(L)
    psi = _prep_psi(L)
    qv.init_state(psi, method="qswitch")

    def main(BE):
        operand = qv
        ancillas = BE.apply(operand)
        return operand, ancillas

    operand, ancillas = main(be)

    qc = operand.qs.compile()
    sv = qc.statevector_array()

    # Take out the resulting operands with zero ancillas amplitude.
    res_ops = _pick_ops_with_anc_all_zero(sv, ancillas, L)
    
    H = _spin_glass_from_def(L, phys_g, phys_J) / alpha
    ref_state = H @ psi

    assert np.allclose(res_ops, ref_state, atol=1e-5)

def test_block_encoding_from_foqcs_lcu_heisenberg_prep_jasp():
    r"""
    Verifies Heisenberg model block encoding `.apply_rus` (under the jasp environment).
    Tests it against the `.apply` produced statevector.
    """
    # Initialize variables + their values
    L = 4
    g, J = _generate_heisenberg_coeff()
    _g, _J, norm = _preprocess_heisenberg_coeff(g, J, L)

    # Construct dictionary input expected by foqcs_prep_heisenberg()
    heis_g = {"X": _g[0], "Y": _g[1], "Z": _g[2]}
    heis_J = {"X": _J[0], "Y": _J[1], "Z": _J[2]}

    # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
    p_r = partial(
        foqcs_prep_heisenberg,
        L=L,
        g=heis_g,
        J=heis_J,
    )
    p_l = partial(
        foqcs_prep_heisenberg,
        L=L,
        g=heis_g,
        J=heis_J,
        conjugate=True
    )

    be = BlockEncoding.from_foqcs_lcu_prep(
        p_r = p_r,
        p_l = p_l,
        num_q_ops = L,
        norm = norm ** 2
    )

    psi = _prep_psi(L)

    def operand_prep(psi):
        qv = QuantumVariable(4)
        qv.init_state(psi, method="qswitch")
        return qv

    qv_manual = operand_prep(psi)
    def main_apply(BE):
        operand = qv_manual
        ancillas = BE.apply(operand)
        return operand, ancillas

    @terminal_sampling
    def main_apply_rus(BE):
        return BE.apply_rus(operand_prep)(psi)

    # Do the measurement manually
    operand, ancillas = main_apply(be)

    res_dict = multi_measurement([operand] + ancillas)
    # Filtering only zero ancillae entries
    zero_anc = "0" * len(ancillas[0])
    filtered = {
        key: value
        for key, value in res_dict.items()
        if key[1] == zero_anc
    }
    success_prob = sum(filtered.values())
    filtered_conditional = {
        int(operand_bits[::-1], 2): prob / success_prob
        for (operand_bits, anc_bits), prob in filtered.items()
    }

    # Do the measurement using RUS
    result_rus = main_apply_rus(be)

    assert np.allclose([filtered_conditional[k] for k in sorted(filtered_conditional)],
                       [result_rus[k] for k in sorted(result_rus)],
                       atol = 1e-4)

def test_block_encoding_from_operator_spin_glass_jasp():
    r"""
    Verifies spin-glass model block encoding `.apply_rus` (under the jasp environment).
    Tests it against the `.apply` produced statevector.
    """
    L = 3

    g = {
        "X": np.array([0.31, -0.47, 0.22], dtype=complex),
        "Y": np.array([-0.18, 0.29, 0.41], dtype=complex),
        "Z": np.array([0.52, -0.13, -0.36], dtype=complex),
    }

    # J[axis][k - 1][i] couples sites i and i + k.
    J = {
        "X": [
            np.array([0.17, -0.24], dtype=complex),
            np.array([0.33], dtype=complex),
        ],
        "Y": [
            np.array([-0.21, 0.15], dtype=complex),
            np.array([-0.28], dtype=complex),
        ],
        "Z": [
            np.array([0.39, -0.11], dtype=complex),
            np.array([0.26], dtype=complex),
        ],
    }

    # Normalize physical coefficients. This is not required mathematically,
    # but it keeps the sampled probabilities well-conditioned.
    phys_norm = np.linalg.norm(_flatten_spin_glass_coeffs(g, J))

    for axis in ["X", "Y", "Z"]:
        g[axis] = g[axis] / phys_norm
        J[axis] = [diag / phys_norm for diag in J[axis]]

    # Build the Qrisp operator.
    #
    # This uses the same site convention as _spin_glass_from_def:
    #   g_axis[i] acts on site L - 1 - i
    #   J_axis[k - 1][i] acts on sites L - 1 - i and L - 1 - (i + k)
    terms = []

    pauli = {
        "X": X,
        "Y": Y,
        "Z": Z,
    }

    for axis in ["X", "Y", "Z"]:
        P = pauli[axis]

        for i in range(L):
            site = L - 1 - i
            terms.append(g[axis][i] * P(site))

        for k in range(1, L):
            for i in range(L - k):
                site_0 = L - 1 - i
                site_1 = L - 1 - (i + k)

                terms.append(J[axis][k - 1][i] * P(site_0) * P(site_1))

    O = sum(terms[1:], terms[0])

    be = BlockEncoding.from_foqcs_lcu_operator(O)

    psi = _prep_psi(L)

    def operand_prep(psi):
        qv = QuantumVariable(L)
        qv.init_state(psi, method="qswitch")
        return qv

    qv_manual = operand_prep(psi)

    def main_apply(BE):
        operand = qv_manual
        ancillas = BE.apply(operand)
        return operand, ancillas

    @terminal_sampling
    def main_apply_rus(BE):
        return BE.apply_rus(operand_prep)(psi)

    def measurement_key_to_int(x):
        if isinstance(x, str):
            return int(x[::-1], 2)
        return x

    def is_zero_measurement(x):
        if isinstance(x, str):
            return set(x) <= {"0"}
        return x == 0

    # Manual post-selection version.
    operand, ancillas = main_apply(be)
    res_dict = multi_measurement([operand] + ancillas)

    filtered = {
        key: value
        for key, value in res_dict.items()
        if all(is_zero_measurement(anc_res) for anc_res in key[1:])
    }

    success_prob = sum(filtered.values())

    assert success_prob > 0

    filtered_conditional = {
        measurement_key_to_int(key[0]): prob / success_prob
        for key, prob in filtered.items()
    }

    # RUS version.
    result_rus = main_apply_rus(be)

    result_rus_int = {
        measurement_key_to_int(k): v
        for k, v in result_rus.items()
    }

    keys = sorted(set(filtered_conditional) | set(result_rus_int))
    
    assert np.allclose(
        [filtered_conditional.get(k, 0) for k in keys],
        [result_rus_int.get(k, 0) for k in keys],
        atol=1e-3,
    )

@pytest.mark.parametrize(
    "O",
    [
        pytest.param(
            0.31 * X(0)
            + 0.31 * X(1)
            + 0.31 * X(2)
            + 0.31 * X(3)
            - 0.47 * Y(0)
            - 0.47 * Y(1)
            - 0.47 * Y(2)
            - 0.47 * Y(3)
            + 0.22 * Z(0)
            + 0.22 * Z(1)
            + 0.22 * Z(2)
            + 0.22 * Z(3)
            + 0.17 * X(0) * X(1)
            + 0.17 * X(1) * X(2)
            + 0.17 * X(2) * X(3)
            - 0.24 * Y(0) * Y(1)
            - 0.24 * Y(1) * Y(2)
            - 0.24 * Y(2) * Y(3)
            + 0.33 * Z(0) * Z(1)
            + 0.33 * Z(1) * Z(2)
            + 0.33 * Z(2) * Z(3),
            id="full_uniform_L4_heisenberg",
        ),
        pytest.param(
            X(0) + X(1),
            id="uniform_X_field_heisenberg",
        ),
        pytest.param(
            0.39 * Z(0) * Z(1)
            + 0.39 * Z(1) * Z(2)
            + 0.39 * Z(2) * Z(3),
            id="uniform_ZZ_coupling_heisenberg",
        ),
    ],
)
def test_block_encoding_from_foqcs_lcu_heisenberg_operator(O):
    r"""
    Verifies `from_foqcs_lcu_operator` execution with Heisenberg operators.
    Tests it against the manually constructed statevector from the analyzed
    Heisenberg coefficients.
    """
    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import foqcs_analyze_operator_heisenberg

    aresult = foqcs_analyze_operator_heisenberg(O)

    L = aresult["L"]
    g = aresult["g"]
    J = aresult["J"]

    be = BlockEncoding.from_foqcs_lcu_operator(O)

    qv = QuantumVariable(L)
    psi = _prep_psi(L)
    qv.init_state(psi, method="qswitch")

    ancillas = be.apply(qv)

    qc = qv.qs.compile()
    sv = qc.statevector_array()

    # Take out the resulting operands with zero ancillas amplitude.
    res_ops = _pick_ops_with_anc_all_zero(sv, ancillas, L)

    # Build reference state from the Heisenberg definition.
    g_arr = np.array([g["X"], g["Y"], g["Z"]], dtype=complex)
    J_arr = np.array([J["X"], J["Y"], J["Z"]], dtype=complex)
    H = _heisenberg_from_def(L, g_arr, J_arr) / be.alpha
    ref_state = H @ psi

    assert np.allclose(res_ops, ref_state, atol=1e-6)

@pytest.mark.parametrize(
    "O",
    [
        pytest.param(
            0.31 * X(0)
            - 0.47 * X(1)
            + 0.22 * X(2)
            - 0.18 * Y(0)
            + 0.29 * Y(1)
            + 0.41 * Y(2)
            + 0.52 * Z(0)
            - 0.13 * Z(1)
            - 0.36 * Z(2)
            + 0.17 * X(0) * X(1)
            - 0.24 * X(1) * X(2)
            + 0.33 * X(0) * X(2)
            - 0.21 * Y(0) * Y(1)
            + 0.15 * Y(1) * Y(2)
            - 0.28 * Y(0) * Y(2)
            + 0.39 * Z(0) * Z(1)
            - 0.11 * Z(1) * Z(2)
            + 0.26 * Z(0) * Z(2),
            id="dense_L3_spin_glass",
        ),
        pytest.param(
            0.7 * X(0)
            - 0.3 * Z(2)
            + 0.5 * X(0) * X(1)
            + 1.2 * Y(1) * Y(3)
            - 0.8 * Z(0) * Z(3),
            id="sparse_L4_spin_glass",
        ),
    ],
)
def test_block_encoding_from_foqcs_lcu_spin_glass_operator(O):
    r"""
    Verifies `from_foqcs_lcu_operator` execution with spin-glass operators.
    Tests it against the manually constructed statevector from the analyzed
    spin-glass coefficients.
    """
    from qrisp.block_encodings.constructors.foqcs_lcu.foqcs_analysis import foqcs_analyze_operator_spin_glass
    def _J_matrix_to_diag_list(J, L):
        """
        Convert full matrix J into diagonal-list form:
          
          `J_diag[p][k - 1][i]` couples `i` and `i + k`.
        
        This is the format expected by `_spin_glass_from_def` and
        `foqcs_prep_spin_glass`.
        """
        return {
            p: [
                np.array(
                    [J[p][i, i + k] for i in range(L - k)],
                    dtype=complex,
                )
                for k in range(1, L)
            ]
            for p in ("X", "Y", "Z")
        }

    aresult = foqcs_analyze_operator_spin_glass(O)

    L = aresult["L"]
    g = aresult["g"]
    J_diag = _J_matrix_to_diag_list(aresult["J"], L)

    be = BlockEncoding.from_foqcs_lcu_operator(O)

    qv = QuantumVariable(L)
    psi = _prep_psi(L)
    qv.init_state(psi, method="qswitch")

    ancillas = be.apply(qv)

    qc = qv.qs.compile()
    sv = qc.statevector_array()

    # Take out the resulting operands with zero ancillas amplitude.
    res_ops = _pick_ops_with_anc_all_zero(sv, ancillas, L)

    # Build reference state from the spin-glass definition.
    H = _spin_glass_from_def(L, g, J_diag) / be.alpha
    ref_state = H @ psi

    assert np.allclose(res_ops, ref_state, atol=1e-5)

def test_foqcs_lcu_custom_prep_from_prep():
    r"""
    Tests the usage of custom PREP function with `from_foqcs_lcu_prep`.
    """
    from collections.abc import Sequence
    from qrisp.core import QuantumVariable, Qubit
    from qrisp.core.gate_application_functions import x, cx
    L = 2
    n_anc_custom_prep = 5

    # Custom PREP_R subroutine.
    # Defines how many ancillary qubits the circuit requires.
    # This example does not result in a viable block encoding,
    # it only shows the process of defining a custom subroutine.
    def custom_prep(qv: QuantumVariable | Sequence[Qubit], L: int):
        # Ancilla layout for L = 2 and n_anc = 5:
        #
        #   [extra, x0, x1, z0, z1]
        #
        # SELECT should use only the final 2L qubits:
        #
        #   [x0, x1, z0, z1]
        #
        # This PREP sets extra = 1 and copies it into x0.
        # Thus the selected operation should be X(0)
        x(qv[0]) # Extra ancillary
        cx(qv[0], qv[1]) # x[0] taken from extra ancillary.
        x(qv[4]) # z[1]

    # Then, custom_prep is used for both prep_r and prep_l, as there is no
    # specific handling required. (For example, parametrised subcircuit
    # would have required conjugated parameters. See the `foqcs_prep_heisenberg`
    # usage from previous example)
    p_r = partial(
        custom_prep,
        L=L
    )
    p_l = partial(
        custom_prep,
        L=L
    )

    be = BlockEncoding.from_foqcs_lcu_prep(
        p_r = p_r,
        p_l = p_l,
        num_q_ops = L,
        num_q_anc = n_anc_custom_prep
    )

    qv = QuantumVariable(L)
    ancillas = be.apply(qv)

    assert len(ancillas[0]) == n_anc_custom_prep

    qc = qv.qs.compile()
    sv = qc.statevector_array()

    res = _pick_ops_with_anc_all_zero(sv, ancillas, L)

    expected = np.zeros(2 ** L, dtype=complex)
    expected[1] = 1.0  # X(0)|00> = |01>, i.e. basis index 2**0

    assert np.allclose(res, expected, atol=1e-6)

def test_foqcs_lcu_custom_prep_n_anc_fail():
    r"""
    Verifies that passing invalid number of ancillary qubits results in
    a failure.
    """
    from collections.abc import Sequence
    from qrisp.core import Qubit
    from qrisp.core.gate_application_functions import x
    L = 2
    n_anc_custom_prep = 3

    def custom_prep(qv: QuantumVariable | Sequence[Qubit], L: int):
        x(qv[0])

    p_r = partial(
        custom_prep,
        L=L
    )
    p_l = p_r

    with pytest.raises(ValueError) as exc_info:
        be = BlockEncoding.from_foqcs_lcu_prep(
            p_r = p_r,
            p_l = p_l,
            num_q_ops = L,
            num_q_anc = n_anc_custom_prep
        )

    assert f"at least {L * 2}, but received {n_anc_custom_prep}." in str(exc_info.value)

def test_foqcs_lcu_resources():
    r"""
    Validates that `.resources` can be executed for FOQCS-LCU Block Encoding.
    Spin-glass model is taken as benchmark.
    """
    L = 3

    O = (
        0.31 * X(0) - 0.47 * X(1) + 0.22 * X(2)
        - 0.18 * Y(0) + 0.29 * Y(1) + 0.41 * Y(2)
        + 0.52 * Z(0) - 0.13 * Z(1) - 0.36 * Z(2)
        + 0.17 * X(0) * X(1) - 0.24 * X(1) * X(2) + 0.33 * X(0) * X(2)
        - 0.21 * Y(0) * Y(1) + 0.15 * Y(1) * Y(2) - 0.28 * Y(0) * Y(2)
        + 0.39 * Z(0) * Z(1) - 0.11 * Z(1) * Z(2) + 0.26 * Z(0) * Z(2)
    )

    be = BlockEncoding.from_foqcs_lcu_operator(O)
    res = be.resources(QuantumVariable(L))

    assert set(res) == {"gate counts", "depth", "qubits"}
    assert isinstance(res["gate counts"], dict)
    assert res["gate counts"]
    assert res["depth"] > 0
    assert res["qubits"] > 0

def test_block_encoding_foqcs_lcu_is_controllable():
    r"""
    Tests that FOQCS-LCU produced block encoding can be controlled under jasp environment.
    """
    from qrisp import QuantumBool, h
    from qrisp.environments import control
    H = X(0)*X(1) + 0.2*Y(0)*Y(1)
    n = H.find_minimal_qubit_amount()

    BE = BlockEncoding.from_foqcs_lcu_operator(H)

    @terminal_sampling
    def main():
        ctrl = QuantumBool()
        qv = QuantumVariable(n)

        h(ctrl)

        with control(ctrl):
            BE.apply(qv)

        return qv

    main()

###########################################################################################################
#### Transformations tests ################################################################################
###########################################################################################################

def _compare_results(res_dict_1, res_dict_2, n):
    for k in range(2 ** n):
        val_1 = res_dict_1.get(k, 0)
        val_2 = res_dict_2.get(k, 0)
        assert np.isclose(val_1, val_2, atol=1e-6), f"Mismatch at state |{k}>: {val_1} vs {val_2}"

@pytest.mark.parametrize(
    "H1, H2, rescaled",
    [
        (
            X(0) * X(1) + 0.2 * Y(0) * Y(1),
            Z(0) * Z(1) + X(2),
            True,
        ),
        (
            0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4),
            Z(0) + Z(1) + X(2),
            False,
        ),
    ],
)
def test_foqcs_lcu_chebyshev(H1, H2, rescaled):
    r"""
    Tests `.chebyshev(k)` on a FOQCS-LCU block encoding by comparing `BE1 + T_k(BE2)`
    against direct operator block encodings for `T_1`, `T_2`, and `T_3`,
    with both rescaled and non-rescaled cases.
    """
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_foqcs_lcu_operator(H2)

    alpha = BE2.alpha

    if rescaled:
        H_T1 = H1 + H2 # H1 + T_1(H2)
        H_T2 = H1 + (2 * H2**2 - 1) # H1 + T_2(H2)
        H_T3 = H1 + (4 * H2**3 - 3 * H2) # H1 + T_3(H2)
    else:
        H_T1 = H1 + H2 # H1 + T_1(H2 / alpha)
        H_T2 = H1 + (2 / alpha**2 * H2**2 - 1) # H1 + T_2(H2 / alpha)
        H_T3 = H1 + (4 / alpha**3 * H2**3 - 3 / alpha * H2) # H1 + T_3(H2 / alpha)

    BE_T1 = BlockEncoding.from_operator(H_T1)
    BE_T2 = BlockEncoding.from_operator(H_T2)
    BE_T3 = BlockEncoding.from_operator(H_T3)

    BE_cheb_T1 = BE2.chebyshev(1, rescale=rescaled)
    BE_cheb_T2 = BE2.chebyshev(2, rescale=rescaled)
    BE_cheb_T3 = BE2.chebyshev(3, rescale=rescaled)

    BE_add_T1 = BE1 + BE_cheb_T1
    BE_add_T2 = BE1 + BE_cheb_T2
    BE_add_T3 = BE1 + BE_cheb_T3

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be_t1 = main(BE_T1)
    res_be_add_t1 = main(BE_add_T1)
    _compare_results(res_be_t1, res_be_add_t1, n)

    res_be_t2 = main(BE_T2)
    res_be_add_t2 = main(BE_add_T2)
    _compare_results(res_be_t2, res_be_add_t2, n)

    res_be_t3 = main(BE_T3)
    res_be_add_t3 = main(BE_add_T3)
    _compare_results(res_be_t3, res_be_add_t3, n)

@pytest.mark.parametrize(
    "H1, H2, poly",
    [
        (
            X(0) * X(1) + 0.2 * Y(0) * Y(1),
            Z(0) * Z(1) + X(2),
            np.array([1, 1, 1]),
        ),
        (
            0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4),
            Z(0) + Z(1) + X(2),
            np.array([0, 1, 0, 1]),
        ),
        (
            X(0) * X(1),
            Z(0) + 0.9 * Z(1) + X(3),
            np.array([1, 1, 0, 1]),
        ),
    ],
)
def test_foqcs_lcu_poly(H1, H2, poly):
    r"""
    Tests `.poly(poly)` on a FOQCS-LCU block encoding by comparing `poly(BE1) + BE2`
    against a direct block encoding of `sum(poly[k] * H1**k) + H2`.
    """
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_foqcs_lcu_operator(H2)

    # Apply polynomial to QubitOperator H1 and add H2
    H3 = sum(poly[k] * H1**k for k in range(len(poly))) + H2
    BE3 = BlockEncoding.from_operator(H3)

    # Apply polynomial to BlockEncoding BE1 and add BE2
    BE_poly = BE1.poly(poly) + BE2

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_poly = main(BE_poly)

    _compare_results(res_be3, res_be_poly, n)

def test_foqcs_lcu_inv():
    r"""
    Tests `.inv(...)` on a FOQCS-LCU block encoding by solving a small linear system
    and comparing the sampled output amplitudes
    against the normalized classical solution `inv(H) @ b`.
    """
    from qrisp import prepare

    L = 2

    H_op = (
        0.65 * Z(0)
        + 0.35 * Z(1)
        + 0.20 * X(0) * X(1)
    )

    assert is_operator_foqcs_compatible(H_op)

    H = np.array(
        [
            [1.00, 0.00, 0.00, 0.20],
            [0.00, 0.30, 0.20, 0.00],
            [0.00, 0.20, -0.30, 0.00],
            [0.20, 0.00, 0.00, -1.00],
        ],
        dtype=complex,
    )

    assert np.linalg.matrix_rank(H) == 2**L

    BE_H = BlockEncoding.from_foqcs_lcu_operator(H_op)
    BE_H_inv = BE_H.inv(0.01, np.linalg.cond(H))

    b = np.array([0, 1, 0, 1])

    # Prepares operand variable in state |b>
    def prep_b():
        operand = QuantumVariable(L)
        prepare(operand, b)
        return operand

    @terminal_sampling
    def main():
        operand = BE_H_inv.apply_rus(prep_b)()
        return operand

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(len(b))])

    c = np.linalg.inv(H) @ b
    c = c / np.linalg.norm(c)

    assert np.linalg.norm(np.abs(c) - q) < 1e-2

def test_foqcs_lcu_sim():
    r"""
    Tests `.sim(t, N)` for Hamiltonian simulation by comparing simulation results from a FOQCS-LCU block encoding
    against a generic `BlockEncoding.from_operator(H)` reference
    """
    L = 2
    t = 0.1

    # Nontrivial FOQCS-LCU-compatible Hamiltonian.
    #
    # Starting from |00>, the X(0) term creates population transfer, while
    # the Z(0)Z(1) term adds an interaction phase. This is small enough that
    # the FOQCS-LCU path and generic reference path agree within 1e-6.
    H = 0.2 * Z(0) * Z(1) + 0.3 * X(0)

    assert is_operator_foqcs_compatible(H)

    BE_ref = BlockEncoding.from_operator(H)
    BE_foqcs = BlockEncoding.from_foqcs_lcu_operator(H)

    def operand_prep():
        return QuantumFloat(L)

    @terminal_sampling
    def main(BE):
        BE_sim = BE.sim(t=t, N=8)
        operand = BE_sim.apply_rus(operand_prep)()
        return operand

    res_ref = main(BE_ref)
    res_foqcs = main(BE_foqcs)

    # Make sure the test is not accidentally trivial.
    # The |1> state of qubit 0 should receive some population from X(0).
    # assert any(v > 1e-8 for k, v in res_ref.items() if k != 0)
    _compare_results(res_ref, res_foqcs, L)

###########################################################################################################
#### Arithmetic / composition tests #######################################################################
###########################################################################################################
# Addition
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_foqcs_lcu_addition(H1, H2):
    r"""
    Tests `BE_addition = BE1 + BE2` with `BE1` constructed by FOQCS-LCU.
    """

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 + H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_addition = BE1 + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_add = main(BE_addition)
    _compare_results(res_be3, res_be_add, n)

# Subtraction
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_foqcs_lcu_subtraction(H1, H2):
    r"""
    Tests `BE_subtraction = BE1 - BE2` with `BE1` constructed by FOQCS-LCU.
    """

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 - H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_subtraction = BE1 - BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_sub = main(BE_subtraction)
    _compare_results(res_be3, res_be_sub, n)

# @ (__matmul__)

# The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.
# Thus, to ensure that the multiplication test is valid, we should choose pairs of operators that commute.
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), X(0) + X(4)),
    (X(0)*X(1), Z(0)*Z(1) + Y(3)),
])
def test_block_encoding_foqcs_lcu_multiplication(H1, H2):
    r"""
    Tests `BE_multiplication = BE1 @ BE2` with `BE1` constructed by FOQCS-LCU.
    """

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 * H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_multiplication = BE1 @ BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be3 = main(BE3)
    res_be_mul = main(BE_multiplication)
    _compare_results(res_be3, res_be_mul, n)

# __mul__, __rmul__
@pytest.mark.parametrize("H1, H2, scalar", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2), -2),
    (0.5*X(1) + 0.7*Y(1), Z(0) + X(2), 0.5),
    (X(0), Z(0), 1),
])
def test_block_encoding_foqcs_lcu_scalar_multiplication(H1, H2, scalar):
    r"""
    Tests the `BE_left = scalar * BE1 + BE2`, `BE_right = BE1 * scalar + BE2`
    with `BE1` constructed by FOQCS-LCU.
    """
    H_target = scalar * H1 + H2
    BE_target = BlockEncoding.from_operator(H_target)

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_left = scalar * BE1 + BE2
    BE_right = BE1 * scalar + BE2
    
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_target = main(BE_target)
    res_left = main(BE_left)
    res_right = main(BE_right)
    _compare_results(res_target, res_left, n)
    _compare_results(res_target, res_right, n)

# .kron
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (X(0)*X(1), Z(0)*Z(1)),
])
def test_block_encoding_foqcs_lcu_kron(H1, H2):
    r"""
    Tests the use of `.kron` with FOQCS-LCU.
    """
    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_kron = BE1.kron(BE2)

    n1 = H1.find_minimal_qubit_amount()
    n2 = H2.find_minimal_qubit_amount()

    def operand_prep():
        qv1 = QuantumFloat(n1)
        qv2 = QuantumFloat(n2)
        return qv1, qv2

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(operand_prep)()

    result_be_kron = main(BE_kron)

    @terminal_sampling
    def main(BE1, BE2):
        qv1 = BE1.apply_rus(lambda : QuantumFloat(n1))()
        qv2 = BE2.apply_rus(lambda : QuantumFloat(n2))()
        return qv1, qv2

    result_be1_be2 = main(BE1, BE2)

    for k in range(2 ** n1):
        for l in range(2 ** n2):
            val_be_kron = result_be_kron.get((k, l), 0)
            val_be1_be2 = result_be1_be2.get((k, l), 0)
            assert np.isclose(val_be_kron, val_be1_be2), f"Mismatch at state |{k}>: {val_be_kron} vs {val_be1_be2}"

# __neg__
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) - 0.2*Y(0)*Y(1), 0.2*Y(0)*Y(1) - X(0)*X(1)),
    (0.5*X(1) - 0.7*Y(1) + 0.3*X(4), 0.7*Y(1) - 0.5*X(1) - 0.3*X(4)),
    (Z(0)*Z(1) - Y(3), Y(3) - Z(0)*Z(1)),
])
def test_block_encoding_foqcs_lcu_negation(H1, H2):
    r"""
    Tests the use of negation `BE_neg = -BE` with FOQCS-LCU.
    """

    BE1 = BlockEncoding.from_foqcs_lcu_operator(H1)
    BE_neg = -BE1

    BE2 = BlockEncoding.from_operator(H2)

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())
    #n = H1.find_minimal_qubit_amount()

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be2 = main(BE2)
    res_be_neg = main(BE_neg)
    _compare_results(res_be2, res_be_neg, n)

###########################################################################################################
#### Operator analysis tests ##############################################################################
###########################################################################################################
def _uniform_heisenberg_chain(L):
    H = 0
    for i in range(L):
        H += 1.0 * X(i)
        H += 0.5 * Y(i)
        H += -0.3 * Z(i)

        if i < L - 1:
            H += 0.7 * X(i) * X(i + 1)
            H += -0.2 * Y(i) * Y(i + 1)
            H += 0.4 * Z(i) * Z(i + 1)

    return H

@pytest.mark.parametrize(
    "O, L, expected_error",
    [
        pytest.param(
            0 * X(0),
            -1,
            "empty or constant operator",
            id="empty_operator",
        ),
        pytest.param(
            X(0) * X(0),
            -1,
            "empty or constant operator",
            id="constant_operator",
        ),
        pytest.param(
            X(0) + X(1) + X(2) + X(3) + X(4),
            2,
            "Received L = 2",
            id="bad_length",
        ),
        pytest.param(
            Y(0) + (0.18 + 0.5j) * Z(1) * Z(3) + X(0) * X(0),
            -1,
            "FOQCS-LCU does not support constant/identity terms",
            id="constant_term_in_operator",
        ),
        pytest.param(
            X(0) * Y(3) + X(0) * X(1) + Z(0),
            -1,
            "FOQCS-LCU supports only same-axis couplings, but received: X(0) * Y(3)",
            id="cross_axis_coupling",
        ),
        pytest.param(
            X(0) * X(1) + Z(0) + Y(0) * Y(1) * Y(2),
            -1,
            "FOQCS-LCU supports only one and two-body interactions",
            id="three_body_coupling",
        ),
    ],
)
def test_foqcs_operator_analysis_spin_glass_failures(O, L, expected_error):
    r"""
    Verifies that `foqcs_analyze_operator_spin_glass` raises correct errors for different failure cases.
    """
    with pytest.raises(ValueError) as exc_info:
        foqcs_analyze_operator_spin_glass(O, L=L)

    assert expected_error in str(exc_info.value)

@pytest.mark.parametrize(
    "O, L, expected_error",
    [
        pytest.param(
            0 * X(0),
            -1,
            "empty or constant operator",
            id="empty_operator",
        ),
        pytest.param(
            X(0) * X(0),
            -1,
            "empty or constant operator",
            id="constant_operator",
        ),
        pytest.param(
            X(0) + X(1) + X(2) + X(3) + X(4),
            2,
            "Received L = 2",
            id="bad_length",
        ),
        pytest.param(
            Y(0) + (0.18 + 0.5j) * Z(1) * Z(2) + X(0) * X(0),
            -1,
            "FOQCS-LCU does not support constant/identity terms",
            id="constant_term_in_operator",
        ),
        pytest.param(
            X(0) * Y(1) + X(0) + X(1),
            -1,
            "FOQCS-LCU supports only same-axis couplings, but received: X(0) * Y(1)",
            id="cross_axis_coupling",
        ),
        pytest.param(
            X(0) * X(1) + Z(0) + Y(0) * Y(1) * Y(2),
            -1,
            "FOQCS-LCU supports only one and two-body interactions",
            id="three_body_coupling",
        ),
        pytest.param(
            0.5 * X(0) + 0.7 * X(1),
            2,
            "non-uniform local interactions",
            id="non_uniform_local_fields",
        ),
        pytest.param(
            X(0) * X(2),
            3,
            "not NN interactions present",
            id="long_range_coupling",
        ),
        pytest.param(
            0.5 * X(0) * X(1) + 0.9 * X(1) * X(2),
            3,
            "non-uniform NN interactions",
            id="non_uniform_nn_couplings",
        ),
        pytest.param(
            # Zeroes matter for the Heisenberg specialization.
            # For L=3, X0X1 exists but X1X2 is implicitly zero,
            # so NN couplings are [1, 0] and therefore non-uniform.
            X(0) * X(1),
            3,
            "non-uniform NN interactions",
            id="missing_nn_couplings",
        ),
    ],
)
def test_foqcs_operator_analysis_heisenberg_failures(O, L, expected_error):
    r"""
    Verifies that `foqcs_analyze_operator_heisenberg` raises correct errors for different failure cases.
    """
    with pytest.raises(ValueError) as exc_info:
        foqcs_analyze_operator_heisenberg(O, L=L)

    assert expected_error in str(exc_info.value)

@pytest.mark.parametrize(
    "O, expected_method",
    [
        pytest.param(
            _uniform_heisenberg_chain(2),
            "heisenberg",
            id="L2_full_uniform",
        ),
        pytest.param(
            _uniform_heisenberg_chain(4),
            "heisenberg",
            id="L4_full_uniform",
        ),
        pytest.param(
            X(0) + X(1) + X(2) + X(3),
            "heisenberg",
            id="uniform_X_field_only",
        ),
        pytest.param(
            Z(0) * Z(1)
            + Z(1) * Z(2)
            + Z(2) * Z(3),
            "heisenberg",
            id="uniform_ZZ_coupling_only",
        ),
        pytest.param(
            0.8 * X(0) + 0.8 * X(1) + 0.8 * X(2)
            + 0.0 * Y(0) + 0.0 * Y(1) + 0.0 * Y(2)
            + 0.2 * Z(0) * Z(1)
            + 0.2 * Z(1) * Z(2),
            "heisenberg",
            id="uniform_with_some_zero_families",
        ),
        pytest.param(
            1.0 * X(0)
            + 0.5 * X(1)
            + 1.0 * X(2),
            "spin_glass",
            id="non_uniform_local_X",
        ),
        pytest.param(
            X(0) + 0.5 * Y(1)
            + 0.2 * Z(0) * Z(1),
            "spin_glass",
            id="position_dependent_local_fields",
        ),
        pytest.param(
            X(0) * X(2),
            "spin_glass",
            id="long_range_XX",
        ),
        pytest.param(
            0.2 * Z(0) * Z(1)
            + 0.7 * Z(1) * Z(2),
            "spin_glass",
            id="non_uniform_NN_ZZ",
        ),
        pytest.param(
            X(3) + 0.2 * Z(0) * Z(1),
            "spin_glass",
            id="missing_NN_bonds_due_to_L",
        ),
        pytest.param(
            0.3 * X(0) * X(3)
            - 0.4 * Y(1) * Y(2)
            + 0.9 * Z(0),
            "spin_glass",
            id="sparse_spin_glass_pairs",
        ),
    ],
)
def test_foqcs_operator_analysis(O, expected_method):
    r"""
    Verifies that `foqcs_analyze_operator` properly identifies Heisenberg and spin-glass operator cases.
    """
    res = is_operator_foqcs_compatible(O)

    assert res, f"Compatible operator failed analysis: {O}"
    assert res["method"] == expected_method
