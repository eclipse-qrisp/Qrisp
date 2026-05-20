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
from qrisp.block_encodings import BlockEncoding
from qrisp import QuantumVariable, terminal_sampling, multi_measurement
from qrisp.block_encodings import foqcs_prep_heisenberg_1D, foqcs_prep_spin_glass
from functools import partial
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_W_state

def _heisenberg_from_def(L: int, g: dict, J: dict):

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

def test_foqcs_lcu_prep():
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    #g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    #J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    d_state = QuantumVariable(L * 2 + 6)

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    foqcs_prep_heisenberg_1D(d_state, L, heis_g, heis_J)
    #sv = d_state.qs.statevector("function")
    # Take out the statevector from the compiled circuit
    qc = d_state.qs.compile()
    statev = qc.statevector_array()

    # Modify the original coefficients for the manual state building.
    g[0] = np.sqrt(g[0] * L)
    g[1] = np.sqrt(g[1] * L * -1j)
    g[2] = np.sqrt(g[2] * L)
    J[0] = np.sqrt(J[0] * (L - 1))
    J[1] = np.sqrt(J[1] * -(L - 1))
    J[2] = np.sqrt(J[2] * (L - 1))

    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

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

    # g[0]
    ref_state += g[0] * np.kron(
            [1 if i == 2 ** (6 - 1) else 0 for i in range(2**6)],
            np.kron(dicke_1, zero_n),
        )
    
    # g[1]
    ref_state += g[1] * np.kron(
        [1 if i == 2 ** (6 - 2) else 0 for i in range(2**6)], dicke_double
    )
    
    # g[2]
    ref_state += g[2] * np.kron(
        [1 if i == 2 ** (6 - 3) else 0 for i in range(2**6)],
        np.kron(zero_n, dicke_1),
    )

    # J[0]
    ref_state += J[0] * np.kron(
        [1 if i == 2 ** (6 - 4) else 0 for i in range(2**6)],
        np.kron(dicke_2NN, zero_n),
    )

    # J[1]
    ref_state += J[1] * np.kron(
        [1 if i == 2 ** (6 - 5) else 0 for i in range(2**6)], dicke_2NN_double
    )

    # J[2]
    ref_state += J[2] * np.kron(
        [1 if i == 2 ** (6 - 6) else 0 for i in range(2**6)],
        np.kron(zero_n, dicke_2NN),
    )

    #comp_arr = []
    #comp_arr_ref = []

    # print("State:")
    # for i in range(2**14):
    #     bits = format(i, "014b")
    #     amp = sv({d_state: bits})
    #     if np.isclose(amp, 0j, atol=1e-6) == False:
    #         print(bits, amp)
    #         comp_arr.append(amp)

    # print()
    # print("Ref state:")
    # init_ref = {}
    # q = 0
    # for i in ref_state:
    #     bits = format(q, "014b")
    #     if i != 0:
    #         init_ref[bits] = i
    #         print(bits, i)
    #         comp_arr_ref.append(i)
    #     q += 1

    # print(f"Array: {comp_arr}")
    # print(f"Array REF: {comp_arr_ref}")
    # print(f"Comparison of shortened arrays: {np.allclose(comp_arr, comp_arr_ref, atol=1e-06)}")

    # print(init_ref)

    # expected = QuantumVariable(14)
    # expected.init_state(init_ref, method="qiskit")

    # print(f"State:\n{d_state}")
    # print(f"Expected state:\n{expected}")

    #Zero out entries that are close to zero
    #statev[np.isclose(statev, 0j, atol=1e-6)] = 0

    #for i in range(0, len(statev)):
    #    if statev[i] != 0:
    #        print(f"s[{i}] = {statev[i]}")
    #    if ref_state[i] != 0:
    #        print(f"r[{i}] = {ref_state[i]}")

    # Test that the state received is the same as the reference up to a global phase.
    idx = np.argmax(np.abs(ref_state))
    if np.isclose(ref_state[idx], 0, atol=1e-06):
        assert np.allclose(statev, ref_state, atol=1e-06)

    phase = statev[idx] / ref_state[idx]
    phase /= abs(phase)

    assert np.allclose(statev, phase * ref_state, atol=1e-06), (
        f"States differ beyond global phase {phase}"
    )

def test_foqcs_lcu_spin_glass_prep():
    # g = {"X" : [], "Y" : [], "Z" : []}
    # J = {"X" : [[]], "Y" : [[]], "Z" : [[]]}

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

    # g = np.array(np.random.uniform(-1, 1, (3, L)), dtype=object)
    # Jx = np.array([np.random.uniform(-1, 1, size=i) for i in range(L - 1, 0, -1)], dtype=object)
    # Jx = [row.tolist() for row in Jx]
    # Jz = np.array([np.random.uniform(-1, 1, size=i) for i in range(L - 1, 0, -1)], dtype=object)
    # Jz = [row.tolist() for row in Jz]
    # Jy = np.array([np.random.uniform(-1, 1, size=i) for i in range(L - 1, 0, -1)], dtype=object)
    # Jy = [row.tolist() for row in Jy]
    # J = {"X": Jx, "Y": Jy, "Z": Jz}

    # Fix coefficients for debugging
    g = np.array( [[-0.4808829 +0.j, -0.86150457+0.j,  0.22114172+0.j],
                   [-0.1736148 +0.j,  0.49011868+0.j,  0.78437336+0.j],
                   [-0.33825609+0.j, -0.19728503+0.j, -0.5482909 +0.j]])
    J = np.array([[[ 0.        +0.j,  0.45669928+0.j, -0.29835039+0.j],
                   [ 0.45669928+0.j,  0.        +0.j, -0.66928103+0.j],
                   [-0.29835039+0.j, -0.66928103+0.j,  0.        +0.j]],
                  [[ 0.        +0.j, -0.24574681+0.j,  0.07099358+0.j],
                   [-0.24574681+0.j,  0.        +0.j, -0.08502734+0.j],
                   [ 0.07099358+0.j, -0.08502734+0.j,  0.        +0.j]],
                  [[ 0.        +0.j, -0.49783128+0.j,  0.68088956+0.j],
                   [-0.49783128+0.j,  0.        +0.j, -0.82130807+0.j],
                   [ 0.68088956+0.j, -0.82130807+0.j,  0.        +0.j]]])

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

    # sv = d_state.qs.statevector("function")
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

    # Modify the original coefficients for the manual state building.
    for k in range(L):
        for i in range(0, L - k):
            # g terms
            if k == 0:
                ket = np.zeros(2**L)
                ket[2 ** (L - i - 1)] = 1
                # gx
                ref_state += g[0, i] * np.kron(
                    [
                        1 if j == 2 ** (3 * L - 1) else 0
                        for j in range(2 ** (3 * L))
                    ],
                    np.kron(ket, zero_n),
                )
                # gz
                ref_state += g[2, i] * np.kron(
                    [1 if j == 2 ** (L - 1) else 0 for j in range(2 ** (3 * L))],
                    np.kron(zero_n, ket),
                )
                double_ket = np.zeros(2 ** (2 * L))
                double_ket[2 ** (L - i - 1) + 2 ** (2 * L - i - 1)] = 1
                # gy
                ref_state += g[1, i] * np.kron(
                    [
                        1 if j == 2 ** (2 * L - 1) else 0
                        for j in range(2 ** (3 * L))
                    ],
                    double_ket,
                )
            # J terms
            else:
                ket = np.zeros(2**L)
                ket[2 ** (L - i - 1) + 2 ** (L - i - k - 1)] = 1
                # gx
                ref_state += J[0, i, i + k] * np.kron(
                    [
                        1 if j == 2 ** (3 * L - k - 1) else 0
                        for j in range(2 ** (3 * L))
                    ],
                    np.kron(ket, zero_n),
                )
                # gz
                ref_state += J[2, i, i + k] * np.kron(
                    [
                        1 if j == 2 ** (L - k - 1) else 0
                        for j in range(2 ** (3 * L))
                    ],
                    np.kron(zero_n, ket),
                )
                double_ket = np.zeros(2 ** (2 * L))
                double_ket[
                    2 ** (L - i - 1)
                    + 2 ** (L - i - k - 1)
                    + 2 ** (2 * L - i - 1)
                    + 2 ** (2 * L - i - k - 1)
                ] = 1
                # gy
                ref_state += J[1, i, i + k] * np.kron(
                    [
                        1 if j == 2 ** (2 * L - k - 1) else 0
                        for j in range(2 ** (3 * L))
                    ],
                    double_ket,
                )
    ref_state = ref_state / np.linalg.norm(ref_state)

    statev[np.isclose(statev, 0j, atol=1e-6)] = 0

    for i in range(0, len(statev)):
        if statev[i] != 0:
            print(f"s[{i}] = {statev[i]}")
        if ref_state[i] != 0:
            print(f"r[{i}] = {ref_state[i]}")

    # Test that the state received is the same as the reference up to a global phase.
    idx = np.argmax(np.abs(ref_state))
    if np.isclose(ref_state[idx], 0, atol=1e-06):
        assert np.allclose(statev, ref_state, atol=1e-06)

    phase = statev[idx] / ref_state[idx]
    phase /= abs(phase)

    assert np.allclose(statev, phase * ref_state, atol=1e-06), (
        f"States differ beyond global phase {phase}"
    )

    # Test that the state received is the same as the reference.
    #assert np.allclose(statev, ref_state, atol=1e-06)

def test_foqcs_lcu_spin_glass_subprep():

    L = 5
    g = {"X" : [2, 1, 3, 2, 4], "Y" : [1, 2, 3, 4, 5], "Z" : [6, 7, 8, 9, 10]}
    J = {"X" : [[3, 5, 6, 7], [1, 2, 1], [1, 0], [4]], "Y" : [[1, 2, 3, 4], [5, 6, 7], [9, 10], [11]], "Z" : [[12, 13, 14, 15], [16, 17, 18], [19, 20], [21]]}

    g_betas = [] # Squared normalization factors for all g components (X, Y, Z) --> [g_beta_X, g_beta_Y, g_beta_Z]
    J_betas = [[], [], []] # Squared normalization factors for all J components and diagonals (X, Y, Z) --> [[J_beta_X1, J_beta_X2, ...], [J_beta_Y1, J_beta_Y2, ...], [J_beta_Z1, J_beta_Z2, ...]]
    g_hats = [[], [], []] # Normalized g coefficients
    J_hats = [[], [], []] # Normalized J coefficients
    components = ["X", "Y", "Z"]

    # Normalization for state preparation
    for i in range(3):

        for j in range(len(J["X"])):

            J_hats[i].append([])

    for i in range(3):

        s_sum = 0
        dimension = components[i]

        for j in range(len(g[dimension])):

            s_sum += abs(g[dimension][j]) ** 2

        g_betas.append(s_sum)

    for i in range(3):

        dimension = components[i]

        for j in range(len(J[dimension])):

            s_sum = 0

            for k in range(len(J[dimension][j])):

                s_sum += abs(J[dimension][j][k]) ** 2

            J_betas[i].append(s_sum)

    for i in range(3):

        dimension = components[i]

        for j in range(len(g[dimension])):

            new_g = g[dimension][j] / (g_betas[i] ** 0.5)
            g_hats[i].append(new_g)

    for i in range(3):

        dimension = components[i]

        for j in range(len(J[dimension])):

            for k in range(len(J[dimension][j])):

                new_J = J[dimension][j][k] / ((J_betas[i][j]) ** 0.5)
                J_hats[i][j].append(new_J)

    final_betas = []

    for i in range(3):

        final_betas.append(g_betas[i])

        for j in range(len(J_betas[i])):

            final_betas.append(J_betas[i][j])

    final_betas = np.sqrt(np.array(final_betas))
    final_betas = final_betas / np.linalg.norm(final_betas)

    # SUBPREP
    extra_anc = len(g_betas) + (3 * len(J_betas[0]))
    prep_qv = QuantumVariable(extra_anc)
    unbalanced_W_state(prep_qv, final_betas, reversed=True)

    qc = prep_qv.qs.compile()
    statev = qc.statevector_array()

    ref_state = np.zeros(2 ** (3 * L))
    components = ["X", "Y", "Z"]

    for x in range(3):

        # g term
        ket = np.zeros(2 ** (3 * L))
        ket[2 ** (x * L)] = 1
        ref_state += np.linalg.norm(g[components[x]]) * ket

        # J terms
        for k in range(1, L):

            J_kNN = J[components[x]][k - 1]

            ket = np.zeros(2 ** (3 * L))
            ket[2 ** (k + x * L)] = 1

            ref_state += np.linalg.norm(J_kNN) * ket

    ref_state = ref_state / np.linalg.norm(ref_state)

    idx = np.argmax(np.abs(statev))
    phase = ref_state[idx] / statev[idx]
    phase /= abs(phase)
    
    statev[np.isclose(statev, 0j, atol=1e-6)] = 0

    for i in range(0, len(statev)):
        if statev[i] != 0:
            print(f"s[{i}] = {statev[i]}")
        if ref_state[i] != 0:
            print(f"r[{i}] = {ref_state[i]}")

    assert np.allclose(statev * phase, ref_state)

def test_block_encoding_from_foqcs_lcu_prep():
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    # g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    # J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # actual parameters for the PREP
    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")
    for i in range(3):
        _g[i] = np.sqrt(g[i] * L)
        _J[i] = np.sqrt(J[i] * (L - 1))
    # Correction for XZ = -iY
    _J[1] = 1j * _J[1]
    _g[1] = (1 - 1j) * _g[1] / np.sqrt(2)
    # normalization for state preparation
    norm = np.linalg.norm(np.block([_g, _J]))

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
    prep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
    )
    unprep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
        conjugate=True
    )

    be = BlockEncoding.from_foqcs_lcu_prep(prep, L, unprep=unprep, norm=norm ** 2)

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

    def bit_reverse(i: int, n: int) -> int:
        return int(f"{i:0{n}b}"[::-1], 2)

    # Take out the resulting operands with zero ancillas amplitude.
    res_ops = []
    for i in range(0, 2 ** L):
        qi = bit_reverse(i, L)
        ind = qi << (len(ancillas[0]))
        res_ops.append(sv[ind])

    # Construct reference state vector
    H = _heisenberg_from_def(L, g, J) / (norm ** 2)
    ref_state = H @ psi

    print(f"Ref state = {ref_state}")
    print(f"Resulting operands = {res_ops}")

    assert np.allclose(res_ops, ref_state, atol=1e-6)

def test_block_encoding_from_foqcs_lcu_prep_jax():
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    # g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    # J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # actual parameters for the PREP
    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")
    for i in range(3):
        _g[i] = np.sqrt(g[i] * L)
        _J[i] = np.sqrt(J[i] * (L - 1))
    # Correction for XZ = -iY
    _J[1] = 1j * _J[1]
    _g[1] = (1 - 1j) * _g[1] / np.sqrt(2)
    # normalization for block encoding
    norm = np.linalg.norm(np.block([_g, _J]))

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
    prep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
    )
    unprep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
        conjugate=True
    )

    be = BlockEncoding.from_foqcs_lcu_prep(prep=prep, num_q_ops=L, unprep=unprep, norm=norm ** 2)

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

    print(f"\n\nFiltered dict = {filtered_conditional}")
    print(f"Filtered dict amps sum: {sum(filtered_conditional.values())}")

    # Do the measurement using RUS
    result_rus = main_apply_rus(be)
    print(result_rus)
    print(sum(result_rus.values()))

    assert np.allclose([filtered_conditional[k] for k in sorted(filtered_conditional)],
                       [result_rus[k] for k in sorted(result_rus)],
                       atol = 1e-4)

def test_block_encoding_from_foqcs_lcu_bench_lcu():
    from qrisp.operators import X, Y, Z
    from qiskit import transpile
    # Test FOQCS-LCU resources agains normal LCU
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    # g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    # J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # actual parameters for the PREP
    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")
    for i in range(3):
        _g[i] = np.sqrt(g[i] * L)
        _J[i] = np.sqrt(J[i] * (L - 1))
    # Correction for XZ = -iY
    _J[1] = 1j * _J[1]
    _g[1] = (1 - 1j) * _g[1] / np.sqrt(2)
    # normalization for block encoding
    norm = np.linalg.norm(np.block([_g, _J]))

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
    prep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
    )
    unprep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
        conjugate=True
    )

    be1 = BlockEncoding.from_foqcs_lcu_prep(prep=prep, num_q_ops=L, unprep=unprep, norm=norm ** 2)

    H = sum(
        heis_g["X"] * X(i)
        + heis_g["Y"] * Y(i)
        + heis_g["Z"] * Z(i)
        for i in range(L)
    )

    H += sum(
        heis_J["X"] * X(i) * X(i + 1)
        + heis_J["Y"] * Y(i) * Y(i + 1)
        + heis_J["Z"] * Z(i) * Z(i + 1)
        for i in range(L - 1)
    )

    be2 = BlockEncoding.from_operator(H)

    psi = _prep_psi(L)
    qv = QuantumVariable(4)
    qv.init_state(psi, method="qswitch")

    res1 = be1.resources(qv)
    res2 = be2.resources(qv)

    print(f"Resources comp\nFOQCS-LCU:\n{res1}\nLCU:\n{res2}")

    be1.apply(qv)
    qc1 = qv.qs.compile().to_qiskit()
    be2.apply(qv)
    qc2 = qv.qs.compile().to_qiskit()

    tqc1 = transpile(
                qc1,
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=0,
            )

    tqc2 = transpile(
                qc2,
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=0,
            )

    counts1 = tqc1.count_ops()
    counts2 = tqc2.count_ops()

    print("FOQCS-LCU")
    print("CX count:", counts1.get("cx", 0))
    print("Counts:", counts1)
    print("Depth:", tqc1.depth())
    print("Qubits:", tqc1.num_qubits)
    print("LCU")
    print("CX count:", counts2.get("cx", 0))
    print("Counts:", counts2)
    print("Depth:", tqc2.depth())
    print("Qubits:", tqc2.num_qubits)

    assert True