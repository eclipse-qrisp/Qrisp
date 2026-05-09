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
from qrisp.block_encodings import BlockEncoding
from qrisp import QuantumVariable, terminal_sampling, multi_measurement
from qrisp.block_encodings import foqcs_prep_heisenberg_1D, is_operator_foqcs_compatible, foqcs_analyze_operator
from functools import partial
from qrisp.operators import X, Y, Z

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

    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    d_state = QuantumVariable(L * 2 + 6)

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    foqcs_prep_heisenberg_1D(d_state, L, heis_g, heis_J)
    #sv = d_state.qs.statevector("function")
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

    # Test that the state received is the same as the reference.
    assert np.allclose(statev, ref_state, atol=1e-06)

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

    # Modify the original coefficients for the manual state building.
    _g[0] = np.sqrt(g[0] * L)
    _g[1] = np.sqrt(g[1] * L * -1j)
    _g[2] = np.sqrt(g[2] * L)
    _J[0] = np.sqrt(J[0] * (L - 1))
    _J[1] = np.sqrt(J[1] * -(L - 1))
    _J[2] = np.sqrt(J[2] * (L - 1))

    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": _g[0], "Y": _g[1], "Z": _g[2]}
    heis_J = {"X": _J[0], "Y": _J[1], "Z": _J[2]}

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
    # Modify the original coefficients for the manual state building.
    _g[0] = np.sqrt(g[0] * L)
    _g[1] = np.sqrt(g[1] * L * -1j)
    _g[2] = np.sqrt(g[2] * L)
    _J[0] = np.sqrt(J[0] * (L - 1))
    _J[1] = np.sqrt(J[1] * -(L - 1))
    _J[2] = np.sqrt(J[2] * (L - 1))

    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": _g[0], "Y": _g[1], "Z": _g[2]}
    heis_J = {"X": _J[0], "Y": _J[1], "Z": _J[2]}

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

def test_block_encoding_from_foqcs_lcu_operator():
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

    # Build the operator
    terms = []
    for i in range(L):
        terms.append(g[0] * X(i))
        terms.append(g[1] * Y(i))
        terms.append(g[2] * Z(i))
    for i in range(L - 1):
        terms.append(J[0] * X(i) * X(i + 1))
        terms.append(J[1] * Y(i) * Y(i + 1))
        terms.append(J[2] * Z(i) * Z(i + 1))
    O = sum(terms[1:], terms[0])

    be = BlockEncoding.from_foqcs_lcu_operator(O, L)

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
    H = _heisenberg_from_def(L, g, J) / be.alpha # Normalisation can be taken from BE
    ref_state = H @ psi

    print(f"Ref state = {ref_state}")
    print(f"Resulting operands = {res_ops}")

    assert np.allclose(res_ops, ref_state, atol=1e-6)

def test_block_encoding_from_foqcs_lcu_operator_jax():
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

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # Build the operator
    terms = []
    for i in range(L):
        terms.append(g[0] * X(i))
        terms.append(g[1] * Y(i))
        terms.append(g[2] * Z(i))
    for i in range(L - 1):
        terms.append(J[0] * X(i) * X(i + 1))
        terms.append(J[1] * Y(i) * Y(i + 1))
        terms.append(J[2] * Z(i) * Z(i + 1))
    O = sum(terms[1:], terms[0])

    be = BlockEncoding.from_foqcs_lcu_operator(O, L)

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

###########################################################################################################
#### Operator analysis tests ##############################################################################
###########################################################################################################

def test_foqcs_operator_analysis_failures():
    def empty_must_fail():
        O = 0 * X(0)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O)

        print(exc_info.value)
        assert f"empty or constant operator: {O}" in str(exc_info.value)
    
    def constant_must_fail():
        O = X(0) * X(0)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O)

        print(exc_info.value)
        assert f"empty or constant operator: {O}" in str(exc_info.value)
    
    def bad_length_must_fail():
        O = X(0) + X(1) + X(2) + X(3) + X(4)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O, L=2)

        print(exc_info.value)
        assert f"Received L = {2}" in str(exc_info.value)

    def const_in_op_must_fail():
        O = Y(0) + (0.18 + 0.5j) * Z(1) * Z(3) + X(0) * X(0)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O)

        print(exc_info.value)
        assert f"FOQCS-LCU does not support constant/identity terms" in str(exc_info.value)
    
    def cross_axis_couple_must_fail():
        O = X(0) * Y(3) + X(0) * X(1) + Z(0)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O)

        print(exc_info.value)
        assert f"FOQCS-LCU supports only same-axis couplings, but received: X({0}) * Y({3})" in str(exc_info.value)

    def three_body_couple_must_fail():
        O = X(0) * X(1) + Z(0) + Y(0) * Y(1) * Y(2)
        with pytest.raises(ValueError) as exc_info:
            foqcs_analyze_operator(O)

        print(exc_info.value)
        assert f"FOQCS-LCU supports only one and two-body interactions" in str(exc_info.value)

    print("\n")
    empty_must_fail()
    constant_must_fail()
    bad_length_must_fail()
    const_in_op_must_fail()
    cross_axis_couple_must_fail()
    three_body_couple_must_fail()

def test_foqcs_operator_heisenberg():
    def make_heisenberg_pass_cases():
        cases = {}
        # Full uniform nearest-neighbor XYZ/Heisenberg chain with 2 qubits.
        cases["L2_full_uniform"] = (
            X(0) + X(1)
            + 0.5 * Y(0) + 0.5 * Y(1)
            - 0.3 * Z(0) - 0.3 * Z(1)
            + 0.7 * X(0) * X(1)
            - 0.2 * Y(0) * Y(1)
            + 0.4 * Z(0) * Z(1)
        )
        # Full uniform nearest-neighbor XYZ/Heisenberg chain with 4 qubits.
        H = 0
        L = 4
        H += 1.0 * X(0)
        H += 0.5 * Y(0)
        H += -0.3 * Z(0)
        for i in range(1, L):
            H += 1.0 * X(i)
            H += 0.5 * Y(i)
            H += -0.3 * Z(i)
            H += 0.7 * X(i - 1) * X(i)
            H += -0.2 * Y(i - 1) * Y(i)
            H += 0.4 * Z(i - 1) * Z(i)
        cases["L4_full_uniform"] = H
        # Uniform local X field only.
        cases["uniform_X_field_only"] = X(0) + X(1) + X(2) + X(3)
        # Uniform ZZ NN coupling only.
        cases["uniform_ZZ_coupling_only"] = (
            Z(0) * Z(1)
            + Z(1) * Z(2)
            + Z(2) * Z(3)
        )
        # Zero coefficient in one Pauli family.
        cases["uniform_with_some_zero_families"] = (
            0.8 * X(0) + 0.8 * X(1) + 0.8 * X(2)
            + 0.0 * Y(0) + 0.0 * Y(1) + 0.0 * Y(2)
            + 0.2 * Z(0) * Z(1)
            + 0.2 * Z(1) * Z(2)
        )
        return cases

    def make_heisenberg_fail_cases():
        cases = {}
        # Non-uniform local fields
        cases["non_uniform_local_X"] = (
            1.0 * X(0)
            + 0.5 * X(1)
            + 1.0 * X(2)
        )
        # Local fields on different sites/axes only.
        cases["position_dependent_local_fields"] = (
            X(0)
            + 0.5 * Y(1)
            + 0.2 * Z(0) * Z(1)
        )
        # Long-range same-axis coupling
        cases["long_range_XX"] = X(0) * X(2)
        # Non-uniform nearest-neighbor couplings
        cases["non_uniform_NN_ZZ"] = (
            0.2 * Z(0) * Z(1)
            + 0.7 * Z(1) * Z(2)
        )
        # Missing one NN bond while L is inferred as 4 because of X(3).
        cases["missing_NN_bonds_due_to_L"] = (
            X(3)
            + 0.2 * Z(0) * Z(1)
        )
        # Same-axis arbitrary pair couplings.
        cases["sparse_spin_glass_pairs"] = (
            0.3 * X(0) * X(3)
            - 0.4 * Y(1) * Y(2)
            + 0.9 * Z(0)
        )
        return cases

    for name, O in make_heisenberg_pass_cases().items():
        check, res = is_operator_foqcs_compatible(O)
        assert check, f"Passing operator failed analysis: {O} "
        assert res["method"] == "heisenberg", name

    for name, O in make_heisenberg_fail_cases().items():
        check, res = is_operator_foqcs_compatible(O)
        assert check, f"Passing operator failed analysis: {O} "
        assert res["method"] == "spin_glass", name

# Manual verification test, for now
def test_foqcs_operator_analysis():
    from qrisp.operators import A, C, P1
    H_good = X(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(2)
    H_heis = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)
    H_bad = X(0) * Z(1)
    H = 1+2*X(0)+3*X(0)*Y(1)*A(2)+C(4)*P1(0)

    print(f"\n-----------------------------------------\nFrom H_good = {H_good}")
    comp, res = is_operator_foqcs_compatible(H_good)
    print(f"H_good: {comp}, res = {res}")

    print(f"-----------------------------------------\nFrom H_bad = {H_bad}")
    comp, res = is_operator_foqcs_compatible(H_bad)
    print(f"H_bad: {comp}, res = {res}")

    print(f"-----------------------------------------\nFrom H_heis = {H_heis}")
    comp, res = is_operator_foqcs_compatible(H_heis)
    print(f"H_heis: {comp}, res = {res}")

    print(f"-----------------------------------------\nFrom H = {H}")
    comp, res = is_operator_foqcs_compatible(H)
    print(f"H: {comp}, res = {res}")
    
    assert True



# FIXME! I need to be made into a test and put in some appropriate place. How cruel of you to leave me like this!
def test_pauli_coeff_dict_extraction():
    from qrisp.operators import A, C, P1

    H_good = X(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(2)
    H_bad = X(0) * Z(1)
    H = 1+2*X(0)+3*X(0)*Y(1)*A(2)+C(4)*P1(0)

    res_good = H_good.to_pauli_coeff_dict()
    print(res_good)
    res_bad = H_bad.to_pauli_coeff_dict()
    print(res_bad)
    res = H.to_pauli_coeff_dict()
    print(res)

    assert True
