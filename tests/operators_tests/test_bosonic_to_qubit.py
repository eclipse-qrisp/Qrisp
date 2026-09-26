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

from qrisp.operators.bosonic import a_b as a, c_b as c
from qrisp.operators.bosonic.bosonic_term import gray_code, standard_binary, one_hot
import numpy as np
from scipy import sparse

a_matrix = np.diag(np.sqrt(np.arange(1, 8)), k=1).astype(complex)
c_matrix = np.diag(np.sqrt(np.arange(1, 8)), k=-1).astype(complex)

a_matrix_small = np.diag(np.sqrt(np.arange(1, 3)), k=1).astype(complex)
c_matrix_small = np.diag(np.sqrt(np.arange(1, 3)), k=-1).astype(complex)


def _gray_to_standard_binary_permutation(N):
    gray = gray_code(N)
    std_bin = standard_binary(N)

    index_map = {tuple(val): idx for idx, val in enumerate(std_bin)}

    return np.array([index_map[tuple(val)] for val in gray])


def _test_bosonic_to_qubit_bin_reps(operator, matrix):
    """Check whether single bosonic operators are tanslated correctly for the three binary encodings"""

    ##### standard binary #####
    op_pauli_stb = operator.to_qubit_operator(binary_encoding="standard_binary").to_pauli()
    np.testing.assert_array_almost_equal(matrix, np.asarray(op_pauli_stb.to_sparse_matrix().todense()))

    ##### Gray code #####
    op_pauli_gray = operator.to_qubit_operator(binary_encoding="gray_code").to_pauli()
    # the .to_sparse_matrix() method uses a standard binary representation, so in order to check for equality we need to permute first
    _perm = _gray_to_standard_binary_permutation(3)
    if np.shape(matrix)[0] == 8:
        perm = _perm
    else:
        perm = []
        for i in range(8):
            for j in range(8):
                perm.append(_perm[i] * 8 + _perm[j])
        perm = np.array(perm)
    np.testing.assert_array_almost_equal(
        matrix, np.asarray(op_pauli_gray.to_sparse_matrix().todense()[np.ix_(perm, perm)])
    )


def _test_bosonic_to_qubit_onehot(operator, matrix):
    op_pauli_onehot = operator.to_qubit_operator(binary_encoding="one_hot", truncation=3).to_pauli()
    ind = np.array([2 ** (2 - i) for i in range(3)])
    diff = abs(sparse.csr_matrix(matrix) - op_pauli_onehot.to_sparse_matrix()[np.ix_(ind, ind)])
    assert diff.max() < 1e-5


def test_bosonic_to_qubit_single_operators():
    """Check whether single bosonic operators are tanslated correctly for the three binary encodings"""
    _test_bosonic_to_qubit_bin_reps(a(0), a_matrix)
    _test_bosonic_to_qubit_bin_reps(c(0), c_matrix)

    # one-hot representation gets very heavy to compute very quickly, use smaller truncation number (N=3)
    _test_bosonic_to_qubit_onehot(a(0), a_matrix_small)
    _test_bosonic_to_qubit_onehot(c(0), c_matrix_small)


def test_bosonic_to_qubit_multiple_operators():
    """Check whether single bosonic operators are tanslated correctly for the three binary encodings"""
    _test_bosonic_to_qubit_bin_reps(c(0) * a(0), c_matrix @ a_matrix)
    _test_bosonic_to_qubit_bin_reps(c(0) * a(1), np.kron(c_matrix, np.identity(8)) @ np.kron(np.identity(8), a_matrix))

    # one-hot representation gets very heavy to compute very quickly, use smaller truncation number (N=3)
    _test_bosonic_to_qubit_onehot(c(0) * a(0), c_matrix_small @ a_matrix_small)
