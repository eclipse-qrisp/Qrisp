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
import sympy as sp

# Checks if a gate is permeable on a set of qubits
# Permeable means that the gate has the form
# U = \sum_{x = 0}^{2^n} |x><x| U_x
# where the braket expression represents the qubits
# of the list "qubit_indices"
# {U_x | x <= 2^n} is a set of operations on the remaining qubits

# Applying a gate which is permeable on a register R to a computational basis state in R
# leaves the state of R invariant i.e. if R is in state |x>
# U |x>|y> = |x> U_x |y>
# this property will for instance be used to make sure that uncomputations,
# which depend on the state of R indeed stil recieve the original state of R
# when the uncomputation is performed. An error will be thrown if in the meantime,
# there have been non-permeable operations on R

# But what is the general strategy in determining this property?
# To understand this, note that if we order the computational basis vectors,
# such that all the qubits which are supposed to be checked for permeability are at the
# end.

# In this basis, the unitary is block diagonal, where the blocks on the diagonal are the
# U_x matrices, we check the block diagonality by first calculating the block matrices
# and then create a boolean 2^n x 2^n matrix, where the entry is False if the
# corresponding block matrix is 0


# We then check the diagonality of this matrix and return True if so
def is_permeable(gate, qubit_indices):
    for i in qubit_indices:
        if gate.permeability[i] is False:
            return False
        elif gate.permeability[i] is None:
            break
    else:
        return True

    from qrisp.simulator import calc_embedded_unitary

    if gate.definition:
        qc = gate.definition
        for instr in qc.data:
            relevant_qubits = set([qc.qubits[k] for k in qubit_indices]).intersection(
                instr.qubits
            )

            if not relevant_qubits:
                continue

            if instr.op.name == "measure":
                return False

            local_qubit_indices = [instr.qubits.index(qb) for qb in relevant_qubits]

            if not is_permeable(instr.op, local_qubit_indices):
                break

        else:
            for i in qubit_indices:
                gate.permeability[i] = True

            return True

    for par in gate.params:
        if isinstance(par, sp.Expr):
            return False

    # Create qubit order
    qubit_order = []
    for i in range(gate.num_qubits):
        if i not in qubit_indices:
            qubit_order.append(i)

    # The qubits which are to be checked come first
    qubit_order = qubit_indices + qubit_order

    # Invert permutation
    qubit_order = [qubit_order.index(i) for i in range(len(qubit_order))]

    # Calculate unitary
    unitary = calc_embedded_unitary(gate, gate.num_qubits, qubit_order)

    if not isinstance(unitary, np.ndarray):
        unitary = unitary.to_array()

    # Calculate boolean matrix
    bbm = get_boolean_block_matrix(unitary, 2 ** (gate.num_qubits - len(qubit_indices)))
    # Check non-zero off diagonal elements
    off_diagonal_non_zero_count = np.count_nonzero(bbm - np.diag(np.diagonal(bbm)))
    # Return result

    result = not bool(off_diagonal_non_zero_count)

    if result:
        for i in qubit_indices:
            gate.permeability[i] = True

    return result


# Tests if a gate is qfree where qfree is the property that is introduced in SILQ
# meaning that an operation maps any computational basis state to a computational basis
# state.
def is_qfree(gate):
    if gate.is_qfree is not None:
        return gate.is_qfree

    if gate.definition:
        for instr in gate.definition.data:
            if not is_qfree(instr.op):
                break
        else:
            gate.is_qfree = True
            return True

    unitary = np.around(gate.get_unitary(), 5)
    # Check if the unitary has only one non-zero entry in each line
    for i in range(unitary.shape[0]):
        if np.count_nonzero(unitary[i, :]) != 1:
            gate.is_qfree = False
            return False

    gate.is_qfree = True
    return True


# Splits the given matrix into blocks of size block_size
# Returns an object array where each entry contains the corresponding block
def get_block_matrix(matrix, block_size):
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception("Tried to decompose non-square blockmatrix")

    if matrix.shape[0] % block_size:
        raise Exception("Matrix shape is no integer multiple of block size")

    block_amount = int(matrix.shape[0] / block_size)
    result = np.zeros((block_amount, block_amount), dtype="object")

    for i in range(block_amount):
        for j in range(block_amount):
            result[i, j] = matrix[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]

    return result


# Recieves a matrix which is turned into a block matrix and then
# returns a matrix of boolean entries where the entry is False, if
# the corresponding block is the 0 matrix and True otherwise
def get_boolean_block_matrix(matrix, block_size):
    block_matrix = get_block_matrix(matrix, block_size)

    block_amount = int(matrix.shape[0] / block_size)

    result = np.zeros((block_amount, block_amount))

    for i in range(block_amount):
        for j in range(block_amount):
            if np.linalg.norm(block_matrix[i, j]) > 1e-4:
                result[i, j] = 1

    return result
