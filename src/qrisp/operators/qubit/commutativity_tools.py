"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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
********************************************************************************/
"""

import numpy as np


def inverse_mod2(matrix):
    """
    Calculates the inverse of a binary matrix over the field of integers modulo 2.

    Parameters
    ----------
    matrix : numpy.array
        An invertible binary matrix.
    result : numpy.array
        The inverse of the given matrix over the field of integers modulo 2.

    """
    rows, columns = matrix.shape
    if rows != columns:
        raise Exception("The matrix must be a square matrix")
    
    matrix = matrix.copy()
    result = np.eye(rows, dtype=int)

    for i in range(rows):
        # Find the pivot row
        max_row = i + np.argmax(matrix[i:,i])

         # Swap the current row with the pivot row
        matrix[[i, max_row]] = matrix[[max_row, i]]
        result[[i, max_row]] = result[[max_row, i]]
        
        # Eliminate all rows below the pivot
        for j in range(i + 1, rows):
            if matrix[j, i] == 1:
                matrix[j] = (matrix[j] + matrix[i]) % 2
                result[j] = (result[j] + result[i]) % 2

    # Backward elimination
    for i in range(rows - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if matrix[j, i] == 1:
                matrix[j] = (matrix[j] + matrix[i]) % 2
                result[j] = (result[j] + result[i]) % 2

    return result


def gaussian_elimination_mod2(matrix, type='row', reduced=False, show_pivots=False):
    r"""
    Performs Gaussian elimination in the field $F_2$.
    
    Parameters
    ----------
    matrix : numpy.array
        A binary matrix.
    type : str, optional
        Available are ``row`` for row echelon form, and ``column`` for column echelon form. 
        The default is ``row``.
    reduced : Boolean, optional
        If ``True``, the reduced row (column) echelon form is calculated.
        The default is ``False``.
    show_pivots : Boolean, optional
        If ``True``, the pivot columns (rows) are returned.

    Returns
    -------
    matrix : numpy.array
        The row (column) echelon form of the given matrix.
    pivots : list[int], optional
        A list of indices for the pivot columns (rows).
    
    """
    
    matrix = matrix.copy()
    
    rows, columns = matrix.shape
    column_index = 0
    row_index = 0
    pivots = [] # the pivot columns (type='row') or rows (type='column')

    if type=='row':
        while row_index<rows and column_index<columns:
            # Find the pivot row
            max_row = row_index + np.argmax(matrix[row_index:,column_index])

            if matrix[max_row,column_index]==0:
                column_index+=1
            else:
                # Pivot
                pivots.append(column_index)

                # Swap the current row with the pivot row
                matrix[[row_index, max_row]] = matrix[[max_row, row_index]]
        
                # Eliminate all rows after the pivot
                for j in range(row_index + 1, rows):
                    if matrix[j, column_index] == 1:
                        matrix[j] = (matrix[j] + matrix[row_index]) % 2

                row_index+=1
                column_index+=1

        # Backward elimination (optional, for reduced row echelon form)
        if reduced:
            for i in range(min(rows, columns) - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    if matrix[j, i] == 1:
                        matrix[j] = (matrix[j] + matrix[i]) % 2


    elif type=='column':
        while row_index<rows and column_index<columns:
            # Find the pivot column
            max_column = column_index + np.argmax(matrix[row_index,column_index:])

            if matrix[row_index,max_column]==0:
                row_index+=1
            else:
                # Pivot
                pivots.append(row_index)

                # Swap the current column with the pivot column
                matrix[:,[column_index, max_column]] = matrix[:,[max_column, column_index]]
        
                # Eliminate all columns after the pivot
                for j in range(column_index + 1, columns):
                    if matrix[row_index, j] == 1:
                        matrix[:,j] = (matrix[:,j] + matrix[:,column_index]) % 2

                row_index+=1
                column_index+=1     

        # Backward elimination (optional, for column row echelon form)
        if reduced:
            for i in range(min(rows, columns) - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    if matrix[j, i] == 1:
                        matrix[:,j] = (matrix[:,j] + matrix[:,i]) % 2

    if show_pivots:
        return matrix, pivots 
    else:
        return matrix


def construct_change_of_basis(S):
    """
    Implements the CZ construction outlined in https://quantum-journal.org/papers/q-2021-01-20-385/.

    Parameters
    ----------
    S : numpy.array
        A matrix representing a list of commuting Pauli operators: Each column is a Pauli operator in binary representation.

    Returns
    -------
    A : numpy.array
        Adjacency matrix for the graph state.
    R_inv : numpy.array
        A matrix representing a list of new Pauli operators.
    h_list : numpy.array
        A list indicating the qubits on which a Hadamard gate is applied.
    s_list : numpy.array
        A list indicating the qubits on which an S gate is applied.
    perm_vec : numpy.array
        A vector repesenting a permutation.

    """

    n = int(S.shape[0]/2)

    ####################
    # Step 0: Calculate S_0: Independent columns (i.e., Pauli terms) of S
    ####################

    S_reduced, independent_cols = gaussian_elimination_mod2(S, reduced=True, show_pivots=True)
    k = len(independent_cols)

    S0 = S[:,independent_cols]
    R0_inv = S_reduced[:k, :]

    ####################
    # Step 1: Calculate S_1: The first k rows of the X component have full rank k
    ####################

    # Find independent rows in X component of S0
    S0X_reduced, independent_rows = gaussian_elimination_mod2(S0[-n:, :], type='column', show_pivots=True)

    S1 = S0.copy()
    h_list = []
    # Construct S1 by applying a Hadamard (i.e., a swap) to the rows of S0 not in independent_rows
    if len(independent_rows)<k:
        h_list = [i for i in range(n) if i not in independent_rows]
        for i in h_list:
            S1[[i, n+i]] = S1[[n+i, i]]

    # Find independent rows in X component of S1
    S1X_reduced, independent_rows = gaussian_elimination_mod2(S1[-n:, :], type="column", show_pivots=True)

    # Construct permutation achieving that the first k rows in X component of S1 are independent
    perm = np.arange(0,n)
    for index1, index2 in enumerate(independent_rows):
        curr = perm[index1]
        perm[index1] = index2
        perm[index2] = curr

    # Apply permutation to rows of S1 for Z and X component
    S1 = np.vstack((S1[:n,:][perm],S1[-n:,:][perm]))

    #################### 
    # Step 2: Calculate S2: The first k rows of the X component are the identity matrix
    ####################

    R1_inv = S1[n:n+k,:]
    R1 = inverse_mod2(R1_inv)
    S2 = S1 @ R1 % 2

    ####################
    # Step 3: Calculate S3: Basis extension if n>k
    ####################

    if n>k:

        C = S2[:k, :]
        D = S2[k:n, :]
        F = S2[-(n-k):, :]

        S3 = np.block([[C, np.transpose(D)],
                        [D, np.zeros((n-k,n-k), dtype=int)],
                        [np.eye(k, dtype=int), np.zeros((k,n-k), dtype=int)],
                        [F, np.eye(n-k, dtype=int)]])
        R2_inv = np.block([[np.eye(k, dtype=int)],
                            [np.zeros((n-k,k), dtype=int)]])   
    
    else:

        S3 = S2
        R2_inv = np.eye(n, dtype=int)

    ####################
    # Step 4: Calculate S4 
    ####################

    R3_inv = S3[-n:,:]
    R3 = inverse_mod2(R3_inv)

    S4 = S3 @ R3 % 2

    # Remove diagonal entries in upper left block 
    s_list = []
    for i in range(n):
        if S4[:n, :][i,i]==1:
            s_list.append(i)

    Q2 = np.block([[np.eye(n, dtype=int),S4[:n, :]*np.eye(n, dtype=int)],
                   [np.zeros((n,n), dtype=int),np.eye(n, dtype=int)]])
    
    S4 = Q2 @ S4 % 2

    R_inv = R3_inv @ R2_inv @ R1_inv @ R0_inv % 2

    # Adjacency matrix for the graph 
    A = S4[:n, :]

    return A, R_inv, h_list, s_list, perm