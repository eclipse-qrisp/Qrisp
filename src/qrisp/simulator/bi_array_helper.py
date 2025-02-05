"""
\********************************************************************************
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
********************************************************************************/
"""

# -*- coding: utf-8 -*-

# This file contains some functions that enhance the speed
# of SparseBiArray manipulation

import numpy as np
from numba import uint64, uint32, int64, int32, njit, prange, vectorize
from qrisp.simulator.numerics_config import float_tresh, cutoff_ratio
from scipy.sparse import coo_array

# It can happen that the coo-matrix multiplication puts two data entries
# for the same index. These function recognizes and sums these duplicates
def sum_duplicates(nz_indices, data):
    new_indices, reconstr_indices = np.unique(nz_indices, return_inverse=True)

    if len(nz_indices) > 1e4:
        new_data = sum_duplicates_aux_jitted(new_indices, data, reconstr_indices)
    else:
        new_data = sum_duplicates_aux(new_indices, data, reconstr_indices)

    return new_indices, new_data


def sum_duplicates_aux(new_indices, old_data, reconstr_indices):
    new_data = np.zeros(new_indices.shape, dtype=old_data.dtype)

    for i in range(len(reconstr_indices)):
        new_data[reconstr_indices[i]] += old_data[i]

    return new_data


@njit(nogil=True, cache=True)
def sum_duplicates_aux_jitted(new_indices, old_data, reconstr_indices):
    new_data = np.zeros(new_indices.shape, dtype=old_data.dtype)

    for i in range(len(reconstr_indices)):
        new_data[reconstr_indices[i]] += old_data[i]

    return new_data


# This function rounds entries, which are below a certain float thresh and removes the
# tracked indices.
# @njit(parallel = True)
def elim_zeros(nz_indices, data, decimals=8):
    # return nz_indices, data
    # data = np.around(data, decimals=decimals, out=data)

    # mask = np.abs(data) > 10**-decimals
    mask = np.nonzero(data)[0]

    return nz_indices[mask], data[mask]


# This function applies an index permutation onto
# flattened indices.
# Consider the situation that we have the two flattened indices
# [1,3]
# of a 2x2 matrix.
# If we now want to permute the axes, we permute the bit level
# representation of these numbers
# ["01", "10"] ==> ["10", "01"]


@njit(nogil=True, cache=True)
def permute_axes(index, permutation):
    permutation = -permutation + len(permutation) - 1
    permutation = permutation[::-1]
    new_index = np.zeros(index.size, dtype=index.dtype)
    for i in range(len(permutation)):
        if permutation[i] == i:
            # new_index = new_index | (index&(1<<permutation[i]))
            for j in prange(len(new_index)):
                new_index[j] = new_index[j] | (index[j] & (1 << permutation[i]))
        else:
            # new_index = new_index |
            # (((index&(1<<permutation[i]))>>permutation[i])<<(i))
            for j in prange(len(new_index)):
                new_index[j] = new_index[j] | (
                    ((index[j] & (1 << permutation[i])) >> permutation[i]) << (i)
                )

    return new_index


# This function retrieves the coordinates from
# flattened coordinades for a given shape of length two
def get_coordinates(indices, shape):
    if shape[0] < 1<<31:
        coordinates_0 = get_coordinates_0_32(indices, int(np.log2(shape[1])))
    else:
        coordinates_0 = get_coordinates_0_64(indices, int(np.log2(shape[1])))
        
    if shape[1] < 1<<31:
        coordinates_1 = get_coordinates_1_32(indices, int(np.log2(shape[1])))
    else:
        coordinates_1 = get_coordinates_1_64(indices, int(np.log2(shape[1])))

    return coordinates_0, coordinates_1
        


@vectorize([uint64(int64, int64)])
def get_coordinates_0_64(index, bit_shape_1):
    return index >> bit_shape_1


@vectorize([uint64(int64, int64)])
def get_coordinates_1_64(index, bit_shape_0):
    return index & ((1 << bit_shape_0) - 1)

@vectorize([uint32(int64, int64)])
def get_coordinates_0_32(index, bit_shape_1):
    return index >> bit_shape_1

@vectorize([uint32(int64, int64)])
def get_coordinates_1_32(index, bit_shape_0):
    return index & ((1 << bit_shape_0) - 1)



# This function basically inverts get_coordinates.
# Ie. it generates flat coordinates from row and column indices
def gen_flat_coords(col, row, shape):
    return gen_flat_coords_vec(col, row, int(np.log2(shape[1])))
    
@vectorize([int64(int64, int64, int64)], cache=True)
def gen_flat_coords_vec(col, row, bit_shape_1):
    return row << bit_shape_1 | col



# This function constructs a flattened numpy array from
# given flattened coordinates and data
@njit(nogil=True, cache=True)
def construct_flat_array(flat_coords, data, size):
    res = np.zeros(size, dtype=data.dtype)

    for i in range(len(flat_coords)):
        res[flat_coords[i]] += data[i]

    return res

def construct_flat_array_no_jit(flat_coords, data, size):
    res = np.zeros(size, dtype=data.dtype)

    for i in range(len(flat_coords)):
        res[flat_coords[i]] += data[i]

    return res


@njit(cache=True)
def extract_bit_range(index, start, stop):
    breadth = stop - start
    return (index & (((1 << breadth) - 1) << start)) >> start


@njit(cache=True)
def insert_bit_range(target, bit_string, start):
    target |= bit_string << start


@njit(nogil=True, cache=True)
def jitted_permuter(index, new_index, bit_partition, permuted_bit_partition, perm):
    # for i in range(len(perm)):
    #     # bit_range = extract_bit_range(index, bit_partition[i], bit_partition[i+1])
    #     # insert_bit_range(new_index, bit_range, permuted_bit_partition[perm[i]])
    #     #Inlined version more efficient:
    #     new_index |= (((index&((((1<<(bit_partition[i+1] -
    #     bit_partition[i]))-1))<<bit_partition[i]))>>bit_partition[i])<<
    #     permuted_bit_partition[perm[i]])

    for j in prange(index.size):
        for i in range(len(perm)):
            # bit_range = extract_bit_range(index, bit_partition[i], bit_partition[i+1])
            # insert_bit_range(new_index, bit_range, permuted_bit_partition[perm[i]])
            # Inlined version more efficient:
            new_index[j] |= (
                (
                    index[j]
                    & (
                        ((1 << (bit_partition[i + 1] - bit_partition[i])) - 1)
                        << bit_partition[i]
                    )
                )
                >> bit_partition[i]
            ) << permuted_bit_partition[perm[i]]


def permuter(index, new_index, bit_partition, permuted_bit_partition, perm):
    for i in range(len(perm)):
        # bit_range = extract_bit_range(index, bit_partition[i], bit_partition[i+1])
        # insert_bit_range(new_index, bit_range, permuted_bit_partition[perm[i]])
        # Inlined version more efficient:
        if isinstance(new_index, np.ndarray):
            new_index |= (
                (
                    index
                    & (
                        (int(1 << (bit_partition[i + 1] - bit_partition[i])) - 1)
                        << bit_partition[i]
                    )
                )
                >> bit_partition[i]
            ) << permuted_bit_partition[perm[i]]
        else:
            for j in range(len(index)):
                new_index[j] |= (
                    (
                        index[j]
                        & (
                            (int(1 << (bit_partition[i + 1] - bit_partition[i])) - 1)
                            << int(bit_partition[i])
                        )
                    )
                    >> int(bit_partition[i])
                ) << int(permuted_bit_partition[perm[i]])
            


def invert_perm(perm):
    return [perm.index(i) for i in range(len(perm))]


def permute_axes(index, index_bit_permutation, jit = True):
    n = len(index_bit_permutation)

    index_bit_permutation = -np.array(index_bit_permutation) + n - 1
    index_bit_permutation = list(index_bit_permutation)[::-1]

    index_bit_permutation = list(index_bit_permutation)

    # Get the permutation of the axes

    perm = [index_bit_permutation.index(i) for i in range(n)]

    # Initialize the shape
    perm_shape = n * [1]

    i = 1
    while i < len(perm):
        # If this condition is met, both indices form an interval,
        # that does not need further permutation
        if perm[i] - 1 == perm[i - 1]:
            # Remove the index from the perm and perm_shape list
            perm.pop(i)
            perm_shape.pop(i)

            # Update the shape
            perm_shape[i - 1] += 1
            # Update the permutation indices
            for j in range(len(perm)):
                if perm[j] > perm[i - 1]:
                    perm[j] -= 1
        else:
            i += 1

    if isinstance(index, np.ndarray):
        new_index = np.zeros(index.size, dtype=index.dtype)
        dtype = index.dtype
    else:
        new_index = [0 for _ in range(len(index))]
        dtype = np.int64

    bit_partition = [sum(perm_shape[:i]) for i in range(len(perm_shape) + 1)]

    perm_shape_permuted = [perm_shape[i] for i in invert_perm(perm)]
    permuted_bit_partition = [
        sum(perm_shape_permuted[:i]) for i in range(len(perm_shape) + 1)
    ]

    if len(perm) * len(index) > 2**10 and jit:
        jitted_permuter(
            index,
            new_index,
            np.array(bit_partition),
            np.array(permuted_bit_partition),
            np.array(perm),
        )
    else:
        
        permuter(
            index,
            new_index,
            np.array(bit_partition, dtype = dtype),
            np.array(permuted_bit_partition, dtype = dtype),
            np.array(perm, dtype = dtype),
        )

    return new_index


@njit(cache=True)
def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    s = np.empty(len(p), dtype=np.int64)
    s[p] = np.arange(p.size)
    return s


@njit(nogil=True, cache=True)
def bi_array_moveaxis(data_array, index_perm, f_index_array):
    for i in range(len(index_perm)):
        shifted_value = 1 << index_perm[i]
        start_index = 1 << i

        for j in range(start_index):
            f_index_array[start_index ^ j] = shifted_value ^ f_index_array[j]

    # return f_index_array
    return data_array[f_index_array]


@njit(parallel = True, cache = True)
def dense_measurement_brute(input_array, mes_amount, outcome_index):
    
    n = int(np.log2(len(input_array)))
    mes_amount = int(mes_amount)
    reshaped_array = input_array.reshape((2**(mes_amount), 2**(n-mes_amount)))
    
    p_array = np.zeros(2**mes_amount, dtype = input_array.dtype)
    
    for i in prange(2**mes_amount):
        new_array = reshaped_array[i,:]
        p_array[i] = np.vdot(new_array, new_array)
    
    p_array = np.abs(p_array)
    max_p_array = np.max(p_array)
    
    indices = np.nonzero(p_array > max_p_array*cutoff_ratio)[0]
    
    return reshaped_array[indices,:], p_array[indices], indices

    new_arrays = []
    p_values = []
    outcome_indices = []
    
    for i in range(2**mes_amount):
        p = p_array[i]
        if p > max_p_array*cutoff_ratio:
            new_arrays.append(reshaped_array[i, :])
            p_values.append(p)
            outcome_indices.append(outcome_index+i)
    
    return new_arrays, p_values, outcome_indices


@njit(nogil=True, cache=True)
def dense_measurement_smart(input_array, mes_amount, outcome_index):
    
    p = np.abs(np.vdot(input_array, input_array) )
    
    if p < float_tresh:
        return [input_array], [p], [-1]
    
    if mes_amount == 0:
        return [input_array], [p], [outcome_index]
    
    N = input_array.shape[0]
    new_arrays = []
    p_list = []
    outcome_index_list = []
    
    a, b, c = dense_measurement_smart(input_array[:N//2], mes_amount - 1, outcome_index)
    
    if c[0] != -1:
        new_arrays.extend(a)
        p_list.extend(b)
        outcome_index_list.extend(c)
    
    a, b, c = dense_measurement_smart(input_array[N//2:], mes_amount - 1, outcome_index + 2**(mes_amount-1))
    
    if c[0] != -1:
        new_arrays.extend(a)
        p_list.extend(b)
        outcome_index_list.extend(c)
    
    if len(outcome_index_list):
        return new_arrays, p_list, outcome_index_list
    else:
        return new_arrays, p_list, [-1]


@njit(cache = True)
def sort_indices_jitted(row, col, data, shape_1):
    
    shifted_row = row << (int(np.log2(shape_1))+1)
    sorting_array = shifted_row ^ col
    
    array_sort = np.argsort(sorting_array, kind = "mergesort")
    
    new_row = row[array_sort]
    new_col = col[array_sort]
    new_data = data[array_sort]
    
    return new_row, new_col, new_data


def coo_sparse_matrix_mult_inner(A_row, A_col, A_data, B_row, B_col, B_data, A_shape, B_shape):
    """
    This function describes a novel sparse matrix multiplication algorithm operating
    on the COO format. The scipy and MKL implementation of sparse matrix multiplication
    operate on the CSR/CSC format, which requires a conversion step, that can
    be costly for very sparse matrices.
    
    Given are the inputs A, B in sparse format and we want to extract A*B in sparse format
    
    The steps in the algorithm presented here are the following
    
    1. For both inputs, sort the data according to the indices, prioritizing columns
    for A and rows for B
    
    For instance if we have a sparse matrix with
    
    col = [1,2,1,3]
    row = [4,3,3,1]
    data = [0.1, 0.2, 0.3, 0.4]
    
    The sorting with columns prioritized is
    
    col = [1, 1, 2, 3]
    row = [3, 4, 3, 1]
    data = [0.3, 0.1, 0.2, 0.4]
    
    2. Find the indices, where the indices of the prioritized array increases.
    We call this the "unique marker". For instance in the example above,
    the first unique marker is at index 2, because at this index, the col array
    changes from 1 to 2 and the next one is at index 3 because the col array changes
    from 2 to 3.
    
    Each of these interval now represent one column of the matrix. For instance 
    the first column is represented by 
    
    col = [1,1]
    row = [3,4]
    data = [0.3, 0.1]
    
    3. We now identified the columns for A and the rows for B.
    Matrix multiplication is essentially the scalar product of the columns
    of A with the rows of B.
    
    To compute the dot product of a given column with a given row, we need to find
    the index agreements. This is because only on these indices, there can be
    a non-zero contribution. 
    
    For example assume that we have
    
    Column A:
        
    col_a = [1,1]
    row_a = [3,4]
    data_a = [0.3, 0.1]

    Row B:
        
    col_b = [1,4]
    row_b = [3,3]
    data_b = [0.7, 0.8]
    
    This computes the scalar product of column 1 of A with row 3 of B (therefore
    the result will be the (1,3) entry of A*B).
    
    To compute the dot product we see that there is only a non-zero contribution
    for index 4:
        
      A[1,0]*B[0,3] +A[1,1]*B[1,3] +A[1,2]*B[2,3] +A[1,3]*B[3,3] +A[1,4]*B[4,3]
    = 0*0           +0*0.7         +0*0           +0.3*0         + 0.8*0.1
    
    If we label the array of agreeing indices as agreements_a and agreements_b,
    the entry (A*B)[1,3] is therefore determined as
    
    dot(data_a[agreements_a], data_b[agreements_b])
    
    4. After having done this for all unique markers, we end up with a K times L
    array R, where K is the amount of unique markers in A and L is the amount of unique
    markers in B.
    
    To retrieve the COO representation of R, we first filter all non-zero indices.
    For this we can apply some floating point error tolerance mechanism.
    
    This will give as the index arrays I, J where R is non-zero.
    The rows array of the COO representation of A*B is now given as unique_marker_A[I].
    The colums array of the COO representation of A*B is now given as unique_marker_B[j].
    The data is R[I,J]

    """
    
    A_row, A_col, A_data = sort_indices_jitted(A_row, A_col, A_data, A_shape[1])
    B_col, B_row, B_data = sort_indices_jitted(B_col, B_row, B_data, B_shape[0])
    
    unique_marker_a = find_unique_markers(A_row)
    unique_marker_b = find_unique_markers(B_col)

    R = coo_mult_kernel(A_col, A_data, B_row, B_data, unique_marker_a, unique_marker_b)
    
    abs_R = np.abs(R)
    max_abs = np.max(R.ravel())
    
    I, J = np.nonzero(abs_R > (cutoff_ratio * max_abs))
    
    res_row = A_row[unique_marker_a[I]]
    res_col = B_col[unique_marker_b[J]]
    
    R_flat = R.ravel()
    res_data = R_flat[J + R.shape[1]*I]

    return res_row, res_col, res_data

@njit(parallel = True, cache = True)                
def coo_mult_kernel(A_col, A_data, B_row, B_data, unique_marker_a, unique_marker_b):
        
    res = np.zeros((len(unique_marker_a)-1, len(unique_marker_b)-1), dtype = A_data.dtype)
    
    for i in prange(len(unique_marker_a)-1):
        comparison_block_a = A_col[unique_marker_a[i]:unique_marker_a[i+1]]        
        for j in range(len(unique_marker_b)-1):
                
            if comparison_block_a[0] > B_row[unique_marker_b[j+1]-1]:
                continue
            
            if B_row[unique_marker_b[j]] > comparison_block_a[-1]:
                continue
            
            comparison_block_b = B_row[unique_marker_b[j]:unique_marker_b[j+1]]
            
            agreeing_ind_a, agreeing_ind_b = find_agreements(comparison_block_a, comparison_block_b)
            
            res[i, j] = np.dot(A_data[unique_marker_a[i] + agreeing_ind_a], B_data[unique_marker_b[j] + agreeing_ind_b])

    return res

@njit(cache = True)
def find_unique_markers(arr):
    
    unique_marker = [0]
    
    cur_value = arr[0]
    for i in range(len(arr)):
        if cur_value != arr[i]:
            unique_marker.append(i)
            cur_value = arr[i]
            
    unique_marker.append(len(arr))
            
    return np.array(unique_marker, dtype = np.int64)
    
@njit(cache = True)
def find_agreements(a, b):
    A = np.broadcast_to(b, (a.shape[0], b.shape[0]))
    B = np.broadcast_to(a, (b.shape[0], a.shape[0]))
    return np.nonzero(A==B.transpose())


def coo_sparse_matrix_mult(A, B):
    if A.shape[0] < B.shape[1]:
        new_row, new_col, new_data = coo_sparse_matrix_mult_inner(A.row, A.col, A.data, B.row, B.col, B.data, A.shape, B.shape)
        
    else:
        
        new_col, new_row, new_data = coo_sparse_matrix_mult_inner(B.col, B.row, B.data, A.col, A.row, A.data, B.shape[::-1], A.shape[::-1])
        
    if len(new_data) == 0:
        return coo_array(([], ([], [])), shape = (A.shape[0], B.shape[1]))

    if np.max(new_row) < 1<<31 and not new_row.dtype == np.int32:
        new_row = np.array(new_row, dtype = np.int32)
        
    if np.max(new_col) < 1<<31 and not new_col.dtype == np.int32:
        new_col = np.array(new_col, dtype = np.int32)
            
    return coo_array((new_data, (new_row, new_col)), shape = (A.shape[0], B.shape[1]))


def sparse_matrix_mult(A, B):
    
    if A.shape[0]*A.shape[1] < B.shape[0]*B.shape[1]:

        log_shape_0_a = np.log2(A.shape[0])
        log_shape_1_b = np.log2(B.shape[1])
        
        log_sparsity_a = -np.log2(A.nnz/(A.shape[0]*A.shape[1]))
        log_sparsity_b = -np.log2(B.nnz/(B.shape[0]*B.shape[1]))
        
    else:
        log_shape_0_a = np.log2(B.shape[1])
        log_shape_1_b = np.log2(A.shape[0])
        
        log_sparsity_a = -np.log2(B.nnz/(B.shape[0]*B.shape[1]))
        log_sparsity_b = -np.log2(A.nnz/(A.shape[0]*A.shape[1]))
    
    if get_prediction(log_shape_0_a, log_shape_1_b, log_sparsity_a, log_sparsity_b):
        return (A @ B).tocoo()
    else:
        return  coo_sparse_matrix_mult(A, B)
    

def get_prediction(log_shape_0_a, log_shape_1_b, log_sparsity_a, log_sparsity_b):
    
    data_tuple = (log_shape_0_a, log_shape_1_b, log_sparsity_a, log_sparsity_b)
    
    # Normalize and add bias to the input data
    x_mean = np.array([ 5.488, 28.741,  7.679, 26.106])
    x_std = np.array([ 2.26844793, 11.88553402,  3.73764083, 12.94182228])
    theta = np.array([-3.37310532,  1.05192584, -0.98338696, -0.6838338 , -2.10351854])
    normalized_data = (np.array(data_tuple) - x_mean) / x_std
    data_with_bias = np.hstack(([1], normalized_data))

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Calculate the probability using the trained theta values
    probability = sigmoid(np.dot(data_with_bias, theta))

    # Make a binary prediction (True/False) based on the probability threshold (0.5)
    prediction = probability >= 0.5

    return prediction    