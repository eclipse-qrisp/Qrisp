"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
from numba import int64, int32, njit, prange, vectorize
from qrisp.simulator.numerics_config import float_tresh
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
        


@vectorize([int64(int64, int64)])
def get_coordinates_0_64(index, bit_shape_1):
    return index >> bit_shape_1


@vectorize([int64(int64, int64)])
def get_coordinates_1_64(index, bit_shape_0):
    return index & ((1 << bit_shape_0) - 1)

@vectorize([int32(int64, int64)])
def get_coordinates_0_32(index, bit_shape_1):
    return index >> bit_shape_1

@vectorize([int32(int64, int64)])
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


def invert_perm(perm):
    return [perm.index(i) for i in range(len(perm))]


def permute_axes(index, index_bit_permutation):
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

    # print(perm)
    new_index = np.zeros(index.size, dtype=index.dtype)

    bit_partition = [sum(perm_shape[:i]) for i in range(len(perm_shape) + 1)]

    # print(bit_partition)
    perm_shape_permuted = [perm_shape[i] for i in invert_perm(perm)]
    permuted_bit_partition = [
        sum(perm_shape_permuted[:i]) for i in range(len(perm_shape) + 1)
    ]

    if len(perm) * index.size > 2**10:
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
            np.array(bit_partition, dtype = index.dtype),
            np.array(permuted_bit_partition, dtype = index.dtype),
            np.array(perm, dtype = index.dtype),
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

@njit(nogil=True, cache=True)
def dense_measurement(input_array, mes_amount, outcome_index):
    
    p = np.abs(np.vdot(input_array, input_array))
    
    if p < float_tresh:
        return [input_array], [p], [-1]
    
    if mes_amount == 0:
        return [input_array], [p], [outcome_index]
    
    N = input_array.shape[0]
    new_arrays = []
    p_list = []
    outcome_index_list = []
    
    a, b, c = dense_measurement(input_array[:N//2], mes_amount - 1, outcome_index)
    
    if c[0] != -1:
        new_arrays.extend(a)
        p_list.extend(b)
        outcome_index_list.extend(c)
    
    a, b, c = dense_measurement(input_array[N//2:], mes_amount - 1, outcome_index + 2**(mes_amount-1))
    
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
    
    array_sort = np.argsort(sorting_array)
    
    new_row = row[array_sort]
    new_col = col[array_sort]
    new_data = data[array_sort]
    
    return new_row, new_col, new_data

@njit(parallel = True, cache = True)
def coo_sparse_matrix_mult_jitted(A_row, A_col, A_data, B_row, B_col, B_data, A_shape, B_shape):
    
    A_row, A_col, A_data = sort_indices_jitted(A_row, A_col, A_data, A_shape[1])
    B_col, B_row, B_data = sort_indices_jitted(B_col, B_row, B_data, B_shape[0])
    
    unique_marker_a = find_unique_markers(A_row)
    unique_marker_b = find_unique_markers(B_col)
    
    res = np.zeros((len(unique_marker_a), len(unique_marker_b)), dtype = A_data.dtype)
    
    for i in prange(len(unique_marker_a)-1):
        comparison_block_a = A_col[unique_marker_a[i]:unique_marker_a[i+1]]        
        for j in range(len(unique_marker_b)-1):
                
            if comparison_block_a[0] > B_row[unique_marker_b[j+1]-1]:
                continue
            
            if B_row[unique_marker_b[j]] > comparison_block_a[-1]:
                continue
            
            comparison_block_b = B_row[unique_marker_b[j]:unique_marker_b[j+1]]
            
            agreeing_ind_a, agreeing_ind_b = find_agreements(comparison_block_a, comparison_block_b)
            
            res[i, j] += np.dot(A_data[unique_marker_a[i] + agreeing_ind_a], B_data[unique_marker_b[j] + agreeing_ind_b])
    
    new_row = []
    new_col = []
    new_data = []
    
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if np.abs(res[i,j]) > 1E-10:
            # if res[i,j] != 0:
                new_row.append(A_row[unique_marker_a[i]])
                new_col.append(B_col[unique_marker_b[j]])
                new_data.append(res[i,j])
                
    return np.array(new_row, dtype = np.int64), np.array(new_col, dtype = np.int64), np.array(new_data, dtype = A_data.dtype)
                
            
@njit(cache = True)
def find_unique_markers(arr):
    
    unique_marker = [0]
    
    cur_value = arr[0]
    for i in range(len(arr)):
        if cur_value != arr[i]:
            unique_marker.append(i)
            cur_value = arr[i]
            
    unique_marker.append(len(arr))
            
    return unique_marker
    

@njit
def find_agreements(block_a, block_b):

    a_pointer = 0
    b_pointer = 0
    
    a_agreements = []
    b_agreements = []
    
    while a_pointer < len(block_a) and b_pointer < len(block_b):
        
        if block_a[a_pointer] < block_b[b_pointer]:
            a_pointer += 1
        elif block_a[a_pointer] > block_b[b_pointer]:
            b_pointer += 1
        else:
            a_agreements.append(a_pointer)
            b_agreements.append(b_pointer)
            
            a_pointer += 1
            b_pointer += 1
            
    return np.array(a_agreements, dtype = np.int32), np.array(b_agreements, dtype = np.int32)


def coo_sparse_matrix_mult(A, B):
    if A.shape[0] < B.shape[1]:
        new_row, new_col, new_data = coo_sparse_matrix_mult_jitted(A.row, A.col, A.data, B.row, B.col, B.data, A.shape, B.shape)
        
    else:
        
        new_col, new_row, new_data = coo_sparse_matrix_mult_jitted(B.col, B.row, B.data, A.col, A.row, A.data, B.shape[::-1], A.shape[::-1])
        
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