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
from numba import int64, njit, prange, vectorize


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
    data = np.around(data, decimals=decimals, out=data)

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
    new_index = np.zeros(index.size, dtype=np.int64)
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
    return get_coordinates_0(indices, int(np.log2(shape[1]))), get_coordinates_1(
        indices, int(np.log2(shape[1]))
    )


@vectorize([int64(int64, int64)])
def get_coordinates_0(index, bit_shape_1):
    return index >> bit_shape_1


@vectorize([int64(int64, int64)])
def get_coordinates_1(index, bit_shape_0):
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
                    ((1 << (bit_partition[i + 1] - bit_partition[i])) - 1)
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
    new_index = np.zeros(index.size, dtype=np.int64)

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
            np.array(bit_partition),
            np.array(permuted_bit_partition),
            np.array(perm),
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
