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

import math
import numpy as np
from numba import njit, prange


def partition(values, num_qubits):
    """
    Partitions a list of integers into a list of lists of integers with size 64 bit.

    Parameters
    ----------
    values : list[int]
        A list of integers.
    num_qubits : int
        The maximal number of bits.

    Returns
    -------
    partition : list[np.array]
        A list of NumPy numpy.uint64 arrays.

    """
    

    M = math.ceil(num_qubits/64)
    N = len(values)

    zeros = [[0]*N for k in range(M)]
    partition = [values]
    partition.extend(zeros)

    lower_mask = (1<<64) - 1
    
    for j in range(1,M):
        for i in range(N):
            partition[j][i] = partition[j-1][i] & lower_mask
            partition[j+1][i] = partition[j-1][i] >> 64

    if M==1:
        return [np.array(partition[0], dtype=np.uint64)]
    else:
        return [np.array(part, dtype=np.uint64) for part in partition[1:]]


def evaluate_observable(observable: int, x: int):
    """
    This method evaluates an observable that is a tensor product of Pauli-:math:`Z` operators
    with respect to a measurement outcome. 
        
    A Pauli operator of the form :math:`\prod_{i\in I}Z_i`, for some finite set of indices :math:`I\subset \mathbb N`, 
    is identified with an integer:
    We identify the Pauli operator with the binary string that has ones at positions :math:`i\in I`
    and zeros otherwise, and then convert this binary string to an integer.
        
    Parameters
    ----------
        
    observable : int
        The observable represented as integer.
     x : int 
        The measurement outcome represented as integer.
        
    Returns
    -------
    int
        The value of the observable with respect to the measurement outcome.
        
    """
        
    if bin(observable & x).count('1') % 2 == 0:
        return 1
    else:
        return -1  
    

@njit(cache = True)
def evaluate_observable_jitted(observable, x):
    """

    """

    value = observable & x
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return 1 if count % 2 == 0 else -1


@njit(parallel = True, cache = True)
def evaluate_observables_parallel(observables_parts, outcome_parts, probabilities): #M=#observables, N=#measurements
    """

    Parameters
    ----------
    observables_parts : list[numpy.array[numpy.unit64]]
        The observables.
    outcome_parts : list[numpy.array[numpy.unit64]]
        The measurement outcomes.
    probabilities : numpy.array[numpy.float64]
        The measurment probabilities.

    Returns
    -------

    """

    K = len(observables_parts) # Number of observables PARTS
    M = len(observables_parts[0]) # Number of observables
    N = len(probabilities) # Number of measurement results

    res = np.zeros(M, dtype=np.float64) 

    for j in range(M):

        res_array = np.zeros(N, dtype=np.float64) 
        for i in prange(N):
            temp = 1
            for k in range(K):
                temp *= evaluate_observable_jitted(observables_parts[k][j], outcome_parts[k][i])
                
            res_array[i] += temp
        
        res[j] = res_array @ probabilities

    return res

