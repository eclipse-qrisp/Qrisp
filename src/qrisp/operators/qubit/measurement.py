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

from qrisp import QuantumVariable, QuantumArray
from qrisp.core.compilation import qompiler
import math
import numpy as np
from numba import njit, prange


def get_measurement(
    hamiltonian,
    qarg,
    precision=0.01,
    backend=None,
    shots=1000000,
    compile=True,
    compilation_kwargs={},
    subs_dic={},
    precompiled_qc=None,
    _measurement=None # measurement settings
    ):
    r"""
    This method returns the expected value of a Hamiltonian for the state of a quantum argument.

    Parameters
    ----------
    qarg : QuantumVariable, QuantumArray or list[QuantumVariable]
        The quantum argument to evaluate the Hamiltonian on.
    precision: float, optional
        The precision with which the expectation of the Hamiltonian is to be evaluated.
        The default is 0.01. The number of shots scales quadratically with the inverse precision.
    backend : BackendClient, optional
        The backend on which to evaluate the quantum circuit. The default can be
        specified in the file default_backend.py.
    shots : integer, optional
        The maximum amount of shots to evaluate the expectation of the Hamiltonian. 
        The default is 1000000.
    compile : bool, optional
        Boolean indicating if the .compile method of the underlying QuantumSession
        should be called before. The default is True.
    compilation_kwargs  : dict, optional
        Keyword arguments for the compile method. For more details check
        :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`. The default
        is ``{}``.
    subs_dic : dict, optional
        A dictionary of Sympy symbols and floats to specify parameters in the case
        of a circuit with unspecified, :ref:`abstract parameters<QuantumCircuit>`.
        The default is {}.
    precompiled_qc : QuantumCircuit, optional
            A precompiled quantum circuit.

    Raises
    ------
    Exception
        If the containing QuantumSession is in a quantum environment, it is not
        possible to execute measurements.

    Returns
    -------
    float
        The expected value of the Hamiltonian.

    """

    from qrisp import QuantumSession, merge

    if isinstance(qarg,QuantumVariable):
        if qarg.is_deleted():
            raise Exception("Tried to get measurement from deleted QuantumVariable")
        qs = qarg.qs
            
    elif isinstance(qarg,QuantumArray):
        for qv in qarg.flatten():
            if qv.is_deleted():
                raise Exception(
                    "Tried to measure QuantumArray containing deleted QuantumVariables"
                )
        qs = qarg.qs
    elif isinstance(qarg,list):
        qs = QuantumSession()
        for arg in qarg:
            if isinstance(arg, QuantumVariable) and qv.is_deleted():
                raise Exception(
                    "Tried to measure QuantumArray containing deleted QuantumVariables"
                ) 
            merge(qs,arg)

    if backend is None:
        if qs.backend is None:
            from qrisp.default_backend import def_backend

            backend = def_backend
        else:
            backend = qarg.qs.backend

    if len(qs.env_stack) != 0:
        raise Exception("Tried to get measurement within open environment")


    # Copy circuit in over to prevent modification
    if precompiled_qc is None:        
        if compile:
            qc = qompiler(
                qs, **compilation_kwargs
            )
        else:
            qc = qs.copy()
    else:
        qc = precompiled_qc.copy()

    # Bind parameters
    if subs_dic:
        qc = qc.bind_parameters(subs_dic)
        from qrisp.core.compilation import combine_single_qubit_gates
        qc = combine_single_qubit_gates(qc)

    qc = qc.transpile()
    
    hamiltonian = hamiltonian.hermitize()
    hamiltonian = hamiltonian.eliminate_ladder_conjugates()
    
    from qrisp.misc import get_measurement_from_qc
    results = []
    
    groups = hamiltonian.commuting_qw_groups()
        
    # Collect standard deviation
    stds = []
    for group in groups:
        stds.append(np.sqrt(group.to_pauli().hermitize().get_operator_variance()))
        
    N = sum(stds)
    shots_list = [int(N*s/precision**2) for s in stds]
    tot_shots = sum(x for x in shots_list)
    
    if tot_shots>shots:
        raise Warning("Warning: The total amount of shots required: " + str(tot_shots) +" for the target precision: " + str(precision) + " exceeds the allowed maxium amount of shots. Decrease the precision or increase the maxium amount of shots.")

    
    meas_ops = []
    meas_coeffs = []
    
    for group in groups:
        
        conjugation_circuit = group.get_conjugation_circuit()
        
        curr = qc.copy()
        qubits = [qarg[i] for i in range(conjugation_circuit.num_qubits())]
        curr.append(conjugation_circuit.to_gate(), qubits)
        
        res = get_measurement_from_qc(curr.transpile(), list(qarg), backend, shots_list.pop(0))
        results.append(res)
        
        temp_meas_ops = []
        temp_coeff = []
        for term, coeff in group.terms_dict.items():
            temp_meas_ops.append(term.serialize())
            temp_coeff.append(coeff)
            
        meas_coeffs.append(temp_coeff)
        meas_ops.append(temp_meas_ops)
    
    return evaluate_expectation(results, meas_ops, meas_coeffs)

#
# Evaluate expectation
#

    
def evaluate_observable(observable: tuple, x: int):
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
    
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable
    
    sign_flip = bin(z_int & x).count('1')
    
    temp = (x ^ AND_ctrl_state)
    
    if contains_ladder:
        prefactor = 0.5
    else:
        prefactor = 1
    
    if AND_bits == 0:
        return prefactor*(-1)**sign_flip
    
    if temp & AND_bits == 0:
        return prefactor*(-1)**sign_flip
    else:
        return 0
    
    
    return (-1)**(sign_flip)
    

def evaluate_expectation(results, operators, coefficients):
    """
    Evaluate the expectation.
    
    """

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            for outcome,probability in results[index1].items():
                expectation += probability*evaluate_observable(op,outcome)*np.real(coefficients[index1][index2])
    
    return expectation


#
# Numba accelearation
#


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


def evaluate_expectation_numba(results, operators, coefficients, qubits):
    """
    Evaluate the expectation.
    
    """
    N = len(operators)   

    # Partition outcomes in uint64
    #outcomes_parts = [partition(result.keys(), len(meas_qubits[k])) for k, result in enumerate(results)]
    outcomes_parts = [partition(list(results[k].keys()), len(qubits[k])) if len(qubits[k])>64 else [np.array(list(results[k].keys()), dtype=np.uint64)] for k in range(N)]

    probabilities = [np.array(list(result.values()), dtype=np.float64) for result in results]

    # Partition observables in uint64
    observables_parts = [partition(observable, len(qubits[k])) if len(qubits[k])>64 else [np.array(observable, dtype=np.uint64)] for k, observable in enumerate(operators)]

    coefficients = [np.array(coeffs, dtype=np.float64) for coeffs in coefficients]

    # Evaluate the observable
    expectation = 0
            
         
    for k in range(N):

        result = evaluate_observables_parallel(observables_parts[k], outcomes_parts[k], probabilities[k])
        #print(result)
        #print(coefficients[k])

        expectation += result @ coefficients[k]
    
    return expectation