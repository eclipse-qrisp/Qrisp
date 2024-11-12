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
    
    
    samples = create_padded_array([list(res.keys()) for res in results]).astype(np.int64)
    probs = create_padded_array([list(res.values()) for res in results])
    meas_ops = np.array(meas_ops, dtype = np.int64)
    meas_coeffs = np.array(meas_coeffs)
    
    return evaluate_expectation_jitted(samples, probs, meas_ops, meas_coeffs)


def create_padded_array(list_of_lists):
    """
    Create a padded numpy array from a list of lists with varying lengths.
    
    Parameters:
    list_of_lists (list): A list of lists with potentially different lengths.
    
    Returns:
    numpy.ndarray: A 2D numpy array with padded rows.
    """
    # Find the maximum length of any list in the input
    max_length = max(len(lst) for lst in list_of_lists)
    
    # Create a padded list of lists
    padded_lists = [
        lst + [0] * (max_length - len(lst))
        for lst in list_of_lists
    ]
    
    # Convert to numpy array
    return np.array(padded_lists)


#
# Evaluate expectation
#


def evaluate_expectation(samples, probs, operators, coefficients):
    """
    Evaluate the expectation.
    
    """
    # print(results)
    # print(operators)
    # print(coefficients)
    # raise

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            for i in range(len(samples[index1])):
                outcome,probability = samples[index1, i], probs[index1, i]
                expectation += probability*evaluate_observable(op,outcome)*np.real(coefficients[index1][index2])
    
    return expectation


def evaluate_observable(observable: tuple, x: int):
    # This function evaluates how to compute the energy of a measurement sample x.
    # Since we are also considering ladder operators, this energy can either be
    # 0, -1 or 1. For more details check out the comments of QubitOperator.get_conjugation_circuit
    
    # The observable is given as tuple, containing for integers and a boolean.
    # To understand the meaning of these integers check QubitTerm.serialize.
    
    # Unwrap the tuple
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable

    # Compute whether the sign should be sign flipped based on the Z operators
    sign_flip_int = z_int & x
    sign_flip = 0
    while sign_flip_int:
        sign_flip += sign_flip_int & 1
        sign_flip_int >>= 1
    
    # If there is a ladder operator in the term, we need to half the energy 
    # because we want to measure (|110><110| - |111><111|)/2
    if contains_ladder:
        prefactor = 0.5
    else:
        prefactor = 1
    
    # If there are no and bits, we return the result
    if AND_bits == 0:
        return prefactor*(-1)**sign_flip

    # Otherwise we apply the AND_ctrl_state to flip the appropriate bits.
    corrected_x = (x ^ AND_ctrl_state)
    
    # If all bits are in the 0 state the AND is true.
    if corrected_x & AND_bits == 0:
        return prefactor*(-1)**sign_flip
    else:
        return 0
    

evaluate_observable_jitted = njit(cache = True)(evaluate_observable)

@njit(cache = True)
def evaluate_expectation_jitted(samples, probs, operators, coefficients):
    """
    Evaluate the expectation.
    
    """
    # print(results)
    # print(operators)
    # print(coefficients)
    # raise

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            for i in range(len(samples[index1])):
                outcome,probability = samples[index1, i], probs[index1, i]
                expectation += probability*evaluate_observable_jitted(op,outcome)*np.real(coefficients[index1][index2])
    
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
    