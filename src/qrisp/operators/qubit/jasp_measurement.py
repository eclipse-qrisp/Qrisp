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

import numpy as np
from qrisp.core import QuantumVariable, measure
from qrisp.jasp import sample

import jax
import jax.numpy as jnp
from jax.lax import while_loop, fori_loop


def get_jasp_measurement(
    hamiltonian,
    state_prep,
    state_args=(),
    precision=0.01,
    diagonalisation_method="commuting_qw"
    ):
    r"""
    This method returns the expected value of a Hamiltonian for the state of a quantum argument.

    Parameters
    ----------
    hamiltonian : QubitOperator
        The Hamiltonian for which the exptectaion value is measuered.
    state_prep : callable
        A function returning a QuantumVariable. 
        The expectation of the Hamiltonian for the state from this QuantumVariable will be measured. 
        The state preparation function can only take classical values as arguments. 
        This is because a quantum value would need to be copied for each sampling iteration, which is prohibited by the no-cloning theorem.
    state_args : tuple
        A tuple of arguments of the ``state_prep`` function.    
    precision: float, optional
        The precision with which the expectation of the Hamiltonian is to be evaluated.
        The default is 0.01. The number of shots scales quadratically with the inverse precision.
    diagonalisation_method : str, optional
        Specifies the method for grouping and diagonalizing the QubitOperator. 
        Available are ``commuting_qw``, i.e., the operator is grouped based on qubit-wise commutativity of terms, 
        and ``commuting``, i.e., the operator is grouped based on commutativity of terms.
        The default is ``commuting_qw``.

    Returns
    -------
    float
        The expected value of the Hamiltonian.

    """

    hamiltonian = hamiltonian.hermitize()
    hamiltonian = hamiltonian.eliminate_ladder_conjugates()
    hamiltonian = hamiltonian.apply_threshold(0)
    if len(hamiltonian.terms_dict) == 0:
        return 0

    if diagonalisation_method=="commuting_qw":
        temp_groups = hamiltonian.commuting_qw_groups()
        groups = []
        # In order for the change of basis function (below) to work properly,
        # the ladder terms either need to completely agree or completely disagree
        for group in temp_groups:
            groups.extend(group.group_up(lambda a, b : a.ladders_agree(b) or not a.ladders_intersect(b)))
            
    elif diagonalisation_method=="commuting":
        temp_groups = hamiltonian.group_up(lambda a, b: a.commute_pauli(b))
        groups = []
        # In order for the change of basis function (below) to work properly,
        # the ladder terms either need to completely agree or completely disagree
        for group in temp_groups:
            groups.extend(group.group_up(lambda a, b : a.ladders_agree(b) or not a.ladders_intersect(b)))

    samples = []
    meas_ops = []
    meas_coeffs = []
    stds = []

    # Compute amounts of shots
    for group in groups:
        # Collect standard deviation
        n = hamiltonian.find_minimal_qubit_amount()
        stds.append(np.sqrt(group.get_operator_variance(n = n)))

    N = sum(stds)
    shots_list = [N*s for s in stds]

    for index, group in enumerate(groups):

        # Calculate the new measurement operators (after change of basis)
        meas_op = group.change_of_basis(method=diagonalisation_method)
        
        def new_state_prep(state_args):
            qv = state_prep(*state_args)
            group.change_of_basis(qv, method=diagonalisation_method)
            return qv

        shots = int(shots_list[index]/precision**2)
        res = sample(new_state_prep, shots=shots)(state_args)

        samples.append(jnp.int64(res))
            
        temp_meas_ops = []
        temp_coeff = []
        for term, coeff in meas_op.terms_dict.items():
            temp_meas_ops.append(term.serialize())
            temp_coeff.append(coeff)
                
        meas_coeffs.append(temp_coeff)
        meas_ops.append(temp_meas_ops)

    expectation = jasp_evaluate_expectation_jitted(samples, meas_ops, meas_coeffs)   
    return expectation


@jax.jit
def jasp_evaluate_expectation_jitted(samples, operators, coefficients):
    """
    Evaluate the expectation.
    
    """
    
    def body_fun(i, val):
        expectation, N, op, samples, coefficient = val
        expectation += 1/N*jasp_evaluate_observable_jitted(op, samples[i])*jnp.real(coefficient)
        return expectation, N, op, samples, coefficient

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            N = len(samples[index1])
            expectation, _, _, _, _ = fori_loop(0, N, body_fun, (expectation, N, op, samples[index1], coefficients[index1][index2]))
    
    return expectation


@jax.jit
def jasp_evaluate_observable_jitted(observable: tuple, x: int):
    # This function evaluates how to compute the energy of a measurement sample x.
    # Since we are also considering ladder operators, this energy can either be
    # 0, -1 or 1. For more details check out the comments of QubitOperator.get_conjugation_circuit
    
    # The observable is given as tuple, containing four integers.
    # To understand the meaning of these integers check QubitTerm.serialize.
    
    # Unwrap the tuple
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable

    # Compute whether the sign should be sign flipped based on the Z operators
    sign_flip_int = z_int & x
    sign_flip = 0

    def cond_fun(state):
        sign_flip_int, sign_flip = state
        return sign_flip_int > 0
    
    def body_fun(state):
        sign_flip_int, sign_flip = state
        sign_flip += sign_flip_int & 1
        sign_flip_int >>= 1
        return sign_flip_int, sign_flip
    
    sign_flip_int, sign_flip = while_loop(cond_fun, body_fun, (sign_flip_int, sign_flip)) 
    
    # If there is a ladder operator in the term, we need to half the energy 
    # because we want to measure (|110><110| - |111><111|)/2
    prefactor = 1 - 0.5*contains_ladder
    
    # If there are no AND bits, we return the result
    # Otherwise we apply the AND_ctrl_state to flip the appropriate bits.
    corrected_x = (x ^ AND_ctrl_state)

    # If all bits are in the 0 state the AND is true.
    return prefactor*(-1)**sign_flip * jnp.int64((AND_bits == 0) | (corrected_x & AND_bits == 0))
    