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

from qrisp.operators import Hamiltonian
from qrisp.operators.fermionic.fermionic_term import FermionicTerm
from qrisp.operators.hamiltonian_tools import group_up_iterable
from qrisp import merge, IterationEnvironment, conjugate
from qrisp.operators.qubit import QubitOperator

import sympy as sp

threshold = 1e-9


def fermionic_trotterization(H, forward_evolution = True):
    
    reduced_H = H.reduce(assume_hermitian=True)
    
    groups = reduced_H.group_up(denominator = lambda a,b : a.unipolars_agree(b))
    
    def meta_group_denominator(H0, H1):
        term_0 = next(iter(H0.terms_dict.keys()))
        term_1 = next(iter(H1.terms_dict.keys()))
        return not term_0.unipolars_intersect(term_1)
    
    meta_groups = group_up_iterable(groups, meta_group_denominator)
    groups = [sum(meta_group, 0) for meta_group in meta_groups]
    
    def trotter_step(qarg, t, steps):
        
        for group in groups:
            
            
            # We now treat the fermionic swaps.
            # The problem here is that terms like 
            # a(0)*a(2) + a(1)*a(3)
            # Have the JW embedding
            # -A(0)*Z(1)*A(2) - A(1)*Z(2)*A(3)
            # Implying the both need access to qubit 1&2 which makes them block
            # each other.
            # The goal is therefore to reorder the terms via fermionic swaps
            # to unblock. For an overview over the fermionic swapping topic
            # please check https://arxiv.org/abs/2310.12256
            
            # Obviously we want to reduce the amount of fermionic swaps to a minimum.
            # We approach this by noticing that (contrary to Selingers approach),
            # it is not necessary to group all ladder operators together. It is
            # sufficient to match them into "couples".
            # a(0)*a(1)*a(7)*a(8)
            # => A(0)*A(1)*A(7)*A(8)

            # Matching the ladder operator into couples takes significantly less
            # swaps them grouping them up in one big chunk.

            # For that reason we will now go through the ladder terms that need
            # to be simulated and match them into couples and singles.
            # Singles are ladder terms that can't be matched because the corresponding
            # operator targets an odd amount of qubits.
            # Consider for instance
            # a(1)*a(3)*a(4)
            # Here, 3&4 are a couple and 1 is a single             
            
            terms = list(group.terms_dict.keys())
            
            singles = []
            couples = {}
            
            for term in terms:
                    
                # For non-unipolar factors (i.e. a(i)*c(i) for instance), no matching is necessary.
                index_list = term.get_unipolars()
                
                # If there is an od amount of ladder operators, remove the last one
                # (has the lowest index)
                if len(index_list)%2:
                    singles.append(index_list.pop(-1))
                
                # We now group the ladder operators into couples
                for i in range(len(index_list)//2):
                    couples[index_list[2*i+1]] = index_list[2*i]
                
            # This function computes the swaps that are necessary to match
            # all couples and moves the singles to the lowest positions.
            swaps, permutation = kai_pflaume(singles, couples, len(qarg))
            
            
            # This function applies the CZ gates on the quantum argument to
            # perform the fermionic swap
            with conjugate(apply_fermionic_swaps)(qarg, swaps) as new_qarg:
                for ferm_term in terms:
                    coeff = reduced_H.terms_dict[ferm_term]
                    
                    # This function permutes the indices of the fermionic term
                    qubit_operator = ferm_term.fermionic_swap(permutation).to_qubit_term()
                    qubit_term = list(qubit_operator.terms_dict.keys())[0]
                    
                    if not len(ferm_term.get_unipolars())%2:
                        for factor in qubit_term.factor_dict.values():
                                if factor == "Z":
                                    raise Exception("Fermionic matching failed: Z Operator found")
                    
                    qubit_term.simulate(-coeff*t/steps*qubit_operator.terms_dict[qubit_term]*(-1)**int(forward_evolution), new_qarg)
            

    def U(qarg, t=1, steps=1, iter=1):
        merge([qarg])
        with IterationEnvironment(qarg.qs, iter*steps):
            trotter_step(qarg, t, steps)

    return U

# This function takes a list of indices (singles) and a dictionary of indices (couples)
# and generates the swaps to move the K singles to the K lowest positions. It furthermore
# generates the swaps to move all couples adjacent to each other.
def kai_pflaume(singles, couples, n):
    
    permutation = list(range(n))
    swaps = []
    
    singles.sort()
    
    k = 0
    for s in singles:
        # Move the k-th single to position k
        for i in range(k, s)[::-1]:
            swap = (i+1, i)
            permutation[swap[0]], permutation[swap[1]] = permutation[swap[1]], permutation[swap[0]]
            swaps.append(swap)
        k += 1
    
    # The female indices are the indices that are moved towards the males
    # Imagine we have the female 3 and the male 6
    # We start with the permutation
    # [0,1,2,3,4,5,6,7]
    # We now need to move 3 adjacent to 6 and record the necessary swaps
    # We end up in
    # [0,1,2,4,5,3,6,7]
    
    # We begin by moving the female with the highest index first to avoid
    # accidentally moving other females away from their partner.
    females = list(couples.keys())
    females.sort()
    females.reverse()
    
    for f in females:
        
        m = couples[f]
        # Move the female adjacent to the male
        for i in range(permutation.index(f), permutation.index(m)-1):
            swap = (i, i+1)
            permutation[swap[0]], permutation[swap[1]] = permutation[swap[1]], permutation[swap[0]]
            swaps.append(swap)
            
    return swaps, permutation
  
def apply_fermionic_swaps(qarg, swaps):
    from qrisp import cz
    qb_list = list(qarg)
    
    for swap in swaps:
        cz(qb_list[swap[0]], qb_list[swap[1]])
        qb_list[swap[0]], qb_list[swap[1]] = qb_list[swap[1]], qb_list[swap[0]]
        
    return qb_list
    

                    
def apply_fermionic_swap(qv, permutation):
    from qrisp import cz
    qb_list = list(qv)
    swaps = get_swaps_for_permutation(permutation)
    for swap in swaps[::-1]:
        cz(qb_list[swap[0]], qb_list[swap[1]])
        qb_list[swap[0]], qb_list[swap[1]] = qb_list[swap[1]], qb_list[swap[0]]
        
    return qb_list
        
def get_swaps_for_permutation(permutation):
    swaps = []
    permutation = list(permutation)
    for i in range(len(permutation)):
        j = permutation.index(i)
        while j != i:
            permutation[j], permutation[j-1] = permutation[j-1], permutation[j]
            swaps.append((j, j-1))
            j -= 1
    return swaps