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
from qrisp.operators.hamiltonian_tools import group_up_terms
from qrisp import merge, IterationEnvironment, conjugate
from qrisp.operators.qubit import QubitOperator

import sympy as sp

threshold = 1e-9


def fermionic_trotterization(H, forward_evolution = True):
    
    reduced_H = H.reduce(assume_hermitian=True)
    
    groups = reduced_H.group_up(denominator = lambda a,b : a.indices_agree(b) or not a.intersect(b))
    
    
    def trotter_step(qarg, t, steps):
        
        for group in groups:
            
            permutation = []
            terms = list(group.terms_dict.keys())
            
            def sorting_key(term):
                if len(term.ladder_list):
                    return term.ladder_list[-1][-1]
                else:
                    return 0
            terms = sorted(terms, key = sorting_key)
            
            singles = []
            couples = {}
            
            for term in terms:
                
                ladder_list = list(term.ladder_list)
                
                i = 0
                while i < len(ladder_list)-1:
                    if ladder_list[i][0] == ladder_list[i+1][0]:
                        ladder_list.pop(i)
                        ladder_list.pop(i)
                        continue
                    i += 1
                
                if len(ladder_list)%2:
                    singles.append(ladder_list.pop(-1)[0])
                
                for i in range(len(ladder_list)//2):
                    couples[ladder_list[2*i+1][0]] = ladder_list[2*i][0]
                
                
                for ladder in term.ladder_list[::-1]:
                    if ladder[0] not in permutation:
                        permutation.append(ladder[0])
                    # else:
                        # permutation.remove(ladder[0])
            
            for k in range(len(qarg)):
                if k not in permutation:
                    permutation.append(k)
            
            # permutation = permutation[::-1]
            
            swaps, permutation = generate_fermionic_swaps(singles, couples, len(qarg))
            # print(permutation)
            with conjugate(apply_fermionic_swaps)(qarg, swaps) as new_qarg:
                for ferm_term in terms:
                    coeff = reduced_H.terms_dict[ferm_term]
                    pauli_hamiltonian = ferm_term.fermionic_swap(permutation).to_qubit_term()
                    pauli_term = list(pauli_hamiltonian.terms_dict.keys())[0]
                    if len(ferm_term.ladder_list) > 1:
                        for factor in pauli_term.factor_dict.values():
                                if factor == "Z":
                                    raise Exception("Fermionic matching failed: Z Operator found")
                    # print(pauli_term)
                    
                    pauli_term.simulate(-coeff*t/steps*pauli_hamiltonian.terms_dict[pauli_term]*(-1)**int(forward_evolution), new_qarg)
            

    def U(qarg, t=1, steps=1, iter=1):
        merge([qarg])
        with IterationEnvironment(qarg.qs, iter*steps):
            trotter_step(qarg, t, steps)

    return U


def generate_fermionic_swaps(singles, couples, n):
    
    singles.sort()
    permutation = list(range(n))
    
    swaps = []
    k = 0
    for s in singles:
        for i in range(k, s)[::-1]:
            swap = (i+1, i)
            permutation[swap[0]], permutation[swap[1]] = permutation[swap[1]], permutation[swap[0]]
            swaps.append(swap)
        k += 1
    
    females = list(couples.keys())
    females.sort()
    
    for f in females[::-1]:
        for i in range(permutation.index(f), permutation.index(couples[f])-1):
            # print(permutation)
            swap = (i, i+1)
            # print(swap)
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