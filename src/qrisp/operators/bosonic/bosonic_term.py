"""********************************************************************************
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
********************************************************************************
"""

#
# BosonicTerm
#
import numpy as np

from qrisp.operators.fermionic.visualization import a_, c_
from qrisp.operators.qubit import A, C, Z, P0, P1


class BosonicTerm:
    r""" """

    def __init__(self, ladder_list=[]):

        self.ladder_list = ladder_list

        # Compute the hash value such that
        # terms receive the same hash as their hermitean conjugate
        # this way the BosonicOperator does not have
        # to track both the term and it's dagger
        index_list = [index for index, is_creator in ladder_list]
        is_creator_hash = 0
        for i in range(len(ladder_list)):
            is_creator_hash += ladder_list[i][1] * 2**i

        self.hash_value = hash(tuple(index_list + [is_creator_hash]))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value

    def copy(self):
        return BosonicTerm(self.ladder_list.copy())

    def dagger(self):
        return BosonicTerm([(index, not is_creator) for index, is_creator in self.ladder_list[::-1]])

    #
    # Printing
    #

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)

    def __repr__(self):
        return str(self)

    def to_expr(self):
        """Returns a SymPy expression representing the BosonicTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the BosonicTerm.

        """

        def to_ladder(value, index):
            if value:
                return c_(index)
            else:
                return a_(index)

        expr = 1
        for index, value in self.ladder_list[::-1]:
            expr *= to_ladder(value, str(index))

        return expr

    #
    # Arithmetic
    #

    def __mul__(self, other):
        result_ladder_list = other.ladder_list + self.ladder_list
        return BosonicTerm(result_ladder_list)

    def order(self):
        """Not that important, since relevant Hamiltonians (e.g., electronic structure) consist of ordered terms.
        What is needed for trotterization?

        Bosonic commutation relations:

        [a_i,a_j^dagger] = a_i*a_j^dagger - a_j^dagger*a_i = delta_{ij}
        [a_i^dagger,a_j^dagger] = [a_i,a_j] = 0


        Order ladder terms such that
            1) Raising operators preceed lowering operators
            2) Operators are ordered in descending order of fermionic modes

        Example: a_5^dagger a_2^dagger a_3 a_1

        """
        pass

    def sort(self):
        # Sort ladder operators (ladder operator semantics are order in-dependent)
        sorting_list = [-index for index, is_creator in self.ladder_list]
        perm = np.argsort(sorting_list, kind="stable")
        ladder_list = [self.ladder_list[i] for i in perm]

        return BosonicTerm(ladder_list)

    def bosonic_swap(self, permutation):

        permutation = [permutation.index(i) for i in range(len(permutation))]
        new_ladder_list = [(permutation[i], is_creator) for i, is_creator in self.ladder_list]

        return BosonicTerm(new_ladder_list)

    def unipolars_intersect(self, other):
        """Checks if two terms have intersecting unipolar factos.
        Unipolar factors are factors that are not of the form a(i)*c(i),
        i.e. the index i appears only once.
        """
        return len(set(self.get_unipolars()).intersection(other.get_unipolars())) != 0

    def unipolars_agree(self, other):
        """Checks if two terms have intersecting unipolar factos.
        Unipolar factors are factors that are not of the form a(i)*c(i),
        i.e. the index i appears only once.
        """
        return set(self.get_unipolars()) == set(other.get_unipolars())

    def get_unipolars(self):
        if hasattr(self, "unipolars"):
            return list(self.unipolars)
        else:
            index_list = [index for index, is_creator in self.ladder_list]
            index_list.sort()

            i = 0
            while i < len(index_list) - 1:
                if index_list[i] == index_list[i + 1]:
                    index_list.pop(i)
                    index_list.pop(i)
                    continue
                i += 1

            self.unipolars = index_list[::-1]
            return list(self.unipolars)

    def to_qubit_term(self, truncation=8, binary_encoding="gray_code"):
        """Maps a bosonic term to a qubit term. 
        Since bosonic operators act on an infinite-dimensional space, a truncation to a finite
        number of bosonic occupation numbers is necessary (provided by the "truncation" argument).
        For their embedding into qubits, an arbitrary binary encoding can be chosen,
        but the Gray encoding appears tailor-made for the structure of ladder operators.
        """
        if not np.isclose(np.log2(truncation)%1, 0):
            raise Warning("truncation is not a power of 2, could be chosen larger with same amount of qubits.")
        if binary_encoding == "gray_code":
            encoder = gray_code
        else:
            raise Exception(f"Don't know binary encoding type {binary_encoding}")
        
        indices_present = set([x[0] for x in ladder_list])
        
        n_qubits = int(np.log2(truncation)) + 1
        
        binary_rep = encoder(n_qubits)
        
        res = 1
        
        for ind in indices_present:
            ladder_ops = [x[1] for x in self.ladder_list if x[0] == ind]
            #bring term into matrix form
            M = np.identity(truncation)
            for l in ladder_ops:
                if l:
                    M = c_matrix(truncation) @ M
                else:
                    M = a_matrix(truncation) @ M

            temp = 0
            for i in range(truncation):
                for j in range(truncation):
                    if not np.isclose(M[i][j], 0.):
                        for k in range(n_qubits):
                            c1, c2 = binary_rep[i][k], binary_rep[j][k]
                            qb_ind = ind*n_qubits+k
                            if (c1, c2) == (0, 0):
                                temp += M[i][j] * P0(qb_ind)
                            elif (c1, c2) == (1, 1):
                                temp += M[i][j] * P1(qb_ind)
                            elif (c1, c2) == (0, 1):
                                temp += M[i][j] * A(qb_ind)
                            elif (c1, c2) == (1, 0):
                                temp += M[i][j] * C(qb_ind)
            
            res *= temp
      
        return res
  
#Bosonic annihilation operator in matrix representation    
def a_matrix(N):
    return np.diag(
            np.sqrt(np.arange(1, N)), 
            k=1
        ).astype(complex)

#Bosonic creation operator in matrix representation            
def c_matrix(N):
    return np.diag(
            np.sqrt(np.arange(1, N)),
            k=-1
        ).astype(complex)

#Compute the Gray code
def gray_code(n):
    code = []
    for i in range(n):
        temp = []
        block1 = (2**i)*[0]+(2**i)*[1]
        block2 = (2**i)*[1]+(2**i)*[0]
        for j in range(2**(n-i-1)):
            if j%2==0:
                temp += block1
            else:
                temp += block2
        code.append(temp)

    return np.transpose(np.asarray(code)).tolist()
