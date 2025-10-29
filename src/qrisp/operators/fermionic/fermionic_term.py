"""
********************************************************************************
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

import numpy as np

from qrisp.operators.fermionic.visualization import a_, c_
from qrisp.operators.qubit import A, C, Z

#
# FermionicTerm
#


class FermionicTerm:
    r""" """

    def __init__(self, ladder_list=[]):

        self.ladder_list = ladder_list

        # Compute the hash value such that
        # terms receive the same hash as their hermitean conjugate
        # this way the FermionicOperator does not have
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
        return FermionicTerm(self.ladder_list.copy())

    def dagger(self):
        return FermionicTerm(
            [(index, not is_creator) for index, is_creator in self.ladder_list[::-1]]
        )

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
        """
        Returns a SymPy expression representing the FermionicTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the FermionicTerm.

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
        return FermionicTerm(result_ladder_list)

    def order(self):
        """
        Not that important, since relevant Hamiltonians (e.g., electronic structure) consist of ordered terms.
        What is needed for trotterization?

        Fermionic commutation relations:

        {a_i,a_j^dagger} = a_i*a_j^dagger + a_j^dagger*a_i = delta_{ij}
        {a_i^dagger,a_j^dagger} = {a_i,a_j} = 0


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
        flip_sign = permutation_signature(perm)

        for i in range(len(ladder_list) - 1):
            if ladder_list[i][0] == ladder_list[i + 1][0]:
                if ladder_list[i][1] == ladder_list[i + 1][1]:
                    flip_sign = 0
                    break
                if i > 1 and ladder_list[i][0] == ladder_list[i - 1][0]:
                    flip_sign = 0
                    break

        return FermionicTerm(ladder_list), flip_sign

    def fermionic_swap(self, permutation):

        permutation = [permutation.index(i) for i in range(len(permutation))]
        new_ladder_list = [
            (permutation[i], is_creator) for i, is_creator in self.ladder_list
        ]

        return FermionicTerm(new_ladder_list)

    def unipolars_intersect(self, other):
        """
        Checks if two terms have intersecting unipolar factos.
        Unipolar factors are factors that are not of the form a(i)*c(i),
        i.e. the index i appears only once.
        """
        return len(set(self.get_unipolars()).intersection(other.get_unipolars())) != 0

    def unipolars_agree(self, other):
        """
        Checks if two terms have intersecting unipolar factos.
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

    def to_qubit_term(self, mapping_type="jordan_wigner"):
        if mapping_type == "jordan_wigner":
            res = 1
            for i in range(len(self.ladder_list)):
                temp = 1
                for j in range(self.ladder_list[i][0]):
                    temp = Z(j) * temp

                if self.ladder_list[i][1]:
                    temp = temp * C(self.ladder_list[i][0])
                else:
                    temp = temp * A(self.ladder_list[i][0])

                res = temp * res
            if res == 1:
                from qrisp.operators import QubitOperator, QubitTerm

                return QubitOperator({QubitTerm(): 1})

            return res
        else:
            raise Exception(f"Don't know fermionic mapping type {mapping_type}")


def permutation_signature(perm):

    k = 0
    for i in range(len(perm)):
        for j in range(i):
            k += int(perm[i] < perm[j])

    return (-1) ** (k % 2)
