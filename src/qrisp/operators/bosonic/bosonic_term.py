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
import warnings
from typing import Self

from qrisp.operators.bosonic.visualization import a_, c_
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

    def __eq__(self, other: Self):
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

    def __mul__(self, other: Self):
        result_ladder_list = other.ladder_list + self.ladder_list
        return BosonicTerm(result_ladder_list)

    def sort(self):
        # Sort ladder operators (ladder operator semantics are order independent)
        sorting_list = [-index for index, is_creator in self.ladder_list]
        perm = np.argsort(sorting_list, kind="stable")
        ladder_list = [self.ladder_list[i] for i in perm]

        return BosonicTerm(ladder_list)

    def to_qubit_term(self, truncation: int = 8, binary_encoding: str = "gray_code"):
        """Maps a bosonic term to a qubit term.
        Since bosonic operators act on an infinite-dimensional space, a truncation to a finite
        number of bosonic occupation numbers is necessary (provided by the "truncation" argument).
        For their embedding into qubits, an arbitrary binary encoding can be chosen,
        but the Gray encoding appears tailor-made for the structure of ladder operators.
        Apart from the Gray encoding, a standard binary and a one-hot encoding are provided.
        https://www.nature.com/articles/s41534-020-0278-0 contains more details on their respective advantages and disadvantages.

        Parameters
        ----------
        truncation: int, optional
            The number of bosonic occupation numbers one wants to describe.
            Note that 0 also counts as an occupation number, so occupation numbers from 0 to truncation-1 are represented.
        binary_encoding: str, optional
            How to embed the bosonic matrix into qubits. Possible values are "gray_code", "standard_binary" and "one_hot".
        """
        if not np.isclose(np.log2(truncation) % 1, 0) and binary_encoding != "one_hot":
            warnings.warn("truncation is not a power of 2, could be chosen larger with same amount of qubits.")
        if binary_encoding == "gray_code":
            encoder = gray_code
        elif binary_encoding == "standard_binary":
            encoder = standard_binary
        elif binary_encoding == "one_hot":
            encoder = one_hot
        else:
            raise Exception(f"Don't know binary encoding type {binary_encoding}")

        gate_mapping = {(0, 0): P0, (1, 1): P1, (0, 1): A, (1, 0): C}

        indices_present = set([x[0] for x in self.ladder_list])

        if binary_encoding != "one_hot":
            n_qubits = int(np.ceil(np.log2(truncation)))
        else:
            n_qubits = truncation

        binary_rep = encoder(n_qubits)

        res = 1

        for ind in indices_present:
            ladder_ops = [x[1] for x in self.ladder_list if x[0] == ind]
            # bring term into matrix form
            M = np.identity(truncation)
            for l in ladder_ops:
                if l:
                    M = c_matrix(truncation) @ M
                else:
                    M = a_matrix(truncation) @ M

            temp = 0
            for i in range(truncation):
                for j in range(truncation):
                    if not np.isclose(M[i][j], 0.0):
                        temp2 = 1
                        for k in range(n_qubits):
                            c1, c2 = binary_rep[i][k], binary_rep[j][k]
                            qb_ind = ind * n_qubits + k
                            temp2 *= gate_mapping[(c1, c2)](qb_ind)
                        temp += M[i][j] * temp2

            res *= temp

        return res


# Bosonic annihilation operator in matrix representation
def a_matrix(N: int):
    return np.diag(np.sqrt(np.arange(1, N)), k=1).astype(complex)


# Bosonic creation operator in matrix representation
def c_matrix(N: int):
    return np.diag(np.sqrt(np.arange(1, N)), k=-1).astype(complex)


# Return the Gray code
def gray_code(n: int):
    code = []
    for i in range(n):
        temp = []
        block1 = (2 ** (n - i - 1)) * [0] + (2 ** (n - i - 1)) * [1]
        block2 = (2 ** (n - i - 1)) * [1] + (2 ** (n - i - 1)) * [0]
        for j in range(2**i):
            if j % 2 == 0:
                temp += block1
            else:
                temp += block2
        code.append(temp)

    return np.transpose(np.asarray(code)).tolist()


from itertools import product


def standard_binary(n: int):
    # product returns tuples like (0, 1, 0), so we convert them to lists
    return [list(bits) for bits in product((0, 1), repeat=n)]


# Return one-hot encoding
def one_hot(n: int):
    code = []
    for i in range(n):
        code.append(i * [0] + [1] + (n - i - 1) * [0])
    return code
