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

from typing import Self

import numpy as np
import sympy as sp

from qrisp.operators import Hamiltonian
from qrisp.operators.bosonic.bosonic_term import BosonicTerm
from qrisp.operators.hamiltonian_tools import group_up_iterable
from qrisp.operators.qubit import QubitOperator

threshold = 1e-9

#
# BosonicOperator
#


class BosonicOperator(Hamiltonian):
    r"""This class provides an efficient implementation of bosonic ladder term operators, i.e.,
    operators of the form

    .. math::
        
        O=\sum\limits_{j}\alpha_jO_j 
            
    where each term $O_j$ is a product of bosonic raising $a_i^{\dagger}$ and lowering $a_i$ operators acting on the $i$ th bosonic mode.

    The ladder operators satisfy the commutation relations

    .. math::

        [a_i,a_j^{\dagger}] &= a_ia_j^{\dagger}-a_j^{\dagger}a_i = \delta_{ij}\\
        [a_i^{\dagger},a_j^{\dagger}] &= [a_i,a_j] = 0

    Examples
    --------
    A ladder term operator can be specified conveniently in terms of ``a_b`` (lowering, i.e., annihilation), ``c_b`` (raising, i.e., creation) operators:

    ::
        
        from qrisp.operators.bosonic import a_b, c_b

        O = a_b(2)*c_b(1)+a_b(3)*c_b(2)
        O

    Yields $a_2c_1+a_3c_2$.

    """

    def __init__(self, terms_dict: dict = {}):

        self.terms_dict = dict(terms_dict)

    def reduce(self, assume_hermitian: bool = False):
        """Applies the bosonic commutation laws to bring the operator into
        a standard form. This can reduce the amount of terms because several
        terms might be the permuted version of each other and therefore their
        coefficients add up.

        This function can reduce the amount of terms even further if the user
        can guarantee that the operator will be hermitized. In this case more
        identifications can be made.

        Parameters
        ----------
        assume_hermitian : bool, optional
            If set to True the function will assume that the result will be
            hermitized. The default is False.

        Returns
        -------
        BosonicOperator
            The reduced BosonicOperator.

        Examples
        --------
        We create a BosonicOperator with redundant term definitions:

        ::

            from qrisp.operators.bosonic import *

            O = a_b(0)*a_b(1) - a_b(1)*a_b(0)
            print(O.reduce())
            # Yields: 2*a0*a1

        To demonstrate the ``assume_hermitian`` feature, we create a BosonicOperator
        that has redundant terms, if hermitized.


        >>> O = a_b(0)*a_b(1) + c_b(1)*c_b(0)
        >>> reduced_O = O.reduce(assume_hermitian = True)
        >>> print(reduced_O)
        2*a0*a1

        Hermitizing gives the original operator.

        >>> print(reduced_O.hermitize())
        1.0*a0*a1 + 1.0*c1*c0

        """
        # code is adapted from the FermionicOperator class
        new_terms_dict = {}

        for term, coeff in self.terms_dict.items():
            sorted_term = term.sort()
            if sorted_term not in new_terms_dict and assume_hermitian:
                daggered_sorted_term = term.dagger().sort()
                if daggered_sorted_term in new_terms_dict:
                    sorted_term = daggered_sorted_term

            # Compute the new coefficient.
            new_terms_dict[sorted_term] = coeff + new_terms_dict.get(sorted_term, 0)

        for term, coeff in list(new_terms_dict.items()):
            if isinstance(coeff, (int, float)):
                if coeff == 0:
                    del new_terms_dict[term]

        return BosonicOperator(new_terms_dict)

    def len(self):
        return len(self.terms_dict)

    def coeffs(self):
        """Returns the coefficients of the operator.

        Returns
        -------
        ndarray
            The coefficients.

        Examples
        --------
        >>> from qrisp.operators import a_b, c_b
        >>> O = a_b(0)*c_b(1)+a_b(1)*c_b(0)+0.5*a_b(1)+0.5*c_b(1)
        >>> O.coeffs()
        array([1. , 1. , 0.5, 0.5])

        """
        return np.array(list(self.terms_dict.values()))

    #
    # Printing
    #

    def _repr_latex_(self):
        # Convert the sympy expression to LaTeX and return it
        expr = self.to_expr()
        return f"${sp.latex(expr)}$"

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)

    def to_expr(self):
        """Returns a SymPy expression representing the operator.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the operator.

        """
        expr = 0
        for ladder_term, coeff in self.terms_dict.items():
            expr += coeff * ladder_term.to_expr()
        return expr

    #
    # Arithmetic
    #

    def dagger(self):
        r"""Returns the daggered/adjoint version of self.

        Returns
        -------
        BosonicOperator
            The Operator $O^\dagger$.

        Examples
        --------
        We create a BosonicOperator and dagger it:

        ::

            from qrisp.operators import *

            O = a_b(0)*c_b(1)*a_b(2) + a_b(3)
            print(O.dagger())
            # Yields: c2*a1*c0 + c3

        """
        terms_dict = {}
        for term, coeff in self.terms_dict.items():
            terms_dict[term.dagger()] = np.conj(coeff)
        return BosonicOperator(terms_dict)

    def hermitize(self):
        r"""Returns the hermitized version of self.

        Returns
        -------
        BosonicOperator
            The Operator $(O + O^\dagger)/2$.

        Examples
        --------
        We create a BosonicOperator and hermitize it:

        ::

            from qrisp.operators import *

            O = a_b(0)*c_b(1)*a_b(2) + a_b(3)
            print(O.hermitize())
            # Yields: 0.5*a0*c1*a2 + 0.5*a3 + 0.5*c2*a1*c0 + 0.5*c3

        """
        return 0.5 * (self + self.dagger())

    def __eq__(self, other: Self):
        reduced_self = self.reduce()
        reduced_other = other.reduce()

        if len(reduced_self.terms_dict) != len(reduced_other.terms_dict):
            return False

        for term, coeff in reduced_self.terms_dict.items():
            if term not in other.terms_dict:
                daggered_sorted_term = term.dagger().sort()
                if daggered_sorted_term not in reduced_other.terms_dict:
                    return False
                elif reduced_self.terms_dict[term] != reduced_other.terms_dict[daggered_sorted_term]:
                    return False
                continue

            if reduced_self.terms_dict[term] != reduced_other.terms_dict[term]:
                return False

        return True

    def __neg__(self):
        return -1 * self

    def __add__(self, other: Self):
        """Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to add to the operator self.

        Returns
        -------
        result : BosonicOperator
            The sum of the operator self and other.

        """
        if isinstance(other, (int, float, complex)):
            other = BosonicOperator({BosonicTerm(): other})
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot add BosonicOperator and " + str(type(other)))

        res_terms_dict = {}

        for ladder_term, coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) + coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        for ladder_term, coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) + coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        result = BosonicOperator(res_terms_dict)
        return result

    def __sub__(self, other: Self):
        """Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to substract from the operator self.

        Returns
        -------
        result : BosonicOperator
            The difference of the operator self and other.

        """
        if isinstance(other, (int, float, complex)):
            other = BosonicOperator({BosonicTerm(): other})
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot substract BosonicOperator and " + str(type(other)))

        res_terms_dict = {}

        for ladder_term, coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) + coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        for ladder_term, coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) - coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        result = BosonicOperator(res_terms_dict)
        return result

    def __rsub__(self, other: Self):
        """Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to substract from the operator self from.

        Returns
        -------
        result : BosonicOperator
            The difference of the operator other and self.

        """
        if isinstance(other, (int, float, complex)):
            other = BosonicOperator({BosonicTerm(): other})
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot substract BosonicOperator and " + str(type(other)))

        res_terms_dict = {}

        for ladder_term, coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) - coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        for ladder_term, coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term, 0) + coeff
            if abs(res_terms_dict[ladder_term]) < threshold:
                del res_terms_dict[ladder_term]

        result = BosonicOperator(res_terms_dict)
        return result

    def __mul__(self, other: Self):
        """Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to multiply with the operator self.

        Returns
        -------
        result : BosonicOperator
            The product of the operator self and other.

        """
        if isinstance(other, (int, float, complex)):
            other = BosonicOperator({BosonicTerm(): other})
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot multipliy BosonicOperator and " + str(type(other)))

        res_terms_dict = {}

        for ladder_term1, coeff1 in self.terms_dict.items():
            for ladder_term2, coeff2 in other.terms_dict.items():
                curr_ladder_term = ladder_term1 * ladder_term2
                res_terms_dict[curr_ladder_term] = res_terms_dict.get(curr_ladder_term, 0) + coeff1 * coeff2

        result = BosonicOperator(res_terms_dict)
        return result

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, exp: int):
        """Returns the operator self exponentiated by int.

        Parameters
        ----------
        exp : int
            A positive integer with which self is exponentiated.

        Returns
        -------
        result : BosonicOperator
            The exponentiated operator self.

        """
        if not (isinstance(exp, int) and exp > 0):
            raise TypeError("Operators can be exponentiated only with positive integers.")

        res = self
        for _ in range(exp - 1):
            res = res * self

        return res

    #
    # Inplace arithmetic
    #

    def __iadd__(self, other: Self):
        """Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to add to the operator self.

        """
        if isinstance(other, (int, float, complex)):
            self.terms_dict[BosonicTerm()] = self.terms_dict.get(BosonicTerm(), 0) + other
            return self
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot add BosonicOperator and " + str(type(other)))

        for ladder_term, coeff in other.terms_dict.items():
            self.terms_dict[ladder_term] = self.terms_dict.get(ladder_term, 0) + coeff
            if abs(self.terms_dict[ladder_term]) < threshold:
                del self.terms_dict[ladder_term]
        self.terms_dict = BosonicOperator(self.terms_dict).terms_dict
        return self

    def __isub__(self, other: Self):
        """Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to substract from the operator self.

        """
        if isinstance(other, (int, float, complex)):
            self.terms_dict[BosonicTerm()] = self.terms_dict.get(BosonicTerm(), 0) - other
            return self
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot add BosonicOperator and " + str(type(other)))

        for ladder_term, coeff in other.terms_dict.items():
            self.terms_dict[ladder_term] = self.terms_dict.get(ladder_term, 0) - coeff
            if abs(self.terms_dict[ladder_term]) < threshold:
                del self.terms_dict[ladder_term]
        return self

    def __imul__(self, other: Self):
        """Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or BosonicOperator
            A scalar or a BosonicOperator to multiply with the operator self.

        """
        if isinstance(other, (int, float, complex)):
            other = BosonicOperator({BosonicTerm(): other})
        if not isinstance(other, BosonicOperator):
            raise TypeError("Cannot multipliy BosonicOperator and " + str(type(other)))

        res_terms_dict = {}

        for ladder_term1, coeff1 in self.terms_dict.items():
            for ladder_term2, coeff2 in other.terms_dict.items():
                curr_ladder_term = ladder_term1 * ladder_term2
                res_terms_dict[curr_ladder_term] = res_terms_dict.get(curr_ladder_term, 0) + coeff1 * coeff2

        self.terms_dict = res_terms_dict

    #
    # Miscellaneous
    #

    def apply_threshold(self, threshold: float):
        """Removes all ladder_term terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the ladder_term terms.

        """
        delete_list = []
        for ladder_term, coeff in self.terms_dict.items():
            if abs(coeff) < threshold:
                delete_list.append(ladder_term)
        for ladder_term in delete_list:
            del self.terms_dict[ladder_term]

    def to_sparse_matrix(self, truncation: int = 8, binary_encoding: str = "gray_code"):
        """Returns a matrix representing the operator.

        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.
        truncation: How many bosonic occupation numbers to take into account.
        binary_encoding : string, optional
            How to embed the bosonic terms into a QubitOperator. Possible values are "gray_code", "standard_binary" and "one_hot".

        """
        return self.to_qubit_operator(truncation=truncation, binary_encoding=binary_encoding).to_sparse_matrix()

    def ground_state_energy(self, truncation: int = 8):
        """Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.

        Returns
        -------
        float
            The ground state energy.

        """
        return self.to_qubit_operator(truncation=truncation).ground_state_energy()

    def to_qubit_operator(self, truncation: int = 8, binary_encoding: str = "gray_code"):
        """Transforms the BosonicOperator to a :ref:`QubitOperator`.

        Parameters
        ----------
        truncation : int, optional
            How many bosonic occupation numbers to take into account
        binary_encoding : str, optional
            How to embed the bosonic terms into a QubitOperator. Possible values are "gray_code", "standard_binary" and "one_hot".

        Returns
        -------
        O : :ref:`QubitOperator`
            The resulting QubitOperator.
        """
        if binary_encoding in ["gray_code", "standard_binary", "one_hot"]:
            res = QubitOperator({})
            for term, coeff in self.terms_dict.items():
                res += coeff * term.to_qubit_term(truncation=truncation, binary_encoding=binary_encoding)
            return res
        else:
            raise Exception(f"Don't know bosonic mapping {binary_encoding}.")

    def expectation_value(
        self, state_prep: callable, truncation: int = 8, binary_encoding: str = "gray_code", **measurement_kwargs
    ):
        r"""The ``expectation value`` function allows to estimate the expectation value of a Hamiltonian for a state that is specified by a preparation procedure.
        This preparation procedure can be supplied via a Python function that returns a :ref:`QuantumVariable`.

        Note that this method measures the **hermitized** version of the operator:

        .. math::

            H = (O + O^\dagger)/2


        Parameters
        ----------
        state_prep : callable
            A function returning a QuantumVariable.
            The expectation of the Hamiltonian for the state of this QuantumVariable will be measured.
            The state preparation function can only take classical values as arguments.
            This is because a quantum value would need to be copied for each sampling iteration, which is prohibited by the no-cloning theorem.
        truncation: How many bosonic occupation numbers to take into account.
        binary_encoding : string, optional
            How to embed the bosonic terms into a QubitOperator. Possible values are "gray_code", "standard_binary" and "one_hot".
        measurement_kwargs : dict, optional
            The keyword arguments of :meth:`QubitOperator.expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>`.

        Returns
        -------
        callable
            A function returning an array containing the expectation value.

        """
        qubit_operator = self.to_qubit_operator(truncation=truncation, binary_encoding=binary_encoding)
        return qubit_operator.expectation_value(state_prep, **measurement_kwargs)

    #
    # Trotterization
    #

    def trotterization(self, truncation: int = 8, binary_encoding: str = "gray_code", forward_evolution: bool = True):
        r"""Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $U(t) = e^{-itH}$ via Trotterization."""
        qubit_operator = self.to_qubit_operator(truncation=truncation, binary_encoding=binary_encoding)
        return qubit_operator.trotterization(forward_evolution=forward_evolution)

    def group_up(self, denominator: callable):
        term_groups = group_up_iterable(list(self.terms_dict.keys()), denominator)
        if len(term_groups) == 0:
            return [self]
        groups = []
        for term_group in term_groups:
            O = BosonicOperator({term: self.terms_dict[term] for term in term_group})
            groups.append(O)

        return groups


def get_bosonic_encoding_qubit_number(truncation, binary_encoding):
    if binary_encoding != "one_hot":
        return int(np.ceil(np.log2(truncation)))
    else:
        return truncation


from qrisp.operators.bosonic.bosonic_term import gray_code, standard_binary, one_hot
from qrisp import QuantumVariable, x


def prepare_bosonic_fock_state(n: int, truncation: int = 8, binary_encoding: str = "gray_code"):
    if not 0 <= n < truncation:
        raise ValueError("n must be between 0 an truncation-1")

    n_qubits = get_bosonic_encoding_qubit_number(truncation, binary_encoding)

    qv = QuantumVariable(n_qubits)

    if binary_encoding == "gray_code":
        qubits = gray_code(n_qubits)[n]
    elif binary_encoding == "standard_binary":
        qubits = standard_binary(n_qubits)[n]
    elif binary_encoding == "one_hot":
        qubits = one_hot(n_qubits)[n]
    else:
        raise Exception(f"Don't know binary encoding type {binary_encoding}")

    for i, q in enumerate(qubits):
        if q:
            x(qv[i])

    return qv
