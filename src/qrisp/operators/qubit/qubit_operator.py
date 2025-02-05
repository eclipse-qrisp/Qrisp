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

from itertools import product

import sympy as sp
import numpy as np
import jax.numpy as jnp

from qrisp.operators.hamiltonian_tools import group_up_iterable
from qrisp.operators.hamiltonian import Hamiltonian
from qrisp.operators.qubit.qubit_term import QubitTerm
from qrisp.operators.qubit.measurement import get_measurement
from qrisp.operators.qubit.commutativity_tools import construct_change_of_basis
from qrisp import cx, cz, h, s, sx_dg, IterationEnvironment, conjugate, merge

from qrisp.jasp import check_for_tracing_mode, jrange


threshold = 1e-9

#
# QubitOperator
#

class QubitOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of QubitOperators, i.e.
    Operators, that act on a qubit space :math:`(\mathbb{C}^2)^{\otimes n}`.
    Supported are operators of the following form:
    
    .. math::
        
        O=\sum\limits_{j}\alpha_j O_j 
        
    where :math:`O_j=\bigotimes_i m_i^j` is a product of the following operators:
    
    .. list-table::
       :header-rows: 1
       :widths: 20 40 40
    
       * - Operator
         - Ket-Bra Realization
         - Description
       * - $X$
         - :math:`\ket{0}\bra{1} + \ket{1}\bra{0}`
         - Pauli-X operator (bit flip)
       * - $Y$
         - :math:`-i\ket{0}\bra{1} + i\ket{1}\bra{0}`
         - Pauli-Y operator (bit flip with phase)
       * - $Z$
         - :math:`\ket{0}\bra{0} - \ket{1}\bra{1}`
         - Pauli-Z operator (phase flip)
       * - $A$
         - :math:`\ket{0}\bra{1}`
         - Annihilation operator
       * - $C$
         - :math:`\ket{1}\bra{0}`
         - Creation operator
       * - $P_0$
         - :math:`\ket{0}\bra{0}`
         - Projector onto the :math:`\ket{0}` state
       * - $P_1$
         - :math:`\ket{1}\bra{1}`
         - Projector onto the :math:`\ket{1}` state
       * - $I$
         - :math:`\ket{1}\bra{1} + \ket{0}\bra{0}`
         - Identity operator
    
    If you already have some experience you might wonder why to include the
    non-Pauli operators - after all they can be represented as a linear
    combination of ``X``, ``Y`` and ``Z``. 
    
    .. math::
        
        \begin{align}
        A_0 C_1 &= (X_0 - i Y_0)(X_1 + Y_1)/4 \\
        & = (X_0X_1 + X_0Y_1 - Y_0X_1 + Y_0Y_1)/4
        \end{align}
    
    Recently, a much more efficient method of simulating ``A`` and ``C`` `has 
    been proposed by Kornell and Selinger <https://arxiv.org/abs/2310.12256>`_,
    which avoids decomposing these Operators into Paulis strings
    but instead simulates 
    
    .. math::
        H = A_0C_1 + h.c. 
        
    within a single step.
    This idea is deeply integrated into the Operators module of Qrisp. For an
    example circuit see below.

    Examples
    --------
    
    A QubitOperator can be specified conveniently in terms of arithmetic 
    combinations of the mentioned operators:

    ::
        
        from qrisp.operators.qubit import X,Y,Z,A,C,P0,P1

        H = 1+2*X(0)+3*X(0)*Y(1)*A(2)+C(4)*P1(0)
        H

    Yields $1 + P^1_0C_4 + 2X_0 + 3X_0Y_1A_2$.

    We create a QubitOperator and perform Hamiltonian simulation via :meth:`trotterization <QubitOperator.trotterization>`:

    ::
        
        from sympy import Symbol
        from qrisp.operators import A,C,Z,Y
        from qrisp import QuantumVariable
        O = A(0)*C(1)*Z(2)*A(3) + Y(3)
        U = O.trotterization()
        qv = QuantumVariable(4)
        t = Symbol("t")
        U(qv, t = t)
        
    >>> print(qv.qs)
    QuantumCircuit:
    ---------------
              ┌───┐                                                                »
        qv.0: ┤ X ├────────────o──────────────────────────────────────o────────────»
              └─┬─┘┌───┐       │                                      │       ┌───┐»
        qv.1: ──┼──┤ X ├───────■──────────────────────────────────────■───────┤ X ├»
                │  └─┬─┘       │                                      │       └─┬─┘»
        qv.2: ──┼────┼─────────┼────■────────────────────────────■────┼─────────┼──»
                │    │  ┌───┐  │  ┌─┴─┐     ┌────────────┐     ┌─┴─┐  │  ┌───┐  │  »
        qv.3: ──■────■──┤ H ├──┼──┤ X ├──■──┤ Rz(-0.5*t) ├──■──┤ X ├──┼──┤ H ├──■──»
                        └───┘┌─┴─┐└───┘┌─┴─┐├───────────┬┘┌─┴─┐└───┘┌─┴─┐└───┘     »
    hs_anc.0: ───────────────┤ X ├─────┤ X ├┤ Rz(0.5*t) ├─┤ X ├─────┤ X ├──────────»
                             └───┘     └───┘└───────────┘ └───┘     └───┘          »
    «          ┌───┐                            
    «    qv.0: ┤ X ├────────────────────────────
    «          └─┬─┘                            
    «    qv.1: ──┼──────────────────────────────
    «            │                              
    «    qv.2: ──┼──────────────────────────────
    «            │  ┌────┐┌────────────┐┌──────┐
    «    qv.3: ──■──┤ √X ├┤ Rz(-2.0*t) ├┤ √Xdg ├
    «               └────┘└────────────┘└──────┘
    «hs_anc.0: ─────────────────────────────────
    «                                           
    Live QuantumVariables:
    ----------------------
    QuantumVariable qv
    
    Call the simulator:
        
    >>> print(qv.get_measurement(subs_dic = {t : 0.5}))
    {'0000': 0.77015, '0001': 0.22985}

    """

    def __init__(self, terms_dict={}):
        self.terms_dict = dict(terms_dict)

    def len(self):
        return len(self.terms_dict)
    
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

    def __repr__(self):
        # Convert the sympy expression to a string and return it
        return str(self)
    
    def to_expr(self):
        """
        Returns a SymPy expression representing the operator.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the operator.

        """
        
        expr = 0  
        for term, coeff in self.terms_dict.items():
            expr += coeff*term.to_expr()
        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        res = 1
        for i in range(e):
            res = res * self
        return res    

    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to add to the operator self.

        Returns
        -------
        result : QubitOperator
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitOperator({QubitTerm():other})
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot add QubitOperator and "+str(type(other)))

        res_terms_dict = {}

        for term,coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitOperator(res_terms_dict)
        return result
    
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to substract from the operator self.

        Returns
        -------
        result : QubitOperator
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitOperator({QubitTerm():other})
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot substract QubitOperator and "+str(type(other)))

        res_terms_dict = {}

        for term, coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)-coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitOperator(res_terms_dict)
        return result
    
    def __rsub__(self,other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to substract the operator self from.

        Returns
        -------
        result : QubitOperator
            The difference of the operator other and self.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitOperator({QubitTerm():other})
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot substract QubitOperator and "+str(type(other)))

        res_terms_dict = {}

        for term,coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)-coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitOperator(res_terms_dict)
        return result

    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to multiply with the operator self.

        Returns
        -------
        result : QubitOperator
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitOperator({QubitTerm():other})
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot multipliy QubitOperator and "+str(type(other)))

        res_terms_dict = {}

        for term1, coeff1 in self.terms_dict.items():
            for term2, coeff2 in other.terms_dict.items():
                curr_term, curr_coeff = term1*term2
                res_terms_dict[curr_term] = res_terms_dict.get(curr_term,0) + curr_coeff*coeff1*coeff2

        result = QubitOperator(res_terms_dict)
        return result

    __radd__ = __add__
    __rmul__ = __mul__

    #
    # Inplace arithmetic
    #

    def __iadd__(self,other):
        """
        Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[QubitTerm()] = self.terms_dict.get(QubitTerm(),0)+other
            return self
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot add QubitOperator and "+str(type(other)))

        for term,coeff in other.terms_dict.items():
            self.terms_dict[term] = self.terms_dict.get(term,0)+coeff
            if abs(self.terms_dict[term])<threshold:
                del self.terms_dict[term]       
        return self         

    def __isub__(self,other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[QubitTerm()] = self.terms_dict.get(QubitTerm(),0)-other
            return self
        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot add QubitOperator and "+str(type(other)))

        for term,coeff in other.terms_dict.items():
            self.terms_dict[term] = self.terms_dict.get(term,0)-coeff
            if abs(self.terms_dict[term])<threshold:
                del self.terms_dict[term]  
        return self
    
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or QubitOperator
            A scalar or a QubitOperator to multiply with the operator self.

        """

        if isinstance(other,(int,float,complex)):
            #other = QubitOperator({QubitTerm():other})
            for term in self.terms_dict:
                self.terms_dict[term] *= other
            return self

        if not isinstance(other,QubitOperator):
            raise TypeError("Cannot multipliy QubitOperator and "+str(type(other)))

        res_terms_dict = {}

        for term1, coeff1 in self.terms_dict.items():
            for term2, coeff2 in other.terms_dict.items():
                curr_term, curr_coeff = term1*term2
                res_terms_dict[curr_term] = res_terms_dict.get(curr_term,0) + curr_coeff*coeff1*coeff2

        self.terms_dict = res_terms_dict    
        return self

    #
    # Substitution
    #

    def subs(self, subs_dict):
        """
        
        Parameters
        ----------
        subs_dict : dict
            A dictionary with indices (int) as keys and numbers (int, float, complex) as values.

        Returns
        -------
        result : QubitOperator
            The resulting QubitOperator.
        
        """

        res_terms_dict = {}

        for term, coeff in self.terms_dict.items():
            curr_term, curr_coeff = term.subs(subs_dict)
            res_terms_dict[curr_term] = res_terms_dict.get(curr_term,0) + curr_coeff*coeff

        return QubitOperator(res_terms_dict)

    #
    # Miscellaneous
    #
    
    def find_minimal_qubit_amount(self):
        indices = sum([list(term.factor_dict.keys()) for term in self.terms_dict.keys()], [])
        if len(indices) == 0:
            return 0
        return max(indices)+1
    
    def commutator(self, other):
        """
        Computes the commutator.
        
        .. math::
            
            [A,B] = AB - BA

        Parameters
        ----------
        other : QubitOperator
            The second argument of the commutator.

        Returns
        -------
        commutator : QubitOperator
            The commutator operator.
            
        Examples
        --------
        
        We compute the commutator of a ladder operator with a Pauli string.
        
        >>> from qrisp.operators import A,C,X,Z
        >>> O_0 = A(0)*C(1)*A(2)
        >>> O_1 = Z(0)*X(1)*X(1)
        >>> print(O_0.commutator(O_1))
        2*A_0*C_1*A_2
        

        """
        
        res = 0
        
        for term_self, coeff_self in self.terms_dict.items():
            for term_other, coeff_other in other.terms_dict.items():
                res += coeff_self*coeff_other*term_self.commutator(term_other)
                
        min_coeff_self = min([abs(coeff) for coeff in self.terms_dict.values()])
        min_coeff_other = min([abs(coeff) for coeff in other.terms_dict.values()])
        
        res.apply_threshold(min_coeff_self*min_coeff_other/2)
        
        return res    

    def apply_threshold(self,threshold):
        """
        Removes all terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the terms.

        """

        delete_list = []
        new_terms_dict = dict(self.terms_dict)
        for term,coeff in self.terms_dict.items():
            if abs(coeff)<threshold:
                delete_list.append(term)
        for term in delete_list:
            del new_terms_dict[term]
        return QubitOperator(new_terms_dict)
    
    @classmethod
    def from_matrix(self, matrix):
        r"""
        Represents a matrix as an operator
            
        .. math::

            O=\sum_i\alpha_i\bigotimes_{j=0}^{n-1}O_{ij}

        where $O_{ij}\in\{A,C,P_0,P_1\}$.

        Parameters
        ----------
        matrix : numpy.ndarray or scipy.sparse.csr_matrix
            The matrix.

        Returns
        -------
        QubitOperator
            The operator represented by the matrix.


        Examples
        --------

        ::

            from scipy.sparse import csr_matrix
            from qrisp.operators import QubitOperator

            sparse_matrix = csr_matrix([[0, 5, 0, 1],
                                        [5, 0, 0, 0],
                                        [0, 0, 0, 2],
                                        [1, 0, 2, 0]])

            O = QubitOperator.from_matrix(sparse_matrix)
            print(O)
            # Yields: A_0*A_1 + C_0*C_1 + 5*P^0_0*A_1 + 5*P^0_0*C_1 + 2*P^1_0*A_1 + 2*P^1_0*C_1

        """
        from scipy.sparse import csr_matrix
        from numpy import ndarray
        import numpy as np

        OPERATOR_TABLE = {(0,0):"P0",(0,1):"A",(1,0):"C",(1,1):"P1"}

        if isinstance(matrix,ndarray):
            new_matrix = csr_matrix(matrix)
        elif isinstance(matrix,csr_matrix):
            new_matrix = matrix.copy()
        else:
            raise Exception("Cannot construct QubitOperator from type " + str(type(matrix)))
        
        M, N = new_matrix.shape
        n = max(int(np.ceil(np.log2(M))),int(np.ceil(np.log2(N))))

        new_matrix.eliminate_zeros()
    
        rows, cols = new_matrix.nonzero()
        values = new_matrix.data

        O = QubitOperator({})
        for row, col, value in zip(rows, cols, values):
            factor_dict = {}
            for k in range(n):
                i = (row >> k) & 1
                j = (col >> k) & 1
                factor_dict[n-k-1]=OPERATOR_TABLE[(i,j)]

            O.terms_dict[QubitTerm(factor_dict)] = value
        return O
    
    @classmethod
    def from_numpy_array(cls, numpy_array, threshold = np.inf):
        
        from qrisp.operators import X, Y, Z
        
        n = int(np.log2(numpy_array.shape[0]))
        H = 0
        
        for pauli_indicator_tuple in product(range(4), repeat = n):
            
            temp_H = 1
            for i in range(n):
                
                if pauli_indicator_tuple[i] == 1:
                    temp_H = X(i)*temp_H
                if pauli_indicator_tuple[i] == 2:
                    temp_H = Y(i)*temp_H
                if pauli_indicator_tuple[i] == 3:
                    temp_H = Z(i)*temp_H
            
            if isinstance(temp_H, int) and temp_H == 1:
                temp_H_array = np.eye(2**n)
            else:
                temp_H_array = temp_H.to_array(n)
            
            coefficient = np.dot(temp_H_array.flatten().conjugate(), numpy_array.flatten())
            
            H += (coefficient/2**(n)) * temp_H
            
        return H


    @classmethod
    def from_matrix(self, matrix):
        r"""
        Represents a matrix as an operator
            
        .. math::

            O=\sum_i\alpha_i\bigotimes_{j=0}^{n-1}O_{ij}

        where $O_{ij}\in\{A,C,P_0,P_1\}$.

        Parameters
        ----------
        matrix : numpy.ndarray or scipy.sparse.csr_matrix
            The matrix.

        Returns
        -------
        QubitOperator
            The operator represented by the matrix.


        Examples
        --------

        ::

            from scipy.sparse import csr_matrix
            from qrisp.operators import QubitOperator

            sparse_matrix = csr_matrix([[0, 5, 0, 1],
                                        [5, 0, 0, 0],
                                        [0, 0, 0, 2],
                                        [1, 0, 2, 0]])

            O = QubitOperator.from_matrix(sparse_matrix)
            print(O)
            # Yields: A_0*A_1 + C_0*C_1 + 5*P^0_0*A_1 + 5*P^0_0*C_1 + 2*P^1_0*A_1 + 2*P^1_0*C_1

        """
        from scipy.sparse import csr_matrix
        from numpy import ndarray
        import numpy as np

        OPERATOR_TABLE = {(0,0):"P0",(0,1):"A",(1,0):"C",(1,1):"P1"}

        if isinstance(matrix,ndarray):
            new_matrix = csr_matrix(matrix)
        elif isinstance(matrix,csr_matrix):
            new_matrix = matrix.copy()
        else:
            raise Exception("Cannot construct QubitOperator from type " + str(type(matrix)))
        
        M, N = new_matrix.shape
        n = max(int(np.ceil(np.log2(M))),int(np.ceil(np.log2(N))))

        new_matrix.eliminate_zeros()
    
        rows, cols = new_matrix.nonzero()
        values = new_matrix.data

        O = QubitOperator({})
        for row, col, value in zip(rows, cols, values):
            factor_dict = {}
            for k in range(n):
                i = (row >> k) & 1
                j = (col >> k) & 1
                factor_dict[n-k-1]=OPERATOR_TABLE[(i,j)]

            O.terms_dict[QubitTerm(factor_dict)] = value
        return O

    def to_sparse_matrix(self, factor_amount=None):
        r"""
        Returns a scipy matrix representing the operator

        .. math::

            O=\sum_i\alpha_i\bigotimes_{j=0}^{n-1}O_{ij}

        where $O_{ij}\in\{X,Y,Z,A,C,P_0,P_1,I\}$.

        Parameters
        ----------
        factor_amount : int, optional
            The amount of factors $n$ to represent this matrix. The matrix will have 
            the dimension $2^n \times 2^n$, where n is the amount of factors. 
            By default the minimal number $n$ is chosen.
    
        Returns
        -------
        scipy.sparse.csr_matrix
            The sparse matrix representing the operator.

        """

        import scipy.sparse as sp
        from scipy.sparse import kron as TP, csr_matrix

        def get_matrix(P):
            if P=="I":
                return csr_matrix([[1,0],[0,1]])
            if P=="X":
                return csr_matrix([[0,1],[1,0]])
            if P=="Y":
                return csr_matrix([[0,-1j],[1j,0]])
            if P == "Z":
                return csr_matrix([[1,0],[0,-1]])
            if P == "A":
                return csr_matrix([[0,1],[0,0]])
            if P == "C":
                return csr_matrix([[0,0],[1,0]])
            if P == "P0":
                return csr_matrix([[1,0],[0,0]])
            if P == "P1":
                return csr_matrix([[0,0],[0,1]])

        def recursive_TP(keys,term_dict):
            if len(keys)==1:
                return get_matrix(term_dict.get(keys[0],"I"))
            return TP(get_matrix(term_dict.get(keys.pop(0),"I")),recursive_TP(keys,term_dict))

        term_dicts = []
        coeffs = []

        participating_indices = set()
        for term,coeff in self.terms_dict.items():
            curr_dict = term.factor_dict
            term_dicts.append(curr_dict)    
            coeffs.append(coeff)
            participating_indices = participating_indices.union(term.non_trivial_indices())

        if factor_amount is None:
            if len(participating_indices):
                factor_amount = max(participating_indices) + 1
            else:
                res = 1
                for coeff in coeffs:
                    res *= coeff
                M = sp.csr_matrix((1,1))
                M[0,0] = res
                return M
        elif participating_indices and factor_amount < max(participating_indices) + 1:
            raise Exception("Tried to construct matrix with insufficient factor_amount")

        keys = list(range(factor_amount))
        
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k])*recursive_TP(keys.copy(),term_dicts[k])
        # res = ((M + M.transpose().conjugate())/2)
        # res.sum_duplicates()
        return M
    
    def to_array(self, factor_amount=None):
        r"""
        Returns a numpy array representing the operator

        .. math::

            O=\sum_i\alpha_i\bigotimes_{j=0}^{n-1}O_{ij}

        where $O_{ij}\in\{X,Y,Z,A,C,P_0,P_1,I\}$.

        Parameters
        ----------
        factor_amount : int, optional
            The amount of factors $n$ to represent this matrix. The matrix will have 
            the dimension $2^n \times 2^n$, where n is the amount of factors. 
            By default the minimal number $n$ is chosen.

        Returns
        -------
        np.ndarray
            The array representing the operator.

        Examples
        --------

        >>> from qrisp.operators import *
        >>> O = X(0)*X(1) + 2*P0(0)*P0(1) + 3*P1(0)*P1(1)
        >>> O.to_array()
        matrix([[2.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [1.+0.j, 0.+0.j, 0.+0.j, 3.+0.j]])

        """
        return np.array(self.to_sparse_matrix(factor_amount).todense())
    
    def to_pauli(self):
        """
        Returns an equivalent operator, which however only contains Pauli factors.

        Returns
        -------
        QubitOperator
            An operator that contains only Pauli factors.

        Examples
        --------
        
        We create a QubitOperator containing A and C terms and convert it to a
        Pauli based representation.
        
        >>> from qrisp.operators import A,C,Z
        >>> H = A(0)*C(1)*Z(2)
        >>> print(H.to_pauli())
        0.25*X_0*X_1*Z_2 + 0.25*I*X_0*Y_1*Z_2 - 0.25*I*Y_0*X_1*Z_2 + 0.25*Y_0*Y_1*Z_2
            
        """
        
        res = 0
        for term, coeff in self.terms_dict.items():
            res += coeff*term.to_pauli()
        if isinstance(res, (float, int)):
            return QubitOperator({QubitTerm({}) : res})
        return res
    
    def adjoint(self):
        """
        Returns an the adjoint operator.

        Returns
        -------
        QubitOperator
            The adjoint operator.

        Examples
        --------
        
        We create a QubitOperator and inspect its adjoint.
        
        >>> from qrisp.operators import A,C,Z
        >>> H = A(0)*C(1)*Z(2)
        >>> print(H.adjoint())
        C_0*A_1*Z_2
        """
        new_terms_dict = {}
        for term, coeff in self.terms_dict.items():
            new_terms_dict[term.adjoint()] = np.conjugate(coeff)
        return QubitOperator(new_terms_dict)
    
    def hermitize(self):
        """
        Returns the hermitian part of self.
        
        $H = (O + O^\dagger)/2$

        Returns
        -------
        QubitOperator
            The hermitian part.

        """
        return 0.5*(self + self.adjoint())
        
    def eliminate_ladder_conjugates(self):
        new_terms_dict = {}
        for term, coeff in self.terms_dict.items():
            for factor in term.factor_dict.values():
                if factor in ["A", "C"]:
                    break
            else:
                new_terms_dict[term] = coeff
                continue
            
            if term.adjoint() in new_terms_dict:
                new_terms_dict[term.adjoint()] += coeff
            else:
                new_terms_dict[term] = coeff
        
        return QubitOperator(new_terms_dict)
        
    
    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.
    
        Returns
        -------
        float
            The ground state energy. 

        """

        from scipy.sparse.linalg import eigsh

        M = self.hermitize().to_sparse_matrix()
        # Compute the smallest eigenvalue
        eigenvalues, _ = eigsh(M, k=1, which='SA')  # 'SA' stands for smallest algebraic
        E = eigenvalues[0]

        return E
    
    #
    # Partitions 
    #

    # Commutativity: Partitions the QubitOperator into QubitOperators with pairwise commuting QubitTerms
    def commuting_groups(self):
        r"""
        Partitions the QubitOperator into QubitOperators with pairwise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually commuting.

        Returns
        -------
        groups : list[QubitOperator]
            The partition of the Hamiltonian.
        
        """

        groups = [] # Groups of commuting QubitTerms 

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        for term,coeff in sorted_terms:

            commute_bool = False
            if len(groups) > 0:
                for group in groups:
                    for term_,coeff_ in group.terms_dict.items():
                        commute_bool = term_.commute(term)
                        if not commute_bool:
                            break
                    if commute_bool:
                        group.terms_dict[term]=coeff
                        break
            if len(groups)==0 or not commute_bool: 
                groups.append(QubitOperator({term:coeff}))

        return groups
    
    def group_up(self, group_denominator):
        term_groups = group_up_iterable(list(self.terms_dict.keys()), group_denominator)
        if len(term_groups) == 0:
            return [self]
        groups = []
        for term_group in term_groups:
            H = QubitOperator({term : self.terms_dict[term] for term in term_group})
            groups.append(H)
            
        return groups
        

    # Qubit-wise commutativity: Partitions the QubitOperator into QubitOperators with pairwise qubit-wise commuting QubitTerms
    def commuting_qw_groups(self, show_bases=False, use_graph_coloring = True):
        r"""
        Partitions the QubitOperator into QubitOperators with pairwise qubit-wise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually qubit-wise commuting.

        Returns
        -------
        groups : list[QubitOperator]
            The partition of the Hamiltonian.
        
        """

        groups = [] # Groups of qubit-wise commuting QubitTerms
        bases = [] # Bases as termTerms

        if use_graph_coloring:        
            
            term_groups = group_up_iterable(list(self.terms_dict.keys()), lambda a, b : a.commute_qw(b))
            for term_group in term_groups:
                H = QubitOperator({term : self.terms_dict[term] for term in term_group})
                groups.append(H)
                
                if show_bases:
                    factor_dict = {}
                    
                    for term in term_group:
                        for index, factor in term.factor_dict.items():
                            if factor in ["X", "Y", "Z"]:
                                factor_dict[index] = factor
                    
                    bases.append(QubitTerm(factor_dict))
                    
            if show_bases:
                return groups, bases
            else:
                return groups

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        for term,coeff in sorted_terms:

            commute_bool = False
            if len(groups)>0:
                n = len(groups)
                for i in range(n):
                    commute_bool = bases[i].commute_qw(term)
                    if commute_bool:
                        bases[i].update(term.factor_dict)
                        groups[i].terms_dict[term]=coeff
                        break
            if len(groups)==0 or not commute_bool:
                groups.append(QubitOperator({term:coeff}))
                bases.append(term.copy())

        if show_bases:
            return groups, bases
        else:
            return groups
    
    #
    # Measurement settings and measurement
    #
    
    def change_of_basis(self, qarg, method="commuting_qw"):
        """
        Performs several operations on a quantum argument such that the hermitian
        part of self is diagonal when conjugated with these operations.

        Parameters
        ----------
        qarg : QuantumVariable or list[Qubit]
            The quantum argument to apply the change of basis on.
        method : str, optional
            The method for calculating the change of basis. 
            Available are ``commuting`` (all QubitTerms must mutually commute) and ``commuting_qw`` (all QubitTerms must mutually commute qubit-wise).
            The default is ``commuting_qw``.

        Returns
        -------
        res : QubitOperator
            A qubit operator that contains only diagonal entries (I, Z, P0, P1).

        """
        
        # Assuming all terms of self commute qubit-wise,
        # the basis change for Pauli factor is trivial:
        # Z stays the same, for X we apply an h gate and for Y and s_dg.
        
        # For ladder operators, the situation is more intricate.
        
        # Take for instance the ladder operators A(0)*A(1)*A(2) + h.c.
        
        # In Bra-Ket form, this is |000><111| + |111><000|
        
        # The considerations from Selingers Paper https://arxiv.org/abs/2310.12256
        
        # In this work, the above term is simulated by the following circuit
        
        #                ┌───┐                                                        ┌───┐
        #   qv_0.0: ─────┤ X ├────────────■─────────────────────────■─────────────────┤ X ├─────
        #                └─┬─┘┌───┐       │                         │            ┌───┐└─┬─┘
        #   qv_0.1: ───────┼──┤ X ├───────■─────────────────────────■────────────┤ X ├──┼───────
        #           ┌───┐  │  └─┬─┘┌───┐  │  ┌───┐┌──────────────┐  │  ┌───┐┌───┐└─┬─┘  │  ┌───┐
        #   qv_0.2: ┤ X ├──■────■──┤ X ├──┼──┤ H ├┤ Rz(-1.0*phi) ├──┼──┤ H ├┤ X ├──■────■──┤ X ├
        #           └───┘          └───┘┌─┴─┐└───┘└──────┬───────┘┌─┴─┐└───┘└───┘          └───┘
        # hs_anc.0: ────────────────────┤ X ├────────────■────────┤ X ├─────────────────────────
        #                               └───┘                     └───┘
        
        
        # From this we conclude that H can be expressed as a conjugation of the following form.
        
        # H = U^dg (|110><110| - |111><111|)/2 U
        
        # Where U is the following circuit:
        
        #             ┌───┐               
        # qb_90: ─────┤ X ├───────────────
        #             └─┬─┘┌───┐          
        # qb_91: ───────┼──┤ X ├──────────
        #        ┌───┐  │  └─┬─┘┌───┐┌───┐
        # qb_92: ┤ X ├──■────■──┤ X ├┤ H ├
        #        └───┘          └───┘└───┘
                        
        
        # This is because
        
        # exp(i*t*H) = U^dg MCRZ(i*t) U
        #            = U^dg exp(i*t*(|110><110| - |111><111|)/2) U
        
        # The bra-ket term is already diagonal but how to express it via operators?
        
        # The answer is P1(0)*P1(1)*Z(2)
        
        # From this we conclude the underlying rule here. For ladder terms we can
        # pick an arbitrary qubit that we call "anchor qubit" which is conjugated
        # with an H gate.
        
        # After performing the conjugation with the CX gates to complete the inverse
        # GHZ preparation, the ladder operator transforms into a chain of projectors
        # whereas the anchor qubit becomes a Z gate.
        
        n = self.find_minimal_qubit_amount()
        if not check_for_tracing_mode() and len(qarg) < n:
            raise Exception("Tried to change the basis of an Operator on a quantum argument with insufficient qubits.")
        
     
        # This dictionary will contain the new terms/coefficient comination for the
        # diagonal operator
        new_terms_dict = {}

        new_factor_dicts = []
        prefactors = []
        
        ladder_conjugation_performed = False
        ladder_indices = []

        if method=="commuting_qw":
            # We track which qubit is in which basis to raise an error if a
            # violation with the requirement of qubit wise commutativity is detected.
            basis_dict = {}            
    
            # We iterate through the terms and apply the appropriate basis transformation
            for term, coeff in self.terms_dict.items():

                factor_dict = term.factor_dict
                # This dictionary will contain the factors of the new term
                new_factor_dict = {}
                
                new_factor_dicts.append(new_factor_dict)
            
                prefactor = 1
                prefactors.append(prefactor)
            
                for j in range(n):
                
                    # If there is no entry in the factor dict, this corresponds to
                    # identity => no basis change required.
                    if j not in factor_dict:
                        continue
                
                    # If j is already in the basis dict, we assert that the bases agree
                    # (otherwise there is a violation of qubit-wise commutativity)
                    if j in basis_dict:
                        if basis_dict[j] != factor_dict[j]:
                            assert basis_dict[j] in ["Z", "P0", "P1"]
                        new_factor_dict[j] = "Z"
                        continue
                    
                    # We treat ladder operators in the next section
                    if factor_dict[j] not in ["X", "Y", "Z"]:
                        continue
                
                    # Update the basis dict
                    basis_dict[j] = factor_dict[j]
                
                    # Append the appropriate basis-change gate
                    if factor_dict[j]=="X":
                        h(qarg[j])
                    
                    if factor_dict[j]=="Y":
                        sx_dg(qarg[j])
            
                    new_factor_dict[j] = "Z"
                    
                    

        if method=="commuting":

            # Calculate S: Matrix where the colums correspond to the binary representation (Z/X) of the Pauli terms
            x_vectors = []
            z_vectors = []
            for term, coeff in self.terms_dict.items():
                x_vector, z_vector = term.binary_representation(n)
                x_vectors.append(x_vector)
                z_vectors.append(z_vector)
            x_matrix = np.stack(x_vectors, axis=1)
            z_matrix = np.stack(z_vectors, axis=1)

            # Find qubits (rows) on which Pauli X,Y,Z operatos act
            qb_indices = []
            for k in range(n):
                if not (np.all(x_matrix[k] == 0) and np.all(z_matrix[k] == 0)):
                    qb_indices.append(k)
            m = len(qb_indices)
            
            if m==0:
                new_factor_dicts = [{} for _ in range(self.len())]
                prefactors = [1]*self.len()
            else:
                S = np.vstack((z_matrix[qb_indices], x_matrix[qb_indices]))
            
                # Construct and apply change of basis
                A, R_inv, h_list, s_list, perm = construct_change_of_basis(S)

                def inv_graph_state(qarg):
                    for i in range(m):
                        for j in range(i):
                            if A[i,j]==1:
                                cz(qarg[qb_indices[perm[i]]],qarg[qb_indices[perm[j]]])
                    for i in qb_indices:
                        h(qarg[i])

                def change_of_basis(qarg):
                    for i in h_list:
                        h(qarg[qb_indices[i]])
                    for i in s_list:
                        s(qarg[qb_indices[perm[i]]])
                    inv_graph_state(qarg)

                change_of_basis(qarg)

                # Construct new QubitOperator
                #
                # Factor (-1) appears if S gate is applied to X, or Hadamard gate H is applied to Y:
                # S^dagger X S = -Y
                # S^dagger Y S = X
                # S^dagger Z S = Z
                # H X H = Z
                # H Y H = -Y
                # H Z H = X
                # For the original Pauli terms this translates to: Factor (-1) appears if S gate is applied to Y, or Hadamard gate H is applied to Y
                # No factor (-1) occurs if H S^{-1} P S H is applied (i.e., H and S) for any P in {X,Y,Z}

                s_vector = np.zeros(m, dtype=int)
                s_vector[s_list] = 1
                h_vector = np.zeros(m, dtype=int)
                h_vector[h_list] = 1
                sh_vector = s_vector[perm] + h_vector % 2 
                sign_vector = sh_vector @ (x_matrix[qb_indices]*z_matrix[qb_indices]) % 2

                # Lower triangular part of A
                A_low = np.tril(A)

                for index,z_vector in enumerate(R_inv.T):

                    # Determine the sign of the product of the selected graph state stabilizers:
                    # 
                    # Consider product of stabilizers S_{i_1}*S_{i_2}*...*S_{i_m} with (w.l.o.g.) i_1<i_2<...<i_m
                    # For each i: Swap X_i with all Z_i's from stabilizers if index > i such that all Z_i's are on the left of X_i
                    # Calculate the paritiy n1 of the sum of the numbers of 1's with position j>i for each row of the square submatrix A defined by z_vector
                    # Yields a factor (-1)^n1

                    n1 = sum((z_vector @ A_low)*z_vector) % 2

                    # For each i: Count the number of Z_i's: if even, no factor, if odd: factor i (ZX=iY)
                    # Count the number n2 of rows of the square submatrix of A defined by z_vector, such that the number of 1's in each row is odd
                    # This number is always even since A is a symmetric matrix with 0's on the diagonal
                    # Yields a factor i^n2=(-1)^(n2/2)

                    n2 = sum((z_vector @ A)*z_vector % 2)

                    new_factor_dict = {qb_indices[perm[i]]:"Z" for i in range(m) if z_vector[i]==1}
                    new_factor_dicts.append(new_factor_dict)
                    
                    prefactor = (-1)**sign_vector[index]*(-1)**(n1+n2/2)
                    prefactors.append(prefactor)

        # Ladder operators 
        for term, coeff in self.terms_dict.items():    

            prefactor = prefactors.pop(0)    
            new_factor_dict = new_factor_dicts.pop(0)
            
            # Next we treat the ladder operators
            ladder_operators = [base for base in term.factor_dict.items() if base[1] in ["A", "C"]]
            ladder_operators.sort(key = lambda x : x[0])
            
            if len(ladder_operators):
                
                # The anchor factor is the "last" ladder operator. 
                # This is the qubit where the H gate will be executed.
                anchor_factor = ladder_operators[-1]
                new_factor_dict[ladder_operators[-1][0]] = "Z"
                
                # Perform the cnot gates
                for j in range(len(ladder_operators)-1):
                    if not ladder_conjugation_performed:
                        ladder_indices.append(ladder_operators[j][0])
                        cx(qarg[anchor_factor[0]], qarg[ladder_operators[j][0]])
                    
                    if anchor_factor[1] == "C":
                        if ladder_operators[j][1] == "A":
                            new_factor_dict[ladder_operators[j][0]] = "P1"
                        else:
                            new_factor_dict[ladder_operators[j][0]] = "P0"
                    else:
                        if ladder_operators[j][1] == "A":
                            new_factor_dict[ladder_operators[j][0]] = "P0"
                        else:
                            new_factor_dict[ladder_operators[j][0]] = "P1"
                
                if not ladder_conjugation_performed:
                    # Execute the H-gate
                    ladder_indices.append(anchor_factor[0])
                    h(qarg[anchor_factor[0]])
                else:
                    if set(ladder_indices) != set(ladder_factor[0] for ladder_factor in ladder_operators):
                        raise Exception("Tried to perform change of basis on operator containing non-matching ladder indices")
                
                ladder_conjugation_performed = True
                
                prefactor *= 0.5
                
            for k, v in term.factor_dict.items():
                if v in ["P0", "P1"]:
                    new_factor_dict[k] = v

            new_term = QubitTerm(new_factor_dict)
            new_terms_dict[new_term] = prefactor*self.terms_dict[term]
        
        return QubitOperator(new_terms_dict) 
        
    
    def get_conjugation_circuit(self):
        # This method returns a QuantumCircuit that should be applied
        # before a measurement of self is peformed.
        # The method assumes that all terms within this Operator commute qubit-
        # wise. For instance, if an X operator is supposed to be measured,
        # the conjugation circuit will contain an H gate at that point,
        # because the X operator can be measured by measuring the Z Operator
        # in the H-transformed basis.
        
        # For the ladder operators, the conjugation circuit not this straight-
        # forward. To understand how we measure the ladder operators, consider
        # the operator
        
        # H = (A(0)*A(1)*A(2) + h.c.)
        #   = (|000><111| + |111><000|)
        
        # The considerations from Selingers Paper https://arxiv.org/abs/2310.12256
        # motivate that H can be expressed as a conjugation of the following form.
        
        # H = U^dg (|110><110| - |111><111|)/2 U
        
        # This is because
        
        # exp(i*t*H) = U^dg MCRZ(i*t) U
        #            = U^dg exp(i*t*(|110><110| - |111><111|)/2) U
        
        # We use this insight because the Operator 
        # |111><111| - |110><110| = |11><11| (x) (|0><0| - |1><1|) 
        # = |11><11| (x) Z
        # can be measured via postprocessing.
        
        # The postprocessing to do is essentially measuring the last qubit as
        # a regular Z operator and only add the result to the expectation value
        # if the first two qubits are measured to be in the |1> state.
        # If they are in any other state nothing should be added.
        
        # From this we can also conclude how the conjugation circuit needs to
        # look like: Essentially like the conjugation circuit from the paper.
        
        # For our example above (when simulated) gives:
            
        #                ┌───┐                                                        ┌───┐
        #   qv_0.0: ─────┤ X ├────────────■─────────────────────────■─────────────────┤ X ├─────
        #                └─┬─┘┌───┐       │                         │            ┌───┐└─┬─┘
        #   qv_0.1: ───────┼──┤ X ├───────■─────────────────────────■────────────┤ X ├──┼───────
        #           ┌───┐  │  └─┬─┘┌───┐  │  ┌───┐┌──────────────┐  │  ┌───┐┌───┐└─┬─┘  │  ┌───┐
        #   qv_0.2: ┤ X ├──■────■──┤ X ├──┼──┤ H ├┤ Rz(-1.0*phi) ├──┼──┤ H ├┤ X ├──■────■──┤ X ├
        #           └───┘          └───┘┌─┴─┐└───┘└──────┬───────┘┌─┴─┐└───┘└───┘          └───┘
        # hs_anc.0: ────────────────────┤ X ├────────────■────────┤ X ├─────────────────────────
        #                               └───┘                     └───┘
        
        # Where the construction of the MCRZ gate is is encoded into the Toffolis
        # and the controlled RZ-Gate.
        
        # The conjugation circuit therefore needs to look like this:
                    
        #             ┌───┐               
        # qb_90: ─────┤ X ├───────────────
        #             └─┬─┘┌───┐          
        # qb_91: ───────┼──┤ X ├──────────
        #        ┌───┐  │  └─┬─┘┌───┐┌───┐
        # qb_92: ┤ X ├──■────■──┤ X ├┤ H ├
        #        └───┘          └───┘└───┘
                
        # To learn more about how the post-processing is implemented check the
        # comments of QubitTerm.serialize
        
        # ===============
        
        # Create a QuantumCircuit that contains the conjugation
        from qrisp import QuantumCircuit
        n = self.find_minimal_qubit_amount()
        qc = QuantumCircuit(n)
        
        # We track which qubit is in which basis to raise an error if a
        # violation with the requirement of qubit wise commutativity is detected.
        basis_dict = {}
        
        # We iterate through the terms and apply the appropriate basis transformation
        for term, coeff in self.terms_dict.items():
            
            factor_dict = term.factor_dict
            for j in range(n):
                
                # If there is no entry in the factor dict, this corresponds to
                # identity => no basis change required.
                if j not in factor_dict:
                    continue
                
                # If j is already in the basis dict, we assert that the bases agree
                # (otherwise there is a violation of qubit-wise commutativity)
                if j in basis_dict:
                    assert basis_dict[j] == factor_dict[j]
                    continue
                
                # We treat ladder operators in the next section
                if factor_dict[j] not in ["X", "Y", "Z"]:
                    continue
                
                # Update the basis dict
                basis_dict[j] = factor_dict[j]
                
                # Append the appropriate basis-change gate
                if factor_dict[j]=="X":
                    qc.h(j)
                if factor_dict[j]=="Y":
                    qc.sx(j)
            
            # Next we treat the ladder operators
            ladder_operators = [base for base in term.factor_dict.items() if base[1] in ["A", "C"]]
            
            if len(ladder_operators):
                
                # The anchor factor is the "last" ladder operator. 
                # This is the qubit where the H gate will be executed.
                anchor_factor = ladder_operators[-1]
            
                # Flip the anchor qubit if the ladder operator is an annihilator
                if anchor_factor[1] == "C":
                    qc.x(anchor_factor[0])
                
                # Perform the cnot gates
                for j in range(len(ladder_operators)-1):
                    qc.cx(anchor_factor[0], ladder_operators[j][0])

                # Flip the anchor qubit back
                if anchor_factor[1] == "C":
                    qc.x(anchor_factor[0])
            
                # Execute the H-gate
                qc.h(anchor_factor[0])
        
        return qc, QubitOperator(self.terms_dict)
    
    def get_operator_variance(self, n = 1):
        """
        Calculates the optimal distribution and number of shots following https://quantum-journal.org/papers/q-2021-01-20-385/pdf/.
        
        Normally to compute the variance of an operator, the distribution has to be known.
        Since the distribution is not known without querying the quantum device,
        the authors estimate the variance as the expectation value of a distribution
        of quantum states. This distribution is uniform across the unit sphere.
        
        For an arbitrary Pauli-Operator P != I they conclude
        
        E(Var(P)) = alpha_n = 1 - 1/(2^n + 1)
        
        Where 2^n is the dimension of the comprising space
        
        Since the QubitOperator class also contains A, C and P operators, we have to
        do more work.
        
        To understand how the variance can be estimated, recall that every
        QubitOperator O can be transformed to a sum of Pauli strings
        
        Var(O) = Var(sum_i(c_i*P_i))
               = sum_i(Var(c_i*P_i)) + 2*sum_[0<=i<j<=n](Cov(c_i*P_i,c_j*P_j))
        
        The last line can be found in https://arxiv.org/pdf/1907.13623 section 10.1.
        Theorem 2 of that very same source states that for the above distribution
        of states, we have E(Cov(P_i, P_j)) = 0 if P_i != P_j
        
        From that we conclude
        
        E(Var(O)) = sum_i(E(Var(c_i*P_i)))
                  = sum_i(abs(c_i)**2*E(Var(P_i)))
                  = alpha_n * sum_i(abs(c_i)**2)
                  
        It therefore suffices to compute the variance of the Pauli form of the
        QubitOperator.
        
        
        """
        var = 0
        pauli_form = self.hermitize().to_pauli()
        for term, coeff in pauli_form.terms_dict.items():
            if len(term.factor_dict) != 0:
                var += abs(coeff)**2
        alpha_n = 1 - 1/(2**n + 1)
        
        return var*alpha_n
        
    def get_measurement(
        self,
        qarg,
        precision=0.01,
        backend=None,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        precompiled_qc=None,
        diagonalisation_method="commuting_qw",
        measurement_data=None # measurement settings
    ):
        r"""
        This method returns the expected value of a Hamiltonian for the state 
        of a quantum argument. Note that this method measures the **hermitized**
        version of the operator:
            
        .. math::
            
            H = (O + O^\dagger)/2


        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or list[Qubit]
            The quantum argument to evaluate the Hamiltonian on.
        precision : float, optional
            The precision with which the expectation of the Hamiltonian is to be evaluated.
            The default is 0.01. The number of shots scales quadratically with the inverse precision.
        backend : :ref:`BackendClient`, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        compile : bool, optional
            Boolean indicating if the .compile method of the underlying QuantumSession
            should be called before. The default is ``True``.
        compilation_kwargs  : dict, optional
            Keyword arguments for the compile method. For more details check
            :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`. The default
            is ``{}``.
        subs_dic : dict, optional
            A dictionary of Sympy symbols and floats to specify parameters in the case
            of a circuit with unspecified, :ref:`abstract parameters<QuantumCircuit>`.
            The default is ``{}``.
        precompiled_qc : QuantumCircuit, optional
            A precompiled quantum circuit.
        diagonalisation_method : str, optional
            Specifies the method for grouping and diagonalizing the QubitOperator. 
            Available are ``commuting_qw``, i.e., the operator is grouped based on qubit-wise commutativity of terms, 
            and ``commuting``, i.e., the operator is grouped based on commutativity of terms.
            The default is ``commuting_qw``.
        measurement_data : QubitOperatorMeasurement
            Cached data to accelerate the measurement procedure. Automatically generated by default.

        Raises
        ------
        Exception
            If the containing QuantumSession is in a quantum environment, it is not
            possible to execute measurements.

        Returns
        -------
        float
            The expected value of the Hamiltonian.

        Examples
        --------

        We define a Hamiltonian, and measure its expected value for the state of a :ref:`QuantumVariable`.

        ::

            from qrisp import QuantumVariable, h
            from qrisp.operators.qubit import X,Y,Z
            qv = QuantumVariable(2)
            h(qv)
            H = Z(0)*Z(1)
            res = H.get_measurement(qv)
            print(res)
            #Yields 0.0011251406425802912

        """
        return get_measurement(self, 
                                qarg, 
                                precision=precision, 
                                backend=backend, 
                                compile=compile, 
                                compilation_kwargs=compilation_kwargs, 
                                subs_dic=subs_dic,
                                precompiled_qc=precompiled_qc, 
                                diagonalisation_method=diagonalisation_method,
                                measurement_data=measurement_data)

    #
    # Trotterization
    #

    def trotterization(self, method='commuting_qw', forward_evolution = True):
        r"""
        .. _ham_sim:
        
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $U(t) = e^{-itH}$ via Trotterization.
        Note that this method will always simulate the **hermitized** operator, i.e.
        
        .. math::
            
            H = (O + O^\dagger)/2


        Parameters
        ----------
        method : str, optional
            The method for grouping the QubitTerms. 
            Available are ``commuting`` (groups such that all QubitTerms mutually commute) and ``commuting_qw`` (groups such that all QubitTerms mutually commute qubit-wise).
            The default is ``commuting_qw``.
        forward_evolution : bool, optional
            If set to False $U(t)^\dagger = e^{itH}$ will be executed (usefull for quantum phase estimation). The default is True.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{-itH}$ is 
            approximated by 
            
            .. math::

                e^{-itH}\approx U(t,N)=\left(e^{-iH_1t/N}\dotsb e^{-iH_mt/N}\right)^N

            This function receives the following arguments:

            * qarg : QuantumVariable 
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U(t,N)$ is applied. The default is 1.
                
        Examples
        --------
        
        We simulate a simple QubitOperator.
        
        >>> from sympy import Symbol
        >>> from qrisp.operators import A,C,Z,Y
        >>> from qrisp import QuantumVariable
        >>> O = A(0)*C(1)*Z(2) + Y(3)
        >>> U = O.trotterization()
        >>> qv = QuantumVariable(4)
        >>> t = Symbol("t")
        >>> U(qv, t = t)
        >>> print(qv.qs)
        QuantumCircuit:
        ---------------
              ┌───┐                      ┌───┐┌────────────┐┌───┐          ┌───┐
        qv.0: ┤ X ├──────────────────────┤ X ├┤ Rz(-0.5*t) ├┤ X ├──────────┤ X ├
              └─┬─┘     ┌───┐     ┌───┐  └─┬─┘├───────────┬┘└─┬─┘┌───┐┌───┐└─┬─┘
        qv.1: ──■───────┤ H ├─────┤ X ├────■──┤ Rz(0.5*t) ├───■──┤ X ├┤ H ├──■──
                        └───┘     └─┬─┘       └───────────┘      └─┬─┘└───┘     
        qv.2: ──────────────────────■──────────────────────────────■────────────
              ┌────┐┌───────────┐┌──────┐                                       
        qv.3: ┤ √X ├┤ Rz(2.0*t) ├┤ √Xdg ├───────────────────────────────────────
              └────┘└───────────┘└──────┘                                       
        Live QuantumVariables:
        ----------------------
        QuantumVariable qv
        
        Execute a simulation:
            
        >>> print(qv.get_measurement(subs_dic = {t : 0.5}))
        {'0000': 0.77015, '0001': 0.22985}
        
        """
        O = self.hermitize().eliminate_ladder_conjugates()
        commuting_groups = O.group_up(lambda a, b: a.commute(b))
        
        if method=='commuting_qw':
            def trotter_step(qarg, t, steps):
                for com_group in commuting_groups:
                    qw_groups= com_group.group_up(lambda a,b : a.commute_qw(b) and a.ladders_agree(b))
                    for qw_group in qw_groups:
                        
                        with conjugate(qw_group.change_of_basis)(qarg) as diagonal_operator:
                            intersect_groups = diagonal_operator.group_up(lambda a, b: not a.intersect(b))
                            for intersect_group in intersect_groups:
                                for term,coeff in intersect_group.terms_dict.items():
                                    coeff = jnp.real(coeff)
                                    
                                    term.simulate(-coeff*t/steps*(-1)**int(forward_evolution), qarg)
        
        if method=='commuting':
            def trotter_step(qarg, t, steps):
                for com_group in commuting_groups:
                    with conjugate(com_group.change_of_basis)(qarg,method="commuting") as diagonal_operator:
                        intersect_groups = diagonal_operator.group_up(lambda a, b: not a.intersect(b))
                        for intersect_group in intersect_groups:
                            for term,coeff in intersect_group.terms_dict.items():
                                term.simulate(-coeff*t/steps*(-1)**int(forward_evolution), qarg)

        def U(qarg, t=1, steps=1, iter=1):
            if check_for_tracing_mode():
                for i in jrange(iter*steps):
                    trotter_step(qarg, t, steps)
            else:
                merge([qarg])
                with IterationEnvironment(qarg.qs, iter*steps):
                    trotter_step(qarg, t, steps)

        return U

