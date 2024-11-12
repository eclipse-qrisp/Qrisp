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
from qrisp.operators.hamiltonian_tools import find_qw_commuting_groups
from qrisp.operators.hamiltonian import Hamiltonian
from qrisp.operators.qubit.qubit_term import QubitTerm
from qrisp.operators.qubit.measurement import get_measurement
from qrisp import h, s, x, IterationEnvironment, conjugate, merge

import sympy as sp
import numpy as np

threshold = 1e-9

#
# QubitOperator
#

class QubitOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of Qubit Hamiltonians, i.e.
    Hamiltonians, that operate on a qubit space :math:`(\mathbb{C}^2)^{\otimes n}`.
    Supported are operators of the following form:
    
    .. math::
        
        H=\sum\limits_{j}\alpha_j O_j 
        
    where :math:`O_j=\prod_i o_i^j` is a product of the following operators:
    
    .. list-table::
       :header-rows: 1
       :widths: 20 40 40
    
       * - Operator
         - Ket-Bra Realization
         - Description
       * - X
         - :math:`\ket{0}\bra{1} + \ket{1}\bra{0}`
         - Pauli-X operator (bit flip)
       * - Y
         - :math:`-i\ket{0}\bra{1} + i\ket{1}\bra{0}`
         - Pauli-Y operator (bit flip with phase)
       * - Z
         - :math:`\ket{0}\bra{0} - \ket{1}\bra{1}`
         - Pauli-Z operator (phase flip)
       * - A
         - :math:`\ket{0}\bra{1}`
         - Annihilation operator (removes :math:`\ket{1}` state)
       * - C
         - :math:`\ket{1}\bra{0}`
         - Creation operator (adds :math:`\ket{1}` state)
       * - P0
         - :math:`\ket{0}\bra{0}`
         - Projector onto the :math:`\ket{0}` state
       * - P1
         - :math:`\ket{1}\bra{1}`
         - Projector onto the :math:`\ket{1}` state
    
    Arbitrary combinations of these operators can be efficiently simulated.
    
    Parameters
    ----------
    terms_dict : dict, optional
        A dictionary representing a QubitOperator.

    Examples
    --------

    A QubitOperator can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::
        
        from qrisp.operators.qubit import X,Y,Z,A,C,P0,P1

        H = 1+2*X(0)+3*X(0)*Y(1)*A(2)+C(4)*P1(0)
        H

    Yields $1 + P^1_0*C_4 + 2*X_0 + 3*X_0*Y_1*A_2$.
    
    Investigate the simulation circuit by simulating for a symbolic amount of time:

    ::        

        from qrisp import QuantumVariable
        from sympy import Symbol
        
        H = A(0)*C(1)*C(2)*Z(3)*X(4)
        U = H.trotterization()

        qv = QuantumVariable(5)
        phi = Symbol("phi")

        U(qv, t = phi)
        print(qv.qs)
    
    ::
        
                  ┌───┐                                                                  ┌───┐
            qv.0: ┤ X ├────────────■───────────────────────────────────■─────────────────┤ X ├
                  └─┬─┘┌───┐       │                                   │            ┌───┐└─┬─┘
            qv.1: ──┼──┤ X ├───────o───────────────────────────────────o────────────┤ X ├──┼──
                    │  └─┬─┘┌───┐  │  ┌───┐┌───┐┌──────────────┐┌───┐  │  ┌───┐┌───┐└─┬─┘  │
            qv.2: ──■────■──┤ H ├──┼──┤ X ├┤ X ├┤ Rz(-1.0*phi) ├┤ X ├──┼──┤ X ├┤ H ├──■────■──
                            └───┘  │  └─┬─┘└─┬─┘└──────┬───────┘└─┬─┘  │  └─┬─┘└───┘
            qv.3: ─────────────────┼────■────┼─────────┼──────────┼────┼────■─────────────────
                  ┌───┐            │         │         │          │    │  ┌───┐
            qv.4: ┤ H ├────────────┼─────────■─────────┼──────────■────┼──┤ H ├───────────────
                  └───┘          ┌─┴─┐                 │             ┌─┴─┐└───┘
        hs_anc.0: ───────────────┤ X ├─────────────────■─────────────┤ X ├────────────────────
                                 └───┘                               └───┘
    """

    def __init__(self, terms_dict={}):
        self.terms_dict = terms_dict
        pass

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

        result = QubitOperator(res_terms_dict)
        return result

    #
    # Miscellaneous
    #
    
    def find_minimal_qubit_amount(self):
        indices = sum([list(term.factor_dict.keys()) for term in self.terms_dict.keys()], [])
        return max(indices)+1
    
    def commutator(self, other):
        
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
        for term,coeff in self.terms_dict.items():
            if abs(coeff)<threshold:
                delete_list.append(term)
        for term in delete_list:
            del self.terms_dict[term]

    def to_sparse_matrix(self, factor_amount = None):
        """
        Returns a matrix representing the operator.
    
        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.

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
                return csr_matrix([[0,0],[1,0]])
            if P == "C":
                return csr_matrix([[0,1],[0,0]])
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
        elif participating_indices and factor_amount < max(participating_indices):
            raise Exception("Tried to compute Hermitian matrix with factor_amount variable lower than the largest factor index")

        keys = list(range(factor_amount))
        
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k])*recursive_TP(keys.copy(),term_dicts[k])
        # res = ((M + M.transpose().conjugate())/2)
        # res.sum_duplicates()
        return M
    
    def to_array(self, factor_amount = None):
        """
        Returns a numpy array describing the operator.

        Parameters
        ----------
        factor_amount : int, optional
            The amount of factors to represent this array. The array will have 
            the dimension $2^n \times 2^n$, where n is the amount of factors. 
            By default the minimal number is chosen.

        Returns
        -------
        np.ndarray
            The array describing the operator.

        """
        return self.to_sparse_matrix(factor_amount).todense()
    
    def to_pauli(self):
        """
        Returns an equivalent operator, which however only contains Pauli factors.

        Returns
        -------
        QubitOperator
            An operator that contains only Pauli-Factor.

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
        
        We create a QubitOperator and inspect it' adjoint.
        
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
        E : float
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

    # Qubit-wise commutativity: Partitions the QubitOperator into QubitOperators with pairwise qubit-wise commuting QubitTerms
    def commuting_qw_groups(self, show_bases=False):
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

        term_groups = find_qw_commuting_groups(self)
        
        groups = [] # Groups of qubit-wise commuting QubitTerms
        bases = [] # Bases as termTerms
        
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
                if anchor_factor[1] == "A":
                    qc.x(anchor_factor[0])
                
                # Perform the cnot gates
                for j in range(len(ladder_operators)-1):
                    qc.cx(anchor_factor[0], ladder_operators[j][0])

                # Flip the anchor qubit back
                if anchor_factor[1] == "A":
                    qc.x(anchor_factor[0])
            
                # Execute the H-gate
                qc.h(anchor_factor[0])
        
        return qc
    
    def get_operator_variance(self):
        """
        Calculates the optimal distribution and number of shots following https://quantum-journal.org/papers/q-2021-01-20-385/pdf/.
        
        """
        var = 0
        for term, coeff in self.terms_dict.items():
            if len(term.factor_dict) != 0:
                var += abs(coeff)**2
        return var
        
    def get_measurement(
        self,
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
            #Yields 0.0

        We define a Hamiltonian, and measure its expected value for the state of a :ref:`QuantumArray`.

        ::

            from qrisp import QuantumVariable, QuantumArray, h
            from qrisp.operators.qubit import X,Y,Z
            qtype = QuantumVariable(2)
            q_array = QuantumArray(qtype, shape=(2))
            h(q_array)
            H = Z(0)*Z(1) + X(2)*X(3)
            res = H.get_measurement(q_array)
            print(res)
            #Yields 1.0

        """
        return get_measurement(self, 
                                qarg, 
                                precision=precision, 
                                backend=backend, 
                                shots=shots, 
                                compile=compile, 
                                compilation_kwargs=compilation_kwargs, 
                                subs_dic=subs_dic,
                                precompiled_qc=precompiled_qc, 
                                _measurement=_measurement)

    #
    # Trotterization
    #

    def trotterization(self):
        r"""
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $e^{itH}$ via Trotterization.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{itH}$ is 
            approximated by 
            
            .. math::

                e^{itH}\approx U_1(t,N)=\left(e^{iH_1t/N}\dotsb e^{iH_mt/N}\right)^N

            This function receives the following arguments:

            * qarg : QuantumVariable 
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """

        def change_of_basis(qarg, terms_dict):
            for index, factor in terms_dict.items():
                if factor=="X":
                    h(qarg[index])
                if factor=="Y":
                    s(qarg[index])
                    h(qarg[index])
                    x(qarg[index])
                    

        groups, bases = self.commuting_qw_groups(show_bases=True)

        def trotter_step(qarg, t, steps):
            for index,basis in enumerate(bases):
                with conjugate(change_of_basis)(qarg, basis.factor_dict):
                    for term,coeff in groups[index].terms_dict.items():
                        term.simulate(coeff*t/steps, qarg, do_change_of_basis = False)

        def U(qarg, t=1, steps=1, iter=1):
            merge([qarg])
            with IterationEnvironment(qarg.qs, iter*steps):
                trotter_step(qarg, t, steps)

        return U

