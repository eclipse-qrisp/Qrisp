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
from qrisp.operators.qubit.pauli_measurement import PauliMeasurement
from qrisp.operators.qubit.measurement import get_measurement
from qrisp import h, s, x, IterationEnvironment, conjugate, merge

import sympy as sp

threshold = 1e-9

#
# QubitHamiltonian
#

class QubitHamiltonian(Hamiltonian):
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
        A dictionary representing a QubitHamiltonian.

    Examples
    --------

    A QubitHamiltonian can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::
        
        from qrisp.operators.qubit import X,Y,Z,A,C,P0,P1

        H = 1+2*X(0)+3*X(0)*Y(1)*A(2)+C(4)*P1(0)
        H

    Yields $3*A_2*X_0*Y_1 + C_4*P1_0 + 1 + 2*X_0$.
    
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
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to add to the operator self.

        Returns
        -------
        result : QubitHamiltonian
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitHamiltonian({QubitTerm():other})
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot add QubitHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for term,coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitHamiltonian(res_terms_dict)
        return result
    
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to substract from the operator self.

        Returns
        -------
        result : QubitHamiltonian
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitHamiltonian({QubitTerm():other})
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot substract QubitHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for term, coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)-coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitHamiltonian(res_terms_dict)
        return result
    
    def __rsub__(self,other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to substract the operator self from.

        Returns
        -------
        result : QubitHamiltonian
            The difference of the operator other and self.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitHamiltonian({QubitTerm():other})
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot substract QubitHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for term,coeff in self.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)-coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
    
        for term,coeff in other.terms_dict.items():
            res_terms_dict[term] = res_terms_dict.get(term,0)+coeff
            if abs(res_terms_dict[term])<threshold:
                del res_terms_dict[term]
        
        result = QubitHamiltonian(res_terms_dict)
        return result

    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to multiply with the operator self.

        Returns
        -------
        result : QubitHamiltonian
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = QubitHamiltonian({QubitTerm():other})
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot multipliy QubitHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for term1, coeff1 in self.terms_dict.items():
            for term2, coeff2 in other.terms_dict.items():
                curr_term, curr_coeff = term1*term2
                res_terms_dict[curr_term] = res_terms_dict.get(curr_term,0) + curr_coeff*coeff1*coeff2

        result = QubitHamiltonian(res_terms_dict)
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
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[QubitTerm()] = self.terms_dict.get(QubitTerm(),0)+other
            return self
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot add QubitHamiltonian and "+str(type(other)))

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
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[termTerm()] = self.terms_dict.get(termTerm(),0)-other
            return self
        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot add QubitHamiltonian and "+str(type(other)))

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
        other : int, float, complex or QubitHamiltonian
            A scalar or a QubitHamiltonian to multiply with the operator self.

        """

        if isinstance(other,(int,float,complex)):
            #other = QubitHamiltonian({QubitTerm():other})
            for term in self.terms_dict:
                self.terms_dict[term] *= other
            return self

        if not isinstance(other,QubitHamiltonian):
            raise TypeError("Cannot multipliy QubitHamiltonian and "+str(type(other)))

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
        result : QubitHamiltonian
            The resulting QubitHamiltonian.
        
        """

        res_terms_dict = {}

        for term, coeff in self.terms_dict.items():
            curr_term, curr_coeff = term.subs(subs_dict)
            res_terms_dict[curr_term] = res_terms_dict.get(curr_term,0) + curr_coeff*coeff

        result = QubitHamiltonian(res_terms_dict)
        return result

    #
    # Miscellaneous
    #

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
        elif factor_amount < max(participating_indices):
            raise Exception("Tried to compute Hermitian matrix with factor_amount variable lower than the largest factor index")

        keys = list(range(factor_amount))
        
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k])*recursive_TP(keys.copy(),term_dicts[k])

        res = ((M + M.transpose().conjugate())/2)
        res.sum_duplicates()
        return res
    
    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.
    
        Returns
        -------
        E : float
            The ground state energy. 

        """

        from scipy.sparse.linalg import eigsh

        M = self.to_sparse_matrix()
        # Compute the smallest eigenvalue
        eigenvalues, _ = eigsh(M, k=1, which='SA')  # 'SA' stands for smallest algebraic
        E = eigenvalues[0]

        return E
    
    #
    # Partitions 
    #

    # Commutativity: Partitions the QubitHamiltonian into QubitHamiltonians with pairwise commuting QubitTerms
    def commuting_groups(self):
        r"""
        Partitions the QubitHamiltonian into QubitHamiltonians with pairwise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually commuting.

        Returns
        -------
        groups : list[QubitHamiltonian]
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
                groups.append(QubitHamiltonian({term:coeff}))

        return groups

    # Qubit-wise commutativity: Partitions the QubitHamiltonian into QubitHamiltonians with pairwise qubit-wise commuting QubitTerms
    def commuting_qw_groups(self, show_bases=False):
        r"""
        Partitions the QubitHamiltonian into QubitHamiltonians with pairwise qubit-wise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually qubit-wise commuting.

        Returns
        -------
        groups : list[QubitHamiltonian]
            The partition of the Hamiltonian.
        
        """

        term_groups = find_qw_commuting_groups(self)
        
        groups = [] # Groups of qubit-wise commuting QubitTerms
        bases = [] # Bases as termTerms
        
        for term_group in term_groups:
            H = QubitHamiltonian({term : self.terms_dict[term] for term in term_group})
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
                groups.append(QubitHamiltonian({term:coeff}))
                bases.append(term.copy())

        if show_bases:
            return groups, bases
        else:
            return groups
    
    #
    # Measurement settings and measurement
    #

    def pauli_measurement(self):
        return PauliMeasurement(self)
    
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
