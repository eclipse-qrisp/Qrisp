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
from qrisp.operators.hamiltonian import Hamiltonian
from qrisp.operators.pauli.helper_functions import *
from qrisp.operators.pauli.bound_pauli_term import BoundPauliTerm
from qrisp.operators.pauli.pauli_measurement import PauliMeasurement

import sympy as sp

from sympy import init_printing
# Initialize automatic LaTeX rendering
init_printing()

threshold = 1e-9

#
# BoundPauliHamiltonian
#

class BoundPauliHamiltonian(Hamiltonian):
    r"""
    This class provides an efficient implementation of Pauli Hamiltonians acting on QuantumVariables, i.e.,
    Hamiltonians of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j 
            
    where $P_j=\prod_i\sigma_i^j$ is a Pauli product, 
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Parameters
    ----------
    terms_dict : dict, optional
        A dictionary representing a BoundPauliHamiltonian.

    Examples
    --------

    A BoundPauliHamiltonian can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::

        from qrisp import QuantumVariable
        from qrisp.operators import BoundPauliHamiltonian, X,Y,Z
        
        qv = QuantumVariable(2)
        H = 1+2*X(qv[0])+3*X(qv[0])*Y(qv[1])

    Yields $1+2X(qv.0)+3X(qv.0)Y(qv.1)$.

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
        for pauli,coeff in self.terms_dict.items():
            expr += coeff*pauli.to_expr()
        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if self.len()==1:
            if isinstance(e, int) and e>=0:
                if e%2==0:
                    return BoundPauliHamiltonian({BoundPauliTerm():1})
                else:
                    return self
            else:
                raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to add to the operator self.

        Returns
        -------
        result : BoundPauliHamiltonian
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = BoundPauliHamiltonian({BoundPauliTerm():other})
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot add BoundPauliHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for pauli,coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
    
        for pauli,coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
        
        result = BoundPauliHamiltonian(res_terms_dict)
        return result
    
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to substract from the operator self.

        Returns
        -------
        result : BoundPauliHamiltonian
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = BoundPauliHamiltonian({BoundPauliTerm():other})
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot substract BoundPauliHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for pauli,coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
    
        for pauli,coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)-coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
        
        result = BoundPauliHamiltonian(res_terms_dict)
        return result

    def __rsub__(self,other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to substract the operator self from.

        Returns
        -------
        result : BoundPauliHamiltonian
            The difference of the operator other and self.

        """

        if isinstance(other,(int,float,complex)):
            other = BoundPauliHamiltonian({BoundPauliTerm():other})
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot substract BoundPauliHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for pauli,coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)-coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
    
        for pauli,coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
        
        result = BoundPauliHamiltonian(res_terms_dict)
        return result
    
    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to multiply with the operator self.

        Returns
        -------
        result : BoundPauliHamiltonian
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = BoundPauliHamiltonian({BoundPauliTerm():other})
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot multipliy BoundPauliHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1*pauli2
                res_terms_dict[curr_pauli] = res_terms_dict.get(curr_pauli,0) + curr_coeff*coeff1*coeff2

        result = BoundPauliHamiltonian(res_terms_dict)
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
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[BoundPauliTerm()] = self.terms_dict.get(BoundPauliTerm(),0)+other
            return self
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot add BoundPauliHamiltonian and "+str(type(other)))

        for pauli,coeff in other.terms_dict.items():
            self.terms_dict[pauli] = self.terms_dict.get(pauli,0)+coeff
            if abs(self.terms_dict[pauli])<threshold:
                del self.terms_dict[pauli]       
        return self         

    def __isub__(self,other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[BoundPauliTerm()] = self.terms_dict.get(BoundPauliTerm(),0)-other
            return self
        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot add BoundPauliHamiltonian and "+str(type(other)))

        for pauli,coeff in other.terms_dict.items():
            self.terms_dict[pauli] = self.terms_dict.get(pauli,0)-coeff
            if abs(self.terms_dict[pauli])<threshold:
                del self.terms_dict[pauli]  
        return self
    
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or BoundPauliHamiltonian
            A scalar or a BoundPauliHamiltonian to multiply with the operator self.

        """

        if isinstance(other,(int,float,complex)):
            #other = BoundPauliHamiltonian({BoundPauliTerm():other})
            for term in self.terms_dict:
                self.terms_dict[term] *= other
            return self

        if not isinstance(other,BoundPauliHamiltonian):
            raise TypeError("Cannot multipliy BoundPauliHamiltonian and "+str(type(other)))

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1*pauli2
                res_terms_dict[curr_pauli] = res_terms_dict.get(curr_pauli,0) + curr_coeff*coeff1*coeff2

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
        result : BoundPauliHamiltonian
            The resulting BoundPauliHamiltonian.
        
        """

        res_terms_dict = {}

        for pauli, coeff in self.terms_dict.items():
            curr_pauli, curr_coeff = pauli.subs(subs_dict)
            res_terms_dict[curr_pauli] = res_terms_dict.get(curr_pauli,0) + curr_coeff*coeff

        result = BoundPauliHamiltonian(res_terms_dict)
        return result
    
    #
    # Miscellaneous
    #

    def apply_threshold(self,threshold):
        """
        Removes all Pauli terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the Pauli terms.

        """

        delete_list = []
        for pauli,coeff in self.terms_dict.items():
            if abs(coeff)<threshold:
                delete_list.append(pauli)
        for pauli in delete_list:
            del self.terms_dict[pauli]

    def to_sparse_matrix(self):
        """
        Returns a matrix representing the operator.
    
        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.

        """

        import scipy.sparse as sp
        from scipy.sparse import kron as TP, csr_matrix

        I = csr_matrix([[1,0],[0,1]])

        def get_matrix(P):
            if P=="I":
                return csr_matrix([[1,0],[0,1]])
            if P=="X":
                return csr_matrix([[0,1],[1,0]])
            if P=="Y":
                return csr_matrix([[0,-1j],[1j,0]])
            else:
                return csr_matrix([[1,0],[0,-1]])

        def recursive_TP(keys,pauli_dict):
            if len(keys)==1:
                return get_matrix(pauli_dict.get(keys[0],"I"))
            return TP(get_matrix(pauli_dict.get(keys.pop(0),"I")),recursive_TP(keys,pauli_dict))

        pauli_dicts = []
        coeffs = []

        keys = set()
        for pauli,coeff in self.terms_dict.items():
            curr_dict = pauli.pauli_dict
            keys.update(set(curr_dict.keys()))
            pauli_dicts.append(curr_dict)    
            coeffs.append(coeff)

        keys = set()
        for item in pauli_dicts:
            keys.update(set(item.keys()))
        keys = sorted(keys)
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k])*recursive_TP(keys.copy(),pauli_dicts[k])

        return M

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

    # Commutativity: Partitions the BoundPauliHamiltonian into BoundPauliHamiltonians with pairwise commuting BoundPauliTerms
    def commuting_groups(self):
        r"""
        Partitions the PauliHamiltonian into PauliHamiltonians with pairwise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually commuting.

        Returns
        -------
        groups : list[PauliHamiltonian]
            The partition of the Hamiltonian.
        
        """

        groups = [] # Groups of commuting BoundPauliTerms 

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        for pauli,coeff in sorted_terms:

            commute_bool = False
            if len(groups) > 0:
                for group in groups:
                    for pauli_,coeff_ in group.terms_dict.items():
                        commute_bool = pauli_.commute(pauli)
                        if not commute_bool:
                            break
                    if commute_bool:
                        group.terms_dict[pauli]=coeff
                        break
            if len(groups)==0 or not commute_bool: 
                groups.append(BoundPauliHamiltonian({pauli:coeff}))

        return groups

    # Qubit-wise commutativity: Partitions the BoundPauliHamiltonian into BoundPauliHamiltonians with pairwise qubit-wise commuting BoundPauliTerms
    def commuting_qw_groups(self, show_bases=False):
        r"""
        Partitions the BoundPauliHamiltonian into BoundPauliHamiltonians with pairwise qubit-wise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually qubit-wise commuting.

        Returns
        -------
        groups : list[BoundPauliHamiltonian]
            The partition of the Hamiltonian.
        
        """

        groups = [] # Groups of qubit-wise commuting BoundPauliTerms
        bases = [] # Bases as BoundPauliTerms

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        for pauli,coeff in sorted_terms:

            commute_bool = False
            if len(groups)>0:
                n = len(groups)
                for i in range(n):
                    commute_bool = bases[i].commute_qw(pauli)
                    if commute_bool:
                        bases[i].update(pauli.pauli_dict)
                        groups[i].terms_dict[pauli]=coeff
                        break
            if len(groups)==0 or not commute_bool:
                groups.append(BoundPauliHamiltonian({pauli:coeff}))
                bases.append(pauli.copy())

        if show_bases:
            return groups, bases
        else:
            return groups
    
    #
    # Measurement settings
    #

    def pauli_measurement(self):
        return PauliMeasurement(self)

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

            * qarg : QuantumVariable or list[QuantumVariable]
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """

        from qrisp import conjugate, rx, ry, rz, cx, h, IterationEnvironment, gphase, QuantumSession, merge

        pauli_measurement = self.pauli_measurement()
        bases = pauli_measurement.bases
        indices = pauli_measurement.operators_ind # Indices (Qubits) of Z's in PauliTerms (after change of basis)
        operators_int = pauli_measurement.operators_int
        coeffs = pauli_measurement.coefficients

        def trotter_step(qarg, t, steps):

            if isinstance(qarg,list):
                qubit = qarg[0][0]
            else:
                qubit = qarg[0]

            N = len(bases)
            for k in range(N):
                basis = bases[k].pauli_dict

                with conjugate(change_of_basis_bound)(basis):
                    M = len(indices[k])

                    for l in range(M):
                        if(operators_int[k][l]>0): # Not identity
                            with conjugate(parity_bound)(indices[k][l]):
                                rz(-2*coeffs[k][l]*t/steps,indices[k][l][-1])
                        else: # Identity
                            gphase(coeffs[k][l]*t/steps,qubit)

        def U(qarg, t=1, steps=1, iter=1):

            if isinstance(qarg,list):
                qs = QuantumSession()
                for qv in qarg:
                    merge(qs,qv.qs)
            else:
                qs = qarg.qs

            with IterationEnvironment(qs, iter):
                for i in range(steps):
                    trotter_step(qarg, t, steps)

        return U
    

