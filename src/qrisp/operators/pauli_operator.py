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
from qrisp.operators import Hamiltonian
from qrisp.operators.helper_functions import *
from qrisp.operators.pauli_term import PauliTerm
from qrisp.operators.pauli_measurement import PauliMeasurement

import sympy as sp

from sympy import init_printing
# Initialize automatic LaTeX rendering
init_printing()

threshold = 1e-9

#
# PauliOperator
#

class PauliOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of Pauli operators, i.e.,
    operators of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j 
            
    where $P_j=\prod_i\sigma_i^j$ is a Pauli product, 
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Parameters
    ----------
    arg : dict, optional
        A dictionary representing a Pauli operator.

    Examples
    --------

    A Pauli operator can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::
        
        from qrisp.operators import PauliOperator, X,Y,Z

        P1 = 1+2*X(0)+3*X(0)*Y(1)

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
                    return PauliOperator({PauliTerm():1})
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
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to add to the operator self.

        Returns
        -------
        result : PauliOperator
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({PauliTerm():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

        res_terms_dict = {}

        for pauli,coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
    
        for pauli,coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
        
        result = PauliOperator(res_terms_dict)
        return result
    
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to substract from the operator self.

        Returns
        -------
        result : PauliOperator
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({PauliTerm():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot substract PauliOperator and "+str(type(other)))

        res_terms_dict = {}

        for pauli,coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)+coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
    
        for pauli,coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli,0)-coeff
            if abs(res_terms_dict[pauli])<threshold:
                del res_terms_dict[pauli]
        
        result = PauliOperator(res_terms_dict)
        return result

    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to multiply with the operator self.

        Returns
        -------
        result : PauliOperator
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({PauliTerm():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot multipliy PauliOperator and "+str(type(other)))

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1*pauli2
                res_terms_dict[curr_pauli] = res_terms_dict.get(curr_pauli,0) + curr_coeff*coeff1*coeff2

        result = PauliOperator(res_terms_dict)
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
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[PauliTerm()] = self.terms_dict.get(PauliTerm(),0)+other
            return self
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

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
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[PauliTerm()] = self.terms_dict.get(PauliTerm(),0)-other
            return self
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot add PauliOperator and "+str(type(other)))

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
        other : int, float, complex or PauliOperator
            A scalar or a PauliOperator to multiply with the operator self.

        """

        if isinstance(other,(int,float,complex)):
            other = PauliOperator({PauliTerm():other})
        if not isinstance(other,PauliOperator):
            raise TypeError("Cannot multipliy PauliOperator and "+str(type(other)))

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1*pauli2
                res_terms_dict[curr_pauli] = res_terms_dict.get(curr_pauli,0) + curr_coeff*coeff1*coeff2

        self.terms_dict = res_terms_dict    

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

    # Commutativity: Partitions the PauliOperator into PauliOperators with pairwise commuting PauliTerms
    def commuting_groups(self):

        groups = [] # Groups of commuting PauliTerms 

        for pauli,coeff in self.terms_dict.items():

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
                groups.append(PauliOperator({pauli:coeff}))

        return groups

    # Qubit-wise commutativity: Partitions the PauliOperator into PauliOperators with pairwise qubit-wise commuting PauliTerms
    def commuting_qw_groups(self, show_bases=False):

        groups = [] # Groups of qubit-wise commuting PauliTerms
        bases = [] # Bases as PauliTerms

        for pauli,coeff in self.terms_dict.items():

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
                groups.append(PauliOperator({pauli:coeff}))
                bases.append(pauli.copy())

        if show_bases:
            return groups, bases
        else:
            return groups
    
    #
    # Measurement settings
    #

    def commuting_measurement(self):
        return 0
    
    def commuting_qw_measurement(self):

        groups, bases = self.commuting_qw_groups(show_bases=True)
        operators_ind = []
        operators_int = []
        coefficients = []

        for group in groups:
            curr_ind = []
            curr_int = []
            curr_coeff = []
            for pauli,coeff in group.terms_dict.items():
                ind = list(pauli.pauli_dict.keys())
                curr_ind.append(ind)
                curr_int.append(get_integer_from_indices(ind))
                curr_coeff.append(coeff)
            operators_ind.append(curr_ind)
            operators_int.append(curr_int)
            coefficients.append(curr_coeff)

        return PauliMeasurement(bases,operators_ind,operators_int,coefficients)
    
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

            This function recieves the following arguments:

            * qarg : QuantumVariable or QuantumArray
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """

        from qrisp import conjugate, rx, ry, rz, cx, h, IterationEnvironment, gphase

        #bases, ops, indices, coeffs, constant = self.qubit_wise_commutativity()

        pauli_measurement = self.commuting_qw_measurement()
        bases = pauli_measurement.bases
        indices = pauli_measurement.operators_ind
        ops = pauli_measurement.operators_int
        coeffs = pauli_measurement.coefficients

        def trotter_step(qarg, t, steps):

            #if constant != 0:
            #    gphase(t/steps*constant,qarg[0])

            N = len(bases)
            for k in range(N):
                basis = bases[k].pauli_dict
                with conjugate(change_of_basis)(qarg, basis):
                    M = len(ops[k])
                    for l in range(M):
                        with conjugate(parity)(qarg, indices[k][l]):
                            rz(-2*coeffs[k][l]*t/steps,qarg[indices[k][l][-1]])

        def U(qarg, t=1, steps=1, iter=1):
            with IterationEnvironment(qarg.qs, iter):
                for i in range(steps):
                    trotter_step(qarg, t, steps)

        return U
    


def X(arg):
    return PauliOperator({PauliTerm({arg:"X"}):1})

def Y(arg):
    return PauliOperator({PauliTerm({arg:"Y"}):1})

def Z(arg):
    return PauliOperator({PauliTerm({arg:"Z"}):1})
