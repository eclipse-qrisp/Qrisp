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
import warnings

from qrisp.operators import Hamiltonian
from qrisp.operators.fermionic.fermionic_term import FermionicTerm
from qrisp.operators.fermionic.trotterization import fermionic_trotterization
from qrisp.operators.hamiltonian_tools import group_up_iterable
from qrisp.operators.qubit import QubitOperator

import sympy as sp

threshold = 1e-9

#
# FermionicOperator
#

class FermionicOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of ladder term operators, i.e.,
    operators of the form

    .. math::
        
        O=\sum\limits_{j}\alpha_jO_j 
            
    where each term $O_j$ is a product of fermionic raising $a_i^{\dagger}$ and lowering $a_i$ operators acting on the $i$ th fermionic mode.

    The ladder operators satisfy the commutation relations

    .. math::

        \{a_i,a_j^{\dagger}\} &= a_ia_j^{\dagger}+a_j^{\dagger}a_i = \delta_{ij}\\
        \{a_i^{\dagger},a_j^{\dagger}\} &= \{a_i,a_j\} = 0

    Examples
    --------

    A ladder term operator can be specified conveniently in terms of ``a`` (lowering, i.e., annihilation), ``c`` (raising, i.e., creation) operators:

    ::
        
        from qrisp.operators.fermionic import a, c

        O = a(2)*c(1)+a(3)*c(2)
        O

    Yields $a_2c_1+a_3c_2$.

    """

    def __init__(self, terms_dict={}):
        
        self.terms_dict = dict(terms_dict)

    def reduce(self, assume_hermitian = False):
        """
        Applies the fermionic anticommutation laws to bring the operator into
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
        FermionicOperator
            The reduced FermionicOperator.

        Examples
        --------
        
        We create a FermionicOperator with redundant term definitions:
            
        ::
            
            from qrisp.operators import *
            
            O = a(0)*a(1) - a(1)*a(0)
            print(O.reduce())
            # Yields: 2*a0*a1

        To demonstrate the ``assume_hermitian`` feature, we create a FermionicOperator
        that has redundant terms, if hermitized.
        
            
        >>> O = a(0)*a(1) + c(1)*c(0)
        >>> reduced_O = O.reduce(assume_hermitian = True)
        >>> print(reduced_O)
        2*a0*a1
        
        Hermitizing gives the original operator.
        
        >>> print(reduced_O.hermitize())
        1.0*a0*a1 + 1.0*c1*c0
            
        """
        
        # This function performs some non trivial logic.
        
        # The problem here is that each fermionic term can
        # be reshaped into several different forms and still express
        # the same operator. 
        # For instance, the operator can be arbitrarily reordered
        # if the permutation sign is take care of and no creators/annihilators
        # with the same index are swapped.
        # Furthermore the Hamiltonian must be Hermitian, so each
        # term must be equivalent to is Hermitian conjugate.
        
        # This function implements a storage system, that combines
        # the coefficients of differing terms representing the same operator
        # into a single term.
        
        # This dictionary will contain the new terms, where redundancies
        # are taken care of.
        new_terms_dict = {}
        
        for term, coeff in self.terms_dict.items():
            
            # We only store the sorted version of each term.
            # Sorting here means permuting the creators/annihilators
            # while considering the sign of the permutation applied by the sort.
            # The sort is performed in a stable manner, so terms like a(0)*c(0)
            # don't get permuted (this would be a non-trivial anti-commutator).
            sorted_term, flip_sign = term.sort()
            if sorted_term not in new_terms_dict and assume_hermitian:
                # If the sorted term is not in the terms dict, the sorted version
                # of the daggering might be.
                daggered_sorted_term, daggered_flip_sign = term.dagger().sort()
                if daggered_sorted_term in new_terms_dict:
                    sorted_term = daggered_sorted_term
                    flip_sign = daggered_flip_sign
                                
            # Compute the new coefficient.
            new_terms_dict[sorted_term] = flip_sign*coeff + new_terms_dict.get(sorted_term, 0)
            
        for term, coeff in list(new_terms_dict.items()):
            if isinstance(coeff, (int, float)):
                if coeff == 0:
                    del new_terms_dict[term]
            
        return FermionicOperator(new_terms_dict)

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
        for ladder_term,coeff in self.terms_dict.items():
            expr += coeff*ladder_term.to_expr()
        return expr

    #
    # Arithmetic
    #
    
    def dagger(self):
        r"""
        Returns the daggered/adjoint version of self.

        Returns
        -------
        FermionicOperator
            The Operator $O^\dagger$.

        Examples
        --------
        
        We create a FermionicOperator and dagger it:
            
        ::
            
            from qrisp.operators import *
            
            O = a(0)*c(1)*a(2) + a(3)
            print(O.dagger())
            # Yields: c2*a1*c0 + c3
        """
        terms_dict = {}
        for term, coeff in self.terms_dict.items():
            terms_dict[term.dagger()] = np.conj(coeff)
        return FermionicOperator(terms_dict)    
    
    def hermitize(self):
        r"""
        Returns the hermitized version of self.

        Returns
        -------
        FermionicOperator
            The Operator $(O + O^\dagger)/2$.

        Examples
        --------
        
        We create a FermionicOperator and hermitize it:
            
        ::
            
            from qrisp.operators import *
            
            O = a(0)*c(1)*a(2) + a(3)
            print(O.hermitize())
            # Yields: 0.5*a0*c1*a2 + 0.5*a3 + 0.5*c2*a1*c0 + 0.5*c3
            
        """
        return 0.5*(self + self.dagger())
    
    def __eq__(self, other):
        reduced_self = self.reduce()
        reduced_other = other.reduce()
        
        if len(reduced_self.terms_dict) != len(reduced_other.terms_dict):
            return False
        
        for term, coeff in reduced_self.terms_dict.items():
            if not term in other.terms_dict:
                daggered_sorted_term, flip_sign = term.dagger().sort()
                if daggered_sorted_term not in reduced_other.terms_dict:
                    return False
                elif reduced_self.terms_dict[term] != flip_sign*reduced_other.terms_dict[daggered_sorted_term]:
                    return False
                continue
                    
            if reduced_self.terms_dict[term] != reduced_other.terms_dict[term]:
                return False
        
        return True
    
    def __neg__(self):
        return -1*self
            
    #def __pow__(self, e):
    #    if self.len()==1:
    #        if isinstance(e, int) and e>=0:
    #            if e%2==0:
    #                return FermionicOperator({FermionicTerm():1})
    #            else:
    #                return self
    #        else:
    #            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))
    #    else:
    #        raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to add to the operator self.

        Returns
        -------
        result : FermionicOperator
            The sum of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = FermionicOperator({FermionicTerm():other})
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot add FermionicOperator and "+str(type(other)))

        res_terms_dict = {}

        for ladder_term,coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)+coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
    
        for ladder_term,coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)+coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
        
        result = FermionicOperator(res_terms_dict)
        return result
    
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to substract from the operator self.

        Returns
        -------
        result : FermionicOperator
            The difference of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = FermionicOperator({FermionicTerm():other})
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot substract FermionicOperator and "+str(type(other)))

        res_terms_dict = {}

        for ladder_term,coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)+coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
    
        for ladder_term,coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)-coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
        
        result = FermionicOperator(res_terms_dict)
        return result
    
    def __rsub__(self,other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to substract from the operator self from.

        Returns
        -------
        result : FermionicOperator
            The difference of the operator other and self.

        """

        if isinstance(other,(int,float,complex)):
            other = FermionicOperator({FermionicTerm():other})
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot substract FermionicOperator and "+str(type(other)))

        res_terms_dict = {}

        for ladder_term,coeff in self.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)-coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
    
        for ladder_term,coeff in other.terms_dict.items():
            res_terms_dict[ladder_term] = res_terms_dict.get(ladder_term,0)+coeff
            if abs(res_terms_dict[ladder_term])<threshold:
                del res_terms_dict[ladder_term]
        
        result = FermionicOperator(res_terms_dict)
        return result

    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to multiply with the operator self.

        Returns
        -------
        result : FermionicOperator
            The product of the operator self and other.

        """

        if isinstance(other,(int,float,complex)):
            other = FermionicOperator({FermionicTerm():other})
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot multipliy FermionicOperator and "+str(type(other)))

        res_terms_dict = {}

        for ladder_term1, coeff1 in self.terms_dict.items():
            for ladder_term2, coeff2 in other.terms_dict.items():
                curr_ladder_term = ladder_term1*ladder_term2
                res_terms_dict[curr_ladder_term] = res_terms_dict.get(curr_ladder_term,0) + coeff1*coeff2

        result = FermionicOperator(res_terms_dict)
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
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to add to the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[FermionicTerm()] = self.terms_dict.get(FermionicTerm(),0)+other
            return self
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot add FermionicOperator and "+str(type(other)))

        for ladder_term,coeff in other.terms_dict.items():
            self.terms_dict[ladder_term] = self.terms_dict.get(ladder_term,0)+coeff
            if abs(self.terms_dict[ladder_term])<threshold:
                del self.terms_dict[ladder_term]
        self.terms_dict = FermionicOperator(self.terms_dict).terms_dict
        return self         

    def __isub__(self,other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to substract from the operator self.

        """

        if isinstance(other,(int,float,complex)):
            self.terms_dict[FermionicTerm()] = self.terms_dict.get(FermionicTerm(),0)-other
            return self
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot add FermionicOperator and "+str(type(other)))

        for ladder_term,coeff in other.terms_dict.items():
            self.terms_dict[ladder_term] = self.terms_dict.get(ladder_term,0)-coeff
            if abs(self.terms_dict[ladder_term])<threshold:
                del self.terms_dict[ladder_term]  
        return self
    
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or FermionicOperator
            A scalar or a FermionicOperator to multiply with the operator self.

        """

        if isinstance(other,(int,float,complex)):
            other = FermionicOperator({FermionicTerm():other})
        if not isinstance(other,FermionicOperator):
            raise TypeError("Cannot multipliy FermionicOperator and "+str(type(other)))

        res_terms_dict = {}

        for ladder_term1, coeff1 in self.terms_dict.items():
            for ladder_term2, coeff2 in other.terms_dict.items():
                curr_ladder_term = ladder_term1*ladder_term2
                res_terms_dict[curr_ladder_term] = res_terms_dict.get(curr_ladder_term,0) + coeff1*coeff2

        self.terms_dict = res_terms_dict    

    #
    # Miscellaneous
    #

    def apply_threshold(self,threshold):
        """
        Removes all ladder_term terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the ladder_term terms.

        """

        delete_list = []
        for ladder_term,coeff in self.terms_dict.items():
            if abs(coeff)<threshold:
                delete_list.append(ladder_term)
        for ladder_term in delete_list:
            del self.terms_dict[ladder_term]

    def to_sparse_matrix(self, mapping_type = "jordan_wigner"):
        """
        Returns a matrix representing the operator.
    
        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.
        mapping_type : string, optional
            How to embedd the fermionic terms into a QubitOperator. Currently 
            only ``jordan_wigner`` is supported.

        """
        return self.to_qubit_operator(mapping_type=mapping_type).to_sparse_matrix()

    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.
    
        Returns
        -------
        float
            The ground state energy. 

        """
        return self.to_qubit_operator().ground_state_energy()
    
    @classmethod
    def from_pyscf(self, pyscf_molecular_data):
        """
        .. _pscf_loading:
            
        Loads the data of a `PySCF molecule <https://pyscf.org/user/gto.html>`_ 
        into a FermionicOperator.

        Parameters
        ----------
        pyscf_molecular_data : pyscf.gto.mole.Mole
            The molecule to load.

        Returns
        -------
        molecule_hamiltonian : FermionicOperator
            The molecule as an operator.
            
        Examples
        --------
        
        We load the Hydrogen molecule and perform a hamiltonian simulation:

        >>> from pyscf import gto
        >>> mol = gto.M(atom = '''H 0 0 0; H 0 0 0.74''', basis = 'sto-3g')
        >>> H = FermionicOperator.from_pyscf(mol)
        >>> print(H)
        -0.181210462015197*a0*a1*c2*c3 + 0.181210462015197*a0*c1*c2*a3 
        - 1.25330978664598*c0*a0 + 0.674755926814448*c0*a0*c1*a1 
        + 0.482500939335616*c0*a0*c2*a2 + 0.663711401350814*c0*a0*c3*a3 
        + 0.181210462015197*c0*a1*a2*c3 - 0.181210462015197*c0*c1*a2*a3
        - 1.25330978664598*c1*a1 + 0.663711401350814*c1*a1*c2*a2 
        + 0.482500939335616*c1*a1*c3*a3 - 0.475068848772178*c2*a2 
        + 0.697651504490463*c2*a2*c3*a3 - 0.475068848772178*c3*a3
        
        Create a :ref:`QuantumVariable` and initialize two electrons in the upper
        orbitals. 

        >>> from qrisp import QuantumVariable
        >>> electron_state = QuantumVariable(4)
        >>> electron_state[:] = "0011"
        
        Simulate for $t = 100$ `Angstrom seconds <https://en.wikipedia.org/wiki/Angstrom>`_.
        
        >>> U = H.trotterization()
        >>> U(electron_state, t = 100, steps = 20)
        >>> print(electron_state)
        {'0011': 0.75331, '1100': 0.24669}
        
        We see that the electrons decayed to one of the lower levels. How cool is that?!
        
        """
        from qrisp.algorithms.vqe.problems.electronic_structure import create_electronic_hamiltonian
        return create_electronic_hamiltonian(pyscf_molecular_data)

    #
    # Transformations
    #

    def to_qubit_operator(self, mapping_type='jordan_wigner'):
        """
        Transforms the FermionicOperator to a :ref:`QubitOperator`.

        Parameters
        ----------
        mapping : str, optional
            The mapping to transform the Hamiltonian. Available is ``jordan_wigner``.
            The default is ``jordan_wigner``.

        Returns
        -------
        O : :ref:`QubitOperator`
            The resulting QubitOperator.
            
        Examples
        --------
        
        We map a singular fermionic ladder operator to a QubitOperator to see
        the Jordan-Wigner embedding.
            
        >>> from qrisp.operators import a
        >>> O = a(4)
        >>> print(O.to_qubit_operator())
        Z_0*Z_1*Z_2*Z_3*A_4
        
        """
        
        if mapping_type=="jordan_wigner":
            res = QubitOperator({})
            for term, coeff in self.terms_dict.items():
                res += coeff*term.to_qubit_term(mapping_type="jordan_wigner")
            return res
        else:
            raise Exception(f"Don't know fermionic mapping {mapping_type}.")
    
    #
    # Measurement
    #

    def get_measurement(
        self,
        qarg,
        mapping_type="jordan_wigner",
        **measurement_kwargs
    ):
        r"""

        .. warning::

            This method will no longer be supported in a later release of Qrisp. Instead please migrate to :meth:`expectation_value <qrisp.operators.fermionic.FermionicOperator.expectation_value>`.


        This method returns the expected value of a Hamiltonian for the state 
        of a quantum argument. Note that this method measures the **hermitized**
        version of the operator:
            
        .. math::
            
            H = (O + O^\dagger)/2

        Parameters
        ----------
        qarg : :ref:`QuantumVariable` or list[Qubit]
            The quantum argument to evaluate the Hamiltonian on.
        mapping_type : str, optional
            The strategy on how to map the FermionicOperator to a QubitOperator. Default is ``jordan_wigner``.
        measurement_kwargs : dict, optional
            The keyword arguments of :meth:`QubitOperator.get_measurement`.

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

        We create a FermionicOperator and perform a measurement.
        
        >>> from qrisp.operators import *
        >>> from qrisp import QuantumVariable
        >>> qv = QuantumVariable(4)
        >>> O = a(0)*a(1) + a(2)*c(1) + c(2)*a(3)
        >>> print(O.get_measurement(qv))
        -0.007968127490039834

        """

        warnings.warn("DeprecationWarning: This method will no longer be supported in a later release of Qrisp. Instead please migrate to .expectation_value.")

        qubit_operator = self.to_qubit_operator(mapping_type)
        return qubit_operator.get_measurement(qarg, **measurement_kwargs)
    

    def expectation_value(
        self,
        state_prep,
        mapping_type = "jordan_wigner",
        **measurement_kwargs
        ):
        r"""
        The ``expectation value`` function allows to estimate the expectation value of a Hamiltonian for a state that is specified by a preparation procedure.
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
        mapping_type : str, optional
            The strategy on how to map the FermionicOperator to a QubitOperator. Default is ``jordan_wigner``.
        measurement_kwargs : dict, optional
            The keyword arguments of :meth:`QubitOperator.expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>`.

        Returns
        -------
        callable
            A function returning an array containing the expectaion value.

        Examples
        --------

        We define a Fermionic Hamiltonian, and measure its expectation value for the state of a :ref:`QuantumFloat`.

        We prepare the state

        .. math::

            \ket{\psi_{\theta}} = (\cos(\theta)\ket{0}+\sin(\theta)\ket{1})^{\otimes 2}

        ::
            
            from qrisp import *
            from qrisp.operators import a,c
            import numpy as np

            def state_prep(theta):
                qv = QuantumFloat(2)

                ry(theta,qv)
    
                return qv

        And compute the expectation value of the Hamiltonion $H=a_0^{\dagger}a_1+a_1^{\dagger}a_0$ for the state $\ket{\psi_{\theta}}$

        ::

            H = c(0)*a(1) + c(1)*a(0)

            ev_function = H.expectation_value(state_prep)
            
            print(ev_function(np.pi/2))
            # Yields: 0.5027499999999724

        Similiarly, expectation values can be calculated with Jasp

        ::

            @jaspify(terminal_sampling=True)
            def main():
            
                H = c(0)*a(1) + c(1)*a(0)

                ev_function = H.expectation_value(state_prep)

                return ev_function(np.pi/2)

            print(main())
            # Yields: 0.5027499999999724

        """
        qubit_operator = self.to_qubit_operator(mapping_type)
        return qubit_operator.expectation_value(state_prep, **measurement_kwargs)
        
    #
    # Trotterization
    #
    
    def trotterization(self, forward_evolution = True):
        r"""
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $U(t) = e^{-itH}$ via Trotterization.
        Note that this method will always simulate the **hermitized** operator, i.e.
        
        .. math::
            
            H = (O + O^\dagger)/2
            
        Parameters
        ----------
        forward_evolution, bool, optional
            If set to False $U(t)^\dagger = e^{itH}$ will be executed (usefull for quantum phase estimation). The default is True.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{-itH}$ is 
            approximated by 
            
            .. math::

                e^{-itH}\approx U(t,N)=\left(e^{-iH_1t/N}\dotsb e^{-iH_mt/N}\right)^N

            This function recieves the following arguments:

            * qarg : QuantumVariable or QuantumArray
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U(t,N)$ is applied. The default is 1.
                
        Examples
        --------
        
        We simulate a simple FermionicOperator.
        
        >>> from sympy import Symbol
        >>> from qrisp.operators import a,c
        >>> from qrisp import QuantumVariable
        >>> O = a(0)*a(1) + a(2)
        >>> U = O.trotterization()
        >>> qv = QuantumVariable(3)
        >>> t = Symbol("t")
        >>> U(qv, t = t)
        >>> print(qv.qs)
        QuantumCircuit:
        ---------------
                    ┌───┐             ┌───┐┌────────────┐┌───┐     ┌───┐      
        qv.0: ────■─┤ X ├─────────────┤ X ├┤ Rz(-0.5*t) ├┤ X ├─────┤ X ├─■────
                  │ └─┬─┘    ┌───┐    └─┬─┘├────────────┤└─┬─┘┌───┐└─┬─┘ │    
        qv.1: ─■──┼───■──────┤ H ├──────■──┤ Rz(-0.5*t) ├──■──┤ H ├──■───┼──■─
               │  │ ┌───┐┌───┴───┴───┐┌───┐└────────────┘     └───┘      │  │ 
        qv.2: ─■──■─┤ H ├┤ Rz(1.0*t) ├┤ H ├──────────────────────────────■──■─
                    └───┘└───────────┘└───┘                                   
        Live QuantumVariables:
        ----------------------
        QuantumVariable qv
        
        Execute a simulation:
            
        >>> print(qv.get_measurement(subs_dic = {t : 0.5}))
        {'000': 0.9242, '001': 0.06026, '110': 0.01459, '111': 0.00095}
            
        """
        return fermionic_trotterization(self, forward_evolution)
    
    def group_up(self, denominator):
        term_groups = group_up_iterable(list(self.terms_dict.keys()), denominator)
        if len(term_groups) == 0:
            return [self]
        groups = []
        for term_group in term_groups:
            O = FermionicOperator({term : self.terms_dict[term] for term in term_group})
            groups.append(O)
            
        return groups
    
    @classmethod
    def from_openfermion(cls, of_fermionic_hamiltonian):
        """
        Imports a FermionicOperator from `OpenFermion <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`_.

        Parameters
        ----------
        of_fermionic_operator : openfermion.FermionOperator
            The OpenFermion operator.

        Returns
        -------
        FermionicOperator
            The equivlanet Qrisp operator.
            
        Examples
        --------
        
        We load the $H_2$ molecule from PySCF via OpenFermion and import it into Qrisp:
            
        ::
            
            import openfermion as of
            import openfermionpyscf as ofpyscf
            # Set molecule parameters
            geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.8))]
            basis = "sto-3g"
            multiplicity = 1
            charge = 0
            
            # Perform electronic structure calculations and
            # obtain Hamiltonian as an InteractionOperator
            hamiltonian = ofpyscf.generate_molecular_hamiltonian(
                geometry, basis, multiplicity, charge
            )
            # Convert to a FermionOperator
            hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
            
            from qrisp.operators import FermionicOperator
            H = FermionicOperator.from_openfermion(hamiltonian_ferm_op)

        """
        
        terms_dict = {}
        
        for term, coeff in of_fermionic_hamiltonian.terms.items():
            
            ladder_list = []
            
            for tup in term[::-1]:
                ladder_list.append((tup[0], bool(tup[1])))
            
            terms_dict[FermionicTerm(ladder_list)] = coeff
            
        return FermionicOperator(terms_dict)
    
    def find_minimal_qubit_amount(self):
        indices = sum([[tup[0] for tup in term.ladder_list] for term in self.terms_dict.keys()], [])
        if len(indices) == 0:
            return 0
        return max(indices)+1

    
                    
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