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

from qrisp import gphase, rz, cx, conjugate, lifted
from qrisp.operators.qubit.visualization import X_,Y_,Z_

PAULI_TABLE = {("I","I"):("I",1),("I","X"):("X",1),("I","Y"):("Y",1),("I","Z"):("Z",1),
            ("X","I"):("X",1),("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",-1j),
            ("Y","I"):("Y",1),("Y","X"):("Z",-1j),("Y","Y"):("I",1),("Y","Z"):("X",1j),
            ("Z","I"):("Z",1),("Z","X"):("Y",1j),("Z","Y"):("X",-1j),("Z","Z"):("I",1)}

#
# BoundQubitTerm
#

class BoundQubitTerm:
    r"""
    
    """

    def __init__(self, factor_dict={}):
        self.factor_dict = factor_dict
        
        self.hash_value = hash(tuple(sorted(factor_dict.items(), key = lambda x : hash(x))))

    def update(self, update_dict):
        self.factor_dict.update(update_dict)
        self.hash_value = hash(tuple(sorted(self.factor_dict.items())))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
    
    def copy(self):
        return BoundQubitTerm(self.factor_dict.copy())
    
    def is_identity(self):
        return len(self.factor_dict)==0
    
    #
    # Simulation
    #
    
    # Assume that the operator is diagonal after change of basis
    # Implements exp(i*coeff*\prod_j Z_j) where the product goes over all qubits j in self.factor_dict
    @lifted
    def simulate(self, coeff, qubit):

        def parity(qubits):
            n = len(qubits)
            for i in range(n-1):
                cx(qubits[i],qubits[i+1])

        if not self.is_identity():
            qubits = list(self.factor_dict.keys())
            with conjugate(parity)(qubits):
                rz(-2*coeff,qubits[-1])
        else:
            gphase(coeff,qubit)

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
        Returns a SymPy expression representing the BoundQubitTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the BoundQubitTerm.

        """

        def to_spin(P, index):
            if P=="I":
                return 1
            if P=="X":
                return X_(index)
            if P=="Y":
                return Y_(index)
            else:
                return Z_(index)
        
        expr = 1
        for index,P in self.factor_dict.items():
            expr *= to_spin(P,"("+str(index)+")")

        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return BoundQubitTerm({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

    def __mul__(self, other):
        result_factor_dict={}
        result_coeff = 1
        a = self.factor_dict
        b = other.factor_dict

        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            pauli, coeff = PAULI_TABLE[a.get(key,"I"),b.get(key,"I")]
            if pauli!="I":
                result_factor_dict[key]=pauli
                result_coeff *= coeff
        return BoundQubitTerm(result_factor_dict), result_coeff

    def subs(self, subs_dict):
        """
        
        Parameters
        ----------
        subs_dict : dict
            A dictionary with indices (Qubit) as keys and numbers (int, float, complex) as values.

        Returns
        -------
        BoundQubitTerm
            The resulting BoundQubitTerm.
        result_coeff : int, float, complex
            The resulting coefficient.
        
        """
        result_factor_dict=self.factor_dict.copy()
        result_coeff = 1

        for key, value in subs_dict.items():
            if key in result_factor_dict:
                del result_factor_dict[key]
                result_coeff *= value

        return BoundQubitTerm(result_factor_dict), result_coeff
        
    #
    # Commutativity checks
    #

    def commute(self, other):
        """
        Checks if two BoundQubitTerms commute.

        """
        a = self.factor_dict
        b = other.factor_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        # Count non-commuting Pauli operators
        commute = True

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                commute = not commute
        return commute

    def commute_qw(self, other):
        """
        Checks if two BoundQubitTerms commute qubit-wise.

        """
        a = self.factor_dict
        b = other.factor_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                return False
        return True