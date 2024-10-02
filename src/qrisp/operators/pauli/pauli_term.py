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

from qrisp.operators.pauli.spin import X_,Y_,Z_

#
# Helper functions
#

def mul_helper(P1,P2):
    pauli_table = {("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",-1j),
            ("Y","X"):("Z",-1j),("Y","Y"):("I",1),("Y","Z"):("X",1j),
            ("Z","X"):("Y",1j),("Z","Y"):("X",-1j),("Z","Z"):("I",1)}
    
    if P1=="I":
        return (P2,1)
    if P2=="I":
        return (P1,1)
    return pauli_table[(P1,P2)]

#
# PauliTerm
#

class PauliTerm:
    r"""
    
    """

    def __init__(self, pauli_dict={}):
        self.pauli_dict = pauli_dict
        self.hash_value = hash(tuple(sorted(pauli_dict.items())))

    def update(self, update_dict):
        self.pauli_dict.update(update_dict)
        self.hash_value = hash(tuple(sorted(self.pauli_dict.items())))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
    
    def copy(self):
        return PauliTerm(self.pauli_dict.copy())
    
    #
    # Printing
    #

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)

    def to_expr(self):
        """
        Returns a SymPy expression representing the PauliTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the PauliTerm.

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
        for index,P in self.pauli_dict.items():
            expr *= to_spin(P,str(index))

        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return PauliTerm({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

    def __mul__(self, other):
        result_pauli_dict={}
        result_coeff = 1
        a = self.pauli_dict
        b = other.pauli_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))
        for key in sorted(keys):
            pauli, coeff = mul_helper(a.get(key,"I"),b.get(key,"I"))
            if pauli!="I":
                result_pauli_dict[key]=pauli
                result_coeff *= coeff
        return PauliTerm(result_pauli_dict), result_coeff
    
    #
    # Commutativity checks
    #

    def commute(self, other):
        """
        Checks if two PauliTerms commute.

        """
        a = self.pauli_dict
        b = other.pauli_dict

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
        Checks if two PauliTerms commute qubit-wise.

        """
        a = self.pauli_dict
        b = other.pauli_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                return False
        return True
    

