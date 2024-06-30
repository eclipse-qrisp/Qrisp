"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


import sympy as sp
from sympy import Symbol, Quaternion, I

class Spin(Symbol):

    __slots__ = ("axes", "index")

    def __new__(cls, axes, index):
        if axes not in ["X", "Y", "Z"]:
            raise IndexError("Invalid Pauli spin")
        obj = Symbol.__new__(cls, "%s%d" %(axes,index), commutative=False, hermitian=True)
        obj.axes = axes
        obj.index = index
        return obj

    def get_quaternion(self):
        if self.axes == "X":
            return Quaternion(0,I,0,0)
        elif self.axes == "Y":
            return Quaternion(0,0,I,0)
        else:
            return Quaternion(0,0,0,I)


def set_bit(n,k):
    return n | (1 << k)        

def evaluate_observable(observable: int, x: int):
    """
    This method evaluates an observable that is a tensor product of Pauli-:math:`Z` operators
    with respect to a measurement outcome. 
        
    A Pauli operator of the form :math:`\prod_{i\in I}Z_i`, for some finite set of indices :math:`I\subset \mathbb N`, 
    is identified with an integer:
    we identify the Pauli operator with the binary string that has ones at positions :math:`i\in I`
    and zeros otherwise, and then convert this binary string to an integer.
        
    Parameters
    ----------
        
    observable : int
        The observable represented as integer.
     x : int 
         The measurement outcome represented as integer.
        
    Returns
    -------
    int
        The value of the observable with respect to the measurement outcome.
        
    """
        
    if bin(observable & x).count('1') % 2 == 0:
        return 1
    else:
        return -1    
        
def convert_to_spin(quaternion, index):
    return quaternion.a-I*quaternion.b*Spin("X",index)-I*quaternion.c*Spin("Y",index)-I*quaternion.d*Spin("Z",index)


def simplify_spin(expr):
    simplified_expr = 0

    for monomial in expr.as_ordered_terms():
        factors = monomial.as_ordered_factors()

        simplified_factor = 1
        pauli_indices = []
        pauli_dict = {}

        for arg in factors:
            if isinstance(arg, Spin):
                if arg.index in pauli_indices:
                    pauli_dict[arg.index] *= arg.get_quaternion()
                else:
                    pauli_dict[arg.index] = arg.get_quaternion()   
                    pauli_indices.append(arg.index) 

            elif isinstance(arg, sp.core.power.Pow,) and isinstance(arg.args[0], Spin):
                if arg.args[1]%2!=0:
                    if arg.args[0].index in pauli_indices:
                        pauli_dict[arg.args[0].index] *= arg.args[0].get_quaternion()
                    else:
                        pauli_dict[arg.args[0].index] = arg.args[0].get_quaternion()  
                        pauli_indices.append(arg.args[0].index)

            else:
                simplified_factor *= arg

        sorted_pauli_dict = dict(sorted(pauli_dict.items()))

        for index,quaternion in sorted_pauli_dict.items():
            simplified_factor *= convert_to_spin(quaternion, index)
        
        simplified_expr += simplified_factor

    return simplified_expr