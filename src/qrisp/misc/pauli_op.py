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

from qrisp.misc.spin import *

threshold = 1e-9

#pauli_table = {(1,1):(0,1),(1,2):(3,1j),(1,3):(2,-1j),
#            (2,1):(3,-1j),(2,2):(0,1),(2,3):(1,1j),
#            (3,1):(2,1j),(3,2):(1,-1j),(3,3):(1,1)}

pauli_table = {("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",-1j),
            ("Y","X"):("Z",-1j),("Y","Y"):("I",1),("Y","Z"):("X",1j),
            ("Z","X"):("Y",1j),("Z","Y"):("X",-1j),("Z","Z"):("I",1)}

#
# Helper functions
#

def pauli_mul_np3(P1,P2):
    if P1=="I":
        return (P2,1)
    if P2=="I":
        return (P1,1)
    return pauli_table[(P1,P2)]

def mul2(pauli1,pauli2):
    result_list = []
    result_coeff = 1
    pauli_dict1 = dict(pauli1)
    pauli_dict2 = dict(pauli2)
    keys = set()
    keys.update(set(pauli_dict1.keys()))
    keys.update(set(pauli_dict2.keys()))
    for key in sorted(keys):
        pauli, coeff = pauli_mul_np3(pauli_dict1.get(key,"I"),pauli_dict2.get(key,"I"))
        if pauli!="I":
            result_list.append((key,pauli))
        result_coeff *= coeff
    return tuple(result_list), result_coeff


class PauliOperator:

    def __init__(self, pauli_dict=None, constant=0):
        """

        """
        if pauli_dict is None:
            self.pauli_dict = {}
        else:
            self.pauli_dict = pauli_dict
        self.constant = constant

    @classmethod
    def from_expr(cls, expr):
        pauli_dict, constant = to_Pauli_dict(expr)
        return cls(pauli_dict, constant)

    
    def inpl_add(self,other,factor=1):
        for pauli,coeff in other.pauli_dict.items():
            self.pauli_dict[pauli] = self.pauli_dict.get(pauli,0)+coeff*factor
            if abs(self.pauli_dict[pauli])<threshold:
                del self.pauli_dict[pauli]
        self.constant += other.constant*factor
    

    def __add__(self,other):

        result = PauliOperator()
        res_pauli_dict = {}
        result.constant = self.constant+other.constant

        for pauli,coeff in self.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)+coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
    
        for pauli,coeff in other.pauli_dict.items():
            res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0)+coeff
            if abs(res_pauli_dict[pauli])<threshold:
                del res_pauli_dict[pauli]
        
        result.pauli_dict = res_pauli_dict
        return result

    def __mul__(self,other):

        result = PauliOperator()
        res_pauli_dict = {}
        result.constant = self.constant*other.constant

        for pauli1, coeff1 in self.pauli_dict.items():
            for pauli2, coeff2 in other.pauli_dict.items():
                curr_tuple, curr_coeff = mul2(pauli1,pauli2)
                if len(curr_tuple)>0:
                    res_pauli_dict[curr_tuple] = res_pauli_dict.get(curr_tuple,0) + curr_coeff*coeff1*coeff2
                else:
                    result.constant += curr_coeff*coeff1*coeff2

        if self.constant!=0:
            for pauli, coeff in other.pauli_dict.items():
                res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0) + coeff*self.constant
    
        if other.constant!=0:
            for pauli, coeff in other.pauli_dict.items():
                res_pauli_dict[pauli] = res_pauli_dict.get(pauli,0) + coeff*other.constant

        result.pauli_dict = res_pauli_dict
        return result

    def inpl_scalar_mul(self,constant):
        for pauli,coeff in self.pauli_dict.items():
            self.pauli_dict[pauli] *= constant
        return self

    def scalar_mul(self,constant):
        result = PauliOperator()
        res_pauli_dict = {}
        result.constant = self.constant*constant
        for pauli,coeff in self.pauli_dict.items():
            res_pauli_dict[pauli] = coeff*constant
        result.pauli_dict = res_pauli_dict
        return result

    def to_expr(self):

        return self.constant+from_Pauli_dict(self.pauli_dict)

    
