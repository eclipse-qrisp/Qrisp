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


import numpy as np

from qrisp.qtypes.quantum_float import QuantumFloat
from qrisp.qtypes.quantum_bool import QuantumBool
from qrisp.arithmetic import multi_controlled_U_g, hybrid_mult, U_g
from qrisp.core.library import QFT, h, cx, swap
from qrisp.environments import conjugate, control, invert
from qrisp.circuit import Operation
from qrisp.arithmetic.modular_arithmetic.mod_tools import modinv, montgomery_decoder, montgomery_encoder


def qft_basis_adder(addend, target):
    
    if isinstance(addend, int):
        U_g(addend, target)
    elif isinstance(addend, QuantumFloat):
        if addend.signed:
            raise Exception("Signed addition not supported")
        for i in range(*addend.mshape):
            multi_controlled_U_g(target, [addend.significant(i)], 2**i)

# Performs the modular inplace addition b += a
# where a and b don't need to have the same montgomery shift
def montgomery_addition(a, b):
    
    for i in range(len(a)):
        with control(a[i]):
            b += pow(2, i-a.m, a.modulus)

def beauregard_adder(a, b, modulus):
    
    if modulus > 2**a.size:
        raise Exception("Tried to perform modular addition on QuantumFloat with to few qubits")
    if modulus == 2**a.size:
        with conjugate(QFT)(a, exec_swap = False):
            qft_basis_adder(b, a)
        return
    
    reduction_not_necessary = QuantumBool()
    sign = QuantumBool()
    
    
    if isinstance(b, int):
        b = b%modulus
    
    temp = a.duplicate()
    
    a = list(a) + [sign[0]]
    
    
    with conjugate(QFT)(a, exec_swap = False):

        qft_basis_adder(b, a)
        
        with invert():
            qft_basis_adder(modulus, a)

        
        with conjugate(QFT)(a, exec_swap = False, inv = True):
            cx(sign, reduction_not_necessary)
        
        
        with control(reduction_not_necessary):
            qft_basis_adder(modulus, a)
            
        with invert():
            qft_basis_adder(b, a)
        
    
        with conjugate(QFT)(a, exec_swap = False, inv = True):
            cx(sign, reduction_not_necessary)
            reduction_not_necessary.flip()
        
        qft_basis_adder(b, a)
    
    sign.delete()
    reduction_not_necessary.delete()


