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


from qrisp import cx, x, control
from qrisp.alg_primitives.arithmetic.adders import fourier_adder
from qrisp.qtypes.quantum_float import QuantumFloat

def q_int_mult(factor_1, factor_2, inpl_adder = fourier_adder, target_qf = None):
    
    if factor_1.size < factor_2.size:
        factor_1, factor_2 = factor_2, factor_1
    
    
    if factor_1.signed or factor_2.signed:
        raise Exception("Signed ripple multiplication currently not supported")

    n = factor_1.size-1
    if target_qf is None:
        s = QuantumFloat(factor_1.size + factor_2.size + 1, 
                         exponent = factor_1.exponent + factor_2.exponent)
        for i in range(factor_2.size):
            cx(factor_2[i], s[i+1+n])

    else:
        target_qf.extend(1, 0)
        s = target_qf
        inpl_adder(factor_2[:s.size-n-1], s[n+1:])
        
    x(s)
    inpl_adder(factor_2, s)
    
    cx(factor_1[0], s)
    for i in range(factor_1.size):
        
        inpl_adder(factor_2, s[i:])
        
        if i != factor_1.size-1:
            pass
            cx(factor_1[i], factor_1[i+1])
            cx(factor_1[i+1], s)
            cx(factor_1[i], factor_1[i+1])

    cx(factor_1[-1], s)
    x(s)
    s.reduce(s[0], verify = False)

    return s

def inpl_q_int_mult(operand, cl_int, inpl_adder = fourier_adder):
    if not cl_int%2:
        raise Exception("In-place multiplication with even integers not supported")
    
    for i in range(operand.size-1):
        with control(operand[operand.size-2-i]):
            inpl_adder(cl_int//2, operand[operand.size-1-i:])