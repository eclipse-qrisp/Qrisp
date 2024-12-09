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

from qrisp import *
from qrisp.jasp import *

@qache
def qft(qv):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    n = qv.size
    
    for i in jrange(n):
        # pass
        h(qv[n - 1- i])
        for k in jrange(n - i-1):
            cp(2. * np.pi / pow(2., (k + 2)), qv[n - 1 - (k + i + 1)], qv[n - 1 -i])
    
    for i in jrange(n//2):
        swap(qv[i], qv[n-i-1])
            

# from qrisp import conjugate, QuantumFloat, p, QFT

@qache
def jasp_fourier_adder(a, b):

    n = b.size
    
    with conjugate(qft)(b):
        if isinstance(a, QuantumFloat):
            for i in jrange(a.size):
                with control(a[i]):
                    for j in jrange(b.size):
                        p(np.pi*2.**(j-b.size+1+i), b[j])
    
                    
        else: 
            
            for i in jrange(b.size):
                p(a*np.pi*2.**(i-b.size+1), b[i])
                
@qache(static_argnames = "inpl_adder")
def mod_adder(a, b, modulus, inpl_adder = None, ctrl = None):
    
    reduction_not_necessary = QuantumBool()
    # sign = QuantumBool()
    sign = b[-1]
    
    
    if isinstance(a, int):
        a = a%modulus
    
    # b = list(b) + [sign[0]]
    
    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)
            
    with invert():
        inpl_adder(modulus, b)

    cx(sign, reduction_not_necessary[0])
    
    with control(reduction_not_necessary[0]):
        inpl_adder(modulus, b)
        
    with invert():
        if ctrl is None:
            inpl_adder(a, b)
        else:
            with control(ctrl):
                inpl_adder(a, b)
    
    cx(sign, reduction_not_necessary[0])
    reduction_not_necessary.flip()
    
    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)
    
    reduction_not_necessary.delete()
    
def call_mod_adder():
    
    a = QuantumFloat(5)
    b = QuantumFloat(5)
    
    a[:] = 5
    b[:] = 4
    mod_adder(a, b, 8, inpl_adder = jasp_fourier_adder)
    
    return measure(b)

def test_modular_adder():

    def call_mod_adder():
        
        a = QuantumFloat(5)
        b = QuantumFloat(5)
        
        a[:] = 5
        b[:] = 4
        mod_adder(a, b, 8, inpl_adder = jasp_fourier_adder)
        
        return measure(b)

    jaspr = make_jaspr(call_mod_adder)()
    assert jaspr() == 1

def test_fourier_adder():

    def call_fourier_adder():
        
        a = QuantumFloat(5)
        b = QuantumFloat(5)
        
        a[:] = 5
        b[:] = 4
        jasp_fourier_adder(a, b)
        
        return measure(b)

    jaspr = make_jaspr(call_fourier_adder)()
    assert jaspr() == 9
    
    