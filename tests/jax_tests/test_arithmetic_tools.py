"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************
"""
from qrisp import *
from qrisp.jasp import *
from math import floor, ceil

def test_min_max(exhaustive = False):
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 3
    #Maximal overlap
    for N in range(2, up_bound):
        for k in range(2**N):
            for j in range(2**N):
                
                a = QuantumFloat(N)
                b = QuantumFloat(N)
                a[:] = j
                b[:] = k
                res_min = q_min(a,b).get_measurement()
                res_max = q_max(a,b).get_measurement()
                assert min(j,k) == list(res_min.keys())[0]
                assert max(j,k) == list(res_max.keys())[0]
    #Superposition    
    c = QuantumFloat(2)
    d = QuantumFloat(2)
    c += 2
    h(c[0])
    d+=1

    res_min = q_min(c,d)
    res_max = q_max(c,d)

    assert multi_measurement([c,d,res_min]) == {(2, 1, 1): 0.5, (3, 1, 1): 0.5}     
    assert multi_measurement([c,d,res_max]) == {(2, 1, 2): 0.5, (3, 1, 3): 0.5}
    
    #Partial overlap
    c = QuantumFloat(4,-2)
    d = QuantumFloat(3,-3)
    c[:] = {0.25: 0.25**0.5, 1.25: 0.75**0.5}
    d[:] = 0.125

    res_min = q_min(c,d)
    res_max = q_max(c,d)

    assert multi_measurement([c,d,res_min]) == {(1.25, 0.125, 0.125): 0.75, (0.25, 0.125, 0.125): 0.25}     
    assert multi_measurement([c,d,res_max]) == {(1.25, 0.125, 1.25): 0.75, (0.25, 0.125, 0.25): 0.25}
    
    #No overlap
    c = QuantumFloat(4,-2)
    d = QuantumFloat(2,-4)
    c[:] = {0.25: 0.25**0.5, 1.25: 0.75**0.5}
    d[:] = 0.125

    res_min = q_min(c,d)
    res_max = q_max(c,d)

    assert multi_measurement([c,d,res_min]) == {(1.25, 0.125, 0.125): 0.75, (0.25, 0.125, 0.125): 0.25}     
    assert multi_measurement([c,d,res_max]) == {(1.25, 0.125, 1.25): 0.75, (0.25, 0.125, 0.25): 0.25}

    #Signed
    c = QuantumFloat(2, signed=True)
    d = QuantumFloat(2, signed=True)
    c[:] = {0: 0.5**0.5, 1: 0.5**0.5}
    d[:] = -2

    res_max = q_max(c,d)
    res_min = q_min(c,d)

    assert multi_measurement([c,d,res_max]) == {(0, -2, 0): 0.5, (1, -2, 1): 0.5}
    assert multi_measurement([c,d,res_min]) == {(0, -2, -2): 0.5, (1, -2, -2): 0.5}
    
def test_min_max_jasp(exhaustive = False):
    @boolean_simulation
    def main(N, j, k):
        a = QuantumFloat(N)
        b = QuantumFloat(N)
        a[:] = j
        b[:] = k
        res_max = q_max(a,b)
        res_min = q_min(a,b)
        return measure(res_max), measure(res_min)
    
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 5
        
    for N in range(2, up_bound):
        for k in range(2**N):
            for j in range(2**N):
                
                meas_res = main(N,j,k)
                
                assert max(j,k) == meas_res[0]
                assert min(j,k) == meas_res[1]


def test_q_floor(exhaustive = False):    
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 4
                
    for N in range(1, up_bound):
        for e in range(N):
            for i in range(2**N):
                a = QuantumFloat(N, -e)

                a[:] = i*2**(-e)
                res_floor = q_floor(a).get_measurement()
                assert floor(i*2**(-e)) == list(res_floor.keys())[0]
                
    c = QuantumFloat(4, -2)
    c[:] = {0.25: 0.25**0.5, 1.25: 0.75**0.5}
    assert q_floor(c).get_measurement() == {1.0: 0.75, 0.0: 0.25}   
    
def test_q_ceil(exhaustive = False):
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 4
                
    for N in range(1, up_bound):
        for e in range(N):
            for i in range(2**N - 2**e + 1):
                a = QuantumFloat(N, -e)

                a[:] = i*2**(-e)
                res_ceil = q_ceil(a).get_measurement()

                assert ceil(i*2**(-e)) == list(res_ceil.keys())[0]

            for i in range(2**N - 2**e + 1, 2**N):
                a = QuantumFloat(N, -e)
                a[:] = i*2**(-e)
                res_ceil = q_ceil(a).get_measurement()

                assert list(res_ceil.keys())[0] == 0.0
                
    c = QuantumFloat(4, -2)
    c[:] = {0.25: 0.25**0.5, 1.25: 0.75**0.5}
    assert q_ceil(c).get_measurement() == {2.0: 0.75, 1.0: 0.25}   

def test_q_fractional(exhaustive = False):
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 4
                
    for N in range(1, up_bound):
        for e in range(N):
            for i in range(2**N):
                a = QuantumFloat(N, -e)

                a[:] = i*2**(-e)
                
                res_frac = q_fractional(a).get_measurement()
                assert (i*2**(-e))%1 == list(res_frac.keys())[0]
                
    c = QuantumFloat(4, -2)
    c[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    assert q_fractional(c).get_measurement() == {0.75: 0.75, 0.25: 0.25}   

def test_q_floor_jasp():

    @boolean_simulation
    def main(N, e, i):
        a = QuantumFloat(N, -e)
        a[:] = i*2.0**(-e)

        b = q_floor(a)
        return measure(b)
    for N in range(1, 8):
        for e in range(N):
            for i in range(2**N):
                
                assert floor(i*2**(-e)) == main(N,e, i)