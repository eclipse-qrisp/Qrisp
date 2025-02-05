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

from qrisp import QuantumFloat, h, t, x, conjugate, measure, control, QuantumBool, cx
from qrisp.jasp import jaspify, sample, jrange, expectation_value

def double(*args):
    if len(args) == 1:
        return 2*args[0]
    return tuple([2*x for x in args])

def test_sampling():
    
    def inner_f(i):
        qf = QuantumFloat(4)
        
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])
                
        return qf

    @jaspify
    def main():
        res = sample(inner_f, 500)(2)
        return res
    
    assert set(int(i) for i in main()) == {0,1}
    
    @jaspify(terminal_sampling = True)
    def main():
        res = sample(inner_f, 500)(2)
        return res
    
    assert set(int(i) for i in main()) == {0,1}

    @jaspify
    def main():
        res = sample(inner_f, 500, post_processor = double)(2)
        return res
    
    assert set(int(i) for i in main()) == {0,2}
    
    
    @jaspify(terminal_sampling = True)
    def main():
        res = sample(inner_f, 500, post_processor = double)(2)
        return res
    
    assert set(int(i) for i in main()) == {0,2}
    
    
    def inner_f(i):
        qf = QuantumFloat(4)
        qf_2 = QuantumFloat(4)
        qf_3 = QuantumFloat(4)
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])
                
        return qf, qf_2, qf_3

    
    @jaspify
    def main():
        
        res = sample(inner_f, 10)(2)
        
        return res
    
    assert main().shape == (10, 3)
    
    @jaspify
    def main():
        
        res = sample(inner_f, 10, post_processor=double)(2)
        
        return res
    
    assert main().shape == (10, 3)
    
    @jaspify(terminal_sampling = True)
    def main():
        
        res = sample(inner_f, 10)(2)
        
        return res
    
    assert main().shape == (10, 3)
    
    @jaspify(terminal_sampling = True)
    def main():
        
        res = sample(inner_f, 10, post_processor=double)(2)
        
        return res
    
    assert main().shape == (10, 3)
    

    @sample
    def main():
        
        qbl = QuantumBool()
        qf = QuantumFloat(4)
        
        # Bring qbl into superposition
        h(qbl)
        
        # Perform a measure
        cl_bl = measure(qbl)
        
        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])
        
        return qf, qbl

    assert main() in [{(1.0, True): 0.5, (5.0, True): 0.5}, {(0.0, False): 1.0}]
    
    @sample
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return qf, a, qbl

    for i in range(3):
        for j in range(3):
            assert main(i, j) == {(0.0, 0.0, False): 0.5, (2**i, 2**j, True): 0.5}

    @sample(500)
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return qf, a, qbl

    assert sum(main(2,2).values()) == 500
    
def test_expectation_value():
    
    def inner_f(i):
        qf = QuantumFloat(4)
        
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])
                
        return qf

    @jaspify(terminal_sampling = True)
    def main():
        res = expectation_value(inner_f, 10000)(2)
        return res
    
    assert abs(main()-0.5) < 0.05
    
    @jaspify
    def main():
        res = expectation_value(inner_f, 500)(2)
        return res
    
    assert abs(main()-0.5) < 0.2
    
    @jaspify(terminal_sampling = True)
    def main():
        res = expectation_value(inner_f, 10000, post_processor=double)(2)
        return res
    
    assert abs(main()-1) < 0.05
    
    @jaspify
    def main():
        res = expectation_value(inner_f, 500, post_processor=double)(2)
        return res
    
    assert abs(main()-1) < 0.2
    
    
    def inner_f(i):
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        
        with conjugate(h)(a):
            for k in jrange(i):
                t(a[0])
                x(b[0])
                
        return a, b

    @jaspify(terminal_sampling = True)
    def main():
        res = expectation_value(inner_f, 10000)(2)
        return res
    
    ev_res = main()
    assert abs(ev_res[0]-0.5) < 0.05 and ev_res[1] == 0
    
    @jaspify
    def main():
        res = expectation_value(inner_f, 500)(2)
        return res
    
    ev_res = main()
    assert abs(ev_res[0]-0.5) < 0.2 and ev_res[1] == 0
    
    @jaspify(terminal_sampling = True)
    def main():
        res = expectation_value(inner_f, 10000, post_processor=double)(2)
        return res
    
    ev_res = main()
    assert abs(ev_res[0]-1) < 0.05 and ev_res[1] == 0
    
    @jaspify
    def main():
        res = expectation_value(inner_f, 500, post_processor=double)(2)
        return res
    
    ev_res = main()
    assert abs(ev_res[0]-1) < 0.2 and ev_res[1] == 0
    
    

    

