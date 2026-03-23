"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

def test_prefix_control():
    
    @jaspify
    def main(k):
        
        qf_ = QuantumFloat(6)
        
        def body_fun(i, val):
            acc, qf = val
            x(qf[i])
            acc += measure(qf[i])
            return acc, qf
        
        acc, qf = q_fori_loop(0, k, body_fun, (0, qf_))
        
        return acc, measure(qf)

    assert main(6) == (6, 63)

    @jaspify
    def main(k):
        
        qf = QuantumFloat(6)
        
        def body_fun(i, qf):
            x(qf[i])
            return qf
        
        qf = q_fori_loop(0, k, body_fun, qf)
        
        return measure(qf)

    assert main(6) == 63


    @jaspify
    def main(k):
        
        qf = QuantumFloat(6)
        
        def body_fun(val):
            i, acc, qf = val
            x(qf[i])
            acc += measure(qf[i])
            i += 1
            return i, acc, qf
        
        def cond_fun(val):
            return val[0] < 5
        
        i, acc, qf = q_while_loop(cond_fun, body_fun, (0, 0, qf))
        
        return acc, measure(qf)

    assert main(6) == (5, 31)
    
    @jaspify
    def main():
        
        qf = QuantumFloat(7)
        
        def body_fun(val):
            i, acc, qf = val
            x(qf[i])
            QFT(qf)
            acc += measure(qf)
            
            i += 1
            return (i, acc, qf)
        
        def cond_fun(val):
            return val[0] < 5
        
        i, acc, qf = q_while_loop(cond_fun, body_fun, (0, 0, qf))
        return acc ,measure(qf)

    main()

    @jaspify
    def main(k):
        
        qf = QuantumFloat(6)
        
        def body_fun(val):
            i, qf = val
            x(qf[i])
            i += 1
            return (i, qf)
        
        def cond_fun(val):
            return val[0] < 5
        
        i, qf = q_while_loop(cond_fun, body_fun, (0, qf))
        
        return measure(qf)

    assert main(6) == 31


    @jaspify
    def main():
        
        def false_fun(qbl):
            qbl.flip()
            return qbl
        
        def true_fun(qbl):
            return qbl
        
        qbl = QuantumBool()
        h(qbl)
        pred = measure(qbl)
        
        qbl = q_cond(pred, 
                    true_fun, 
                    false_fun, 
                    qbl)
        
        return measure(qbl)

    assert main()
    
    # Test constant handling problems from https://github.com/eclipse-qrisp/Qrisp/issues/308    
    @jaspify
    def test0():
        N_outer = jnp.array([3])
        bi0 = jnp.array([0])
        return lax.while_loop(lambda x: jnp.all(x==bi0), lambda x:x + N_outer, bi0)
    
    test0()
    
    
    @jaspify
    def test5():
        N_outer = jnp.array([3])
        bi0 = jnp.array([0])
        return q_while_loop(lambda x: jnp.all(x==bi0), lambda x:x + N_outer, bi0)
    
    test5()
