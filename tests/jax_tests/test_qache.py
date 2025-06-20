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
from jax import make_jaxpr

def test_qache():
    
    class TracingCounter:
        def __init__(self):
            self.count = 0
    
        def increment(self):
            self.count += 1

    counter = TracingCounter()
    
    @qache
    def inner_function(qv):
        counter.increment()
        h(qv[0])
        cx(qv[0], qv[1])
        res_bl = measure(qv[0])
        return res_bl

    def outer_function():
        qv_0 = QuantumVariable(2)
        qv_1 = QuantumFloat(2)
        
        temp_0 = inner_function(qv_0)
        temp_1 = inner_function(qv_1)
        temp_2 = inner_function(qv_0)
        temp_3 = inner_function(qv_1)
        return temp_0 & temp_1 & temp_2
    
    print(make_jaspr(outer_function)())
    
    # The function has been called four times but only for two different types
    assert counter.count == 2
    
    # Test whether the Jasprs of the qached functions appear only once
    # (i.e. only a single copy is kept and the rest are referenciations)
    
    def main(i):
        
        a = QuantumFloat(i)
        a[:] = 4
        b = QuantumFloat(i)
        b[:] = 5
        
        s = jasp_multiplyer(a, b, inpl_adder = gidney_adder)
    
        return measure(s)
    
    jaspr = make_jaspr(main)(3)
    assert jaspr(4) == 20
    jaspr = jaspr.flatten_environments()
    
    
    # Set up an interpreter that associates a function name and it signature
    # to a Jaspr object id
    jaspr_id_dict = {}
    def eqn_evaluator(eqn, context_dic):
        
        if eqn.primitive.name == "pjit":
            name = eqn.params["name"]
            if isinstance(eqn.params["jaxpr"].jaxpr, Jaspr) and name != "ctrl_env":
                jaspr = eqn.params["jaxpr"].jaxpr
                signature_hash = [name] + [type(invar.aval) for invar in jaspr.invars]
                signature_hash = tuple(signature_hash)
                signature_hash = hash(signature_hash)
                if signature_hash not in jaspr_id_dict:
                    jaspr_id_dict[signature_hash] = id(jaspr)
                    
                if not jaspr_id_dict[signature_hash] == id(jaspr):
                    print(name)
                    assert False
                    
            invalues = extract_invalues(eqn, context_dic)
            outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
            if not isinstance(outvalues, (list, tuple)):
                outvalues = [outvalues]
            insert_outvalues(eqn, context_dic, outvalues)
            return
        
        # If no pjit primitive is found, we return True, which triggers the default
        # interpreter behavior
        return True
    
    # We perform a simulation with the above interpreter
    from qrisp.jasp.evaluation_tools import BufferedQuantumState
    args = [5, BufferedQuantumState()]
    eval_jaxpr(jaspr, eqn_evaluator = eqn_evaluator)(*args)


