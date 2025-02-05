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

from qrisp.jasp import qfunc_def
from qrisp import *

def test_function_1(i):
    
    @qfunc_def
    def fan_out(a, b):
        
        cx(a[0], b[0])
        cx(a[0], b[1])
        rz(0.5, a[0])

    a = QuantumVariable(i)
    b = QuantumVariable(i+1)
    
    fan_out(a,b)
    fan_out(a,b)
    
    return measure(b[0])

from jax import make_jaxpr

jaxpr = make_jaxpr(test_function_1)(4)

# Jaxpr darstellen
print(jaxpr)


def compile_inv_environments(closed_jaxpr):
    
    jaxpr = closed_jaxpr.jaxpr
    
    environment_stack = [[]]
    # Loop through equations and compile inversion environments accordingly
    for eqn in jaxpr.eqns:
        
        op_name = eqn.primitive.name
        
        if op_name == "enter_inv" and not mlir_implementation_available[op_name.split("_")[1]]:
            environment_stack.append([])
        elif op_name == "exit_inv" and not mlir_implementation_available[op_name.split("_")[1]]:
            content = environment_stack.pop(-1)
            inv_content = get_adjoint(content)
            
            environment_stack[-1].extend(inv_content)
        else:
            environment_stack[-1].append(eqn)
    
    return core.Jaxpr(closed_jaxpr.consts, jaxpr.invars, jaxpr.outvars, environment_stack[0])


#%%
from qrisp import *
from qrisp.jasp import qfunc_def, evaluate_eqn
def test_function_1(i):
    
    @qfunc_def
    def fan_out(a, b):
        
        cx(a[0], b[0])
        cx(a[0], b[1])
        rz(0.5, a[0])

    a = QuantumVariable(i)
    b = QuantumVariable(i+1)
    
    fan_out(a,b)
    fan_out(a,b)
    
    return measure(b[0])

from jax import make_jaxpr

jaxpr = make_jaxpr(test_function_1)(4)
eqns = jaxpr.eqns

# Jaxpr darstellen
print(jaxpr)

def test(i):
    
    context_dic = {jaxpr.jaxpr.invars[0] : i}
    for eqn in eqns:
        for outvar in eqn.outvars:
            context_dic[outvar] = eqn
    
    return evaluate_eqn(eqn, context_dic)

make_jaxpr(test)(1)