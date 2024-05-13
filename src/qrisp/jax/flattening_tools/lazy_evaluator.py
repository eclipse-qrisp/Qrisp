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

from jax.core import JaxprEqn, Literal

def evaluate_eqn(eqn, context_dic):
    
    if eqn.primitive.name == "qcall":
        from qrisp.jax import evaluate_qcall
        return evaluate_qcall(eqn, context_dic)
    
    else:
        return evaluate_default(eqn, context_dic)
        

def evaluate_default(eqn, context_dic):
    
    invalues = []
    
    for i in range(len(eqn.invars)):
        
        invar = eqn.invars[i]
        
        if isinstance(invar, Literal):
            invalues.append(invar.val)
            continue
        elif isinstance(context_dic[invar], JaxprEqn):
            
            sub_eq = context_dic[invar]
            
            res = evaluate_eqn(sub_eq, context_dic)
            
            if sub_eq.primitive.multiple_results:
                for i in range(len(sub_eq.outvars)):
                    context_dic[sub_eq.outvars[i]] = res[i]
            else:
                context_dic[sub_eq.outvars[0]] = res
            
        invalues.append(context_dic[invar])
        
    return eqn.primitive.bind(*invalues, **eqn.params)
    
    
            