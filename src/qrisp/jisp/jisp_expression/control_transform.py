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

from jax.core import JaxprEqn, ClosedJaxpr, Var, Jaxpr
from qrisp.jisp import eval_jaxpr, TracingQuantumSession, OperationPrimitive

control_var_count = np.zeros(1)

def copy_jaxpr(jaxpr):
    return Jaxpr(constvars = list(jaxpr.constvars),
                      invars = list(jaxpr.invars),
                      outvars = list(jaxpr.outvars),
                      eqns = list(jaxpr.eqns),
                      effects = jaxpr.effects)
    

def control_eqn(eqn, ctrl_qubit_var):
    """
    Receives and equation that describes either an operation or a pjit primitive
    and returns an equation that describes the inverse.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be inverted.

    Returns
    -------
    inverted_eqn
        The equation with inverted operation.

    """
    from qrisp.jisp import Jispr
    if eqn.primitive.name == "pjit":
        
        new_params = dict(eqn.params)
        
        new_params["jaxpr"] = ClosedJaxpr(control_jispr(eqn.params["jaxpr"].jaxpr),
                                          eqn.params["jaxpr"].consts)
        
        return JaxprEqn(primitive = eqn.primitive,
                        invars = [eqn.invars[0], ctrl_qubit_var] + eqn.invars[1:],
                        outvars = eqn.outvars,
                        params = new_params,
                        source_info = eqn.source_info,
                        effects = eqn.effects,)
        return eqn
    elif eqn.primitive.name == "while":
        
        new_params = dict(eqn.params)
        
        try:
            new_params["body_jaxpr"] = ClosedJaxpr(control_jispr(Jispr(eqn.params["body_jaxpr"].jaxpr)),
                                              eqn.params["body_jaxpr"].consts)
            
            new_params["body_jaxpr"].jaxpr.outvars.insert(1, new_params["body_jaxpr"].jaxpr.invars[1])
        except:
            
            new_jaxpr = copy_jaxpr(new_params["body_jaxpr"].jaxpr)
            new_jaxpr.invars.insert(1, ctrl_qubit_var)
            new_params["body_jaxpr"] = ClosedJaxpr(new_jaxpr,
                                                   eqn.params["body_jaxpr"].consts)
        
        try:
            new_params["cond_jaxpr"] = ClosedJaxpr(control_jispr(Jispr(eqn.params["cond_jaxpr"].jaxpr)),
                                              eqn.params["cond_jaxpr"].consts)
        except:
            new_jaxpr = copy_jaxpr(new_params["cond_jaxpr"].jaxpr)
            new_jaxpr.invars.insert(1, ctrl_qubit_var)
            new_params["cond_jaxpr"] = ClosedJaxpr(new_jaxpr,
                                                   eqn.params["cond_jaxpr"].consts)
            
        
        temp = JaxprEqn(primitive = eqn.primitive,
                        invars = [eqn.invars[0], ctrl_qubit_var] + eqn.invars[1:],
                        outvars = eqn.outvars,
                        params = new_params,
                        source_info = eqn.source_info,
                        effects = eqn.effects,)
        
        return temp
    else:
        return JaxprEqn(primitive = eqn.primitive.control(),
                        invars = [eqn.invars[0], ctrl_qubit_var] + eqn.invars[1:],
                        outvars = eqn.outvars,
                        params = eqn.params,
                        source_info = eqn.source_info,
                        effects = eqn.effects,)

def control_jispr(jispr):
    """
    Takes a jaxpr returning a quantum circuit and returns a jaxpr, which returns
    the inverse quantum circuit

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        A jaxpr returning a QuantumCircuit.

    Returns
    -------
    inverted_jaxpr : jaxpr.core.Jaxpr
        A jaxpr returning the inverse QuantumCircuit.

    """
    
    from qrisp.circuit import Operation
    from qrisp.jisp import Jispr, AbstractQubit
    
    ctrl_qubit_var = Var(suffix = "", aval = AbstractQubit(), count = control_var_count[0])
    control_var_count[0] += 1
    
    new_eqns = []
    for eqn in jispr.eqns:
        if isinstance(eqn.primitive, OperationPrimitive) or eqn.primitive.name in ["pjit", "while"]:
            new_eqns.append(control_eqn(eqn, ctrl_qubit_var))
        elif eqn.primitive.name == "measure":
            raise Exception("Tried to applied quantum control to a measurement")
        else:
            new_eqns.append(eqn)
    
    permeability = dict(jispr.permeability)
    permeability[ctrl_qubit_var] = True
    
    return Jispr(permeability = permeability,
                 isqfree = jispr.isqfree,
                 constvars = jispr.constvars, 
                 invars = jispr.invars[:1] + [ctrl_qubit_var] + jispr.invars[1:], 
                 outvars = jispr.outvars, 
                 eqns = new_eqns)
        
def multi_control_jispr(jispr, num_ctrl = 1, ctrl_state = -1):
    
    if num_ctrl == 1:
        return control_jispr(jispr)
    
    from qrisp.jisp import Jispr, AbstractQubit, make_jispr
    
    ctrl_vars = [Var(suffix = "", aval = AbstractQubit(), count = control_var_count[0] + _) for _ in range(num_ctrl)]
    control_var_count[0] += num_ctrl
    ctrl_avals = [x.aval for x in ctrl_vars]
    
    temp_jaxpr = make_jispr(exec_multi_controlled_jispr(jispr, num_ctrl))(*(ctrl_avals + [var.aval for var in jispr.invars[1:] + jispr.constvars]))
    
    invars = temp_jaxpr.invars[num_ctrl:-len(jispr.constvars)]
    constvars = temp_jaxpr.invars[:num_ctrl] + temp_jaxpr.invars[-len(jispr.constvars):]
    
    res = Jispr(temp_jaxpr)
    
    return res
    
    
def exec_multi_controlled_jispr(jispr, num_ctrls):
    
    def multi_controlled_jispr_executor(*args):
        
        if num_ctrls == 1:
            controlled_jispr = control_jispr(jispr)
            return eval_jaxpr(controlled_jispr)(args[0], *args)
            
        else:
            from qrisp.circuit import XGate
            from qrisp import QuantumBool
            
            

            qs = TracingQuantumSession.get_instance()
            # args = list(args)
            # qs.abs_qc = args.pop(0)
            
            # invalues = []
            # for i in range(len(jispr.invars)-1):
                # invalues.append(args.pop(0))
            # constvalues = args
            ctrls = list(args)[:num_ctrls]
            invalues = list(args)[num_ctrls:]
            
            controlled_jispr = control_jispr(jispr)
            mcx_operation = XGate().control(num_ctrls)
            
            ctrl_qbl = QuantumBool()
            ctrl_qb = ctrl_qbl[0]
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
                        
            res = controlled_jispr.inline(*(invalues + [ctrl_qb]))
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
            
            return res
            
    return multi_controlled_jispr_executor
        
