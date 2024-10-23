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

from qrisp.jasp.jasp_expression.centerclass import Jaspr
from qrisp.jasp import eval_jaxpr, TracingQuantumSession, OperationPrimitive

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
    from qrisp.jasp import Jaspr
    if eqn.primitive.name == "pjit":
        
        new_params = dict(eqn.params)
        
        if isinstance(eqn.params["jaxpr"].jaxpr, Jaspr):
            new_params["jaxpr"] = ClosedJaxpr(new_params["jaxpr"].jaxpr.control(1),
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
            new_params["body_jaxpr"] = ClosedJaxpr(control_jaspr(Jaspr(eqn.params["body_jaxpr"].jaxpr)),
                                              eqn.params["body_jaxpr"].consts)
            
            new_params["body_jaxpr"].jaxpr.outvars.insert(1, new_params["body_jaxpr"].jaxpr.invars[1])
        except:
            
            new_jaxpr = copy_jaxpr(new_params["body_jaxpr"].jaxpr)
            new_jaxpr.invars.insert(1, ctrl_qubit_var)
            new_params["body_jaxpr"] = ClosedJaxpr(new_jaxpr,
                                                   eqn.params["body_jaxpr"].consts)
        
        try:
            new_params["cond_jaxpr"] = ClosedJaxpr(control_jaspr(Jaspr(eqn.params["cond_jaxpr"].jaxpr)),
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

def control_jaspr(jaspr):
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
    from qrisp.jasp import Jaspr, AbstractQubit
    
    ctrl_qubit_var = Var(suffix = "", aval = AbstractQubit(), count = control_var_count[0])
    control_var_count[0] += 1
    
    new_eqns = []
    for eqn in jaspr.eqns:
        if isinstance(eqn.primitive, OperationPrimitive) or eqn.primitive.name in ["pjit", "while"]:
            new_eqns.append(control_eqn(eqn, ctrl_qubit_var))
        elif eqn.primitive.name == "measure":
            raise Exception("Tried to applied quantum control to a measurement")
        else:
            new_eqns.append(eqn)
    
    permeability = dict(jaspr.permeability)
    permeability[ctrl_qubit_var] = True
    
    return Jaspr(permeability = permeability,
                 isqfree = jaspr.isqfree,
                 constvars = jaspr.constvars, 
                 invars = jaspr.invars[:1] + [ctrl_qubit_var] + jaspr.invars[1:], 
                 outvars = jaspr.outvars, 
                 eqns = new_eqns)
        
def multi_control_jaspr(jaspr, num_ctrl, ctrl_state):
    
    from qrisp.jasp import AbstractQubit, make_jaspr
    
    ctrl_vars = [Var(suffix = "", aval = AbstractQubit(), count = control_var_count[0] + _) for _ in range(num_ctrl)]
    control_var_count[0] += num_ctrl
    ctrl_avals = [x.aval for x in ctrl_vars]
    
    return make_jaspr(exec_multi_controlled_jaspr(jaspr, num_ctrl, ctrl_state))(*(ctrl_avals + [var.aval for var in jaspr.invars[1:] + jaspr.constvars]))
    
    
def exec_multi_controlled_jaspr(jaspr, num_ctrls, ctrl_state):
    
    def multi_controlled_jaspr_executor(*args):
        
        qs = TracingQuantumSession.get_instance()
        ctrls = list(args)[:num_ctrls]
        invalues = list(args)[num_ctrls:]
        controlled_jaspr = control_jaspr(jaspr)
        
        from qrisp.circuit import XGate
        
        if num_ctrls == 1:
            
            if ctrl_state == "0":
                qs.append(XGate(), ctrls[0])
            temp = controlled_jaspr.inline(*args)
            if ctrl_state == "0":
                qs.append(XGate(), ctrls[0])
            return temp
            
        else:
            
            from qrisp import QuantumBool
            
            
            # args = list(args)
            # qs.abs_qc = args.pop(0)
            
            # invalues = []
            # for i in range(len(jaspr.invars)-1):
                # invalues.append(args.pop(0))
            # constvalues = args

            mcx_operation = XGate().control(num_ctrls, ctrl_state = ctrl_state)
            
            ctrl_qbl = QuantumBool(name = "ctrl_qbl*")
            ctrl_qb = ctrl_qbl[0]
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
                        
            res = controlled_jaspr.inline(*([ctrl_qb] + invalues))
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
            
            return res
            
    return multi_controlled_jaspr_executor


class ControlledJaspr(Jaspr):
    
    __slots__ = ("base_jaspr", "ctrl_state")
    
    def __init__(self, base_jaspr, ctrl_state):
        
        self.base_jaspr = base_jaspr
        self.ctrl_state = str(ctrl_state)
         
        if ctrl_state == "1" and base_jaspr.ctrl_jaspr is not None:
            controlled_jaspr = base_jaspr.ctrl_jaspr
        else:
            controlled_jaspr = multi_control_jaspr(base_jaspr, len(ctrl_state), ctrl_state)
        
        Jaspr.__init__(self, controlled_jaspr)
        
    def control(self, num_ctrl, ctrl_state = -1):
        
        if isinstance(ctrl_state, int):
            if ctrl_state < 0:
                ctrl_state += 2**num_ctrl
                
            ctrl_state = bin(ctrl_state)[2:].zfill(num_ctrl)
        else:
            ctrl_state = str(ctrl_state)
        
        return ControlledJaspr(self.base_jaspr, ctrl_state + self.ctrl_state)
    
    def inverse(self):
        return ControlledJaspr(self.base_jaspr.inverse(), self.ctrl_state)
        
        
        
