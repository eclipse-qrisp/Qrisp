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
from functools import lru_cache

from jax import make_jaxpr
from jax.core import Jaxpr, ClosedJaxpr, Literal

from qrisp.jisp.jisp_expression import invert_jispr, multi_control_jispr, collect_environments
from qrisp.jisp import AbstractQuantumCircuit, eval_jaxpr, flatten_pjit, pjit_to_gate, flatten_environments

class Jispr(Jaxpr):
    
    __slots__ = "permeability", "isqfree", "hashvalue"
    
    def __init__(self, *args, permeability = None, isqfree = None, **kwargs):
        
        if len(args) == 1:
            kwargs["jaxpr"] = args[0]
        
        if "jaxpr" in kwargs:
            jaxpr = kwargs["jaxpr"]

            self.hashvalue = hash(jaxpr)
        
            Jaxpr.__init__(self,
                           constvars = jaxpr.constvars,
                           invars = jaxpr.invars,
                           outvars = jaxpr.outvars,
                           eqns = jaxpr.eqns,
                           effects = jaxpr.effects,
                           debug_info = jaxpr.debug_info
                           )
        else:
            self.hashvalue = id(self)
            
            Jaxpr.__init__(self, **kwargs)
            
        self.permeability = {}
        if permeability is None:
            permeability = {}
        for var in self.constvars + self.invars + self.outvars:
            if isinstance(var, Literal):
                continue
            self.permeability[var] = permeability.get(var, None)
        
        self.isqfree = isqfree
            
        if not isinstance(self.invars[0].aval, AbstractQuantumCircuit):
            raise Exception(f"Tried to create a Jispr from data that doesn't have a QuantumCircuit as first argument (got {type(self.invars[0].aval)} instead)")
        
        if not isinstance(self.outvars[0].aval, AbstractQuantumCircuit):
            raise Exception(f"Tried to create a Jispr from data that doesn't have a QuantumCircuit as first entry of return type (got {type(self.outvars[0].aval)} instead)")
        
    def __hash__(self):
        
        return self.hashvalue
    
    def __eq__(self, other):
        if not isinstance(other, Jaxpr):
            return False
        return self.hashvalue == hash(other)
    
    def inverse(self):
        return invert_jispr(self)
    
    def control(self, num_ctrl, ctrl_state = -1):
        return multi_control_jispr(self, num_ctrl, ctrl_state)
    
    def extract_qc(self, *args):
        from qrisp import QuantumCircuit
        jispr = flatten_environments(self)
        eqn_eval_dic = {"pjit" : pjit_to_gate}
        
        res = eval_jaxpr(jispr, eqn_eval_dic = eqn_eval_dic)(*([QuantumCircuit()] + list(args)))
        
        return res
        
        
    def __call__(self, *args):
        
        if len(self.outvars) == 1:
            return None
        
        from qrisp.simulator import BufferedQuantumState
        args = [BufferedQuantumState()] + list(args)
        
        jispr = flatten_environments(self)
        eqn_eval_dic = {"pjit" : pjit_to_gate}
        
        res = eval_jaxpr(jispr, eqn_eval_dic = eqn_eval_dic)(*args)
        
        if len(self.outvars) == 2:
            return res[1]
        else:
            return res[1:]
        
    def inline(self, *args):
        
        from qrisp.jisp import TracingQuantumSession
        
        qs = TracingQuantumSession.get_instance()
        abs_qc = qs.abs_qc
        
        res = eval_jaxpr(self)(*([abs_qc] + list(args)))
        
        if isinstance(res, tuple):
            new_abs_qc = res[0]
            res = res[1:]
        else:
            new_abs_qc = res
            res = None
        qs.abs_qc = new_abs_qc
        return res
        
        
    @classmethod
    @lru_cache(maxsize = int(1E5))
    def from_cache(cls, jaxpr):
        return Jispr(jaxpr = jaxpr)
    
    def to_qir(self, *args):
        from qrisp.jisp import jispr_to_qir
        
        return jispr_to_qir(self, args)
    
    def to_mlir(self, *args):
        from qrisp.jisp import jispr_to_mlir
        return jispr_to_mlir(self, args)
    
    def to_catalyst_jaxpr(self, *args):
        from qrisp.jisp import jispr_to_catalyst_jaxpr
        return jispr_to_catalyst_jaxpr(self, args)
    
    
    


def make_jispr(fun):
    from qrisp.jisp import AbstractQuantumCircuit, TracingQuantumSession
    def jispr_creator(*args, **kwargs):
        
        def ammended_function(abs_qc, *args, **kwargs):
            
            qs = TracingQuantumSession(abs_qc)
            
            res = fun(*args, **kwargs)
            res_qc = qs.abs_qc
            
            TracingQuantumSession.release()
            
            return res_qc, res
        
        jaxpr = make_jaxpr(ammended_function)(AbstractQuantumCircuit(), *args, **kwargs).jaxpr
        
        return recursive_convert(jaxpr)
    
    return jispr_creator

def recursive_convert(jaxpr):
    
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pjit" and isinstance(eqn.outvars[0].aval, AbstractQuantumCircuit):
            eqn.params["jaxpr"] = ClosedJaxpr(recursive_convert(eqn.params["jaxpr"].jaxpr), eqn.params["jaxpr"].consts)
    
    # We "collect" the QuantumEnvironments.
    # Collect means that the enter/exit statements are transformed into Jispr
    # which are subsequently called. Example:
        
    # from qrisp import *
    # from qrisp.jisp import *
    # import jax

    # def outer_function(x):
    #     qv = QuantumVariable(x)
    #     with QuantumEnvironment():
    #         cx(qv[0], qv[1])
    #         h(qv[0])
    #     return qv

    # jaxpr = make_jaxpr(outer_function)(2).jaxpr
    
    # This piece of code results in the following jaxpr
    
    # { lambda ; a:i32[]. let
    #     b:QuantumCircuit = qdef 
    #     c:QuantumCircuit d:QubitArray = create_qubits b a
    #     e:QuantumCircuit = q_env[stage=enter type=quantumenvironment] c
    #     f:Qubit = get_qubit d 0
    #     g:Qubit = get_qubit d 1
    #     h:QuantumCircuit = cx e f g
    #     i:Qubit = get_qubit d 0
    #     j:QuantumCircuit = h h i
    #     _:QuantumCircuit = q_env[stage=exit type=quantumenvironment] j
    #   in (d,) }
    
    jaxpr = collect_environments(jaxpr)
    
    return Jispr.from_cache(jaxpr)

    
        
        
            
            
    
