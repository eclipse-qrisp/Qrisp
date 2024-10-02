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

from jax.lax import cond

from qrisp.environments import QuantumEnvironment
from qrisp.jisp import extract_invalues, insert_outvalues



class ClControlEnvironment(QuantumEnvironment):
    
    def __init__(self, ctrl_bls, ctrl_state=-1, ctrl_method=None, invert = False):
        
        self.ctrl_bls = ctrl_bls
        
        QuantumEnvironment.__init__(self)
        
        self.env_args = ctrl_bls
    
    def jcompile(self, eqn, context_dic):
        
        args = extract_invalues(eqn, context_dic)
        
        ctrl_vars = args[1:len(self.ctrl_bls)+1]
        env_vars = [args[0]] + args[len(self.ctrl_bls)+1:]
        
        body_jispr = eqn.params["jispr"]
        
        if len(body_jispr.outvars) > 1:
            raise Exception("Found ClControlEnvironment with carry value")
        
        if len(ctrl_vars) > 1:
            cond_bl = True
            for i in range(len(ctrl_vars)):
                cond_bl = cond_bl & ctrl_vars[i]
        else:
            cond_bl = ctrl_vars[0]
        
        def false_fun(*args):
            return args[0]
        
        res_abs_qc = cond(cond_bl, body_jispr.eval, false_fun, *env_vars)
        
        insert_outvalues(eqn, context_dic, res_abs_qc)