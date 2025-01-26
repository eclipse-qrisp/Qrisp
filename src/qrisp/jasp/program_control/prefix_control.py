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

from jax.lax import fori_loop, while_loop, cond

from qrisp.jasp.tracing_logic import TracingQuantumSession

def q_fori_loop(lower, upper, body_fun, init_val):
    
    def new_body_fun(i, val):
        qs.start_tracing(val[0])
        res = body_fun(i, val[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_init_val = (abs_qc, init_val)
    fori_res = fori_loop(lower, upper, new_body_fun, new_init_val)
    
    qs.abs_qc = fori_res[0]
    return fori_res[1]
    
    
def q_while_loop(cond_fun, body_fun, init_val):
    
    def new_cond_fun(val):
        temp_qc = qs.abs_qc
        res = cond_fun(val[1])
        if not qs.abs_qc is temp_qc:
            raise Exception("Tried to modify quantum state during while condition evaluation")
        return res
    
    def new_body_fun(val):
        qs.start_tracing(val[0])
        res = body_fun(val[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_init_val = (abs_qc, init_val)
    while_res = while_loop(new_cond_fun, new_body_fun, new_init_val)
    
    qs.abs_qc = while_res[0]
    return while_res[1]

def q_cond(pred, true_fun, false_fun, *operands):
    
    def new_false_fun(*operands):
        qs.start_tracing(operands[0])
        res = false_fun(*operands[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    def new_true_fun(*operands):
        qs.start_tracing(operands[0])
        res = false_fun(*operands[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_operands = (abs_qc, operands)
    
    cond_res = cond(pred, new_true_fun, new_true_fun, *new_operands)
        
    qs.abs_qc = cond_res[0]
    return cond_res[1]
