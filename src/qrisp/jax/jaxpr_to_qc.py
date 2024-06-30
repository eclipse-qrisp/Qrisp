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

from jax.core import Literal
from qrisp.jax import QuantumPrimitive, flatten_pjit, eval_jaxpr

def jaxpr_to_qc(jaxpr):
    """
    Converts a Qrisp-generated Jaxpr into a QuantumCircuit.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The Jaxpr to be converted.
    in_place : TYPE, optional
        If set to False, the AbstractCircuit is copied and the copy is continued to be processed. The default is True.

    Returns
    -------
    qc : QuantumCircuit
        The converted circuit.

    """
    
    def qc_eval_function(*args):
        
        if len(jaxpr.invars) != len(args):
            raise Exception(f"Supplied inaccurate amount of arguments ({len(args)}) for Jaxpr (requires {len(jaxpr.invars)}).")
        
        context_dic = {jaxpr.invars[i] : args[i] for i in range(len(jaxpr.invars))}
        eval_jaxpr(jaxpr, context_dic)
        
        from qrisp.circuit import QuantumCircuit
        for val in context_dic.values():
            if isinstance(val, QuantumCircuit):
                return val
            
        raise Exception("Could not find QuantumCircuit in Jaxpr")
        
    return qc_eval_function
            
