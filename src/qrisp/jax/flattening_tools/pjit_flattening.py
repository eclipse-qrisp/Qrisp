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
from jax import jit, make_jaxpr

def eval_jaxpr(jaxpr, context_dic = {}):
    """
    Evaluates a Jaxpr using the given context dic to replace variables

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The jaxpr to evaluate.
    context_dic : dict, optional
        The dictionary converting variables to values. The default is {}.

    Returns
    -------
    None.

    """
    
    # Iterate through the equations    
    for eqn in jaxpr.eqns:
        
        
        # Get the invalues (either Literals or from the context_dic)
        invalues = []
        for i in range(len(eqn.invars)):
            
            invar = eqn.invars[i]
            if isinstance(invar, Literal):
                invalues.append(invar.val)
                continue
            invalues.append(context_dic[invar])
        
        
        # Evaluate the primitive
        if eqn.primitive.name == "pjit":
            res = evaluate_pjit_eqn(eqn, context_dic)
        else:
            res = eqn.primitive.bind(*invalues, **eqn.params)
        
        # Insert the values into the context_dic
        if eqn.primitive.multiple_results:
            for i in range(len(eqn.outvars)):
                context_dic[eqn.outvars[i]] = res[i]
        else:
            context_dic[eqn.outvars[0]] = res


def evaluate_pjit_eqn(pjit_eqn, context_dic):
    """
    Inlines a pjit primitive if called in traced mode or executes a gate-wrapped
    circuit if called in QuantumCircuit generation mode.

    Parameters
    ----------
    pjit_eqn : An equation with a pjit primitive
        The equation to execute.
    context_dic : dict
        The context dictionary to execute the.

    Returns
    -------
    return values
        The values/tracers of the pjit execution.

    """
    from qrisp.jax import check_for_tracing_mode
    
    # Set alias for the function definition
    definition_jaxpr = pjit_eqn.params["jaxpr"].jaxpr
    
    invalues = []
    
    # Collect the invalues
    for i in range(len(definition_jaxpr.invars)):
        definition_invar = definition_jaxpr.invars[i]
        invar = pjit_eqn.invars[i]
        
        if isinstance(invar, Literal):
            inval = invar.val
        else:
            inval = context_dic[invar]
            
        invalues.append(inval)
    
    if check_for_tracing_mode():
        return jit(exec_jaxpr, static_argnums = [0,1], inline = True)(definition_jaxpr, *invalues)
    else:
        
        from qrisp.circuit import QuantumCircuit
        # Create new context dic and fill with invalues
        new_context_dic = {}
        for i in range(len(definition_jaxpr.invars)):
            new_context_dic[pjit_eqn.invars[i]] = invalues[i]
        
        # Exchange the QuantumCircuit to an empty one to "track" the function
        if isinstance(invalues[0], QuantumCircuit):
            old_qc = context_dic[pjit_eqn.invars[0]]
            new_qc = old_qc.copy()
            new_context_dic[definition_jaxpr.invars[0]] = new_qc

        # Evaluate the definition
        eval_jaxpr(definition_jaxpr, new_context_dic)
        
        # Add new qubits/clbits to the circuit        
        for qb in set(new_qc.qubits) - set(old_qc.qubits):
            old_qc.add_qubit(qb)

        for cb in set(new_qc.clbits) - set(old_qc.clbits):
            old_qc.add_clbit(cb)
        
        # Append the wrapped old circuit to the new circuit
        old_qc.append(new_qc.to_op(name = pjit_eqn.params["name"]), old_qc.qubits, old_qc.clbits)
        
        
        # Collect the return values
        return_values = []
        for i in range(len(definition_jaxpr.outvars)):
            outvar = definition_jaxpr.outvars[i]
            
            if isinstance(outvar, Literal):
                outval = outvar.val
            else:
                outval = new_context_dic[outvar]
            
            return_values.append(outval)
        
        # Adjust the return value of the QuantumCircuit
        return_values[0] = old_qc
        
        return tuple(return_values)

# Executes a jaxpr
def exec_jaxpr(jaxpr, *args, **kwargs):
    context_dic = {jaxpr.invars[i] : args[i] for i in range(len(args))}
    eval_jaxpr(jaxpr, context_dic)
    return [context_dic[jaxpr.outvars[i]] for i in range(len(jaxpr.outvars))]

# Flattens/Inlines a pjit calls in a jaxpr
def flatten_pjit(jaxpr):
    return make_jaxpr(eval_jaxpr, static_argnums = [0])(jaxpr)
        
 
    