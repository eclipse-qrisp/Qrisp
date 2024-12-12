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

from jax.lax import while_loop, cond
import jax

from qrisp.jasp import TracingQuantumSession, delete_qubits_p, AbstractQubitArray

def RUS(trial_function):
    r"""
    Decorator to deploy repeat-until-success (RUS) components. At the core,
    RUS repeats a given quantum subroutine followed by a qubit measurement until 
    the measurement returns the value ``1``. This step is prevalent
    in many important algorithms, among them the 
    `HHL algorithm <https://arxiv.org/abs/0811.3171>`_ or the  
    `LCU procedure <https://arxiv.org/abs/1202.5822>`_.
    
    Within Jasp, RUS steps can be realized by providing the quantum subroutine
    as a "trial function", which returns a boolean value (the repetition condition) and
    possibly other return values.
    
    It is important to note that the trial function can not receive quantum
    arguments. This is because after each trial, a new copy of these arguments
    would be required to perform the next iteration, which is prohibited by
    the no-clone theorem. It is however legal to provide classical arguments.

    Parameters
    ----------
    trial_function : callable
        A function returning a boolean value as the first return value. More 
        return values are possible.

    Returns
    -------
    callable
        A function that performs the RUS protocol with the trial function. The
        return values of this function are the return values of the trial function
        WITHOUT the boolean value.
        
    Examples
    --------
    
    To demonstrate the RUS behavior, we initialize a GHZ state
    
    .. math::
        
        \ket{\psi} = \frac{\ket{00000} + \ket{11111}}{\sqrt{2}}

    and measure the first qubit into a boolean value. This will be the value
    to cancel the repetition. This will collapse the GHZ state into either 
    $\ket{00000}$ (which will cause a new repetition) or $\ket{11111}$, which
    cancels the loop. After the repetition is canceled we are therefore
    guaranteed to have the latter state.
    
    ::
        
        from qrisp.jasp import RUS, make_jaspr
        from qrisp import QuantumFloat, h, cx, measure
        
        @RUS
        def rus_trial_function():
            qf = QuantumFloat(5)
            h(qf[0])
            
            for i in range(1, 5):
                cx(qf[0], qf[i])
            
            cancelation_bool = measure(qf[0])
            return cancelation_bool, qf
        
        def call_RUS_example():
            
            qf = rus_trial_function()
            
            return measure(qf)
        
    Create the ``jaspr`` and simulate:
        
    ::
        
        jaspr = make_jaspr(call_RUS_example)()
        print(jaspr())
        # Yields, 31 which is the decimal version of 11111
            
    """
    
    def return_function(*trial_args):
        
        def body_fun(args):
            
            previous_trial_res = args[1:len(res_vals)+1]
            
            abs_qc = args[0]
            
            for res_val in previous_trial_res:
                if isinstance(res_val.aval, AbstractQubitArray):
                    abs_qc = delete_qubits_p.bind(abs_qc, res_val)
            
            abs_qs.abs_qc = abs_qc
            
            trial_args = [abs_qc] + list(args[len(res_vals)+1:])
            trial_res = trial_func_jaspr.eval(*trial_args)
            
            return tuple(list(trial_res) + list(trial_args)[1:])
        
        def cond_fun(val):
            return ~val[1]
        
        # We now prepare the "init_val" keyword of the loop.
        
        # For that we extract the invalues for the first iteration
        # Note that the treshold is given as the last argument
        
        from qrisp.jasp import make_jaspr
        trial_func_jaspr = make_jaspr(trial_function)(*trial_args)
        trial_func_jaspr = trial_func_jaspr.flatten_environments()
        
        first_iter_res = trial_function(*trial_args)
        
        arg_vals, arg_tree_def = jax.tree.flatten(trial_args)
        res_vals, res_tree_def = jax.tree.flatten(first_iter_res)
        
        abs_qs = TracingQuantumSession.get_instance()
        
        combined_args = tuple([abs_qs.abs_qc] + list(res_vals) + list(arg_vals))
    
        def true_fun(combined_args):
            return combined_args
        
        def false_fun(combined_args):
            return while_loop(cond_fun, body_fun, init_val = combined_args)
    
        combined_res = cond(first_iter_res[0], true_fun, false_fun, combined_args)
        abs_qs.abs_qc = combined_res[0]
        
        flat_trial_function_res = combined_res[1:len(res_vals)+1]
        
        replacement_dic = {}
        
        for i in range(len(flat_trial_function_res)):
            replacement_dic[res_vals[i]] = flat_trial_function_res[i]
        
        from qrisp import QuantumVariable
        for res_val in first_iter_res:
            if isinstance(res_val, QuantumVariable):
                res_val.reg = replacement_dic[res_val.reg]
        
        trial_function_res = jax.tree.unflatten(res_tree_def, flat_trial_function_res)
        
        if len(first_iter_res) == 2:
            return trial_function_res[1]
        else:
            return trial_function_res[1:]
    
    return return_function