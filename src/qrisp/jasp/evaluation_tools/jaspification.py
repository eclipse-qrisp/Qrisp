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

from jax.tree_util import tree_flatten, tree_unflatten

from qrisp.jasp.interpreter_tools import extract_invalues, insert_outvalues, eval_jaxpr
from qrisp.simulator import BufferedQuantumState

def jaspify(func = None, terminal_sampling = False):
    """
    This simulator is the established Qrisp simulator linked to the Jasp infrastructure.
    Among a variety of simulation tricks, the simulator can leverage state sparsity,
    allowing simulations with up to hundreds of qubits!
    
    To be called as a decorator of a Jasp-traceable function.
    
    .. note::
        
        If you are developing a hybrid algorithm like QAOA or VQE that relies
        heavily on sampling, please activate the ``terminal_sampling`` feature.

    Parameters
    ----------
    func : callable
        The function to simulate.
    terminal_sampling : bool, optional
        Whether to leverage the terminal sampling strategy. Significantly fast 
        for all sampling tasks but can yield incorrect results in some situations.
        Check out :ref:`terminal_sampling` form more details. The default is False.

    Returns
    -------
    callable
        A function performing the simulation.
        
    Examples
    --------
    
    We simulate a function creating a simple GHZ state:
        
    ::
        
        from qrisp import *
        from qrisp.jasp import *
        
        @jaspify
        def main():
            
            qf = QuantumFloat(5)
            
            h(qf[0])
            
            for i in range(1, 5):
                cx(qf[0], qf[i])
                
            return measure(qf)

        print(main())            
        # Yields either 0 or 31
        
    To highlight the speed of the terminal sampling feature, we :ref:`sample` from a
    uniform superposition
    
    ::
            
        def state_prep():
            qf = QuantumFloat(5)
            h(qf)
            return qf
        
        @jaspify
        def without_terminal_sampling():
            sampling_func = sample(state_prep, shots = 10000)
            return sampling_func()
        
        @jaspify(terminal_sampling = True)
        def with_terminal_sampling():
            sampling_func = sample(state_prep, shots = 10000)
            return sampling_func()
        
        
    Benchmark the time difference:
        
    ::
        
        import time
        
        t0 = time.time()
        res = without_terminal_sampling()
        print(time.time() - t0)
        # Yields
        # 43.78982925
        
        t0 = time.time()
        res = with_terminal_sampling()
        print(time.time() - t0)
        # Yields
        # 0.550775527


    """
    
    if isinstance(func, bool):
        terminal_sampling = func
        func = None
    
    if func is None:
        return lambda x : jaspify(x, terminal_sampling = terminal_sampling)
    
    from qrisp.jasp import make_jaspr
    
    treedef_container = []
    def tracing_function(*args):
        res = func(*args)
        flattened_values, tree_def = tree_flatten(res)
        treedef_container.append(tree_def)
        return flattened_values
    
    def return_function(*args):
        jaspr = make_jaspr(tracing_function, garbage_collection = "manual")(*args)
        jaspr_res = simulate_jaspr(jaspr, *args, terminal_sampling = terminal_sampling)
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(treedef_container[0], jaspr_res)
        return jaspr_res
    return return_function


def simulate_jaspr(jaspr, *args, terminal_sampling = True):
    
    if len(jaspr.outvars) == 1:
        return None
    
    args = [BufferedQuantumState()] + list(tree_flatten(args)[0])
            
    flattened_jaspr = jaspr
    
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "pjit":
            
            if terminal_sampling:
                
                function_name = eqn.params["name"]
                
                translation_dic = {"expectation_value_eval_function" : "ev",
                                   "sampling_eval_function" : "array",
                                   "dict_sampling_eval_function" : "dict"}
                
                from qrisp.jasp.interpreter_tools import terminal_sampling_evaluator
                
                if function_name in translation_dic:
                    terminal_sampling_evaluator(translation_dic[function_name])(eqn, context_dic, eqn_evaluator = eqn_evaluator)
                    return
            
                
            invalues = extract_invalues(eqn, context_dic)
            outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
            if not isinstance(outvalues, (list, tuple)):
                outvalues = [outvalues]
            insert_outvalues(eqn, context_dic, outvalues)
        elif eqn.primitive.name == "jasp.quantum_kernel":
            insert_outvalues(eqn, context_dic, BufferedQuantumState())
        else:
            return True
    
    res = eval_jaxpr(flattened_jaspr, eqn_evaluator = eqn_evaluator)(*(args + jaspr.consts))
    
    if len(jaspr.outvars) == 2:
        return res[1]
    else:
        return res[1:]