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

import jax.numpy as jnp
from jax import jit
from jax.core import eval_jaxpr

from qrisp.jasp import make_jaspr

from qrisp.jasp.interpreter_tools.interpreters.cl_func_interpreter import jaspr_to_cl_func_jaxpr

def boolean_simulation(*func, bit_array_padding = 2**20):
    """
    Decorator to simulate Jasp functions containing only classical logic (like X, CX, CCX etc.).
    This decorator transforms the function into a Jax-Expression without any
    quantum primitives and leverages the Jax compilation pipeline to compile
    a highly efficient simulation.
    
    .. note::
        
        The ``boolean_simulation`` decorator will check if deleted 
        :ref:`QuantumVariables <QuantumVariables>`
        have been properly uncomputed and submit a warning otherwise. 
        It therefore provides a valuable tool for verifying the correctness 
        of your algorithms at scale.
    
    
    Parameters
    ----------
    func : callable
        A Python function performing Jasp logic.
    bit_array_padding : int, optional
        An integer specifying the size of the classical array containing the
        (classical) bits which are simulated. Since Jax doesn't allow dynamically
        sized arrays but Jasp supports dynamically sized QuantumVariables, the
        array has to be "padded". The padding therefore indicates an upper boundary
        for how many qubits are required to execute ``func``. A large padding 
        slows down the simulation but prevents overflow errors. The simulation 
        is performed without any memory management, therefore even qubits that
        are deallocated count into the padding. The default is ``2**20``. The
        minimum value is 64.

    Returns
    -------
    simulator_function
        A function performing the simulation for the given input parameters.

    Examples
    --------
    
    We create a simple script that demonstrates the functionality:
        
    ::
        
        from qrisp import QuantumFloat, measure
        from qrisp.jasp import boolean_simulation, jrange
        
        @boolean_simulation
        def main(i, j):
            
            a = QuantumFloat(10)
            
            b = QuantumFloat(10)
            
            a[:] = i
            b[:] = j
            
            c = QuantumFloat(30)
            
            for i in jrange(150): 
                c += a*b
            
            return measure(c)
    
    This script evaluates the multiplication of the two inputs 150 times and adds
    them into the same QuantumFloat. The respected result is therefore ``i*j*150``.
    
    >>> main(1,2)
    Array(300., dtype=float64)
    >>> main(3,4)
    Array(1800., dtype=float64)
    
    Next we demonstrate the behavior under a faulty uncomputation:
        
    ::
        
        @boolean_simulation
        def main(i):
            
            a = QuantumFloat(10)
            a[:] = i
            a.delete()
            return
        
    
    >>> main(0)
    >>> main(1)
    WARNING: Faulty uncomputation found during simulation.
    >>> main(3)
    WARNING: Faulty uncomputation found during simulation.
    WARNING: Faulty uncomputation found during simulation.
    
    For the first case, the deletion is valid, because ``a`` is initialized in the
    $\ket{0}$ state. For the second case, the first qubit is in the $\ket{1}$ 
    state, so the deletion is not valid. The third case has both the first and
    the second qubit in the $\ket{1}$ state (because 3 = ``11`` in binary) so
    there are two warnings.
    
    **Padding**
    
    We demonstrate the effects of the padding feature. For this we recreate the
    above script but with different padding selections.
    
    ::

        @boolean_simulation(bit_array_padding = 64)   
        def main(i, j):
            
            a = QuantumFloat(10)
            
            b = QuantumFloat(10)
            
            a[:] = i
            b[:] = j
            
            c = QuantumFloat(30)
            
            for i in jrange(150): 
                c += a*b
            
            return measure(c)
    
    >>> main(1,2)
    Array(8.92323439e+08, dtype=float64)
    
    A faulty result because the script needs more than 64 qubits.
    
    Increasing the padding ensures that enough qubits are available at the cost
    of simulation speed.
    

    """
    
    if len(func) == 0:
        return lambda x : boolean_simulation(x, bit_array_padding = bit_array_padding)
    else:
        func = func[0]
    
    if bit_array_padding < 64:
        raise Exception("Tried to initialize boolean_simulation with less than 64 bits")
    
    @jit    
    def return_function(*args):
        
        jaspr = make_jaspr(func)(*args)
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(jaspr.flatten_environments(), bit_array_padding)
        
        aval = cl_func_jaxpr.invars[0].aval
        res = eval_jaxpr(cl_func_jaxpr, 
                         [], 
                         jnp.zeros(aval.shape, dtype = aval.dtype), 
                         jnp.array(0, dtype = jnp.int64), *args)
        
        
        if len(res) == 3:
            return res[2]
        elif len(res) == 2:
            return None
        else:
            return res[2:]
    
    return return_function
