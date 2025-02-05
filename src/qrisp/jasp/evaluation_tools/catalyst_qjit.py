"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
from qrisp.jasp.jasp_expression import make_jaspr

def qjit(function):
    """
    Decorator to leverage the jasp + Catalyst infrastructure to compile the given
    function to QIR and run it on the Catalyst QIR runtime.

    Parameters
    ----------
    function : callable
        A function performing Qrisp code.

    Returns
    -------
    callable
        A function executing the compiled code.
        
    Examples
    --------
    
    We write a simple function using the QuantumFloat quantum type and execute
    via ``qjit``:
        
    ::
        
        from qrisp import *
        from qrisp.jasp import qjit

        @qjit
        def test_fun(i):
            qv = QuantumFloat(i, -2)
            with invert():
                cx(qv[0], qv[qv.size-1])
                h(qv[0])
            meas_res = measure(qv)
            return meas_res + 3
            
    
    We execute the function a couple of times to demonstrate the randomness
    
    >>> test_fun(4)
    [array(5.25, dtype=float64)]
    >>> test_fun(5)
    [array(3., dtype=float64)]
    >>> test_fun(5)
    [array(7.25, dtype=float64)]

    """
    
    
    def jitted_function(*args):
        
        if not hasattr(function, "jaspr_dict"):
            function.jaspr_dict = {}
        
        args = list(args)
        
        signature = tuple([type(arg) for arg in args])
        if not signature in function.jaspr_dict:
            function.jaspr_dict[signature] = make_jaspr(function)(*args)
        
        return function.jaspr_dict[signature].qjit(*args, function_name = function.__name__)
    
    return jitted_function