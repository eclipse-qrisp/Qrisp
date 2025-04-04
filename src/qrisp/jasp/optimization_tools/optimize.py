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

from qrisp.jasp.optimization_tools.spsa import spsa

def minimize(fun, x0, args=(), method='SPSA', options={}):
    r"""

    Minimization of scalar functions of one ore more variables via gradient-free solvers.

    The API for this function matches SciPy with some minor deviations.

    * Various optional arguments in the SciPy interface have not yet been implemented.
    
    Parameters
    ----------
    fun : callable
        The objective function to be minimized, ``fun(x, *args) -> float``, where ``x`` is a
        1-D array with shape ``(n,)`` and ``args`` is a tuple of parameters needed to specify the function.
    x0 : jax.Array
        Initial guess. Array of real elements of size ``(n,)``, where ``n`` is the number of independent variables.
    args : tuple
        Extra arguments passed to the objective function.
    method : str, optional
        The solver type. Currently only ``SPSA`` is supported.
    options : dict, optional
        A dictionary of solver options. All methods accept the following generic options:

        * maxiter : int 
            Maximum number of iterations to perform. Depending on the method each iteration may use several function evaluations.

    Returns
    ------- 
    results
        An `OptimizeResults <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults>`_ object.


    Examples
    --------

    We prepare the state 

    .. math::

        \ket{\psi_{\theta}} = \cos(\theta)\ket{0} + \sin(\theta)\ket{1}

    ::
    
        from qrisp import QuantumFloat, ry
        from qrisp.jasp import expectation_value, minimize, jaspify
        import jax.numpy as jnp

        def state_prep(theta):
            qv = QuantumFloat(1)
            ry(theta[0], qv)
            return qv

    Next, we define the objective function calculating the expectation value from the prepared state  

    ::
        
        def objective(theta, state_prep):
            return expectation_value(state_prep, shots=100)(theta)

    Finally, we use ``optimize`` to find the optimal choice of the parameter $\theta_0$ that minimizes the objective function
            
    ::    

        @jaspify(terminal_sampling=True)
        def main():

            x0 = jnp.array([1.0])

            return minimize(objective,x0,args=(state_prep,))

        results = main()
        print(results.x)
        print(results.fun)
    
    """

    if method=='SPSA':
        return spsa(fun, x0, args, **options)
    else:
        raise Exception(f'Optimization method {method} is not available.')