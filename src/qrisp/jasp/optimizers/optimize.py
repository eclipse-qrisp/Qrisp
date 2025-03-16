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

from qrisp.jasp.optimizers.spsa import spsa

def minimize(fun, x0, args=(), method='SPSA', options=None):
    """

    Minimization of scalar functions of one ore more variables via gradient-free solvers.

    The API for this functions matches SciPy with some minor deviations.
    
    Parameters
    ----------
        fun : callable
            The objective function to be minimized, ``fun(x, *args) -> float``, where ``x`` is a
            1-D array with shape ``(n,)``and ``args``is a tuple of parameters needed to specify the function.
        x0 : jax.Array
            Initial guess. Array of real elements of soze ``(n,)``, where ``n``is the number of independent variables.
        args : tuple
            Extra arguments passed to the objective function.
        method : str
            The solver type. Currently only ``SPSA`` is supported.
        options : dict
            A dictionary of solver options. All methods accept the following generic options:
            * maxiter : int 
                Maximum number of iterations to perform. Depending on the method each iteration may use several function evaluations.

    ``minimize`` supports ``jax.jit``compilation.
    
    """

    if method=='SPSA':
        return spsa(fun, x0, args, **options)
    else:
        raise Exception(f'Optimization method {method} is not available.')