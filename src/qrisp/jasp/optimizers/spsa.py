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

import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop

# https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
# Conditions: alpha <= 1; 1/6 <= gamma <= 1/2; 2*(alpha-gamma) > 1
def spsa(fun, x0, args, maxiter=50, a=2.0, c=0.1, alpha=0.702, gamma=0.201, seed=3):
    r"""
    
    Minimize a scalar function of one or more variables using the Simultaneous Perturbation Stochastic Approximation algorithm.

    This algorithm aims at finding the optimal control $x^*$ minimizing a given loss fuction $f$:

    .. math::

        x^* = \argmin_{x} f(x)

    This is done by an iterative process starting from an initial guess $x0$:

    .. math::

        x_{k+1} = x_k - a_kg_k(x_k)

    where $a_k=\frac{a}{n^{\alpha}}$ for scaling parameters $a, \alpha>0$.

    For each step $x_k$ the gradient is approximated by   

    .. math::

        (g_k(x_k))_i = \frac{f(x_k+c_k\Delta_k)-f(x_k-c_k\Delta_k)-}{2c_k(\Delta_k)_i}

    where $c_k=\frac{c}{n^{\gamma}}$ for scaling parameters $c, \gamma>0$, and $\Delta_k$ is a random perturbation vector.

    Options
    -------
        a : float
            Scaling parameter for update rule.
        alpha : float
            Scaling exponent for update rule.
        c : float 
            Scaling parameter for gradient estimation.
        gamma : float
            Scaling exponent for gradient estimation.   

    Returns
    -------
    x : jax.Array
        The solution of the optimization.
    fx : jax.Array
        The value of the objective function at x.

    """
    
    rng = random.PRNGKey(seed)

    def body_fun(k, state):

        x, rng = state

        # Generate random perturbation delta with components +/-1
        rng, rng_input = random.split(rng)
        delta = random.choice(rng, jnp.array([1, -1]), shape=(*x.shape,))
    
        ak = a / (k + 1) ** alpha
        ck = c / (k + 1) ** gamma

        # Evaluate loss function at perturbed points
        x_plus = x + ck * delta
        x_minus = x - ck * delta

        loss_plus = fun(x_plus, *args)
        loss_minus = fun(x_minus, *args)

        # Approximate gradient
        gk = (loss_plus - loss_minus) / (2.0 * ck * delta)

        # Update parameters
        x = x - ak * gk

        return x, rng
    
    from qrisp.jasp import make_tracer
    x, rng = fori_loop(0, make_tracer(maxiter), body_fun, (x0, rng))
    fx = fun(x, *args)

    return x, fx