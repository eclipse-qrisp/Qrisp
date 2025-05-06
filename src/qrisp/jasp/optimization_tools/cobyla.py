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

import jax
import jax.numpy as jnp
from jax.scipy.optimize import OptimizeResults

# https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
# Conditions: alpha <= 1; 1/6 <= gamma <= 1/2; 2*(alpha-gamma) > 1
def cobyla(fun, x0, 
           args, 
           maxiter=50,
           cons=[], rhobeg=1.0, rhoend=1e-6,seed=3):
    r"""
    
    Minimize a scalar function of one or more variables using the `Constrained Optimization By Linear Approximation (COBYLA) algorithm`_.

    

    Parameters
    ----------
        rhobeg : float
            Scaling parameter for gradient estimation.
        rhoend : float
            Scaling exponent for gradient estimation, defines endpoint of gradient estimation.
          

    Returns
    -------
    results
        An `OptimizeResults <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults>`_ object.

    """
    
    def arg_fun(x):
        return fun(x, *args) 
    
    n = len(x0)
    sim = jnp.zeros((n + 1, n))
    sim = sim.at[0].set(x0)
    sim = sim.at[1:].set(x0 + jnp.eye(n) * rhobeg)
    
    # Initialize function values and constraint values
    f = jax.lax.map(arg_fun, sim)
    c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)

    def body_fun(state):
        #CHANGE --> add callback_list
        sim, f, c, rho, nfeval = state
        
        # Find the best and worst points
        best = jnp.argmin(f)
        worst = jnp.argmax(f)
        
        # Calculate the centroid of the simplex excluding the worst point
        # Calculate the centroid of the simplex excluding the worst point
        mask = jnp.arange(n + 1) != worst
        centroid = jnp.sum(sim * mask[:, None], axis=0) / n
        
        # Reflect the worst point
        xr = 2 * centroid - sim[worst]
        fr = arg_fun(xr)
        cr = jnp.array([con(xr) for con in cons])
        nfeval += 1
        
        # Expansion
        xe = 2 * xr - centroid
        fe = arg_fun(xe)
        ce = jnp.array([con(xe) for con in cons])
        nfeval += 1
        
        # Contraction
        xc = 0.5 * (centroid + sim[worst])
        fc = arg_fun(xc)
        cc = jnp.array([con(xc) for con in cons])
        nfeval += 1
        
        # Update simplex based on conditions
        cond_reflect = (fr < f[best]) & jnp.all(cr >= 0)
        cond_expand = (fe < fr) & cond_reflect
        cond_contract = (fc < f[worst]) & jnp.all(cc >= 0)
        
        sim = jnp.where(cond_expand, sim.at[worst].set(xe), 
                jnp.where(cond_reflect, sim.at[worst].set(xr), 
                    jnp.where(cond_contract, sim.at[worst].set(xc), 
                        0.5 * (sim + sim[best]))))
        
        # f = jax.vmap(func)(sim)
        f = jax.lax.map(arg_fun, sim)
        c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
        #CHANGE --> uncomment
        #callb = callb.at[nfeval].set(f[best])
        #callb[nfeval] = f[best]

        rho *= 0.5
        #CHANGE --> uncomment
        return sim, f, c, rho, nfeval #, callb
    
    def cond_fun(state):
        #CHANGE --> add callback_list
        _, _, _, rho, nfeval = state
        return (rho > rhoend) & (nfeval < maxiter)
    
    from qrisp.jasp import make_tracer
    state = (sim, f, c, rhobeg, make_tracer(n + 1)) # (sim, f, c, rhobeg, n + 1, callback_list)
    sim, f, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, state)
    best = jnp.argmin(f)
    x = sim[best]
    fx = arg_fun(x)
    #x = fori_loop(0, make_tracer(maxiter), body_fun, (x0))
    #fx = fun(x, *args)

    return OptimizeResults(x, True, 0, fx, None, None, 2*maxiter+1, maxiter, maxiter)