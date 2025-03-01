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
def spsa(objective, init_point, max_iter, a=2.0, c=0.1, alpha=0.702, gamma=0.201, seed=3):
    
    rng = random.PRNGKey(seed)
    # Generate random perturbation delta with components +/-1
    delta = random.choice(rng, jnp.array([1, -1]), shape=(max_iter,*init_point.shape))
    
    objective_values = jnp.zeros(max_iter)

    def body_fun(k, state):

        objective_values, params, delta, a, c, alpha, gamma = state
    
        ak = a / (k + 1) ** alpha
        ck = c / (k + 1) ** gamma

        # Evaluate loss function at perturbed points
        params_plus = params + ck * delta[k]
        params_minus = params - ck * delta[k]

        loss_plus = objective(params_plus)
        loss_minus = objective(params_minus)

        # Approximate gradient
        gk = (loss_plus - loss_minus) / (2.0 * ck * delta[k])

        # Update parameters
        params_new = params - ak * gk

        loss = objective(params_new)
        callback = objective_values.at[k].set(loss)

        return callback, params_new, delta, a, c, alpha, gamma
    

    objective_values, optimal_params, delta, a, c, alpha, gamma = fori_loop(0, max_iter, body_fun, (objective_values, init_point, delta, a, c, alpha, gamma))

    optimal_value = objective(optimal_params)

    return optimal_params, optimal_value, objective_values