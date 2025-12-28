"""
********************************************************************************
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
********************************************************************************
"""

import jax.numpy as jnp
import numpy as np
from qrisp import *
from qrisp.gqsp import state_preparation

# Gaussian 
def f(x):
    return jnp.exp(-2 * x ** 2)

# Converts the function to be executed within a repeat-until-success (RUS) procedure.
@RUS(static_argnums=1)
def preprare_gaussian(n, k):

    # Evaluate f at equidistant sample points
    delta = 2.0 ** (-2 * k)
    x_val = jnp.arange(-1, 1 ,delta)
    y_val = f(x_val)
    y_val = y_val / jnp.linalg.norm(y_val)

    qv = QuantumFloat(n)
    qbl = state_preparation(qv, y_val, k=k)
    success_bool = measure(qbl) == 0
    return success_bool, qv

# The terminal_sampling decorator performs a hybrid simulation,
# and afterwards samples from the resulting quantum state.
@terminal_sampling
def main():
    qv =  preprare_gaussian(10, 4)
    return qv   

def test_qsp_gaussian():

    # Convert the resulting measurement probabilities to amplitudes by appling the square root.
    res_dict = main()
    for k,v in res_dict.items():
        res_dict[k] = v ** 0.5 
    y_val_sim = np.array([res_dict.get(key, 0) for key in sorted(res_dict.keys())])
    y_val_sim = y_val_sim / np.linalg.norm(y_val_sim)

    # Compare to classical values
    x_val = np.linspace(-1, 1, len(y_val_sim))
    y_val = f(x_val)
    y_val = y_val / np.linalg.norm(y_val)

    # Evaluate trace distance
    F = np.abs(np.conj(y_val) @ y_val_sim) ** 2
    D = np.sqrt(1 - F)
    assert D < 1e-2