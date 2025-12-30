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
import pytest
from qrisp import *
from qrisp.gqsp import fourier_series_loader

# Gaussian 
def f(x, alpha):
    return jnp.exp(-alpha * x ** 2)

# Converts the function to be executed within a repeat-until-success (RUS) procedure.
@RUS(static_argnames=["k"])
def prepare_gaussian(n, alpha, k):
    # Use 32 sampling points to evaluate f
    N_samples = 32
    x_val = jnp.arange(-1.0, 1.0, 2.0 / N_samples)
    y_val = f(x_val, alpha)
    y_val = y_val / jnp.linalg.norm(y_val)

    qv = QuantumFloat(n)
    qbl = fourier_series_loader(qv, y_val, k=k)
    success_bool = measure(qbl) == 0
    return success_bool, qv

# The terminal_sampling decorator performs a hybrid simulation,
# and afterwards samples from the resulting quantum state.
@terminal_sampling
def main(n, alpha):
    qv =  prepare_gaussian(n, alpha, 4)
    return qv   


@pytest.mark.parametrize("n, alpha", [
    (6, 4),
    (6, 10),
])
def test_qsp_gaussian(n, alpha):

    # Run the simulation for n-qubit state
    res_dict = main(n, alpha)

    # Convert the resulting measurement probabilities to amplitudes by appling the square root.
    for k,v in res_dict.items():
        res_dict[k] = v ** 0.5 
    y_val_sim = np.array([res_dict.get(key, 0) for key in range(2 ** n)])
    y_val_sim = y_val_sim / np.linalg.norm(y_val_sim)

    # Compare to target values
    x_val = np.arange(-1, 1, 2 ** (-n + 1))
    y_val = f(x_val, alpha)
    y_val = y_val / np.linalg.norm(y_val)

    # Evaluate trace distance
    F = np.abs(np.conj(y_val) @ y_val_sim) ** 2
    D = np.sqrt(1 - F)
    assert D < 1e-2