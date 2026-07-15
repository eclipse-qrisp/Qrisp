"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from typing import TYPE_CHECKING

import jax.numpy as jnp

from qrisp import (
    QuantumVariable,
    control,
    p,
    ry,
    x,
)
from qrisp.jasp import jrange

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def _unary_angles(mags: "ArrayLike") -> "ArrayLike":
    """Computes rotation angles for the real magnitudes."""
    rev_cumsum = jnp.cumsum(mags[::-1])[::-1]
    y_args = jnp.sqrt(rev_cumsum[1:])
    x_args = jnp.sqrt(mags[:-1])
    
    phi = jnp.arctan2(y_args, x_args)
    return 2 * phi


def _unary_phases(coeffs: "ArrayLike", conjugate: bool = False) -> "ArrayLike":
    """Computes the phase differences beta_j for each qubit.
    
    Parameters
    ----------
    coeffs : ArrayLike
        1-D array of arbitrary complex Chebyshev coefficients.
    conjugate : bool
        If True, computes phases to prepare the state with complex conjugate 
        coefficients sqrt(c_k^*).

    Returns
    -------
    ArrayLike
        1-D array of phase angles beta_j to be applied to each qubit.
    """
    # jnp.angle safely handles the complex plane and branch cuts
    theta = jnp.angle(coeffs) / 2.0
    
    if conjugate:
        theta = -theta
        
    # beta[0] = theta[0]
    # beta[j] = theta[j] - theta[j-1]
    beta = jnp.concatenate([jnp.array([theta[0]]), jnp.diff(theta)])
    
    return beta


def _unary_prep(case: QuantumVariable, coeffs: "ArrayLike", conjugate: bool = False) -> None:
    """Prepares the unary-encoded state of complex Chebyshev coefficients.

    Parameters
    ----------
    case : QuantumVariable
        Variable with j_0 qubits on which the unary state preparation will be performed.
    coeffs : ArrayLike
        1-D array of complex Chebyshev coefficients.
    conjugate : bool, optional
        If True, prepares the state corresponding to the complex conjugates 
        sqrt(c_k^*). Defaults to False.
    """
    # 1. Decouple magnitudes and phases
    mags = jnp.abs(coeffs)
    phi = _unary_angles(mags)
    beta = _unary_phases(coeffs, conjugate)

    # 2. Prepare the real-valued amplitudes using the magnitudes
    x(case[0])
    with control(case.size > 1):
        ry(phi[0], case[1])
    for i in jrange(1, case.size - 1):
        with control(case[i]):
            ry(phi[i], case[i + 1])

    # 3. Apply the phase differences unconditionally
    # These single-qubit phase gates commute perfectly with the target 
    # states of the unary encoding, avoiding any need for extra controls.
    for j in jrange(case.size):
        p(beta[j], case[j])
