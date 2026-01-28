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

import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# numpy.polynomial.chebyshev.poly2cheb
# To be deprecated when available in jax.numpy
@jax.jit
def poly2cheb(poly: "ArrayLike") -> "ArrayLike":
    """
    Convert a polynomial to a Chebyshev series.
    JAX version of `numpy.polynomial.chebyshev.poly2cheb <https://numpy.org/doc/2.3/reference/generated/numpy.polynomial.chebyshev.poly2cheb.html>`_.

    Convert an array representing the coefficients of a polynomial (relative to the “standard” basis) ordered from lowest degree to highest, 
    to an array of the coefficients of the equivalent Chebyshev series, ordered from lowest to highest degree.
    
    Parameters
    ----------
    poly : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    cheb : ArrayLike
        1-D array containing the coefficients of the equivalent Chebyshev series ordered from lowest order term to highest.

    Examples
    --------

    >>> import jax.numpy as jnp
    >>> from qrisp.gqsp import poly2cheb 
    >>> poly = jnp.array([-2., -8.,  4., 12.])
    >>> cheb = poly2cheb(poly)
    >>> cheb
    [0., 1., 2., 3.]

    """
    N = len(poly)
    
    # Build the transformation matrix C such that P_power = C @ P_cheb
    # This matrix contains the power-basis coefficients of T_n(x)
    C = jnp.zeros((N, N), dtype=poly.dtype)
    C = C.at[0, 0].set(1) # T_0(x) = 1
    if N > 1:
        C = C.at[1, 1].set(1) # T_1(x) = x
        # Use the recurrence T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        for n in range(2, N):
            # 2 * x * T_{n-1}(x): shift coefficients right by 1 and multiply by 2
            prev = C[n-1]
            prev_shifted = jnp.roll(prev, 1) * 2
            # Handle the roll boundary condition manually to match 2 * x * T_{n-1}(x)
            prev_shifted = prev_shifted.at[0].set(0) 
            C = C.at[n, :].set(prev_shifted - C[n-2, :])
            
    # Solve the linear system for the Chebyshev coefficients
    # The matrix C is triangular/well-behaved, making the solve stable
    cheb = jnp.linalg.solve(C.T, poly)
    
    return cheb


# numpy.polynomial.chebyshev.cheb2poly
# To be deprecated when available in jax.numpy
@jax.jit
def cheb2poly(cheb: "ArrayLike") -> "ArrayLike":
    """
    Convert a Chebyshev series to a polynomial.
    JAX version of `numpy.polynomial.chebyshev.cheb2poly <https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.cheb2poly.html>`_.

    Convert an array representing the coefficients of a Chebyshev series, ordered from lowest degree to highest, 
    to an array of the coefficients of the equivalent polynomial (relative to the “standard” basis) ordered from lowest to highest degree.
    
    Parameters
    ----------
    cheb : ArrayLike
        1-D array containing the Chebyshev series coefficients, ordered from lowest order term to highest.

    Returns
    -------
    poly : ArrayLike
        1-D array containing the coefficients of the equivalent polynomial (relative to the “standard” basis), ordered from lowest order term to highest.

    Examples
    --------

    >>> import jax.numpy as jnp
    >>> from qrisp.gqsp import cheb2poly 
    >>> poly = jnp.array([0., 1., 2., 3.])
    >>> poly = cheb2poly(cheb)
    >>> poly
    [-2., -8.,  4., 12.]

    """
    N = len(cheb)
    
    # Build the transformation matrix C such that P_power = C @ P_cheb
    # This matrix contains the power-basis coefficients of T_n(x)
    C = jnp.zeros((N, N), dtype=cheb.dtype)
    C = C.at[0, 0].set(1) # T_0(x) = 1
    if N > 1:
        C = C.at[1, 1].set(1) # T_1(x) = x
        # Use the recurrence T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        for n in range(2, N):
            # 2 * x * T_{n-1}(x): shift coefficients right by 1 and multiply by 2
            prev = C[n-1]
            prev_shifted = jnp.roll(prev, 1) * 2
            # Handle the roll boundary condition manually to match 2 * x * T_{n-1}(x)
            prev_shifted = prev_shifted.at[0].set(0) 
            C = C.at[n, :].set(prev_shifted - C[n-2, :])

    # Resulting power coefficients
    poly = jnp.dot(cheb, C) # or jnp.dot(C.T, coeffs) if coeffs was a column vector

    return poly