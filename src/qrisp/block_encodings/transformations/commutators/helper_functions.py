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

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import jax.numpy as jnp
import math
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def _chebyshev_commutator_coeffs(d):
    r"""
    Calculates the coefficient matrix for the Chebyshev expansion of the nested commutator $\text{ad}_A^d(B)$.

    Parameters
    ----------
    d : int
        The order of the nested commutator, which determines the size of the coefficient matrix.

    Returns
    -------
    C : ndarray, shape (d+1, d+1)
        The coefficient matrix for the Chebyshev expansion of $\text{ad}_A^d(B)$,
        where $C_{m,n}$ is the coefficient for the term $T_m(A) B T_n(A)$ in the expansion.

    """
    from numpy.polynomial.chebyshev import poly2cheb

    # Initialize the coefficient matrix C_{m,n} with zeros
    C = np.zeros((d + 1, d + 1))

    for k in range(d + 1):
        # 1. Calculate the binomial coefficient and alternating sign
        term_weight = ((-1) ** k) * math.comb(d, k)

        # 2. Create standard polynomial arrays for A^{d-k} and A^k
        # In NumPy, index `i` corresponds to the coefficient of x^i
        left_poly = np.zeros(d - k + 1)
        left_poly[d - k] = 1.0

        right_poly = np.zeros(k + 1)
        right_poly[k] = 1.0

        # 3. Convert monomials to Chebyshev basis coefficients
        left_cheb = poly2cheb(left_poly)
        right_cheb = poly2cheb(right_poly)

        # 4. Pad the Chebyshev arrays to length d+1 to align matrix dimensions
        left_cheb_padded = np.pad(left_cheb, (0, d + 1 - len(left_cheb)))
        right_cheb_padded = np.pad(right_cheb, (0, d + 1 - len(right_cheb)))

        # 5. Compute the outer product to get the cross-terms T_m(A) B T_n(A)
        # and add it to the total coefficient matrix
        C += term_weight * np.outer(left_cheb_padded, right_cheb_padded)

    return C


def _chebyshev_sum_commutator_coeffs(coeffs):
    """
    Constructs the coefficient matrix for the weighted sum of nested commutators given the coefficients for each order of the commutator.

    Parameters
    ----------
    coeffs : ArrayLike, shape (d,)
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators, where $d$ is the length of the coeffs array.

    Returns
    -------
    C : ndarray, shape (d+1, d+1)
        The coefficient matrix for the weighted sum of nested commutators, where $C_{m,n}$ is the coefficient for the term $T_m(A) B T_n(A)$ in the expansion.

    """
    d = len(coeffs)
    C = np.zeros((d + 1, d + 1), dtype=np.complex128)
    for k in range(1, d + 1):
        Ck_matrix = _chebyshev_commutator_coeffs(k)
        rows, cols = Ck_matrix.shape
        C[:rows, :cols] += coeffs[k - 1] * Ck_matrix
    return C
