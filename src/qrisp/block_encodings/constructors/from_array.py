"""
********************************************************************************
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

import numpy.typing as npt
from scipy.sparse import csr_array, csr_matrix
from typing import Any, Union

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.block_encodings.constructors.from_operator import build_from_operator
from qrisp.operators import QubitOperator


MatrixType = Union[npt.NDArray[Any], csr_array, csr_matrix]


def build_from_array(cls: BlockEncoding, A: MatrixType) -> BlockEncoding:
    r"""
    Constructs a BlockEncoding from a 2-D array.

    Parameters
    ----------
    A : ndarray | csr_array | csr_matrix
        2-D array of shape ``(N,N,)`` for a power of two ``N``.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing the Hermitian part $(A+A^{\dagger})/2$.

    Raises
    ------
    ValueError
        If ``A`` is not a 2-D square matrix.
    ValueError
        If the dimension of ``A`` is not a power of two.

    Notes
    -----
    - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real positive coefficients
      and $P_i$ are Pauli strings (including the respective sign).
    - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

    Examples
    --------

    >>> import numpy as np
    >>> from qrisp.block_encodings import BlockEncoding
    >>> A = np.array([[0,1,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0]])
    >>> B = BlockEncoding.from_array(A)

    """

    shape = A.shape
    # 1. Check if the array is 2D and square
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Array must be square (N, N), but got {shape}.")

    # 2. Check if N is a power of two
    N = shape[0]
    if not (N > 0 and (N & (N - 1)) == 0):
        raise ValueError(f"Size N={N} must be a power of two.")

    O = QubitOperator.from_matrix(A, reverse_endianness=True)
    return build_from_operator(cls, O)
