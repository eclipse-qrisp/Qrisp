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

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.block_encodings.constructors.from_lcu import build_from_lcu
from qrisp.operators import QubitOperator, FermionicOperator


def build_from_operator(
    cls: BlockEncoding, O: QubitOperator | FermionicOperator
) -> BlockEncoding:
    r"""
    Constructs a BlockEncoding from an operator.

    Parameters
    ----------
    O : QubitOperator | FermionicOperator
        The operator to be block-encoded.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing the Hermitian part $(O+O^{\dagger})/2$.

    Notes
    -----
    - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real positive coefficients
      and $P_i$ are Pauli strings (including the respective sign).
    - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

    Examples
    --------

    >>> from qrisp.block_encodings import BlockEncoding
    >>> from qrisp.operators import X, Y, Z
    >>> H = X(0)*X(1) + 0.2*Y(0)*Y(1)
    >>> B = BlockEncoding.from_operator(H)

    """
    if isinstance(O, FermionicOperator):
        O = O.to_qubit_operator()

    unitaries, coeffs = O.unitaries()
    return build_from_lcu(cls, coeffs, unitaries, is_hermitian=True)
