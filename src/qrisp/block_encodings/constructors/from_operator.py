# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Builds a BlockEncoding from a QubitOperator or FermionicOperator via its Pauli-string LCU decomposition."""

import numpy as np

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.block_encodings.constructors.from_lcu import build_from_lcu
from qrisp.operators import FermionicOperator, QubitOperator
from qrisp.operators.qubit.qubit_operator import threshold


def build_from_operator(
    cls: type[BlockEncoding],
    O: QubitOperator | FermionicOperator,  # noqa: E741 -- public, keyword-callable API name
) -> BlockEncoding:
    r"""Constructs a BlockEncoding from an operator.

    Parameters
    ----------
    cls : type[BlockEncoding]
        The class on which this constructor is invoked.
    O : QubitOperator | FermionicOperator
        The operator to be block-encoded.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing $O$.

    Notes
    -----
    - Block encoding based on Pauli decomposition $O=\sum_i\alpha_ie^{i\phi_i}P_i$ where $\alpha_i$ are real nonnegative coefficients
      and $P_i$ are Pauli strings. The phases $e^{i\phi_i}$ are applied together with the respective Pauli strings.
    - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.
    - The block-encoding unitary is Hermitian if $O$ is Hermitian.

    Examples
    --------
    >>> from qrisp.block_encodings import BlockEncoding
    >>> from qrisp.operators import A, C, X, Y, Z
    >>> H = X(0)*X(1) + 0.2*Y(0)*Y(1)
    >>> B = BlockEncoding.from_operator(H)

    Non-Hermitian operators are block-encoded as well:

    >>> O = A(0)*C(1) + 0.5j*Z(0)
    >>> B = BlockEncoding.from_operator(O)

    """
    if isinstance(O, FermionicOperator):
        O = O.to_qubit_operator()

    O = O.to_pauli()
    # Pauli strings are Hermitian, hence O is Hermitian iff all coefficients are real
    is_hermitian = all(abs(np.imag(coeff)) < threshold for coeff in O.terms_dict.values())

    unitaries, coeffs = O.unitaries()
    return build_from_lcu(cls, coeffs, unitaries, is_hermitian=is_hermitian)
