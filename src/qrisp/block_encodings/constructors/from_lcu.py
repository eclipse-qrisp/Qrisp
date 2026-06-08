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

import numpy as np
import numpy.typing as npt
from typing import Any, Callable

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.environments import conjugate
from qrisp.jasp import qache
from qrisp.qtypes import QuantumFloat


def build_from_lcu(
    cls: BlockEncoding,
    coeffs: npt.NDArray[np.number],
    unitaries: list[Callable[..., Any]],
    num_ops: int = 1,
    is_hermitian: bool = False,
) -> BlockEncoding:
    r"""
    Constructs a BlockEncoding using the Linear Combination of Unitaries (LCU) protocol.

    For an LCU block encoding, consider a linear combination of unitaries:

    .. math::

        O = \sum_{i=0}^{M-1} \alpha_i U_i

    where $\alpha_i$ are real non-negative coefficients such that $\sum_i \alpha_i = \alpha$,
    and $U_i$ are unitaries acting on the same operand quantum variables.

    The block encoding unitary is constructed via the LCU protocol:

    .. math::

        U = \text{PREP} \cdot \text{SEL} \cdot \text{PREP}^{\dagger}

    where:

    * **SEL** (Select, in Qrisp: :ref:`q_switch <qswitch>`) applies each unitary $U_i$ conditioned on the auxiliary variable state $\ket{i}_a$:

    .. math::

        \text{SEL} = \sum_{i=0}^{M-1} \ket{i}\bra{i} \otimes U_i

    * **PREP** (Prepare) prepares the state representing the coefficients:

    .. math::

        \text{PREP} \ket{0}_a = \sum_{i=0}^{M-1} \sqrt{\frac{\alpha_i}{\alpha}} \ket{i}_a

    Parameters
    ----------
    coeffs : ArrayLike
        1-D array of non-negative coefficients $\alpha_i$.
    unitaries : list[Callable]
        List of functions, where each ``U(*operands)`` applies a unitary
        transformation in-place to the provided quantum variables.
        All functions must accept the same signature and operate on the
        same set of operands.
    num_ops : int
        The number of operand quantum variables. The default is 1.
    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian. The default is False.
        Set to True, if all provided unitaries are Hermitian.

    Returns
    -------
    BlockEncoding
        A BlockEncoding using LCU.

    Raises
    ------
    ValueError
        If any entry in ``coeffs`` is negative, as the LCU protocol only supports positive coefficients.

    Notes
    -----
    - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

    Examples
    --------

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        def f0(x): x-=1
        def f1(x): x+=1
        BE = BlockEncoding.from_lcu(np.array([1., 1.]), [f0, f1])

        @terminal_sampling
        def main():
            return BE.apply_rus(lambda : QuantumFloat(2))()

        main()
        # {1.0: 0.5, 3.0: 0.5}

    """
    from qrisp.alg_primitives.state_preparation import prepare
    from qrisp.jasp import q_switch

    m = len(coeffs)
    n = (m - 1).bit_length()  # Number of qubits for index variable
    # Ensure coeffs has size 2 ** n by zero padding
    coeffs = np.pad(coeffs, (0, (1 << n) - m))
    alpha = np.sum(coeffs)

    if np.any(coeffs < 0):
        raise ValueError(
            f"Negative coefficients detected: {coeffs}. Only positive values are supported."
        )

    if m == 1:
        return cls(
            alpha, [], unitaries[0], num_ops=num_ops, is_hermitian=is_hermitian
        )

    @qache
    def unitary(*args):
        # LCU = PREP SEL PREP_dg
        with conjugate(prepare)(args[0], np.sqrt(coeffs / alpha)):
            q_switch(args[0], unitaries, *args[1:])

    return cls(
        alpha,
        [QuantumFloat(n).template()],
        unitary,
        num_ops=num_ops,
        is_hermitian=is_hermitian,
    )
