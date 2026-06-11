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

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.environments import conjugate, invert
from qrisp.jasp import qache, q_switch
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

    where $\alpha_i$ are complex coefficients, $\alpha = \sum_i |\alpha_i|$,
    and $U_i$ are unitaries acting on the same operand quantum variables.

    The block encoding unitary is constructed via the LCU protocol.
    If all coefficients are real and non-negative, the block encoding unitary can be expressed as

    .. math::

        U = \text{PREP} \cdot \text{SEL} \cdot \text{PREP}^{\dagger}

    where:

    * $SEL$ (Select, in Qrisp: :ref:`q_switch <qswitch>`) applies each unitary $U_i$ conditioned on the auxiliary variable state $\ket{i}_a$:

    .. math::

        \text{SEL} = \sum_{i=0}^{M-1} \ket{i}\bra{i} \otimes U_i

    * $PREP$ (Prepare) prepares the state representing the coefficients:

    .. math::

        \text{PREP} \ket{0}_a = \sum_{i=0}^{M-1} \sqrt{\frac{\alpha_i}{\alpha}} \ket{i}_a

    If the coefficients are complex or negative, a state preparation pair $(PREP_R, PREP_L)$ is used:

    .. math::

        U = \text{PREP}_R \cdot \text{SEL} \cdot \text{PREP}_L^{\dagger}

    * $PREP_R$ and $PREP_L$ prepare the states representing the coefficients and their complex conjugates, respectively:

    .. math::

        \text{PREP}_R \ket{0}_a = \sum_{i=0}^{M-1} \sqrt{\frac{\alpha_i}{\alpha}} \ket{i}_a, \qquad
        \text{PREP}_L \ket{0}_a = \sum_{i=0}^{M-1} \sqrt{\frac{\alpha_i}{\alpha}}^* \ket{i}_a

    Parameters
    ----------
    coeffs : np.ndarray
        1-D array containing the complex coefficients $\alpha_i$.
    unitaries : list[Callable]
        List of functions, where each ``unitary(*operands)`` applies a unitary
        transformation in-place to the provided quantum variables.
        All functions must accept the same signature and operate on the
        same set of operands.
    num_ops : int, optional
        The number of operand quantum variables. The defaults to 1.
    is_hermitian : bool, optional
        Indicates whether the block-encoding unitary is Hermitian. The defaults to `False`.
        Set to `True`, if all provided unitaries are Hermitian.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing the operator defined as the linear combination of unitaries.

    Notes
    -----
    - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i |\alpha_i|$.

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

    m = len(coeffs)
    n = (m - 1).bit_length()  # Number of qubits for index variable
    # Ensure coeffs has size 2 ** n by zero padding
    coeffs = np.pad(coeffs, (0, (1 << n) - m))
    alpha = np.sum(np.abs(coeffs))

    # Block encoding of a single unitary (up to normalization)
    if m == 1:
        return cls(
            coeffs[0], [], unitaries[0], num_ops=num_ops, is_hermitian=is_hermitian
        )
    
    # Block encoding of a linear combination of unitaries via the LCU protocol
    # If all coefficients are real and non-negative: LCU = PREP SEL PREP_dg
    if _is_real_non_negative_array(coeffs):

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
    
     # If coefficients are complex or negative, we use a state preparation pair (PREP_R, PREP_L): LCU = PREP_R SEL PREP_L_dg
    complex_coeffs = coeffs.astype(complex)
    complex_coeffs_r = np.sqrt(complex_coeffs / alpha)
    complex_coeffs_l = np.sqrt(complex_coeffs / alpha).conjugate()

    @qache
    def unitary(*args):
        # LCU = PREP_R SEL PREP_L_dg
        prepare(args[0], complex_coeffs_r)
        q_switch(args[0], unitaries, *args[1:])
        with invert():
            prepare(args[0], complex_coeffs_l)

    return cls(
        alpha,
        [QuantumFloat(n).template()],
        unitary,
        num_ops=num_ops,
        is_hermitian=is_hermitian,
    )


def _is_real_non_negative_array(arr: npt.NDArray[np.number], tol: float=1e-12):
    """Checks if all entries in an array are non-negative and have negligible imaginary parts."""
    # 1. Check if the array is a complex type
    if np.issubdtype(arr.dtype, np.complexfloating):
        abs_imag_part = np.abs(arr.imag)
        real_part = arr.real
        
        # Check if imaginary parts are within tolerance AND real parts are non-negative
        return (abs_imag_part < tol).all() and (real_part >= 0).all()

    else:
        return np.all(arr >= 0)

