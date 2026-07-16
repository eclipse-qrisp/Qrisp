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

from qrisp import custom_control
from qrisp.core import QuantumVariable
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.core.gate_application_functions import x, z, mcx, swap, h, cx, p
from qrisp.environments import conjugate, control, invert
from qrisp.jasp import (
    jrange,
    qache,
    q_switch,
)
from qrisp.qtypes import QuantumBool, QuantumFloat

from .helper_functions import _chebyshev_commutator_coeffs, _chebyshev_sum_commutator_coeffs
from .unary_prep import _unary_prep

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def create_unary_preps_walk(
    d: int,
    coeffs: npt.NDArray[Any] = None,
) -> None:
    r"""
    Return a state preparation pair that prepares a state that encodes the coefficients of the Chebyshev expansion of a weighted sum of nested commutators in a two-dimensional grid of unary-encoded indices
    by simulating a symmetric quantum walk on a 1D line from $-d$ to $d$.

    Each nested commutator $\text{ad}_A^k(B)$ can be expressed as a sum of terms of the form $C_{k,m,n}T_m(A)BT_n(A)$, where $T_m(A)$ are Chebyshev polynomials of the first kind evaluated at $A$.
    The state prepared by this function encodes the square root of the weighted sum of these coefficients in the amplitudes of a superposition over the indices $m$ and $n$,
    which can then be used to apply the corresponding operators in superposition.

    .. math::

        \sum_{k=1}^d\text{ad}_A^k(B) = \sum_{k=1}^d c_k\sum_{m,n}C_{k,m,n}T_m(A)BT_n(A)

    .. math::

        \text{PREP}\ket{0}_k\ket{0}_m\ket{0}_n \propto \sum_{k=1}^d\sqrt{c_k}\sum_{m,n}\sqrt{C_{k,m,n}}\ket{k}\ket{m}\ket{n}

    Parameters
    ----------
    d : int
        The depth of the commutator expansion, which determines the size of the walk.
    coeffs : ArrayLike, shape (d,), optional
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators.
        If None, defaults to a delta distribution on the highest order commutator.

    Returns
    -------
    prep_right : Callable
        A function that prepares the right side of the state encoding the coefficients of the Chebyshev expansion of the weighted sum of nested commutators.
        The function takes the following arguments:

        steps : QuantumVariable
            A unary-encoded ancilla QuantumVariable of size d, used to control the walk steps.
        coins1 : QuantumVariable
            An ancilla QuantumVariable of size d, used as the first set of coin variables to control the walk steps.
        coins2 : QuantumVariable
            An ancilla QuantumVariable of size d, used as the second set of coin variables to control the walk steps.
        m_line : QuantumVariable
            A one-hot-encoded QuantumVariable of size 2d+1, representing the position of the walk along the $m$-axis, which encodes the index of $T_m(A)$.
        n_line : QuantumVariable
            A one-hot-encoded QuantumVariable of size 2d+1, representing the position of the walk along the $n$-axis, which encodes the index of $T_n(A)$.
        qm : QuantumVariable
            A unary-encoded QuantumVariable of size d, representing the $m$ index.
        qn : QuantumVariable
            A unary-encoded QuantumVariable of size d, representing the $n$ index.
    prep_left : Callable
        A function that prepares the left side of the state encoding the coefficients of the Chebyshev expansion of the weighted sum of nested commutators.
        The function takes the same arguments as `prep_right`.
    prep_anc_templates : List[QuantumVariableTemplate]
        A list of QuantumVariable templates for the ancilla variables used in the state preparation.
        The templates correspond to the following ancilla variables:

        - steps : QuantumVariable of size d, for controlling the walk steps.
        - coins1 : QuantumVariable of size d, for the first set of coin variables.
        - coins2 : QuantumVariable of size d, for the second set of coin variables.
        - m_line : QuantumVariable of size 2d+1, for the position along the $m$-axis.
        - n_line : QuantumVariable of size 2d+1, for the position along the $n$-axis.

    Notes
    -----
    - **Complexity**: This state preparation requires $\mathcal O(d)$ qubits and $\mathcal O(d)$ depth.
    - This function simulates a symmetric quantum walk on two 1D lines from $-d$ to $d$, where the position of the walk encodes the indices of the Chebyshev polynomials in the expansion of the nested commutators.
    - The walk is based on the recurrence relations for Chebyshev polynomials: $xT_k(x) = \frac{1}{2}(T_{k+1}(x) - T_{k-1}(x))$.
    - The walk is implemented using two sets of coin variables (``coins1`` and ``coins2``) and two position variables (``m_line`` and ``n_line``) to achieve perfect parallelism in the shift operations,
      resulting in $\mathcal O(d)$ depth for $d$ walk steps.
    - The crucial minus signs for the commutator are implemented via Z gates on the first coin variable,
      which are applied following the Hadamard gates to create the necessary interference patterns in the walk.

    """

    # 1. Define the 1D line size.
    # For depth d, the furthest the particle can walk is d steps.
    # We need a line from -d to +d, giving a total size of 2d + 1.
    size = 2 * d + 1
    origin = d  # The center of the array represents m=0 and n=0

    # Define the parallel, O(1) depth shift operator
    def apply_symmetric_walk(coin: QuantumVariable, qv: QuantumVariable) -> None:
        # Layer 1: Swap all Even-Odd index pairs (0-1, 2-3, 4-5...)
        with control(coin):
            for i in jrange(size // 2):
                swap(qv[2 * i], qv[2 * i + 1])

        # Layer 2: Swap all Odd-Even index pairs (1-2, 3-4, 5-6...)
        with control(coin, ctrl_state=0):
            for i in jrange((size - 1) // 2):
                swap(qv[2 * i + 1], qv[2 * i + 2])

    def inner_walk(
        steps: QuantumVariable,
        coins1: QuantumVariable,
        coins2: QuantumVariable,
        m_line: QuantumVariable,
        n_line: QuantumVariable,
        step: int,
    ) -> None:

        # Initialize the particles directly at the origin (m=0, n=0)
        x(m_line[origin])
        x(n_line[origin])

        # Initialize the coins
        h(coins1)
        z(coins1)  # Applies the minus sign for the commutator
        h(coins2)

        for step in jrange(d):
            c1 = coins1[step]
            c2 = coins2[step]

            # Apply the walk step
            with control(steps[step]):
                with control(c1, ctrl_state=0):
                    apply_symmetric_walk(c2, m_line)

                with control(c1):
                    apply_symmetric_walk(c2, n_line)

    if coeffs is None:
        coeffs = np.zeros(d)
        coeffs[d - 1] = 1
    else:
        # Rescale coefficients
        coeffs = np.array(coeffs) * np.array([np.sum(np.abs(_chebyshev_commutator_coeffs(k))) for k in range(d)])
        coeffs = coeffs / np.sum(coeffs)

    @custom_control
    def prep_right(
        steps: QuantumVariable,
        coins1: QuantumVariable,
        coins2: QuantumVariable,
        m_line: QuantumVariable,
        n_line: QuantumVariable,
        qm: QuantumVariable,
        qn: QuantumVariable,
        ctrl=None,
    ) -> None:

        if ctrl is not None:
            with control(ctrl):
                if d > 1:
                    _unary_prep(steps, coeffs)
                else:
                    x(steps)
        else:
            if d > 1:
                _unary_prep(steps, coeffs)
            else:
                x(steps)

        inner_walk(steps, coins1, coins2, m_line, n_line, d)

        for i in jrange(1, d + 1):
            cx(m_line[origin - i], m_line[origin + i])
            cx(n_line[origin - i], n_line[origin + i])

        # Copy the position of the particles to the output variables in unary encoding
        for i in jrange(1, d + 1):
            with control(m_line[origin + i]):
                x(qm[:i])
            with control(n_line[origin + i]):
                x(qn[:i])

        z(qn)  # Applies the minus sign for the commutator whenever n is odd via Z gates on the outer right ancilla.

    @custom_control
    def prep_left(
        steps: QuantumVariable,
        coins1: QuantumVariable,
        coins2: QuantumVariable,
        m_line: QuantumVariable,
        n_line: QuantumVariable,
        qm: QuantumVariable,
        qn: QuantumVariable,
        ctrl=None,
    ) -> None:

        if ctrl is not None:
            with control(ctrl):
                if d > 1:
                    _unary_prep(steps, coeffs, conjugate=True)
                else:
                    x(steps)
        else:
            if d > 1:
                _unary_prep(steps, coeffs, conjugate=True)
            else:
                x(steps)

        inner_walk(steps, coins1, coins2, m_line, n_line, d)

        for i in jrange(1, d + 1):
            cx(m_line[origin - i], m_line[origin + i])
            cx(n_line[origin - i], n_line[origin + i])

        # Copy the position of the particles to the output variables in unary encoding
        for i in jrange(1, d + 1):
            with control(m_line[origin + i]):
                x(qm[:i])
            with control(n_line[origin + i]):
                x(qn[:i])

    prep_anc_templates = [
        QuantumVariable(d).template(),  # step ancilla variable for walk
        QuantumVariable(d).template(),  # coin ancilla variable 1 for walk
        QuantumVariable(d).template(),  # coin ancilla variable 2 for walk
        QuantumVariable(2 * d + 1).template(),  # position ancilla variable m_line for walk
        QuantumVariable(2 * d + 1).template(),  # position ancilla variable n_line for walk
    ]
    return prep_right, prep_left, prep_anc_templates
