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

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def create_unary_preps(
    d: int,
    coeffs: npt.NDArray[Any] = None,
) -> None:
    r"""
    Coherently prepares a state that encodes the coefficients of the Chebyshev expansion of a weighted sum of nested commutators in a two-dimensional grid of unary-encoded indices.

    Each nested commutator $\text{ad}_A^k(B)$ can be expressed as a sum of terms of the form $C_{k,m,n}T_m(A)BT_n(A)$, where $T_m(A)$ are Chebyshev polynomials of the first kind evaluated at $A$.
    The state prepared by this function encodes the square root of the weighted sum of these coefficients in the amplitudes of a superposition over the indices $m$ and $n$,
    which can then be used to apply the corresponding operators in superposition.

    .. math::

            \sum_{k=1}^d\text{ad}_A^k(B) = \sum_{k=1}^d c_k\sum_{m,n}C_{k,m,n}T_m(A)BT_n(A)

    .. math::

            \text{PREP}\ket{0}_a\ket{0}_m\ket{0}_n \propto \sum_{m,n}\sqrt{\sum_{k=1}^d c_kC_{k,m,n}}\ket{m}\ket{n}

    Parameters
    ----------
    d : int
        The depth of the commutator expansion, which determines the size of the state.
    coeffs : ArrayLike, shape (d,), optional
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators.
        If None, defaults to a delta distribution on the highest order commutator.

    Returns
    -------
    prep_right : Callable
        A function that prepares the right side of the state encoding the coefficients of the Chebyshev expansion of the weighted sum of nested commutators.
        The function takes the following arguments:
            anc : QuantumVariable
                A binary-encoded ancilla QuantumVariable of size $2\lceil\log_2(d)\rceil$.
                Used to prepare the superposition over the $m$ and $n$ indices in $\mathcal O(d^2)$ depth.
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
            - anc : QuantumVariable of size $2\lceil\log_2(d)\rceil$.

    Notes
    -----
    - **Complexity**: This state preparation requires $\mathcal O(d)$ qubits and $\mathcal O(d^2)$ depth.
    - This function is designed to be used as a state preparation oracle within the nested_commutator function, and is not intended for standalone use.
    - The state prepared by this function encodes the coefficients of the Chebyshev expansion of the nested commutators in a two-dimensional grid of unary-encoded indices,
      which can then be used to apply the corresponding operators in superposition.

    """

    n = int(np.ceil(np.log2(d + 1)))

    def target_state(d, coeffs):

        target = np.zeros(2 ** (2 * n), dtype=np.complex128)

        C_matrix = np.sqrt(_chebyshev_sum_commutator_coeffs(coeffs))

        # Flatten the 2D coefficient matrix into a 1D array corresponding to the amplitudes of the target state
        for i in range(d + 1):
            for j in range(d + 1):
                target[i + (j << n)] += C_matrix[i, j]

        return target

    if coeffs is None:
        coeffs = np.zeros(d, dtype=np.complex128)
        coeffs[d - 1] = 1

    @custom_control
    def prep_right(anc: QuantumVariable, qm: QuantumVariable, qn: QuantumVariable, ctrl=None) -> None:

        target = target_state(d, coeffs)

        if ctrl is not None:
            with control(ctrl):
                prepare(anc, target)
        else:
            prepare(anc, target)

        def case_func(i, qv):
            x(qv[:i])

        n = int(np.ceil(np.log2(d + 1)))
        q_switch(anc[:n], case_func, qm, branch_amount=d + 1)
        q_switch(anc[n:], case_func, qn, branch_amount=d + 1)

    @custom_control
    def prep_left(anc: QuantumVariable, qm: QuantumVariable, qn: QuantumVariable, ctrl=None) -> None:

        target = target_state(d, coeffs).conj()

        if ctrl is not None:
            with control(ctrl):
                prepare(anc, target)
        else:
            prepare(anc, target)

        def case_func(i, qv):
            x(qv[:i])

        n = int(np.ceil(np.log2(d + 1)))
        q_switch(anc[:n], case_func, qm, branch_amount=d + 1)
        q_switch(anc[n:], case_func, qn, branch_amount=d + 1)

    prep_anc_templates = [QuantumVariable(2 * n).template()]  # binary-encoded ancilla for coefficient preparation
    return prep_right, prep_left, prep_anc_templates
