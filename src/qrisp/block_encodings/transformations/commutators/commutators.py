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

from .commutators_unary_prep import create_unary_preps
from .commutators_unary_prep_walk import create_unary_preps_walk
from .helper_functions import _chebyshev_sum_commutator_coeffs

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def apply_nested_commutators(
    self,
    B: BlockEncoding,
    coeffs: "ArrayLike",
    method: Literal["default", "walk"] = "default",
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding of a weighted sum of nested commutators.

    For block-encoded **Hermitian** operators $A$ and $B$, this function returns a BlockEncoding
    of the operator

    .. math::

        \mathcal A = \sum_{k=1}^d \gamma_kc_k \text{ad}_A^k(B)

    where each $\text{ad}_A^k(B)$ is a nested commutator $[A,[A,\dotsc[A,B]]$ of order $k$,
    $c_k$ are real non-negative coefficients, and $\gamma_k=i$ if $k$ is odd and $\gamma_k=1$ if $k$ is even.

    Parameters
    ----------
    B : BlockEncoding
        A block-encoded Hermitian operator.
    coeffs : ArrayLike, shape (d,)
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators.
    method : str, optional
        The method to use for constructing the block encoding.
            - "default": Uses a state preparation method with $\mathcal O(d^2)$ depth.
            - "walk": Uses a quantum walk-based state preparation method with $\mathcal O(d)$ depth.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the sum of nested commutators $\mathcal A$.

    Notes
    -----
    - **Complexity**: This implementation requires $\mathcal O(d)$ qubits, $\mathcal O(d)$ calls to the block-encoding $A$,
      and utilizes a state preparation (PREP) oracle of depth $\mathcal O(d^2)$ ("default"), or of depth $\mathcal O(d)$ ("walk").

    Examples
    --------

    ::

        import numpy as np
        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        A = 0.5*X(0)*Z(1) + 0.5*Y(0)*Y(1)
        B = 0.5*Z(0)*Z(1) + 0.5*X(0)*Y(1)

        ad1 = (A*B - B*A)
        ad2 = A*ad1 - ad1*A
        ad3 = A*ad2 - ad2*A

        ad13 = ad1 + ad3
        B_ad13 = BlockEncoding.from_operator(1.j * (ad1 + ad3))

        B_A = BlockEncoding.from_operator(A)
        B_B = BlockEncoding.from_operator(B)

        # BlockEncoding of sum of odd nested commutators
        B_C = B_A.nested_commutators(B_B, np.array([1., 0., 1.,]))

        b = np.array([1., 1., 0., 1.])
        # Prepare variable in state |b>
        def prep_b():
            qv = QuantumFloat(2)
            prepare(qv, b)
            return qv

        @terminal_sampling
        def main():
            return B_C.apply_rus(prep_b)()

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
        print("qrisp:", amps)

    Compare this to the result obtained by first computing the sum of nested commutators
    and subsequently constructing its block encoding.

    ::

        @terminal_sampling
        def main():
            return B_ad13.apply_rus(prep_b)()

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
        print("qrisp:", amps)


    """
    A = self

    ALLOWED_METHODS = {"default", "walk"}
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Invalid method specified: '{method}'. Allowed methods are: {', '.join(ALLOWED_METHODS)}")

    # Rescale coefficients by the appropriate power of the normalization factor for A.
    d = len(coeffs)
    alpha = A.alpha
    beta = B.alpha
    coeffs = np.array(coeffs) * (alpha ** np.arange(1, d + 1))

    if method == "default":
        prep_func_right, prep_func_left, prep_anc_templates = create_unary_preps(d, coeffs)
    elif method == "walk":
        prep_func_right, prep_func_left, prep_anc_templates = create_unary_preps_walk(d, coeffs)

    num_prep_ancs = len(prep_anc_templates)
    use_prep_pair = prep_func_left is not None

    A_walk = A.qubitization()

    num_ops = A.num_ops
    num_ancs_A = A_walk.num_ancs
    num_ancs_B = B.num_ancs

    @custom_control
    def new_unitary(*args, ctrl=None):

        outer_ancs = args[:num_prep_ancs]
        outer_anc_left = args[num_prep_ancs]
        outer_anc_right = args[num_prep_ancs + 1]

        # Ancilla QuantumBool for ensuring that the ancillas for the block-encoding of A are in state |0> after left application of T_m(A) and before right application of T_n(A).
        # This is necessary to reuse these ancillas for both applications of T_k(A) from the left and right.
        anc_qbl = args[num_prep_ancs + 2]

        ancs_A = args[num_prep_ancs + 3 : num_prep_ancs + num_ancs_A + 3]
        # qubits_A = sum([anc.reg for anc in ancs_A], [])

        ancs_B = args[num_prep_ancs + num_ancs_A + 3 : num_prep_ancs + num_ancs_A + num_ancs_B + 3]
        operands = args[-num_ops:]

        # Apply weighted sum of nested commutators expansion in Chebyshev basis.
        # sum_{m,n} (-1)^n C_{m,n} T_m(A) B T_n(A)
        # with conjugate(prep_func)(*outer_ancs, outer_anc_left, outer_anc_right, d, coeffs):

        def select(outer_anc_left, outer_anc_right, anc_qbl, ancs_A, ancs_B, operands):

            def parity(qv1, qv2, qbl):
                for i in jrange(d):
                    cx(qv1[i], qbl[0])
                    cx(qv2[i], qbl[0])

            # Apply phase -i whenever k = m + n is odd.
            with conjugate(parity)(outer_anc_left, outer_anc_right, anc_qbl):
                p(np.pi / 2, anc_qbl)

            # Apply minus sign for the term T_m(A)BT_n(A) whenever n is odd via Z gates on the outer right ancilla.
            if not use_prep_pair:
                z(outer_anc_right)

            # Apply T_n(A) from the right.
            for i in jrange(d):
                # |0000...> = T_0(A), |1000> = T_1(A)
                with control(outer_anc_right[i]):
                    A_walk.unitary(*ancs_A, *operands)

            # To reuse ancillas for the block-encoding of A for applying T_k(A) from the left,
            # we must ensure that they are in state |0>.
            qubits_A = sum([anc.reg for anc in ancs_A], [])
            mcx(qubits_A, anc_qbl, ctrl_state=0)

            # Apply B
            if ctrl is not None:
                with control(ctrl):
                    B.unitary(*ancs_B, *operands)
            else:
                B.unitary(*ancs_B, *operands)

            # Apply T_m(A) from the left.
            for i in jrange(d):
                # |0000...> = T_0(A), |1000> = T_1(A)
                with control(outer_anc_left[i]):
                    # Ensure that ancillas for block-encoding of A are in state |0>.
                    with control(anc_qbl):
                        A_walk.unitary(*ancs_A, *operands)

            # Ensure that measurment in |0> yields the correct result.
            x(anc_qbl)

        if use_prep_pair:
            if ctrl is not None:
                with control(ctrl):
                    prep_func_right(*outer_ancs, outer_anc_left, outer_anc_right)
            else:
                prep_func_right(*outer_ancs, outer_anc_left, outer_anc_right)

            select(outer_anc_left, outer_anc_right, anc_qbl, ancs_A, ancs_B, operands)

            with invert():
                if ctrl is not None:
                    with control(ctrl):
                        prep_func_left(*outer_ancs, outer_anc_left, outer_anc_right)
                else:
                    prep_func_left(*outer_ancs, outer_anc_left, outer_anc_right)
        else:
            with conjugate(prep_func_right)(*outer_ancs, outer_anc_left, outer_anc_right):
                select(outer_anc_left, outer_anc_right, anc_qbl, ancs_A, ancs_B, operands)

    new_anc_templates = (
        prep_anc_templates
        + [
            QuantumVariable(d).template(),  # unary-encoded m index for T_m(A)
            QuantumVariable(d).template(),  # unary-encoded n index for T_n(A)
            QuantumBool().template(),  # ancilla for reusing qubits for left application of T_k(A)
        ]
        + A_walk._anc_templates
        + B._anc_templates
    )

    new_alpha = np.sum(np.abs(_chebyshev_sum_commutator_coeffs(coeffs))) * beta

    return BlockEncoding(new_alpha, new_anc_templates, new_unitary, num_ops=num_ops)
