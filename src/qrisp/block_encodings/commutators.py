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

import jax.numpy as jnp
import math
import numpy as np
import numpy.typing as npt
from qrisp.core import QuantumVariable
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings import BlockEncoding
from qrisp.core.gate_application_functions import x, z, mcx, swap, h, cx, p
from qrisp.environments import conjugate, control
from qrisp.jasp import (
    jrange,
    qache,
    q_switch,
)
from qrisp.qtypes import QuantumBool, QuantumFloat
from typing import Any, Literal, TYPE_CHECKING

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
    C = np.zeros((d + 1, d + 1))
    for k in range(1, d + 1):
        Ck_matrix = _chebyshev_commutator_coeffs(k)
        rows, cols = Ck_matrix.shape
        C[:rows, :cols] += coeffs[k - 1] * Ck_matrix
    return C


def unary_prep(
    anc: QuantumVariable,
    qm: QuantumVariable,
    qn: QuantumVariable,
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
    anc : QuantumVariable
        A binary-encoded ancilla QuantumVariable of size $2\lceil\log_2(d)\rceil$.
        Used to prepare the superposition over the $m$ and $n$ indices in $\mathcal O(d^2)$ depth.
    qm : QuantumVariable
        A unary-encoded QuantumVariable of size d, representing the $m$ index.
    qn : QuantumVariable
        A unary-encoded QuantumVariable of size d, representing the $n$ index.
    d : int
        The depth of the commutator expansion, which determines the size of the state.
    coeffs : ArrayLike, shape (d,), optional
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators.
        If None, defaults to a delta distribution on the highest order commutator.

    Notes
    -----
    - **Complexity**: This state preparation requires $\mathcal O(d)$ qubits and $\mathcal O(d^2)$ depth.
    - This function is designed to be used as a state preparation oracle within the nested_commutator function, and is not intended for standalone use.
    - The state prepared by this function encodes the coefficients of the Chebyshev expansion of the nested commutators in a two-dimensional grid of unary-encoded indices,
      which can then be used to apply the corresponding operators in superposition.

    """

    if coeffs is None:
        coeffs = np.zeros(d)
        coeffs[d - 1] = 1

    def target_state(d, coeffs):

        n = int(np.ceil(np.log2(d + 1)))
        target = np.zeros(2 ** (2 * n))

        C_matrix = np.sqrt(np.abs(_chebyshev_sum_commutator_coeffs(coeffs)))

        # Flatten the 2D coefficient matrix into a 1D array corresponding to the amplitudes of the target state
        for i in range(d + 1):
            for j in range(d + 1):
                target[i + (j << n)] += C_matrix[i, j]

        return target

    target = target_state(d, coeffs)
    prepare(anc, target)

    def case_func(i, qv):
        x(qv[:i])

    n = int(np.ceil(np.log2(d + 1)))
    q_switch(anc[:n], case_func, qm, branch_amount=d+1)
    q_switch(anc[n:], case_func, qn, branch_amount=d+1)


def unary_walk_prep(
    steps: QuantumVariable,
    coins1: QuantumVariable,
    coins2: QuantumVariable,
    m_line: QuantumVariable,
    n_line: QuantumVariable,
    qm: QuantumVariable,
    qn: QuantumVariable,
    d: int,
    coeffs: npt.NDArray[Any] = None,
) -> None:
    r"""
    Coherently prepares a state that encodes the coefficients of the Chebyshev expansion of a weighted sum of nested commutators in a two-dimensional grid of unary-encoded indices
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
    d : int
        The depth of the commutator expansion, which determines the size of the walk.
    coeffs : ArrayLike, shape (d,), optional
        The non-negative coefficients $c_1,c_2,\dots,c_d$ for the weighted sum of commutators.
        If None, defaults to a delta distribution on the highest order commutator.

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
    from qrisp.algorithms.cks import unary_prep

    # 1. Define the 1D line size.
    # For depth d, the furthest the particle can walk is d steps.
    # We need a line from -d to +d, giving a total size of 2d + 1.
    size = 2 * d + 1
    origin = d  # The center of the array represents m=0 and n=0

    if coeffs is None:
        coeffs = np.zeros(d)
        coeffs[d - 1] = 1
    else:
        # Rescale coefficients
        coeffs = np.array(coeffs) * np.array(
            [np.sum(np.abs(_chebyshev_commutator_coeffs(k))) for k in range(d)]
        )
        coeffs = coeffs / np.sum(coeffs)

    if d > 1:
        unary_prep(steps, coeffs)
    else:
        x(steps)

    def inner_walk(coins1, coins2, m_line, n_line, step):

        # Initialize the particles directly at the origin (m=0, n=0)
        x(m_line[origin])
        x(n_line[origin])

        for step in range(d):
            c1 = coins1[step]
            c2 = coins2[step]

            h(c1)
            z(c1)  # Applies the crucial minus sign for the commutator
            h(c2)

            # Define the perfectly parallel, O(1) depth shift operator
            def apply_symmetric_walk(reg):
                # Layer 1: Swap all Even-Odd index pairs (0-1, 2-3, 4-5...)
                with control(c2):
                    for i in range(0, size - 1, 2):
                        swap(reg[i], reg[i + 1])

                # Layer 2: Swap all Odd-Even index pairs (1-2, 3-4, 5-6...)
                with control(c2, ctrl_state=0):
                    for i in range(1, size - 1, 2):
                        swap(reg[i], reg[i + 1])

            # Apply the walk to the chosen register
            with control(steps[step]):

                with control(c1, ctrl_state=0):
                    apply_symmetric_walk(m_line)

                with control(c1):
                    apply_symmetric_walk(n_line)

    inner_walk(coins1, coins2, m_line, n_line, d)

    for i in range(1, d + 1):
        cx(m_line[origin - i], m_line[origin + i])
        cx(n_line[origin - i], n_line[origin + i])

    # Copy the position of the particles to the output variables in unary encoding
    for i in range(1, d + 1):
        with control(m_line[origin + i]):
            x(qm[:i])
        with control(n_line[origin + i]):
            x(qn[:i])


def nested_commutators(
    A: BlockEncoding,
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
    A : BlockEncoding
        A block-encoded Hermitian operator.
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
        B_C = nested_commutators(B_A, B_B, np.array([1., 0., 1.,]))

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

    ALLOWED_METHODS = {"default", "walk"}
    if method not in ALLOWED_METHODS:
        raise ValueError(
            f"Invalid method specified: '{method}'. "
            f"Allowed methods are: {', '.join(ALLOWED_METHODS)}"
        )

    if method == "default":
        prep_func = unary_prep
        num_prep_ancs = 1
    elif method == "walk":
        prep_func = unary_walk_prep
        num_prep_ancs = 5

    A_walk = A.qubitization()

    num_ops = A.num_ops
    num_ancs_A = A_walk.num_ancs
    num_ancs_B = B.num_ancs

    d = len(coeffs)
    n = int(np.ceil(np.log2(d + 1)))

    # Rescale coefficients by the appropriate power of the normalization factor for A.
    alpha = A.alpha
    beta = B.alpha
    coeffs = np.array(coeffs) * (alpha ** np.arange(1, d + 1))

    def new_unitary(*args):

        outer_ancs = args[:num_prep_ancs]
        outer_anc_left = args[num_prep_ancs]
        outer_anc_right = args[num_prep_ancs + 1]

        # Ancilla QuantumBool for ensuring that the ancillas for the block-encoding of A are in state |0> after left application of T_m(A) and before right application of T_n(A).
        # This is necessary to reuse these ancillas for both applications of T_k(A) from the left and right.
        anc_qbl = args[num_prep_ancs + 2]

        ancs_A = args[num_prep_ancs + 3 : num_prep_ancs + num_ancs_A + 3]
        qubits_A = sum([anc.reg for anc in ancs_A], [])

        ancs_B = args[
            num_prep_ancs + num_ancs_A + 3 : num_prep_ancs + num_ancs_A + num_ancs_B + 3
        ]
        operands = args[-num_ops:]

        # Apply weighted sum of nested commutators expansion in Chebyshev basis.
        # sum_{m,n} (-1)^n C_{m,n} T_m(A) B T_n(A)
        with conjugate(prep_func)(
            *outer_ancs, outer_anc_left, outer_anc_right, d, coeffs
        ):

            def parity(qv1, qv2, qbl):
                for i in jrange(d):
                    cx(qv1[i], qbl[0])
                    cx(qv2[i], qbl[0])

            # Apply phase -i whenever k = m + n is odd.
            with conjugate(parity)(outer_anc_left, outer_anc_right, anc_qbl):
                p(np.pi / 2, anc_qbl)

            # Apply minus sign for the term T_m(A)BT_n(A) whenever n is odd via Z gates on the outer right ancilla.
            z(outer_anc_right)

            # Apply T_n(A) from the right.
            for i in jrange(d):
                # |0000...> = T_0(A), |1000> = T_1(A)
                with control(outer_anc_right[i]):
                    A_walk.unitary(*ancs_A, *operands)

            # To reuse ancillas for the block-encoding of A for applying T_k(A) from the left,
            # we must ensure that they are in state |0>.
            mcx(qubits_A, anc_qbl, ctrl_state=0)

            # Apply B
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

    if method == "default":

        new_anc_templates = (
            [
                QuantumVariable(
                    2 * n
                ).template(),  # binary-encoded ancilla for coefficient preparation
                QuantumVariable(d).template(),  # unary-encoded m index for T_m(A)
                QuantumVariable(d).template(),  # unary-encoded n index for T_n(A)
                QuantumBool().template(),  # ancilla for reusing qubits for left application of T_k(A)
            ]
            + A_walk._anc_templates
            + B._anc_templates
        )

    elif method == "walk":

        new_anc_templates = (
            [
                QuantumVariable(d).template(),  # step ancilla variable for walk
                QuantumVariable(d).template(),  # coin ancilla variable 1 for walk
                QuantumVariable(d).template(),  # coin ancilla variable 2 for walk
                QuantumVariable(
                    2 * d + 1
                ).template(),  # position ancilla variable m_line for walk
                QuantumVariable(
                    2 * d + 1
                ).template(),  # position ancilla variable n_line for walk
                QuantumVariable(d).template(),  # unary-encoded m index for T_m(A)
                QuantumVariable(d).template(),  # unary-encoded n index for T_n(A)
                QuantumBool().template(),  # ancilla for reusing qubits for left application of T_k(A)
                QuantumVariable(num_ancs_A).template(),
                QuantumVariable(num_ancs_B).template(),
            ]
            + A_walk._anc_templates
            + B._anc_templates
        )

    new_alpha = np.sum(np.abs(_chebyshev_sum_commutator_coeffs(coeffs))) * beta

    return BlockEncoding(new_alpha, new_anc_templates, new_unitary, num_ops=num_ops)
