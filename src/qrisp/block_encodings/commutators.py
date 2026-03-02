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
import numpy as np
import numpy.typing as npt
from qrisp.core import QuantumArray, QuantumVariable
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.core.gate_application_functions import x, z, mcx, swap, h, cx
from qrisp.environments import conjugate, control, invert
from qrisp.jasp import (
    jrange,
    qache,
    q_switch,
)
from qrisp.qtypes import QuantumBool, QuantumFloat
from scipy.sparse import csr_array, csr_matrix
from typing import Any, Callable, Literal, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from jax.typing import ArrayLike

MatrixType = Union[npt.NDArray[Any], csr_array, csr_matrix]

from qrisp.block_encodings import BlockEncoding

import numpy as np
import math
from numpy.polynomial.chebyshev import poly2cheb


def _get_chebyshev_commutator_coeffs(d):
    """
    Calculates the coefficient matrix C_{m,n} for the Chebyshev 
    expansion of the nested commutator ad_A^d(B).
    
    Returns a (d+1) x (d+1) numpy array.
    """
    # Initialize the coefficient matrix C_{m,n} with zeros
    C = np.zeros((d + 1, d + 1))
    
    for k in range(d + 1):
        # 1. Calculate the binomial coefficient and alternating sign
        term_weight = ((-1)**k) * math.comb(d, k)
        
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


def unary_prep(
    anc: QuantumVariable,
    qm: QuantumVariable,
    qn: QuantumVariable,
    d: int,
    coeffs: npt.NDArray[Any],
)-> None:
    r"""
    Coherently prepares a state that encodes the coefficients of the Chebyshev expansion of the nested commutators in a two-dimensional grid of unary-encoded indices.

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
        A binary-encoded ancilla QuantumVariable of size $2\lceil\log2(d)\rceil$. 
        Used to prepare the superposition over the $m$ and $n$ indices in $mathcal O(d^2)$ depth.
    qm : QuantumVariable
        A unary-encoded QuantumVariable of size d+1, representing the $m$ index.
    qn : QuantumVariable
        A unary-encoded QuantumVariable of size d+1, representing the $n$ index.
    d : int
        The depth of the commutator expansion, which determines the size of the state.
    coeffs : ArrayLike, shape (d,)
        The non-negative coefficients for the weighted sum of commutators.

    Notes
    -----
    - **Complexity**: This state preparation requires $\mathcal O(d)$ qubits and$\mathcal O(d^2)$ depth.
    - This function is designed to be used as a state preparation oracle within the nested_commutator function, and is not intended for standalone use.
    - The state prepared by this function encodes the coefficients of the Chebyshev expansion of the nested commutators in a two-dimensional grid of unary-encoded indices, 
      which can then be used to apply the corresponding operators in superposition.
    
    """

    def target_state(d, coeffs):

        n = int(np.ceil(np.log2(d + 1)))
        target = np.zeros(2**(2 * n))

        C_matrix = np.zeros((d + 1, d + 1))

        # Only odd terms
        for k in range(1, d + 1, 2):
            Ck_matrix = _get_chebyshev_commutator_coeffs(k)
            rows, cols = Ck_matrix.shape
            C_matrix[:rows, :cols] += coeffs[k] * Ck_matrix

        C_matrix = np.sqrt(np.abs(C_matrix))

        for i in range(k+1):
            for j in range(k+1):
                target[i + (j<<n)] += coeffs[k] * C_matrix[i,j]

        return target

    target = target_state(d, coeffs)
    n = int(np.ceil(np.log2(d + 1)))

    prepare(anc, target)

    def case_func(i, qv):
        x(qv[: i + 1])

    q_switch(anc[:n], case_func, qm)
    q_switch(anc[n:], case_func, qn)


def unary_walk_prep(
    anc: QuantumVariable,
    qm: QuantumVariable,
    qn: QuantumVariable,
    d: int,
    coeffs: npt.NDArray[Any] = None,
) -> None:
    r"""
    Coherently prepares a state that encodes the coefficients of the Chebyshev expansion of the nested commutators in a two-dimensional grid of unary-encoded indices
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
    anc : QuantumVariable
        A unary-encoded ancilla QuantumVariable of size d, used to control the walk steps.
    qm : QuantumVariable
        A unary-encoded QuantumVariable of size d+1, representing the $m$ index.
    qn : QuantumVariable
        A unary-encoded QuantumVariable of size d+1, representing the $n$ index.
    d : int
        The depth of the commutator expansion, which determines the size of the walk.
    coeffs : ArrayLike, shape (d,), optional
        The non-negative coefficients for the weighted sum of commutators. 
        If None, defaults to a delta distribution on the highest order commutator.

    Notes
    -----
    - **Complexity**: This state preparation requires $\mathcal O(d)$ qubits and$\mathcal O(d)$ depth.
    - The walk is implemented using two sets of coin variables and two position variables (m_line and n_line) to achieve perfect parallelism in the shift operations, 
      resulting in $\mathcal O(d)$ depth regardless of the number of steps.
    - The crucial minus signs for the commutator are implemented via Z gates on the coin variables, 
      which are applied in parallel with the Hadamard gates to create the necessary interference patterns in the walk.
    
    """
    from qrisp.algorithms.cks import unary_prep

    # 1. Define the 1D line size. 
    # For depth d, the furthest the particle can walk is d steps.
    # We need a line from -d to +d, giving a total size of 2d + 1.
    size = 2 * d + 1
    origin = d  # The center of the array represents m=0 and n=0

    if coeffs is None:
        coeffs = np.zeros(d)
        coeffs[d-1] = 1
    else:
        # Rescale coefficients 
        coeffs = np.array(coeffs) * np.array([np.sum(np.abs(_get_chebyshev_commutator_coeffs(k))) for k in range(d)])
        coeffs = coeffs / np.sum(coeffs)

    unary_prep(anc, coeffs)

    coins1 = QuantumArray(QuantumBool(), shape=(d,))
    coins2 = QuantumArray(QuantumBool(), shape=(d,))
    m_line = QuantumVariable(size, name="m_line")
    n_line = QuantumVariable(size, name="n_line")

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
                        swap(reg[i], reg[i+1])
                        
                # Layer 2: Swap all Odd-Even index pairs (1-2, 3-4, 5-6...)
                with control(c2, ctrl_state=0):
                    for i in range(1, size - 1, 2):
                        swap(reg[i], reg[i+1])
                        
            # Apply the walk to the chosen register
            with control(anc[step]):

                with control(c1, ctrl_state=0):
                    apply_symmetric_walk(m_line)
                    
                with control(c1):
                    apply_symmetric_walk(n_line)


    inner_walk(coins1, coins2, m_line, n_line, d)   

    for i in range(size):
        cx(m_line[i], qm[abs(i - origin)])
        cx(n_line[i], qn[abs(i - origin)])

    with invert():
        inner_walk(coins1, coins2, m_line, n_line, d)

    coins1.delete()
    coins2.delete()
    m_line.delete()
    n_line.delete()


def nested_commutators(
    A: BlockEncoding,
    B: BlockEncoding,
    coeffs: "ArrayLike",
    method: Literal["default", "walk"] = "default",
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding of a weighted sum odd nested commutators.

    For block-encoded Hermitian operators $A$ and $B$, this function returns a BlockEncoding 
    of the operator 

    .. math:: 

        \mathcal A = \sum_{k=1}^{\lfloor d/2\rfloor} c_{2k-1} \text{ad}_A^k(B)

    where each $\text{ad}_A^k(B)$ is a nested commutator $[A,[A,\dotsc[A,B]]$ of order $k$.

    Parameters
    ----------
    A : BlockEncoding
        A block-encoded Hermitian operator.
    B : BlockEncoding
        A block-encoded Hermitian operator.
    coeffs : ArrayLike, shape (d,)
        The non-negative coefficients $c_k\geq0$.
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
      and utilizes a state preparation (PREP) oracle of detph $\mathcal O(d^2)$.

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
        B_C = nested_commutator(B_A, B_B, np.array([0., 1., 0., 1.,]))

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
    elif method == "walk":
        prep_func = unary_walk_prep

    A_walk = A.qubitization()

    num_ops = A.num_ops
    num_ancs_A = A_walk.num_ancs
    num_ancs_B = B.num_ancs

    d = len(coeffs)
    n = int(np.ceil(np.log2(d + 1)))

    def new_unitary(*args):

        outer_anc = args[0]
        outer_anc_left = args[1]
        outer_anc_right = args[2]

        # Reuse ancillas
        anc_qbl = args[3]

        ancs_A = args[4 : num_ancs_A + 4]
        qubits_A = sum([anc.reg for anc in ancs_A], [])

        ancs_B = args[num_ancs_A + 4 : num_ancs_A + num_ancs_B + 4]
        operands = args[-num_ops:]

        # sum_{m,n} c_{m,n} T_m(A) B T_n(A)
        with conjugate(prep_func)(outer_anc, outer_anc_left, outer_anc_right, d, coeffs):

            # Signs
            z(outer_anc_left[1 : d + 1])
        
            # Apply T_k(A) from the left (first d+1 qubits auf ancilla)
            for i in jrange(d):
                # |1000...> = T_0(A), |1100> = T_1(A)
                with control(outer_anc_left[i + 1]):
                    A_walk.unitary(*ancs_A, *operands)

            # To reuse ancillas for applying T_k(A) from the right,
            # we must ensure that they are in state |0>
            mcx(qubits_A, anc_qbl, ctrl_state=0)

            # Apply B
            B.unitary(*ancs_B, *operands)

            # Apply T_k(A) from the right (second d+1 qubits auf ancilla)
            for i in jrange(d):
                # |1000...> = T_0(A), |1100> = T_1(A)
                with control(outer_anc_right[i + 1]):
                    # Ensure that ancillas are in state |0>
                    with control(anc_qbl):
                        A_walk.unitary(*ancs_A, *operands)

            # flip |1> -> |0>
            # Ensure that measurment in |0> yields the correct result
            x(anc_qbl)

    new_anc_templates = [QuantumVariable(2 * n).template() if method == "default" else QuantumVariable(d).template(), 
                         QuantumVariable(d + 1).template(), 
                         QuantumVariable(d + 1).template(), 
                         QuantumBool().template(),
                        ] + A_walk._anc_templates + B._anc_templates
    new_alpha = 1 # TBD
    return BlockEncoding(new_alpha, new_anc_templates, new_unitary)
