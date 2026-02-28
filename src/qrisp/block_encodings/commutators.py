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
from qrisp.core import QuantumVariable
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.core.gate_application_functions import x, z, mcx
from qrisp.environments import conjugate, control
from qrisp.jasp import (
    jrange,
    qache,
    q_switch,
)
from qrisp.qtypes import QuantumBool, QuantumFloat
from scipy.sparse import csr_array, csr_matrix
from typing import Any, Callable, TYPE_CHECKING, Union

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


def _state_prep(qa, qb, qc, d, coeffs):
    r"""
    Prepares the state $C_{m,n}|m>|n>$ for unary encoded variables in $\mathcal O(d^2)$.
    
    
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

    prepare(qa, target)

    def case_func(i, qv):
        x(qv[: i + 1])

    q_switch(qa[:n], case_func, qb)
    q_switch(qa[n:], case_func, qc)


def nested_commutators(A: BlockEncoding, B: BlockEncoding, coeffs) -> BlockEncoding:
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
        inner_anc = args[3]

        ancs_A = args[4 : num_ancs_A + 4]
        qubits_A = sum([anc.reg for anc in ancs_A], [])

        ancs_B = args[num_ancs_A + 4 : num_ancs_A + num_ancs_B + 4]
        operands = args[-num_ops:]

        # sum_{m,n} c_{m,n} T_m(A) B T_n(A)
        with conjugate(_state_prep)(outer_anc, outer_anc_left, outer_anc_right, d, coeffs):

            # Signs
            z(outer_anc_left[1 : d + 1])
        
            # Apply T_k(A) from the left (first d+1 qubits auf ancilla)
            for i in jrange(d):
                # |1000...> = T_0(A), |1100> = T_1(A)
                with control(outer_anc_left[i + 1]):
                    A_walk.unitary(*ancs_A, *operands)

            # To reuse ancillas for applying T_k(A) from the right,
            # we must ensure that they are in state |0>
            mcx(qubits_A, inner_anc, ctrl_state=0)

            # Apply B
            B.unitary(*ancs_B, *operands)

            # Apply T_k(A) from the right (second d+1 qubits auf ancilla)
            for i in jrange(d):
                # |1000...> = T_0(A), |1100> = T_1(A)
                with control(outer_anc_right[i + 1]):
                    # Ensure that ancillas are in state |0>
                    with control(inner_anc):
                        A_walk.unitary(*ancs_A, *operands)

            # flip |1> -> |0>
            # Ensure that measurment in |0> yields the correct result
            x(inner_anc)

    new_anc_templates = [QuantumVariable(2 * n).template(), 
                         QuantumVariable(d + 1).template(), 
                         QuantumVariable(d + 1).template(), 
                         QuantumBool().template(),
                        ] + A_walk._anc_templates + B._anc_templates
    new_alpha = 1 # TBD
    return BlockEncoding(new_alpha, new_anc_templates, new_unitary)
