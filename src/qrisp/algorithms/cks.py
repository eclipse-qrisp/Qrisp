"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
from scipy.special import comb
import jax.numpy as jnp
from qrisp import (
    QuantumVariable,
    QuantumFloat,
    x,
    z,
    ry,
    control,
    conjugate,
    invert,
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange
import numpy.typing as npt
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def cks_params(eps: float, kappa: float, max_beta: int = None) -> Tuple[int, int]:
    """
    Computes the complexity parameter :math:`\\beta` and the truncation order :math:`j_0` for the
    truncated Chebyshev approximation of :math:`1/x`, as described in the `Childs–Kothari–Somma paper <https://arxiv.org/abs/1511.02306>`_.

    Given the condition number :math:`\\kappa` and the target precision
    :math:`\\epsilon`, the parameters are computed as:

    .. math::

        \\beta = \kappa^2 \log\!\left(\\frac{\kappa}{\epsilon}\\right),
        \quad
        j_0 = \sqrt{\\beta \log\!\left(\\frac{4\\beta}{\epsilon}\\right)}.

    If ``max_beta`` is provided, :math:`\\beta` is capped to
    :math:`\\min(\\beta, \\beta_{\\max})`. The returned values are cast to integers
    via the floor operation.

    Parameters
    ----------
    eps : float
        Target precision :math:`\epsilon`.
    kappa : float
        An upper bound for the condition number :math:`\\kappa` of :math:`A`. This value defines the "gap"
        around zero where the function :math:`1/x` is not approximated.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    j0 : int
        Truncation order of the Chebyshev expansion
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    beta : int
        Complexity parameter :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.
    """

    if max_beta == None:
        beta = kappa**2 * np.log(kappa / eps)
    else:
        beta = min(kappa**2 * np.log(kappa / eps), max_beta)
    j0 = np.sqrt(beta * np.log(4 * beta / eps))
    return int(j0), int(beta)


def cks_coeffs(j0: int, b: int) -> npt.NDArray[float]:
    """
    Computes the positive coefficients :math:`\\alpha_i` for the truncated
    Chebyshev expansion of :math:`1/x` up to order :math:`2j_0+1`.

    The approximation is expressed as a linear combination
    of odd Chebyshev polynomials truncated at index :math:`j_0` (Lemma 14):

    .. math::

        g(x) = 4 \\sum_{j=0}^{j_0} (-1)^j
        \\left[ \\sum_{i=j+1}^{b} \\frac{\\binom{2b}{b+i}}{2^{2b}} \\right]
        T_{2j+1}(x)

    The Linear Combination of Unitaries (LCU) lemma requires strictly
    positive coefficients :math:`\\alpha_i > 0`, their absolute values are
    used. The alternating factor :math:`(-1)^j` is later implemented as a set of
    Z-gates within the CKS circuit.

    Parameters
    ----------
    j0 : int
        Truncation order of the Chebyshev expansion,
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    b : float
        Complexity parameter :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.

    Returns
    -------
    coeffs : ndarray
        1-D array of positive Chebyshev coefficients :math:`{\\alpha_{2j+1}}`, 
        corresponding to the odd degree Chebyshev polynomials of the first kind :math:`T_1, T_3, \\dotsc, T_{2j_0+1}`.

    """
    coeffs = []
    for j in range(j0 + 1):
        sum_i = 0
        for i in range(j + 1, b + 1):
            sum_i += comb(2 * b, b + i)

        coeff = 4 * (2 ** (-2 * b)) * sum_i
        coeffs.append(coeff)
    return np.array(coeffs)


def _unary_angles(coeffs: "ArrayLike") -> "ArrayLike":
    """
    Computes rotation angles :math:`\\phi_i` to prepare the unary state :math:`\ket{\\text{unary}}`,
    corresponding to the square-root-amplitude encoding of the Chebyshev coefficients.

    The prepared unary state is

    .. math::

        \ket{\\text{unary}} \propto
        \\sqrt{\\alpha_1}\ket{100\\dots00} + \\sqrt{\\alpha_3}\ket{110\\dots00} +
        \\cdots + \\sqrt{\\alpha_{2j_0+1}}\ket{111\\dots11}

    Parameters
    ----------
    coeffs : ArrayLike
        1-D array of positive Chebyshev coefficients :math:`\\alpha_i`.

    Returns
    -------
    ArrayLike
        1-D array of rotation angles :math:`\\phi_i` for unary state preparation.
    """

    alpha = jnp.sqrt(coeffs) # coeffs need not to be normalized since unary angles only depend on their ratio
    phi = jnp.zeros(len(alpha) - 1)
    phi = phi.at[-1].set(
        jnp.arctan(alpha[-1] / alpha[-2])
    )  # Last angle is determined by ratio of final two amplitudes
    for i in range(len(phi) - 2, -1, -1):
        phi = phi.at[i].set(
            jnp.arctan(alpha[i + 1] / alpha[i] / jnp.cos(phi[i + 1]))
        )  # Recursively compute previous rotation angles
    return 2 * phi  # Compensate for factor 1/2 in ry


def unary_prep(case: QuantumVariable, coeffs: "ArrayLike") -> None:
    """
    Prepares the unary-encoded state :math:`\ket{\\text{unary}}` of Chebyshev coefficients.

    When applied to a variable in state :math:`\ket{0}`, the resulting superposition is

    .. math::

        \ket{\\text{unary}} \propto
        \\sqrt{\\alpha_1}\ket{100\\dots00} + \\sqrt{\\alpha_3}\ket{110\\dots00} +
        \\cdots + \\sqrt{\\alpha_{2j_0+1}}\ket{111\\dots11}

    Parameters
    ----------
    case : QuantumVariable
        Variable with :math:`j_0` qubits on which the unary state preparation will be performed.
        If the variable is in state :math:`\ket{0}`, the state :math:`\ket{\\text{unary}}` is prepared.
    coeffs : ArrayLike
        1-D array of :math:`j_0` Chebyshev coefficients :math:`\\alpha_1,\\alpha_3,\dotsc,\\alpha_{2j_0+1}`.
    
    """
    phi = _unary_angles(coeffs)

    x(case[0])
    ry(phi[0], case[1])
    for i in jrange(1, case.size - 1):
        with control(case[i]):
            ry(phi[i], case[i + 1])


def CKS(A: BlockEncoding, eps: float, kappa: float, max_beta: float = None) -> BlockEncoding:
    """
    Performs the `Childs–Kothari–Somma (CKS) quantum algorithm <https://arxiv.org/abs/1511.02306>`_ to solve the Quantum Linear System Problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}`, using the Chebyshev approximation of :math:`1/x`.    
    When applied to a state $\ket{b}$, the algorithm prepares a state :math:`\ket{\\tilde{x}} \propto A^{-1} \ket{b}`
    within target precision :math:`\epsilon` of the ideal solution :math:`\ket{x}`. 

    For a block-encoded **Hermitian** matrix :math:`A`, this function returns a BlockEncoding of an 
    operator :math:`\\tilde{A}^{-1}` such that :math:`\|\\tilde{A}^{-1} - A^{-1}\| \leq \epsilon`. 
    The inversion is implemented using a polynomial approximation of :math:`1/x` over the domain :math:`D_{\\kappa} = [-1, -1/\\kappa] \\cup [1/\\kappa, 1]`.
    
    The asymptotic complexity is
    :math:`\\mathcal{O}\\!\\left(\\log(N)s \\kappa^2 \\text{polylog}\\!\\frac{s\\kappa}{\\epsilon}\\right)`, where :math:`N` is the matrix size, :math:`s` its sparsity, and 
    :math:`\kappa` its condition number. This represents an exponentially
    better precision scaling compared to the HHL algorithm.

    This function integrates all core components of the CKS approach:
    Chebyshev polynomial approximation, Linear Combination of Unitaries (LCU),
    `qubitization with reflection operators <https://arxiv.org/abs/2208.00567>`_, and the :ref:`Repeat-Until-Success (RUS) <RUS>` protocol.
    The semantics of the approach can be illustrated with the following circuit schematics:

    .. image:: /_static/CKS_schematics.png
       :align: center

    Implementation overview:
      1. Compute the CKS parameters :math:`j_0` and :math:`\\beta` (:func:`cks_params`).
      2. Generate Chebyshev coefficients and the auxiliary unary state (:func:`cks_coeffs`, :func:`unary_prep`).
      3. Build the core LCU structure via qubitization operator.

    The goal of this algorithm is to apply the non-unitary operator :math:`A^{-1}` to the
    input state :math:`\ket{b}`. Following the Chebyshev approach introduced in the CKS paper, we express :math:`A^{-1}` as
    a linear combination of odd Chebyshev polynomials: 

    .. math::
    
        A^{-1}\propto\sum_{j=0}^{j_0}\\alpha_{2j+1}T_{2j+1}(A),

    where :math:`T_k(A)` are Chebyshev polynomials of the first kind and :math:`\\alpha_{2j+1} > 0` are computed
    via :func:`cks_coeffs`. These operators can be efficiently implemented with qubitization, which relies on a unitary
    block encoding :math:`U` of the matrix :math:`A`, and a :ref:`reflection operator <reflection>` :math:`R`.

    If the block encoding unitary is $U$ is Hermitian (:math:`U^2=I`),
    the fundamental iteration step is defined as :math:`(RU)`, where :math:`R` reflects
    around the auxiliary block-encoding state :math:`\ket{G}`, prepared as the ``inner_case`` QuantumFloat.
    Repeated applications of these unitaries, :math:`(RU)^k`, yield a block encoding of the :math:`k`-th Chebyshev polynomial 
    of the first kind :math:`T_k(A)`.
    
    We then construct a linear combination of these block-encoded polynomials using the LCU structure 

    .. math::
    
       \\text{LCU}\ket{0}\ket{\psi}=\\text{PREP}^{\dagger}\cdot \\text{SEL}\cdot \\text{PREP}\ket{0}\ket{\psi}=\\tilde{A}\ket{0}\ket{\psi}.
    
    Here, the :math:`\\text{PREP}` operation prepares an auxiliary ``out_case`` Quantumfloat in the unary state :math:`\ket{\\text{unary}}`
    that encodes the square root of the Chebyshev coefficients :math:`\sqrt{\\alpha_j}`. The :math:`\\text{SEL}` operation selects and applies the
    appropriate Chebyshev polynomial operator :math:`T_k(A)`, implemented by :math:`(RU)^k`, controlled on ``out_case`` in the unary
    state :math:`\ket{\\text{unary}}`. Based on the Hamming-weight :math:`k` of :math:`\ket{\\text{unary}}`,
    the polynomial :math:`T_{2k-1}` is block encoded and applied to the circuit. 
    
    To construct a linear combination of Chebyshev polynomials up to the :math:`2j_0+1`-th order, as in the original paper, 
    our implementation requires :math:`j_0+1` qubits in the ``out_case`` state :math:`\ket{\\text{unary}}`.

    The Chebyshev coefficients alternate in sign :math:`(-1)^j\\alpha_j`.
    Since the LCU lemma requires :math:`\\alpha_j>0`, negative terms are accounted for
    by applying Z-gates on each index qubit in ``out_case``.

    Parameters
    ----------
    A : BlockEncoding
        The block-encoded Hermitian matrix to be inverted.
    eps : float
        Target precision :math:`\epsilon`.
    kappa : float
        An upper bound for the condition number :math:`\\kappa` of :math:`A`. This value defines the "gap"
        around zero where the function :math:`1/x` is not approximated.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing an approximation of the inverse $A^{-1}$.
        
    Examples
    --------

    The following examples demonstrate how the CKS algorithm can be applied to solve the
    quantum linear systems problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}`, using either a direct Hermitian matrix input or
    a preconstructed block-encoding representation.
    
    **Example 1: Solving a 4×4 Hermitian system**

    First, we define a small Hermitian matrix :math:`A` and a right-hand side vector :math:`\\vec{b}`:

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                      [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                      [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                      [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

        kappa = np.linalg.cond(A)
        print("Condition number of A: ", kappa)
        # Condition number of A:  1.8448536035491883

    Next, we solve this linear system using the CKS quantum algorithm:

    ::

        from qrisp import prepare, QuantumFloat
        from qrisp.algorithms.cks import CKS
        from qrisp.block_encodings import BlockEncoding
        from qrisp.jasp import terminal_sampling

        def prep_b():
            operand = QuantumFloat(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():

            BA = BlockEncoding.from_array(A)
            x = CKS(BA, 0.01, kappa).apply_rus(prep_b)()
            return x

        res_dict = main()

        # The resulting dictionary contains the measurement probabilities 
        # for each computational basis state.
        # To extract the corresponding quantum amplitudes (up to sign):
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
        print("QUANTUM SIMULATION\\n", amps)
        # QUANTUM SIMULATION
        # [0.02714082 0.55709921 0.53035395 0.63845794]

    We compare this to the classical normalized solution:

    ::

        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("CLASSICAL SOLUTION\\n", c)  
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    We see that we obtained the same result in our quantum simulation up to precision :math:`\epsilon`!

    To perform quantum resource estimation, replace the ``@terminal_sampling``
    decorator with ``@count_ops(meas_behavior="0")``:

    ::
        
        from qrisp.jasp import count_ops
    
        @count_ops(meas_behavior="0")
        def main():

            BA = BlockEncoding.from_array(A)
            x = CKS(BA, 0.01, kappa).apply_rus(prep_b)()
            return x

        res_dict = main()
        print(res_dict)
        # {'gphase': 51, 't_dg': 1548, 'cz': 150, 'ry': 2, 'z': 12, 'h': 890, 'cy': 50, 'cx': 3516, 'x': 377, 's_dg': 70, 'p': 193, 's': 70, 't': 1173, 'u3': 753, 'measure': 18}

    The printed dictionary lists the estimated quantum gate counts required for the CKS algorithm. Since the simulation itself is not executed, 
    this approach enables scalable gate count estimation for linear systems of arbitrary size.
    
    **Example 2: Using a custom block encoding**

    The previous example displays how to solve the linear system for any Hermitian matrix :math:`A`,
    where the block-encoding is constructed from the Pauli decomposition of :math:`A`. While this approach is efficient when :math:`A`
    corresponds to, e.g., an Ising or a Heisenberg Hamiltonian, the number of Pauli terms may not scale favorably in general.
    For certain sparse matrices, their structure can be exploited to construct more efficient block-encodings.

    In this example, we construct a block-encoding representation for the following following tridiagonal sparse matrix:

    ::

        import numpy as np

        def tridiagonal_shifted(n, mu=1.0, dtype=float):
            I = np.eye(n, dtype=dtype)
            return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)

        N = 8
        A = tridiagonal_shifted(N, mu=3)
        b = np.random.randint(0, 2, size=N)

        print(A)
        # [[ 5.  0.  0.  0. -2.  0.  0.  0.]
        #  [ 0.  5.  0.  0.  0. -2.  0.  0.]
        #  [ 0.  0.  5.  0.  0.  0. -2.  0.]
        #  [ 0.  0.  0.  5.  0.  0.  0. -2.]
        #  [-2.  0.  0.  0.  5.  0.  0.  0.]
        #  [ 0. -2.  0.  0.  0.  5.  0.  0.]
        #  [ 0.  0. -2.  0.  0.  0.  5.  0.]
        #  [ 0.  0.  0. -2.  0.  0.  0.  5.]]

    This matrix can be decomposed using three unitaries: the identity $I$, and two shift operators $V\colon\ket{k}\\rightarrow-\ket{k+N/2 \mod N}$ and $V^{\dagger}\colon\ket{k}\\rightarrow-\ket{k-N/2 \mod N}$.
    We define their corresponding functions and the block-encoding using Linear Combination of Unitaries (LCU):

    ::

        from qrisp import gphase

        def I(qv):
            pass

        def V(qv):
            qv += N//2
            gphase(np.pi, qv[0])

        def V_dg(qv):
            qv -= N//2
            gphase(np.pi, qv[0])

        unitaries = [I, V, V_dg]
        coeffs = np.array([5,1,1,0])
        BA = BlockEncoding.from_lcu(coeffs, unitaries)

    Next, we solve this linear system using the CKS quantum algorithm:

    ::

        def prep_b():
            operand = QuantumFloat(3)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():

            x = CKS(BA, 0.01, kappa).apply_rus(prep_b)()
            return x

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    We, again, compare the solution obtained by quantum simulation with the classical solution

    ::

        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\\n", amps, "\\nCLASSICAL SOLUTION\\n", c)
        # QUANTUM SIMULATION
        # [0.43917486 0.31395935 0.12522139 0.43917505 0.43917459 0.12522104 0.31395943 0.43917502]
        # CLASSICAL SOLUTION
        # [0.43921906 0.3137279  0.12549116 0.43921906 0.43921906 0.12549116 0.3137279  0.43921906]

    Quantum resource estimation can be performed in the same way as in the first example:

    ::

        @count_ops(meas_behavior="0")
        def main():

            x = CKS(BA, 0.01, kappa).apply_rus(prep_b)()
            return x
    
        res_dict = main()
        print(res_dict)
        # {'gphase': 51, 's_dg': 70, 's': 120, 'z': 12, 'x': 127, 't': 298, 'u3': 157, 'cx': 1194, 'p': 118, 'ry': 2, 'h': 390, 't_dg': 348, 'measure': 116}

    """

    j_0, beta = cks_params(eps, kappa, max_beta=max_beta)
    cheb_coeffs = cks_coeffs(j_0, beta)

    m = len(A._anc_templates)

    # Following https://math.berkeley.edu/~linlin/qasc/qasc_notes.pdf (page 104):
    # T1 = (U R), T2 = (U R U_dg R) -> T2^k T1 is block encoding T_{2k+1}(A) 
    # This is valid even if the block encoding unitary is not Hermitian.
    # The first control-(U R) can be simplified to U since:
    # 1) The first qubit of the |unary> state is always 1.
    # 2) The R acts as identity on |0>.

    def inv_unitary(*args):
        with invert():
            A.unitary(*args)

    def T2(*args):
        reflection(args[:m])
        with conjugate(inv_unitary)(*args):
            reflection(args[:m])

    def new_unitary(*args):
        # Core LCU protocol: PREP, SELECT, PREP^†
        with conjugate(unary_prep)(args[0], cheb_coeffs):

            A.unitary(*args[1:])    

            for i in jrange(1, j_0 + 1):
                z(args[0][i])
                with control(args[0][i]):
                    T2(*args[1:])

    new_anc_templates = [QuantumFloat(j_0 + 1).template()] + A._anc_templates
    new_alpha = np.sum(np.abs(cheb_coeffs)) / A.alpha
    return BlockEncoding(new_alpha, new_anc_templates, new_unitary)