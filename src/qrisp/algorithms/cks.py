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
    measure,
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.alg_primitives.prepare import prepare
from qrisp.jasp import jrange, RUS
from qrisp.operators import QubitOperator


def CKS_parameters(A, eps, kappa=None, max_beta=None):
    """
    Computes the Chebyshev-series complexity parameter :math:`\\beta` and the truncation order :math:`j_0` for the
    truncated Chebyshev approximation of :math:`1/x` used in the Childs–Kothari–Somma (CKS) algorithm.

    To avoid confusion with the vector :math:`\ket{b}`, we represent the complexity parameter as :math:`\\beta`.

    This function distinguishes two input cases:

    1. **Matrix input** — If ``A`` is a NumPy array (Hermitian matrix), the
       condition number :math:`\\kappa` is computed internally as
       :math:`\\kappa = \\mathrm{cond}(A)`.
    2. **Tuple input** — If ``A`` is a tuple of length 3, the function assumes
       external context provides or manages :math:`\\kappa` directly. In this
       case, ``kappa`` must be specified explicitly or precomputed elsewhere.
       Additionally, the block encoding unitary :math:`U` supplied must satisfy
       the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
       required for the correctness of the Chebyshev polynomial block encoding
       and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.

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
    A : np.ndarray or tuple
        The :math:`N \\times N` Hermitian matrix :math:`A` of the linear system
        :math:`A\\vec{x} = \\vec{b}`, or a 3-tuple representing the block encoding of A in which :math:`\\kappa` is handled externally.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within
        :math:`\epsilon` of :math:`\ket{x}.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required if ``A`` is a tuple.
    max_beta : float, optional
        Optional upper bound on :math:`\\beta` for the complexity parameter.

    Returns
    -------
    j_0 : int
        Truncation order of the Chebyshev expansion,
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    beta : float
        Complexity parameter, :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.
    """
    if isinstance(A, tuple) and len(A) == 3:
        kappa = kappa
    else:
        kappa = np.linalg.cond(A)

    if max_beta == None:
        beta = kappa**2 * np.log(kappa / eps)
    else:
        beta = min(kappa**2 * np.log(kappa / eps), max_beta)
    j_0 = np.sqrt(beta * np.log(4 * beta / eps))
    return int(j_0), int(beta)


def cheb_coefficients(j0, b):
    """
    Calculates the positive coefficients :math:`\\alpha_i` for the truncated
    Chebyshev expansion of :math:`1/x` up to order :math:`2j_0+1`.

    The approximation is expressed as a linear combination
    of odd Chebyshev polynomials truncated at index :math:`j_0`:

    .. math::

        g(x) = 4 \\sum_{j=0}^{j_0} (-1)^j
        \\left[ \\sum_{i=j+1}^{b} \\frac{\\binom{2b}{b+i}}{2^{2b}} \\right]
        T_{2j+1}(x)

    The Linear Combination of Unitaries (LCU) lemma requires strictly
    positive coefficients :math:`\\alpha_i > 0`, their absolute values are
    used. The alternating factor :math:`(-1)^j` is later implemented as a set of
    conditional Z gates within the Chebyshev circuit (see :func:`inner_CKS`).

    Parameters
    ----------
    j0 : int
        Truncation order of the Chebyshev expansion,
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    b : float
        Complexity parameter, :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.

    Returns
    -------
    coeffs : np.ndarray
        Array of positive Chebyshev coefficients
        :math:`{\\alpha_i}`, corresponding to the odd degrees of Chebyshev polynomials of the first order :math:`T_1, T_3, \\dots, T_{2j_0+1}`.
    """
    coeffs = []
    for j in range(j0 + 1):
        sum_i = 0
        for i in range(j + 1, b + 1):
            sum_i += comb(2 * b, b + i)

        coeff = 4 * (2 ** (-2 * b)) * sum_i
        coeffs.append(coeff)
    return np.array(coeffs)


def unary_angles(coeffs):
    """
    Computes rotation angles :math:`\\phi_i` to prepare the unary state :math:`\ket{\\text{unary}}`,
    corresponding to the square-root-amplitude encoding of the Chebyshev
    coefficients.

    The prepared unary state is

    .. math::

        |\\text{unary}\\rangle =
        \\alpha_1|100\\dots00\\rangle + \\alpha_3|110\\dots00\\rangle +
        \\cdots + \\alpha_{2j_0+1}|111\\dots11\\rangle

    Angles are determined recursively via :math:`\\arctan` ratios of
    target amplitudes :math:`\\sqrt{\\alpha_i}`. The resulting angles are multiplied by 2 to match the ``ry`` gate convention
    :math:`\\mathrm{ry}(\\theta) = e^{-i\\theta Y/2}`.

    Parameters
    ----------
    coeffs : np.ndarray
        Positive Chebyshev coefficients :math:`\\alpha_i`.

    Returns
    -------
    phi : np.ndarray
        Rotation angles :math:`\\phi_i` for unary state preparation.
    """

    alpha = np.sqrt(coeffs / len(coeffs))
    phi = jnp.zeros(len(alpha) - 1)
    phi = phi.at[-1].set(
        jnp.arctan(alpha[-1] / alpha[-2])
    )  # Last angle is determined by ratio of final two amplitudes
    for i in range(len(phi) - 2, -1, -1):
        phi = phi.at[i].set(
            jnp.arctan(alpha[i + 1] / alpha[i] / jnp.cos(phi[i + 1]))
        )  # Recursively compute previous rotation angles
    return 2 * phi  # Compensate for factor 1/2 in ry


def unary_prep(out_case, phi):
    """
    Prepares the unary-encoded state :math:`\ket{\\text{unary}}` of Chebyshev coefficients.

    The resulting superposition is

    .. math::

        |\\text{unary}\\rangle =
        \\alpha_1|100\\dots00\\rangle + \\alpha_3|110\\dots00\\rangle +
        \\cdots + \\alpha_{2j_0+1}|111\\dots11\\rangle.

    The operation is implemented as a cascade of controlled ``ry`` rotations,
    each conditioned on the activation of previous qubits in the unary chain.

    Parameters
    ----------
    out_case : QuantumVariable
        Unary-encoded state :math:`\ket{\\textt{unary}}`.
    phi : np.ndarray
        Rotation angles :math:`\\vec{\phi}` derived from Chebyshev coefficients.
    """
    x(out_case[0])
    ry(phi[0], out_case[1])
    for i in jrange(1, out_case.size - 1):
        with control(out_case[i]):
            ry(phi[i], out_case[i + 1])


def inner_CKS(A, b, eps, kappa=None, max_beta=None):
    """
    Core circuit implementation of the Childs–Kothari–Somma (CKS) linear
    systems algorithm using the Chebyshev approximation of :math:`1/x`.

    This constructs the circuit for the block encoding (LCU) protocol.

    .. math::

        LCU\ket{0}\ket{\psi}=PREP^{\dagger}SELPREP\ket{0}\ket{\psi}=\\tilde{A}\ket{0}\ket{\psi} ,

    where :math:`PREP` prepares the LCU state and :math:`SEL` applies the
    Chebyshev terms :math:`T_k` controlled on a unary register. Based of the Hamming-weight :math:`k` of :math:`\ket{\\text{unary}}`,
    the polynomial :math:`T_{2k-1}` is block encoded and applied to the circuit.

    The Chebyshev coefficients alternate in sign :math:`(-1)^j\\alpha_j`.
    Since the LCU lemma requires :math:`\\alpha_i>0`, negative terms are
    implemented by conditional Z-gates applied on each index qubit in
    ``out_case``.

    This function supports interchangeable and independent input cases for both
    the matrix ``A`` and vector ``b`` from the QLSP :math:`A\\vec{x} = \\vec{b}`:

    **Case distinctions for A**

        1. **Matrix input** — If ``A`` is a NumPy array (Hermitian matrix),
           the block encoding :math:`SEL` and state-preparation unitary
           :math:`PREP` are constructed internally from ``A`` via Pauli
           decomposition.

        2. **Tuple input** — If ``A`` is a tuple of length 3, the function assumes
            external context provides or manages :math:`\\kappa` directly. In this
            case, ``kappa`` must be specified explicitly or precomputed elsewhere.
            Additionally, the block encoding unitary :math:`U` supplied must satisfy
            the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
            required for the correctness of the Chebyshev polynomial block encoding
            and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.


    **Case distinctions for b**

        1. **Vector input** — If ``b`` is a NumPy array, a new operand QuantumFloat is constructed, and state preparation is performed via
           ``prepare(operand, b)``.

        2. **Callable input** — If ``b`` is a callable, it is assumed to
           prepare the operand state when called. The function invokes it to
           generate the operand QuantumFloat directly via ``operand = b()``.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block encoding
        unitary, state preparation routine, and problem size.
    b : numpy.ndarray or callable
        Vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a tuple rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    operand : QuantumVariable
        Operand QuantumFloat after applying the LCU protocol.
    in_case : QuantumVariable
        Auxiliary QuantumFloat after applying the LCU protocol.
        Must be measured in state $\ket{0}$ for the LCU protocol to be successful.
    out_case : QuantumVariable
        Auxiliary QuantumFloat after applying the LCU protocol.
        Must be measured in state $\ket{0}$ for the LCU protocol to be successful.

    """
    if isinstance(A, tuple) and len(A) == 3:
        U, state_prep, n = A
    else:
        H = QubitOperator.from_matrix(A, reversed=True)
        U, state_prep, n = (H.pauli_block_encoding())  # Construct block encoding of A as a set of Pauli unitaries

    j_0, beta = CKS_parameters(A, eps, kappa, max_beta)
    phi = unary_angles(cheb_coefficients(j_0, beta))

    def RU(in_case, operand):
        """
        Applies one qubitization step :math:`RU` (or its inverse) associated
        with the Chebyshev polynomial block encoding of :math:`A`.

        This operation alternates between application of the block-encoded
        unitary :math:`U` and a reflection about the prepared auxiliary state
        :math:`|G\\rangle_a`, realizing the recursive structure

        .. math::

            T_k(H) = (RU)^k,

        where :math:`R` reflects about the auxiliary state :math:`|G\\rangle_a`.

        The specific implementation of :math:`U` and :math:`R` depends on the
        input case chosen for ``A``:

        1. **Matrix input** — If ``A`` is a NumPy array (Hermitian matrix),
           the block encoding :math:`SEL` and state-preparation unitary
           :math:`PREP` are constructed internally from ``A`` via Pauli
           decomposition.

        2. **Tuple input** — If ``A`` is a tuple of length 3, the function assumes
            external context provides or manages :math:`\\kappa` directly. In this
            case, ``kappa`` must be specified explicitly or precomputed elsewhere.
            Additionally, the block encoding unitary :math:`U` supplied must satisfy
            the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
            required for the correctness of the Chebyshev polynomial block encoding
            and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.

        Parameters
        ----------
        in_case : QuantumVariable
            Inner case QuantumFloat encoding the Chebyshev index :math:`i` or
            similar indicator state used to control block-encoded applications
            of :math:`A`.
        operand : QuantumVariable
            Operand (solution) QuantumFloat upon which the block-encoded matrix
            :math:`A` acts. If ``b`` was provided as a callable, this register
            corresponds to that prepared state directly; otherwise, it
            represents the amplitude-prepared encoding of the classical vector
            :math:`\\vec{b}`.
        """
        U(operand, in_case)
        reflection(in_case, state_function=state_prep)  # reflection operator R about $\ket{G}$.

    out_case = QuantumFloat(j_0)
    in_case = QuantumFloat(n)

    if callable(b):
        operand = b()
    else:
        operand = QuantumFloat(int(np.log2(b.shape[0])))
        prepare(operand, b)

    # Core LCU protocol: PREP, SELECT, PREP^†
    with conjugate(unary_prep)(out_case, phi):
        with conjugate(state_prep)(in_case):
            with control(out_case[0]):
                RU(in_case, operand)
            for i in jrange(1, j_0):
                z(out_case[i])
                with control(out_case[i]):
                    RU(in_case, operand)
                    RU(in_case, operand)

    return operand, in_case, out_case

def inner_CKS_wrapper(qlsp, eps, kappa=None, max_beta=None):
    """
    Wrapper for the full Childs–Kothari–Somma (CKS) algorithm, performing post-selection using the RUS (Repeat-Until-Success) protocol.

    This function calls a supplied problem generator ``qlsp`` to obtain the linear system,
    constructs the required circuit via :func:`inner_CKS`, and returns the solution operand QuantumFloat
    if post-selection on auxiliary QuantumFloats succeeds.

    The algorithm succeeds if both auxiliary QuantumFloats ``in_case`` and ``out_case`` are measured
    in the :math:`|0\\rangle` state. If post-selection fails, the simulation
    restarts until successful as managed by the :func:`RUS` decorator.

    Parameters
    ----------
    qlsp : callable
        Function returning a tuple :math:`(A, \\vec{b})`, where ``A`` is either a Hermitian matrix
        or a 3-tuple with preconstructed block encoding (see :func:`inner_CKS`), and ``b`` is either
        a NumPy array or a callable that prepares a quantum state.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within
        :math:`\epsilon` of :math:`\ket{x}.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a tuple rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    success_bool : bool
        Indicates whether post-selection succeeded. Only when ``success_bool`` is ``True``
        does ``operand`` contain the proper solution state; otherwise, the simulation
        will run again according to the RUS protocol.
    operand : QuantumVariable
        QuantumFloat containing the resulting (approximate) solution state
        :math:`|\\tilde x\\rangle \\propto A^{-1}|b\\rangle` if ``success_bool`` is ``True``.
    """

    A, b = qlsp()

    operand, in_case, out_case = inner_CKS(A, b, eps, kappa, max_beta)

    success_bool = (measure(out_case) == 0) & (measure(in_case) == 0)

    return success_bool, operand


# Apply the RUS decorator with the workaround in order to show in documentation
temp_docstring = inner_CKS_wrapper.__doc__
inner_CKS_wrapper = RUS(static_argnums=[1, 2])(inner_CKS_wrapper)
inner_CKS_wrapper.__doc__ = temp_docstring


def CKS(A, b, eps, kappa=None, max_beta=None):
    """
    Solves the Quantum Linear Systems Problem (QLSP)

        :math:`A \\vec{x} = \\vec{b}`

    using the Chebyshev-based **Childs–Kothari–Somma (CKS)** algorithm.

    This high-level routine integrates all core components of the CKS method:
    Chebyshev polynomial approximation, linear combination of unitaries (LCU),
    qubitization with reflection operators, and the Repeat-Until-Success (RUS) protocol.

    The resulting QuantumFloat approximates the solution state

    .. math::

        |\\tilde{x}\\rangle = A^{-1} |b\\rangle ,

    within target precision :math:`\epsilon` of the ideal solution :math:`|x\rangle`.

    The asymptotic complexity is

    .. math::

        \\mathcal{O}\\!\\left(d \\kappa^2 \\log^2\\!\\frac{d\\kappa}{\\epsilon}\\right),

    providing exponentially better precision scaling compared to HHL.

    This function supports interchangeable and independent input cases for both
    the matrix ``A`` and vector``b`` from the QLSP :math:`A\\vec{x} = \\vec{b}`:

    **Case distinctions for A**

        1. **Matrix input** — If ``A`` is a NumPy array (Hermitian matrix),
           the block encoding :math:`SEL` and state-preparation unitary
           :math:`PREP` are constructed internally from ``A`` via Pauli
           decomposition.

        2. **Tuple input** — If ``A`` is a tuple of length 3, the function assumes
            external context provides or manages :math:`\\kappa` directly. In this
            case, ``kappa`` must be specified explicitly or precomputed elsewhere.
            Additionally, the block encoding unitary :math:`U` supplied must satisfy
            the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
            required for the correctness of the Chebyshev polynomial block encoding
            and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.

    **Case distinctions for b**

        1. **Vector input** — If ``b`` is a NumPy array, a new operand QuantumFloat is constructed, and state preparation is performed via
           ``prepare(operand, b)``.

        2. **Callable input** — If ``b`` is a callable, it is assumed to
           prepare the operand state when called. The function invokes it to
           generate the operand QuantumFloat directly via ``operand = b()``.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block encoding
        unitary, state preparation routine, and problem size.
    b : numpy.ndarray or callable
        Vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a tuple rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    operand : QuantumVariable
        Quantum variable containing the final (approximate) solution state
        :math:`|\\tilde{x}\\rangle \propto A^{-1}|b\\rangle`. When the internal
        :func:`RUS` decorator reports success (``success_bool = True``), this
        variable contains the valid post-selected solution. If ``success_bool``
        is ``False``, the simulation automatically repeats until success.
    """

    def qlsp():
        return A, b

    operand = inner_CKS_wrapper(qlsp, eps, kappa, max_beta)
    return operand
