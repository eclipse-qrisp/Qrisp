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

    1. **Matrix input** — If ``A`` is either a NumPy array (Hermitian matrix), or a SciPy sparse CSR matrix (Hermitian matrix),the
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

        \ket{\\text{unary}} =
        \\alpha_1\ket{100\\dots00} + \\alpha_3\ket{110\\dots00} +
        \\cdots + \\alpha_{2j_0+1}\ket{111\\dots11}

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

        \ket{\\text{unary}} =
        \\alpha_1\ket{100\\dots00} + \\alpha_3\ket{110\\dots00} +
        \\cdots + \\alpha_{2j_0+1}\ket{111\\dots11}.

    The operation is implemented as a cascade of controlled ``ry`` rotations,
    each conditioned on the activation of previous qubits in the unary chain.

    Parameters
    ----------
    out_case : QuantumFloat
        Empty QuantumFloat with $j_0$ qubits on which the unary-encoded state :math:`\ket{\\text{unary}}` will be performed.
    phi : np.ndarray
        Rotation angles :math:`\\vec{\phi}` derived from Chebyshev coefficients.
    
    Returns
    -------
    out_case : QuantumFloat
        Unary-encoded state :math:`\ket{\\text{unary}}`.
    """
    x(out_case[0])
    ry(phi[0], out_case[1])
    for i in jrange(1, out_case.size - 1):
        with control(out_case[i]):
            ry(phi[i], out_case[i + 1])


def inner_CKS(A, b, eps, kappa=None, max_beta=None):
    """
    Core implementation of the CKS algorithm. 
    
    This function integrates core components of the CKS approach to construct the circuit:
    Chebyshev polynomial approximation, linear combination of unitaries (LCU), and
    `qubitization with reflection operators <https://arxiv.org/abs/2208.00567>`_ and the :ref:`RUS` (RUS) protocol.
    The semantics of the approach can be illustrated with the following circuit schematics:

    .. image:: /_static/CKS_circuit.png
       :align: center

    Implementation overview:
      1. Compute the CKS parameters :math:`j_0` and :math:`\\beta` (:func:`CKS_parameters`).
      2. Generate Chebyshev coefficients and unary state angles
         (:func:`cheb_coefficients`, :func:`unary_angles`, :func:`unary_prep`).
      3. Build the core LCU structure and qubitization operator (:func:`inner_CKS`).

    This function constructs the routine for the block-encoding (LCU) protocol

    .. math::

        LCU\ket{0}\ket{\psi}=PREP^{\dagger}\cdot SEL\cdot PREP\ket{0}\ket{\psi}=\\tilde{A}\ket{0}\ket{\psi}

    where :math:`PREP` prepares the LCU state and :math:`SEL` applies the
    Chebyshev terms constructed with $k$ qubitization steps :math:`T_k=(RU)^k` controlled on the qubits of a QuantumVariable in a unary-encoded state :math:`\ket{\\text{unary}}`. 
    Based of the Hamming-weight :math:`k` of :math:`\ket{\\text{unary}}`,
    the polynomial :math:`T_{2k-1}` is block encoded and applied to the circuit.

    The Chebyshev coefficients alternate in sign :math:`(-1)^j\\alpha_j`.
    Since the LCU lemma requires :math:`\\alpha_i>0`, negative terms are
    implemented by Z-gates applied on each index qubit in ``out_case``.

    This function supports interchangeable and independent input cases for both
    the matrix ``A`` and vector ``b`` from the QLSP :math:`A\\vec{x} = \\vec{b}`:

    **Case distinctions for A**

    - Matrix input: 
        ``A`` is either a NumPy array (Hermitian matrix), or SciPy Compressed Sparse Row matrix (Hermitian matrix).
        The block encoding unitary :math:`SEL` and state preparation unitary
        :math:`PREP` are constructed internally from ``A`` via
        :meth:`Pauli decomposition <qrisp.operators.qubit.QubitOperator.pauli_block_encoding>`.

    - Block-encoding input: 
        ``A`` is a 3-tuple ``(U, state_prep, n)`` representing a block-encoding: 
        
        * U : callable
            A callable ``U(operand, case)`` applying the block-encoding unitary :math:`SEL`.
        * state_prep : callable
            A callable ``state_prep(case)`` applying the block-encoding state preparatiion unitary :math:`PREP`
            to an auxiliary ``case`` QuantumVariable .
        * n : int
            The size of the auxiliary ``case`` QuantumVariable.
            
        In this case, (an upper bound for) the condition number ``kappa`` must be specified.
        Additionally, the block-encoding unitary :math:`U` supplied must satisfy
        the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
        required for the correctness of the Chebyshev polynomial block encoding
        and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.

    **Case distinctions for b**

    - Vector input: 
        if ``b`` is a NumPy array, a new ``operand`` QuantumFloat is constructed, and state preparation is performed via
        ``prepare(operand, b)``.

    - Callable input:
        if ``b`` is a callable, it is assumed to prepare the operand state when called. 
        The ``operand`` QuantumFloat is generated directly via ``operand = b()``.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block-encoding.
    b : numpy.ndarray or callable
        Vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a tuple (block-encoding) rather than a matrix.
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
        
    Examples
    --------

    The example shows how to solve a QLSP using the ``inner_CKS`` function directly,
    without employing the :ref:`RUS` protocol in Qrisp.

    First, define a Hermitian matrix :math:`A` and a right-hand side vector :math:`\\vec{b}`

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                      [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                      [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                      [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

    Then, constuct the CKS circuit and perform a multi measurement: 

    ::

        from qrisp.algorithms.cks import inner_CKS
        from qrisp import multi_measurement

        operand, in_case, out_case = inner_CKS(A, b, 0.001)
        res_dict = multi_measurement([operand, in_case, out_case])

    This performs the measurement on all three of our QuantumVariables ``out_case``, ``in_case``, and ``operand``.
    Since the CKS (and all other LCU based approaches) is correctly performed only when
    auxiliary QuantumVariables are measured in :math:`\ket{0}`, we need
    to construct a new dictionary, and collect the measurements when this is true.

    ::

        new_dict = dict()
        success_prob = 0

        for key, prob in res_dict.items():
            if key[1]==0 and key[2]==0:
                new_dict[key[0]] = prob
                success_prob += prob

        for key in new_dict.keys():
            new_dict[key] = new_dict[key]/success_prob

        for k, v in new_dict.items():
            new_dict[k] = v**0.5
    
    Finally, compare the quantum simulation result with the classical solution:

    ::

        q = np.array([new_dict.get(key, 0) for key in range(len(b))])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\\n", q, "\\nCLASSICAL SOLUTION\\n", c)
        # QUANTUM SIMULATION
        # [0.0289281  0.55469649 0.53006773 0.64070521] 
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    This approach enables execution of the compiled CKS circuit on :ref:`compatible quantum
    hardware or simulators <BackendInterface>`.  

    """
    if isinstance(A, tuple) and len(A) == 3:
        U, state_prep, n = A
    else:
        H = QubitOperator.from_matrix(A, reversed=True)
        U, state_prep, n = (H.pauli_block_encoding())  # Construct block encoding of A as a set of Pauli unitaries

    j_0, beta = CKS_parameters(A, eps, kappa, max_beta)
    phi = unary_angles(cheb_coefficients(j_0, beta))

    def RU(case, operand):
        """
        Applies one qubitization step :math:`RU` (or its inverse) associated
        with the Chebyshev polynomial block-encoding of an operator :math:`A`.

        This operation alternates between application of the block-encoding
        unitary :math:`U` and a reflection about the block-encoding state
        :math:`\ket{G}`, realizing

        .. math::

            T_k(A) = (RU)^k,

        where :math:`R` reflects about the block-encoding state :math:`\ket{G}`,
        and :math:`T_k(A)` is the Chebyshev polynomials of the first kind of degree :math:`k`.

        Parameters
        ----------
        case : QuantumVariable
            Auxiliary case variable encoding the block-encoding state :math:`\ket{G}`.
        operand : QuantumVariable
            Operand (solution) variable upon which the block-encoded matrix
            :math:`A` acts. 

        """
        U(operand, case)
        reflection(case, state_function=state_prep)  # reflection operator R about $\ket{G}$.

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
    in the :math:`\ket{0}` state. If post-selection fails, the simulation
    restarts until successful as managed by the :func:`RUS` decorator.

    Parameters
    ----------
    qlsp : callable
        Function returning a tuple :math:`(A, \\vec{b})`, where ``A`` is either a Hermitian matrix
        or a 3-tuple with preconstructed block-encoding (see :func:`inner_CKS`), and ``b`` is either
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
        :math:`\ket{\\tilde x} \\propto A^{-1}\ket{b}` if ``success_bool`` is ``True``.
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
    Performs the `Childs–Kothari–Somma (CKS) quantum algorithm <https://arxiv.org/abs/1511.02306>`_ to solve the Quantum Linear Systems Problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}`, using the Chebyshev approximation of :math:`1/x`. 
    
    The algorithm prepares a quantum state :math:`\ket{x} \propto A^{-1} \ket{b}`
    within target precision :math:`\epsilon` of the ideal solution :math:`\ket{x}`. 
    
    The asymptotic complexity is
    :math:`\\mathcal{O}\\!\\left(\\log(N)d \\kappa^2 \\text{polylog}\\!\\frac{d\\kappa}{\\epsilon}\\right)` where :math:`d` is the sparsity of the matrix, and 
    :math:`\kappa` is the condition number of the matrix. This represents an exponentially
    better precision scaling compared to the HHL algorithm.

    This function integrates all core components of the CKS approach:
    Chebyshev polynomial approximation, linear combination of unitaries (LCU),
    `qubitization with reflection operators <https://arxiv.org/abs/2208.00567>`_, and the Repeat-Until-Success (RUS) protocol.
    The semantics of the approach can be illustrated with the following circuit schematics:

    .. image:: /_static/CKS_schematics.png
       :align: center

    Implementation overview:
      1. Compute the CKS parameters :math:`j_0` and :math:`\\beta` (:func:`CKS_parameters`).
      2. Generate Chebyshev coefficients and unary state angles
         (:func:`cheb_coefficients`, :func:`unary_angles`, :func:`unary_prep`).
      3. Build the core LCU structure and qubitization operator (:func:`inner_CKS`).
      4. Apply the RUS post-selection step to project onto the valid success outcome.

    This function supports interchangeable and independent input types for both
    the matrix ``A`` and vector ``b`` from the QLSP :math:`A\\vec{x} = \\vec{b}`.

    **Case distinctions for A**

    - Matrix input: 
        ``A`` is either a NumPy array (Hermitian matrix), or SciPy Compressed Sparse Row matrix (Hermitian matrix).
        The block encoding unitary :math:`SEL` and state preparation unitary
        :math:`PREP` are constructed internally from ``A`` via
        :meth:`Pauli decomposition <qrisp.operators.qubit.QubitOperator.pauli_block_encoding>`.

    - Block-encoding input: 
        ``A`` is a 3-tuple ``(U, state_prep, n)`` representing a block-encoding: 
        
        * U : callable
            A callable ``U(operand, case)`` applying the block-encoding unitary :math:`SEL`.
        * state_prep : callable
            A callable ``state_prep(case)`` applying the block-encoding state preparatiion unitary :math:`PREP`
            to an auxiliary ``case`` QuantumVariable .
        * n : int
            The size of the auxiliary ``case`` QuantumVariable.
            
        In this case, (an upper bound for) the condition number ``kappa`` must be specified.
        Additionally, the block-encoding unitary :math:`U` supplied must satisfy
        the property :math:`U^2 = I`, i.e., it is self-inverse. This condition is
        required for the correctness of the Chebyshev polynomial block encoding
        and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_.

    **Case distinctions for b**

    - Vector input: 
        if ``b`` is a NumPy array, a new ``operand`` QuantumFloat is constructed, and state preparation is performed via
        ``prepare(operand, b)``.

    - Callable input:
        if ``b`` is a callable, it is assumed to prepare the operand state when called. 
        The ``operand`` QuantumFloat is generated directly via ``operand = b()``.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block-encoding.
    b : numpy.ndarray or callable
        Vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a tuple (block-encoding) rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    operand : QuantumVariable
        Quantum variable containing the final (approximate) solution state
        :math:`\ket{\\tilde{x}} \propto A^{-1}\ket{b}`. When the internal
        :func:`RUS` decorator reports success (``success_bool = True``), this
        variable contains the valid post-selected solution. If ``success_bool``
        is ``False``, the simulation automatically repeats until success.
        
    Examples
    --------

    The following examples demonstrate how to use the CKS algorithm to solve the
    quantum linear systems problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}` using both a direct Hermitian matrix input and
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

    Next, we solve the linear system with the CKS quantum algorithm:

    ::

        from qrisp.algorithms.cks import CKS
        from qrisp.jasp import terminal_sampling

        @terminal_sampling
        def main():

            x = CKS(A, b, 0.01)
            return x

        res_dict = main()

    The measured dictionary contains probabilities corresponding to each computational basis state.
    To extract the quantum amplitudes (up to sign):

    ::

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        q = np.array([res_dict.get(key, 0) for key in range(len(b))])
        print("QUANTUM SIMULATION\\n", q)
        # QUANTUM SIMULATION
        # [0.02714082 0.55709921 0.53035395 0.63845794]
    
    We compare this to the classical normalized solution:

    ::

        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("CLASSICAL SOLUTION\\n", c)  
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    We see that we obtained the same result obsered in the quantum simulation up to precision :math:`\epsilon`!

    To perform quantum resource estimation, replace the ``@terminal_sampling``
    decorator with the ``@count_ops(meas_behavior="0")`` decorator:

    ::
        
        from qrisp.jasp import count_ops
    
        @count_ops(meas_behavior="0")
        def main():

            x = CKS(A, b, 0.01)
            return x

        res_dict = main()
        print(res_dict)
        # {'cx': 3628, 't': 1288, 't_dg': 1702, 'p': 178, 'cz': 138, 'cy': 46, 'h': 960, 'x': 324, 's': 66, 's_dg': 66, 'u3': 723, 'gphase': 49, 'ry': 2, 'z': 11, 'measure': 16}

    The printed dictionary lists the estimated quantum gate counts for the full simulation. Since this approach doesn't run
    the simulation itself, one can estimate the quantum gate counts for arbitrary linear system problem sizes.
    
    **Example 2: Using a custom block encoding**

    The previous example displays how to solve the linear system for any Hermitian matrix :math:`A`,
    where the block-encoding is constructed from :math:`A` via Pauli decomposition. While this approach is efficient when :math:`A`
    corresponds to, e.g., an Ising od a Heisenberg Hamiltonian, in gerneral the number of Pauli terms may not scale favorably, and
    for certain sparse matrices their structure can be expoited to construct more efficient block-encodings.

    In this example, we construct a block-encoding representation for a tridiagonal 3-sparse matrix defined by

    ::

        import numpy as np

        def tridiagonal_shifted(n, mu=1.0, dtype=float):
            I = np.eye(n, dtype=dtype)
            return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)

        N = 8
        A = tridiagonal_shifted(N, mu=3)
        b = np.random.randint(0, 2, size=N)

    This matrix can be decomposed using three unitaries: the identity $I$, and two shift operators $V\colon\ket{k}\\rightarrow-\ket{k+N/2 \mod N}$ and $V^{\dagger}\colon\ket{k}\\rightarrow-\ket{k-N/2 \mod N}$.
    We define their corresponding functions:

    ::

        from qrisp import gphase, prepare, qswitch

        def I(qv):
            pass

        def V(qv):
            qv += N//2
            gphase(np.pi, qv[0])

        def V_dg(qv):
            qv -= N//2
            gphase(np.pi, qv[0])

        unitaries = [I, V, V_dg]

    Additionally, the block encoding unitary :math:`U` supplied must satisfy the property :math:`U^2 = I`, i.e., it is self-inverse. In this case, 
    This condition is required for the correctness of the Chebyshev polynomial block encoding
    and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_. In this case, the fact that $V^2=(V^{\dagger})^2=I$
    ensures that defining the block encoding unitary via a :ref:`quantum switch case <qswitch>` satsifies :math:`U^2 = I`.

    We now define the block_encoding ``(U, state_prep, n)``:

    ::

        coeffs = np.array([5,1,1])
        alpha = np.sum(coeffs)

        def U(operand, case):
            qswitch(operand, case, unitaries)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs/alpha))

        block_encoding = (U, state_prep, 2)

    We now solve the problem by passing the block-encoding tuple defined above as ``A`` in the CKS function

    ::

        @terminal_sampling
        def main():

            x = CKS(block_encoding, b, 0.01, np.linalg.cond(A))
            return x

        res_dict = main()

    We, again, compare the solution we obtain by quantum simulation with the classical solution

    ::

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        q = np.array([res_dict.get(key, 0) for key in range(N)])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\\n", q, "\\nCLASSICAL SOLUTION\\n", c)
        # QUANTUM SIMULATION
        # [0.43917486 0.31395935 0.12522139 0.43917505 0.43917459 0.12522104 0.31395943 0.43917502]
        # CLASSICAL SOLUTION
        # [0.43921906 0.3137279  0.12549116 0.43921906 0.43921906 0.12549116 0.3137279  0.43921906]

    Performing quantum resource estimation can be performed in the same manner as in the first example:

    ::

        @count_ops(meas_behavior="0")
        def main():

            x = CKS(block_encoding, b, 0.01, np.linalg.cond(A))
            return x
    
        res_dict = main()
        print(res_dict)
        # {'h': 676, 't_dg': 806, 'cx': 1984, 't': 651, 's': 152, 's_dg': 90, 'u3': 199, 'gphase': 65, 'p': 149, 'z': 15, 'x': 126, 'measure': 142, 'ry': 2}

    Important to note, that one can also solve a linear system without calling the :ref:`RUS` decorator,
    and perform the necessary post-selection by hand. Such an example can be found in :func:`inner_CKS`.

    """

    def qlsp():
        return A, b

    operand = inner_CKS_wrapper(qlsp, eps, kappa, max_beta)
    return operand
