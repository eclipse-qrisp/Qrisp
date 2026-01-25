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
import scipy as sp
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
    reset
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.jasp import jrange, RUS
from qrisp.operators import QubitOperator
import warnings


def CKS_parameters(A, eps, kappa=None, max_beta=None):
    """
    Computes the Chebyshev-series complexity parameter :math:`\\beta` and the truncation order :math:`j_0` for the
    truncated Chebyshev approximation of :math:`1/x` used in the `Childs–Kothari–Somma quantum algorithm <https://arxiv.org/abs/1511.02306>`_.

    To avoid confusion with the vector :math:`\ket{b}`, we represent the complexity parameter as :math:`\\beta`.

    This function distinguishes two input cases:

    - **Matrix input** 
        If ``A`` is either a NumPy array (Hermitian matrix), or a SciPy Compressed Sparse Row matrix (Hermitian matrix), the
        condition number :math:`\\kappa` is computed internally as :math:`\\kappa = \\mathrm{cond}(A)` if it is not specified.
    - **Block-encoding input** 
        If ``A`` is a 3-tuple representing a block-encoding, ``kappa`` must be specified.

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
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        representing a preconstructed block-encoding.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    j_0 : int
        Truncation order of the Chebyshev expansion
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    beta : float
        Complexity parameter :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.
    """
    if kappa != None:
        # kappa is specified by the user
        pass 
    elif isinstance(A, tuple): # block-encoding
        raise Exception("Condition number must be specified if A is a block-encoding")
    else: # matrix
        from scipy.sparse import csr_matrix
        from numpy import ndarray

        if isinstance(A, ndarray):
            kappa = np.linalg.cond(A)
        elif isinstance(A, csr_matrix):
            lam_max = sp.sparse.linalg.eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
            lam_min = sp.sparse.linalg.eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
            kappa = float(abs(lam_max) / abs(lam_min))
        else:
            raise TypeError(f"Unsupported type {type(A)} for the matrix A")

    if max_beta == None:
        beta = kappa**2 * np.log(kappa / eps)
    else:
        beta = min(kappa**2 * np.log(kappa / eps), max_beta)
    j_0 = np.sqrt(beta * np.log(4 * beta / eps))
    return int(j_0), int(beta)


def cheb_coefficients(j0, b):
    """
    Calculates the positive coefficients :math:`\\alpha_i` for the truncated
    Chebyshev expansion of :math:`1/x` up to order :math:`2j_0+1`, as described in the `Childs–Kothari–Somma paper <https://arxiv.org/abs/1511.02306>`_.

    The approximation is expressed as a linear combination
    of odd Chebyshev polynomials truncated at index :math:`j_0` (Lemma 14):

    .. math::

        g(x) = 4 \\sum_{j=0}^{j_0} (-1)^j
        \\left[ \\sum_{i=j+1}^{b} \\frac{\\binom{2b}{b+i}}{2^{2b}} \\right]
        T_{2j+1}(x)

    The Linear Combination of Unitaries (LCU) lemma requires strictly
    positive coefficients :math:`\\alpha_i > 0`, their absolute values are
    used. The alternating factor :math:`(-1)^j` is later implemented as a set of
    Z-gates within the CKS circuit (see :func:`inner_CKS`).

    Parameters
    ----------
    j0 : int
        Truncation order of the Chebyshev expansion,
        :math:`j_0 = \\lfloor\sqrt{\\beta \log(4\\beta/\epsilon)}\\rfloor`.
    b : float
        Complexity parameter, :math:`\\beta = \\lfloor\kappa^2 \log(\kappa/\epsilon)\\rfloor`.

    Returns
    -------
    coeffs : numpy.ndarray
        Array of positive Chebyshev coefficients
        :math:`{\\alpha_{2j+1}}`, corresponding to the odd degrees of Chebyshev polynomials of the first kind :math:`T_1, T_3, \\dots, T_{2j_0+1}`.
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
    corresponding to the square-root-amplitude encoding of the Chebyshev coefficients.

    The prepared unary state is

    .. math::

        \ket{\\text{unary}} \propto
        \\sqrt{\\alpha_1}\ket{100\\dots00} + \\sqrt{\\alpha_3}\ket{110\\dots00} +
        \\cdots + \\sqrt{\\alpha_{2j_0+1}}\ket{111\\dots11}

    Parameters
    ----------
    coeffs : numpy.ndarray
        Positive Chebyshev coefficients :math:`\\alpha_i`.

    Returns
    -------
    phi : numpy.ndarray
        Rotation angles :math:`\\phi_i` for unary state preparation.
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


def unary_prep(case, coeffs):
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
    coeffs : numpy.ndarray
        An array of :math:`j_0` Chebyshev coefficients :math:`\\alpha_1,\\alpha_3,\dotsc,\\alpha_{2j_0+1}`.
    
    """
    phi = unary_angles(coeffs)

    x(case[0])
    ry(phi[0], case[1])
    for i in jrange(1, case.size - 1):
        with control(case[i]):
            ry(phi[i], case[i + 1])


def inner_CKS(A, b, eps, kappa=None, max_beta=None):
    """
    Core implementation of the `Childs–Kothari–Somma (CKS) quantum algorithm <https://arxiv.org/abs/1511.02306>`_. 
    
    This function integrates core components of the CKS approach to construct the circuit:
    Chebyshev polynomial approximation, linear combination of unitaries (LCU), and
    `qubitization with reflection operators <https://arxiv.org/abs/2208.00567>`_. This function does
    not include the Repeat-Until-Success (RUS) protocol and thus serves as a subroutine
    for generating the underlying circuit of the CKS algorithm.

    The semantics of the approach can be illustrated with the following circuit schematics:

    .. image:: /_static/CKS_circuit.png
       :align: center

    Implementation overview:
      1. Compute the CKS parameters :math:`j_0` and :math:`\\beta` (:func:`CKS_parameters`).
      2. Generate Chebyshev coefficients and unary state angles
         (:func:`cheb_coefficients`, :func:`unary_prep`).
      3. Build the core LCU structure and qubitization operator (:func:`inner_CKS`).

    The goal of this algorithm is to apply the non-unitary operator :math:`A^{-1}` to the
    input state :math:`\ket{b}`. Following the Chebyshev approach introduced in the CKS paper, we express :math:`A^{-1}` as
    a linear combination of odd Chebyshev polynomials: 

    .. math::
    
        A^{-1}\propto\sum_{j=0}^{j_0}\\alpha_{2j+1}T_{2j+1}(A),

    where :math:`T_k(A)` are Chebyshev polynomials of the first kind and :math:`\\alpha_{2j+1} > 0` are computed
    via :func:`cheb_coefficients`. These operators can be efficiently implemented with qubitization, which relies on a unitary
    block encoding :math:`U` of the matrix :math:`A`, and a :ref:`reflection operator <reflection>` :math:`R`.
 
    .. note::

        Qubitization requires that the block encoding unitary :math:`U` is self-inverse (:math:`U^2=I`).

    The fundamental iteration step is defined as :math:`(RU)`, where :math:`R` reflects
    around the auxiliary block-encoding state :math:`\ket{G}`, prepared as the ``inner_case`` QuantumFloat.

    Repeated applications of these unitaries, :math:`(RU)^k`, yield a block encoding of the :math:`k`-th Chebyshev polynomial 
    of the first kind :math:`T_k(A)`.
    
    We then construct a linear combination of these block-encoded polynomials using the LCU structure 

    .. math::
    
        LCU\ket{0}\ket{\psi}=PREP^{\dagger}\cdot SEL\cdot PREP\ket{0}\ket{\psi}=\\tilde{A}\ket{0}\ket{\psi}.
    
    Here, the :math:`PREP` operation prepares an auxiliary ``out_case`` Quantumfloat in the unary state :math:`\ket{\\text{unary}}`
    that encodes the square root of the Chebyshev coefficients :math:`\sqrt{\\alpha_j}`. The :math:`SEL` operation selects and applies the
    appropriate Chebyshev polynomial operator :math:`T_k(A)`, implemented by :math:`(RU)^k`, controlled on ``out_case`` in the unary
    state :math:`\ket{\\text{unary}}`. Based on the Hamming-weight :math:`k` of :math:`\ket{\\text{unary}}`,
    the polynomial :math:`T_{2k-1}` is block encoded and applied to the circuit. 
    
    To construct a linear combination of Chebyshev polynomials up to the :math:`2j_0+1`-th order, as in the original paper, 
    our implementation requires :math:`j_0+1` qubits in the ``out_case`` state :math:`\ket{\\text{unary}}`.

    The Chebyshev coefficients alternate in sign :math:`(-1)^j\\alpha_j`.
    Since the LCU lemma requires :math:`\\alpha_j>0`, negative terms are accounted for
    by applying Z-gates on each index qubit in ``out_case``.

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
            Callable ``U(case, operand)`` applies the block-encoding unitary :math:`SEL`.
        * state_prep : callable
            Callable ``state_prep(case)`` applies the block-encoding state preparatiion unitary :math:`PREP`
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
        If ``b`` is a NumPy array, a new ``operand`` QuantumFloat is constructed, and state preparation is performed via
        ``prepare(operand, b)``.

    - Callable input:
        If ``b`` is a callable, it is assumed to prepare the operand state when called. 
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
        Operand variable after applying the LCU protocol.
    in_case : QuantumFloat
        Auxiliary variable after applying the LCU protocol.
        Must be measured in state $\ket{0}$ for the LCU protocol to be successful.
    out_case : QuantumFloat
        Auxiliary variable after applying the LCU protocol.
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
    from qrisp.operators.block_encoding import BlockEncoding

    if isinstance(A, BlockEncoding):
        BE = A
    elif isinstance(A, tuple) and len(A) == 3:
        warnings.warn("The interface (U, state_prep, n) is deprecated. Please use a BlockEncoding object instead.")
        
        U, state_prep, n = A
        def encoding_unitary(case, operand):
            with conjugate(state_prep)(case):
                U(case, operand)
        BE = BlockEncoding(encoding_unitary, [QuantumFloat(n).template()], 1.0)
    else:
        H = QubitOperator.from_matrix(A, reverse_endianness=True)
        BE = H.pauli_block_encoding() # Construct block encoding of A as a set of Pauli unitaries

    j_0, beta = CKS_parameters(A, eps, kappa, max_beta)
    cheb_coeffs = cheb_coefficients(j_0, beta)

    def RU(cases, operand):
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
        BE.unitary(*cases, operand)
        reflection(cases, state_function=lambda x : None)  # reflection operator R about $\ket{0}$.

    out_case = QuantumFloat(j_0 + 1)
    in_case_list = BE.create_ancillas()

    if callable(b):
        operand = b()
    else:
        operand = QuantumFloat(int(np.log2(b.shape[0])))
        prepare(operand, b)

    # Core LCU protocol: PREP, SELECT, PREP^†
    with conjugate(unary_prep)(out_case, cheb_coeffs):
        with control(out_case[0]):
            RU(in_case_list, operand)
        for i in jrange(1, j_0 + 1):
            z(out_case[i])
            with control(out_case[i]):
                RU(in_case_list, operand)
                RU(in_case_list, operand)

    return operand, in_case_list[0], out_case

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
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.
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

    vars = inner_CKS(A, b, eps, kappa, max_beta)
    operand = vars[0]
    ancillas = vars[1:]

    bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
    success_bool = jnp.all(bools)

    # garbage collection
    [reset(anc) for anc in ancillas]
    [anc.delete() for anc in ancillas]

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
    :math:`\\mathcal{O}\\!\\left(\\log(N)s \\kappa^2 \\text{polylog}\\!\\frac{s\\kappa}{\\epsilon}\\right)`, where :math:`N` is the matrix size, :math:`s` its sparsity, and 
    :math:`\kappa` its condition number. This represents an exponentially
    better precision scaling compared to the HHL algorithm.

    This function integrates all core components of the CKS approach:
    Chebyshev polynomial approximation, linear combination of unitaries (LCU),
    `qubitization with reflection operators <https://arxiv.org/abs/2208.00567>`_, and the :ref:`Repeat-Until-Success (RUS) <RUS>` protocol.
    The semantics of the approach can be illustrated with the following circuit schematics:

    .. image:: /_static/CKS_schematics.png
       :align: center

    Implementation overview:
      1. Compute the CKS parameters :math:`j_0` and :math:`\\beta` (:func:`CKS_parameters`).
      2. Generate Chebyshev coefficients and the auxiliary unary state (:func:`cheb_coefficients`, :func:`unary_prep`).
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
            Callable ``U(operand, case)`` applies the block-encoding unitary :math:`SEL`.
        * state_prep : callable
            Callable ``state_prep(case)`` applies the block-encoding state preparatiion unitary :math:`PREP`
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
        If ``b`` is a NumPy array, a new ``operand`` QuantumFloat is constructed, and state preparation is performed via
        ``prepare(operand, b)``.

    - Callable input:
        If ``b`` is a callable, it is assumed to prepare the operand state when called. 
        The ``operand`` QuantumFloat is generated directly via ``operand = b()``.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block-encoding.
    b : numpy.ndarray or callable
        Either a vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state ``operand``.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.
    max_beta : float, optional
        Optional upper bound on the complexity parameter :math:`\\beta`.

    Returns
    -------
    operand : QuantumVariable
        Quantum variable containing the final (approximate) solution state
        :math:`\ket{\\tilde{x}} \propto A^{-1}\ket{b}`. When the internal
        :ref:`RUS <RUS>` decorator reports success (``success_bool = True``), this
        variable contains the valid post-selected solution. If ``success_bool``
        is ``False``, the simulation automatically repeats until success.
        
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

    Next, we solve this linear system using the CKS quantum algorithm:

    ::

        from qrisp.algorithms.cks import CKS
        from qrisp.jasp import terminal_sampling

        @terminal_sampling
        def main():

            x = CKS(A, b, 0.01)
            return x

        res_dict = main()

    The resulting dictionary contains the measurement probabilities for each computational basis state.
    To extract the corresponding quantum amplitudes (up to sign):

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

    We see that we obtained the same result in our quantum simulation up to precision :math:`\epsilon`!

    To perform quantum resource estimation, replace the ``@terminal_sampling``
    decorator with ``@count_ops(meas_behavior="0")``:

    ::
        
        from qrisp.jasp import count_ops
    
        @count_ops(meas_behavior="0")
        def main():

            x = CKS(A, b, 0.01)
            return x

        res_dict = main()
        print(res_dict)
        # {'cx': 3628, 't': 1288, 't_dg': 1702, 'p': 178, 'cz': 138, 'cy': 46, 'h': 960, 'x': 324, 's': 66, 's_dg': 66, 'u3': 723, 'gphase': 49, 'ry': 2, 'z': 11, 'measure': 16}

    The printed dictionary lists the estimated quantum gate counts required for the CKS algorithm. Since the simulation itself is not executed, 
    this approach enables scalable gate count estimation for linear systems of arbitrary size.
    
    **Example 2: Using a custom block encoding**

    The previous example displays how to solve the linear system for any Hermitian matrix :math:`A`,
    where the block-encoding is constructed from the Pauli decomposition of :math:`A`. While this approach is efficient when :math:`A`
    corresponds to, e.g., an Ising or a Heisenberg Hamiltonian, the number of Pauli terms may not scale favorably in general.
    For certain sparse matrices, their structure can be exploited to construct more efficient block-encodings.

    In this example, we construct a block-encoding representation for the following tridiagonal sparse matrix:

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

    Additionally, the block encoding unitary :math:`U` supplied must satisfy the property :math:`U^2 = I`, i.e., it is self-inverse.
    This condition is required for the correctness of the Chebyshev polynomial block encoding
    and qubitization step. Further details can be found `here <https://arxiv.org/abs/2208.00567>`_. In this case, the fact that $V^2=(V^{\dagger})^2=I$
    ensures that defining the block encoding unitary via a :ref:`quantum switch case <qswitch>` satsifies :math:`U^2 = I`.

    We now define the block_encoding ``(U, state_prep, n)``:

    ::

        coeffs = np.array([5,1,1])
        alpha = np.sum(coeffs)

        def U(case, operand):
            qswitch(operand, case, unitaries)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs/alpha))

        block_encoding = (U, state_prep, 2)

    We solve the linear system by passing this block-encoding tuple as ``A`` into the CKS function:

    ::

        @terminal_sampling
        def main():

            x = CKS(block_encoding, b, 0.01, np.linalg.cond(A))
            return x

        res_dict = main()

    We, again, compare the solution obtained by quantum simulation with the classical solution

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

    Quantum resource estimation can be performed in the same way as in the first example:

    ::

        @count_ops(meas_behavior="0")
        def main():

            x = CKS(block_encoding, b, 0.01, np.linalg.cond(A))
            return x
    
        res_dict = main()
        print(res_dict)
        # {'h': 676, 't_dg': 806, 'cx': 1984, 't': 651, 's': 152, 's_dg': 90, 'u3': 199, 'gphase': 65, 'p': 149, 'z': 15, 'x': 126, 'measure': 142, 'ry': 2}

    Finally, note that it is also possible to solve a linear system without using the :ref:`RUS` decorator,
    by performing the necessary post-selection manually. Such an example can be found in :func:`inner_CKS`.

    """

    def qlsp():
        return A, b

    operand = inner_CKS_wrapper(qlsp, eps, kappa, max_beta)
    return operand
