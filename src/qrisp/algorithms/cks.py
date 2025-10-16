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
import jax.numpy as jnp
from scipy.special import comb
from qrisp import *
from qrisp.operators import *


def CKS_parameters(A, eps, max_beta = None):
    """
    Computes the Chebyshev complexity parameter :math:`b` and the truncation order :math:´j_0´ for the
    truncated Chebyshev series approximation of :math:`\\frac{1}{x}`, used in the
    Childs–Kothari–Somma (CKS) algorithm. To avoid confusion with the vector :math:`\ket{b}`, we represent the complexity parameter as :math:`\\beta`.

    Parameters, :math:`(j_0, \\beta)` determine both precision and
    gate complexity of the Chebyshev CKS approach.

    .. math::

        \\beta = \kappa^2 \log\!\left(\\frac{\kappa}{\epsilon}\\right),
        \quad
        j_0 = \sqrt{\\beta \log\!\left(\\frac{4\\beta}{\epsilon}\\right)},

    where :math:`\kappa` denotes the condition number of :math:`A`. 

    It is also possible for the user to manually set the maximal complexity parameter :math:`\\beta_{\\text{max}}` as a
    keyword argument. 

    Parameters
    ----------
    A : np.ndarray
        The :math:`N \\times N` Hermitian matrix :math:`A` in the linear system
        :math:`A\\vec{x} = \\vec{b}`.
    eps : float
        Desired precision :math:`\epsilon`, such that
        :math:`\|\,|\\tilde x\\rangle - |x\\rangle\,\| \leq \epsilon`.
    max_beta : float, optional
        Optional upper bound on :math:`\\beta`. If provided,
        :math:`\\beta` is capped as the minimum of the provided :math:`\\beta_{\\text{max}}` and 
        calculated values.

    Returns
    -------
    j_0 : int
        Truncation order of the Chebyshev expansion,
        :math:`j_0 = \sqrt{\\beta \log(4\\beta/\epsilon)}`.
    beta : float
        Complexity parameter :math:`\\beta = \kappa^2 \log(\kappa/\epsilon)`.
    """
    kappa = np.linalg.cond(A)
    # Determine complexity parameter beta with optional cap
    if max_beta == None:
        beta = kappa ** 2 * np.log(kappa / eps)
    else:
        beta = min(kappa ** 2 * np.log(kappa / eps), max_beta)
    j_0 = np.sqrt(beta * np.log(4 * beta / eps))
    return int(j_0), int(beta)

def cheb_coefficients(j0, b):
    """
    Calculates the positive coefficients :math:`\\alpha_i` for the truncated
    Chebyshev expansion used in approximating :math:`\frac{1}{x}`.

    The approximation function :math:`g(x)` is expressed as a linear combination
    of odd Chebyshev polynomials :math:`T_{2j+1}(x)` truncated at index
    :math:`j_0`:

    .. math::

        g(x) = 4 \\sum_{j=0}^{j_0} (-1)^j
        \\left[ \\sum_{i=j+1}^{b} \\frac{\\binom{2b}{b+i}}{2^{2b}} \\right]
        T_{2j+1}(x)

    Since the Linear Combination of Unitaries (LCU) framework requires strictly
    positive coefficients :math:`\\alpha_i > 0`, their absolute values are
    used. The alternating factor :math:`(-1)^j` is later implemented as a set of
    conditional Z-phase gates within the Chebyshev circuit (see
    :func:`inner_CKS`).

    Parameters
    ----------
    j0 : int
        Truncation order :math:`j_0` of the Chebyshev expansion.
    b : float
        Complexity parameter :math:`\\beta`.

    Returns
    -------
    coeffs : np.ndarray
        Array of coefficients
        :math:`{\\alpha_1, \\alpha_3, \\dots, \\alpha_{2j_0+1}}`
        corresponding to :math:`T_1, T_3, \\dots, T_{2j_0+1}`.
    """
    coeffs = []
    for j in range(j0 + 1):
        sum_i = 0
        for i in range(j+1, b+1):
            sum_i += comb(2*b, b + i)

        coeff = 4 * (2 ** (-2 * b)) * sum_i
        coeffs.append(coeff)
    return np.array(coeffs)

def unary_angles(coeffs):
    """
    Computes the rotation angles :math:`\\phi_i` for preparing the unary state
    corresponding to the square-root-amplitude encoding of the Chebyshev
    coefficients.

    The unary-encoded state prepared corresponds to

    .. math::

        |\\text{unary}\\rangle =
        \\alpha_1|100\\dots00\\rangle + \\alpha_3|110\\dots00\\rangle +
        \\cdots + \\alpha_{2j_0+1}|111\\dots11\\rangle

    Angles are determined by successive :math:`\\arctan` operations on ratios of
    target amplitudes :math:`\\sqrt{\\alpha_i}` [16]. The resulting vector
    of :math:`\\phi_i` angles is then multiplied by 2 to compensate for the
    :math:`1/2` scaling convention in the elementary ``ry`` gate.

    Parameters
    ----------
    coeffs : np.ndarray
        Positive Chebyshev coefficients :math:`\\alpha_i`.

    Returns
    -------
    phi : np.ndarray
        Rotation angles :math:`\\phi_i` defining the unary state preparation
        register.
    """
    
    alpha = np.sqrt(coeffs/len(coeffs))
    phi = jnp.zeros(len(alpha)-1)
    phi = phi.at[-1].set(jnp.arctan(alpha[-1] / alpha[-2])) # Last angle is determined by ratio of final two amplitudes
    for i in range(len(phi)-2,-1,-1):
        phi = phi.at[i].set(jnp.arctan(alpha[i+1]/alpha[i]/jnp.cos(phi[i+1]))) # Recursively compute previous rotation angles
    return 2 * phi # Compensate for factor 1/2 in ry 

def unary_prep(out_case, phi):
    """
    Prepares the unary-encoded
    superposition of Chebyshev coefficients in the unary register.

    The unary-encoded state prepared corresponds to

    .. math::

        |\\text{unary}\\rangle =
        \\alpha_1|100\\dots00\\rangle + \\alpha_3|110\\dots00\\rangle +
        \\cdots + \\alpha_{2j_0+1}|111\\dots11\\rangle.

    The operation is implemented as a cascade of controlled ``ry`` rotations,
    each conditioned on the activation of previous qubits in the unary chain.

    Parameters
    ----------
    out_case : QuantumVariable
        Unary-encoded state representing the LCU index variable :math:`i`.
    phi : np.ndarray
        Rotation angles derived from Chebyshev coefficients.
    """
    x(out_case[0])
    ry(phi[0], out_case[1])
    for i in jrange(1, out_case.size - 1):
        with control(out_case[i]):
            ry(phi[i], out_case[i + 1])

def inner_CKS(A, b, eps, max_beta=None):
    """
    Core circuit implementation of the Childs–Kothari–Somma (CKS) linear
    systems algorithm using the Chebyshev approximation of :math:`1/x`.

    This constructs the composite unitary

    .. math::

        W = V^\\dagger U V ,

    where :math:`V` prepares the LCU state and :math:`U` applies the
    Chebyshev terms :math:`T_{2j+1}(H)` conditioned on a unary register.

    Handling of Coefficients and Phases
    ----------------------------------
    The Chebyshev coefficients alternate in sign :math:`(-1)^j\\alpha_j`.
    Since the LCU lemma requires :math:`\\alpha_i>0`, negative terms are
    implemented by conditional Z-gates applied on each index qubit in
    ``out_case``.

    Parameters
    ----------
    A : np.ndarray
        Hermitian matrix being inverted.
    b : function
        State preparation procedure generating :math:`|b\\rangle`.
    eps : float
        Desired precision :math:`\\epsilon`.
    max_beta : float, optional
        Maximum allowable :math:`\\beta` cap.

    Returns
    -------
    operand : QuantumVariable
        Register holding the resulting solution state.
    in_case : QuantumVariable
        Auxiliary case indicator register.
    out_case : QuantumVariable
        Unary register encoding :math:`T_{2j+1}` terms.
    """

    # Construct block encoding of A as a set of Pauli unitaries
    H = QubitOperator.from_matrix(A).to_pauli()
    unitaries, coeffs = H.unitaries()
    n = int(np.ceil(np.log2(len(unitaries))))

    # Compute Chebyshev parameters and rotation angle
    j_0, beta = CKS_parameters(A, eps, max_beta)
    phi = unary_angles(cheb_coefficients(j_0, beta))
    
    def state_prep(case):
        prepare(case, np.sqrt(coeffs))

    def RU(case_indicator, operand, unitaries):
        """
        Applies one qubitization step :math:`RU` (or its inverse) corresponding
        to the Chebyshev polynomial block encoding.

        .. math::

            T_k(H) = (RU)^k

        where :math:`R` reflects about the auxiliary state :math:`|G\\rangle_a`.

        Parameters
        ----------
        case_indicator : QuantumVariable
            Register encoding the Chebyshev index :math:`i`.
        operand : QuantumVariable
            Target state register upon which block-encoded :math:`H` acts.
        """
        qswitch(operand, case_indicator, unitaries) #qswitch applies $U = \sum_i\ket{i}\bra{i}\otimes P_i$.
        reflection(case_indicator, state_function=state_prep) # diffuser implements the reflection operator R about $\ket{G}$.
    
    out_case = QuantumVariable(j_0)
    in_case = QuantumFloat(n)
    operand = QuantumVariable(H.find_minimal_qubit_amount())

    # Core LCU protocol: PREP, SELECT, PREP^† following V^† U V structure
    with conjugate(unary_prep)(out_case, phi):
        with conjugate(state_prep)(in_case):
            prepare(operand, b, reversed=True)
            with control(out_case[0]):
                RU(in_case, operand, unitaries)
            for i in jrange(1, j_0):
                z(out_case[i])
                with control(out_case[i]):
                    RU(in_case, operand, unitaries)
                    RU(in_case, operand, unitaries)

    return operand, in_case, out_case

@RUS(static_argnums=[1,2])
def inner_CKS_wrapper(qlsp, eps, max_beta=None):
    """
    Wraps the full CKS circuit construction and post-selection measurement.

    Post-selection
    --------------
    The algorithm succeeds if both auxiliary registers
    (:data:`in_case`, :data:`out_case`) are measured in the
    :math:`|0\\rangle` state. Amplitude amplification is optionally used
    to improve the success probability from :math:`\\Omega(1)` to constant.

    Parameters
    ----------
    qlsp : function
        Function returning the linear system specification
        :math:`(A, \\vec{b})`.
    eps : float
        Target precision :math:`\\epsilon`.
    max_beta : float, optional
        Optional cap on complexity parameter :math:`\\beta`.

    Returns
    -------
    success_bool : bool
        Indicates whether post-selection succeeded.
    operand : QuantumVariable
        Register containing the resulting solution
        :math:`|\\tilde x\\rangle \\propto A^{-1}|b\\rangle` on success.
    """

    A, b = qlsp()

    operand, in_case, out_case = inner_CKS(A, b, eps, max_beta)

    success_bool = (measure(out_case) == 0) & (measure(in_case) == 0)

    for i in jrange(operand.size//2):
        swap(operand[i], operand[operand.size-i-1])

    return success_bool, operand   


def CKS(A, b, eps, max_beta=None):
    """
    Solves the Quantum Linear Systems Problem (QLSP)
    :math:`A\\vec{x} = \\vec{b}` using the Chebyshev-based
    Childs–Kothari–Somma algorithm.

    This routine initializes the QLSP instance, constructs required quantum
    registers, and invokes the CKS wrapper to produce the output state

    .. math::

        |\\tilde x\\rangle \\propto A^{-1}|b\\rangle .

    The asymptotic complexity is

    .. math::

        \\mathcal{O}\\!\\left(d \\kappa^2 \\log^2\\!\\frac{d\\kappa}{\\epsilon}\\right),

    providing exponentially better precision scaling compared to HHL.

    Parameters
    ----------
    A : np.ndarray
        Hermitian matrix defining the system.
    b : function
        Quantum state preparation function generating :math:`|b\\rangle`.
    eps : float
        Desired precision :math:`\\epsilon`.
    max_beta : float, optional
        Optional upper bound for the complexity parameter :math:`\\beta`.

    Returns
    -------
    operand : QuantumVariable
        Quantum register containing the final solution
        :math:`|\\tilde x\\rangle \\propto A^{-1}|b\\rangle`.
    """

    def qlsp():
        return A, b
    
    operand = inner_CKS_wrapper(qlsp, eps, max_beta)
    return operand