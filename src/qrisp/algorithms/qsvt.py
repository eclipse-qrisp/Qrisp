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
from qrisp import (
    QuantumVariable,
    QuantumFloat,
    QuantumBool,
    h,
    conjugate,
    measure,
    invert,
    mcx,
    rz,
)
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.jasp import jrange, q_cond, RUS, check_for_tracing_mode
from qrisp.operators import QubitOperator
from qrisp.block_encodings import BlockEncoding
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def QSVT(A, phi_qsvt):
    """
    Core implementation of the Quantum Singular Value Transformation (QSVT) algorithm.

    This function constructs the quantum circuit performing singular value transformation 
    on a block-encoded matrix A, applied to an input quantum state b, using 
    a sequence of phase modulation angles ``phi_qsvt``. The implementation follows the 
    alternating phase modulation framework of QSVT based on controlled reflections and 
    block-encoding unitaries, as described by `Gilyén et al. <https://arxiv.org/abs/1806.01838>`_.

    The algorithm applies a sequence of phase modulated reflection operators interleaved 
    with applications of the block-encoding unitary :math:`U` and its state preparation auxiliary, 
    implementing a polynomial transformation of the singular values of A determined by 
    the phase angles phi_qsvt.

    This function implements the core QSVT circuit without repeat-until-success or post-selection.

    Parameters
    ----------
    A : tuple or numpy.ndarray
        Either a 3-tuple (U, state_prep, n) representing the block-encoding unitary and its 
        associated state preparation callable and ancilla qubit size, or a Hermitian matrix 
        that is internally block-encoded.
    operand_prep : callable
        Function returning the (operand) QuantumVariable in the initial system state $\ket{\psi_0}$, i.e.,
        ``operand=operand_prep()``.
    phi_qsvt : tuple or list of floats
        Phase modulation angles phi used in the singular value transformation polynomial.

    Returns
    -------
    operand : QuantumVariable
        Operand variable after applying the QSVT protocol.
    case : QuantumFloat
        Auxiliary variable used for the block-encoding after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    temp : QuantumBool
        Auxiliary variable used for the reflections after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    """

    m = len(A.anc_templates)

    def reflection(args, phase):
        qubits = sum([arg.reg for arg in args[1:m + 1]], []) 
        with conjugate(mcx)(qubits, args[0], ctrl_state=0):
            rz(phase, args[0])

    def even(args):
        A.unitary(*args[1:])

    def odd(args):
        with invert():
            A.unitary(*args[1:])

    def new_unitary(*args):
        h(args[0]) 

        phlipped_qsvt = jnp.flip(phi_qsvt)
        d = len(phi_qsvt) - 1
                
        for i in jrange(0, d): 
            reflection(args, phase = 2 * phlipped_qsvt[i])
            q_cond(i%2==0, even, odd, args) 
        reflection(args, phase=2 * phlipped_qsvt[d])
            
        h(args[0])

    new_anc_templates = [QuantumBool().template()] + A.anc_templates
    new_alpha = 1 # TBD

    return BlockEncoding(new_unitary, new_anc_templates, new_alpha, is_hermitian = False)



def QSVT_inversion(A, eps, kappa):
    """
    Quantum Linear System solver via Quantum Singular Value Transformation (QSVT).

    This function implements a QSVT-based quantum circuit that approximates the matrix 
    inversion operation :math:`A^{-1}` applied to an input quantum state :math:`\ket{b}` representing. 
    Using polynomial phase factor sequences generated to approximate 
    the inverse function within error tolerance ``eps``, it constructs the quantum singular 
    value transformation that effectively solves the Quantum Linear System Problem (QLSP) :math:`A\\vec{x}=\\vec{b}`.

    The block-encoding of matrix :math:`A` can be provided either explicitly as a block-encoding tuple,
    or implicitly via a Hermitian matrix which will be block-encoded internally via :meth:`Pauli decomposition <qrisp.operators.qubit.QubitOperator.pauli_block_encoding>`. 
    Note that in the former case, the condition number :math:`\kappa` of :math:`A` also has to be provided. 

    This function implements the core QSVT circuit without repeat-until-success or post-selection.

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

    Returns
    -------
    operand : QuantumVariable
        Operand variable after applying the QSVT protocol.
    case : QuantumFloat
        Auxiliary variable used for the block-encoding after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    temp : QuantumBool
        Auxiliary variable used for the reflections after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    """

    def inversion_angles(eps, kappa):
        """
        Compute phase angles for QSVT polynomial approximation of matrix inversion.

        This helper function generates a sequence of phase modulation angles ``phi_qsvt`` 
        that define a quantum singular value transformation polynomial approximating 
        the function :math:`f(x) = 1/x` on the domain :math:`\[1/\kappa, 1\]` within precision
        ``eps``. It leverages the `pyqsp library <https://github.com/ichuang/pyqsp>`_'s polynomial construction for rational 
        function approximation, ensuring boundedness and numerical stability.

        The function first constructs a polynomial approximation of the inverse function 
        using the condition number ``kappa`` and specified precision ```eps``. Then, it translates 
        the QSP phase sequence into the corresponding QSVT angles by applying appropriate 
        phase shifts, matching the alternating phase modulation framework of QSVT.

        Parameters
        ----------
        eps : float
            Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
            :math:`\epsilon` of :math:`\ket{x}`.
        kappa : float
            Condition number :math:`\\kappa` of the matrix :math:`A`.

        Returns
        -------
        phi_qsvt : numpy.ndarray
            Array of phase modulation angles encoding the QSVT polynomial for matrix inversion.
        s : float
            Scaling factor from the polynomial generation process, related to norm bounding.
        """
        from pyqsp.poly import PolyOneOverX
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases

        pcoefs, s = PolyOneOverX().generate(kappa, eps, return_coef=True, ensure_bounded=True, return_scale=True)
        phi_qsp = QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.00001)
        phi_qsp = np.array(phi_qsp)

        phi_qsvt = np.empty(len(phi_qsp))
        phi_qsvt[0] = phi_qsp[0] + 3 * np.pi / 4 - (3 + len(phi_qsp) % 4) * np.pi / 2
        phi_qsvt[1:-1] = phi_qsp[1:-1] + np.pi / 2
        phi_qsvt[-1] = phi_qsp[-1] - np.pi / 4

        return np.array(phi_qsvt), s


    phi_inversion, _ = inversion_angles(eps, kappa)
    print(phi_inversion)
    return QSVT(A, phi_inversion)