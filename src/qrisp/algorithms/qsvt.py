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
    Performs `Quantum Singular Value Transformation (QSVT) <https://arxiv.org/abs/1806.01838>_`

    Applies polynomial transformations on the singular values of an operator that block encods the matrix A,
    by tuning a sequence of phase angles that define the transformation.

    The Quantum Singular Value Transformation is described as follows:
    Suppose $A = \sum_i \sigma_i \ket{u_i}\bra{v_i}$ is the singular value decomposition (SVD) of $A$,
      where $\sigma_i \in [0, 1]$ are the singular values. A block-encoding $U_A$ of $A$ satisfies 
      $(\bra{0} \otimes I) U_A (\ket{0} \otimes I) = A / \alpha$ for some normalization constant $\alpha$.
    
    QSVT constructs a transformed unitary $U_P$ such that
    
      .. math::
          (\bra{0} \otimes I) U_P (\ket{0} \otimes I) = P(A / \alpha)
    
      where $P$ is a polynomial whose coefficients are determined implicitly by the chosen sequence 
      of phase modulation angles $\phi = (\phi_0, \phi_1, \dots, \phi_d)$.

    

    Parameters
    ----------
    A : BlockEncoding
        Block encoding of the input operator (not necessarily Hermitian).

    phi_qsvt : tuple or list of floats
        Sequence of phase modulation angles $\phi$ defining the transformation polynomial.

    Returns
    -------
    BlockEncoding
        A block encoding of the transformed operator $P(A/\alpha)$.
    
    Examples
    --------

    ::

        from qrisp import *
        from qrisp.algorithms.qsvt import QSVT_inversion
        from qrisp.operators import QubitOperator
        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
              [0.14516978, 0.68701415, -0.04929867, -0.00999921],
              [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
              [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

        H = QubitOperator.from_matrix(A, reverse_endianness=True)
        BA = H.pauli_block_encoding()

        BA_inv = QSVT_inversion(BA, 0.01, 2)

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumVariable(2)
            prepare(operand, b)
            return operand

        @terminal_sampling()
        def main():
            operand = BA_inv.apply_rus(prep_b)()
            return operand

        res_dict = main()

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        q = np.array([res_dict.get(key, 0) for key in range(len(b))])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\n", q, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.02846016 0.55539072 0.53011626 0.6400843 ] 
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]


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

    This function performs QSVT on a block encoding that approximates the matrix 
    inversion operation :math:`A^{-1}` applied to an input quantum state :math:`\ket{b}`. 
    Using polynomial phase factor sequences generated to approximate 
    the inverse function within error tolerance ``eps``, it constructs the quantum singular 
    value transformation that effectively solves the Quantum Linear System Problem (QLSP) :math:`A\\vec{x}=\\vec{b}`.

    The condition number :math:`\kappa` of the block encoded matrix :math:`A` also has be provided. 

    Parameters
    ----------
    A : BlockEncoding
        Block encoding of the input operator (not necessarily Hermitian).
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.

    Returns
    -------
    BlockEncoding
        A block encoding of the transformed operator $A^{-1}$.
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