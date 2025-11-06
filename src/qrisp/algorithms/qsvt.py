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
from qrisp import (
    QuantumVariable,
    QuantumFloat,
    QuantumBool,
    h,
    conjugate,
    measure,
    invert,
    multi_measurement,
    mcx,
    rz,
    terminal_sampling
)
from qrisp.alg_primitives.prepare import prepare
from qrisp.jasp import jrange, RUS
from qrisp.operators import QubitOperator

""" from pyqsp.poly import PolyOneOverX
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

def qsvt_angles_scaling(kappa, eps):   

    pcoefs, s = PolyOneOverX().generate(kappa, eps, return_coef=True, ensure_bounded=True, return_scale=True)
    phi_qsp = QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.00001)

    phi_qsvt = np.empty(len(phi_qsp))

    phi_qsvt[0] = phi_qsp[0] + 3 * np.pi / 4 - (3 + len(phi_qsp) % 4) * np.pi / 2
    phi_qsvt[1:-1] = phi_qsp[1:-1] + np.pi / 2
    phi_qsvt[-1] = phi_qsp[-1] - np.pi / 4

    return phi_qsvt, s """


def inner_QSVT_inversion(A_scaled, b, phi_qsvt):
    """
    Core implementation of the Quantum Singular Value Transformation (QSVT) algorithm.

    This function constructs the quantum circuit performing singular value transformation 
    on a block-encoded matrix A, applied to an input quantum state b, using 
    a sequence of phase modulation angles phi_qsvt. The implementation follows the 
    alternating phase modulation framework of QSVT based on controlled reflections and 
    block-encoding unitaries, as described by `Gilyén et al. <https://arxiv.org/abs/1806.01838>`_.

    The algorithm applies a sequence of phase modulated reflection operators interleaved 
    with applications of the block-encoding unitary U and its state preparation auxiliary, 
    implementing a polynomial transformation of the singular values of A determined by 
    the phase angles phi_qsvt.

    This function implements the core QSVT circuit without repeat-until-success or post-selection.

    Parameters
    ----------
    A_scaled : tuple or numpy.ndarray
        Either a 3-tuple (U, state_prep, n) representing the block-encoding unitary and its 
        associated state preparation callable and ancilla qubit size, or a Hermitian matrix 
        that is internally block-encoded.
    b : numpy.ndarray
        Vector representing the input quantum state |b> to which the transformed operator 
        is applied.
    phi_qsvt : tuple or list of floats
        Phase modulation angles phi used in the singular value transformation polynomial.

    Returns
    -------
    operand : QuantumVariable
        Quantum register containing the operand state after applying the QSVT circuit.
    in_case : QuantumFloat
        Auxiliary quantum float register encoding the internal block-encoding state. Must be 
        measured to |0> for successful transformation.
    temp : QuantumBool
        Ancillary qubit used in reflections and measurement for success post-selection.
    """
    if isinstance(A_scaled, tuple) and len(A_scaled) == 3:
        U, state_prep, n = A_scaled
    else:
        H = QubitOperator.from_matrix(A_scaled, reverse_endianness=True)
        U, state_prep, n = (H.pauli_block_encoding())  # Construct block encoding of A as a set of Pauli unitaries

    def reflection(case, temp, phase):
        with conjugate(mcx)(case,temp,ctrl_state=0):
            rz(phase, temp)
    
    def U_tilde(case, operand, state_function, dg = False):
        with conjugate(state_function)(case):
            if dg:
                with invert():
                    U(case, operand)
            
            else:
                U(case, operand)
    
    temp = QuantumBool()
    in_case = QuantumFloat(n)
    operand = QuantumFloat(int(np.log2(b.shape[0])))

    prepare(operand, b)
    h(temp) 

    d = len(phi_qsvt) - 1
    dg_flag = False
    for i in range(d, -1, -1):
        reflection(in_case, temp, phase=2 * phi_qsvt[i])
        if i != 0:
            U_tilde(in_case, operand, state_prep, dg=dg_flag)
        dg_flag = not dg_flag
        
    h(temp)

    return operand, in_case, temp

@RUS
def inner_QSVT_inversion_wrapper(qlsp, angles):
    """
    Wrapper for the Quantum Singular Value Transformation (QSVT) implementation, performing post-selection using the RUS (Repeat-Until-Success) protocol.
    
    This function calls a supplied problem generator ``qlsp`` to obtain the linear system, 
    ``angles`` to obtain the QSVT phase angles, and constructs the required circuit via
    :func:`inner_QSVT`, and returns the solution operand QuantumFloat
    if post-selection on auxiliary QuantumFloats succeeds.

    The algorithm succeeds if both auxiliary QuantumFloats ``in_case`` and ``temp`` are measured
    in the :math:`\ket{0}` state. If post-selection fails, the simulation
    restarts until successful as managed by the :func:`RUS` decorator.

    Parameters
    ----------
    qlsp : callable
        A function that returns a tuple (A_scaled, b) representing the block-encoded operator 
        and input state vector for the quantum linear system problem.
    angles : callable
        A function returning the phase angles tuple used in the QSVT polynomial transformation.

    Returns
    -------
    success_bool : bool
        Indicates whether post-selection succeeded. Only when ``success_bool`` is ``True``
        does ``operand`` contain the proper solution state; otherwise, the simulation
        will run again according to the RUS protocol.
    operand : QuantumVariable
        The quantum state after successful QSVT circuit execution corresponding to the polynomial 
        transformed state.
    """
    A_scaled, b = qlsp()

    phi_qsvt = angles()
    operand, in_case, temp = inner_QSVT_inversion(A_scaled, b, phi_qsvt)

    success_bool = (measure(temp) == 0) & (measure(in_case) == 0)

    return success_bool, operand

def QSVT_inversion(A_scaled, b, phi_qsvt):
    """
    Performs the Quantum Singular Value Transformation (QSVT) for matrix inversion.

    This function serves as the simplified entry point to perform quantum singular value 
    transformation on a block-encoded operator and an input quantum state vector to solve
    a QLSP problem :math:`A \\vec{x} = \\vec{b}`.

    It wraps the core QSVT circuit execution with post-selection handling, abstracting the 
    lower-level details of block-encoding and phase modulation angle management.

    Parameters
    ----------
    A_scaled : tuple or numpy.ndarray
        A block-encoded operator tuple (U, state_prep, n) or a Hermitian matrix representing 
        the operator to transform.
    b : numpy.ndarray
        Vector representing the input quantum state to which the polynomial transformation is applied.
    phi_qsvt : tuple or list of floats
        Phase angles defining the polynomial transformation in the QSVT.

    Returns
    -------
    operand : QuantumVariable
        Quantum variable containing the final (approximate) solution state
        :math:`\ket{\\tilde{x}} \propto A^{-1}\ket{b}`. When the internal
        :ref:`RUS <RUS>` decorator reports success (``success_bool = True``), this
        variable contains the valid post-selected solution. If ``success_bool``
        is ``False``, the simulation automatically repeats until success.
    """
    def qlsp():
        return A_scaled, b
    
    def angles():
        return tuple(phi_qsvt)
    
    operand = inner_QSVT_inversion_wrapper(qlsp, angles)
    return operand