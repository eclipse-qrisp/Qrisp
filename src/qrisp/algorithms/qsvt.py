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
from qrisp.alg_primitives.prepare import prepare
from qrisp.jasp import jrange, q_cond, RUS, check_for_tracing_mode
from qrisp.operators import QubitOperator


def inner_QSVT(A, operand_prep, phi_qsvt):
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
    in_case : QuantumFloat
        Auxiliary variable used for the block-encoding after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    temp : QuantumBool
        Auxiliary variable used for the reflections after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    """
    if isinstance(A, tuple) and len(A) == 3:
        U, state_prep, n = A
    else:
        H = QubitOperator.from_matrix(A, reverse_endianness=True)
        U, state_prep, n = (H.pauli_block_encoding())  # Construct block encoding of A as a set of Pauli unitaries

    def reflection(case, temp, phase):
        with conjugate(mcx)(case,temp,ctrl_state=0):
            rz(phase, temp)
    
    def U_tilde(case, operand, state_function):
        with conjugate(state_function)(case):
            U(case, operand)

            
    def U_tilde_dg(case, operand, state_function):
        with conjugate(state_function)(case):
            with invert():
                U(case, operand)
    
    temp = QuantumBool()
    in_case = QuantumFloat(n)
    operand = operand_prep()

    h(temp) 

    phi_qsvt = jnp.flip(phi_qsvt)
    d = len(phi_qsvt) - 1

    # Jasp mode
    if check_for_tracing_mode():
        x_cond = q_cond
    else:
        def x_cond(pred, true_fun, false_fun, *operands):
            if pred:
                return true_fun(*operands)
            else:
                return false_fun(*operands)
            
    for i in jrange(0, d): 
        reflection(in_case, temp, phase=2 * phi_qsvt[i])
        x_cond(i%2==0, U_tilde, U_tilde_dg, in_case, operand, state_prep) 
    reflection(in_case, temp, phase=2 * phi_qsvt[d])
        
    h(temp)

    return operand, in_case, temp


@RUS
def inner_QSVT_wrapper(matrix, operand_prep, phi_qsvt):
    """
    Wrapper for Quantum Singular Value Transformation (QSVT), performing post-selection using the RUS (Repeat-Until-Success) protocol.
    
    This function calls a supplied problem generator ``matrix`` to obtain the matrix :math`A`, 
    an initial state preparation function ``operand_prep``, and a sequence of phase angles ``phi_qsvt``. 
    It invokes :func:`inner_QSVT`, and returns the solution operand QuantumFloat
    if post-selection on auxiliary QuantumFloats succeeds.

    The algorithm succeeds if both auxiliary QuantumFloats ``in_case`` and ``temp`` are measured
    in the :math:`\ket{0}` state. If post-selection fails, the :func:`RUS` decorator repeats the
    simulation until success.

    Parameters
    ----------
    matrix : callable
        A function that returns the block-encoded matrix :math:`A` used in
        the QSVT procedure.
    operand_prep : callable
        A callable that prepares the corresponding quantum state ``operand``.
    phi_qsvt : jnp.array
        Sequence of phase angles defining the polynomial transformation implemented
        by the QSVT circuit.

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

    A = matrix()

    operand, in_case, temp = inner_QSVT(A, operand_prep, phi_qsvt)

    success_bool = (measure(temp) == 0) & (measure(in_case) == 0)

    return success_bool, operand


def QSVT(A, operand_prep, phi_qsvt):
    """
    Performs the Quantum Singular Value Transformation (QSVT) for matrix inversion.

    This function serves as the simplified entry point to perform quantum singular value 
    transformation on a block-encoded operator and an input quantum state vector to solve
    a QLSP problem :math:`A \\vec{x} = \\vec{b}`.

    It wraps the core QSVT circuit execution with post-selection handling, abstracting the 
    lower-level details of block-encoding and phase modulation angle management.

    Parameters
    ----------
    A : tuple or numpy.ndarray
        A block-encoded operator tuple (U, state_prep, n) or a Hermitian matrix representing 
        the operator to transform.
    operand_prep : callable
        Function returning the (operand) QuantumVariable in the initial system state $\ket{\psi_0}$, i.e.,
        ``operand=operand_prep()``.

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
    """
    def matrix():
        return A
       
    operand = inner_QSVT_wrapper(matrix, operand_prep, phi_qsvt)
    return operand


def inner_QSVT_inversion(A, b, eps, kappa = None):
    """
    Quantum Linear System solver via Quantum Singular Value Transformation (QSVT).

    This function implements a QSVT-based quantum circuit that approximates the matrix 
    inversion operation :math:`A^{-1}` applied to an input quantum state :math:`\ket{b}` representing. 
    Using polynomial phase factor sequences generated to approximate 
    the inverse function within error tolerance ``eps``, it constructs the quantum singular 
    value transformation that effectively solves the Quantum Linear System Problem (QLSP) :math:`A\\vec{x}=\\vec{b}`.

    The block-encoding of matrix :math:`A` can be provided either explicitly as a block encoding tuple,
    or implicitly via a Hermitian matrix which will be block-encoded internally via :meth:`Pauli decomposition <qrisp.operators.qubit.QubitOperator.pauli_block_encoding>`. 
    Note that in the former case, the condition number :math:`\kappa` of :math:`A` also has to be provided. 

    This function implements the core QSVT circuit without repeat-until-success or post-selection.

    Parameters
    ----------
    A : tuple or numpy.ndarray
        Either a 3-tuple (U, state_prep, n) representing the block-encoding unitary and its 
        associated state preparation callable and ancilla qubit size, or a Hermitian matrix 
        that is internally block-encoded.
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
    in_case : QuantumFloat
        Auxiliary variable used for the block-encoding after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    temp : QuantumBool
        Auxiliary variable used for the reflections after applying the QSVT protocol.
        Must be measured in state $\ket{0}$ for the QSVT protocol to be successful.
    """
    if isinstance(A, tuple) and len(A) == 3:
        kappa = kappa
    else:
        kappa = np.linalg.cond(A)


    def inversion_angles(eps, kappa):
        """
        Compute phase angles for QSVT polynomial approximation of matrix inversion.

        This helper function generates a sequence of phase modulation angles ``phi_qsvt`` 
        that define a quantum singular value transformation polynomial approximating 
        the function :math:`f(x) = 1/x` on the domain :math:`\[1/\kappa, 1\]` within precision
        ``eps``. It leverages the `pyqsp library <https://github.com/ichuang/pyqsp>`_'s polynomial construction for rational 
        function approximation, ensuring boundedness and numerical stability.

        The function first constructs a polynomial approximation of the inverse function 
        using the condition number ``kappa`` and specified precision `eps`. Then, it translates 
        the QSP phase sequence into the corresponding QSVT angles by applying appropriate 
        phase shifts, matching the alternating phase modulation framework of QSVT.

        Parameters
        ----------
        eps : float
            Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
            :math:`\epsilon` of :math:`\ket{x}`.
        kappa : float, optional
            Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
            a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.

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

        return jnp.array(phi_qsvt), s


    phi_inversion, _ = inversion_angles(eps, kappa)
    
    if callable(b):
        def bprep():
            return b()

    else:
        def bprep():
            operand = QuantumFloat(int(np.log2(b.shape[0])))
            prepare(operand, b)
            return operand
    
    operand, in_case, temp = inner_QSVT(A, bprep, phi_inversion)

    return operand, in_case, temp


@RUS(static_argnums=[1, 2])
def inner_QSVT_inversion_wrapper(qlsp, eps, kappa = None):
    """
    Wrapper for the Quantum Linear System solver via Quantum Singular Value Transformation (QSVT), performing post-selection using the RUS (Repeat-Until-Success) protocol.
    
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
        A function that returns a tuple (A, b) representing the block-encoded operator 
        and input state vector for the quantum linear system problem.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.

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
    A, b = qlsp()

    if isinstance(A, tuple) and len(A) == 3:
        kappa = kappa
    else:
        kappa = np.linalg.cond(A)

    operand, in_case, temp = inner_QSVT_inversion(A, b, eps, kappa)

    success_bool = (measure(temp) == 0) & (measure(in_case) == 0)

    return success_bool, operand


def QSVT_inversion(A, b, eps, kappa=None):
    """
    Performs the Quantum Singular Value Transformation (QSVT) for matrix inversion.

    This function serves as the simplified entry point to perform quantum singular value 
    transformation on a block-encoded operator and an input quantum state vector to solve
    a QLSP problem :math:`A \\vec{x} = \\vec{b}`.

    It wraps the core QSVT circuit execution with post-selection handling, abstracting the 
    lower-level details of block-encoding and phase modulation angle management.

    Parameters
    ----------
    A : tuple or numpy.ndarray
        A block-encoded operator tuple (U, state_prep, n) or a Hermitian matrix representing 
        the operator to transform.
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
        Quantum variable containing the final (approximate) solution state
        :math:`\ket{\\tilde{x}} \propto A^{-1}\ket{b}`. When the internal
        :ref:`RUS <RUS>` decorator reports success (``success_bool = True``), this
        variable contains the valid post-selected solution. If ``success_bool``
        is ``False``, the simulation automatically repeats until success.

    Examples
    --------

    The following examples demonstrate how QSVT can be applied for matrix inversion to solve the
    quantum linear systems problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}`, using either a direct Hermitian matrix input or
    a preconstructed block-encoding representation.
    
    **Example 1: Solving a 4×4 Hermitian system**

    First, we define a small Hermitian matrix :math:`A` and a right-hand side vector :math:`\\vec{b}`, as 
    well as our desired precision :math:`\epsilon`:

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                      [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                      [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                      [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

        eps = 0.001

    We now solve this linear system using QSVT for matrix inversion:

    ::

        from qrisp.algorithms.qsvt import QSVT_inversion
        from qrisp.jasp import terminal_sampling

        @terminal_sampling
        def main():

            x = QSVT_inversion(A, b, eps)
            return x

        res_dict = main()

    The resulting dictionary contains the measurement probabilities for each computational basis state.
    To extract the corresponding quantum amplitudes (up to sign) and compare to the classical normalized
    solution:

    ::

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        q = np.array([res_dict.get(key, 0) for key in range(4)])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
        print("QUANTUM SIMULATION\\n", q, "\\nCLASSICAL SOLUTION\\n", c)
        # QUANTUM SIMULATION
        #  [0.02898903 0.55472887 0.53005196 0.64068747] 
        # CLASSICAL SOLUTION
        #  [0.02944539 0.55423278 0.53013239 0.64102936]

    We see that we obtained the same result in our quantum simulation up to precision :math:`\epsilon`!

    
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
        b = np.array([0, 1, 1, 1, 0, 0, 1, 1])

        eps = 0.01     
        kappa = np.linalg.cond(A)

        print(A)
        # [[ 5.  0.  0.  0. -2.  0.  0.  0.]
        #  [ 0.  5.  0.  0.  0. -2.  0.  0.]
        #  [ 0.  0.  5.  0.  0.  0. -2.  0.]
        #  [ 0.  0.  0.  5.  0.  0.  0. -2.]
        #  [-2.  0.  0.  0.  5.  0.  0.  0.]
        #  [ 0. -2.  0.  0.  0.  5.  0.  0.]
        #  [ 0.  0. -2.  0.  0.  0.  5.  0.]
        #  [ 0.  0.  0. -2.  0.  0.  0.  5.]]

   
    This matrix can be decomposed using three unitaries: the identity :math:`I`, and two shift operators :math:`V\colon\ket{k}\\rightarrow-\ket{k+N/2 \mod N}` and :math:`V^{\dagger}\colon\ket{k}\\rightarrow-\ket{k-N/2 \mod N}`.
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

    We now define the block_encoding ``(U, state_prep, n)``. 
    In this case, we have to also pass the condition number :math:`\kappa`:

    ::

        coeffs = np.array([5,1,1])
        alpha = np.sum(coeffs)

        def U(case, operand):
            qswitch(operand, case, unitaries)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs/alpha))

        block_encoding = (U, state_prep, 2)

    We solve the linear system by passing this block-encoding tuple as ``A`` into the QSVT function.
    We can also pass ``b`` as a function. In this case we define ``b_prep`` as

    ::

        from qrisp import QuantumFloat
        def b_prep():
            operand = QuantumFloat(int(np.log2(b.shape[0])))
            prepare(operand, b)
            return operand

    
    and solve the linear system:

    ::

        @terminal_sampling
        def main():

            x = QSVT_inversion(block_encoding, b_prep, eps, kappa)
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
        # [0.         0.3336103  0.46663903 0.46663891 0.         0.13302811 0.46663899 0.46663891] 
        # CLASSICAL SOLUTION
        # [0.         0.33333333 0.46666667 0.46666667 0.         0.13333333 0.46666667 0.46666667]

    To perform quantum resource estimation, replace the ``@terminal_sampling``
    decorator with ``@count_ops(meas_behavior="0")``:

    ::

        from qrisp import count_ops

        @count_ops(meas_behavior="0")
        def main():

            x = QSVT_inversion(A, b_prep, eps, kappa)
            return x

        res_dict = main()
        print(res_dict)
        # {'rz': 58, 'x': 403, 'cx': 291, 'gphase': 115, 'u3': 121, 'p': 57, 'h': 2, 'measure': 2}   

        
    **Example 3: Using a custom block encoding**

    In this example, we construct a block-encoding representation for the following tridiagonal sparse matrix:

    ::

        import numpy as np

        def tridiagonal_shifted(n, mu=1.0, dtype=float):
            I = np.eye(n, dtype=dtype)
            return (2 + mu) * I - np.eye(n, k=1, dtype=dtype) - np.eye(n, k=-1, dtype=dtype)

        N = 8
        A = tridiagonal_shifted(N, mu=3)
        A[N-1,0] = -1
        A[0,N-1] = -1
        b = np.array([0, 1, 1, 1, 0, 0, 1, 1])

        eps = 0.01
        kappa = np.linalg.cond(A)

        print(A)
        #[[ 5. -1.  0.  0.  0.  0.  0. -1.]
        # [-1.  5. -1.  0.  0.  0.  0.  0.]
        # [ 0. -1.  5. -1.  0.  0.  0.  0.]
        # [ 0.  0. -1.  5. -1.  0.  0.  0.]
        # [ 0.  0.  0. -1.  5. -1.  0.  0.]
        # [ 0.  0.  0.  0. -1.  5. -1.  0.]
        # [ 0.  0.  0.  0.  0. -1.  5. -1.]
        # [-1.  0.  0.  0.  0.  0. -1.  5.]]

    This matrix can be decomposed using three unitaries: the identity :math:`I`, and two shift operators :math:`V\colon\ket{k}\\rightarrow-\ket{k+1 \mod N}` and :math:`V^{\dagger}\colon\ket{k}\\rightarrow-\ket{k-1 \mod N}`.
    We define their corresponding functions:

    ::

        from qrisp import gphase, prepare, qswitch

        def I(qv):
            pass

        def V(qv):
            qv += 1
            gphase(np.pi, qv[0])

        def V_dg(qv):
            qv -= 1
            gphase(np.pi, qv[0])

        unitaries = [I, V, V_dg]

    We now define the block_encoding ``(U, state_prep, n)``: 
    
    ::

        coeffs = np.array([5,1,1])
        alpha = np.sum(coeffs)

        def U(case, operand):
            qswitch(operand, case, unitaries)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs/alpha))

        block_encoding = (U, state_prep, 2)

    Note that, unlike the previous example, the block-encoding unitary does not satisfy $U^2=I$.

    ::

        @terminal_sampling
        def main():

            x = QSVT_inversion(block_encoding, b, eps, kappa)
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
        # [0.17153117 0.43691595 0.47894578 0.42359613 0.10474241 0.10286244 0.41215706 0.42359611] 
        # CLASSICAL SOLUTION
        # [0.17207112 0.43684477 0.47875139 0.42351084 0.1054015  0.10349665 0.41208177 0.42351084]

    """
    def qlsp():
        return A, b
       
    operand = inner_QSVT_inversion_wrapper(qlsp, eps, kappa)
    return operand