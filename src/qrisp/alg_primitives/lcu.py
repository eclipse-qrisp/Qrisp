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

from qrisp import QuantumFloat, conjugate, measure
from qrisp.jasp import make_jaspr, RUS
from qrisp.alg_primitives.switch_case import qswitch
from qrisp.algorithms.grover.grover_tools import tag_state
from qrisp.alg_primitives.amplitude_amplification import amplitude_amplification
import numpy as np


def inner_LCU(operand_prep, state_prep, unitaries, num_unitaries=None, oaa_iter=0):
    r"""
    Core implementation of the Linear Combination of Unitaries (LCU) protocol without
    Repeat-Until-Success (RUS) protocol. The LCU method is a foundational quantum algorithmic
    primitive that enables the application of a non-unitary operator $A$, expressed as a weighted
    sum of unitaries $U_i$ as $A=\sum_i\alpha_i U_i$, to a quantum state, by embedding $A$ into a larger unitary circuit. This is
    central to quantum algorithms for `Hamiltonian simulation <https://www.taylorfrancis.com/chapters/edit/10.1201/9780429500459-11/simulating-physics-computers-richard-feynman>`_, `Linear Combination of Hamiltonian Simulation (LCHS) <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.131.150603>`_, Quantum Linear Systems (e.g. `HHL algorithm <https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst>`_), `Quantum Signal
    Processing (QSP) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`_, and `Quantum Singular Value Transformation (QSVT) <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`_.

    This function implements the prepare-select-unprepare structure, also known as block encoding:

    .. math::
        \mathrm{LCU} = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    - **PREPARE**: Prepares an ancilla quantum variable in a superposition encoding the normalized coefficients $\alpha_i\geq0$ of the target operator

    .. math ::

            \mathrm{PREPARE}|0\rangle=\sum_i\sqrt{\frac{\alpha_i}{\lambda}}|i\rangle

    - **SELECT**: Applies the unitary $U_i$ to the input state $\ket{\psi}$, controlled on the ancilla variable being in state $|i\rangle$.

    .. math ::

        \mathrm{SELECT}|i\rangle|\psi\rangle=|i\rangle U_i|\psi\rangle

    - **PREPARE**$^\dagger$: Applies the inverse prepartion to the ancilla.

    .. note::

        The LCU protocol is deemed successful only if the ancilla variable is measured in the $\ket{0}$ state, which occurs with a probability proportional to :math:`\frac{\langle\psi|A^{\dagger}A|\psi\rangle}{\lambda^2}` where $\lambda=\sum_i\alpha_i$.
        This function does not perform the measurement; it returns the ancilla variable and the transformed target variable.

    For a complete implementation of LCU with the Repeat-Until-Success protocol, see :func:`LCU`.

    For more details on the LCU protocol, refer to `Childs and Wiebe (2012) <https://arxiv.org/abs/1202.5822>`_, or `related seminars provided by Nathan Wiebe <https://www.youtube.com/watch?v=irMKrOIrHP4>`_.

    Parameters
    ----------
    operand_prep : callable
        A function preparing the input state $\ket{\psi}$. This function must return a :ref:`QuantumVariable` (the ``operand``).
    state_prep : callable
        A function preparing the coefficient state from the $\ket{0}$ state.
        This function receives a :ref:`QuantumFloat` with $\lceil\log_2m\rceil$ qubits for $m$ unitiaries $U_0,\dotsc,U_{m-1}$ as argument and applies

        .. math ::

            \text{PREPARE}\ket{0} = \sum_i\sqrt{\frac{\alpha_i}{\lambda}}\ket{i}

    unitaries : list[callable] or callable
        Either:
          - A list of functions performing some in-place operation on ``operand``, or
          - A function ``unitaries(i, operand)`` performing some in-place operation on ``operand`` depending on a nonnegative integer index ``i`` specifying the case.

    num_unitaries : int, optional
        Required when ``unitaries`` is a callable to specify the number $m$ of unitaries.
    oaa_iter : int, optional
        The number of iterations of oblivious amplitude amplification to perform. The default is 0.

    Returns
    -------
    tuple(:ref:`QuantumFloat`, :ref:`QuantumVariable`)
          - **case_indicator** : Ancilla variable encoding which unitary was selected.
          - **operand** : Target quantum variable after the LCU operation.

    Raises
    ------
    TypeError
        If ``unitaries`` is not a list or a callable.
    ValueError
        If ``num_unitaries`` is not specified when ``unitaries`` is a callable.

    See Also
    --------
    LCU : Full LCU implementation using the RUS protocol.
    view_LCU : Generates the quantum circuit for visualization.

    Examples
    --------

    As a **first example**, we apply the non-unitary operator $A$ to the operand $\ket{\psi}=\ket{1}$ where

    .. math::

        A = \begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix} = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix} + \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix} = I + X

    That is,

    .. math:: A = \alpha_0U_0 + \alpha_1U_1

    where $\alpha_0=\alpha_1=1$, and $U_0=I$, $U_1=X$.

    Accordingly, we define the unitaries

    ::

        from qrisp import *

        def U0(operand):
            pass

        def U1(operand):
            x(operand)

        unitaries = [U0, U1]

    and the ``state_prep`` function implementing

    .. math ::

        \text{PREPARE}\ket{0} = \frac{1}{\sqrt{2}}\left(\ket{0}+\ket{1}\right)

    ::

        def state_prep(case):
            h(case)

    Next, we define the ``operand_prep`` function preparing the state $\ket{\psi}=\ket{1}$

    ::

        def operand_prep():
            operand = QuantumVariable(1)
            x(operand)
            return operand

    Finally, we apply ``inner_LCU``

    >>> case_indicator, operand = inner_LCU(operand_prep, state_prep, unitaries)
    >>> operand.qs.statevector()

    As result we obtain the state

    .. math ::

        \frac12((\ket{0}+\ket{1})\ket{0}_{\text{case}} - \frac12(\ket{0}-\ket{1})\ket{1}_{\text{case}}

    If we now measure the ``case_indicator`` in the state $\ket{0}$, the operand will be in state $\frac{1}{\sqrt{2}}(\ket{0}+\ket{1})$
    implementing (up to rescaling) the non-unitary operator $A$ acting on the input state $\ket{\psi}=\ket{1}$.

    """

    operand = operand_prep()

    if not callable(unitaries):
        if not isinstance(unitaries, list):
            raise TypeError("unitaries must be callable or list[callable].")
        num_unitaries = len(unitaries)
    else:
        if num_unitaries == None:
            raise ValueError(
                "The number of unitiaries must be specified if unitaries is callable."
            )

    # Specify the QunatumVariable that indicates which case to execute
    n = np.int64(np.ceil(np.log2(num_unitaries)))
    case_indicator = QuantumFloat(n)

    # LCU protocol with conjugate preparation
    def LCU_state_prep(case_indicator, operand):
        with conjugate(state_prep)(case_indicator):
            qswitch(operand, case_indicator, unitaries)

    def oracle_func(case_indicator, operand):
        tag_state({case_indicator: 0})

    LCU_state_prep(case_indicator, operand)

    if oaa_iter > 0:
        amplitude_amplification(
            [case_indicator, operand],
            LCU_state_prep,
            oracle_func,
            reflection_indices=[0],
            iter=oaa_iter,
        )

    return case_indicator, operand


def LCU(operand_prep, state_prep, unitaries, num_unitaries=None, oaa_iter=0):
    r"""
    Full implementation of the Linear Combination of Unitaries (LCU) algorithmic primitive using the
    Repeat-Until-Success (RUS) protocol.

    This function constructs and executes the LCU protocol using the provided state preparation function
    and unitaries. It utilizes the :func:`qrisp.jasp.RUS` decorator from Jasp to handle the repeat-until-success
    mechanism, which repeatedly applies the LCU operation until the ancilla register is measured in the $|0\rangle$ state, indicating a successful implementation. The LCU algorithm enables the implementation of linear combinations of unitary operations
    on a quantum variable by probabilistically projecting onto the desired transformation. The terminal_sampling decorator is utilized to evaluate the LCU.

    For more details on the LCU primitive, refer to `Childs and Wiebe (2012) <https://arxiv.org/abs/1202.5822>`_.

    For more information on the inner workings of this LCU implementation, see :func:`inner_LCU`.

    Parameters
    ----------
    operand_prep : callable
        A function preparing the input state $\ket{\psi}$. This function must return a :ref:`QuantumVariable` (the ``operand``).
    state_prep : callable
        A function preparing the coefficient state from the $\ket{0}$ state.
        This function receives a :ref:`QuantumFloat` with $\lceil\log_2m\rceil$ qubits for $m$ unitiaries $U_0,\dotsc,U_{m-1}$ as argument and applies

        .. math ::

            \text{PREPARE}\ket{0} = \sum_i\sqrt{\frac{\alpha_i}{\lambda}}\ket{i}

    unitaries : list[callable] or callable
        Either:
          - A list of functions performing some in-place operation on ``operand``, or
          - A function ``unitaries(i, operand)`` performing some in-place operation on ``operand`` depending on a nonnegative integer index ``i`` specifying the case.

    num_unitaries : int, optional
        Required when ``unitaries`` is a callable to specify the number $m$ of unitaries.
    oaa_iter : int, optional
        The number of iterations of oblivious amplitude amplification to perform. The default is 0.

    Returns
    -------
    :ref:`QuantumVariable`
        A variable representing the output state $A\ket{\psi}$ after successful application of LCU to the input state $\ket{\psi}$.

    Raises
    ------
    TypeError
        If ``unitaries`` is not a list or callable.
    ValueError
        If ``num_unitaries`` is not specified when ``unitaries`` is a callable.

    See Also
    --------
    inner_LCU : Core LCU implementation without RUS.
    view_LCU : Generates the quantum circuit for visualization.

    Examples
    --------

    As a **first example**, we apply the non-unitary operator $A$ to the operand $\ket{\psi}=\ket{1}$ where

    .. math::

        A = \begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix} = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix} + \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix} = I + X

    This is,

    .. math:: A = \alpha_0U_0 + \alpha_1U_1

    where $\alpha_0=\alpha_1=1$, and $U_0=I$, $U_1=X$.

    Accordingly, we define the unitaries

    ::

        from qrisp import *

        def U0(operand):
            pass

        def U1(operand):
            x(operand)

        unitaries = [U0, U1]

    and the ``state_prep`` function implementing

    .. math ::

        \text{PREPARE}\ket{0} = \frac{1}{\sqrt{2}}\left(\ket{0}+\ket{1}\right)

    ::

        def state_prep(case):
            h(case)

    Next, we define the ``operand_prep`` function preparing the state $\ket{\psi}=\ket{1}$

    ::

        def operand_prep():
            operand = QuantumVariable(1)
            x(operand)
            return operand

    Finally, we apply LCU

    ::

        @terminal_sampling
        def main():

            qv = LCU(operand_prep, state_prep, unitaries)
            return qv

    and simulate

    >>> main()
    {0: 0.5, 1: 0.5}


    As a **second example**, we apply the operator

    .. math::

        \cos(H) = \frac{e^{iH}+e^{-iH}}{2}

    for some :ref:`Hermitian operator <operators>` $H$ to the input state $\ket{\psi}=\ket{0}$.

    First, we define an operator $H$ and unitaries performing the Hamiltonian evolutions $e^{iH}$ and $e^{-iH}$.
    (In this case, Trotterization will perform Hamiltonian evolution exactly since the individual terms commute.)

    ::

        from qrisp import *
        from qrisp.operators import X,Y,Z

        H = Z(0)*Z(1) + X(0)*X(1)

        def U0(operand):
            H.trotterization(forward_evolution=False)(operand)

        def U1(operand):
            H.trotterization(forward_evolution=True)(operand)

        unitaries = [U0, U1]

    Next, we define the ``state_prep`` and ``operand_prep`` functions

    ::

        def state_prep(case):
            h(case)

        def operand_prep():
            operand = QuantumVariable(2)
            return operand

    Finally, we apply LCU

    ::

        @terminal_sampling
        def main():

            qv = LCU(operand_prep, state_prep, unitaries)
            return qv

    and simulate

    >>> main()
    {3: 0.85471756539818, 0: 0.14528243460182003}

    Let's compare to the classically calculated result:

    >>> A = H.to_array()
    >>> from scipy.linalg import cosm
    >>> print(cosm(A))
    [[ 0.29192658+0.j  0.        +0.j  0.        +0.j -0.70807342+0.j]
    [ 0.        +0.j  0.29192658+0.j  0.70807342+0.j  0.        +0.j]
    [ 0.        +0.j  0.70807342+0.j  0.29192658+0.j  0.        +0.j]
    [-0.70807342+0.j  0.        +0.j  0.        +0.j  0.29192658+0.j]]

    That is, starting in state $\ket{\psi}=\ket{0}=(1,0,0,0)$, we obtain

    >>> result = cosm(A)@(np.array([1,0,0,0]).transpose())
    >>> result = result/np.linalg.norm(result) # normalise
    >>> result = result**2 # compute measurement probabilities
    >>> print(result)
    [0.1452825+0.j 0.       +0.j 0.       +0.j 0.8547175-0.j]

    which are exactly the probabilities we obsered in the quantum simulation!

    """

    case_indicator, qv = inner_LCU(
        operand_prep, state_prep, unitaries, num_unitaries, oaa_iter
    )

    # Success condition
    success_bool = measure(case_indicator) == 0
    return success_bool, qv


# Apply the RUS decorator with the workaround in order to show in documentation
temp_docstring = LCU.__doc__
LCU = RUS(static_argnums=[3, 4])(LCU)
LCU.__doc__ = temp_docstring


def view_LCU(operand_prep, state_prep, unitaries, num_unitaries=None):
    r"""
    Generate and return the quantum circuit for the LCU algorithm without utilizing
    the Repeat-Until-Success (RUS) protocol.

    This function constructs the LCU primitive and returns the corresponding quantum
    circuit representation. It's useful for visualization and analysis of the
    underlying quantum circuit implementing the LCU algorithm.

    Parameters
    ----------
    operand_prep : callable
        A function preparing the input state $\ket{\psi}$. This function must return a :ref:`QuantumVariable` (the ``operand``).
    state_prep : callable
        A function preparing the coefficient state from the $\ket{0}$ state.
        This function receives a :ref:`QuantumFloat` with $\lceil\log_2m\rceil$ qubits for $m$ unitiaries $U_0,\dotsc,U_{m-1}$ as argument and applies

        .. math ::

            \text{PREPARE}\ket{0} = \sum_i\sqrt{\frac{\alpha_i}{\lambda}}\ket{i}

    unitaries : list[callable] or callable
        Either:
          - A list of functions performing some in-place operation on ``operand``, or
          - A function ``unitaries(i, operand)`` performing some in-place operation on ``operand`` depending on a nonnegative integer index ``i`` specifying the case.

    num_unitaries : int, optional
        Required when ``unitaries`` is a callable to specify the number $m$ of unitaries.

    Returns
    -------
    :ref:`QuantumCircuit`
        Quantum circuit object showing the LCU implementation details.

    See Also
    --------
    inner_LCU : Core LCU implementation.
    LCU : Full LCU implementation using the RUS protocol.

    Examples
    --------

    ::

        from qrisp import *

        def U0(operand):
        y(operand)

        def U1(operand):
            x(operand)

        unitaries = [U0, U1]

        def state_prep(case):
            h(case)

        def operand_prep():
            operand = QuantumVariable(1)
            y(operand)
            return operand

        qc = view_LCU(operand_prep, state_prep, unitaries)

    >>> print(qc)
            ┌───┐                     ┌───┐                        »
    qb_287: ┤ Y ├─────────────────────┤ Y ├────────────────────────»
            ├───┤┌───────────────────┐└─┬─┘┌──────────────────────┐»
    qb_288: ┤ H ├┤0                  ├──┼──┤0                     ├»
            └───┘│  jasp_balauca_mcx │  │  │  jasp_balauca_mcx_dg │»
    qb_289: ─────┤1                  ├──■──┤1                     ├»
                 └───────────────────┘     └──────────────────────┘»
    «                             ┌───┐
    «qb_287: ─────────────────────┤ X ├─────────────────────────────
    «        ┌───────────────────┐└─┬─┘┌──────────────────────┐┌───┐
    «qb_288: ┤0                  ├──┼──┤0                     ├┤ H ├
    «        │  jasp_balauca_mcx │  │  │  jasp_balauca_mcx_dg │└───┘
    «qb_289: ┤1                  ├──■──┤1                     ├─────
    «        └───────────────────┘     └──────────────────────┘

    We can see that the operand and case variables are prepared, the unitaries for the two cases (i.e., X and Y) are executed, and the inverse preparation (H gate) of the case variable is applied.
    The inner workings of the circuit can be further analyzed by calling ``qc.transpile(level: int)``.

    """

    if not callable(unitaries):
        if not isinstance(unitaries, list):
            raise TypeError("unitaries must be callable or list[callable].")
        num_unitaries = len(unitaries)
    else:
        if num_unitaries == None:
            raise ValueError(
                "The number of unitiaries must be specified if unitaries is callable."
            )

    jaspr = make_jaspr(inner_LCU)(operand_prep, state_prep, unitaries, num_unitaries)

    # Convert Jaspr to quantum circuit and return the circuit
    return jaspr.to_qc(num_unitaries)[-1].transpile(3)
