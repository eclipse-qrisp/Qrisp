"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import *
from qrisp.jasp import *
import jax.numpy as jnp
import numpy as np


def inner_LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
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

    - **PREPARE**: Prepares an ancilla register in a superposition encoding the normalized coefficients $\alpha_i$ of the target operator $\mathrm{PREPARE}|0\rangle=\sum_i\sqrt{\frac{\alpha_i}{\lambda}}|i\rangle$.
    - **SELECT**: Applies the unitary $U_i$ to the target register, controlled on the ancilla register being in state $|i\rangle$. $\mathrm{SELECT}|i\rangle|\psi\rangle=|i\rangle U_i|\psi\rangle$.
    - **PREPARE**$^\dagger$: Uncomputes the ancilla.

    .. note::

        The LCU protocol is deemed successful only if the ancilla register is measured in the :math:`|0\rangle` state, which occurs with a probability proportional to :math:`\frac{|\alpha|_1^2}{\lambda^2}`. This function does not perform the measurement; it returns the ancilla register and the transformed target register.

    For a complete implementation of LCU with the Repeat-Until-Success protocol, see :func:`LCU`.

    For more details on the LCU protocol, refer to `Childs and Wiebe (2012) <https://arxiv.org/abs/1202.5822>`_, or `related seminars provided by Nathan Wiebe <https://www.youtube.com/watch?v=irMKrOIrHP4>`_.

    Parameters
    ----------
    state_prep : callable
        Function that prepares the ancilla register in the coefficient superposition. Must accept a :ref:`QuantumVariable` and apply the appropriate quantum gates.
    unitaries : list/tuple or callable
        Either:
          - A list or tuple of unitary functions, each acting on a :class:`QuantumVariable`, or
          - A callable that accepts an integer index and returns the corresponding unitary function.
    num_qubits : int
        Number of qubits for the target :ref:`QuantumFloat`.
    num_unitaries : int, optional
        Required if ``unitaries`` is a callable. Specifies the number of unitary terms in the sum.

    Returns
    -------
    tuple of (:ref:`QuantumVariable`, :ref:`QuantumVariable`)
          - **case_indicator**: Ancilla register encoding which unitary was selected.
          - **qv**: Target quantum register after the LCU operation.

    Raises
    ------
    TypeError
        If ``unitaries`` is not a list/tuple or a callable.
    ValueError
        If ``num_unitaries`` is not specified when ``unitaries`` is a callable.

    See Also
    --------
    LCU : Full LCU implementation using the RUS protocol.
    view_LCU : Generates the quantum circuit for visualization.
    """

    qv = QuantumFloat(num_qubits)

    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda index: unitaries[index]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError(
                "num_unitaries must be specified for dynamic unitaries"
            )
        unitary_func = unitaries

    else:
        raise TypeError("unitaries must be a list/tuple or a callable function")

    # Specify the QunatumVariable that indicates which case to execute
    n = jnp.int64(jnp.ceil(jnp.log2(num_unitaries)))
    case_indicator = QuantumFloat(n)

    # LCU protocol with conjugate preparation
    with conjugate(state_prep)(case_indicator):
        for i in range(num_unitaries):
            qb = QuantumBool()
            with conjugate(mcx)(case_indicator, qb, ctrl_state=i):
                with control(qb):
                    unitary_func(i)(qv)

    return case_indicator, qv


def LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
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
    state_prep : callable
        Quantum circuit function preparing the coefficient state.
    unitaries : list/tuple or callable
        Either:
          - A list or tuple of pre-defined unitary operations, each acting on a :class:`QuantumVariable`, or
          - A callable function that accepts an integer index and returns a unitary function.
    num_qubits : int
        Number of qubits for the target quantum register.
    num_unitaries : int, optional
        Required when ``unitaries`` is a callable to specify the number of unitary terms.

    Returns
    -------
    tuple (:ref:`QuantumBool`, :ref:`QuantumFloat`)
        - success_bool : Indicator of whether the protocol was successful (always `True` due to RUS).
        - qv : Output state after successful application of LCU to the initial state.

    Raises
    ------
    TypeError
        If ``unitaries`` is not a list, tuple, or callable.
    ValueError
        If ``num_unitaries`` is not specified when ``unitaries`` is a callable.

    See Also
    --------
    inner_LCU : Core LCU implementation without RUS.
    view_LCU : Generates the quantum circuit for visualization.

    """
    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda i: unitaries[i]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError(
                "num_unitaries must be specified for dynamic unitaries"
            )
        unitary_func = unitaries

    else:
        raise TypeError("unitaries must be a list/tuple or a callable function")

    case_indicator, qv = inner_LCU(state_prep, unitary_func, num_qubits, num_unitaries)

    # Success condition
    success_bool = measure(case_indicator) == 0
    return success_bool, qv

# Apply the RUS decorator with the workaround in order to show in documentation
temp_docstring = LCU.__doc__
LCU = RUS(static_argnums=[1, 2, 3])(LCU)
LCU.__doc__ = temp_docstring

def view_LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
    r"""
    Generate and return the quantum circuit for the LCU algorithm without utilizing
    the Repeat-Until-Success (RUS) protocol.

    This function constructs the LCU primitive and returns the corresponding quantum
    circuit representation. It's useful for visualization and analysis of the
    underlying quantum circuit implementing the LCU algorithm.

    Parameters
    ----------
    state_prep : callable
        State preparation function for LCU coefficients.
    unitaries : list/tuple or callable
        Either:
        - A list or tuple of unitary operations to visualize, or
        - A callable that accepts an integer index and returns a unitary
    num_qubits : int
        Number of qubits for the target quantum register.
    num_unitaries : int, optional
        Required when using callable-based unitaries to specify the number of unitary terms.

    Returns
    -------
    :ref:`QuantumCircuit`
        Quantum circuit object showing the LCU implementation details.

    See Also
    --------
    inner_LCU : Core LCU implementation.
    LCU : Full LCU implementation using the RUS protocol.

    """
    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda index: unitaries[index]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError(
                "num_unitaries must be specified for dynamic unitaries"
            )
        unitary_func = unitaries

    else:
        raise TypeError("unitaries must be list/tuple, or callable")

    jaspr = make_jaspr(inner_LCU)(state_prep, unitary_func, num_qubits, num_unitaries)

    # Convert Jaspr to quantum circuit and return the circuit
    return jaspr.to_qc(num_qubits, num_unitaries)[-1]
