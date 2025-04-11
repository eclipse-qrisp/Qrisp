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

@RUS(static_argnums=[1, 2, 3]) 
def LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
    """
    Implements the Linear Combination of Unitaries (LCU) protocol 
    https://arxiv.org/abs/1202.5822 utilizing the RUS (Repeat-Until-Success) 
    Jasp feature. This function constructs and executes the LCU protocol using 
    the provided state preparation function and unitaries, facilitating the
    implementation of linear combination of unitary operations on a QuantumVariable.

    The terminal_sampling decorator is utilized to evaluate the LCU.

    Parameters
    ----------
    state_prep : callable
        Quantum circuit preparing the initial state for LCU. 
    unitaries : list of tuple of callables, or callable
        Either a list or tuple of unitary functions, or a callable that returns a
        unitary function given an index. Specifies the unitary operations to combine. 
        Each unitary should be a callable acting on a QuantumVariable of size num_qubits.
    num_qubits : int
        Number of qubits required for the target QuantumVariable that unitaries
        act upon.
    num_unitaries : int, optional
        The number of unitaries. Must be specified if ``unitaries`` is a callable 
        function. If ``unitaries`` is a list or tuple, ``num_unitaries`` defaults 
        to ``len(unitaries)``.

    Returns
    -------
    success_bool : bool
        ``True`` if the LCU protocol succeeds (the measurement of the ancilla qubits yields zero); ``False`` otherwise.
    qv : :ref:`QuantumVariable`
        The quantum variable after the application of the LCU protocol.

    Raises
    ------
    ValueError
        If ``num_unitaries`` is not specified when ``unitaries`` is a callable function.
    TypeError
        If ``unitaries`` is neither a list, tuple, nor a callable function.
    """
    qv = QuantumFloat(num_qubits)

    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda index: unitaries[index]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError("num_unitaries must be specified for dynamic unitaries")
        unitary_func = unitaries
    
    else:
        raise TypeError("unitaries must be a list/tuple or a callable function")
    
    # Specify the QunatumVariable that indicates which case to execute
    n = int(np.ceil(np.log2(len(unitaries))))
    case_indicator = QuantumFloat(n)
    
    # Turn into a list of qubits
    case_indicator_qubits = [case_indicator[i] for i in range(n)]
    
    # LCU protocol with conjugate preparation
    with conjugate(state_prep)(case_indicator):
        # SELECT
        for i in range(num_unitaries):
            with control(case_indicator_qubits, ctrl_state=i):
                unitary = unitary_func(i)
                unitary(qv)  # Apply i-th unitary from unitary list
    
    # Success condition
    success_bool = (measure(case_indicator) == 0)
    
    return success_bool, qv
