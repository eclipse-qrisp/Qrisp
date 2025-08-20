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
    QuantumArray,
    QuantumVariable,
    gate_wrap,
    gphase,
    h,
    mcx,
    mcp,
    z,
    conjugate,
    invert,
    control,
)
from qrisp.jasp import jlen


@gate_wrap(permeability=[], is_qfree=False)
def reflection(qargs, state_function=None, phase=np.pi, reflection_indices=None):
    r"""
    Applies a reflection around a state $\ket{\psi}$ of (multiple) QuantumVariables, i.e.,

    .. math::

        R = ((1-e^{i\phi})\ket{\psi}\bra{\psi}-\mathbb I) = U^{\dagger}((1-e^{i\phi})\ket{0}\bra{0}-I)U,

    where $\ket{psi} = U\ket{0}$.

    
    Parameters
    ----------
    qargs : QuantumVariable | QuantumArray | list[QuantumVariable | QuantumArray]
        The (list of) QuantumVariables to apply the reflection on.
    state_function : function, optional
        A Python function preparing the state $\ket{\psi}$ around which to reflect.
        By default, the function prepares the uniform superposition state.
    phase : float or sympy.Symbol, optional
        Specifies the phase shift. The default is $\pi$.
    refection_indices : list[int], optional
        A list of indices indicating with respect to which variables the reflection is performed. 
        This is used for `oblivious amplitude amplification <https://arxiv.org/pdf/1312.1414>`_.
        Indices correspond to the flattened ``qargs``, e.g., if ``qargs = QuantumArray(QuantumFloat(3), (6,))``,
        ``reflection_indices=[0,1,2,3]`` corresponds to the first four variables in the array.
        By default, the reflection is performed with respect to all variables in ``qargs``.

    Examples
    --------

    

    """
    
    # Convert qargs into a list
    if isinstance(qargs, (QuantumVariable, QuantumArray)):
        qargs = [qargs]

    # Generate a (flat) list of all QuantumVariables in input_object
    flattened_qargs = []

    for arg in qargs:
        if isinstance(arg, QuantumVariable):
            flattened_qargs.append(arg)

        elif isinstance(arg, QuantumArray):
            flattened_qargs.extend([qv for qv in arg.flatten()])

        else:
            raise TypeError("Arguments must be of type QuantumVariable or QuantumArray")
        

    if reflection_indices is None:
        reflection_indices = range(len(flattened_qargs))

    qubits = sum([flattened_qargs[i].reg for i in reflection_indices], [])  


    if state_function is not None:

        def inv_state_function(args):
            with invert():
                state_function(*args)

    else:

        def inv_state_function(args):
            [h(qv) for qv in args]
        

    with conjugate(inv_state_function)(qargs):

        with control(phase == np.pi):

            with control(jlen(qubits) == 1):
                z(qubits[0])

            with control(jlen(qubits) > 1):
                h(qubits[-1])
                mcx(qubits[:-1], qubits[-1], ctrl_state=0)
                h(qubits[-1])

        with control(phase != np.pi):

            mcp(phase, qubits, ctrl_state=0)


    gphase(np.pi, qargs[0][0])

