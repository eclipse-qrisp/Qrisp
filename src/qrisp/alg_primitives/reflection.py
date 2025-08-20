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
    QuantumFloat,
    gate_wrap,
    gphase,
    h,
    mcx,
    mcp,
    mcz,
    p,
    x,
    z,
    merge,
    recursive_qs_search,
    conjugate,
    invert,
    control,
    IterationEnvironment,
)
from qrisp.jasp import check_for_tracing_mode, jrange


# Applies the grover diffuser onto the (list of) quantum variable input_object
def reflection(input_object, phase=np.pi, state_function=None, reflection_indices=None):
    r"""
    Applies a reflection onto (multiple) QuantumVariables.

    Parameters
    ----------
    input_object : QuantumVariable or list[QuantumVariable]
        The (list of) QuantumVariables to apply the Grover diffuser on.
    phase : float or sympy.Symbol, optional
        Specifies the phase shift. The default is $\pi$, i.e. a
        multi-controlled Z gate.
    state_function : function, optional
        A Python function preparing the initial state.
        By default, the function prepares the uniform superposition state.
    refection_indices : list[int], optional
        A list indicating with respect to which variables the reflection is performed.
        By default, the reflection is performed with respect to all variables in ``input_object``.

    Examples
    --------

    We apply the Grover diffuser onto several QuantumChars:

    >>> from qrisp import QuantumChar
    >>> from qrisp.grover import diffuser
    >>> q_ch_list = [QuantumChar(), QuantumChar(), QuantumChar()]
    >>> diffuser(q_ch_list)
    >>> print(q_ch_list[0].qs)

    .. code-block:: none

                  ┌────────────┐
        q_ch_0.0: ┤0           ├
                  │            │
        q_ch_0.1: ┤1           ├
                  │            │
        q_ch_0.2: ┤2           ├
                  │            │
        q_ch_0.3: ┤3           ├
                  │            │
        q_ch_0.4: ┤4           ├
                  │            │
        q_ch_1.0: ┤5           ├
                  │            │
        q_ch_1.1: ┤6           ├
                  │            │
        q_ch_1.2: ┤7  diffuser ├
                  │            │
        q_ch_1.3: ┤8           ├
                  │            │
        q_ch_1.4: ┤9           ├
                  │            │
        q_ch_2.0: ┤10          ├
                  │            │
        q_ch_2.1: ┤11          ├
                  │            │
        q_ch_2.2: ┤12          ├
                  │            │
        q_ch_2.3: ┤13          ├
                  │            │
        q_ch_2.4: ┤14          ├
                  └────────────┘

    """
    
    

    if isinstance(input_object, (QuantumVariable, QuantumArray)):
        input_object = [input_object]

    # Generate a list of all QuantumVariables in input_object
    flattened_qarg_list = []

    for arg in input_object:
        if isinstance(arg, QuantumVariable):
            flattened_qarg_list.append(arg)

        elif isinstance(arg, QuantumArray):
            flattened_qarg_list.extend([qv for qv in arg.flatten()])

        else:
            raise TypeError("Arguments must be of type QuantumVariable or QuantumArray")


    if isinstance(input_object, (list, tuple)) and reflection_indices is None:
        reflection_indices = [i for i in range(len(input_object))]

 
    def inv_state_function(args):
        with invert():
            state_function(*args)


    if isinstance(input_object, (list, tuple)):
        with conjugate(inv_state_function)(input_object):
            if check_for_tracing_mode():
                tag_state({input_object[i]: 0 for i in reflection_indices}, phase=phase)
            else:
                tag_state(
                    {
                        input_object[i]: "0" * input_object[i].size
                        for i in reflection_indices
                    },
                    binary_values=True,
                    phase=phase,
                )
        gphase(np.pi, input_object[0][0])
    else:
        with conjugate(inv_state_function)(input_object):
            if check_for_tracing_mode():
                tag_state({input_object: 0}, phase=phase)
            else:
                tag_state(
                    {input_object: input_object.size * "0"},
                    binary_values=True,
                    phase=phase,
                )
        gphase(np.pi, input_object[0])

    return input_object