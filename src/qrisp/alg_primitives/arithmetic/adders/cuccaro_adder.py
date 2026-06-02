"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumFloat
from qrisp.environments import conjugate, custom_control
from qrisp.misc import int_encoder
from qrisp.jasp import jrange, jlen
import jax.numpy as jnp

@custom_control
def cuccaro_adder(a, b, c_out=None, ctrl = None):
    """In-place adder as introduced in https://arxiv.org/abs/quant-ph/0410184

    This function works in both static and dynamic modes. The allowed inputs are both quantum types or one classical
    type and one quantum type. Note that when the first input is larger than the second input, the function will perform
    modulo addition (relative to the size of the second input) after the first input is truncated to be the same size as
    the second input.

    The custom control implementation is based on Theorem 2.12 of https://arxiv.org/abs/2407.20167

    .. note::
    
        If the first input is quantum and the second classical, the function cannot work as addition is 
        performed "in-place" on the second input. 

    
    Parameters
    ----------
    a : int or QuantumVariable
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_out : QuantumVariable, optional
        An optional carry out value. The default is None.

    Raises
    ------
    ValueError
        If the inputs are not valid quantum or classical types.
    
    Returns
    -------
    None
        The function modifies the second input in place.
    
    Examples
    --------
    Static mode with both quantum inputs:

    >>> from qrisp import QuantumFloat, cuccaro_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> cuccaro_adder(a,b)
    >>> print(b)
    {9: 1.0}
    """

    # convert the classical input to a quantum input
    if not isinstance(a, QuantumVariable):
        # create a QuantumFloat of the same size as the other quantum input
        q_a = b.duplicate()

        with conjugate(int_encoder)(q_a, a):
            # begin with q_a in the state |a>
            if c_out is not None:
                cuccaro_adder(q_a, b, c_out = c_out)
            elif ctrl is not None:
                cuccaro_adder(q_a, b, ctrl = ctrl)
            elif ctrl is not None and c_out is not None:
                cuccaro_adder(q_a, b, c_out = c_out, ctrl = ctrl)
            else:
                cuccaro_adder(q_a, b)
        
        # outside the conjugation, q_a is back in the state |0> and the addition has been performed on b
        # delete the temporary quantum variable created for the classical input
        q_a.delete()
        return
    
    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")
    

    # when the inputs are of unequal length
    # pad the size of the input with the smaller size
    dim_a = a.size
    dim_b = b.size

    max_size = jnp.maximum(dim_a, dim_b)

    # reduce the size of a to the size of b if a is larger than b
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # create an extension ancilla to change the size of a when it is smaller than b
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a

    dim_a = jlen(a)
    dim_b = jlen(b)

    ancilla = QuantumFloat(max_size)

    if c_out is not None:
        ancilla2 = c_out

    # first maj gate application
    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    # iterator maj gate application
    from qrisp.jasp import jrange
    
    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    # cnot
    if c_out is not None:
        cx(a[-1], ancilla2[0])

    if ctrl is None:

        # iterator uma gate application
        for j in jrange(dim_a - 1):
            # reverse the iteration
            i = dim_a - j - 1

            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        # last uma gate application
        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])
    
    else:

        # iterator uma gate application
        for j in jrange(dim_a - 1):
            # reverse the iteration
            i = dim_a - j - 1

            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i-1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])
        
        # last uma gate application
        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])
    

    # delete the ancilla used for carry bits
    ancilla.delete()

    # delete the extension ancillas when the inputs are of unequal length
    extension_anc_a.delete()