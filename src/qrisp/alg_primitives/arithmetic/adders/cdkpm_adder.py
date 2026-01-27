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
from qrisp import *
from qrisp.jasp import jrange
import jax.numpy as jnp

ALLOWED_CDKPM_ADDER_QUANTUM_TYPES = (QuantumFloat, QuantumBool, QuantumModulus)


def cdkpm_adder(a, b, c_in=None, c_out=None):
    """In-place adder as introduced in https://arxiv.org/abs/quant-ph/0410184

    This function works in both static and dynamic modes. The allowed inputs are both quantum types or one quantum
    type and one classical type. 

    .. note::
    
        If the first input is quantum and the second classical, the addition is performed
        on the first input and the inputs are switched internally.

    
    Parameters
    ----------
    a : int or QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_in : Qubit, optional
        An optional carry in value. The default is None.
    c_out : Qubit, optional
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

    >>> from qrisp import QuantumFloat, cdkpm_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> cdkpm_adder(a,b)
    >>> print(b)
    {9: 1.0}
    """

    if not (
        # verify at least one input is classical and the other is quantum
        (
            isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)
            ^ isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)
        )
        or
        # verify both inputs are quantum
        (
            isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)
            and isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES)
        )
    ):
        raise ValueError("Attempted to call the CDKPM adder on invalid inputs.")

    # convert the classical input to a quantum input
    if not isinstance(a, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        # create a QuantumFloat of the same size as the other quantum input
        q_a = QuantumFloat(b.size)
        q_a[:] = a
        a = q_a
    
    # if the first input is quantum and the second is classical
    # we switch the inputs to make the second input always quantum given that
    # the sum is implemented on the second input
    elif not isinstance(b, ALLOWED_CDKPM_ADDER_QUANTUM_TYPES):
        q_b = QuantumFloat(a.size)
        q_b[:] = b
        b = q_b
        # switch the inputs
        a, b = b, a

    # when the inputs are of unequal length
    # pad the size of the input with the smaller size
    dim_a = a.size
    dim_b = b.size

    max_size = jnp.maximum(dim_a, dim_b)
    a.extend(abs(max_size - dim_a))
    b.extend(abs(max_size - dim_b))

    # carry bit is initialized to 0
    if c_in is None:
        ancilla = QuantumFloat(1)
        ancilla[:] = 0
    else:
        ancilla = [c_in]

    if c_out is None:
        ancilla2 = QuantumFloat(1)
    else:
        ancilla2 = [c_out]

    # first maj gate application
    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    # iterator maj gate application
    for i in jrange(1, a.size):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    # cnot
    cx(a[-1], ancilla2[-1])

    # iterator uma gate application
    for j in jrange(a.size - 1):
        # reverse the iteration
        i = a.size - j - 1

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

    if c_in is None:
        ancilla.delete()

    if c_out is None:
        ancilla2.delete()