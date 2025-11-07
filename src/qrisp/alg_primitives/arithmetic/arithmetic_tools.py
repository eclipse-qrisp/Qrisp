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

from qrisp.core import Qubit
from qrisp.qtypes import QuantumFloat, QuantumBool
from qrisp.environments import control, invert, conjugate
from qrisp.core.gate_application_functions import x, cx, mcx
from qrisp.jasp import jrange, jnp
from typing import Tuple


def q_max(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:
    """
    Computes the maximum of two QuantumFloats ``a`` and ``b``.
    
    Parameters
    ----------
    a : QuantumFloat
    b : QuantumFloat

    Returns
    -------
    QuantumFloat
        The maximum value between ``a`` and ``b``.

    Examples
    --------
    >>> from qrisp import *
    >>> a = QuantumFloat(2)
    >>> b = QuantumFloat(2)
    >>> a += 2
    >>> h(a[0])
    >>> b+=1
    >>> res_max = q_max(a,b)
    >>> multi_measurement([a,b,res_max])
    {(2, 1, 2): 0.5, (3, 1, 3): 0.5}
    """

    res_mshape = [
        jnp.minimum(a.mshape[0], b.mshape[0]),
        jnp.maximum(a.mshape[1], b.mshape[1]),
    ]
    new_msize = res_mshape[1] - res_mshape[0]
    new_exponent = res_mshape[0]
    res = QuantumFloat(new_msize, new_exponent, signed=a.signed | b.signed)

    # We calculate the overlap of bits with the same significance of a and b.
    l = jnp.maximum(a.mshape[0], b.mshape[0])
    r = jnp.minimum(a.mshape[1], b.mshape[1])

    c = QuantumBool()
    
    compare_func = lambda x, y: x >= y
    injecteted_comp_func = c << compare_func

    with conjugate(injecteted_comp_func)(a, b):
        cx(a[l - a.exponent : r - a.exponent], res[l - new_exponent : r - new_exponent])

        with conjugate(cx)(
            a[l - a.exponent : r - a.exponent], b[l - b.exponent : r - b.exponent]
        ):
            with control(c, 0):
                cx(
                    b[l - b.exponent : r - b.exponent],
                    res[l - new_exponent : r - new_exponent],
                )

        with control(c):
            up_bd = jnp.minimum(a.size,l - a.exponent)
            cx(a[:up_bd], res[:up_bd])
            lw_bd = jnp.maximum(r, a.exponent)
            cx(a[lw_bd - a.exponent : a.msize], res[lw_bd - new_exponent : a.mshape[1] - new_exponent])
            if a.signed:
                cx(a.sign(), res.sign())

        with control(c, 0):
            up_bd = jnp.minimum(b.size,l - b.exponent)
            cx(b[: up_bd], res[: up_bd])
            lw_bd = jnp.maximum(r, b.exponent)
            cx(b[lw_bd - b.exponent : b.msize], res[lw_bd - new_exponent : b.mshape[1] - new_exponent])
            if b.signed:
                cx(b.sign(), res.sign())

    c.delete()

    return res

def q_min(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:
    """
    Computes the minimum of two QuantumFloats ``a`` and ``b``.
    
    Parameters
    ----------
    a : QuantumFloat
    b : QuantumFloat

    Returns
    -------
    QuantumFloat
        The minimum value between ``a`` and ``b``.

    Examples
    -------- 
    >>> from qrisp import *
    >>> a = QuantumFloat(2)
    >>> b = QuantumFloat(2)
    >>> a += 2
    >>> h(a[0])
    >>> b+=1
    >>> res_min = q_min(a,b)
    >>> multi_measurement([a,b,res_min])
    {(2, 1, 1): 0.5, (3, 1, 1): 0.5}
    """
    res_mshape = [
        jnp.minimum(a.mshape[0], b.mshape[0]),
        jnp.maximum(a.mshape[1], b.mshape[1]),
    ]
    new_msize = res_mshape[1] - res_mshape[0]
    new_exponent = res_mshape[0]
    res = QuantumFloat(new_msize, new_exponent, signed=a.signed | b.signed)

    # We calculate the overlap of bits with the same significance of a and b.
    l = jnp.maximum(a.mshape[0], b.mshape[0])
    r = jnp.minimum(a.mshape[1], b.mshape[1])

    c = QuantumBool()

    compare_func = lambda x, y: x <= y
    injecteted_comp_func = c << compare_func

    with conjugate(injecteted_comp_func)(a, b):
        cx(a[l - a.exponent : r - a.exponent], res[l - new_exponent : r - new_exponent])

        with conjugate(cx)(
            a[l - a.exponent : r - a.exponent], b[l - b.exponent : r - b.exponent]
        ):
            with control(c, 0):
                cx(
                    b[l - b.exponent : r - b.exponent],
                    res[l - new_exponent : r - new_exponent],
                )

        with control(c):
            up_bd = jnp.minimum(a.size,l - a.exponent)
            cx(a[:up_bd], res[:up_bd])
            lw_bd = jnp.maximum(r, a.exponent)
            cx(a[lw_bd - a.exponent : a.msize], res[lw_bd - new_exponent : a.mshape[1] - new_exponent])
            if a.signed:
                cx(a.sign(), res.sign())

        with control(c, 0):
            up_bd = jnp.minimum(b.size,l - b.exponent)
            cx(b[: up_bd], res[: up_bd])
            lw_bd = jnp.maximum(r, b.exponent)
            cx(b[lw_bd - b.exponent : b.msize], res[lw_bd - new_exponent : b.mshape[1] - new_exponent])
            if b.signed:
                cx(b.sign(), res.sign())

    c.delete()

    return res

def q_floor(a: QuantumFloat) -> QuantumFloat:
    """
    Computes out-of-place the floor of a QuantumFloat.
    
    Parameters
    ----------
    a : QuantumFloat

    Returns
    -------
    QuantumFloat
        The floor of ``a``.

    Examples
    --------
    >>> from qrisp import *
    >>> a = QuantumFloat(4,-2)
    >>> a[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    >>> b = q_floor(a)
    >>> b.get_measurement()
    {1.0: 0.75, 0.0: 0.25}
    """

    b = a.duplicate()
    for i in jrange(-a.exponent, a.size):
        cx(a[i], b[i])

    return b


def q_ceil(a: QuantumFloat) -> QuantumFloat:
    """
    Computes out-of-place the ceiling of a QuantumFloat.
    
    Parameters
    ----------
    a : QuantumFloat

    Returns
    -------
    QuantumFloat
        The ceiling of ``a``.

    Examples
    --------
    >>> from qrisp import *
    >>> a = QuantumFloat(4,-2)
    >>> a[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    >>> b = q_ceil(a)
    >>> b.get_measurement()
    {2.0: 0.75, 1.0: 0.25}

    .. warning::
        Ceiling operations that would result in overflow, raise no errors. Instead, the operations
        are performed `modular <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.
    """
    b = q_floor(a)

    with control(a.exponent < 0):
        flag = QuantumFloat(1)

        # Logical OR on the fractional part
        logical_OR(a[:-a.exponent], flag[0])

        with control(flag[0]):
            b += 1

        logical_OR(a[:-a.exponent], flag[0])
        flag.delete()
    return b


def logical_OR(operands: list[Qubit], flag: Qubit):
    # handle the case fo an empty list
    # Implement the case of a Quantum Variable with more the 2 qubits
    x(operands)
    x(flag)
    mcx(operands, flag)
    x(operands)

def q_round(a: QuantumFloat) -> QuantumFloat:
    """
    Computes out-of-place the rounding of a QuantumFloat to the first significant digit.
    
    Parameters
    ----------
    a : QuantumFloat

    Returns
    -------
    QuantumFloat
       The rounding of ``a``.

    Examples
    --------

    >>> from qrisp import *
    >>> a = QuantumFloat(4,-2)
    >>> a[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    >>> b = q_round(a)
    >>> b.get_measurement()
    {2.0: 0.75, 0.0: 0.25}

    .. warning::
        Rounding operations that would result in overflow, raise no errors. Instead, the operations
        are performed `modular <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.
    """
    b = q_floor(a)

    with control(a.exponent < 0):
        with control(a[-a.exponent - 1]): 
            #Control on the first fractional bit (the one holding the value b*0.5)
            #specifies if the number is rounded up or down.
            b += 1

    return b


def q_fractional(a: QuantumFloat) -> QuantumFloat:
    """
    Computes out-of-place the fractional part of a QuantumFloat.
    
    Parameters
    ----------
    a : QuantumFloat

    Returns
    -------
    QuantumFloat
       The fractional part of ``a``.

    Examples
    --------

    >>> from qrisp import *
    >>> a = QuantumFloat(4,-2)
    >>> a[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    >>> b = q_fractional(a)
    >>> b.get_measurement()
    {0.75: 0.75, 0.25: 0.25}

    """
    b = a.duplicate()
    for i in jrange(a.size):
        cx(a[i], b[i])

    with control(a.signed):
        flag = QuantumFloat(1)
        cx(a[-1], flag)
        with control(flag): 
            #If the QuantumFloat is signed and the value negative: substract the ceiling
            c = q_ceil(a)
            b -= c
            injected_qceil = c << q_ceil
            with invert():
                injected_qceil(a)
            c.delete()
            
        with control(flag, 0): 
            #If the QuantumFloat is signed and the value positive: substract the floor
            c = q_floor(a)
            b -= c
            injected_qfloor = c << q_floor
            with invert():
                injected_qfloor(a)
            c.delete()
        cx(a[-1], flag)
        flag.delete()
        
    with control(not a.signed):  
        #If the QuantumFloat is unsigned: substract the floor
        c = q_floor(a)
        b -= c
        injected_qfloor = c << q_floor
        with invert():
            injected_qfloor(a)
        c.delete()
    return b


def q_modf(a: QuantumFloat) -> Tuple[QuantumFloat, QuantumFloat]:
    """
    Computes out-of-place the integer and fractional parts of a QuantumFloat.

    Parameters
    ----------
    a : QuantumFloat

    Returns
    -------
    Tuple[QuantumFloat, QuantumFloat]
        A pair `(i, f)` where:

        - `i =` :func:`q_floor(a) <qrisp.q_floor>` is the integer part
        - `f =` :func:`q_fractional(a) <qrisp.q_fractional>` is the fractional part

        Both are new QuantumFloats with the same configuration as ``a``.

    Examples
    --------
    >>> from qrisp import *
    >>> a = QuantumFloat(4, -2)
    >>> a[:] = {0.25: 0.25**0.5, 1.75: 0.75**0.5}
    >>> i, f = qmodf(a)
    >>> i.get_measurement()
    {1.0: 0.75, 0.0: 0.25}
    >>> f.get_measurement()
    {0.75: 0.75, 0.25: 0.25}
    """
    return (q_floor(a), q_fractional(a))
