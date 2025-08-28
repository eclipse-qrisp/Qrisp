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

from typing import Union
from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder
from qrisp.qtypes import QuantumFloat, QuantumModulus
from qrisp.jasp import jrange, check_for_tracing_mode, jlen
from qrisp.environments import control, invert, custom_control
from .jasp_bigintiger import BigInteger
from .jasp_mod_tools import *
from qrisp.core import swap, cx, x


def compute_aux_radix_exponent(modulus, n=0):
    """
    Compute the exponent of the auxiliary radix for the montgomery reduction.

    Parameters
    ----------
    modulus : int
        The original modulus
    n : int
        The qubit size of the QuantumFloat

    Returns
    ----------
    The exponent of the $m$ of the auxiliary radix $R=2^m$.

    """
    return smallest_power_of_two(modulus)


def q_montgomery_reduction(qf: QuantumFloat, N: Union[int, BigInteger], m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery reduction of a QuantumFloat in-place

    Parameters
    ----------
    qf : QuantumFloat
        The QuantumFloat of which to compute the montgomery reduction.
    N : int
        Original modulus of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation, defaults to ``gidney_adder``.

    """

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range

    # Estimation stage
    for j in xrange(m):
        with control(qf[j]):
            with invert():
                inpl_adder((N-1) >> 1, qf[j+1:])

    # Correction stage
    with control(qf[-1]):
        inpl_adder(N, qf[m:-1])

    for i in xrange(jlen(qf)-m-+1):
        swap(qf[-1-i], qf[-1-i-1])
    cx(qf[m+1], qf[m])


def cq_montgomery_multiply(X: Union[int, BigInteger], y: QuantumFloat, N: Union[int, BigInteger], m: int, inpl_adder=gidney_adder,
                           x_is_montgomery: bool = False, res=None):
    """
    Perform the montgomery product of an integer and a QuantumFloat.


    Parameters
    ----------
    X : Union[int, BigInteger]
        Integer factor of the montgomery product.
    y : QuantumFloat
        Quantum factor of the montgomery product.
    N : Union[int, BigInteger]
        Original modulus of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
    x_is_montgomery: bool
        If the classical input X is already in montgomery form. Defaults to False.

    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    X_type = isinstance(X, BigInteger)
    N_type = isinstance(N, BigInteger)
    if (X_type or N_type) and not (X_type and N_type):
        raise ValueError("Both inputs must be BigInteger or none of them!")

    if X_type:
        R = BigInteger.from_int(1, X.digits.shape[0]) << m
    else:
        R = 1 << m
    if not x_is_montgomery:
        X = montgomery_encoder(X, R, N)
    N1 = modinv(N, R << 1)

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range
    n = jlen(y)
    if res is None:
        res = QuantumFloat(n)
    aux = QuantumFloat(m+1)
    wqf = aux[:] + res[:]

    # Multiplication
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder((X << j) % N, wqf)

    # Reduction
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)

    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            with invert():
                inpl_adder(((X << j) % N)*N1, aux[:])

    aux.delete()

    return res


@custom_control
def cq_montgomery_multiply_inplace(X: Union[int, BigInteger], y: QuantumFloat, N: Union[int, BigInteger], m: int, inpl_adder=gidney_adder,
                                   x_is_montgomery: bool = False, ctrl=None):
    """
    Perform the montgomery product of an integer and a QuantumFloat in-place

    Parameters
    ----------
    X : Union[int, BigInteger]
        Integer factor of the montgomery product.
    y : QuantumFloat
        Quantum factor of the montgomery product.
    N : Union[int, BigInteger]
        Original modulus of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
    x_is_montgomery: bool
        If the classical input X is already in montgomery form. Defaults to True.
    """

    X_type = isinstance(X, BigInteger)
    N_type = isinstance(N, BigInteger)
    if (X_type or N_type) and not (X_type and N_type):
        raise ValueError("Both inputs must be BigInteger or none of them!")

    tmp = QuantumFloat(y.size)

    if ctrl is not None:
        x(ctrl)
        with control(ctrl):  # TODO: Add invert=True
            for i in jrange(y.size):
                swap(tmp[i], y[i])
        x(ctrl)

    cq_montgomery_multiply(X, y, N, m, inpl_adder, x_is_montgomery, tmp)

    if x_is_montgomery:
        if X_type:
            R = BigInteger.from_int(1, X.digits.shape[0]) << m
            X = montgomery_decoder(X, R, N)
            X1 = modinv(X, N)
        else:
            X = montgomery_decoder(X, 1 << m, N)
            X1 = modinv(X, N)
    else:
        X1 = modinv(X, N)

    if ctrl is not None:
        x(ctrl)
        with control(ctrl):  # TODO: Add invert=True
            for i in jrange(y.size):
                swap(tmp[i], y[i])
        x(ctrl)

    with invert():
        cq_montgomery_multiply(X1, tmp, N, m, inpl_adder, False, y)

    if ctrl is not None:
        with control(ctrl, invert=False):
            for i in jrange(y.size):
                swap(tmp[i], y[i])
    else:
        for i in jrange(y.size):
            swap(tmp[i], y[i])

    tmp.delete()


def qq_montgomery_multiply(x: QuantumFloat, y: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of two QuantumFloats. Note that both QuantumFloats must be in montgomery form.

    Parameters
    ----------
    x : QuantumFloat
        First factor of the montgomery product.
    y : QuantumFloat
        Second factor of the montgomery product.
    N : int
        Original modulus of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation

    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range

    def qq_mul(ox: QuantumFloat, oy: QuantumFloat, ores: QuantumFloat):
        for i in xrange(jlen(oy)):
            with control(oy[i]):
                inpl_adder(ox[:], ores[i:])

    def qc_mul_inplace(operand, cl_int):
        size = jlen(operand)
        for i in xrange(size - 1):
            with control(operand[size - 2 - i]):
                inpl_adder(cl_int // 2, operand[size - 1 - i:])

    n = jlen(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    qq_mul(x, y, wqf[:-1])
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    qc_mul_inplace(aux, N)
    with invert():
        qq_mul(x, y, aux[:])
    aux.delete()

    return res


def qq_montgomery_multiply_modulus(x: QuantumModulus, y: QuantumModulus):
    """
    Perform the montgomery product of two QuantumModuli. Note that both QuantumModuli must be in montgomery form.
    Similiar to `qq_montgomery_multiply` but extracts the fields of the Moduli.
    Assumes that both QuantumModuli share the same modulus and the montgomery shift.

    Parameters
    ----------
    x : QuantumModulus
        First factor of the montgomery product.
    y : QuantumModulus
        Second factor of the montgomery product.

    Returns
    ----------
    QuantumModulus
        The mongomery product of the inputs.
    """

    inpl_adder = x.inpl_adder
    N = x.modulus
    m = x.m

    # res = qq_montgomery_multiply(x, y, N, m, inpl_adder)
    # return res

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range

    def qq_mul(ox: QuantumFloat, oy: QuantumFloat, ores: QuantumFloat):
        for i in xrange(jlen(oy)):
            with control(oy[i]):
                inpl_adder(ox[:], ores[i:])

    def qc_mul_inplace(operand, cl_int):
        size = jlen(operand)
        for i in xrange(size - 1):
            with control(operand[size - 2 - i]):
                inpl_adder(cl_int // 2, operand[size - 1 - i:])

    res = QuantumModulus(N)
    res.m = m
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    qq_mul(x, y, wqf[:-1])
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    qc_mul_inplace(aux, N)
    with invert():
        qq_mul(x, y, aux[:])
    aux.delete()

    return res


def montgomery_product(x, y, N, m, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of the two inputs

    ::

        x*y*2**(-m) mod N

    Parameters
    ----------
    x : QuantumFloat or Quantummodulus or integer
        First factor of the montgomery product.
    y : QuantumFloat or Quantummodulus or integer
        Second factor of the montgomery product.
    N : int
        Original modulus of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place quantum adder to use during computation.

    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    if isinstance(x, [int, float]) and isinstance(x, [int, float]):
        raise TypeError("Provide at least one Quantum input")
    elif isinstance(x, [int, float]):
        return cq_montgomery_multiply(x, y, N, m, inpl_adder=inpl_adder)
    elif isinstance(y, [int, float]):
        return cq_montgomery_multiply(y, x, N, m, inpl_adder=inpl_adder)
    else:
        return qq_montgomery_multiply(x, y, N, m, inpl_adder=inpl_adder)
