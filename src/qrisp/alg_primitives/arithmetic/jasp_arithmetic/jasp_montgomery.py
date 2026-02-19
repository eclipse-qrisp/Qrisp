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

from typing import Union

from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder
from qrisp.qtypes import QuantumFloat, QuantumModulus
from qrisp.jasp import jrange, check_for_tracing_mode, jlen, q_cond
from qrisp.environments import control, invert, custom_control
from qrisp.core import swap, cx, x

from .jasp_bigintiger import BigInteger
from .jasp_mod_tools import (
    montgomery_encoder,
    montgomery_decoder,
    modinv,
    best_montgomery_shift,
)


def q_montgomery_reduction(
    qf: QuantumFloat, N: Union[int, BigInteger], m: int, inpl_adder=gidney_adder
):
    """
    Perform the Montgomery reduction of a concatenated QuantumFloat in-place.

    Layout
    ------
    qf = aux[:] + res[:]
    - aux has m+1 qubits (for u-tilde including the folded sign bit),
    - res has n qubits (holds the reduced result).

    Algorithm
    ---------
    - Estimation (m steps): For each LSB, conditionally subtract floor(N/2) on the
      truncated slice (implicit right-shift).
    - Correction: If negative, add N to the result part; fold the sign bit into
      aux via a single CNOT to produce u-tilde (m+1 bits).

    Parameters
    ----------
    qf : QuantumFloat
        The concatenated QuantumFloat (aux || res).
    N : int or BigInteger
        Modulus (must be odd).
    m : int
        Exponent m of the auxiliary radix R = 2^m.
    inpl_adder : Callable
        In-place adder to use during computation (defaults to gidney_adder).
    """
    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range

    # Estimation stage
    for j in xrange(m):
        with control(qf[j]):
            with invert():
                inpl_adder((N - 1) >> 1, qf[j + 1 :])

    # Correction stage
    with control(qf[-1]):
        inpl_adder(N, qf[m:-1])

    # Move sign bit across the result into the aux MSB (n swaps)
    for i in xrange(jlen(qf) - m - 1):
        swap(qf[-1 - i], qf[-2 - i])

    # Fold sign with LSB of estimate to form u-tilde
    cx(qf[m + 1], qf[m])


def cq_montgomery_multiply(
    X: Union[int, BigInteger],
    y: QuantumFloat,
    N: Union[int, BigInteger],
    m: int,
    inpl_adder=gidney_adder,
    x_is_montgomery: bool = False,
    res=None,
):
    """
    Montgomery product of a classical X and a QuantumFloat y: X*y*R^{-1} mod N.

    Outline
    -------
    - If needed, Montgomery-encode X using R = 2^m.
    - Accumulate reduced partial products (X*2^j mod N) into wqf = aux || res.
    - Apply q_montgomery_reduction(wqf, N, m).
    - Uncompute aux (size m+1) by adding ((X*2^j mod N) * N^{-1} mod 2^{m+1})
      controlled by y_j, with invert().

    Important
    ---------
    In the quantum-classical case (Rines & Chuang, 2018), the quantum input y
    can remain in standard (non-Montgomery) representation, and the output is
    returned in standard representation:
      output = X*y mod N

    Parameters
    ----------
    X : int or BigInteger
        Classical multiplicand. If not already in Montgomery form, set
        x_is_montgomery=False (default), which will encode it.
    y : QuantumFloat
        Quantum multiplicand (n qubits, standard representation).
    N : int or BigInteger
        Odd modulus.
    m : int
        Exponent m of the auxiliary radix R = 2^m.
    inpl_adder : Callable
        In-place adder to use during computation (defaults to gidney_adder).
    x_is_montgomery : bool
        If the classical X is already in Montgomery form. Defaults to False.
    res : QuantumFloat or None
        Optional target result register (n qubits). Allocated if None.

    Returns
    -------
    QuantumFloat
        The Montgomery product X*y mod N in standard representation.
    """
    # Build R = 2^m with width matching X if BigInteger
    if isinstance(X, BigInteger):
        R = BigInteger.create(1, X.digits.shape[0]) << m
    else:
        R = 1 << m

    if not x_is_montgomery:
        X = montgomery_encoder(X, R, N)

    # N^{-1} modulo 2^{m+1}
    N1 = modinv(N, R << 1)

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range
    n = jlen(y)
    if res is None:
        res = QuantumFloat(n)
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    # Multiplication: sum_j y_j * (X*2^j mod N) into wqf
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder((X << j) % N, wqf)

    # Reduction
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)

    # Uncompute aux: add ((X*2^j mod N)*N^{-1} mod 2^{m+1})
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            with invert():
                inpl_adder(((X << j) % N) * N1, aux[:])

    aux.delete()
    return res


@custom_control
def cq_montgomery_multiply_inplace(
    X: Union[int, BigInteger],
    y: QuantumFloat,
    N: Union[int, BigInteger],
    m: int,
    inpl_adder=gidney_adder,
    x_is_montgomery: bool = False,
    ctrl=None,
):
    """
    Montgomery product of a classical X and a QuantumFloat y, in-place on y.

    Notes
    -----
    - y remains in standard representation; the output is standard (X*y mod N).

    Parameters
    ----------
    X : int or BigInteger
        Integer factor of the Montgomery product.
    y : QuantumFloat
        Quantum factor of the Montgomery product (overwritten in-place).
    N : int or BigInteger
        Modulus of the Montgomery reduction.
    m : int
        Exponent m of the auxiliary radix R = 2^m.
    inpl_adder : Callable
        In-place adder to use during computation.
    x_is_montgomery : bool
        If the classical input X is already in Montgomery form. Defaults to False.
    ctrl : QuantumBit or None
        Optional external control for the in-place operation.
    """

    with control(X != 1):
        tmp = QuantumFloat(y.size)

        if ctrl is not None:
            x(ctrl)
            with control(ctrl):  # TODO: Add invert=True
                for i in jrange(y.size):
                    swap(tmp[i], y[i])
            x(ctrl)

        cq_montgomery_multiply(X, y, N, m, inpl_adder, x_is_montgomery, tmp)

        if x_is_montgomery:
            if isinstance(X, BigInteger):
                R = BigInteger.create(1, X.digits.shape[0]) << m
                X = montgomery_decoder(X, R, N)
            else:
                X = montgomery_decoder(X, 1 << m, N)
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


def qq_montgomery_multiply(
    x: QuantumFloat, y: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder
):
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
        Exponent $m$ of the auxiliary radix $R=2^m$ (see best_montgomery_shift).
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
                inpl_adder(cl_int // 2, operand[size - 1 - i :])

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
                inpl_adder(cl_int // 2, operand[size - 1 - i :])

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


def cq_montgomery_mat_multiply(A, B, out):
    if check_for_tracing_mode():
        xrange = jrange
        x_cond = q_cond
    else:
        xrange = range

        def x_cond(condition, tf, ff):
            if condition:
                tf()
            else:
                ff()

    n1 = A.shape[0]
    n2 = B.shape[1]

    m = A.shape[1]

    # out = QuantumArray(qtype=A[0,0], shape=(n1, n2))

    for k in xrange(m):
        for j in xrange(n2):
            for i in xrange(n1):

                def true_fun():
                    best_montgomery_shift(B[k, j], A[i, k].modulus)
                    shift = best_montgomery_shift(B[k, j], A[i, k].modulus)
                    aux = cq_montgomery_multiply(
                        B[k, j], A[i, k], A[i, k].modulus, shift
                    )
                    out[i, j] += aux
                    with invert():
                        cq_montgomery_multiply(
                            B[k, j], A[i, k], A[i, k].modulus, shift, res=aux
                        )
                    aux.delete()

                x_cond(B[k, j] != 0, true_fun, lambda: None)

    return out
