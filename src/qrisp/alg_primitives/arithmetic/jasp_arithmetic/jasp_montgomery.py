# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Implements Jasp-traceable Montgomery reduction and multiplication (Rines & Chuang, 2018)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from qrisp.alg_primitives.arithmetic.adders import gidney_adder
from qrisp.core import QuantumArray, cx, swap, x
from qrisp.environments import control, custom_control, invert
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode, jlen, jrange, q_cond
from qrisp.qtypes import QuantumBool, QuantumFloat, QuantumModulus

from .jasp_bigintiger import BigInteger
from .jasp_mod_tools import best_montgomery_shift, modinv, montgomery_decoder, montgomery_encoder, smallest_power_of_two


def q_montgomery_reduction(
    qf: QuantumFloat | DynamicQubitArray, N: int | BigInteger, m: int, inpl_adder: Callable = gidney_adder
) -> None:
    """Perform the Montgomery reduction of a concatenated QuantumFloat in-place.

    Implements the quantum Montgomery reduction from Rines & Chuang (2018),
    https://arxiv.org/abs/1801.01081.

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


# The aux register of the Montgomery multiplication has m + 1 qubits. Up to this
# width, the constants added to it are computed with 64-bit integers.
_MAX_MACHINE_AUX_BITS = 32


def _low_bits(value: int | BigInteger | Array, num_bits: int | Array) -> int | Array:
    """Return ``value`` modulo ``2**num_bits``, for ``num_bits`` up to 32.

    Parameters
    ----------
    value : int, BigInteger or Array
        A non-negative integer.
    num_bits : int or Array
        The number of low bits to keep, at most 32.

    Returns
    -------
    int or Array
        The low bits, as a Python integer for Python inputs and an int64 array otherwise.

    """
    low = value.digits[0] if isinstance(value, BigInteger) else value
    if isinstance(low, (int, np.integer)) and isinstance(num_bits, (int, np.integer)):
        return int(low) & ((1 << int(num_bits)) - 1)
    mask = (jnp.uint64(1) << jnp.asarray(num_bits).astype(jnp.uint64)) - jnp.uint64(1)
    return (jnp.asarray(low).astype(jnp.uint64) & mask).astype(jnp.int64)


def _inverse_mod_power_of_two(N: int | BigInteger | Array, num_bits: int | Array) -> int | Array:
    """Return ``N**-1`` modulo ``2**num_bits`` for an odd ``N``, from its lowest bits.

    Parameters
    ----------
    N : int, BigInteger or Array
        An odd integer.
    num_bits : int or Array
        The number of bits of the modulus, at most 32.

    Returns
    -------
    int or Array
        The inverse, as a Python integer for Python inputs and an int64 array otherwise.

    """
    low = _low_bits(N, num_bits)
    if isinstance(low, int):
        return pow(low, -1, 1 << int(num_bits))
    low = low.astype(jnp.uint64)
    # An odd number is its own inverse modulo 8, and every Newton step doubles the
    # number of correct bits (3, 6, 12, 24, 48), which covers the 32 bits needed
    inverse = low
    for _ in range(4):
        inverse = inverse * (jnp.uint64(2) - low * inverse)
    return _low_bits(inverse, num_bits)


def _times_mod_power_of_two(
    a: int | BigInteger | Array, b: int | BigInteger | Array, num_bits: int | Array
) -> int | Array:
    """Return ``a * b`` modulo ``2**num_bits``, for ``num_bits`` up to 32, from the low bits of the factors.

    Parameters
    ----------
    a : int, BigInteger or Array
        A non-negative factor.
    b : int, BigInteger or Array
        A non-negative factor.
    num_bits : int or Array
        The number of bits of the modulus, at most 32.

    Returns
    -------
    int or Array
        The product modulo ``2**num_bits``.

    """
    a, b = _low_bits(a, num_bits), _low_bits(b, num_bits)
    if isinstance(a, int) and isinstance(b, int):
        return (a * b) & ((1 << int(num_bits)) - 1)
    # The product of two numbers below 2**32 fits into 64 unsigned bits
    product = jnp.asarray(a).astype(jnp.uint64) * jnp.asarray(b).astype(jnp.uint64)
    return _low_bits(product, num_bits)


@jax.jit
def _partial_product_table(X: BigInteger | Array, N: BigInteger | Array, n: Array) -> Array:
    """Return the table of the reduced partial products ``X * 2**j mod N``, for ``j = 0, ..., n - 1``.

    Every row is the previous one doubled modulo ``N``, which is cheap, instead of
    the long division ``(X << j) % N``. The table has a row for every possible
    ``j`` (the bits of ``X``), and the rows from ``n`` on are not used.

    Parameters
    ----------
    X : BigInteger or Array
        The classical factor, smaller than ``N``.
    N : BigInteger or Array
        The modulus. A JAX integer modulus must be smaller than ``2**62``, so that
        twice a partial product fits into 64 bits.
    n : Array
        The number of partial products.

    Returns
    -------
    Array
        The limbs of every partial product for a BigInteger ``X``, its values otherwise.

    """
    if isinstance(X, BigInteger):
        width = X.digits.shape[0]
        table = jnp.zeros((_LIMB_BITS * width, width), dtype=X.digits.dtype)

        def store_and_double(j: Array, carry: tuple[Array, BigInteger]) -> tuple[Array, BigInteger]:
            """Store the partial product of ``j`` and double it modulo ``N``.

            Parameters
            ----------
            j : Array
                The index of the partial product.
            carry : tuple
                The table and the partial product of ``j``.

            Returns
            -------
            tuple
                The table and the partial product of ``j + 1``.

            """
            table, product = carry
            table = table.at[j].set(product.digits)
            doubled = product << 1
            return table, BigInteger(jnp.where(doubled >= N, (doubled - N).digits, doubled.digits))

        table, _ = lax.fori_loop(0, n, store_and_double, (table, X % N))
        return table

    table = jnp.zeros(64, dtype=jnp.int64)

    def store_and_double_int(j: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        """Store the partial product of ``j`` and double it modulo ``N``.

        Parameters
        ----------
        j : Array
            The index of the partial product.
        carry : tuple
            The table and the partial product of ``j``.

        Returns
        -------
        tuple
            The table and the partial product of ``j + 1``.

        """
        table, product = carry
        table = table.at[j].set(product)
        doubled = product * 2
        return table, jnp.where(doubled >= N, doubled - N, doubled)

    modulus = jnp.asarray(N, dtype=jnp.int64)
    table, _ = lax.fori_loop(0, n, store_and_double_int, (table, jnp.asarray(X, dtype=jnp.int64) % modulus))
    return table


_LIMB_BITS = 32


def _montgomery_radix(X: int | BigInteger, m: int) -> int | BigInteger:
    """Build the Montgomery radix R = 2^m, matching X's limb width if X is a BigInteger."""
    if isinstance(X, BigInteger):
        return BigInteger.create(1, X.digits.shape[0]) << m
    return 1 << m


def cq_montgomery_multiply(
    X: int | BigInteger,
    y: QuantumFloat,
    N: int | BigInteger,
    m: int,
    inpl_adder: Callable = gidney_adder,
    x_is_montgomery: bool = False,
    res: QuantumFloat | None = None,
) -> QuantumFloat:
    """Montgomery product of a classical X and a QuantumFloat y: X*y*R^{-1} mod N.

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

    Examples
    --------
    Computes 5*11 mod 97 = 55:

    >>> from qrisp import QuantumFloat, boolean_simulation, gidney_adder, measure
    >>> from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import best_montgomery_shift
    >>> @boolean_simulation
    ... def cq(X, y, n, N):
    ...     qy = QuantumFloat(n)
    ...     qy[:] = y
    ...     m = best_montgomery_shift(X, N)
    ...     res = cq_montgomery_multiply(X, qy, N, m, gidney_adder)
    ...     return measure(res)
    >>> cq(5, 11, (11).bit_length(), 97)  # doctest: +SKIP
    55

    """
    # Build R = 2^m with width matching X if BigInteger
    R = _montgomery_radix(X, m)

    # TODO (this is a bug that should be solved): under @boolean_simulation,
    # x_is_montgomery is traced instead of staying a static Python bool, so
    # this branch raises TracerBoolConversionError when called with
    # x_is_montgomery=True (e.g. via cq_montgomery_multiply_inplace).
    if not x_is_montgomery:
        X = montgomery_encoder(X, R, N)

    # N^{-1} modulo 2^{m+1}. The aux register has m + 1 qubits, so only the low
    # m + 1 bits of the constants added to it matter, which machine integers hold
    # for the shifts used in practice (m <= 31).
    small_aux = not isinstance(m, (int, np.integer)) or m + 1 <= _MAX_MACHINE_AUX_BITS
    if small_aux:
        N1 = _inverse_mod_power_of_two(N, m + 1)
    else:
        N1 = modinv(N, R << 1)

    if check_for_tracing_mode():
        xrange = jrange
    else:
        xrange = range
    n = jlen(y)
    if res is None:
        res = QuantumModulus(N)
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    # The reduced partial products X*2^j mod N. Under tracing they are computed
    # once, by doubling, into a table indexed by j: the loops below run backwards
    # under invert(), so they cannot carry a running value.
    table = _partial_product_table(X, N, n) if check_for_tracing_mode() else None

    def partial_product(j: int | Array) -> int | BigInteger | Array:
        """Return the reduced partial product ``X * 2**j mod N``.

        Parameters
        ----------
        j : int or Array
            The index of the partial product.

        Returns
        -------
        int, BigInteger or Array
            The partial product.

        """
        if table is None:
            return (X << j) % N
        return BigInteger(table[j]) if isinstance(X, BigInteger) else table[j]

    # Multiplication: sum_j y_j * (X*2^j mod N) into wqf
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder(partial_product(j), wqf)

    # Reduction
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)

    # Uncompute aux: add ((X*2^j mod N)*N^{-1} mod 2^{m+1})
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            with invert():
                if small_aux:
                    inpl_adder(_times_mod_power_of_two(partial_product(j), N1, m + 1), aux[:])
                else:
                    inpl_adder(partial_product(j) * N1, aux[:])

    aux.delete()
    return res


@custom_control
def cq_montgomery_multiply_inplace(
    X: int | BigInteger,
    y: QuantumFloat,
    N: int | BigInteger,
    m: int,
    inpl_adder: Callable = gidney_adder,
    x_is_montgomery: bool = False,
    ctrl: QuantumBool | None = None,
    X_inverse: int | BigInteger | Array | None = None,
) -> None:
    """Montgomery product of a classical X and a QuantumFloat y, in-place on y.

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
    X_inverse : int, BigInteger or Array, optional
        The inverse of X modulo N, in standard representation. The uncomputation
        multiplies by it, and computes it with ``modinv`` if it is not given, which
        is the most expensive classical step of the multiplication.

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

        inverse = X_inverse
        if inverse is None:
            if x_is_montgomery:
                X = montgomery_decoder(X, _montgomery_radix(X, m), N)
            inverse = modinv(X, N)

        if ctrl is not None:
            x(ctrl)
            with control(ctrl):  # TODO: Add invert=True
                for i in jrange(y.size):
                    swap(tmp[i], y[i])
            x(ctrl)

        with invert():
            cq_montgomery_multiply(inverse, tmp, N, m, inpl_adder, False, y)

        if ctrl is not None:
            with control(ctrl, invert=False):
                for i in jrange(y.size):
                    swap(tmp[i], y[i])
        else:
            for i in jrange(y.size):
                swap(tmp[i], y[i])

        tmp.delete()


def _qq_mul(ox: QuantumFloat, oy: QuantumFloat, ores: QuantumFloat | DynamicQubitArray, inpl_adder: Callable) -> None:
    """Accumulate ox * oy into ores via repeated controlled in-place addition."""
    xrange = jrange if check_for_tracing_mode() else range
    for i in xrange(jlen(oy)):
        with control(oy[i]):
            inpl_adder(ox[:], ores[i:])


def _qc_mul_inplace(operand: QuantumFloat, cl_int: int | BigInteger, inpl_adder: Callable) -> None:
    """Multiply operand by the classical cl_int in place via repeated controlled in-place addition."""
    xrange = jrange if check_for_tracing_mode() else range
    size = jlen(operand)
    for i in xrange(size - 1):
        with control(operand[size - 2 - i]):
            inpl_adder(cl_int // 2, operand[size - 1 - i :])


def qq_montgomery_multiply(
    x: QuantumFloat, y: QuantumFloat, N: int | BigInteger, m: int, inpl_adder: Callable = gidney_adder
) -> QuantumFloat:
    """Perform the montgomery product of two QuantumFloats. Note that both QuantumFloats must be in montgomery form.

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
    -------
    QuantumFloat
        The mongomery product of the inputs.

    """
    n = jlen(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    _qq_mul(x, y, wqf[:-1], inpl_adder)
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    _qc_mul_inplace(aux, N, inpl_adder)
    with invert():
        _qq_mul(x, y, aux[:], inpl_adder)
    aux.delete()

    return res


def qq_montgomery_multiply_modulus(x: QuantumModulus, y: QuantumModulus) -> QuantumModulus:
    """Perform the montgomery product of two QuantumModuli.

    Compatible with ``montgomery_mod_mul``: inputs can be in any Montgomery
    representation (including standard form where ``m=0``).

    The reduction shift *m* is computed from the modulus size (not from the
    inputs' ``.m`` attributes), and the result's Montgomery shift is set to
    ``x.m + y.m - m``, which matches the non-JASP ``montgomery_mod_mul``
    semantics.  When both inputs are in standard form (``m=0``), the output
    is also in standard form.

    Parameters
    ----------
    x : QuantumModulus
        First factor of the montgomery product.
    y : QuantumModulus
        Second factor of the montgomery product.

    Returns
    -------
    QuantumModulus
        The montgomery product of the inputs.

    Examples
    --------
    Computes 12*7 mod 97 = 84:

    >>> from qrisp import QuantumModulus, gidney_adder, multi_measurement
    >>> a = QuantumModulus(97, inpl_adder=gidney_adder)
    >>> b = QuantumModulus(97, inpl_adder=gidney_adder)
    >>> a[:] = 12
    >>> b[:] = 7
    >>> res = qq_montgomery_multiply_modulus(a, b)
    >>> multi_measurement([a, b, res])  # doctest: +SKIP
    {(12, 7, 84): 1.0}

    """
    from qrisp.qtypes.quantum_modulus import _moduli_neq

    if not check_for_tracing_mode() and _moduli_neq(x.modulus, y.modulus):
        raise ValueError("Tried to multiply two QuantumModulus with differing modulus")

    inpl_adder: Callable = x.inpl_adder
    N = x.modulus

    # Compute the reduction shift m = ceil(log2((N-1)^2 + 1)) - n.
    # When N is a BigInteger with traced digits, both n and m will be
    # JAX tracers.  That is fine: jrange / jlen handle traced loop bounds,
    # QuantumFloat accepts traced sizes, and jdecoder handles a traced
    # Montgomery shift (res.m) via BigInteger arithmetic.
    n = smallest_power_of_two(N)
    m = smallest_power_of_two((N - 1) ** 2 + 1) - n

    res = QuantumModulus(N)
    # The result's Montgomery shift after reduction: (x.m + y.m) - m
    # (the reduction divides by 2^m, subtracting m from the accumulated shift)
    res.m = x.m + y.m - m
    res.inpl_adder = inpl_adder
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]

    _qq_mul(x, y, wqf[:-1], inpl_adder)
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    _qc_mul_inplace(aux, N, inpl_adder)
    with invert():
        _qq_mul(x, y, aux[:], inpl_adder)
    aux.delete()

    return res


def cq_montgomery_mat_multiply(A: QuantumArray, B: np.ndarray | Array, out: QuantumArray) -> QuantumArray:
    """Multiply a classical matrix B into a QuantumArray of QuantumModulus entries A, accumulating into out.

    Parameters
    ----------
    A : QuantumArray
        2D array of QuantumModulus entries (quantum matrix).
    B : numpy.ndarray or jax.Array
        2D array of classical entries (classical matrix).
    out : QuantumArray
        2D array of QuantumModulus entries that accumulates A @ B.

    Returns
    -------
    QuantumArray
        The updated ``out`` array.

    """
    if check_for_tracing_mode():
        xrange = jrange
        x_cond = q_cond
    else:
        xrange = range

        def x_cond(pred, true_fun, false_fun):
            if pred:
                true_fun()
            else:
                false_fun()

    n1 = A.shape[0]
    n2 = B.shape[1]

    m = A.shape[1]

    # out = QuantumArray(qtype=A[0,0], shape=(n1, n2))

    for k in xrange(m):
        for j in xrange(n2):
            for i in xrange(n1):

                def true_fun():
                    shift = best_montgomery_shift(B[k, j], A[i, k].modulus)
                    aux = cq_montgomery_multiply(B[k, j], A[i, k], A[i, k].modulus, shift)
                    out[i, j] += aux
                    with invert():
                        cq_montgomery_multiply(B[k, j], A[i, k], A[i, k].modulus, shift, res=aux)
                    aux.delete()

                x_cond(B[k, j] != 0, true_fun, lambda: None)

    return out
