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

"""Defines the QuantumFloat type for arbitrary-precision signed/unsigned quantum floating-point numbers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array, jit
from jax.core import Tracer

from qrisp.core import QuantumVariable, cx, x
from qrisp.environments import conjugate, invert
from qrisp.jasp import check_for_tracing_mode
from qrisp.misc import gate_wrap
from qrisp.typing import FloatLike

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qrisp.circuit.qubit import Qubit
    from qrisp.qtypes.quantum_bool import QuantumBool
    from qrisp.qtypes.quantum_modulus import QuantumModulus


def _signed_int_iso(value: int | Array, n: int) -> Array:
    """Compute the signed integer isomorphism for a given bit-width.

    This function maps an integer ``value`` from the signed range
    [-2^n, 2^n - 1] into the unsigned range [0, 2^(n+1) - 1].
    This is equivalent to the mathematical operation: value % 2^(n+1).

    Parameters
    ----------
    value : int or jax.Array
        The signed integer or array of integers to be transformed.
    n : int
        The bit-width for the signed integer representation.

    Returns
    -------
    jax.Array
        A jnp.int64 array where each element of ``value`` has been mapped to
        the unsigned range [0, 2^(n+1) - 1].

    """
    # 1. Modular wrap: Ensure value is within [0, 2**(n+1) - 1]
    mask = (jnp.int64(1) << (n + 1)) - 1
    return jnp.int64(value) & mask


@jit
def _signed_int_iso_inv(y: int | Array, n: int) -> Array:
    """Compute the inverse signed integer isomorphism for a given bit-width.

    This function maps an integer `y` from the unsigned range [0, 2^(n+1) - 1]
    back into the signed range [-2^n, 2^n - 1]. It performs a manual
    sign-extension by treating the n-th bit of `y` as the sign bit.

    Parameters
    ----------
    y : int or jax.Array
        The unsigned integer or array of integers to be transformed.
    n : int
        The bit-width for the signed integer representation.

    Returns
    -------
    jax.Array
        A jnp.int64 array where each element of `y` has been mapped to
        the signed range [-2^n, 2^n - 1].

    """
    # 1. Modular wrap: Ensure y is within [0, 2**(n+1) - 1]
    mask = (jnp.int64(1) << (n + 1)) - 1
    y_wrapped = jnp.int64(y) & mask

    # 2. Sign extension: If bit 'n' is set, the number is negative.
    # In two's complement, we subtract 2**(n+1) from values >= 2**n.
    sign_bit = jnp.int64(1) << n
    return jnp.where(y_wrapped & sign_bit, y_wrapped - (jnp.int64(1) << (n + 1)), y_wrapped)


def trunc_poly(poly: sp.Expr, trunc_bounds: tuple[int, int]) -> sp.Expr:
    """Truncate a polynomial to the given power-of-2 bounds.

    Truncates a polynomial of the form ``p(x) = 2**k_0*x**i_0 +
    2**k_1*x**i_1 + ...`` by removing every summand whose coefficient's
    power of 2 does not lie within ``trunc_bounds``.

    Parameters
    ----------
    poly : sympy.Expr
        The polynomial to truncate.
    trunc_bounds : tuple[int, int]
        The (lower, upper) power-of-2 bounds to truncate to.

    Returns
    -------
    sympy.Expr
        The truncated polynomial, expanded.

    """
    # sympy's type stubs don't model Poly's dynamic attribute surface, so
    # pyright can't see .trunc()/.expr here even though they're real members.
    # Convert to sympy polynomial
    poly_repr = sp.poly(poly)

    # Clip upper bound
    poly_repr = poly_repr.trunc(2.0 ** (trunc_bounds[1]))  # pyright: ignore[reportAttributeAccessIssue]

    # Clip lower bound
    poly_repr = poly_repr / 2.0 ** trunc_bounds[0]
    poly_repr = poly_repr - sp.poly(poly_repr).trunc(1)
    poly_repr = poly_repr * 2.0 ** trunc_bounds[0]

    return poly_repr.expr.expand()


class QuantumFloat(QuantumVariable):
    r"""This subclass of :ref:`QuantumVariable` represents signed or unsigned floats to arbitrary precision.

    The technical details of the employed arithmetic can be found in this
    `article <https://ieeexplore.ieee.org/document/9815035>`_.

    To create a QuantumFloat we call the constructor:

    >>> from qrisp import QuantumFloat
    >>> a = QuantumFloat(3, -1, signed = False)

    Here, the 3 indicates the number of mantissa qubits and the -1 indicates the
    exponent.

    For unsigned QuantumFloats, the decoder function is given by

    .. math::

        f_{k}(i) = i2^{k}

    Where $k$ is the exponent.

    We can check which values can be represented:

    >>> for i in range(2**a.size): print(a.decoder(i))
    0.0
    0.5
    1.0
    1.5
    2.0
    2.5
    3.0
    3.5

    We see $2^3 = 8$ values, because we have 3 mantissa qubits. The exponent is -1,
    implying the precision is $0.5 = 2^{-1}$.

    For signed QuantumFloats, the decoder function is

    .. math::

        f_{k}^{n}(i) = \begin{cases} i2^{k} & \text{if } i < 2^n \\ (i - 2^{n+1})2^k &
        \text{else} \end{cases}

    Where $k$ is again the exponent and $n$ is the mantissa size.


    Another example:

    >>> b = QuantumFloat(2, -2, signed = True)
    >>> for i in range(2**b.size): print(b.decoder(i))
    0.0
    0.25
    0.5
    0.75
    -1.0
    -0.75
    -0.5
    -0.25

    Here, we have $2^2 = 4$ values and their signed equivalents. Their precision is
    $0.25 = 2^{-2}$.


    **Arithmetic**

    Many operations known from classical arithmetic work for QuantumFloats in infix
    notation.

    Addition:

    >>> a[:] = 1.5
    >>> b[:] = 0.25
    >>> c = a + b
    >>> print(c)
    {1.75: 1.0}

    Subtraction:

    >>> d = a - c
    >>> print(d)
    {-0.25: 1.0}

    Multiplication:

    >>> e = d * b
    >>> print(e)
    {-0.0625: 1.0}

    And even division:

    >>> a = QuantumFloat(3)
    >>> b = QuantumFloat(3)
    >>> a[:] = 7
    >>> b[:] = 2
    >>> c = a/b
    >>> print(c)
    {3.5: 1.0}

    Floor division:

    >>> d = a//b
    >>> print(d)
    {3: 1.0}

    Inversion:

    >>> a = QuantumFloat(3, -1)
    >>> a[:] = 3.5
    >>> b = a**-1
    >>> print(b)
    {0.25: 1.0}

    Note that the latter is only an approximate result. This is because in many cases,
    the results of division cannot be stored in a finite number of qubits, forcing us
    to approximate.
    To get a better approximation we can use the :meth:`q_div <qrisp.q_div>` and
    :meth:`qf_inversion <qrisp.qf_inversion>` functions and specify the precision:

    >>> from qrisp import q_div, qf_inversion
    >>> a = QuantumFloat(3)
    >>> a[:] = 1
    >>> b = QuantumFloat(3)
    >>> b[:] = 7
    >>> c = q_div(a, b, prec = 6)
    >>> print(c)
    {0.140625: 1.0}

    Comparing with the classical result (0.1428571428):

    >>> 1/7 - 0.140625
    0.002232142857142849

    We see that the result is inside the expected precision of $2^{-6} = 0.015625$.


    **In-place Operations**

    Further supported operations are in-place addition, subtraction (with both classical
    and quantum values):

    >>> a = QuantumFloat(4, signed = True)
    >>> a[:] = 4
    >>> b = QuantumFloat(4)
    >>> b[:] = 3
    >>> a += b
    >>> print(a)
    {7: 1.0}
    >>> a -= 2
    >>> print(a)
    {5: 1.0}

    .. warning::
        Additions that would result in overflow, raise no errors. Instead, the additions
        are performed `modular <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.

        >>> c = QuantumFloat(3)
        >>> c += 9
        >>> print(c)
        {1: 1.0}

    For in-place multiplications, only classical values are allowed:

    >>> a *= -3
    >>> print(a)
    {-15: 1.0}

    .. note::
        In-place multiplications can change the mantissa size to prevent overflow
        errors. If you want to prevent this behavior, look into
        :meth:`inpl_mult <qrisp.inpl_mult>`.

        >>> a.size
        7

    **Bitshifts**

    Bitshifts can be executed for free (i.e. not requiring any quantum gates). We can
    either use the :meth:`exp_shift <qrisp.QuantumFloat.exp_shift>` method or use the
    infix operators. Note that the bitshifts work in-place.


    >>> a.exp_shift(3)
    >>> print(a)
    {-120: 1.0}
    >>> a >>= 5
    >>> print(a)
    {-3.75: 1.0}

    **Comparisons**

    QuantumFloats can be compared to Python floats using the established operators. The
    return values are :ref:`QuantumBools <QuantumBool>`:

    >>> from qrisp import h
    >>> a = QuantumFloat(4)
    >>> _ = h(a[2])
    >>> print(a)
    {0: 0.5, 4: 0.5}
    >>> comparison_qbl_0 = (a < 4 )
    >>> print(comparison_qbl_0)
    {False: 0.5, True: 0.5}

    Comparison to other QuantumFloats also works:

    >>> b = QuantumFloat(3)
    >>> b[:] = 4
    >>> comparison_qbl_1 = (a == b)
    >>> comparison_qbl_1.qs.statevector()
    sqrt(2)*(|0>*|True>*|4>*|False> + |4>*|False>*|4>*|True>)/2

    The first tensor factor containing a boolean value corresponds to
    ``comparison_qbl_0`` and the second one is ``comparison_qbl_1``.

    """

    signed: bool
    exponent: int | Array
    traced_attributes: list[str]
    static_attributes: list[str]

    def __init__(
        self,
        msize: int | Array,
        exponent: int | Array = 0,
        qs: Any = None,
        name: str | None = None,
        signed: bool = False,
    ) -> None:
        """Construct a QuantumFloat with the given mantissa size, exponent, and sign.

        Parameters
        ----------
        msize : int or jax.Array
            The number of mantissa qubits.
        exponent : int or jax.Array, optional
            The exponent, determining the precision. The default is 0.
        qs : QuantumSession, optional
            A QuantumSession object, where the QuantumFloat is supposed to be
            registered. The default is None.
        name : str, optional
            A name which uniquely identifies the QuantumFloat. The default is None.
        signed : bool, optional
            If ``True``, an additional qubit is allocated to represent the
            sign. The default is False.

        """
        # Boolean to indicate if the float is signed
        self.signed = signed
        # Exponent
        self.exponent = exponent

        # Initialize QuantumVariable
        if signed:
            super().__init__(msize + 1, qs, name=name)
        else:
            super().__init__(msize, qs, name=name)

        self.traced_attributes = ["exponent"]
        self.static_attributes = ["signed"]

    @property
    def msize(self) -> int:
        """The number of mantissa qubits (excludes the sign qubit, if any).

        Returns
        -------
        int
            The mantissa size.

        """
        return self.size - self.signed

    @property
    def mshape(self) -> tuple[int | Array, int | Array]:
        """The (log2(min), log2(max)) bounds of the absolute values this QuantumFloat can represent.

        Returns
        -------
        tuple[int, int]
            The (minimal, maximal) exponent of the representable magnitude.

        """
        return (self.exponent, self.exponent + self.msize)

    # Define outcome_labels
    def decoder(self, i: int | Array) -> int | float | Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Convert a measurement outcome (integer) back to a human-readable value.

        Parameters
        ----------
        i : int or jax.Array
            The integer outcome of a measurement of this QuantumFloat's qubits.

        Returns
        -------
        int, float, or jax.Array
            The decoded value: an ``int`` or ``float`` outside of tracing
            mode (depending on whether the exponent is non-negative), or a
            traced ``jax.Array`` while tracing.

        """
        if self.signed:
            res = _signed_int_iso_inv(i, self.msize) * jnp.float64(2) ** self.exponent
        else:
            res = i * jnp.float64(2) ** self.exponent

        if check_for_tracing_mode():
            return res
        if self.exponent >= 0:
            return int(res)
        return float(res)

    def jdecoder(self, i: int | Array) -> int | float | Array:
        """JAX-traceable version of :meth:`decoder`, used internally during tracing.

        Parameters
        ----------
        i : int or jax.Array
            The integer outcome of a measurement of this QuantumFloat's qubits.

        Returns
        -------
        int, float, or jax.Array
            The decoded value, see :meth:`decoder`.

        """
        return self.decoder(i)

    def encoder(self, i: int | float | bool | np.integer | np.floating | Tracer) -> int | Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Convert a human-readable value to an integer that represents the measurement result.

        Also validates that the input value can be represented within the bounds of the provided
        QuantumFloat in static mode.

        .. note::

            Unlike the base :meth:`QuantumVariable.encoder
            <qrisp.QuantumVariable.encoder>`, this parameter is named ``i``
            (not ``value``) for historical reasons specific to QuantumFloat.

            Unlike the base's ``ScalarLike``, this does not accept ``complex``:
            the bounds/sign checks below order ``i`` with ``<``/``>``, which
            complex values don't support.

        Parameters
        ----------
        i : int, float, bool, np.integer, np.floating, or jax.core.Tracer
            A human-readable, real-valued number.

        Returns
        -------
        int or jax.Array
            The integer encoding the given value.

        """
        # check if the encoding number is negative while the QuantumFloat is unsigned.
        # We do this before converting to integer to prevent wrapping.
        if not check_for_tracing_mode() and not self.signed and i < 0:  # pyright: ignore[reportOperatorIssue]
            raise ValueError("Tried to encode negative number in an unsigned QuantumFloat")

        # the following check is based on the math for fixed point arithmetic which varies according to the
        # size, exponent, and whether the QuantumFloat is signed or unsigned.

        # calculate the integer bounds based on mantissa size (msize)
        max_int = (1 << self.msize) - 1
        if self.signed:
            # Signed range: -2^msize to 2^msize - 1
            min_int = -(1 << self.msize)
        else:
            # Unsigned range: 0 to 2^msize - 1
            min_int = 0

        # convert those integer bounds into actual Float values
        # using the exponent.
        scaling_factor = 2**self.exponent
        max_float = max_int * scaling_factor
        min_float = min_int * scaling_factor

        # compare the input 'i' against the float limits.
        # we do this before converting to integer to prevent wrapping.
        if not check_for_tracing_mode():
            is_out_of_bounds = (i > max_float) or (i < min_float)

            # add a check that the provided value is safe to be encoded in the provided QuantumFloat
            if is_out_of_bounds:
                sign_description = "signed" if self.signed else "unsigned"
                raise ValueError(
                    f"Not enough qubits to encode value {i} in {sign_description} QuantumFloat"
                    + f" of {self.msize} qubits and exponent {self.exponent}."
                )

        if self.signed:
            res = _signed_int_iso(i / jnp.float64(2**self.exponent), self.msize)
        else:
            res = i / jnp.float64(2) ** self.exponent

        if isinstance(res, (int, float)):
            return int(res)
        return res.astype(int)

    def sb_poly(self, m: int = 0) -> sp.Expr:
        """Returns the semi-boolean polynomial of this `QuantumFloat` where `m` specifies the image extension parameter.

        For the technical details we refer to:
        https://ieeexplore.ieee.org/document/9815035


        Parameters
        ----------
        m : int, optional
            Image extension parameter. The default is 0.

        Returns
        -------
        Sympy expression
            The semi-boolean polynomial of this QuantumFloat.

        Examples
        --------
        The polynomial's symbols are named after this QuantumFloat's ``hash``
        (to guarantee uniqueness across QuantumFloats), so we inspect its
        coefficients rather than its literal string representation:

        >>> from qrisp import QuantumFloat
        >>> import sympy as sp
        >>> x = QuantumFloat(3, -1, signed = True, name = "x")
        >>> [float(c) for c in sp.Poly(x.sb_poly(5)).coeffs()]
        [0.5, 1.0, 2.0, 28.0]

        """
        if m == 0:
            m = self.size

        symbols = sp.symbols(f"{hash(self)}_0:{self.size}")

        poly = sum(2.0**i * symbols[i] for i in range(self.size))

        if self.signed:
            poly += (2.0 ** (m + 1) - 2.0 ** (self.size)) * symbols[-1]

        # sympy's Symbol arithmetic isn't precisely typed, and self.exponent
        # can be a traced jax.Array -- both are real Expr-producing operations
        # at runtime.
        return 2**self.exponent * poly  # pyright: ignore[reportReturnType]

    def encode(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        encoding_number: int | float | bool | np.integer | np.floating | Tracer,
        rounding: bool = False,
        permit_dirtyness: bool = False,
    ) -> None:
        """Initialize a QuantumFloat to a specific value.

        .. note::

            Unlike the base :meth:`QuantumVariable.encode
            <qrisp.QuantumVariable.encode>`, this method additionally accepts
            ``rounding`` (inserted before ``permit_dirtyness``, for
            historical reasons specific to QuantumFloat).

            Unlike the base's ``ScalarLike``, this does not accept ``complex``
            (see :meth:`encoder <qrisp.QuantumFloat.encoder>`, which this
            delegates to).

        Parameters
        ----------
        encoding_number : int, float, bool, np.integer, np.floating, or jax.core.Tracer
            The value to encode.
        rounding : bool, optional
            If ``True``, round ``encoding_number`` to the value this
            QuantumFloat can represent that is closest to it, before
            encoding. The default is False.
        permit_dirtyness : bool, optional
            Suppresses the error message when calling encode on dirty
            qubits. The default is False.

        Returns
        -------
        None

        """
        value = encoding_number
        if rounding:
            # Round to the closest representable value the same way truncate()
            # does: representable values form a uniform grid, so the nearest
            # one is found directly by rounding and clipping, in O(1) --
            # no need to enumerate all 2**size outcomes to search for it.
            # truncate() is typed strictly as float, but a bool/np.integer/
            # np.floating/Tracer here all support the same arithmetic at runtime.
            value = self.truncate(encoding_number)  # pyright: ignore[reportArgumentType]

        super().encode(value, permit_dirtyness=permit_dirtyness)

    @gate_wrap(permeability="args", is_qfree=True)
    def __mul__(self, other: QuantumFloat | int | np.integer) -> QuantumFloat:
        """Multiply this QuantumFloat by another QuantumFloat or a classical int."""
        if check_for_tracing_mode():
            # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
            from qrisp.alg_primitives.arithmetic import jasp_multiplyer, jasp_squaring

            if isinstance(other, QuantumFloat):
                if self is other:
                    return jasp_squaring(self)
                return jasp_multiplyer(other, self)
            raise TypeError(f"Tried to multiply class {type(other)} with QuantumFloat")

        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import polynomial_encoder, q_mult

        if isinstance(other, QuantumFloat):
            return q_mult(self, other)

        if isinstance(other, (int, np.integer)):
            if other == 0:
                # Multiplying by the classical scalar 0 always yields 0, regardless
                # of self's state, so no entanglement with self is needed. Handled
                # separately since the bit-shift/log2 logic below assumes other != 0.
                return QuantumFloat(1, self.exponent, signed=self.signed)

            bit_shift = 0
            while not other % 2:
                other = other >> 1
                bit_shift += 1

            output_qf = QuantumFloat(
                self.msize + int(np.ceil(np.log2(abs(other)))),
                self.exponent,
                signed=bool(self.signed or other < 0),
            )

            # int.__mul__ doesn't know about Symbol's __rmul__, but this works fine at runtime.
            polynomial_encoder([self], output_qf, other * sp.Symbol("x"))  # pyright: ignore[reportOperatorIssue]

            output_qf.exp_shift(bit_shift)

            return output_qf

        raise TypeError(
            f"QuantumFloat multiplication for type {type(other)} not implemented (available are QuantumFloat and int)"
        )

    @gate_wrap(permeability="args", is_qfree=True)
    def __add__(self, other: QuantumFloat | int | float | Tracer) -> QuantumFloat:
        """Add another QuantumFloat or a classical scalar to this QuantumFloat."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import sbp_add

        if isinstance(other, QuantumFloat):
            if check_for_tracing_mode():
                res = self.duplicate()
                cx(self, res)
                res += other
                return res
            return sbp_add(self, other)

        if isinstance(other, (int, float, Tracer)):
            res = self.duplicate()
            cx(self, res)
            res += other
            return res
        raise TypeError(f"Addition with type {type(other)} not implemented")

    @gate_wrap(permeability="args", is_qfree=True)
    def __sub__(self, other: QuantumFloat | int | float | Tracer) -> QuantumFloat:
        """Subtract another QuantumFloat or a classical scalar from this QuantumFloat."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import sbp_sub

        if isinstance(other, QuantumFloat):
            if check_for_tracing_mode():
                res = self.duplicate()
                cx(self, res)
                res -= other
                return res
            return sbp_sub(self, other)

        if isinstance(other, (int, float, Tracer)):
            res = self.duplicate()
            cx(self, res)
            res -= other
            return res
        raise TypeError(f"Subtraction with type {type(other)} not implemented")

    __radd__ = __add__
    __rmul__ = __mul__

    @gate_wrap(permeability="args", is_qfree=True)
    def __rsub__(self, other: QuantumFloat | int | float) -> QuantumFloat:
        """Subtract this QuantumFloat from a classical scalar or QuantumFloat."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import sbp_sub

        if isinstance(other, QuantumFloat):
            return sbp_sub(other, self)
        if isinstance(other, (int, float)):
            res = self.duplicate(init=True)
            if not res.signed:
                res.add_sign()
            x(res)
            res += other + 2**res.exponent
            return res
        raise TypeError(f"Subtraction with type {type(other)} not implemented")

    @gate_wrap(permeability="args", is_qfree=True)
    def __truediv__(self, other: QuantumFloat) -> QuantumFloat:
        """Divide this QuantumFloat by another QuantumFloat."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import q_div

        return q_div(self, other)

    @gate_wrap(permeability="args", is_qfree=True)
    def __floordiv__(self, other: QuantumFloat) -> QuantumFloat:
        """Floor-divide this (unsigned, integer) QuantumFloat by another one."""
        if self.signed or other.signed:
            raise NotImplementedError("Floor division not implemented for signed QuantumFloats")

        if self.exponent < 0 or other.exponent < 0:
            raise ValueError("Tried to perform floor division on non-integer QuantumFloats")
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import q_div

        return q_div(self, other, prec=0)

    @gate_wrap(permeability="args", is_qfree=True)
    def __pow__(self, power: int) -> QuantumFloat:
        """Raise this QuantumFloat to an integer power (-1 means inversion)."""
        if not isinstance(power, (int, np.integer)):
            raise TypeError(f"QuantumFloat exponentiation requires an integer power, got {type(power)}")

        if power == -1:
            # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
            from qrisp.alg_primitives.arithmetic import qf_inversion

            return qf_inversion(self)

        if power < 0:
            raise NotImplementedError(
                f"QuantumFloat exponentiation only supports inversion (power=-1) for negative powers, got power={power}"
            )

        if power == 0:
            res = self.duplicate()
            res[:] = 1
            return res

        temp_results = [QuantumFloat((i + 1) * self.size) for i in range(power)]

        res = QuantumFloat(self.size * power)
        with conjugate(_power_conjugator)(self, power, temp_results):
            cx(temp_results[-1], res)

        for qv in temp_results:
            qv.delete()

        return res

    @gate_wrap(permeability=[1], is_qfree=True)
    def __iadd__(self, other: QuantumFloat | FloatLike) -> QuantumFloat:
        """Add another QuantumFloat or a classical scalar to this QuantumFloat, in place."""
        if check_for_tracing_mode():
            # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
            from qrisp.alg_primitives.arithmetic.adders import gidney_adder

            if isinstance(other, QuantumFloat):
                starting_digit = jnp.maximum(other.exponent, self.exponent)

                gidney_adder(
                    other[starting_digit - other.exponent :],
                    self[starting_digit - self.exponent :],
                )
            elif isinstance(other, (int, float, np.integer, np.floating)) or (
                isinstance(other, Tracer) and isinstance(other, Array)
            ):
                # gidney_adder's stub predates encoder() returning a traced
                # jax.Array here; a concrete int or a traced Array both work
                # at runtime.
                gidney_adder(self.encoder(other), self)  # pyright: ignore[reportArgumentType]
            else:
                raise TypeError(f"Don't know how to handle quantum addition with type {type(other)}")

            return self

        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import polynomial_encoder

        if isinstance(other, QuantumFloat):
            input_qf_list = [other]
            poly = sp.symbols("x")

            polynomial_encoder(input_qf_list, self, poly)

        elif isinstance(other, (int, float, np.number)):
            scaled = other / 2**self.exponent
            if int(scaled) != scaled:
                raise ValueError(
                    "Tried to perform in-place addition with invalid number. QuantumFloat precision too low."
                )

            input_qf_list = []
            poly = sp.sympify(other)

            polynomial_encoder(input_qf_list, self, poly)

        else:
            raise TypeError(f"In-place addition for type {type(other)} not implemented")

        return self

    @gate_wrap(permeability=[1], is_qfree=True)
    def __isub__(self, other: QuantumFloat | FloatLike) -> QuantumFloat:
        """Subtract another QuantumFloat or a classical scalar from this QuantumFloat, in place."""
        if check_for_tracing_mode():
            with invert():
                self.__iadd__(other)
            return self

        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import polynomial_encoder

        if isinstance(other, QuantumFloat):
            input_qf_list = [other]
            poly = -sp.symbols("x")

            polynomial_encoder(input_qf_list, self, poly)

        elif isinstance(other, (int, float, np.integer, np.floating)):
            scaled = other / 2**self.exponent
            if int(scaled) != scaled:
                raise ValueError(
                    "Tried to perform in-place subtraction with invalid number. QuantumFloat precision too low."
                )

            input_qf_list = []
            poly = -sp.sympify(other)

            polynomial_encoder(input_qf_list, self, poly)

        else:
            raise TypeError(f"In-place subtraction for type {type(other)} not implemented")

        return self

    @gate_wrap(permeability=[], is_qfree=True)
    def __imul__(self, other: FloatLike) -> QuantumFloat:
        """Multiply this QuantumFloat by a classical scalar, in place."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import inpl_mult

        inpl_mult(self, other)

        return self

    def __irshift__(self, k: int) -> QuantumFloat:
        """Shift this QuantumFloat's exponent down by k (a free, gate-less bitshift)."""
        self.exp_shift(-k)
        return self

    def __ilshift__(self, k: int) -> QuantumFloat:
        """Shift this QuantumFloat's exponent up by k (a free, gate-less bitshift)."""
        self.exp_shift(k)
        return self

    def __lt__(self, other: QuantumFloat | FloatLike) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (<)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import gidney_adder, lt, uint_lt

        if check_for_tracing_mode():
            return uint_lt(self, other, gidney_adder)  # pyright: ignore[reportReturnType]
        if not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return lt(self, other)  # pyright: ignore[reportReturnType]

    def __gt__(self, other: QuantumFloat | FloatLike) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (>)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import gidney_adder, gt, uint_gt

        if check_for_tracing_mode():
            return uint_gt(self, other, gidney_adder)  # pyright: ignore[reportReturnType]
        if not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return gt(self, other)  # pyright: ignore[reportReturnType]

    def __le__(self, other: QuantumFloat | FloatLike) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (<=)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import gidney_adder, leq, uint_le

        if check_for_tracing_mode():
            return uint_le(self, other, gidney_adder)
        if not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return leq(self, other)

    def __ge__(self, other: QuantumFloat | FloatLike) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (>=)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import geq, gidney_adder, uint_ge

        if check_for_tracing_mode():
            return uint_ge(self, other, gidney_adder)
        if not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return geq(self, other)

    def __eq__(self, other: object) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (==)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import eq

        if not check_for_tracing_mode() and not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return eq(self, other)

    def __ne__(self, other: object) -> "QuantumBool":
        """Compare this QuantumFloat to another QuantumFloat or a classical scalar (!=)."""
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import neq

        if not check_for_tracing_mode() and not isinstance(other, (QuantumFloat, int, float)):
            raise TypeError(f"Comparison with type {type(other)} not implemented")

        return neq(self, other)

    def exp_shift(self, shift: int) -> None:
        """Performs an internal bit shift.

        Note that this method doesn't cost any quantum gates. For the quantum
        version of this method, see
        :meth:`quantum_bit_shift<qrisp.QuantumFloat.quantum_bit_shift>`.

        Parameters
        ----------
        shift : int
            The amount to shift.

        Raises
        ------
        TypeError
            Tried to shift QuantumFloat exponent by non-integer value

        Examples
        --------
        We create a QuantumFloat and perform a bitshift:

        >>> from qrisp import QuantumFloat
        >>> a = QuantumFloat(4)
        >>> a[:] = 2
        >>> a.exp_shift(2)
        >>> print(a)
        {8: 1.0}
        >>> print(a.qs)
        QuantumCircuit:
        ---------------
        a.0: ─────
             ┌───┐
        a.1: ┤ X ├
             └───┘
        a.2: ─────
        <BLANKLINE>
        a.3: ─────
        <BLANKLINE>
        Live QuantumVariables:
        ----------------------
        QuantumFloat a

        """
        if not isinstance(shift, int):
            raise TypeError("Tried to shift QuantumFloat exponent by non-integer value")

        self.exponent += shift

    def add_sign(self) -> None:
        """Turns an unsigned QuantumFloat into its signed version.

        Raises
        ------
        ValueError
            Tried to add sign to signed QuantumFloat.

        Examples
        --------
        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(4)
        >>> qf.signed
        False
        >>> qf.add_sign()
        >>> qf.signed
        True

        """
        if self.signed:
            raise ValueError("Tried to add sign to signed QuantumFloat")

        self.extend(1, self.size)
        self.signed = True

    def sign(self) -> "Qubit":
        r"""Returns the sign qubit.

        This qubit is in state $\ket{1}$ if the QuantumFloat holds a negative value and
        in state $\ket{0}$ otherwise.

        For more information about the encoding of negative numbers check the
        `publication <https://ieeexplore.ieee.org/document/9815035>`_.

        .. warning::

            Performing an X gate on this qubit does not flip the sign! Use in-place
            multiplication instead.

            >>> from qrisp import QuantumFloat
            >>> qf = QuantumFloat(3, signed = True)
            >>> qf[:] = 3
            >>> qf *= -1
            >>> print(qf)
            {-3: 1.0}

        Raises
        ------
        ValueError
            Tried to retrieve sign qubit of unsigned QuantumFloat.

        Returns
        -------
        Qubit
            The qubit holding the sign.

        Examples
        --------
        We create a QuantumFloat, initiate a state that has probability 2/3 of being
        negative and entangle a QuantumBool with the sign qubit.

        >>> from qrisp import QuantumFloat, QuantumBool, cx
        >>> qf = QuantumFloat(4, signed = True)
        >>> n_amp = 1/3**0.5
        >>> qf[:] = {-1 : n_amp, -2 : n_amp, 1 : n_amp}
        >>> qbl = QuantumBool()
        >>> _ = cx(qf.sign(), qbl)
        >>> print(qbl)
        {True: 0.66667, False: 0.33333}

        """
        if not self.signed:
            raise ValueError("Tried to retrieve sign qubit of unsigned QuantumFloat")

        return self[-1]  # pyright: ignore[reportReturnType]

    def init_from(
        self, other: QuantumFloat, ignore_rounding_errors: bool = False, ignore_overflow_errors: bool = False
    ) -> None:
        """Initialize this (zero-valued) QuantumFloat with the value of another one.

        Parameters
        ----------
        other : QuantumFloat
            The QuantumFloat to copy the value from.
        ignore_rounding_errors : bool, optional
            If ``True``, don't raise if ``other`` has more precision than
            this QuantumFloat can represent. The default is False.
        ignore_overflow_errors : bool, optional
            If ``True``, don't raise if ``other`` can represent larger
            magnitudes than this QuantumFloat. The default is False.

        """
        copy_qf(
            self,
            other,
            ignore_rounding_errors=ignore_rounding_errors,
            ignore_overflow_errors=ignore_overflow_errors,
        )

    def incr(self, value: FloatLike | None = None) -> None:
        """Increment this QuantumFloat in place by a classical value.

        Parameters
        ----------
        value : FloatLike, optional
            The value to increment by. The default is this QuantumFloat's
            smallest representable increment, ``2**self.exponent``.

        """
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic.adders.incrementation import increment

        if value is None:
            value = 2**self.exponent
        increment(self, value)

    def __hash__(self) -> int:
        """Hash by object identity (QuantumFloats define __eq__, which disables the default hash)."""
        return id(self)

    def significant(self, k: int) -> "Qubit":
        """Returns the qubit with significance $k$.

        Parameters
        ----------
        k : int
            The significance.

        Raises
        ------
        ValueError
            Tried to retrieve invalid significant from QuantumFloat

        Returns
        -------
        Qubit
            The Qubit with significance $k$.

        Examples
        --------
        We create a QuantumFloat and flip a qubit of specified significance.

        >>> from qrisp import QuantumFloat, x
        >>> qf = QuantumFloat(6, -3)
        >>> _ = x(qf.significant(-2))
        >>> print(qf)
        {0.25: 1.0}

        The qubit with significance $-2$ corresponds to the value $0.25 = 2^{-2}$.

        >>> _ = x(qf.significant(2))
        >>> print(qf)
        {4.25: 1.0}

        The qubit with significance $2$ corresponds to the value $4 = 2^{2}$.

        """
        min_sig, max_sig = self.mshape

        if not min_sig <= k < max_sig:
            raise ValueError(
                f"Tried to retrieve invalid significant {k} from QuantumFloat with mantissa shape {self.mshape}"
            )

        return self[k - min_sig]  # pyright: ignore[reportReturnType]

    def truncate(self, value: float) -> float:
        """Receives a regular float and returns the float that is closest to the input but can still be encoded.

        Parameters
        ----------
        value : float
            A float that is supposed to be truncated.

        Returns
        -------
        float
            The truncated float.

        Examples
        --------
        We create a QuantumFloat and round a value to the closest one it can
        represent. Note that directly encoding an unrepresentable value (like
        ``0.5102341`` below, which doesn't fit this QuantumFloat's precision
        of $2^{-1} = 0.5$) already truncates silently, so ``truncate`` is
        most useful when you want to know the resulting value ahead of time:

        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(4, -1)
        >>> value = 0.5102341
        >>> rounded_value = qf.truncate(value)
        >>> rounded_value
        0.5
        >>> qf[:] = rounded_value
        >>> print(qf)
        {0.5: 1.0}

        """
        # Clip in floating point before converting to int64: converting a
        # float far outside int64's range is platform-dependent behavior, not
        # a guaranteed saturating clamp.
        res = jnp.round(value / jnp.float64(2) ** self.exponent)
        res = jnp.minimum(2.0**self.msize - 1, res)

        if self.signed:
            res = jnp.maximum(-(2.0**self.msize), res)
            res = _signed_int_iso(jnp.int64(res), self.size)
        else:
            res = jnp.maximum(0.0, res)
            res = jnp.int64(res)

        return self.decoder(res)  # pyright: ignore[reportReturnType]

    def get_ev(self, **mes_kwargs: Any) -> float:
        """Retrieves the expectation value of self.

        Parameters
        ----------
        **mes_kwargs : dict
            Keyword arguments for the measurement. See :meth:`qrisp.QuantumVariable.get_measurement` for more information.

        Returns
        -------
        float
            The expectation value.

        Examples
        --------
        We set up a QuantumFloat in uniform superposition and retrieve the expectation value:

        >>> from qrisp import QuantumFloat, h
        >>> qf = QuantumFloat(4)
        >>> _ = h(qf)
        >>> qf.get_ev()
        7.5

        """
        mes_res = self.get_measurement(**mes_kwargs)

        return sum(k * v for k, v in mes_res.items())  # pyright: ignore[reportReturnType]

    def quantum_bit_shift(self, shift_amount: int | QuantumFloat) -> None:
        """Performs a bit shift in the quantum device.

        While :meth:`exp_shift<qrisp.QuantumFloat.exp_shift>` performs a bit shift
        in the compiler (thus costing no quantum gates), this method performs the
        bit shift on the hardware.

        This has the advantage that it can be controlled if called within a
        :ref:`ControlEnvironment` and furthermore admits bit shifts based on the
        state of a QuantumFloat.

        .. note::

            Bit shifts based on a QuantumFloat are currently only possible
            if both self and ``shift_amount`` are unsigned.

        .. warning::

            Quantum bit shifting extends the QuantumFloat (ie. it allocates
            additional qubits).

        Parameters
        ----------
        shift_amount : int or QuantumFloat
            The amount to shift.

        Raises
        ------
        TypeError
            Tried to shift QuantumFloat exponent by non-integer value
        Exception
            Quantum-quantum bitshifting is currently only supported for unsigned arguments

        Examples
        --------
        We create a QuantumFloat and a QuantumBool to perform a controlled bit
        shift, then evaluate the resulting (superposed) state:

        ::

            from qrisp import QuantumFloat, QuantumBool, h
            qf = QuantumFloat(4)
            qf[:] = 1
            qbl = QuantumBool()
            h(qbl)

            with qbl:
                qf.quantum_bit_shift(2)

            print(qf.qs.statevector())
            # Yields
            # sqrt(2)*(|1>*|False> + |4>*|True>)/2

        """
        # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
        from qrisp.alg_primitives.arithmetic import quantum_bit_shift

        quantum_bit_shift(self, shift_amount)


def _power_conjugator(base: QuantumFloat, power: int, temp_results: list[QuantumFloat]) -> None:
    """Conjugator for QuantumFloat.__pow__: fills temp_results[i] with base**(i + 1).

    Parameters
    ----------
    base : QuantumFloat
        The QuantumFloat being raised to a power.
    power : int
        The power ``base`` is being raised to.
    temp_results : list[QuantumFloat]
        Freshly allocated QuantumFloats, one per power from 1 to ``power``,
        filled in place: ``temp_results[i]`` ends up holding ``base**(i + 1)``.

    """
    # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
    from qrisp.alg_primitives.arithmetic import jasp_multiplyer

    cx(base, temp_results[0])
    for i in range(power - 1):
        (temp_results[i + 1] << jasp_multiplyer)(base, temp_results[i])


def _addsub_bounds(op0: QuantumFloat, op1: QuantumFloat) -> tuple[int | Array, int | Array]:
    """Compute the (exponent, max_sig) bounds for an add/sub output QuantumFloat.

    Comparisons like min/max need concrete values to branch on, which a jax
    tracer can't provide -- so this only uses jnp when actually tracing (see
    :func:`check_for_tracing_mode`). Outside of tracing, jnp.minimum/maximum
    would silently turn a plain-int exponent into a 0-d jax.Array, which then
    breaks any later ``2**exponent`` with a negative exponent (jax's
    integer_pow rejects negative integer exponents).

    Parameters
    ----------
    op0 : QuantumFloat
        The first operand.
    op1 : QuantumFloat
        The second operand.

    Returns
    -------
    tuple[int or jax.Array, int or jax.Array]
        The (exponent, max_sig) bounds for sizing the output QuantumFloat.

    """
    if check_for_tracing_mode():
        exponent = jnp.minimum(op0.exponent, op1.exponent)
        max_sig = jnp.maximum(op0.mshape[1], op1.mshape[1]) + 1
    else:
        exponent = min(op0.exponent, op1.exponent)
        max_sig = max(op0.mshape[1], op1.mshape[1]) + 1
    return exponent, max_sig


def _prod(values: Iterable[sp.Expr]) -> sp.Expr:
    """Multiply an iterable of sympy expressions together.

    Seeded from the first element (rather than ``math.prod``'s default seed
    of ``1``) so this also accepts non-numeric multiplicative types; here,
    the powers of sympy Symbols making up one monomial of a polynomial.

    Parameters
    ----------
    values : Iterable[sympy.Expr]
        The values to multiply together. Must be non-empty.

    Returns
    -------
    sympy.Expr
        The product of every element in ``values``.

    """
    values = list(values)
    res = values[0]
    for value in values[1:]:
        # sympy's Expr stubs don't model __imul__ precisely enough for pyright
        # here, even though multiplying two Exprs is a real, supported operation.
        res *= value  # pyright: ignore[reportOperatorIssue]

    return res


def _all_quantum_modulus(operands: list[QuantumFloat]) -> TypeGuard[list[QuantumModulus]]:
    """Check whether every operand is a QuantumModulus, narrowing the list element type.

    A plain ``all(isinstance(operand, QuantumModulus) for operand in operands)``
    is just as correct at runtime, but pyright can't propagate a narrowed
    element type out of that expression -- wrapping it in a
    :data:`~typing.TypeGuard`-annotated function is what lets callers use
    ``operands[i].m``/``.modulus`` afterwards without a type: ignore.

    """
    # NOTE: Local import to avoid a circular import (QuantumModulus subclasses QuantumFloat, so
    # qrisp.qtypes can only expose QuantumModulus after this module has finished loading).
    from qrisp.qtypes import QuantumModulus

    return all(isinstance(operand, QuantumModulus) for operand in operands)


def _polynomial_output_qf(operands: list[QuantumFloat], op: sp.Expr) -> QuantumFloat:
    """Size the output QuantumFloat for a polynomial-encoding operation.

    Parameters
    ----------
    operands : list[QuantumFloat]
        The QuantumFloats participating in the polynomial. Sorted in place
        by name as a side effect (matching ``op``'s generator order).
    op : sympy.Expr
        The polynomial expression being encoded.

    Returns
    -------
    QuantumFloat
        A freshly allocated QuantumFloat, sized to hold the result of ``op``
        without overflow.

    """
    # NOTE: Local import to avoid a circular import (qrisp.alg_primitives.arithmetic imports from qrisp.qtypes).
    from qrisp.alg_primitives.arithmetic.poly_tools import expr_to_list

    # Only called for its validation side effect (raises if op isn't
    # actually a polynomial); sp.Poly() below doesn't catch that on its own.
    _ = expr_to_list(op)

    operands.sort(key=lambda operand: operand.name)

    # sympy's type stubs don't model Poly's/Abs's dynamic attribute
    # surface, so pyright can't see .gens/.coeffs()/.monoms()/.subs()
    # here even though they're all real Poly/Basic members.
    poly = sp.Poly(op)  # pyright: ignore[reportAttributeAccessIssue]
    monom_list = [
        a * _prod(sym**k for sym, k in zip(poly.gens, mon))  # pyright: ignore[reportAttributeAccessIssue]
        for a, mon in zip(poly.coeffs(), poly.monoms())  # pyright: ignore[reportAttributeAccessIssue]
    ]

    max_value_dic = {sp.Symbol(qf.name): 2.0 ** qf.mshape[1] for qf in operands}
    min_value_dic = {sp.Symbol(qf.name): 2.0 ** qf.mshape[0] for qf in operands}

    abs_poly = sum((sp.Abs(monom) for monom in monom_list), 0)  # pyright: ignore[reportCallIssue, reportArgumentType]

    min_poly_value = min(float(sp.Abs(monom).subs(min_value_dic)) for monom in monom_list)  # pyright: ignore[reportAttributeAccessIssue]

    max_poly_value = float(abs_poly.subs(max_value_dic))

    min_sig = int(np.floor(np.log2(min_poly_value)))
    max_sig = int(np.ceil(np.log2(max_poly_value)))

    return QuantumFloat(
        max_sig - min_sig,
        exponent=min_sig,
        signed=any(operand.signed for operand in operands),
    )


def create_output_qf(operands: list[QuantumFloat], op: str | sp.Expr) -> QuantumFloat:
    """Determine the appropriately-sized output QuantumFloat for an arithmetic operation.

    Parameters
    ----------
    operands : list[QuantumFloat]
        The QuantumFloats participating in the operation.
    op : str or sympy.Expr
        Either one of "add", "sub", "mul", or a sympy expression describing
        a polynomial encoding (see :func:`polynomial_encoder <qrisp.polynomial_encoder>`).

    Returns
    -------
    QuantumFloat
        A freshly allocated QuantumFloat, sized to hold the result of ``op``
        without overflow.

    """
    if isinstance(op, sp.Expr):
        return _polynomial_output_qf(operands, op)

    if _all_quantum_modulus(operands):
        res = operands[0].duplicate()
        if op == "mul":
            res.m = (
                operands[0].m
                + operands[1].m
                - (int(np.ceil(np.log2((operands[0].modulus - 1) ** 2) + 1)) - operands[0].size)
            )
        return res

    if op == "add":
        signed = operands[0].signed or operands[1].signed
        exponent, max_sig = _addsub_bounds(operands[0], operands[1])
        msize = max_sig - exponent + 1

        return QuantumFloat(msize, exponent, operands[0].qs, signed=signed, name="add_res*")

    if op == "mul":
        signed = operands[0].signed or operands[1].signed

        if operands[0].reg == operands[1].reg and (operands[0].signed and operands[1].signed):
            signed = False

        return QuantumFloat(
            operands[0].msize + operands[1].msize + operands[0].signed * operands[1].signed,
            operands[0].exponent + operands[1].exponent,
            operands[0].qs,
            signed=signed,
            name="mul_res*",
        )

    if op == "sub":
        exponent, max_sig = _addsub_bounds(operands[0], operands[1])
        msize = max_sig - exponent + 1

        return QuantumFloat(msize, exponent, operands[0].qs, signed=True, name="sub_res*")

    raise ValueError(f"Don't know how to create output QuantumFloat for operation {op}")


# Initiates the value of qf2 into qf1 where qf1 has to hold the value 0
def copy_qf(
    qf1: QuantumFloat, qf2: QuantumFloat, ignore_overflow_errors: bool = False, ignore_rounding_errors: bool = False
) -> None:
    """Initiate the value of qf2 into qf1, where qf1 has to hold the value 0.

    Parameters
    ----------
    qf1 : QuantumFloat
        The (zero-valued) QuantumFloat to copy the value into.
    qf2 : QuantumFloat
        The QuantumFloat to copy the value from.
    ignore_overflow_errors : bool, optional
        If ``True``, don't raise if qf2 can represent larger magnitudes than
        qf1. The default is False.
    ignore_rounding_errors : bool, optional
        If ``True``, don't raise if qf2 has more precision than qf1 can
        represent. The default is False.

    """
    # Each QuantumFloat's qubit i has significance qf.exponent + i, a
    # contiguous run -- so its bounds are plain arithmetic, no list needed.
    qf1_sig_range = range(qf1.exponent, qf1.exponent + qf1.size)
    qf2_sig_range = range(qf2.exponent, qf2.exponent + qf2.size)

    # Check overflow/underflow
    if max(qf1_sig_range) < max(qf2_sig_range) and not ignore_overflow_errors:
        raise ValueError("Copy operation would result in overflow (use ignore_overflow_errors = True)")

    if min(qf1_sig_range) > min(qf2_sig_range) and not ignore_rounding_errors:
        raise ValueError("Copy operation would result in rounding (use ignore_rounding_errors = True)")

    qs = qf1.qs

    # Qubit counts to copy, excluding the sign qubit (the last one) when qf2
    # is signed -- it's handled on its own below.
    qf1_len = qf1.size
    qf2_len = qf2.size

    if qf2.signed:
        if not qf1.signed:
            raise ValueError("Tried to copy signed into unsigned float")

        qf1_len -= 1
        qf2_len -= 1

    qf1_start = qf1.exponent
    qf2_start = qf2.exponent

    # Highest significance in qf2's own mantissa range. For a signed,
    # zero-mantissa qf2 (qf2_len == 0), this is qf2_start - 1, so every
    # significance in qf1 at or above qf2_start (there being no actual qf2
    # mantissa bit to overlap with) is still sign-extended below.
    qf2_max = qf2_start + qf2_len - 1

    # QuantumVariable.qs/__getitem__ aren't typed precisely enough for pyright
    # to see qs as a QuantumSession (with .cx) here rather than the
    # TracingQuantumSession union member, or single-index __getitem__ as
    # returning a Qubit rather than DynamicQubitArray -- both hold in this
    # non-tracing, single-qubit-index context.
    for i in range(qf1_len):
        significance = qf1_start + i

        # If we are in a realm where both floats have overlapping significance
        # => CNOT into each other
        rel_index = significance - qf2_start
        if 0 <= rel_index < qf2_len:
            qs.cx(qf2[rel_index], qf1[i])  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            continue

        # Otherwise copy the sign bit into the bits of higher significance than qf2
        if qf2.signed and significance > qf2_max:
            qs.cx(qf2[-1], qf1[i])  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    # Copy the sign bit
    if qf2.signed:
        qs.cx(qf2[-1], qf1[-1])  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
