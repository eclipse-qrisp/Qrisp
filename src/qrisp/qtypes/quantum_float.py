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

import numpy as np
import sympy as sp
import jax.numpy as jnp
from jax import jit, Array
from jax.core import Tracer

from qrisp.core import QuantumVariable, cx
from qrisp.misc import gate_wrap
from qrisp.environments import invert, conjugate
from qrisp.jasp import check_for_tracing_mode


@jit
def _signed_int_iso(x, n):
    """
    Computes the signed integer isomorphism for a given bit-width.

    This function maps an integer `x` from the signed range [-2^n, 2^n - 1]
    into the unsigned range [0, 2^(n+1) - 1].
    This is equivalent to the mathematical operation: x % 2^(n+1).

    Parameters
    ----------
    x : int or jax.Array
        The signed integer or array of integers to be transformed.
    n : int
        The bit-width for the signed integer representation.

    Returns
    -------
    jax.Array
        A jnp.int64 array where each element of `x` has been mapped to
        the unsigned range [0, 2^(n+1) - 1].
    """
    # 1. Modular wrap: Ensure x is within [0, 2**(n+1) - 1]
    mask = (jnp.int64(1) << (n + 1)) - 1
    return jnp.int64(x) & mask


@jit
def _signed_int_iso_inv(y, n):
    """
    Computes the inverse signed integer isomorphism for a given bit-width.

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
    return jnp.where(
        y_wrapped & sign_bit, y_wrapped - (jnp.int64(1) << (n + 1)), y_wrapped
    )


# def signed_int_iso(x, n):
#    if int(x) < -(2**n) or int(x) >= 2**n:
#        raise Exception("Applying signed integer isomorphism resulted in overflow")

#    if x >= 0:
#        return x % 2**n
#    else:
#        return -abs(x) % 2 ** (n + 1)


# def signed_int_iso_inv(y, n):
#    y = y % 2 ** (n + 1)
#    if y < 2**n:
#        return y
#    else:
#        return -(2 ** (n + 1)) + y


# Truncates a polynomial of the form p(x) = 2**k_0*x*i_0 + 2**k_1*x**i_1 ...
# where every summand where the power of the coefficients does not lie in the interval
# trunc bounds is removed
def trunc_poly(poly, trunc_bounds):
    # Convert to sympy polynomial
    poly = sp.poly(poly)

    # Clip upper bound
    poly = poly.trunc(2.0 ** (trunc_bounds[1]))

    # Clip lower bound
    poly = poly / 2.0 ** trunc_bounds[0]
    poly = poly - sp.poly(poly).trunc(1)
    poly = poly * 2.0 ** trunc_bounds[0]

    return poly.expr.expand()


class QuantumFloat(QuantumVariable):
    r"""
    This subclass of :ref:`QuantumVariable` can represent floating point numbers
    (signed and unsigned) up to an arbitrary precision.

    The technical details of the employed arithmetic can be found in this
    `article <https://ieeexplore.ieee.org/document/9815035>`_.

    To create a QuantumFloat we call the constructor:

    >>> from qrisp import QuantumFloat
    >>> a = QuantumFloat(3, -1, signed = False)

    Here, the 3 indicates the amount of mantissa qubits and the -1 indicates the
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

    Where $k$ is again, the exponent and $n$ is the mantissa size.


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
    {3.0: 1.0}

    Inversion:

    >>> a = QuantumFloat(3, -1)
    >>> a[:] = 3.5
    >>> b = a**-1
    >>> print(b)
    {0.25: 1.0}

    Note that the latter is only an approximate result. This is because in many cases,
    the results of division can not be stored in a finite amount of qubits, forcing us
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

    We see that the result is inside the expected precision of $2^{-6} =  0.015625$.


    **In-place Operations**

    Further supported operations are inplace addition, subtraction (with both classical
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

    For inplace multiplications, only classical values are allowed:

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
    >>> h(a[2])
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

    The first tensor factor containing a boolean value is corresponding to
    ``comparison_qbl_0`` and the second one is ``comparison_qbl_1``.

    """

    def __init__(self, msize, exponent=0, qs=None, name=None, signed=False):
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
    def msize(self):
        return self.size - self.signed

    @property
    def mshape(self):
        # Tuple that consists of (log2(min), log2(max)) where min and max are the
        # minimal and maximal values of the absolutes that the QuantumFloat can
        # represent.
        return (self.exponent, self.exponent + self.msize)

    # Define outcome_labels
    def decoder(self, i):

        if self.signed:
            res = _signed_int_iso_inv(i, self.msize) * jnp.float64(2) ** self.exponent
        else:
            res = i * jnp.float64(2) ** self.exponent

        if check_for_tracing_mode():
            return res
        else:
            if self.exponent >= 0:
                return int(res)
            else:
                return float(res)

    def jdecoder(self, i):
        return self.decoder(i)

    def encoder(self, i):

        if self.signed:
            res = _signed_int_iso(i / jnp.float64(2) ** self.exponent, self.msize)
        else:
            res = i / jnp.float64(2) ** self.exponent

        if isinstance(res, (int, float)):
            return int(res)
        else:
            return res.astype(int)

    def sb_poly(self, m=0):
        """
        Returns the semi-boolean polynomial of this `QuantumFloat` where `m` specifies
        the image extension parameter.

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

        >>> from qrisp import QuantumFloat
        >>> x = QuantumFloat(3, -1, signed = True, name = "x")
        >>> print(x.sb_poly(5))
        0.5*x_0 + 1.0*x_1 + 2.0*x_2 + 28.0*x_3

        """

        if m == 0:
            m = self.size

        symbols = [sp.symbols(str(hash(self)) + "_" + str(i)) for i in range(self.size)]

        poly = sum([2.0 ** (i) * symbols[i] for i in range(self.size)])

        if self.signed:
            poly += (2.0 ** (m + 1) - 2.0 ** (self.size)) * symbols[-1]

        return 2**self.exponent * poly

    def encode(self, encoding_number, rounding=False, permit_dirtyness=False):
        if rounding:
            # Round value to closest fitting number
            outcome_labels = [self.decoder(i) for i in range(2**self.size)]
            encoding_number = outcome_labels[
                np.argmin(np.abs(encoding_number - np.array(outcome_labels)))
            ]
        try:
            super().encode(encoding_number, permit_dirtyness=permit_dirtyness)
        except Exception as e:
            raise ValueError(f"Not enough qubits to encode integer {encoding_number} in QuantumFloat"
                             +f" of {self.size} qubits and exponent {self.exponent}.")

    @gate_wrap(permeability="args", is_qfree=True)
    def __mul__(self, other):

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():
            from qrisp.alg_primitives.arithmetic import jasp_multiplyer, jasp_squaring

            if isinstance(other, QuantumFloat):
                if self is other:
                    return jasp_squaring(self)
                else:
                    return jasp_multiplyer(other, self)
            else:
                raise Exception(
                    f"Tried to multiply class {type(other)} with QuantumFloat"
                )

        from qrisp.alg_primitives.arithmetic import q_mult, polynomial_encoder

        if isinstance(other, QuantumFloat):
            return q_mult(self, other)
        elif isinstance(other, (int, np.integer)):
            bit_shift = 0
            while not other % 2:
                other = other >> 1
                bit_shift += 1

            if self.signed or other < 0:
                output_qf = QuantumFloat(
                    self.msize + int(np.ceil(np.log2(abs(other)))),
                    self.exponent,
                    signed=True,
                )
            else:
                output_qf = QuantumFloat(
                    self.msize + int(np.ceil(np.log2(abs(other)))),
                    self.exponent,
                    signed=False,
                )

            polynomial_encoder([self], output_qf, other * sp.Symbol("x"))

            output_qf.exp_shift(bit_shift)

            return output_qf
        else:
            raise Exception(
                "QuantumFloat multiplication for type " + str(type(other)) + ""
                " not implemented (available are QuantumFloat and int)"
            )

    @gate_wrap(permeability="args", is_qfree=True)
    def __add__(self, other):

        from qrisp.alg_primitives.arithmetic import sbp_add
        from qrisp import check_for_tracing_mode

        if isinstance(other, QuantumFloat):
            if check_for_tracing_mode():
                res = self.duplicate()
                cx(self, res)
                res += other
                return res
            else:
                return sbp_add(self, other)

        elif isinstance(other, (int, float)):
            res = self.duplicate()
            cx(self, res)
            res += other
            return res
        else:
            raise Exception(
                "Addition with type " + str(type(other)) + " not implemented"
            )

    @gate_wrap(permeability="args", is_qfree=True)
    def __sub__(self, other):
        from qrisp.alg_primitives.arithmetic import sbp_sub
        from qrisp import check_for_tracing_mode

        if isinstance(other, QuantumFloat):
            if check_for_tracing_mode():
                res = self.duplicate()
                cx(self, res)
                res -= other
                return res
            else:
                return sbp_sub(self, other)

        elif isinstance(other, (int, float)):
            res = self.duplicate()
            cx(self, res)
            res -= other
            return res
        else:
            raise Exception(
                "Subtraction with type " + str(type(other)) + " not implemented"
            )

    __radd__ = __add__
    __rmul__ = __mul__

    @gate_wrap(permeability="args", is_qfree=True)
    def __rsub__(self, other):
        from qrisp import x
        from qrisp.alg_primitives.arithmetic import sbp_sub

        if isinstance(other, QuantumFloat):
            return sbp_sub(other, self)
        elif isinstance(other, (int, float)):
            res = self.duplicate(init=True)
            if not res.signed:
                res.add_sign()
            x(res)
            res += other + 2**res.exponent
            return res
        else:
            raise Exception(
                "Subtraction with type " + str(type(other)) + " not implemented"
            )

    @gate_wrap(permeability="args", is_qfree=True)
    def __truediv__(self, other):
        from qrisp.alg_primitives.arithmetic import q_div

        return q_div(self, other)

    @gate_wrap(permeability="args", is_qfree=True)
    def __floordiv__(self, other):
        if self.signed or other.signed:
            raise Exception("Floor division not implemented for signed QuantumFloats")

        if self.exponent < 0 or other.exponent < 0:
            raise Exception(
                "Tried to perform floor division on non-integer QuantumFloats"
            )
        from qrisp.alg_primitives.arithmetic import q_div

        return q_div(self, other, prec=0)

    @gate_wrap(permeability="args", is_qfree=True)
    def __pow__(self, power):
        if power == -1:
            from qrisp.alg_primitives.arithmetic import qf_inversion

            return qf_inversion(self)
        elif power == 0:
            res = self.duplicate()
            res[:] = 1
            return res
        else:
            from qrisp import jasp_multiplyer

            def power_conjugator(base, power, temp_results):
                cx(base, temp_results[0])
                for i in range(power - 1):
                    (temp_results[i + 1] << jasp_multiplyer)(base, temp_results[i])
                    # (temp_results[i+1] << (lambda a, b : a * b))(base, temp_results[i])

            temp_results = [QuantumFloat((i + 1) * self.size) for i in range(power)]

            res = QuantumFloat(self.size * power)
            with conjugate(power_conjugator)(self, power, temp_results):
                cx(temp_results[-1], res)

            for qv in temp_results:
                qv.delete()

            return res

    @gate_wrap(permeability=[1], is_qfree=True)
    def __iadd__(self, other):

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():
            from qrisp.alg_primitives.arithmetic import gidney_adder

            if isinstance(other, QuantumFloat):
                starting_digit = jnp.maximum(other.exponent, self.exponent)

                gidney_adder(
                    other[starting_digit - other.exponent :],
                    self[starting_digit - self.exponent :],
                )
            elif isinstance(other, (int, float)) or (
                isinstance(other, Tracer) and isinstance(other, Array)
            ):
                gidney_adder(self.encoder(other), self)
            else:
                print(isinstance(other, Tracer))
                print(type(other.dtype))
                raise Exception(
                    f"Don't know how to handle quantum addition with type {type(other)}"
                )

            return self

        from qrisp.alg_primitives.arithmetic import polynomial_encoder

        if isinstance(other, QuantumFloat):
            input_qf_list = [other]
            poly = sp.symbols("x")

            polynomial_encoder(input_qf_list, self, poly)

        elif isinstance(other, (int, float, np.number)):
            # self.incr(other)

            if not int(other / 2**self.exponent) == other / 2**self.exponent:
                raise Exception(
                    "Tried to perform in-place addition with invalid number. "
                    "QuantumFloat precision too low."
                )

            input_qf_list = []
            poly = sp.sympify(other)

            polynomial_encoder(input_qf_list, self, poly)

        else:
            raise Exception(
                "In-place addition for type " + str(type(other)) + " not implemented"
            )

        return self

    @gate_wrap(permeability=[1], is_qfree=True)
    def __isub__(self, other):

        from qrisp.alg_primitives.arithmetic import polynomial_encoder

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():
            with invert():
                self.__iadd__(other)
            return self

        if isinstance(other, QuantumFloat):
            input_qf_list = [other]
            poly = -sp.symbols("x")

            polynomial_encoder(input_qf_list, self, poly)

        elif isinstance(other, (int, float)):
            if not int(other / 2**self.exponent) == other / 2**self.exponent:
                raise Exception(
                    "Tried to perform in-place subtraction with invalid number. "
                    "QuantumFloat precision too low."
                )

            input_qf_list = []
            poly = -sp.sympify(other)

            polynomial_encoder(input_qf_list, self, poly)

        else:
            raise Exception(
                "In-place substraction for type "
                + str(type(other))
                + " not implemented"
            )

        return self

    @gate_wrap(permeability=[], is_qfree=True)
    def __imul__(self, other):

        from qrisp.alg_primitives.arithmetic import inpl_mult

        inpl_mult(self, other)

        return self

    def __irshift__(self, k):
        self.exp_shift(-k)
        return self

    def __ilshift__(self, k):
        self.exp_shift(k)
        return self

    def __lt__(self, other):
        from qrisp.alg_primitives.arithmetic import lt, uint_lt, gidney_adder

        if check_for_tracing_mode():
            return uint_lt(self, other, gidney_adder)
        else:
            if not isinstance(other, (QuantumFloat, int, float)):
                raise Exception(f"Comparison with type {type(other)} not implemented")

            return lt(self, other)

    def __gt__(self, other):
        from qrisp.alg_primitives.arithmetic import gt, uint_gt, gidney_adder

        if check_for_tracing_mode():
            return uint_gt(self, other, gidney_adder)
        else:
            if not isinstance(other, (QuantumFloat, int, float)):
                raise Exception(f"Comparison with type {type(other)} not implemented")

            return gt(self, other)

    def __le__(self, other):
        from qrisp.alg_primitives.arithmetic import leq, uint_le, gidney_adder

        if check_for_tracing_mode():
            return uint_le(self, other, gidney_adder)
        else:
            if not isinstance(other, (QuantumFloat, int, float)):
                raise Exception(f"Comparison with type {type(other)} not implemented")

            return leq(self, other)

    def __ge__(self, other):
        from qrisp.alg_primitives.arithmetic import geq, uint_ge, gidney_adder

        if check_for_tracing_mode():
            return uint_ge(self, other, gidney_adder)
        else:

            if not isinstance(other, (QuantumFloat, int, float)):
                raise Exception(f"Comparison with type {type(other)} not implemented")

            return geq(self, other)

    def __eq__(self, other):

        from qrisp.alg_primitives.arithmetic import eq

        if not check_for_tracing_mode() and not isinstance(
            other, (QuantumFloat, int, float)
        ):
            raise Exception(f"Comparison with type {type(other)} not implemented")

        return eq(self, other)

    def __ne__(self, other):

        from qrisp.alg_primitives.arithmetic import neq

        if not check_for_tracing_mode() and not isinstance(
            other, (QuantumFloat, int, float)
        ):
            raise Exception(f"Comparison with type {type(other)} not implemented")

        return neq(self, other)

    def exp_shift(self, shift):
        """
        Performs an internal bit shift. Note that this method doesn't cost any
        quantum gates. For the quantum version of this method, see
        :meth:`quantum_bit_shift<qrisp.QuantumFloat.quantum_bitshift>`.

        Parameters
        ----------
        shift : int
            The amount to shift.

        Raises
        ------
        Exception
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

        .. code-block:: none

            QuantumCircuit:
            --------------
            a.0: ─────
                 ┌───┐
            a.1: ┤ X ├
                 └───┘
            a.2: ─────
            <BLANKLINE>
            a.3: ─────

            Live QuantumVariables:
            ---------------------
            QuantumFloat a

        """
        if not isinstance(shift, int):
            raise Exception("Tried to shift QuantumFloat exponent by non-integer value")

        self.exponent += shift

    def add_sign(self):
        """
        Turns an unsigned QuantumFloat into its signed version.

        Raises
        ------
        Exception
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
            raise Exception(r"Tried to add sign to signed QuantumFloat")

        self.extend(1, self.size)
        self.signed = True

    def sign(self):
        r"""
        Returns the sign qubit.

        This qubit is in state $\ket{1}$ if the QuantumFloat holds a negative value and
        in state $\ket{0}$ otherwise.

        For more information about the encoding of negative numbers check the
        `publication <https://ieeexplore.ieee.org/document/9815035>`_.

        .. warning::

            Performing an X gate on this qubit does not flip the sign! Use inplace
            multiplication instead.

            >>> from qrisp import QuantumFloat
            >>> qf = QuantumFloat(3, signed = True)
            >>> qf[:] = 3
            >>> qf *= -1
            >>> print(qf)
            {-3: 1.0}

        Raises
        ------
        Exception
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
        >>> cx(qf.sign(), qbl)
        >>> print(qbl)
        {True: 0.6667, False: 0.3333}

        """
        if not self.signed:
            raise Exception("Tried to retrieve sign qubit of unsigned QuantumFloat")

        return self[-1]

    def init_from(
        self, other, ignore_rounding_errors=False, ignore_overflow_errors=False
    ):
        copy_qf(
            self,
            other,
            ignore_rounding_errors=ignore_rounding_errors,
            ignore_overflow_errors=ignore_overflow_errors,
        )

    def incr(self, x=None):
        from qrisp.alg_primitives.arithmetic.adders.incrementation import increment

        if x is None:
            x = 2**self.exponent
        increment(self, x)

    def __hash__(self):
        return id(self)

    def significant(self, k):
        """
        Returns the qubit with significance $k$.

        Parameters
        ----------
        k : int
            The significance.

        Raises
        ------
        Exception
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
        >>> x(qf.significant(-2))
        >>> print(qf)
        {0.25: 1.0}

        The qubit with significance $-2$ corresponds to the value $0.25 = 2^{-2}$.

        >>> x(qf.significant(2))
        {4.25: 1.0}

        The qubit with significance $2$ corresponds to the value $4 = 2^{2}$.

        """

        sig_list = list(range(self.mshape[0], self.mshape[1]))

        if k not in sig_list:
            raise Exception(
                f"Tried to retrieve invalid significant {k} "
                f"from QuantumFloat with mantissa shape {self.mshape}"
            )

        return self[sig_list.index(k)]

    def truncate(self, x):
        """
        Receives a regular float and returns the float that is closest to the input but
        can still be encoded.

        Parameters
        ----------
        x : float
            A float that is supposed to be truncated.

        Returns
        -------
        float
            The truncated float.

        Examples
        --------

        We create a QuantumFloat and round a value to fit the encoder and subsequently
        initiate:

        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(4, -1)
        >>> value = 0.5102341
        >>> qf[:] = value
        Exception: Value 0.5102341 not supported by encoder.
        >>> rounded_value = qf.truncate(value)
        >>> rounded_value
        0.5
        >>> qf[:] = rounded_value
        >>> print(qf)
        {0.5: 1.0}

        """

        res = jnp.int64(jnp.round(x / jnp.float64(2) ** self.exponent))
        res = jnp.minimum(2**self.msize - 1, res)

        if self.signed:
            res = jnp.maximum(-(2**self.msize), res)
            res = _signed_int_iso(res, self.size)
        else:
            res = jnp.maximum(0, res)

        return self.decoder(res)

    def get_ev(self, **mes_kwargs):
        """
        Retrieves the expectation value of self.

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
        >>> h(qf)
        >>> qf.get_ev()
        7.5

        """

        mes_res = self.get_measurement(**mes_kwargs)

        return sum([k * v for k, v in mes_res.items()])

    def quantum_bit_shift(self, shift_amount):
        """
        Performs a bit shift in the quantum device.
        While :meth:`exp_shift<qrisp.QuantumFloat.exp_shift>` performs a bit shift
        in the compiler (thus costing no quantum gates) this method performs the
        bitshift on the hardware.

        This has the advantage, that it can be controlled if called within a
        :ref:`ControlEnvironment` and furthermore admits bit shifts based on the
        state of a QuantumFloat

        .. note::

            Bit bit shifts based on a QuantumFloat are currently only possible
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
        Exception
            Tried to shift QuantumFloat exponent by non-integer value
        Exception
            Quantum-quantum bitshifting is currently only supported for unsigned arguments

        Examples
        --------

        We create a QuantumFloat and a QuantumBool to perform a controlled bit shift.

        ::

            from qrisp import QuantumFloat, QuantumBool, h
            qf = QuantumFloat(4)
            qf[:] = 1
            qbl = QuantumBool()
            h(qbl)

            with qbl:
                qf.quantum_bit_shift(2)

        Evaluate the result

        >>> print(qf.qs.statevector())
        sqrt(2)*(|1>*|False> + |4>*|True>)/2

        """

        from qrisp.alg_primitives.arithmetic import quantum_bit_shift

        quantum_bit_shift(self, shift_amount)


def create_output_qf(operands, op):
    if isinstance(op, sp.core.expr.Expr):
        from qrisp.alg_primitives.arithmetic.poly_tools import expr_to_list

        expr_list = expr_to_list(op)

        for i in range(len(expr_list)):
            if not isinstance(expr_list[i][0], sp.Symbol):
                expr_list[i].pop(0)

        operands.sort(key=lambda x: x.name)

        def prod(iter):
            iter = list(iter)
            a = iter[0]
            for i in range(1, len(iter)):
                a *= iter[i]

            return a

        from sympy import Abs, Poly, Symbol

        poly = Poly(op)
        monom_list = [
            a * prod(x**k for x, k in zip(poly.gens, mon))
            for a, mon in zip(poly.coeffs(), poly.monoms())
        ]

        max_value_dic = {Symbol(qf.name): 2.0 ** qf.mshape[1] for qf in operands}
        min_value_dic = {Symbol(qf.name): 2.0 ** qf.mshape[0] for qf in operands}

        abs_poly = sum([Abs(monom) for monom in monom_list], 0)

        min_poly_value = min(
            [float(Abs(monom).subs(min_value_dic)) for monom in monom_list]
        )

        max_poly_value = float(abs_poly.subs(max_value_dic))

        min_sig = int(np.floor(np.log2(min_poly_value)))
        max_sig = int(np.ceil(np.log2(max_poly_value)))

        msize = max_sig - min_sig
        exponent = min_sig

        signed = bool(sum([int(operand.signed) for operand in operands]))

        return QuantumFloat(msize, exponent=exponent, signed=signed)

    from qrisp.qtypes import QuantumModulus

    if all(isinstance(operand, QuantumModulus) for operand in operands):
        res = operands[0].duplicate()
        if op == "mul":
            res.m = (
                operands[0].m
                + operands[1].m
                - (
                    int(np.ceil(np.log2((operands[0].modulus - 1) ** 2) + 1))
                    - operands[0].size
                )
            )
        return res

    if op == "add":
        signed = operands[0].signed or operands[1].signed
        exponent = jnp.minimum(operands[0].exponent, operands[1].exponent)
        max_sig = jnp.maximum(operands[0].mshape[1], operands[1].mshape[1]) + 1
        msize = max_sig - exponent + 1

        return QuantumFloat(
            msize, exponent, operands[0].qs, signed=signed, name="add_res*"
        )

    if op == "mul":
        signed = operands[0].signed or operands[1].signed

        if operands[0].reg == operands[1].reg and (
            operands[0].signed and operands[1].signed
        ):
            signed = False

        return QuantumFloat(
            operands[0].msize
            + operands[1].msize
            + operands[0].signed * operands[1].signed,
            operands[0].exponent + operands[1].exponent,
            operands[0].qs,
            signed=signed,
            name="mul_res*",
        )

    if op == "sub":
        exponent = jnp.minimum(operands[0].exponent, operands[1].exponent)
        max_sig = jnp.maximum(operands[0].mshape[1], operands[1].mshape[1]) + 1
        msize = max_sig - exponent + 1

        return QuantumFloat(
            msize, exponent, operands[0].qs, signed=True, name="sub_res*"
        )


# Initiates the value of qf2 into qf1 where qf1 has to hold the value 0
def copy_qf(qf1, qf2, ignore_overflow_errors=False, ignore_rounding_errors=False):
    # Lists that translate Qubit index => Significance
    qf1_sign_list = [qf1.exponent + i for i in range(qf1.size)]
    qf2_sign_list = [qf2.exponent + i for i in range(qf2.size)]

    # Check overflow/underflow
    if max(qf1_sign_list) < max(qf2_sign_list) and not ignore_overflow_errors:
        raise Exception(
            "Copy operation would result in overflow "
            "(use ignore_overflow_errors = True)"
        )

    if min(qf1_sign_list) > min(qf2_sign_list) and not ignore_rounding_errors:
        raise Exception(
            "Copy operation would result in rounding "
            "(use ignore_rounding_errors = True)"
        )

    qs = qf1.qs

    if qf2.signed:
        if not qf1.signed:
            raise Exception("Tried to copy signed into unsigend float")

        # Remove last entry from significance list (last qubit is the sign qubit)
        qf2_sign_list.pop(-1)
        qf1_sign_list.pop(-1)

    for i in range(len(qf1_sign_list)):
        # If we are in a realm where both floats have overlapping significance
        # => CNOT into each other
        if qf1_sign_list[i] in qf2_sign_list:
            qf2_index = qf2_sign_list.index(qf1_sign_list[i])
            qs.cx(qf2[qf2_index], qf1[i])
            continue

        # Otherwise copy the sign bit into the bits of higher significance than qf2
        if qf1_sign_list[i] > max(qf2_sign_list) and qf2.signed:
            qs.cx(qf2[-1], qf1[i])

    # Copy the sign bit
    if qf2.signed:
        qs.cx(qf2[-1], qf1[-1])
